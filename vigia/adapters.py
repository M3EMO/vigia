"""Registro declarativo de arquitecturas MoE de HF transformers (v5).

Cada arquitectura es un dato (`RouterSpec`), no una clase con lógica propia. Agregar un modelo es
agregar una fila con `register`. La detección de capas MoE mira los módulos reales: una capa cuyo
bloque no tiene router (capa densa de DeepSeek-V3 o de Qwen con `mlp_only_layers`) queda fuera.
Los expertos compartidos (`shared_experts_attr`) nunca entran en las señales del router.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from vigia.contracts import ModelInfo, UnsupportedModelError
from vigia.features import Gating


@dataclass(frozen=True)
class RouterSpec:
    model_type: str
    block_attr: str  # atributo del bloque MoE dentro de la capa del decoder
    router_attr: str  # atributo del router dentro del bloque
    experts_attr: str  # módulo de expertos fusionado (clase *Experts)
    gating: Gating
    num_experts_key: str  # clave del config con N
    top_k_key: str = "num_experts_per_tok"
    renormalize: bool = False  # renormalización top-k si el router no expone `norm_topk_prob`
    shared_experts_attr: str | None = None  # experto siempre activo, excluido de F1


_REGISTRY: dict[str, RouterSpec] = {}


def register(spec: RouterSpec) -> None:
    """Agrega o reemplaza la fila de un `model_type`."""
    _REGISTRY[spec.model_type] = spec


for _spec in (
    RouterSpec("mixtral", "mlp", "gate", "experts", "softmax_topk", "num_local_experts", renormalize=True),
    RouterSpec("gpt_oss", "mlp", "router", "experts", "topk_softmax", "num_local_experts"),
    RouterSpec(
        "qwen2_moe",
        "mlp",
        "gate",
        "experts",
        "softmax_topk",
        "num_experts",
        shared_experts_attr="shared_expert",
    ),
    RouterSpec("qwen3_moe", "mlp", "gate", "experts", "softmax_topk", "num_experts"),
    RouterSpec("olmoe", "mlp", "gate", "experts", "softmax_topk", "num_experts"),
    RouterSpec("granitemoe", "block_sparse_moe", "router", "experts", "topk_softmax", "num_local_experts"),
    RouterSpec(
        "deepseek_v3",
        "mlp",
        "gate",
        "experts",
        "sigmoid_group",
        "n_routed_experts",
        shared_experts_attr="shared_experts",
    ),
    RouterSpec("phimoe", "mlp", "router", "experts", "sparsemixer", "num_local_experts"),
):
    register(_spec)


def supported_model_types() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def _unsupported(msg: str) -> UnsupportedModelError:
    return UnsupportedModelError(f"{msg}. Arquitecturas soportadas: {', '.join(supported_model_types())}")


def get_spec(model_type: str) -> RouterSpec:
    try:
        return _REGISTRY[model_type]
    except KeyError:
        raise _unsupported(f"model_type '{model_type}' no soportado") from None


@dataclass(frozen=True)
class MoEBlock:
    layer: int  # índice de la capa en el decoder
    block: nn.Module
    router: nn.Module
    experts: nn.Module | None
    spec: RouterSpec


def decoder_layers(model: nn.Module) -> nn.ModuleList:
    """Capas del decoder de un `*ForCausalLM` o de su modelo base."""
    for candidate in (getattr(model, "model", None), getattr(model, "base_model", None), model):
        layers = getattr(candidate, "layers", None)
        if isinstance(layers, nn.ModuleList):
            return layers
    raise _unsupported(f"no se encontraron capas de decoder en {type(model).__name__}")


def final_norm(model: nn.Module) -> nn.Module | None:
    """Norma final del decoder (para el logit lens), si existe."""
    for candidate in (getattr(model, "model", None), getattr(model, "base_model", None)):
        norm = getattr(candidate, "norm", None)
        if isinstance(norm, nn.Module):
            return norm
    return None


def _is_router(module: Any, n_experts: int) -> bool:
    weight = getattr(module, "weight", None)
    return (
        isinstance(module, nn.Module)
        and "Router" in type(module).__name__
        and isinstance(weight, torch.Tensor)
        and weight.dim() == 2
        and weight.shape[0] == n_experts
    )


def _config(model: nn.Module) -> Any:
    config = getattr(model, "config", None)
    if config is None:
        raise _unsupported(f"{type(model).__name__} no tiene config")
    return config


def find_moe_blocks(model: nn.Module) -> list[MoEBlock]:
    """Bloques MoE del modelo en orden de capa. Falla con la lista de soportados si no hay ninguno."""
    config = _config(model)
    model_type = getattr(config, "model_type", "")
    spec = get_spec(model_type)
    n_experts = int(getattr(config, spec.num_experts_key))
    blocks = []
    for i, layer in enumerate(decoder_layers(model)):
        block = getattr(layer, spec.block_attr, None)
        router = getattr(block, spec.router_attr, None)
        if block is None or not _is_router(router, n_experts):
            continue
        blocks.append(MoEBlock(i, block, router, getattr(block, spec.experts_attr, None), spec))
    if not blocks:
        raise _unsupported(
            f"no se encontraron capas MoE en el modelo (model_type '{model_type}', se buscó "
            f"'{spec.block_attr}.{spec.router_attr}' con un router de {n_experts} expertos)"
        )
    return blocks


def top_k(model: nn.Module) -> int:
    config = _config(model)
    return int(getattr(config, get_spec(config.model_type).top_k_key))


def routing_params(block: MoEBlock) -> dict[str, Any]:
    """Parámetros de `vigia.features.route` leídos del router real (no del config)."""
    router = block.router
    params: dict[str, Any] = {"renormalize": bool(getattr(router, "norm_topk_prob", block.spec.renormalize))}
    if block.spec.gating in ("sigmoid_group", "sigmoid_topk"):
        bias = getattr(router, "e_score_correction_bias", None)
        params["correction_bias"] = None if bias is None else bias.detach().float().cpu().numpy()
        params["n_group"] = int(getattr(router, "num_group", 1))
        params["topk_group"] = int(getattr(router, "topk_group", 1))
        params["scaling"] = float(getattr(router, "routed_scaling_factor", 1.0))
    if block.spec.gating == "sparsemixer":
        params["jitter_eps"] = float(getattr(router, "router_jitter_noise", 0.01))
    return params


def _quantization(config: Any) -> str:
    q = getattr(config, "quantization_config", None)
    if q is None:
        return "none"
    method = q.get("quant_method") if isinstance(q, dict) else getattr(q, "quant_method", None)
    return str(getattr(method, "value", method) or "unknown")


def model_info(model: nn.Module, model_id: str) -> ModelInfo:
    """`ModelInfo` del contrato a partir del modelo cargado."""
    config = _config(model)
    blocks = find_moe_blocks(model)
    spec = blocks[0].spec
    param = next(model.parameters(), None)
    dtype = str(param.dtype).replace("torch.", "") if param is not None else "float32"
    return ModelInfo(
        model_id=model_id,
        model_type=config.model_type,
        n_layers=len(decoder_layers(model)),
        moe_layers=tuple(b.layer for b in blocks),
        n_experts=int(getattr(config, spec.num_experts_key)),
        top_k=top_k(model),
        vocab_size=int(config.vocab_size),
        gating=spec.gating,
        dtype=dtype,
        quantization=_quantization(config),
        revision=str(getattr(config, "_commit_hash", None) or ""),
    )
