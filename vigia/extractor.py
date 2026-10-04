"""Extractor de señales por token con hooks sobre los bloques MoE (ADR A-01).

Protocolo de un paso (prefill o un token de decodificación):

    with SignalExtractor(model, tiers=(Tier.F0, Tier.F1)) as ex:
        ex.begin_step(attention_mask=mask, score_mask=score)   # opcional
        out = model(input_ids=ids, attention_mask=mask, ...)
        rows = ex.end_step(out.logits, targets)               # list[list[TokenSignals]] por fila

Alineación: la posición p de un forward produce las features del token muestreado desde los
logits de p (`targets[b, p]`). Así F0 y F1 describen el mismo cómputo que eligió el token.

- F1: un `forward_pre_hook` sobre cada bloque MoE recalcula `F.linear(x, W_router, b)` en float32,
  así que funciona aunque el bloque no llame al router (MXFP4). La reducción a escalares ocurre
  dentro del hook; nunca se guardan tensores por experto.
- U/F1: el uso acumulado y `r_switch` son causales por fila: solo cuentan posiciones puntuadas
  (tokens generados) hasta t. `r_switch` compara con la posición válida anterior.
- F2 (opt-in): un forward hook sobre el módulo de expertos recalcula la salida de cada experto
  elegido sin tocar la salida del modelo. Con expertos cuantizados queda deshabilitado.
- H: residual stream tras `probe_layer` capas del decoder (convención `hidden_states[i]` de HF).
- Las posiciones con `attention_mask == 0` no generan features ni actualizan estado (AT-S05).
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from vigia import features as fx
from vigia.adapters import MoEBlock, decoder_layers, final_norm, find_moe_blocks, routing_params, top_k
from vigia.contracts import (
    AGGREGATES,
    F1_LAYER_FEATURES,
    F2_LAYER_FEATURES,
    USAGE_LAYER_FEATURES,
    HookNotFiredError,
    Tier,
    agg_feature,
    layer_feature,
)

_ROUTER_TIERS = frozenset({Tier.F1, Tier.U})
_F2_CHUNK = 64  # filas por lote al recalcular expertos
_LENS_CHUNK = 8  # filas por lote en el logit lens (k × |V| floats por fila)


@dataclass
class TokenSignals:
    features: dict[str, float]
    expert_ids: list[list[int]] | None = None  # [capa_moe][k]


@dataclass
class _Step:
    valid: np.ndarray | None = None  # (B, T) bool
    scored: np.ndarray | None = None  # (B, T) bool
    shape: tuple[int, int] | None = None  # (B, T) visto por los hooks
    router: dict[int, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)  # (packed, idx)
    experts: dict[int, dict[str, np.ndarray]] = field(default_factory=dict)
    hidden: tuple[np.ndarray, np.ndarray] | None = None  # (filas planas, valores (n, d))
    fired: set[tuple[str, int]] = field(default_factory=set)

    def score_rows(self, n: int) -> torch.Tensor | None:
        if self.scored is None or self.scored.size != n:
            return None
        return torch.from_numpy(np.flatnonzero(self.scored.reshape(-1)))


def _inference() -> Any:
    """`inference_mode` solo si no está activo: entrar al contexto cuesta ~100 µs por hook."""
    return nullcontext() if torch.is_inference_mode_enabled() else torch.inference_mode()


def _is_float_3d(t: Any) -> bool:
    return isinstance(t, torch.Tensor) and t.dim() == 3 and t.is_floating_point()


def f2_unsupported_reason(blocks: list[MoEBlock]) -> str | None:
    """Motivo legible si no se pueden recalcular las salidas por experto; None si se puede."""
    for b in blocks:
        ex = b.experts
        if ex is None:
            return f"F2: el bloque de la capa {b.layer} no tiene módulo de expertos"
        up = getattr(ex, "gate_up_proj" if getattr(ex, "has_gate", False) else "up_proj", None)
        flags = all(hasattr(ex, a) for a in ("has_gate", "has_bias", "is_transposed"))
        if not (flags and _is_float_3d(up) and _is_float_3d(getattr(ex, "down_proj", None))):
            return f"F2: expertos cuantizados o con kernel fusionado ({type(ex).__name__})"
    return None


def expert_outputs(experts: nn.Module, hidden: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Salidas sin ponderar de los expertos elegidos: (n, H) y (n, k) → (n, k, H) en float32.

    Replica el forward eager de los módulos `*Experts` de transformers v5 (pesos 3D, gate opcional,
    bias opcional, layout transpuesto) experto por experto, sin materializar pesos por token.
    """
    n, k = idx.shape
    x = hidden.float()
    out = torch.zeros((n, k, x.shape[-1]), dtype=torch.float32, device=x.device)
    gated = experts.has_gate
    up_w = experts.gate_up_proj if gated else experts.up_proj
    up_b = (experts.gate_up_proj_bias if gated else experts.up_proj_bias) if experts.has_bias else None
    down_b = experts.down_proj_bias if experts.has_bias else None

    def linear(h: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None) -> torch.Tensor:
        y = h @ w.float() if experts.is_transposed else F.linear(h, w.float())
        return y if b is None else y + b.float()

    for e in torch.unique(idx).tolist():
        rows, slots = torch.where(idx == e)
        proj = linear(x[rows], up_w[e], None if up_b is None else up_b[e])
        proj = experts._apply_gate(proj) if gated else experts.act_fn(proj)
        out[rows, slots] = linear(proj, experts.down_proj[e], None if down_b is None else down_b[e])
    return out


def resolve_probe_layer(probe_layer: int | None, n_layers: int) -> int:
    """Capa de la sonda H: por defecto round(0.95·L). Debe caer en [1, L]."""
    p = round(0.95 * n_layers) if probe_layer is None else int(probe_layer)
    if not 1 <= p <= n_layers:
        raise ValueError(f"probe_layer={p} fuera de [1, {n_layers}]")
    return p


def _as_bool(mask: Any) -> np.ndarray | None:
    if mask is None:
        return None
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError("las máscaras deben tener forma (batch, tiempo)")
    return arr.astype(bool)


class SignalExtractor:
    """Context manager que instala los hooks y produce `TokenSignals` por token puntuado."""

    def __init__(
        self,
        model: nn.Module,
        tiers: tuple[Tier | str, ...] = (Tier.F0, Tier.F1),
        probe_layer: int | None = None,
        *,
        keep_router_logits: bool = False,
    ) -> None:
        self.model = model
        self.blocks = find_moe_blocks(model)
        self.layers = tuple(b.layer for b in self.blocks)
        self._layer_rows = np.arange(len(self.layers))[:, None]
        self._nan_layers = np.full(len(self.layers), np.nan)
        self._names = {
            bases: (
                [layer_feature(base, layer) for base in bases for layer in self.layers],
                [agg_feature(agg, base) for base in bases for agg in AGGREGATES],
            )
            for bases in (F1_LAYER_FEATURES + USAGE_LAYER_FEATURES, USAGE_LAYER_FEATURES)
        }
        self.n_experts = int(self.blocks[0].router.weight.shape[0])
        self.top_k = top_k(model)
        self._route_kw = [routing_params(b) for b in self.blocks]
        self.disabled: list[str] = []
        wanted = {Tier(t) for t in tiers}
        if Tier.F2 in wanted:
            reason = f2_unsupported_reason(self.blocks)
            if reason:
                self.disabled.append(reason)
                wanted.discard(Tier.F2)
        self.tiers = frozenset(wanted)
        self._lens: tuple[nn.Module, nn.Module] | None = None
        if Tier.F2 in self.tiers:
            norm, head = final_norm(model), model.get_output_embeddings()
            if norm is None or head is None:
                self.disabled.append("F2: e_jsd sin norma final o cabeza de salida")
            else:
                self._lens = (norm, head)
        self.probe_layer = (
            resolve_probe_layer(probe_layer, len(decoder_layers(model))) if Tier.H in self.tiers else None
        )
        self.keep_router_logits = keep_router_logits
        self.router_logits: dict[int, torch.Tensor] = {}  # capa -> (B, T, N) del último paso
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._step: _Step | None = None
        self.reset()

    # --- ciclo de vida ----------------------------------------------------------------------

    def __enter__(self) -> SignalExtractor:
        try:
            if self.tiers & _ROUTER_TIERS:
                for j, b in enumerate(self.blocks):
                    self._handles.append(
                        b.block.register_forward_pre_hook(partial(self._router_hook, j), with_kwargs=True)
                    )
            if Tier.F2 in self.tiers:
                for j, b in enumerate(self.blocks):
                    self._handles.append(
                        b.experts.register_forward_hook(partial(self._experts_hook, j), with_kwargs=True)
                    )
            if self.probe_layer is not None:
                layer = decoder_layers(self.model)[self.probe_layer - 1]
                self._handles.append(layer.register_forward_hook(self._probe_hook))
        except BaseException:
            self._remove_hooks()
            raise
        return self

    def __exit__(self, *exc: object) -> None:
        self._remove_hooks()

    def _remove_hooks(self) -> None:
        try:
            for h in self._handles:
                h.remove()
        finally:
            self._handles.clear()
            self._step = None

    def reset(self) -> None:
        """Borra el estado causal (uso acumulado, selección previa). Llamar entre secuencias."""
        self._counts: list[np.ndarray] | None = None
        self._prev_sel: list[np.ndarray | None] = []

    # --- hooks ------------------------------------------------------------------------------

    def _current(self) -> _Step:
        if self._step is None:
            self._step = _Step()
        return self._step

    def _router_hook(self, j: int, module: nn.Module, args: tuple, kwargs: dict) -> None:
        x = args[0] if args else next(v for v in kwargs.values() if isinstance(v, torch.Tensor))
        block = self.blocks[j]
        with _inference():
            h = x.detach().reshape(-1, x.shape[-1]).float()
            bias = getattr(block.router, "bias", None)
            logits = F.linear(
                h, block.router.weight.detach().float(), None if bias is None else bias.detach().float()
            )
            stats = fx.router_stats(logits.cpu().numpy(), block.spec.gating, self.top_k, **self._route_kw[j])
        step = self._current()
        step.shape = tuple(x.shape[:-1]) if x.dim() == 3 else (1, h.shape[0])
        # (N, 5 + k): las 5 features F1 en el orden del contrato y los pesos top-k; idx aparte.
        step.router[j] = (
            np.column_stack([stats[name] for name in F1_LAYER_FEATURES] + [stats["w"]]),
            stats["idx"],
        )
        step.fired.add(("router", j))
        if self.keep_router_logits:
            self.router_logits[block.layer] = logits.reshape(*step.shape, -1).clone()

    def _experts_hook(self, j: int, module: nn.Module, args: tuple, kwargs: dict, output: Any) -> None:
        hidden = args[0] if args else kwargs["hidden_states"]
        idx = args[1] if len(args) > 1 else kwargs.get("top_k_index", kwargs.get("router_indices"))
        w = args[2] if len(args) > 2 else kwargs.get("top_k_weights", kwargs.get("routing_weights"))
        step = self._current()
        n = hidden.shape[0]
        rows = step.score_rows(n)
        rows = torch.arange(n) if rows is None else rows
        result = {name: np.full(n, np.nan) for name in (*F2_LAYER_FEATURES, "e_jsd")}
        last = j == len(self.blocks) - 1
        with _inference():
            for start in range(0, len(rows), _F2_CHUNK):
                sel = rows[start : start + _F2_CHUNK].to(hidden.device)
                e = expert_outputs(module, hidden.detach()[sel], idx.detach()[sel])
                w_np = w.detach()[sel].float().cpu().numpy()
                e_np = e.cpu().numpy()
                r = sel.cpu().numpy()
                result["e_cos"][r] = fx.expert_cosine(e_np, w_np)
                result["e_disp"][r] = fx.expert_dispersion(e_np, w_np)
                if last and self._lens is not None:
                    result["e_jsd"][r] = self._jsd(e, w_np)
        step.experts[j] = result
        step.fired.add(("experts", j))

    def _jsd(self, e: torch.Tensor, w: np.ndarray) -> np.ndarray:
        """JSD entre las distribuciones logit-lens (norma final + lm_head) de los expertos elegidos."""
        norm, head = self._lens  # type: ignore[misc]
        scale = float(getattr(self.model.config, "logits_scaling", 1.0) or 1.0)
        norm_dtype = next(norm.parameters(), e).dtype
        out = np.empty(e.shape[0])
        for start in range(0, e.shape[0], _LENS_CHUNK):
            chunk = e[start : start + _LENS_CHUNK]
            normed = norm(chunk.to(norm_dtype)).float()
            bias = None if getattr(head, "bias", None) is None else head.bias.float()
            logits = F.linear(normed, head.weight.float(), bias) / scale
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            out[start : start + _LENS_CHUNK] = fx.weighted_jsd(probs, w[start : start + _LENS_CHUNK])
        return out

    def _probe_hook(self, module: nn.Module, args: tuple, output: Any) -> None:
        hs = output[0] if isinstance(output, tuple) else output
        flat = hs.detach().reshape(-1, hs.shape[-1])
        step = self._current()
        rows = step.score_rows(flat.shape[0])
        rows = torch.arange(flat.shape[0]) if rows is None else rows
        with _inference():
            step.hidden = (rows.numpy(), flat[rows.to(flat.device)].float().cpu().numpy())
        step.fired.add(("probe", 0))

    # --- protocolo por paso -----------------------------------------------------------------

    def begin_step(self, attention_mask: Any = None, score_mask: Any = None) -> None:
        """Declara las máscaras (B, T) del próximo forward. Sin máscaras, todo es válido y puntuado."""
        valid = _as_bool(attention_mask)
        scored = _as_bool(score_mask)
        if valid is not None and scored is not None and valid.shape != scored.shape:
            raise ValueError("attention_mask y score_mask deben tener la misma forma")
        if scored is not None and valid is not None:
            scored = scored & valid
        elif scored is None:
            scored = valid
        self.router_logits.clear()
        self._step = _Step(valid=valid, scored=scored)

    def _expected_hooks(self) -> set[tuple[str, int]]:
        expected: set[tuple[str, int]] = set()
        if self.tiers & _ROUTER_TIERS:
            expected |= {("router", j) for j in range(len(self.blocks))}
        if Tier.F2 in self.tiers:
            expected |= {("experts", j) for j in range(len(self.blocks))}
        if self.probe_layer is not None:
            expected.add(("probe", 0))
        return expected

    def end_step(self, logits: torch.Tensor | None = None, targets: Any = None) -> list[list[TokenSignals]]:
        """Reduce el último forward a `TokenSignals` por fila, en orden de posición.

        `logits` (B, T', V) y `targets` (B, T') cubren las últimas T' posiciones del paso (como
        `logits_to_keep` de HF). Hacen falta si el nivel F0 está activo.
        """
        step = self._step or _Step()
        self._step = None
        missing = self._expected_hooks() - step.fired
        if missing:
            where = sorted(
                {self.blocks[j].layer if kind != "probe" else self.probe_layer for kind, j in missing}
            )
            raise HookNotFiredError(
                f"los hooks no dispararon en las capas {where}: el forward no pasó por los bloques "
                "instrumentados (¿kernel fusionado, torch.compile o forward fuera del contexto?)"
            )
        shape = step.shape
        for m in (step.scored, step.valid):
            if m is not None:
                if shape is not None and tuple(m.shape) != tuple(shape):
                    raise ValueError(f"máscara {tuple(m.shape)} no coincide con el forward {shape}")
                shape = tuple(m.shape)
        if shape is None:
            if logits is None:
                raise ValueError("no hubo forward ni logits para este paso")
            shape = tuple(logits.shape[:2])
        bsz, seq = shape
        valid = step.valid if step.valid is not None else np.ones(shape, dtype=bool)
        scored = step.scored if step.scored is not None else valid
        if self._counts is None:
            self._counts = [np.zeros((len(self.blocks), self.n_experts)) for _ in range(bsz)]
            self._prev_sel = [None] * bsz
        elif len(self._counts) != bsz:
            raise ValueError("el tamaño de batch cambió: llamar a reset() entre secuencias")

        f0 = self._f0(logits, targets, scored, seq) if Tier.F0 in self.tiers else None
        router = self._stack_router(step) if self.tiers & _ROUTER_TIERS else None
        hidden_pos = {}
        if step.hidden is not None:
            hidden_pos = {int(r): i for i, r in enumerate(step.hidden[0])}

        out: list[list[TokenSignals]] = [[] for _ in range(bsz)]
        k_f0 = 0
        for b in range(bsz):
            pending: int | None = None
            for p in range(seq):
                if not valid[b, p]:
                    continue
                n = b * seq + p
                if not scored[b, p]:
                    pending = n
                    continue
                if router is not None and pending is not None:
                    self._prev_sel[b] = router["idx"][:, pending]
                pending = None
                feats: dict[str, float] = {}
                expert_ids = None
                if f0 is not None:
                    for name, values in f0.items():
                        feats[name] = float(values[k_f0])
                    k_f0 += 1
                if router is not None:
                    expert_ids = self._router_features(b, n, router, feats)
                if step.experts:
                    self._f2_features(n, step, feats)
                if step.hidden is not None and n in hidden_pos:
                    vec = step.hidden[1][hidden_pos[n]]
                    feats.update({f"h_{i}": float(v) for i, v in enumerate(vec)})
                out[b].append(TokenSignals(feats, expert_ids))
            if router is not None and pending is not None:
                self._prev_sel[b] = router["idx"][:, pending]
        return out

    # --- reducciones ------------------------------------------------------------------------

    def _f0(
        self, logits: torch.Tensor | None, targets: Any, scored: np.ndarray, seq: int
    ) -> dict[str, np.ndarray]:
        if logits is None or targets is None:
            raise ValueError("el nivel F0 necesita logits y targets en end_step")
        tgt = targets.detach().cpu().numpy() if isinstance(targets, torch.Tensor) else np.asarray(targets)
        kept = logits.shape[1]
        if tgt.shape != tuple(logits.shape[:2]):
            raise ValueError("targets debe tener la forma (B, T') de logits")
        offset = seq - kept
        if offset < 0:
            raise ValueError("logits cubre más posiciones que el paso")
        bs, ps = np.nonzero(scored)
        if len(ps) and ps.min() < offset:
            raise ValueError("hay posiciones puntuadas sin logits (logits_to_keep demasiado chico)")
        rows = logits.detach()[torch.as_tensor(bs), torch.as_tensor(ps - offset)].float().cpu().numpy()
        return fx.output_features(rows, tgt[bs, ps - offset])

    def _stack_router(self, step: _Step) -> dict[str, np.ndarray]:
        packed = [step.router[j] for j in range(len(self.blocks))]
        return {"packed": np.stack([p[0] for p in packed]), "idx": np.stack([p[1] for p in packed])}

    def _router_features(
        self, b: int, n: int, router: dict[str, np.ndarray], feats: dict[str, float]
    ) -> list:
        idx = router["idx"][:, n]  # (Lm, k)
        packed = router["packed"][:, n]  # (Lm, 5 + k)
        n_f1 = len(F1_LAYER_FEATURES)
        weighted = Tier.F1 in self.tiers
        counts = self._counts[b]  # type: ignore[index]
        np.add.at(counts, (self._layer_rows, idx), packed[:, n_f1:] if weighted else 1.0)
        prev = self._prev_sel[b]
        usage = fx.usage_stats(counts)
        bases = F1_LAYER_FEATURES + USAGE_LAYER_FEATURES if weighted else USAGE_LAYER_FEATURES
        values = np.empty((len(bases), len(self.layers)))
        head = n_f1 if weighted else 0
        if weighted:
            values[:n_f1] = packed[:, :n_f1].T
        values[head] = usage["u_gini"]
        values[head + 1] = usage["u_ihhi"]
        values[head + 2] = usage["u_ent"]
        values[head + 3] = self._nan_layers if prev is None else fx.route_switch(prev, idx)
        self._prev_sel[b] = idx
        aggs = fx.layer_aggregates(values, axis=1)
        agg_values = np.empty((len(bases), len(AGGREGATES)))
        for i, agg in enumerate(AGGREGATES):
            agg_values[:, i] = aggs[agg]
        names, agg_names = self._names[bases]
        feats.update(zip(names, values.ravel().tolist(), strict=True))
        feats.update(zip(agg_names, agg_values.ravel().tolist(), strict=True))
        return idx.tolist()

    def _f2_features(self, n: int, step: _Step, feats: dict[str, float]) -> None:
        for j, block in enumerate(self.blocks):
            res = step.experts[j]
            for base in F2_LAYER_FEATURES:
                feats[layer_feature(base, block.layer)] = float(res[base][n])
        feats["e_jsd"] = float(step.experts[len(self.blocks) - 1]["e_jsd"][n])
