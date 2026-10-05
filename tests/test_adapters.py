"""Registro de arquitecturas y detección de capas MoE (AT-S07)."""

import pytest
import torch
import transformers as tf
from transformers import AutoModelForCausalLM

from vigia import adapters
from vigia.contracts import UnsupportedModelError

BASE = dict(
    vocab_size=256,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
)
MOE = dict(num_experts_per_tok=2)


def tiny_config(model_type: str, **over):
    c = {
        "mixtral": lambda: tf.MixtralConfig(**BASE, **MOE, num_local_experts=4),
        "gpt_oss": lambda: tf.GptOssConfig(**BASE, **MOE, num_local_experts=4, head_dim=8),
        "qwen2_moe": lambda: tf.Qwen2MoeConfig(
            **BASE, **MOE, num_experts=4, moe_intermediate_size=32, shared_expert_intermediate_size=32
        ),
        "qwen3_moe": lambda: tf.Qwen3MoeConfig(
            **BASE, **MOE, num_experts=4, moe_intermediate_size=32, head_dim=8
        ),
        "olmoe": lambda: tf.OlmoeConfig(**BASE, **MOE, num_experts=4),
        "granitemoe": lambda: tf.GraniteMoeConfig(**BASE, **MOE, num_local_experts=4),
        "phimoe": lambda: tf.PhimoeConfig(**BASE, **MOE, num_local_experts=4),
        "deepseek_v3": lambda: tf.DeepseekV3Config(
            vocab_size=256,
            hidden_size=32,
            intermediate_size=64,
            moe_intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            n_routed_experts=4,
            num_experts_per_tok=2,
            n_group=1,
            topk_group=1,
            first_k_dense_replace=0,
            q_lora_rank=None,
            kv_lora_rank=16,
            qk_rope_head_dim=8,
            qk_nope_head_dim=8,
            v_head_dim=8,
        ),
    }[model_type]()
    for k, v in over.items():
        setattr(c, k, v)
    return c


def tiny(model_type: str, **over):
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(tiny_config(model_type, **over)).eval()


GATING = {
    "mixtral": "softmax_topk",
    "gpt_oss": "topk_softmax",
    "qwen2_moe": "softmax_topk",
    "qwen3_moe": "softmax_topk",
    "olmoe": "softmax_topk",
    "granitemoe": "topk_softmax",
    "phimoe": "sparsemixer",
    "deepseek_v3": "sigmoid_group",
}


def test_registry_covers_p01_architectures():
    assert set(adapters.supported_model_types()) >= set(GATING)


@pytest.mark.parametrize("model_type", sorted(GATING))
def test_find_blocks_and_model_info(model_type):
    model = tiny(model_type)
    blocks = adapters.find_moe_blocks(model)
    assert [b.layer for b in blocks] == [0, 1]
    assert all(b.experts is not None and b.router.weight.shape == (4, 32) for b in blocks)
    info = adapters.model_info(model, f"tiny/{model_type}")
    assert info.model_type == model_type
    assert info.moe_layers == (0, 1)
    assert (info.n_layers, info.n_experts, info.top_k, info.vocab_size) == (2, 4, 2, 256)
    assert info.gating == GATING[model_type]
    assert info.dtype == "float32" and info.quantization == "none"


def test_dense_layers_are_skipped():
    """Las capas densas (DeepSeek first_k_dense_replace, Qwen mlp_only_layers) quedan fuera."""
    ds = tiny("deepseek_v3", first_k_dense_replace=1)
    assert adapters.model_info(ds, "ds").moe_layers == (1,)
    qw = tiny("qwen2_moe", mlp_only_layers=[0])
    assert [b.layer for b in adapters.find_moe_blocks(qw)] == [1]


def test_routing_params_read_from_router():
    ds = adapters.find_moe_blocks(tiny("deepseek_v3"))[0]
    p = adapters.routing_params(ds)
    assert (
        p["scaling"] == pytest.approx(2.5) and p["renormalize"] is True and p["correction_bias"].shape == (4,)
    )
    assert adapters.routing_params(adapters.find_moe_blocks(tiny("mixtral"))[0])["renormalize"] is True
    assert adapters.routing_params(adapters.find_moe_blocks(tiny("olmoe"))[0])["renormalize"] is False
    assert "jitter_eps" in adapters.routing_params(adapters.find_moe_blocks(tiny("phimoe"))[0])


def test_at_s07_unsupported_model_type_lists_supported():
    """AT-S07: un model_type no soportado da un error que lista las arquitecturas soportadas."""
    torch.manual_seed(0)
    dense = AutoModelForCausalLM.from_config(tf.LlamaConfig(**BASE)).eval()
    with pytest.raises(UnsupportedModelError) as err:
        adapters.find_moe_blocks(dense)
    msg = str(err.value)
    assert "llama" in msg
    assert all(t in msg for t in adapters.supported_model_types())


def test_at_s07_supported_type_without_moe_layers():
    """AT-S07: un modelo sin capas MoE (todas densas) da error claro con la lista de soportadas."""
    model = tiny("qwen2_moe", mlp_only_layers=[0, 1])
    with pytest.raises(UnsupportedModelError, match="no se encontraron capas MoE") as err:
        adapters.model_info(model, "denso")
    assert "mixtral" in str(err.value)


def test_register_adds_architecture_by_data():
    spec = adapters.RouterSpec(
        "mixtral_test", "mlp", "gate", "experts", "softmax_topk", "num_local_experts", renormalize=True
    )
    adapters.register(spec)
    try:
        model = tiny("mixtral")
        model.config.model_type = "mixtral_test"
        assert adapters.model_info(model, "x").model_type == "mixtral_test"
    finally:
        adapters._REGISTRY.pop("mixtral_test")


# --- fingerprint de los routers (control de deriva, docs/spec/mf-moe-revision.md) ----------------


def _router_detector(info):
    from vigia.detector import Detector

    return Detector.build(
        model=info.fingerprint(),
        feature_names=["out_entropy"],
        mean=[0.0],
        std=[1.0],
        weights=[1.0],
        bias=0.0,
        platt={"a": 1.0, "b": 0.0},
        response_calibrator={"a": 1.0, "b": 0.0},
        thresholds=[{"alpha": 0.1, "tau": 0.5, "expected_flag_rate": 0.2, "expected_precision": None}],
        default_alpha=0.1,
        metrics={},
        training={
            "dataset_sha256": "0" * 64,
            "n_tokens": 10,
            "n_responses": 2,
            "label_source": "computable",
            "judge": None,
            "created_at": "2026-10-04T00:00:00+00:00",
        },
    )


@pytest.mark.parametrize("model_type", ["mixtral", "gpt_oss", "deepseek_v3"])
def test_router_sha256_is_stable_and_enters_the_fingerprint(model_type):
    a = adapters.model_info(tiny(model_type), "m")
    b = adapters.model_info(tiny(model_type), "m")
    assert a.router_sha256 is not None and len(a.router_sha256) == 64
    assert a.router_sha256 == b.router_sha256  # mismos pesos -> mismo hash
    assert a.fingerprint()["router_sha256"] == a.router_sha256


def test_fine_tuned_router_invalidates_the_detector():
    """Un fine-tune que toca un router y conserva el model_id invalida la calibración."""
    from vigia.contracts import DetectorMismatchError

    model = tiny("mixtral")
    info = adapters.model_info(model, "cliente/modelo")
    detector = _router_detector(info)
    detector.check_model(info)
    with torch.no_grad():  # fine-tune simulado: un paso chico sobre el router de la última capa
        adapters.find_moe_blocks(model)[-1].router.weight.add_(1e-3)
    tuned = adapters.model_info(model, "cliente/modelo")
    assert tuned.fingerprint()["model_id"] == info.model_id and tuned.router_sha256 != info.router_sha256
    with pytest.raises(DetectorMismatchError, match="router"):
        detector.check_model(tuned)


def test_router_bias_changes_the_hash():
    model = tiny("gpt_oss")  # el router de gpt-oss tiene bias
    before = adapters.model_info(model, "m").router_sha256
    with torch.no_grad():
        adapters.find_moe_blocks(model)[0].router.bias.add_(1e-3)
    assert adapters.model_info(model, "m").router_sha256 != before


def test_detector_without_router_hash_stays_valid():
    """Compatibilidad: un detector calibrado antes de router_sha256 sigue valiendo con el modelo nuevo,
    y un backend que no conoce el hash (proxy vLLM) acepta un detector que sí lo tiene."""
    import dataclasses

    info = adapters.model_info(tiny("mixtral"), "m")
    legacy = _router_detector(dataclasses.replace(info, router_sha256=None))
    assert "router_sha256" not in legacy.model
    legacy.check_model(info)
    _router_detector(info).check_model(dataclasses.replace(info, router_sha256=None))


def test_meta_router_weights_give_unknown_hash():
    model = tiny("mixtral")
    blocks = adapters.find_moe_blocks(model)
    blocks[0].router.weight = torch.nn.Parameter(torch.empty(4, 32, device="meta"))
    assert adapters.router_sha256(blocks) is None
