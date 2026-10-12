"""Extractor de señales con hooks sobre bloques MoE (AT-S01, S02, S04, S05, S06, S11)."""

import math
from types import MethodType

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import transformers as tf
from torch import nn
from transformers import AutoModelForCausalLM

from vigia import features as fx
from vigia.adapters import find_moe_blocks, routing_params
from vigia.contracts import (
    AGGREGATES,
    F0_FEATURES,
    F1_LAYER_FEATURES,
    USAGE_LAYER_FEATURES,
    HookNotFiredError,
    Tier,
    agg_feature,
    layer_feature,
)
from vigia.extractor import SignalExtractor, expert_outputs, f2_unsupported_reason, resolve_probe_layer

BASE = dict(
    vocab_size=256,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
)
MOE = dict(num_experts_per_tok=2)
CONFIGS = {
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
}
# Posición de (logits, pesos, índices) en la tupla que devuelve cada router de transformers 5.
ROUTER_OUTPUT = {"granitemoe": (2, 1, 0)}
ALL_TIERS = (Tier.F0, Tier.F1, Tier.F2, Tier.H)


def tiny(model_type: str) -> nn.Module:
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(CONFIGS[model_type]()).eval()
    if model_type == "deepseek_v3":
        with torch.no_grad():
            for b in find_moe_blocks(model):
                b.router.e_score_correction_bias.copy_(torch.tensor([0.0, 0.3, -0.2, 0.1]))
    return model


def ids_of(n: int, seed: int = 1, batch: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (batch, n), generator=g)


def run_once(model, ids, tiers=(Tier.F0, Tier.F1), targets=None, **kw):
    """Una pasada puntuando todas las posiciones; targets por defecto = siguiente token."""
    if targets is None:
        targets = torch.cat([ids[:, 1:], torch.zeros_like(ids[:, :1])], dim=1)
    with torch.inference_mode(), SignalExtractor(model, tiers, **kw) as ex:
        ex.begin_step()
        out = model(input_ids=ids)
        return ex.end_step(out.logits, targets), ex


def assert_same_features(a: dict, b: dict, atol=1e-5):
    assert a.keys() == b.keys()
    va = np.array([a[k] for k in a])
    vb = np.array([b[k] for k in a])
    assert np.allclose(va, vb, atol=atol, rtol=1e-5, equal_nan=True)


@pytest.mark.parametrize("model_type", sorted(CONFIGS))
def test_at_s01_recomputed_logits_match_router(model_type):
    """AT-S01: logits del pre-hook allclose a los del router y top-k reconstruido igual al del modelo."""
    model = tiny(model_type)
    blocks = find_moe_blocks(model)
    seen = {}
    li, wi, ii = ROUTER_OUTPUT.get(model_type, (0, 1, 2))
    hooks = [
        b.router.register_forward_hook(lambda m, a, o, layer=b.layer: seen.__setitem__(layer, o))
        for b in blocks
    ]
    try:
        ids = ids_of(7)
        with torch.inference_mode(), SignalExtractor(model, (Tier.F1,), keep_router_logits=True) as ex:
            ex.begin_step()
            model(input_ids=ids)
            recomputed = dict(ex.router_logits)
            signals = ex.end_step()[0]
    finally:
        for h in hooks:
            h.remove()
    for b in blocks:
        out = seen[b.layer]
        ref_logits, ref_w, ref_idx = out[li].float(), out[wi].float(), out[ii]
        assert torch.allclose(recomputed[b.layer].reshape(-1, 4), ref_logits.reshape(-1, 4), atol=1e-5)
        li_moe = [blk.layer for blk in blocks].index(b.layer)
        idx = np.array([s.expert_ids[li_moe] for s in signals])  # selección pública por token
        z = recomputed[b.layer].reshape(-1, 4).numpy()
        stats = fx.router_stats(z, b.spec.gating, 2, **routing_params(b))
        assert np.array_equal(stats["idx"], idx)
        w = stats["w"]
        ref_idx_np, ref_w_np = ref_idx.reshape(-1, 2).numpy(), ref_w.reshape(-1, 2).numpy()
        assert np.array_equal(np.sort(idx, -1), np.sort(ref_idx_np, -1))
        order, ref_order = np.argsort(idx, -1), np.argsort(ref_idx_np, -1)
        assert np.allclose(
            np.take_along_axis(w, order, -1), np.take_along_axis(ref_w_np, ref_order, -1), atol=1e-5
        )


def test_at_s02_prehook_captures_when_block_skips_router():
    """AT-S02: con un MLP parcheado que no llama al router (MXFP4 simulado) el pre-hook captura igual."""
    model = tiny("gpt_oss")
    blocks = find_moe_blocks(model)

    def mxfp4_forward(self, hidden_states):
        b, t, h = hidden_states.shape
        x = hidden_states.reshape(-1, h)
        logits = F.linear(x, self.router.weight, self.router.bias)
        top, idx = torch.topk(logits, self.router.top_k, dim=-1)
        w = torch.softmax(top, dim=1)
        return self.experts(x, idx, w).reshape(b, t, h), w

    calls, inputs = [], {}
    handles = []
    for blk in blocks:
        blk.block.forward = MethodType(mxfp4_forward, blk.block)
        handles.append(blk.router.register_forward_hook(lambda *a: calls.append(1)))
        handles.append(
            blk.block.register_forward_pre_hook(
                lambda m, a, layer=blk.layer: inputs.__setitem__(layer, a[0].detach().clone())
            )
        )
    try:
        with torch.inference_mode(), SignalExtractor(model, (Tier.F1,), keep_router_logits=True) as ex:
            ex.begin_step()
            model(input_ids=ids_of(6))
            signals = ex.end_step()
            captured = dict(ex.router_logits)
    finally:
        for h in handles:
            h.remove()
    assert calls == []  # el router nunca corrió
    for blk in blocks:
        with torch.no_grad():
            expected = F.linear(inputs[blk.layer], blk.router.weight, blk.router.bias)
        assert torch.allclose(captured[blk.layer], expected, atol=1e-6)
    assert len(signals[0]) == 6 and not math.isnan(signals[0][3].features["r_ent_n@1"])


def test_feature_names_and_expert_ids():
    model = tiny("mixtral")
    signals, ex = run_once(model, ids_of(5))
    feats = signals[0][2].features
    expected = set(F0_FEATURES)
    for base in F1_LAYER_FEATURES + USAGE_LAYER_FEATURES:
        expected |= {layer_feature(base, layer) for layer in (0, 1)}
        expected |= {agg_feature(agg, base) for agg in AGGREGATES}
    assert set(feats) == expected
    assert math.isnan(signals[0][0].features["r_switch@0"])  # sin posición previa
    assert not math.isnan(signals[0][1].features["r_switch@0"])
    ids = signals[0][2].expert_ids
    assert len(ids) == 2 and all(len(row) == 2 for row in ids)
    assert ex.layers == (0, 1)


def test_u_tier_uses_unweighted_usage():
    model = tiny("olmoe")
    signals, _ = run_once(model, ids_of(4), tiers=(Tier.U,))
    feats = signals[0][3].features
    assert "r_ent_n@0" not in feats and "out_entropy" not in feats
    # Con 4 tokens y k=2 sin ponderar, el uso son conteos enteros: Gini = 1 − Σ (c/8)².
    counts = np.zeros(4)
    for s in signals[0]:
        for e in s.expert_ids[0]:
            counts[e] += 1
    u = counts / counts.sum()
    assert feats["u_gini@0"] == pytest.approx(1 - (u * u).sum())


def test_at_s04_features_are_causal():
    """AT-S04: agregar tokens al final no cambia las features de los primeros t tokens."""
    model = tiny("mixtral")
    long_ids = ids_of(12)
    short_ids = long_ids[:, :8]
    sig_long, _ = run_once(model, long_ids, tiers=ALL_TIERS, probe_layer=1)
    sig_short, _ = run_once(model, short_ids, tiers=ALL_TIERS, probe_layer=1, targets=long_ids[:, 1:9])
    for a, b in zip(sig_short[0], sig_long[0][:8], strict=False):
        assert_same_features(a.features, b.features)
        assert a.expert_ids == b.expert_ids


def test_at_s04_incremental_kv_matches_single_pass():
    """AT-S04: decodificar paso a paso con cache KV da las mismas features que una sola pasada."""
    model = tiny("qwen3_moe")
    ids = ids_of(9)
    targets = torch.cat([ids[:, 1:], torch.zeros_like(ids[:, :1])], dim=1)
    full, _ = run_once(model, ids, tiers=ALL_TIERS, probe_layer=1)
    steps = []
    with torch.inference_mode(), SignalExtractor(model, ALL_TIERS, probe_layer=1) as ex:
        ex.begin_step()
        out = model(input_ids=ids[:, :4], use_cache=True)
        steps += ex.end_step(out.logits, targets[:, :4])[0]
        for p in range(4, 9):
            ex.begin_step()
            out = model(input_ids=ids[:, p : p + 1], past_key_values=out.past_key_values, use_cache=True)
            steps += ex.end_step(out.logits, targets[:, p : p + 1])[0]
    assert len(steps) == 9
    for a, b in zip(steps, full[0], strict=True):
        assert_same_features(a.features, b.features, atol=1e-4)


def test_at_s05_left_padding_generates_no_features():
    """AT-S05: con padding a izquierda, las posiciones de padding no generan features."""
    model = tiny("mixtral")
    row0 = ids_of(6, seed=2)
    row1 = ids_of(4, seed=3)
    batch = torch.cat([row0, torch.cat([torch.zeros(1, 2, dtype=torch.long), row1], 1)], 0)
    mask = torch.tensor([[1] * 6, [0, 0, 1, 1, 1, 1]])
    targets = torch.cat([batch[:, 1:], torch.zeros(2, 1, dtype=torch.long)], 1)
    with torch.inference_mode(), SignalExtractor(model, ALL_TIERS, probe_layer=1) as ex:
        ex.begin_step(attention_mask=mask)
        out = model(input_ids=batch, attention_mask=mask)
        batched = ex.end_step(out.logits, targets)
    assert [len(r) for r in batched] == [6, 4]
    alone0, _ = run_once(model, row0, tiers=ALL_TIERS, probe_layer=1, targets=targets[:1])
    alone1, _ = run_once(model, row1, tiers=ALL_TIERS, probe_layer=1, targets=targets[1:, 2:])
    for got, ref in ((batched[0], alone0[0]), (batched[1], alone1[0])):
        for a, b in zip(got, ref, strict=True):
            assert_same_features(a.features, b.features, atol=1e-4)


def _hook_count(model: nn.Module) -> int:
    return sum(len(m._forward_pre_hooks) + len(m._forward_hooks) for m in model.modules())


def test_at_s06_hooks_removed_even_on_exception():
    """AT-S06: al salir del context manager, aun con excepción, no queda ningún hook."""
    model = tiny("gpt_oss")
    before = _hook_count(model)
    with SignalExtractor(model, ALL_TIERS):
        assert _hook_count(model) > before
    assert _hook_count(model) == before
    with pytest.raises(RuntimeError, match="boom"), SignalExtractor(model, ALL_TIERS):
        raise RuntimeError("boom")
    assert _hook_count(model) == before


def test_hook_not_fired_raises():
    model = tiny("mixtral")
    with SignalExtractor(model, (Tier.F1,)) as ex:
        ex.begin_step()
        with pytest.raises(HookNotFiredError):
            ex.end_step()
    layer = model.model.layers[1]
    layer.forward = MethodType(lambda self, hidden_states, *a, **k: hidden_states, layer)
    with torch.inference_mode(), SignalExtractor(model, (Tier.F1,)) as ex:
        ex.begin_step()
        model(input_ids=ids_of(3))
        with pytest.raises(HookNotFiredError, match=r"\[1\]"):
            ex.end_step()


@pytest.mark.parametrize("model_type", ["mixtral", "gpt_oss", "qwen2_moe", "deepseek_v3", "granitemoe"])
def test_expert_outputs_reproduce_experts_module(model_type):
    """La suma ponderada de las salidas recalculadas reproduce la salida del módulo de expertos."""
    model = tiny(model_type)
    blk = find_moe_blocks(model)[0]
    x = torch.randn(5, 32)
    idx = torch.tensor([[0, 1], [2, 3], [1, 3], [0, 2], [3, 0]])
    w = torch.rand(5, 2)
    with torch.inference_mode():
        ref = blk.experts(x, idx, w)
        e = expert_outputs(blk.experts, x, idx)
    assert torch.allclose((w[..., None] * e).sum(1), ref, atol=1e-5)


@pytest.mark.parametrize("impl", ["eager", "grouped_mm"])
def test_at_s11_f2_does_not_change_model_output(impl):
    """AT-S11: con F2 activado la salida del modelo es idéntica a la salida sin instrumentar."""
    model = tiny("gpt_oss")
    model.set_experts_implementation(impl)
    ids = ids_of(6)
    with torch.inference_mode():
        plain = model(input_ids=ids).logits
        with SignalExtractor(model, ALL_TIERS) as ex:
            ex.begin_step()
            hooked = model(input_ids=ids).logits
            signals = ex.end_step(hooked, torch.zeros_like(ids))
    assert torch.equal(plain, hooked)
    feats = signals[0][-1].features
    assert -1.0 <= feats["e_cos@1"] <= 1.0 and feats["e_disp@0"] >= 0.0
    assert 0.0 <= feats["e_jsd"] <= 1.0


def test_f2_restricted_to_scored_positions():
    model = tiny("mixtral")
    ids = ids_of(5)
    score = torch.zeros(1, 5, dtype=torch.bool)
    score[0, -1] = True
    with torch.inference_mode(), SignalExtractor(model, ALL_TIERS, probe_layer=1) as ex:
        ex.begin_step(score_mask=score)
        out = model(input_ids=ids, logits_to_keep=1)
        last = ex.end_step(out.logits, torch.zeros(1, 1, dtype=torch.long))
    full, _ = run_once(
        model, ids, tiers=ALL_TIERS, probe_layer=1, targets=torch.zeros(1, 5, dtype=torch.long)
    )
    assert len(last[0]) == 1
    keys = [k for k in last[0][0].features if k.startswith(("e_", "h_", "out_"))]
    assert_same_features(
        {k: last[0][0].features[k] for k in keys}, {k: full[0][-1].features[k] for k in keys}
    )


class _FusedExperts(nn.Module):
    """Imita un kernel fusionado: delega el forward pero no expone pesos por experto."""

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, *args, **kwargs):
        return self.inner(*args, **kwargs)


def test_f2_disabled_with_fused_experts_without_failing():
    model = tiny("gpt_oss")
    for blk in find_moe_blocks(model):
        blk.block.experts = _FusedExperts(blk.block.experts)
    assert "F2" in f2_unsupported_reason(find_moe_blocks(model))
    signals, ex = run_once(model, ids_of(4), tiers=(Tier.F0, Tier.F1, Tier.F2))
    assert Tier.F2 not in ex.tiers and ex.disabled and ex.disabled[0].startswith("F2")
    assert "e_cos@0" not in signals[0][0].features and "r_ent_n@0" in signals[0][0].features


def test_h_tier_matches_hidden_states():
    model = tiny("mixtral")
    ids = ids_of(4)
    signals, ex = run_once(model, ids, tiers=(Tier.H,), probe_layer=1)
    with torch.inference_mode():
        hs = model(input_ids=ids, output_hidden_states=True).hidden_states[1]
    vec = np.array([signals[0][2].features[f"h_{i}"] for i in range(32)])
    assert np.allclose(vec, hs[0, 2].numpy(), atol=1e-6)
    assert resolve_probe_layer(None, 24) == 23
    with pytest.raises(ValueError):
        resolve_probe_layer(0, 24)


def test_f0_requires_logits_and_batch_size_is_fixed():
    model = tiny("mixtral")
    with torch.inference_mode(), SignalExtractor(model, (Tier.F0, Tier.F1)) as ex:
        ex.begin_step()
        model(input_ids=ids_of(3))
        with pytest.raises(ValueError, match="F0"):
            ex.end_step()
        ex.begin_step()
        out = model(input_ids=ids_of(3, batch=2))
        with pytest.raises(ValueError, match="batch"):
            ex.end_step(out.logits, torch.zeros(2, 3, dtype=torch.long))


def test_hook_only_stores_step_router_logits():
    """El pre-hook guarda los logits (n_tokens, N) float32 del paso; la reducción es de end_step."""
    model = tiny("mixtral")
    with torch.inference_mode(), SignalExtractor(model, (Tier.F1,)) as ex:
        ex.begin_step()
        model(input_ids=ids_of(5))
        stored = dict(ex._step.router)
        assert sorted(stored) == [0, 1]
        for logits in stored.values():
            assert logits.shape == (5, 4) and logits.dtype == torch.float32 and not logits.requires_grad
        ex.end_step()
        assert ex._step is None  # ningún logit sobrevive al paso
        assert ex._counts.shape == (1, 2, 4)


def _reference_router_features(model, ids, scored):
    """Oráculo token a token: stats por capa, uso acumulado y r_switch contra la posición previa."""
    blocks = find_moe_blocks(model)
    with torch.inference_mode(), SignalExtractor(model, (Tier.F1,), keep_router_logits=True) as ex:
        ex.begin_step()
        model(input_ids=ids)
        logits = {layer: t[0].numpy() for layer, t in ex.router_logits.items()}
        ex.end_step()
    stats = [fx.router_stats(logits[b.layer], b.spec.gating, 2, **routing_params(b)) for b in blocks]
    counts = np.zeros((len(blocks), 4))
    prev = None
    out = []
    for p in range(ids.shape[1]):
        idx = np.stack([s["idx"][p] for s in stats])
        if scored[p]:
            w = np.stack([s["w"][p] for s in stats])
            for layer in range(len(blocks)):
                for slot in range(2):
                    counts[layer, idx[layer, slot]] += w[layer, slot]
            feats = {}
            for li, b in enumerate(blocks):
                u = counts[li] / counts[li].sum()
                feats[layer_feature("u_gini", b.layer)] = 1 - (u * u).sum()
                feats[layer_feature("u_ihhi", b.layer)] = 1 / (u * u).sum() / 4
                nz = u[u > 0]
                feats[layer_feature("u_ent", b.layer)] = -(nz * np.log(nz)).sum() / np.log(4)
                shared = 0 if prev is None else len(set(idx[li]) & set(prev[li]))
                feats[layer_feature("r_switch", b.layer)] = math.nan if prev is None else 1 - shared / 2
                for name in F1_LAYER_FEATURES:
                    feats[layer_feature(name, b.layer)] = stats[li][name][p]
            out.append((feats, idx.tolist()))
        prev = idx
    return out


@pytest.mark.parametrize("model_type", ["mixtral", "deepseek_v3"])
def test_vectorized_reduction_matches_token_by_token_oracle(model_type):
    """end_step vectorizado = oráculo token a token, con huecos en la máscara y dos pasos con KV."""
    model = tiny(model_type)
    if model_type == "deepseek_v3":  # sesgos distintos por capa: ruteo no compartido
        with torch.no_grad():
            for i, b in enumerate(find_moe_blocks(model)):
                b.router.e_score_correction_bias.copy_(torch.tensor([0.0, 0.3, -0.2, 0.1]) * (i + 1))
    ids = ids_of(8, seed=5)
    scored = np.array([0, 1, 1, 0, 1, 0, 1, 1], dtype=bool)
    expected = _reference_router_features(model, ids, scored)
    got = []
    with torch.inference_mode(), SignalExtractor(model, (Tier.F1,)) as ex:
        assert ex._shared_route is (model_type == "mixtral")
        ex.begin_step(score_mask=torch.as_tensor(scored[None, :5]))
        out = model(input_ids=ids[:, :5], use_cache=True)
        got += ex.end_step()[0]
        ex.begin_step(score_mask=torch.as_tensor(scored[None, 5:]))
        model(input_ids=ids[:, 5:], past_key_values=out.past_key_values, use_cache=True)
        got += ex.end_step()[0]
    assert len(got) == len(expected) == scored.sum()
    for sig, (feats, ids_ref) in zip(got, expected, strict=True):
        assert sig.expert_ids == ids_ref
        assert_same_features({k: sig.features[k] for k in feats}, feats, atol=1e-5)
        for base in USAGE_LAYER_FEATURES + F1_LAYER_FEATURES:
            vals = np.array([sig.features[layer_feature(base, layer)] for layer in (0, 1)])
            agg = fx.layer_aggregates(vals)
            for name in AGGREGATES:
                a, b = sig.features[agg_feature(name, base)], float(agg[name])
                assert (math.isnan(a) and math.isnan(b)) or a == pytest.approx(b, abs=1e-12)


def test_output_features_on_device_match_numpy_reference():
    """F0 se calcula en el dispositivo de los logits; tiene que coincidir con `features.output_features`."""
    import numpy as np
    import torch

    from vigia import features as fx
    from vigia.extractor import SignalExtractor

    rng = np.random.default_rng(0)
    for vocab in (2, 25, 5000):
        z = rng.normal(0, 4, (7, vocab)).astype(np.float32)
        tgt = rng.integers(0, vocab, 7)
        got = SignalExtractor._output_features(torch.from_numpy(z), tgt)
        ref = fx.output_features(z, tgt)
        for k, v in ref.items():
            np.testing.assert_allclose(got[k], v, rtol=1e-9, atol=1e-12, err_msg=k)
