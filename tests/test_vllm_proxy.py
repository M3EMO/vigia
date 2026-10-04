"""Tests del backend de modo reducido contra un vLLM falso (P-15)."""

from __future__ import annotations

import base64
import json
import math

import httpx
import numpy as np
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from vigia.api import RequestContext, create_api_router
from vigia.backends.vllm_proxy import (
    VllmProxyBackend,
    VllmProxyError,
    parse_routed_experts,
    routed_usage_features,
    token_offsets,
    top_logprob_features,
)
from vigia.contracts import Policy, Tier

N_EXPERTS = 4
# 3 tokens, 2 capas MoE, top-2
ROUTED = np.array([[[0, 1], [2, 3]], [[0, 1], [0, 2]], [[2, 3], [0, 2]]], dtype=np.int64)
TOKENS = [
    {"token": "Hola", "logprob": -0.1, "bytes": list(b"Hola")},
    {"token": " ni", "logprob": -1.2, "bytes": list(b" ni")},
    {"token": "ño", "logprob": -2.3, "bytes": list("ño".encode())},
]


def encode(arr: np.ndarray, dtype: str) -> dict:
    np_dtype = {"uint8": "<u1", "uint16": "<u2", "int16": "<i2"}[dtype]
    data = base64.b64encode(arr.astype(np_dtype).tobytes()).decode()
    return {"dtype": dtype, "shape": list(arr.shape), "data": data}


def vllm_response(routed: object, tokens: list[dict] | None = None) -> dict:
    tokens = tokens if tokens is not None else TOKENS
    content = []
    for t in tokens:
        top = [{"token": t["token"], "logprob": t["logprob"]}] + [
            {"token": f"alt{i}", "logprob": t["logprob"] - 1.0 - i} for i in range(19)
        ]
        content.append({**t, "top_logprobs": top})
    choice = {
        "index": 0,
        "message": {"role": "assistant", "content": "".join(t["token"] for t in tokens)},
        "finish_reason": "stop",
        "logprobs": {"content": content},
    }
    if routed is not None:
        choice["routed_experts"] = routed
    return {"id": "x", "object": "chat.completion", "choices": [choice], "usage": {"prompt_tokens": 5}}


def fake_vllm(payload: dict, seen: list[dict], status: int = 200) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(
            {
                "url": str(request.url),
                "auth": request.headers.get("authorization"),
                "body": json.loads(request.content),
            }
        )
        return httpx.Response(status, json=payload)

    return httpx.Client(transport=httpx.MockTransport(handler))


def make_backend(payload: dict, seen: list[dict] | None = None, **kw) -> VllmProxyBackend:
    seen = seen if seen is not None else []
    return VllmProxyBackend(
        "http://vllm.local:8000/v1",
        "granite-moe",
        "sk-test",
        fake_vllm(payload, seen),
        n_experts=N_EXPERTS,
        **kw,
    )


def run(backend: VllmProxyBackend, tiers=(Tier.F0, Tier.U)):
    return list(backend.stream([{"role": "user", "content": "hola"}], max_tokens=8, tiers=tiers))


@pytest.mark.parametrize("fmt", ["uint8", "uint16", "int16", "list"])
def test_AT_A09_reduced_mode_produces_u_features(fmt):
    """AT-A09: con routed_experts uint8/uint16/int16 o listas, el modo reducido produce features U."""
    routed = ROUTED.tolist() if fmt == "list" else encode(ROUTED, fmt)
    seen: list[dict] = []
    backend = make_backend(vllm_response(routed), seen)
    events = run(backend)
    assert len(events) == 3
    body = seen[0]["body"]
    assert body["logprobs"] is True and body["top_logprobs"] == 20
    assert body["return_routed_experts"] is True and body["stream"] is False
    assert seen[0]["url"] == "http://vllm.local:8000/v1/chat/completions"
    assert seen[0]["auth"] == "Bearer sk-test"
    for ev in events:
        for name in ("u_gini@0", "u_ihhi@1", "u_ent@0", "r_switch@1", "agg_mean:u_ent", "agg_max:u_gini"):
            assert name in ev.features
        assert math.isnan(ev.features["out_entropy"])
        assert math.isfinite(ev.features["out_top20_entropy"])
    assert events[0].expert_ids == [[0, 1], [2, 3]]
    assert backend.mode == "reduced"
    caps = backend.capabilities()
    assert caps.tiers == (Tier.F0, Tier.U) and caps.streaming is False


def test_AT_A09_closed_form_usage_values():
    """AT-A09: los valores U coinciden con el cálculo a mano sobre el uso acumulado."""
    feats = routed_usage_features(ROUTED, N_EXPERTS)
    # t=0, capa 0: uso {0:1, 1:1} -> u = [.5,.5,0,0]
    assert feats[0]["u_gini@0"] == pytest.approx(0.5)
    assert feats[0]["u_ihhi@0"] == pytest.approx(2 / 4)
    assert feats[0]["u_ent@0"] == pytest.approx(math.log(2) / math.log(4))
    assert math.isnan(feats[0]["r_switch@0"])
    # t=2, capa 0: conteos [2,2,1,1] -> uniforme parcial
    u = np.array([2, 2, 1, 1]) / 6
    assert feats[2]["u_gini@0"] == pytest.approx(1 - (u**2).sum())
    assert feats[2]["r_switch@0"] == pytest.approx(1.0)  # {0,1} -> {2,3}
    assert feats[1]["r_switch@1"] == pytest.approx(0.5)  # {2,3} -> {0,2}
    assert feats[1]["agg_mean:r_switch"] == pytest.approx(0.25)
    # uso uniforme total: u_ent = 1, u_ihhi = 1
    uniform = routed_usage_features(np.array([[[0, 1]], [[2, 3]]]), N_EXPERTS)
    assert uniform[1]["u_ent@0"] == pytest.approx(1.0)
    assert uniform[1]["u_ihhi@0"] == pytest.approx(1.0)


def test_usage_features_are_causal():
    short = routed_usage_features(ROUTED[:2], N_EXPERTS)
    full = routed_usage_features(ROUTED, N_EXPERTS)
    for t in range(2):
        for k, v in short[t].items():
            assert (math.isnan(v) and math.isnan(full[t][k])) or v == full[t][k]


def test_top_logprob_features():
    feats = top_logprob_features(-0.5, [-0.5, -1.5, -3.0])
    p = np.exp([-0.5, -1.5, -3.0])
    q = p / p.sum()
    assert feats["out_nll"] == pytest.approx(0.5)
    assert feats["out_margin"] == pytest.approx(p[0] - p[1])
    assert feats["out_top20_entropy"] == pytest.approx(float(-(q * np.log(q)).sum()))
    empty = top_logprob_features(None, [])
    assert all(math.isnan(v) for v in empty.values())


def test_utf8_split_bytes_offsets_have_no_gaps_or_overlaps():
    n_tilde = "ñ".encode()
    pieces = ["a", "�", "�", "b"]
    raw = [list(b"a"), [n_tilde[0]], [n_tilde[1]], list(b"b")]
    text, offsets = token_offsets(pieces, raw)
    assert text == "añb"
    assert offsets == [(0, 1), (1, 1), (1, 2), (2, 3)]


def test_offsets_and_text_from_backend():
    events = run(make_backend(vllm_response(ROUTED.tolist())))
    assert "".join(e.text for e in events) == "Hola niño"
    assert [(e.start, e.end) for e in events] == [(0, 4), (4, 7), (7, 9)]


def test_routed_experts_one_short_leaves_last_token_without_u():
    events = run(make_backend(vllm_response(ROUTED[:2].tolist())))
    assert math.isfinite(events[1].features["u_gini@0"])
    assert math.isnan(events[2].features["u_gini@0"]) and events[2].expert_ids is None


@pytest.mark.parametrize(
    "raw",
    [
        {"dtype": "float32", "shape": [3, 2, 2], "data": ""},
        {"dtype": "uint8", "shape": [3, 2, 2], "data": base64.b64encode(b"\x00" * 5).decode()},
        {"dtype": "uint8", "shape": [3, 2], "data": ""},
        {"dtype": "int16", "shape": [1, 1, 1], "data": "!!!"},
        encode(np.array([[[-1, 0]]]), "int16"),
        [[1, 2], [3]],
        "abc",
    ],
)
def test_parse_routed_experts_rejects_invalid(raw):
    with pytest.raises(VllmProxyError):
        parse_routed_experts(raw)


def test_errors_from_vllm():
    with pytest.raises(VllmProxyError, match="routed_experts"):
        run(make_backend(vllm_response(None)))
    no_logprobs = vllm_response(ROUTED.tolist())
    no_logprobs["choices"][0]["logprobs"] = None
    with pytest.raises(VllmProxyError, match="logprobs"):
        run(make_backend(no_logprobs))
    with pytest.raises(VllmProxyError, match="posiciones"):
        run(make_backend(vllm_response([[[0, 1]]])))
    with pytest.raises(VllmProxyError, match="n_experts"):
        run(make_backend(vllm_response((ROUTED + 4).tolist())))
    backend = VllmProxyBackend("http://vllm.local", "m", None, fake_vllm({}, [], status=500), n_experts=4)
    with pytest.raises(VllmProxyError, match="HTTP 500") as exc:
        run(backend)
    assert "sk-" not in str(exc.value)


def test_f0_only_does_not_request_routed_experts():
    seen: list[dict] = []
    events = run(make_backend(vllm_response(None), seen), tiers=(Tier.F0,))
    assert "return_routed_experts" not in seen[0]["body"]
    assert "u_gini@0" not in events[0].features


def test_unknown_n_experts_is_declared():
    backend = VllmProxyBackend("http://vllm.local", "m", None, fake_vllm(vllm_response(ROUTED.tolist()), []))
    assert any("n_experts" in d for d in backend.capabilities().disabled)
    assert backend.model_info().n_experts == 0
    assert len(run(backend)) == 3


def test_invalid_base_url():
    with pytest.raises(ValueError):
        VllmProxyBackend("file:///etc/passwd", "m")


def test_AT_A09_api_marks_reduced_mode():
    """AT-A09: servido por la API, el resultado marca mode="reduced"."""
    backend = make_backend(vllm_response(encode(ROUTED, "uint16")))

    async def get_context(request: Request) -> RequestContext:
        return RequestContext(backend=backend, scorer=None, policy=Policy())

    app = FastAPI()
    app.include_router(create_api_router(get_context))
    resp = TestClient(app).post(
        "/v1/chat/completions",
        json={"model": "granite-moe", "messages": [{"role": "user", "content": "hola"}], "vigia": {}},
    )
    assert resp.status_code == 200
    choice = resp.json()["choices"][0]
    assert choice["message"]["content"] == "Hola niño"
    assert choice["vigia"]["mode"] == "reduced"


def test_reduced_mode_formulas_match_hand_computation():
    """Las features U del modo reducido (vía vigia.features) igualan al cálculo a mano token a token."""
    rng = np.random.default_rng(7)
    t_len, n_layers, n_exp = 7, 3, 6
    routed = np.stack(
        [np.stack([rng.choice(n_exp, 2, replace=False) for _ in range(n_layers)]) for _ in range(t_len)]
    )
    feats = routed_usage_features(routed, n_exp)
    counts = np.zeros((n_layers, n_exp))
    for t in range(t_len):
        per_base = {b: [] for b in ("u_gini", "u_ihhi", "u_ent", "r_switch")}
        for layer in range(n_layers):
            counts[layer, routed[t, layer]] += 1
            u = counts[layer] / counts[layer].sum()
            nz = u[u > 0]
            shared = 0 if t == 0 else len(set(routed[t, layer]) & set(routed[t - 1, layer]))
            ref = {
                "u_gini": 1 - (u * u).sum(),
                "u_ihhi": 1 / (u * u).sum() / n_exp,
                "u_ent": -(nz * np.log(nz)).sum() / np.log(n_exp),
                "r_switch": math.nan if t == 0 else 1 - shared / 2,
            }
            for base, v in ref.items():
                got = feats[t][f"{base}@{layer}"]
                assert (math.isnan(v) and math.isnan(got)) or got == pytest.approx(v, abs=1e-12)
                per_base[base].append(v)
        for base, vals in per_base.items():
            arr = np.array([v for v in vals if not math.isnan(v)])
            for agg, fn in (("mean", np.mean), ("max", np.max), ("std", np.std)):
                got = feats[t][f"agg_{agg}:{base}"]
                assert math.isnan(got) if arr.size == 0 else got == pytest.approx(fn(arr), abs=1e-12)
    assert routed_usage_features(np.zeros((0, 2, 2), dtype=np.int64), 4) == []


def test_top_logprob_features_edge_cases():
    feats = top_logprob_features(-0.2, [-0.2, float("-inf"), -2.0, float("nan")])
    p = np.exp([-0.2, -2.0])
    q = p / p.sum()
    assert feats["out_margin"] == pytest.approx(p[0] - p[1])
    assert feats["out_top20_entropy"] == pytest.approx(float(-(q * np.log(q)).sum()))
    only_inf = top_logprob_features(float("-inf"), [float("-inf")])
    assert all(math.isnan(v) for v in only_inf.values())
    single = top_logprob_features(-0.1, [-0.1])
    assert math.isnan(single["out_margin"]) and single["out_top20_entropy"] == pytest.approx(0.0)
