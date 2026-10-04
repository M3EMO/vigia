"""Tests de la API OpenAI-compatible con backend y detector falsos (P-13, P-14)."""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import threading
import time
from collections.abc import Iterator
from dataclasses import replace
from typing import Any

import anyio
import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict
from starlette.concurrency import run_in_threadpool

from vigia.api import RequestContext, TraceRecord, create_api_router
from vigia.contracts import (
    Capabilities,
    DetectorMismatchError,
    ModelInfo,
    Policy,
    Tier,
    TokenEvent,
)

MODEL = "tiny-moe"
INFO = ModelInfo(
    model_id=MODEL,
    model_type="mixtral",
    n_layers=2,
    moe_layers=(0, 1),
    n_experts=4,
    top_k=2,
    vocab_size=100,
    gating="softmax_topk",
)
ANSWER = ["Belgrano", " nació", " en", " 18", "20", " en", " Rosario", "."]
RISKY = {"18": 0.9, "20": 0.7}  # texto -> p_error


class FakeBackend:
    def __init__(self, pieces: list[str] | None = None, *, fail: bool = False, harmony: bool = False):
        self.pieces = pieces if pieces is not None else ANSWER
        self.fail = fail
        self.harmony = harmony
        self.calls: list[dict[str, Any]] = []

    def model_info(self) -> ModelInfo:
        return INFO

    def capabilities(self) -> Capabilities:
        return Capabilities(tiers=(Tier.F0, Tier.F1, Tier.F2), streaming=True)

    def count_prompt_tokens(self, messages: list[dict[str, str]]) -> int:
        return sum(len(m["content"].split()) for m in messages)

    def _pieces_for(self, messages: list[dict[str, str]]) -> list[str]:
        if messages[0]["role"] == "system" and "contexto recuperado" in messages[0]["content"]:
            return ["Belgrano", " nació", " en", " 1770", "."]
        return self.pieces

    def stream(self, messages, *, max_tokens, temperature=0.0, top_p=1.0, seed=None, stop=None, tiers=()):
        self.calls.append({"messages": messages, "max_tokens": max_tokens, "tiers": tiers})
        if self.fail:
            raise RuntimeError("secreto interno: /home/model/weights")
        idx = 0
        if self.harmony:
            yield TokenEvent(idx, 1, "pienso...", 0, 0, {"out_nll": 1.0}, channel="analysis")
            idx += 1
            yield TokenEvent(idx, 2, "<|message|>", 0, 0, {}, is_special=True)
            idx += 1
        pos = 0
        for piece in self._pieces_for(messages)[:max_tokens]:
            yield TokenEvent(idx, 10 + idx, piece, pos, pos + len(piece), {"out_nll": 0.5}, logprob=-0.25)
            pos += len(piece)
            idx += 1


class FakeScorer:
    id = "abcd1234abcd1234"

    def __init__(self, *, fail_on: str | None = None, response: float | None = None):
        self.fail_on = fail_on
        self.response = response

    def check_model(self, info: ModelInfo) -> None:
        if self.fail_on == "check":
            raise DetectorMismatchError("otro modelo")

    def tau(self, alpha: float) -> float:
        return 0.5

    def score_tokens(self, tokens: list[TokenEvent]) -> list[float | None]:
        if self.fail_on == "score":
            raise RuntimeError("detector roto")
        return [
            None if (t.is_special or t.channel != "final") else RISKY.get(t.text.strip(), 0.1) for t in tokens
        ]

    def score_response(self, ps: list[float | None]) -> float | None:
        if self.response is not None:
            return self.response
        vals = [p for p in ps if p is not None]
        return max(vals) if vals else None


class FakeRetriever:
    def __init__(self, fail: bool = False):
        self.queries: list[str] = []
        self.fail = fail

    def retrieve(self, query: str) -> list[str]:
        self.queries.append(query)
        if self.fail:
            raise RuntimeError("caído")
        return ["Manuel Belgrano nació en 1770 en Buenos Aires."]


def make_client(ctx: RequestContext, **router_kwargs: Any) -> TestClient:
    async def get_context(request: Request) -> RequestContext:
        return ctx

    app = FastAPI()
    app.include_router(create_api_router(get_context, **router_kwargs))
    return TestClient(app)


def make_ctx(**kw: Any) -> tuple[RequestContext, list[TraceRecord]]:
    traces: list[TraceRecord] = []
    base: dict[str, Any] = {
        "backend": FakeBackend(),
        "scorer": FakeScorer(),
        "policy": Policy(),
        "org_id": "org1",
        "key_id": "key1",
        "sink": traces.append,
    }
    base.update(kw)
    return RequestContext(**base), traces


def chat(client: TestClient, **body: Any):
    payload = {"model": MODEL, "messages": [{"role": "user", "content": "¿Dónde nació Belgrano?"}], **body}
    return client.post("/v1/chat/completions", json=payload)


def sse_events(resp) -> list[Any]:
    out: list[Any] = []
    for line in resp.text.splitlines():
        if line.startswith("data: "):
            data = line[len("data: ") :]
            out.append(data if data == "[DONE]" else json.loads(data))
    return out


# --- AT-A01 / AT-A02 ----------------------------------------------------------------------------


def test_AT_A01_without_vigia_response_is_plain_openai():
    """AT-A01: sin `vigia` la respuesta tiene exactamente los campos de OpenAI y la traza se guarda."""
    ctx, traces = make_ctx()
    resp = chat(make_client(ctx))
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"id", "object", "created", "model", "choices", "usage"}
    assert body["id"].startswith("chatcmpl-")
    assert body["object"] == "chat.completion"
    assert isinstance(body["created"], int)
    assert body["model"] == MODEL
    (choice,) = body["choices"]
    assert set(choice) == {"index", "message", "finish_reason", "logprobs"}
    assert choice["message"] == {"role": "assistant", "content": "".join(ANSWER)}
    assert choice["finish_reason"] == "stop"
    assert choice["logprobs"] is None
    assert body["usage"] == {"prompt_tokens": 3, "completion_tokens": 8, "total_tokens": 11}
    trace_id = resp.headers["X-Vigia-Trace-Id"]
    assert len(traces) == 1 and traces[0].trace_id == trace_id
    assert traces[0].org_id == "org1" and traces[0].key_id == "key1"
    assert traces[0].generation.text == "".join(ANSWER)
    assert traces[0].result.action == "human_review"


def test_AT_A02_with_vigia_each_choice_has_scores():
    """AT-A02: con `vigia` cada choice trae tokens, spans, calibration_id y action."""
    ctx, traces = make_ctx()
    resp = chat(make_client(ctx), vigia={})
    assert resp.status_code == 200
    v = resp.json()["choices"][0]["vigia"]
    assert v["trace_id"] == resp.headers["X-Vigia-Trace-Id"]
    assert [t["text"] for t in v["tokens"]] == ANSWER
    assert [t["flag"] for t in v["tokens"]] == [False, False, False, True, True, False, False, False]
    assert v["spans"] == [
        {"start": 18, "end": 22, "token_start": 3, "token_end": 5, "score": 0.9, "text": "1820"}
    ]
    assert v["calibration_id"] == FakeScorer.id
    assert v["calibrated"] is True
    assert v["action"] == "human_review"
    assert v["response_p_error"] == 0.9
    assert v["tau"] == 0.5 and v["alpha"] == 0.1
    assert v["mode"] == "full" and v["degraded"] is False
    assert v["schema_version"] == "1"


def test_vigia_options_tiers_and_alpha_are_forwarded():
    backend = FakeBackend()
    ctx, _ = make_ctx(backend=backend)
    resp = chat(make_client(ctx), vigia={"tiers": ["F0", "F2"], "alpha": 0.2})
    assert resp.json()["choices"][0]["vigia"]["alpha"] == 0.2
    assert backend.calls[0]["tiers"] == (Tier.F0, Tier.F2)
    chat(make_client(ctx))
    assert backend.calls[1]["tiers"] == (Tier.F0, Tier.F1)  # por defecto solo niveles baratos


def test_unsupported_tier_is_422():
    ctx, _ = make_ctx()
    resp = chat(make_client(ctx), vigia={"tiers": ["U"]})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "unsupported_tier"


def test_logprobs_when_requested():
    ctx, _ = make_ctx()
    lp = chat(make_client(ctx), logprobs=True).json()["choices"][0]["logprobs"]
    assert [e["token"] for e in lp["content"]] == ANSWER
    assert lp["content"][0]["logprob"] == -0.25


# --- AT-A03 streaming ---------------------------------------------------------------------------


def test_AT_A03_streaming_chunks_carry_tokens_and_final_summary():
    """AT-A03: cada chunk trae sus tokens, el final trae spans y acción, y cierra con [DONE]."""
    ctx, traces = make_ctx()
    resp = chat(make_client(ctx), stream=True, vigia={}, stream_options={"include_usage": True})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = sse_events(resp)
    assert events[-1] == "[DONE]"
    chunks = events[:-1]
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    token_chunks = chunks[1:-2]
    assert "".join(c["choices"][0]["delta"]["content"] for c in token_chunks) == "".join(ANSWER)
    indexes = [c["choices"][0]["vigia"]["tokens"][0]["index"] for c in token_chunks]
    assert indexes == list(range(len(ANSWER)))
    assert token_chunks[3]["choices"][0]["vigia"]["tokens"][0]["p_error"] == 0.9
    final = chunks[-2]["choices"][0]
    assert final["finish_reason"] == "stop"
    assert final["vigia"]["spans"][0]["text"] == "1820"
    assert final["vigia"]["action"] == "human_review"
    assert final["vigia"]["response_p_error"] == 0.9
    assert chunks[-1]["choices"] == [] and chunks[-1]["usage"]["completion_tokens"] == 8
    assert len(traces) == 1 and traces[0].trace_id == resp.headers["X-Vigia-Trace-Id"]


def test_AT_A03_streaming_without_vigia_has_no_extra_fields():
    """AT-A03 / AT-A01: sin `vigia` los chunks no llevan campos extra."""
    ctx, traces = make_ctx()
    events = sse_events(chat(make_client(ctx), stream=True))
    assert events[-1] == "[DONE]"
    for c in events[:-1]:
        assert set(c) == {"id", "object", "created", "model", "choices"}
        assert set(c["choices"][0]) == {"index", "delta", "logprobs", "finish_reason"}
    assert len(traces) == 1


def test_AT_A03_buffer_abstain_never_leaks_original_text():
    """AT-A03: con stream_policy=buffer y abstain el cliente no recibe ningún token original."""
    pol = Policy(abstain_at=0.8, stream_policy="buffer", abstain_message="Sin respuesta confiable.")
    ctx, traces = make_ctx(policy=pol)
    resp = chat(make_client(ctx), stream=True, vigia={})
    assert "Belgrano" not in resp.text and "Rosario" not in resp.text
    events = sse_events(resp)
    contents = [c["choices"][0]["delta"].get("content") for c in events[:-1]]
    assert "Sin respuesta confiable." in contents
    assert events[-2]["choices"][0]["vigia"]["action"] == "abstain"
    assert traces[0].generation.text == "".join(ANSWER)  # el original queda solo en la traza


def test_streaming_text_completion():
    ctx, _ = make_ctx()
    client = make_client(ctx)
    resp = client.post("/v1/completions", json={"model": MODEL, "prompt": "Belgrano", "stream": True})
    events = sse_events(resp)
    assert events[-1] == "[DONE]"
    assert all(c["object"] == "text_completion" for c in events[:-1])
    assert "".join(c["choices"][0]["text"] for c in events[:-1]) == "".join(ANSWER)


# --- AT-A04 cliente estilo SDK ------------------------------------------------------------------


class _SdkModel(BaseModel):
    model_config = ConfigDict(extra="allow")  # el SDK oficial usa extra="allow"


class _SdkMessage(_SdkModel):
    role: str
    content: str | None


class _SdkChoice(_SdkModel):
    index: int
    message: _SdkMessage
    finish_reason: str
    logprobs: Any = None


class _SdkUsage(_SdkModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class _SdkCompletion(_SdkModel):
    id: str
    object: str
    created: int
    model: str
    choices: list[_SdkChoice]
    usage: _SdkUsage


def test_AT_A04_sdk_like_client_exposes_model_extra_vigia():
    """AT-A04: un cliente que parsea como el SDK (extra="allow") expone model_extra["vigia"]."""
    ctx, _ = make_ctx()
    client = make_client(ctx)
    extra_body = {"vigia": {"alpha": 0.1}}
    body = {"model": MODEL, "messages": [{"role": "user", "content": "hola"}], **extra_body}
    resp = client.post(
        "/v1/chat/completions",
        json=body,
        headers={"Authorization": "Bearer vg_live_x", "User-Agent": "OpenAI/Python 2.0"},
    )
    parsed = _SdkCompletion.model_validate(resp.json())
    assert parsed.object == "chat.completion"
    assert parsed.choices[0].message.content == "".join(ANSWER)
    vigia = parsed.choices[0].model_extra["vigia"]
    assert vigia["action"] == "human_review" and vigia["spans"][0]["text"] == "1820"
    plain = _SdkCompletion.model_validate(chat(client).json())
    assert plain.choices[0].model_extra == {}


def test_AT_A04_official_openai_sdk():
    """AT-A04: el SDK oficial `openai` con base_url local parsea y expone model_extra["vigia"]."""
    openai = pytest.importorskip("openai")
    ctx, _ = make_ctx()
    sdk = openai.OpenAI(base_url="http://testserver/v1", api_key="vg_live_x", http_client=make_client(ctx))
    out = sdk.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": "hola"}], extra_body={"vigia": {}}
    )
    assert out.choices[0].model_extra["vigia"]["action"] == "human_review"


# --- AT-A06 / AT-A07 políticas ------------------------------------------------------------------


def test_AT_A06_abstain_replaces_content_and_keeps_original_in_trace():
    """AT-A06: abstain gana sobre human_review; el cliente recibe el mensaje de abstención."""
    pol = Policy(abstain_at=0.85, human_review_at=0.5, abstain_message="No tengo una respuesta confiable.")
    ctx, traces = make_ctx(policy=pol)
    body = chat(make_client(ctx), vigia={}).json()
    choice = body["choices"][0]
    assert choice["message"]["content"] == "No tengo una respuesta confiable."
    assert choice["vigia"]["action"] == "abstain"
    assert choice["vigia"]["tokens"] == [] and choice["vigia"]["spans"] == []
    assert "Belgrano" not in json.dumps(body)
    assert traces[0].generation.text == "".join(ANSWER)
    assert traces[0].result.tokens  # la traza conserva el detalle


def test_AT_A06_annotate_when_below_review_threshold():
    """AT-A06: debajo de los umbrales de respuesta, con fragmentos, la acción es annotate."""
    ctx, _ = make_ctx(policy=Policy(human_review_at=0.95))
    assert chat(make_client(ctx), vigia={}).json()["choices"][0]["vigia"]["action"] == "annotate"
    ctx, _ = make_ctx(policy=Policy(human_review_at=0.95, annotate=False))
    assert chat(make_client(ctx), vigia={}).json()["choices"][0]["vigia"]["action"] == "pass"


def test_AT_A07_rag_retry_requeries_once_with_context():
    """AT-A07: rag_retry consulta al retriever una vez y regenera con el contexto recuperado."""
    backend = FakeBackend()
    retriever = FakeRetriever()
    pol = Policy(rag_retry_at=0.8, rag_url="https://rag.example.com/q", rag_max_retries=1)
    ctx, traces = make_ctx(backend=backend, policy=pol, retriever=retriever)
    body = chat(make_client(ctx), vigia={}).json()
    assert retriever.queries == ["¿Dónde nació Belgrano?\nBelgrano nació en ___ en Rosario."]
    assert len(backend.calls) == 2
    sys_msg = backend.calls[1]["messages"][0]
    assert sys_msg["role"] == "system" and "1770" in sys_msg["content"]
    choice = body["choices"][0]
    assert choice["message"]["content"] == "Belgrano nació en 1770."
    assert choice["vigia"]["action"] == "pass"
    assert traces[0].rag_status == "retried"
    assert traces[0].prompt_messages == [{"role": "user", "content": "¿Dónde nació Belgrano?"}]


def test_AT_A07_rag_retry_fallback_after_max_retries_and_on_error():
    """AT-A07: si el reintento sigue sobre el umbral o el retriever falla, se aplica el respaldo."""

    class AlwaysBad(FakeBackend):
        def _pieces_for(self, messages):
            return ["Dato", " 18"]

    always_bad = AlwaysBad()
    retriever = FakeRetriever()
    pol = Policy(rag_retry_at=0.8, rag_url="https://rag.example.com/q", rag_max_retries=1)
    ctx, traces = make_ctx(backend=always_bad, policy=pol, retriever=retriever)
    v = chat(make_client(ctx), vigia={}).json()["choices"][0]["vigia"]
    assert len(retriever.queries) == 1 and len(always_bad.calls) == 2
    assert v["action"] == "human_review"

    failing = FakeRetriever(fail=True)
    ctx, traces = make_ctx(policy=pol, retriever=failing)
    v = chat(make_client(ctx), vigia={}).json()["choices"][0]["vigia"]
    assert v["action"] == "human_review" and traces[0].rag_status == "error"

    ctx, traces = make_ctx(policy=pol, retriever=None)
    v = chat(make_client(ctx), vigia={}).json()["choices"][0]["vigia"]
    assert v["action"] == "human_review" and traces[0].rag_status == "unavailable"


# --- AT-A08 fallas del detector -----------------------------------------------------------------


@pytest.mark.parametrize("fail_on", ["score", "check"])
def test_AT_A08_fail_closed_returns_503(fail_on):
    """AT-A08: con fail_closed una excepción del detector devuelve 503 en formato OpenAI."""
    ctx, _ = make_ctx(scorer=FakeScorer(fail_on=fail_on), policy=Policy(on_detector_error="fail_closed"))
    client = make_client(ctx)
    for kwargs in ({}, {"stream": True}):
        resp = chat(client, vigia={}, **kwargs)
        assert resp.status_code == 503
        err = resp.json()["error"]
        assert err["code"] == "detector_unavailable"
        assert err["request_id"] and "detector roto" not in resp.text


def test_AT_A08_fail_open_returns_null_scores_and_degraded():
    """AT-A08: con fail_open la respuesta sale con score null y degraded true."""
    ctx, traces = make_ctx(scorer=FakeScorer(fail_on="score"), policy=Policy(on_detector_error="fail_open"))
    resp = chat(make_client(ctx), vigia={})
    assert resp.status_code == 200
    v = resp.json()["choices"][0]["vigia"]
    assert v["degraded"] is True and v["calibrated"] is False
    assert v["response_p_error"] is None and v["action"] == "pass"
    assert all(t["p_error"] is None and t["flag"] is False for t in v["tokens"])
    assert traces[0].result.degraded is True


def test_AT_A08_fail_open_streaming():
    """AT-A08: en streaming con fail_open los chunks traen p_error null y el final degraded true."""
    ctx, _ = make_ctx(scorer=FakeScorer(fail_on="score"), policy=Policy(on_detector_error="fail_open"))
    events = sse_events(chat(make_client(ctx), vigia={}, stream=True))
    assert events[-2]["choices"][0]["vigia"]["degraded"] is True
    assert events[1]["choices"][0]["vigia"]["tokens"][0]["p_error"] is None


def test_without_scorer_scores_are_null_and_action_pass():
    ctx, traces = make_ctx(scorer=None)
    v = chat(make_client(ctx), vigia={}).json()["choices"][0]["vigia"]
    assert v["response_p_error"] is None and v["calibrated"] is False
    assert v["calibration_id"] is None and v["tau"] is None
    assert v["action"] == "pass" and v["spans"] == []
    assert traces[0].result.action == "pass"


# --- Validación y errores -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("body", "param"),
    [
        ({"messages": []}, "messages"),
        ({"messages": [{"role": "hacker", "content": "x"}]}, "messages.0.role"),
        ({"messages": [{"role": "user", "content": "  "}]}, None),
        ({"max_tokens": 0}, "max_tokens"),
        ({"max_tokens": 100_000}, "max_tokens"),
        ({"temperature": 3}, "temperature"),
        ({"top_p": 0}, "top_p"),
        ({"n": 2}, "n"),
        ({"stop": ["a", "b", "c", "d", "e"]}, "stop"),
        ({"vigia": {"alpha": 1.5}}, "vigia.alpha"),
        ({"vigia": {"unknown": 1}}, "vigia.unknown"),
    ],
)
def test_input_validation_is_422_openai_style(body, param):
    ctx, traces = make_ctx()
    payload = {"model": MODEL, "messages": [{"role": "user", "content": "hola"}], **body}
    resp = make_client(ctx, max_tokens_limit=1024).post("/v1/chat/completions", json=payload)
    assert resp.status_code == 422
    err = resp.json()["error"]
    assert set(err) == {"message", "type", "code", "param", "request_id"}
    assert err["type"] == "invalid_request_error"
    if param:
        assert err["param"] == param
    assert resp.headers["X-Request-Id"] == err["request_id"]
    assert traces == []


def test_invalid_json_and_unknown_model():
    ctx, _ = make_ctx()
    client = make_client(ctx)
    resp = client.post(
        "/v1/chat/completions", content=b"{no json", headers={"Content-Type": "application/json"}
    )
    assert resp.status_code == 400 and resp.json()["error"]["code"] == "invalid_json"
    resp = client.post("/v1/chat/completions", json=[1, 2])
    assert resp.status_code == 400
    resp = chat(client, model="gpt-4")
    assert resp.status_code == 404 and resp.json()["error"]["code"] == "model_not_found"


def test_internal_error_is_generic_without_stack_trace():
    ctx, _ = make_ctx(backend=FakeBackend(fail=True))
    resp = chat(make_client(ctx))
    assert resp.status_code == 500
    assert resp.json()["error"]["code"] == "internal_error"
    assert "secreto" not in resp.text and "Traceback" not in resp.text


def test_request_id_header_is_propagated():
    ctx, _ = make_ctx()
    client = make_client(ctx)
    resp = client.post(
        "/v1/chat/completions", json={"model": MODEL, "messages": []}, headers={"X-Request-Id": "abc-123"}
    )
    assert resp.json()["error"]["request_id"] == "abc-123"


def test_max_tokens_length_finish_reason_and_completion_tokens_alias():
    ctx, _ = make_ctx()
    body = chat(make_client(ctx), max_completion_tokens=3).json()
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["choices"][0]["message"]["content"] == "Belgrano nació en"


def test_harmony_analysis_and_special_tokens_are_hidden():
    ctx, traces = make_ctx(backend=FakeBackend(harmony=True))
    body = chat(make_client(ctx), vigia={}).json()
    choice = body["choices"][0]
    assert choice["message"]["content"] == "".join(ANSWER)
    assert "pienso" not in json.dumps(body)
    assert [t["index"] for t in choice["vigia"]["tokens"]] == list(range(2, 2 + len(ANSWER)))


# --- Completions, modelos, trazas ---------------------------------------------------------------


def test_completions_endpoint():
    ctx, _ = make_ctx()
    client = make_client(ctx)
    body = client.post("/v1/completions", json={"model": MODEL, "prompt": "Belgrano", "vigia": {}}).json()
    assert body["id"].startswith("cmpl-") and body["object"] == "text_completion"
    assert body["choices"][0]["text"] == "".join(ANSWER)
    assert body["choices"][0]["vigia"]["action"] == "human_review"
    assert client.post("/v1/completions", json={"model": MODEL, "prompt": ["a", "b"]}).status_code == 422


def test_models_endpoint():
    ctx, _ = make_ctx()
    body = make_client(ctx).get("/v1/models").json()
    assert body["object"] == "list"
    assert body["data"][0]["id"] == MODEL and body["data"][0]["object"] == "model"


def test_trace_lookup_endpoint():
    store: dict[str, dict] = {}
    ctx, traces = make_ctx()
    ctx = replace(ctx, trace_lookup=store.get)
    client = make_client(ctx)
    resp = chat(client, vigia={})
    trace_id = resp.headers["X-Vigia-Trace-Id"]
    store[trace_id] = {"trace_id": trace_id, "action": traces[0].result.action}
    got = client.get(f"/v1/vigia/traces/{trace_id}")
    assert got.status_code == 200 and got.json()["action"] == "human_review"
    missing = client.get("/v1/vigia/traces/" + "0" * 32)
    assert missing.status_code == 404 and missing.json()["error"]["code"] == "trace_not_found"
    assert client.get("/v1/vigia/traces/..%2F..%2Fetc").status_code == 404


def test_sink_failure_does_not_break_response():
    def broken(_: TraceRecord) -> None:
        raise OSError("disco lleno")

    ctx, _ = make_ctx(sink=broken)
    assert chat(make_client(ctx)).status_code == 200


def test_stream_iterator_closed_on_completion():
    closed: list[bool] = []

    class ClosingBackend(FakeBackend):
        def stream(self, messages, **kw) -> Iterator[TokenEvent]:
            try:
                yield from super().stream(messages, **kw)
            finally:
                closed.append(True)

    ctx, _ = make_ctx(backend=ClosingBackend())
    sse_events(chat(make_client(ctx), stream=True))
    assert closed == [True]


# --- Regresiones de la verificación api-edge (ronda 1) ------------------------------------------


class LockingBackend(FakeBackend):
    """Imita a HFBackend: un threading.Lock tomado durante toda la generación."""

    def __init__(self, n: int = 300, delay: float = 0.0) -> None:
        super().__init__([f" t{i}" for i in range(n)])
        self.lock = threading.Lock()
        self.delay = delay

    def stream(self, messages, **kw):
        return self._locked(super().stream(messages, **kw))

    def _locked(self, inner):
        with self.lock:
            for ev in inner:
                if self.delay:
                    time.sleep(self.delay)
                yield ev


class FinishBackend(FakeBackend):
    """Backend que, como GenerationStream, informa su propio finish_reason al agotarse."""

    def __init__(self, pieces: list[str], finish: str) -> None:
        super().__init__(pieces)
        self.finish = finish

    def stream(self, messages, **kw):
        backend = self

        class _Stream:
            finish_reason = None

            def __init__(self) -> None:
                self._it = FakeBackend.stream(backend, messages, **kw)

            def __iter__(self):
                return self

            def __next__(self):
                try:
                    return next(self._it)
                except StopIteration:
                    self.finish_reason = backend.finish
                    raise

            def close(self) -> None:
                self._it.close()

        return _Stream()


class FeatureScorer(FakeScorer):
    """Detector con feature_names y alphas calibrados, como vigia.detector.Detector."""

    feature_names = ("agg_mean:r_ent_n", "out_entropy")
    thresholds = [{"alpha": 0.05, "tau": 0.6}, {"alpha": 0.3, "tau": 0.3}]


def _asgi_app(ctx: RequestContext, **router_kwargs: Any) -> FastAPI:
    async def get_context(request: Request) -> RequestContext:
        return ctx

    app = FastAPI()
    app.include_router(create_api_router(get_context, **router_kwargs))
    return app


async def _stream_then_disconnect(app: FastAPI, body: dict[str, Any], n_chunks: int) -> int:
    """Llama la app por ASGI, lee `n_chunks` chunks del body y simula que el cliente cierra."""
    raw = json.dumps(body).encode()
    disconnect = asyncio.Event()
    state = {"sent": False, "chunks": 0}

    async def receive() -> dict[str, Any]:
        if not state["sent"]:
            state["sent"] = True
            return {"type": "http.request", "body": raw, "more_body": False}
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        if message["type"] == "http.response.body" and message.get("body"):
            state["chunks"] += 1
            if state["chunks"] >= n_chunks:
                disconnect.set()

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "root_path": "",
        "query_string": b"",
        "headers": [(b"content-type", b"application/json"), (b"host", b"testserver")],
        "client": ("127.0.0.1", 1234),
        "server": ("testserver", 80),
    }
    await asyncio.wait_for(app(scope, receive, send), 10)
    return state["chunks"]


def test_client_disconnect_closes_stream_and_releases_backend_lock():
    """Un cliente que corta un stream deja tomado el lock del backend y bloquea a todos los tenants
    hasta un GC no determinista: al desconectarse, la generación se cierra y el lock se libera ya."""
    backend = LockingBackend(n=300, delay=0.002)
    ctx, traces = make_ctx(backend=backend)
    app = _asgi_app(ctx)
    body = {"model": MODEL, "messages": [{"role": "user", "content": "hola"}], "stream": True, "vigia": {}}
    gc.disable()  # el arreglo no puede depender del GC cíclico
    try:
        chunks = asyncio.run(_stream_then_disconnect(app, body, n_chunks=5))
        assert chunks < 300
        assert not backend.lock.locked()
        assert len(traces) == 1 and traces[0].generation.finish_reason == "error"
    finally:
        gc.enable()
    assert chat(make_client(ctx), max_tokens=2).status_code == 200  # el backend sigue atendiendo


def test_many_concurrent_streams_do_not_exhaust_threadpool():
    """40 o más streams concurrentes agotan el threadpool y dejan el proceso en deadlock permanente
    (también /readyz): las generaciones en exceso esperan en el event loop, no en un hilo."""
    backend = LockingBackend(n=6, delay=0.005)
    ctx, traces = make_ctx(backend=backend)
    app = _asgi_app(ctx)

    @app.get("/readyz")
    async def readyz():
        return await run_in_threadpool(lambda: {"ok": True})

    async def scenario() -> tuple[list[int], int]:
        # pool chico: con el bug, 3 hilos esperando el lock alcanzan para el deadlock
        anyio.to_thread.current_default_thread_limiter().total_tokens = 3
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            payload = {
                "model": MODEL,
                "messages": [{"role": "user", "content": "hola"}],
                "stream": True,
                "max_tokens": 6,
            }
            reqs = [client.post("/v1/chat/completions", json=payload) for _ in range(8)]
            results = await asyncio.wait_for(asyncio.gather(*reqs, client.get("/readyz")), 30)
        return [r.status_code for r in results[:-1]], results[-1].status_code

    codes, ready_code = asyncio.run(scenario())
    assert codes == [200] * 8 and ready_code == 200
    assert len(traces) == 8 and all(t.generation.finish_reason == "length" for t in traces)


def test_generation_queue_full_is_503_with_retry_after():
    """40 o más streams concurrentes agotan el threadpool y dejan el proceso en deadlock permanente:
    con la cola de generación llena, el exceso recibe 503 con Retry-After."""
    backend = LockingBackend(n=20, delay=0.01)
    ctx, _ = make_ctx(backend=backend)
    app = _asgi_app(ctx, max_queued_generations=0)

    async def scenario() -> list[httpx.Response]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            payload = {"model": MODEL, "messages": [{"role": "user", "content": "hola"}]}
            reqs = [client.post("/v1/chat/completions", json=payload) for _ in range(3)]
            return await asyncio.wait_for(asyncio.gather(*reqs), 30)

    resps = asyncio.run(scenario())
    codes = sorted(r.status_code for r in resps)
    assert codes[0] == 200 and 503 in codes
    busy = next(r for r in resps if r.status_code == 503)
    assert busy.json()["error"]["code"] == "server_overloaded"
    assert int(busy.headers["Retry-After"]) >= 1


def test_requested_tiers_cannot_drop_detector_features():
    """vigia.tiers del cliente puede excluir features del detector: p_error se imputa con la media,
    la acción cambia y sigue calibrated=true (bypass de la política)."""
    for requested, expected in (
        (["F0"], (Tier.F0, Tier.F1)),
        (["F2"], (Tier.F2, Tier.F0, Tier.F1)),
        (None, (Tier.F0, Tier.F1)),
    ):
        backend = FakeBackend()
        ctx, _ = make_ctx(backend=backend, scorer=FeatureScorer())
        opts = {} if requested is None else {"tiers": requested}
        assert chat(make_client(ctx), vigia=opts).status_code == 200
        assert backend.calls[0]["tiers"] == expected


@pytest.mark.parametrize("stream", [False, True])
def test_rag_fallback_action_abstain_is_honored(stream):
    """policy.rag_fallback_action se ignora: siempre cae a human_review aunque el tenant eligió abstain."""
    pol = Policy(
        rag_retry_at=0.8,
        rag_url="https://rag.example.com/q",
        human_review_at=None,
        rag_fallback_action="abstain",
        abstain_message="Sin respuesta confiable.",
        stream_policy="buffer",
    )
    for retriever in (FakeRetriever(fail=True), None):
        ctx, traces = make_ctx(policy=pol, retriever=retriever)
        resp = chat(make_client(ctx), vigia={}, stream=stream)
        assert "Rosario" not in resp.text
        if stream:
            final = sse_events(resp)[-2]["choices"][0]
        else:
            final = resp.json()["choices"][0]
            assert final["message"]["content"] == "Sin respuesta confiable."
        assert final["vigia"]["action"] == "abstain"
        assert traces[0].result.action == "abstain"


def test_live_stream_rag_retry_maps_to_fallback():
    """Streaming annotate_only con rag_retry devuelve action 'rag_retry' sin re-consultar y la guarda
    así en la traza: la acción final es el respaldo con rag_status='unavailable'."""
    retriever = FakeRetriever()
    pol = Policy(rag_retry_at=0.5, rag_url="https://rag.example.com/q", human_review_at=None)
    ctx, traces = make_ctx(policy=pol, retriever=retriever)
    final = sse_events(chat(make_client(ctx), vigia={}, stream=True))[-2]["choices"][0]["vigia"]
    assert final["action"] == "human_review" and final["rag_status"] == "unavailable"
    assert traces[0].result.action == "human_review" and traces[0].rag_status == "unavailable"
    assert retriever.queries == []


def test_rag_status_reaches_the_client():
    """vigia.rag_status nunca se completa en la respuesta al cliente (siempre null)."""
    pol = Policy(rag_retry_at=0.8, rag_url="https://rag.example.com/q", rag_max_retries=1)
    cases = ((FakeRetriever(), "retried"), (FakeRetriever(fail=True), "error"), (None, "unavailable"))
    for retriever, expected in cases:
        ctx, traces = make_ctx(policy=pol, retriever=retriever)
        v = chat(make_client(ctx), vigia={}).json()["choices"][0]["vigia"]
        assert v["rag_status"] == expected
        assert traces[0].result.rag_status == expected
    no_retry = replace(pol, rag_max_retries=0)
    ctx, _ = make_ctx(policy=no_retry, retriever=FakeRetriever())
    assert chat(make_client(ctx), vigia={}).json()["choices"][0]["vigia"]["rag_status"] == "unavailable"


class RejectingBackend(FakeBackend):
    """Como HFBackend: valida el prompt al llamar a stream(), antes de generar."""

    def stream(self, messages, **kw):
        if len(messages[-1]["content"]) > 1000:
            raise ValueError("el prompt (1118 tokens) excede el contexto del modelo (1024)")
        return super().stream(messages, **kw)


class TypedRejectingBackend(FakeBackend):
    def stream(self, messages, **kw):
        from vigia.contracts import InvalidRequestError

        raise InvalidRequestError("el prompt excede el contexto del modelo", code="context_length_exceeded")


@pytest.mark.parametrize("stream", [False, True])
def test_backend_rejected_prompt_is_400_not_500(stream):
    """Prompts que el backend rechaza (más largos que el contexto o con surrogates sueltos)
    devuelven 500 internal_error."""
    ctx, traces = make_ctx(backend=RejectingBackend())
    client = make_client(ctx)
    resp = chat(client, messages=[{"role": "user", "content": "a" * 1100}], stream=stream)
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "invalid_request_error"
    assert resp.json()["error"]["code"] == "invalid_prompt"
    flag = "true" if stream else "false"
    raw = '{"model":"tiny-moe","messages":[{"role":"user","content":"\\ud800x"}],"stream":' + flag + "}"
    resp = client.post("/v1/chat/completions", content=raw, headers={"Content-Type": "application/json"})
    assert resp.status_code == 400 and resp.json()["error"]["code"] == "invalid_value"
    ctx, _ = make_ctx(backend=TypedRejectingBackend())
    resp = chat(make_client(ctx), stream=stream)
    assert resp.status_code == 400 and resp.json()["error"]["code"] == "context_length_exceeded"
    assert traces == []


@pytest.mark.parametrize("stream", [False, True])
def test_finish_reason_comes_from_backend(stream):
    """finish_reason es 'stop' cuando la generación se corta por el contexto del modelo."""
    ctx, _ = make_ctx(backend=FinishBackend(ANSWER[:3], "length"))
    resp = chat(make_client(ctx), max_tokens=50, stream=stream)
    if stream:
        fr = sse_events(resp)[-2]["choices"][0]["finish_reason"]
    else:
        fr = resp.json()["choices"][0]["finish_reason"]
    assert fr == "length"
    ctx, _ = make_ctx(backend=FinishBackend(ANSWER, "stop"))
    assert chat(make_client(ctx), max_tokens=8).json()["choices"][0]["finish_reason"] == "stop"


@pytest.mark.parametrize("stream", [False, True])
def test_alpha_below_calibrated_minimum_is_422(stream):
    """Un vigia.alpha por debajo del mínimo calibrado se trata como falla del detector: 503 con
    fail_closed o degradado silencioso con fail_open."""
    for on_error in ("fail_closed", "fail_open"):
        ctx, traces = make_ctx(scorer=FeatureScorer(), policy=Policy(on_detector_error=on_error))
        resp = chat(make_client(ctx), vigia={"alpha": 0.01}, stream=stream)
        assert resp.status_code == 422
        err = resp.json()["error"]
        assert err["param"] == "vigia.alpha" and "0.05" in err["message"]
        assert traces == []
        assert chat(make_client(ctx), vigia={"alpha": 0.05}, stream=stream).status_code == 200


def test_unsupported_logprobs_options_are_422():
    """top_logprobs se acepta pero siempre devuelve []; logprobs en /v1/completions se acepta y
    devuelve null."""
    ctx, _ = make_ctx()
    client = make_client(ctx)
    resp = chat(client, logprobs=True, top_logprobs=2)
    assert resp.status_code == 422 and resp.json()["error"]["param"] == "top_logprobs"
    assert resp.json()["error"]["code"] == "unsupported_value"
    assert chat(client, logprobs=True, top_logprobs=0).status_code == 200
    resp = client.post("/v1/completions", json={"model": MODEL, "prompt": "Belgrano", "logprobs": 2})
    assert resp.status_code == 422 and resp.json()["error"]["param"] == "logprobs"


# --- Regresiones ronda 2 (api-edge) ----------------------------------------------------------------


class PromptCountingBackend(FakeBackend):
    """Como GenerationStream: el stream informa prompt_tokens; el backend no tiene count_prompt_tokens."""

    count_prompt_tokens = None  # type: ignore[assignment]

    def stream(self, messages, **kw):
        inner = super().stream(messages, **kw)

        class _Stream:
            prompt_tokens = 42
            finish_reason = None

            def __iter__(self):
                return self

            def __next__(self):
                return next(inner)

            def close(self):
                inner.close()

        return _Stream()


def test_usage_prompt_tokens_comes_from_backend_stream():
    """usage.prompt_tokens siempre es 0 con los backends reales (HF y vLLM): el conteo del stream
    del backend llega a chat, completions, streaming con include_usage y la traza."""
    ctx, traces = make_ctx(backend=PromptCountingBackend())
    client = make_client(ctx)
    expected = {"prompt_tokens": 42, "completion_tokens": 8, "total_tokens": 50}
    assert chat(client).json()["usage"] == expected
    body = {"model": MODEL, "prompt": "Belgrano", "max_tokens": 8}
    assert client.post("/v1/completions", json=body).json()["usage"] == expected
    events = sse_events(chat(client, stream=True, stream_options={"include_usage": True}))
    assert events[-2]["usage"] == expected
    assert [t.generation.prompt_tokens for t in traces] == [42, 42, 42]


def test_usage_prompt_tokens_falls_back_to_count_prompt_tokens():
    """Un stream sin prompt_tokens (generador) sigue usando count_prompt_tokens del backend."""
    ctx, _ = make_ctx()
    assert chat(make_client(ctx)).json()["usage"]["prompt_tokens"] == 3


class RemoteBackend(FakeBackend):
    """Proxy en modo reducido (como VllmProxyBackend): sin lock, mide cuántas generaciones solapan."""

    mode = "reduced"

    def __init__(self) -> None:
        super().__init__()
        self.inflight = self.max_inflight = 0
        self._mutex = threading.Lock()

    def stream(self, messages, **kw):
        with self._mutex:
            self.inflight += 1
            self.max_inflight = max(self.max_inflight, self.inflight)
        try:
            time.sleep(0.2)
            yield from super().stream(messages, **kw)
        finally:
            with self._mutex:
                self.inflight -= 1


def test_reduced_mode_proxy_is_not_serialized():
    """El proxy vLLM serializa todas las generaciones del proceso (cupo 1): un backend en modo
    reducido sin max_concurrency deja pasar requests concurrentes; el backend en proceso sigue en 1."""
    from vigia.api import REMOTE_MAX_CONCURRENCY, generation_gate

    assert REMOTE_MAX_CONCURRENCY > 1
    assert generation_gate(FakeBackend()).slots == 1
    declared = RemoteBackend()
    declared.max_concurrency = 3  # type: ignore[attr-defined]
    assert generation_gate(declared).slots == 3
    backend = RemoteBackend()
    ctx, _ = make_ctx(backend=backend)
    app = _asgi_app(ctx)

    async def scenario() -> list[int]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            payload = {"model": MODEL, "messages": [{"role": "user", "content": "hola"}]}
            reqs = [client.post("/v1/chat/completions", json=payload) for _ in range(4)]
            return [r.status_code for r in await asyncio.wait_for(asyncio.gather(*reqs), 30)]

    assert asyncio.run(scenario()) == [200] * 4
    assert backend.max_inflight > 1


@pytest.mark.parametrize("param", ["temperature", "top_p", "n", "stream", "logprobs"])
def test_nullable_openai_params_accept_null(param):
    """Parámetros OpenAI nullable con null devuelven 422 (temperature, top_p, n, stream): null
    equivale al default."""
    ctx, _ = make_ctx()
    client = make_client(ctx)
    resp = chat(client, **{param: None})
    assert resp.status_code == 200, resp.text
    assert resp.json()["object"] == "chat.completion"
    if param != "logprobs":
        body = {"model": MODEL, "prompt": "Belgrano", param: None}
        assert client.post("/v1/completions", json=body).status_code == 200


def test_deeply_nested_json_is_400_not_500():
    """JSON profundamente anidado en /v1/* devuelve 500 en lugar de 400."""
    ctx, traces = make_ctx()
    client = make_client(ctx)
    nested = "[" * 100_000 + "]" * 100_000
    raw = f'{{"model":"{MODEL}","messages":[{{"role":"user","content":"x"}}],"x":{nested}}}'
    resp = client.post("/v1/chat/completions", content=raw, headers={"content-type": "application/json"})
    assert resp.status_code == 400 and resp.json()["error"]["code"] == "invalid_json"
    assert traces == []


@pytest.mark.parametrize("stream", [False, True])
def test_AT_A08_fail_open_in_score_response_leaves_no_flags(stream):
    """fail_open con falla de score_response deja tokens con p_error y flag=true aunque tau=null y
    degraded=true: la respuesta y la traza no traen flags ni fragmentos de la señal degradada."""

    class BrokenResponse(FakeScorer):
        def score_response(self, ps):
            raise RuntimeError("detector roto")

    policy = Policy(on_detector_error="fail_open", stream_policy="buffer")
    ctx, traces = make_ctx(scorer=BrokenResponse(), policy=policy)
    resp = chat(make_client(ctx), vigia={}, stream=stream)
    assert resp.status_code == 200
    if stream:
        events = sse_events(resp)
        final = events[-2]["choices"][0]["vigia"]
        tokens = [t for e in events[1:-2] for t in e["choices"][0]["vigia"]["tokens"]]
    else:
        final = resp.json()["choices"][0]["vigia"]
        tokens = final["tokens"]
    assert final["degraded"] is True and final["tau"] is None and final["response_p_error"] is None
    assert final["spans"] == [] and final["action"] == "pass"
    assert tokens and all(t["p_error"] is None and t["flag"] is False for t in tokens)
    stored = traces[0].result
    assert stored.spans == [] and all(t.p_error is None and not t.flag for t in stored.tokens)


def _base_body(path: str) -> dict[str, Any]:
    if path == "/v1/chat/completions":
        return {"model": MODEL, "messages": [{"role": "user", "content": "hola"}]}
    return {"model": MODEL, "prompt": "hola"}


@pytest.mark.parametrize(
    ("path", "param", "value"),
    [
        ("/v1/chat/completions", "tools", [{"type": "function", "function": {"name": "f"}}]),
        ("/v1/chat/completions", "tool_choice", "required"),
        ("/v1/chat/completions", "functions", [{"name": "f"}]),
        ("/v1/chat/completions", "function_call", "auto"),
        ("/v1/chat/completions", "response_format", {"type": "json_object"}),
        ("/v1/chat/completions", "logit_bias", {"50256": -100}),
        ("/v1/completions", "echo", True),
        ("/v1/completions", "best_of", 3),
        ("/v1/completions", "suffix", "fin"),
        ("/v1/completions", "logit_bias", {"50256": -100}),
        ("/v1/chat/completions", "modalities", ["text", "audio"]),
        ("/v1/chat/completions", "modalities", ["audio"]),
        ("/v1/chat/completions", "audio", {"voice": "x", "format": "mp3"}),
        ("/v1/chat/completions", "reasoning_effort", "high"),
    ],
)
def test_unsupported_semantic_params_are_422(path, param, value):
    """Parámetros OpenAI no soportados se ignoran en silencio (tools, response_format, echo,
    best_of, logit_bias): se rechazan con 422 unsupported_value."""
    ctx, traces = make_ctx()
    resp = make_client(ctx).post(path, json={**_base_body(path), param: value})
    assert resp.status_code == 422
    err = resp.json()["error"]
    assert err["code"] == "unsupported_value" and err["param"] == param
    assert traces == []


@pytest.mark.parametrize(
    ("path", "extra"),
    [
        ("/v1/chat/completions", {"tools": [], "tool_choice": "none", "response_format": {"type": "text"}}),
        ("/v1/chat/completions", {"tools": None, "logit_bias": {}, "functions": None}),
        ("/v1/completions", {"echo": False, "best_of": 1, "suffix": None, "logit_bias": None}),
        ("/v1/chat/completions", {"modalities": ["text"], "audio": None, "reasoning_effort": None}),
        ("/v1/chat/completions", {"modalities": None}),
    ],
)
def test_default_values_of_unsupported_params_are_accepted(path, extra):
    """Los valores inocuos (null, vacío o el default de OpenAI) de esos parámetros no se rechazan."""
    ctx, _ = make_ctx()
    assert make_client(ctx).post(path, json={**_base_body(path), **extra}).status_code == 200


# --- Regresiones de la verificación api-edge y ops-resilience (ronda 3) ---------------------------


def test_modalities_audio_reasoning_effort_are_rejected():
    """Parámetros OpenAI que cambian la respuesta siguen ignorándose en silencio: modalities ['audio'],
    audio y reasoning_effort."""
    ctx, traces = make_ctx()
    client = make_client(ctx)
    for extra in (
        {"modalities": ["audio"], "audio": {"voice": "x", "format": "mp3"}},
        {"reasoning_effort": "high"},
    ):
        r = chat(client, **extra)
        assert r.status_code == 422 and r.json()["error"]["code"] == "unsupported_value"
    assert traces == []


async def _asgi_call(
    app: FastAPI, body: dict[str, Any], *, disconnect_after: float | None = None
) -> dict[str, Any]:
    """Llama la app por ASGI. Con `disconnect_after`, el cliente se desconecta a esos segundos."""
    raw = json.dumps(body).encode()
    loop = asyncio.get_running_loop()
    started = loop.time()
    done = asyncio.Event()
    state: dict[str, Any] = {"sent": False, "status": None, "headers": {}, "body": b""}

    async def receive() -> dict[str, Any]:
        if not state["sent"]:
            state["sent"] = True
            return {"type": "http.request", "body": raw, "more_body": False}
        if disconnect_after is None:
            await done.wait()
        elif (remaining := started + disconnect_after - loop.time()) > 0:
            await asyncio.sleep(remaining)  # como uvicorn: ya desconectado, receive no se suspende
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        if message["type"] == "http.response.start":
            state["status"] = message["status"]
            state["headers"] = {k.decode().lower(): v.decode() for k, v in message.get("headers", [])}
        elif message["type"] == "http.response.body":
            state["body"] += message.get("body", b"")
            if not message.get("more_body", False):
                done.set()

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "root_path": "",
        "query_string": b"",
        "headers": [(b"content-type", b"application/json"), (b"host", b"testserver")],
        "client": ("127.0.0.1", 1234),
        "server": ("testserver", 80),
    }
    await asyncio.wait_for(app(scope, receive, send), 30)
    state["returned"] = loop.time() - started
    return state


class CountingBackend(LockingBackend):
    """LockingBackend que cuenta los tokens que llegó a generar."""

    def __init__(self, n: int, delay: float) -> None:
        super().__init__(n=n, delay=delay)
        self.generated = 0

    def _locked(self, inner):
        for ev in super()._locked(inner):
            self.generated += 1
            yield ev


class GatedBackend(LockingBackend):
    """Retiene el cupo hasta que el test lo suelta: el orden no depende de la carga de la máquina."""

    def __init__(self) -> None:
        super().__init__(n=5)
        self.started = threading.Event()
        self.go = threading.Event()

    def _locked(self, inner):
        with self.lock:
            self.started.set()
            assert self.go.wait(20)
            yield from inner


def _slow_body(**extra: Any) -> dict[str, Any]:
    return {"model": MODEL, "messages": [{"role": "user", "content": "hola"}], "max_tokens": 200, **extra}


async def _hold_slot(app: FastAPI, backend: GatedBackend) -> asyncio.Future[dict[str, Any]]:
    """Lanza un request que toma el cupo y espera a que empiece a generar."""
    first = asyncio.ensure_future(_asgi_call(app, _slow_body()))
    assert await asyncio.get_running_loop().run_in_executor(None, backend.started.wait, 20)
    return first


def test_abandoned_request_in_queue_is_not_generated():
    """Un request sin streaming que el cliente abandona se genera completo igual, incluso si estaba en
    la cola del GenerationGate: al desconectarse mientras espera, deja la cola sin generar."""
    backend = GatedBackend()
    ctx, traces = make_ctx(backend=backend)
    app = _asgi_app(ctx)

    async def scenario() -> tuple[dict[str, Any], list[dict[str, Any]]]:
        first = await _hold_slot(app, backend)
        try:
            gone = [_asgi_call(app, _slow_body(), disconnect_after=0.05) for _ in range(4)]
            gone_states = list(await asyncio.gather(*gone))
        finally:
            backend.go.set()
        return await first, gone_states

    first, gone = asyncio.run(scenario())
    assert first["status"] == 200
    assert [g["status"] for g in gone] == [499] * 4  # se fueron de la cola sin esperar el cupo
    assert len(backend.calls) == 1 and len(traces) == 1 and not backend.lock.locked()
    assert chat(make_client(ctx), max_tokens=2).status_code == 200  # el cupo no quedó perdido


def test_abandoned_request_is_cancelled_between_tokens():
    """Un request sin streaming que el cliente abandona se genera completo igual: la generación en curso
    se corta entre tokens, libera el backend y no guarda una traza que nadie pidió."""
    backend = CountingBackend(n=600, delay=0.01)
    ctx, traces = make_ctx(backend=backend)
    app = _asgi_app(ctx)
    gc.disable()  # el corte no puede depender del GC cíclico
    try:
        state = asyncio.run(_asgi_call(app, _slow_body(max_tokens=600), disconnect_after=0.1))
        assert state["status"] == 499
        assert 0 < backend.generated < 600
        assert not backend.lock.locked() and traces == []
    finally:
        gc.enable()
    assert chat(make_client(ctx), max_tokens=2).status_code == 200


@pytest.mark.parametrize("stream", [False, True])
def test_queued_generation_during_shutdown_is_503_without_generating(stream):
    """Las generaciones en cola cuando llega CTRL+C/SIGTERM igual arrancan y terminan cortadas: un
    request que espera el cupo al empezar el cierre responde enseguida 503 JSON con Retry-After."""
    backend = GatedBackend()
    closing = threading.Event()
    ctx, traces = make_ctx(backend=backend, closing=closing.is_set)
    app = _asgi_app(ctx)

    async def scenario() -> tuple[dict[str, Any], list[dict[str, Any]]]:
        first = await _hold_slot(app, backend)
        try:
            queued = [asyncio.ensure_future(_asgi_call(app, _slow_body(stream=stream))) for _ in range(3)]
            await asyncio.sleep(0.3)
            closing.set()
            queued_states = list(await asyncio.gather(*queued))
        finally:
            backend.go.set()  # la generación en curso recién termina después de los 503
        return await first, queued_states

    first, queued = asyncio.run(scenario())
    assert first["status"] == 200  # la generación en curso termina
    for q in queued:
        assert q["status"] == 503 and q["headers"]["retry-after"] == "5"
        err = json.loads(q["body"])["error"]
        assert err["code"] == "shutting_down" and err["request_id"]
    assert len(backend.calls) == 1 and len(traces) == 1


def test_request_reaching_slot_during_shutdown_does_not_generate():
    """Las generaciones en cola cuando llega CTRL+C/SIGTERM igual arrancan: con el cierre activo, el
    request no arranca la generación aunque el cupo esté libre."""
    ctx, traces = make_ctx(closing=lambda: True)
    r = chat(make_client(ctx))
    assert r.status_code == 503 and r.json()["error"]["code"] == "shutting_down"
    assert r.headers["Retry-After"] == "5" and ctx.backend.calls == [] and traces == []
    ctx2, _ = make_ctx(closing=lambda: False)
    assert chat(make_client(ctx2)).status_code == 200


def test_status_cancelled_on_stream_cut_and_not_degraded():
    """VigiaResult.status nunca se setea: un stream cortado por el cliente queda status='cancelled' y
    degraded=false (degraded es solo la falla del detector)."""
    backend = LockingBackend(n=300, delay=0.002)
    ctx, traces = make_ctx(backend=backend)
    body = {"model": MODEL, "messages": [{"role": "user", "content": "hola"}], "stream": True, "vigia": {}}
    asyncio.run(_stream_then_disconnect(_asgi_app(ctx), body, n_chunks=3))
    assert len(traces) == 1
    res = traces[0].result
    assert traces[0].generation.finish_reason == "error"
    assert res.status == "cancelled" and res.degraded is False


@pytest.mark.parametrize("stream", [False, True])
def test_status_detector_error_on_fail_open(stream):
    """VigiaResult.status nunca se setea: con fail_open y el detector caído, status='detector_error'."""
    ctx, traces = make_ctx(scorer=FakeScorer(fail_on="score"), policy=Policy(on_detector_error="fail_open"))
    r = chat(make_client(ctx), vigia={}, stream=stream)
    final = sse_events(r)[-2]["choices"][0]["vigia"] if stream else r.json()["choices"][0]["vigia"]
    assert final["degraded"] is True and final["status"] == "detector_error"
    assert traces[0].result.status == "detector_error"


@pytest.mark.parametrize(
    ("pieces", "harmony", "max_tokens", "status"),
    [
        (None, False, 16, "ok"),
        ([], False, 16, "empty_generation"),
        (["  "], False, 16, "empty_generation"),
        ([], True, 16, "no_final_channel"),
        ([], True, 2, "truncated_before_final"),
    ],
)
@pytest.mark.parametrize("stream", [False, True])
def test_status_reflects_generation_outcome(pieces, harmony, max_tokens, status, stream):
    """VigiaResult.status nunca se setea: siempre 'ok' aunque la generación quedó vacía o una respuesta
    harmony no llegó al canal final."""
    ctx, traces = make_ctx(backend=FakeBackend(pieces, harmony=harmony))
    r = chat(make_client(ctx), vigia={}, stream=stream, max_tokens=max_tokens)
    assert r.status_code == 200
    final = sse_events(r)[-2]["choices"][0]["vigia"] if stream else r.json()["choices"][0]["vigia"]
    assert final["status"] == status and final["degraded"] is False
    assert traces[-1].result.status == status


class UpstreamDownBackend(RemoteBackend):
    """Proxy en modo reducido cuyo vLLM no responde (conexión, timeout o 5xx)."""

    def stream(self, messages, **kw):
        from vigia.backends.vllm_proxy import VllmProxyError

        raise VllmProxyError("vLLM respondió HTTP 503")


@pytest.mark.parametrize("stream", [False, True])
def test_upstream_failure_is_502_with_retry_after_and_logged_type(stream, caplog):
    """vLLM caído, 5xx o timeout sale como 500 'Error interno del servidor' sin Retry-After y el log
    no dice la causa: responde 502 upstream_unavailable con Retry-After y loguea error_type."""
    ctx, traces = make_ctx(backend=UpstreamDownBackend(), scorer=None)
    with caplog.at_level(logging.WARNING, logger="vigia.api"):
        r = chat(make_client(ctx), stream=stream)
    assert r.status_code == 502 and r.headers["Retry-After"] == "5"
    err = r.json()["error"]
    assert err["code"] == "upstream_unavailable" and err["request_id"]
    assert any(getattr(rec, "error_type", None) == "VllmProxyError" for rec in caplog.records)
    assert traces == []


def test_internal_error_log_carries_error_type(caplog):
    """vLLM caído ... el log no dice la causa: el 500 genérico también registra error_type."""
    ctx, _ = make_ctx(backend=FakeBackend(fail=True))
    with caplog.at_level(logging.ERROR, logger="vigia.api"):
        assert chat(make_client(ctx)).status_code == 500
    assert any(getattr(rec, "error_type", None) == "RuntimeError" for rec in caplog.records)
