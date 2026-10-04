"""Tests de la API OpenAI-compatible con backend y detector falsos (P-13, P-14)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import replace
from typing import Any

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict

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
