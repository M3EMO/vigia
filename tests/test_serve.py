"""Servidor single-tenant del núcleo abierto: auth Bearer, trazas LRU, health y límites (P-13, P-14)."""

from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient

from vigia.cli import make_tiny_backend
from vigia.contracts import Capabilities, DetectorMismatchError, ModelInfo, Policy, Tier, TokenEvent
from vigia.serve import TraceStore, create_app, is_loopback

KEY = "clave-de-prueba-1234567890"
LOCAL = ("127.0.0.1", 50000)
INFO = ModelInfo(
    model_id="fake-moe",
    model_type="mixtral",
    n_layers=2,
    moe_layers=(0, 1),
    n_experts=4,
    top_k=2,
    vocab_size=100,
    gating="softmax_topk",
)
PIECES = ["Belgrano", " nació", " en", " 1820", "."]


class FakeBackend:
    def __init__(self, fail_info: bool = False):
        self.fail_info = fail_info

    def model_info(self) -> ModelInfo:
        if self.fail_info:
            raise RuntimeError("caído")
        return INFO

    def capabilities(self) -> Capabilities:
        return Capabilities(tiers=(Tier.F0, Tier.F1), streaming=True)

    def stream(self, messages, *, max_tokens, temperature=0.0, top_p=1.0, seed=None, stop=None, tiers=()):
        pos = 0
        for i, piece in enumerate(PIECES[:max_tokens]):
            yield TokenEvent(i, 10 + i, piece, pos, pos + len(piece), {"out_nll": 0.5})
            pos += len(piece)


class FakeScorer:
    id = "abcd1234abcd1234"

    def __init__(self, model_ok: bool = True, p: float = 0.1):
        self.model_ok = model_ok
        self.p = p

    def check_model(self, info: ModelInfo) -> None:
        if not self.model_ok:
            raise DetectorMismatchError("otro modelo")

    def tau(self, alpha: float) -> float:
        return 0.5

    def score_tokens(self, tokens):
        return [0.9 if t.text.strip() == "1820" else self.p for t in tokens]

    def score_response(self, ps):
        vals = [p for p in ps if p is not None]
        return max(vals) if vals else None


def chat(client: TestClient, headers=None, **extra):
    body = {"model": "fake-moe", "messages": [{"role": "user", "content": "¿Dónde nació Belgrano?"}], **extra}
    return client.post("/v1/chat/completions", json=body, headers=headers or {})


def auth(key: str = KEY) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def test_bearer_auth_required_and_constant_time_compare():
    client = TestClient(create_app(FakeBackend(), api_key=KEY))
    r = chat(client)
    assert r.status_code == 401 and r.headers["WWW-Authenticate"].startswith("Bearer")
    assert r.json()["error"]["code"] == "invalid_api_key" and r.json()["error"]["request_id"]
    assert chat(client, auth("clave-equivocada-1234567890")).status_code == 401
    assert chat(client, {"Authorization": f"Basic {KEY}"}).status_code == 401
    assert chat(client, auth(KEY[:-1])).status_code == 401
    ok = chat(client, auth())
    assert (
        ok.status_code == 200 and ok.json()["choices"][0]["message"]["content"] == "Belgrano nació en 1820."
    )
    assert client.get("/v1/models", headers=auth()).json()["data"][0]["id"] == "fake-moe"
    r = client.get("/v1/models?api_key=" + KEY)
    assert r.status_code == 400 and r.json()["error"]["code"] == "credentials_in_query"


def test_api_key_must_be_strong():
    with pytest.raises(ValueError, match="16 caracteres"):
        create_app(FakeBackend(), api_key="corta")
    with pytest.raises(ValueError, match="espacios"):
        create_app(FakeBackend(), api_key="clave con espacios 12345")


def test_without_key_only_loopback_and_warns(caplog):
    with caplog.at_level(logging.WARNING, logger="vigia.serve"):
        app = create_app(FakeBackend())
    assert "sin API key" in caplog.text
    remote = TestClient(app, client=("10.0.0.7", 40000))
    r = chat(remote)
    assert r.status_code == 403 and r.json()["error"]["code"] == "forbidden"
    assert chat(TestClient(app, client=LOCAL)).status_code == 200
    assert chat(TestClient(app, client=("::1", 40000))).status_code == 200


def test_is_loopback():
    assert is_loopback("127.0.0.1") and is_loopback("127.8.0.1") and is_loopback("::1")
    assert is_loopback("[::1]") and is_loopback("localhost")
    assert not is_loopback("0.0.0.0")  # noqa: S104 and not is_loopback("10.0.0.1") and not is_loopback(None)
    assert not is_loopback("testclient")


def test_traces_by_id_and_vigia_field():
    """P-14: la traza se recupera por el header X-Vigia-Trace-Id con el mismo campo vigia."""
    client = TestClient(create_app(FakeBackend(), detector=FakeScorer(), api_key=KEY))
    r = chat(client, auth(), vigia={})
    assert r.status_code == 200
    trace_id = r.headers["X-Vigia-Trace-Id"]
    vigia = r.json()["choices"][0]["vigia"]
    assert vigia["calibration_id"] == FakeScorer.id and vigia["spans"][0]["text"].strip() == "1820"
    t = client.get(f"/v1/vigia/traces/{trace_id}", headers=auth())
    assert t.status_code == 200
    body = t.json()
    assert body["id"] == trace_id and body["response"] == "Belgrano nació en 1820."
    assert body["vigia"]["spans"] == vigia["spans"] and body["vigia"]["action"] == vigia["action"]
    assert client.get(f"/v1/vigia/traces/{trace_id}").status_code == 401
    assert client.get("/v1/vigia/traces/no-existe-123", headers=auth()).status_code == 404


def test_abstain_trace_does_not_leak_original_text():
    policy = Policy(abstain_at=0.5, human_review_at=None)
    client = TestClient(create_app(FakeBackend(), detector=FakeScorer(), policy=policy, api_key=KEY))
    r = chat(client, auth(), vigia={})
    assert r.json()["choices"][0]["message"]["content"] == policy.abstain_message
    body = client.get(f"/v1/vigia/traces/{r.headers['X-Vigia-Trace-Id']}", headers=auth()).json()
    assert body["vigia"]["action"] == "abstain"
    assert body["response"] is None and body["vigia"]["tokens"] == [] and body["vigia"]["spans"] == []
    assert "1820" not in str(body)


def test_trace_store_is_lru_bounded():
    app = create_app(FakeBackend(), api_key=KEY, max_traces=3)
    client = TestClient(app)
    ids = [chat(client, auth()).headers["X-Vigia-Trace-Id"] for _ in range(3)]
    assert client.get(f"/v1/vigia/traces/{ids[0]}", headers=auth()).status_code == 200  # ids[0] pasa al final
    ids.append(chat(client, auth()).headers["X-Vigia-Trace-Id"])
    store: TraceStore = app.state.vigia_traces
    assert len(store) == 3
    assert store.get(ids[1]) is None and store.get(ids[0]) is not None and store.get(ids[3]) is not None
    with pytest.raises(ValueError):
        TraceStore(0)


def test_health_routes_and_no_public_docs():
    client = TestClient(create_app(FakeBackend(), detector=FakeScorer(), api_key=KEY))
    assert client.get("/livez").json() == {"status": "ok"}
    ready = client.get("/readyz")
    assert ready.status_code == 200 and ready.json() == {
        "status": "ready",
        "model": "fake-moe",
        "calibrated": True,
    }
    for path in ("/docs", "/redoc", "/openapi.json"):
        r = client.get(path)
        assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"


def test_readyz_reports_backend_failure():
    backend = FakeBackend()
    client = TestClient(create_app(backend, api_key=KEY))
    backend.fail_info = True
    r = client.get("/readyz")
    assert r.status_code == 503 and r.json() == {"status": "not_ready"}


def test_detector_for_other_model_fails_at_startup():
    with pytest.raises(DetectorMismatchError):
        create_app(FakeBackend(), detector=FakeScorer(model_ok=False), api_key=KEY)


def test_body_limit_and_security_headers():
    """AT-P07 en el servidor single-tenant: body enorme -> 413; max_tokens enorme -> 422."""
    client = TestClient(
        create_app(
            FakeBackend(), api_key=KEY, max_body_bytes=2048, max_tokens_limit=64, default_max_tokens=16
        )
    )
    big = chat(client, auth(), padding="x" * 5000)
    assert big.status_code == 413 and big.json()["error"]["code"] == "body_too_large"
    assert chat(client, auth(), max_tokens=10_000).status_code == 422
    r = chat(client, auth())
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert r.headers["Cache-Control"] == "no-store" and r.headers["X-Frame-Options"] == "DENY"
    stream = chat(client, auth(), stream=True)
    assert stream.status_code == 200 and stream.headers["Cache-Control"] == "no-cache"
    assert stream.text.rstrip().endswith("data: [DONE]")


def test_serve_with_tiny_model_end_to_end():
    backend = make_tiny_backend("gpt_oss")
    client = TestClient(create_app(backend), client=LOCAL)
    body = {
        "model": "tiny:gpt_oss",
        "messages": [{"role": "user", "content": "Hola"}],
        "max_tokens": 6,
        "vigia": {},
    }
    r = client.post("/v1/chat/completions", json=body)
    assert r.status_code == 200
    choice = r.json()["choices"][0]
    assert choice["vigia"]["calibrated"] is False and choice["vigia"]["mode"] == "full"
    assert all(t["p_error"] is None for t in choice["vigia"]["tokens"])
    trace = client.get(f"/v1/vigia/traces/{r.headers['X-Vigia-Trace-Id']}").json()
    assert trace["response"] == choice["message"]["content"]
