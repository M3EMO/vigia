"""Tests de vigia_ee.security: headers, errores genéricos, límite de body, rate limit, CSRF y logs."""

from __future__ import annotations

import io
import json
import logging
import re

import pytest
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.testclient import TestClient
from pydantic import BaseModel

from vigia_ee import security
from vigia_ee.security import (
    RateLimiter,
    RateLimitExceeded,
    csrf_token_for,
    enforce_csrf,
    enforce_rate_limit,
    install_security,
    sanitize_log_value,
)

LIMITER = RateLimiter(rate=1.0, burst=2)
SECRET, SID_HASH = "a" * 64, "b" * 64


class Item(BaseModel):
    n: int


def make_app() -> FastAPI:
    app = FastAPI()
    install_security(app, max_body_bytes=1024)

    @app.get("/page")
    def page(request: Request):
        return {"nonce": request.state.csp_nonce, "rid": request.state.request_id}

    @app.get("/private")
    def private(request: Request):
        request.state.authenticated = True
        return {"ok": True}

    @app.get("/boom")
    def boom():
        raise RuntimeError("password=hunter2 dato secreto del cliente")

    @app.post("/echo")
    async def echo(request: Request):
        return {"len": len(await request.body())}

    @app.post("/items")
    def items(item: Item):
        return item

    @app.get("/limited")
    def limited():
        enforce_rate_limit(LIMITER, "k", scope="key")
        return {"ok": True}

    @app.get("/teapot")
    def teapot():
        raise HTTPException(401, "Necesitás credenciales.", headers={"WWW-Authenticate": "Bearer"})

    async def csrf_dep(request: Request):
        await enforce_csrf(request, SECRET, SID_HASH)

    @app.post("/form", dependencies=[Depends(csrf_dep)])
    async def form():
        return {"ok": True}

    return app


@pytest.fixture
def client():
    return TestClient(make_app(), base_url="https://testserver")


# --- headers --------------------------------------------------------------------------------------


def test_at_p12_security_headers_and_strict_csp(client):
    """AT-P12: CSP sin unsafe-inline/unsafe-eval ni hosts externos y headers de seguridad presentes."""
    r = client.get("/page")
    csp = r.headers["content-security-policy"]
    assert "unsafe-inline" not in csp and "unsafe-eval" not in csp
    assert not re.search(r"https?:|\*", csp)
    for directive in (
        "default-src 'self'",
        "'strict-dynamic'",
        "object-src 'none'",
        "base-uri 'none'",
        "frame-ancestors 'none'",
        "form-action 'self'",
        "style-src 'self'",
        "img-src 'self' data:",
    ):
        assert directive in csp
    assert r.headers["strict-transport-security"].startswith("max-age=63072000")
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["referrer-policy"] == "no-referrer"
    assert "camera=()" in r.headers["permissions-policy"]
    assert r.headers["cross-origin-opener-policy"] == "same-origin"
    assert r.headers["cross-origin-resource-policy"] == "same-origin"
    assert r.headers["x-frame-options"] == "DENY"
    assert "server" not in r.headers


def test_at_p12_nonce_is_unique_per_response_and_exposed_in_state(client):
    """AT-P12: cada respuesta trae un nonce distinto y es el mismo de request.state.csp_nonce."""
    r1, r2 = client.get("/page"), client.get("/page")
    n1, n2 = r1.json()["nonce"], r2.json()["nonce"]
    assert n1 != n2 and len(n1) >= 22
    assert f"'nonce-{n1}'" in r1.headers["content-security-policy"]
    assert f"'nonce-{n2}'" in r2.headers["content-security-policy"]


def test_cache_control_no_store_only_for_authenticated(client):
    assert "cache-control" not in client.get("/page").headers
    assert client.get("/private").headers["cache-control"] == "no-store"
    assert client.get("/page", headers={"Authorization": "Bearer x"}).headers["cache-control"] == "no-store"
    client.cookies.set(security.SESSION_COOKIE, "x")
    assert client.get("/page").headers["cache-control"] == "no-store"


def test_request_id_header_matches_state(client):
    r = client.get("/page")
    assert r.headers["x-request-id"] == r.json()["rid"]
    assert (
        client.get("/page", headers={"X-Request-ID": "inyectado-123"}).headers["x-request-id"]
        != "inyectado-123"
    )


# --- errores --------------------------------------------------------------------------------------


def test_at_p17_internal_error_is_generic_with_request_id(client):
    """AT-P17: un 500 devuelve JSON genérico con request_id, sin stack trace ni mensaje interno."""
    r = client.get("/boom")
    assert r.status_code == 500
    body = r.json()
    assert body == {
        "error": "internal_error",
        "message": "Error interno.",
        "request_id": r.headers["x-request-id"],
    }
    assert "Traceback" not in r.text and "hunter2" not in r.text and "RuntimeError" not in r.text
    assert "content-security-policy" in r.headers  # los headers de seguridad también van en el 500


def test_at_p17_internal_error_log_has_no_exception_message():
    """AT-P17: el log guarda tipo y frames, no el mensaje de la excepción (puede traer datos)."""
    stream = io.StringIO()
    security.configure_json_logging(stream=stream)
    TestClient(make_app()).get("/boom")
    lines = [json.loads(line) for line in stream.getvalue().splitlines()]
    err = next(line for line in lines if line.get("event") == "unhandled_error")
    assert err["error_type"] == "RuntimeError" and "test_security.py" in err["frames"]
    assert "hunter2" not in stream.getvalue()


def test_http_exception_keeps_headers_and_has_request_id(client):
    r = client.get("/teapot")
    assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"
    assert r.json()["error"] == "unauthorized" and r.json()["request_id"] == r.headers["x-request-id"]
    assert client.get("/no-existe").json()["error"] == "not_found"


def test_validation_error_does_not_echo_input(client):
    r = client.post("/items", json={"n": "DNI 30123456 secreto"})
    assert r.status_code == 422
    assert "30123456" not in r.text and r.json()["fields"][0]["loc"] == ["body", "n"]


# --- límite de body -------------------------------------------------------------------------------


def test_at_p07_body_over_limit_returns_413(client):
    """AT-P07: un body enorme recibe 413, por Content-Length o por streaming sin largo."""
    assert client.post("/echo", content=b"x" * 1000).json() == {"len": 1000}
    r = client.post("/echo", content=b"x" * 5000)
    assert r.status_code == 413 and r.json()["error"] == "payload_too_large"

    def chunks():
        for _ in range(10):
            yield b"y" * 500

    assert client.post("/echo", content=chunks()).status_code == 413


# --- rate limiting --------------------------------------------------------------------------------


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def test_at_p06_token_bucket_blocks_and_refills():
    """AT-P06: una ráfaga agota el bucket; Retry-After indica cuánto esperar y luego se recupera."""
    clock = FakeClock()
    lim = RateLimiter(rate=0.5, burst=3, clock=clock)
    assert [lim.hit("k") for _ in range(3)] == [None, None, None]
    wait = lim.hit("k")
    assert wait == pytest.approx(2.0)
    assert lim.hit("otra") is None  # buckets independientes por clave
    clock.t += 2.0
    assert lim.blocked("k") is None and lim.hit("k") is None
    assert lim.hit("k") is not None


def test_rate_limiter_bounds_memory():
    lim = RateLimiter(rate=1, burst=1, clock=FakeClock(), max_keys=10)
    for i in range(100):
        lim.hit(f"ip-{i}")
    assert len(lim._buckets) == 10


def test_at_p06_burst_returns_429_with_retry_after(client):
    """AT-P06: la ráfaga recibe 429 con Retry-After entero."""
    LIMITER.reset("k")
    codes = [client.get("/limited").status_code for _ in range(4)]
    assert codes[:2] == [200, 200] and codes[-1] == 429
    r = client.get("/limited")
    assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1
    assert r.json()["error"] == "rate_limited"


def test_enforce_rate_limit_raises_and_none_limiter_is_noop():
    enforce_rate_limit(None, "x")
    lim = RateLimiter(rate=1, burst=1, clock=FakeClock())
    enforce_rate_limit(lim, "x")
    with pytest.raises(RateLimitExceeded) as exc:
        enforce_rate_limit(lim, "x")
    assert exc.value.status_code == 429 and exc.value.headers["Retry-After"] == "1"


# --- CSRF -----------------------------------------------------------------------------------------


def test_at_p05_csrf_token_required_and_bound_to_session(client):
    """AT-P05: POST sin token o con token de otra sesión → 403; con token válido → 200."""
    good = csrf_token_for(SECRET, SID_HASH)
    other = csrf_token_for(SECRET, "c" * 64)
    assert client.post("/form", data={"x": "1"}).status_code == 403
    assert client.post("/form", data={"csrf_token": other}).status_code == 403
    assert client.post("/form", data={"csrf_token": good}).status_code == 200
    assert client.post("/form", json={}, headers={"X-CSRF-Token": good}).status_code == 200
    assert client.post("/form", json={"csrf_token": good}).status_code == 403  # JSON exige header


def test_at_p05_cross_site_fetch_is_rejected_even_with_token(client):
    """AT-P05: Sec-Fetch-Site: cross-site en un POST → 403."""
    good = csrf_token_for(SECRET, SID_HASH)
    r = client.post("/form", data={"csrf_token": good}, headers={"Sec-Fetch-Site": "cross-site"})
    assert r.status_code == 403
    assert (
        client.post("/form", data={"csrf_token": good}, headers={"Sec-Fetch-Site": "same-origin"}).status_code
        == 200
    )


# --- logs -----------------------------------------------------------------------------------------


def test_sanitize_log_value_escapes_crlf_and_controls():
    assert sanitize_log_value("a\r\nb\x00c") == "a\\r\\nb\\u0000c"
    assert "\n" not in sanitize_log_value("x y")
    assert sanitize_log_value("z" * 600).endswith("…")


def test_json_logger_uses_allowlist_and_single_lines():
    stream = io.StringIO()
    logger = security.configure_json_logging(stream=stream)
    security.log_event(logger, "login_failed", user_id="u1\nINYECTADO", password="hunter2", prompt="texto")  # noqa: S106
    line = stream.getvalue().strip().splitlines()[-1]
    data = json.loads(line)
    assert data["event"] == "login_failed" and data["user_id"] == "u1\\nINYECTADO"
    assert "password" not in data and "prompt" not in data and "hunter2" not in line
    assert data["level"] == "info"


def test_access_log_omits_query_string():
    stream = io.StringIO()
    security.configure_json_logging(stream=stream, level=logging.INFO)
    TestClient(make_app()).get("/page?api_key=vg_live_secreto&email=ana@example.com")
    out = stream.getvalue()
    assert '"path": "/page"' in out
    assert "vg_live_secreto" not in out and "ana@example.com" not in out
