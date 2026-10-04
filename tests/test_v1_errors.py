"""Errores de `/v1` en la plataforma con el sobre de OpenAI; el panel conserva su formato.

Defecto: "Los errores de auth, body límite y 405 en /v1 de la plataforma no tienen la forma de
error de OpenAI".
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel
from vigia_platform_support import make_runtime

from vigia_ee.app import create_app
from vigia_ee.auth import create_api_key
from vigia_ee.security import install_security

ENVELOPE_KEYS = {"message", "type", "code", "param", "request_id"}
BODY = {"model": "tiny:mixtral", "messages": [{"role": "user", "content": "hola"}], "max_tokens": 2}


@pytest.fixture
def rt(tmp_path):
    runtime = make_runtime(tmp_path, max_body_bytes=4096, key_rate_per_minute=2)
    yield runtime
    runtime.close()


@pytest.fixture
def client(rt):
    return TestClient(create_app(runtime=rt))


def api_key(rt):
    org = rt.create_org("Acme")
    return create_api_key(rt.auth, org, "tests", ["ingest", "read"])[0]


def openai_error(resp, status, type_, code):
    assert resp.status_code == status, resp.text
    body = resp.json()
    assert set(body) == {"error"} and set(body["error"]) == ENVELOPE_KEYS, body
    err = body["error"]
    assert (err["type"], err["code"]) == (type_, code), err
    assert isinstance(err["message"], str) and err["message"]
    assert err["request_id"] == resp.headers["x-request-id"]
    return err


def test_v1_auth_errors_use_openai_envelope(client, rt):
    """Defecto "Los errores de auth, body límite y 405 en /v1 ...": 401 y 400 de auth."""
    url = "/v1/chat/completions"
    r = client.post(url, json=BODY)
    openai_error(r, 401, "authentication_error", "invalid_api_key")
    assert r.headers["www-authenticate"] == "Bearer"
    bad = client.post(url, json=BODY, headers={"Authorization": "Bearer vg_live_no_existe"})
    openai_error(bad, 401, "authentication_error", "invalid_api_key")
    scheme = client.post(url, json=BODY, headers={"Authorization": "Basic abc"})
    openai_error(scheme, 401, "authentication_error", "invalid_api_key")
    query = client.post(url + "?api_key=x", json=BODY)
    openai_error(query, 400, "invalid_request_error", "credentials_in_query")
    token = create_api_key(rt.auth, rt.create_org("Beta"), "solo-read", ["read"])[0]
    scope = client.post(url, json=BODY, headers={"Authorization": f"Bearer {token}"})
    openai_error(scope, 403, "invalid_request_error", "insufficient_scope")


def test_v1_body_limit_405_404_and_rate_limit_use_openai_envelope(client, rt):
    """Defecto "Los errores de auth, body límite y 405 en /v1 ...": 413, 405, 404 y 429."""
    token = api_key(rt)
    auth = {"Authorization": f"Bearer {token}"}
    big = client.post("/v1/chat/completions", content=b"x" * 10_000, headers=auth)
    openai_error(big, 413, "invalid_request_error", "payload_too_large")
    openai_error(client.get("/v1/chat/completions"), 405, "invalid_request_error", "method_not_allowed")
    openai_error(client.get("/v1/no-existe"), 404, "invalid_request_error", "not_found")
    codes = [client.get("/v1/models", headers=auth) for _ in range(3)]
    assert [r.status_code for r in codes[:2]] == [200, 200]
    limited = openai_error(codes[2], 429, "rate_limit_error", "rate_limited")
    assert limited["param"] is None and int(codes[2].headers["retry-after"]) >= 1


class Item(BaseModel):
    n: int


def test_v1_internal_error_503_and_422_use_openai_envelope():
    """Defecto "Los errores de auth, body límite y 405 en /v1 ...": 500, 503 y 422 genéricos."""
    app = FastAPI()
    install_security(app)

    @app.post("/v1/items")
    def items(item: Item):
        return item

    @app.get("/v1/boom")
    def boom():
        raise RuntimeError("detalle interno secreto")

    @app.get("/v1/down")
    def down():
        raise HTTPException(503, "No disponible.")

    c = TestClient(app, raise_server_exceptions=False)
    err = openai_error(c.get("/v1/boom"), 500, "server_error", "internal_error")
    assert "secreto" not in err["message"]
    openai_error(c.get("/v1/down"), 503, "service_unavailable_error", "unavailable")
    bad = c.post("/v1/items", json={"n": "DNI 30123456"})
    assert openai_error(bad, 422, "invalid_request_error", "validation_error")["param"] == "n"
    assert "30123456" not in bad.text


def test_panel_errors_keep_flat_format(client):
    """El panel (fuera de /v1) conserva `{error, message, request_id}`."""
    r = client.post("/login", content=b"x" * 10_000_000)
    assert r.status_code == 413
    assert r.json() == {
        "error": "payload_too_large",
        "message": "El cuerpo del request supera el tamaño máximo.",
        "request_id": r.headers["x-request-id"],
    }
    assert client.get("/v1x").json()["error"] == "not_found"  # solo el prefijo exacto /v1


def test_openai_sdk_reads_code_of_platform_errors(client, rt):
    """Defecto "Los errores de auth, body límite y 405 en /v1 ...": el SDK lee `e.code`."""
    openai = pytest.importorskip("openai")
    sdk = openai.OpenAI(base_url="http://testserver/v1", api_key="vg_live_x", http_client=client)
    with pytest.raises(openai.AuthenticationError) as auth_err:
        sdk.chat.completions.create(**BODY)
    assert auth_err.value.code == "invalid_api_key"
    sdk = sdk.with_options(api_key=api_key(rt), max_retries=0)
    huge = [{"role": "user", "content": "x" * 10_000}]
    with pytest.raises(openai.APIStatusError) as big_err:
        sdk.chat.completions.create(model="tiny:mixtral", messages=huge, max_tokens=2)
    assert big_err.value.status_code == 413 and big_err.value.code == "payload_too_large"
    assert big_err.value.body["type"] == "invalid_request_error"  # el SDK desenvuelve el sobre
