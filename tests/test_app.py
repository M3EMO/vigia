"""Composición de la app: permisos, health, métricas, errores, límites, calibración de punta a punta y CLI."""

import json
import re

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from vigia_platform_support import (
    PASSWORD,
    PEPPERS,
    add_user,
    constant_detector,
    install_detector,
    learnable_jsonl,
    make_runtime,
)

from vigia_ee import app as appmod
from vigia_ee.app import create_app, main
from vigia_ee.auth import (
    PermissionSpec,
    create_api_key,
    route_permission,
    undeclared_routes,
    verify_api_key,
    viewer_mutation_violations,
)
from vigia_ee.db import Database, lookup_user_by_email

METRICS_TOKEN = "m" * 32
CSRF_RE = re.compile(r'name="csrf_token" value="([0-9a-f]{64})"')


@pytest.fixture
def rt(tmp_path):
    runtime = make_runtime(tmp_path, metrics_token=METRICS_TOKEN, max_body_bytes=4096)
    yield runtime
    runtime.close()


@pytest.fixture
def app(rt):
    return create_app(runtime=rt)


def api_key(rt, org, scopes=("ingest", "read")):
    return create_api_key(rt.auth, org, "tests", list(scopes))[0]


def login(app, email):
    client = TestClient(app, base_url="https://testserver")
    resp = client.post(
        "/login", data={"email": email, "password": PASSWORD, "next": "/"}, follow_redirects=False
    )
    assert resp.status_code == 303, resp.text
    csrf = CSRF_RE.search(client.get("/account").text).group(1)
    return client, csrf


def chat(client, token, **extra):
    body = {
        "model": "tiny:mixtral",
        "messages": [{"role": "user", "content": "hola"}],
        "max_tokens": 3,
        **extra,
    }
    return client.post("/v1/chat/completions", json=body, headers={"Authorization": f"Bearer {token}"})


# --- AT-P02: permisos por introspección -----------------------------------------------------------


def test_AT_P02_every_route_declares_permission_and_viewer_cannot_mutate(app):
    """AT-P02: deny-by-default por introspección sobre la app completa (API, health y panel)."""
    assert undeclared_routes(app) == []
    assert viewer_mutation_violations(app) == []
    specs = {r.path: route_permission(r) for r in app.routes if isinstance(r, APIRoute)}
    assert len(specs) >= 40
    assert specs["/v1/chat/completions"] == PermissionSpec("api_key", scope="ingest")
    assert specs["/v1/completions"] == PermissionSpec("api_key", scope="ingest")
    assert specs["/v1/models"] == PermissionSpec("api_key", scope="ingest")
    assert specs["/v1/vigia/traces/{trace_id}"] == PermissionSpec("api_key", scope="read")
    for path in ("/livez", "/startupz", "/readyz", "/metrics"):
        assert specs[path].kind == "public"
    assert [r for r in app.routes if not isinstance(r, APIRoute)] == [], (
        "sin mounts ni docs fuera del chequeo"
    )


def test_trace_route_requires_read_scope(rt, app):
    org = rt.create_org("Acme")
    client = TestClient(app)
    ingest_only = api_key(rt, org, ("ingest",))
    resp = chat(client, ingest_only)
    assert resp.status_code == 200
    trace_id = resp.headers["X-Vigia-Trace-Id"]
    path = f"/v1/vigia/traces/{trace_id}"
    assert client.get(path, headers={"Authorization": f"Bearer {ingest_only}"}).status_code == 403
    found = client.get(path, headers={"Authorization": f"Bearer {api_key(rt, org)}"})
    assert found.status_code == 200 and found.json()["id"] == trace_id
    assert client.get(path).status_code == 401


# --- AT-P14: health y métricas --------------------------------------------------------------------


def test_AT_P14_health_routes_follow_their_contract(rt, app, monkeypatch):
    """AT-P14: livez solo el proceso; startupz modelo y detectores; readyz base, worker y cola."""
    org = rt.create_org("Organización Secreta SA")
    client = TestClient(app)
    assert client.get("/livez").json() == {"status": "ok"}
    assert client.get("/startupz").json() == {"status": "ok", "checks": {"model": True, "detectors": True}}
    not_ready = client.get("/readyz")
    assert not_ready.status_code == 503
    assert not_ready.json() == {
        "status": "unavailable",
        "checks": {"db": True, "worker": False, "queue": True},
    }
    with TestClient(app) as running:  # el lifespan arranca el worker y la purga
        ready = running.get("/readyz")
        assert ready.status_code == 200 and ready.json()["checks"] == {
            "db": True,
            "worker": True,
            "queue": True,
        }
        for path in ("/livez", "/startupz", "/readyz"):
            body = running.get(path).text
            assert org not in body and "Secreta" not in body
    assert rt.jobs.is_alive() is False, "el lifespan detiene el worker al salir"
    rt.startup_errors = 1
    assert client.get("/startupz").status_code == 503
    rt.startup_errors = 0

    def db_down(*args):
        raise RuntimeError("base caída")

    monkeypatch.setattr(rt, "readiness", lambda: {"db": False, "worker": True, "queue": True})
    assert client.get("/readyz").status_code == 503
    monkeypatch.setattr(rt, "repo", db_down)
    assert client.get("/livez").status_code == 200, "livez nunca chequea dependencias"


def test_AT_P14_metrics_require_token_and_have_no_tenant_labels(rt, app):
    """AT-P14: /metrics con Bearer propio; formato Prometheus sin ids de tenant, usuario ni traza."""
    org = rt.create_org("Acme")
    install_detector(rt, org, constant_detector(rt.model_info, 0.9))
    user = add_user(rt, org, "persona@acme.com.ar", "owner")
    client = TestClient(app)
    resp = chat(client, api_key(rt, org), vigia={})
    trace_id = resp.headers["X-Vigia-Trace-Id"]
    assert client.get("/metrics").status_code == 401
    assert (
        client.get("/metrics", headers={"Authorization": "Bearer otro-token-incorrecto"}).status_code == 401
    )
    ok = client.get("/metrics", headers={"Authorization": f"Bearer {METRICS_TOKEN}"})
    assert ok.status_code == 200 and ok.headers["content-type"].startswith("text/plain; version=0.0.4")
    assert ok.headers["cache-control"] == "no-store"
    text = ok.text
    assert 'vigia_requests_total{route="/v1/chat/completions",status="2xx"} 1' in text
    assert 'vigia_actions_total{action="human_review"} 1' in text
    assert "vigia_auth_failures_total 2" in text
    for secret in (org, trace_id, user.user_id, "persona@acme.com.ar", "Acme"):
        assert secret not in text
    labels = set(re.findall(r"(\w+)=\"", text))
    assert labels <= {"route", "status", "le", "action", "detector_id", "version"}


def test_AT_P14_metrics_disabled_without_token(tmp_path):
    rt = make_runtime(tmp_path)
    try:
        client = TestClient(create_app(runtime=rt))
        assert client.get("/metrics", headers={"Authorization": f"Bearer {METRICS_TOKEN}"}).status_code == 404
    finally:
        rt.close()


# --- AT-P17: errores internos ---------------------------------------------------------------------


def test_AT_P17_internal_errors_are_generic_with_request_id(rt, app, monkeypatch):
    """AT-P17: una excepción interna responde genérico con request_id y sin stack trace."""
    org = rt.create_org("Acme")
    token = api_key(rt, org)
    client = TestClient(app)
    secret = "detalle-interno-/srv/vigia/secreto.py"  # noqa: S105 - texto testigo, no una credencial

    def boom(*args, **kwargs):
        raise RuntimeError(secret)

    monkeypatch.setattr(rt.backend, "stream", boom)
    engine = chat(client, token)  # falla dentro de vigia.api: error OpenAI con request_id
    assert engine.status_code == 500
    err = engine.json()["error"]
    assert err["code"] == "internal_error" and err["request_id"] == engine.headers["x-request-id"]
    monkeypatch.setattr(rt, "context_for", boom)
    dependency = chat(client, token)  # falla en la dependencia: lo ataja el middleware de seguridad
    assert dependency.status_code == 500
    assert dependency.json() == {
        "error": "internal_error",
        "message": "Error interno.",
        "request_id": dependency.headers["x-request-id"],
    }
    monkeypatch.setattr(rt, "readiness", boom)
    health = client.get("/readyz")
    assert health.status_code == 500 and health.json()["request_id"] == health.headers["x-request-id"]
    for r in (engine, dependency, health):
        assert secret not in r.text and "Traceback" not in r.text and "RuntimeError" not in r.text
        assert "content-security-policy" in r.headers


# --- Límites de body ------------------------------------------------------------------------------


def test_body_limit_is_strict_except_on_dataset_upload(rt, app):
    org = rt.create_org("Acme")
    add_user(rt, org, "owner@acme.com.ar", "owner")
    client = TestClient(app)
    big = {"model": "tiny:mixtral", "messages": [{"role": "user", "content": "x" * 5000}]}
    resp = client.post(
        "/v1/chat/completions", json=big, headers={"Authorization": f"Bearer {api_key(rt, org)}"}
    )
    assert resp.status_code == 413
    panel, csrf = login(app, "owner@acme.com.ar")
    assert panel.post("/policies", data={"csrf_token": csrf, "pad": "x" * 5000}).status_code == 413
    content = learnable_jsonl(80)
    assert len(content) > 4096
    up = panel.post(
        "/calibration/datasets",
        data={"csrf_token": csrf, "name": "grande"},
        files={"file": ("grande.jsonl", content, "application/jsonl")},
        follow_redirects=False,
    )
    assert up.status_code == 303, up.text[:300]
    assert [d["n_items"] for d in rt.repo(org).list_datasets()] == [80]


# --- AT-P16: calibración de punta a punta ---------------------------------------------------------


def test_AT_P16_upload_train_activate_and_next_request_uses_new_detector(rt, app):
    """AT-P16: subir JSONL → job → candidato con reporte → activar → el SDK openai ve el nuevo detector."""
    openai = pytest.importorskip("openai")
    org = rt.create_org("Acme")
    add_user(rt, org, "owner@acme.com.ar", "owner")
    install_detector(rt, org, constant_detector(rt.model_info, 0.3))  # detector previo
    previous = rt.detector_for(org).id
    panel, csrf = login(app, "owner@acme.com.ar")
    up = panel.post(
        "/calibration/datasets",
        data={"csrf_token": csrf, "name": "demo"},
        files={"file": ("demo.jsonl", learnable_jsonl(40), "application/jsonl")},
        follow_redirects=False,
    )
    assert up.status_code == 303, up.text[:300]
    dataset_id = rt.repo(org).list_datasets()[0]["id"]
    start = panel.post(
        "/calibration/start", data={"csrf_token": csrf, "dataset_id": dataset_id}, follow_redirects=False
    )
    assert start.status_code == 303, start.text[:300]
    job_id = start.headers["location"].split("/calibration/jobs/")[1].split("?")[0]
    assert panel.get(f"/calibration/jobs/{job_id}/status").json()["state"] == "queued"
    assert rt.jobs.run_until_idle() >= 1
    status = panel.get(f"/calibration/jobs/{job_id}/status").json()
    assert status["state"] == "done", status
    detector_id = status["detector_id"]
    assert rt.repo(org).get_detector(detector_id)["status"] == "candidate"
    report_page = panel.get(f"/calibration/detectors/{detector_id}")
    assert report_page.status_code == 200 and "AUROC" in report_page.text
    sdk = openai.OpenAI(
        base_url="http://testserver/v1", api_key=api_key(rt, org), http_client=TestClient(app)
    )

    def calibration_id():
        out = sdk.chat.completions.create(
            model="tiny:mixtral",
            messages=[{"role": "user", "content": "hola"}],
            max_tokens=3,
            extra_body={"vigia": {}},
        )
        return out.choices[0].model_extra["vigia"]["calibration_id"]

    assert calibration_id() == previous
    act = panel.post(
        f"/calibration/detectors/{detector_id}/activate",
        data={"csrf_token": csrf, "confirm": "yes"},
        follow_redirects=False,
    )
    assert act.status_code == 303, act.text[:300]
    assert calibration_id() == detector_id
    actions = [r["action"] for r in rt.db.query("SELECT action FROM audit_log WHERE org_id = ?", (org,))]
    assert {"dataset_upload", "calibration_start", "calibration_completed", "detector_activate"} <= set(
        actions
    )


# --- CLI ------------------------------------------------------------------------------------------


def cli_env(tmp_path, **extra):
    env = {
        "VIGIA_MODEL": "tiny:mixtral",
        "VIGIA_DATA_DIR": str(tmp_path),
        "VIGIA_PEPPERS": "k1:" + "p" * 40,
    }
    env.update(extra)
    return env


def test_cli_init_reads_password_from_file_and_creates_owner(tmp_path, capsys):
    secret = tmp_path / "owner.txt"
    secret.write_text(PASSWORD + "\n", encoding="utf-8")
    env = cli_env(tmp_path, VIGIA_OWNER_PASSWORD_FILE=str(secret))
    assert main(["init", "--org", "Acme", "--email", "duena@acme.com.ar"], environ=env) == 0
    out = capsys.readouterr().out
    org_id = re.search(r"Organización creada: ([0-9a-f]{32})", out).group(1)
    db = Database(tmp_path / "vigia.db")
    try:
        user = lookup_user_by_email(db, "duena@acme.com.ar")
        assert user["org_id"] == org_id and user["role"] == "owner"
        assert PASSWORD not in out and PASSWORD not in user["password_hash"]
        actions = [r["action"] for r in db.query("SELECT action FROM audit_log WHERE org_id = ?", (org_id,))]
        assert actions == ["org_create", "user_create"]
    finally:
        db.close()
    secret.write_text("corta", encoding="utf-8")
    assert main(["init", "--org", "Otra", "--email", "otra@acme.com.ar"], environ=env) == 2
    assert "15 caracteres" in capsys.readouterr().err
    db = Database(tmp_path / "vigia.db")
    try:
        assert db.query_one("SELECT COUNT(*) AS n FROM orgs")["n"] == 1, (
            "un owner inválido no deja una org huérfana"
        )
    finally:
        db.close()


def test_cli_init_asks_password_with_getpass(tmp_path, monkeypatch, capsys):
    answers = iter([PASSWORD, PASSWORD + " distinta"])
    monkeypatch.setattr(appmod.getpass, "getpass", lambda prompt: next(answers))
    assert main(["init", "--org", "Acme", "--email", "a@acme.com.ar"], environ=cli_env(tmp_path)) == 2
    assert "no coinciden" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(
            ["init", "--org", "Acme", "--email", "a@acme.com.ar", "--password", "x"],
            environ=cli_env(tmp_path),
        )


def test_cli_create_key_prints_token_once(tmp_path, capsys):
    secret = tmp_path / "owner.txt"
    secret.write_text(PASSWORD, encoding="utf-8")
    env = cli_env(tmp_path, VIGIA_OWNER_PASSWORD_FILE=str(secret))
    main(["init", "--org", "Acme", "--email", "a@acme.com.ar"], environ=env)
    org_id = re.search(r"Organización creada: (\w+)", capsys.readouterr().out).group(1)
    assert (
        main(["create-key", "--org-id", org_id, "--name", "prod", "--scopes", "ingest,read"], environ=env)
        == 0
    )
    out = capsys.readouterr().out
    tokens = re.findall(r"vg_live_\w+", out)
    assert len(tokens) == 1
    db = Database(tmp_path / "vigia.db")
    try:
        from vigia_ee.auth import AuthState

        principal = verify_api_key(AuthState(db=db, peppers=PEPPERS), tokens[0])
        assert principal.org_id == org_id and principal.scopes == {"ingest", "read"}
        assert tokens[0] not in json.dumps(db.query("SELECT * FROM api_keys"))
    finally:
        db.close()
    assert main(["create-key", "--org-id", "f" * 32, "--name", "x"], environ=env) == 2
    assert "no existe" in capsys.readouterr().err
    assert main(["create-key", "--org-id", org_id, "--name", "x", "--scopes", "root"], environ=env) == 2
    assert main(["create-key", "--org-id", org_id, "--name", "x"], environ={}) == 2
    assert "VIGIA_MODEL" in capsys.readouterr().err


def test_cli_serve_hides_server_header_and_binds_localhost(tmp_path, monkeypatch):
    import uvicorn

    seen = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: seen.update(kw, app=app))
    monkeypatch.setattr(appmod, "configure_json_logging", lambda: None)  # no tocar el logging global
    assert main(["serve", "--port", "8123"], environ=cli_env(tmp_path)) == 0
    assert seen["server_header"] is False and seen["host"] == "127.0.0.1" and seen["port"] == 8123
    assert seen["proxy_headers"] is False and seen["workers"] == 1
    assert undeclared_routes(seen["app"]) == []
    seen["app"].state.runtime.close()
