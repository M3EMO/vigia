"""Composición de la app: permisos, health, métricas, errores, límites, calibración de punta a punta y CLI."""

import json
import os
import re
import signal
import stat

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
    assert client.get("/startupz").json() == {
        "status": "ok",
        "checks": {"model": True, "detectors": True},
        "detector_errors": 0,
    }
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
    rt.startup_errors = 1  # un detector roto de una org no deja al pod reiniciando en bucle
    started = client.get("/startupz")
    assert started.status_code == 200 and started.json()["detector_errors"] == 1
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
    assert dependency.json() == {  # bajo /v1, el sobre de OpenAI (mismo formato que vigia.api)
        "error": {
            "message": "Error interno.",
            "type": "server_error",
            "code": "internal_error",
            "param": None,
            "request_id": dependency.headers["x-request-id"],
        }
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


def test_routing_and_body_limit_errors_render_the_panel_page_but_v1_keeps_the_openai_envelope(rt, app):
    """Regresión (R5): create_app no registraba la página de error del panel. Un 404/405 del ruteo o el
    413 del middleware llegaban al navegador como JSON crudo. /v1 sigue con el sobre de OpenAI."""
    org = rt.create_org("Acme")
    add_user(rt, org, "owner@acme.com.ar", "owner")
    panel, csrf = login(app, "owner@acme.com.ar")
    nav = {"Accept": "text/html,application/xhtml+xml"}
    for path, status, back in (
        ("/settings/keys", 405, 'href="/settings"'),
        ("/calibration/datasets", 405, 'href="/calibration"'),
        ("/generaciones", 404, None),
    ):
        resp = panel.get(path, headers=nav)
        assert resp.status_code == status and resp.headers["content-type"].startswith("text/html"), path
        assert 'id="vg-nav"' in resp.text and 'role="alert"' in resp.text, path
        assert back is None or back in resp.text, path
    big = panel.post("/policies", data={"csrf_token": csrf, "pad": "x" * 5000}, headers=nav)
    assert big.status_code == 413 and big.headers["content-type"].startswith("text/html")
    assert "supera" in big.text and 'href="/policies"' in big.text  # vuelve al formulario
    assert panel.get("/generaciones").json()["error"] == "not_found"  # un fetch sigue en JSON
    v1 = panel.post(
        "/v1/chat/completions",
        json={"model": "tiny:mixtral", "messages": [{"role": "user", "content": "x" * 5000}]},
        headers=nav,
    )
    assert v1.status_code == 413 and v1.json()["error"]["code"] == "payload_too_large"
    missing = panel.get("/v1/no-existe", headers=nav)
    assert missing.status_code == 404 and missing.json()["error"]["code"] == "not_found"


def test_deeply_nested_dataset_upload_is_422_not_500(rt, app):
    """Regresión: 'Subida de dataset de calibración con JSON profundamente anidado lanza
    RecursionError no capturada y devuelve HTTP 500 en vez de 422' (AT-P16, AT-P17)."""
    org = rt.create_org("Acme")
    add_user(rt, org, "owner@acme.com.ar", "owner")
    panel, csrf = login(app, "owner@acme.com.ar")
    prefix = '{"id":"a","question":"q","response":"r","label":0,"x":'
    deep = (prefix + "[" * 100_000 + "]" * 100_000 + "}").encode()
    up = panel.post(
        "/calibration/datasets",
        data={"csrf_token": csrf, "name": "hondo"},
        files={"file": ("hondo.jsonl", deep, "application/jsonl")},
        follow_redirects=False,
    )
    assert up.status_code == 422, up.text[:300]
    assert "anidamiento excesivo" in up.text
    assert rt.repo(org).list_datasets() == []


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


def _capture_server(monkeypatch):
    """Reemplaza `uvicorn.Server.run`: guarda el servidor armado por `serve` sin abrir el puerto."""
    import uvicorn

    seen = {}
    monkeypatch.setattr(uvicorn.Server, "run", lambda self, sockets=None: seen.update(server=self))
    monkeypatch.setattr(appmod, "configure_json_logging", lambda: None)  # no tocar el logging global
    return seen


def test_cli_serve_hides_server_header_and_binds_localhost(tmp_path, monkeypatch):
    seen = _capture_server(monkeypatch)
    assert main(["serve", "--port", "8123"], environ=cli_env(tmp_path)) == 0
    config = seen["server"].config
    assert config.server_header is False and config.host == "127.0.0.1" and config.port == 8123
    assert config.proxy_headers is False and config.workers == 1
    assert undeclared_routes(config.app) == []
    config.app.state.runtime.close()


def test_cli_serve_bounds_shutdown_and_rejects_new_generations(tmp_path, monkeypatch):
    """Regresión: 'Ctrl+C espera sin tope a todas las generaciones en cola y hasta arranca
    generaciones nuevas después de la señal'. El cierre tiene tope (VIGIA_SHUTDOWN_TIMEOUT) y la
    señal marca el runtime: la API responde 503 y el playground no genera."""
    import signal

    seen = _capture_server(monkeypatch)
    env = cli_env(tmp_path, VIGIA_SHUTDOWN_TIMEOUT="7")
    assert main(["serve", "--port", "8123"], environ=env) == 0
    server = seen["server"]
    app = server.config.app
    rt = app.state.runtime
    try:
        assert server.config.timeout_graceful_shutdown == 7
        rt.auth.scrypt_n = 2**10  # scrypt barato: el test no mide el costo del hash
        org = rt.create_org("Acme")
        token, _ = create_api_key(rt.auth, org, "k", ["ingest"])
        owner = add_user(rt, org, "o@acme.com.ar", "owner")
        client = TestClient(app)
        body = {"model": "tiny:mixtral", "messages": [{"role": "user", "content": "hola"}], "max_tokens": 2}
        headers = {"Authorization": f"Bearer {token}"}
        assert client.post("/v1/chat/completions", json=body, headers=headers).status_code == 200
        assert not rt.shutting_down
        server.handle_exit(signal.SIGINT, None)
        assert rt.shutting_down and server.should_exit
        assert client.post("/v1/chat/completions", json=body, headers=headers).status_code == 503
        events = list(rt.playground_stream(owner, [{"role": "user", "content": "hola"}], max_tokens=2))
        assert [e["type"] for e in events] == ["error"] and "cerrando" in events[0]["message"]
    finally:
        rt.close()


_STOP_SIGNALS = [name for name in ("SIGINT", "SIGTERM", "SIGBREAK") if hasattr(signal, name)]


@pytest.mark.parametrize("name", _STOP_SIGNALS)
def test_cli_serve_exits_ok_after_an_orderly_stop_signal(tmp_path, monkeypatch, name):
    """Regresión: 'Un cierre ordenado con Ctrl+C (SIGINT) termina 'serve' con código de salida 1, como
    si fuera un error' y 'Un cierre ordenado por SIGTERM (STOPSIGNAL del Dockerfile) o Ctrl+Break no
    sale con código 0: uvicorn vuelve a lanzar la señal y el proceso muere por el manejador por
    defecto'. Corre el `capture_signals` real de uvicorn: la señal no se relanza al salir y `serve`
    devuelve 0. Un manejador de prueba registra lo relanzado, así la señal nunca mata a pytest."""
    import uvicorn

    sig = getattr(signal, name)
    relaunched = []
    previous = signal.signal(sig, lambda s, f: relaunched.append(s))
    seen = {}

    def run(self, sockets=None):  # el ciclo real de señales de uvicorn, sin abrir el puerto
        seen.update(server=self)
        with self.capture_signals():
            self.handle_exit(sig, None)  # llega la señal de parada

    monkeypatch.setattr(uvicorn.Server, "run", run)
    monkeypatch.setattr(appmod, "configure_json_logging", lambda: None)
    try:
        assert main(["serve", "--port", "8123"], environ=cli_env(tmp_path)) == 0
        assert seen["server"].config.app.state.runtime.shutting_down
        assert relaunched == []
    finally:
        signal.signal(sig, previous)
        seen["server"].config.app.state.runtime.close()


def test_cli_serve_keyboard_interrupt_without_shutdown_is_an_error(tmp_path, monkeypatch):
    """Un KeyboardInterrupt que no viene de un cierre ordenado (runtime sin marcar) sigue siendo error."""
    import uvicorn

    seen = {}

    def run(self, sockets=None):
        seen.update(server=self)
        raise KeyboardInterrupt

    monkeypatch.setattr(uvicorn.Server, "run", run)
    monkeypatch.setattr(appmod, "configure_json_logging", lambda: None)
    try:
        assert main(["serve", "--port", "8123"], environ=cli_env(tmp_path)) == 1
    finally:
        seen["server"].config.app.state.runtime.close()


def test_cli_serve_routes_uvicorn_and_asyncio_logs_to_json(tmp_path, monkeypatch, capsys):
    """Regresión: 'Los logs de uvicorn y asyncio salen en texto plano, fuera del formato JSON, con
    stack traces'."""
    import io
    import logging

    from vigia_ee.security import configure_json_logging as real_configure

    stream = io.StringIO()
    names = ("vigia", *appmod.FOREIGN_LOGGERS)
    saved = {n: (list(logging.getLogger(n).handlers), logging.getLogger(n).propagate) for n in names}
    for n in names:
        logging.getLogger(n).handlers = []
    try:
        seen = _capture_server(monkeypatch)
        monkeypatch.setattr(appmod, "configure_json_logging", lambda: real_configure(stream=stream))
        assert main(["serve", "--port", "8123"], environ=cli_env(tmp_path)) == 0
        seen["server"].config.app.state.runtime.close()
        logging.getLogger("uvicorn.error").warning("Invalid HTTP request received.")
        try:
            raise AssertionError("detalle interno")
        except AssertionError:
            logging.getLogger("asyncio").error("Exception in callback x", exc_info=True)
        lines = [ln for ln in stream.getvalue().splitlines() if ln.strip()]
        records = [json.loads(ln) for ln in lines]  # cada línea es un objeto JSON
        assert [r["logger"] for r in records[-2:]] == ["uvicorn.error", "asyncio"]
        assert records[-2]["msg"] == "Invalid HTTP request received."
        assert "Traceback" not in stream.getvalue() and "detalle interno" not in stream.getvalue()
        assert "Invalid HTTP request" not in capsys.readouterr().err
    finally:
        for n, (handlers, propagate) in saved.items():
            logging.getLogger(n).handlers = handlers
            logging.getLogger(n).propagate = propagate


def _storage_env(tmp_path, case):
    """Arma cada caso de almacenamiento roto y devuelve (env, variable que el mensaje debe nombrar)."""
    secret = tmp_path / "owner.txt"
    secret.write_text(PASSWORD, encoding="utf-8")
    if case == "missing_parent":
        missing = tmp_path / "no-existe.txt"
        missing.write_text("x", encoding="utf-8")  # un archivo como carpeta: mkdir falla (OSError)
        env = cli_env(tmp_path, VIGIA_DATA_DIR=str(missing / "sub"), VIGIA_OWNER_PASSWORD_FILE=str(secret))
        return env, "VIGIA_DATA_DIR"
    if case == "not_sqlite":
        bad = tmp_path / "basura.db"
        bad.write_bytes(b"esto no es una base SQLite " * 64)
        env = cli_env(tmp_path, VIGIA_DB_PATH=str(bad), VIGIA_OWNER_PASSWORD_FILE=str(secret))
        return env, "VIGIA_DB_PATH"
    assert case == "readonly"
    data = tmp_path / "ro"
    env = cli_env(data, VIGIA_OWNER_PASSWORD_FILE=str(secret))
    assert main(["init", "--org", "A", "--email", "a@acme.com.ar"], environ=env) == 0
    for extra in data.glob("vigia.db-*"):
        extra.unlink()
    os.chmod(data / "vigia.db", stat.S_IREAD)
    return env, "VIGIA_DATA_DIR"


@pytest.mark.parametrize("case", ["missing_parent", "not_sqlite", "readonly"])
@pytest.mark.parametrize("command", ["init", "create-key", "serve"])
def test_cli_storage_errors_exit_2_with_clear_message(tmp_path, monkeypatch, capsys, case, command):
    """Regresión: 'Un error de almacenamiento al arrancar (carpeta inválida, base de solo lectura o
    corrupta) sale como traceback crudo en vez de un mensaje claro'."""
    import uvicorn

    monkeypatch.setattr(
        uvicorn.Server, "run", lambda self, sockets=None: pytest.fail("serve no debe arrancar")
    )
    monkeypatch.setattr(appmod, "configure_json_logging", lambda: None)
    env, variable = _storage_env(tmp_path, case)
    capsys.readouterr()
    argv = {
        "init": ["init", "--org", "B", "--email", "b@acme.com.ar"],
        "create-key": ["create-key", "--org-id", "f" * 32, "--name", "x"],
        "serve": ["serve", "--port", "8123"],
    }[command]
    try:
        assert main(argv, environ=env) == 2
    finally:
        for db_file in tmp_path.rglob("vigia.db"):
            os.chmod(db_file, stat.S_IREAD | stat.S_IWRITE)
    err = capsys.readouterr().err
    assert err.startswith("error: ") and variable in err and "Traceback" not in err
    if case == "readonly":
        assert "solo lectura" in err
    if case == "not_sqlite":
        assert "no es una base SQLite" in err


@pytest.mark.parametrize("variable", ["VIGIA_MODEL", "VIGIA_BASE_DETECTOR"])
def test_cli_serve_bad_model_or_base_detector_exits_2(tmp_path, monkeypatch, capsys, variable):
    """Regresión: 'VIGIA_MODEL inválido (ruta o repo inexistente) aborta el arranque con traceback
    crudo y sin nombrar la variable' y 'VIGIA_BASE_DETECTOR inexistente o ilegible no frena el
    arranque y /startupz dice detectors:true'."""
    _capture_server(monkeypatch)
    missing = tmp_path / "no-existe"
    env = cli_env(tmp_path / "datos", VIGIA_BACKEND="hf", **{variable: str(missing)})
    capsys.readouterr()
    assert main(["serve", "--port", "8123"], environ=env) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: ") and f"{variable}={missing}" in err and "Traceback" not in err
