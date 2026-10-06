"""Verificación (ronda 1): aislamiento entre tenants y matriz rol/scope -> permiso.

Complementa `test_isolation.py` (que barre solo con el owner): acá cada rol del panel y cada scope de
API key de la org A intenta leer y mutar recursos de la org B. Reglas que se fijan:

- Con ids ajenos e ids inexistentes la respuesta es la misma: 404 si el rol tiene el permiso, 403 si
  no lo tiene; nunca 200 ni datos de B (ADR A-15).
- La matriz sale de `vigia_ee.auth.PERMISSIONS`: viewer no muta, reviewer solo revisa y usa el
  playground, admin no toca `settings:write` ni `data:delete`, el último owner queda protegido.
- Exports, métricas, jobs, webhooks, políticas, rate limit por tenant y cache de detectores por org.
- Cambiar el rol o deshabilitar una cuenta invalida sus sesiones.
"""

from __future__ import annotations

import json
import re
import secrets

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from vigia_platform_support import (
    PASSWORD,
    add_user,
    constant_detector,
    install_detector,
    learnable_jsonl,
    make_runtime,
)

from vigia_ee import webhooks
from vigia_ee.app import create_app
from vigia_ee.auth import PERMISSIONS, ROLE_RANK, create_api_key, has_permission, route_permission

ROLES = ("owner", "admin", "reviewer", "viewer")
SCOPES = ("ingest", "read", "admin")
CSRF_RE = re.compile(r'name="csrf_token" value="([0-9a-f]{64})"')
RID_RE = re.compile(r"[0-9a-f]{16}")
NONCE_RE = re.compile(r'nonce="[^"]+"')
CSRF_VALUE_RE = re.compile(r'value="[0-9a-f]{64}"')
METRICS_TOKEN = "metrics-token-verify-0001"  # noqa: S105 - token de prueba
FORM = {
    "confirm": "yes",
    "role": "viewer",
    "label": "correct",
    "disabled": "1",
    "target": "response",
    "min_reviewers": "1",
}


def seed(rt, tag):
    marker = f"MARCA-{tag}-{secrets.token_hex(3)}"
    org = rt.create_org(f"Org {tag} {marker}")
    rt.repo(org).update_org(store_text="full")
    users = {role: add_user(rt, org, f"{role}-{tag.lower()}@{tag.lower()}.com.ar", role) for role in ROLES}
    owner = users["owner"]
    install_detector(rt, org, constant_detector(rt.model_info, 0.9 if tag == "A" else 0.8))
    done = [
        e
        for e in rt.playground_stream(
            owner, [{"role": "user", "content": f"Consulta {marker}"}], max_tokens=4
        )
        if e["type"] == "done"
    ][0]
    repo = rt.repo(org)
    item = repo.list_review_items()[0]
    dataset = rt.calibration.upload_dataset(owner, f"dataset {marker}", learnable_jsonl(20, prefix=marker))
    job = rt.calibration.start(owner, dataset)
    candidate = constant_detector(rt.model_info, 0.61 if tag == "A" else 0.71)
    repo.insert_detector(candidate.id, candidate.to_json())
    keys = {scope: create_api_key(rt.auth, org, f"key {scope} {marker}", [scope]) for scope in SCOPES}
    webhook = repo.insert_webhook(
        f"https://hooks-{tag.lower()}.example/recibir", "whsec_" + "A" * 43, ["trace.flagged"]
    )
    return {
        "org": org,
        "marker": marker,
        "users": users,
        "emails": {role: f"{role}-{tag.lower()}@{tag.lower()}.com.ar" for role in ROLES},
        "keys": keys,
        "dataset": dataset,
        "ids": {
            "trace_id": done["trace_id"],
            "item_id": item["id"],
            "job_id": job,
            "detector_id": candidate.id,
            "user_id": users["viewer"].user_id,
            "key_id": keys["read"][1],
            "webhook_id": webhook,
        },
    }


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    rt = make_runtime(
        tmp_path_factory.mktemp("verify-isolation"),
        metrics_token=METRICS_TOKEN,
        key_rate_per_minute=100_000,
        tenant_rate_per_minute=100_000,
    )
    a = seed(rt, "A")
    b = seed(rt, "B")
    app = create_app(runtime=rt)
    yield rt, app, a, b
    rt.close()


class Panel:
    def __init__(self, app, email):
        self.client = TestClient(app, base_url="https://testserver")
        resp = self.client.post(
            "/login", data={"email": email, "password": PASSWORD, "next": "/"}, follow_redirects=False
        )
        assert resp.status_code == 303, resp.text[:300]
        self.csrf = CSRF_RE.search(self.client.get("/account").text).group(1)

    def get(self, path):
        return self.client.get(path, follow_redirects=False)

    def post(self, path, data=None):
        return self.client.post(path, data={"csrf_token": self.csrf, **(data or {})}, follow_redirects=False)

    def request(self, method, path, data=None):
        return self.get(path) if method == "GET" else self.post(path, data)


def routes(app):
    for route in app.routes:
        if isinstance(route, APIRoute):
            for method in sorted(route.methods - {"HEAD"}):
                yield route, method, route_permission(route)


def params_of(path):
    return re.findall(r"\{([^}:]+)\}", path)


def fill(path, ids):
    return re.sub(r"\{([^}:]+)\}", lambda m: ids[m.group(1)], path)


def missing_ids():
    return {
        "trace_id": secrets.token_hex(16),
        "item_id": secrets.token_hex(16),
        "job_id": secrets.token_hex(16),
        "detector_id": secrets.token_hex(8),
        "user_id": secrets.token_hex(16),
        "key_id": secrets.token_hex(8),
        "webhook_id": secrets.token_hex(16),
    }


def normalized(resp, path):
    text = resp.text.replace(path, "P")
    return RID_RE.sub("R", NONCE_RE.sub("N", CSRF_VALUE_RE.sub("C", text)))


def leaks(rt, b):
    values = [b["marker"], b["org"], *b["emails"].values(), *b["ids"].values(), b["dataset"]]
    values += [k[1] for k in b["keys"].values()] + [u.user_id for u in b["users"].values()]
    values.append(rt.repo(b["org"]).get_trace(b["ids"]["trace_id"])["response_text"])
    return [v for v in values if v and len(v) >= 6]


def assert_no_leak(label, resp, values, allowed=()):
    for value in values:
        assert value in allowed or value not in resp.text, f"{label} filtró {value[:24]!r}"


def api(app, token, method, path):
    client = TestClient(app)
    headers = {"Authorization": f"Bearer {token}"}
    if method == "POST":
        body = {"model": "tiny:mixtral", "messages": [{"role": "user", "content": "hola"}], "max_tokens": 1}
        if path.endswith("/completions") and "chat" not in path:
            body = {"model": "tiny:mixtral", "prompt": "hola", "max_tokens": 1}
        return client.post(path, headers=headers, json=body)
    return client.request(method, path, headers=headers)


# --- barrido rol x ruta --------------------------------------------------------------------------


@pytest.mark.parametrize("role", ROLES)
def test_every_role_gets_identical_404_or_403_for_foreign_ids(world, role):
    """Cada rol de A: ids de B e ids inexistentes dan la misma respuesta (404 con permiso, 403 sin)."""
    rt, app, a, b = world
    panel = Panel(app, a["emails"][role])
    values = leaks(rt, b)
    missing = missing_ids()
    swept = 0
    for route, method, spec in routes(app):
        names = params_of(route.path)
        if spec.kind != "session" or not names or route.path == "/static/{name}":
            continue
        other_path, absent_path = fill(route.path, b["ids"]), fill(route.path, missing)
        other = panel.request(method, other_path, FORM)
        absent = panel.request(method, absent_path, FORM)
        expected = 404 if has_permission(role, spec.permission) else 403
        label = f"{role} {method} {route.path}"
        assert (other.status_code, absent.status_code) == (expected, expected), (label, other.status_code)
        assert normalized(other, other_path) == normalized(absent, absent_path), label
        assert_no_leak(label, other, values, allowed=[b["ids"][n] for n in names])
        swept += 1
    assert swept >= 15


@pytest.mark.parametrize("scope", SCOPES)
def test_every_key_scope_gets_identical_answer_for_foreign_ids(world, scope):
    """Cada scope de key de A: la traza de B y una inexistente responden igual (404 con scope, 403 sin)."""
    rt, app, a, b = world
    token = a["keys"][scope][0]
    values = leaks(rt, b)
    missing = missing_ids()
    checked = 0
    for route, method, spec in routes(app):
        if spec.kind != "api_key":
            continue
        has = scope in ("admin", spec.scope)
        names = params_of(route.path)
        if names:
            other_path, absent_path = fill(route.path, b["ids"]), fill(route.path, missing)
            other = api(app, token, method, other_path)
            absent = api(app, token, method, absent_path)
            expected = 404 if has else 403
            assert (other.status_code, absent.status_code) == (expected, expected), (scope, route.path)
            assert normalized(other, other_path) == normalized(absent, absent_path)
            assert_no_leak(f"key {scope} {route.path}", other, values, allowed=[b["ids"][n] for n in names])
        else:
            resp = api(app, token, method, route.path)
            assert resp.status_code == (200 if has else 403), (scope, method, route.path, resp.status_code)
            assert_no_leak(f"key {scope} {route.path}", resp, values)
        checked += 1
    assert checked >= 4


def test_own_trace_is_readable_only_with_read_scope(world):
    rt, app, a, b = world
    path = f"/v1/vigia/traces/{a['ids']['trace_id']}"
    assert api(app, a["keys"]["read"][0], "GET", path).status_code == 200
    assert api(app, a["keys"]["admin"][0], "GET", path).status_code == 200
    assert api(app, a["keys"]["ingest"][0], "GET", path).status_code == 403
    assert api(app, b["keys"]["read"][0], "GET", path).status_code == 404


@pytest.mark.parametrize("role", ROLES)
def test_pages_without_params_follow_the_permission_matrix(world, role):
    """GET sin path params: 200 con permiso y 403 sin él, sin datos de B. POST sin permiso: 403."""
    rt, app, a, b = world
    panel = Panel(app, a["emails"][role])
    values = leaks(rt, b)
    for route, method, spec in routes(app):
        if spec.kind != "session" or params_of(route.path) or route.path == "/logout":
            continue
        allowed = has_permission(role, spec.permission)
        if method == "GET":
            resp = panel.get(route.path)
            assert resp.status_code == (200 if allowed else 403), (role, route.path, resp.status_code)
            assert_no_leak(f"{role} GET {route.path}", resp, values)
        elif not allowed:
            resp = panel.post(route.path, {})
            assert resp.status_code == 403, (role, route.path, resp.status_code)


def test_permission_matrix_matches_the_documented_roles():
    """viewer no muta; reviewer solo revisa y usa playground; admin no toca settings:write ni data:delete."""
    reviewer_only = {
        p for p in PERMISSIONS if has_permission("reviewer", p) and not has_permission("viewer", p)
    }
    assert reviewer_only == {"review:write", "playground:use"}
    owner_only = {p for p in PERMISSIONS if not has_permission("admin", p)}
    assert owner_only == {"settings:write", "data:delete"}
    assert all(has_permission("owner", p) for p in PERMISSIONS)
    assert ROLE_RANK == {"viewer": 0, "reviewer": 1, "admin": 2, "owner": 3}


# --- exports, métricas, jobs ----------------------------------------------------------------------


@pytest.mark.parametrize("role", ["owner", "admin"])
def test_exports_contain_only_own_tenant(world, role):
    rt, app, a, b = world
    panel = Panel(app, a["emails"][role])
    values = leaks(rt, b)
    resp = panel.get("/data/export")
    assert resp.status_code == 200
    assert_no_leak("data export", resp, values)
    data = json.loads(resp.text)
    assert a["ids"]["trace_id"] in {t["id"] for t in data["traces"]}
    tables = (
        "users",
        "api_keys",
        "traces",
        "review_items",
        "detectors",
        "datasets",
        "webhooks",
        "jobs",
        "audit_log",
    )
    assert {row["org_id"] for name in tables for row in data[name]} == {a["org"]}
    for fmt in ("json", "csv"):
        resp = panel.get(f"/audit/export?formato={fmt}")
        assert resp.status_code == 200
        assert_no_leak(f"audit {fmt}", resp, values)
    events = json.loads(panel.get("/audit/export?formato=json").text)["events"]
    assert events and {e["org_id"] for e in events} == {a["org"]}


def test_metrics_and_health_carry_no_tenant_data(world):
    rt, app, a, b = world
    client = TestClient(app)
    values = leaks(rt, a) + leaks(rt, b)
    for path in ("/livez", "/startupz", "/readyz"):
        assert_no_leak(path, client.get(path), values)
    resp = client.get("/metrics", headers={"Authorization": f"Bearer {METRICS_TOKEN}"})
    assert resp.status_code == 200
    assert "org_id" not in resp.text
    assert_no_leak("/metrics", resp, [v for v in values if not re.fullmatch(r"[0-9a-f]{16}", v)])


def test_jobs_and_webhooks_never_cross_tenants(world):
    rt, app, a, b = world
    assert rt.jobs.get(a["org"], b["ids"]["job_id"]) is None
    assert rt.calibration.job(a["org"], b["ids"]["job_id"]) is None
    jobs = webhooks.emit(rt.db, rt.jobs, a["org"], "trace.flagged", {"trace_id": a["ids"]["trace_id"]})
    assert jobs
    for job_id in jobs:
        job = rt.jobs.get(a["org"], job_id)
        assert job["org_id"] == a["org"]
        assert job["payload"]["webhook_id"] == a["ids"]["webhook_id"]
        assert rt.jobs.get(b["org"], job_id) is None
    assert all(j["org_id"] == b["org"] for j in rt.jobs.list(b["org"]))


def test_policy_change_in_one_tenant_does_not_reach_the_other(world):
    rt, app, a, b = world
    owner_a = a["users"]["owner"]
    rt.save_policy(owner_a, {"abstain_at": 0.0, "abstain_message": "ABSTENCION A"})
    try:
        out_a = api(app, a["keys"]["ingest"][0], "POST", "/v1/chat/completions")
        assert out_a.status_code == 200
        body = {
            "model": "tiny:mixtral",
            "messages": [{"role": "user", "content": "hola"}],
            "max_tokens": 2,
            "vigia": {},
        }
        auth = {"Authorization": f"Bearer {b['keys']['ingest'][0]}"}
        out_b = TestClient(app).post("/v1/chat/completions", headers=auth, json=body).json()
        assert out_b["choices"][0]["vigia"]["action"] != "abstain"
        auth = {"Authorization": f"Bearer {a['keys']['ingest'][0]}"}
        out_a = TestClient(app).post("/v1/chat/completions", headers=auth, json=body).json()
        assert out_a["choices"][0]["vigia"]["action"] == "abstain"
    finally:
        rt.save_policy(owner_a, {"abstain_at": None})


# --- cache de detectores por org ------------------------------------------------------------------


def _calibration_id(app, token):
    body = {
        "model": "tiny:mixtral",
        "messages": [{"role": "user", "content": "hola"}],
        "max_tokens": 2,
        "vigia": {},
    }
    resp = TestClient(app).post(
        "/v1/chat/completions", headers={"Authorization": f"Bearer {token}"}, json=body
    )
    assert resp.status_code == 200, resp.text[:200]
    return resp.json()["choices"][0]["vigia"]["calibration_id"]


def test_detector_cache_is_per_tenant_even_with_the_same_artifact(tmp_path):
    """El mismo artefacto (mismo id) en dos orgs: activarlo en A nunca puntúa los requests de B."""
    rt = make_runtime(tmp_path, key_rate_per_minute=100_000, tenant_rate_per_minute=100_000)
    try:
        app = create_app(runtime=rt)
        org_a, org_b = rt.create_org("A"), rt.create_org("B")
        shared = constant_detector(rt.model_info, 0.9)
        other = constant_detector(rt.model_info, 0.6)
        for org in (org_a, org_b):
            rt.repo(org).insert_detector(shared.id, shared.to_json())
        rt.repo(org_a).insert_detector(other.id, other.to_json())
        key_a = create_api_key(rt.auth, org_a, "a", ["ingest"])[0]
        key_b = create_api_key(rt.auth, org_b, "b", ["ingest"])[0]
        owner_a = add_user(rt, org_a, "owner-a@a.com.ar", "owner")
        owner_b = add_user(rt, org_b, "owner-b@b.com.ar", "owner")
        assert _calibration_id(app, key_b) is None
        rt.calibration.activate(owner_a, shared.id)
        assert _calibration_id(app, key_a) == shared.id
        assert _calibration_id(app, key_b) is None, "el detector de A puntuó un request de B"
        rt.calibration.activate(owner_b, shared.id)
        rt.calibration.activate(owner_a, other.id)
        assert _calibration_id(app, key_a) == other.id
        assert _calibration_id(app, key_b) == shared.id
        assert rt.repo(org_b).get_detector(shared.id)["status"] == "active"
        assert rt.repo(org_a).get_detector(shared.id)["status"] == "retired"
    finally:
        rt.close()


def test_tenant_rate_limit_is_per_tenant(tmp_path):
    rt = make_runtime(tmp_path, key_rate_per_minute=100_000, tenant_rate_per_minute=2)
    try:
        app = create_app(runtime=rt)
        org_a, org_b = rt.create_org("A"), rt.create_org("B")
        key_a = create_api_key(rt.auth, org_a, "a", ["ingest"])[0]
        key_b = create_api_key(rt.auth, org_b, "b", ["ingest"])[0]
        codes = [api(app, key_a, "GET", "/v1/models").status_code for _ in range(4)]
        assert codes[-1] == 429
        assert api(app, key_b, "GET", "/v1/models").status_code == 200
    finally:
        rt.close()


# --- sesiones y último owner ----------------------------------------------------------------------


@pytest.fixture
def org_env(tmp_path):
    rt = make_runtime(tmp_path)
    org = rt.create_org("Solo")
    users = {role: add_user(rt, org, f"{role}@solo.com.ar", role) for role in ROLES}
    app = create_app(runtime=rt)
    yield rt, app, org, users
    rt.close()


def test_role_change_and_disable_invalidate_sessions(org_env):
    rt, app, org, users = org_env
    owner = Panel(app, "owner@solo.com.ar")
    admin = Panel(app, "admin@solo.com.ar")
    viewer = Panel(app, "viewer@solo.com.ar")
    assert admin.get("/settings").status_code == 200
    assert (
        owner.post(f"/settings/users/{users['admin'].user_id}/role", {"role": "reviewer"}).status_code == 303
    )
    resp = admin.get("/review")
    assert resp.status_code == 303 and resp.headers["location"].startswith("/login")
    assert (
        owner.post(f"/settings/users/{users['viewer'].user_id}/disable", {"disabled": "1"}).status_code == 303
    )
    assert viewer.get("/").status_code == 303
    assert owner.get("/settings").status_code == 200, "el owner que actuó sigue con sesión"


def test_last_owner_is_protected_and_admin_cannot_touch_owners(org_env):
    rt, app, org, users = org_env
    owner_id = users["owner"].user_id
    owner = Panel(app, "owner@solo.com.ar")
    admin = Panel(app, "admin@solo.com.ar")
    assert owner.post(f"/settings/users/{owner_id}/role", {"role": "admin"}).status_code == 422
    assert owner.post(f"/settings/users/{owner_id}/disable", {"disabled": "1"}).status_code == 422
    assert admin.post(f"/settings/users/{owner_id}/role", {"role": "viewer"}).status_code == 403
    assert admin.post(f"/settings/users/{owner_id}/disable", {"disabled": "1"}).status_code == 403
    assert admin.post(f"/settings/users/{users['admin'].user_id}/role", {"role": "owner"}).status_code == 403
    assert admin.post("/settings/users", {"email": "otro@solo.com.ar", "role": "owner"}).status_code == 403
    assert admin.post("/settings/org", {"store_text": "none", "retention_days": "1"}).status_code == 403
    assert admin.post("/data/delete", {"confirm": "yes", "confirm_name": "Solo"}).status_code == 403
    repo = rt.repo(org)
    assert repo.get_user(owner_id)["role"] == "owner" and not repo.get_user(owner_id)["disabled"]
    assert repo.get_org()["store_text"] == "redacted"
    # con un segundo owner deshabilitado, el owner habilitado sigue siendo el último
    second = add_user(rt, org, "owner2@solo.com.ar", "owner")
    assert owner.post(f"/settings/users/{second.user_id}/disable", {"disabled": "1"}).status_code == 303
    assert owner.post(f"/settings/users/{owner_id}/role", {"role": "admin"}).status_code == 422


def test_deleting_one_tenant_leaves_the_other_intact(tmp_path):
    rt = make_runtime(tmp_path)
    try:
        app = create_app(runtime=rt)
        org_a, org_b = rt.create_org("Alfa"), rt.create_org("Beta")
        add_user(rt, org_a, "owner@alfa.com.ar", "owner")
        add_user(rt, org_b, "owner@beta.com.ar", "owner")
        key_a = create_api_key(rt.auth, org_a, "a", ["ingest"])[0]
        key_b = create_api_key(rt.auth, org_b, "b", ["ingest"])[0]
        panel_a = Panel(app, "owner@alfa.com.ar")
        panel_b = Panel(app, "owner@beta.com.ar")
        assert panel_b.post("/data/delete", {"confirm": "yes", "confirm_name": "Beta"}).status_code == 303
        assert api(app, key_b, "GET", "/v1/models").status_code == 401
        assert api(app, key_a, "GET", "/v1/models").status_code == 200
        assert panel_a.get("/settings").status_code == 200
        assert rt.repo(org_a).get_org() is not None and rt.repo(org_b).get_org() is None
    finally:
        rt.close()


# --- ronda 2 --------------------------------------------------------------------------------------


def test_route_sweep_sees_every_route_after_flattening_included_routers(world):
    """El barrido usa `app.routes`; con routers perezosos (FastAPI >= 0.140) podría saltear rutas.
    Fija que el aplanado de `_walk_routes` no agrega rutas que el barrido no vea."""
    from vigia_ee.auth import _walk_routes

    rt, app, a, b = world
    flat = {
        (route.path, method)
        for route in _walk_routes(app.routes)
        if isinstance(route, APIRoute)
        for method in route.methods - {"HEAD"}
    }
    swept = {(route.path, method) for route, method, _ in routes(app)}
    assert flat == swept
    assert len([1 for _, _, spec in routes(app) if spec is None]) == 0


@pytest.mark.parametrize("role", ["owner", "admin"])
def test_foreign_ids_in_the_form_body_are_404_and_enqueue_nothing(world, role):
    """El id ajeno puede viajar en el cuerpo y no en la ruta: `dataset_id` de B en /calibration/start."""
    rt, app, a, b = world
    panel = Panel(app, a["emails"][role])
    before = {org: len(rt.jobs.list(org)) for org in (a["org"], b["org"])}
    for dataset_id in (b["dataset"], secrets.token_hex(16)):
        resp = panel.post("/calibration/start", {"dataset_id": dataset_id, "alphas": "0,1", "seed": "0"})
        assert resp.status_code == 404, resp.text[:200]
        assert_no_leak(f"{role} start", resp, leaks(rt, b), allowed=[b["dataset"]])
    assert {org: len(rt.jobs.list(org)) for org in (a["org"], b["org"])} == before


def test_mixed_credentials_never_reach_the_other_tenant(world):
    """Cookie de A más Bearer de B: el panel usa solo la sesión y la API solo la key."""
    rt, app, a, b = world
    panel = Panel(app, a["emails"]["owner"])
    b_admin = {"Authorization": f"Bearer {b['keys']['admin'][0]}"}
    resp = panel.client.get(f"/generations/{b['ids']['trace_id']}", headers=b_admin, follow_redirects=False)
    assert resp.status_code == 404
    assert_no_leak("cookie A + key B", resp, leaks(rt, b), allowed=[b["ids"]["trace_id"]])
    a_read = {"Authorization": f"Bearer {a['keys']['read'][0]}"}
    resp = panel.client.get(f"/v1/vigia/traces/{b['ids']['trace_id']}", headers=a_read)
    assert resp.status_code == 404


def test_api_trace_of_one_tenant_is_404_for_every_key_of_the_other(world):
    """La traza creada por la API (no por el playground) tampoco se lee con keys de otra org."""
    rt, app, a, b = world
    body = {"model": "tiny:mixtral", "messages": [{"role": "user", "content": "hola"}], "max_tokens": 2}
    resp = TestClient(app).post(
        "/v1/chat/completions", headers={"Authorization": f"Bearer {b['keys']['ingest'][0]}"}, json=body
    )
    assert resp.status_code == 200
    trace_id = resp.headers["X-Vigia-Trace-Id"]
    path = f"/v1/vigia/traces/{trace_id}"
    assert api(app, b["keys"]["read"][0], "GET", path).status_code == 200
    for scope in ("read", "admin"):
        other = api(app, a["keys"][scope][0], "GET", path)
        absent = api(app, a["keys"][scope][0], "GET", f"/v1/vigia/traces/{secrets.token_hex(16)}")
        assert (other.status_code, absent.status_code) == (404, 404)
        assert other.json()["error"]["code"] == absent.json()["error"]["code"]


def test_enabling_again_does_not_revive_old_sessions(org_env):
    rt, app, org, users = org_env
    owner = Panel(app, "owner@solo.com.ar")
    reviewer = Panel(app, "reviewer@solo.com.ar")
    rid = users["reviewer"].user_id
    assert owner.post(f"/settings/users/{rid}/disable", {"disabled": "1"}).status_code == 303
    assert owner.post(f"/settings/users/{rid}/disable", {"disabled": "0"}).status_code == 303
    resp = reviewer.get("/review")
    assert resp.status_code == 303 and resp.headers["location"].startswith("/login")
    assert Panel(app, "reviewer@solo.com.ar").get("/review").status_code == 200


def test_admin_demoting_another_admin_logs_it_out_and_cannot_promote_itself(org_env):
    rt, app, org, users = org_env
    other = add_user(rt, org, "admin2@solo.com.ar", "admin")
    admin = Panel(app, "admin@solo.com.ar")
    admin2 = Panel(app, "admin2@solo.com.ar")
    assert admin.post(f"/settings/users/{other.user_id}/role", {"role": "viewer"}).status_code == 303
    assert admin2.get("/settings").status_code == 303
    viewer2 = Panel(app, "admin2@solo.com.ar")
    assert viewer2.get("/settings").status_code == 403
    assert viewer2.post(f"/settings/users/{other.user_id}/role", {"role": "admin"}).status_code == 403
    assert admin.post(f"/settings/users/{users['admin'].user_id}/role", {"role": "owner"}).status_code == 403
    assert rt.repo(org).get_user(other.user_id)["role"] == "viewer"
    assert rt.repo(org).get_user(users["admin"].user_id)["role"] == "admin"


def test_two_owners_demoting_each_other_leave_one_owner(org_env):
    """Último owner: el segundo cambio cruzado ya no tiene sesión o ve que quedaría sin owners."""
    rt, app, org, users = org_env
    second = add_user(rt, org, "owner2@solo.com.ar", "owner")
    first_id = users["owner"].user_id
    p1 = Panel(app, "owner@solo.com.ar")
    p2 = Panel(app, "owner2@solo.com.ar")
    assert p1.post(f"/settings/users/{second.user_id}/role", {"role": "admin"}).status_code == 303
    assert p2.post(f"/settings/users/{first_id}/role", {"role": "admin"}).status_code in (401, 403, 422)
    owners = [u for u in rt.repo(org).list_users() if u["role"] == "owner" and not u["disabled"]]
    assert [u["id"] for u in owners] == [first_id]


# --- ronda 3 --------------------------------------------------------------------------------------

FORM_TAG_RE = re.compile(r"<form[^>]*>", re.S)
HREF_RE = re.compile(r'href="(/[^"#?]*)')


def _own_pages(a):
    ids = a["ids"]
    return [
        "/",
        "/generations",
        f"/generations/{ids['trace_id']}",
        "/review",
        f"/review/{ids['item_id']}",
        "/calibration",
        f"/calibration/jobs/{ids['job_id']}",
        f"/calibration/detectors/{ids['detector_id']}",
        "/policies",
        "/settings",
        "/audit",
        "/data",
        "/model",
        "/performance",
        "/account",
        "/playground",
    ]


def _spec_for(app, method, path):
    from starlette.routing import Match

    for route, m, spec in routes(app):
        if m == method and route.matches({"type": "http", "path": path, "method": method})[0] == Match.FULL:
            return spec
    return None


@pytest.mark.parametrize("role", ROLES)
def test_every_form_a_role_sees_posts_to_a_route_it_may_use(world, role):
    """AT-E03 por barrido: ningún formulario visible apunta a una ruta sin permiso para ese rol."""
    rt, app, a, b = world
    panel = Panel(app, a["emails"][role])
    checked = 0
    for page in _own_pages(a):
        resp = panel.get(page)
        if resp.status_code != 200:
            continue
        for tag in FORM_TAG_RE.findall(resp.text):
            method_m = re.search(r'method="(\w+)"', tag, re.I)
            method = method_m.group(1).upper() if method_m else "GET"
            if method == "DIALOG":
                continue
            action_m = re.search(r'action="([^"]*)"', tag)
            action = (action_m.group(1) if action_m else page).split("?")[0]
            spec = _spec_for(app, method, action)
            assert spec is not None, (role, page, method, action)
            if spec.kind == "session":
                assert has_permission(role, spec.permission), (role, page, method, action, spec.permission)
            checked += 1
    assert checked >= 3


@pytest.mark.parametrize("role", ROLES)
def test_no_link_a_role_sees_leads_to_a_403(world, role):
    """Los enlaces del panel (navegación incluida) solo llevan a páginas que el rol puede abrir."""
    rt, app, a, b = world
    panel = Panel(app, a["emails"][role])
    seen: set[str] = set()
    for page in _own_pages(a):
        resp = panel.get(page)
        if resp.status_code != 200:
            continue
        for href in set(HREF_RE.findall(resp.text)) - seen:
            seen.add(href)
            if href.startswith("/static/"):
                continue
            assert panel.get(href).status_code != 403, (role, page, href)
    assert seen


def test_head_options_and_id_variants_do_not_distinguish_foreign_ids(world):
    """HEAD, OPTIONS y el id ajeno en mayúsculas responden igual que un id inexistente."""
    rt, app, a, b = world
    panel = Panel(app, a["emails"]["owner"])
    missing = missing_ids()
    for template in (
        "/generations/{trace_id}",
        "/review/{item_id}",
        "/calibration/jobs/{job_id}",
        "/calibration/jobs/{job_id}/status",
        "/calibration/detectors/{detector_id}",
    ):
        other, absent = fill(template, b["ids"]), fill(template, missing)
        for method in ("HEAD", "OPTIONS"):
            r_other = panel.client.request(method, other, follow_redirects=False)
            r_absent = panel.client.request(method, absent, follow_redirects=False)
            assert r_other.status_code == r_absent.status_code, (method, template)
            assert normalized(r_other, other) == normalized(r_absent, absent), (method, template)
        upper = panel.get(fill(template, {k: v.upper() for k, v in b["ids"].items()}))
        assert upper.status_code == 404, template
        assert_no_leak(f"upper {template}", upper, leaks(rt, b), allowed=list(b["ids"].values()))


def test_exports_and_api_traces_are_never_cacheable(world):
    """Exports del tenant y trazas de la API llevan `no-store`: un cache compartido no los sirve a otro."""
    rt, app, a, b = world
    panel = Panel(app, a["emails"]["owner"])
    for path in ("/data/export", "/audit/export?formato=csv", "/audit/export?formato=json"):
        resp = panel.get(path)
        assert resp.status_code == 200 and resp.headers["cache-control"] == "no-store", path
    path = f"/v1/vigia/traces/{a['ids']['trace_id']}"
    assert api(app, a["keys"]["read"][0], "GET", path).headers["cache-control"] == "no-store"
    foreign = api(app, a["keys"]["read"][0], "GET", f"/v1/vigia/traces/{b['ids']['trace_id']}")
    assert foreign.status_code == 404 and foreign.headers["cache-control"] == "no-store"


def test_tenant_export_carries_no_secrets(world):
    """El export JSON del tenant no lleva hashes de contraseña, HMAC de keys, secretos ni sesiones."""
    rt, app, a, b = world
    panel = Panel(app, a["emails"]["owner"])
    text = panel.get("/data/export").text
    data = json.loads(text)
    assert "sessions" not in data
    for row in data["users"]:
        assert "password_hash" not in row
    for row in data["api_keys"]:
        assert "hmac" not in row
    for row in data["webhooks"]:
        assert "secret" not in row
    assert "whsec_" not in text and "csrf_secret" not in text
    for token, _ in a["keys"].values():
        assert token not in text


# --- ronda 4 --------------------------------------------------------------------------------------


def _tenant_snapshot(rt, org):
    """Estado persistido de una org: lo que un intento ajeno nunca debe cambiar."""
    q = rt.db.query
    o = (org,)
    return {
        "org": q(
            "SELECT name, store_text, retention_days, policy_json, active_detector_id FROM orgs WHERE id = ?",
            o,
        ),
        "users": q("SELECT id, role, disabled FROM users WHERE org_id = ? ORDER BY id", o),
        "keys": q("SELECT id, revoked_at FROM api_keys WHERE org_id = ? ORDER BY id", o),
        "traces": q("SELECT id FROM traces WHERE org_id = ? ORDER BY id", o),
        "items": q("SELECT id, state, reserved_by FROM review_items WHERE org_id = ? ORDER BY id", o),
        "decisions": q("SELECT id FROM review_decisions WHERE org_id = ? ORDER BY id", o),
        "detectors": q("SELECT id, status FROM detectors WHERE org_id = ? ORDER BY id", o),
        "datasets": q("SELECT id FROM datasets WHERE org_id = ? ORDER BY id", o),
        "webhooks": q("SELECT id, disabled FROM webhooks WHERE org_id = ? ORDER BY id", o),
        "jobs": q("SELECT id, state FROM jobs WHERE org_id = ? ORDER BY id", o),
        "audit": q("SELECT COUNT(*) AS n FROM audit_log WHERE org_id = ?", o),
    }


def test_foreign_mutations_leave_the_other_tenant_untouched(world):
    """Cada rol y cada key de A intenta mutar con ids de B: el estado de B (auditoría incluida) no cambia."""
    rt, app, a, b = world
    before = _tenant_snapshot(rt, b["org"])
    tried = 0
    for role in ROLES:
        panel = Panel(app, a["emails"][role])
        for route, method, spec in routes(app):
            if method != "POST" or spec.kind != "session" or not params_of(route.path):
                continue
            assert panel.post(fill(route.path, b["ids"]), FORM).status_code in (403, 404)
            tried += 1
        if has_permission(role, "calibration:write"):
            resp = panel.post("/calibration/start", {"dataset_id": b["dataset"], "alphas": "", "seed": "0"})
            assert resp.status_code == 404
    for token, _ in a["keys"].values():
        for route, method, spec in routes(app):
            if method == "POST" and spec.kind == "session" and params_of(route.path):
                assert not 200 <= api(app, token, "POST", fill(route.path, b["ids"])).status_code < 300
    assert tried >= 4 * 10
    assert _tenant_snapshot(rt, b["org"]) == before


@pytest.mark.parametrize("scope", SCOPES)
def test_api_keys_never_open_panel_routes(world, scope):
    """Una API key (de cualquier scope) no abre rutas del panel, ni con ids propios ni ajenos."""
    rt, app, a, b = world
    token = a["keys"][scope][0]
    values = leaks(rt, b)
    checked = 0
    for route, method, spec in routes(app):
        if spec.kind != "session" or route.path == "/static/{name}":
            continue
        for ids in (a["ids"], b["ids"]):
            path = fill(route.path, ids)
            client = TestClient(app, base_url="https://testserver", follow_redirects=False)
            resp = client.request(
                method, path, headers={"Authorization": f"Bearer {token}"}, data={"confirm": "yes"}
            )
            assert not 200 <= resp.status_code < 300, (scope, method, path, resp.status_code)
            assert_no_leak(f"key {scope} {method} {path}", resp, values, allowed=list(b["ids"].values()))
            checked += 1
    assert checked >= 80


def test_trace_api_url_variants_do_not_distinguish_foreign_ids(world):
    """Barra final, query, HEAD y OPTIONS en la traza de la API: id ajeno e inexistente, igual respuesta."""
    rt, app, a, b = world
    client = TestClient(app, follow_redirects=False)
    headers = {"Authorization": f"Bearer {a['keys']['read'][0]}"}
    other, absent = b["ids"]["trace_id"], secrets.token_hex(16)
    for template in ("/v1/vigia/traces/{}", "/v1/vigia/traces/{}/", "/v1/vigia/traces/{}?x=1"):
        for method in ("GET", "HEAD", "OPTIONS"):
            r_other = client.request(method, template.format(other), headers=headers)
            r_absent = client.request(method, template.format(absent), headers=headers)
            assert r_other.status_code == r_absent.status_code, (method, template)
            assert r_other.status_code != 200, (method, template)
            assert normalized(r_other, other) == normalized(r_absent, absent), (method, template)
            location = (r_other.headers.get("location") or "").replace(other, "X")
            assert location == (r_absent.headers.get("location") or "").replace(absent, "X")


def test_role_change_in_both_directions_and_self_demotion_close_sessions(org_env):
    """Subir de rol también invalida la sesión vieja; un admin que se degrada a sí mismo queda afuera."""
    rt, app, org, users = org_env
    owner = Panel(app, "owner@solo.com.ar")
    reviewer = Panel(app, "reviewer@solo.com.ar")
    rid, aid = users["reviewer"].user_id, users["admin"].user_id
    assert owner.post(f"/settings/users/{rid}/role", {"role": "admin"}).status_code == 303
    resp = reviewer.get("/review")
    assert resp.status_code == 303 and resp.headers["location"].startswith("/login")
    assert Panel(app, "reviewer@solo.com.ar").get("/settings").status_code == 200
    admin = Panel(app, "admin@solo.com.ar")
    assert admin.post(f"/settings/users/{aid}/role", {"role": "viewer"}).status_code == 303
    resp = admin.get("/settings")
    assert resp.status_code == 303 and resp.headers["location"].startswith("/login")
    assert Panel(app, "admin@solo.com.ar").get("/settings").status_code == 403
    assert rt.repo(org).get_user(aid)["role"] == "viewer"


def test_flipping_the_active_detector_of_one_tenant_never_scores_the_other(tmp_path):
    """A alterna su detector activo entre requests de B: B puntúa con el suyo y A con el recién activado."""
    rt = make_runtime(tmp_path, key_rate_per_minute=100_000, tenant_rate_per_minute=100_000)
    try:
        app = create_app(runtime=rt)
        org_a, org_b = rt.create_org("A"), rt.create_org("B")
        owner_a = add_user(rt, org_a, "owner-a@flip.com.ar", "owner")
        det_b = constant_detector(rt.model_info, 0.8)
        install_detector(rt, org_b, det_b)
        first, second = constant_detector(rt.model_info, 0.9), constant_detector(rt.model_info, 0.6)
        for det in (first, second):
            rt.repo(org_a).insert_detector(det.id, det.to_json())
        key_a = create_api_key(rt.auth, org_a, "a", ["ingest"])[0]
        key_b = create_api_key(rt.auth, org_b, "b", ["ingest"])[0]
        for i in range(6):
            active = (first, second)[i % 2]
            rt.calibration.activate(owner_a, active.id)
            assert _calibration_id(app, key_b) == det_b.id
            assert _calibration_id(app, key_a) == active.id
    finally:
        rt.close()


# --- ronda 5 --------------------------------------------------------------------------------------


@pytest.fixture
def pair(tmp_path):
    """Dos orgs con los cuatro roles, el mismo detector activo y una key ingest+read cada una."""
    rt = make_runtime(
        tmp_path, key_rate_per_minute=100_000, tenant_rate_per_minute=100_000, review_min_reviewers=1
    )
    app = create_app(runtime=rt)
    orgs = {}
    shared = constant_detector(rt.model_info, 0.9)
    for tag in ("a", "b"):
        org = rt.create_org(f"Org {tag}")
        rt.repo(org).update_org(store_text="full")
        users = {role: add_user(rt, org, f"{role}@{tag}5.com.ar", role) for role in ROLES}
        install_detector(rt, org, shared)
        key = create_api_key(rt.auth, org, f"k {tag}", ["ingest", "read"])[0]
        orgs[tag] = {"org": org, "users": users, "key": key}
    yield rt, app, orgs["a"], orgs["b"], shared
    rt.close()


def _chat(app, key, content, max_tokens=3):
    body = {
        "model": "tiny:mixtral",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "vigia": {},
    }
    resp = TestClient(app).post("/v1/chat/completions", headers={"Authorization": f"Bearer {key}"}, json=body)
    assert resp.status_code == 200, resp.text[:200]
    return resp


def _audit_count(rt, org):
    return rt.db.query_one("SELECT COUNT(*) AS n FROM audit_log WHERE org_id = ?", (org,))["n"]


def test_review_next_and_dataset_from_reviews_stay_in_the_tenant(pair):
    """Cola propia vacía: `/review/next` no reserva items de B y `/review/dataset` no trae sus trazas."""
    rt, app, a, b, _ = pair
    marker = f"SECRETO-B-{secrets.token_hex(3)}"
    trace_b = _chat(app, b["key"], f"pregunta {marker}").headers["X-Vigia-Trace-Id"]
    trace_a = _chat(app, a["key"], "pregunta de A").headers["X-Vigia-Trace-Id"]
    for tag, org, label in (("b", b, "error"), ("a", a, "correct")):
        reviewer = Panel(app, f"reviewer@{tag}5.com.ar")
        items = rt.repo(org["org"]).list_review_items()
        assert items, "la acción del detector constante encola revisión"
        for item in items:
            assert reviewer.post(f"/review/{item['id']}/decide", {"label": label}).status_code == 303
    pending_b = rt.repo(b["org"]).create_review_item(trace_b)
    state_b = [(i["id"], i["state"], i["reserved_by"]) for i in rt.repo(b["org"]).list_review_items()]
    resp = Panel(app, "reviewer@a5.com.ar").post("/review/next")
    assert resp.status_code == 303 and resp.headers["location"].startswith("/review?")
    assert pending_b not in resp.headers["location"]
    assert [(i["id"], i["state"], i["reserved_by"]) for i in rt.repo(b["org"]).list_review_items()] == state_b
    assert Panel(app, "admin@a5.com.ar").post("/review/dataset", {"name": "revisiones A"}).status_code == 303
    (dataset,) = rt.repo(a["org"]).list_datasets()
    content = bytes(rt.repo(a["org"]).get_dataset(dataset["id"])["content"]).decode("utf-8")
    assert trace_a in content
    assert trace_b not in content and marker not in content
    assert rt.repo(b["org"]).list_datasets() == []


def test_pending_calibration_cap_is_counted_per_tenant(pair):
    """Una calibración en cola de B no bloquea la de A: el tope de jobs pendientes es por organización."""
    rt, app, a, b, _ = pair
    datasets = {
        tag: rt.calibration.upload_dataset(org["users"]["owner"], f"ds {tag}", learnable_jsonl(20))
        for tag, org in (("a", a), ("b", b))
    }
    owner_b = Panel(app, "owner@b5.com.ar")
    form = {"alphas": "", "seed": "0"}
    assert owner_b.post("/calibration/start", {"dataset_id": datasets["b"], **form}).status_code == 303
    assert owner_b.post("/calibration/start", {"dataset_id": datasets["b"], **form}).status_code == 422
    admin_a = Panel(app, "admin@a5.com.ar")
    assert admin_a.post("/calibration/start", {"dataset_id": datasets["a"], **form}).status_code == 303
    assert len(rt.calibration.list_jobs(a["org"])) == 1
    assert len(rt.calibration.list_jobs(b["org"])) == 1


def test_foreign_webhook_test_and_delete_enqueue_nothing_in_either_tenant(pair):
    rt, app, a, b, _ = pair
    hook_b = rt.repo(b["org"]).insert_webhook(
        "https://hooks-b5.example/x", "whsec_" + "A" * 43, ["trace.flagged"]
    )
    orgs = (a["org"], b["org"])
    owner_a = Panel(app, "owner@a5.com.ar")  # el login audita: se cuenta después
    jobs_before = {org: len(rt.jobs.list(org, limit=200)) for org in orgs}
    audit_before = {org: _audit_count(rt, org) for org in orgs}
    assert owner_a.post(f"/settings/webhooks/{hook_b}/test").status_code == 404
    assert owner_a.post(f"/settings/webhooks/{hook_b}/delete").status_code == 404
    assert {org: len(rt.jobs.list(org, limit=200)) for org in orgs} == jobs_before
    assert {org: _audit_count(rt, org) for org in orgs} == audit_before
    assert rt.repo(b["org"]).get_webhook(hook_b) is not None


def test_inviting_a_foreign_email_in_any_letter_case_is_generic_and_creates_nothing(pair):
    """Email único global (riesgo aceptado, seguridad.md §3): mayúsculas o espacios no cambian el mensaje."""
    rt, app, a, b, _ = pair
    admin = Panel(app, "admin@a5.com.ar")
    orgs = (a["org"], b["org"])
    users_before = {org: len(rt.repo(org).list_users()) for org in orgs}
    for email in ("owner@b5.com.ar", "OWNER@B5.com.ar", "  Viewer@b5.COM.ar  ", "admin@a5.com.ar"):
        resp = admin.post("/settings/users", {"email": email, "role": "viewer"})
        assert resp.status_code == 422, email
        assert "No se pudo crear una cuenta con ese email." in resp.text, email
        assert b["org"] not in resp.text
    assert {org: len(rt.repo(org).list_users()) for org in orgs} == users_before


def test_concurrent_detector_flips_in_one_tenant_never_score_the_other(pair):
    """Hilos de API de A y B mientras A alterna su detector activo: B siempre puntúa con el suyo."""
    import threading
    from collections import Counter

    rt, app, a, b, shared = pair
    other = constant_detector(rt.model_info, 0.6)
    rt.repo(a["org"]).insert_detector(other.id, other.to_json())
    own_b = constant_detector(rt.model_info, 0.7)
    rt.repo(b["org"]).insert_detector(own_b.id, own_b.to_json())
    rt.calibration.activate(b["users"]["owner"], own_b.id)
    stop = threading.Event()
    seen: Counter = Counter()
    errors: list[str] = []

    def flip():
        i = 0
        while not stop.is_set():
            rt.calibration.activate(a["users"]["owner"], (shared.id, other.id)[i % 2])
            i += 1

    def ingest(tag, key):
        for _ in range(8):
            try:
                resp = _chat(app, key, "hola", max_tokens=2)
                seen[(tag, resp.json()["choices"][0]["vigia"]["calibration_id"])] += 1
            except Exception as exc:  # noqa: BLE001 - se informa en el assert
                errors.append(repr(exc))

    flipper = threading.Thread(target=flip)
    flipper.start()
    workers = [threading.Thread(target=ingest, args=args) for args in (("a", a["key"]), ("b", b["key"]))]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    stop.set()
    flipper.join()
    assert not errors
    assert {cal for tag, cal in seen if tag == "b"} == {own_b.id}
    assert {cal for tag, cal in seen if tag == "a"} <= {shared.id, other.id}
    assert sum(n for (tag, _), n in seen.items() if tag == "b") == 8


def test_signals_of_a_detector_shared_by_two_tenants_count_only_own_traces(pair):
    """El mismo artefacto activo en A y B: las señales de drift de B no cuentan las trazas de A."""
    rt, app, a, b, shared = pair
    for i in range(4):
        _chat(app, a["key"], f"consulta {i}")
    assert rt.drift_for(a["org"])["detector_id"] == rt.drift_for(b["org"])["detector_id"] == shared.id
    assert rt.repo(a["org"]).recent_token_signals(shared.id)["n_traces"] == 4
    assert rt.repo(b["org"]).recent_token_signals(shared.id) == {"n_traces": 0, "tokens": []}
    report_b = rt.drift_for(b["org"])
    assert report_b["n_traces"] == 0 and report_b["n_tokens"] == 0
    assert Panel(app, "viewer@b5.com.ar").get("/performance").status_code == 200
