"""Aislamiento entre tenants por barrido automático de TODAS las rutas de la app (AT-P01, ADR A-15).

Dos organizaciones con datos de cada tipo. Para cada ruta registrada (API y panel):

- Con path params: el owner de A pide los ids de B y unos ids inexistentes. Las dos respuestas deben
  ser 404 e indistinguibles (salvo el request_id); con los ids propios de A, la ruta responde.
- Sin path params (GET): la respuesta de A nunca contiene datos de B.
- Sin credenciales: 401/403, o 303 al login en páginas del panel; nunca datos.
Después del barrido, los datos de B siguen intactos.
"""

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

from vigia_ee.app import create_app
from vigia_ee.auth import create_api_key, route_permission

PARAM_KINDS = ("trace_id", "item_id", "job_id", "detector_id", "user_id", "key_id", "webhook_id")
CSRF_RE = re.compile(r'name="csrf_token" value="([0-9a-f]{64})"')
RID_RE = re.compile(r"[0-9a-f]{16}")
NONCE_RE = re.compile(r'nonce="[^"]+"')


def seed_tenant(rt, name, marker):
    org = rt.create_org(f"Org {name} {marker}")
    rt.repo(org).update_org(store_text="full")
    owner = add_user(rt, org, f"owner-{marker.lower()}@ejemplo.com.ar", "owner")
    viewer = add_user(rt, org, f"viewer-{marker.lower()}@ejemplo.com.ar", "viewer")
    install_detector(rt, org, constant_detector(rt.model_info, 0.9 if name == "A" else 0.85))
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
    candidate = constant_detector(rt.model_info, 0.66 if name == "A" else 0.77)
    repo.insert_detector(candidate.id, candidate.to_json())
    token, key_id = create_api_key(rt.auth, org, f"clave {marker}", ["ingest", "read"])
    webhook = repo.insert_webhook(
        f"https://hooks.{marker.lower()}.example/recibir", "whsec_" + "A" * 32, ["trace.flagged"]
    )
    return {
        "org": org,
        "owner": owner,
        "email": f"owner-{marker.lower()}@ejemplo.com.ar",
        "marker": marker,
        "token": token,
        "ids": {
            "trace_id": done["trace_id"],
            "item_id": item["id"],
            "job_id": job,
            "detector_id": candidate.id,
            "user_id": viewer.user_id,
            "key_id": key_id,
            "webhook_id": webhook,
        },
    }


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    rt = make_runtime(tmp_path_factory.mktemp("isolation"))
    a = seed_tenant(rt, "A", "MARCA-A-91c")
    b = seed_tenant(rt, "B", "MARCA-B-7f3")
    app = create_app(runtime=rt)
    yield rt, app, a, b
    rt.close()


def login(app, email):
    client = TestClient(app, base_url="https://testserver")
    resp = client.post(
        "/login", data={"email": email, "password": PASSWORD, "next": "/"}, follow_redirects=False
    )
    assert resp.status_code == 303, resp.text
    match = CSRF_RE.search(client.get("/account").text)
    assert match, "sin token CSRF"
    return client, match.group(1)


def routes(app):
    for route in app.routes:
        if isinstance(route, APIRoute):
            for method in sorted(route.methods - {"HEAD"}):
                yield route, method


def params_of(route):
    return re.findall(r"\{([^}:]+)(?::[^}]*)?\}", route.path)


def fill(route, ids):
    return re.sub(r"\{([^}:]+)(?::[^}]*)?\}", lambda m: ids[m.group(1)], route.path)


def b_secrets(rt, b):
    """Todo lo que identifica datos de B y que A nunca debería ver."""
    values = [b["marker"], b["email"], b["org"], *b["ids"].values()]
    values.append(rt.repo(b["org"]).get_trace(b["ids"]["trace_id"])["response_text"])
    return [v for v in values if v and len(v) >= 6]


def assert_no_leak(resp, secrets_, allowed=()):
    body = resp.text
    for value in secrets_:
        if value in allowed:
            continue
        assert value not in body, f"{resp.request.method} {resp.request.url.path} filtró {value[:20]!r}"


def call(client, method, path, *, token=None, csrf=None, api=False):
    if api:
        headers = {"Authorization": f"Bearer {token}"}
        if method == "POST":
            return client.post(
                path,
                headers=headers,
                json={
                    "model": "tiny:mixtral",
                    "messages": [{"role": "user", "content": "hola"}],
                    "max_tokens": 1,
                },
            )
        return client.request(method, path, headers=headers)
    data = {"csrf_token": csrf, "confirm": "yes"} if method != "GET" else None
    return client.request(method, path, data=data, follow_redirects=False)


def normalized(resp):
    """Cuerpo sin lo que cambia por request: request_id y nonce de CSP."""
    return RID_RE.sub("RID", NONCE_RE.sub('nonce="N"', resp.text))


def test_AT_P01_every_route_with_ids_returns_404_for_other_tenant(world):
    """AT-P01: barrido de todas las rutas con ids del otro tenant: 404 indistinguible y sin datos."""
    rt, app, a, b = world
    panel, csrf = login(app, a["email"])
    api = TestClient(app)
    missing = {k: secrets.token_hex(8 if k in ("detector_id", "key_id") else 16) for k in PARAM_KINDS}
    leaks = b_secrets(rt, b)
    swept = 0
    for route, method in routes(app):
        names = params_of(route)
        spec = route_permission(route)
        if not names or spec.kind == "public":
            continue
        assert set(names) <= set(PARAM_KINDS), f"parámetro sin mapear en {route.path}: {names}"
        is_api = spec.kind == "api_key"
        client = api if is_api else panel
        other = call(client, method, fill(route, b["ids"]), token=a["token"], csrf=csrf, api=is_api)
        absent = call(client, method, fill(route, missing), token=a["token"], csrf=csrf, api=is_api)
        label = f"{method} {route.path}"
        assert other.status_code == 404, (label, other.status_code, other.text[:200])
        assert absent.status_code == 404, (label, absent.status_code)
        assert normalized(other).replace(fill(route, b["ids"]), "X") == normalized(absent).replace(
            fill(route, missing), "X"
        ), f"{label}: la respuesta distingue un recurso ajeno de uno inexistente"
        assert_no_leak(other, leaks, allowed=[b["ids"][n] for n in names])
        swept += 1
    assert swept >= 15, f"el barrido cubrió solo {swept} rutas con ids"
    for route, method in routes(app):  # con los ids propios, las páginas de lectura responden
        names = params_of(route)
        spec = route_permission(route)
        if method != "GET" or not names or spec.kind == "public":
            continue
        is_api = spec.kind == "api_key"
        resp = call(api if is_api else panel, method, fill(route, a["ids"]), token=a["token"], api=is_api)
        assert resp.status_code == 200, (route.path, resp.status_code, resp.text[:200])


def test_AT_P01_list_pages_and_exports_only_show_own_tenant(world):
    """AT-P01: listados, reportes y exports sin path params nunca muestran datos de B."""
    rt, app, a, b = world
    panel, _ = login(app, a["email"])
    api = TestClient(app)
    leaks = b_secrets(rt, b)
    checked = 0
    for route, method in routes(app):
        spec = route_permission(route)
        if method != "GET" or params_of(route) or spec.kind == "public":
            continue
        is_api = spec.kind == "api_key"
        resp = call(api if is_api else panel, "GET", route.path, token=a["token"], api=is_api)
        assert resp.status_code == 200, (route.path, resp.status_code)
        assert_no_leak(resp, leaks)
        checked += 1
    assert checked >= 10
    own = panel.get("/generations").text
    assert a["ids"]["trace_id"] in own, "el listado propio sí muestra la traza de A"


def test_AT_P01_unauthenticated_requests_get_no_data(world):
    """AT-P01: sin credenciales, cada ruta protegida responde 401/403 (o 303 al login) y sin datos."""
    rt, app, a, b = world
    client = TestClient(app, base_url="https://testserver")
    leaks = b_secrets(rt, b) + b_secrets(rt, a)
    for route, method in routes(app):
        spec = route_permission(route)
        if spec.kind == "public":
            continue
        ids = b["ids"]
        path = fill(route, ids) if params_of(route) else route.path
        resp = client.request(method, path, follow_redirects=False)
        expected = {401, 403} if method != "GET" or spec.kind == "api_key" else {303, 401}
        assert resp.status_code in expected, (method, route.path, resp.status_code)
        assert_no_leak(resp, leaks, allowed=[ids[n] for n in params_of(route)])


def test_AT_P01_other_tenant_data_is_untouched_after_sweep(world):
    """AT-P01: los POST con ids de B no cambiaron nada de B."""
    rt, app, a, b = world
    test_AT_P01_every_route_with_ids_returns_404_for_other_tenant(world)
    repo = rt.repo(b["org"])
    ids = b["ids"]
    assert repo.get_trace(ids["trace_id"]) is not None
    assert repo.get_review_item(ids["item_id"])["state"] == "pending"
    assert repo.list_review_decisions(ids["item_id"]) == []
    assert repo.get_detector(ids["detector_id"])["status"] == "candidate"
    assert repo.get_api_key(ids["key_id"])["revoked_at"] is None
    assert repo.get_user(ids["user_id"])["role"] == "viewer" and not repo.get_user(ids["user_id"])["disabled"]
    assert repo.get_webhook(ids["webhook_id"]) is not None
    assert rt.jobs.get(b["org"], ids["job_id"])["state"] == "queued"
    assert repo.get_org() is not None
