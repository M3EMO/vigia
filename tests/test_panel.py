"""Tests del panel: login, páginas por rol, CSRF, CSP, escape de texto del modelo, detalle y playground."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from _panel_support import (
    PW,
    activate_detector,
    add_user,
    csrf_of,
    login,
    make_app,
    make_runtime,
    play,
)
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from vigia_ee import panel
from vigia_ee.auth import has_permission, route_permission
from vigia_ee.security import SESSION_COOKIE

TEMPLATES = Path(panel.TEMPLATES_DIR)
PAGES = {
    "/": "metrics:read",
    "/generations": "traces:read",
    "/review": "review:read",
    "/playground": "playground:use",
    "/model": "model:read",
    "/performance": "metrics:read",
    "/calibration": "calibration:read",
    "/policies": "policy:read",
    "/settings": "settings:read",
    "/audit": "audit:read",
    "/data": "data:export",
    "/account": "account:self",
}
ROLES = ("owner", "admin", "reviewer", "viewer")
POST_FORM_RE = re.compile(r'<form[^>]*method="post"[^>]*action="([^"]+)"', re.I)


@pytest.fixture
def env(tmp_path):
    runtime = make_runtime(tmp_path)
    org = runtime.create_org("Acme")
    users = {role: add_user(runtime, org, role) for role in ROLES}
    app = make_app(runtime)
    yield runtime, org, users, app
    runtime.close()


def _clients(app, users):
    return {role: login(app, email) for role, email in users.items()}


# --- login, sesión y redirecciones ---------------------------------------------------------------


def test_page_without_session_redirects_to_login_with_next(env):
    _, _, _, app = env
    client = TestClient(app, base_url="https://testserver")
    resp = client.get("/generations?accion=pass", follow_redirects=False)
    assert resp.status_code == 303
    assert resp.headers["location"] == "/login?next=%2Fgenerations%3Faccion%3Dpass"
    page = client.get(resp.headers["location"])
    assert page.status_code == 200
    assert 'name="next" value="/generations?accion=pass"' in page.text


def test_login_rejects_bad_password_and_open_redirect(env):
    _, _, users, app = env
    client = TestClient(app, base_url="https://testserver")
    bad = client.post("/login", data={"email": users["owner"], "password": "otra frase cualquiera larga"})
    assert bad.status_code == 401
    assert 'role="alert"' in bad.text and "incorrectos" in bad.text
    ok = client.post(
        "/login",
        data={"email": users["owner"], "password": PW, "next": "//evil.example/x"},
        follow_redirects=False,
    )
    assert ok.status_code == 303 and ok.headers["location"] == "/"
    cookie = ok.headers["set-cookie"]
    assert SESSION_COOKIE in cookie and "HttpOnly" in cookie and "samesite=strict" in cookie.lower()


def test_login_with_secure_cookies_sets_secure_flag(tmp_path):
    runtime = make_runtime(tmp_path, secure_cookies=True)
    try:
        org = runtime.create_org("Seg")
        email = add_user(runtime, org, "owner")
        client = TestClient(make_app(runtime), base_url="https://testserver")
        resp = client.post("/login", data={"email": email, "password": PW}, follow_redirects=False)
        assert "Secure" in resp.headers["set-cookie"]
        assert client.get("/account").status_code == 200
    finally:
        runtime.close()


def test_logout_invalidates_server_session(env):
    _, _, users, app = env
    client = login(app, users["viewer"])
    token = csrf_of(client)
    raw = client.cookies.get(SESSION_COOKIE)
    resp = client.post("/logout", data={"csrf_token": token}, follow_redirects=False)
    assert resp.status_code == 303 and resp.headers["location"].startswith("/login")
    other = TestClient(app, base_url="https://testserver", cookies={SESSION_COOKIE: raw})
    assert other.get("/account", follow_redirects=False).status_code == 303


def test_account_password_change_closes_sessions(env):
    runtime, _, users, app = env
    client = login(app, users["reviewer"])
    token = csrf_of(client)
    wrong = client.post(
        "/account/password",
        data={
            "csrf_token": token,
            "current_password": "no es la actual pero larga",
            "new_password": "x" * 20,
            "confirm_password": "x" * 20,
        },
    )
    assert wrong.status_code == 422 and 'aria-invalid="true"' in wrong.text
    new = "otra frase propia y bastante larga"
    resp = client.post(
        "/account/password",
        data={"csrf_token": token, "current_password": PW, "new_password": new, "confirm_password": new},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert client.get("/account", follow_redirects=False).status_code == 303
    login(app, users["reviewer"], new)


# --- páginas por rol y AT-E03 --------------------------------------------------------------------


@pytest.mark.parametrize("role", ROLES)
def test_each_page_renders_or_is_forbidden_per_role(env, role):
    _, _, users, app = env
    client = login(app, users[role])
    for path, perm in PAGES.items():
        resp = client.get(path)
        expected = 200 if has_permission(role, perm) else 403
        assert resp.status_code == expected, (role, path, resp.status_code)
        assert '<html lang="es-AR">' in resp.text
        if expected == 403:
            assert 'role="alert"' in resp.text


def test_nav_only_lists_allowed_sections(env):
    _, _, users, app = env
    viewer = login(app, users["viewer"]).get("/").text
    nav = re.search(r'<nav class="vg-nav".*?</nav>', viewer, re.S).group(0)
    assert 'href="/generations"' in nav and 'href="/settings"' not in nav and 'href="/playground"' not in nav
    owner = login(app, users["owner"]).get("/").text
    assert 'href="/settings"' in re.search(r'<nav class="vg-nav".*?</nav>', owner, re.S).group(0)


def api_routes(routes):
    """APIRoutes de la app, entrando en los routers incluidos (FastAPI >= 0.140 los guarda lazy)."""
    for route in routes:
        inner = getattr(route, "original_router", None)
        if inner is not None:
            yield from api_routes(inner.routes)
        elif isinstance(route, APIRoute):
            yield route


def _post_routes(app):
    for route in api_routes(app.routes):
        if "POST" in route.methods:
            spec = route_permission(route)
            path = re.sub(r"\{[^}]+\}", "0123456789abcdef", route.path)
            yield route.path, path, spec


def test_AT_E03_viewer_sees_no_mutation_controls_and_direct_post_is_403(env):
    runtime, org, users, app = env
    activate_detector(runtime, org)
    done = play(runtime, users["owner"])
    client = login(app, users["viewer"])
    pages = [
        p
        for p, perm in PAGES.items()
        if perm
        in (
            "metrics:read",
            "traces:read",
            "review:read",
            "model:read",
            "calibration:read",
            "policy:read",
            "account:self",
        )
    ]
    pages += [f"/generations/{done['trace_id']}"]
    for path in pages:
        actions = POST_FORM_RE.findall(client.get(path).text)
        allowed = {"/logout", "/account/password"}
        assert set(actions) <= allowed, (path, actions)
    token = csrf_of(client)
    checked = 0
    for template, path, spec in _post_routes(app):
        if spec is None or spec.kind == "public" or spec.permission == "account:self":
            continue
        resp = client.post(path, data={"csrf_token": token}, follow_redirects=False)
        assert resp.status_code == 403, (template, resp.status_code)
        checked += 1
    assert checked >= 20
    # Ninguna ruta que muta es usable por un viewer salvo las de su propia cuenta (deny-by-default).
    for route in api_routes(app.routes):
        spec = route_permission(route)
        if route.methods & {"POST", "PUT", "PATCH", "DELETE"} and spec.kind != "public":
            assert spec.permission == "account:self" or not has_permission("viewer", spec.permission), (
                route.path
            )


def test_every_panel_route_declares_permission(env):
    _, _, _, app = env
    routes = list(api_routes(app.routes))
    assert len(routes) >= 45
    for route in routes:
        assert route_permission(route) is not None, route.path


# --- CSRF (AT-P05) -------------------------------------------------------------------------------


def test_AT_P05_every_post_requires_csrf_token(env):
    _, _, users, app = env
    client = login(app, users["owner"])
    token = csrf_of(client)
    checked = 0
    for template, path, spec in _post_routes(app):
        if spec.kind == "public":
            continue
        assert client.post(path, data={}, follow_redirects=False).status_code == 403, template
        assert client.post(path, data={"csrf_token": "0" * 64}, follow_redirects=False).status_code == 403
        cross = client.post(
            path, data={"csrf_token": token}, headers={"Sec-Fetch-Site": "cross-site"}, follow_redirects=False
        )
        assert cross.status_code == 403, template
        checked += 1
    assert checked >= 24


def test_every_post_form_in_every_page_carries_csrf(env):
    runtime, org, users, app = env
    activate_detector(runtime, org)
    done = play(runtime, users["owner"])
    client = login(app, users["owner"])
    for path in [*PAGES, f"/generations/{done['trace_id']}"]:
        html = client.get(path).text
        for form in re.findall(r'<form[^>]*method="post".*?</form>', html, re.S | re.I):
            assert 'name="csrf_token"' in form, (path, form[:120])


# --- CSP y plantillas (AT-P12) -------------------------------------------------------------------


FORBIDDEN_TEMPLATE = [
    (re.compile(r"\sstyle\s*=", re.I), "atributo style="),
    (re.compile(r"<style", re.I), "<style>"),
    (re.compile(r"\son[a-z]+\s*=", re.I), "handler on*="),
    (re.compile(r"javascript:", re.I), "javascript:"),
    (re.compile(r"hx-on", re.I), "hx-on"),
    (re.compile(r"\|\s*safe\b"), "|safe"),
    (re.compile(r"(src|href)\s*=\s*\"(https?:)?//", re.I), "recurso externo"),
]


def test_AT_P12_templates_have_no_inline_styles_handlers_or_external_resources():
    files = sorted(TEMPLATES.glob("*.html"))
    assert len(files) >= 15
    for f in files:
        text = f.read_text(encoding="utf-8")
        for pattern, what in FORBIDDEN_TEMPLATE:
            assert not pattern.search(text), f"{f.name}: {what}"
        for script in re.findall(r"<script[^>]*>", text):
            assert 'nonce="{{ nonce }}"' in script and 'src="/static/' in script, f.name


def test_AT_P12_static_js_has_no_eval_or_inner_html():
    js = (Path(panel.STATIC_DIR) / "app.js").read_text(encoding="utf-8")
    for bad in ("innerHTML", "outerHTML", "eval(", "new Function", "document.write", "insertAdjacentHTML"):
        assert bad not in js, bad


def test_AT_P12_rendered_pages_have_strict_csp_and_nonced_scripts(env):
    runtime, org, users, app = env
    activate_detector(runtime, org)
    done = play(runtime, users["owner"])
    client = login(app, users["owner"])
    nonces = set()
    for path in ["/login", *PAGES, f"/generations/{done['trace_id']}"]:
        resp = client.get(path)
        csp = resp.headers["content-security-policy"]
        assert "unsafe-inline" not in csp and "unsafe-eval" not in csp and "style-src 'self'" in csp
        nonce = re.search(r"'nonce-([^']+)'", csp).group(1)
        nonces.add(nonce)
        for script in re.findall(r"<script[^>]*>", resp.text):
            assert f'nonce="{nonce}"' in script
        assert not re.search(r"\sstyle=", resp.text)
        assert not re.search(r"<style", resp.text)
        assert not re.search(r'(src|href)="(https?:)?//', resp.text)
        assert resp.headers["x-content-type-options"] == "nosniff"
        if path != "/login":
            assert resp.headers["cache-control"] == "no-store"
    assert len(nonces) == len(PAGES) + 2


def test_static_assets_are_served_and_unknown_names_404(env):
    _, _, _, app = env
    client = TestClient(app, base_url="https://testserver")
    css = client.get("/static/app.css")
    assert css.status_code == 200 and css.headers["content-type"].startswith("text/css")
    assert "--vg-hl-high-bg" in css.text
    assert "vg-chart" in client.get("/static/charts.css").text
    assert client.get("/static/app.js").headers["content-type"].startswith("text/javascript")
    assert client.get("/static/favicon.svg").status_code == 200
    assert client.get("/static/..%2Fpanel.py").status_code == 404
    assert client.get("/static/panel.py").status_code == 404


# --- escape del texto del modelo (AT-P13) --------------------------------------------------------


EVIL = "<script>alert(1)</script> ‮admin‬ fin\x07 <img src=x onerror=alert(2)>"


def test_AT_P13_model_text_with_script_bidi_and_control_is_escaped(env):
    runtime, org, users, app = env
    runtime.repo(org).update_org(store_text="full")
    activate_detector(runtime, org)
    trace_id = runtime.repo(org).insert_trace(
        model_id="tiny-panel",
        prompt_text="user: " + EVIL,
        response_text=EVIL,
        action="human_review",
        tokens=[{"index": 0, "text": EVIL[:20], "start": 0, "end": 20, "p_error": 0.9, "flag": True}],
        spans=[{"start": 0, "end": 20, "score": 0.9}],
        calibration_id="f" * 16,
        alpha=0.1,
        tau=0.5,
        response_p_error=0.9,
    )
    item_id = runtime.repo(org).create_review_item(trace_id)
    client = login(app, users["owner"])
    for path in (f"/generations/{trace_id}", f"/review/{item_id}"):
        html = client.get(path).text
        assert "<script>alert(1)" not in html and "<img src=x" not in html, path
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
        assert "‮" not in html and "‬" not in html and "\x07" not in html
        assert "admin" in html


def test_AT_P13_playground_echo_is_escaped(env):
    runtime, org, users, app = env
    client = login(app, users["owner"])
    token = csrf_of(client, "/playground")
    resp = client.post(
        "/playground", data={"csrf_token": token, "prompt": EVIL, "max_tokens": "8", "temperature": "0"}
    )
    assert resp.status_code == 200
    assert "<script>alert(1)" not in resp.text and "&lt;script&gt;" in resp.text


# --- resumen -------------------------------------------------------------------------------------


def test_overview_empty_state_and_uncalibrated_banner(env):
    _, _, users, app = env
    html = login(app, users["viewer"]).get("/").text
    assert "No hay un detector calibrado activo" in html
    assert "Todavía no hay generaciones" in html


def test_overview_kpis_and_trend_from_real_traces(env):
    runtime, org, users, app = env
    play(runtime, users["owner"])  # sin detector: sin fragmentos
    activate_detector(runtime, org)
    for _ in range(3):
        play(runtime, users["owner"])
    html = login(app, users["viewer"]).get("/").text
    assert "No hay un detector calibrado activo" not in html
    kpis = re.search(r'<dl class="vg-kpis">.*?</dl>', html, re.S).group(0)
    assert ">4</dd>" in kpis  # requests
    assert "75,0 %" in kpis  # 3 de 4 con fragmentos
    assert "meta 5 %" in kpis
    assert 'class="vg-chart vg-trend"' in html and 'role="img"' in html
    assert "<details" in html  # tabla alternativa del gráfico
    assert "Revisión humana" in html


def test_summary_stats_trend_marks_recalibration_and_skips_empty_days():
    rows = [
        {
            "created_at": "2026-10-01T15:00:00.000000Z",
            "action": "pass",
            "latency_ms": 100.0,
            "overhead_ms": 2.0,
            "calibration_id": "a",
            "n_tokens": 10,
            "degraded": 0,
            "n_flagged": 1,
            "n_spans": 1,
        },
        {
            "created_at": "2026-10-03T15:00:00.000000Z",
            "action": "abstain",
            "latency_ms": 110.0,
            "overhead_ms": 10.0,
            "calibration_id": "b",
            "n_tokens": 10,
            "degraded": 0,
            "n_flagged": 5,
            "n_spans": 2,
        },
    ]
    stats = panel.summary_stats(rows)
    assert stats["trend_labels"] == ["01/10", "03/10"]
    assert stats["trend_rates"] == [0.1, 0.5]
    assert stats["trend_recal"] == [1]
    assert stats["flagged_pct"] == 1.0
    assert stats["ovh_alert"] is True  # 10 / 100 = 10 % > 5 %


# --- generaciones --------------------------------------------------------------------------------


def _insert(runtime, org, n, **kw):
    repo = runtime.repo(org)
    return [
        repo.insert_trace(
            model_id="tiny-panel",
            response_text=f"respuesta {i}",
            action=kw.get("action", "pass"),
            source=kw.get("source", "api"),
            response_p_error=kw.get("p", 0.2),
            latency_ms=10.0,
            overhead_ms=0.2,
        )
        for i in range(n)
    ]


def test_generations_filters_cursor_pagination_and_querystring_without_text(env):
    runtime, org, users, app = env
    _insert(runtime, org, 55)
    _insert(runtime, org, 3, action="abstain", p=0.95)
    client = login(app, users["viewer"])
    first = client.get("/generations")
    assert first.text.count('href="/generations/') == 50
    nxt = re.search(r'href="(/generations\?cursor=[^"]+)" rel="next"', first.text).group(1)
    second = client.get(nxt.replace("&amp;", "&"))
    assert second.text.count('href="/generations/') == 8
    assert "Primera página" in second.text and 'rel="next"' not in second.text
    filtered = client.get("/generations?accion=abstain&p_min=0.9&degradado=no")
    assert filtered.text.count('href="/generations/') == 3
    assert "Limpiar filtros" in filtered.text
    for href in re.findall(r'href="(/generations\?[^"]+)"', first.text + filtered.text):
        keys = {k.split("=")[0] for k in href.split("?", 1)[1].replace("&amp;", "&").split("&")}
        assert keys <= {"accion", "origen", "desde", "hasta", "p_min", "degradado", "cursor"}


def test_generations_invalid_filters_show_alert_and_empty_filter_state(env):
    runtime, org, users, app = env
    _insert(runtime, org, 2)
    client = login(app, users["viewer"])
    bad = client.get("/generations?accion=borrar&p_min=7&desde=ayer&cursor=xx")
    assert bad.status_code == 200
    assert 'role="alert"' in bad.text and "P(error) mínimo" in bad.text and "paginación" in bad.text
    empty = client.get("/generations?accion=abstain")
    assert "Ninguna generación cumple estos filtros" in empty.text


def test_generations_date_filter_uses_local_days():
    f, errors = panel.parse_filters({"desde": "2026-10-04", "hasta": "2026-10-04"})
    assert not errors
    assert panel.local_day_start(f.since).isoformat() == "2026-10-04T03:00:00+00:00"


# --- detalle -------------------------------------------------------------------------------------


def test_detail_triple_encoding_signals_threshold_and_warning(env):
    runtime, org, users, app = env
    activate_detector(runtime, org)
    done = play(runtime, users["owner"])
    client = login(app, users["viewer"])
    html = client.get(f"/generations/{done['trace_id']}").text
    assert "El texto sin marcar no está verificado" in html
    marks = re.findall(
        r'<mark class="vg-mark vg-lvl-(\w+)" id="frag-(\d+)"[^>]*aria-describedby="([^"]+)"', html
    )
    assert marks == [("high", "1", "frag-1-desc")]
    assert re.search(r'<span id="frag-1-desc">Fragmento dudoso 1, nivel alto, P\(error\) 0,98\.</span>', html)
    assert ">1820</mark>" in html
    assert 'id="sig-1"' in html
    assert re.search(r"<code>\s*18</code>", html) and re.search(r"<code>\s*20</code>", html)
    assert 'name="umbral"' in html and 'type="range"' in html
    detector_id = runtime.detector_for(org).id
    assert f"<code>{detector_id}</code>" in html and "0,500" in html  # versión del calibrador y tau
    # Umbral de vista bajo: todas las palabras (p = 0,27) quedan marcadas en un solo fragmento.
    looser = client.get(f"/generations/{done['trace_id']}?umbral=0.2").text
    assert ">Belgrano nació en 1820 en Rosario</mark>." in looser
    # La agregación es por palabra: con 0,9 "1820" sigue marcada entera por el subtoken "18".
    tighter = client.get(f"/generations/{done['trace_id']}?umbral=0.9").text
    assert ">1820</mark>" in tighter
    none = client.get(f"/generations/{done['trace_id']}?umbral=0.999").text
    assert "<mark class" not in none.split("data-marked-text>")[1].split("</div>")[0]
    bad = client.get(f"/generations/{done['trace_id']}?umbral=3")
    assert "El umbral de vista debe estar entre 0 y 1" in bad.text
    assert runtime.policy_for(org).alpha == 0.1  # el umbral de vista nunca cambia la política


def test_detail_uncalibrated_trace_has_no_highlight(env):
    runtime, _, users, app = env
    done = play(runtime, users["owner"])
    html = login(app, users["viewer"]).get(f"/generations/{done['trace_id']}").text
    assert "no tiene puntajes calibrados" in html
    assert '<mark class="vg-mark' not in html.split("data-marked-text>")[1].split("</div>")[0]


def test_detail_with_store_text_none_explains_missing_text(env):
    runtime, org, users, app = env
    runtime.repo(org).update_org(store_text="none")
    activate_detector(runtime, org)
    done = play(runtime, users["owner"])
    html = login(app, users["viewer"]).get(f"/generations/{done['trace_id']}").text
    assert "store_text = none" in html and "Belgrano" not in html


def test_detail_send_to_review_and_delete(env):
    runtime, org, users, app = env
    activate_detector(runtime, org)
    done = play(runtime, users["owner"])
    reviewer = login(app, users["reviewer"])
    token = csrf_of(reviewer)
    resp = reviewer.post(
        f"/generations/{done['trace_id']}/review",
        data={"csrf_token": token, "target": "0", "min_reviewers": "2"},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    page = reviewer.get(resp.headers["location"])
    assert "quedó en la cola" in page.text and 'href="/review/' in page.text
    bad = reviewer.post(
        f"/generations/{done['trace_id']}/review",
        data={"csrf_token": token, "target": "99", "min_reviewers": "2"},
    )
    assert bad.status_code == 422
    # Solo el owner borra (supresión, Ley 25.326 art. 16).
    assert (
        reviewer.post(
            f"/generations/{done['trace_id']}/delete", data={"csrf_token": token, "confirm": "yes"}
        ).status_code
        == 403
    )
    owner = login(app, users["owner"])
    otoken = csrf_of(owner)
    assert (
        owner.post(f"/generations/{done['trace_id']}/delete", data={"csrf_token": otoken}).status_code == 422
    )
    gone = owner.post(
        f"/generations/{done['trace_id']}/delete",
        data={"csrf_token": otoken, "confirm": "yes"},
        follow_redirects=False,
    )
    assert gone.status_code == 303
    assert owner.get(f"/generations/{done['trace_id']}").status_code == 404


def test_other_tenant_resources_are_404(env):
    runtime, org, users, app = env
    other = runtime.create_org("Otra")
    other_owner = add_user(runtime, other, "owner", "owner@otra.com.ar")
    activate_detector(runtime, other)
    done = play(runtime, other_owner)
    item = runtime.repo(other).create_review_item(done["trace_id"])
    client = login(app, users["owner"])
    token = csrf_of(client)
    assert client.get(f"/generations/{done['trace_id']}").status_code == 404
    assert client.get(f"/review/{item}").status_code == 404
    assert (
        client.post(
            f"/generations/{done['trace_id']}/review",
            data={"csrf_token": token, "target": "response", "min_reviewers": "1"},
        ).status_code
        == 404
    )
    assert (
        client.post(f"/review/{item}/decide", data={"csrf_token": token, "label": "error"}).status_code == 404
    )
    assert runtime.repo(other).get_review_item(item)["state"] == "pending"


def test_panel_errors_render_as_html_with_request_id(env):
    _, _, users, app = env
    resp = login(app, users["viewer"]).get("/generations/no-existe-123456")
    assert resp.status_code == 404
    assert resp.headers["content-type"].startswith("text/html")
    assert 'role="alert"' in resp.text and "Código para soporte" in resp.text and "Reintentar" in resp.text


# --- playground (AT-E07 sin JS y SSE con JS) ------------------------------------------------------


def test_AT_E07_playground_works_without_javascript(env):
    runtime, org, users, app = env
    activate_detector(runtime, org)
    client = login(app, users["reviewer"])
    token = csrf_of(client, "/playground")
    resp = client.post(
        "/playground",
        data={
            "csrf_token": token,
            "prompt": "¿Dónde nació Belgrano?",
            "max_tokens": "16",
            "temperature": "0",
        },
    )
    assert resp.status_code == 200
    assert ">1820</mark>" in resp.text and "nivel alto" in resp.text
    assert "Listo: 8 tokens, 2 marcados." in resp.text
    trace_id = re.search(r'href="/generations/([0-9a-f]+)"', resp.text).group(1)
    assert runtime.repo(org).get_trace(trace_id)["source"] == "playground"


def test_playground_validation_errors_are_inline(env):
    _, _, users, app = env
    client = login(app, users["reviewer"])
    token = csrf_of(client, "/playground")
    resp = client.post(
        "/playground", data={"csrf_token": token, "prompt": " ", "max_tokens": "99999", "temperature": "9"}
    )
    assert resp.status_code == 422
    assert resp.text.count('aria-invalid="true"') == 3
    assert "Escribí un mensaje." in resp.text and "1.024" in resp.text


def test_playground_stream_sse_requires_csrf_header_and_emits_events(env):
    runtime, org, users, app = env
    activate_detector(runtime, org)
    client = login(app, users["reviewer"])
    token = csrf_of(client, "/playground")
    body = {"prompt": "hola", "max_tokens": 16, "temperature": 0}
    assert client.post("/playground/stream", json=body).status_code == 403
    resp = client.post("/playground/stream", json=body, headers={"X-CSRF-Token": token})
    assert resp.status_code == 200 and resp.headers["content-type"].startswith("text/event-stream")
    events = [json.loads(line[6:]) for line in resp.text.splitlines() if line.startswith("data: ")]
    tokens = [e for e in events if e["type"] == "token"]
    assert len(tokens) == 8 and sum(e["flag"] for e in tokens) == 2
    assert events[-1]["type"] == "done" and events[-1]["spans"]
    invalid = client.post("/playground/stream", json={"prompt": ""}, headers={"X-CSRF-Token": token})
    assert invalid.status_code == 422 and "prompt" in invalid.json()["fields"]
    viewer = login(app, users["viewer"])
    vtoken = csrf_of(viewer)
    assert viewer.post("/playground/stream", json=body, headers={"X-CSRF-Token": vtoken}).status_code == 403


# --- modelo y rendimiento ------------------------------------------------------------------------


def test_model_heatmap_with_scroll_container_and_table(env):
    runtime, org, users, app = env
    client = login(app, users["viewer"])
    assert "Todavía no hay datos de uso de expertos" in client.get("/model").text
    play(runtime, users["owner"])
    html = client.get("/model").text
    assert 'class="vg-heatmap-wrap" role="region"' in html and 'tabindex="0"' in html
    assert 'class="vg-chart vg-heatmap"' in html and "Experto 3" in html
    assert "<details" in html and "tiny-panel" in html


def test_usage_matrix_normalizes_per_layer_and_skips_other_shapes(env):
    runtime, org, _, _ = env
    repo = runtime.repo(org)
    old = panel.utcnow() - panel.timedelta(hours=1)
    # Una traza vieja de otra arquitectura no se mezcla: manda la forma del modelo más reciente.
    repo.insert_trace(
        model_id="m",
        response_text="c",
        action="pass",
        usage={"layers": [0], "counts": [[9, 9, 9]]},
        created_at=old,
    )
    repo.insert_trace(
        model_id="m", response_text="a", action="pass", usage={"layers": [0, 1], "counts": [[1, 3], [2, 2]]}
    )
    repo.insert_trace(
        model_id="m", response_text="b", action="pass", usage={"layers": [0, 1], "counts": [[3, 1], [0, 4]]}
    )
    matrix, rows, cols, n = panel.usage_matrix(repo)
    assert rows == ["0", "1"] and cols == ["0", "1"]
    assert n == 2 and matrix == [[0.5, 0.5], [0.25, 0.75]]


def test_performance_latency_chart_against_target(env):
    runtime, org, users, app = env
    client = login(app, users["viewer"])
    assert "Todavía no hay requests con latencia medida" in client.get("/performance").text
    repo = runtime.repo(org)
    for lat, ovh in ((100.0, 2.0), (200.0, 4.0), (120.0, 30.0)):
        repo.insert_trace(
            model_id="m", response_text="x", action="pass", latency_ms=lat, overhead_ms=ovh, n_tokens=10
        )
    html = client.get("/performance").text
    assert 'class="vg-chart vg-latency"' in html and "Meta +5 %" in html
    assert "no cumple" in html  # p99 queda arriba del 5 %
    base, full = panel.latency_percentiles(panel.trace_window(repo, panel.utcnow() - panel.timedelta(days=1)))
    assert base["p50"] == pytest.approx(98.0) and full["p50"] == pytest.approx(120.0)


# --- formato es-AR -------------------------------------------------------------------------------


def test_number_and_date_formatting_es_ar():
    assert panel.fmt_num(1234.5, 1) == "1.234,5"
    assert panel.fmt_count(1234567) == "1.234.567"
    assert panel.fmt_pct(0.0526) == "5,3 %"
    assert panel.fmt_num(None) == "s/d" and panel.fmt_num(float("nan")) == "s/d"
    assert panel.fmt_dt("2026-10-04T02:30:00.000000Z") == "03/10/2026 23:30"
    assert (
        panel.safe_next("//evil.example") == "/"
        and panel.safe_next("/review?estado=pending") == "/review?estado=pending"
    )


def test_segments_split_by_code_points_and_strip_bidi_per_segment():
    text = "año 🇦🇷 ‮dato‬ final"
    marks = panel.marks_from_spans([{"start": 6, "end": 13, "score": 0.6}])
    segs = panel.segments_for(text, marks)
    assert [s["text"] for s in segs] == ["año 🇦🇷", " dato", " final"]
    assert segs[1]["mark"].level == "medium"


def test_full_app_composition_serves_panel(tmp_path, monkeypatch):
    """La app real (`vigia_ee.app.create_app`) monta el panel con la página de errores HTML."""
    app_mod = pytest.importorskip("vigia_ee.app")
    runtime = make_runtime(tmp_path)
    try:
        app = app_mod.create_app(runtime=runtime)
        client = TestClient(app, base_url="https://testserver")
        resp = client.get("/login")
        assert resp.status_code == 200 and "Ingresar al panel" in resp.text
        assert "unsafe-inline" not in resp.headers["content-security-policy"]
        assert client.get("/", follow_redirects=False).status_code == 303
    finally:
        runtime.close()
