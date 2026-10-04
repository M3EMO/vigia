"""Regresiones del panel, ronda 3: ayudas de formulario, overhead, página de error, ROC, CSS y /model."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from _panel_support import add_user, csrf_of, login, make_app, make_runtime, play
from starlette.requests import Request

from vigia_ee import panel, panel_admin

TEMPLATES = Path(panel.TEMPLATES_DIR)
STATIC = TEMPLATES.parent / "static"
ROLES = ("owner", "admin", "reviewer", "viewer")


@pytest.fixture
def env(tmp_path):
    runtime = make_runtime(tmp_path)
    org = runtime.create_org("Acme")
    users = {role: add_user(runtime, org, role) for role in ROLES}
    app = make_app(runtime)
    yield runtime, org, users, app
    runtime.close()


def _describedby(html: str, selector_re: str) -> list[str]:
    tag = re.search(selector_re, html)
    assert tag, selector_re
    attr = re.search(r'aria-describedby="([^"]*)"', tag.group(0))
    return attr.group(1).split() if attr else []


def _text_of(html: str, element_id: str) -> str:
    m = re.search(r'<(\w+)[^>]*\bid="' + re.escape(element_id) + r'"[^>]*>(.*?)</\1>', html, re.S)
    assert m, f"no existe el elemento #{element_id}"
    return re.sub(r"<[^>]+>", "", m.group(2))


def _css_rule(css: str, selector: str) -> str:
    match = re.search(r"(?:^|\})\s*" + re.escape(selector) + r"\s*\{([^}]*)\}", css, re.M)
    assert match, f"falta la regla {selector}"
    return match.group(1)


# --- ayudas de formulario -----------------------------------------------------------------------


def test_form_hints_are_linked_to_their_fields_with_aria_describedby(env):
    """Regresión: 'Las ayudas de formulario no están asociadas a sus campos con aria-describedby
    (contraseña de 15 caracteres, umbrales, URL RAG, retención, formato JSONL)'."""
    _, _, users, app = env
    owner = login(app, users["owner"])
    pages = {
        "/account": [
            (r'<input id="new_password"[^>]*>', "Mínimo 15 caracteres"),
            (r'<input id="confirm_password"[^>]*>', "Mínimo 15 caracteres"),
        ],
        "/policies": [
            (r'<input id="human_review_at"[^>]*>', "Vacío desactiva la acción"),
            (r'<input id="rag_retry_at"[^>]*>', "Vacío desactiva la acción"),
            (r'<input id="abstain_at"[^>]*>', "Vacío desactiva la acción"),
            (r'<input id="rag_url"[^>]*>', "allowlist"),
        ],
        "/settings": [
            (r'<input id="retention_days"[^>]*>', "se purgan solas"),
            (r'<input type="checkbox" name="external_judge_allowed"[^>]*>', "transferencia internacional"),
        ],
        "/calibration": [(r'<input id="file"[^>]*>', "JSONL, una línea por item")],
    }
    for path, fields in pages.items():
        html = owner.get(path).text
        for selector, hint_text in fields:
            ids = _describedby(html, selector)
            assert ids, (path, selector)
            assert any(hint_text in _text_of(html, i) for i in ids), (path, selector, ids)


def test_field_error_keeps_the_hint_in_aria_describedby(env):
    """Regresión: 'Las ayudas de formulario no están asociadas a sus campos con aria-describedby':
    el macro invalid() reemplazaba la ayuda por el error; ahora suma los dos ids."""
    _, _, users, app = env
    client = login(app, users["reviewer"])
    token = csrf_of(client)
    resp = client.post(
        "/account/password",
        data={
            "csrf_token": token,
            "current_password": "x",
            "new_password": "corta",
            "confirm_password": "otra",
        },
    )
    assert resp.status_code == 422
    for field in ("new_password", "confirm_password"):
        tag = re.search(r'<input id="' + field + r'"[^>]*>', resp.text).group(0)
        ids = _describedby(resp.text, r'<input id="' + field + r'"[^>]*>')
        if 'aria-invalid="true"' in tag:
            assert ids == [f"err-{field}", "pw-hint"], ids
            assert _text_of(resp.text, f"err-{field}")
        else:
            assert ids == ["pw-hint"], ids
    assert 'aria-invalid="true"' in resp.text


def test_invalid_macro_without_hint_or_error_adds_nothing():
    """El macro invalid() sin error ni ayuda no emite atributos (los campos sin ayuda no cambian)."""
    macro = panel.ENV.get_template("_macros.html").module.invalid
    assert str(macro({}, "x")) == ""
    assert str(macro({}, "x", "h")) == ' aria-describedby="h"'
    assert str(macro({"x": "mal"}, "x")) == ' aria-invalid="true" aria-describedby="err-x"'
    assert str(macro({"x": "mal"}, "x", "h")) == ' aria-invalid="true" aria-describedby="err-x h"'


# --- overhead -----------------------------------------------------------------------------------


def _kpi(html: str, name: str) -> str:
    m = re.search(r"<dt>" + re.escape(name) + r"</dt><dd>([^<]*)", html)
    assert m, name
    return m.group(1)


def test_overview_and_performance_report_the_same_overhead(env):
    """Regresión: 'Resumen y Rendimiento muestran Overhead p95 distinto para los mismos requests
    (2,4 % contra 0,0 %)'. Las dos páginas usan el percentil del overhead de cada request."""
    runtime, org, users, app = env
    repo = runtime.repo(org)
    # Overhead alto en requests rápidos y nulo en los lentos: el cociente de percentiles de latencia
    # (p95 con / p95 sin - 1) da 0 %, mientras que el p95 del overhead por request da 50 %.
    for lat, ovh in [(15.0, 5.0)] * 3 + [(1000.0, 0.0)] * 17:
        repo.insert_trace(
            model_id="m", response_text="x", action="pass", latency_ms=lat, overhead_ms=ovh, n_tokens=10
        )
    client = login(app, users["viewer"])
    overview = client.get("/").text
    perf = client.get("/performance").text
    ovh_overview = _kpi(overview, "Overhead de Vigía p95")
    ovh_perf = _kpi(perf, "Overhead de Vigía p95")
    assert ovh_overview == ovh_perf, (ovh_overview, ovh_perf)
    assert ovh_perf.startswith("50,0")
    assert "p50 0,0 %" in overview and _kpi(perf, "Overhead de Vigía p50").startswith("0,0")
    assert 'id="ovh-def"' in perf and "igual que en el resumen" in perf
    assert _kpi(perf, "Requests medidos") == "20"


# --- página de error ----------------------------------------------------------------------------


def _main_links(html: str) -> list[tuple[str, str]]:
    main = re.search(r"<main[^>]*>.*?</main>", html, re.S).group(0)
    return re.findall(r'<a class="([^"]*)" href="([^"]+)">', main)


def test_404_page_of_a_get_links_to_the_parent_listing_not_to_retry(env):
    """Regresión: 'La página de error 404 de un GET ofrece Reintentar como acción principal'."""
    _, _, users, app = env
    client = login(app, users["viewer"])
    resp = client.get("/generations/" + "f" * 32)
    assert resp.status_code == 404
    assert "Reintentar" not in resp.text
    links = _main_links(resp.text)
    assert links[0] == ("vg-btn", "/generations"), links
    assert "Volver a generaciones" in resp.text
    assert client.get("/generations").status_code == 200


def _request(method: str, path: str, principal=None, query: str = "") -> Request:
    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "query_string": query.encode(),
        "headers": [],
        "scheme": "https",
        "server": ("testserver", 443),
        "state": {"principal": principal} if principal is not None else {},
    }
    return Request(scope)


def test_retry_link_keeps_retry_only_for_transient_errors(env):
    """Regresión: 'La página de error 404 de un GET ofrece Reintentar como acción principal'.
    Un 5xx o un 429 se reintentan; un 403 o 404 llevan a la sección padre si el rol la ve."""
    runtime, _, users, _ = env
    from _panel_support import principal_for

    viewer = principal_for(runtime, users["viewer"])
    detail = "/generations/abc"
    assert panel.retry_link(_request("GET", detail, viewer, "umbral=0.5"), 500) == (
        f"{detail}?umbral=0.5",
        "Reintentar",
    )
    assert panel.retry_link(_request("GET", detail, viewer), 429)[1] == "Reintentar"
    assert panel.retry_link(_request("GET", detail, viewer), 404) == ("/generations", "Volver a generaciones")
    assert panel.retry_link(_request("GET", detail, viewer), 403) == ("/generations", "Volver a generaciones")
    # La sección misma no tiene padre; una sección que el rol no ve no se ofrece.
    assert panel.retry_link(_request("GET", "/generations", viewer), 404) is None
    assert panel.retry_link(_request("GET", "/calibration/jobs/x", viewer), 404) == (
        "/calibration",
        "Volver a calibración",
    )
    assert panel.retry_link(_request("GET", "/audit/export", viewer), 422) is None
    owner = principal_for(runtime, users["owner"])
    assert panel.retry_link(_request("GET", "/audit/export", owner), 422) == ("/audit", "Volver a auditoría")


# --- ROC del detector ---------------------------------------------------------------------------


def test_detector_roc_legend_and_table_use_readable_labels():
    """Regresión: 'La leyenda y la tabla del ROC del detector muestran ids internos tier:F0,
    tier:todos'."""
    curve = {"fpr": [0.0, 0.5, 1.0], "tpr": [0.0, 0.8, 1.0], "auc": 0.8}
    names = ["detector", "tier:F0", "tier:F1", "tier:F0+F1", "tier:todos", "baseline:out_entropy"]
    report = {"curves": {"roc": [dict(curve, name=n) for n in names]}}
    html = "".join(str(f) for f in panel_admin._report_figures(report))
    assert "tier:" not in html and "baseline:" not in html
    for label in (
        "Detector",
        "Nivel F0",
        "Nivel F1",
        "Nivel F0+F1",
        "Todos los niveles",
        "Baseline out_entropy",
    ):
        assert label in html, label
    assert panel_admin.roc_label("otra") == "otra" and panel_admin.roc_label(None) == ""


# --- CSS ----------------------------------------------------------------------------------------


def test_threshold_apply_button_matches_the_input_height():
    """Regresión: 'El botón Aplicar del umbral de vista mide 28 px junto a un input de 40 px'."""
    css = (STATIC / "app.css").read_text(encoding="utf-8")
    assert "min-height: 2.5rem" in _css_rule(css, ".vg-threshold .vg-btn")
    assert "height: 2.5rem" in css  # el input number mide 2.5rem (regla de inputs)


def test_shell_fills_the_viewport_below_the_topbar_whatever_its_height():
    """Regresión: 'La barra lateral y el shell terminan 6 px antes del borde inferior en páginas
    cortas'. Un calc con la altura supuesta de la barra (3.2rem) no coincide con la real (2.8rem)."""
    css = (STATIC / "app.css").read_text(encoding="utf-8")
    body = _css_rule(css, "body")
    assert "min-height: 100vh" in body and "display: flex" in body and "flex-direction: column" in body
    shell = _css_rule(css, ".vg-shell")
    assert "flex: 1 0 auto" in shell and "calc(" not in shell


def test_code_inside_pre_has_no_inline_padding():
    """Regresión: 'El bloque de código del estado vacío del Resumen indenta solo la primera línea'."""
    css = (STATIC / "app.css").read_text(encoding="utf-8")
    rule = _css_rule(css, "pre code")
    assert "padding: 0" in rule and "background: none" in rule


# --- /model -------------------------------------------------------------------------------------


def _accessible_names(html: str) -> list[str]:
    """Nombres de las regiones de la página: cada <section> con nombre y cada role="region"."""
    names = []
    for tag in re.findall(r"<section[^>]*>|<div[^>]*role=\"region\"[^>]*>", html):
        label = re.search(r'aria-label="([^"]+)"', tag)
        ref = re.search(r'aria-labelledby="([^"]+)"', tag)
        if label:
            names.append(label.group(1))
        elif ref:
            names.append(" ".join(_text_of(html, i) for i in ref.group(1).split()))
    return names


def test_model_page_regions_have_unique_names(env):
    """Regresión: '/model tiene dos regiones con el mismo nombre accesible (axe landmark-unique)'."""
    runtime, _, users, app = env
    play(runtime, users["owner"])
    html = login(app, users["viewer"]).get("/model").text
    assert 'class="vg-heatmap-wrap" role="region"' in html
    names = _accessible_names(html)
    assert "Uso por capa y experto" in names
    assert len(names) == len(set(names)), names
