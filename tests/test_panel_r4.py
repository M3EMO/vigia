"""Regresiones del panel, ronda de cierre: nombres de regiones, columna Objeto, KPIs y nota del overhead.

La alineación y los espacios se miden en el navegador en `tests/e2e/test_e2e_panel_r4.py`. Acá se
fija la causa en el HTML y el CSS que sirve la plataforma.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from _panel_support import add_user, login, make_app, make_runtime, play

from vigia_ee import panel

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


def _css_rule(css: str, selector: str) -> str:
    match = re.search(r"(?:^|\})\s*" + re.escape(selector) + r"\s*\{([^}]*)\}", css, re.M)
    assert match, f"falta la regla {selector}"
    return match.group(1)


def region_names(html: str) -> list[str]:
    """Nombre accesible de cada `role="region"` de la página (aria-label o aria-labelledby)."""
    names = []
    for tag in re.findall(r'<div[^>]*role="region"[^>]*>', html):
        label = re.search(r'aria-label="([^"]+)"', tag)
        ref = re.search(r'aria-labelledby="([^"]+)"', tag)
        if label:
            names.append(label.group(1))
        elif ref:
            texts = []
            for element_id in ref.group(1).split():
                m = re.search(r'<(\w+)[^>]*\bid="' + re.escape(element_id) + r'"[^>]*>(.*?)</\1>', html, re.S)
                texts.append(re.sub(r"<[^>]+>", "", m.group(2)) if m else "")
            names.append(" ".join(texts))
        else:
            names.append("")
    return names


def test_chart_and_its_data_table_regions_have_distinct_names(env):
    """Regresión: 'Gráficos con 'Ver datos' abierto: la región del SVG y la región de la tabla tienen
    el mismo nombre accesible (axe landmark-unique)'. El gráfico se llama «Gráfico: <título>» y la
    tabla conserva el caption."""
    runtime, _, users, app = env
    play(runtime, users["owner"])
    client = login(app, users["viewer"])
    expected = {"/": "Tasa de tokens marcados por día", "/performance": "Latencia por request"}
    for path, title in expected.items():
        names = region_names(client.get(path).text)
        assert f"Gráfico: {title}" in names, (path, names)
        assert title in names, (path, names)  # la región de la tabla de datos
        assert len(names) == len(set(names)), (path, names)


def test_chart_region_name_escapes_the_svg_title():
    """El título del SVG ya viene escapado: el nombre de la región no lo escapa dos veces."""
    svg = '<svg role="img"><title id="vg-x-t">Tasa &lt;p&gt; &amp; &quot;x&quot;</title></svg>'
    region = panel.scrollable_chart(svg)
    assert 'aria-label="Gráfico: Tasa &lt;p&gt; &amp; &quot;x&quot;"' in region
    assert 'aria-label="Gráfico"' in panel.scrollable_chart("<svg></svg>")


def test_audit_object_ids_do_not_break_anywhere(env):
    """Regresión: 'Auditoría: la columna Objeto mide 142 px y parte cada id de 32 caracteres en 3
    líneas'. Objeto no usa .vg-break (overflow-wrap:anywhere): su min-content es el id entero. Solo
    Detalle (JSON largo) puede cortarse en cualquier carácter."""
    _, _, users, app = env
    html = login(app, users["owner"]).get("/audit").text
    objects = re.findall(r'<td data-label="Objeto"[^>]*>', html)
    assert objects, "la auditoría debería tener eventos de login"
    assert all("vg-break" not in td for td in objects), objects
    assert '<td data-label="Detalle" class="vg-break">' in html


def test_kpi_values_share_a_row_line_when_a_label_wraps():
    """Regresión: 'Rendimiento: el valor del KPI 'Tokens por segundo (p50)' queda 21 px más abajo que
    los demás'. Cada KPI ocupa dos filas de la grilla con subgrid: el rótulo más alto agranda la fila
    de rótulos de todas las tarjetas y los valores quedan en la misma línea."""
    css = (STATIC / "app.css").read_text(encoding="utf-8")
    kpi = _css_rule(css, ".vg-kpi")
    assert "grid-template-rows: subgrid" in kpi and "grid-row: span 2" in kpi
    assert "display: grid" in kpi
    assert "margin: 0;" in _css_rule(css, ".vg-kpi dd")  # sin margen propio: el row-gap separa


def test_overhead_note_has_room_before_the_next_card(env):
    """Regresión: 'Rendimiento: la nota que define el overhead queda pegada a la tarjeta siguiente
    (0 px) y parece pertenecer al gráfico'. La nota queda más cerca de los KPIs que del gráfico."""
    runtime, _, users, app = env
    play(runtime, users["owner"])
    html = login(app, users["viewer"]).get("/performance").text
    assert re.search(r'<p class="[^"]*\bvg-kpis-note\b[^"]*" id="ovh-def">', html)
    css = (STATIC / "app.css").read_text(encoding="utf-8")
    assert "margin-bottom: 1rem" in _css_rule(css, ".vg-hint.vg-kpis-note")
    assert "margin-bottom: .5rem" in _css_rule(css, ".vg-kpis:has(+ .vg-kpis-note)")
