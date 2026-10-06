"""Regresiones del panel (ronda de cierre) medidas en el navegador: regiones, auditoría y KPIs."""

from __future__ import annotations

import pytest
from _e2e import OWNER_PASSWORD, login

pytestmark = pytest.mark.e2e

REGION_NAMES = """() => [...document.querySelectorAll('[role=region]')].map((el) => {
  const label = el.getAttribute('aria-label');
  if (label) return label;
  return (el.getAttribute('aria-labelledby') || '').split(' ')
    .map((id) => (document.getElementById(id) || {}).textContent || '').join(' ').trim();
})"""


def _page_with_traffic(platform, new_context):
    platform.chat("respuesta para medir rendimiento", max_tokens=8)
    page = new_context(viewport={"width": 1280, "height": 900}).new_page()
    login(page, platform, platform.owner_email, OWNER_PASSWORD)
    return page


def test_chart_regions_are_unique_with_data_tables_open(platform, new_context):
    """Regresión: 'Gráficos con 'Ver datos' abierto: la región del SVG y la región de la tabla tienen
    el mismo nombre accesible (axe landmark-unique)'."""
    page = _page_with_traffic(platform, new_context)
    for path in ("/", "/performance"):
        page.goto(platform.url(path))
        page.evaluate("document.querySelectorAll('details').forEach((d) => { d.open = true; })")
        names = page.evaluate(REGION_NAMES)
        assert len(names) == len(set(names)), (path, names)
    assert "Gráfico: Latencia por request" in names and "Latencia por request" in names, names


def test_performance_kpi_values_share_a_line_and_the_note_sits_with_them(platform, new_context):
    """Regresión: 'Rendimiento: el valor del KPI 'Tokens por segundo (p50)' queda 21 px más abajo
    que los demás' y 'Rendimiento: la nota que define el overhead queda pegada a la tarjeta siguiente
    (0 px) y parece pertenecer al gráfico'."""
    page = _page_with_traffic(platform, new_context)
    page.goto(platform.url("/performance"))
    m = page.evaluate(
        """() => {
          const tops = [...document.querySelectorAll('.vg-kpis .vg-kpi')]
            .map((k) => [k.getBoundingClientRect().top, k.querySelector('dd').getBoundingClientRect().top]);
          const note = document.getElementById('ovh-def').getBoundingClientRect();
          const kpis = document.querySelector('.vg-kpis').getBoundingClientRect();
          const card = document.querySelector('section[aria-labelledby=lat-title]').getBoundingClientRect();
          return {tops, above: note.top - kpis.bottom, below: card.top - note.bottom};
        }"""
    )
    rows: dict[int, list[float]] = {}
    for card_top, dd_top in m["tops"]:
        rows.setdefault(round(card_top), []).append(dd_top)
    assert len(m["tops"]) == 5 and len(rows) == 1, m  # a 1280 px los cinco KPIs van en una fila
    for dd_tops in rows.values():
        assert max(dd_tops) - min(dd_tops) < 0.5, m
    assert m["below"] >= 12, m
    assert m["above"] < m["below"], m


def test_audit_object_ids_fit_on_one_line(platform, new_context):
    """Regresión: 'Auditoría: la columna Objeto mide 142 px y parte cada id de 32 caracteres en 3
    líneas'."""
    page = new_context(viewport={"width": 1280, "height": 900}).new_page()
    login(page, platform, platform.owner_email, OWNER_PASSWORD)
    page.goto(platform.url("/audit"))
    lines = page.evaluate(
        "[...document.querySelectorAll('td[data-label=Objeto] code')]"
        ".map((c) => [c.textContent.length, c.getClientRects().length])"
    )
    ids = [n for length, n in lines if length <= 40]
    assert ids, lines
    assert all(n == 1 for n in ids), lines
