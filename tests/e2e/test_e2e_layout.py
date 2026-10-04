"""Regresiones visuales y de teclado del panel medidas en Chromium (ronda 1 de reparación del panel).

Cada test cita en su docstring el título del defecto que cubre. Las medidas salen del layout real:
getBoundingClientRect, estilos computados y contraste calculado como en WCAG.
"""

from __future__ import annotations

import pytest
from _e2e import OWNER_PASSWORD, login

pytestmark = pytest.mark.e2e

CONTRAST_JS = """(sel) => {
  const el = document.querySelector(sel);
  const parse = c => c.match(/[\\d.]+/g).map(Number);
  const f = v => { v /= 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); };
  const lum = rgb => {
    return 0.2126 * f(rgb[0]) + 0.7152 * f(rgb[1]) + 0.0722 * f(rgb[2]); };
  const ph = getComputedStyle(el, '::placeholder');
  const fg = parse(ph.color), bg = parse(getComputedStyle(el).backgroundColor);
  const alpha = (fg.length > 3 ? fg[3] : 1) * parseFloat(ph.opacity || '1');
  const mix = [0, 1, 2].map(i => fg[i] * alpha + bg[i] * (1 - alpha));
  const a = lum(mix), b = lum(bg);
  return (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);
}"""

SVG_JS = """() => [...document.querySelectorAll('svg[role=img]')].map(s => {
  const base = s.getAttribute('font-size') || '12';
  const sizes = [...s.querySelectorAll('text')].map(t => parseFloat(t.getAttribute('font-size') || base));
  const scale = s.getBoundingClientRect().width / s.viewBox.baseVal.width;
  const cls = s.getAttribute('class');
  return {cls, min_px: Math.min(...sizes) * scale, max_px: Math.max(...sizes) * scale};
})"""

UNFOCUSABLE_SCROLL_JS = """() => [...document.querySelectorAll('main *')].filter(e => {
  const cs = getComputedStyle(e);
  const scrolls = ['auto', 'scroll'].includes(cs.overflowX) && e.scrollWidth > e.clientWidth + 1;
  const inner = e.querySelector('a[href], button, input, select, textarea, [tabindex]');
  return scrolls && e.tabIndex < 0 && !inner;
}).map(e => e.className || e.tagName)"""


def _owner_page(platform, new_context, **ctx_kw):
    page = new_context(**ctx_kw).new_page()
    login(page, platform, platform.owner_email, OWNER_PASSWORD)
    return page


def test_playground_keyboard_focus_stays_on_the_button(platform, new_context):
    """Regresión: 'El foco se pierde al body al generar en el playground con teclado'."""
    page = _owner_page(platform, new_context)
    page.goto(platform.url("/playground"))
    page.fill("#prompt", "hola")
    page.fill("#max_tokens", "24")
    # Con el modelo diminuto la generación dura menos que un sondeo de Playwright: un
    # MutationObserver registra el foco en el instante exacto en que cambia aria-busy.
    page.evaluate(
        """() => { window.__busy = []; const out = document.querySelector('[data-playground-output]');
          new MutationObserver(() => { const a = document.activeElement;
            window.__busy.push([out.getAttribute('aria-busy'), a.tagName, a.getAttribute('aria-disabled')]);
          }).observe(out, {attributes: true, attributeFilter: ['aria-busy']}); }"""
    )
    streams: list[str] = []
    page.on("request", lambda r: streams.append(r.url) if r.url.endswith("/playground/stream") else None)
    page.focus("[data-playground-submit]")
    page.keyboard.press("Enter")
    page.keyboard.press("Enter")  # un segundo Enter durante la generación no lanza otro request
    page.wait_for_function("() => window.__busy.some(b => b[0] === 'false')", timeout=120_000)
    busy = page.evaluate("window.__busy")
    after = page.evaluate(
        "() => [document.activeElement.tagName, document.activeElement.hasAttribute('aria-disabled')]"
    )
    assert busy[0] == ["true", "BUTTON", "true"], busy
    assert busy[-1][:2] == ["false", "BUTTON"], busy
    assert after == ["BUTTON", False], after
    assert len(streams) == 1, streams
    assert page.inner_text("[data-playground-status]").startswith("Listo:")


@pytest.mark.parametrize("path", ["/generations", "/audit"])
def test_filter_controls_share_height_and_label_line(platform, new_context, path):
    """Regresión: 'Los controles del filtro de Generaciones y Auditoría tienen alturas distintas y
    desalinean los labels'."""
    page = _owner_page(platform, new_context, viewport={"width": 1280, "height": 900})
    page.goto(platform.url(path))
    boxes = page.evaluate(
        """() => { const f = document.querySelector('main form[method="get"]');
          const r = s => [...f.querySelectorAll(s)].map(e => e.getBoundingClientRect());
          return {labels: r('.vg-field label').map(b => Math.round(b.top)),
                  controls: r('input:not([type=hidden]), select').map(b => Math.round(b.height))}; }"""
    )
    assert len(set(boxes["controls"])) == 1, boxes
    assert len(set(boxes["labels"])) == 1, boxes


def test_placeholder_contrast_in_dark_theme(platform, new_context):
    """Regresión: 'El placeholder de los inputs no cumple 4,5:1 en tema oscuro (3,75:1)'."""
    for scheme in ("dark", "light"):
        page = _owner_page(platform, new_context, color_scheme=scheme)
        page.goto(platform.url("/policies"))
        ratio = page.evaluate(CONTRAST_JS, "#rag_url")
        assert ratio >= 4.5, (scheme, ratio)


def test_form_card_heading_has_no_extra_top_gap(platform, new_context):
    """Regresión: 'La tarjeta Editar la política tiene un hueco superior de unos 42 px: el h2 conserva
    su margin-top'."""
    page = _owner_page(platform, new_context, viewport={"width": 1280, "height": 900})
    page.goto(platform.url("/policies"))
    gaps = page.evaluate(
        """() => ['#alpha-title', '#pol-title'].map(s => { const h = document.querySelector(s);
          const c = h.closest('.vg-card');
          return Math.round(h.getBoundingClientRect().top - c.getBoundingClientRect().top); })"""
    )
    assert gaps[0] == gaps[1], gaps


def test_chart_text_is_readable_on_model_and_performance(platform, new_context):
    """Regresión: 'Los gráficos SVG escalan al ancho del contenedor: texto de 6,7 px en el reporte del
    detector, de 21 px en Rendimiento y de 58 px en el heatmap'."""
    platform.chat("una generación para tener uso de expertos y latencia", max_tokens=8)
    page = _owner_page(platform, new_context, viewport={"width": 1280, "height": 900})
    for path in ("/model", "/performance"):
        page.goto(platform.url(path))
        charts = page.evaluate(SVG_JS)
        assert charts, path
        for c in charts:
            assert 9.5 <= c["min_px"] and c["max_px"] <= 15, (path, c)


def test_detector_report_and_alpha_tables(platform, new_context, calibrated):
    """Regresiones: 'Tablas angostas de alphas sin foco: violación serious scrollable-region-focusable
    a 375 px y a 320 px (reflow)' y el texto de 6,7 px de los gráficos del reporte del detector."""
    detector = f"/calibration/detectors/{calibrated['detector_id']}"
    page = _owner_page(platform, new_context, viewport={"width": 1280, "height": 900})
    page.goto(platform.url(detector))
    charts = page.evaluate(SVG_JS)
    assert charts and all(9.5 <= c["min_px"] and c["max_px"] <= 15 for c in charts), charts
    for width in (375, 320):
        narrow = _owner_page(platform, new_context, viewport={"width": width, "height": 800})
        for path in ("/policies", detector):
            narrow.goto(platform.url(path))
            assert narrow.evaluate(UNFOCUSABLE_SCROLL_JS) == [], (width, path)
            alpha_cells = narrow.locator("table:has(caption:text-matches('Umbral')) tbody th")
            assert alpha_cells.count() > 0, path
            for i in range(alpha_cells.count()):
                cell = alpha_cells.nth(i).bounding_box()
                assert cell and cell["height"] < 40, (
                    width,
                    path,
                    cell,
                )  # una línea, sin un carácter por renglón
