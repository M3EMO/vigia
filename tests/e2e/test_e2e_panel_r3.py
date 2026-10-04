"""Regresiones del panel (ronda 3) medidas en el navegador: badge del playground, alturas y alineación."""

from __future__ import annotations

import pytest
from _e2e import OWNER_B_PASSWORD, OWNER_PASSWORD, ConsoleLog, login
from playwright.sync_api import expect

pytestmark = pytest.mark.e2e


def test_playground_result_shows_the_action_as_a_badge(platform, new_context):
    """Regresión: 'El resultado del playground con JS muestra la acción como texto plano y sin JS
    como badge'."""
    page = new_context().new_page()
    console = ConsoleLog(page)
    login(page, platform, platform.owner_email, OWNER_PASSWORD)
    page.goto(platform.url("/playground"))
    page.fill("#prompt", "Decí una palabra.")
    page.fill("#max_tokens", "4")
    page.click("[data-playground-submit]")
    expect(page.locator("[data-playground-status]")).to_contain_text("Listo", timeout=180_000)
    badge = page.locator("[data-playground-result] dl.vg-meta dd span.vg-badge")
    expect(badge).to_have_count(1)
    assert badge.inner_text() in ("Pasa", "Anotar", "Revisión humana", "Re-consulta RAG", "Abstención")
    assert console.errors == [], console.errors


def test_shell_and_nav_reach_the_bottom_of_the_viewport(platform, new_context):
    """Regresión: 'La barra lateral y el shell terminan 6 px antes del borde inferior en páginas
    cortas'."""
    page = new_context(viewport={"width": 1280, "height": 900}).new_page()
    login(page, platform, platform.owner_email, OWNER_PASSWORD)
    for path in ("/account", "/model", "/performance"):
        page.goto(platform.url(path))
        bottoms = page.evaluate(
            "({shell: document.querySelector('.vg-shell').getBoundingClientRect().bottom,"
            " nav: document.querySelector('.vg-nav').getBoundingClientRect().bottom,"
            " scroll: document.documentElement.scrollHeight, inner: innerHeight})"
        )
        if bottoms["scroll"] <= bottoms["inner"]:  # página corta: sin scroll vertical
            assert abs(bottoms["shell"] - bottoms["inner"]) < 0.5, (path, bottoms)
            assert abs(bottoms["nav"] - bottoms["inner"]) < 0.5, (path, bottoms)


def test_empty_overview_code_block_lines_are_aligned(platform, new_context):
    """Regresión: 'El bloque de código del estado vacío del Resumen indenta solo la primera línea'."""
    page = new_context().new_page()
    login(page, platform, platform.owner_b_email, OWNER_B_PASSWORD)
    page.goto(platform.url("/"))
    lefts = page.evaluate(
        """() => {
          const code = document.querySelector('main pre code');
          const node = code.firstChild;
          const text = node.textContent;
          const second = text.indexOf('\\n') + 1;
          const at = (i) => { const r = document.createRange(); r.setStart(node, i); r.setEnd(node, i + 1);
                              return r.getBoundingClientRect().left; };
          return [at(0), at(second)];
        }"""
    )
    assert abs(lefts[0] - lefts[1]) < 0.5, lefts


def test_threshold_apply_button_is_as_tall_as_the_number_input(platform, new_context, calibrated):
    """Regresión: 'El botón Aplicar del umbral de vista mide 28 px junto a un input de 40 px'."""
    trace_id = platform.chat("respuesta para medir el umbral", max_tokens=12)["choices"][0]["vigia"][
        "trace_id"
    ]
    page = new_context(viewport={"width": 1280, "height": 900}).new_page()
    login(page, platform, platform.owner_email, OWNER_PASSWORD)
    page.goto(platform.url(f"/generations/{trace_id}?umbral=0.50"))
    form = page.locator("[data-threshold-form]")
    number = form.locator("input[type=number]").bounding_box()
    button = form.locator("button[type=submit]").bounding_box()
    assert abs(number["height"] - button["height"]) < 0.5, (number, button)


def test_model_page_landmarks_are_unique_for_axe(platform, new_context):
    """Regresión: '/model tiene dos regiones con el mismo nombre accesible (axe landmark-unique)'."""
    axe_module = pytest.importorskip("axe_playwright_python.sync_playwright")
    platform.chat("una generación para el mapa de calor", max_tokens=6)
    page = new_context().new_page()
    login(page, platform, platform.owner_email, OWNER_PASSWORD)
    page.goto(platform.url("/model"))
    expect(page.locator(".vg-heatmap-wrap")).to_be_visible()
    results = axe_module.Axe().run(page, options={"runOnly": ["landmark-unique"]})
    violations = results.response["violations"]
    assert violations == [], violations
