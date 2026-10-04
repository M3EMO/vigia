"""Atajos de teclado y preferencias del navegador: detalle (j/k/Enter/?), tema y menú móvil."""

from __future__ import annotations

import pytest
from _e2e import OWNER_PASSWORD, ConsoleLog, login
from playwright.sync_api import expect

pytestmark = pytest.mark.e2e


def test_detail_fragments_with_keyboard(platform, new_context, calibrated):
    trace_id = platform.chat("respuesta para recorrer con el teclado", max_tokens=24)["choices"][0]["vigia"][
        "trace_id"
    ]
    page = new_context().new_page()
    console = ConsoleLog(page)
    login(page, platform, platform.owner_email, OWNER_PASSWORD)
    page.goto(platform.url(f"/generations/{trace_id}?umbral=0"))
    marks = page.locator("[data-marked-text] mark[data-fragment]")
    assert marks.count() > 0
    page.keyboard.press("j")
    focused = page.evaluate("document.activeElement.getAttribute('data-fragment')")
    assert focused == "1", focused
    current = page.evaluate(
        "[...document.querySelectorAll('[data-signals].is-current')].map(e => e.dataset.signals)"
    )
    assert current == ["1"], current
    page.keyboard.press("Enter")
    assert page.evaluate("document.activeElement.closest('#sig-1') !== null"), "Enter no llevó a las señales"
    page.keyboard.press("k")
    assert page.evaluate("document.activeElement.tagName") == "MARK"
    page.keyboard.press("?")
    expect(page.locator("dialog#shortcuts")).to_be_visible()
    page.keyboard.press("Escape")
    expect(page.locator("dialog#shortcuts")).to_be_hidden()
    assert console.errors == [], console.errors


def test_theme_choice_cycles_and_persists_across_pages(platform, new_context):
    page = new_context(color_scheme="light").new_page()
    login(page, platform, platform.owner_email, OWNER_PASSWORD)
    toggle = page.locator("[data-theme-toggle]")
    expect(toggle).to_have_text("Tema: sistema")
    light_bg = page.evaluate("getComputedStyle(document.body).backgroundColor")
    toggle.click()
    expect(toggle).to_have_text("Tema: claro")
    toggle.click()
    expect(toggle).to_have_text("Tema: oscuro")
    page.goto(platform.url("/generations"))
    expect(page.locator("[data-theme-toggle]")).to_have_text("Tema: oscuro")
    assert page.evaluate("document.documentElement.dataset.theme") == "dark"
    assert page.evaluate("getComputedStyle(document.body).backgroundColor") != light_bg
    page.locator("[data-theme-toggle]").click()
    expect(page.locator("[data-theme-toggle]")).to_have_text("Tema: sistema")
    page.reload()
    assert page.evaluate("document.documentElement.hasAttribute('data-theme')") is False


def test_mobile_menu_opens_with_keyboard_and_focuses_first_link(platform, new_context):
    page = new_context(viewport={"width": 375, "height": 812}).new_page()
    login(page, platform, platform.owner_email, OWNER_PASSWORD)
    nav = page.locator("#vg-nav")
    expect(nav).to_be_hidden()
    page.focus("[data-nav-toggle]")
    page.keyboard.press("Enter")
    expect(nav).to_be_visible()
    assert page.evaluate("document.activeElement.closest('#vg-nav') !== null")
    page.keyboard.press("Tab")
    with page.expect_navigation():
        page.keyboard.press("Enter")
    assert page.url.endswith("/generations"), page.url


def test_threshold_slider_moves_several_steps_with_the_keyboard(platform, new_context, calibrated):
    """Con el teclado, cada flecha mueve el deslizador del umbral un paso. Si cada flecha recarga la
    página y el foco cae al body, la persona no puede mover el umbral más de un paso seguido."""
    trace_id = platform.chat("respuesta para mover el umbral", max_tokens=24)["choices"][0]["vigia"][
        "trace_id"
    ]
    page = new_context().new_page()
    login(page, platform, platform.owner_email, OWNER_PASSWORD)
    page.goto(platform.url(f"/generations/{trace_id}?umbral=0.50"))
    slider = page.locator("[data-threshold-range]")
    expect(slider).to_be_visible()
    slider.focus()
    for _ in range(3):
        page.keyboard.press("ArrowRight")
        page.wait_for_timeout(500)
        page.wait_for_load_state()
    on_slider = page.evaluate(
        "document.activeElement && document.activeElement.matches('[data-threshold-range]')"
    )
    value = page.locator("[data-threshold-range]").input_value()
    assert on_slider and abs(float(value) - 0.53) < 1e-9, (on_slider, value, page.url)
