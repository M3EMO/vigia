"""Panel, ronda de cierre: tabla de Auditoría con zoom de 200 % y atajos de una tecla (WCAG 2.1.4).

Cada test usa una org propia para no depender de la cola ni de la auditoría de otros módulos.
"""

from __future__ import annotations

import pytest
from _e2e import ConsoleLog, create_org, enqueue_for_review, login
from playwright.sync_api import Page, expect

pytestmark = pytest.mark.e2e

PASSWORD = "Clave-Larga-De-La-Ronda-De-Cierre-2026"  # noqa: S105 - credencial de prueba local
WCAG_TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"]


def _org_with_review_item(platform, new_context, slug: str) -> tuple[Page, str, str]:
    """Org nueva con una generación en la cola (un revisor). Devuelve (página del owner, trace, item)."""
    email = f"duena@{slug}.test"
    _org_id, key = create_org(platform, f"Org {slug}", email, PASSWORD)
    trace_id = platform.chat(f"respuesta de la org {slug}", key=key)["choices"][0]["vigia"]["trace_id"]
    page = new_context().new_page()
    login(page, platform, email, PASSWORD)
    page.goto(platform.url(f"/generations/{trace_id}"))
    enqueue_for_review(page, platform, trace_id, min_reviewers=1)
    item_id = platform.sql("SELECT id FROM review_items WHERE trace_id = ?", (trace_id,))[0]["id"]
    return page, trace_id, item_id


def _take_item(page: Page, platform, item_id: str) -> None:
    page.goto(platform.url("/review"))
    with page.expect_navigation():
        page.get_by_role("button", name="Tomar el siguiente item").click()
    assert page.url.endswith(f"/review/{item_id}"), page.url
    expect(page.locator("[data-review-form]")).to_be_visible()


def _set_char_keys(page: Page, platform, on: bool) -> None:
    page.goto(platform.url("/account"))
    toggle = page.locator("[data-shortcuts-toggle]")
    expect(toggle).to_be_visible()
    expect(toggle).to_be_checked()
    if not on:
        toggle.uncheck()
        expect(page.locator("[data-shortcuts-status]")).to_have_text("Atajos de una tecla apagados.")


def test_audit_table_fits_at_1024_and_is_operable_at_200_percent_zoom(platform, new_context):
    """Regresión: 'Auditoría: entre 640 y 1080 px de ancho la tabla desborda su contenedor, la columna
    Detalle queda fuera de vista y axe marca scrollable-region-focusable (serious)' (AT-E04).

    El id de 32 caracteres de Objeto se corta y la tabla entra a 1024 px; si igual desborda (640 px,
    zoom de 200 % sobre 1280), la región desplazable recibe el foco y tiene nombre."""
    axe_module = pytest.importorskip("axe_playwright_python.sync_playwright")
    page, _trace_id, _item_id = _org_with_review_item(platform, new_context, "cierreaudit")
    console = ConsoleLog(page)

    page.set_viewport_size({"width": 1024, "height": 768})
    page.goto(platform.url("/audit"))
    wrap = page.locator(".vg-table-wrap")
    expect(wrap).to_have_count(1)
    assert page.locator("td[data-label='Objeto'] code").count() > 0, "la auditoría no tiene objetos"
    over = wrap.evaluate("w => w.scrollWidth - w.clientWidth")
    assert over <= 0, f"la tabla de auditoría desborda {over}px a 1024 px"
    right = page.evaluate(
        "() => Math.max(...[...document.querySelectorAll('.vg-table thead th')]"
        ".map(th => th.getBoundingClientRect().right))"
    )
    assert right <= 1024, f"la columna Detalle termina en x={right} sobre 1024 px"

    axe = axe_module.Axe()
    for width, height in ((640, 450), (1024, 768)):
        page.set_viewport_size({"width": width, "height": height})
        page.goto(platform.url("/audit"))
        results = axe.run(page, options={"runOnly": WCAG_TAGS}).response
        serious = [
            (v["id"], [n["target"] for n in v["nodes"][:3]])
            for v in results["violations"]
            if v.get("impact") in ("serious", "critical")
        ]
        assert not serious, f"axe en /audit a {width}x{height}: {serious}"
    assert console.errors == [], console.errors


def test_char_key_shortcuts_can_be_turned_off(platform, new_context):
    """Regresión: 'Los atajos de una sola tecla (j, k, ?, 1, 2, 3) actúan en toda la página y no se
    pueden desactivar (WCAG 2.1.4 Character Key Shortcuts, nivel A)' (AT-E02, AT-E04).

    Prendidos (el default) siguen andando; apagados desde «Mi cuenta», ni ? ni 2 hacen nada aunque
    el foco esté en un link de la página. Ctrl+Enter no es una tecla sola y sigue andando."""
    page, trace_id, item_id = _org_with_review_item(platform, new_context, "cierreatajos")
    console = ConsoleLog(page)

    # Prendidos: ? abre la hoja del detalle y 2 marca «error» en la revisión.
    page.goto(platform.url(f"/generations/{trace_id}"))
    page.keyboard.press("?")
    expect(page.locator("dialog#shortcuts")).to_be_visible()
    expect(page.locator("dialog#shortcuts a[href='/account']")).to_have_text("Mi cuenta")
    page.keyboard.press("Escape")
    _take_item(page, platform, item_id)
    page.keyboard.press("2")
    expect(page.locator("[data-review-form] input[name=label]:checked")).to_have_attribute("data-key", "2")

    _set_char_keys(page, platform, on=False)

    page.goto(platform.url(f"/generations/{trace_id}"))
    page.locator("main a").first.focus()
    focused = page.evaluate("document.activeElement.outerHTML")
    for key in ("?", "j", "k"):
        page.keyboard.press(key)
    expect(page.locator("dialog#shortcuts")).to_be_hidden()
    assert page.evaluate("document.activeElement.outerHTML") == focused, "un atajo movió el foco"

    page.goto(platform.url(f"/review/{item_id}"))
    expect(page.locator("[data-review-form]")).to_be_visible()
    page.locator("main a").first.focus()
    for key in ("1", "2", "3", "?"):
        page.keyboard.press(key)
    expect(page.locator("[data-review-form] input[name=label]:checked")).to_have_count(0)
    expect(page.locator("dialog#shortcuts")).to_be_hidden()

    # La preferencia persiste en «Mi cuenta» y se vuelve a prender.
    page.goto(platform.url("/account"))
    toggle = page.locator("[data-shortcuts-toggle]")
    expect(toggle).not_to_be_checked()
    toggle.check()
    page.goto(platform.url(f"/review/{item_id}"))
    page.keyboard.press("3")
    expect(page.locator("[data-review-form] input[name=label]:checked")).to_have_attribute("data-key", "3")
    assert console.errors == [], console.errors
