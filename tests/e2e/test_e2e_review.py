"""AT-E02: dos revisores completan la cola usando solo el teclado; las etiquetas quedan guardadas."""

from __future__ import annotations

import pytest
from _e2e import OWNER_PASSWORD, ConsoleLog, enqueue_for_review, login
from playwright.sync_api import Page, expect

pytestmark = pytest.mark.e2e

MAX_ITEMS = 60


def tab_until(page: Page, predicate: str, *, max_tabs: int = 80) -> None:
    """Aprieta Tab hasta que el elemento con foco cumple `predicate` (JS que recibe el elemento)."""
    for _ in range(max_tabs):
        page.keyboard.press("Tab")
        if page.evaluate(f"(el => {predicate})(document.activeElement)"):
            return
    raise AssertionError(f"ningún elemento enfocable cumple: {predicate}")


def keyboard_login(page: Page, base_url: str, email: str, password: str) -> None:
    page.goto(base_url + "/login")
    tab_until(page, "el && el.id === 'email'")
    page.keyboard.type(email)
    page.keyboard.press("Tab")
    assert page.evaluate("document.activeElement.id") == "password"
    page.keyboard.type(password)
    with page.expect_navigation():
        page.keyboard.press("Enter")
    assert "/login" not in page.url, page.url


def complete_queue_with_keyboard(page: Page, key: str) -> int:
    """Desde cualquier página del panel: va a la cola, toma items y decide con `key` + Ctrl+Enter."""
    tab_until(page, "el && el.tagName === 'A' && el.textContent.trim() === 'Cola de revisión'")
    with page.expect_navigation():
        page.keyboard.press("Enter")
    tab_until(page, "el && el.textContent.trim().startsWith('Tomar el siguiente item')")
    with page.expect_navigation():
        page.keyboard.press("Enter")
    decided = 0
    while "/review/" in page.url and decided < MAX_ITEMS:
        expect(page.locator("[data-review-form]")).to_be_visible()
        page.keyboard.press(key)
        checked = page.locator("[data-review-form] input[name=label]:checked")
        expect(checked).to_have_attribute("data-key", key)
        with page.expect_navigation():
            page.keyboard.press("Control+Enter")
        decided += 1
    assert "ok=queue_empty" in page.url, page.url
    return decided


def test_AT_E02_reviewers_complete_queue_keyboard_only(platform, new_context, accounts, calibrated):
    owner = new_context().new_page()
    login(owner, platform, platform.owner_email, OWNER_PASSWORD)
    trace_ids = [
        platform.chat(f"respuesta para revisar {i}")["choices"][0]["vigia"]["trace_id"] for i in range(3)
    ]
    for trace_id in trace_ids:
        owner.goto(platform.url(f"/generations/{trace_id}"))
        enqueue_for_review(owner, platform, trace_id, min_reviewers=2)
    marks = ",".join("?" for _ in trace_ids)
    # Además de los 3 items manuales puede haber items automáticos (acción «Revisión humana»).
    min_rev = {
        r["id"]: r["min_reviewers"]
        for r in platform.sql(
            f"SELECT id, min_reviewers FROM review_items WHERE trace_id IN ({marks})",  # noqa: S608
            tuple(trace_ids),
        )
    }
    items = list(min_rev)
    assert sum(1 for n in min_rev.values() if n == 2) >= 3, min_rev

    expected = {}
    for email, key, label in (("revisora1@acme.test", "1", "correct"), ("revisor2@acme.test", "2", "error")):
        page = new_context().new_page()
        console = ConsoleLog(page)
        keyboard_login(page, platform.base_url, email, accounts[email])
        decided = complete_queue_with_keyboard(page, key)
        assert decided >= len(items), f"{email} decidió {decided} items"
        assert console.errors == [], console.errors
        expected[email] = label

    for item_id in items:
        state = platform.sql("SELECT state FROM review_items WHERE id = ?", (item_id,))[0]["state"]
        assert state == "completed", (item_id, state)
        rows = platform.sql(
            "SELECT u.email, d.label FROM review_decisions d JOIN users u ON u.id = d.user_id "
            "WHERE d.item_id = ?",
            (item_id,),
        )
        got = {r["email"]: r["label"] for r in rows}
        assert len(got) == min_rev[item_id] and all(expected[e] == lab for e, lab in got.items()), got

    # Con el item completo, las decisiones de ambos revisores se ven en el panel.
    two = next(i for i, n in min_rev.items() if n == 2)
    owner.goto(platform.url(f"/review/{two}"))
    expect(owner.locator("main")).to_contain_text("revisora1@acme.test: Correcto")
    expect(owner.locator("main")).to_contain_text("revisor2@acme.test: Error")
