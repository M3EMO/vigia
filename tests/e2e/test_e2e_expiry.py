"""Sesiones que vencen con una página abierta: formularios, playground y "Salir".

El vencimiento se fuerza moviendo `last_seen_at` 31 minutos atrás (el reloj no se puede esperar).
Ninguna acción puede dejar a la persona en una página sin salida (un 405 en JSON, por ejemplo).
"""

from __future__ import annotations

import pytest
from _e2e import OWNER_PASSWORD, ConsoleLog, invite_user, iso_ago, login
from playwright.sync_api import Page, expect

pytestmark = pytest.mark.e2e


@pytest.fixture
def expiring_admin(platform, new_context):
    """Una cuenta admin propia por test y una función que vence sus sesiones."""

    def make(email: str) -> tuple[Page, str, object]:
        owner = new_context().new_page()
        login(owner, platform, platform.owner_email, OWNER_PASSWORD)
        password = invite_user(owner, platform, email, "admin")
        owner.close()
        page = new_context().new_page()
        login(page, platform, email, password)
        uid = platform.sql("SELECT id FROM users WHERE email = ?", (email,))[0]["id"]

        def expire() -> None:
            platform.sql("UPDATE sessions SET last_seen_at = ? WHERE user_id = ?", (iso_ago(31), uid))

        return page, password, expire

    return make


def error_page_links(page: Page, platform) -> dict[str, int]:
    """Sigue cada enlace de la página de error y devuelve {href: status} de los que no responden HTML."""
    hrefs = page.eval_on_selector_all("main a[href]", "as => as.map(a => a.getAttribute('href'))")
    assert hrefs, "la página de error no ofrece ningún enlace"
    broken = {}
    for href in hrefs:
        resp = page.goto(platform.url(href))
        content_type = (resp.headers.get("content-type", "") if resp else "") or ""
        if resp is None or resp.status >= 400 or not content_type.startswith("text/html"):
            broken[href] = resp.status if resp else 0
    return broken


def test_form_post_after_expiry_offers_a_working_way_back(platform, expiring_admin):
    page, _password, expire = expiring_admin("vence-politica@acme.test")
    page.goto(platform.url("/policies"))
    expire()
    page.click("form[action='/policies'] button[type=submit]")
    page.wait_for_load_state()
    if "/login" in page.url:
        return  # redirigir al login también es una salida válida
    expect(page.locator("h1")).to_have_text("Sesión vencida")
    assert error_page_links(page, platform) == {}


def test_error_page_after_a_failed_post_has_no_dead_retry_link(platform, expiring_admin):
    """Un POST que falla (acá, una decisión de revisión con la sesión vencida) muestra la página de
    error; su "Reintentar" no puede apuntar a la ruta POST-only (GET → 405 en JSON)."""
    page, _password, expire = expiring_admin("vence-revision@acme.test")
    trace_id = platform.chat("item que se decide con la sesión vencida")["choices"][0]["vigia"]["trace_id"]
    page.goto(platform.url(f"/generations/{trace_id}"))
    page.fill("#min_reviewers", "1")
    page.click(f"form[action='/generations/{trace_id}/review'] button[type=submit]")
    page.wait_for_load_state()
    query = "SELECT id FROM review_items WHERE trace_id = ? ORDER BY created_at DESC"
    rows = platform.sql(query, (trace_id,))
    item_id = rows[0]["id"]
    page.goto(platform.url(f"/review/{item_id}"))
    page.check("[data-review-form] input[name=label][value=correct]")
    expire()
    page.click("[data-review-form] button[name=next][value='0']")
    page.wait_for_load_state()
    if "/login" in page.url:
        return
    assert error_page_links(page, platform) == {}


def test_logout_with_an_expired_session_lands_on_login(platform, expiring_admin):
    page, _password, expire = expiring_admin("vence-salir@acme.test")
    page.goto(platform.url("/account"))
    expire()
    page.get_by_role("button", name="Salir").click()
    page.wait_for_load_state()
    assert "/login" in page.url, (page.url, page.locator("main").inner_text()[:200])


def test_playground_stream_reports_the_expired_session(platform, expiring_admin):
    page, password, expire = expiring_admin("vence-playground@acme.test")
    console = ConsoleLog(page)
    page.goto(platform.url("/playground"))
    expire()
    page.fill("#prompt", "hola")
    page.fill("#max_tokens", "4")
    page.click("[data-playground-submit]")
    expect(page.locator("[data-playground-error]")).to_contain_text("La sesión venció", timeout=60_000)
    expect(page.locator("[data-playground-status]")).to_have_text("La generación no terminó.")
    expect(page.locator("[data-playground-output]")).to_have_attribute("aria-busy", "false")
    assert all("401" in e for e in console.errors), console.errors
    # Volver a entrar desde el mismo dispositivo deja todo andando.
    page.goto(platform.url("/playground"))
    assert "/login" in page.url and "next=%2Fplayground" in page.url, page.url
    page.fill("#email", "vence-playground@acme.test")
    page.fill("#password", password)
    page.click("form[action='/login'] button[type=submit]")
    page.wait_for_load_state()
    assert page.url.endswith("/playground"), page.url
