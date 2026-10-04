"""Varios usuarios y varios dispositivos sobre la misma organización.

Cada contexto de navegador es un dispositivo con su propia cookie. Lo que una persona hace en un
dispositivo tiene que verse en la sesión abierta de otra persona (o de la misma en otro equipo).
"""

from __future__ import annotations

import httpx
import pytest
from _e2e import (
    OWNER_B_PASSWORD,
    OWNER_PASSWORD,
    ConsoleLog,
    create_key_in_panel,
    create_org,
    enqueue_for_review,
    invite_user,
    login,
    post_form,
)
from playwright.sync_api import expect

pytestmark = pytest.mark.e2e

MOBILE = {"viewport": {"width": 375, "height": 812}, "is_mobile": True, "has_touch": True}


@pytest.fixture
def owner_page(platform, new_context):
    page = new_context().new_page()
    login(page, platform, platform.owner_email, OWNER_PASSWORD)
    return page


def user_id(platform, email: str) -> str:
    return platform.sql("SELECT id FROM users WHERE email = ?", (email,))[0]["id"]


def api_status(platform, key: str) -> int:
    resp = httpx.post(
        platform.url("/v1/chat/completions"),
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": platform.env["VIGIA_MODEL"],
            "messages": [{"role": "user", "content": "x"}],
            "max_tokens": 2,
        },
        timeout=120,
    )
    return resp.status_code


def test_role_change_closes_the_open_session_and_applies_the_new_role(platform, new_context, owner_page):
    password = invite_user(owner_page, platform, "degradada@acme.test", "admin")
    other = new_context(**MOBILE).new_page()
    login(other, platform, "degradada@acme.test", password)
    other.goto(platform.url("/settings"))
    expect(other.locator("h1")).to_have_text("Configuración")

    uid = user_id(platform, "degradada@acme.test")
    owner_page.goto(platform.url("/settings"))
    owner_page.select_option(f"#role-{uid}", "viewer")
    owner_page.click(f"form[action='/settings/users/{uid}/role'] button[type=submit]")
    owner_page.wait_for_load_state()
    assert "ok=role_changed" in owner_page.url, owner_page.url

    # La sesión abierta del otro dispositivo se cerró; al volver a entrar rige el rol nuevo.
    other.goto(platform.url("/settings"))
    assert "/login" in other.url, other.url
    login(other, platform, "degradada@acme.test", password)
    expect(other.locator(".vg-user")).to_contain_text("Lectura")
    assert "Configuración" not in other.locator("#vg-nav").inner_text()
    resp = other.goto(platform.url("/settings"))
    assert resp is not None and resp.status == 403


def test_disabling_a_user_closes_its_session_and_blocks_login(platform, new_context, owner_page):
    password = invite_user(owner_page, platform, "deshabilitada@acme.test", "reviewer")
    other = new_context().new_page()
    login(other, platform, "deshabilitada@acme.test", password)
    other.goto(platform.url("/review"))
    expect(other.locator("h1")).to_be_visible()

    uid = user_id(platform, "deshabilitada@acme.test")
    toggle = f"form[action='/settings/users/{uid}/disable'] button[type=submit]"
    owner_page.goto(platform.url("/settings"))
    owner_page.click(toggle)
    owner_page.wait_for_load_state()
    assert "ok=user_disabled" in owner_page.url, owner_page.url

    other.reload()
    assert "/login" in other.url, other.url
    login(other, platform, "deshabilitada@acme.test", password, expect_ok=False)
    assert "/login" in other.url, "una cuenta deshabilitada volvió a entrar"

    owner_page.goto(platform.url("/settings"))
    owner_page.click(toggle)
    owner_page.wait_for_load_state()
    assert "ok=user_enabled" in owner_page.url, owner_page.url
    login(other, platform, "deshabilitada@acme.test", password)


def test_key_created_and_revoked_in_the_panel(platform, owner_page):
    console = ConsoleLog(owner_page)
    key = create_key_in_panel(owner_page, platform, "key para revocar", ("ingest",))
    assert api_status(platform, key) == 200
    key_id = platform.sql("SELECT id FROM api_keys WHERE name = ?", ("key para revocar",))[0]["id"]
    owner_page.goto(platform.url("/settings"))
    owner_page.click(f"form[action='/settings/keys/{key_id}/revoke'] button[type=submit]")
    owner_page.wait_for_load_state()
    assert "ok=key_revoked" in owner_page.url, owner_page.url
    assert api_status(platform, key) == 401
    assert console.errors == [], console.errors


def test_two_reviewers_on_two_devices_respect_the_reservation(platform, new_context, accounts, owner_page):
    trace_id = platform.chat("item que reservan dos revisores")["choices"][0]["vigia"]["trace_id"]
    owner_page.goto(platform.url(f"/generations/{trace_id}"))
    enqueue_for_review(owner_page, platform, trace_id, min_reviewers=2)
    item_id = platform.sql(
        "SELECT id FROM review_items WHERE trace_id = ? AND min_reviewers = 2 ORDER BY created_at DESC",
        (trace_id,),
    )[0]["id"]

    first = new_context().new_page()
    second = new_context(**MOBILE).new_page()
    login(first, platform, "revisora1@acme.test", accounts["revisora1@acme.test"])
    login(second, platform, "revisor2@acme.test", accounts["revisor2@acme.test"])

    first.goto(platform.url(f"/review/{item_id}"))
    first.click(f"form[action='/review/{item_id}/reserve'] button[type=submit]")
    first.wait_for_load_state()
    assert "ok=review_reserved" in first.url, first.url

    # El otro revisor ve la reserva, no tiene formulario y un POST directo da 409 sin guardar nada.
    second.goto(platform.url(f"/review/{item_id}"))
    expect(second.locator("main")).to_contain_text("Otro revisor tiene reservado este item")
    expect(second.locator("[data-review-form]")).to_have_count(0)
    assert post_form(second, f"/review/{item_id}/decide", {"label": "error"}, platform)["status"] == 409

    first.check("[data-review-form] input[name=label][value=correct]")
    first.click("[data-review-form] button[name=next][value='0']")
    first.wait_for_load_state()
    assert "ok=review_saved" in first.url, first.url

    # Decidir libera la reserva: el segundo revisor decide sin esperar los 15 minutos.
    second.goto(platform.url(f"/review/{item_id}"))
    second.check("[data-review-form] input[name=label][value=error]")
    second.click("[data-review-form] button[name=next][value='0']")
    second.wait_for_load_state()
    assert "ok=review_saved" in second.url, second.url
    expect(second.locator("main")).to_contain_text("revisora1@acme.test: Correcto")
    expect(second.locator("main")).to_contain_text("revisor2@acme.test: Error")


def test_deleting_the_org_closes_every_device_and_its_keys(platform, new_context):
    password = "Clave-Larga-Org-Borrable-2026"  # noqa: S105 - credencial de prueba local
    org_id, key = create_org(platform, "Org Borrable", "duena@borrable.test", password)
    platform.chat("dato de la org borrable", key=key, max_tokens=3)

    owner = new_context().new_page()
    login(owner, platform, "duena@borrable.test", password)
    viewer_pw = invite_user(owner, platform, "lectora@borrable.test", "viewer")
    phone = new_context(**MOBILE).new_page()
    login(phone, platform, "lectora@borrable.test", viewer_pw)
    phone.goto(platform.url("/generations"))
    expect(phone.locator("h1")).to_have_text("Generaciones")

    owner.goto(platform.url("/data"))
    owner.fill("#confirm_name", "Org Borrable")
    owner.check("form[action='/data/delete'] input[name=confirm]")
    owner.click("form[action='/data/delete'] button[type=submit]")
    owner.wait_for_load_state()
    assert "/login" in owner.url and "ok=org_deleted" in owner.url, owner.url

    phone.reload()
    assert "/login" in phone.url, phone.url
    login(phone, platform, "lectora@borrable.test", viewer_pw, expect_ok=False)
    assert "/login" in phone.url, "una cuenta de la org borrada volvió a entrar"
    assert api_status(platform, key) == 401
    left = platform.sql(
        "SELECT (SELECT COUNT(*) FROM users WHERE org_id = ?) AS u, (SELECT COUNT(*) FROM traces "
        "WHERE org_id = ?) AS t, (SELECT COUNT(*) FROM sessions WHERE org_id = ?) AS s",
        (org_id, org_id, org_id),
    )[0]
    assert (left["u"], left["t"], left["s"]) == (0, 0, 0), dict(left)


def test_concurrent_playground_streams_from_three_devices_all_finish(platform, new_context, calibrated):
    """El lock del modelo es uno solo: tres streams a la vez (dos orgs) terminan todos, en orden."""
    pages = []
    for email, password in (
        (platform.owner_email, OWNER_PASSWORD),
        (platform.owner_b_email, OWNER_B_PASSWORD),
        (platform.owner_email, OWNER_PASSWORD),
    ):
        page = new_context().new_page()
        console = ConsoleLog(page)
        login(page, platform, email, password)
        page.goto(platform.url("/playground"))
        page.fill("#prompt", f"generación concurrente de {email}")
        page.fill("#max_tokens", "120")
        pages.append((page, console))
    for page, _ in pages:
        page.click("[data-playground-submit]")
    for page, console in pages:
        expect(page.locator("[data-playground-status]")).to_contain_text("Listo: 120 tokens", timeout=300_000)
        expect(page.get_by_role("link", name="Abrir el detalle con señales")).to_be_visible()
        assert console.errors == [], console.errors
    # Cada org ve su propia generación del playground y no la de la otra.
    calibrated_flags = [p.locator("[data-playground-result] dl").inner_text() for p, _ in pages]
    assert "sí" in calibrated_flags[0] and "sí" in calibrated_flags[2], calibrated_flags
    assert "no" in calibrated_flags[1], calibrated_flags
