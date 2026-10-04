"""Sesiones en navegador real: logout, expiración por inactividad y tope absoluto, varios dispositivos
y aislamiento entre organizaciones (AT-P04 y AT-P01 de punta a punta)."""

from __future__ import annotations

import pytest
from _e2e import (
    OWNER_B_PASSWORD,
    OWNER_PASSWORD,
    ConsoleLog,
    browser_fetch,
    invite_user,
    iso_ago,
    login,
)
from playwright.sync_api import expect

pytestmark = pytest.mark.e2e

COOKIE = "__Host-vigia_session"
NEW_PASSWORD = "Nueva-Clave-Bastante-Larga-2026"  # noqa: S105 - credencial de prueba local


@pytest.fixture
def fresh_user(platform, new_context):
    """Una cuenta propia por test, así cerrar sus sesiones no afecta a otros tests."""

    def make(email: str, role: str = "viewer") -> str:
        page = new_context().new_page()
        login(page, platform, platform.owner_email, OWNER_PASSWORD)
        password = invite_user(page, platform, email, role)
        page.close()
        return password

    return make


def session_cookie(ctx) -> dict | None:
    return next((c for c in ctx.cookies() if c["name"] == COOKIE), None)


def test_login_sets_hardened_cookie_and_logout_invalidates_server_side(platform, new_context, fresh_user):
    password = fresh_user("salida@acme.test")
    ctx = new_context()
    page = ctx.new_page()
    console = ConsoleLog(page)
    login(page, platform, "salida@acme.test", password)
    cookie = session_cookie(ctx)
    assert cookie and cookie["secure"] and cookie["httpOnly"] and cookie["sameSite"] == "Strict", cookie
    assert cookie["path"] == "/" and not cookie["domain"].startswith("."), cookie

    page.get_by_role("button", name="Salir").click()
    page.wait_for_load_state()
    assert "/login" in page.url and "ok=logged_out" in page.url, page.url
    assert session_cookie(ctx) is None
    page.goto(platform.url("/generations"))
    assert "/login" in page.url, page.url

    # Reusar la cookie vieja desde otro navegador no sirve: el servidor borró la sesión.
    thief = new_context()
    thief.add_cookies(
        [{k: cookie[k] for k in ("name", "value", "domain", "path", "secure", "httpOnly", "sameSite")}]
    )
    other = thief.new_page()
    other.goto(platform.url("/generations"))
    assert "/login" in other.url, other.url
    assert console.errors == [], console.errors


@pytest.mark.parametrize(
    ("column", "value"),
    [("last_seen_at", iso_ago(31)), ("expires_at", iso_ago(1))],
    ids=["inactividad-30-min", "tope-absoluto"],
)
def test_expired_session_redirects_to_login_and_returns_to_page(
    platform, new_context, fresh_user, column, value
):
    email = f"expira-{column.split('_')[0]}@acme.test"
    password = fresh_user(email)
    ctx = new_context()
    page = ctx.new_page()
    login(page, platform, email, password)
    page.goto(platform.url("/generations"))
    expect(page.locator("h1")).to_have_text("Generaciones")
    user_id = platform.sql("SELECT id FROM users WHERE email = ?", (email,))[0]["id"]
    platform.sql(f"UPDATE sessions SET {column} = ? WHERE user_id = ?", (value, user_id))  # noqa: S608
    page.goto(platform.url("/model"))
    assert "/login" in page.url and "next=%2Fmodel" in page.url, page.url
    assert platform.sql("SELECT COUNT(*) AS n FROM sessions WHERE user_id = ?", (user_id,))[0]["n"] == 0
    page.fill("#email", email)
    page.fill("#password", password)
    page.click("form[action='/login'] button[type=submit]")
    page.wait_for_load_state()
    assert page.url.endswith("/model"), page.url


def test_two_devices_and_password_change_closes_the_other(platform, new_context, fresh_user):
    password = fresh_user("dos-equipos@acme.test")
    desktop = new_context(viewport={"width": 1440, "height": 900}).new_page()
    phone = new_context(viewport={"width": 375, "height": 812}, is_mobile=True, has_touch=True).new_page()
    login(desktop, platform, "dos-equipos@acme.test", password)
    login(phone, platform, "dos-equipos@acme.test", password)
    for page in (desktop, phone):
        page.goto(platform.url("/generations"))
        expect(page.locator("h1")).to_have_text("Generaciones")

    phone.goto(platform.url("/account"))
    phone.fill("#current_password", password)
    phone.fill("#new_password", NEW_PASSWORD)
    phone.fill("#confirm_password", NEW_PASSWORD)
    phone.click("form[action='/account/password'] button[type=submit]")
    phone.wait_for_load_state()
    assert "/login" in phone.url and "ok=password_changed" in phone.url, phone.url

    desktop.reload()
    assert "/login" in desktop.url, "la sesión del otro dispositivo sigue viva tras cambiar la contraseña"
    login(desktop, platform, "dos-equipos@acme.test", password, expect_ok=False)
    assert "/login" in desktop.url  # la contraseña vieja ya no entra
    login(desktop, platform, "dos-equipos@acme.test", NEW_PASSWORD)


def test_other_tenant_sees_404_not_403(platform, new_context, calibrated):
    trace_id = platform.chat("dato privado de la org A")["choices"][0]["vigia"]["trace_id"]
    page = new_context().new_page()
    login(page, platform, platform.owner_b_email, OWNER_B_PASSWORD)
    page.goto(platform.url("/generations"))
    expect(page.locator(f"a[href='/generations/{trace_id}']")).to_have_count(0)
    for path in (
        f"/generations/{trace_id}",
        f"/calibration/detectors/{calibrated['detector_id']}",
        f"/calibration/jobs/{calibrated['job_id']}",
    ):
        resp = page.goto(platform.url(path))
        assert resp is not None and resp.status == 404, (path, resp and resp.status)
    status = browser_fetch(
        page, platform.url(f"/calibration/jobs/{calibrated['job_id']}/status"), method="GET"
    )
    assert status["status"] == 404, status
