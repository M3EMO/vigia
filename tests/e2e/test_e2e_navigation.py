"""Navegación de sesión en un navegador real: botón Atrás tras salir, destino `next` del login y
pestañas viejas en un equipo compartido donde entra otra persona."""

from __future__ import annotations

from urllib.parse import urlencode

import pytest
from _e2e import OWNER_PASSWORD, ConsoleLog, invite_user, login
from playwright.sync_api import expect

pytestmark = pytest.mark.e2e


@pytest.fixture
def admin_account(platform, new_context):
    """Crea (desde el panel del owner) una cuenta admin propia por test y devuelve su contraseña."""

    def make(email: str) -> str:
        owner = new_context().new_page()
        login(owner, platform, platform.owner_email, OWNER_PASSWORD)
        password = invite_user(owner, platform, email, "admin")
        owner.close()
        return password

    return make


def test_back_button_after_logout_does_not_show_private_pages(platform, new_context, admin_account):
    password = admin_account("atras@acme.test")
    page = new_context().new_page()
    login(page, platform, "atras@acme.test", password)
    page.goto(platform.url("/settings"))
    expect(page.locator("h1")).to_have_text("Configuración")
    page.get_by_role("button", name="Salir").click()
    page.wait_for_load_state()
    assert "/login" in page.url, page.url
    page.go_back()
    page.wait_for_load_state()
    # Con Cache-Control: no-store el navegador vuelve a pedir la página y el servidor manda al login.
    assert "/login" in page.url, page.url
    expect(page.locator("main")).not_to_contain_text("API keys")


def test_login_keeps_next_after_a_failed_attempt(platform, new_context, admin_account):
    password = admin_account("siguiente@acme.test")
    page = new_context().new_page()
    page.goto(platform.url("/model"))
    assert "next=%2Fmodel" in page.url, page.url
    page.fill("#email", "siguiente@acme.test")
    page.fill("#password", "una-clave-equivocada-larga")
    page.click("form[action='/login'] button[type=submit]")
    page.wait_for_load_state()
    expect(page.locator("[role=alert]")).to_be_visible()
    page.fill("#password", password)
    page.click("form[action='/login'] button[type=submit]")
    page.wait_for_load_state()
    assert page.url == platform.url("/model"), page.url


@pytest.mark.parametrize(
    "target",
    [
        "//evil.test/robo",
        "https://evil.test/robo",
        "/\\evil.test/robo",
        "/%2F%2Fevil.test",
        "javascript:alert(1)",
    ],
)
def test_login_never_redirects_off_site(platform, new_context, target):
    page = new_context().new_page()
    page.goto(platform.url("/login?") + urlencode({"next": target}))
    page.fill("#email", platform.owner_email)
    page.fill("#password", OWNER_PASSWORD)
    page.click("form[action='/login'] button[type=submit]")
    page.wait_for_load_state()
    assert page.url.startswith(platform.base_url + "/"), page.url


def test_stale_tab_cannot_act_as_the_next_person_on_a_shared_device(platform, new_context, admin_account):
    """Equipo compartido: A deja abierta Configuración, sale en otra pestaña y entra B. El formulario
    viejo de A no puede crear nada en nombre de B; la página de error ofrece una salida que funciona."""
    pw_a = admin_account("compartido-a@acme.test")
    pw_b = admin_account("compartido-b@acme.test")
    ctx = new_context()
    tab_a = ctx.new_page()
    console = ConsoleLog(tab_a)
    login(tab_a, platform, "compartido-a@acme.test", pw_a)
    tab_a.goto(platform.url("/settings"))
    tab_b = ctx.new_page()
    tab_b.goto(platform.url("/account"))
    tab_b.get_by_role("button", name="Salir").click()
    tab_b.wait_for_load_state()
    login(tab_b, platform, "compartido-b@acme.test", pw_b)

    tab_a.fill("#key-name", "clave-de-la-pestana-vieja")
    tab_a.click("form[action='/settings/keys'] button[type=submit]")
    tab_a.wait_for_load_state()
    made = platform.sql("SELECT id FROM api_keys WHERE name = ?", ("clave-de-la-pestana-vieja",))
    assert made == [], "el formulario viejo de A creó una key con la sesión de B"
    expect(tab_a.locator("h1")).to_have_text("Sin permiso")
    hrefs = tab_a.eval_on_selector_all("main a[href]", "as => as.map(a => a.getAttribute('href'))")
    assert hrefs, "la página de error no ofrece salida"
    for href in hrefs:
        resp = tab_a.goto(platform.url(href))
        assert resp is not None and resp.status == 200, (href, resp and resp.status)
    assert all("403" in e for e in console.errors), console.errors
