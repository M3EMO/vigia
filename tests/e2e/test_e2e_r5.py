"""Ronda 5: flujos multiusuario y multidispositivo que la suite previa no cubría.

- Cada rol en un teléfono (375 px) con un email largo: sin scroll horizontal en sus páginas.
- Los secretos que muestra Configuración (API key, contraseña temporal) en un teléfono.
- Páginas de error (404, 403) en un teléfono.
- Límites de rol entre admin y owner por POST directo.
- Una persona invitada entra con la contraseña temporal y la cambia desde «Mi cuenta».
"""

from __future__ import annotations

import pytest
from _e2e import (
    API_KEY_RE,
    TEMP_PASSWORD_RE,
    ConsoleLog,
    create_org,
    csrf_token,
    enqueue_for_review,
    h_overflow,
    invite_user,
    login,
    post_form,
)
from playwright.sync_api import expect

pytestmark = pytest.mark.e2e

PASSWORD = "Clave-Larga-De-La-Ronda-Cinco-2026"  # noqa: S105 - credencial de prueba local
NEW_PASSWORD = "Otra-Clave-Nueva-Ronda-Cinco-2026"  # noqa: S105 - credencial de prueba local
MOBILE = {"viewport": {"width": 375, "height": 812}, "is_mobile": True, "has_touch": True}
LONG_LOCAL = "persona.con.un.nombre.de.cuenta.larguisimo"


def _org(platform, slug: str) -> tuple[str, str, str]:
    email = f"duena@{slug}.test"
    org_id, key = create_org(platform, f"Org {slug}", email, PASSWORD)
    return org_id, email, key


def _wide(page) -> int:
    for details in page.locator("main details").all():
        details.evaluate("d => d.open = true")
    return h_overflow(page)


@pytest.mark.parametrize("role", ["admin", "reviewer", "viewer"])
def test_each_role_with_a_long_email_on_a_phone(platform, new_context, role):
    slug = f"r5{role}"
    _org_id, owner_email, key = _org(platform, slug)
    owner = new_context().new_page()
    login(owner, platform, owner_email, PASSWORD)
    email = f"{LONG_LOCAL}.{role}@{slug}.test"
    temp = invite_user(owner, platform, email, role)
    trace_id = platform.chat("respuesta para cada rol en el teléfono", key=key)["choices"][0]["vigia"][
        "trace_id"
    ]
    owner.goto(platform.url(f"/generations/{trace_id}"))
    enqueue_for_review(owner, platform, trace_id, min_reviewers=1)
    item_id = platform.sql("SELECT id FROM review_items WHERE trace_id = ?", (trace_id,))[0]["id"]

    page = new_context(**MOBILE).new_page()
    console = ConsoleLog(page)
    login(page, platform, email, temp)
    links = page.eval_on_selector_all("#vg-nav a[href]", "as => as.map(a => a.getAttribute('href'))")
    paths = sorted({*links, f"/generations/{trace_id}", f"/review/{item_id}", "/account"})
    wide = {}
    for path in paths:
        resp = page.goto(platform.url(path))
        assert resp is not None and resp.status == 200, (path, resp and resp.status)
        px = _wide(page)
        if px > 0:
            wide[path] = px
    assert not wide, f"{role} con email largo: scroll horizontal a 375 px {wide}"
    assert console.errors == [], console.errors


def test_secrets_shown_in_settings_fit_on_a_phone(platform, new_context):
    _org_id, owner_email, _key = _org(platform, "r5secretos")
    page = new_context(**MOBILE).new_page()
    console = ConsoleLog(page)
    login(page, platform, owner_email, PASSWORD)

    page.goto(platform.url("/settings"))
    page.fill("#key-name", "key creada desde el teléfono")
    page.click("form[action='/settings/keys'] button[type=submit]")
    page.wait_for_load_state()
    assert API_KEY_RE.search(page.content())
    key_px = _wide(page)

    page.goto(platform.url("/settings"))
    page.fill("#invite-email", f"{LONG_LOCAL}@r5secretos.test")
    page.select_option("#invite-role", "viewer")
    page.click("form[action='/settings/users'] button[type=submit]")
    page.wait_for_load_state()
    assert TEMP_PASSWORD_RE.search(page.content())
    invite_px = _wide(page)
    assert (key_px, invite_px) == (0, 0), (
        f"scroll horizontal con el secreto a la vista: key={key_px} invite={invite_px}"
    )
    assert console.errors == [], console.errors


def test_error_pages_fit_on_a_phone(platform, new_context, accounts):
    page = new_context(**MOBILE).new_page()
    login(page, platform, "lectora@acme.test", accounts["lectora@acme.test"])
    wide = {}
    for path, status in (
        ("/generations/no-existe-esta-generacion", 404),
        ("/settings", 403),
        ("/audit", 403),
    ):
        resp = page.goto(platform.url(path))
        assert resp is not None and resp.status == status, (path, resp and resp.status)
        px = h_overflow(page)
        if px > 0:
            wide[path] = px
        expect(page.locator("main")).to_be_visible()
    assert not wide, wide


def test_admin_cannot_touch_the_owner_by_direct_post(platform, new_context, accounts):
    owner_id = platform.sql("SELECT id FROM users WHERE email = ?", (platform.owner_email,))[0]["id"]
    admin_id = platform.sql("SELECT id FROM users WHERE email = ?", ("admin@acme.test",))[0]["id"]
    page = new_context().new_page()
    login(page, platform, "admin@acme.test", accounts["admin@acme.test"])
    page.goto(platform.url("/settings"))
    for path, data in (
        (f"/settings/users/{owner_id}/role", {"role": "viewer"}),
        (f"/settings/users/{owner_id}/disable", {"disabled": "1"}),
        (f"/settings/users/{admin_id}/role", {"role": "owner"}),
    ):
        resp = post_form(page, path, data, platform)
        assert resp["status"] == 403, (path, resp)
    rows = platform.sql(
        "SELECT role, disabled FROM users WHERE id IN (?, ?) ORDER BY role", (owner_id, admin_id)
    )
    assert [(r["role"], r["disabled"]) for r in rows] == [("admin", 0), ("owner", 0)]
    # Lo mismo con un formulario sin token CSRF: 403 y nada cambia.
    resp = page.evaluate(
        "async (u) => (await fetch(u, {method: 'POST',"
        " body: new URLSearchParams({role: 'viewer'})})).status",
        platform.url(f"/settings/users/{owner_id}/role"),
    )
    assert resp == 403, resp


def test_invited_person_signs_in_and_changes_the_temporary_password(platform, new_context):
    _org_id, owner_email, _key = _org(platform, "r5invitada")
    owner = new_context().new_page()
    login(owner, platform, owner_email, PASSWORD)
    email = "invitada@r5invitada.test"
    temp = invite_user(owner, platform, email, "reviewer")

    phone = new_context(**MOBILE).new_page()
    console = ConsoleLog(phone)
    login(phone, platform, email, temp)
    phone.goto(platform.url("/account"))
    phone.fill("#current_password", temp)
    phone.fill("#new_password", NEW_PASSWORD)
    confirm = phone.locator("#confirm_password, #new_password2, input[name=confirm_password]")
    if confirm.count():
        confirm.first.fill(NEW_PASSWORD)
    phone.click("form[action='/account/password'] button[type=submit]")
    phone.wait_for_load_state()
    assert "/login" in phone.url or "ok=" in phone.url, (
        phone.url,
        phone.locator(".vg-alert").all_inner_texts(),
    )
    assert console.errors == [], console.errors

    laptop = new_context().new_page()
    login(laptop, platform, email, temp, expect_ok=False)
    assert "/login" in laptop.url, "la contraseña temporal sigue sirviendo después del cambio"
    login(laptop, platform, email, NEW_PASSWORD)
    expect(laptop.locator(".vg-user")).to_contain_text(email)
    assert csrf_token(laptop)


def test_address_bar_urls_left_by_a_post_and_unknown_urls_render_the_panel_error_page(platform, new_context):
    """Tras crear una key o invitar, la barra muestra /settings/keys o /settings/users. Volver a abrir
    esa URL (Enter en la barra, un marcador) o tipear mal una ruta del panel debe dar la página de
    error del panel en español, no un JSON en inglés."""
    _org_id, owner_email, _key = _org(platform, "r5barra")
    page = new_context().new_page()
    login(page, platform, owner_email, PASSWORD)
    page.goto(platform.url("/settings"))
    page.fill("#key-name", "key para la barra")
    page.click("form[action='/settings/keys'] button[type=submit]")
    page.wait_for_load_state()
    left_by_post = page.url.replace(platform.base_url, "")
    assert left_by_post == "/settings/keys", left_by_post
    raw = {}
    for path in (
        left_by_post,
        "/settings/users",
        "/account/password",
        "/calibration/datasets",
        "/generaciones",
    ):
        resp = page.goto(platform.url(path))
        ctype = (resp.headers.get("content-type", "") if resp else "") or ""
        if not ctype.startswith("text/html") or page.locator("nav#vg-nav").count() == 0:
            raw[path] = (resp.status if resp else 0, ctype, page.locator("body").inner_text()[:80])
    assert not raw, f"URLs del panel que el navegador muestra como JSON crudo: {raw}"


def test_too_large_calibration_upload_shows_the_panel_page_not_raw_json(platform, new_context):
    """Un JSONL más grande que VIGIA_MAX_UPLOAD_BYTES (20 MB por defecto) debe volver a Calibración con
    el aviso «El archivo supera el máximo…», no con el JSON del middleware de tamaño."""
    _org_id, owner_email, _key = _org(platform, "r5grande")
    big = platform.data_dir / "dataset-demasiado-grande.jsonl"
    line = b'{"question": "pregunta de relleno", "answers": ["respuesta"]}\n'
    big.write_bytes(line * (21 * 1024 * 1024 // len(line) + 1))
    try:
        page = new_context().new_page()
        login(page, platform, owner_email, PASSWORD)
        page.goto(platform.url("/calibration"))
        page.set_input_files("#file", str(big))
        with page.expect_navigation() as nav:
            page.click("form[action='/calibration/datasets'] button[type=submit]")
        resp = nav.value
        ctype = resp.headers.get("content-type", "")
        assert resp.status == 413, resp.status
        assert ctype.startswith("text/html"), (ctype, page.locator("body").inner_text()[:200])
        expect(page.locator("main")).to_contain_text("supera el máximo")
    finally:
        big.unlink(missing_ok=True)
