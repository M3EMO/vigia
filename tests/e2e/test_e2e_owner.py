"""AT-E01: el owner invita cuentas, crea una API key y usa el playground, sin errores de consola."""

from __future__ import annotations

import re

import pytest
from _e2e import OWNER_PASSWORD, ConsoleLog, create_key_in_panel, invite_user, login
from playwright.sync_api import expect

pytestmark = pytest.mark.e2e


def test_AT_E01_owner_invites_creates_key_and_uses_playground(platform, new_context, calibrated):
    page = new_context().new_page()
    console = ConsoleLog(page)
    login(page, platform, platform.owner_email, OWNER_PASSWORD)
    expect(page.locator("h1")).to_have_text("Resumen")

    reviewer_pw = invite_user(page, platform, "revisor-e01@acme.test", "reviewer")
    viewer_pw = invite_user(page, platform, "lector-e01@acme.test", "viewer")
    users = page.locator("#users table").inner_text()
    assert "revisor-e01@acme.test" in users and "lector-e01@acme.test" in users

    key = create_key_in_panel(page, platform, "backend e2e", ("ingest", "read"))
    expect(page.locator("#keys table")).to_contain_text("backend e2e")
    out = platform.chat("hola desde la key creada en el panel", key=key)
    vigia = out["choices"][0]["vigia"]
    assert vigia["calibration_id"] == calibrated["detector_id"]
    page.goto(platform.url("/generations"))
    expect(page.locator(f"a[href='/generations/{vigia['trace_id']}']")).to_be_visible()

    # Playground con streaming: tokens en vivo y, al final, fragmentos resaltados.
    page.goto(platform.url("/playground"))
    page.fill("#prompt", "¿Cuál es la capital de Francia?")
    page.fill("#max_tokens", "24")
    page.click("[data-playground-submit]")
    status = page.locator("[data-playground-status]")
    expect(status).to_contain_text(re.compile(r"^Listo: 24 tokens"), timeout=180_000)
    marks = page.locator("[data-playground-result] mark")
    assert marks.count() > 0, "el playground no muestra fragmentos con un detector calibrado activo"
    expect(page.locator("[data-playground-result] dl")).to_contain_text("sí")
    page.get_by_role("link", name="Abrir el detalle con señales").click()
    page.wait_for_load_state()
    assert re.search(r"/generations/[A-Za-z0-9_-]+$", page.url), page.url
    expect(page.locator("[data-marked-text] mark[data-fragment]").first).to_be_visible()
    expect(page.locator("[data-signals]").first).to_be_visible()

    # Las cuentas nuevas entran desde otro dispositivo con la contraseña temporal.
    for email, password, role in (
        ("revisor-e01@acme.test", reviewer_pw, "Revisor"),
        ("lector-e01@acme.test", viewer_pw, "Lectura"),
    ):
        other = new_context().new_page()
        other_console = ConsoleLog(other)
        login(other, platform, email, password)
        expect(other.locator(".vg-user")).to_contain_text(email)
        expect(other.locator(".vg-user")).to_contain_text(re.compile(role, re.IGNORECASE))
        assert other_console.errors == [], other_console.errors

    assert console.errors == [], console.errors
