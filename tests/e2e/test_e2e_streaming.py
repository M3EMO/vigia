"""Varios usuarios y un modelo compartido: cerrar la pestaña en medio del streaming del playground no
puede dejar tomado el backend (el lock del modelo es uno solo para todos los tenants).

Este módulo corre al final de la sesión (ver conftest): si el backend queda tomado, los tests que
generan después se colgarían.
"""

from __future__ import annotations

import time

import httpx
import pytest
from _e2e import OWNER_B_PASSWORD, login
from playwright.sync_api import expect

pytestmark = pytest.mark.e2e

MAX_TOKENS = 1000


def test_closing_tab_mid_stream_releases_the_model_for_other_users(platform, new_context):
    t0 = time.monotonic()
    platform.chat("medir una generación completa", max_tokens=MAX_TOKENS, vigia=False)
    full_s = time.monotonic() - t0

    ctx = new_context()
    page = ctx.new_page()
    login(page, platform, platform.owner_b_email, OWNER_B_PASSWORD)
    page.goto(platform.url("/playground"))
    page.fill("#prompt", "una respuesta larga")
    page.fill("#max_tokens", str(MAX_TOKENS))
    page.click("[data-playground-submit]")
    expect(page.locator("[data-playground-status]")).to_contain_text("Generando:", timeout=60_000)
    ctx.close()  # la usuaria cierra la pestaña a mitad de la generación

    # Aun si la generación abandonada siguiera hasta el final, a esta altura ya habría terminado.
    time.sleep(max(5.0, 2 * full_s))
    t1 = time.monotonic()
    try:
        out = platform.chat("otra organización pide 2 tokens", max_tokens=2, timeout=max(60.0, 3 * full_s))
    except httpx.ReadTimeout:
        pytest.fail(
            f"tras cerrar la pestaña, un request de 2 tokens de otro tenant no terminó en "
            f"{time.monotonic() - t1:.0f} s (una generación completa de {MAX_TOKENS} tokens "
            f"tarda {full_s:.1f} s)"
        )
    assert out["choices"][0]["message"]["content"] is not None
