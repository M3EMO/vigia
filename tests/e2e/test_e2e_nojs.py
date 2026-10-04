"""AT-E07: con JavaScript desactivado, las páginas y los formularios principales siguen funcionando."""

from __future__ import annotations

import re

import pytest
from _e2e import API_KEY_RE, login
from playwright.sync_api import expect

pytestmark = pytest.mark.e2e

ADMIN = "admin@acme.test"


@pytest.fixture
def nojs_page(platform, new_context, accounts, calibrated):
    page = new_context(java_script_enabled=False).new_page()
    login(page, platform, ADMIN, accounts[ADMIN])
    assert page.evaluate("1 + 1") == 2  # evaluate de Playwright sigue andando; el JS de la página no
    expect(page.locator("html")).not_to_have_class(re.compile(r"\bjs\b"))
    return page


def test_AT_E07_generations_detail_and_threshold_without_js(platform, nojs_page):
    page = nojs_page
    trace_id = platform.chat("generación para ver sin javascript", max_tokens=24)["choices"][0]["vigia"][
        "trace_id"
    ]
    page.goto(platform.url("/generations"))
    expect(page.locator("h1")).to_have_text("Generaciones")
    page.select_option("#f-origen", index=1)
    page.click("form[action='/generations'] button[type=submit]")
    page.wait_for_load_state()
    assert "origen=" in page.url, page.url
    page.goto(platform.url("/generations"))
    page.click(f"a[href='/generations/{trace_id}']")
    page.wait_for_load_state()
    expect(page.locator("[data-marked-text]")).to_be_visible()
    expect(page.locator("[data-marked-text] mark[data-fragment]").first).to_be_visible()
    page.fill("#umbral", "1")
    page.click("form[data-threshold-form] button[type=submit]")
    page.wait_for_load_state()
    assert "umbral=1" in page.url, page.url
    expect(page.locator("[data-marked-text] mark")).to_have_count(0)
    # Las señales de todos los fragmentos quedan visibles sin JS (no hay pestañas que dependan de JS).
    page.goto(platform.url(f"/generations/{trace_id}?umbral=0"))
    signals = page.locator("[data-signals]")
    for i in range(signals.count()):
        expect(signals.nth(i)).to_be_visible()


def test_AT_E07_playground_by_post_without_js(platform, nojs_page):
    page = nojs_page
    page.goto(platform.url("/playground"))
    page.fill("#prompt", "¿Cuál es la capital de Francia?")
    page.fill("#max_tokens", "12")
    page.click("form[action='/playground'] button[type=submit]")
    page.wait_for_load_state()
    expect(page.locator("[data-playground-status]")).to_contain_text("Listo: 12 tokens")
    expect(page.locator("[data-playground-result] mark").first).to_be_visible()
    page.get_by_role("link", name="Abrir el detalle con señales").click()
    page.wait_for_load_state()
    assert re.search(r"/generations/[A-Za-z0-9_-]+$", page.url), page.url


def test_AT_E07_review_forms_without_js(platform, nojs_page):
    page = nojs_page
    trace_id = platform.chat("generación para revisar sin javascript")["choices"][0]["vigia"]["trace_id"]
    page.goto(platform.url(f"/generations/{trace_id}"))
    page.fill("#min_reviewers", "1")
    page.click(f"form[action='/generations/{trace_id}/review'] button[type=submit]")
    page.wait_for_load_state()
    assert "ok=review_created" in page.url, page.url
    item_id = platform.sql("SELECT id FROM review_items WHERE trace_id = ?", (trace_id,))[0]["id"]
    page.goto(platform.url(f"/review/{item_id}"))
    page.click(f"form[action='/review/{item_id}/reserve'] button[type=submit]")
    page.wait_for_load_state()
    assert "ok=review_reserved" in page.url, page.url
    page.check("[data-review-form] input[name=label][value=error]")
    page.fill("#note", "revisado sin javascript")
    page.click("[data-review-form] button[name=next][value='0']")
    page.wait_for_load_state()
    assert "ok=review_saved" in page.url, page.url
    row = platform.sql("SELECT label, note FROM review_decisions WHERE item_id = ?", (item_id,))
    assert [(r["label"], r["note"]) for r in row] == [("error", "revisado sin javascript")]


def test_AT_E07_admin_forms_without_js(platform, nojs_page):
    page = nojs_page
    page.goto(platform.url("/policies"))
    page.click("form[action='/policies'] button[type=submit]")
    page.wait_for_load_state()
    assert "/policies" in page.url and "ok=" in page.url, (
        page.url,
        page.locator(".vg-alert").all_inner_texts(),
    )

    page.goto(platform.url("/settings"))
    page.fill("#key-name", "key sin javascript")
    page.click("form[action='/settings/keys'] button[type=submit]")
    page.wait_for_load_state()
    assert API_KEY_RE.search(page.content()), "no se mostró el token de la key creada sin JS"

    dataset = platform.data_dir / "calibracion-e2e.jsonl"
    page.goto(platform.url("/calibration"))
    page.set_input_files("#file", str(dataset))
    page.fill("form[action='/calibration/datasets'] #ds-name", "subido sin javascript")
    page.click("form[action='/calibration/datasets'] button[type=submit]")
    page.wait_for_load_state()
    assert "ok=dataset_uploaded" in page.url, page.locator(".vg-alert").all_inner_texts()
    expect(page.locator("main")).to_contain_text("subido sin javascript")

    page.goto(platform.url("/account"))
    page.click("form[action='/logout'] button[type=submit]")
    page.wait_for_load_state()
    assert "/login" in page.url, page.url


def test_AT_E07_mobile_nav_reachable_without_js(platform, new_context, accounts):
    page = new_context(java_script_enabled=False, viewport={"width": 375, "height": 812}).new_page()
    login(page, platform, ADMIN, accounts[ADMIN])
    nav = page.locator("#vg-nav")
    expect(nav).to_be_visible()
    nav.get_by_role("link", name="Cola de revisión").click()
    page.wait_for_load_state()
    assert page.url.endswith("/review"), page.url
