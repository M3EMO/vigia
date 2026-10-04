"""AT-E05: sin scroll horizontal en cada página a 375, 768 y 1440 px; el detalle se apila en móvil."""

from __future__ import annotations

import pytest
from _e2e import OWNER_PASSWORD, ConsoleLog, enqueue_for_review, h_overflow, login
from playwright.sync_api import expect

pytestmark = pytest.mark.e2e

VIEWPORTS = {"movil-375": (375, 812), "tablet-768": (768, 1024), "escritorio-1440": (1440, 900)}


@pytest.fixture
def pages_to_check(platform, new_context, calibrated):
    """Rutas de todas las páginas del panel, con ids reales (una generación con fragmentos y un item)."""
    if "responsive_paths" not in platform.cache:
        page = new_context().new_page()
        login(page, platform, platform.owner_email, OWNER_PASSWORD)
        out = platform.chat("una respuesta larga para revisar el diseño en pantallas chicas", max_tokens=48)
        trace_id = out["choices"][0]["vigia"]["trace_id"]
        page.goto(platform.url(f"/generations/{trace_id}"))
        enqueue_for_review(page, platform, trace_id, min_reviewers=1)
        item_id = platform.sql("SELECT id FROM review_items WHERE trace_id = ?", (trace_id,))[0]["id"]
        platform.cache["responsive_paths"] = [
            "/",
            "/generations",
            f"/generations/{trace_id}",
            "/review",
            f"/review/{item_id}",
            "/playground",
            "/model",
            "/performance",
            "/calibration",
            f"/calibration/jobs/{calibrated['job_id']}",
            f"/calibration/detectors/{calibrated['detector_id']}",
            "/policies",
            "/settings",
            "/audit",
            "/data",
            "/account",
        ]
        platform.cache["responsive_detail"] = f"/generations/{trace_id}"
    return platform.cache["responsive_paths"]


@pytest.mark.parametrize("viewport", list(VIEWPORTS), ids=list(VIEWPORTS))
def test_AT_E05_no_horizontal_scroll(platform, new_context, pages_to_check, viewport):
    width, height = VIEWPORTS[viewport]
    mobile = width < 768
    ctx = new_context(viewport={"width": width, "height": height}, is_mobile=mobile, has_touch=mobile)
    page = ctx.new_page()
    console = ConsoleLog(page)
    page.goto(platform.url("/login"))
    overflow = {"/login": h_overflow(page)}
    login(page, platform, platform.owner_email, OWNER_PASSWORD)
    for path in pages_to_check:
        resp = page.goto(platform.url(path))
        assert resp is not None and resp.status == 200, (path, resp and resp.status)
        overflow[path] = h_overflow(page)
    wide = {path: px for path, px in overflow.items() if px > 0}
    assert not wide, f"scroll horizontal a {width}px (px de más por página): {wide}"
    assert console.errors == [], console.errors


def test_AT_E05_detail_stacks_and_menu_toggles_on_mobile(platform, new_context, pages_to_check):
    ctx = new_context(viewport={"width": 375, "height": 812}, is_mobile=True, has_touch=True)
    page = ctx.new_page()
    login(page, platform, platform.owner_email, OWNER_PASSWORD)
    page.goto(platform.url(platform.cache["responsive_detail"]))
    main_box = page.locator(".vg-detail > div").first.bounding_box()
    aside_box = page.locator(".vg-detail > .vg-detail-aside").bounding_box()
    assert main_box and aside_box
    assert aside_box["y"] >= main_box["y"] + main_box["height"] - 1, (main_box, aside_box)
    assert aside_box["width"] >= main_box["width"] - 1, (main_box, aside_box)
    nav = page.locator("#vg-nav")
    toggle = page.locator("[data-nav-toggle]")
    expect(nav).to_be_hidden()
    toggle.click()
    expect(nav).to_be_visible()
    expect(toggle).to_have_attribute("aria-expanded", "true")
    nav.get_by_role("link", name="Generaciones").click()
    page.wait_for_load_state()
    assert page.url.endswith("/generations"), page.url
