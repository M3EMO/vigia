"""Ronda 5: estados que AT-E04 no barría (errores de formulario, páginas de error, vista de cada rol)
y foco visible al recorrer con Tab cada página."""

from __future__ import annotations

import pytest
from _e2e import OWNER_PASSWORD, enqueue_for_review, login
from playwright.sync_api import Page

pytestmark = pytest.mark.e2e

WCAG_TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"]


def _serious(page: Page) -> list[str]:
    axe = pytest.importorskip("axe_playwright_python.sync_playwright").Axe()
    out = []
    for v in axe.run(page, options={"runOnly": WCAG_TAGS}).response["violations"]:
        if v.get("impact") in ("serious", "critical"):
            out.append(f"{v['id']} ({v['impact']}): {[', '.join(n['target']) for n in v['nodes'][:3]]}")
    return out


@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_axe_on_error_states_and_other_roles(platform, new_context, accounts, calibrated, scheme):
    found: dict[str, list[str]] = {}
    page = new_context(color_scheme=scheme).new_page()
    # Login con error.
    login(page, platform, platform.owner_email, "una-clave-equivocada-larga", expect_ok=False)
    found["/login (error)"] = _serious(page)
    login(page, platform, platform.owner_email, OWNER_PASSWORD)
    trace_id = platform.chat("respuesta para los estados de error")["choices"][0]["vigia"]["trace_id"]
    page.goto(platform.url(f"/generations/{trace_id}"))
    enqueue_for_review(page, platform, trace_id, min_reviewers=1)
    item_id = platform.sql("SELECT id FROM review_items WHERE trace_id = ?", (trace_id,))[0]["id"]
    # Configuración con errores en línea (key sin nombre, invitación con email inválido).
    page.goto(platform.url("/settings"))
    page.evaluate("document.querySelectorAll('[required]').forEach(e => e.removeAttribute('required'))")
    page.click("form[action='/settings/keys'] button[type=submit]")
    page.wait_for_load_state()
    found["/settings (key sin nombre)"] = _serious(page)
    page.goto(platform.url("/account"))
    page.evaluate(
        "document.querySelectorAll('[required],[minlength]').forEach(e => {"
        "e.removeAttribute('required'); e.removeAttribute('minlength')})"
    )
    page.fill("#current_password", "no-es-la-clave")
    page.fill("#new_password", "corta")
    page.fill("#confirm_password", "distinta")
    page.click("form[action='/account/password'] button[type=submit]")
    page.wait_for_load_state()
    found["/account (errores)"] = _serious(page)
    page.goto(platform.url("/calibration"))
    page.evaluate("document.querySelectorAll('[required]').forEach(e => e.removeAttribute('required'))")
    page.fill("#seed", "-5")
    page.click("form[action='/calibration/start'] button[type=submit]")
    page.wait_for_load_state()
    found["/calibration (semilla inválida)"] = _serious(page)
    page.goto(platform.url("/generations/no-existe"))
    found["404"] = _serious(page)

    for email in ("revisora1@acme.test", "lectora@acme.test"):
        other = new_context(color_scheme=scheme).new_page()
        login(other, platform, email, accounts[email])
        for path in ("/", "/review", f"/review/{item_id}", "/settings"):
            other.goto(platform.url(path))
            found[f"{email} {path}"] = _serious(other)
    found = {k: v for k, v in found.items() if v}
    assert not found, f"violaciones serias/críticas de axe ({scheme}): {found}"


FOCUS_JS = """() => {
  const el = document.activeElement;
  if (!el || el === document.body) return null;
  // Las partes internas de un control nativo (el botón del calendario de un input date) toman el foco
  // sin que el input lo tenga: su indicador es el del navegador y no se evalúa.
  if (!el.matches(":focus")) return {skip: true};
  const cs = getComputedStyle(el);
  const outline = cs.outlineStyle !== 'none' && parseFloat(cs.outlineWidth) > 0;
  const shadow = cs.boxShadow && cs.boxShadow !== 'none';
  const r = el.getBoundingClientRect();
  return {tag: el.tagName, id: el.id, text: (el.innerText || el.value || '').slice(0, 30),
          visible: outline || shadow, w: r.width, h: r.height};
}"""


@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_every_tab_stop_has_a_visible_focus_indicator(platform, new_context, scheme):
    page = new_context(color_scheme=scheme).new_page()
    login(page, platform, platform.owner_email, OWNER_PASSWORD)
    bad: dict[str, list] = {}
    for path in (
        "/",
        "/generations",
        "/playground",
        "/calibration",
        "/policies",
        "/settings",
        "/account",
        "/data",
    ):
        page.goto(platform.url(path))
        seen = 0
        for _ in range(80):
            page.keyboard.press("Tab")
            info = page.evaluate(FOCUS_JS)
            if info is None:
                break
            seen += 1
            if info.get("skip"):
                continue
            if not info["visible"] or info["w"] == 0 or info["h"] == 0:
                bad.setdefault(path, []).append(info)
        assert seen > 0, path
    assert not bad, f"paradas de Tab sin foco visible ({scheme}): {bad}"
