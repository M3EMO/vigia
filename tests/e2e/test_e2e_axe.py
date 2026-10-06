"""AT-E04: axe-core sobre cada página del panel, tema claro y oscuro, escritorio y 375 px.

El oráculo es axe-core: cero violaciones de impacto `serious` o `critical` con las reglas de
WCAG 2.0/2.1/2.2 A y AA. Las páginas son las mismas que barre AT-E05 (ids reales)."""

from __future__ import annotations

import pytest
from _e2e import OWNER_PASSWORD, login
from test_e2e_responsive import pages_to_check  # noqa: F401 - fixture compartido

pytestmark = pytest.mark.e2e

WCAG_TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"]
CASES = {
    "claro-1280": ("light", 1280, 900),
    "oscuro-1280": ("dark", 1280, 900),
    "claro-375": ("light", 375, 812),
    "oscuro-375": ("dark", 375, 812),
}


def _serious(results: dict) -> list[str]:
    out = []
    for v in results["violations"]:
        if v.get("impact") in ("serious", "critical"):
            targets = [", ".join(n["target"]) for n in v["nodes"][:3]]
            out.append(f"{v['id']} ({v['impact']}): {targets}")
    return out


@pytest.mark.parametrize("case", list(CASES), ids=list(CASES))
def test_AT_E04_axe_has_no_serious_violations(platform, new_context, pages_to_check, case):  # noqa: F811
    axe_module = pytest.importorskip("axe_playwright_python.sync_playwright")
    axe = axe_module.Axe()
    scheme, width, height = CASES[case]
    mobile = width < 768
    page = new_context(
        color_scheme=scheme, viewport={"width": width, "height": height}, is_mobile=mobile, has_touch=mobile
    ).new_page()
    found: dict[str, list[str]] = {}
    page.goto(platform.url("/login"))
    login_issues = _serious(axe.run(page, options={"runOnly": WCAG_TAGS}).response)
    if login_issues:
        found["/login"] = login_issues
    login(page, platform, platform.owner_email, OWNER_PASSWORD)
    for path in pages_to_check:
        resp = page.goto(platform.url(path))
        assert resp is not None and resp.status == 200, (path, resp and resp.status)
        issues = _serious(axe.run(page, options={"runOnly": WCAG_TAGS}).response)
        if issues:
            found[path] = issues
    assert not found, f"violaciones serias/críticas de axe ({case}): {found}"
