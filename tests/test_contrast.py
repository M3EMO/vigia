"""AT-E06: contraste del texto sobre cada nivel de resaltado, en modo claro y oscuro.

Los colores se leen de `vigia_ee/static/app.css` (variables de `:root` y de `:root[data-theme="dark"]`),
no de una copia: si alguien cambia la paleta, este test mide la nueva.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

CSS_PATH = Path(__file__).resolve().parents[1] / "vigia_ee" / "static" / "app.css"
LEVELS = ("low", "medium", "high")
VAR_RE = re.compile(r"(--vg-[a-z0-9-]+)\s*:\s*(#[0-9a-fA-F]{6})\b")


def _block(css: str, selector: str) -> str:
    start = css.index(selector)
    open_brace = css.index("{", start)
    depth, i = 0, open_brace
    while True:
        if css[i] == "{":
            depth += 1
        elif css[i] == "}":
            depth -= 1
            if depth == 0:
                return css[open_brace + 1 : i]
        i += 1


def _palette(block: str) -> dict[str, str]:
    return {name: value.lower() for name, value in VAR_RE.findall(block)}


def _luminance(hex_color: str) -> float:
    def channel(c: int) -> float:
        s = c / 255
        return s / 12.92 if s <= 0.04045 else ((s + 0.055) / 1.055) ** 2.4

    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)


def contrast(a: str, b: str) -> float:
    la, lb = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


@pytest.fixture(scope="module")
def palettes() -> dict[str, dict[str, str]]:
    css = CSS_PATH.read_text(encoding="utf-8")
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    light = _palette(_block(css, ":root {"))
    dark = _palette(_block(css, ':root[data-theme="dark"]'))
    media = _palette(_block(css, ':root:not([data-theme="light"])'))
    return {"light": light, "dark": dark, "dark_media": media}


def test_contrast_helper_matches_wcag_reference_values():
    assert contrast("#000000", "#ffffff") == pytest.approx(21.0)
    assert contrast("#777777", "#ffffff") == pytest.approx(4.48, abs=0.01)


@pytest.mark.parametrize("mode", ["light", "dark"])
@pytest.mark.parametrize("level", LEVELS)
def test_AT_E06_text_over_each_highlight_level_is_at_least_4_5(palettes, mode, level):
    pal = palettes[mode]
    ratio = contrast(pal["--vg-hl-ink"], pal[f"--vg-hl-{level}-bg"])
    assert ratio >= 4.5, f"{mode}/{level}: {ratio:.2f}:1"


def test_AT_E06_dark_mode_by_media_query_and_by_toggle_use_the_same_palette(palettes):
    hl = {k: v for k, v in palettes["dark"].items() if k.startswith("--vg-hl-") or k == "--vg-text"}
    assert hl and all(palettes["dark_media"].get(k) == v for k, v in hl.items())


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_AT_E06_levels_are_a_monotonic_single_hue_ramp(palettes, mode):
    """El fondo crece en intensidad con el nivel (secuencial), así el nivel no depende solo del tono."""
    pal = palettes[mode]
    lum = [_luminance(pal[f"--vg-hl-{level}-bg"]) for level in LEVELS]
    if mode == "light":
        assert lum[0] > lum[1] > lum[2]
    else:
        assert lum[0] < lum[1] < lum[2]


@pytest.mark.parametrize("mode", ["light", "dark"])
@pytest.mark.parametrize(
    ("fg", "bg"),
    [
        ("--vg-text", "--vg-surface"),
        ("--vg-text", "--vg-page-bg"),
        ("--vg-text-muted", "--vg-surface"),
        ("--vg-accent", "--vg-surface"),
        ("--vg-accent-ink", "--vg-accent"),
        ("--vg-danger-ink", "--vg-danger"),
        ("--vg-warn-ink", "--vg-warn-soft"),
        ("--vg-danger", "--vg-surface"),
    ],
)
def test_ui_text_pairs_meet_4_5(palettes, mode, fg, bg):
    pal = palettes[mode]
    ratio = contrast(pal[fg], pal[bg])
    assert ratio >= 4.5, f"{mode} {fg} sobre {bg}: {ratio:.2f}:1"


@pytest.mark.parametrize("mode", ["light", "dark"])
@pytest.mark.parametrize("level", LEVELS)
def test_underline_of_each_level_is_visible_against_its_background(palettes, mode, level):
    """WCAG 1.4.11: el subrayado (segunda codificación) llega a 3:1 sobre su fondo."""
    pal = palettes[mode]
    ratio = contrast(pal[f"--vg-hl-{level}-line"], pal[f"--vg-hl-{level}-bg"])
    assert ratio >= 3.0, f"{mode}/{level}: {ratio:.2f}:1"


def test_panel_tokens_do_not_collide_with_chart_css():
    """`charts.css` se carga después de `app.css`: una variable repetida pisaría la paleta del panel."""
    from vigia.charts import CHART_CSS

    mine = set(re.findall(r"(--vg-[a-z0-9-]+)\s*:", CSS_PATH.read_text(encoding="utf-8")))
    theirs = set(re.findall(r"(--vg-[a-z0-9-]+)\s*:", CHART_CSS))
    assert mine and theirs and not (mine & theirs)
