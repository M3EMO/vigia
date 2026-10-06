"""Regresiones del panel, cierre: tabla de Auditoría a 1024 px y atajos de una tecla (WCAG 2.1.4).

El comportamiento en el navegador (anchos, axe, teclas) se mide en
`tests/e2e/test_e2e_panel_cierre.py`. Acá se fija la causa en el HTML y el JS que sirve la plataforma.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from _panel_support import add_user, login, make_app, make_runtime

from vigia_ee import audit, panel

TEMPLATES = Path(panel.TEMPLATES_DIR)
STATIC = TEMPLATES.parent / "static"


@pytest.fixture
def env(tmp_path):
    runtime = make_runtime(tmp_path)
    org = runtime.create_org("Acme")
    owner = add_user(runtime, org, "owner")
    app = make_app(runtime)
    yield runtime, org, owner, app
    runtime.close()


def test_audit_table_region_is_focusable_and_object_ids_break_every_16_chars(env):
    """Regresión: 'Auditoría: entre 640 y 1080 px de ancho la tabla desborda su contenedor, la columna
    Detalle queda fuera de vista y axe marca scrollable-region-focusable (serious)'.

    La región desplazable tiene foco y nombre. El id de Objeto ofrece un corte cada 16 caracteres
    con <wbr>: el texto copiado no cambia y el min-content de la columna baja a la mitad."""
    runtime, org, owner, app = env
    target = "0123456789abcdef" * 2 + "&<x>"
    audit.append(runtime.db, org, None, "prueba_cierre", target)
    html = login(app, owner).get("/audit").text

    wrap = re.search(r'<div class="vg-table-wrap"[^>]*>', html)
    assert wrap, "falta la tabla de auditoría"
    assert 'tabindex="0"' in wrap.group(0) and 'role="region"' in wrap.group(0)
    assert 'aria-label="Eventos de auditoría, desplazable"' in wrap.group(0)

    cells = re.findall(r'<td data-label="Objeto">(.*?)</td>', html)
    assert "<code>0123456789abcdef<wbr>0123456789abcdef<wbr>&amp;&lt;x&gt;</code>" in cells, cells
    # Los ids cortos (≤ 16) no llevan cortes.
    assert all("<wbr><wbr>" not in c and not c.endswith("<wbr></code>") for c in cells), cells


def test_char_key_shortcuts_are_gated_by_a_preference():
    """Regresión: 'Los atajos de una sola tecla (j, k, ?, 1, 2, 3) actúan en toda la página y no se
    pueden desactivar (WCAG 2.1.4 Character Key Shortcuts, nivel A)'.

    Cada atajo de una tecla imprimible pasa por charKeysOn(); Mi cuenta ofrece apagarlos y las hojas
    de atajos lo documentan."""
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    for key in ('"j"', '"k"', '"?"'):
        uses = re.findall(r"if \((?:chars && )?ev\.key === " + re.escape(key), js)
        assert uses and all(u.startswith("if (chars && ") for u in uses), (key, uses)
    assert 'if (form && chars && (ev.key === "1" || ev.key === "2" || ev.key === "3"))' in js
    assert js.count("var chars = charKeysOn();") == 2

    account = (TEMPLATES / "account.html").read_text(encoding="utf-8")
    assert "data-shortcuts-pref hidden" in account and "data-shortcuts-toggle" in account
    for name in ("detail.html", "review_item.html"):
        sheet = (TEMPLATES / name).read_text(encoding="utf-8")
        assert 'se apagan en <a href="/account">Mi cuenta</a>' in sheet, name
