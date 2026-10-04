"""Flujos completos sobre una organización propia por test: paginación con filtros, exports que el
navegador descarga, revisión con decisiones ocultas, borrado con un revisor trabajando y contenido
largo sin espacios en pantallas chicas."""

from __future__ import annotations

import csv
import io
import json
import re

import pytest
from _e2e import ConsoleLog, create_org, enqueue_for_review, h_overflow, invite_user, login
from playwright.sync_api import expect

pytestmark = pytest.mark.e2e

PASSWORD = "Clave-Larga-De-Una-Org-Propia-2026"  # noqa: S105 - credencial de prueba local
MOBILE = {"viewport": {"width": 375, "height": 812}, "is_mobile": True, "has_touch": True}


@pytest.fixture
def own_org(platform):
    """Crea una org nueva con owner y API key; devuelve (org_id, email, api_key)."""

    def make(slug: str) -> tuple[str, str, str]:
        email = f"duena@{slug}.test"
        org_id, key = create_org(platform, f"Org {slug}", email, PASSWORD)
        return org_id, email, key

    return make


def trace_links(page) -> list[str]:
    hrefs = page.eval_on_selector_all(
        "main table a[href^='/generations/']", "as => as.map(a => a.getAttribute('href'))"
    )
    return [h.rsplit("/", 1)[1] for h in hrefs]


def test_generations_pagination_keeps_filters_and_shows_everything_once(platform, new_context, own_org):
    _org, email, key = own_org("paginas")
    made = [
        platform.chat(f"generación número {i}", key=key, max_tokens=1)["choices"][0]["vigia"]["trace_id"]
        for i in range(53)
    ]
    page = new_context().new_page()
    console = ConsoleLog(page)
    login(page, platform, email, PASSWORD)
    page.goto(platform.url("/generations"))
    page.select_option("#f-origen", "api")
    page.click("form[action='/generations'] button[type=submit]")
    page.wait_for_load_state()
    first = trace_links(page)
    assert len(first) == 50, len(first)
    page.get_by_role("link", name="Página siguiente").click()
    page.wait_for_load_state()
    assert "origen=api" in page.url, page.url
    expect(page.locator("#f-origen")).to_have_value("api")
    second = trace_links(page)
    assert sorted(first + second) == sorted(made), (len(first), len(second))
    expect(page.get_by_role("link", name="Página siguiente")).to_have_count(0)
    page.get_by_role("link", name="Primera página").click()
    page.wait_for_load_state()
    assert "origen=api" in page.url and "cursor=" not in page.url, page.url
    assert trace_links(page) == first
    assert console.errors == [], console.errors


def test_owner_downloads_tenant_export_and_audit_csv(platform, new_context, own_org):
    org_id, email, key = own_org("exporta")
    mine = platform.chat("dato de la org que exporta", key=key)["choices"][0]["vigia"]["trace_id"]
    foreign = platform.chat("dato de la org A")["choices"][0]["vigia"]["trace_id"]
    page = new_context(accept_downloads=True).new_page()
    console = ConsoleLog(page)
    login(page, platform, email, PASSWORD)

    page.goto(platform.url("/data"))
    with page.expect_download() as info:
        page.get_by_role("link", name="Descargar export JSON").click()
    download = info.value
    name = download.suggested_filename
    assert re.fullmatch(r"vigia-export-\d{8}T\d{6}Z\.json", name), name
    data = json.loads(open(download.path(), encoding="utf-8").read())
    trace_ids = {t["id"] for t in data["traces"]}
    assert mine in trace_ids and foreign not in trace_ids, trace_ids
    assert data["org"]["id"] == org_id
    assert "password_hash" not in json.dumps(data)

    page.goto(platform.url("/audit"))
    with page.expect_download() as info:
        page.get_by_role("button", name="Exportar CSV").click()
    download = info.value
    assert download.suggested_filename.endswith(".csv"), download.suggested_filename
    rows = list(csv.DictReader(io.StringIO(open(download.path(), encoding="utf-8").read())))
    actions = {r["action"] for r in rows}
    assert {"login", "data_export"} <= actions, actions
    assert console.errors == [], console.errors


def test_review_decisions_stay_hidden_until_the_item_completes(platform, new_context, own_org):
    _org, email, key = own_org("ciega")
    owner = new_context().new_page()
    login(owner, platform, email, PASSWORD)
    pw1 = invite_user(owner, platform, "rev1@ciega.test", "reviewer")
    pw2 = invite_user(owner, platform, "rev2@ciega.test", "reviewer")
    pw_view = invite_user(owner, platform, "lectura@ciega.test", "viewer")
    trace_id = platform.chat("respuesta que revisan a ciegas", key=key)["choices"][0]["vigia"]["trace_id"]
    owner.goto(platform.url(f"/generations/{trace_id}"))
    enqueue_for_review(owner, platform, trace_id, min_reviewers=2)
    item_id = platform.sql(
        "SELECT id FROM review_items WHERE trace_id = ? AND min_reviewers = 2", (trace_id,)
    )[0]["id"]

    rev1 = new_context().new_page()
    login(rev1, platform, "rev1@ciega.test", pw1)
    rev1.goto(platform.url(f"/review/{item_id}"))
    rev1.check("[data-review-form] input[name=label][value=error]")
    rev1.fill("#note", "nota-privada-de-rev1")
    rev1.click("[data-review-form] button[name=next][value='0']")
    rev1.wait_for_load_state()
    assert "ok=review_saved" in rev1.url, rev1.url

    rev2 = new_context(**MOBILE).new_page()
    login(rev2, platform, "rev2@ciega.test", pw2)
    viewer = new_context().new_page()
    login(viewer, platform, "lectura@ciega.test", pw_view)
    for page in (rev2, viewer, owner):
        page.goto(platform.url(f"/review/{item_id}"))
        text = page.locator("main").inner_text()
        assert "nota-privada-de-rev1" not in text and "rev1@ciega.test" not in text, text[:400]
        page.goto(platform.url(f"/generations/{trace_id}"))
        assert "nota-privada-de-rev1" not in page.locator("main").inner_text()

    rev2.goto(platform.url(f"/review/{item_id}"))
    rev2.check("[data-review-form] input[name=label][value=correct]")
    rev2.click("[data-review-form] button[name=next][value='0']")
    rev2.wait_for_load_state()
    assert "ok=review_saved" in rev2.url, rev2.url
    for page in (rev1, rev2, viewer):
        page.goto(platform.url(f"/review/{item_id}"))
        expect(page.locator("main")).to_contain_text("rev1@ciega.test: Error")
        expect(page.locator("main")).to_contain_text("rev2@ciega.test: Correcto")


def test_generation_deleted_while_a_reviewer_has_it_open(platform, new_context, own_org):
    _org, email, key = own_org("borra")
    owner = new_context().new_page()
    login(owner, platform, email, PASSWORD)
    pw = invite_user(owner, platform, "revisora@borra.test", "reviewer")
    trace_id = platform.chat("generación que se borra", key=key)["choices"][0]["vigia"]["trace_id"]
    owner.goto(platform.url(f"/generations/{trace_id}"))
    enqueue_for_review(owner, platform, trace_id, min_reviewers=1)

    reviewer = new_context(**MOBILE).new_page()
    console = ConsoleLog(reviewer)
    login(reviewer, platform, "revisora@borra.test", pw)
    reviewer.goto(platform.url("/review"))
    reviewer.get_by_role("button", name="Tomar el siguiente item").click()
    reviewer.wait_for_load_state()
    assert re.search(r"/review/[0-9a-f]+", reviewer.url), reviewer.url
    reviewer.check("[data-review-form] input[name=label][value=correct]")

    owner.goto(platform.url(f"/generations/{trace_id}"))
    owner.check("form[action$='/delete'] input[name=confirm]")
    owner.get_by_role("button", name="Borrar", exact=True).click()
    owner.wait_for_load_state()
    assert "ok=trace_deleted" in owner.url, owner.url
    assert owner.goto(platform.url(f"/generations/{trace_id}")).status == 404

    reviewer.click("[data-review-form] button[name=next][value='1']")
    reviewer.wait_for_load_state()
    assert platform.sql("SELECT COUNT(*) AS n FROM review_items WHERE trace_id = ?", (trace_id,))[0]["n"] == 0
    # La revisora termina en una página con salida y la cola sigue usable.
    reviewer.goto(platform.url("/review"))
    reviewer.get_by_role("button", name="Tomar el siguiente item").click()
    reviewer.wait_for_load_state()
    assert "ok=queue_empty" in reviewer.url, reviewer.url
    assert all("404" in e for e in console.errors), console.errors


LONG_URL = "https://docs.ejemplo.test/" + "segmento-sin-espacios-" * 12 + "fin"
LONG_NAME = "nombre_sin_espacios_" * 5


def test_long_unbroken_content_does_not_overflow_on_mobile(platform, new_context, own_org):
    _org, email, key = own_org("larga")
    owner = new_context().new_page()
    login(owner, platform, email, PASSWORD)
    invite_user(owner, platform, "persona.con.un.email.muy.largo.y.sin.espacios.fin@larga.test", "viewer")
    owner.goto(platform.url("/settings"))
    owner.fill("#key-name", LONG_NAME)
    owner.click("form[action='/settings/keys'] button[type=submit]")
    owner.wait_for_load_state()
    trace_id = platform.chat(f"Leé {LONG_URL} y resumí", key=key, max_tokens=24)["choices"][0]["vigia"][
        "trace_id"
    ]
    owner.goto(platform.url(f"/generations/{trace_id}"))
    enqueue_for_review(owner, platform, trace_id, min_reviewers=1)
    item_id = platform.sql("SELECT id FROM review_items WHERE trace_id = ?", (trace_id,))[0]["id"]

    page = new_context(**MOBILE).new_page()
    login(page, platform, email, PASSWORD)
    wide = {}
    for path in ("/settings", f"/generations/{trace_id}", f"/review/{item_id}", "/audit", "/generations"):
        page.goto(platform.url(path))
        for details in page.locator("main details").all():
            details.evaluate("d => d.open = true")
        px = h_overflow(page)
        if px > 0:
            wide[path] = px
    page.goto(platform.url("/playground"))
    page.fill("#prompt", f"Leé {LONG_URL}")
    page.fill("#max_tokens", "8")
    page.click("[data-playground-submit]")
    expect(page.locator("[data-playground-status]")).to_contain_text("Listo", timeout=180_000)
    if h_overflow(page) > 0:
        wide["/playground (resultado)"] = h_overflow(page)
    assert not wide, f"scroll horizontal a 375 px con contenido largo: {wide}"


def test_keyboard_note_with_digits_keeps_the_chosen_label(platform, new_context, own_org):
    """En la cola, 1/2/3 eligen etiqueta, pero dentro de la nota se escriben como texto."""
    _org, email, key = own_org("nota")
    owner = new_context().new_page()
    login(owner, platform, email, PASSWORD)
    pw = invite_user(owner, platform, "revisor@nota.test", "reviewer")
    trace_id = platform.chat("respuesta con nota", key=key)["choices"][0]["vigia"]["trace_id"]
    owner.goto(platform.url(f"/generations/{trace_id}"))
    enqueue_for_review(owner, platform, trace_id, min_reviewers=1)
    item_id = platform.sql("SELECT id FROM review_items WHERE trace_id = ?", (trace_id,))[0]["id"]

    page = new_context().new_page()
    login(page, platform, "revisor@nota.test", pw)
    page.goto(platform.url(f"/review/{item_id}"))
    page.keyboard.press("2")
    expect(page.locator("[data-review-form] input[name=label]:checked")).to_have_attribute("value", "error")
    page.focus("#note")
    page.keyboard.type("falla en 1 de 3 datos")
    with page.expect_navigation():
        page.keyboard.press("Control+Enter")
    rows = platform.sql("SELECT label, note FROM review_decisions WHERE item_id = ?", (item_id,))
    assert [(r["label"], r["note"]) for r in rows] == [("error", "falla en 1 de 3 datos")]
