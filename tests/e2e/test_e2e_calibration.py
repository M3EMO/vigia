"""Calibración por el panel (AT-P16 en navegador): subir JSONL con `answers`, esperar el job, ver el
reporte y activar. El request siguiente usa el detector nuevo."""

from __future__ import annotations

import json
import time

import pytest
from _e2e import OWNER_PASSWORD, ConsoleLog, login
from playwright.sync_api import expect

pytestmark = pytest.mark.e2e


def test_calibration_flow_upload_job_report_activate(platform, new_context, calibrated):
    assert calibrated["console"] == [], calibrated["console"]
    assert "AUROC" in calibrated["report_text"]
    detector_id = calibrated["detector_id"]

    page = new_context().new_page()
    console = ConsoleLog(page)
    login(page, platform, platform.owner_email, OWNER_PASSWORD)
    status = page.evaluate(
        "async url => (await fetch(url, {credentials: 'same-origin'})).json()",
        platform.url(f"/calibration/jobs/{calibrated['job_id']}/status"),
    )
    assert status["state"] == "done" and status["detector_id"] == detector_id, status
    page.goto(platform.url("/calibration"))
    row = page.locator("tr", has=page.locator(f"th a[href='/calibration/detectors/{detector_id}']"))
    expect(row).to_contain_text("Activo")
    page.goto(platform.url(f"/calibration/detectors/{detector_id}"))
    expect(page.locator("form[action$='/activate']")).to_have_count(0)  # ya activo: no se ofrece activar

    out = platform.chat("después de calibrar", max_tokens=8)
    vigia = out["choices"][0]["vigia"]
    assert vigia["calibration_id"] == detector_id
    assert any(t["p_error"] is not None for t in vigia["tokens"]), vigia["tokens"][:3]
    page.goto(platform.url("/"))
    expect(page.locator("main")).not_to_contain_text("No hay un detector activo")
    assert console.errors == [], console.errors


def test_calibration_job_page_without_js_offers_manual_refresh(platform, new_context, calibrated):
    page = new_context(java_script_enabled=False).new_page()
    login(page, platform, platform.owner_email, OWNER_PASSWORD)
    page.goto(platform.url(f"/calibration/jobs/{calibrated['job_id']}"))
    expect(page.locator("main")).to_contain_text("Terminado")
    expect(page.get_by_role("link", name="Ver el reporte del detector")).to_be_visible()


@pytest.mark.parametrize("js", [True, False], ids=["con-js", "sin-js"])
def test_failed_calibration_job_shows_the_reason(platform, new_context, js):
    """Un dataset chico hace fallar el job: la página lo dice (con JS por sondeo, sin JS al recargar),
    respeta los alphas elegidos y no activa nada."""
    path = platform.data_dir / "calibracion-chica.jsonl"
    path.write_text(
        "".join(
            json.dumps({"id": f"chico-{i}", "question": f"pregunta {i}", "answers": ["nunca"]}) + "\n"
            for i in range(6)
        ),
        encoding="utf-8",
    )
    active_before = platform.sql("SELECT active_detector_id FROM orgs WHERE id = ?", (platform.org_a,))[0][0]
    page = new_context(java_script_enabled=js).new_page()
    console = ConsoleLog(page)
    login(page, platform, platform.owner_email, OWNER_PASSWORD)
    page.goto(platform.url("/calibration"))
    page.set_input_files("#file", str(path))
    page.fill("form[action='/calibration/datasets'] #ds-name", f"dataset chico {js}")
    page.click("form[action='/calibration/datasets'] button[type=submit]")
    page.wait_for_load_state()
    assert "ok=dataset_uploaded" in page.url, page.locator(".vg-alert").all_inner_texts()
    option = page.locator("#dataset_id option", has_text=f"dataset chico {js}").first
    page.select_option("#dataset_id", option.get_attribute("value"))
    page.fill("#alphas", "0,05; 0,2")
    page.click("form[action='/calibration/start'] button[type=submit]")
    page.wait_for_load_state()
    assert "/calibration/jobs/" in page.url, page.url
    status = page.locator("[data-job-status]")
    if js:
        expect(status).to_have_attribute("aria-busy", "false", timeout=600_000)
    else:
        deadline = time.monotonic() + 600
        while "Falló" not in status.inner_text() and time.monotonic() < deadline:
            time.sleep(3)
            page.get_by_role("link", name="Actualizar estado").click()
            page.wait_for_load_state()
    expect(status).to_contain_text("Falló")
    expect(status.locator("[role=alert]")).to_contain_text("al menos")
    expect(status).to_contain_text("0,05; 0,20")
    expect(page.get_by_role("link", name="Ver el reporte del detector")).to_have_count(0)
    active_after = platform.sql("SELECT active_detector_id FROM orgs WHERE id = ?", (platform.org_a,))[0][0]
    assert active_after == active_before
    assert console.errors == [], console.errors


def test_balanced_dataset_calibrates_with_the_default_seed(calibrated):
    """Regresión: "El split de calibración por pregunta no se estratifica: un dataset balanceado de
    24 respuestas falla con la semilla por defecto". 24 respuestas (12 correctas y 12 erróneas, más
    que el mínimo de 20) calibran con la semilla que trae el formulario: el split estratificado no
    deja a `cal` o `test` con una sola clase."""
    assert calibrated["seed_failures"] == [], calibrated["seed_failures"]
