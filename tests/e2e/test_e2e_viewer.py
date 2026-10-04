"""AT-E03: un viewer no ve controles de mutación y un POST directo con CSRF válido da 403."""

from __future__ import annotations

import json

import pytest
from _e2e import OWNER_PASSWORD, ConsoleLog, browser_fetch, csrf_token, enqueue_for_review, login, post_form
from playwright.sync_api import expect

pytestmark = pytest.mark.e2e

VIEWER = "lectora@acme.test"
ALLOWED_POST = {"/logout", "/account/password"}


def test_AT_E03_viewer_has_no_mutation_controls_and_direct_posts_get_403(
    platform, new_context, accounts, calibrated
):
    owner = new_context().new_page()
    login(owner, platform, platform.owner_email, OWNER_PASSWORD)
    trace_id = platform.chat("respuesta que mira la lectora")["choices"][0]["vigia"]["trace_id"]
    owner.goto(platform.url(f"/generations/{trace_id}"))
    enqueue_for_review(owner, platform, trace_id, min_reviewers=1)
    item_id = platform.sql("SELECT id FROM review_items WHERE trace_id = ?", (trace_id,))[0]["id"]
    dataset_id = platform.sql("SELECT id FROM datasets WHERE org_id = ? LIMIT 1", (platform.org_a,))[0]["id"]
    key_id = platform.sql("SELECT id FROM api_keys WHERE org_id = ? LIMIT 1", (platform.org_a,))[0]["id"]
    owner_id = platform.sql("SELECT id FROM users WHERE email = ?", (platform.owner_email,))[0]["id"]
    detector_id, job_id = calibrated["detector_id"], calibrated["job_id"]

    page = new_context().new_page()
    console = ConsoleLog(page)
    login(page, platform, VIEWER, accounts[VIEWER])
    nav = page.locator("#vg-nav").inner_text()
    for hidden in ("Playground", "Configuración", "Auditoría", "Datos del tenant"):
        assert hidden not in nav, f"el viewer ve el enlace {hidden!r}"

    readable = [
        "/",
        "/generations",
        f"/generations/{trace_id}",
        "/review",
        f"/review/{item_id}",
        "/model",
        "/performance",
        "/calibration",
        f"/calibration/jobs/{job_id}",
        f"/calibration/detectors/{detector_id}",
        "/policies",
        "/account",
    ]
    for path in readable:
        resp = page.goto(platform.url(path))
        assert resp is not None and resp.status == 200, (path, resp and resp.status)
        actions = page.eval_on_selector_all(
            "form",
            "fs => fs.filter(f => (f.method || '').toLowerCase() === 'post')"
            ".map(f => f.getAttribute('action'))",
        )
        extra = set(actions) - ALLOWED_POST
        assert not extra, f"{path}: el viewer ve formularios de mutación {sorted(extra)}"
        for label in (
            "Tomar el siguiente item",
            "Enviar a revisión",
            "Borrar",
            "Activar detector",
            "Guardar la política",
            "Validar y subir",
            "Calibrar",
            "Generar",
        ):
            expect(page.get_by_role("button", name=label, exact=True)).to_have_count(0)

    for path in ("/playground", "/settings", "/audit", "/data"):
        resp = page.goto(platform.url(path))
        assert resp is not None and resp.status == 403, (path, resp and resp.status)

    page.goto(platform.url("/account"))
    posts = {
        "/review/next": {},
        f"/review/{item_id}/reserve": {},
        f"/review/{item_id}/decide": {"label": "correct"},
        "/review/dataset": {"name": "x"},
        f"/generations/{trace_id}/review": {"target": "response", "min_reviewers": "1"},
        f"/generations/{trace_id}/delete": {"confirm": "yes"},
        "/playground": {"prompt": "hola", "max_tokens": "4", "temperature": "0"},
        "/calibration/start": {"dataset_id": dataset_id},
        f"/calibration/detectors/{detector_id}/activate": {"confirm": "yes"},
        "/policies": {"alpha": "0.1", "gap_max": "1", "min_tokens": "1"},
        "/settings/org": {"store_text": "full", "retention_days": "30"},
        "/settings/users": {"email": "intrusa@acme.test", "role": "viewer"},
        f"/settings/users/{owner_id}/role": {"role": "viewer"},
        f"/settings/users/{owner_id}/disable": {"disabled": "1"},
        "/settings/keys": {"name": "intrusa", "scopes": "ingest"},
        f"/settings/keys/{key_id}/revoke": {},
        "/settings/webhooks": {"url": "https://example.com/h", "events": "trace.flagged"},
        "/data/delete": {"confirm": "yes"},
    }
    got = {path: post_form(page, path, data, platform)["status"] for path, data in posts.items()}
    token = csrf_token(page)
    got["/playground/stream"] = browser_fetch(
        page,
        platform.url("/playground/stream"),
        json=json.dumps({"prompt": "hola", "max_tokens": "4", "temperature": "0"}),
        headers={"X-CSRF-Token": token},
    )["status"]
    got["/calibration/datasets"] = browser_fetch(
        page,
        platform.url("/calibration/datasets"),
        form={"csrf_token": token, "name": "intrusa"},
        files={"file": {"name": "x.jsonl", "type": "application/jsonl", "text": '{"id":"a"}\n'}},
    )["status"]
    wrong = {path: status for path, status in got.items() if status != 403}
    assert not wrong, f"POST del viewer que no dieron 403: {wrong}"

    # Nada cambió: la generación, el item y la key siguen como estaban.
    assert platform.sql("SELECT COUNT(*) AS n FROM traces WHERE id = ?", (trace_id,))[0]["n"] == 1
    assert (
        platform.sql("SELECT COUNT(*) AS n FROM review_decisions WHERE item_id = ?", (item_id,))[0]["n"] == 0
    )
    assert platform.sql("SELECT revoked_at FROM api_keys WHERE id = ?", (key_id,))[0]["revoked_at"] is None
    assert console.errors == [] or all("403" in e for e in console.errors), console.errors
