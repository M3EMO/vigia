"""Tests del panel de operación: cola de revisión (AT-P15), calibración, políticas, configuración,
auditoría (AT-P13 en CSV) y datos del tenant."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
from _panel_support import (
    INFO,
    PW,
    activate_detector,
    add_user,
    csrf_of,
    login,
    make_app,
    make_detector,
    make_runtime,
    play,
)
from fastapi.testclient import TestClient

from vigia.calibrate import train_detector
from vigia.contracts import ModelInfo
from vigia.evaluate import evaluate_report
from vigia_ee import audit, webhooks
from vigia_ee.auth import LoginFailed, Principal, verify_api_key
from vigia_ee.auth import login as auth_login
from vigia_ee.security import SESSION_COOKIE

SECRET_RE = re.compile(r'<code id="secret-value">([^<]+)</code>')


@pytest.fixture
def env(tmp_path):
    runtime = make_runtime(tmp_path, webhook_allowed_hosts=("hooks.example",), max_upload_bytes=4096)
    org = runtime.create_org("Acme")
    users = {
        "owner": add_user(runtime, org, "owner"),
        "admin": add_user(runtime, org, "admin"),
        "rev1": add_user(runtime, org, "reviewer", "rev1@ejemplo.com.ar"),
        "rev2": add_user(runtime, org, "reviewer", "rev2@ejemplo.com.ar"),
        "viewer": add_user(runtime, org, "viewer"),
    }
    app = make_app(runtime)
    yield runtime, org, users, app
    runtime.close()


def post(client: TestClient, path: str, data: dict | None = None, **kw):
    token = csrf_of(client)
    return client.post(path, data={"csrf_token": token, **(data or {})}, follow_redirects=False, **kw)


# --- cola de revisión ----------------------------------------------------------------------------


def test_AT_P15_two_reviewers_hidden_decisions_and_labels_exported_to_calibration(env):
    runtime, org, users, app = env
    activate_detector(runtime, org)
    webhooks.create_webhook(
        runtime.db,
        org,
        "http://hooks.example/vigia",
        ["review.completed"],
        None,
        allowed_hosts=("hooks.example",),
    )
    done = play(runtime, users["owner"])
    assert done["action"] == "human_review"
    items = runtime.repo(org).list_review_items()
    assert len(items) == 1 and items[0]["min_reviewers"] == 2 and items[0]["span_idx"] == 0
    item_id = items[0]["id"]

    a, b = login(app, users["rev1"]), login(app, users["rev2"])
    resp = post(a, "/review/next")
    assert resp.headers["location"] == f"/review/{item_id}"
    assert runtime.repo(org).get_review_item(item_id)["state"] == "reserved"
    # B no puede tomar ni decidir un item reservado por A.
    assert post(b, "/review/next").headers["location"] == "/review?ok=queue_empty"
    page_b = b.get(f"/review/{item_id}").text
    assert "Otro revisor tiene reservado este item" in page_b and 'action="/review/' not in page_b.split(
        "Decisión</h2>"
    )[1].split("</section>")[0].replace("/reserve", "")
    assert post(b, f"/review/{item_id}/decide", {"label": "error"}).status_code == 409

    page_a = a.get(f"/review/{item_id}").text
    for key, label in (("1", "correct"), ("2", "error"), ("3", "uncertain")):
        assert f'value="{label}" required data-key="{key}"' in page_a
    assert "data-submit-next" in page_a and "Ctrl</kbd>+<kbd>Enter" in page_a
    assert (
        post(a, f"/review/{item_id}/decide", {"label": "error", "note": "1820 es incorrecto"}).status_code
        == 303
    )
    assert runtime.repo(org).get_review_item(item_id)["state"] == "needs_other"
    page_a = a.get(f"/review/{item_id}").text
    assert (
        "Tu decisión: <strong>Error</strong>" in page_a and "se muestran cuando el item se completa" in page_a
    )

    # B toma el item y no ve la decisión ni la nota de A.
    assert post(b, "/review/next").headers["location"] == f"/review/{item_id}"
    page_b = b.get(f"/review/{item_id}").text
    assert "1820 es incorrecto" not in page_b and users["rev1"] not in page_b and "1 de 2" in page_b
    resp = post(b, f"/review/{item_id}/decide", {"label": "error", "next": "1"})
    assert resp.headers["location"] == "/review?ok=queue_empty"
    assert runtime.repo(org).get_review_item(item_id)["state"] == "completed"

    # Completo: las dos decisiones quedan visibles para ambos.
    for client in (a, b):
        html = client.get(f"/review/{item_id}").text
        assert users["rev1"] in html and users["rev2"] in html and "1820 es incorrecto" in html
        assert "data-review-form" not in html
    events = [j["payload"]["event_type"] for j in runtime.jobs.list(org, webhooks.JOB_KIND)]
    assert "review.completed" in events

    # Las etiquetas realimentan la calibración.
    viewer = login(app, users["viewer"])
    assert "Crear dataset desde revisiones" not in viewer.get("/review").text
    admin = login(app, users["admin"])
    resp = post(admin, "/review/dataset", {"name": "revisiones de prueba"})
    assert resp.status_code == 303 and resp.headers["location"] == "/calibration?ok=dataset_uploaded"
    ds = runtime.repo(org).list_datasets()
    assert [d["name"] for d in ds] == ["revisiones de prueba"] and ds[0]["n_items"] == 1
    line = json.loads(runtime.repo(org).get_dataset(ds[0]["id"])["content"].decode().splitlines()[0])
    assert line["hallucinated_spans"][0]["text"] == "1820"
    assert "revisiones de prueba" in admin.get("/calibration").text


def test_review_list_filters_and_empty_states(env):
    runtime, org, users, app = env
    client = login(app, users["viewer"])
    assert "La cola está vacía" in client.get("/review").text
    activate_detector(runtime, org)
    play(runtime, users["owner"])
    html = client.get("/review").text
    assert "Pendiente (1)" in html and "Fragmento 1" in html and "0 de 2" in html
    assert "Tomar el siguiente item" not in html  # el viewer no muta
    assert "No hay items en el estado" in client.get("/review?estado=completed").text
    assert 'role="alert"' in client.get("/review?estado=borrado").text


def test_review_decision_validation_and_reservation_conflicts(env):
    runtime, org, users, app = env
    activate_detector(runtime, org)
    play(runtime, users["owner"])
    item_id = runtime.repo(org).list_review_items()[0]["id"]
    a = login(app, users["rev1"])
    assert post(a, f"/review/{item_id}/decide", {"label": "quizas"}).status_code == 422
    assert post(a, f"/review/{item_id}/decide", {"label": "error", "note": "x" * 2001}).status_code == 422
    assert post(a, f"/review/{item_id}/reserve").status_code == 303
    b = login(app, users["rev2"])
    conflict = post(b, f"/review/{item_id}/reserve")
    assert conflict.status_code == 409 and 'role="alert"' in conflict.text
    assert post(a, f"/review/{item_id}/decide", {"label": "correct"}).status_code == 303
    assert post(a, f"/review/{item_id}/decide", {"label": "error"}).status_code == 409  # ya decidió


# --- calibración ---------------------------------------------------------------------------------


def _report_detector():
    rng = np.random.default_rng(0)
    n_q, per_q = 60, 10
    n = n_q * per_q
    groups = np.repeat(np.arange(n_q), per_q)
    responses = np.repeat(np.arange(n_q * 2), per_q // 2)
    a, b = rng.normal(size=n), rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(2.0 * a - b - 1.0)))).astype(int)
    feats = {"out_nll": a, "out_entropy": b}
    det, split_report = train_detector(feats, y, groups, responses, INFO, c_grid=(1.0,))
    report = evaluate_report(
        feats, y, groups, responses, INFO, detector=det, split=split_report["groups"], n_boot=20
    )
    report["latency"] = {
        "baseline": {"p50": 10.0, "p95": 12.0, "p99": 15.0},
        "vigia": {"p50": 10.3, "p95": 12.4, "p99": 16.5},
        "target": 0.05,
    }
    return det, report


def test_calibration_upload_validation_and_limits(env):
    runtime, org, users, app = env
    admin = login(app, users["admin"])
    bad = post(
        admin,
        "/calibration/datasets",
        files={"file": ("datos.jsonl", b"esto no es json\n", "application/jsonl")},
    )
    assert bad.status_code == 422 and 'role="alert"' in bad.text and 'aria-invalid="true"' in bad.text
    big = post(
        admin, "/calibration/datasets", files={"file": ("grande.jsonl", b"{}\n" * 2000, "application/jsonl")}
    )
    assert big.status_code == 413 and "supera el máximo" in big.text
    empty = post(admin, "/calibration/datasets", data={"name": "x"})
    assert empty.status_code == 422 and "Elegí un archivo JSONL" in empty.text
    rows = [
        {"id": f"q{i}", "question": f"pregunta {i}", "response": f"respuesta {i}", "label": i % 2}
        for i in range(3)
    ]
    content = ("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n").encode()
    ok = post(
        admin, "/calibration/datasets", files={"file": ("mis-datos.jsonl", content, "application/jsonl")}
    )
    assert ok.status_code == 303
    ds = runtime.repo(org).list_datasets()
    assert ds[0]["name"] == "mis-datos" and ds[0]["n_items"] == 3
    reviewer = login(app, users["rev1"])
    assert "Validar y subir" not in reviewer.get("/calibration").text
    assert (
        post(
            reviewer, "/calibration/datasets", files={"file": ("a.jsonl", content, "application/jsonl")}
        ).status_code
        == 403
    )


def test_calibration_job_lifecycle_status_and_failure_message(env):
    runtime, org, users, app = env
    repo = runtime.repo(org)
    content = b'{"id": "q1", "question": "hola", "response": "chau", "label": 0}\n'
    dataset_id = repo.insert_dataset("chico", content, n_items=1)
    admin = login(app, users["admin"])
    bad = post(admin, "/calibration/start", {"dataset_id": dataset_id, "alphas": "0,05; 0,9"})
    assert bad.status_code == 422 and "Los alphas van de 0,001 a 0,5" in bad.text
    assert post(admin, "/calibration/start", {"dataset_id": "0" * 32}).status_code == 404
    resp = post(admin, "/calibration/start", {"dataset_id": dataset_id, "alphas": "0,1; 0,2", "seed": "3"})
    assert resp.status_code == 303
    job_url = resp.headers["location"].split("?")[0]
    page = admin.get(job_url).text
    assert "En cola" in page and 'data-job-status="' in page and 'aria-live="polite"' in page
    assert "0,10; 0,20" in page and "Actualizar estado" in page
    status = admin.get(job_url + "/status").json()
    assert status["state"] == "queued" and status["label"] == "En cola"
    runtime.jobs.run_until_idle()
    page = admin.get(job_url).text
    assert "Falló" in page and 'role="alert"' in page
    assert admin.get(job_url + "/status").json()["state"] == "failed"
    assert admin.get("/calibration/jobs/" + "0" * 32).status_code == 404


def test_detector_report_charts_tables_and_activation(env):
    runtime, org, users, app = env
    det, report = _report_detector()
    repo = runtime.repo(org)
    repo.insert_detector(det.id, det.to_json(), report_json=json.dumps(report))
    admin = login(app, users["admin"])
    listing = admin.get("/calibration").text
    assert f"/calibration/detectors/{det.id}" in listing and "Candidato" in listing
    page = admin.get(f"/calibration/detectors/{det.id}").text
    for kind in ("reliability", "roc", "pr", "hist", "latency"):
        assert f'class="vg-chart vg-{kind}"' in page, kind
    assert page.count('role="img"') >= 5 and page.count('<details class="vg-data">') >= 5
    assert "AUROC por token" in page and "IC 95 %" in page and "Umbrales por alpha" in page
    no_confirm = post(admin, f"/calibration/detectors/{det.id}/activate")
    assert no_confirm.status_code == 422 and "Confirmá la activación" in no_confirm.text
    resp = post(admin, f"/calibration/detectors/{det.id}/activate", {"confirm": "yes"})
    assert resp.status_code == 303 and runtime.detector_for(org).id == det.id
    assert "Activo" in admin.get("/calibration").text
    assert "Activar detector" not in admin.get(f"/calibration/detectors/{det.id}").text
    viewer = login(app, users["viewer"])
    assert "Activar detector" not in viewer.get(f"/calibration/detectors/{det.id}").text


def test_activation_of_detector_for_other_model_shows_error(env):
    runtime, org, users, app = env
    other = ModelInfo("otro-modelo", "mixtral", 2, (0, 1), 4, 2, 100, "softmax_topk")
    base = make_detector()
    artifact = base.artifact
    artifact["model"] = other.fingerprint()
    artifact.pop("sha256", None)
    from vigia.detector import Detector, artifact_sha256

    artifact["sha256"] = artifact_sha256(artifact)
    foreign = Detector(artifact)
    runtime.repo(org).insert_detector(foreign.id, foreign.to_json())
    admin = login(app, users["admin"])
    resp = post(admin, f"/calibration/detectors/{foreign.id}/activate", {"confirm": "yes"})
    assert resp.status_code == 422 and "No se puede activar" in resp.text
    assert (
        post(admin, "/calibration/detectors/" + "0" * 16 + "/activate", {"confirm": "yes"}).status_code == 404
    )


# --- políticas -----------------------------------------------------------------------------------


def test_policies_viewer_is_read_only_and_admin_saves_with_audit(env):
    runtime, org, users, app = env
    activate_detector(runtime, org)
    viewer = login(app, users["viewer"]).get("/policies").text
    assert "Política vigente" in viewer and 'action="/policies"' not in viewer
    assert "Tasa de marcado esperada" in viewer and "25,0 %" in viewer  # alphas del detector activo
    # Detector.tau es un escalón (no interpola): el texto de ayuda lo dice así.
    assert "interpola" not in viewer and "alpha guardado inferior" in viewer
    admin = login(app, users["admin"])
    form = {
        "alpha": "0,3",
        "gap_max": "2",
        "min_tokens": "1",
        "annotate": "on",
        "human_review_at": "0,6",
        "rag_retry_at": "",
        "abstain_at": "0.95",
        "abstain_message": "Prefiero no responder.",
        "rag_url": "",
        "rag_max_retries": "1",
        "rag_fallback_action": "abstain",
        "on_detector_error": "fail_open",
        "stream_policy": "buffer",
    }
    resp = post(admin, "/policies", form)
    assert resp.status_code == 303
    policy = runtime.policy_for(org)
    assert (policy.alpha, policy.human_review_at, policy.abstain_at, policy.rag_retry_at) == (
        0.3,
        0.6,
        0.95,
        None,
    )
    assert policy.on_detector_error == "fail_open" and policy.version == 2
    assert [r["action"] for r in audit.export_range(runtime.db, org)][-1] == "policy_change"


def test_policies_inline_errors(env):
    runtime, org, users, app = env
    activate_detector(runtime, org, make_detector(min_alpha=0.2))
    admin = login(app, users["admin"])
    base = {
        "alpha": "0,3",
        "gap_max": "1",
        "min_tokens": "1",
        "human_review_at": "",
        "rag_retry_at": "",
        "abstain_at": "",
        "abstain_message": "No sé.",
        "rag_url": "",
        "rag_max_retries": "1",
        "rag_fallback_action": "human_review",
        "on_detector_error": "fail_closed",
        "stream_policy": "annotate_only",
    }
    bad = post(admin, "/policies", {**base, "alpha": "abc", "human_review_at": "2", "gap_max": "-1"})
    assert bad.status_code == 422
    assert 'id="alpha" name="alpha"' in bad.text and bad.text.count('aria-invalid="true"') == 3
    assert 'value="abc"' in bad.text  # se conserva lo que escribió la persona
    low = post(admin, "/policies", {**base, "alpha": "0,1"})
    assert low.status_code == 422 and "umbrales desde alpha 0,20" in low.text
    ssrf = post(admin, "/policies", {**base, "rag_url": "http://169.254.169.254/latest"})
    # El error y la ayuda del campo se anuncian juntos (test_panel_r3).
    assert ssrf.status_code == 422 and 'aria-describedby="err-rag_url rag-url-hint"' in ssrf.text
    assert runtime.policy_for(org).version == 1
    assert post(login(app, users["rev1"]), "/policies", base).status_code == 403


# --- configuración -------------------------------------------------------------------------------


def test_invite_user_with_temporary_password_shown_once(env):
    runtime, org, users, app = env
    admin = login(app, users["admin"])
    page = admin.get("/settings").text
    assert '<option value="owner"' not in page.split('id="invite-role"')[1].split("</select>")[0]
    assert (
        post(admin, "/settings/users", {"email": "jefa@ejemplo.com.ar", "role": "owner"}).status_code == 403
    )
    resp = post(admin, "/settings/users", {"email": "nueva@ejemplo.com.ar", "role": "reviewer"})
    assert resp.status_code == 200 and "Se muestra una sola vez" in resp.text
    temp = SECRET_RE.search(resp.text).group(1)
    assert len(temp) == 23
    assert temp not in admin.get("/settings").text
    login(app, "nueva@ejemplo.com.ar", temp)
    dup = post(admin, "/settings/users", {"email": "nueva@ejemplo.com.ar", "role": "viewer"})
    assert dup.status_code == 422 and "No se pudo crear una cuenta con ese email" in dup.text


def test_change_role_disable_and_owner_protections(env):
    runtime, org, users, app = env
    repo = runtime.repo(org)
    ids = {u["email"]: u["id"] for u in repo.list_users()}
    viewer = login(app, users["viewer"])
    owner = login(app, users["owner"])
    admin = login(app, users["admin"])
    assert (
        post(owner, f"/settings/users/{ids[users['viewer']]}/role", {"role": "reviewer"}).status_code == 303
    )
    assert repo.get_user(ids[users["viewer"]])["role"] == "reviewer"
    assert viewer.get("/account", follow_redirects=False).status_code == 303  # sesión regenerada
    assert post(admin, f"/settings/users/{ids[users['owner']]}/role", {"role": "viewer"}).status_code == 403
    assert post(admin, f"/settings/users/{ids[users['owner']]}/disable", {"disabled": "1"}).status_code == 403
    assert post(owner, f"/settings/users/{ids[users['owner']]}/disable", {"disabled": "1"}).status_code == 422
    assert post(owner, f"/settings/users/{ids[users['owner']]}/role", {"role": "admin"}).status_code == 422
    rev = login(app, users["rev1"])
    assert post(admin, f"/settings/users/{ids[users['rev1']]}/disable", {"disabled": "1"}).status_code == 303
    assert rev.get("/account", follow_redirects=False).status_code == 303
    with pytest.raises(LoginFailed):
        auth_login(runtime.auth, users["rev1"], PW, ip="127.0.0.1")
    assert "Deshabilitada" in admin.get("/settings").text
    assert post(admin, f"/settings/users/{ids[users['rev1']]}/disable", {"disabled": "0"}).status_code == 303
    assert post(admin, "/settings/users/" + "0" * 32 + "/role", {"role": "viewer"}).status_code == 404


def test_api_key_created_once_works_and_revokes(env):
    runtime, org, users, app = env
    admin = login(app, users["admin"])
    bad = post(admin, "/settings/keys", {"name": "", "expires_days": "0"})
    assert bad.status_code == 422 and bad.text.count('class="vg-field-error"') == 3
    resp = post(
        admin, "/settings/keys", {"name": "backend prod", "scopes": ["ingest", "read"], "expires_days": "30"}
    )
    token = SECRET_RE.search(resp.text).group(1)
    assert token.startswith("vg_live_")
    principal = verify_api_key(runtime.auth, token)
    assert principal is not None and principal.org_id == org and principal.scopes == {"ingest", "read"}
    page = admin.get("/settings").text
    assert token not in page and "backend prod" in page
    assert post(admin, f"/settings/keys/{principal.key_id}/revoke").status_code == 303
    assert verify_api_key(runtime.auth, token) is None
    assert post(admin, f"/settings/keys/{principal.key_id}/revoke").status_code == 404
    assert "Revocada" in admin.get("/settings").text


def test_webhooks_ssrf_secret_once_test_and_delete(env):
    runtime, org, users, app = env
    owner = login(app, users["owner"])
    blocked = post(
        owner, "/settings/webhooks", {"url": "http://localhost:8080/x", "events": ["trace.flagged"]}
    )
    assert blocked.status_code == 422 and "Destino no permitido" in blocked.text
    no_events = post(owner, "/settings/webhooks", {"url": "http://hooks.example/x"})
    assert no_events.status_code == 422
    resp = post(owner, "/settings/webhooks", {"url": "http://hooks.example/x", "events": ["trace.flagged"]})
    secret = SECRET_RE.search(resp.text).group(1)
    assert secret.startswith("whsec_") and secret not in owner.get("/settings").text
    hook_id = runtime.repo(org).list_webhooks()[0]["id"]
    assert post(owner, f"/settings/webhooks/{hook_id}/test").status_code == 303
    jobs = runtime.jobs.list(org, webhooks.JOB_KIND)
    assert jobs and jobs[0]["payload"]["event_type"] == "webhook.test"
    assert "En cola" in owner.get("/settings").text
    admin = login(app, users["admin"])
    assert "Crear webhook" not in admin.get("/settings").text
    assert post(admin, f"/settings/webhooks/{hook_id}/delete").status_code == 403
    assert post(owner, f"/settings/webhooks/{hook_id}/delete").status_code == 303
    assert runtime.repo(org).list_webhooks() == []


def test_org_settings_owner_only_with_audit(env):
    runtime, org, users, app = env
    admin = login(app, users["admin"])
    assert "Solo un owner cambia estos datos" in admin.get("/settings").text
    form = {"store_text": "none", "retention_days": "7", "external_judge_allowed": "on"}
    assert post(admin, "/settings/org", form).status_code == 403
    owner = login(app, users["owner"])
    assert post(owner, "/settings/org", {**form, "retention_days": "9999"}).status_code == 422
    assert post(owner, "/settings/org", form).status_code == 303
    org_row = runtime.repo(org).get_org()
    assert (org_row["store_text"], org_row["retention_days"], org_row["external_judge_allowed"]) == (
        "none",
        7,
        1,
    )
    last = audit.export_range(runtime.db, org)[-1]
    assert last["action"] == "settings_change" and last["detail"]["changes"]["store_text"] == "none"


# --- auditoría -----------------------------------------------------------------------------------


def test_audit_view_verify_and_tamper_detection(env):
    runtime, org, users, app = env
    admin = login(app, users["admin"])
    html = admin.get("/audit").text
    assert "<code>login</code>" in html and users["admin"] in html
    assert "La cadena es íntegra" in admin.get("/audit?verificar=1").text
    assert login(app, users["rev1"]).get("/audit").status_code == 403
    runtime.db.execute("DROP TRIGGER audit_log_no_update")
    runtime.db.execute(
        "UPDATE audit_log SET action = 'otra' WHERE seq = (SELECT MIN(seq) FROM audit_log WHERE org_id = ?)",
        (org,),
    )
    broken = admin.get("/audit?verificar=1").text
    assert "La cadena está rota" in broken and 'role="alert"' in broken


def test_AT_P13_audit_csv_export_neutralizes_formulas(env):
    runtime, org, users, app = env
    audit.append(runtime.db, org, None, "trace_delete", '=HYPERLINK("http://x")')
    audit.append(runtime.db, org, None, "trace_delete", "@SUM(A1)")
    admin = login(app, users["admin"])
    resp = admin.get("/audit/export?formato=csv")
    assert resp.status_code == 200 and "attachment" in resp.headers["content-disposition"]
    body = resp.text
    assert "'=HYPERLINK" in body and "'@SUM(A1)" in body
    assert not re.search(r'(^|,)"?=HYPERLINK', body, re.M)
    data = admin.get("/audit/export?formato=json").json()
    assert data["org_id"] == org and data["events"][-1]["action"] in ("audit_export", "trace_delete")
    assert audit.verify_rows(data["events"]).ok
    assert admin.get("/audit/export?formato=xml").status_code == 422
    assert admin.get("/audit/export?desde=ayer").status_code == 422


# --- datos del tenant ----------------------------------------------------------------------------


def test_data_export_and_owner_only_delete(env):
    runtime, org, users, app = env
    play(runtime, users["owner"])
    admin = login(app, users["admin"])
    page = admin.get("/data").text
    assert "Descargar export JSON" in page and "Borrar la organización" not in page
    resp = admin.get("/data/export")
    assert resp.status_code == 200 and "attachment" in resp.headers["content-disposition"]
    data = resp.json()
    assert data["org_id"] == org and len(data["traces"]) == 1
    assert "password_hash" not in resp.text and "hmac" not in json.dumps(data["api_keys"])
    assert post(admin, "/data/delete", {"confirm": "yes", "confirm_name": "Acme"}).status_code == 403
    owner = login(app, users["owner"])
    wrong = post(owner, "/data/delete", {"confirm": "yes", "confirm_name": "acme"})
    assert wrong.status_code == 422 and "nombre exacto" in wrong.text and "Generaciones" in wrong.text
    raw = owner.cookies.get(SESSION_COOKIE)
    gone = post(owner, "/data/delete", {"confirm": "yes", "confirm_name": "Acme"})
    assert gone.status_code == 303 and gone.headers["location"] == "/login?ok=org_deleted"
    assert runtime.repo(org).get_org() is None
    stale = TestClient(app, base_url="https://testserver", cookies={SESSION_COOKIE: raw})
    assert stale.get("/", follow_redirects=False).status_code == 303
    assert any(r["action"] == "org_delete" for r in audit.export_range(runtime.db, org))


def test_data_delete_forgets_the_tenant_detector_in_runtime_and_metrics(env):
    """Regresión: 'Borrar la organización (POST /data/delete) deja el detector del tenant borrado en
    la cache del runtime y en /metrics'."""
    runtime, org_a, _, app = env
    org_b = runtime.create_org("Org Beta")
    owner_b = add_user(runtime, org_b, "owner", "owner@beta.com.ar")
    det_a = activate_detector(runtime, org_a)
    det_b = activate_detector(runtime, org_b, make_detector(min_alpha=0.05))
    assert det_a.id != det_b.id
    assert runtime.detector_for(org_a).id == det_a.id and runtime.detector_for(org_b).id == det_b.id
    assert det_b.id in runtime.metrics.render()
    runtime.save_policy(Principal(org_id=org_b, role="owner"), {"gap_max": 2})
    assert runtime.policy_for(org_b).gap_max == 2 and org_b in runtime._policies
    owner = login(app, owner_b)
    gone = post(owner, "/data/delete", {"confirm": "yes", "confirm_name": "Org Beta"})
    assert gone.status_code == 303 and runtime.repo(org_b).get_org() is None
    text = runtime.metrics.render()
    assert det_b.id not in text, "vigia_calibration_info sigue listando el detector del tenant borrado"
    assert f'vigia_calibration_info{{detector_id="{det_a.id}"}} 1' in text
    assert org_b not in runtime._detectors
    # forget_org: tampoco quedan su política, sus alertas, su retriever ni la marca del detector base.
    for state in (runtime._policies, runtime._policy_alerts, runtime._retrievers, runtime._detector_alerts):
        assert org_b not in state
    assert org_b not in runtime._base_checked
    assert runtime.detector_for(org_a).id == det_a.id  # la otra org no se toca


# --- regresiones de la ronda 2 del panel -----------------------------------------------------------


def test_calibration_job_page_title_follows_the_job_state(env):
    """Regresión: 'El título de la página del job de calibración siempre dice Calibración en curso,
    aunque el job terminó o falló' (WCAG 2.4.2)."""
    runtime, org, users, app = env
    content = b'{"id": "q1", "question": "hola", "response": "chau", "label": 0}\n'
    dataset_id = runtime.repo(org).insert_dataset("chico", content, n_items=1)
    admin = login(app, users["admin"])
    resp = post(admin, "/calibration/start", {"dataset_id": dataset_id})
    job_url = resp.headers["location"].split("?")[0]
    short = job_url.rsplit("/", 1)[1][:12]
    queued = admin.get(job_url).text
    assert f"<title>Calibración en cola ({short}) · Vigía</title>" in queued
    assert f'data-title-done="Calibración terminada ({short}) · Vigía"' in queued
    assert f'data-title-failed="Calibración fallida ({short}) · Vigía"' in queued
    runtime.jobs.run_until_idle()
    failed = admin.get(job_url).text
    assert f"<title>Calibración fallida ({short}) · Vigía</title>" in failed
    assert "Calibración en curso" not in failed
    js = (Path(__file__).parents[1] / "vigia_ee" / "static" / "app.js").read_text(encoding="utf-8")
    assert 'box.getAttribute("data-title-" + job.state)' in js and "document.title = title" in js


def test_detector_report_charts_scroll_instead_of_shrinking(env):
    """Regresión: 'El texto de los gráficos SVG baja de 10 px: los 4 gráficos del reporte miden 317 px
    a 375 px'. Cada SVG del reporte va en una región con scroll, foco y nombre."""
    runtime, org, users, app = env
    det, report = _report_detector()
    runtime.repo(org).insert_detector(det.id, det.to_json(), report_json=json.dumps(report))
    page = login(app, users["admin"]).get(f"/calibration/detectors/{det.id}").text
    regions = re.findall(
        r'<div class="vg-chart-scroll" tabindex="0" role="region" aria-label="Gráfico: ([^"]+)">', page
    )
    assert len(regions) >= 5
    assert all(re.search(r'<title id="[^"]+">' + re.escape(t) + "</title>", page) for t in regions)


def test_detector_report_regions_have_unique_names(env):
    """Regresión: 'Gráficos con 'Ver datos' abierto: la región del SVG y la región de la tabla tienen
    el mismo nombre accesible (axe landmark-unique)'. En el reporte del detector, la región de la
    curva ROC y la de su tabla se llamaban las dos «Curva ROC»."""
    runtime, org, users, app = env
    det, report = _report_detector()
    runtime.repo(org).insert_detector(det.id, det.to_json(), report_json=json.dumps(report))
    page = login(app, users["admin"]).get(f"/calibration/detectors/{det.id}").text
    names = re.findall(r'<div[^>]*role="region"[^>]*aria-label="([^"]+)"', page)
    assert "Gráfico: Curva ROC" in names and "Curva ROC" in names, names
    assert 'role="region" aria-labelledby=' not in page
    assert len(names) == len(set(names)), names
