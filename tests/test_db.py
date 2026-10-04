"""Tests de vigia_ee.db: migraciones, repositorio con tenant obligatorio, trazas, revisión y export."""

from __future__ import annotations

import json
import threading
from datetime import timedelta

import pytest

from vigia.contracts import ScoredToken, Span
from vigia_ee import db as dbm
from vigia_ee.db import Database, TenantScopedRepository, create_org, new_id, utcnow


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "vigia.db")
    yield database
    database.close()


@pytest.fixture
def two_orgs(db):
    a = create_org(db, "Org A", store_text="full")
    b = create_org(db, "Org B", store_text="full")
    return TenantScopedRepository(db, a), TenantScopedRepository(db, b)


def _trace(repo, text="hola mundo", **kw):
    tokens = [
        {"idx": 0, "text": "hola", "start": 0, "end": 4, "p_error": 0.1, "flag": False},
        {"idx": 1, "text": " mundo", "start": 4, "end": 10, "p_error": 0.9, "flag": True},
    ]
    spans = [{"start": 5, "end": 10, "score": 0.9}]
    return repo.insert_trace(
        model_id="tiny",
        response_text=text,
        action="annotate",
        tokens=tokens,
        spans=spans,
        prompt_text="decí hola",
        **kw,
    )


# --- conexión y migraciones -----------------------------------------------------------------------


def test_migrations_apply_once_and_pragmas_are_set(tmp_path):
    path = tmp_path / "m.db"
    first = Database(path)
    assert first.schema_version() == 1
    assert first.query_one("PRAGMA foreign_keys")["foreign_keys"] == 1
    assert first.query_one("PRAGMA journal_mode")["journal_mode"] == "wal"
    assert int(first.query_one("PRAGMA busy_timeout")["timeout"]) == 5000
    first.close()
    second = Database(path)
    assert second.migrate() == 1
    assert len(second.query("SELECT * FROM schema_migrations")) == 1
    second.close()


def test_memory_database_works():
    mem = Database(":memory:")
    org = create_org(mem, "x")
    assert TenantScopedRepository(mem, org).get_org()["name"] == "x"


def test_new_id_is_random_and_not_sequential():
    ids = {new_id() for _ in range(200)}
    assert len(ids) == 200
    assert all(len(i) == 32 and int(i, 16) >= 0 for i in ids)


def test_transaction_rolls_back_on_error(db):
    org = create_org(db, "x")
    with pytest.raises(RuntimeError), db.transaction():
        db.execute("UPDATE orgs SET name = 'cambiado' WHERE id = ?", (org,))
        raise RuntimeError("falla")
    assert TenantScopedRepository(db, org).get_org()["name"] == "x"


def test_repository_requires_org_id(db):
    with pytest.raises(ValueError):
        TenantScopedRepository(db, "")
    with pytest.raises(ValueError):
        TenantScopedRepository(db, None)  # type: ignore[arg-type]


def test_concurrent_writes_from_threads(db):
    repo = TenantScopedRepository(db, create_org(db, "x", store_text="none"))
    errors = []

    def worker():
        try:
            for _ in range(10):
                _trace(repo)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(repo.list_traces(limit=200)[0]) == 40


# --- organización ---------------------------------------------------------------------------------


def test_org_settings_and_policy(db):
    repo = TenantScopedRepository(db, create_org(db, "x"))
    assert repo.get_org()["store_text"] == "redacted"
    assert repo.update_org(retention_days=7, store_text="none", external_judge_allowed=True)
    org = repo.get_org()
    assert (org["retention_days"], org["store_text"], org["external_judge_allowed"]) == (7, "none", 1)
    with pytest.raises(ValueError):
        repo.update_org(retention_days=0)
    with pytest.raises(ValueError):
        repo.update_org(store_text="todo")
    with pytest.raises(ValueError):
        repo.update_org(active_detector_id="x")
    assert repo.set_policy({"alpha": 0.05})
    assert repo.get_policy() == {"alpha": 0.05}


# --- aislamiento entre tenants --------------------------------------------------------------------


def test_at_p01_other_tenant_resources_are_invisible(two_orgs):
    """AT-P01: todo recurso de otro tenant devuelve None/False/vacío desde el repositorio."""
    a, b = two_orgs
    user = a.create_user("ana@example.com", "scrypt$x", "admin")
    trace = _trace(a)
    item = a.create_review_item(trace)
    dataset = a.insert_dataset("d", b'{"q":1}\n', n_items=1)
    a.insert_detector("det-a-0001", "{}", dataset_id=dataset)
    webhook = a.insert_webhook("https://example.com/h", "whsec_x", ["trace.flagged"])
    a.insert_api_key("0123456789abcdef", name="k", hmac_hex="00", pepper_kid="k1", scopes=["read"])

    assert b.get_user(user) is None and b.list_users() == []
    assert b.get_trace(trace) is None and b.list_traces()[0] == []
    assert b.get_review_item(item) is None and b.list_review_items() == []
    assert b.list_review_decisions(item) is None
    assert b.get_dataset(dataset) is None and b.list_datasets() == []
    assert b.get_detector("det-a-0001") is None and b.list_detectors() == []
    assert b.get_webhook(webhook) is None and b.list_webhooks() == []
    assert b.get_api_key("0123456789abcdef") is None and b.list_api_keys() == []

    assert b.update_user(user, role="viewer") is False
    assert b.delete_user(user) is False
    assert b.delete_trace(trace) is False
    assert b.create_review_item(trace) is None
    assert b.add_review_decision(item, user, "correct") is None
    assert b.activate_detector("det-a-0001") is False
    assert b.revoke_api_key("0123456789abcdef") is False
    assert b.set_webhook_disabled(webhook, True) is False
    assert b.delete_webhook(webhook) is False
    assert b.delete_dataset(dataset) is False
    assert b.purge_before(utcnow() + timedelta(days=1)) == 0
    with pytest.raises(ValueError):
        b.insert_detector("det-b-0001", "{}", dataset_id=dataset)

    assert a.get_trace(trace) is not None and a.get_user(user)["role"] == "admin"


def test_user_queries_never_return_password_hash(two_orgs):
    a, _ = two_orgs
    uid = a.create_user("ana@example.com", "scrypt$secreto", "viewer")
    assert "password_hash" not in a.get_user(uid)
    assert all("password_hash" not in u for u in a.list_users())
    assert dbm.lookup_user_by_email(a.db, "ANA@example.com")["password_hash"] == "scrypt$secreto"  # noqa: S105
    with pytest.raises(ValueError):
        a.create_user("ana@example.com", "x", "viewer")
    with pytest.raises(ValueError):
        a.create_user("otro@example.com", "x", "superuser")


# --- trazas ---------------------------------------------------------------------------------------


def test_insert_and_get_trace_with_tokens_and_spans(two_orgs):
    a, _ = two_orgs
    tid = _trace(a, usage={"layers": [0], "counts": [[1, 2]]}, alpha=0.1, tau=0.5)
    t = a.get_trace(tid)
    assert t["response_text"] == "hola mundo" and t["prompt_text"] == "decí hola"
    assert t["n_tokens"] == 2 and t["alpha"] == 0.1
    assert [tok["text"] for tok in t["tokens"]] == ["hola", " mundo"]
    assert t["tokens"][1]["flag"] == 1
    assert t["spans"] == [{"idx": 0, "start": 5, "end": 10, "score": 0.9}]
    assert json.loads(t["usage_json"]) == {"counts": [[1, 2]], "layers": [0]}


def test_insert_trace_accepts_contract_dataclasses(two_orgs):
    a, _ = two_orgs
    tokens = [ScoredToken(0, "ok", 0, 2, 0.3, False)]
    tid = a.insert_trace(
        model_id="m",
        response_text="ok",
        action="pass",
        tokens=tokens,
        spans=[Span(0, 2, 0, 1, 0.3)],
        trace_id="trace-fixed-01",
    )
    assert tid == "trace-fixed-01"
    assert a.get_trace(tid)["tokens"][0]["p_error"] == 0.3
    with pytest.raises(ValueError):
        a.insert_trace(model_id="m", response_text="x", action="pass", trace_id="bad id!")


def test_at_p08_store_text_redacted_keeps_markers_and_never_the_data(db):
    """AT-P08: con store_text=redacted la base guarda marcadores; el dato no aparece en ninguna tabla."""
    repo = TenantScopedRepository(db, create_org(db, "x"))  # default = redacted
    response = "Tu CUIT es 20-12345678-6 y tu DNI 30.123.456, escribí a ana@example.com."
    # Tokens de 4 caracteres para que los datos queden partidos.
    tokens = [
        {
            "idx": i,
            "text": response[s : s + 4],
            "start": s,
            "end": min(s + 4, len(response)),
            "p_error": 0.2,
            "flag": False,
        }
        for i, s in enumerate(range(0, len(response), 4))
    ]
    tid = repo.insert_trace(
        model_id="m",
        response_text=response,
        prompt_text="mi mail es ana@example.com",
        action="pass",
        tokens=tokens,
        spans=[{"start": 11, "end": 24, "score": 0.8}],
    )
    t = repo.get_trace(tid)
    assert t["response_text"] == "Tu CUIT es [CUIT] y tu DNI [DNI], escribí a [EMAIL]."
    assert t["prompt_text"] == "mi mail es [EMAIL]"
    assert "".join(tok["text"] for tok in t["tokens"]) == t["response_text"]
    for tok in t["tokens"]:
        assert t["response_text"][tok["start"] : tok["end"]] == tok["text"]
    assert t["response_text"][t["spans"][0]["start"] : t["spans"][0]["end"]] == "[CUIT]"
    dump = json.dumps(repo.export_all(), ensure_ascii=False)
    for secret in ("20-12345678-6", "30.123.456", "ana@example.com", "12345678"):
        assert secret not in dump


def test_store_text_none_stores_no_text(db):
    repo = TenantScopedRepository(db, create_org(db, "x", store_text="none"))
    tid = _trace(repo, text="hola mundo")
    t = repo.get_trace(tid)
    assert t["response_text"] is None and t["prompt_text"] is None
    assert all(tok["text"] is None for tok in t["tokens"])
    assert t["tokens"][1]["start"] == 4 and len(t["text_sha256"]) == 64


def test_list_traces_cursor_pagination_and_filters(two_orgs):
    a, b = two_orgs
    base = utcnow() - timedelta(hours=1)
    ids = [_trace(a, created_at=base + timedelta(minutes=i), response_p_error=i / 10) for i in range(7)]
    _trace(b)
    seen, cursor = [], None
    while True:
        rows, cursor = a.list_traces(limit=3, cursor=cursor)
        seen.extend(r["id"] for r in rows)
        if cursor is None:
            break
    assert seen == list(reversed(ids))
    rows, _ = a.list_traces(min_p_error=0.5)
    assert {r["id"] for r in rows} == set(ids[5:])
    rows, _ = a.list_traces(since=base + timedelta(minutes=2), until=base + timedelta(minutes=4))
    assert [r["id"] for r in rows] == [ids[3], ids[2]]
    assert a.list_traces(action="abstain")[0] == []
    with pytest.raises(ValueError):
        a.list_traces(cursor="no-es-un-cursor")


def test_delete_trace_cascades_tokens_spans_and_review(two_orgs):
    a, _ = two_orgs
    tid = _trace(a)
    item = a.create_review_item(tid)
    assert a.delete_trace(tid)
    assert a.get_review_item(item) is None
    assert a.db.query("SELECT * FROM trace_tokens WHERE trace_id = ?", (tid,)) == []
    assert a.db.query("SELECT * FROM spans WHERE trace_id = ?", (tid,)) == []


def test_at_p10_purge_before_removes_only_expired_traces_of_the_tenant(two_orgs):
    """AT-P10: la purga borra lo vencido del tenant y nada más."""
    a, b = two_orgs
    now = utcnow()
    old = _trace(a, created_at=now - timedelta(days=40))
    fresh = _trace(a, created_at=now - timedelta(days=5))
    other_old = _trace(b, created_at=now - timedelta(days=40))
    assert a.purge_before(now - timedelta(days=30)) == 1
    assert a.get_trace(old) is None
    assert a.get_trace(fresh) is not None
    assert b.get_trace(other_old) is not None


def test_at_p10_export_contains_everything_of_the_tenant_and_nothing_foreign(two_orgs):
    """AT-P10: el export JSON incluye todo lo del tenant, nada ajeno y ningún secreto."""
    a, b = two_orgs
    uid = a.create_user("ana@example.com", "scrypt$HASH-SECRETO", "owner")
    tid = _trace(a)
    a.create_review_item(tid)
    a.insert_webhook("https://example.com/h", "whsec_SECRETO", ["trace.flagged"])
    a.insert_api_key("0123456789abcdef", name="k", hmac_hex="HMAC-SECRETO", pepper_kid="k1", scopes=["read"])
    a.insert_dataset("d", b'{"q":"hola"}\n', n_items=1)
    foreign = _trace(b, text="dato ajeno")
    b.create_user("beto@example.com", "scrypt$x", "owner")

    export = a.export_all()
    dump = json.dumps(export, ensure_ascii=False)
    assert export["org_id"] == a.org_id and export["schema_version"] == "1"
    assert [t["id"] for t in export["traces"]] == [tid]
    assert len(export["traces"][0]["tokens"]) == 2 and len(export["review_items"]) == 1
    assert export["users"][0]["id"] == uid and export["datasets"][0]["content"] == '{"q":"hola"}\n'
    assert b.org_id not in dump and foreign not in dump and "beto@example.com" not in dump
    for secret in ("HASH-SECRETO", "whsec_SECRETO", "HMAC-SECRETO"):
        assert secret not in dump


def test_delete_org_removes_all_tenant_data(two_orgs):
    a, b = two_orgs
    _trace(a)
    a.create_user("ana@example.com", "x", "owner")
    assert a.delete_org()
    assert a.get_org() is None
    assert a.db.query("SELECT * FROM traces WHERE org_id = ?", (a.org_id,)) == []
    assert a.db.query("SELECT * FROM trace_tokens WHERE org_id = ?", (a.org_id,)) == []
    assert b.get_org() is not None


# --- revisión -------------------------------------------------------------------------------------


def test_review_flow_with_two_reviewers(two_orgs):
    a, _ = two_orgs
    r1 = a.create_user("r1@example.com", "x", "reviewer")
    r2 = a.create_user("r2@example.com", "x", "reviewer")
    item = a.create_review_item(_trace(a), min_reviewers=2)
    now = utcnow()
    assert a.reserve_review_item(item, r1, until=now + timedelta(minutes=5), now=now)
    assert not a.reserve_review_item(item, r2, until=now + timedelta(minutes=5), now=now)
    assert a.add_review_decision(item, r2, "error", now=now) is None  # reservado por r1
    assert a.add_review_decision(item, r1, "error", now=now)
    assert a.get_review_item(item)["state"] == "needs_other"
    assert a.add_review_decision(item, r1, "correct") is None  # ya decidió
    assert not a.reserve_review_item(item, r1, until=now + timedelta(minutes=5), now=now)
    assert a.add_review_decision(item, r2, "correct")
    done = a.get_review_item(item)
    assert done["state"] == "completed" and done["completed_at"]
    assert [d["label"] for d in a.list_review_decisions(item)] == ["error", "correct"]
    assert [i["id"] for i in a.list_review_items(state="completed")] == [item]
    with pytest.raises(ValueError):
        a.add_review_decision(item, r1, "quizás")


def test_expired_reservation_can_be_taken(two_orgs):
    a, _ = two_orgs
    r1 = a.create_user("r1@example.com", "x", "reviewer")
    r2 = a.create_user("r2@example.com", "x", "reviewer")
    item = a.create_review_item(_trace(a))
    now = utcnow()
    assert a.reserve_review_item(item, r1, until=now + timedelta(minutes=1), now=now)
    later = now + timedelta(minutes=2)
    assert a.reserve_review_item(item, r2, until=later + timedelta(minutes=1), now=later)


# --- detectores, datasets y webhooks --------------------------------------------------------------


def test_activate_detector_retires_previous(two_orgs):
    a, _ = two_orgs
    a.insert_detector("det-0000000001", '{"v":1}')
    a.insert_detector("det-0000000002", '{"v":2}', report_json='{"auroc":0.8}')
    assert a.get_active_detector() is None
    assert a.activate_detector("det-0000000001")
    assert a.get_active_detector()["id"] == "det-0000000001"
    assert a.activate_detector("det-0000000002")
    statuses = {d["id"]: d["status"] for d in a.list_detectors()}
    assert statuses == {"det-0000000001": "retired", "det-0000000002": "active"}
    assert a.get_org()["active_detector_id"] == "det-0000000002"
    assert "artifact_json" not in a.list_detectors()[0]
    assert a.set_detector_report("det-0000000002", '{"auroc":0.9}')
    with pytest.raises(ValueError):
        a.insert_detector("det-0000000001", "{}")


def test_dataset_and_webhook_crud(two_orgs):
    a, _ = two_orgs
    ds = a.insert_dataset("calib", b"{}\n{}\n", n_items=2)
    row = a.get_dataset(ds)
    assert row["n_items"] == 2 and len(row["sha256"]) == 64
    assert "content" not in a.list_datasets()[0]
    wh = a.insert_webhook("https://example.com/h", "whsec_abc", ["trace.flagged", "review.completed"])
    assert "secret" not in a.list_webhooks()[0]
    assert a.list_webhooks(event="review.completed", with_secret=True)[0]["secret"] == "whsec_abc"  # noqa: S105
    assert a.list_webhooks(event="calibration.activated") == []
    assert a.set_webhook_disabled(wh, True)
    assert a.list_webhooks(enabled_only=True) == []
    assert a.delete_webhook(wh) and a.delete_dataset(ds)


def test_api_key_rows_hide_hmac_and_revoke_once(two_orgs):
    a, _ = two_orgs
    a.insert_api_key("0123456789abcdef", name="k", hmac_hex="ff", pepper_kid="k1", scopes=["read", "ingest"])
    assert "hmac" not in a.get_api_key("0123456789abcdef")
    assert a.get_api_key("0123456789abcdef")["scopes"] == "read,ingest"
    assert a.revoke_api_key("0123456789abcdef")
    assert not a.revoke_api_key("0123456789abcdef")
    assert dbm.lookup_api_key(a.db, "0123456789abcdef")["hmac"] == "ff"
