"""Tests de vigia_ee.retention: purga por tenant, job recurrente, supresión y export (AT-P10)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from vigia_ee import audit, retention
from vigia_ee.db import Database, TenantScopedRepository, create_org
from vigia_ee.jobs import JobQueue

ACTOR = {"user_id": "u-admin"}


class FakeClock:
    def __init__(self) -> None:
        self.t = datetime(2026, 10, 4, 3, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.t

    def advance(self, **kw: float) -> None:
        self.t += timedelta(**kw)


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "ret.db")
    yield database
    database.close()


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def jobs(db, clock):
    return JobQueue(db, clock=clock)


@pytest.fixture
def repos(db):
    a = create_org(db, "A", store_text="full", retention_days=7)
    b = create_org(db, "B", store_text="full", retention_days=30)
    return TenantScopedRepository(db, a), TenantScopedRepository(db, b)


def _trace(repo, created_at, text):
    return repo.insert_trace(
        model_id="tiny",
        response_text=text,
        action="annotate",
        prompt_text=f"pregunta {text}",
        tokens=[{"idx": 0, "text": text, "start": 0, "end": len(text), "p_error": 0.8, "flag": True}],
        spans=[{"start": 0, "end": len(text), "score": 0.8}],
        created_at=created_at,
    )


def _trace_ids(repo):
    rows, _ = repo.list_traces(limit=200)
    return {r["id"] for r in rows}


# --- purga ----------------------------------------------------------------------------------------


def test_at_p10_purge_org_removes_only_expired_traces_of_that_tenant(db, repos, clock):
    a, b = repos
    now = clock()
    a_old = _trace(a, now - timedelta(days=8), "viejo de A")
    a_new = _trace(a, now - timedelta(days=6), "nuevo de A")
    b_old = _trace(b, now - timedelta(days=8), "viejo de B")
    a.create_review_item(a_old)
    result = retention.purge_org(db, a.org_id, now=now)
    assert result["traces"] == 1
    assert _trace_ids(a) == {a_new}
    assert _trace_ids(b) == {b_old}
    assert a.get_trace(a_old) is None
    leftovers = db.query("SELECT COUNT(*) AS n FROM trace_tokens WHERE trace_id = ?", (a_old,))
    assert leftovers[0]["n"] == 0
    assert a.list_review_items() == []
    [event] = audit.export_range(db, a.org_id)
    assert event["action"] == "retention_purge"
    assert event["detail"]["traces"] == 1 and event["detail"]["retention_days"] == 7
    assert audit.export_range(db, b.org_id) == []


def test_at_p10_purge_all_applies_each_tenant_retention(db, repos, jobs, clock):
    a, b = repos
    now = clock()
    _trace(a, now - timedelta(days=10), "A 10 días")
    b_keep = _trace(b, now - timedelta(days=10), "B 10 días")
    _trace(b, now - timedelta(days=31), "B 31 días")
    totals = retention.purge_all(db, jobs, now=now)
    assert totals["orgs"] == 2 and totals["traces"] == 2
    assert _trace_ids(a) == set()
    assert _trace_ids(b) == {b_keep}
    assert audit.verify_chain(db, a.org_id).ok and audit.verify_chain(db, b.org_id).ok


def test_purge_also_removes_old_finished_jobs_of_the_tenant_only(db, repos, jobs, clock):
    a, b = repos
    jobs.register("calibrate", lambda job: {})
    old_a = jobs.enqueue("calibrate", {"t": 1}, org_id=a.org_id)
    old_b = jobs.enqueue("calibrate", {"t": 1}, org_id=b.org_id)
    old_sys = jobs.enqueue("calibrate", {})
    jobs.run_until_idle()
    queued_a = jobs.enqueue("webhook", {}, org_id=a.org_id)
    clock.advance(days=8)
    totals = retention.purge_all(db, jobs, now=clock())
    assert jobs.get(a.org_id, old_a) is None
    assert jobs.get(b.org_id, old_b) is not None  # B retiene 30 días
    assert jobs.get(a.org_id, queued_a)["state"] == "queued"
    assert jobs.get_system(old_sys) is not None  # los de sistema se guardan 30 días
    assert totals["jobs"] == 1
    clock.advance(days=31)
    retention.purge_all(db, jobs, now=clock())
    assert jobs.get_system(old_sys) is None


def test_purge_org_of_unknown_tenant_raises(db):
    with pytest.raises(LookupError):
        retention.purge_org(db, "no-existe-0000")


# --- job recurrente -------------------------------------------------------------------------------


def test_schedule_is_idempotent_and_reschedules_itself(db, repos, jobs, clock):
    a, _ = repos
    _trace(a, clock() - timedelta(days=8), "vencido")
    first = retention.schedule(jobs, every_hours=24)
    assert retention.schedule(jobs, every_hours=24) == first
    assert len(jobs.list_system("purge", states=("queued",))) == 1
    done = jobs.run_once()
    assert done["id"] == first and done["state"] == "done"
    assert done["result"]["traces"] == 1
    assert _trace_ids(a) == set()
    [nxt] = jobs.list_system("purge", states=("queued",))
    assert datetime.fromisoformat(nxt["run_after"]) == clock() + timedelta(hours=24)
    assert jobs.run_once() is None
    clock.advance(hours=24)
    assert jobs.run_once()["state"] == "done"
    assert len(jobs.list_system("purge", states=("queued",))) == 1


def test_a_failing_purge_keeps_the_cycle_alive(db, repos, jobs, clock, monkeypatch):
    retention.schedule(jobs, every_hours=12)

    def broken(*args, **kwargs):
        raise RuntimeError("disco lleno")

    monkeypatch.setattr(retention, "purge_all", broken)
    job = jobs.run_once()
    assert job["state"] == "queued" and "disco lleno" in job["error"]
    queued = jobs.list_system("purge", states=("queued",))
    assert len(queued) == 2  # el reintento y la corrida siguiente


def test_schedule_validates_interval(jobs):
    for bad in (0, -1, 24 * 31, True):
        with pytest.raises(ValueError):
            retention.schedule(jobs, every_hours=bad)


# --- supresión y export ---------------------------------------------------------------------------


def test_delete_trace_request_is_scoped_and_audited(db, repos, clock):
    a, b = repos
    tid = _trace(a, clock(), "borrame")
    assert not retention.delete_trace_request(b, tid, ACTOR)
    assert a.get_trace(tid) is not None
    assert audit.export_range(db, b.org_id) == []
    assert retention.delete_trace_request(a, tid, ACTOR)
    assert a.get_trace(tid) is None
    [event] = audit.export_range(db, a.org_id)
    assert event["action"] == "trace_delete" and event["target"] == tid
    assert event["actor_user_id"] == "u-admin"
    assert not retention.delete_trace_request(a, tid, ACTOR)


def test_at_p10_export_contains_everything_of_the_tenant_and_nothing_foreign(db, repos, clock):
    a, b = repos
    a_ids = {_trace(a, clock() - timedelta(days=i), f"texto propio {i}") for i in range(3)}
    b_id = _trace(b, clock(), "texto ajeno confidencial")
    secret = "whsec_" + "A" * 44
    a.insert_webhook("https://hooks.example.com/a", secret, ["trace.flagged"])
    data = retention.export_org(a, ACTOR)
    dumped = json.dumps(data, ensure_ascii=False)
    assert {t["id"] for t in data["traces"]} == a_ids
    assert all(t["tokens"] and t["spans"] for t in data["traces"])
    assert data["org_id"] == a.org_id and len(data["webhooks"]) == 1
    assert b_id not in dumped and "texto ajeno confidencial" not in dumped and b.org_id not in dumped
    assert secret not in dumped
    [event] = audit.export_range(db, a.org_id)
    assert event["action"] == "data_export" and event["detail"]["counts"]["traces"] == 3
    assert audit.export_range(db, b.org_id) == []
