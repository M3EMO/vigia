"""Tests de vigia_ee.jobs: aislamiento, claim atómico, backoff, huérfanos y worker."""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime, timedelta

import pytest

from vigia_ee.db import Database, TenantScopedRepository, create_org, new_id, to_iso
from vigia_ee.jobs import MAX_PAYLOAD_BYTES, JobQueue, PermanentJobError


class FakeClock:
    def __init__(self) -> None:
        self.t = datetime(2026, 10, 4, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.t

    def advance(self, **kw: float) -> None:
        self.t += timedelta(**kw)


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "jobs.db")
    yield database
    database.close()


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def jobs(db, clock):
    return JobQueue(db, clock=clock)


@pytest.fixture
def orgs(db):
    return create_org(db, "A"), create_org(db, "B")


# --- encolado y lectura aislada ---------------------------------------------------------------------


def test_get_is_isolated_by_org_and_system_jobs_only_via_get_system(jobs, orgs):
    a, b = orgs
    ja = jobs.enqueue("calibrate", {"x": 1}, org_id=a)
    js = jobs.enqueue("purge", {})
    assert jobs.get(a, ja)["payload"] == {"x": 1}
    assert jobs.get(b, ja) is None
    assert jobs.get_system(ja) is None
    assert jobs.get(a, js) is None
    assert jobs.get_system(js)["kind"] == "purge"
    with pytest.raises(ValueError):
        jobs.get(None, ja)


def test_list_filters_by_org_and_kind(jobs, orgs):
    a, b = orgs
    jobs.enqueue("calibrate", {}, org_id=a)
    jobs.enqueue("webhook", {}, org_id=a)
    jobs.enqueue("calibrate", {}, org_id=b)
    jobs.enqueue("purge", {})
    assert {j["kind"] for j in jobs.list(a)} == {"calibrate", "webhook"}
    assert [j["kind"] for j in jobs.list(a, kind="webhook")] == ["webhook"]
    assert len(jobs.list(b)) == 1
    assert len(jobs.list(a, limit=1)) == 1


@pytest.mark.parametrize(
    ("kind", "payload", "kwargs"),
    [
        ("Bad Kind", {}, {}),
        ("ok", {"v": float("nan")}, {}),
        ("ok", {"v": object()}, {}),
        ("ok", {}, {"max_attempts": 0}),
        ("ok", {}, {"max_attempts": True}),
        ("ok", {"big": "x" * (MAX_PAYLOAD_BYTES + 1)}, {}),
        ("ok", ["no", "dict"], {}),
    ],
)
def test_enqueue_rejects_invalid_input(jobs, kind, payload, kwargs):
    with pytest.raises(ValueError):
        jobs.enqueue(kind, payload, **kwargs)


# --- ejecución ------------------------------------------------------------------------------------


def test_run_once_stores_result_and_respects_run_after(jobs, clock, orgs):
    seen = []
    jobs.register("calibrate", lambda job: seen.append(job["payload"]) or {"ok": True})
    jid = jobs.enqueue("calibrate", {"n": 1}, org_id=orgs[0], run_after=clock() + timedelta(minutes=1))
    assert jobs.run_once() is None
    clock.advance(minutes=1)
    done = jobs.run_once()
    assert done["id"] == jid and done["state"] == "done"
    assert done["result"] == {"ok": True} and done["attempts"] == 1 and done["error"] is None
    assert seen == [{"n": 1}]
    assert jobs.run_once() is None


def test_unregistered_kinds_wait_in_queue(jobs):
    jid = jobs.enqueue("calibrate", {})
    jobs.register("purge", lambda job: None)
    assert jobs.run_once() is None
    assert jobs.get_system(jid)["state"] == "queued"


def test_failure_retries_with_deterministic_exponential_backoff(db, clock):
    jobs = JobQueue(db, clock=clock, backoff_base=10, backoff_max=10_000)

    def boom(job):
        raise ValueError("falló\nla segunda línea")

    jobs.register("calibrate", boom)
    jid = jobs.enqueue("calibrate", {}, max_attempts=3)
    first = jobs.run_once()
    assert first["state"] == "queued" and first["attempts"] == 1
    assert first["error"] == "ValueError: falló la segunda línea"
    assert "Traceback" not in first["error"]
    delay1 = (datetime.fromisoformat(first["run_after"]) - clock()).total_seconds()
    assert 10 <= delay1 <= 12.5
    assert delay1 == pytest.approx(jobs.retry_delay(jid, 1), abs=1e-5)
    twin = JobQueue(db, clock=clock, backoff_base=10, backoff_max=10_000)
    assert twin.retry_delay(jid, 1) == jobs.retry_delay(jid, 1)
    assert jobs.run_once() is None  # todavía no venció
    clock.advance(seconds=delay1)
    second = jobs.run_once()
    delay2 = (datetime.fromisoformat(second["run_after"]) - clock()).total_seconds()
    assert 20 <= delay2 <= 25
    clock.advance(seconds=delay2)
    third = jobs.run_once()
    assert third["state"] == "failed" and third["attempts"] == 3


def test_backoff_is_capped(db):
    jobs = JobQueue(db, backoff_base=5, backoff_max=60)
    assert jobs.retry_delay("x", 20) == 60


def test_retry_schedule_overrides_backoff(jobs, clock):
    def boom(job):
        raise RuntimeError("x")

    jobs.register("webhook", boom, retry_schedule=(7, 70))
    jobs.enqueue("webhook", {}, max_attempts=4)
    delays = []
    for _ in range(3):
        job = jobs.run_once()
        delay = (datetime.fromisoformat(job["run_after"]) - clock()).total_seconds()
        delays.append(delay)
        clock.advance(seconds=delay)
    assert delays == [7, 70, 70]
    assert jobs.run_once()["state"] == "failed"


def test_permanent_error_fails_without_retry(jobs):
    def nope(job):
        raise PermanentJobError("destino bloqueado")

    jobs.register("webhook", nope)
    jobs.enqueue("webhook", {})
    job = jobs.run_once()
    assert job["state"] == "failed" and job["attempts"] == 1
    assert job["error"] == "PermanentJobError: destino bloqueado"


def test_non_serializable_result_counts_as_failure(jobs):
    jobs.register("calibrate", lambda job: {"bad": object()})
    jobs.enqueue("calibrate", {}, max_attempts=1)
    assert jobs.run_once()["state"] == "failed"


def test_org_deleted_while_running_does_not_break_the_worker(db, jobs, orgs):
    a = orgs[0]
    jobs.register("calibrate", lambda job: TenantScopedRepository(db, a).delete_org() and {"ok": 1})
    jobs.enqueue("calibrate", {}, org_id=a)
    assert jobs.run_once() is None
    assert db.query("SELECT * FROM jobs") == []


# --- huérfanos ------------------------------------------------------------------------------------


def _insert_running(db, *, updated_at, attempts=1, max_attempts=5):
    stamp = to_iso(updated_at)
    db.execute(
        "INSERT INTO jobs (id, org_id, kind, payload_json, state, attempts, max_attempts, run_after, "
        "created_at, updated_at) VALUES (?, NULL, 'purge', '{}', 'running', ?, ?, ?, ?, ?)",
        (new_id(), attempts, max_attempts, stamp, stamp, stamp),
    )


def test_recover_orphans_requeues_expired_leases_only(db, clock):
    jobs = JobQueue(db, clock=clock, lease=timedelta(minutes=10))
    _insert_running(db, updated_at=clock() - timedelta(minutes=30))
    _insert_running(db, updated_at=clock() - timedelta(minutes=30), attempts=5, max_attempts=5)
    _insert_running(db, updated_at=clock() - timedelta(minutes=1), attempts=2)
    assert jobs.recover_orphans() == 2
    states = sorted((r["attempts"], r["state"]) for r in db.query("SELECT * FROM jobs"))
    assert states == [(1, "queued"), (2, "running"), (5, "failed")]
    queued = db.query_one("SELECT * FROM jobs WHERE state = 'queued'")
    assert queued["error"].startswith("LeaseExpired")
    jobs.register("purge", lambda job: {"ok": True})
    assert jobs.run_once()["state"] == "done"


# --- claim atómico entre conexiones ------------------------------------------------------------------


def test_claim_is_atomic_across_connections(tmp_path):
    path = tmp_path / "shared.db"
    seed = Database(path)
    producer = JobQueue(seed)
    ids = {producer.enqueue("calibrate", {"i": i}) for i in range(30)}
    executed: list[str] = []
    lock = threading.Lock()

    def handler(job):
        with lock:
            executed.append(job["id"])
        return {}

    conns = [Database(path) for _ in range(4)]
    queues = [JobQueue(c) for c in conns]
    for q in queues:
        q.register("calibrate", handler)
    threads = [threading.Thread(target=q.run_until_idle) for q in queues]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert sorted(executed) == sorted(ids)
    assert len(executed) == len(set(executed))
    for c in [*conns, seed]:
        c.close()


# --- profundidad y limpieza --------------------------------------------------------------------------


def test_depth_counts_due_and_running(jobs, clock, db):
    jobs.enqueue("calibrate", {})
    jobs.enqueue("calibrate", {}, run_after=clock() + timedelta(hours=1))
    _insert_running(db, updated_at=clock())
    assert jobs.depth() == 2
    assert jobs.depth(include_scheduled=True) == 3


def test_purge_finished_only_touches_old_finished_jobs_of_the_given_owner(jobs, clock, orgs):
    a, b = orgs
    jobs.register("calibrate", lambda job: {})
    old_a = jobs.enqueue("calibrate", {}, org_id=a)
    old_b = jobs.enqueue("calibrate", {}, org_id=b)
    jobs.run_until_idle()
    pending = jobs.enqueue("other", {}, org_id=a)
    clock.advance(days=10)
    assert jobs.purge_finished(clock() - timedelta(days=5), org_id=a) == 1
    assert jobs.get(a, old_a) is None
    assert jobs.get(b, old_b) is not None
    assert jobs.get(a, pending)["state"] == "queued"


# --- worker en hilo ----------------------------------------------------------------------------------


def _wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_worker_thread_runs_jobs_survives_errors_and_stops(db):
    jobs = JobQueue(db, poll_interval=0.01)
    assert not jobs.is_alive()

    def flaky(job):
        if job["payload"].get("fail"):
            raise PermanentJobError("no")
        return {"ok": True}

    jobs.register("calibrate", flaky)
    bad = jobs.enqueue("calibrate", {"fail": True})
    good = jobs.enqueue("calibrate", {"fail": False})
    jobs.start()
    jobs.start()  # idempotente
    try:
        assert _wait_for(lambda: jobs.get_system(good)["state"] == "done")
        assert jobs.get_system(bad)["state"] == "failed"
        assert jobs.is_alive()
        assert jobs.last_beat is not None
    finally:
        assert jobs.stop(timeout=5)
    assert not jobs.is_alive()


def test_is_alive_false_when_heartbeat_is_stale(db):
    jobs = JobQueue(db, poll_interval=0.01)
    jobs.start()
    try:
        assert jobs.is_alive()
        jobs._beat = lambda: None  # el hilo sigue vivo pero deja de latir
        jobs.last_beat = time.monotonic() - 60
        assert not jobs.is_alive(max_age=1.0)
    finally:
        assert jobs.stop()


def test_constructor_validates_parameters(db):
    with pytest.raises(ValueError):
        JobQueue(db, poll_interval=0)
    with pytest.raises(ValueError):
        JobQueue(db, lease=timedelta(0))
    with pytest.raises(ValueError):
        JobQueue(db, backoff_base=10, backoff_max=1)
