"""Tests de vigia_ee.webhooks: firma Standard Webhooks, emisión, entrega, reintentos y SSRF (AT-P11)."""

from __future__ import annotations

import base64
import json
import socket
import threading
import time
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from vigia.policy import UnsafeURLError
from vigia_ee import audit, webhooks
from vigia_ee import jobs as job_queue
from vigia_ee.db import Database, TenantScopedRepository, create_org
from vigia_ee.jobs import JobQueue
from vigia_ee.webhooks import (
    MAX_ATTEMPTS,
    RETRY_SCHEDULE_S,
    WebhookVerificationError,
    create_webhook,
    emit,
    generate_secret,
    register_delivery,
    sign,
    verify,
)

PUBLIC_IP = "93.184.216.34"
ACTOR = {"user_id": "u-owner"}


def public_dns(host: str) -> list[str]:
    return [PUBLIC_IP]


class FakeClock:
    def __init__(self) -> None:
        self.t = datetime(2026, 10, 4, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.t

    def advance(self, **kw: float) -> None:
        self.t += timedelta(**kw)


class Receiver:
    """Destino falso: responde la secuencia de códigos y guarda cada request."""

    def __init__(self, statuses=(200,), *, error: Exception | None = None) -> None:
        self.statuses = list(statuses)
        self.error = error
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        status = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
        headers = {"location": "https://otro.example/"} if 300 <= status < 400 else {}
        return httpx.Response(status, headers=headers, text="cuerpo que no se lee")


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "wh.db")
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


def _setup(jobs, receiver, **kw):
    client = httpx.Client(transport=httpx.MockTransport(receiver))
    register_delivery(jobs, http_client=client, resolve=public_dns, **kw)
    return client


def _hook(db, org, events=("trace.flagged",), url="https://hooks.example.com/vigia"):
    return create_webhook(db, org, url, events, ACTOR, resolve=public_dns)


# --- firma ----------------------------------------------------------------------------------------


def test_at_p11_signature_matches_standard_webhooks_reference_vector():
    secret = "whsec_MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw"  # noqa: S105 - vector público de la spec
    payload = b'{"test": 2432232314}'
    assert (
        sign(secret, "msg_p5jXN8AQM9LWM0D4loKWxJek", 1614265330, payload)
        == "v1,g0hM9SsE+OTPJTGt/tmIKtSyZlE3uFJELVlNIOLJ1OE="
    )


def test_at_p11_verify_accepts_valid_and_rejects_tampered_or_replayed():
    secret = generate_secret()
    body = b'{"type":"trace.flagged","data":{}}'
    sig = sign(secret, "msg_1", 1_000_000, body)
    assert verify(secret, "msg_1", "1000000", body, sig, now=1_000_100) == json.loads(body)
    with pytest.raises(WebhookVerificationError, match="firma"):
        verify(secret, "msg_1", 1_000_000, body.replace(b"{}", b"[]"), sig, now=1_000_000)
    with pytest.raises(WebhookVerificationError, match="firma"):
        verify(secret, "msg_2", 1_000_000, body, sig, now=1_000_000)
    with pytest.raises(WebhookVerificationError, match="tolerancia"):
        verify(secret, "msg_1", 1_000_000, body, sig, now=1_000_000 + 6 * 60)
    with pytest.raises(WebhookVerificationError, match="tolerancia"):
        verify(secret, "msg_1", 1_000_000, body, sig, now=1_000_000 - 6 * 60)
    with pytest.raises(WebhookVerificationError):
        verify(secret, "msg_1", "no-es-numero", body, sig, now=1_000_000)
    with pytest.raises(WebhookVerificationError):
        verify(generate_secret(), "msg_1", 1_000_000, body, sig, now=1_000_000)


def test_verify_accepts_any_of_several_signatures_for_secret_rotation():
    old, new = generate_secret(), generate_secret()
    body = b"{}"
    header = f"{sign(old, 'm', 5, body)} {sign(new, 'm', 5, body)}"
    assert verify(new, "m", 5, body, header, now=5) == {}
    assert verify(old, "m", 5, body, header, now=5) == {}


def test_generate_secret_format_and_sign_input_validation():
    secret = generate_secret()
    assert secret.startswith("whsec_")
    assert len(base64.b64decode(secret[6:])) == 32
    assert secret != generate_secret()
    with pytest.raises(ValueError):
        sign("sin-prefijo", "m", 1, b"{}")
    with pytest.raises(ValueError):
        sign("whsec_%%%", "m", 1, b"{}")
    with pytest.raises(ValueError):
        sign(secret, "con.punto", 1, b"{}")
    with pytest.raises(ValueError):
        sign(secret, "m", "1", b"{}")


# --- alta con validación SSRF ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "https://localhost/hook",
        "https://127.0.0.1/hook",
        "https://169.254.169.254/latest/meta-data",
        "https://10.0.0.5/hook",
        "https://[::1]/hook",
        "http://hooks.example.com/hook",
        "ftp://hooks.example.com/hook",
        "https://user:pass@hooks.example.com/hook",
    ],
)
def test_at_p11_create_blocks_unsafe_destinations(db, orgs, url):
    with pytest.raises(UnsafeURLError):
        create_webhook(db, orgs[0], url, ["trace.flagged"], ACTOR, resolve=public_dns)
    assert TenantScopedRepository(db, orgs[0]).list_webhooks() == []


def test_create_blocks_hostname_resolving_to_private_ip(db, orgs):
    with pytest.raises(UnsafeURLError):
        create_webhook(
            db,
            orgs[0],
            "https://interno.example.com/h",
            ["trace.flagged"],
            ACTOR,
            resolve=lambda h: ["10.1.2.3"],
        )


def test_create_accepts_public_https_and_allowlisted_http(db, orgs):
    wid, secret = _hook(db, orgs[0])
    hook = TenantScopedRepository(db, orgs[0]).get_webhook(wid)
    assert hook["secret"] == secret and hook["events"] == "trace.flagged"
    wid2, _ = create_webhook(
        db,
        orgs[0],
        "http://receptor.interno:8080/h",
        ["review.completed"],
        ACTOR,
        allowed_hosts=["receptor.interno"],
    )
    assert TenantScopedRepository(db, orgs[0]).get_webhook(wid2) is not None
    events = audit.export_range(db, orgs[0])
    assert [e["action"] for e in events] == ["webhook_create", "webhook_create"]
    assert secret not in json.dumps(events)


def test_create_rejects_unknown_or_empty_events(db, orgs):
    with pytest.raises(ValueError):
        create_webhook(db, orgs[0], "https://h.example.com", ["trace.inventado"], ACTOR, resolve=public_dns)
    with pytest.raises(ValueError):
        create_webhook(db, orgs[0], "https://h.example.com", [], ACTOR, resolve=public_dns)


def test_delete_and_enable_are_scoped_and_audited(db, orgs):
    a, b = orgs
    wid, _ = _hook(db, a)
    assert not webhooks.delete_webhook(db, b, wid, ACTOR)
    assert not webhooks.set_enabled(db, b, wid, False, ACTOR)
    assert webhooks.set_enabled(db, a, wid, False, ACTOR)
    assert TenantScopedRepository(db, a).get_webhook(wid)["disabled"] == 1
    assert webhooks.set_enabled(db, a, wid, True, ACTOR, resolve=public_dns)
    assert webhooks.delete_webhook(db, a, wid, ACTOR)
    actions = [e["action"] for e in audit.export_range(db, a)]
    assert actions == ["webhook_create", "webhook_disable", "webhook_enable", "webhook_delete"]
    assert audit.export_range(db, b) == []


# --- emisión --------------------------------------------------------------------------------------


def test_emit_enqueues_one_job_per_subscribed_enabled_webhook_of_the_tenant(db, jobs, orgs):
    a, b = orgs
    w1, _ = _hook(db, a)
    w2, _ = _hook(db, a, events=("trace.flagged", "review.completed"))
    _hook(db, a, events=("review.completed",))
    w4, _ = _hook(db, a)
    TenantScopedRepository(db, a).set_webhook_disabled(w4, True)
    _hook(db, b)
    ids = emit(db, jobs, a, "trace.flagged", {"trace_id": "tr_12345678", "action": "annotate"})
    payloads = [jobs.get(a, i)["payload"] for i in ids]
    assert sorted(p["webhook_id"] for p in payloads) == sorted([w1, w2])
    assert len({p["msg_id"] for p in payloads}) == 1
    body = json.loads(payloads[0]["body"])
    assert body == {
        "type": "trace.flagged",
        "timestamp": "2026-10-04T12:00:00.000000Z",
        "data": {"trace_id": "tr_12345678", "action": "annotate"},
    }
    assert all(jobs.get(b, i) is None for i in ids)
    with pytest.raises(ValueError):
        emit(db, jobs, a, "evento.raro", {})


def test_emit_sends_reference_when_payload_exceeds_20kb(db, jobs, orgs):
    _hook(db, orgs[0])
    [jid] = emit(db, jobs, orgs[0], "trace.flagged", {"trace_id": "tr_12345678", "text": "x" * 30_000})
    body = jobs.get(orgs[0], jid)["payload"]["body"]
    assert len(body.encode()) < 20_000
    assert json.loads(body)["data"] == {"trace_id": "tr_12345678", "truncated": True}


def test_emit_without_subscribers_enqueues_nothing(db, jobs, orgs):
    assert emit(db, jobs, orgs[0], "trace.flagged", {}) == []


# --- entrega --------------------------------------------------------------------------------------


def test_at_p11_delivery_is_signed_and_verifiable(db, jobs, orgs, clock):
    receiver = Receiver([200])
    _setup(jobs, receiver)
    wid, secret = _hook(db, orgs[0])
    emit(db, jobs, orgs[0], "trace.flagged", {"trace_id": "tr_12345678"})
    job = jobs.run_once()
    assert job["state"] == "done" and job["result"]["status"] == 200
    [req] = receiver.requests
    assert req.method == "POST" and str(req.url) == "https://hooks.example.com/vigia"
    assert req.headers["content-type"] == "application/json"
    assert req.headers["webhook-timestamp"] == str(int(clock().timestamp()))
    parsed = verify(
        secret,
        req.headers["webhook-id"],
        req.headers["webhook-timestamp"],
        req.content,
        req.headers["webhook-signature"],
        now=clock().timestamp(),
    )
    assert parsed["data"] == {"trace_id": "tr_12345678"}
    assert req.extensions["timeout"]["read"] == 15.0


def test_at_p11_retries_on_500_with_spec_schedule_and_same_webhook_id(db, jobs, orgs, clock):
    receiver = Receiver([500, 500, 200])
    _setup(jobs, receiver)
    _hook(db, orgs[0])
    [jid] = emit(db, jobs, orgs[0], "trace.flagged", {"trace_id": "tr_12345678"})
    first = jobs.run_once()
    assert first["state"] == "queued" and "HTTP 500" in first["error"]
    assert datetime.fromisoformat(first["run_after"]) == clock() + timedelta(seconds=5)
    assert jobs.run_once() is None
    clock.advance(seconds=5)
    second = jobs.run_once()
    assert datetime.fromisoformat(second["run_after"]) == clock() + timedelta(minutes=5)
    clock.advance(minutes=5)
    third = jobs.run_once()
    assert third["state"] == "done" and third["attempts"] == 3
    ids = {r.headers["webhook-id"] for r in receiver.requests}
    stamps = [int(r.headers["webhook-timestamp"]) for r in receiver.requests]
    assert len(receiver.requests) == 3 and len(ids) == 1
    assert stamps == sorted(stamps) and stamps[0] < stamps[-1]


def test_full_retry_schedule_then_failed(db, jobs, orgs, clock):
    receiver = Receiver([503])
    _setup(jobs, receiver)
    _hook(db, orgs[0])
    emit(db, jobs, orgs[0], "trace.flagged", {})
    delays = []
    while True:
        job = jobs.run_once()
        if job["state"] == "failed":
            break
        delay = (datetime.fromisoformat(job["run_after"]) - clock()).total_seconds()
        delays.append(delay)
        clock.advance(seconds=delay)
    assert tuple(delays) == RETRY_SCHEDULE_S
    assert len(receiver.requests) == MAX_ATTEMPTS
    assert RETRY_SCHEDULE_S == (5, 300, 1800, 7200, 18000, 36000, 36000)


def test_redirects_are_not_followed(db, jobs, orgs):
    receiver = Receiver([302])
    _setup(jobs, receiver)
    _hook(db, orgs[0])
    emit(db, jobs, orgs[0], "trace.flagged", {})
    job = jobs.run_once()
    assert job["state"] == "queued" and "HTTP 302" in job["error"]
    assert len(receiver.requests) == 1


def test_connection_error_retries_without_leaking_the_url(db, jobs, orgs):
    receiver = Receiver(error=httpx.ConnectError("no conecta a https://hooks.example.com/vigia?token=abc"))
    _setup(jobs, receiver)
    _hook(db, orgs[0], url="https://hooks.example.com/vigia?token=abc")
    emit(db, jobs, orgs[0], "trace.flagged", {})
    job = jobs.run_once()
    assert job["state"] == "queued"
    assert "ConnectError" in job["error"] and "token" not in job["error"]


@pytest.mark.parametrize(
    ("url", "dns"),
    [
        ("https://169.254.169.254/latest/meta-data", public_dns),
        ("https://localhost/hook", public_dns),
        ("https://rebind.example.com/hook", lambda h: ["127.0.0.1"]),
    ],
)
def test_at_p11_delivery_revalidates_destination_and_blocks_ssrf(db, clock, orgs, url, dns):
    jobs = JobQueue(db, clock=clock)
    receiver = Receiver([200])
    client = httpx.Client(transport=httpx.MockTransport(receiver))
    register_delivery(jobs, http_client=client, resolve=dns)
    # Inserción directa: simula un webhook viejo o un DNS que cambió después del alta.
    TenantScopedRepository(db, orgs[0]).insert_webhook(url, generate_secret(), ["trace.flagged"])
    emit(db, jobs, orgs[0], "trace.flagged", {})
    job = jobs.run_once()
    assert job["state"] == "failed" and "destino bloqueado" in job["error"]
    assert receiver.requests == []


def test_deleted_or_disabled_webhook_is_skipped(db, jobs, orgs):
    receiver = Receiver([200])
    _setup(jobs, receiver)
    w1, _ = _hook(db, orgs[0])
    w2, _ = _hook(db, orgs[0])
    emit(db, jobs, orgs[0], "trace.flagged", {})
    repo = TenantScopedRepository(db, orgs[0])
    repo.delete_webhook(w1)
    repo.set_webhook_disabled(w2, True)
    results = sorted(jobs.run_once()["result"]["skipped"] for _ in range(2))
    assert results == ["webhook_deleted", "webhook_disabled"]
    assert receiver.requests == []


def _fail_one_delivery(db, jobs, org, clock):
    emit(db, jobs, org, "trace.flagged", {})
    while True:
        job = jobs.run_once()
        if job is None:
            clock.advance(hours=11)
            continue
        if job["state"] in ("failed", "done"):
            return job


def test_webhook_is_disabled_and_audited_after_consecutive_failed_deliveries(db, jobs, orgs, clock):
    receiver = Receiver([500])
    _setup(jobs, receiver, max_consecutive_failures=2)
    wid, _ = _hook(db, orgs[0])
    repo = TenantScopedRepository(db, orgs[0])
    assert _fail_one_delivery(db, jobs, orgs[0], clock)["state"] == "failed"
    assert repo.get_webhook(wid)["disabled"] == 0
    assert _fail_one_delivery(db, jobs, orgs[0], clock)["state"] == "failed"
    assert repo.get_webhook(wid)["disabled"] == 1
    event = audit.export_range(db, orgs[0])[-1]
    assert event["action"] == "webhook_auto_disabled" and event["target"] == wid
    assert event["detail"]["consecutive_failures"] == 2
    assert audit.verify_chain(db, orgs[0]).ok
    assert emit(db, jobs, orgs[0], "trace.flagged", {}) == []


def test_a_success_resets_the_consecutive_failure_count(db, jobs, orgs, clock):
    receiver = Receiver([500])
    _setup(jobs, receiver, max_consecutive_failures=2)
    wid, _ = _hook(db, orgs[0])
    _fail_one_delivery(db, jobs, orgs[0], clock)
    receiver.statuses = [200]
    assert _fail_one_delivery(db, jobs, orgs[0], clock)["state"] == "done"
    receiver.statuses = [500]
    _fail_one_delivery(db, jobs, orgs[0], clock)
    assert TenantScopedRepository(db, orgs[0]).get_webhook(wid)["disabled"] == 0


# --- un webhook caído o lento no frena al resto -----------------------------------------------------


def test_broken_destination_is_disabled_after_n_failed_attempts_not_after_28_hours(db, jobs, orgs):
    """Regresión: 'El webhook caído de un solo tenant satura el único worker: atrasa la purga de
    los demás tenants y pone /readyz en rojo para toda la plataforma'. Cada intento fallido cuenta;
    tras N seguidos el webhook se desactiva y el resto de su cola se descarta sin tocar la red."""
    receiver = Receiver(error=httpx.ConnectError("rechazada"))
    _setup(jobs, receiver, max_consecutive_failures=5)
    wid, _ = _hook(db, orgs[0])
    for i in range(40):
        emit(db, jobs, orgs[0], "trace.flagged", {"trace_id": f"tr_{i}"})
    jobs.run_until_idle()
    assert len(receiver.requests) == 5, "solo N intentos llegan a la red"
    assert TenantScopedRepository(db, orgs[0]).get_webhook(wid)["disabled"] == 1
    assert jobs.depth() == 0, "la cola vencida drena entera"
    event = audit.export_range(db, orgs[0])[-1]
    assert event["action"] == "webhook_auto_disabled" and event["detail"]["consecutive_failures"] == 5


def test_slow_webhook_does_not_delay_purge_or_calibration_of_other_tenants(db, orgs):
    """Regresión: 'El webhook caído de un solo tenant satura el único worker: atrasa la purga de
    los demás tenants y pone /readyz en rojo para toda la plataforma'. Los webhooks corren en su
    propio carril: la purga y la calibración no esperan detrás de ellos."""
    gate = threading.Event()

    def slow(request: httpx.Request) -> httpx.Response:
        gate.wait(10)  # destino colgado hasta el final del test
        return httpx.Response(200)

    jobs = JobQueue(db, poll_interval=0.01)
    _setup(jobs, slow)
    _hook(db, orgs[0])
    for i in range(20):
        emit(db, jobs, orgs[0], "trace.flagged", {"trace_id": f"tr_{i}"})
    jobs.register("purge", lambda job: {"ok": True})
    jobs.register("calibrate", lambda job: {"ok": True})
    purge = jobs.enqueue("purge", {})
    calib = jobs.enqueue("calibrate", {}, org_id=orgs[1])
    jobs.start()
    try:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and jobs.get_system(purge)["state"] != "done":
            time.sleep(0.01)
        assert jobs.get_system(purge)["state"] == "done"
        assert jobs.get(orgs[1], calib)["state"] == "done"
        assert jobs.depth(lane=job_queue.DEFAULT_LANE) == 0, "la readiness no cuenta la cola de webhooks"
        assert jobs.depth() == 20, "la métrica sigue viendo todos los webhooks pendientes"
        assert jobs.is_alive(max_age=60)
    finally:
        gate.set()
        assert jobs.stop(timeout=10)


# --- aislamiento del carril entre tenants -----------------------------------------------------------


def _trickle_server(pads: int, interval: float):
    """Destino que responde 200 pero gotea un header cada `interval` s (cada lectura cabe en el
    timeout por lectura de httpx). Devuelve `(url, cerrar)`."""
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(8)
    stop = threading.Event()

    def handle(conn: socket.socket) -> None:
        with conn:
            data = b""
            while b"\r\n\r\n" not in data:
                chunk = conn.recv(65536)
                if not chunk:
                    return
                data += chunk
            try:
                conn.sendall(b"HTTP/1.1 200 OK\r\n")
                for i in range(pads):
                    if stop.wait(interval):
                        return
                    conn.sendall(f"X-Pad-{i}: a\r\n".encode())
                conn.sendall(b"Content-Length: 0\r\n\r\n")
            except OSError:
                return  # el cliente cortó por el plazo total

    def accept_loop() -> None:
        while not stop.is_set():
            try:
                conn, _ = server.accept()
            except OSError:
                return
            threading.Thread(target=handle, args=(conn,), daemon=True).start()

    threading.Thread(target=accept_loop, daemon=True).start()

    def close() -> None:
        stop.set()
        server.close()

    return f"http://127.0.0.1:{server.getsockname()[1]}/hook", close


def test_trickling_destination_is_cut_at_the_total_delivery_deadline(db, jobs, orgs):
    """Regresión: 'Carril de webhooks FIFO compartido y sin plazo total: el destino lento de un
    tenant frena los webhooks de los demás'. httpx acota cada lectura, no la entrega: un destino
    que gotea headers retenía el hilo sin límite y su 200 contaba como éxito. El plazo total corta
    la entrega en `timeout` segundos y la cuenta como fallo."""
    url, close = _trickle_server(pads=12, interval=0.3)  # 3,6 s goteando; cada lectura < 1 s
    try:
        register_delivery(jobs, allowed_hosts=("127.0.0.1",), timeout=1.0)
        create_webhook(db, orgs[0], url, ["trace.flagged"], ACTOR, allowed_hosts=("127.0.0.1",))
        emit(db, jobs, orgs[0], "trace.flagged", {"trace_id": "tr_1"})
        started = time.monotonic()
        job = jobs.run_once(lane=webhooks.JOB_KIND)
        elapsed = time.monotonic() - started
    finally:
        close()
    assert elapsed < 2.5, f"la entrega duró {elapsed:.1f} s con plazo total de 1 s"
    assert job["state"] == "queued" and job["attempts"] == 1, "superar el plazo es un fallo con reintento"
    assert "plazo" in job["error"]


def test_slow_tenant_webhooks_do_not_delay_webhooks_of_other_tenants(db, orgs):
    """Regresión: 'Carril de webhooks FIFO compartido y sin plazo total: el destino lento de un
    tenant frena los webhooks de los demás'. El claim era FIFO global con un solo hilo: el webhook
    de B esperaba detrás de toda la cola colgada de A."""
    gate = threading.Event()
    delivered: list[str] = []

    def receiver(request: httpx.Request) -> httpx.Response:
        if request.url.host == "hooks-a.example.com":
            gate.wait(10)  # destino de A colgado hasta el final del test
        else:
            delivered.append(request.url.host)
        return httpx.Response(200)

    jobs = JobQueue(db, poll_interval=0.01)
    _setup(jobs, receiver)
    _hook(db, orgs[0], url="https://hooks-a.example.com/vigia")
    _hook(db, orgs[1], url="https://hooks-b.example.com/vigia")
    for i in range(10):
        emit(db, jobs, orgs[0], "trace.flagged", {"trace_id": f"a_{i}"})
    (b_job,) = emit(db, jobs, orgs[1], "trace.flagged", {"trace_id": "b_0"})
    jobs.start()
    try:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and jobs.get(orgs[1], b_job)["state"] != "done":
            time.sleep(0.01)
        assert jobs.get(orgs[1], b_job)["state"] == "done", "el webhook de B no espera detrás de A"
        assert delivered == ["hooks-b.example.com"]
        states = [j["state"] for j in jobs.list(orgs[0])]
        assert states.count("running") == 1, "A ocupa a lo sumo un hilo del carril"
    finally:
        gate.set()
        assert jobs.stop(timeout=10)
