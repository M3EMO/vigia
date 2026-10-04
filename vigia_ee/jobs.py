"""Cola de jobs persistente sobre la tabla `jobs` de SQLite y worker en un hilo.

Reglas de este módulo:

- La tabla `jobs` es infraestructura: no pasa por `TenantScopedRepository`, así que cada lectura
  filtra `org_id` a mano. `get(org_id, ...)` nunca devuelve un job de otro tenant ni uno de sistema
  (`org_id` NULL); los de sistema se leen solo con `get_system`.
- El claim es atómico: `BEGIN IMMEDIATE` toma el lock de escritura antes de elegir el job, así dos
  workers (o dos procesos) nunca reclaman el mismo.
- Un job en `running` guarda la hora del claim en `updated_at`. Si el proceso muere, el job queda
  huérfano; `recover_orphans()` (que corre al arrancar el worker) lo devuelve a la cola cuando el
  lease venció, o lo marca `failed` si ya agotó sus intentos.
- Ante una excepción del handler, el job vuelve a la cola con backoff exponencial y jitter
  determinista (mismo job y mismo intento → mismo retraso, así los tests son reproducibles). Tras
  `max_attempts`, queda `failed`. El error guardado es `Tipo: mensaje`, nunca el stack trace.
- `PermanentJobError` marca `failed` sin reintentar (por ejemplo, un destino de webhook bloqueado).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from vigia_ee.db import Database, new_id, to_iso, utcnow
from vigia_ee.security import log_event

JOB_STATES = ("queued", "running", "done", "failed")
MAX_PAYLOAD_BYTES = 256 * 1024
MAX_ERROR_CHARS = 500
MAX_LIST = 200
DEFAULT_LEASE = timedelta(minutes=15)
DEFAULT_BACKOFF_BASE = 5.0  # segundos tras el primer fallo
DEFAULT_BACKOFF_MAX = 3600.0
JITTER_FRACTION = 0.25

_KIND_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,31}$")

Handler = Callable[[dict[str, Any]], Mapping[str, Any] | None]
Clock = Callable[[], datetime]

_log = logging.getLogger("vigia.jobs")


class PermanentJobError(Exception):
    """El handler pide marcar el job como `failed` sin reintentar."""


@dataclass(frozen=True)
class _Registration:
    handler: Handler
    retry_schedule: tuple[float, ...] | None


def _check_kind(kind: str) -> str:
    if not isinstance(kind, str) or not _KIND_RE.match(kind):
        raise ValueError("tipo de job inválido: usar minúsculas, dígitos y _.- (máximo 32)")
    return kind


def format_error(exc: BaseException) -> str:
    """`Tipo: mensaje` en una línea, recortado. Nunca incluye el stack trace."""
    msg = " ".join(str(exc).split())
    text = f"{type(exc).__name__}: {msg}" if msg else type(exc).__name__
    return text[:MAX_ERROR_CHARS]


def _row_to_job(row: Mapping[str, Any]) -> dict[str, Any]:
    job = dict(row)
    job["payload"] = json.loads(job.pop("payload_json"))
    raw = job.pop("result_json")
    job["result"] = json.loads(raw) if raw is not None else None
    return job


class JobQueue:
    """Cola persistente con worker opcional en un hilo daemon.

    `clock` devuelve la hora UTC (inyectable en tests). `lease` es cuánto puede durar un job en
    `running` antes de considerarse huérfano. `poll_interval` es la espera del worker cuando la cola
    está vacía.
    """

    def __init__(
        self,
        db: Database,
        *,
        clock: Clock | None = None,
        lease: timedelta = DEFAULT_LEASE,
        poll_interval: float = 1.0,
        backoff_base: float = DEFAULT_BACKOFF_BASE,
        backoff_max: float = DEFAULT_BACKOFF_MAX,
    ) -> None:
        if poll_interval <= 0:
            raise ValueError("poll_interval debe ser positivo")
        if lease <= timedelta(0):
            raise ValueError("lease debe ser positivo")
        if backoff_base < 0 or backoff_max < backoff_base:
            raise ValueError("backoff inválido")
        self.db = db
        self._clock: Clock = clock or utcnow
        self.lease = lease
        self.poll_interval = float(poll_interval)
        self.backoff_base = float(backoff_base)
        self.backoff_max = float(backoff_max)
        self._handlers: dict[str, _Registration] = {}
        self._handlers_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._state_lock = threading.Lock()
        self.last_beat: float | None = None  # time.monotonic() del último latido del worker
        self._job_started: float | None = None  # monotonic del inicio del job en curso

    # -- reloj --

    def now(self) -> datetime:
        return self._clock()

    # -- registro --

    def register(
        self, kind: str, handler: Handler, *, retry_schedule: Sequence[float] | None = None
    ) -> None:
        """Asocia un handler a un tipo de job.

        `handler(job)` recibe el dict del job (con `payload` ya parseado) y devuelve un dict de
        resultado JSON-serializable o None. Con `retry_schedule`, el retraso tras el fallo n es
        `retry_schedule[n-1]` (el último valor se repite); sin él, backoff exponencial con jitter.
        """
        _check_kind(kind)
        if not callable(handler):
            raise TypeError("el handler debe ser invocable")
        schedule = None
        if retry_schedule is not None:
            schedule = tuple(float(s) for s in retry_schedule)
            if not schedule or any(s < 0 for s in schedule):
                raise ValueError("retry_schedule debe tener retrasos no negativos")
        with self._handlers_lock:
            self._handlers[kind] = _Registration(handler, schedule)

    def registered_kinds(self) -> tuple[str, ...]:
        with self._handlers_lock:
            return tuple(sorted(self._handlers))

    # -- encolado y lectura --

    def enqueue(
        self,
        kind: str,
        payload: Mapping[str, Any],
        *,
        org_id: str | None = None,
        run_after: datetime | None = None,
        max_attempts: int = 5,
    ) -> str:
        """Encola un job y devuelve su id. `org_id=None` es un job de sistema."""
        _check_kind(kind)
        if not isinstance(payload, Mapping):
            raise ValueError("el payload debe ser un objeto")
        if org_id is not None and (not isinstance(org_id, str) or not org_id):
            raise ValueError("org_id inválido")
        valid_attempts = isinstance(max_attempts, int) and not isinstance(max_attempts, bool)
        if not valid_attempts or not 1 <= max_attempts <= 100:
            raise ValueError("max_attempts debe estar entre 1 y 100")
        try:
            payload_json = json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("el payload debe ser JSON serializable") from exc
        if len(payload_json.encode()) > MAX_PAYLOAD_BYTES:
            raise ValueError(f"el payload supera {MAX_PAYLOAD_BYTES} bytes")
        now = to_iso(self.now())
        job_id = new_id()
        self.db.execute(
            "INSERT INTO jobs (id, org_id, kind, payload_json, state, attempts, max_attempts, run_after, "
            "created_at, updated_at) VALUES (?, ?, ?, ?, 'queued', 0, ?, ?, ?, ?)",
            (
                job_id,
                org_id,
                kind,
                payload_json,
                max_attempts,
                to_iso(run_after) if run_after is not None else now,
                now,
                now,
            ),
        )
        return job_id

    def get(self, org_id: str, job_id: str) -> dict[str, Any] | None:
        """Job del tenant `org_id`. Otro tenant o un job de sistema → None (la API responde 404)."""
        if not isinstance(org_id, str) or not org_id:
            raise ValueError("org_id obligatorio")
        row = self.db.query_one("SELECT * FROM jobs WHERE id = ? AND org_id = ?", (job_id, org_id))
        return _row_to_job(row) if row else None

    def get_system(self, job_id: str) -> dict[str, Any] | None:
        """Job de sistema (`org_id` NULL). Nunca devuelve jobs de un tenant."""
        row = self.db.query_one("SELECT * FROM jobs WHERE id = ? AND org_id IS NULL", (job_id,))
        return _row_to_job(row) if row else None

    def list(self, org_id: str, kind: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        """Jobs del tenant, más nuevos primero."""
        if not isinstance(org_id, str) or not org_id:
            raise ValueError("org_id obligatorio")
        limit = max(1, min(int(limit), MAX_LIST))
        sql = "SELECT * FROM jobs WHERE org_id = ?"
        params: list[Any] = [org_id]
        if kind is not None:
            sql += " AND kind = ?"
            params.append(_check_kind(kind))
        rows = self.db.query(sql + " ORDER BY created_at DESC, rowid DESC LIMIT ?", (*params, limit))
        return [_row_to_job(r) for r in rows]

    def list_system(
        self, kind: str | None = None, *, states: Sequence[str] = JOB_STATES
    ) -> list[dict[str, Any]]:
        """Jobs de sistema, opcionalmente por tipo y estado (para la purga recurrente)."""
        bad = set(states) - set(JOB_STATES)
        if bad:
            raise ValueError(f"estados inválidos: {sorted(bad)}")
        marks = ", ".join("?" for _ in states)
        sql = f"SELECT * FROM jobs WHERE org_id IS NULL AND state IN ({marks})"  # noqa: S608 - solo '?'
        params: list[Any] = list(states)
        if kind is not None:
            sql += " AND kind = ?"
            params.append(_check_kind(kind))
        rows = self.db.query(sql + " ORDER BY run_after, rowid LIMIT ?", (*params, MAX_LIST))
        return [_row_to_job(r) for r in rows]

    def depth(self, *, include_scheduled: bool = False) -> int:
        """Jobs pendientes: en cola y vencidos (o también programados a futuro) más los que corren."""
        if include_scheduled:
            row = self.db.query_one("SELECT COUNT(*) AS n FROM jobs WHERE state IN ('queued', 'running')")
        else:
            row = self.db.query_one(
                "SELECT COUNT(*) AS n FROM jobs WHERE state = 'running' "
                "OR (state = 'queued' AND run_after <= ?)",
                (to_iso(self.now()),),
            )
        return int(row["n"]) if row else 0

    def purge_finished(self, before: datetime, *, org_id: str | None) -> int:
        """Borra jobs `done`/`failed` actualizados antes de `before` de un tenant (o de sistema)."""
        return self.db.execute(
            "DELETE FROM jobs WHERE org_id IS ? AND state IN ('done', 'failed') AND updated_at < ?",
            (org_id, to_iso(before)),
        )

    # -- ejecución --

    def retry_delay(self, job_id: str, attempts: int, schedule: tuple[float, ...] | None = None) -> float:
        """Segundos hasta el próximo intento tras el fallo número `attempts` (1 = primer fallo)."""
        n = max(1, attempts)
        if schedule:
            return schedule[min(n, len(schedule)) - 1]
        base = min(self.backoff_max, self.backoff_base * (2 ** (n - 1)))
        digest = hashlib.sha256(f"{job_id}:{n}".encode()).digest()
        jitter = int.from_bytes(digest[:8], "big") / 2**64 * JITTER_FRACTION
        return min(self.backoff_max, base * (1 + jitter))

    def recover_orphans(self) -> int:
        """Devuelve a la cola los jobs `running` con lease vencido. Devuelve cuántos tocó."""
        now = self.now()
        cutoff = to_iso(now - self.lease)
        stamp = to_iso(now)
        with self.db.transaction():
            failed = self.db.execute(
                "UPDATE jobs SET state = 'failed', updated_at = ?, "
                "error = 'LeaseExpired: el job quedó huérfano y agotó sus intentos' "
                "WHERE state = 'running' AND updated_at < ? AND attempts >= max_attempts",
                (stamp, cutoff),
            )
            requeued = self.db.execute(
                "UPDATE jobs SET state = 'queued', run_after = ?, updated_at = ?, "
                "error = 'LeaseExpired: el job quedó huérfano y vuelve a la cola' "
                "WHERE state = 'running' AND updated_at < ?",
                (stamp, stamp, cutoff),
            )
        if failed or requeued:
            log_event(_log, "jobs_orphans_recovered", level=logging.WARNING)
        return failed + requeued

    def _claim(self) -> dict[str, Any] | None:
        kinds = self.registered_kinds()
        if not kinds:
            return None
        now = to_iso(self.now())
        marks = ", ".join("?" for _ in kinds)
        with self.db.transaction():
            row = self.db.query_one(
                f"SELECT id FROM jobs WHERE state = 'queued' AND run_after <= ? AND kind IN ({marks}) "  # noqa: S608
                "ORDER BY run_after, rowid LIMIT 1",
                (now, *kinds),
            )
            if row is None:
                return None
            claimed = self.db.execute(
                "UPDATE jobs SET state = 'running', attempts = attempts + 1, updated_at = ? "
                "WHERE id = ? AND state = 'queued'",
                (now, row["id"]),
            )
            if claimed != 1:
                return None
            full = self.db.query_one("SELECT * FROM jobs WHERE id = ?", (row["id"],))
        return _row_to_job(full) if full else None

    def run_once(self) -> dict[str, Any] | None:
        """Reclama y ejecuta un job vencido. Devuelve el job final o None si no había ninguno."""
        job = self._claim()
        if job is None:
            return None
        with self._handlers_lock:
            reg = self._handlers.get(job["kind"])
        with self._state_lock:
            self._job_started = time.monotonic()
        try:
            if reg is None:  # desregistrado entre el claim y la ejecución
                raise PermanentJobError(f"no hay handler para {job['kind']!r}")
            result = reg.handler(dict(job))
            result_json = json.dumps(dict(result or {}), ensure_ascii=False, sort_keys=True, allow_nan=False)
        except Exception as exc:  # un handler nunca debe tumbar el worker
            self._fail(job, exc, reg.retry_schedule if reg else None)
        else:
            self.db.execute(
                "UPDATE jobs SET state = 'done', updated_at = ?, result_json = ?, error = NULL "
                "WHERE id = ? AND state = 'running'",
                (to_iso(self.now()), result_json, job["id"]),
            )
        finally:
            with self._state_lock:
                self._job_started = None
        row = self.db.query_one("SELECT * FROM jobs WHERE id = ?", (job["id"],))
        return _row_to_job(row) if row else None

    def _fail(self, job: Mapping[str, Any], exc: Exception, schedule: tuple[float, ...] | None) -> None:
        error = format_error(exc)
        now = self.now()
        permanent = isinstance(exc, PermanentJobError)
        if permanent or job["attempts"] >= job["max_attempts"]:
            self.db.execute(
                "UPDATE jobs SET state = 'failed', updated_at = ?, error = ? "
                "WHERE id = ? AND state = 'running'",
                (to_iso(now), error, job["id"]),
            )
            log_event(_log, "job_failed", level=logging.WARNING, org_id=job["org_id"],
                      action=job["kind"], error_type=type(exc).__name__)
            return
        delay = self.retry_delay(job["id"], job["attempts"], schedule)
        self.db.execute(
            "UPDATE jobs SET state = 'queued', updated_at = ?, run_after = ?, error = ? "
            "WHERE id = ? AND state = 'running'",
            (to_iso(now), to_iso(now + timedelta(seconds=delay)), error, job["id"]),
        )
        log_event(_log, "job_retry", org_id=job["org_id"], action=job["kind"],
                  error_type=type(exc).__name__, retry_after=round(delay, 3))

    def run_until_idle(self, max_jobs: int = 1000) -> int:
        """Ejecuta jobs vencidos hasta vaciar la cola (o `max_jobs`). Devuelve cuántos corrió."""
        count = 0
        while count < max_jobs and self.run_once() is not None:
            count += 1
        return count

    # -- worker --

    def _beat(self) -> None:
        with self._state_lock:
            self.last_beat = time.monotonic()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._beat()
            try:
                ran = self.run_once()
            except Exception as exc:  # error de base: reintenta en el próximo ciclo
                log_event(_log, "jobs_worker_error", level=logging.ERROR, error_type=type(exc).__name__)
                ran = None
            if ran is None:
                self._stop.wait(self.poll_interval)

    def start(self) -> None:
        """Recupera huérfanos y arranca el worker en un hilo daemon. Idempotente."""
        if self._thread is not None and self._thread.is_alive():
            return
        self.recover_orphans()
        self._stop.clear()
        self._beat()
        self._thread = threading.Thread(target=self._loop, name="vigia-jobs", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> bool:
        """Pide al worker que termine y espera hasta `timeout`. Devuelve True si terminó."""
        self._stop.set()
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout)
        if thread.is_alive():
            return False
        self._thread = None
        return True

    def is_alive(self, max_age: float | None = None) -> bool:
        """True si el hilo vive y late. Para `/readyz`.

        Un job largo no late; cuenta como vivo mientras no supere el lease. `max_age` por defecto
        es `max(5 s, 3 × poll_interval)`.
        """
        thread = self._thread
        if thread is None or not thread.is_alive():
            return False
        limit = max_age if max_age is not None else max(5.0, 3 * self.poll_interval)
        now = time.monotonic()
        with self._state_lock:
            beat, started = self.last_beat, self._job_started
        if beat is not None and now - beat <= limit:
            return True
        return started is not None and now - started <= self.lease.total_seconds()
