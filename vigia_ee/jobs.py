"""Cola de jobs persistente sobre la tabla `jobs` de SQLite y workers en hilos (uno o más por carril).

Reglas de este módulo:

- La tabla `jobs` es infraestructura: no pasa por `TenantScopedRepository`, así que cada lectura
  filtra `org_id` a mano. `get(org_id, ...)` nunca devuelve un job de otro tenant ni uno de sistema
  (`org_id` NULL); los de sistema se leen solo con `get_system`.
- El claim es atómico: `BEGIN IMMEDIATE` toma el lock de escritura antes de elegir el job, así dos
  workers (o dos procesos) nunca reclaman el mismo.
- Un job en `running` guarda la hora del claim en `updated_at`. Si el proceso muere, el job queda
  huérfano; `recover_orphans()` lo devuelve a la cola cuando el lease venció, o lo marca `failed`
  si ya agotó sus intentos. Corre al arrancar y también cada `recover_interval` segundos dentro del
  worker: un huérfano reciente (lease vigente al reiniciar) vuelve a la cola apenas vence el lease.
  Nunca toca los jobs que este proceso está ejecutando.
- Carriles: cada tipo de job se registra en un carril (`DEFAULT_LANE` si no se indica) y cada carril
  tiene su propio hilo. Un carril lento (por ejemplo, webhooks a un destino colgado) no atrasa la
  purga ni la calibración. `depth(lane=...)` mide un carril solo, para la readiness.
- Dentro de un carril, `configure_lane` da varios hilos y un tope de jobs en curso por tenant. Un
  tenant lento ocupa a lo sumo su tope de hilos; el resto sigue en los hilos libres. El tope es por
  proceso y los jobs de sistema cuentan como un tenant más.
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
DEFAULT_LANE = "default"
MAX_LANE_WORKERS = 32
MAX_RECOVER_INTERVAL_S = 60.0
_SYSTEM_ORG = ""  # clave del tope por tenant para los jobs de sistema (org_id NULL; nunca '')

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
    lane: str


def _check_kind(kind: str) -> str:
    if not isinstance(kind, str) or not _KIND_RE.match(kind):
        raise ValueError("tipo de job inválido: usar minúsculas, dígitos y _.- (máximo 32)")
    return kind


def format_error(exc: BaseException) -> str:
    """`Tipo: mensaje` en una línea, recortado. Nunca incluye el stack trace."""
    msg = " ".join(str(exc).split())
    text = f"{type(exc).__name__}: {msg}" if msg else type(exc).__name__
    return text[:MAX_ERROR_CHARS]


def _worker_key(lane: str, index: int) -> str:
    """Clave del hilo `index` de un carril. El primero usa el nombre del carril ('#' no es válido)."""
    return lane if index == 0 else f"{lane}#{index}"


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
    está vacía. `recover_interval` (atributo, segundos) es cada cuánto el worker busca huérfanos.
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
        self.recover_interval = max(
            self.poll_interval, min(lease.total_seconds() / 4, MAX_RECOVER_INTERVAL_S)
        )
        self._handlers: dict[str, _Registration] = {}
        self._handlers_lock = threading.Lock()
        self._threads: dict[str, threading.Thread] = {}  # clave de hilo (_worker_key) -> hilo
        self._threads_lock = threading.Lock()
        self._lane_workers: dict[str, int] = {}  # carril -> hilos; 1 si no se configuró
        self._started = False
        self._stop = threading.Event()
        self._state_lock = threading.Lock()
        self._beats: dict[str, float] = {}  # clave de hilo -> time.monotonic() del último latido
        self._job_started: dict[str | None, float] = {}  # clave de hilo -> monotonic del job en curso
        # Ids que este proceso ejecuta -> (carril, tenant) si cuenta para el tope por tenant.
        self._inflight: dict[str, tuple[str, str] | None] = {}
        self._org_limits: dict[str, int] = {}  # carril -> jobs en curso por tenant
        self._running_orgs: dict[str, dict[str, int]] = {}  # carril -> tenant -> jobs en curso
        self._next_recover: float | None = None

    @property
    def last_beat(self) -> float | None:
        """`time.monotonic()` del último latido del carril por defecto."""
        with self._state_lock:
            return self._beats.get(DEFAULT_LANE)

    @last_beat.setter
    def last_beat(self, value: float | None) -> None:
        with self._state_lock:
            if value is None:
                self._beats.pop(DEFAULT_LANE, None)
            else:
                self._beats[DEFAULT_LANE] = value

    # -- reloj --

    def now(self) -> datetime:
        return self._clock()

    # -- registro --

    def register(
        self,
        kind: str,
        handler: Handler,
        *,
        retry_schedule: Sequence[float] | None = None,
        lane: str = DEFAULT_LANE,
    ) -> None:
        """Asocia un handler a un tipo de job.

        `handler(job)` recibe el dict del job (con `payload` ya parseado) y devuelve un dict de
        resultado JSON-serializable o None. Con `retry_schedule`, el retraso tras el fallo n es
        `retry_schedule[n-1]` (el último valor se repite); sin él, backoff exponencial con jitter.
        `lane` elige el hilo que lo ejecuta; un carril nuevo arranca su hilo si el worker ya corre.
        """
        _check_kind(kind)
        _check_kind(lane)
        if not callable(handler):
            raise TypeError("el handler debe ser invocable")
        schedule = None
        if retry_schedule is not None:
            schedule = tuple(float(s) for s in retry_schedule)
            if not schedule or any(s < 0 for s in schedule):
                raise ValueError("retry_schedule debe tener retrasos no negativos")
        with self._handlers_lock:
            self._handlers[kind] = _Registration(handler, schedule, lane)
        with self._threads_lock:
            if self._started and not self._stop.is_set():
                self._spawn_locked(lane)

    def registered_kinds(self, lane: str | None = None) -> tuple[str, ...]:
        """Tipos registrados, de todos los carriles o de uno."""
        with self._handlers_lock:
            return tuple(sorted(k for k, r in self._handlers.items() if lane is None or r.lane == lane))

    def lanes(self) -> tuple[str, ...]:
        """Carriles con hilo propio: siempre el por defecto, más los de los tipos registrados."""
        with self._handlers_lock:
            return tuple(sorted({DEFAULT_LANE, *(r.lane for r in self._handlers.values())}))

    def configure_lane(self, lane: str, *, workers: int = 1, max_running_per_org: int | None = None) -> None:
        """Fija los hilos de un carril y el tope de jobs en curso por tenant dentro del carril.

        Con `workers > max_running_per_org`, un tenant con jobs lentos nunca toma todos los hilos:
        los jobs de los otros tenants corren en los hilos libres. `None` quita el tope. Si el
        worker ya corre, arranca los hilos que falten; los sobrantes terminan solos.
        """
        _check_kind(lane)
        if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= MAX_LANE_WORKERS:
            raise ValueError(f"workers debe estar entre 1 y {MAX_LANE_WORKERS}")
        if max_running_per_org is not None and (
            isinstance(max_running_per_org, bool)
            or not isinstance(max_running_per_org, int)
            or max_running_per_org < 1
        ):
            raise ValueError("max_running_per_org debe ser un entero positivo o None")
        with self._state_lock:
            if max_running_per_org is None:
                self._org_limits.pop(lane, None)
            else:
                self._org_limits[lane] = max_running_per_org
        with self._threads_lock:
            self._lane_workers[lane] = workers
            if self._started and not self._stop.is_set() and lane in self.lanes():
                self._spawn_locked(lane)

    def _lane_filter(self, lane: str | None) -> tuple[str, list[Any]] | None:
        """Condición SQL sobre `kind` para un carril. None si el carril no puede tener jobs.

        El carril por defecto incluye además los tipos sin registrar (cuentan para la profundidad).
        """
        if lane is None:
            return "1", []
        if lane == DEFAULT_LANE:
            with self._handlers_lock:
                others = sorted(k for k, r in self._handlers.items() if r.lane != DEFAULT_LANE)
            if not others:
                return "1", []
            return f"kind NOT IN ({', '.join('?' for _ in others)})", list(others)
        mine = self.registered_kinds(lane)
        if not mine:
            return None
        return f"kind IN ({', '.join('?' for _ in mine)})", list(mine)

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

    def depth(self, *, include_scheduled: bool = False, lane: str | None = None) -> int:
        """Jobs pendientes: en cola y vencidos (o también programados a futuro) más los que corren.

        Sin `lane` cuenta todos los carriles (la métrica). Con `lane` cuenta solo ese carril: la
        readiness usa `DEFAULT_LANE` para que la cola de webhooks de un tenant no saque el nodo.
        """
        lane_filter = self._lane_filter(lane)
        if lane_filter is None:
            return 0
        lane_sql, params = lane_filter
        if include_scheduled:
            due = "state IN ('queued', 'running')"
        else:
            due = "(state = 'running' OR (state = 'queued' AND run_after <= ?))"
            params = [to_iso(self.now()), *params]
        sql = f"SELECT COUNT(*) AS n FROM jobs WHERE {due} AND {lane_sql}"  # noqa: S608 - solo '?'
        row = self.db.query_one(sql, tuple(params))
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
        """Devuelve a la cola los jobs `running` con lease vencido. Devuelve cuántos tocó.

        Excluye los jobs que este proceso está ejecutando: un job largo propio no es huérfano.
        """
        now = self.now()
        cutoff = to_iso(now - self.lease)
        stamp = to_iso(now)
        with self._state_lock:
            busy = sorted(self._inflight)
        orphan = "state = 'running' AND updated_at < ?"
        if busy:
            orphan += f" AND id NOT IN ({', '.join('?' for _ in busy)})"
        with self.db.transaction():
            failed = self.db.execute(
                "UPDATE jobs SET state = 'failed', updated_at = ?, "  # noqa: S608 - solo '?'
                "error = 'LeaseExpired: el job quedó huérfano y agotó sus intentos' "
                f"WHERE {orphan} AND attempts >= max_attempts",
                (stamp, cutoff, *busy),
            )
            requeued = self.db.execute(
                "UPDATE jobs SET state = 'queued', run_after = ?, updated_at = ?, "  # noqa: S608
                f"error = 'LeaseExpired: el job quedó huérfano y vuelve a la cola' WHERE {orphan}",
                (stamp, stamp, cutoff, *busy),
            )
        if failed or requeued:
            log_event(_log, "jobs_orphans_recovered", level=logging.WARNING)
        return failed + requeued

    def _claim(self, lane: str | None = None) -> dict[str, Any] | None:
        kinds = self.registered_kinds(lane)
        if not kinds:
            return None
        now = to_iso(self.now())
        marks = ", ".join("?" for _ in kinds)
        with self.db.transaction():
            # Dentro de la transacción: dos hilos del carril nunca ven el mismo conteo por tenant.
            with self._state_lock:
                limit = self._org_limits.get(lane) if lane is not None else None
                running = self._running_orgs.get(lane, {}) if lane is not None else {}
                full_orgs = [] if limit is None else sorted(o for o, n in running.items() if n >= limit)
            org_sql = ""
            if full_orgs:
                org_sql = f" AND COALESCE(org_id, '') NOT IN ({', '.join('?' for _ in full_orgs)})"
            row = self.db.query_one(
                f"SELECT id FROM jobs WHERE state = 'queued' AND run_after <= ? AND kind IN ({marks})"  # noqa: S608
                f"{org_sql} ORDER BY run_after, rowid LIMIT 1",
                (now, *kinds, *full_orgs),
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
            if full is not None:  # dentro de la transacción: la recuperación nunca lo ve sin marcar
                with self._state_lock:
                    slot = None
                    if limit is not None and lane is not None:
                        slot = (lane, full["org_id"] or _SYSTEM_ORG)
                        counts = self._running_orgs.setdefault(lane, {})
                        counts[slot[1]] = counts.get(slot[1], 0) + 1
                    self._inflight[full["id"]] = slot
        return _row_to_job(full) if full else None

    def _release(self, job_id: str) -> None:
        """Saca el job de los que corren y descuenta su tenant del tope. Requiere `_state_lock`."""
        slot = self._inflight.pop(job_id, None)
        if slot is None:
            return
        lane, org = slot
        counts = self._running_orgs.get(lane, {})
        counts[org] = counts.get(org, 1) - 1
        if counts[org] <= 0:
            del counts[org]

    def run_once(self, lane: str | None = None) -> dict[str, Any] | None:
        """Reclama y ejecuta un job vencido (de un carril o de cualquiera).

        Devuelve el job final o None si no había ninguno.
        """
        return self._run_once(lane, lane)

    def _run_once(self, lane: str | None, worker: str | None) -> dict[str, Any] | None:
        """`run_once` desde el hilo `worker` (clave de su latido y de su job en curso)."""
        job = self._claim(lane)
        if job is None:
            return None
        with self._handlers_lock:
            reg = self._handlers.get(job["kind"])
        with self._state_lock:
            self._job_started[worker] = time.monotonic()
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
                self._job_started.pop(worker, None)
                self._release(job["id"])
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
            log_event(
                _log,
                "job_failed",
                level=logging.WARNING,
                org_id=job["org_id"],
                action=job["kind"],
                error_type=type(exc).__name__,
            )
            return
        delay = self.retry_delay(job["id"], job["attempts"], schedule)
        self.db.execute(
            "UPDATE jobs SET state = 'queued', updated_at = ?, run_after = ?, error = ? "
            "WHERE id = ? AND state = 'running'",
            (to_iso(now), to_iso(now + timedelta(seconds=delay)), error, job["id"]),
        )
        log_event(
            _log,
            "job_retry",
            org_id=job["org_id"],
            action=job["kind"],
            error_type=type(exc).__name__,
            retry_after=round(delay, 3),
        )

    def run_until_idle(self, max_jobs: int = 1000) -> int:
        """Ejecuta jobs vencidos hasta vaciar la cola (o `max_jobs`). Devuelve cuántos corrió."""
        count = 0
        while count < max_jobs and self.run_once() is not None:
            count += 1
        return count

    # -- worker --

    def _beat(self, key: str = DEFAULT_LANE) -> None:
        with self._state_lock:
            self._beats[key] = time.monotonic()

    def _maybe_recover(self) -> None:
        """Busca huérfanos si pasó `recover_interval` desde la última vez (entre todos los carriles)."""
        now = time.monotonic()
        with self._state_lock:
            if self._next_recover is not None and now < self._next_recover:
                return
            self._next_recover = now + self.recover_interval
        self.recover_orphans()

    def _loop(self, lane: str, index: int) -> None:
        key = _worker_key(lane, index)
        while not self._stop.is_set():
            if index >= self._lane_workers.get(lane, 1):  # el carril se achicó: este hilo sobra
                with self._threads_lock:
                    if self._threads.get(key) is threading.current_thread():
                        del self._threads[key]
                return
            self._beat(key)
            try:
                self._maybe_recover()
                ran = self._run_once(lane, key)
            except Exception as exc:  # error de base: reintenta en el próximo ciclo
                log_event(_log, "jobs_worker_error", level=logging.ERROR, error_type=type(exc).__name__)
                ran = None
            if ran is None:
                self._stop.wait(self.poll_interval)

    def _spawn_locked(self, lane: str) -> None:
        """Arranca los hilos del carril que no corren. Requiere `_threads_lock`."""
        base = "vigia-jobs" if lane == DEFAULT_LANE else f"vigia-jobs-{lane}"
        for index in range(self._lane_workers.get(lane, 1)):
            key = _worker_key(lane, index)
            current = self._threads.get(key)
            if current is not None and current.is_alive():
                continue
            self._beat(key)
            name = base if index == 0 else f"{base}-{index}"
            thread = threading.Thread(target=self._loop, args=(lane, index), name=name, daemon=True)
            self._threads[key] = thread
            thread.start()

    def start(self) -> None:
        """Recupera huérfanos y arranca los hilos daemon de cada carril. Idempotente."""
        with self._threads_lock:
            running = self._threads.get(DEFAULT_LANE)
            already = self._started and running is not None and running.is_alive()
        if not already:
            self.recover_orphans()
            with self._state_lock:
                self._next_recover = time.monotonic() + self.recover_interval
            self._stop.clear()
        with self._threads_lock:
            self._started = True
            for lane in self.lanes():
                self._spawn_locked(lane)

    def stop(self, timeout: float = 5.0) -> bool:
        """Pide a los hilos que terminen y espera hasta `timeout`. Devuelve True si terminaron."""
        self._stop.set()
        with self._threads_lock:
            self._started = False
            threads = dict(self._threads)
        deadline = time.monotonic() + timeout
        for thread in threads.values():
            thread.join(max(0.0, deadline - time.monotonic()))
        if any(t.is_alive() for t in threads.values()):
            return False
        with self._threads_lock:
            for key, thread in threads.items():
                if self._threads.get(key) is thread:
                    del self._threads[key]
        return True

    def is_alive(self, max_age: float | None = None) -> bool:
        """True si todos los hilos viven y laten. Para `/readyz`.

        Un job largo no late; cuenta como vivo mientras no supere el lease. `max_age` por defecto
        es `max(5 s, 3 × poll_interval)`.
        """
        with self._threads_lock:
            threads = dict(self._threads)
        if DEFAULT_LANE not in threads or not all(t.is_alive() for t in threads.values()):
            return False
        limit = max_age if max_age is not None else max(5.0, 3 * self.poll_interval)
        now = time.monotonic()
        lease_s = self.lease.total_seconds()
        with self._state_lock:
            beats, started = dict(self._beats), dict(self._job_started)
        for key in threads:
            beat, began = beats.get(key), started.get(key)
            if beat is not None and now - beat <= limit:
                continue
            if began is not None and now - began <= lease_s:
                continue
            return False
        return True
