"""Acceso a datos de la plataforma Vigía sobre SQLite (stdlib).

Reglas de este módulo:

- `Database` es la única conexión. Aplica `schema.sql` como migración 1, activa `foreign_keys`,
  WAL y `busy_timeout`. `ponytail:` una conexión por proceso, serializada con un `RLock`. SQLite
  admite un solo escritor, así que el techo real es el mismo; con más de un nodo va Postgres + RLS.
- `TenantScopedRepository(db, org_id)` es la única puerta a datos de cliente. Toda consulta filtra
  por `org_id`. Un recurso de otro tenant devuelve `None` y la capa HTTP lo traduce a 404.
- Las funciones `lookup_*` no filtran por tenant: solo sirven para resolver el principal antes de
  conocer el tenant (login por email, API key por key_id, sesión por hash).
- Fechas en UTC con formato fijo `YYYY-MM-DDTHH:MM:SS.ffffffZ`: el orden de texto es el orden real.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
import secrets
import sqlite3
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vigia.contracts import SCHEMA_VERSION
from vigia_ee import pii

SCHEMA_PATH = Path(__file__).with_name("schema.sql")

# Migraciones versionadas. La 1 es el contrato `schema.sql` (idempotente, CREATE IF NOT EXISTS).
# Una migración nueva se agrega al final con el número siguiente; nunca se edita una aplicada.
# 2: `audit_log.org_seq`, la posición del evento dentro de su organización. El `seq` global queda
# interno (solo ordena): mostrarlo dejaba inferir cuántos eventos registraron los otros tenants.
# El backfill reabre un momento el trigger append-only y lo vuelve a crear dentro de la misma transacción.
_AUDIT_ORG_SEQ = """
ALTER TABLE audit_log ADD COLUMN org_seq INTEGER;
DROP TRIGGER IF EXISTS audit_log_no_update;
UPDATE audit_log SET org_seq = n.pos
FROM (SELECT seq, row_number() OVER (PARTITION BY org_id ORDER BY seq) AS pos FROM audit_log) AS n
WHERE n.seq = audit_log.seq;
CREATE TRIGGER audit_log_no_update BEFORE UPDATE ON audit_log
BEGIN SELECT RAISE(ABORT, 'audit_log es append-only'); END;
CREATE UNIQUE INDEX IF NOT EXISTS ux_audit_org_seq ON audit_log(org_id, org_seq);
"""
# 3: `traces.status` (VigiaResult.status) y `traces.rag_status` (None | retried | error | unavailable),
# para que `GET /v1/vigia/traces/{id}` devuelva lo mismo que la respuesta. Una traza previa queda con
# status 'ok' y rag_status NULL. Las columnas no van en schema.sql: la migración 1 corre en una base
# nueva y esta fallaría con 'duplicate column'.
_TRACE_STATUS = """
ALTER TABLE traces ADD COLUMN status TEXT NOT NULL DEFAULT 'ok';
ALTER TABLE traces ADD COLUMN rag_status TEXT;
"""
MIGRATIONS: tuple[tuple[int, str], ...] = (
    (1, SCHEMA_PATH.read_text(encoding="utf-8")),
    (2, _AUDIT_ORG_SEQ),
    (3, _TRACE_STATUS),
)
# Eventos de auditoría de una org tal como salen hacia el tenant: `seq` es la posición por organización.
AUDIT_SELECT = (
    "SELECT org_seq AS seq, org_id, ts, actor_user_id, actor_key_id, action, target, detail_json, "
    "prev_hash, hash FROM audit_log WHERE org_id IS ?"
)
AUDIT_ORDER = " ORDER BY audit_log.seq"

STORE_TEXT_MODES = ("none", "redacted", "full")
ROLES = ("owner", "admin", "reviewer", "viewer")
REVIEW_LABELS = ("correct", "error", "uncertain")
REVIEW_STATES = ("pending", "reserved", "needs_other", "completed")
DETECTOR_STATUSES = ("candidate", "active", "retired")
TRACE_SOURCES = ("api", "playground")
MAX_PAGE = 200

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
_ISO_FMT = "%Y-%m-%dT%H:%M:%S.%fZ"

# Espera de SQLite por intento ante otro proceso que retiene el lock de archivo. La espera total la
# fija `busy_timeout_ms` de `Database`; entre intentos el hilo suelta el `RLock` (ver `Database._call`).
BUSY_ATTEMPT_MS = 10
_BACKOFF_MIN_S = 0.02
_BACKOFF_MAX_S = 0.25
_BUSY_CODES = frozenset({5, 6})  # SQLITE_BUSY, SQLITE_LOCKED (código primario)


# --- utilidades -----------------------------------------------------------------------------------


def utcnow() -> datetime:
    return datetime.now(UTC)


def to_iso(dt: datetime) -> str:
    """Fecha UTC con ancho fijo; una fecha naive se interpreta como UTC."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).strftime(_ISO_FMT)


def parse_iso(value: str) -> datetime:
    return datetime.strptime(value, _ISO_FMT).replace(tzinfo=UTC)


def new_id(nbytes: int = 16) -> str:
    """ID aleatorio no secuencial (hex de `nbytes` bytes de un CSPRNG)."""
    return secrets.token_hex(nbytes)


def _check_id(value: str, what: str = "id") -> str:
    if not isinstance(value, str) or not _ID_RE.match(value):
        raise ValueError(f"{what} inválido")
    return value


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Lee un campo de un dict o de un objeto (dataclasses de contracts)."""
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _is_busy(exc: sqlite3.OperationalError) -> bool:
    """True si SQLite rechazó la operación porque otra conexión retiene el lock."""
    code = getattr(exc, "sqlite_errorcode", None)
    if isinstance(code, int):
        return (code & 0xFF) in _BUSY_CODES
    text = str(exc).lower()
    return "database is locked" in text or "database is busy" in text or "table is locked" in text


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


# --- conexión -------------------------------------------------------------------------------------


class Database:
    """Conexión SQLite con migraciones, WAL, claves foráneas y transacciones explícitas.

    `busy_timeout_ms` es la espera TOTAL de una operación ante otro proceso que retiene el lock de
    archivo. El `RLock` nunca se retiene durante esa espera: cada intento usa un `busy_timeout` corto
    (`BUSY_ATTEMPT_MS`) y, si SQLite sigue ocupado, el hilo suelta el `RLock`, espera y reintenta
    hasta el tope. Así cinco workers que reintentan no suman sus esperas a la de un request.
    """

    def __init__(self, path: str | Path, *, busy_timeout_ms: int = 5000) -> None:
        self.path = str(path)
        self._lock = threading.RLock()
        self._depth = 0
        self.busy_timeout_s = max(0, int(busy_timeout_ms)) / 1000
        conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        try:
            conn.row_factory = sqlite3.Row
            self._conn = conn
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute(f"PRAGMA busy_timeout={min(BUSY_ATTEMPT_MS, max(0, int(busy_timeout_ms)))}")
            if self.path != ":memory:":
                self._call(lambda: conn.execute("PRAGMA journal_mode=WAL"))
                conn.execute("PRAGMA synchronous=NORMAL")
            self.migrate()
        except BaseException:
            # Un archivo que no es SQLite o una migración que falla no deja la conexión abierta hasta el
            # GC: en Windows eso retiene el archivo.
            conn.close()
            raise

    # -- reintentos ante SQLITE_BUSY --

    def _call(self, op: Callable[[], Any]) -> Any:
        """Corre `op` con el `RLock`. Ante SQLITE_BUSY fuera de una transacción suelta el lock y reintenta.

        Dentro de una transacción no reintenta: el `BEGIN IMMEDIATE` ya tomó el lock de escritura y
        soltar el `RLock` expondría la transacción a otros hilos. Una sentencia que falla por
        SQLITE_BUSY no aplicó nada, así que repetirla es seguro.
        """
        deadline = time.monotonic() + self.busy_timeout_s
        delay = _BACKOFF_MIN_S
        while True:
            with self._lock:
                try:
                    return op()
                except sqlite3.OperationalError as exc:
                    if self._depth or not _is_busy(exc) or time.monotonic() >= deadline:
                        raise
            time.sleep(min(delay, max(0.0, deadline - time.monotonic())))
            delay = min(delay * 2, _BACKOFF_MAX_S)

    def _begin(self) -> bool:
        """Toma el `RLock` y abre `BEGIN IMMEDIATE`. Devuelve True si se une a una transacción abierta.

        Al volver, el hilo retiene el `RLock` (el llamador lo suelta). Entre intentos fallidos por
        SQLITE_BUSY lo suelta, con la misma espera total que `_call`.
        """
        deadline = time.monotonic() + self.busy_timeout_s
        delay = _BACKOFF_MIN_S
        while True:
            self._lock.acquire()
            if self._depth:
                return True
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                return False
            except BaseException as exc:
                self._lock.release()
                if (
                    not isinstance(exc, sqlite3.OperationalError)
                    or not _is_busy(exc)
                    or time.monotonic() >= deadline
                ):
                    raise
            time.sleep(min(delay, max(0.0, deadline - time.monotonic())))
            delay = min(delay * 2, _BACKOFF_MAX_S)

    # -- migraciones --

    def migrate(self) -> int:
        """Aplica las migraciones pendientes, cada una atómica. Devuelve la versión final."""
        self._call(
            lambda: self._conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations "
                "(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
            )
        )
        for version, sql in MIGRATIONS:
            if version <= self.schema_version():
                continue
            # Solo valores generados acá (entero y fecha propia); el SQL viene de MIGRATIONS.
            record = (
                "INSERT INTO schema_migrations (version, applied_at) "  # noqa: S608
                f"VALUES ({int(version)}, '{to_iso(utcnow())}');"
            )
            script = f"BEGIN IMMEDIATE;\n{sql}\n;{record}\nCOMMIT;"

            def run(script: str = script, version: int = version) -> None:
                if self.schema_version() >= version:  # otro proceso la aplicó mientras esperábamos
                    return
                try:
                    self._conn.executescript(script)
                except sqlite3.Error:
                    if self._conn.in_transaction:
                        self._conn.execute("ROLLBACK")
                    raise

            self._call(run)
        return self.schema_version()

    def schema_version(self) -> int:
        row = self._call(lambda: self._conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone())
        return int(row[0] or 0)

    # -- ejecución --

    @contextmanager
    def transaction(self) -> Iterator[Database]:
        """Transacción `BEGIN IMMEDIATE`. Una transacción anidada se une a la exterior."""
        nested = self._begin()
        try:
            if nested:
                self._depth += 1
                try:
                    yield self
                finally:
                    self._depth -= 1
                return
            self._depth = 1
            try:
                yield self
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            try:
                self._conn.execute("COMMIT")
            except BaseException:
                # Un COMMIT fallido no deja la transacción abierta en la conexión compartida.
                if self._conn.in_transaction:
                    self._conn.execute("ROLLBACK")
                raise
        finally:
            if not nested:
                self._depth = 0
            self._lock.release()

    def execute(self, sql: str, params: Iterable[Any] | Mapping[str, Any] = ()) -> int:
        """Ejecuta una sentencia y devuelve las filas afectadas."""
        return self._call(lambda: self._conn.execute(sql, params).rowcount)

    def insert(self, sql: str, params: Iterable[Any] | Mapping[str, Any] = ()) -> int:
        """Ejecuta un INSERT y devuelve el rowid."""
        return self._call(lambda: int(self._conn.execute(sql, params).lastrowid or 0))

    def executemany(self, sql: str, rows: Iterable[Iterable[Any]]) -> None:
        """Ejecuta la sentencia por fila, todo en una transacción (o en la que ya está abierta)."""
        batch = list(rows)
        with self.transaction():
            self._conn.executemany(sql, batch)

    def query(self, sql: str, params: Iterable[Any] | Mapping[str, Any] = ()) -> list[dict[str, Any]]:
        return self._call(lambda: [dict(r) for r in self._conn.execute(sql, params).fetchall()])

    def query_one(self, sql: str, params: Iterable[Any] | Mapping[str, Any] = ()) -> dict[str, Any] | None:
        def op() -> dict[str, Any] | None:
            row = self._conn.execute(sql, params).fetchone()
            return dict(row) if row is not None else None

        return self._call(op)

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# --- funciones sin tenant (creación de org y resolución del principal) ----------------------------


def create_org(
    db: Database,
    name: str,
    *,
    store_text: str = "redacted",
    retention_days: int = 30,
    policy: Mapping[str, Any] | None = None,
) -> str:
    """Crea una organización y devuelve su id."""
    _validate_org_fields({"name": name, "store_text": store_text, "retention_days": retention_days})
    org_id = new_id()
    db.execute(
        "INSERT INTO orgs (id, name, created_at, store_text, retention_days, policy_json) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (org_id, name, to_iso(utcnow()), store_text, retention_days, _json(dict(policy or {}))),
    )
    return org_id


def list_org_ids(db: Database) -> list[str]:
    """IDs de todas las organizaciones (para jobs de sistema como la purga de retención)."""
    return [r["id"] for r in db.query("SELECT id FROM orgs ORDER BY created_at, rowid")]


def lookup_user_by_email(db: Database, email: str) -> dict[str, Any] | None:
    """Usuario por email, con `password_hash`. Solo para login."""
    return db.query_one("SELECT * FROM users WHERE email = ?", (email.strip(),))


def lookup_api_key(db: Database, key_id: str) -> dict[str, Any] | None:
    """Fila completa de una API key por key_id, con `hmac`. Solo para autenticar."""
    return db.query_one("SELECT * FROM api_keys WHERE id = ?", (key_id,))


def lookup_session(db: Database, id_hash: str) -> dict[str, Any] | None:
    """Sesión por hash del id, con rol, email y estado del usuario. Solo para autenticar."""
    return db.query_one(
        "SELECT s.*, u.role AS role, u.email AS email, u.disabled AS user_disabled, "
        "u.org_id AS user_org_id FROM sessions s JOIN users u ON u.id = s.user_id WHERE s.id_hash = ?",
        (id_hash,),
    )


def _validate_org_fields(fields: Mapping[str, Any]) -> None:
    if "name" in fields:
        name = fields["name"]
        if not isinstance(name, str) or not name.strip() or len(name) > 200:
            raise ValueError("nombre de organización inválido")
    if "store_text" in fields and fields["store_text"] not in STORE_TEXT_MODES:
        raise ValueError(f"store_text debe ser uno de {STORE_TEXT_MODES}")
    if "retention_days" in fields:
        days = fields["retention_days"]
        if not isinstance(days, int) or isinstance(days, bool) or not 1 <= days <= 3650:
            raise ValueError("retention_days debe estar entre 1 y 3650")


# --- cursor de paginación -------------------------------------------------------------------------


def _encode_cursor(created_at: str, row_id: str) -> str:
    raw = _json([created_at, row_id]).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _decode_cursor(cursor: str) -> tuple[str, str]:
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        created_at, row_id = json.loads(base64.urlsafe_b64decode(padded.encode()))
        parse_iso(created_at)
        _check_id(row_id)
        return created_at, row_id
    except (ValueError, TypeError, binascii.Error, json.JSONDecodeError):
        raise ValueError("cursor inválido") from None


# --- repositorio con tenant obligatorio -----------------------------------------------------------

_ORG_UPDATABLE = ("name", "store_text", "retention_days", "external_judge_allowed")
_USER_PUBLIC = "id, org_id, email, role, created_at, disabled, failed_logins, locked_until"
_KEY_PUBLIC = (
    "id, org_id, name, pepper_kid, scopes, created_by, created_at, expires_at, last_used_at, revoked_at"
)
_DETECTOR_PUBLIC = "id, org_id, created_at, created_by, status, dataset_id"
_DATASET_PUBLIC = "id, org_id, name, created_at, n_items, sha256"
_WEBHOOK_PUBLIC = "id, org_id, url, events, created_at, disabled"


class TenantScopedRepository:
    """Lecturas y escrituras de datos de cliente, siempre filtradas por `org_id`."""

    def __init__(self, db: Database, org_id: str) -> None:
        if not isinstance(org_id, str) or not org_id:
            raise ValueError("org_id obligatorio")
        self.db = db
        self.org_id = org_id

    # -- organización --

    def get_org(self) -> dict[str, Any] | None:
        return self.db.query_one("SELECT * FROM orgs WHERE id = ?", (self.org_id,))

    def update_org(self, **fields: Any) -> bool:
        unknown = set(fields) - set(_ORG_UPDATABLE)
        if unknown:
            raise ValueError(f"campos no editables: {sorted(unknown)}")
        if not fields:
            return False
        _validate_org_fields(fields)
        if "external_judge_allowed" in fields:
            fields["external_judge_allowed"] = 1 if fields["external_judge_allowed"] else 0
        cols = ", ".join(f"{k} = ?" for k in fields)  # claves de _ORG_UPDATABLE, nunca del usuario
        return (
            self.db.execute(
                f"UPDATE orgs SET {cols} WHERE id = ?",  # noqa: S608
                (*fields.values(), self.org_id),
            )
            == 1
        )

    def get_policy(self) -> dict[str, Any] | None:
        org = self.get_org()
        return json.loads(org["policy_json"]) if org else None

    def set_policy(self, policy: Mapping[str, Any]) -> bool:
        if not isinstance(policy, Mapping):
            raise ValueError("la política debe ser un objeto")
        return (
            self.db.execute(
                "UPDATE orgs SET policy_json = ? WHERE id = ?", (_json(dict(policy)), self.org_id)
            )
            == 1
        )

    def delete_org(self) -> bool:
        """Borra la organización y, por cascada, todos sus datos (supresión del tenant completo).

        `jobs` no tiene clave foránea y se borra a mano. `audit_log` se conserva: es evidencia.
        """
        with self.db.transaction():
            self.db.execute("DELETE FROM jobs WHERE org_id = ?", (self.org_id,))
            return self.db.execute("DELETE FROM orgs WHERE id = ?", (self.org_id,)) == 1

    # -- usuarios --

    def create_user(self, email: str, password_hash: str, role: str) -> str:
        if role not in ROLES:
            raise ValueError(f"rol inválido; opciones: {ROLES}")
        user_id = new_id()
        try:
            self.db.execute(
                "INSERT INTO users (id, org_id, email, password_hash, role, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, self.org_id, email.strip(), password_hash, role, to_iso(utcnow())),
            )
        except sqlite3.IntegrityError:
            # El email es único global porque el login no pide organización.
            raise ValueError("el email ya está registrado") from None
        return user_id

    def get_user(self, user_id: str) -> dict[str, Any] | None:
        return self.db.query_one(
            f"SELECT {_USER_PUBLIC} FROM users WHERE id = ? AND org_id = ?",  # noqa: S608
            (user_id, self.org_id),
        )

    def list_users(self) -> list[dict[str, Any]]:
        return self.db.query(
            f"SELECT {_USER_PUBLIC} FROM users WHERE org_id = ? ORDER BY created_at, rowid",  # noqa: S608
            (self.org_id,),
        )

    def update_user(
        self,
        user_id: str,
        *,
        role: str | None = None,
        disabled: bool | None = None,
        password_hash: str | None = None,
    ) -> bool:
        sets: list[str] = []
        params: list[Any] = []
        if role is not None:
            if role not in ROLES:
                raise ValueError(f"rol inválido; opciones: {ROLES}")
            sets.append("role = ?")
            params.append(role)
        if disabled is not None:
            sets.append("disabled = ?")
            params.append(1 if disabled else 0)
        if password_hash is not None:
            sets.append("password_hash = ?")
            params.append(password_hash)
        if not sets:
            return False
        return (
            self.db.execute(
                f"UPDATE users SET {', '.join(sets)} WHERE id = ? AND org_id = ?",  # noqa: S608
                (*params, user_id, self.org_id),
            )
            == 1
        )

    def record_login_failure(self, user_id: str, *, max_failures: int, lock_until: datetime) -> bool:
        """Suma un fallo; al llegar a `max_failures` bloquea hasta `lock_until`. Devuelve si bloqueó."""
        with self.db.transaction():
            self.db.execute(
                "UPDATE users SET failed_logins = failed_logins + 1 WHERE id = ? AND org_id = ?",
                (user_id, self.org_id),
            )
            row = self.db.query_one(
                "SELECT failed_logins FROM users WHERE id = ? AND org_id = ?", (user_id, self.org_id)
            )
            if row is None or row["failed_logins"] < max_failures:
                return False
            self.db.execute(
                "UPDATE users SET failed_logins = 0, locked_until = ? WHERE id = ? AND org_id = ?",
                (to_iso(lock_until), user_id, self.org_id),
            )
            return True

    def reset_login_failures(self, user_id: str) -> None:
        self.db.execute(
            "UPDATE users SET failed_logins = 0, locked_until = NULL WHERE id = ? AND org_id = ?",
            (user_id, self.org_id),
        )

    def delete_user(self, user_id: str) -> bool:
        return self.db.execute("DELETE FROM users WHERE id = ? AND org_id = ?", (user_id, self.org_id)) == 1

    # -- sesiones --

    def create_session(
        self, id_hash: str, user_id: str, csrf_secret: str, *, created_at: datetime, expires_at: datetime
    ) -> None:
        if self.get_user(user_id) is None:
            raise ValueError("usuario inexistente en esta organización")
        now = to_iso(created_at)
        self.db.execute(
            "INSERT INTO sessions (id_hash, user_id, org_id, csrf_secret, created_at, last_seen_at, "
            "expires_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (id_hash, user_id, self.org_id, csrf_secret, now, now, to_iso(expires_at)),
        )

    def touch_session(self, id_hash: str, now: datetime) -> None:
        self.db.execute(
            "UPDATE sessions SET last_seen_at = ? WHERE id_hash = ? AND org_id = ?",
            (to_iso(now), id_hash, self.org_id),
        )

    def delete_session(self, id_hash: str) -> bool:
        return (
            self.db.execute("DELETE FROM sessions WHERE id_hash = ? AND org_id = ?", (id_hash, self.org_id))
            == 1
        )

    def delete_user_sessions(self, user_id: str) -> int:
        return self.db.execute(
            "DELETE FROM sessions WHERE user_id = ? AND org_id = ?", (user_id, self.org_id)
        )

    # -- API keys --

    def insert_api_key(
        self,
        key_id: str,
        *,
        name: str,
        hmac_hex: str,
        pepper_kid: str,
        scopes: Iterable[str],
        created_by: str | None = None,
        expires_at: datetime | None = None,
    ) -> None:
        self.db.execute(
            "INSERT INTO api_keys (id, org_id, name, hmac, pepper_kid, scopes, created_by, created_at, "
            "expires_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                key_id,
                self.org_id,
                name,
                hmac_hex,
                pepper_kid,
                ",".join(scopes),
                created_by,
                to_iso(utcnow()),
                to_iso(expires_at) if expires_at else None,
            ),
        )

    def get_api_key(self, key_id: str) -> dict[str, Any] | None:
        return self.db.query_one(
            f"SELECT {_KEY_PUBLIC} FROM api_keys WHERE id = ? AND org_id = ?",  # noqa: S608
            (key_id, self.org_id),
        )

    def list_api_keys(self) -> list[dict[str, Any]]:
        return self.db.query(
            f"SELECT {_KEY_PUBLIC} FROM api_keys WHERE org_id = ? ORDER BY created_at, rowid",  # noqa: S608
            (self.org_id,),
        )

    def revoke_api_key(self, key_id: str, *, now: datetime | None = None) -> bool:
        return (
            self.db.execute(
                "UPDATE api_keys SET revoked_at = ? WHERE id = ? AND org_id = ? AND revoked_at IS NULL",
                (to_iso(now or utcnow()), key_id, self.org_id),
            )
            == 1
        )

    def touch_api_key(self, key_id: str, now: datetime) -> None:
        self.db.execute(
            "UPDATE api_keys SET last_used_at = ? WHERE id = ? AND org_id = ?",
            (to_iso(now), key_id, self.org_id),
        )

    # -- trazas --

    def insert_trace(
        self,
        *,
        model_id: str,
        response_text: str,
        action: str,
        prompt_text: str | None = None,
        tokens: Iterable[Any] = (),
        spans: Iterable[Any] = (),
        trace_id: str | None = None,
        n_tokens: int | None = None,
        api_key_id: str | None = None,
        source: str = "api",
        mode: str = "full",
        response_p_error: float | None = None,
        calibration_id: str | None = None,
        alpha: float | None = None,
        tau: float | None = None,
        latency_ms: float | None = None,
        overhead_ms: float | None = None,
        degraded: bool = False,
        finish_reason: str | None = None,
        usage: Mapping[str, Any] | None = None,
        created_at: datetime | None = None,
        status: str = "ok",
        rag_status: str | None = None,
    ) -> str:
        """Guarda una traza con sus tokens y fragmentos en una transacción.

        Recibe el texto crudo y aplica `store_text` de la organización acá, en la frontera:
        `none` no guarda texto, `redacted` guarda marcadores de PII y re-mapea los offsets de
        tokens y fragmentos al texto redactado, `full` guarda todo. Los tokens y fragmentos aceptan
        dicts o los dataclasses `ScoredToken` / `Span` de `vigia.contracts`. `status` y `rag_status` son
        los de `VigiaResult`.
        """
        if source not in TRACE_SOURCES:
            raise ValueError(f"source inválido; opciones: {TRACE_SOURCES}")
        trace_id = _check_id(trace_id, "trace_id") if trace_id is not None else new_id()
        token_list = list(tokens)
        span_list = list(spans)
        with self.db.transaction():
            org = self.get_org()
            if org is None:
                raise LookupError("organización inexistente")
            mode_text = org["store_text"]
            text_sha = hashlib.sha256(((prompt_text or "") + "\x00" + response_text).encode()).hexdigest()
            if mode_text == "full":
                stored_prompt, stored_response, replacements = prompt_text, response_text, []
            elif mode_text == "redacted":
                stored_prompt = pii.redact(prompt_text)[0] if prompt_text is not None else None
                red = pii.redact_with_map(response_text)
                stored_response, replacements = red.text, red.replacements
            else:
                stored_prompt, stored_response, replacements = None, None, []

            def remap(pos: int) -> int:
                return pii.map_offset(replacements, pos) if replacements else pos

            self.db.execute(
                "INSERT INTO traces (id, org_id, created_at, api_key_id, source, model_id, mode, "
                "prompt_text, "
                "response_text, text_sha256, n_tokens, response_p_error, action, calibration_id, alpha, tau, "
                "latency_ms, overhead_ms, degraded, finish_reason, usage_json, status, rag_status) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    trace_id,
                    self.org_id,
                    to_iso(created_at or utcnow()),
                    api_key_id,
                    source,
                    model_id,
                    mode,
                    stored_prompt,
                    stored_response,
                    text_sha,
                    n_tokens if n_tokens is not None else len(token_list),
                    response_p_error,
                    action,
                    calibration_id,
                    alpha,
                    tau,
                    latency_ms,
                    overhead_ms,
                    1 if degraded else 0,
                    finish_reason,
                    _json(dict(usage)) if usage is not None else None,
                    status,
                    rag_status,
                ),
            )
            token_rows = []
            for pos, tok in enumerate(token_list):
                start, end = int(_get(tok, "start")), int(_get(tok, "end"))
                idx = _get(tok, "idx", _get(tok, "index", pos))
                if mode_text == "none":
                    text = None
                elif mode_text == "redacted":
                    start, end = remap(start), remap(end)
                    text = stored_response[start:end] if stored_response is not None else None
                else:
                    text = _get(tok, "text")
                token_rows.append(
                    (
                        trace_id,
                        self.org_id,
                        int(idx),
                        text,
                        start,
                        end,
                        _get(tok, "p_error"),
                        1 if _get(tok, "flag", False) else 0,
                        _get(tok, "out_entropy"),
                        _get(tok, "r_ent_mean"),
                    )
                )
            self.db.executemany(
                "INSERT INTO trace_tokens (trace_id, org_id, idx, text, start, end, p_error, flag, "
                "out_entropy, r_ent_mean) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                token_rows,
            )
            span_rows = []
            for pos, span in enumerate(span_list):
                start, end = int(_get(span, "start")), int(_get(span, "end"))
                if mode_text == "redacted":
                    start, end = remap(start), remap(end)
                span_rows.append(
                    (
                        trace_id,
                        self.org_id,
                        int(_get(span, "idx", pos)),
                        start,
                        end,
                        float(_get(span, "score")),
                    )
                )
            self.db.executemany(
                "INSERT INTO spans (trace_id, org_id, idx, start, end, score) VALUES (?, ?, ?, ?, ?, ?)",
                span_rows,
            )
        return trace_id

    def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        """Traza con `tokens` y `spans` ordenados, o None si no es de este tenant."""
        trace = self.db.query_one("SELECT * FROM traces WHERE id = ? AND org_id = ?", (trace_id, self.org_id))
        if trace is None:
            return None
        trace["tokens"] = self.db.query(
            "SELECT idx, text, start, end, p_error, flag, out_entropy, r_ent_mean FROM trace_tokens "
            "WHERE trace_id = ? AND org_id = ? ORDER BY idx",
            (trace_id, self.org_id),
        )
        trace["spans"] = self.db.query(
            "SELECT idx, start, end, score FROM spans WHERE trace_id = ? AND org_id = ? ORDER BY idx",
            (trace_id, self.org_id),
        )
        return trace

    def list_traces(
        self,
        *,
        limit: int = 50,
        cursor: str | None = None,
        action: str | None = None,
        source: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        min_p_error: float | None = None,
        degraded: bool | None = None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Trazas de la más nueva a la más vieja, con paginación por cursor opaco.

        Devuelve `(filas, siguiente_cursor)`; el cursor es None en la última página.
        """
        limit = max(1, min(int(limit), MAX_PAGE))
        where = ["org_id = ?"]
        params: list[Any] = [self.org_id]
        if action is not None:
            where.append("action = ?")
            params.append(action)
        if source is not None:
            where.append("source = ?")
            params.append(source)
        if since is not None:
            where.append("created_at >= ?")
            params.append(to_iso(since))
        if until is not None:
            where.append("created_at < ?")
            params.append(to_iso(until))
        if min_p_error is not None:
            where.append("response_p_error >= ?")
            params.append(float(min_p_error))
        if degraded is not None:
            where.append("degraded = ?")
            params.append(1 if degraded else 0)
        if cursor:
            c_at, c_id = _decode_cursor(cursor)
            where.append("(created_at < ? OR (created_at = ? AND id < ?))")
            params.extend([c_at, c_at, c_id])
        rows = self.db.query(
            f"SELECT * FROM traces WHERE {' AND '.join(where)} "  # noqa: S608 - condiciones fijas
            "ORDER BY created_at DESC, id DESC LIMIT ?",
            (*params, limit + 1),
        )
        next_cursor = None
        if len(rows) > limit:
            rows = rows[:limit]
            next_cursor = _encode_cursor(rows[-1]["created_at"], rows[-1]["id"])
        return rows, next_cursor

    def recent_token_signals(self, calibration_id: str, *, limit_traces: int = 500) -> dict[str, Any]:
        """Señales por token de las últimas `limit_traces` trazas puntuadas con `calibration_id`.

        Para el monitor de drift: `{n_traces, tokens: [{p_error, flag, r_ent_mean, out_entropy}]}`.
        Excluye trazas degradadas y tokens sin p_error (especiales y canales no puntuados).
        """
        limit = max(1, min(int(limit_traces), 10_000))
        rows = self.db.query(
            "SELECT tt.trace_id, tt.p_error, tt.flag, tt.r_ent_mean, tt.out_entropy FROM trace_tokens tt "
            "WHERE tt.org_id = ? AND tt.p_error IS NOT NULL AND tt.trace_id IN ("
            "SELECT id FROM traces WHERE org_id = ? AND calibration_id = ? AND degraded = 0 "
            "ORDER BY created_at DESC, id DESC LIMIT ?)",
            (self.org_id, self.org_id, calibration_id, limit),
        )
        n_traces = len({r.pop("trace_id") for r in rows})
        return {"n_traces": n_traces, "tokens": rows}

    def delete_trace(self, trace_id: str) -> bool:
        """Borra una traza con sus tokens, fragmentos e items de revisión (cascada)."""
        return self.db.execute("DELETE FROM traces WHERE id = ? AND org_id = ?", (trace_id, self.org_id)) == 1

    def purge_before(self, ts: datetime) -> int:
        """Borra las trazas creadas antes de `ts` (y sus dependientes). Devuelve cuántas borró."""
        return self.db.execute(
            "DELETE FROM traces WHERE org_id = ? AND created_at < ?", (self.org_id, to_iso(ts))
        )

    # -- revisión --

    def create_review_item(
        self, trace_id: str, *, span_idx: int | None = None, min_reviewers: int = 1
    ) -> str | None:
        """Encola una traza (o un fragmento) para revisión. None si la traza no es de este tenant."""
        if not isinstance(min_reviewers, int) or min_reviewers < 1:
            raise ValueError("min_reviewers debe ser >= 1")
        with self.db.transaction():
            if (
                self.db.query_one("SELECT 1 FROM traces WHERE id = ? AND org_id = ?", (trace_id, self.org_id))
                is None
            ):
                return None
            item_id = new_id()
            self.db.execute(
                "INSERT INTO review_items (id, org_id, trace_id, span_idx, state, min_reviewers, created_at) "
                "VALUES (?, ?, ?, ?, 'pending', ?, ?)",
                (item_id, self.org_id, trace_id, span_idx, min_reviewers, to_iso(utcnow())),
            )
            return item_id

    def get_review_item(self, item_id: str) -> dict[str, Any] | None:
        return self.db.query_one(
            "SELECT * FROM review_items WHERE id = ? AND org_id = ?", (item_id, self.org_id)
        )

    def list_review_items(self, *, state: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), MAX_PAGE))
        if state is None:
            return self.db.query(
                "SELECT * FROM review_items WHERE org_id = ? ORDER BY created_at LIMIT ?",
                (self.org_id, limit),
            )
        if state not in REVIEW_STATES:
            raise ValueError(f"estado inválido; opciones: {REVIEW_STATES}")
        return self.db.query(
            "SELECT * FROM review_items WHERE org_id = ? AND state = ? ORDER BY created_at LIMIT ?",
            (self.org_id, state, limit),
        )

    def reserve_review_item(
        self, item_id: str, user_id: str, *, until: datetime, now: datetime | None = None
    ) -> bool:
        """Reserva un item para un revisor que todavía no decidió sobre él."""
        now_s = to_iso(now or utcnow())
        return (
            self.db.execute(
                "UPDATE review_items SET state = 'reserved', reserved_by = ?, reserved_until = ? "
                "WHERE id = ? AND org_id = ? AND ("
                "  state IN ('pending', 'needs_other') OR "
                "  (state = 'reserved' AND (reserved_until < ? OR reserved_by = ?))) "
                "AND EXISTS (SELECT 1 FROM users WHERE id = ? AND org_id = ?) "
                "AND NOT EXISTS (SELECT 1 FROM review_decisions WHERE item_id = ? AND user_id = ?)",
                (
                    user_id,
                    to_iso(until),
                    item_id,
                    self.org_id,
                    now_s,
                    user_id,
                    user_id,
                    self.org_id,
                    item_id,
                    user_id,
                ),
            )
            == 1
        )

    def add_review_decision(
        self, item_id: str, user_id: str, label: str, *, note: str | None = None, now: datetime | None = None
    ) -> str | None:
        """Registra la decisión de un revisor y avanza el estado del item.

        Devuelve el id de la decisión, o None si el item no es de este tenant, ya está completo, lo
        reservó otro revisor con reserva vigente o este revisor ya decidió.
        """
        if label not in REVIEW_LABELS:
            raise ValueError(f"etiqueta inválida; opciones: {REVIEW_LABELS}")
        now_dt = now or utcnow()
        with self.db.transaction():
            item = self.get_review_item(item_id)
            if item is None or item["state"] == "completed":
                return None
            if self.get_user(user_id) is None:
                return None
            if (
                item["state"] == "reserved"
                and item["reserved_by"] != user_id
                and item["reserved_until"]
                and item["reserved_until"] >= to_iso(now_dt)
            ):
                return None
            if self.db.query_one(
                "SELECT 1 FROM review_decisions WHERE item_id = ? AND user_id = ?", (item_id, user_id)
            ):
                return None
            decision_id = new_id()
            self.db.execute(
                "INSERT INTO review_decisions (id, item_id, org_id, user_id, label, note, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (decision_id, item_id, self.org_id, user_id, label, note, to_iso(now_dt)),
            )
            count = self.db.query_one(
                "SELECT COUNT(*) AS n FROM review_decisions WHERE item_id = ? AND org_id = ?",
                (item_id, self.org_id),
            )["n"]
            if count >= item["min_reviewers"]:
                self.db.execute(
                    "UPDATE review_items SET state = 'completed', completed_at = ?, reserved_by = NULL, "
                    "reserved_until = NULL WHERE id = ? AND org_id = ?",
                    (to_iso(now_dt), item_id, self.org_id),
                )
            else:
                self.db.execute(
                    "UPDATE review_items SET state = 'needs_other', reserved_by = NULL, "
                    "reserved_until = NULL "
                    "WHERE id = ? AND org_id = ?",
                    (item_id, self.org_id),
                )
            return decision_id

    def list_review_decisions(self, item_id: str) -> list[dict[str, Any]] | None:
        if self.get_review_item(item_id) is None:
            return None
        return self.db.query(
            "SELECT * FROM review_decisions WHERE item_id = ? AND org_id = ? ORDER BY created_at, rowid",
            (item_id, self.org_id),
        )

    # -- detectores --

    def insert_detector(
        self,
        detector_id: str,
        artifact_json: str,
        *,
        report_json: str = "{}",
        dataset_id: str | None = None,
        created_by: str | None = None,
    ) -> None:
        """Guarda un detector candidato. Falla si el id ya existe o el dataset no es de este tenant."""
        if dataset_id is not None and self.get_dataset(dataset_id) is None:
            raise ValueError("dataset inexistente en esta organización")
        try:
            self.db.execute(
                "INSERT INTO detectors (id, org_id, created_at, created_by, status, artifact_json, "
                "report_json, "
                "dataset_id) VALUES (?, ?, ?, ?, 'candidate', ?, ?, ?)",
                (
                    detector_id,
                    self.org_id,
                    to_iso(utcnow()),
                    created_by,
                    artifact_json,
                    report_json,
                    dataset_id,
                ),
            )
        except sqlite3.IntegrityError:
            raise ValueError("el detector ya existe") from None

    def get_detector(self, detector_id: str) -> dict[str, Any] | None:
        return self.db.query_one(
            "SELECT * FROM detectors WHERE id = ? AND org_id = ?", (detector_id, self.org_id)
        )

    def list_detectors(self) -> list[dict[str, Any]]:
        return self.db.query(
            f"SELECT {_DETECTOR_PUBLIC} FROM detectors WHERE org_id = ? ORDER BY created_at, rowid",  # noqa: S608
            (self.org_id,),
        )

    def set_detector_report(self, detector_id: str, report_json: str) -> bool:
        return (
            self.db.execute(
                "UPDATE detectors SET report_json = ? WHERE id = ? AND org_id = ?",
                (report_json, detector_id, self.org_id),
            )
            == 1
        )

    def activate_detector(self, detector_id: str) -> bool:
        """Activa un detector y retira el activo anterior, en una transacción."""
        with self.db.transaction():
            if self.get_detector(detector_id) is None:
                return False
            self.db.execute(
                "UPDATE detectors SET status = 'retired' WHERE org_id = ? AND status = 'active' AND id != ?",
                (self.org_id, detector_id),
            )
            self.db.execute(
                "UPDATE detectors SET status = 'active' WHERE id = ? AND org_id = ?",
                (detector_id, self.org_id),
            )
            self.db.execute("UPDATE orgs SET active_detector_id = ? WHERE id = ?", (detector_id, self.org_id))
            return True

    def get_active_detector(self) -> dict[str, Any] | None:
        return self.db.query_one(
            "SELECT d.* FROM detectors d JOIN orgs o ON o.active_detector_id = d.id "
            "WHERE o.id = ? AND d.org_id = ? AND d.status = 'active'",
            (self.org_id, self.org_id),
        )

    # -- datasets --

    def insert_dataset(self, name: str, content: bytes, *, n_items: int) -> str:
        if not isinstance(content, bytes):
            raise ValueError("content debe ser bytes")
        dataset_id = new_id()
        self.db.execute(
            "INSERT INTO datasets (id, org_id, name, created_at, n_items, sha256, content) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                dataset_id,
                self.org_id,
                name,
                to_iso(utcnow()),
                int(n_items),
                hashlib.sha256(content).hexdigest(),
                content,
            ),
        )
        return dataset_id

    def get_dataset(self, dataset_id: str) -> dict[str, Any] | None:
        return self.db.query_one(
            "SELECT * FROM datasets WHERE id = ? AND org_id = ?", (dataset_id, self.org_id)
        )

    def list_datasets(self) -> list[dict[str, Any]]:
        return self.db.query(
            f"SELECT {_DATASET_PUBLIC} FROM datasets WHERE org_id = ? ORDER BY created_at, rowid",  # noqa: S608
            (self.org_id,),
        )

    def delete_dataset(self, dataset_id: str) -> bool:
        return (
            self.db.execute("DELETE FROM datasets WHERE id = ? AND org_id = ?", (dataset_id, self.org_id))
            == 1
        )

    # -- webhooks --

    def insert_webhook(self, url: str, secret: str, events: Iterable[str]) -> str:
        """Guarda un webhook. La validación de destino (SSRF) es de `vigia_ee.webhooks`."""
        webhook_id = new_id()
        self.db.execute(
            "INSERT INTO webhooks (id, org_id, url, secret, events, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (webhook_id, self.org_id, url, secret, ",".join(events), to_iso(utcnow())),
        )
        return webhook_id

    def get_webhook(self, webhook_id: str) -> dict[str, Any] | None:
        """Webhook completo, con secreto (para firmar entregas)."""
        return self.db.query_one(
            "SELECT * FROM webhooks WHERE id = ? AND org_id = ?", (webhook_id, self.org_id)
        )

    def list_webhooks(
        self, *, event: str | None = None, enabled_only: bool = False, with_secret: bool = False
    ) -> list[dict[str, Any]]:
        cols = "*" if with_secret else _WEBHOOK_PUBLIC
        sql = f"SELECT {cols} FROM webhooks WHERE org_id = ?"  # noqa: S608 - columnas fijas
        if enabled_only:
            sql += " AND disabled = 0"
        rows = self.db.query(sql + " ORDER BY created_at, rowid", (self.org_id,))
        if event is not None:
            rows = [r for r in rows if event in r["events"].split(",")]
        return rows

    def set_webhook_disabled(self, webhook_id: str, disabled: bool) -> bool:
        return (
            self.db.execute(
                "UPDATE webhooks SET disabled = ? WHERE id = ? AND org_id = ?",
                (1 if disabled else 0, webhook_id, self.org_id),
            )
            == 1
        )

    def delete_webhook(self, webhook_id: str) -> bool:
        return (
            self.db.execute("DELETE FROM webhooks WHERE id = ? AND org_id = ?", (webhook_id, self.org_id))
            == 1
        )

    # -- export del tenant (Ley 25.326 art. 14) --

    def export_all(self) -> dict[str, Any]:
        """Todo lo del tenant en un dict serializable a JSON. Excluye secretos y sesiones.

        Quedan afuera `password_hash`, el HMAC de las keys, el secreto de los webhooks y la tabla
        de sesiones. El contenido de los datasets sale como texto UTF-8 (son JSONL validados).
        """
        org = self.get_org()
        if org is None:
            raise LookupError("organización inexistente")
        q = self.db.query
        o = (self.org_id,)
        traces = q("SELECT * FROM traces WHERE org_id = ? ORDER BY created_at, id", o)
        tokens = q("SELECT * FROM trace_tokens WHERE org_id = ? ORDER BY trace_id, idx", o)
        spans = q("SELECT * FROM spans WHERE org_id = ? ORDER BY trace_id, idx", o)
        by_trace_tokens: dict[str, list[dict[str, Any]]] = {}
        by_trace_spans: dict[str, list[dict[str, Any]]] = {}
        for t in tokens:
            by_trace_tokens.setdefault(t["trace_id"], []).append(t)
        for s in spans:
            by_trace_spans.setdefault(s["trace_id"], []).append(s)
        for tr in traces:
            tr["tokens"] = by_trace_tokens.get(tr["id"], [])
            tr["spans"] = by_trace_spans.get(tr["id"], [])
        datasets = q("SELECT * FROM datasets WHERE org_id = ? ORDER BY created_at, rowid", o)
        for d in datasets:
            d["content"] = bytes(d["content"]).decode("utf-8", errors="replace")
        return {
            "schema_version": SCHEMA_VERSION,
            "exported_at": to_iso(utcnow()),
            "org_id": self.org_id,
            "org": org,
            "users": self.list_users(),
            "api_keys": self.list_api_keys(),
            "traces": traces,
            "review_items": q("SELECT * FROM review_items WHERE org_id = ? ORDER BY created_at, rowid", o),
            "review_decisions": q(
                "SELECT * FROM review_decisions WHERE org_id = ? ORDER BY created_at, rowid", o
            ),
            "detectors": q("SELECT * FROM detectors WHERE org_id = ? ORDER BY created_at, rowid", o),
            "datasets": datasets,
            "webhooks": self.list_webhooks(),
            "jobs": q("SELECT * FROM jobs WHERE org_id = ? ORDER BY created_at, rowid", o),
            "audit_log": q(AUDIT_SELECT + AUDIT_ORDER, o),
        }
