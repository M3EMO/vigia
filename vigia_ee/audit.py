"""Auditoría append-only con cadena de hashes por organización.

Cada fila guarda `prev_hash` (el `hash` de la fila anterior de la misma organización, o
`GENESIS` en la primera) y `hash = sha256(prev_hash || json_canónico(fila sin seq ni hash))`.
Alterar, borrar o reordenar una fila intermedia rompe la verificación. Truncar las últimas filas
no se detecta con la cadena sola: para eso conviene exportar y guardar el último hash afuera.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from vigia_ee.db import Database, to_iso, utcnow

GENESIS = "0" * 64
HASHED_FIELDS = (
    "org_id",
    "ts",
    "actor_user_id",
    "actor_key_id",
    "action",
    "target",
    "detail_json",
    "prev_hash",
)
# Claves que nunca entran al detalle de auditoría: el log es evidencia y no debe guardar secretos.
FORBIDDEN_DETAIL_KEYS = frozenset(
    {"password", "token", "secret", "api_key", "session", "cookie", "hmac", "pepper"}
)


@dataclass(frozen=True)
class ChainCheck:
    ok: bool
    checked: int
    broken_at: int | None = None  # seq de la primera fila que no valida


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def row_hash(row: Mapping[str, Any]) -> str:
    payload = {k: row.get(k) for k in HASHED_FIELDS if k != "prev_hash"}
    return hashlib.sha256((row["prev_hash"] + canonical_json(payload)).encode()).hexdigest()


def _check_detail(detail: Mapping[str, Any]) -> None:
    for key, value in detail.items():
        k = str(key).lower()
        if k in FORBIDDEN_DETAIL_KEYS or any(k.endswith("_" + bad) for bad in FORBIDDEN_DETAIL_KEYS):
            raise ValueError(f"el detalle de auditoría no admite la clave {key!r}")
        if isinstance(value, Mapping):
            _check_detail(value)


def append(
    db: Database,
    org_id: str | None,
    actor: Any,
    action: str,
    target: str | None = None,
    detail: Mapping[str, Any] | None = None,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Agrega un evento al final de la cadena de `org_id` (None = eventos de sistema).

    `actor` es un `Principal` (o cualquier objeto con `user_id` / `key_id`), un dict con esas
    claves, o None para el sistema. Devuelve la fila insertada con `seq` y `hash`.
    """
    if not isinstance(action, str) or not action or len(action) > 64:
        raise ValueError("acción de auditoría inválida")
    detail = dict(detail or {})
    _check_detail(detail)

    def actor_field(name: str) -> str | None:
        if actor is None:
            return None
        if isinstance(actor, Mapping):
            return actor.get(name)
        return getattr(actor, name, None)

    row: dict[str, Any] = {
        "org_id": org_id,
        "ts": to_iso(now or utcnow()),
        "actor_user_id": actor_field("user_id"),
        "actor_key_id": actor_field("key_id"),
        "action": action,
        "target": target,
        "detail_json": canonical_json(detail),
    }
    with db.transaction():
        last = db.query_one(
            "SELECT hash FROM audit_log WHERE org_id IS ? ORDER BY seq DESC LIMIT 1", (org_id,)
        )
        row["prev_hash"] = last["hash"] if last else GENESIS
        row["hash"] = row_hash(row)
        row["seq"] = db.insert(
            "INSERT INTO audit_log (org_id, ts, actor_user_id, actor_key_id, action, target, detail_json, "
            "prev_hash, hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            tuple(row[k] for k in HASHED_FIELDS) + (row["hash"],),
        )
    return row


def verify_rows(rows: Iterable[Mapping[str, Any]], *, expected_prev: str | None = None) -> ChainCheck:
    """Verifica hashes y enlaces de una secuencia ordenada por `seq`.

    Con `expected_prev`, la primera fila debe enlazar con ese hash (GENESIS para la cadena entera).
    Sin él, sirve para un export por rango: se valida desde la primera fila recibida.
    """
    prev = expected_prev
    checked = 0
    for row in rows:
        if prev is not None and row["prev_hash"] != prev:
            return ChainCheck(False, checked, row["seq"])
        if row_hash(row) != row["hash"]:
            return ChainCheck(False, checked, row["seq"])
        prev = row["hash"]
        checked += 1
    return ChainCheck(True, checked)


def verify_chain(db: Database, org_id: str | None) -> ChainCheck:
    """Recorre la cadena completa de `org_id` y devuelve dónde se rompe, si se rompe."""
    rows = db.query("SELECT * FROM audit_log WHERE org_id IS ? ORDER BY seq", (org_id,))
    return verify_rows(rows, expected_prev=GENESIS)


def export_range(
    db: Database, org_id: str | None, *, start: datetime | None = None, end: datetime | None = None
) -> list[dict[str, Any]]:
    """Eventos de `org_id` con `start <= ts < end`, ordenados por `seq`, verificables con `verify_rows`."""
    sql = "SELECT * FROM audit_log WHERE org_id IS ?"
    params: list[Any] = [org_id]
    if start is not None:
        sql += " AND ts >= ?"
        params.append(to_iso(start))
    if end is not None:
        sql += " AND ts < ?"
        params.append(to_iso(end))
    rows = db.query(sql + " ORDER BY seq", params)
    for r in rows:
        r["detail"] = json.loads(r["detail_json"])
    return rows
