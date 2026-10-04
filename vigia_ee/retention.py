"""Retención y derechos del titular (Ley 25.326 art. 4, 14 y 16).

- `purge_org` borra las trazas de un tenant más viejas que su `retention_days` (con tokens,
  fragmentos e items de revisión por cascada) y los jobs terminados del mismo período. Nunca toca
  otro tenant. Cada corrida queda en auditoría con las cantidades.
- `schedule` encola un job de sistema `purge` recurrente: el handler encola la corrida siguiente
  antes de purgar, así un fallo no corta el ciclo.
- `delete_trace_request` atiende un pedido de supresión puntual y `export_org` el de acceso.
- La auditoría se conserva: es evidencia y no tiene datos del titular.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from vigia_ee import audit
from vigia_ee.db import Database, TenantScopedRepository, list_org_ids, to_iso, utcnow
from vigia_ee.jobs import JobQueue
from vigia_ee.security import log_event

PURGE_KIND = "purge"
SYSTEM_JOBS_RETENTION = timedelta(days=30)
PURGE_MAX_ATTEMPTS = 3
MAX_EVERY_HOURS = 24 * 30

_log = logging.getLogger("vigia.retention")


def purge_org(
    db: Database, org_id: str, *, now: datetime | None = None, jobs: JobQueue | None = None
) -> dict[str, Any]:
    """Purga lo vencido de un tenant y lo audita. Devuelve las cantidades borradas."""
    now = now or utcnow()
    repo = TenantScopedRepository(db, org_id)
    org = repo.get_org()
    if org is None:
        raise LookupError("organización inexistente")
    days = int(org["retention_days"])
    cutoff = now - timedelta(days=days)
    with db.transaction():
        traces = repo.purge_before(cutoff)
        finished_jobs = jobs.purge_finished(cutoff, org_id=org_id) if jobs is not None else 0
        audit.append(
            db,
            org_id,
            None,
            "retention_purge",
            None,
            {"traces": traces, "jobs": finished_jobs, "retention_days": days, "cutoff": to_iso(cutoff)},
            now=now,
        )
    return {"traces": traces, "jobs": finished_jobs, "cutoff": to_iso(cutoff)}


def purge_all(db: Database, jobs: JobQueue | None = None, *, now: datetime | None = None) -> dict[str, Any]:
    """Purga todos los tenants. Un tenant que falla no frena al resto; al final se informa el fallo.

    Con `jobs`, también borra los jobs de sistema terminados hace más de 30 días.
    """
    now = now or utcnow()
    totals = {"orgs": 0, "traces": 0, "jobs": 0}
    failed = 0
    for org_id in list_org_ids(db):
        try:
            result = purge_org(db, org_id, now=now, jobs=jobs)
        except LookupError:
            continue  # el tenant se borró entre el listado y la purga
        except Exception as exc:  # se reintenta en la próxima corrida; el resto sigue
            failed += 1
            log_event(_log, "retention_purge_error", level=logging.ERROR, org_id=org_id,
                      error_type=type(exc).__name__)
            continue
        totals["orgs"] += 1
        totals["traces"] += result["traces"]
        totals["jobs"] += result["jobs"]
    if jobs is not None:
        totals["jobs"] += jobs.purge_finished(now - SYSTEM_JOBS_RETENTION, org_id=None)
    if failed:
        raise RuntimeError(f"la purga falló en {failed} organización(es)")
    return totals


def _ensure_next(jobs: JobQueue, every_hours: float, run_after: datetime) -> str:
    """Id del próximo job de purga en cola; si no hay ninguno, lo encola para `run_after`."""
    queued = jobs.list_system(PURGE_KIND, states=("queued",))
    if queued:
        return queued[0]["id"]
    return jobs.enqueue(
        PURGE_KIND,
        {"every_hours": every_hours},
        run_after=run_after,
        max_attempts=PURGE_MAX_ATTEMPTS,
    )


def schedule(jobs: JobQueue, every_hours: float = 24) -> str:
    """Registra el handler `purge` y deja una purga en cola (inmediata si no había ninguna).

    Es idempotente: al reiniciar el proceso no duplica la purga pendiente. Devuelve su id.
    """
    if isinstance(every_hours, bool) or not 0 < float(every_hours) <= MAX_EVERY_HOURS:
        raise ValueError(f"every_hours debe estar entre 0 y {MAX_EVERY_HOURS}")
    every = float(every_hours)

    def handler(job: dict[str, Any]) -> dict[str, Any]:
        interval = float(job["payload"].get("every_hours", every))
        now = jobs.now()
        _ensure_next(jobs, interval, now + timedelta(hours=interval))
        return purge_all(jobs.db, jobs, now=now)

    jobs.register(PURGE_KIND, handler)
    return _ensure_next(jobs, every, jobs.now())


def delete_trace_request(repo: TenantScopedRepository, trace_id: str, actor: Any) -> bool:
    """Borra una traza a pedido (supresión). Otro tenant o inexistente → False (404), sin auditar."""
    with repo.db.transaction():
        if not repo.delete_trace(trace_id):
            return False
        audit.append(repo.db, repo.org_id, actor, "trace_delete", trace_id)
    return True


def export_org(repo: TenantScopedRepository, actor: Any) -> dict[str, Any]:
    """Export JSON de todo lo del tenant (acceso del titular) y su evento `data_export`."""
    data = repo.export_all()
    counts = {
        key: len(data[key])
        for key in ("users", "api_keys", "traces", "review_items", "detectors", "datasets", "webhooks")
    }
    audit.append(repo.db, repo.org_id, actor, "data_export", None, {"counts": counts})
    return data
