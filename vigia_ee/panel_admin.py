"""Páginas de operación y administración del panel: cola de revisión, calibración, políticas,
configuración (organización, usuarios, API keys, webhooks), auditoría y datos del tenant.

`register(router, runtime)` agrega las rutas al router que arma `vigia_ee.panel.create_panel_router`.
Mismas reglas que `panel`: cada ruta declara su permiso, autoescape siempre, los secretos (token de
API key, secreto de webhook, contraseña temporal) se muestran una sola vez en la respuesta del POST
y nunca viajan por URL ni quedan en la base en claro.
"""

from __future__ import annotations

import json
import re
import secrets
from collections.abc import Mapping
from datetime import timedelta
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse
from markupsafe import Markup
from starlette.datastructures import UploadFile

from vigia import charts
from vigia.policy import PolicyError, UnsafeURLError
from vigia_ee import audit, retention, webhooks
from vigia_ee.auth import (
    ROLE_RANK,
    SCOPES,
    PasswordPolicyError,
    Principal,
    can_assign_role,
    change_role,
    check_password_policy,
    create_api_key,
    register_user,
    require_session,
    revoke_api_key,
)
from vigia_ee.db import REVIEW_LABELS, REVIEW_STATES, STORE_TEXT_MODES, new_id, to_iso, utcnow
from vigia_ee.panel import (
    ROLE_LABELS,
    PanelHTTPError,
    clean,
    clear_cookie,
    fmt_num,
    form_text,
    local_day_start,
    marks_from_spans,
    not_found,
    page,
    parse_float,
    parse_int,
    redirect,
    render,
    scrollable_chart,
    segments_for,
)
from vigia_ee.pii import csv_dumps

RESERVATION_TTL = timedelta(minutes=15)
MAX_NOTE_CHARS = 2000
MAX_QUEUE_SCAN = 200
AUDIT_PAGE = 200
TEMP_PASSWORD_ALPHABET = "abcdefghjkmnpqrstuvwxyzACDEFGHJKLMNPQRSTUVWXYZ23456789"  # noqa: S105 - alfabeto, no un secreto
STATE_LABELS = {
    "pending": "Pendiente",
    "reserved": "Reservado",
    "needs_other": "Necesita otro revisor",
    "completed": "Completado",
}
LABEL_TEXT = {"correct": "Correcto", "error": "Error", "uncertain": "Incierto"}
STORE_TEXT_LABELS = {
    "none": "No guardar texto (solo puntajes y offsets)",
    "redacted": "Guardar con PII redactada (recomendado)",
    "full": "Guardar texto completo",
}
POLICY_FIELDS = (
    "alpha",
    "gap_max",
    "min_tokens",
    "annotate",
    "human_review_at",
    "rag_retry_at",
    "abstain_at",
    "abstain_message",
    "rag_url",
    "rag_max_retries",
    "rag_fallback_action",
    "on_detector_error",
    "stream_policy",
)
THRESHOLD_FIELDS = ("human_review_at", "rag_retry_at", "abstain_at")
INT_FIELDS = {"gap_max": (0, 1000), "min_tokens": (1, 10_000), "rag_max_retries": (0, 5)}
DATA_TABLES = (
    ("Generaciones", "traces"),
    ("Items de revisión", "review_items"),
    ("Datasets", "datasets"),
    ("Detectores", "detectors"),
    ("Usuarios", "users"),
    ("API keys", "api_keys"),
    ("Webhooks", "webhooks"),
)
JOB_STATE_LABELS = {"queued": "En cola", "running": "Corriendo", "done": "Terminado", "failed": "Falló"}
# Título de la página del job por estado (WCAG 2.4.2); app.js lo actualiza cuando el polling termina.
JOB_PAGE_TITLES = {
    "queued": "Calibración en cola",
    "running": "Calibración en curso",
    "done": "Calibración terminada",
    "failed": "Calibración fallida",
}


def _now_until() -> tuple[Any, Any]:
    now = utcnow()
    return now, now + RESERVATION_TTL


def _temp_password(email: str) -> str:
    """Contraseña inicial temporal: 4 grupos de 5 caracteres (~114 bits), válida para la política."""
    while True:
        groups = ["".join(secrets.choice(TEMP_PASSWORD_ALPHABET) for _ in range(5)) for _ in range(4)]
        candidate = "-".join(groups)
        try:
            check_password_policy(candidate, email=email)
        except PasswordPolicyError:
            continue
        return candidate


def _detector_auroc(report: Mapping[str, Any]) -> float | None:
    try:
        value = report["detector"]["token"]["auroc"]["value"]
    except (KeyError, TypeError):
        return None
    return value if isinstance(value, int | float) else None


def _load_json(raw: str | None) -> dict[str, Any]:
    try:
        data = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def roc_label(name: Any) -> str:
    """Etiqueta en español de una curva ROC del reporte: los ids internos (`tier:F0`, `baseline:x`) no
    llegan a la leyenda ni a la tabla. `tier:todos` -> "Todos los niveles"; `tier:F0` -> "Nivel F0"."""
    raw = str(name or "")
    if raw == "detector":
        return "Detector"
    if raw == "tier:todos":
        return "Todos los niveles"
    if raw.startswith("tier:"):
        return "Nivel " + raw.removeprefix("tier:")
    if raw.startswith("baseline:"):
        return "Baseline " + raw.removeprefix("baseline:")
    return raw


def _report_figures(report: Mapping[str, Any]) -> list[Markup]:
    """Gráficos del reporte de evaluación (SVG de `vigia.charts`, que escapa todo su texto)."""
    figures: list[Markup] = []
    curves = report.get("curves") or {}
    det = report.get("detector") or {}
    ds = report.get("dataset") or {}

    def add(svg: str, table: str) -> None:
        figures.append(Markup(charts.figure(scrollable_chart(svg), table)))  # noqa: S704 - SVG generado y escapado

    try:
        ece = det["token"]["ece"]["value"]
        if curves.get("reliability") and isinstance(ece, int | float):
            add(
                charts.reliability_svg(curves["reliability"], ece),
                charts.reliability_table(curves["reliability"], ece),
            )
    except (KeyError, TypeError, ValueError):
        pass
    roc = [c for c in curves.get("roc") or [] if isinstance(c, Mapping)]
    if roc:
        named = [dict(c, name=roc_label(c.get("name"))) for c in roc]
        try:
            add(charts.roc_svg(named), charts.roc_table(named))
        except (KeyError, TypeError, ValueError):
            pass
    pr = curves.get("pr")
    if isinstance(pr, Mapping) and isinstance(ds.get("prevalence_token"), int | float):
        try:
            pr_curves = [{"name": "Detector", **{k: pr[k] for k in ("recall", "precision", "ap")}}]
            tp = pr.get("threshold_point")
            add(
                charts.pr_svg(pr_curves, ds["prevalence_token"], threshold_point=tp),
                charts.pr_table(pr_curves, ds["prevalence_token"], threshold_point=tp),
            )
        except (KeyError, TypeError, ValueError):
            pass
    hist = curves.get("histogram")
    if isinstance(hist, Mapping):
        try:
            tau = report.get("tau")
            args = (hist["edges"], hist["correct"], hist["error"])
            add(charts.score_histogram_svg(*args, tau=tau), charts.score_histogram_table(*args, tau=tau))
        except (KeyError, TypeError, ValueError):
            pass
    lat = report.get("latency")
    if isinstance(lat, Mapping) and "baseline" in lat and "vigia" in lat:
        try:
            target = lat.get("target", 0.05)
            add(
                charts.latency_svg(lat["baseline"], lat["vigia"], target=target),
                charts.latency_table(lat["baseline"], lat["vigia"], target=target),
            )
        except (KeyError, TypeError, ValueError):
            pass
    return figures


def _report_metrics(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    det = (report.get("detector") or {}).get("token") or {}
    rows = []
    for key, label, kind in (
        ("auroc", "AUROC por token", "num"),
        ("auprc", "AUPRC por token", "num"),
        ("ece", "ECE", "num"),
        ("brier", "Brier", "num"),
        ("recall_at_fpr", "Recall con FPR 0,1", "num"),
        ("precision", "Precisión al umbral", "num"),
        ("recall", "Recall al umbral", "num"),
        ("flag_rate", "Tasa de marcado", "pct"),
    ):
        entry = det.get(key)
        if isinstance(entry, Mapping) and isinstance(entry.get("value"), int | float):
            ci = entry.get("ci") if isinstance(entry.get("ci"), list) and len(entry["ci"]) == 2 else None
            rows.append({"label": label, "value": entry["value"], "ci": ci, "kind": kind})
    return rows


def _policy_error_field(message: str) -> str | None:
    for name in sorted(POLICY_FIELDS, key=len, reverse=True):
        if name in message:
            return name
    return None


# Dependencias por permiso (a nivel de módulo: FastAPI resuelve las anotaciones por nombre).
ReviewReadP = Annotated[Principal, Depends(page("review:read"))]
ReviewWriteP = Annotated[Principal, Depends(page("review:write"))]
CalReadP = Annotated[Principal, Depends(page("calibration:read"))]
CalWriteP = Annotated[Principal, Depends(page("calibration:write"))]
CalReadApiP = Annotated[Principal, Depends(require_session("calibration:read"))]
PolicyReadP = Annotated[Principal, Depends(page("policy:read"))]
PolicyWriteP = Annotated[Principal, Depends(page("policy:write"))]
SettingsReadP = Annotated[Principal, Depends(page("settings:read"))]
SettingsWriteP = Annotated[Principal, Depends(page("settings:write"))]
UsersWriteP = Annotated[Principal, Depends(page("users:write"))]
KeysWriteP = Annotated[Principal, Depends(page("keys:write"))]
AuditP = Annotated[Principal, Depends(page("audit:read"))]
ExportP = Annotated[Principal, Depends(page("data:export"))]
DeleteP = Annotated[Principal, Depends(page("data:delete"))]


def register(router: APIRouter, runtime: Any) -> None:  # noqa: C901 - registro plano de rutas
    settings = runtime.settings
    secure = bool(getattr(settings, "secure_cookies", True))

    # ==== cola de revisión ========================================================================

    def queue_items(repo: Any, state: str | None) -> list[dict[str, Any]]:
        items = repo.list_review_items(state=state, limit=MAX_QUEUE_SCAN)
        if not items:
            return []
        marks = ",".join("?" for _ in items)
        ids = [i["id"] for i in items]
        counts = {
            r["item_id"]: r["n"]
            for r in repo.db.query(
                f"SELECT item_id, COUNT(*) AS n FROM review_decisions WHERE org_id = ? "  # noqa: S608
                f"AND item_id IN ({marks}) GROUP BY item_id",
                (repo.org_id, *ids),
            )
        }
        traces = {
            r["id"]: r
            for r in repo.db.query(
                f"SELECT id, created_at, action, response_p_error, source FROM traces "  # noqa: S608
                f"WHERE org_id = ? AND id IN ({marks})",
                (repo.org_id, *[i["trace_id"] for i in items]),
            )
        }
        return [dict(i, n_decisions=counts.get(i["id"], 0), trace=traces.get(i["trace_id"])) for i in items]

    def my_decided(repo: Any, user_id: str) -> set[str]:
        return {
            r["item_id"]
            for r in repo.db.query(
                "SELECT item_id FROM review_decisions WHERE org_id = ? AND user_id = ?",
                (repo.org_id, user_id),
            )
        }

    def available_for(item: Mapping[str, Any], user_id: str, decided: set[str], now_iso: str) -> bool:
        if item["id"] in decided or item["state"] == "completed":
            return False
        if item["state"] == "reserved":
            expired = (item["reserved_until"] or "") < now_iso
            return item["reserved_by"] == user_id or expired
        return True

    def next_item(repo: Any, user_id: str, *, after: str | None = None) -> str | None:
        now_iso = to_iso(utcnow())
        decided = my_decided(repo, user_id)
        items = [
            i
            for i in repo.list_review_items(limit=MAX_QUEUE_SCAN)
            if available_for(i, user_id, decided, now_iso) and i["id"] != after
        ]
        now, until = _now_until()
        for item in items:
            if repo.reserve_review_item(item["id"], user_id, until=until, now=now):
                return item["id"]
        return None

    def review_page(
        request: Request, p: Principal, *, status: int = 200, error: str | None = None
    ) -> HTMLResponse:
        repo = runtime.repo(p.org_id)
        state = request.query_params.get("estado") or None
        if state not in (None, *REVIEW_STATES):
            state, error = None, error or "El estado del filtro no existe."
        counts = {s: len(repo.list_review_items(state=s, limit=MAX_QUEUE_SCAN)) for s in REVIEW_STATES}
        return render(
            request,
            "review.html",
            status=status,
            items=queue_items(repo, state),
            state=state,
            counts=counts,
            cap=MAX_QUEUE_SCAN,
            state_labels=STATE_LABELS,
            error=error,
            user_id=p.user_id,
            now_iso=to_iso(utcnow()),
        )

    @router.get("/review", response_class=HTMLResponse)
    def review_list(request: Request, p: ReviewReadP) -> HTMLResponse:
        return review_page(request, p)

    @router.post("/review/next")
    def review_next(request: Request, p: ReviewWriteP) -> Response:
        item_id = next_item(runtime.repo(p.org_id), p.user_id)
        return redirect(f"/review/{item_id}") if item_id else redirect("/review", "queue_empty")

    @router.post("/review/dataset", response_class=HTMLResponse)
    async def review_dataset(request: Request, p: CalWriteP) -> Response:
        from vigia_ee.calibration import DatasetError

        form = await request.form()
        name = form_text(form, "name", max_len=200) or f"revisiones {utcnow().strftime('%Y-%m-%d')}"
        try:
            await run_in_threadpool(runtime.calibration.dataset_from_reviews, p, name)
        except DatasetError as exc:
            return review_page(request, p, status=422, error=str(exc))
        return redirect("/calibration", "dataset_uploaded")

    @router.get("/review/{item_id}", response_class=HTMLResponse)
    def review_item(request: Request, item_id: str, p: ReviewReadP) -> HTMLResponse:
        return review_item_page(request, item_id, p)

    def review_item_page(
        request: Request, item_id: str, p: Principal, *, status: int = 200, error: str | None = None
    ) -> HTMLResponse:
        repo = runtime.repo(p.org_id)
        item = repo.get_review_item(item_id)
        if item is None:
            raise not_found("El item de revisión no existe.")
        trace = repo.get_trace(item["trace_id"])
        if trace is None:
            raise not_found("La generación del item ya no existe.")
        decisions = repo.list_review_decisions(item_id) or []
        mine = next((d for d in decisions if d["user_id"] == p.user_id), None)
        completed = item["state"] == "completed"
        users = {u["id"]: u["email"] for u in repo.list_users()} if completed else {}
        # Las decisiones de otros revisores quedan ocultas hasta completar el item (sin sesgo).
        visible = (
            [dict(d, email=users.get(d["user_id"], "cuenta borrada")) for d in decisions] if completed else []
        )
        text = trace["response_text"]
        spans = trace["spans"]
        if item["span_idx"] is not None:
            spans = [s for s in spans if s["idx"] == item["span_idx"]]
        segments = segments_for(text, marks_from_spans(spans)) if text is not None else []
        now_iso = to_iso(utcnow())
        reserved_by_other = (
            item["state"] == "reserved"
            and item["reserved_by"] != p.user_id
            and (item["reserved_until"] or "") >= now_iso
        )
        reserved_by_me = (
            item["state"] == "reserved"
            and item["reserved_by"] == p.user_id
            and (item["reserved_until"] or "") >= now_iso
        )
        open_ids = [
            i["id"] for i in repo.list_review_items(limit=MAX_QUEUE_SCAN) if i["state"] != "completed"
        ]
        idx = open_ids.index(item_id) if item_id in open_ids else -1
        prev_id = open_ids[idx - 1] if idx > 0 else None
        next_id = open_ids[idx + 1] if 0 <= idx < len(open_ids) - 1 else None
        can_decide = p.can("review:write") and not completed and mine is None and not reserved_by_other
        return render(
            request,
            "review_item.html",
            status=status,
            item=item,
            trace=trace,
            text=text,
            segments=segments,
            prompt=trace["prompt_text"],
            mine=mine,
            visible=visible,
            n_decisions=len(decisions),
            completed=completed,
            reserved_by_me=reserved_by_me,
            reserved_by_other=reserved_by_other,
            can_decide=can_decide,
            prev_id=prev_id,
            next_id=next_id,
            error=error,
            labels=REVIEW_LABELS,
            label_text=LABEL_TEXT,
            state_labels=STATE_LABELS,
        )

    @router.post("/review/{item_id}/reserve")
    def review_reserve(request: Request, item_id: str, p: ReviewWriteP) -> Response:
        repo = runtime.repo(p.org_id)
        if repo.get_review_item(item_id) is None:
            raise not_found("El item de revisión no existe.")
        now, until = _now_until()
        if not repo.reserve_review_item(item_id, p.user_id, until=until, now=now):
            return review_item_page(
                request,
                item_id,
                p,
                status=409,
                error="No se pudo reservar: otro revisor lo tiene reservado, ya decidiste o está completo.",
            )
        return redirect(f"/review/{item_id}", "review_reserved")

    @router.post("/review/{item_id}/decide")
    async def review_decide(request: Request, item_id: str, p: ReviewWriteP) -> Response:
        form = await request.form()
        repo = runtime.repo(p.org_id)
        item = repo.get_review_item(item_id)
        if item is None:
            raise not_found("El item de revisión no existe.")
        label = form_text(form, "label", max_len=20)
        if label not in REVIEW_LABELS:
            return review_item_page(
                request, item_id, p, status=422, error="Elegí una etiqueta: correcto, error o incierto."
            )
        raw_note = form.get("note")
        note = raw_note.strip() if isinstance(raw_note, str) else ""
        if len(note) > MAX_NOTE_CHARS:
            return review_item_page(
                request, item_id, p, status=422, error=f"La nota admite hasta {MAX_NOTE_CHARS} caracteres."
            )
        decision = repo.add_review_decision(item_id, p.user_id, label, note=note or None)
        if decision is None:
            return review_item_page(
                request,
                item_id,
                p,
                status=409,
                error=(
                    "La decisión no se guardó: el item está completo, ya decidiste "
                    "u otro revisor lo tiene reservado."
                ),
            )
        audit.append(runtime.db, p.org_id, p, "review_decision", item_id, {"label": label})
        updated = repo.get_review_item(item_id)
        if updated is not None and updated["state"] == "completed":
            labels = [d["label"] for d in repo.list_review_decisions(item_id) or []]
            runtime.emit_event(
                p.org_id,
                "review.completed",
                {
                    "item_id": item_id,
                    "trace_id": item["trace_id"],
                    "span_idx": item["span_idx"],
                    "labels": {lab: labels.count(lab) for lab in REVIEW_LABELS},
                },
            )
        if form.get("next") == "1":
            nxt = next_item(repo, p.user_id, after=item_id)
            return redirect(f"/review/{nxt}", "review_saved") if nxt else redirect("/review", "queue_empty")
        return redirect(f"/review/{item_id}", "review_saved")

    # ==== calibración =============================================================================

    def calibration_page(
        request: Request,
        p: Principal,
        *,
        status: int = 200,
        error: str | None = None,
        field: str | None = None,
    ) -> HTMLResponse:
        repo = runtime.repo(p.org_id)
        org = repo.get_org() or {}
        detectors = []
        for row in repo.list_detectors():
            full = repo.get_detector(row["id"]) or {}
            report = _load_json(full.get("report_json"))
            detectors.append(dict(row, auroc=_detector_auroc(report)))
        detectors.reverse()
        return render(
            request,
            "calibration.html",
            status=status,
            datasets=list(reversed(repo.list_datasets())),
            jobs=runtime.calibration.list_jobs(p.org_id),
            detectors=detectors,
            active_id=org.get("active_detector_id"),
            error=error,
            error_field=field,
            max_mb=settings.max_upload_bytes / (1024 * 1024),
            job_labels=JOB_STATE_LABELS,
        )

    @router.get("/calibration", response_class=HTMLResponse)
    def calibration(request: Request, p: CalReadP) -> HTMLResponse:
        return calibration_page(request, p)

    @router.post("/calibration/datasets", response_class=HTMLResponse)
    async def calibration_upload(request: Request, p: CalWriteP) -> Response:
        from vigia_ee.calibration import DatasetError

        limit = int(settings.max_upload_bytes)
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > limit + 65_536:
            raise PanelHTTPError(413, f"El archivo supera el máximo de {fmt_num(limit / 1048576, 0)} MB.")
        form = await request.form()
        upload = form.get("file")
        if not isinstance(upload, UploadFile) or not upload.filename:
            return calibration_page(request, p, status=422, error="Elegí un archivo JSONL.", field="file")
        content = await upload.read(limit + 1)
        if len(content) > limit:
            return calibration_page(
                request,
                p,
                status=413,
                field="file",
                error=f"El archivo supera el máximo de {fmt_num(limit / 1048576, 0)} MB.",
            )
        stem = re.sub(r"\.[A-Za-z0-9]{1,8}$", "", upload.filename)[:200]
        name = form_text(form, "name", max_len=200) or clean(stem) or "dataset"
        try:
            await run_in_threadpool(runtime.calibration.upload_dataset, p, name, content)
        except DatasetError as exc:
            return calibration_page(request, p, status=422, error=str(exc), field="file")
        return redirect("/calibration", "dataset_uploaded")

    @router.post("/calibration/start", response_class=HTMLResponse)
    async def calibration_start(request: Request, p: CalWriteP) -> Response:
        from vigia_ee.calibration import DatasetError, DatasetNotFound

        form = await request.form()
        dataset_id = form_text(form, "dataset_id", max_len=64)
        raw_alphas = form_text(form, "alphas", max_len=200)
        alphas = None
        if raw_alphas:
            parts = [x for x in re.split(r"[;\s]+", raw_alphas) if x]
            parsed = [parse_float(x, lo=0.001, hi=0.5) for x in parts]
            if not parsed or any(a is None for a in parsed):
                return calibration_page(
                    request,
                    p,
                    status=422,
                    field="alphas",
                    error="Los alphas van de 0,001 a 0,5, separados por punto y coma (ej.: 0,05; 0,1; 0,2).",
                )
            alphas = sorted(set(parsed))
        seed = parse_int(form_text(form, "seed", max_len=12) or "0", lo=0, hi=2**31 - 1)
        if seed is None:
            return calibration_page(
                request, p, status=422, field="seed", error="La semilla va de 0 a 2147483647."
            )
        try:
            job_id = runtime.calibration.start(p, dataset_id, alphas=alphas, seed=seed)
        except DatasetNotFound:
            raise not_found("El dataset no existe.") from None
        except DatasetError as exc:
            return calibration_page(request, p, status=422, error=str(exc))
        return redirect(f"/calibration/jobs/{job_id}", "calibration_started")

    @router.get("/calibration/jobs/{job_id}", response_class=HTMLResponse)
    def calibration_job(request: Request, job_id: str, p: CalReadP) -> HTMLResponse:
        job = runtime.calibration.job(p.org_id, job_id)
        if job is None:
            raise not_found("El job de calibración no existe.")
        return render(
            request, "calibration_job.html", job=job, job_labels=JOB_STATE_LABELS, job_titles=JOB_PAGE_TITLES
        )

    @router.get("/calibration/jobs/{job_id}/status")
    def calibration_job_status(job_id: str, p: CalReadApiP) -> JSONResponse:
        job = runtime.calibration.job(p.org_id, job_id)
        if job is None:
            return JSONResponse({"error": "not_found", "message": "El job no existe."}, status_code=404)
        result = job.get("result") or {}
        return JSONResponse(
            {
                "state": job["state"],
                "label": JOB_STATE_LABELS.get(job["state"], job["state"]),
                "error": job.get("error"),
                "detector_id": result.get("detector_id") if isinstance(result, Mapping) else None,
            }
        )

    def detector_page(
        request: Request, p: Principal, detector_id: str, *, status: int = 200, error: str | None = None
    ) -> HTMLResponse:
        repo = runtime.repo(p.org_id)
        row = repo.get_detector(detector_id)
        if row is None:
            raise not_found("El detector no existe.")
        report = _load_json(row["report_json"])
        artifact = _load_json(row["artifact_json"])
        return render(
            request,
            "detector.html",
            status=status,
            detector=row,
            report=report,
            figures=_report_figures(report),
            metrics=_report_metrics(report),
            thresholds=artifact.get("thresholds") or [],
            model=artifact.get("model") or {},
            training=artifact.get("training") or {},
            summary=report.get("calibration") or {},
            error=error,
        )

    @router.get("/calibration/detectors/{detector_id}", response_class=HTMLResponse)
    def detector_view(request: Request, detector_id: str, p: CalReadP) -> HTMLResponse:
        return detector_page(request, p, detector_id)

    @router.post("/calibration/detectors/{detector_id}/activate", response_class=HTMLResponse)
    async def detector_activate(request: Request, detector_id: str, p: CalWriteP) -> Response:
        from vigia_ee.calibration import ActivationError, DetectorNotFound

        form = await request.form()
        if form.get("confirm") != "yes":
            return detector_page(
                request, p, detector_id, status=422, error="Confirmá la activación marcando la casilla."
            )
        try:
            await run_in_threadpool(runtime.calibration.activate, p, detector_id)
        except DetectorNotFound:
            raise not_found("El detector no existe.") from None
        except ActivationError as exc:
            return detector_page(request, p, detector_id, status=422, error=str(exc))
        return redirect("/calibration", "detector_activated")

    # ==== políticas ===============================================================================

    def policy_page(
        request: Request,
        p: Principal,
        *,
        values: Mapping[str, Any] | None = None,
        errors: Mapping[str, str] | None = None,
        status: int = 200,
    ) -> HTMLResponse:
        policy = runtime.policy_for(p.org_id)
        detector = runtime.detector_for(p.org_id)
        thresholds = list(detector.thresholds) if detector is not None else []
        current = policy.to_dict()
        shown = dict(current)
        if values:
            shown.update(values)
        return render(
            request,
            "policies.html",
            status=status,
            policy=current,
            values=shown,
            errors=dict(errors or {}),
            thresholds=thresholds,
            detector_id=getattr(detector, "id", None),
        )

    @router.get("/policies", response_class=HTMLResponse)
    def policies(request: Request, p: PolicyReadP) -> HTMLResponse:
        return policy_page(request, p)

    @router.post("/policies", response_class=HTMLResponse)
    async def policies_save(request: Request, p: PolicyWriteP) -> Response:
        form = await request.form()
        raw = {name: form_text(form, name, max_len=2100) for name in POLICY_FIELDS}
        errors: dict[str, str] = {}
        data: dict[str, Any] = {}
        alpha = parse_float(raw["alpha"], lo=0.0, hi=1.0) if raw["alpha"] else None
        if alpha is None or not 0 < alpha < 1:
            errors["alpha"] = "El alpha debe ser un número entre 0 y 1, sin incluirlos (ej.: 0,1)."
        else:
            data["alpha"] = alpha
        for name, (lo, hi) in INT_FIELDS.items():
            value = parse_int(raw[name], lo=lo, hi=hi)
            if value is None:
                errors[name] = f"Debe ser un entero entre {lo} y {hi}."
            else:
                data[name] = value
        for name in THRESHOLD_FIELDS:
            if not raw[name]:
                data[name] = None
                continue
            value = parse_float(raw[name], lo=0.0, hi=1.0)
            if value is None:
                errors[name] = "Debe ser un número entre 0 y 1, o quedar vacío para desactivar la acción."
            else:
                data[name] = value
        data["annotate"] = form.get("annotate") == "on"
        raw_msg = form.get("abstain_message")
        data["abstain_message"] = raw_msg.strip() if isinstance(raw_msg, str) else ""
        data["rag_url"] = raw["rag_url"] or None
        for name in ("rag_fallback_action", "on_detector_error", "stream_policy"):
            data[name] = raw[name]
        detector = runtime.detector_for(p.org_id)
        if detector is not None and "alpha" in data:
            min_alpha = min((t["alpha"] for t in detector.thresholds), default=None)
            if min_alpha is not None and data["alpha"] < min_alpha:
                errors["alpha"] = (
                    f"El detector activo tiene umbrales desde alpha {fmt_num(min_alpha, 2)}. "
                    "Elegí un alpha mayor o recalibrá."
                )
        shown = {**raw, "annotate": data["annotate"], "abstain_message": data["abstain_message"]}
        if errors:
            return policy_page(request, p, values=shown, errors=errors, status=422)
        try:
            await run_in_threadpool(runtime.save_policy, p, data)
        except PolicyError as exc:
            field = _policy_error_field(str(exc)) or "_form"
            return policy_page(request, p, values=shown, errors={field: str(exc)}, status=422)
        return redirect("/policies", "policy_saved")

    # ==== configuración ===========================================================================

    def settings_page(
        request: Request,
        p: Principal,
        *,
        status: int = 200,
        secret: Mapping[str, str] | None = None,
        errors: Mapping[str, str] | None = None,
    ) -> HTMLResponse:
        repo = runtime.repo(p.org_id)
        org = repo.get_org() or {}
        hooks = repo.list_webhooks() if p.can("settings:read") else []
        deliveries: dict[str, dict[str, Any]] = {}
        if hooks:
            for job in runtime.jobs.list(p.org_id, webhooks.JOB_KIND, limit=200):
                wid = (job.get("payload") or {}).get("webhook_id")
                if wid and wid not in deliveries:
                    deliveries[wid] = job
        assignable = [r for r in ROLE_RANK if can_assign_role(p.role, r)]
        return render(
            request,
            "settings.html",
            status=status,
            org=org,
            users=repo.list_users() if p.can("users:read") else [],
            keys=repo.list_api_keys() if p.can("keys:read") else [],
            hooks=hooks,
            deliveries=deliveries,
            secret=secret,
            errors=dict(errors or {}),
            roles=list(ROLE_RANK),
            assignable=assignable,
            role_labels=ROLE_LABELS,
            scopes=SCOPES,
            store_modes=STORE_TEXT_MODES,
            store_labels=STORE_TEXT_LABELS,
            event_types=webhooks.EVENT_TYPES,
            job_labels=JOB_STATE_LABELS,
            now_iso=to_iso(utcnow()),
        )

    @router.get("/settings", response_class=HTMLResponse)
    def settings_view(request: Request, p: SettingsReadP) -> HTMLResponse:
        return settings_page(request, p)

    @router.post("/settings/org", response_class=HTMLResponse)
    async def settings_org(request: Request, p: SettingsWriteP) -> Response:
        form = await request.form()
        repo = runtime.repo(p.org_id)
        org = repo.get_org()
        if org is None:
            raise not_found()
        errors: dict[str, str] = {}
        store_text = form_text(form, "store_text", max_len=20)
        if store_text not in STORE_TEXT_MODES:
            errors["store_text"] = "Elegí un modo de guardado de texto."
        days = parse_int(form_text(form, "retention_days", max_len=6), lo=1, hi=3650)
        if days is None:
            errors["retention_days"] = "La retención va de 1 a 3650 días."
        judge = form.get("external_judge_allowed") == "on"
        if errors:
            return settings_page(request, p, status=422, errors=errors)
        changes = {
            k: v
            for k, v in (
                ("store_text", store_text),
                ("retention_days", days),
                ("external_judge_allowed", judge),
            )
            if (bool(org[k]) if k == "external_judge_allowed" else org[k]) != v
        }
        if changes:
            with runtime.db.transaction():
                repo.update_org(**changes)
                audit.append(runtime.db, p.org_id, p, "settings_change", p.org_id, {"changes": changes})
        return redirect("/settings", "org_saved")

    @router.post("/settings/users", response_class=HTMLResponse)
    async def settings_invite(request: Request, p: UsersWriteP) -> Response:
        form = await request.form()
        email = form_text(form, "email", max_len=254)
        role = form_text(form, "role", max_len=20)
        if not can_assign_role(p.role, role):
            return settings_page(request, p, status=403, errors={"invite_role": "No podés asignar ese rol."})
        password = _temp_password(email)
        try:
            await run_in_threadpool(register_user, runtime.auth, p.org_id, email, password, role, actor=p)
        except PermissionError:
            return settings_page(request, p, status=403, errors={"invite_role": "No podés asignar ese rol."})
        except ValueError as exc:
            message = str(exc)
            if "registrado" in message:
                # El email es único global: un mensaje genérico no confirma cuentas de otros tenants.
                message = "No se pudo crear una cuenta con ese email."
            return settings_page(request, p, status=422, errors={"invite_email": message})
        return settings_page(
            request,
            p,
            secret={
                "kind": "password",
                "title": "Contraseña inicial temporal",
                "label": email,
                "value": password,
                "note": "Compartila por un canal seguro. La persona la cambia desde «Mi cuenta» al entrar.",
            },
        )

    @router.post("/settings/users/{user_id}/role")
    async def settings_role(request: Request, user_id: str, p: UsersWriteP) -> Response:
        form = await request.form()
        role = form_text(form, "role", max_len=20)
        repo = runtime.repo(p.org_id)
        target = repo.get_user(user_id)
        if target is None:
            raise not_found("La cuenta no existe.")
        if role not in ROLE_RANK:
            raise PanelHTTPError(422, "Rol inválido.")
        if not can_assign_role(p.role, role) or (target["role"] == "owner" and p.role != "owner"):
            raise PanelHTTPError(403, "No podés asignar ese rol a esta cuenta.")
        if target["role"] == "owner" and role != "owner" and _owners(repo, exclude=user_id) == 0:
            raise PanelHTTPError(422, "La organización necesita al menos un owner habilitado.")
        try:
            if not change_role(runtime.auth, p.org_id, user_id, role, actor=p):
                raise not_found("La cuenta no existe.")
        except PermissionError as exc:
            raise PanelHTTPError(403, str(exc)) from None
        return redirect("/settings", "role_changed")

    def _owners(repo: Any, *, exclude: str) -> int:
        return sum(
            1 for u in repo.list_users() if u["role"] == "owner" and not u["disabled"] and u["id"] != exclude
        )

    @router.post("/settings/users/{user_id}/disable")
    async def settings_disable(request: Request, user_id: str, p: UsersWriteP) -> Response:
        form = await request.form()
        disable = form.get("disabled") == "1"
        repo = runtime.repo(p.org_id)
        target = repo.get_user(user_id)
        if target is None:
            raise not_found("La cuenta no existe.")
        if user_id == p.user_id:
            raise PanelHTTPError(422, "No podés deshabilitar tu propia cuenta.")
        if target["role"] == "owner" and p.role != "owner":
            raise PanelHTTPError(403, "Solo un owner puede deshabilitar a otro owner.")
        if disable and target["role"] == "owner" and _owners(repo, exclude=user_id) == 0:
            raise PanelHTTPError(422, "La organización necesita al menos un owner habilitado.")
        with runtime.db.transaction():
            repo.update_user(user_id, disabled=disable)
            if disable:
                repo.delete_user_sessions(user_id)
            audit.append(runtime.db, p.org_id, p, "user_disable" if disable else "user_enable", user_id)
        return redirect("/settings", "user_disabled" if disable else "user_enabled")

    @router.post("/settings/keys", response_class=HTMLResponse)
    async def settings_key_create(request: Request, p: KeysWriteP) -> Response:
        form = await request.form()
        name = form_text(form, "name", max_len=100)
        scopes = [s for s in form.getlist("scopes") if isinstance(s, str)]
        raw_days = form_text(form, "expires_days", max_len=5)
        errors: dict[str, str] = {}
        if not name:
            errors["key_name"] = "Poné un nombre que identifique el uso de la key."
        if not scopes or any(s not in SCOPES for s in scopes):
            errors["key_scopes"] = "Elegí al menos un scope válido."
        expires_at = None
        if raw_days:
            days = parse_int(raw_days, lo=1, hi=3650)
            if days is None:
                errors["key_expires"] = "La expiración va de 1 a 3650 días, o vacía para no vencer."
            else:
                expires_at = utcnow() + timedelta(days=days)
        if errors:
            return settings_page(request, p, status=422, errors=errors)
        try:
            token, key_id = create_api_key(
                runtime.auth, p.org_id, name, scopes, actor=p, expires_at=expires_at
            )
        except ValueError as exc:
            return settings_page(request, p, status=422, errors={"key_name": str(exc)})
        except PermissionError as exc:
            raise PanelHTTPError(403, str(exc)) from None
        return settings_page(
            request,
            p,
            secret={
                "kind": "api_key",
                "title": "API key nueva",
                "label": f"{clean(name)} ({key_id})",
                "value": token,
                "note": "Guardala en tu gestor de secretos. Vigía no la puede volver a mostrar.",
            },
        )

    @router.post("/settings/keys/{key_id}/revoke")
    def settings_key_revoke(request: Request, key_id: str, p: KeysWriteP) -> Response:
        try:
            ok = revoke_api_key(runtime.auth, p.org_id, key_id, actor=p)
        except PermissionError as exc:
            raise PanelHTTPError(403, str(exc)) from None
        if not ok:
            raise not_found("La API key no existe o ya estaba revocada.")
        return redirect("/settings", "key_revoked")

    @router.post("/settings/webhooks", response_class=HTMLResponse)
    async def settings_webhook_create(request: Request, p: SettingsWriteP) -> Response:
        form = await request.form()
        url = form_text(form, "url", max_len=2048)
        events = [e for e in form.getlist("events") if isinstance(e, str)]
        try:
            webhook_id, secret = await run_in_threadpool(
                webhooks.create_webhook,
                runtime.db,
                p.org_id,
                url,
                events,
                p,
                allowed_hosts=tuple(getattr(settings, "webhook_allowed_hosts", ()) or ()),
            )
        except UnsafeURLError as exc:
            return settings_page(request, p, status=422, errors={"hook_url": f"Destino no permitido: {exc}."})
        except ValueError as exc:
            return settings_page(request, p, status=422, errors={"hook_events": str(exc).capitalize() + "."})
        return settings_page(
            request,
            p,
            secret={
                "kind": "webhook",
                "title": "Secreto del webhook",
                "label": webhook_id,
                "value": secret,
                "note": "Usalo para verificar la firma Standard Webhooks (header webhook-signature).",
            },
        )

    @router.post("/settings/webhooks/{webhook_id}/delete")
    def settings_webhook_delete(request: Request, webhook_id: str, p: SettingsWriteP) -> Response:
        if not webhooks.delete_webhook(runtime.db, p.org_id, webhook_id, p):
            raise not_found("El webhook no existe.")
        return redirect("/settings", "webhook_deleted")

    @router.post("/settings/webhooks/{webhook_id}/test")
    def settings_webhook_test(request: Request, webhook_id: str, p: SettingsWriteP) -> Response:
        repo = runtime.repo(p.org_id)
        hook = repo.get_webhook(webhook_id)
        if hook is None:
            raise not_found("El webhook no existe.")
        if hook["disabled"]:
            raise PanelHTTPError(422, "El webhook está deshabilitado; no se le envían eventos.")
        body = webhooks.build_body(
            "webhook.test", {"webhook_id": webhook_id, "test": True}, now=runtime.jobs.now()
        )
        runtime.jobs.enqueue(
            webhooks.JOB_KIND,
            {
                "webhook_id": webhook_id,
                "msg_id": f"msg_{new_id()}",
                "event_type": "webhook.test",
                "body": body.decode(),
            },
            org_id=p.org_id,
            max_attempts=1,
        )
        audit.append(runtime.db, p.org_id, p, "webhook_test", webhook_id)
        return redirect("/settings", "webhook_tested")

    # ==== auditoría ===============================================================================

    def audit_range(request: Request) -> tuple[Any, Any, list[str]]:
        from datetime import date

        errors = []
        bounds = []
        for key in ("desde", "hasta"):
            raw = request.query_params.get(key) or ""
            value = None
            if raw:
                try:
                    value = date.fromisoformat(raw)
                except ValueError:
                    errors.append(f"La fecha «{key}» no es válida (aaaa-mm-dd).")
            bounds.append(value)
        start = local_day_start(bounds[0]) if bounds[0] else None
        end = local_day_start(bounds[1] + timedelta(days=1)) if bounds[1] else None
        return start, end, errors

    @router.get("/audit", response_class=HTMLResponse)
    def audit_view(request: Request, p: AuditP) -> HTMLResponse:
        start, end, errors = audit_range(request)
        rows = audit.export_range(runtime.db, p.org_id, start=start, end=end)
        total = len(rows)
        shown = list(reversed(rows[-AUDIT_PAGE:]))
        check = (
            audit.verify_chain(runtime.db, p.org_id) if request.query_params.get("verificar") == "1" else None
        )
        users = {u["id"]: u["email"] for u in runtime.repo(p.org_id).list_users()}
        return render(
            request,
            "audit.html",
            rows=shown,
            total=total,
            page_size=AUDIT_PAGE,
            check=check,
            users=users,
            errors=errors,
            desde=request.query_params.get("desde") or "",
            hasta=request.query_params.get("hasta") or "",
        )

    @router.get("/audit/export")
    def audit_export(request: Request, p: AuditP) -> Response:
        start, end, errors = audit_range(request)
        if errors:
            raise PanelHTTPError(422, " ".join(errors))
        fmt = request.query_params.get("formato") or "json"
        if fmt not in ("json", "csv"):
            raise PanelHTTPError(422, "El formato debe ser json o csv.")
        rows = audit.export_range(runtime.db, p.org_id, start=start, end=end)
        audit.append(runtime.db, p.org_id, p, "audit_export", None, {"format": fmt, "rows": len(rows)})
        stamp = utcnow().strftime("%Y%m%dT%H%M%SZ")
        if fmt == "csv":
            cols = (
                "seq",
                "ts",
                "actor_user_id",
                "actor_key_id",
                "action",
                "target",
                "detail_json",
                "prev_hash",
                "hash",
            )
            body = csv_dumps(([r.get(c) for c in cols] for r in rows), header=cols)
            return Response(
                body.encode("utf-8"),
                media_type="text/csv; charset=utf-8",
                headers={"Content-Disposition": f'attachment; filename="vigia-auditoria-{stamp}.csv"'},
            )
        # `detail_json` viaja tal cual: es parte del hash y hace verificable el export (verify_rows).
        return Response(
            json.dumps({"org_id": p.org_id, "events": rows}, ensure_ascii=False, indent=1).encode("utf-8"),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="vigia-auditoria-{stamp}.json"'},
        )

    # ==== datos del tenant ========================================================================

    def data_counts(repo: Any) -> dict[str, int]:
        return {
            name: repo.db.query_one(
                f"SELECT COUNT(*) AS n FROM {table} WHERE org_id = ?",  # noqa: S608 - tabla de la lista fija
                (repo.org_id,),
            )["n"]
            for name, table in DATA_TABLES
        }

    @router.get("/data", response_class=HTMLResponse)
    def data_view(request: Request, p: ExportP) -> HTMLResponse:
        repo = runtime.repo(p.org_id)
        org = repo.get_org() or {}
        return render(request, "data.html", org=org, counts=data_counts(repo), error=None)

    @router.get("/data/export")
    def data_export(request: Request, p: ExportP) -> Response:
        data = retention.export_org(runtime.repo(p.org_id), p)
        stamp = utcnow().strftime("%Y%m%dT%H%M%SZ")
        return Response(
            json.dumps(data, ensure_ascii=False, indent=1, default=str).encode("utf-8"),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="vigia-export-{stamp}.json"'},
        )

    @router.post("/data/delete", response_class=HTMLResponse)
    async def data_delete(request: Request, p: DeleteP) -> Response:
        form = await request.form()
        repo = runtime.repo(p.org_id)
        org = repo.get_org()
        if org is None:
            raise not_found()
        typed = form_text(form, "confirm_name", max_len=200)
        if form.get("confirm") != "yes" or typed != org["name"]:
            return render(
                request,
                "data.html",
                status=422,
                org=org,
                counts=data_counts(repo),
                error="Para borrar, marcá la casilla y escribí el nombre exacto de la organización.",
            )
        audit.append(runtime.db, p.org_id, p, "org_delete", p.org_id)
        repo.delete_org()
        runtime.forget_org(p.org_id)  # saca su detector, política y retriever de memoria y de /metrics
        response = redirect("/login", "org_deleted")
        clear_cookie(response, secure=secure)
        return response
