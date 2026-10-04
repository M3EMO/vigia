"""Panel web de Vigía: páginas server-rendered con Jinja2 y un JS propio chico (ADR A-11).

Cómo se monta (lo hace `vigia_ee.app`):

    install_panel(app, runtime)          # router + página HTML para errores del panel
    # o, si la app registra los handlers por su cuenta:
    app.include_router(create_panel_router(runtime))

Reglas de este módulo:

- Cada ruta declara su permiso con `page(<permiso>)` (envuelve `require_session` y conserva la
  declaración para `undeclared_routes`) o con `public_route()`. Una página sin sesión redirige a
  `/login`; el resto de los errores se muestra como página HTML si la app instaló el handler.
- Autoescape siempre activo. El texto del modelo o del usuario pasa por `pii.safe_display` y se
  escapa en la plantilla; nunca se marca como seguro. Solo el SVG que generan `vigia.charts` (que
  escapa todo su texto) entra como `Markup`.
- CSP estricta: cero `style=`, cero `<style>`, cero handlers `on*=`. El JS vive en `static/app.js`.
- Los querystrings de filtros nunca llevan texto ni PII: solo acción, origen, fechas y números.
- Números y fechas en es-AR (coma decimal, UTC-3); la base guarda UTC.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import quote, urlencode

import anyio
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape
from markupsafe import Markup
from starlette.types import Receive, Scope, Send

from vigia import charts
from vigia.contracts import ScoredToken
from vigia.spans import build_spans
from vigia_ee import audit, pii, retention
from vigia_ee.auth import (
    PERMISSION_ATTR,
    LoginFailed,
    Principal,
    login,
    logout,
    public_route,
    require_session,
    set_password,
    set_session_cookie,
    verify_password,
)
from vigia_ee.db import TenantScopedRepository, lookup_user_by_email, parse_iso, utcnow
from vigia_ee.security import SESSION_COOKIE, RateLimitExceeded, check_fetch_site, enforce_rate_limit

PANEL_TZ = timezone(timedelta(hours=-3), "UTC-3")
TEMPLATES_DIR = Path(__file__).with_name("templates")
STATIC_DIR = Path(__file__).with_name("static")
STATIC_TYPES = {
    "app.css": "text/css; charset=utf-8",
    "app.js": "text/javascript; charset=utf-8",
    "favicon.svg": "image/svg+xml",
}
OVERHEAD_TARGET = 0.05  # meta del plan: < 5 % de latencia agregada
TREND_DAYS = 14
PERF_DAYS = 30
STATS_MAX_TRACES = 50_000  # `ponytail:` agregados en Python sobre la ventana; con más volumen, SQL por día
USAGE_MAX_TRACES = 500
MAX_PROMPT_CHARS = 8000
MAX_SYSTEM_CHARS = 4000
DEFAULT_VIEW_THRESHOLD = 0.5
SESSION_MAX_AGE = timedelta(hours=8)
PAGE_SIZE = 50

ACTIONS = ("pass", "annotate", "human_review", "rag_retry", "abstain")
ACTION_LABELS = {
    "pass": "Pasa",
    "annotate": "Anotar",
    "human_review": "Revisión humana",
    "rag_retry": "Re-consulta RAG",
    "abstain": "Abstención",
}
SOURCE_LABELS = {"api": "API", "playground": "Playground"}
ROLE_LABELS = {"owner": "Owner", "admin": "Admin", "reviewer": "Revisor", "viewer": "Lectura"}
LEVELS = ((0.8, "high", "alto"), (0.5, "medium", "medio"), (0.0, "low", "bajo"))

OK_MESSAGES = {
    "logged_out": "Cerraste la sesión.",
    "password_changed": "La contraseña cambió. Iniciá sesión con la nueva.",
    "org_deleted": "La organización y todos sus datos se borraron.",
    "review_created": "La respuesta quedó en la cola de revisión.",
    "review_saved": "La decisión quedó guardada.",
    "review_reserved": "Reservaste el item.",
    "queue_empty": "No quedan items disponibles en la cola.",
    "trace_deleted": "La generación se borró.",
    "policy_saved": "La política se guardó y aplica a los requests siguientes.",
    "calibration_started": "La calibración quedó en cola.",
    "detector_activated": "El detector quedó activo para los requests siguientes.",
    "dataset_uploaded": "El dataset se validó y quedó guardado.",
    "org_saved": "La configuración de la organización se guardó.",
    "role_changed": "El rol cambió y las sesiones de esa persona se cerraron.",
    "user_disabled": "La cuenta quedó deshabilitada y sus sesiones se cerraron.",
    "user_enabled": "La cuenta quedó habilitada.",
    "key_revoked": "La API key quedó revocada.",
    "webhook_deleted": "El webhook se borró.",
    "webhook_tested": "El evento de prueba quedó en cola de entrega.",
}

NAV: tuple[tuple[str, tuple[tuple[str, str, str], ...]], ...] = (
    (
        "Monitoreo",
        (
            ("/", "Resumen", "metrics:read"),
            ("/generations", "Generaciones", "traces:read"),
            ("/review", "Cola de revisión", "review:read"),
            ("/playground", "Playground", "playground:use"),
        ),
    ),
    (
        "Modelo",
        (
            ("/model", "Uso de expertos", "model:read"),
            ("/performance", "Rendimiento", "metrics:read"),
            ("/calibration", "Calibración", "calibration:read"),
            ("/policies", "Políticas", "policy:read"),
        ),
    ),
    (
        "Administración",
        (
            ("/settings", "Configuración", "settings:read"),
            ("/audit", "Auditoría", "audit:read"),
            ("/data", "Datos del tenant", "data:export"),
            ("/account", "Mi cuenta", "account:self"),
        ),
    ),
)


# --- formato es-AR ---------------------------------------------------------------------------------


def _finite(x: Any) -> bool:
    return isinstance(x, int | float) and not isinstance(x, bool) and math.isfinite(x)


def fmt_num(x: Any, digits: int = 3) -> str:
    """Número con separador de miles `.` y coma decimal; `s/d` si falta."""
    if not _finite(x):
        return "s/d"
    text = f"{x:,.{digits}f}"
    return text.replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def fmt_count(x: Any) -> str:
    if not _finite(x):
        return "s/d"
    return f"{int(x):,}".replace(",", ".")


def fmt_pct(x: Any, digits: int = 1) -> str:
    return "s/d" if not _finite(x) else fmt_num(x * 100, digits) + " %"


def fmt_ms(x: Any) -> str:
    return "s/d" if not _finite(x) else fmt_num(x, 1) + " ms"


def to_local(value: str | datetime | None) -> datetime | None:
    if value is None:
        return None
    dt = parse_iso(value) if isinstance(value, str) else value
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(PANEL_TZ)


def fmt_dt(value: str | datetime | None) -> str:
    """Fecha guardada en UTC mostrada en UTC-3 (`dd/mm/aaaa hh:mm`)."""
    try:
        local = to_local(value)
    except (TypeError, ValueError):
        return "s/d"
    return local.strftime("%d/%m/%Y %H:%M") if local else "s/d"


def clean(text: Any) -> str:
    """Texto de modelo o usuario apto para mostrar: sin bidi ni controles. El escape es de Jinja."""
    return pii.safe_display(text) if isinstance(text, str) else ""


HEATMAP_SCALE = 1.2  # el texto del heatmap mide 10 unidades: a 1,2 queda en 12 px
_SVG_OPEN_RE = re.compile(r'^<svg [^>]*?viewBox="0 0 (\d+(?:\.\d+)?) (\d+(?:\.\d+)?)" width="100%"')


def fixed_size_svg(svg: str, scale: float = HEATMAP_SCALE) -> str:
    """Cambia el `width="100%"` de un SVG de `vigia.charts` por un tamaño en px proporcional al viewBox.

    Con `width="100%"` el SVG no tiene tamaño intrínseco y se estira al ancho del contenedor: el
    heatmap de 176 unidades llegaba a 937 px con rótulos de 58 px. Con tamaño fijo, el contenedor
    con scroll lo muestra entero a escala `scale`.
    """
    match = _SVG_OPEN_RE.match(svg)
    if match is None:
        return svg
    w, h = (round(float(v) * scale) for v in match.groups())
    head = match.group(0).replace('width="100%"', f'width="{w}" height="{h}"')
    return head + svg[match.end() :]


_SVG_TITLE_ID_RE = re.compile(r'<title id="([^"]+)">')


def scrollable_chart(svg: str) -> str:
    """Envuelve un SVG de `vigia.charts` en una región con scroll horizontal y foco de teclado.

    El CSS fija el SVG en 30rem (escala 1, texto de 10 a 12 px). En una columna angosta la región
    hace scroll en lugar de achicar el texto. La región toma su nombre del `<title>` del SVG.
    """
    match = _SVG_TITLE_ID_RE.search(svg)
    name = f'aria-labelledby="{match.group(1)}"' if match else 'aria-label="Gráfico"'
    return f'<div class="vg-chart-scroll" tabindex="0" role="region" {name}>{svg}</div>'


def token_label(text: Any) -> str:
    """Rótulo visible de un token en la tabla de señales: nunca queda vacío.

    Un token de control, bidi o solo espacios queda vacío tras `clean`; ahí se muestran sus code
    points (`U+001D`) para que el revisor sepa qué token es. Sin texto, `·`.
    """
    shown = clean(text)
    if shown.strip():
        return shown
    if not isinstance(text, str) or not text:
        return "·"
    return " ".join(f"U+{ord(ch):04X}" for ch in text[:8]) + (" …" if len(text) > 8 else "")


def level_for(score: float) -> tuple[str, str]:
    for lo, key, label in LEVELS:
        if score >= lo:
            return key, label
    return LEVELS[-1][1], LEVELS[-1][2]


def make_environment() -> Environment:
    env = Environment(
        loader=FileSystemLoader(TEMPLATES_DIR),
        autoescape=select_autoescape(default=True, default_for_string=True),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters.update(
        num=fmt_num, count=fmt_count, pct=fmt_pct, ms=fmt_ms, dt=fmt_dt, clean=clean, token_label=token_label
    )
    env.globals.update(
        ACTION_LABELS=ACTION_LABELS,
        SOURCE_LABELS=SOURCE_LABELS,
        ROLE_LABELS=ROLE_LABELS,
        OVERHEAD_TARGET=OVERHEAD_TARGET,
    )
    return env


ENV = make_environment()


# --- errores, dependencias y render ----------------------------------------------------------------


class PanelHTTPError(HTTPException):
    """Error de una página del panel. Con `install_panel` se muestra como HTML; si no, como JSON."""


def page(permission: str) -> Callable[..., Any]:
    """Como `require_session(permission)`, pero una página sin sesión redirige al login."""
    inner = require_session(permission)

    async def dependency(request: Request) -> Principal:
        try:
            return await inner(request)
        except HTTPException as exc:
            if exc.status_code == 401 and request.method in ("GET", "HEAD"):
                target = request.url.path + (f"?{request.url.query}" if request.url.query else "")
                location = "/login?" + urlencode({"next": target})
                raise PanelHTTPError(303, "Iniciá sesión.", headers={"Location": location}) from None
            raise PanelHTTPError(exc.status_code, exc.detail, headers=exc.headers) from None

    setattr(dependency, PERMISSION_ATTR, getattr(inner, PERMISSION_ATTR))
    return dependency


def optional_page(permission: str) -> Callable[..., Any]:
    """Como `page(permission)`, pero sin sesión vigente devuelve None en lugar de un 401.

    Con sesión vigente exige CSRF y permiso igual que `page`. Sin sesión solo rechaza un request
    que el navegador declara cross-site. Sirve para acciones idempotentes como "Salir".
    """
    inner = require_session(permission)

    async def dependency(request: Request) -> Principal | None:
        try:
            return await inner(request)
        except HTTPException as exc:
            if exc.status_code != 401:
                raise PanelHTTPError(exc.status_code, exc.detail, headers=exc.headers) from None
        try:
            check_fetch_site(request)
        except HTTPException as exc:
            raise PanelHTTPError(exc.status_code, exc.detail) from None
        return None

    setattr(dependency, PERMISSION_ATTR, getattr(inner, PERMISSION_ATTR))
    return dependency


def nav_for(principal: Principal | None, path: str) -> list[dict[str, Any]]:
    if principal is None:
        return []
    groups = []
    for title, items in NAV:
        links = []
        for href, label, perm in items:
            if not principal.can(perm):
                continue
            current = path == href or (href != "/" and path.startswith(href + "/"))
            links.append({"href": href, "label": label, "current": current})
        if links:
            groups.append({"title": title, "links": links})
    return groups


def render(request: Request, template: str, *, status: int = 200, **context: Any) -> HTMLResponse:
    principal: Principal | None = getattr(request.state, "principal", None)
    session = getattr(request.state, "session", None)
    ok = request.query_params.get("ok")
    base = {
        "request": request,
        "path": request.url.path,
        "principal": principal,
        "email": getattr(session, "email", None),
        "csrf_token": getattr(request.state, "csrf_token", ""),
        "nonce": getattr(request.state, "csp_nonce", ""),
        "nav": nav_for(principal, request.url.path),
        "can": principal.can if principal is not None else (lambda _perm: False),
        "flash": OK_MESSAGES.get(ok) if ok else None,
    }
    base.update(context)
    return HTMLResponse(ENV.get_template(template).render(**base), status_code=status)


def redirect(path: str, ok: str | None = None) -> RedirectResponse:
    if ok:
        path += ("&" if "?" in path else "?") + urlencode({"ok": ok})
    return RedirectResponse(path, status_code=303)


_PATH_PARAM_RE = re.compile(r"\{(\w+)(?::\w+)?\}")


def _get_paths(routes: Sequence[Any]) -> set[str]:
    """Plantillas de ruta que aceptan GET (entra en los routers que FastAPI incluye de forma perezosa)."""
    out: set[str] = set()
    for route in routes:
        inner = getattr(route, "original_router", None)
        if inner is not None:
            out |= _get_paths(inner.routes)
        elif "GET" in (getattr(route, "methods", None) or ()):
            out.add(route.path)
    return out


def form_page_url(request: Request) -> str | None:
    """Página GET que muestra el formulario de un POST: quita segmentos de acción del final de la ruta.

    `/review/{item_id}/decide` -> `/review/{item_id}`; `/settings/users/{id}/role` -> `/settings`;
    `/policies` -> `/policies` (la misma ruta acepta GET). Se compara la plantilla de la ruta y no la
    URL concreta: `/review/next` no debe confundirse con `GET /review/{item_id}`.
    """
    template = getattr(request.scope.get("route"), "path", None)
    if not isinstance(template, str) or not template.startswith("/"):
        return None
    get_paths = _get_paths(request.app.routes)
    params = request.path_params
    segments = template.split("/")
    while segments:
        candidate = "/".join(segments) or "/"
        if candidate in get_paths:
            names = _PATH_PARAM_RE.findall(candidate)
            if all(name in params for name in names):
                return _PATH_PARAM_RE.sub(lambda m: quote(str(params[m.group(1)]), safe=""), candidate)
        segments.pop()
    return None


def section_link(request: Request) -> tuple[str, str] | None:
    """(href, texto) hacia la sección del menú que contiene la ruta pedida, si el rol la ve.

    `/generations/{id}` -> ("/generations", "Volver a generaciones"). La sección misma o una ruta fuera
    del menú no tienen a dónde volver: None (la página de error ya ofrece "Ir al resumen").
    """
    principal: Principal | None = getattr(request.state, "principal", None)
    path = request.url.path
    best: tuple[str, str] | None = None
    for _title, items in NAV:
        for href, label, perm in items:
            if href == "/" or not path.startswith(href + "/"):
                continue
            if principal is None or not principal.can(perm):
                continue
            if best is None or len(href) > len(best[0]):
                best = (href, f"Volver a {label.lower()}")
    return best


def retry_link(request: Request, status: int) -> tuple[str, str] | None:
    """(href, texto) del botón principal de la página de error; None si no hay una página GET válida.

    Un GET se reintenta tal cual solo ante un error transitorio (5xx o 429). Ante un 4xx (404, 403,
    datos inválidos) reintentar repite el mismo error: el botón lleva a la sección padre. Un POST vuelve
    a la página de su formulario (un GET sobre la ruta del POST daría 405): con la sesión vencida, por
    el login; con un 404, no hay a dónde volver.
    """
    if request.method in ("GET", "HEAD"):
        if status != 429 and status < 500:
            return section_link(request)
        # safe_next: una ruta `//host/...` sería un enlace protocol-relative a otro sitio.
        target = request.url.path + (f"?{request.url.query}" if request.url.query else "")
        return safe_next(target), "Reintentar"
    if status == 404:
        return None
    page_url = form_page_url(request)
    if page_url is None:
        return None
    if status == 401:
        return "/login?" + urlencode({"next": page_url}), "Iniciar sesión"
    return page_url, "Volver al formulario"


async def panel_error_handler(request: Request, exc: PanelHTTPError) -> Response:
    """Página HTML para errores del panel; los 3xx son redirecciones."""
    if 300 <= exc.status_code < 400:
        return RedirectResponse((exc.headers or {}).get("Location", "/"), status_code=exc.status_code)
    titles = {
        401: "Sesión vencida",
        403: "Sin permiso",
        404: "No encontrado",
        413: "Archivo demasiado grande",
        429: "Demasiados intentos",
    }
    response = render(
        request,
        "error.html",
        status=exc.status_code,
        title=titles.get(exc.status_code, "Error"),
        message=exc.detail if isinstance(exc.detail, str) else "No se pudo completar la acción.",
        request_id=getattr(request.state, "request_id", None),
        retry=retry_link(request, exc.status_code),
    )
    for key, value in (exc.headers or {}).items():
        response.headers[key] = value
    return response


def install_panel(app: FastAPI, runtime: Any) -> None:
    """Monta el panel y registra la página HTML de errores del panel."""
    app.include_router(create_panel_router(runtime))
    app.add_exception_handler(PanelHTTPError, panel_error_handler)  # type: ignore[arg-type]


def not_found(what: str = "El recurso no existe.") -> PanelHTTPError:
    # Un recurso de otro tenant también es 404 (ADR A-15).
    return PanelHTTPError(404, what)


def set_cookie(response: Response, raw_id: str, *, secure: bool) -> None:
    if secure:
        set_session_cookie(response, raw_id, max_age=SESSION_MAX_AGE)
    else:  # solo desarrollo local sin TLS (Settings.secure_cookies = False)
        response.set_cookie(
            SESSION_COOKIE,
            raw_id,
            max_age=int(SESSION_MAX_AGE.total_seconds()),
            path="/",
            httponly=True,
            samesite="strict",
        )


def clear_cookie(response: Response, *, secure: bool) -> None:
    response.delete_cookie(SESSION_COOKIE, path="/", secure=secure, httponly=True, samesite="strict")


def safe_next(value: str | None) -> str:
    """Destino post-login: solo rutas locales (evita open redirect)."""
    if not value or not value.startswith("/") or value.startswith("//") or "\\" in value:
        return "/"
    return value[:512]


def form_text(form: Mapping[str, Any], name: str, *, max_len: int = 500) -> str:
    value = form.get(name)
    return value.strip()[:max_len] if isinstance(value, str) else ""


def parse_float(raw: str, *, lo: float, hi: float) -> float | None:
    """Número en [lo, hi] desde un input (acepta coma decimal). None si es inválido."""
    try:
        value = float(raw.replace(",", "."))
    except (AttributeError, ValueError):
        return None
    return value if math.isfinite(value) and lo <= value <= hi else None


def parse_int(raw: str, *, lo: int, hi: int) -> int | None:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if lo <= value <= hi else None


# --- estadísticas de trazas (siempre filtradas por org_id) ------------------------------------------


def percentile(values: Sequence[float], q: float) -> float | None:
    data = sorted(v for v in values if _finite(v))
    if not data:
        return None
    pos = (len(data) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return data[lo] + (data[hi] - data[lo]) * (pos - lo)


def trace_window(repo: TenantScopedRepository, since: datetime) -> list[dict[str, Any]]:
    """Trazas del tenant desde `since` con conteo de tokens marcados y fragmentos.

    `ponytail:` consulta directa con `org_id` explícito porque el repositorio no expone agregados;
    pedido de contrato: mover a `TenantScopedRepository.trace_stats`.
    """
    return repo.db.query(
        "SELECT t.id, t.created_at, t.action, t.source, t.latency_ms, t.overhead_ms, t.calibration_id, "
        "t.n_tokens, t.degraded, t.response_p_error, "
        "(SELECT COUNT(*) FROM trace_tokens k WHERE k.trace_id = t.id AND k.org_id = t.org_id "
        " AND k.flag = 1) AS n_flagged, "
        "(SELECT COUNT(*) FROM spans s WHERE s.trace_id = t.id AND s.org_id = t.org_id) AS n_spans "
        "FROM traces t WHERE t.org_id = ? AND t.created_at >= ? ORDER BY t.created_at, t.id LIMIT ?",
        (repo.org_id, since.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ"), STATS_MAX_TRACES),
    )


def overhead_ratio(row: Mapping[str, Any]) -> float | None:
    lat, ovh = row.get("latency_ms"), row.get("overhead_ms")
    if not (_finite(lat) and _finite(ovh)) or lat - ovh <= 0:
        return None
    return ovh / (lat - ovh)


def overhead_percentiles(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, float | None], int]:
    """Percentiles del overhead por request (overhead / latencia sin Vigía) y cuántos requests entran.

    Es la única definición de "Overhead pXX" del panel: el resumen y Rendimiento la comparten. No es
    el cociente de percentiles de latencia (p95 con / p95 sin - 1), que da otro número.
    """
    ratios = [x for x in (overhead_ratio(r) for r in rows) if x is not None]
    qs = {"p50": 0.5, "p95": 0.95, "p99": 0.99}
    return {k: percentile(ratios, q) for k, q in qs.items()}, len(ratios)


def summary_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """KPIs y tendencia diaria (días locales UTC-3 con tráfico; un día sin tráfico no se inventa)."""
    total = len(rows)
    by_action = {a: 0 for a in ACTIONS}
    flagged = degraded = 0
    for r in rows:
        by_action[r["action"]] = by_action.get(r["action"], 0) + 1
        flagged += 1 if r["n_spans"] else 0
        degraded += 1 if r["degraded"] else 0
    lat = [r["latency_ms"] for r in rows if _finite(r["latency_ms"])]
    ovh, _n_measured = overhead_percentiles(rows)
    days: dict[date, list[int]] = {}
    recal_days: list[date] = []
    prev_cal: str | None = None
    for r in rows:
        day = to_local(r["created_at"]).date()
        tok = days.setdefault(day, [0, 0])
        tok[0] += int(r["n_flagged"] or 0)
        tok[1] += int(r["n_tokens"] or 0)
        cal = r["calibration_id"]
        if cal and prev_cal and cal != prev_cal and day not in recal_days:
            recal_days.append(day)
        prev_cal = cal or prev_cal
    trend_days = [d for d in sorted(days) if days[d][1] > 0]
    labels = [d.strftime("%d/%m") for d in trend_days]
    rates = [days[d][0] / days[d][1] for d in trend_days]
    recal_idx = [trend_days.index(d) for d in recal_days if d in trend_days]
    ovh_p95 = ovh["p95"]
    return {
        "total": total,
        "flagged_pct": flagged / total if total else None,
        "degraded": degraded,
        "by_action": [
            {"action": a, "label": ACTION_LABELS.get(a, a), "count": n, "pct": n / total if total else None}
            for a, n in by_action.items()
        ],
        "lat_p50": percentile(lat, 0.5),
        "lat_p95": percentile(lat, 0.95),
        "ovh_p50": ovh["p50"],
        "ovh_p95": ovh_p95,
        "ovh_alert": ovh_p95 is not None and ovh_p95 > OVERHEAD_TARGET,
        "trend_labels": labels,
        "trend_rates": rates,
        "trend_recal": recal_idx,
    }


def latency_percentiles(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, float], dict[str, float]] | None:
    """Percentiles de latencia sin Vigía (latencia - overhead) y con Vigía, en ms."""
    pairs = [
        (r["latency_ms"] - r["overhead_ms"], r["latency_ms"])
        for r in rows
        if _finite(r["latency_ms"]) and _finite(r["overhead_ms"]) and r["latency_ms"] - r["overhead_ms"] > 0
    ]
    if not pairs:
        return None
    base = [b for b, _ in pairs]
    full = [v for _, v in pairs]
    qs = {"p50": 0.5, "p95": 0.95, "p99": 0.99}
    return (
        {k: percentile(base, q) for k, q in qs.items()},  # type: ignore[misc]
        {k: percentile(full, q) for k, q in qs.items()},  # type: ignore[misc]
    )


def usage_matrix(repo: TenantScopedRepository) -> tuple[list[list[float]], list[str], list[str], int] | None:
    """Uso capa x experto agregado de las trazas recientes, normalizado por capa (fracción)."""
    rows = repo.db.query(
        "SELECT usage_json FROM traces WHERE org_id = ? AND usage_json IS NOT NULL "
        "ORDER BY created_at DESC LIMIT ?",
        (repo.org_id, USAGE_MAX_TRACES),
    )
    totals: list[list[float]] | None = None
    layers: list[Any] = []
    used = 0
    for row in rows:
        try:
            usage = json.loads(row["usage_json"])
            counts = [[float(v) for v in layer] for layer in usage["counts"]]
            lay = list(usage.get("layers") or range(len(counts)))
        except (TypeError, ValueError, KeyError, json.JSONDecodeError):
            continue
        if not counts or len(lay) != len(counts) or any(len(c) != len(counts[0]) for c in counts):
            continue
        if totals is None:
            totals, layers = [list(c) for c in counts], lay
        elif lay != layers or len(counts[0]) != len(totals[0]):
            continue  # traza de otra arquitectura: no se mezcla
        else:
            totals = [[a + b for a, b in zip(t, c, strict=True)] for t, c in zip(totals, counts, strict=True)]
        used += 1
    if totals is None:
        return None
    matrix = [[v / s if (s := sum(row)) > 0 else 0.0 for v in row] for row in totals]
    return matrix, [str(x) for x in layers], [str(j) for j in range(len(totals[0]))], used


# --- fragmentos del detalle -------------------------------------------------------------------------


@dataclass
class Fragment:
    n: int
    start: int
    end: int
    score: float
    level: str
    level_label: str
    tokens: list[dict[str, Any]]


def scored_tokens(tokens: Sequence[Mapping[str, Any]], threshold: float) -> list[ScoredToken]:
    out = []
    for t in tokens:
        p = t.get("p_error")
        out.append(
            ScoredToken(
                index=int(t.get("idx", t.get("index", 0))),
                text=t.get("text") or "",
                start=int(t["start"]),
                end=int(t["end"]),
                p_error=p if _finite(p) else None,
                flag=_finite(p) and p >= threshold,
            )
        )
    return out


def fragments_for(
    text: str, tokens: Sequence[Mapping[str, Any]], threshold: float, *, gap_max: int, min_tokens: int
) -> list[Fragment]:
    """Fragmentos dudosos al umbral de vista (no cambia la política)."""
    st = scored_tokens(tokens, threshold)
    by_idx = {int(t.get("idx", t.get("index", 0))): t for t in tokens}
    out = []
    for n, sp in enumerate(build_spans(text, st, gap_max=gap_max, min_tokens=min_tokens), 1):
        level, label = level_for(sp.score)
        toks = [by_idx[i] for i in range(sp.token_start, sp.token_end) if i in by_idx]
        out.append(Fragment(n, sp.start, sp.end, sp.score, level, label, toks))
    return out


def segments_for(text: str, marks: Sequence[Any]) -> list[dict[str, Any]]:
    """Corta el texto en tramos planos y marcados. Cada tramo pasa por `safe_display` por separado."""
    out: list[dict[str, Any]] = []
    pos = 0
    for m in sorted(marks, key=lambda m: m.start):
        start, end = max(pos, min(m.start, len(text))), min(m.end, len(text))
        if end <= start:
            continue
        if start > pos:
            out.append({"text": clean(text[pos:start]), "mark": None})
        out.append({"text": clean(text[start:end]), "mark": m})
        pos = end
    if pos < len(text):
        out.append({"text": clean(text[pos:]), "mark": None})
    return out


@dataclass
class SimpleMark:
    n: int
    start: int
    end: int
    score: float
    level: str
    level_label: str


def marks_from_spans(spans: Sequence[Mapping[str, Any]]) -> list[SimpleMark]:
    out = []
    for n, sp in enumerate(spans, 1):
        score = float(sp["score"])
        level, label = level_for(score)
        out.append(SimpleMark(n, int(sp["start"]), int(sp["end"]), score, level, label))
    return out


# --- filtros de generaciones ------------------------------------------------------------------------


@dataclass
class TraceFilters:
    action: str | None = None
    source: str | None = None
    since: date | None = None
    until: date | None = None
    min_p: float | None = None
    degraded: bool | None = None

    def query(self) -> dict[str, str]:
        """Querystring canónico: solo campos estructurados, nunca texto libre."""
        q: dict[str, str] = {}
        if self.action:
            q["accion"] = self.action
        if self.source:
            q["origen"] = self.source
        if self.since:
            q["desde"] = self.since.isoformat()
        if self.until:
            q["hasta"] = self.until.isoformat()
        if self.min_p is not None:
            q["p_min"] = f"{self.min_p:g}"
        if self.degraded is not None:
            q["degradado"] = "si" if self.degraded else "no"
        return q

    @property
    def active(self) -> bool:
        return bool(self.query())


def local_day_start(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, tzinfo=PANEL_TZ).astimezone(UTC)


def parse_filters(params: Mapping[str, str]) -> tuple[TraceFilters, list[str]]:
    f = TraceFilters()
    errors: list[str] = []
    action = params.get("accion") or ""
    if action:
        if action in ACTIONS:
            f.action = action
        else:
            errors.append("La acción del filtro no existe.")
    source = params.get("origen") or ""
    if source:
        if source in SOURCE_LABELS:
            f.source = source
        else:
            errors.append("El origen del filtro no existe.")
    for key, attr in (("desde", "since"), ("hasta", "until")):
        raw = params.get(key) or ""
        if raw:
            try:
                setattr(f, attr, date.fromisoformat(raw))
            except ValueError:
                errors.append(f"La fecha «{key}» no es válida (aaaa-mm-dd).")
    if f.since and f.until and f.until < f.since:
        errors.append("La fecha «hasta» es anterior a «desde».")
        f.until = None
    raw_p = params.get("p_min") or ""
    if raw_p:
        f.min_p = parse_float(raw_p, lo=0.0, hi=1.0)
        if f.min_p is None:
            errors.append("El P(error) mínimo debe estar entre 0 y 1.")
    deg = params.get("degradado") or ""
    if deg in ("si", "no"):
        f.degraded = deg == "si"
    elif deg:
        errors.append("El filtro «degradado» acepta sí o no.")
    return f, errors


# --- playground -------------------------------------------------------------------------------------


@dataclass
class PlaygroundInput:
    prompt: str
    system: str
    max_tokens: int
    temperature: float

    def messages(self) -> list[dict[str, str]]:
        msgs = [{"role": "system", "content": self.system}] if self.system else []
        return msgs + [{"role": "user", "content": self.prompt}]


def parse_playground(data: Mapping[str, Any], settings: Any) -> tuple[PlaygroundInput, dict[str, str]]:
    errors: dict[str, str] = {}
    prompt = data.get("prompt")
    prompt = prompt.strip() if isinstance(prompt, str) else ""
    if not prompt:
        errors["prompt"] = "Escribí un mensaje."
    elif len(prompt) > MAX_PROMPT_CHARS:
        errors["prompt"] = f"El mensaje admite hasta {fmt_count(MAX_PROMPT_CHARS)} caracteres."
    system = data.get("system")
    system = system.strip() if isinstance(system, str) else ""
    if len(system) > MAX_SYSTEM_CHARS:
        errors["system"] = f"El mensaje de sistema admite hasta {fmt_count(MAX_SYSTEM_CHARS)} caracteres."
    limit = int(settings.max_tokens_limit)
    raw_mt = data.get("max_tokens", settings.default_max_tokens)
    max_tokens = parse_int(str(raw_mt), lo=1, hi=limit)
    if max_tokens is None:
        errors["max_tokens"] = f"Los tokens máximos van de 1 a {fmt_count(limit)}."
        max_tokens = int(settings.default_max_tokens)
    raw_t = data.get("temperature", 0.0)
    temperature = parse_float(str(raw_t), lo=0.0, hi=2.0)
    if temperature is None:
        errors["temperature"] = "La temperatura va de 0 a 2."
        temperature = 0.0
    return PlaygroundInput(prompt, system, max_tokens, temperature), errors


def playground_quota_wait(runtime: Any, principal: Principal) -> int | None:
    """Consume la cuota del tenant (la misma que la API, P-33); segundos de espera si no alcanza.

    El playground usa el modelo compartido igual que la API: sin esta cuota, una org con un
    revisor generaba sin límite y salteaba la de su tenant.
    """
    try:
        enforce_rate_limit(runtime.auth.tenant_limiter, f"org:{principal.org_id}", scope="tenant")
    except RateLimitExceeded as exc:
        return max(1, math.ceil(exc.retry_after))
    return None


def quota_message(retry_after: int) -> str:
    return (
        "Tu organización alcanzó el límite de generaciones por minuto. "
        f"Esperá {fmt_count(retry_after)} s y reintentá."
    )


def run_playground(runtime: Any, principal: Principal, inp: PlaygroundInput) -> dict[str, Any]:
    """Consume el stream completo (camino sin JavaScript, AT-E07)."""
    tokens: list[dict[str, Any]] = []
    done: dict[str, Any] | None = None
    error: str | None = None
    for event in runtime.playground_stream(
        principal, inp.messages(), max_tokens=inp.max_tokens, temperature=inp.temperature
    ):
        kind = event.get("type")
        if kind == "token":
            tokens.append(event)
        elif kind == "done":
            done = event
        elif kind == "error":
            error = str(event.get("message") or "La generación falló.")
    return {"tokens": tokens, "done": done, "error": error}


def sse_events(runtime: Any, principal: Principal, inp: PlaygroundInput) -> Iterator[str]:
    stream = runtime.playground_stream(
        principal, inp.messages(), max_tokens=inp.max_tokens, temperature=inp.temperature
    )
    try:
        for event in stream:
            yield "data: " + json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n\n"
    except Exception:  # noqa: BLE001 - el detalle va al log del runtime, no al navegador
        yield "data: " + json.dumps({"type": "error", "message": "La generación falló."}) + "\n\n"
    finally:
        close = getattr(stream, "close", None)
        if callable(close):
            close()


class ClosingStreamingResponse(StreamingResponse):
    """`StreamingResponse` sobre un iterador sync que lo cierra apenas termina la respuesta.

    Starlette no llama `close()` sobre el iterador sync cuando el cliente se desconecta: el
    generador queda suspendido en un `yield` y retiene el lock del modelo (compartido por todos
    los tenants) hasta que lo finaliza el GC, minutos después. Acá el cierre corre siempre, en un
    hilo y protegido de la cancelación que dispara la desconexión. Si la respuesta terminó
    normal, `close()` no hace nada.
    """

    def __init__(self, content: Iterator[str], **kwargs: Any) -> None:
        super().__init__(content, **kwargs)
        self._sync_iterator = content

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            close = getattr(self._sync_iterator, "close", None)
            if callable(close):
                with anyio.CancelScope(shield=True):
                    await anyio.to_thread.run_sync(close)


# Dependencias por permiso (a nivel de módulo: FastAPI resuelve las anotaciones por nombre).
MetricsP = Annotated[Principal, Depends(page("metrics:read"))]
TracesP = Annotated[Principal, Depends(page("traces:read"))]
ModelP = Annotated[Principal, Depends(page("model:read"))]
PlayP = Annotated[Principal, Depends(page("playground:use"))]
SelfP = Annotated[Principal, Depends(page("account:self"))]
OptionalSelfP = Annotated[Principal | None, Depends(optional_page("account:self"))]
ReviewWriteP = Annotated[Principal, Depends(page("review:write"))]
DeleteP = Annotated[Principal, Depends(page("data:delete"))]
PlayApiP = Annotated[Principal, Depends(require_session("playground:use"))]


# --- router -----------------------------------------------------------------------------------------


def _static_payload(name: str) -> bytes:
    if name == "charts.css":
        return charts.CHART_CSS.encode("utf-8")
    return (STATIC_DIR / name).read_bytes()


def create_panel_router(runtime: Any) -> APIRouter:
    """Rutas del panel. `runtime` es un `vigia_ee.runtime.Runtime` (o algo con su interfaz)."""
    from vigia_ee import panel_admin  # import diferido: panel_admin importa helpers de este módulo

    router = APIRouter()
    settings = runtime.settings
    secure = bool(getattr(settings, "secure_cookies", True))
    static_cache = {name: _static_payload(name) for name in (*STATIC_TYPES, "charts.css")}
    static_types = {**STATIC_TYPES, "charts.css": "text/css; charset=utf-8"}

    # -- estáticos --

    @router.get("/static/{name}", dependencies=[Depends(public_route())], include_in_schema=False)
    def static_file(name: str) -> Response:
        if name not in static_cache:
            raise HTTPException(404, "Archivo inexistente.")
        return Response(
            static_cache[name],
            media_type=static_types[name],
            headers={"Cache-Control": "public, max-age=300"},
        )

    # -- login / logout / cuenta --

    @router.get("/login", dependencies=[Depends(public_route())], response_class=HTMLResponse)
    def login_form(request: Request, next: str = "/") -> HTMLResponse:  # noqa: A002
        return render(request, "login.html", next=safe_next(next), error=None, email_value="")

    @router.post("/login", dependencies=[Depends(public_route())], response_class=HTMLResponse)
    async def login_submit(request: Request) -> Response:
        check_fetch_site(request)
        form = await request.form()
        email = form_text(form, "email", max_len=254)
        password = form.get("password")
        password = password if isinstance(password, str) else ""
        target = safe_next(form_text(form, "next", max_len=512))
        ip = request.client.host if request.client else "desconocida"
        try:
            result = await run_in_threadpool(
                login,
                runtime.auth,
                email,
                password,
                ip=ip,
                previous_session_id=request.cookies.get(SESSION_COOKIE),
            )
        except LoginFailed as exc:
            return render(request, "login.html", status=401, next=target, error=str(exc), email_value=email)
        except RateLimitExceeded as exc:
            resp = render(
                request,
                "login.html",
                status=429,
                next=target,
                error="Demasiados intentos fallidos desde esta red. Esperá unos minutos.",
                email_value=email,
            )
            resp.headers.update(exc.headers or {})
            return resp
        response = RedirectResponse(target, status_code=303)
        set_cookie(response, result.session_id, secure=secure)
        return response

    @router.post("/logout")
    def logout_submit(request: Request, p: OptionalSelfP) -> Response:
        # Idempotente: con la sesión vencida no hay nada que validar; igual se borra la cookie.
        if p is not None:
            logout(runtime.auth, request.cookies.get(SESSION_COOKIE))
        response = redirect("/login", "logged_out")
        clear_cookie(response, secure=secure)
        return response

    @router.get("/account", response_class=HTMLResponse)
    def account(request: Request, p: SelfP) -> HTMLResponse:
        return render(request, "account.html", errors={})

    @router.post("/account/password", response_class=HTMLResponse)
    async def account_password(request: Request, p: SelfP) -> Response:
        form = await request.form()
        current = form.get("current_password")
        new = form.get("new_password")
        confirm = form.get("confirm_password")
        current = current if isinstance(current, str) else ""
        new = new if isinstance(new, str) else ""
        confirm = confirm if isinstance(confirm, str) else ""
        errors: dict[str, str] = {}
        session = request.state.session
        user = lookup_user_by_email(runtime.db, session.email)
        ok = user is not None and await run_in_threadpool(verify_password, current, user["password_hash"])
        if not ok:
            errors["current_password"] = "La contraseña actual no coincide."  # noqa: S105
        elif new != confirm:
            errors["confirm_password"] = "La confirmación no coincide con la nueva."  # noqa: S105
        else:
            try:
                await run_in_threadpool(set_password, runtime.auth, p.org_id, p.user_id, new, actor=p)
            except ValueError as exc:  # PasswordPolicyError
                errors["new_password"] = str(exc)
        if errors:
            return render(request, "account.html", status=422, errors=errors)
        response = redirect("/login", "password_changed")
        clear_cookie(response, secure=secure)
        return response

    # -- resumen --

    @router.get("/", response_class=HTMLResponse)
    def overview(request: Request, p: MetricsP) -> HTMLResponse:
        repo = runtime.repo(p.org_id)
        since = utcnow() - timedelta(days=TREND_DAYS)
        stats = summary_stats(trace_window(repo, since))
        trend = None
        if stats["trend_rates"]:
            args = (stats["trend_labels"], stats["trend_rates"])
            kw = {"recalibrations": stats["trend_recal"], "title": "Tasa de tokens marcados por día"}
            trend = Markup(  # noqa: S704 - SVG generado y escapado por vigia.charts
                scrollable_chart(charts.flag_trend_svg(*args, **kw)) + charts.flag_trend_table(*args, **kw)
            )
        pending = len(repo.list_review_items(state="pending", limit=200)) + len(
            repo.list_review_items(state="needs_other", limit=200)
        )
        return render(
            request,
            "overview.html",
            stats=stats,
            trend=trend,
            pending=pending,
            calibrated=runtime.detector_for(p.org_id) is not None,
            days=TREND_DAYS,
        )

    # -- generaciones --

    @router.get("/generations", response_class=HTMLResponse)
    def generations(request: Request, p: TracesP) -> HTMLResponse:
        repo = runtime.repo(p.org_id)
        filters, errors = parse_filters(request.query_params)
        cursor = request.query_params.get("cursor") or None
        criteria = {
            "action": filters.action,
            "source": filters.source,
            "since": local_day_start(filters.since) if filters.since else None,
            "until": local_day_start(filters.until + timedelta(days=1)) if filters.until else None,
            "min_p_error": filters.min_p,
            "degraded": filters.degraded,
        }
        try:
            rows, next_cursor = repo.list_traces(limit=PAGE_SIZE, cursor=cursor, **criteria)
        except ValueError:
            errors.append("El enlace de paginación venció o es inválido; se muestra la primera página.")
            cursor = None
            rows, next_cursor = repo.list_traces(limit=PAGE_SIZE, **criteria)
        counts = {}
        if rows:
            marks = ",".join("?" for _ in rows)
            for r in repo.db.query(
                f"SELECT trace_id, COUNT(*) AS n FROM spans WHERE org_id = ? AND trace_id IN ({marks}) "  # noqa: S608
                "GROUP BY trace_id",
                (p.org_id, *[r["id"] for r in rows]),
            ):
                counts[r["trace_id"]] = r["n"]
        query = filters.query()
        next_url = "/generations?" + urlencode({**query, "cursor": next_cursor}) if next_cursor else None
        first_url = "/generations" + ("?" + urlencode(query) if query else "")
        return render(
            request,
            "generations.html",
            rows=rows,
            span_counts=counts,
            filters=filters,
            errors=errors,
            next_url=next_url,
            first_url=first_url if cursor else None,
            actions=ACTIONS,
            any_traces=bool(rows) or filters.active or bool(cursor) or _has_traces(repo),
        )

    def _has_traces(repo: TenantScopedRepository) -> bool:
        return bool(repo.list_traces(limit=1)[0])

    @router.get("/generations/{trace_id}", response_class=HTMLResponse)
    def generation_detail(request: Request, trace_id: str, p: TracesP) -> HTMLResponse:
        repo = runtime.repo(p.org_id)
        trace = repo.get_trace(trace_id)
        if trace is None:
            raise not_found("La generación no existe.")
        policy = runtime.policy_for(p.org_id)
        default_t = trace["tau"] if _finite(trace["tau"]) else DEFAULT_VIEW_THRESHOLD
        raw_t = request.query_params.get("umbral")
        threshold = parse_float(raw_t, lo=0.0, hi=1.0) if raw_t else default_t
        threshold_error = None
        if threshold is None:
            threshold, threshold_error = default_t, "El umbral de vista debe estar entre 0 y 1."
        text = trace["response_text"]
        calibrated = bool(trace["calibration_id"]) and any(_finite(t["p_error"]) for t in trace["tokens"])
        frags: list[Fragment] = []
        if text is not None and calibrated:
            frags = fragments_for(
                text, trace["tokens"], threshold, gap_max=policy.gap_max, min_tokens=policy.min_tokens
            )
        segments = segments_for(text, frags) if text is not None else []
        stored_spans = [
            {"idx": s["idx"], "label": clean((text or "")[s["start"] : s["end"]])[:60], "score": s["score"]}
            for s in trace["spans"]
        ]
        reviews = repo.db.query(
            "SELECT id, state, span_idx FROM review_items WHERE org_id = ? AND trace_id = ? "
            "ORDER BY created_at",
            (p.org_id, trace_id),
        )
        return render(
            request,
            "detail.html",
            trace=trace,
            text=text,
            prompt=trace["prompt_text"],
            segments=segments,
            fragments=frags,
            calibrated=calibrated,
            threshold=threshold,
            default_threshold=default_t,
            threshold_error=threshold_error,
            stored_spans=stored_spans,
            reviews=reviews,
            min_reviewers=int(getattr(settings, "review_min_reviewers", 1)),
        )

    @router.post("/generations/{trace_id}/review")
    async def generation_to_review(request: Request, trace_id: str, p: ReviewWriteP) -> Response:
        form = await request.form()
        repo = runtime.repo(p.org_id)
        trace = repo.get_trace(trace_id)
        if trace is None:
            raise not_found("La generación no existe.")
        target = form_text(form, "target", max_len=20) or "response"
        span_idx: int | None = None
        if target != "response":
            span_idx = parse_int(target, lo=0, hi=1_000_000)
            if span_idx is None or span_idx not in {s["idx"] for s in trace["spans"]}:
                raise PanelHTTPError(422, "El fragmento elegido no existe en esta generación.")
        min_reviewers = parse_int(form_text(form, "min_reviewers", max_len=3), lo=1, hi=5)
        if min_reviewers is None:
            raise PanelHTTPError(422, "La cantidad mínima de revisores va de 1 a 5.")
        item_id = repo.create_review_item(trace_id, span_idx=span_idx, min_reviewers=min_reviewers)
        if item_id is None:
            raise not_found("La generación no existe.")
        audit.append(
            runtime.db, p.org_id, p, "review_enqueue", item_id, {"trace_id": trace_id, "span_idx": span_idx}
        )
        return redirect(f"/generations/{trace_id}", "review_created")

    @router.post("/generations/{trace_id}/delete")
    async def generation_delete(request: Request, trace_id: str, p: DeleteP) -> Response:
        form = await request.form()
        if form.get("confirm") != "yes":
            raise PanelHTTPError(422, "Confirmá el borrado marcando la casilla.")
        if not retention.delete_trace_request(runtime.repo(p.org_id), trace_id, p):
            raise not_found("La generación no existe.")
        return redirect("/generations", "trace_deleted")

    # -- playground --

    @router.get("/playground", response_class=HTMLResponse)
    def playground(request: Request, p: PlayP) -> HTMLResponse:
        return render(
            request,
            "playground.html",
            values={"prompt": "", "system": "", "max_tokens": settings.default_max_tokens, "temperature": 0},
            errors={},
            result=None,
            settings=settings,
        )

    @router.post("/playground", response_class=HTMLResponse)
    async def playground_submit(request: Request, p: PlayP) -> HTMLResponse:
        form = await request.form()
        data = {k: form.get(k) for k in ("prompt", "system", "max_tokens", "temperature")}
        inp, errors = parse_playground(data, settings)
        values = {
            "prompt": inp.prompt or form_text(form, "prompt", max_len=MAX_PROMPT_CHARS),
            "system": inp.system,
            "max_tokens": inp.max_tokens,
            "temperature": inp.temperature,
        }
        if errors:
            return render(
                request,
                "playground.html",
                status=422,
                values=values,
                errors=errors,
                result=None,
                settings=settings,
            )
        retry_after = playground_quota_wait(runtime, p)
        if retry_after is not None:
            resp = render(
                request,
                "playground.html",
                status=429,
                values=values,
                errors={},
                result=None,
                stream_error=quota_message(retry_after),
                settings=settings,
            )
            resp.headers["Retry-After"] = str(retry_after)
            return resp
        out = await run_in_threadpool(run_playground, runtime, p, inp)
        result = None
        if out["done"]:
            done = out["done"]
            text = done.get("text") or "".join(t.get("text", "") for t in out["tokens"])
            result = {
                "done": done,
                "segments": segments_for(text, marks_from_spans(done.get("spans") or [])),
                "n_tokens": len(out["tokens"]),
                "n_flagged": sum(1 for t in out["tokens"] if t.get("flag")),
            }
        status = 200 if out["error"] is None else 502
        return render(
            request,
            "playground.html",
            status=status,
            values=values,
            errors={},
            result=result,
            stream_error=out["error"],
            settings=settings,
        )

    @router.post("/playground/stream")
    async def playground_stream(request: Request, p: PlayApiP) -> Response:
        try:
            data = await request.json()
        except (ValueError, UnicodeDecodeError):
            return JSONResponse({"error": "bad_request", "message": "JSON inválido."}, status_code=400)
        if not isinstance(data, dict):
            return JSONResponse({"error": "bad_request", "message": "Se espera un objeto JSON."}, 400)
        inp, errors = parse_playground(data, settings)
        if errors:
            return JSONResponse(
                {"error": "validation_error", "message": "Datos inválidos.", "fields": errors}, 422
            )
        retry_after = playground_quota_wait(runtime, p)
        if retry_after is not None:
            return JSONResponse(
                {"error": "rate_limited", "message": quota_message(retry_after), "retry_after": retry_after},
                status_code=429,
                headers={"Retry-After": str(retry_after)},
            )
        return ClosingStreamingResponse(
            sse_events(runtime, p, inp),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    # -- modelo y rendimiento --

    @router.get("/model", response_class=HTMLResponse)
    def model(request: Request, p: ModelP) -> HTMLResponse:
        repo = runtime.repo(p.org_id)
        agg = usage_matrix(repo)
        heatmap = table = None
        n_traces = 0
        if agg is not None:
            matrix, rows, cols, n_traces = agg
            kw = {"value_label": "fracción de selecciones", "title": "Uso por capa y experto"}
            heatmap = Markup(fixed_size_svg(charts.heatmap_svg(matrix, rows, cols, **kw)))  # noqa: S704
            table = Markup(charts.heatmap_table(matrix, rows, cols, **kw))  # noqa: S704
        info = runtime.model_info
        caps = None
        try:
            caps = runtime.backend.capabilities()
        except Exception:  # noqa: BLE001 - el panel sigue sin capacidades
            caps = None
        return render(
            request, "model.html", heatmap=heatmap, table=table, n_traces=n_traces, info=info, caps=caps
        )

    @router.get("/performance", response_class=HTMLResponse)
    def performance(request: Request, p: MetricsP) -> HTMLResponse:
        repo = runtime.repo(p.org_id)
        rows = trace_window(repo, utcnow() - timedelta(days=PERF_DAYS))
        lat = latency_percentiles(rows)
        figure = None
        overhead, n_measured = overhead_percentiles(rows)
        if lat is not None:
            base, full = lat
            kw = {"target": OVERHEAD_TARGET, "unit": "ms", "title": "Latencia por request"}
            figure = Markup(  # noqa: S704
                scrollable_chart(charts.latency_svg(base, full, **kw))
                + charts.latency_table(base, full, **kw)
            )
        tps = [
            r["n_tokens"] / (r["latency_ms"] / 1000.0)
            for r in rows
            if _finite(r["latency_ms"]) and r["latency_ms"] > 0 and r["n_tokens"]
        ]
        return render(
            request,
            "performance.html",
            figure=figure,
            overhead=overhead,
            n=n_measured,
            tps_p50=percentile(tps, 0.5),
            days=PERF_DAYS,
        )

    panel_admin.register(router, runtime)
    return router
