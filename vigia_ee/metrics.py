"""Métricas Prometheus en memoria con exposición de texto 0.0.4 hecha a mano.

Reglas de cardinalidad (P-34): todo label sale de un conjunto acotado.

- `route` es la plantilla de la ruta (`/v1/vigia/traces/{trace_id}`), nunca el path crudo. Un
  request sin ruta resuelta cuenta como `unmatched`. Más de `MAX_ROUTES` plantillas distintas
  caen en `other`.
- `status` es la clase (`2xx` … `5xx`).
- `action` sale de las acciones de `vigia.contracts`; otro valor cae en `other`.
- `detector_id` solo acepta ids de artefacto (16 hex) y está acotado por los detectores activos.
- Nunca hay `org_id`, `trace_id`, usuario ni email en un label.
"""

from __future__ import annotations

import logging
import math
import re
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from importlib import metadata
from typing import Any, get_args

from vigia.contracts import Action
from vigia_ee.security import log_event

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"
DURATION_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)
OVERHEAD_BUCKETS = (0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
MAX_ROUTES = 200
UNMATCHED_ROUTE = "unmatched"
OTHER = "other"
ACTIONS = tuple(get_args(Action))
_DETECTOR_ID_RE = re.compile(r"^[0-9a-f]{16}$")
_ROUTE_RE = re.compile(r"^/[A-Za-z0-9_./{}:-]{0,199}$")

_log = logging.getLogger("vigia.metrics")


def _version() -> str:
    try:
        return metadata.version("vigia")
    except metadata.PackageNotFoundError:
        return "unknown"


def escape_label_value(value: str) -> str:
    """Escapado de valores de label del formato de texto: barra, comillas y salto de línea."""
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def escape_help(text: str) -> str:
    return str(text).replace("\\", "\\\\").replace("\n", "\\n")


def format_value(value: float) -> str:
    """Número en el formato de Prometheus (`+Inf`, `-Inf`, `NaN`)."""
    v = float(value)
    if math.isnan(v):
        return "NaN"
    if math.isinf(v):
        return "+Inf" if v > 0 else "-Inf"
    if v.is_integer() and abs(v) < 2**53:
        return str(int(v))
    return repr(v)


def _labels(pairs: Sequence[tuple[str, str]]) -> str:
    if not pairs:
        return ""
    return "{" + ",".join(f'{k}="{escape_label_value(v)}"' for k, v in pairs) + "}"


def status_class(status: int) -> str:
    """Clase del código (`2xx` … `5xx`); un código fuera de 100–599 cuenta como `5xx`."""
    return f"{status // 100}xx" if 100 <= status <= 599 else "5xx"


class _Histogram:
    __slots__ = ("buckets", "counts", "sum", "count")

    def __init__(self, buckets: Sequence[float]) -> None:
        self.buckets = tuple(buckets)
        self.counts = [0] * len(self.buckets)
        self.sum = 0.0
        self.count = 0

    def observe(self, value: float) -> None:
        for i, upper in enumerate(self.buckets):
            if value <= upper:
                self.counts[i] += 1
                break
        self.sum += value
        self.count += 1

    def lines(self, name: str, base: Sequence[tuple[str, str]]) -> list[str]:
        out: list[str] = []
        cumulative = 0
        for upper, n in zip(self.buckets, self.counts, strict=True):
            cumulative += n
            out.append(f"{name}_bucket{_labels([*base, ('le', format_value(upper))])} {cumulative}")
        out.append(f"{name}_bucket{_labels([*base, ('le', '+Inf')])} {self.count}")
        out.append(f"{name}_sum{_labels(base)} {format_value(self.sum)}")
        out.append(f"{name}_count{_labels(base)} {self.count}")
        return out


class Metrics:
    """Registro de métricas de Vigía, thread-safe.

    `jobs_depth` y `review_depth` son funciones opcionales que `render()` llama para leer la
    profundidad actual; si fallan, queda el último valor conocido.
    """

    def __init__(
        self,
        *,
        version: str | None = None,
        jobs_depth: Callable[[], int] | None = None,
        review_depth: Callable[[], int] | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self.version = version or _version()
        self._jobs_depth_fn = jobs_depth
        self._review_depth_fn = review_depth
        self._requests: dict[tuple[str, str], int] = {}
        self._durations: dict[str, _Histogram] = {}
        self._routes: set[str] = set()
        self._overhead = _Histogram(OVERHEAD_BUCKETS)
        self._tokens_scored = 0
        self._low_confidence = 0
        self._actions: dict[str, int] = {}
        self._calibration: frozenset[str] = frozenset()
        self._auth_failures = 0
        self._rate_limited = 0
        self._review_depth = 0
        self._jobs_depth = 0

    # -- registro de eventos --

    def _route_label(self, route: str | None) -> str:
        if not route:
            return UNMATCHED_ROUTE
        if not _ROUTE_RE.match(route):
            return OTHER
        if route in self._routes:
            return route
        if len(self._routes) >= MAX_ROUTES:
            return OTHER
        self._routes.add(route)
        return route

    def observe_request(self, route: str | None, status: int, duration_s: float) -> None:
        """Un request HTTP terminado. `route` es la plantilla, nunca el path crudo."""
        duration = max(0.0, float(duration_s))
        with self._lock:
            label = self._route_label(route)
            key = (label, status_class(int(status)))
            self._requests[key] = self._requests.get(key, 0) + 1
            hist = self._durations.get(label)
            if hist is None:
                hist = self._durations[label] = _Histogram(DURATION_BUCKETS)
            hist.observe(duration)

    def observe_overhead(self, ratio: float) -> None:
        """Overhead de Vigía sobre la generación base (0,05 = 5 %)."""
        value = float(ratio)
        if math.isnan(value) or value < 0:
            return
        with self._lock:
            self._overhead.observe(value)

    def add_tokens(self, scored: int, low_confidence: int = 0) -> None:
        if scored < 0 or low_confidence < 0:
            raise ValueError("los conteos de tokens no pueden ser negativos")
        with self._lock:
            self._tokens_scored += int(scored)
            self._low_confidence += int(low_confidence)

    def inc_action(self, action: str) -> None:
        label = action if action in ACTIONS else OTHER
        with self._lock:
            self._actions[label] = self._actions.get(label, 0) + 1

    def set_calibrations(self, detector_ids: Iterable[str]) -> None:
        """Reemplaza el conjunto de detectores activos. Ignora ids que no son de artefacto."""
        valid = frozenset(d for d in detector_ids if isinstance(d, str) and _DETECTOR_ID_RE.match(d))
        with self._lock:
            self._calibration = valid

    def inc_auth_failure(self) -> None:
        with self._lock:
            self._auth_failures += 1

    def inc_rate_limited(self) -> None:
        with self._lock:
            self._rate_limited += 1

    def set_review_queue_depth(self, depth: int) -> None:
        with self._lock:
            self._review_depth = max(0, int(depth))

    def set_jobs_queue_depth(self, depth: int) -> None:
        with self._lock:
            self._jobs_depth = max(0, int(depth))

    # -- exposición --

    def _refresh_gauges(self) -> None:
        for fn, setter in (
            (self._jobs_depth_fn, self.set_jobs_queue_depth),
            (self._review_depth_fn, self.set_review_queue_depth),
        ):
            if fn is None:
                continue
            try:
                setter(fn())
            except Exception as exc:  # una fuente caída no rompe /metrics
                log_event(_log, "metrics_gauge_error", level=logging.WARNING, error_type=type(exc).__name__)

    def render(self) -> str:
        """Texto de exposición Prometheus 0.0.4, terminado en salto de línea."""
        self._refresh_gauges()
        out: list[str] = []

        def header(name: str, kind: str, help_text: str) -> None:
            out.append(f"# HELP {name} {escape_help(help_text)}")
            out.append(f"# TYPE {name} {kind}")

        with self._lock:
            header("vigia_requests_total", "counter", "Requests HTTP por plantilla de ruta y clase.")
            for (route, status), n in sorted(self._requests.items()):
                out.append(f"vigia_requests_total{_labels([('route', route), ('status', status)])} {n}")

            header("vigia_request_duration_seconds", "histogram", "Duración de requests HTTP en segundos.")
            for route, hist in sorted(self._durations.items()):
                out.extend(hist.lines("vigia_request_duration_seconds", [("route", route)]))

            header("vigia_overhead_ratio", "histogram", "Overhead de Vigía relativo a la generación base.")
            out.extend(self._overhead.lines("vigia_overhead_ratio", []))

            header("vigia_tokens_scored_total", "counter", "Tokens puntuados por un detector calibrado.")
            out.append(f"vigia_tokens_scored_total {self._tokens_scored}")

            header("vigia_low_confidence_tokens_total", "counter", "Tokens marcados con p_error >= tau.")
            out.append(f"vigia_low_confidence_tokens_total {self._low_confidence}")

            header("vigia_actions_total", "counter", "Acciones de política elegidas por respuesta.")
            for action, n in sorted(self._actions.items()):
                out.append(f"vigia_actions_total{_labels([('action', action)])} {n}")

            header("vigia_calibration_info", "gauge", "Detectores calibrados activos (valor 1).")
            for det in sorted(self._calibration):
                out.append(f"vigia_calibration_info{_labels([('detector_id', det)])} 1")

            header("vigia_auth_failures_total", "counter", "Fallos de autenticación (key, sesión o CSRF).")
            out.append(f"vigia_auth_failures_total {self._auth_failures}")

            header("vigia_rate_limited_total", "counter", "Requests rechazados con 429.")
            out.append(f"vigia_rate_limited_total {self._rate_limited}")

            header("vigia_review_queue_depth", "gauge", "Items pendientes en la cola de revisión.")
            out.append(f"vigia_review_queue_depth {self._review_depth}")

            header("vigia_jobs_queue_depth", "gauge", "Jobs pendientes o en ejecución.")
            out.append(f"vigia_jobs_queue_depth {self._jobs_depth}")

            header("vigia_build_info", "gauge", "Versión de Vigía (valor 1).")
            out.append(f"vigia_build_info{_labels([('version', self.version)])} 1")
        return "\n".join(out) + "\n"


class MetricsMiddleware:
    """Middleware ASGI que cuenta requests y su duración por plantilla de ruta.

    Lee `scope['route'].path` después de que el router resolvió la ruta; sin ruta, usa `unmatched`.
    Una excepción sin manejar cuenta como 500 y se propaga.
    """

    def __init__(self, app: Any, metrics: Metrics) -> None:
        self.app = app
        self.metrics = metrics

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        start = time.perf_counter()
        status = {"code": 500}

        async def send_wrapper(message: dict[str, Any]) -> None:
            if message.get("type") == "http.response.start":
                status["code"] = int(message.get("status", 500))
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            route = scope.get("route")
            template = getattr(route, "path", None) if route is not None else None
            self.metrics.observe_request(
                template if isinstance(template, str) else None,
                status["code"],
                time.perf_counter() - start,
            )
