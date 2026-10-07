"""Composición de la plataforma: configuración, backend compartido, detector y política por tenant.

`Runtime` es lo único que la app y el panel necesitan para servir un tenant:

    runtime = Runtime(Settings.from_env())
    ctx = runtime.context_for(principal)           # RequestContext para vigia.api
    for event in runtime.playground_stream(principal, messages, max_tokens=128): ...

Reglas:

- El backend (modelo) es uno por proceso y lo comparten todos los tenants (ADR A-07).
- El detector activo de cada tenant se cachea por id; un detector calibrado para otro modelo no se
  usa (`detector_for` devuelve None) y queda una alerta legible en `detector_alert`.
- Las trazas se guardan con el texto crudo: `TenantScopedRepository.insert_trace` aplica
  `store_text` (none | redacted | full) y la redacción de PII.
- Un webhook o una métrica que falla nunca tumba un request.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import secrets
import sqlite3
import threading
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, fields, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from vigia.api import RequestContext, TraceRecord
from vigia.contracts import (
    Backend,
    DetectorMismatchError,
    Generation,
    ModelInfo,
    Policy,
    ScoredToken,
    Span,
    Tier,
    TokenEvent,
    VigiaResult,
)
from vigia.detector import Detector
from vigia.drift import DRIFT_FEATURES, drift_report, worst_level
from vigia.policy import (
    HttpRetriever,
    PolicyError,
    RetrieverError,
    decide,
    policy_from_dict,
    rag_fallback,
)
from vigia.spans import build_spans
from vigia_ee import audit, webhooks
from vigia_ee.auth import AuthState, PepperRing, Principal, require_current
from vigia_ee.calibration import CalibrationService
from vigia_ee.db import Database, TenantScopedRepository, create_org, list_org_ids, utcnow
from vigia_ee.jobs import DEFAULT_LANE, JobQueue
from vigia_ee.metrics import Metrics
from vigia_ee.security import DEV_SESSION_COOKIE, SESSION_COOKIE, RateLimiter, log_event

_log = logging.getLogger("vigia.runtime")

TINY_PREFIX = "tiny:"
TINY_ARCHS = ("mixtral", "gpt_oss")
CHEAP_TIERS = (Tier.F0, Tier.F1, Tier.U)
MAX_PLAYGROUND_MESSAGES = 64
MAX_PLAYGROUND_CHARS = 32_000
SHUTTING_DOWN_MESSAGE = "La plataforma se está cerrando. Reintentá en unos segundos."
DRIFT_TRACES = 500  # el drift compara las últimas N trazas puntuadas con el detector activo
DRIFT_MIN_TOKENS = 200  # con menos tokens el PSI es ruido: no se informa
DRIFT_CACHE_S = 60.0  # /metrics recalcula el drift de cada tenant a lo sumo una vez por minuto
READINESS_TIMEOUT_S = 2.0  # espera máxima de /readyz por la base y la cola; si vence, ambas en False
_TRUE = {"1", "true", "yes", "on", "si", "sí"}
_FALSE = {"0", "false", "no", "off"}


class SettingsError(ValueError):
    """La configuración es inválida. El mensaje nombra la variable."""


# Variable de entorno de cada campo con rango: los errores nombran lo que el operador corrige.
_RANGES: tuple[tuple[str, str, int, int], ...] = (
    ("max_tokens_limit", "VIGIA_MAX_TOKENS_LIMIT", 1, 32_768),
    ("default_max_tokens", "VIGIA_DEFAULT_MAX_TOKENS", 1, 32_768),
    ("max_body_bytes", "VIGIA_MAX_BODY_BYTES", 1024, 64 * 1024 * 1024),
    ("max_upload_bytes", "VIGIA_MAX_UPLOAD_BYTES", 1024, 256 * 1024 * 1024),
    ("key_rate_per_minute", "VIGIA_KEY_RATE_PER_MINUTE", 1, 1_000_000),
    ("tenant_rate_per_minute", "VIGIA_TENANT_RATE_PER_MINUTE", 1, 1_000_000),
    ("max_queue_depth", "VIGIA_MAX_QUEUE_DEPTH", 1, 10_000_000),
    ("review_min_reviewers", "VIGIA_REVIEW_MIN_REVIEWERS", 1, 10),
    ("shutdown_timeout_s", "VIGIA_SHUTDOWN_TIMEOUT", 1, 3600),
    ("vllm_max_concurrency", "VIGIA_VLLM_MAX_CONCURRENCY", 1, 4096),
)
# Forma del modelo servido por vLLM. vLLM no la informa y la normalización de U la necesita fija.
_VLLM_SHAPE: tuple[tuple[str, str, int, int], ...] = (
    ("vllm_n_experts", "VIGIA_VLLM_N_EXPERTS", 2, 4096),
    ("vllm_top_k", "VIGIA_VLLM_TOP_K", 1, 4096),
    ("vllm_n_layers", "VIGIA_VLLM_N_LAYERS", 1, 1024),
)


def _check_int(value: Any, env_name: str, lo: int, hi: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
        raise SettingsError(f"{env_name} debe ser un entero entre {lo} y {hi}.")


# --- Configuración ---------------------------------------------------------------------------------


@dataclass
class Settings:
    """Configuración del proceso. `from_env` la lee de variables `VIGIA_*`."""

    data_dir: Path = Path("var")
    db_path: Path | None = None  # default: data_dir / 'vigia.db'
    model: str = "tiny:mixtral"  # id o ruta HF, o 'tiny:mixtral' / 'tiny:gpt_oss' (demo y tests)
    backend: Literal["hf", "vllm"] = "hf"
    vllm_url: str | None = None
    vllm_model: str | None = None
    max_tokens_limit: int = 1024
    default_max_tokens: int = 256
    max_body_bytes: int = 1_048_576
    max_upload_bytes: int = 20_971_520
    metrics_token: str | None = field(default=None, repr=False)
    rag_allowed_hosts: tuple[str, ...] = ()
    webhook_allowed_hosts: tuple[str, ...] = ()
    secure_cookies: bool = True
    device: str = "cpu"  # VIGIA_DEVICE: cpu | cuda | cuda:N | mps (backend HF)
    dtype: str = "auto"  # VIGIA_DTYPE: auto | float32 | bfloat16 | float16
    base_detector_path: Path | None = None
    # Extras de operación (no forman parte del contrato mínimo, todos con default seguro).
    vllm_api_key: str | None = field(default=None, repr=False)
    peppers: PepperRing | None = field(default=None, repr=False)  # None: PepperRing.from_env()
    key_rate_per_minute: int = 120
    tenant_rate_per_minute: int = 600
    max_queue_depth: int = 1000
    review_min_reviewers: int = 2
    shutdown_timeout_s: int = 30  # tope del cierre ordenado (uvicorn timeout_graceful_shutdown)
    # Obligatorios con backend=vllm: expertos por capa, top-k y capas MoE del modelo servido.
    vllm_n_experts: int | None = None
    vllm_top_k: int | None = None
    vllm_n_layers: int | None = None
    vllm_max_concurrency: int = 16  # generaciones simultáneas hacia vLLM (cupo de vigia.api)

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.db_path = Path(self.db_path) if self.db_path is not None else self.data_dir / "vigia.db"
        if self.base_detector_path is not None:
            self.base_detector_path = Path(self.base_detector_path)
        self.rag_allowed_hosts = tuple(self.rag_allowed_hosts)
        self.webhook_allowed_hosts = tuple(self.webhook_allowed_hosts)
        if self.backend not in ("hf", "vllm"):
            raise SettingsError("VIGIA_BACKEND debe ser 'hf' o 'vllm'.")
        if not isinstance(self.model, str) or not self.model.strip():
            raise SettingsError("VIGIA_MODEL es obligatorio.")
        if self.model.startswith(TINY_PREFIX) and self.model.removeprefix(TINY_PREFIX) not in TINY_ARCHS:
            raise SettingsError(
                f"Modelo diminuto desconocido; opciones: {', '.join(TINY_PREFIX + a for a in TINY_ARCHS)}."
            )
        if self.dtype not in ("auto", "float32", "bfloat16", "float16"):
            raise SettingsError("VIGIA_DTYPE debe ser auto, float32, bfloat16 o float16.")
        if not re.fullmatch(r"cpu|mps|cuda(:\d+)?", self.device):
            raise SettingsError("VIGIA_DEVICE debe ser cpu, mps, cuda o cuda:N.")
        if self.backend == "vllm":
            self._check_vllm()
        for name, env_name, lo, hi in _RANGES:
            _check_int(getattr(self, name), env_name, lo, hi)
        if self.default_max_tokens > self.max_tokens_limit:
            raise SettingsError("VIGIA_DEFAULT_MAX_TOKENS no puede superar VIGIA_MAX_TOKENS_LIMIT.")
        if self.metrics_token is not None and len(self.metrics_token) < 16:
            raise SettingsError("VIGIA_METRICS_TOKEN debe tener al menos 16 caracteres.")

    def _check_vllm(self) -> None:
        """URL y forma del modelo servido. Sin `n_experts` fijo, U de un token dependería del futuro."""
        if not self.vllm_url or not self.vllm_model:
            raise SettingsError("Con VIGIA_BACKEND=vllm hacen falta VIGIA_VLLM_URL y VIGIA_VLLM_MODEL.")
        parts = urlsplit(self.vllm_url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise SettingsError("VIGIA_VLLM_URL debe ser una URL http(s) con host.")
        missing = [env_name for name, env_name, _, _ in _VLLM_SHAPE if getattr(self, name) is None]
        if missing:
            raise SettingsError(
                f"Con VIGIA_BACKEND=vllm hacen falta {', '.join(missing)} (config.json del modelo servido)."
            )
        for name, env_name, lo, hi in _VLLM_SHAPE:
            _check_int(getattr(self, name), env_name, lo, hi)
        if self.vllm_top_k > self.vllm_n_experts:  # type: ignore[operator]
            raise SettingsError("VIGIA_VLLM_TOP_K no puede superar VIGIA_VLLM_N_EXPERTS.")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        """Lee la configuración. Los secretos aceptan la variante `*_FILE` (Docker secrets)."""
        env = os.environ if environ is None else environ
        model = env.get("VIGIA_MODEL", "").strip()
        if not model:
            raise SettingsError("Falta VIGIA_MODEL (id o ruta del modelo HF, o 'tiny:mixtral' para la demo).")
        data_dir = Path(env.get("VIGIA_DATA_DIR") or "var")
        db_path = env.get("VIGIA_DB_PATH")
        base = env.get("VIGIA_BASE_DETECTOR")
        peppers = None
        if env.get("VIGIA_PEPPERS_FILE") or env.get("VIGIA_PEPPERS"):
            peppers = PepperRing.from_env(env)
        return cls(
            data_dir=data_dir,
            db_path=Path(db_path) if db_path else None,
            model=model,
            backend=env.get("VIGIA_BACKEND", "hf").strip().lower(),  # type: ignore[arg-type]
            vllm_url=env.get("VIGIA_VLLM_URL") or None,
            vllm_model=env.get("VIGIA_VLLM_MODEL") or None,
            max_tokens_limit=_env_int(env, "VIGIA_MAX_TOKENS_LIMIT", 1024),
            default_max_tokens=_env_int(env, "VIGIA_DEFAULT_MAX_TOKENS", 256),
            max_body_bytes=_env_int(env, "VIGIA_MAX_BODY_BYTES", 1_048_576),
            max_upload_bytes=_env_int(env, "VIGIA_MAX_UPLOAD_BYTES", 20_971_520),
            metrics_token=_env_secret(env, "VIGIA_METRICS_TOKEN"),
            rag_allowed_hosts=_env_csv(env, "VIGIA_RAG_ALLOWED_HOSTS"),
            webhook_allowed_hosts=_env_csv(env, "VIGIA_WEBHOOK_ALLOWED_HOSTS"),
            secure_cookies=_env_bool(env, "VIGIA_SECURE_COOKIES", True),
            device=(env.get("VIGIA_DEVICE") or "cpu").strip(),
            dtype=(env.get("VIGIA_DTYPE") or "auto").strip().lower(),
            base_detector_path=Path(base) if base else None,
            vllm_api_key=_env_secret(env, "VIGIA_VLLM_API_KEY"),
            peppers=peppers,
            key_rate_per_minute=_env_int(env, "VIGIA_KEY_RATE_PER_MINUTE", 120),
            tenant_rate_per_minute=_env_int(env, "VIGIA_TENANT_RATE_PER_MINUTE", 600),
            max_queue_depth=_env_int(env, "VIGIA_MAX_QUEUE_DEPTH", 1000),
            review_min_reviewers=_env_int(env, "VIGIA_REVIEW_MIN_REVIEWERS", 2),
            shutdown_timeout_s=_env_int(env, "VIGIA_SHUTDOWN_TIMEOUT", 30),
            vllm_n_experts=_env_int(env, "VIGIA_VLLM_N_EXPERTS", None),
            vllm_top_k=_env_int(env, "VIGIA_VLLM_TOP_K", None),
            vllm_n_layers=_env_int(env, "VIGIA_VLLM_N_LAYERS", None),
            vllm_max_concurrency=_env_int(env, "VIGIA_VLLM_MAX_CONCURRENCY", 16),
        )


def _env_int(env: Mapping[str, str], name: str, default: int | None) -> int | None:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip().replace("_", ""))
    except ValueError:
        raise SettingsError(f"{name} debe ser un entero.") from None


def _env_bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return default
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise SettingsError(f"{name} debe ser true o false.")


def _env_csv(env: Mapping[str, str], name: str) -> tuple[str, ...]:
    return tuple(h.strip().lower() for h in (env.get(name) or "").split(",") if h.strip())


def _env_secret(env: Mapping[str, str], name: str) -> str | None:
    path = env.get(name + "_FILE")
    if path:
        try:
            with open(path, encoding="utf-8") as fh:
                value = fh.read().strip()
        except OSError:
            raise SettingsError(f"No se pudo leer {name}_FILE.") from None
        return value or None
    value = (env.get(name) or "").strip()
    return value or None


# --- Backend ---------------------------------------------------------------------------------------


def build_backend(settings: Settings) -> Backend:
    """Backend según la configuración: vLLM (modo reducido), modelo diminuto o checkpoint HF.

    Un checkpoint que no se puede cargar (ruta o repo inexistente, carpeta sin tokenizer) sale como
    `SettingsError` con `VIGIA_MODEL` y la causa en una línea, nunca como traceback crudo. vLLM recibe
    la forma fija del modelo servido (`VIGIA_VLLM_N_EXPERTS`, `_TOP_K`, `_N_LAYERS`): así U es causal.
    """
    if settings.backend == "vllm":
        from vigia.backends.vllm_proxy import VllmProxyBackend

        return VllmProxyBackend(
            settings.vllm_url,
            settings.vllm_model,
            api_key=settings.vllm_api_key,
            n_experts=settings.vllm_n_experts,
            top_k=settings.vllm_top_k or 0,
            n_layers=settings.vllm_n_layers or 0,
            max_concurrency=settings.vllm_max_concurrency,
        )
    if settings.model.startswith(TINY_PREFIX):
        return make_tiny_backend(settings.model.removeprefix(TINY_PREFIX))
    from vigia.backends.hf import HFBackend

    try:
        return HFBackend(
            settings.model, device=settings.device, dtype=None if settings.dtype == "auto" else settings.dtype
        )
    except (OSError, ValueError) as exc:
        raise SettingsError(
            f"No se puede cargar el modelo (VIGIA_MODEL={settings.model}): {_short_cause(exc)}"
        ) from exc


def _short_cause(exc: BaseException, limit: int = 300) -> str:
    """Primera línea no vacía del error, recortada: la causa sin el stack ni el texto de ayuda."""
    first = next((ln.strip() for ln in str(exc).splitlines() if ln.strip()), type(exc).__name__)
    return first if len(first) <= limit else first[: limit - 1] + "…"


def make_tiny_backend(arch: str) -> Backend:
    """Modelo MoE diminuto con pesos aleatorios fijos y tokenizer byte-level (demo y tests).

    Delega en `vigia.cli.make_tiny_backend`: el mismo `tiny:<arch>` que usa la CLI, así un detector
    calibrado con `vigia calibrate` sirve en la plataforma.
    """
    from vigia.cli import make_tiny_backend as core_tiny

    return core_tiny(arch)


# --- Utilidades de trazas --------------------------------------------------------------------------


def render_prompt(messages: list[dict[str, str]]) -> str:
    """Transcripción legible del prompt: bloques `rol: contenido` separados por una línea vacía."""
    return "\n\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(value) else None


def _visible(t: TokenEvent) -> bool:
    return t.channel == "final" and not t.is_special


def _scored(t: TokenEvent, p: float | None, tau: float | None, text_len: int) -> ScoredToken:
    start = min(max(t.start, 0), text_len)
    end = min(max(t.end, start), text_len)
    flag = p is not None and tau is not None and p >= tau
    return ScoredToken(index=t.index, text=t.text, start=start, end=end, p_error=p, flag=flag)


def _usage(info: ModelInfo, tokens: list[TokenEvent]) -> dict[str, Any] | None:
    """Uso capa × experto de la respuesta (para el heatmap del panel)."""
    n = info.n_experts
    layers = list(info.moe_layers)
    if n <= 0 or not layers:
        return None
    counts = [[0] * n for _ in layers]
    seen = False
    for t in tokens:
        if not _visible(t) or not t.expert_ids:
            continue
        for li, ids in enumerate(t.expert_ids[: len(layers)]):
            for e in ids:
                if isinstance(e, int) and 0 <= e < n:
                    counts[li][e] += 1
                    seen = True
    return {"layers": layers, "counts": counts} if seen else None


def _stored_span(row: Mapping[str, Any], tokens: list[ScoredToken], text: str) -> Span:
    """Span guardado (start, end, score) con el rango de tokens y el texto que la base no guarda."""
    start, end = int(row["start"]), int(row["end"])
    inside = [t.index for t in tokens if t.start < end and t.end > start]
    return Span(
        start=start,
        end=end,
        token_start=min(inside) if inside else 0,
        token_end=max(inside) + 1 if inside else 0,
        score=float(row["score"]),
        text=text[start:end],
    )


def _check_messages(messages: Any) -> list[dict[str, str]]:
    if not isinstance(messages, list) or not 1 <= len(messages) <= MAX_PLAYGROUND_MESSAGES:
        raise ValueError(f"Mandá entre 1 y {MAX_PLAYGROUND_MESSAGES} mensajes.")
    out = []
    total = 0
    for m in messages:
        if not isinstance(m, Mapping):
            raise ValueError("Cada mensaje debe tener 'role' y 'content'.")
        role, content = m.get("role"), m.get("content")
        if role not in ("system", "user", "assistant") or not isinstance(content, str):
            raise ValueError(
                "Cada mensaje debe tener un rol system, user o assistant y un contenido de texto."
            )
        total += len(content)
        out.append({"role": role, "content": content})
    if total > MAX_PLAYGROUND_CHARS:
        raise ValueError(f"El prompt supera los {MAX_PLAYGROUND_CHARS} caracteres.")
    if not any(m["content"].strip() for m in out):
        raise ValueError("El prompt está vacío.")
    return out


# --- Almacenamiento -------------------------------------------------------------------------------

_SQLITE_REASONS = (
    ("readonly", "la base es de solo lectura"),
    ("not a database", "el archivo no es una base SQLite"),
    ("unable to open", "no se puede abrir el archivo"),
    ("locked", "otra conexión tiene la base bloqueada"),
    ("disk i/o", "error de entrada/salida del disco"),
)


def _storage_var(settings: Settings) -> str:
    """Nombra la variable que fijó la ruta de la base, para que el operador sepa qué corregir."""
    if Path(settings.db_path) == Path(settings.data_dir) / "vigia.db":
        return f"VIGIA_DATA_DIR={settings.data_dir}"
    return f"VIGIA_DB_PATH={settings.db_path}"


def _sqlite_reason(exc: sqlite3.Error) -> str:
    text = str(exc)
    for needle, reason in _SQLITE_REASONS:
        if needle in text.lower():
            return reason
    return text


def open_database(settings: Settings) -> Database:
    """Crea la carpeta, abre la base, migra y prueba que acepte escrituras.

    Un error de almacenamiento (carpeta inválida, base de solo lectura o corrupta) sale como
    `SettingsError` con la variable y la causa, nunca como traceback crudo.
    """
    where = _storage_var(settings)
    path = settings.db_path
    if str(path) != ":memory:":
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            reason = exc.strerror or str(exc)
            raise SettingsError(f"No se puede crear la carpeta de datos ({where}): {reason}.") from exc
    try:
        db = Database(path)
    except (OSError, sqlite3.Error) as exc:
        reason = _sqlite_reason(exc) if isinstance(exc, sqlite3.Error) else exc.strerror or str(exc)
        raise SettingsError(f"No se puede abrir la base ({where}): {reason}.") from exc
    try:
        with db.transaction():
            db.execute("UPDATE schema_migrations SET applied_at = applied_at WHERE 0")
    except sqlite3.Error as exc:
        db.close()
        raise SettingsError(f"La base no acepta escrituras ({where}): {_sqlite_reason(exc)}.") from exc
    return db


# --- Runtime ---------------------------------------------------------------------------------------


class Runtime:
    """Estado compartido del proceso: base, auth, backend, detectores, jobs y métricas."""

    def __init__(self, settings: Settings, *, backend: Backend | None = None, clock: Any = None) -> None:
        self.settings = settings
        self.clock = clock or utcnow
        self.db = open_database(settings)
        self._closed = False
        self._shutting_down = threading.Event()
        self._probe_lock = threading.Lock()
        self._probe: tuple[threading.Thread, dict[str, bool]] | None = None
        try:
            base_artifact = self._read_base_detector()  # antes del modelo: falla rápido
            peppers = settings.peppers or PepperRing.from_env()
            self.auth = AuthState(
                db=self.db,
                peppers=peppers,
                cookie_name=SESSION_COOKIE if settings.secure_cookies else DEV_SESSION_COOKIE,
                key_limiter=RateLimiter(
                    rate=settings.key_rate_per_minute / 60.0, burst=settings.key_rate_per_minute
                ),
                tenant_limiter=RateLimiter(
                    rate=settings.tenant_rate_per_minute / 60.0, burst=settings.tenant_rate_per_minute
                ),
                clock=self.clock,
            )
            self.backend: Backend = backend if backend is not None else build_backend(settings)
            self.model_info: ModelInfo = self.backend.model_info()
            self.jobs = JobQueue(self.db, clock=self.clock)
            self.metrics = Metrics(
                jobs_depth=self.jobs.depth, review_depth=self._review_depth, drift=self.drift_gauges
            )
            webhooks.register_delivery(self.jobs, allowed_hosts=settings.webhook_allowed_hosts)
            self._lock = threading.RLock()
            self._detectors: dict[str, tuple[str, Detector | None]] = {}
            self._detector_alerts: dict[str, str] = {}
            self._policies: dict[str, tuple[str, Policy, bool]] = {}
            self._policy_alerts: dict[str, str] = {}
            self._retrievers: dict[str, tuple[str, HttpRetriever | None]] = {}
            self._base_checked: set[str] = set()
            self._drift: dict[str, tuple[float, str | None, dict[str, Any]]] = {}
            self.base_detector: Detector | None = self._check_base_detector(base_artifact)
            self.calibration = CalibrationService(self)
            self.startup_errors = self._warm_detectors()
        except BaseException:
            self.db.close()
            raise
        self.started_at = self.clock()
        self.startup_complete = True

    # -- básicos --

    def now(self) -> datetime:
        return self.clock()

    def repo(self, org_id: str) -> TenantScopedRepository:
        return TenantScopedRepository(self.db, org_id)

    @property
    def shutting_down(self) -> bool:
        """True desde que el proceso recibió la señal de cierre (`begin_shutdown`)."""
        return self._shutting_down.is_set()

    def begin_shutdown(self) -> None:
        """Marca el cierre: los requests nuevos de generación responden 503 y el playground corta.

        Lo llama el servidor al recibir SIGINT/SIGTERM, antes de esperar los requests en curso.
        """
        if not self._shutting_down.is_set():
            self._shutting_down.set()
            log_event(_log, "shutdown_started", level=logging.INFO)

    def close(self) -> None:
        """Detiene el worker, libera el backend y cierra la base. Idempotente."""
        if self._closed:
            return
        self._closed = True
        self.jobs.stop()
        close = getattr(self.backend, "close", None)
        if callable(close):
            try:
                close()
            except Exception as exc:  # noqa: BLE001 - cerrar no debe impedir cerrar la base
                log_event(_log, "backend_close_error", level=logging.WARNING, error_type=type(exc).__name__)
        self.db.close()

    def _review_depth(self) -> int:
        row = self.db.query_one(
            "SELECT COUNT(*) AS n FROM review_items WHERE state IN ('pending', 'reserved', 'needs_other')"
        )
        return int(row["n"]) if row else 0

    # -- organizaciones y detector base --

    def create_org(self, name: str, *, store_text: str = "redacted", retention_days: int = 30) -> str:
        """Crea una organización y le instala el detector base si coincide con el modelo."""
        org_id = create_org(self.db, name, store_text=store_text, retention_days=retention_days)
        audit.append(self.db, org_id, None, "org_create", org_id)
        self.ensure_base_detector(org_id)
        return org_id

    def _read_base_detector(self) -> Detector | None:
        """Lee `VIGIA_BASE_DETECTOR`. Archivo inexistente, ilegible o artefacto inválido: SettingsError."""
        path = self.settings.base_detector_path
        if path is None:
            return None
        where = f"VIGIA_BASE_DETECTOR={path}"
        try:
            text = Path(path).read_text(encoding="utf-8")
        except FileNotFoundError:
            raise SettingsError(f"No existe el detector base ({where}).") from None
        except (OSError, UnicodeDecodeError) as exc:
            reason = exc.strerror if isinstance(exc, OSError) and exc.strerror else type(exc).__name__
            raise SettingsError(f"No se puede leer el detector base ({where}): {reason}.") from None
        try:
            return Detector.loads(text)
        except (DetectorMismatchError, ValueError) as exc:
            raise SettingsError(f"El detector base es inválido ({where}): {_short_cause(exc)}") from None

    def _check_base_detector(self, detector: Detector | None) -> Detector | None:
        """Rechazo blando (documentado): un detector base de otro modelo no se instala y queda en el log."""
        if detector is None:
            return None
        try:
            detector.check_model(self.model_info)
        except DetectorMismatchError as exc:
            log_event(_log, "base_detector_rejected", level=logging.WARNING, error_type="model_mismatch")
            _log.warning("detector base descartado: %s", exc)
            return None
        return detector

    def ensure_base_detector(self, org_id: str) -> bool:
        """Instala y activa el detector base en una org sin detectores. Devuelve si lo instaló."""
        if self.base_detector is None:
            return False
        with self._lock:
            if org_id in self._base_checked:
                return False
            self._base_checked.add(org_id)
            repo = self.repo(org_id)
            if repo.get_org() is None or repo.list_detectors():
                return False
            det = self.base_detector
            with self.db.transaction():
                repo.insert_detector(det.id, det.to_json())
                repo.activate_detector(det.id)
                audit.append(self.db, org_id, None, "detector_activate", det.id, {"source": "base"})
        return True

    # -- detector por tenant --

    def detector_for(self, org_id: str) -> Detector | None:
        """Detector activo del tenant, cacheado por id. None si no hay o si es de otro modelo."""
        repo = self.repo(org_id)
        org = repo.get_org()
        if org is None:
            return None
        active_id = org["active_detector_id"]
        if active_id is None and self.ensure_base_detector(org_id):
            org = repo.get_org() or org
            active_id = org["active_detector_id"]
        if active_id is None:
            self.invalidate_detector(org_id)
            return None
        with self._lock:
            cached = self._detectors.get(org_id)
        if cached is not None and cached[0] == active_id:
            return cached[1]
        row = repo.get_active_detector()
        if row is None:
            self.invalidate_detector(org_id)
            return None
        detector: Detector | None = None
        alert = None
        try:
            detector = Detector.loads(row["artifact_json"])
            detector.check_model(self.model_info)
        except DetectorMismatchError as exc:
            detector = None
            alert = f"El detector activo {row['id']} no se usa: {exc}"
            log_event(_log, "detector_rejected", level=logging.WARNING, org_id=org_id, error_type="mismatch")
        with self._lock:
            self._detectors[org_id] = (row["id"], detector)
            if alert:
                self._detector_alerts[org_id] = alert
            else:
                self._detector_alerts.pop(org_id, None)
            active = [d.id for _, d in self._detectors.values() if d is not None]
        self.metrics.set_calibrations(active)
        return detector

    def detector_alert(self, org_id: str) -> str | None:
        """Motivo legible por el que el detector activo no se usa (para el banner del panel)."""
        with self._lock:
            return self._detector_alerts.get(org_id)

    def invalidate_detector(self, org_id: str) -> None:
        with self._lock:
            self._detectors.pop(org_id, None)
            self._detector_alerts.pop(org_id, None)
            active = [d.id for _, d in self._detectors.values() if d is not None]
        self.metrics.set_calibrations(active)

    def forget_org(self, org_id: str) -> None:
        """Saca de memoria todo el estado de una org borrada: detector, alertas, política y retriever."""
        with self._lock:
            self._detectors.pop(org_id, None)
            self._detector_alerts.pop(org_id, None)
            self._policies.pop(org_id, None)
            self._policy_alerts.pop(org_id, None)
            self._retrievers.pop(org_id, None)
            self._base_checked.discard(org_id)
            self._drift.pop(org_id, None)
            active = [d.id for _, d in self._detectors.values() if d is not None]
        self.metrics.set_calibrations(active)

    def _warm_detectors(self) -> int:
        errors = 0
        for org_id in list_org_ids(self.db):
            try:
                self.detector_for(org_id)
            except Exception as exc:  # noqa: BLE001 - un tenant roto no impide arrancar a los demás
                errors += 1
                log_event(
                    _log,
                    "detector_warm_error",
                    level=logging.ERROR,
                    org_id=org_id,
                    error_type=type(exc).__name__,
                )
        return errors

    # -- política --

    def _stored_policy(self, org_id: str) -> tuple[Policy, bool]:
        """(política guardada validada, alpha fijado por la política). Cacheada por contenido.

        Una política guardada inválida cae a los defaults seguros y su alpha no cuenta como fijado.
        """
        raw = self.repo(org_id).get_policy()
        if not raw:
            return Policy(), False
        key = json.dumps(raw, sort_keys=True)
        with self._lock:
            cached = self._policies.get(org_id)
        if cached is not None and cached[0] == key:
            return replace(cached[1]), cached[2]
        try:
            policy = policy_from_dict(dict(raw), rag_allowed_hosts=self.settings.rag_allowed_hosts)
            alert = None
        except PolicyError as exc:
            policy = Policy()
            alert = f"La política guardada es inválida y se usan los valores por defecto: {exc}"
            log_event(_log, "policy_invalid", level=logging.WARNING, org_id=org_id)
        explicit = alert is None and "alpha" in raw
        with self._lock:
            self._policies[org_id] = (key, policy, explicit)
            if alert:
                self._policy_alerts[org_id] = alert
            else:
                self._policy_alerts.pop(org_id, None)
        return replace(policy), explicit

    def _detector_alpha(self, org_id: str) -> float:
        """Alpha recomendado del detector activo; si falta, su default_alpha; sin detector, Policy.alpha."""
        detector = self.detector_for(org_id)
        alpha = getattr(detector, "recommended_alpha", None)  # un Scorer puede no tenerlo
        if alpha is None:
            alpha = getattr(detector, "default_alpha", None)
        if isinstance(alpha, int | float) and not isinstance(alpha, bool) and 0 < alpha < 1:
            return float(alpha)
        return Policy().alpha

    def policy_for(self, org_id: str) -> Policy:
        """Política efectiva del tenant (copia).

        alpha efectivo: el de la política si el owner lo fijó (la política guardada tiene la clave
        `alpha`); si no, `default_alpha` del detector activo, que es el alpha recomendado en
        calibración; sin detector, `Policy.alpha` (0,1). Así una org que no eligió alpha sigue al
        detector cuando se recalibra.
        """
        policy, explicit = self._stored_policy(org_id)
        if explicit:
            return policy
        return replace(policy, alpha=self._detector_alpha(org_id))

    def policy_alpha_explicit(self, org_id: str) -> bool:
        """True si la política guardada fija alpha; False si el alpha sale del detector activo."""
        return self._stored_policy(org_id)[1]

    def policy_alert(self, org_id: str) -> str | None:
        with self._lock:
            return self._policy_alerts.get(org_id)

    def save_policy(self, principal: Principal, data: dict[str, Any]) -> Policy:
        """Valida y guarda la política (merge sobre la vigente), sube la versión y audita.

        Un `alpha` numérico en `data` queda fijado por la política. `alpha: None` lo suelta: vuelve a
        seguir el `default_alpha` del detector activo. Sin `alpha` en `data`, se conserva lo que había
        (fijado o del detector). La política guardada solo lleva la clave `alpha` si está fijado.
        """
        if not principal.can("policy:write"):
            raise PermissionError("Tu rol no tiene permiso para cambiar la política.")
        if not isinstance(data, dict):
            raise PolicyError("la política debe ser un objeto JSON")
        org_id = principal.org_id
        explicit_before = self.policy_alpha_explicit(org_id)
        current = self.policy_for(org_id)
        before = current.to_dict()
        known = {f.name for f in fields(Policy)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise PolicyError(f"campos desconocidos en la política: {', '.join(unknown)}")
        changes = {k: v for k, v in data.items() if k != "version"}
        explicit = explicit_before
        if "alpha" in changes:
            explicit = changes["alpha"] is not None
            if not explicit:
                changes["alpha"] = self._detector_alpha(org_id)
        merged = {**before, **changes}
        merged["version"] = current.version + 1
        policy = policy_from_dict(merged, rag_allowed_hosts=self.settings.rag_allowed_hosts)
        detector = self.detector_for(org_id)
        if detector is not None:
            min_alpha = min(t["alpha"] for t in detector.thresholds)
            if policy.alpha < min_alpha:
                raise PolicyError(
                    f"alpha {policy.alpha:g} es menor que el mínimo calibrado del detector activo "
                    f"({min_alpha:g})"
                )
        after = policy.to_dict()
        changed = sorted(k for k in after if k != "version" and before.get(k) != after[k])
        stored = after if explicit else {k: v for k, v in after.items() if k != "alpha"}
        detail: dict[str, Any] = {"changed": changed, "version": policy.version}
        if explicit != explicit_before:
            detail["alpha_source"] = "policy" if explicit else "detector"
        repo = self.repo(org_id)
        with self.db.transaction():
            # El actor se relee en la transacción: un rol degradado después de resolver el request no
            # confirma el cambio de política.
            actor = require_current(self.auth, principal, "policy:write") or principal
            if not repo.set_policy(stored):
                raise LookupError("organización inexistente")
            audit.append(self.db, org_id, actor, "policy_change", None, detail)
        with self._lock:
            self._policies.pop(org_id, None)
            self._policy_alerts.pop(org_id, None)
        return replace(policy)

    def _retriever_for(self, org_id: str, policy: Policy) -> HttpRetriever | None:
        if not policy.rag_url:
            return None
        with self._lock:
            cached = self._retrievers.get(org_id)
        if cached is not None and cached[0] == policy.rag_url:
            return cached[1]
        try:
            retriever: HttpRetriever | None = HttpRetriever(
                policy.rag_url, allowed_hosts=self.settings.rag_allowed_hosts
            )
        except (RetrieverError, ValueError) as exc:
            retriever = None
            log_event(
                _log,
                "retriever_rejected",
                level=logging.WARNING,
                org_id=org_id,
                error_type=type(exc).__name__,
            )
        with self._lock:
            self._retrievers[org_id] = (policy.rag_url, retriever)
        return retriever

    # -- contexto de request --

    def context_for(
        self, principal: Principal, *, source: Literal["api", "playground"] = "api"
    ) -> RequestContext:
        """RequestContext para `vigia.api` con todo atado al tenant del principal."""
        if source not in ("api", "playground"):
            raise ValueError("source debe ser 'api' o 'playground'")
        org_id = principal.org_id
        policy = self.policy_for(org_id)
        return RequestContext(
            backend=self.backend,
            scorer=self.detector_for(org_id),
            policy=policy,
            org_id=org_id,
            key_id=principal.key_id,
            sink=lambda record: self.store_trace(org_id, record),
            trace_lookup=lambda trace_id: self.trace_view(org_id, trace_id),
            retriever=self._retriever_for(org_id, policy),
            source=source,
            closing=lambda: self.shutting_down,
        )

    def store_trace(self, org_id: str, record: TraceRecord) -> str:
        """Guarda la traza, encola revisión si corresponde, emite el webhook y actualiza métricas."""
        gen, res = record.generation, record.result
        by_index = {t.index: t for t in gen.tokens}
        tokens = []
        for st in res.tokens:
            ev = by_index.get(st.index)
            feats = ev.features if ev is not None else {}
            tokens.append(
                {
                    "index": st.index,
                    "text": st.text,
                    "start": st.start,
                    "end": st.end,
                    "p_error": st.p_error,
                    "flag": st.flag,
                    "out_entropy": _num(feats.get("out_entropy")),
                    "r_ent_mean": _num(feats.get("agg_mean:r_ent_n")),
                }
            )
        try:
            created_at = datetime.fromisoformat(record.created_at)
        except (TypeError, ValueError):
            created_at = self.now()
        repo = self.repo(org_id)
        with self.db.transaction():
            trace_id = repo.insert_trace(
                trace_id=record.trace_id,
                model_id=record.model.model_id,
                prompt_text=render_prompt(record.prompt_messages),
                response_text=gen.text,
                action=res.action,
                tokens=tokens,
                spans=res.spans,
                n_tokens=len(gen.tokens),
                api_key_id=record.key_id,
                source=record.source,
                mode=res.mode,
                response_p_error=res.response_p_error,
                calibration_id=res.calibration_id,
                alpha=res.alpha,
                tau=res.tau,
                latency_ms=record.latency_ms,
                overhead_ms=record.overhead_ms,
                degraded=res.degraded,
                status=res.status,
                rag_status=res.rag_status or record.rag_status,
                finish_reason=gen.finish_reason,
                usage=_usage(record.model, gen.tokens),
                created_at=created_at,
            )
            if res.action == "human_review":
                self._enqueue_review(repo, trace_id, res)
        if res.action != "pass":
            self.emit_event(
                org_id,
                "trace.flagged",
                {
                    "trace_id": trace_id,
                    "action": res.action,
                    "response_p_error": res.response_p_error,
                    "calibration_id": res.calibration_id,
                    "source": record.source,
                },
            )
        self._observe(record)
        return trace_id

    def _enqueue_review(self, repo: TenantScopedRepository, trace_id: str, res: VigiaResult) -> None:
        minimum = self.settings.review_min_reviewers
        tau = res.tau
        picked = [i for i, s in enumerate(res.spans) if tau is not None and s.score >= tau]
        if not picked:
            repo.create_review_item(trace_id, min_reviewers=minimum)
            return
        for idx in picked:
            repo.create_review_item(trace_id, span_idx=idx, min_reviewers=minimum)

    def _observe(self, record: TraceRecord) -> None:
        try:
            res = record.result
            scored = sum(1 for t in res.tokens if t.p_error is not None)
            flagged = sum(1 for t in res.tokens if t.flag)
            self.metrics.add_tokens(scored, flagged)
            self.metrics.inc_action(res.action)
            base_ms = record.latency_ms - record.overhead_ms
            if base_ms > 0 and record.overhead_ms >= 0:
                self.metrics.observe_overhead(record.overhead_ms / base_ms)
        except Exception as exc:  # noqa: BLE001 - una métrica nunca tumba un request
            log_event(_log, "metrics_error", level=logging.WARNING, error_type=type(exc).__name__)

    def emit_event(self, org_id: str, event_type: str, data: Mapping[str, Any]) -> list[str]:
        """Encola el webhook del evento. Nunca lanza: una falla queda en el log."""
        try:
            return webhooks.emit(self.db, self.jobs, org_id, event_type, data)
        except Exception as exc:  # noqa: BLE001
            log_event(
                _log,
                "webhook_emit_error",
                level=logging.WARNING,
                org_id=org_id,
                action=event_type,
                error_type=type(exc).__name__,
            )
            return []

    def trace_view(self, org_id: str, trace_id: str) -> dict[str, Any] | None:
        """Traza del tenant para `GET /v1/vigia/traces/{id}` (sin el prompt). None si es ajena.

        Misma forma que `vigia.serve.trace_view` (A-09, P-14): `{id, object, created_at, model,
        source, finish_reason, response, vigia, latency_ms, overhead_ms, rag_status}`, con `vigia`
        igual a `VigiaResult.to_dict()`. Así un cliente escrito contra `vigia serve` lee la traza de
        la plataforma. Con `abstain` no viajan el texto, los tokens ni los fragmentos. El panel lee la
        traza completa con `repo.get_trace`. `rag_status` y `status` salen de la base
        (migración 3); una traza guardada antes de esa migración vale None y "ok".
        """
        if not isinstance(trace_id, str) or not trace_id:
            return None
        trace = self.repo(org_id).get_trace(trace_id)
        if trace is None:
            return None
        abstained = trace["action"] == "abstain"
        text = trace["response_text"] or ""
        tokens = [
            ScoredToken(
                index=t["idx"],
                text=t["text"],
                start=t["start"],
                end=t["end"],
                p_error=t["p_error"],
                flag=bool(t["flag"]),
            )
            for t in trace["tokens"]
        ]
        spans = [_stored_span(s, tokens, text) for s in trace["spans"]]
        result = VigiaResult(
            trace_id=trace["id"],
            tokens=[] if abstained else tokens,
            spans=[] if abstained else spans,
            response_p_error=trace["response_p_error"],
            action=trace["action"],
            calibration_id=trace["calibration_id"],
            # Una traza parcial (stream cortado) o degradada se guarda con tau=None y no está calibrada.
            calibrated=(
                trace["calibration_id"] is not None and not trace["degraded"] and trace["tau"] is not None
            ),
            mode=trace["mode"],
            degraded=bool(trace["degraded"]),
            status=trace.get("status") or "ok",
            rag_status=trace.get("rag_status"),
            alpha=trace["alpha"],
            tau=trace["tau"],
        )
        return {
            "id": trace["id"],
            "object": "vigia.trace",
            "created_at": trace["created_at"],
            "model": trace["model_id"],
            "source": trace["source"],
            "finish_reason": trace["finish_reason"],
            "response": None if abstained else trace["response_text"],
            "vigia": result.to_dict(),
            "latency_ms": trace["latency_ms"],
            "overhead_ms": trace["overhead_ms"],
            "rag_status": result.rag_status,
        }

    # -- drift (P-36) --

    def drift_for(self, org_id: str, *, n_traces: int = DRIFT_TRACES, max_age: float = 0.0) -> dict[str, Any]:
        """PSI de las últimas `n_traces` trazas del tenant contra la referencia del detector activo.

        Devuelve `{available, reason, detector_id, n_traces, n_tokens, features, level}`, con
        `features = {feature: {psi, level, n}}` (ver `vigia.drift.drift_report`) y `level` el peor.
        Solo cuentan las trazas puntuadas con el detector activo: el P(error) de otro detector no es
        comparable. `flag` se recalcula con el tau de la referencia, no con el alpha de la política.
        `max_age` > 0 reusa un cálculo previo del mismo detector con esa antigüedad en segundos.
        """
        detector = self.detector_for(org_id)
        det_id = detector.id if detector is not None else None
        if max_age > 0:
            with self._lock:
                cached = self._drift.get(org_id)
            if cached is not None and cached[1] == det_id and time.monotonic() - cached[0] < max_age:
                return cached[2]
        result = self._compute_drift(org_id, detector, n_traces)
        with self._lock:
            self._drift[org_id] = (time.monotonic(), det_id, result)
        return result

    def _compute_drift(self, org_id: str, detector: Detector | None, n_traces: int) -> dict[str, Any]:
        out: dict[str, Any] = {
            "available": False,
            "reason": None,
            "detector_id": detector.id if detector is not None else None,
            "n_traces": 0,
            "n_tokens": 0,
            "features": {},
            "level": None,
        }
        if detector is None:
            out["reason"] = "No hay un detector calibrado activo."
            return out
        reference = detector.drift_reference
        if not reference:
            out["reason"] = (
                "El detector activo no guarda la referencia de drift (es anterior a esa función): "
                "recalibrá para medir drift."
            )
            return out
        signals = self.repo(org_id).recent_token_signals(detector.id, limit_traces=n_traces)
        rows = signals["tokens"]
        out["n_traces"], out["n_tokens"] = signals["n_traces"], len(rows)
        if len(rows) < DRIFT_MIN_TOKENS:
            out["reason"] = (
                f"Hacen falta al menos {DRIFT_MIN_TOKENS} tokens puntuados con el detector activo "
                f"para medir drift; hay {len(rows)}."
            )
            return out
        tau = (reference.get("flag") or {}).get("tau")
        p_error = [float(r["p_error"]) for r in rows]
        current = {
            "p_error": p_error,
            "flag": [
                float(p >= tau) if tau is not None else float(bool(r["flag"]))
                for p, r in zip(p_error, rows, strict=True)
            ],
            "agg_mean:r_ent_n": [r["r_ent_mean"] for r in rows if _num(r["r_ent_mean"]) is not None],
            "out_entropy": [r["out_entropy"] for r in rows if _num(r["out_entropy"]) is not None],
        }
        features = drift_report(reference, current, DRIFT_FEATURES)
        out.update(
            available=True, features=features, level=worst_level([f["level"] for f in features.values()])
        )
        return out

    def drift_gauges(self) -> dict[str, float]:
        """PSI máximo entre tenants por feature, para `vigia_drift_psi{feature}` (sin org en labels)."""
        worst: dict[str, float] = {}
        for org_id in list_org_ids(self.db):
            try:
                report = self.drift_for(org_id, max_age=DRIFT_CACHE_S)
            except Exception as exc:  # noqa: BLE001 - un tenant roto no apaga la métrica de los demás
                log_event(
                    _log, "drift_error", level=logging.WARNING, org_id=org_id, error_type=type(exc).__name__
                )
                continue
            for name, entry in report["features"].items():
                worst[name] = max(worst.get(name, 0.0), float(entry["psi"]))
        return worst

    # -- salud --

    def readiness(self) -> dict[str, bool]:
        """Chequeos de `/readyz`: base escribible, worker vivo, cola bajo el tope y upstream vivo.

        La cola mide solo `DEFAULT_LANE` (purga y calibración): el backlog de webhooks de un tenant
        corre en su propio carril y no saca el nodo. La métrica `vigia_jobs_queue_depth` cuenta todo.
        Un backend con `ping()` (proxy vLLM) suma el chequeo `backend`: un upstream caído saca el nodo.
        Base y cola esperan a lo sumo `READINESS_TIMEOUT_S` (ver `_storage_checks`).
        """
        storage = self._storage_checks()
        checks = {"db": storage["db"], "worker": self.jobs.is_alive(), "queue": storage["queue"]}
        ping = getattr(self.backend, "ping", None)
        if callable(ping):
            try:
                checks["backend"] = bool(ping())
            except Exception as exc:  # noqa: BLE001 - una sonda nunca lanza
                log_event(_log, "readyz_backend_error", level=logging.WARNING, error_type=type(exc).__name__)
                checks["backend"] = False
        return checks

    def _storage_checks(self) -> dict[str, bool]:
        """Chequeos `db` y `queue` en un hilo auxiliar, esperados como máximo `READINESS_TIMEOUT_S`.

        Con la base bloqueada, la escritura espera el `RLock` compartido y el `busy_timeout`. Sin tope,
        cada sonda colgaba un hilo del threadpool hasta que se liberaba el lock. Si vence, ambos
        quedan en False. Una sonda que llega con un chequeo en curso lo reusa: no suma hilos colgados.
        """
        with self._probe_lock:
            probe = self._probe
            if probe is None or not probe[0].is_alive():
                result: dict[str, bool] = {"db": False, "queue": False}
                thread = threading.Thread(
                    target=self._probe_storage, args=(result,), name="vigia-readyz-db", daemon=True
                )
                probe = self._probe = (thread, result)
                thread.start()
        thread, result = probe
        thread.join(READINESS_TIMEOUT_S)
        if thread.is_alive():
            log_event(_log, "readyz_db_error", level=logging.WARNING, error_type="TimeoutError")
            return {"db": False, "queue": False}
        return dict(result)

    def _probe_storage(self, result: dict[str, bool]) -> None:
        try:
            with self.db.transaction():
                self.db.execute("UPDATE schema_migrations SET applied_at = applied_at WHERE 0")
            result["db"] = True
        except Exception as exc:  # noqa: BLE001
            log_event(_log, "readyz_db_error", level=logging.WARNING, error_type=type(exc).__name__)
        try:
            result["queue"] = self.jobs.depth(lane=DEFAULT_LANE) < self.settings.max_queue_depth
        except Exception as exc:  # noqa: BLE001
            log_event(_log, "readyz_queue_error", level=logging.WARNING, error_type=type(exc).__name__)

    # -- playground --

    def playground_stream(
        self,
        principal: Principal,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> Iterator[dict[str, Any]]:
        """Genera con streaming y devuelve eventos `token`, `done` o `error`; guarda la traza."""
        if not principal.can("playground:use"):
            raise PermissionError("Tu rol no tiene permiso para usar el playground.")
        return self._playground(principal, messages, max_tokens, temperature)

    def _playground(
        self, principal: Principal, messages: Any, max_tokens: Any, temperature: Any
    ) -> Iterator[dict[str, Any]]:
        try:
            clean = _check_messages(messages)
            limit = self.settings.max_tokens_limit
            if (
                isinstance(max_tokens, bool)
                or not isinstance(max_tokens, int)
                or not 1 <= max_tokens <= limit
            ):
                raise ValueError(f"max_tokens debe estar entre 1 y {limit}.")
            if (
                isinstance(temperature, bool)
                or not isinstance(temperature, int | float)
                or not math.isfinite(temperature)
                or not 0.0 <= temperature <= 2.0
            ):
                raise ValueError("La temperatura debe estar entre 0 y 2.")
        except ValueError as exc:
            yield {"type": "error", "message": str(exc)}
            return
        org_id = principal.org_id
        if self.shutting_down:
            yield {"type": "error", "message": SHUTTING_DOWN_MESSAGE}
            return
        ctx = self.context_for(principal, source="playground")
        policy = ctx.policy
        fail_open = policy.on_detector_error == "fail_open"
        unavailable = {
            "type": "error",
            "message": "El detector de confianza no está disponible y la política exige fail_closed.",
        }
        scorer = ctx.scorer
        tau: float | None = None
        degraded = False
        if scorer is not None:
            try:
                tau = float(scorer.tau(policy.alpha))
            except Exception:  # noqa: BLE001 - cualquier falla del detector sigue on_detector_error
                if not fail_open:
                    yield unavailable
                    return
                scorer, degraded = None, True
        caps = self.backend.capabilities()
        tiers = tuple(t for t in CHEAP_TIERS if t in caps.tiers)
        mode: Literal["full", "reduced"] = "full" if Tier.F1 in caps.tiers else "reduced"
        trace_id = secrets.token_hex(16)
        started = time.perf_counter()
        overhead = 0.0
        tokens: list[TokenEvent] = []
        ps: list[float | None] = []
        stream = None
        try:
            stream = self.backend.stream(
                clean, max_tokens=max_tokens, temperature=float(temperature), tiers=tiers
            )
            text_len = 0
            for ev in stream:
                if self.shutting_down:  # el cierre corta la generación; no se guarda una traza parcial
                    yield {"type": "error", "message": SHUTTING_DOWN_MESSAGE}
                    return
                t0 = time.perf_counter()
                p: float | None = None
                if scorer is not None and _visible(ev):
                    try:
                        p = scorer.score_tokens([ev])[0]
                    except Exception:  # noqa: BLE001
                        if not fail_open:
                            yield unavailable
                            return
                        scorer, degraded, tau = None, True, None
                overhead += time.perf_counter() - t0
                tokens.append(ev)
                ps.append(p)
                if _visible(ev):
                    text_len += len(ev.text)
                    st = _scored(ev, p, tau, text_len)
                    yield {"type": "token", "index": ev.index, "text": ev.text, "p_error": p, "flag": st.flag}
            t0 = time.perf_counter()
            visible = [(t, q) for t, q in zip(tokens, ps, strict=True) if _visible(t)]
            text = "".join(t.text for t, _ in visible)
            scored = [_scored(t, q, tau, len(text)) for t, q in visible]
            response_p: float | None = None
            if scorer is not None:
                try:
                    response_p = scorer.score_response(ps)
                except Exception:  # noqa: BLE001
                    if not fail_open:
                        yield unavailable
                        return
                    scorer, degraded, tau = None, True, None
                    scored = [_scored(t, None, None, len(text)) for t, _ in visible]
            spans: list[Span] = []
            if any(s.flag for s in scored):
                spans = build_spans(text, scored, gap_max=policy.gap_max, min_tokens=policy.min_tokens)
            action = decide(policy, response_p, spans)
            rag_status = None
            if action == "rag_retry":
                # El playground transmite en vivo como `stream_policy=annotate_only`: no re-consulta y
                # usa `policy.rag_fallback_action`. El texto ya viajó, así que no se oculta.
                action, rag_status = rag_fallback(policy), "unavailable"
            overhead += time.perf_counter() - t0
            result = VigiaResult(
                trace_id=trace_id,
                tokens=scored,
                spans=spans,
                response_p_error=response_p,
                action=action,
                calibration_id=ctx.scorer.id if ctx.scorer is not None else None,
                calibrated=scorer is not None,
                mode=mode,
                degraded=degraded,
                rag_status=rag_status,
                alpha=policy.alpha,
                tau=tau,
            )
            generation = Generation(
                text=text,
                tokens=tokens,
                finish_reason=getattr(stream, "finish_reason", None) or "stop",
                prompt_tokens=int(getattr(stream, "prompt_tokens", 0) or 0),
                model=self.model_info,
                mode=mode,
                timings={"overhead_s": overhead},
            )
            record = TraceRecord(
                trace_id=trace_id,
                org_id=org_id,
                key_id=None,
                created_at=datetime.now(UTC).isoformat(),
                model=self.model_info,
                prompt_messages=clean,
                generation=generation,
                result=result,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                overhead_ms=overhead * 1000.0,
                source="playground",
                rag_status=rag_status,
            )
            try:
                self.store_trace(org_id, record)
            except Exception as exc:  # noqa: BLE001 - la respuesta ya se mostró; la falla queda en el log
                log_event(
                    _log,
                    "trace_store_error",
                    level=logging.ERROR,
                    org_id=org_id,
                    error_type=type(exc).__name__,
                )
            yield {
                "type": "done",
                "trace_id": trace_id,
                "action": action,
                "spans": [{"start": s.start, "end": s.end, "score": s.score} for s in spans],
                "response_p_error": response_p,
                "calibrated": scorer is not None,
                "text": text,
            }
        except ValueError as exc:
            yield {"type": "error", "message": f"Entrada inválida: {exc}"}
        except Exception as exc:  # noqa: BLE001 - nunca filtrar detalles internos al panel
            log_event(
                _log, "playground_error", level=logging.ERROR, org_id=org_id, error_type=type(exc).__name__
            )
            yield {"type": "error", "message": "Error interno al generar la respuesta."}
        finally:
            if stream is not None:
                close = getattr(stream, "close", None)
                if callable(close):
                    close()
