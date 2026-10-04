"""API compatible con OpenAI con el campo `vigia` por choice y por chunk (P-13, P-14, ADR A-09).

`create_api_router(get_context)` devuelve un `APIRouter` que la plataforma monta. `get_context` es una
dependencia async `(request) -> RequestContext` que resuelve auth, tenant, backend, detector y política.

- Sin `vigia` en el body (lo que el SDK manda con `extra_body`), la respuesta es idéntica a OpenAI.
  La traza igual se guarda vía `sink` y el header `X-Vigia-Trace-Id` la identifica.
- Con `vigia` (`{}` o `{"tiers": [...], "alpha": 0.1}`), cada choice trae `vigia = VigiaResult`.
- La política del tenant se aplica siempre: `abstain` reemplaza el contenido por `abstain_message`
  y el texto original queda solo en la traza.
- Los errores salen como `{"error": {message, type, code, param, request_id}}`, sin stack trace.
- Un request que espera el cupo del backend no genera si el cliente cortó (499 en el log) o si
  empezó el cierre del proceso (`RequestContext.closing`: 503 `shutting_down` con Retry-After). Sin
  streaming, un cliente que corta durante la generación la interrumpe entre tokens y no deja traza.
- Una falla del servidor vLLM (`VllmProxyError`) responde 502 `upstream_unavailable` con Retry-After.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import secrets
import threading
import time
import weakref
from collections import deque
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any, Literal, TypeVar

import anyio
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool
from starlette.types import Receive, Scope, Send

from vigia.backends.vllm_proxy import VllmProxyError
from vigia.contracts import (
    USAGE_LAYER_FEATURES,
    Action,
    Backend,
    Capabilities,
    Generation,
    InvalidRequestError,
    ModelInfo,
    Policy,
    ScoredToken,
    Scorer,
    Span,
    Tier,
    TokenEvent,
    VigiaResult,
    feature_tier,
)
from vigia.policy import (
    Retriever,
    build_rag_query,
    decide,
    inject_context,
    rag_fallback,
)
from vigia.spans import build_spans

logger = logging.getLogger("vigia.api")

TRACE_HEADER = "X-Vigia-Trace-Id"
REQUEST_ID_HEADER = "X-Request-Id"
CHEAP_TIERS = (Tier.F0, Tier.F1, Tier.U)
REMOTE_MAX_CONCURRENCY = 16  # cupo de un proxy en modo reducido que no declara max_concurrency
RETRY_AFTER_S = 5  # Retry-After ante cierre del proceso o upstream caído
_POLL_S = 0.2  # cada cuánto se mira si el cliente cortó o empezó el cierre mientras se espera
_HARMONY_CHANNELS = frozenset({"analysis", "commentary", "header", "none"})
_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_STARTED = int(time.time())
_M = TypeVar("_M", bound="BaseModel")
_R = TypeVar("_R")


# --- Tipos públicos -----------------------------------------------------------------------------


@dataclass
class TraceRecord:
    """Todo lo que la plataforma necesita para guardar una traza. Contiene el texto original."""

    trace_id: str
    org_id: str | None
    key_id: str | None
    created_at: str  # ISO-8601 UTC
    model: ModelInfo
    prompt_messages: list[dict[str, str]]
    generation: Generation
    result: VigiaResult
    latency_ms: float
    overhead_ms: float
    source: str = "api"
    rag_status: str | None = None  # None | retried | error | unavailable


@dataclass
class RequestContext:
    backend: Backend
    scorer: Scorer | None
    policy: Policy
    org_id: str | None = None
    key_id: str | None = None
    sink: Callable[[TraceRecord], None] | None = None
    trace_lookup: Callable[[str], dict[str, Any] | None] | None = None
    retriever: Retriever | None = None
    source: Literal["api", "playground"] = "api"
    # True desde que el proceso empezó a cerrarse: un request que todavía espera el cupo del backend
    # responde 503 sin generar. La plataforma lo llena con `runtime.shutting_down`.
    closing: Callable[[], bool] | None = None


# --- Errores ------------------------------------------------------------------------------------


class APIError(Exception):
    def __init__(
        self,
        status: int,
        message: str,
        type_: str,
        code: str,
        param: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.type = type_
        self.code = code
        self.param = param
        self.headers = headers or {}


class DetectorUnavailable(Exception):
    """El detector falló con `on_detector_error=fail_closed`."""


def _detector_error() -> APIError:
    return APIError(
        503,
        "El detector de confianza no está disponible y la política exige fail_closed.",
        "service_unavailable_error",
        "detector_unavailable",
    )


class _ClientGone(Exception):
    """El cliente cortó la conexión antes de recibir la respuesta: no se genera ni se guarda nada."""


def _shutting_down_error() -> APIError:
    return APIError(
        503,
        "El servidor se está cerrando. Reintentá en unos segundos.",
        "service_unavailable_error",
        "shutting_down",
        headers={"Retry-After": str(RETRY_AFTER_S)},
    )


def _upstream_error() -> APIError:
    return APIError(
        502,
        "El servidor del modelo no respondió o respondió con un error. Reintentá en unos segundos.",
        "service_unavailable_error",
        "upstream_unavailable",
        headers={"Retry-After": str(RETRY_AFTER_S)},
    )


def _log_upstream(exc: BaseException, request_id: str) -> None:
    logger.warning(
        "falla del servidor del modelo (error_type=%s, request_id=%s): %s",
        type(exc).__name__,
        request_id,
        exc,
        extra={"error_type": type(exc).__name__, "request_id": request_id},
    )


def _error_body(err: APIError, request_id: str) -> dict[str, Any]:
    return {
        "error": {
            "message": err.message,
            "type": err.type,
            "code": err.code,
            "param": err.param,
            "request_id": request_id,
        }
    }


def _error_response(err: APIError, request_id: str) -> JSONResponse:
    return JSONResponse(
        _error_body(err, request_id),
        status_code=err.status,
        headers={**err.headers, REQUEST_ID_HEADER: request_id},
    )


def _invalid_prompt(exc: Exception) -> APIError:
    """El backend rechazó el prompt al abrir el stream: es un error del cliente, no interno."""
    if isinstance(exc, InvalidRequestError):
        return APIError(400, str(exc), "invalid_request_error", exc.code, "messages")
    return APIError(
        400,
        "El backend rechazó el prompt: excede el contexto del modelo o tiene contenido inválido.",
        "invalid_request_error",
        "invalid_prompt",
        "messages",
    )


def _open_stream(backend: Backend, messages: list[dict[str, str]], params: _Params) -> Any:
    """Abre el stream del backend. Lo que el backend rechaza al validar la entrada sale como 400.

    El contrato del backend valida el prompt al llamar a `stream()` (antes de generar); un
    `ValueError` ahí es entrada inválida. Las fallas durante la generación siguen siendo 500.
    """
    try:
        return backend.stream(messages, **params.kwargs())
    except (InvalidRequestError, ValueError) as exc:
        raise _invalid_prompt(exc) from exc


# --- Cupo de generación -------------------------------------------------------------------------


class GenerationBusy(Exception):
    """La cola de generación está llena o la espera venció."""


class GenerationGate:
    """Cupo de generaciones simultáneas de un backend. La espera ocurre en el event loop.

    Una generación toma el cupo antes de pedir un hilo del threadpool y lo suelta al cerrar el
    stream. Así ningún hilo del pool queda bloqueado esperando el lock del backend: el threadpool
    por defecto (auth, /readyz) nunca se agota por generaciones encoladas.
    """

    def __init__(self, slots: int, *, max_waiting: int, timeout_s: float) -> None:
        if slots < 1 or max_waiting < 0 or timeout_s <= 0:
            raise ValueError("slots >= 1, max_waiting >= 0 y timeout_s > 0")
        self.slots = slots
        self.max_waiting = max_waiting
        self.timeout_s = timeout_s
        self._mutex = threading.Lock()
        self._free = slots
        self._waiters: deque[asyncio.Future[None]] = deque()

    @property
    def waiting(self) -> int:
        return len(self._waiters)

    async def acquire(self) -> None:
        """Toma un cupo en orden FIFO. Lanza GenerationBusy si la cola está llena o vence la espera."""
        with self._mutex:
            if self._free > 0 and not self._waiters:
                self._free -= 1
                return
            if len(self._waiters) >= self.max_waiting:
                raise GenerationBusy
            fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
            self._waiters.append(fut)
        try:
            await asyncio.wait_for(asyncio.shield(fut), self.timeout_s)
        except BaseException as exc:
            with self._mutex:
                granted = fut not in self._waiters
                if not granted:
                    self._waiters.remove(fut)
            if granted and isinstance(exc, TimeoutError):
                return  # el cupo llegó justo al vencer la espera: se usa
            if granted:
                self.release()
            if isinstance(exc, TimeoutError):
                raise GenerationBusy from exc
            raise

    def release(self) -> None:
        """Devuelve el cupo; si hay espera, lo pasa al primero. Seguro desde cualquier hilo."""
        with self._mutex:
            while self._waiters:
                fut = self._waiters.popleft()
                loop = fut.get_loop()
                if loop.is_closed():
                    continue
                loop.call_soon_threadsafe(_grant, fut)
                return
            self._free += 1


def _grant(fut: asyncio.Future[None]) -> None:
    if not fut.done():
        fut.set_result(None)


_GATES: weakref.WeakKeyDictionary[Any, GenerationGate] = weakref.WeakKeyDictionary()
_GATES_BY_ID: dict[int, tuple[Any, GenerationGate]] = {}  # backends sin weakref o sin hash
_GATES_LOCK = threading.Lock()


def generation_gate(backend: Backend, *, max_waiting: int = 64, timeout_s: float = 120.0) -> GenerationGate:
    """Cupo compartido por todos los routers que usan el mismo backend.

    El tamaño sale de `backend.max_concurrency` (duck typing). Sin ese dato, un backend en el
    proceso tiene cupo 1 (el backend HF serializa la generación con un lock) y un proxy en modo
    reducido (`mode == "reduced"`, ADR A-08) tiene `REMOTE_MAX_CONCURRENCY`: vLLM hace batching y
    serializar sus requests en Vigía multiplica la latencia. `max_waiting` y `timeout_s` los fija el
    primer llamador.
    """
    with _GATES_LOCK:
        try:
            gate = _GATES.get(backend)
        except TypeError:
            gate = _GATES_BY_ID.get(id(backend), (None, None))[1]
        if gate is None:
            default = REMOTE_MAX_CONCURRENCY if getattr(backend, "mode", None) == "reduced" else 1
            slots = getattr(backend, "max_concurrency", default)
            valid = isinstance(slots, int) and not isinstance(slots, bool) and slots >= 1
            gate = GenerationGate(slots if valid else default, max_waiting=max_waiting, timeout_s=timeout_s)
            try:
                _GATES[backend] = gate
            except TypeError:
                _GATES_BY_ID[id(backend)] = (backend, gate)  # retiene el backend: el id no se reusa
        return gate


def _busy_error(gate: GenerationGate) -> APIError:
    return APIError(
        503,
        "El servidor está saturado de generaciones. Reintentá en unos segundos.",
        "service_unavailable_error",
        "server_overloaded",
        headers={"Retry-After": str(max(1, min(60, gate.waiting + 1)))},
    )


class _ClosingStreamingResponse(StreamingResponse):
    """StreamingResponse que siempre ejecuta `on_close` al terminar, aun si el cliente corta.

    Starlette no cierra el iterador sync cuando el cliente se desconecta; sin esto, el generador del
    backend queda suspendido con su lock tomado hasta que lo recolecte el GC cíclico.
    """

    def __init__(self, events: Iterator[str], on_close: Callable[[], Awaitable[None]], **kwargs: Any) -> None:
        super().__init__(iterate_in_threadpool(events), **kwargs)
        self._on_close = on_close

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            with anyio.CancelScope(shield=True):
                await self._on_close()


# --- Modelos de entrada -------------------------------------------------------------------------


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TextPart(_Strict):
    type: Literal["text"]
    text: str


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: Literal["system", "developer", "user", "assistant"]
    content: str | list[TextPart] | None = None

    def as_backend(self) -> dict[str, str]:
        if isinstance(self.content, list):
            text = "".join(p.text for p in self.content)
        else:
            text = self.content or ""
        return {"role": "system" if self.role == "developer" else self.role, "content": text}


class VigiaOptions(_Strict):
    tiers: list[Tier] | None = Field(default=None, min_length=1, max_length=5)
    alpha: float | None = Field(default=None, gt=0.0, lt=1.0)


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="ignore")
    include_usage: bool = False


class _CommonRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: str = Field(min_length=1, max_length=256)
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    n: int = Field(default=1, ge=1, le=1)
    stream: bool = False
    stream_options: StreamOptions | None = None
    stop: str | list[str] | None = None
    seed: int | None = Field(default=None, ge=-(2**63), le=2**63 - 1)
    vigia: VigiaOptions | None = None

    @field_validator("temperature", "top_p", "n", "stream", mode="before")
    @classmethod
    def _null_is_default(cls, v: Any, info: ValidationInfo) -> Any:
        """La spec de OpenAI declara estos campos nullable: `null` equivale al default."""
        return cls.model_fields[info.field_name].default if v is None else v  # type: ignore[index]

    @field_validator("stop")
    @classmethod
    def _check_stop(cls, v: str | list[str] | None) -> str | list[str] | None:
        items = [v] if isinstance(v, str) else (v or [])
        if len(items) > 4 or any(not 1 <= len(s) <= 256 for s in items):
            raise ValueError("stop admite hasta 4 secuencias de 1 a 256 caracteres")
        return v

    def stop_list(self) -> list[str] | None:
        if self.stop is None:
            return None
        return [self.stop] if isinstance(self.stop, str) else list(self.stop)


class ChatCompletionRequest(_CommonRequest):
    messages: list[ChatMessage] = Field(min_length=1, max_length=512)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    logprobs: bool = False
    top_logprobs: int | None = Field(default=None, ge=0, le=20)

    @field_validator("logprobs", mode="before")
    @classmethod
    def _null_logprobs(cls, v: Any) -> Any:
        return False if v is None else v

    @model_validator(mode="after")
    def _non_empty(self) -> ChatCompletionRequest:
        if not any(m.as_backend()["content"].strip() for m in self.messages):
            raise ValueError("messages no tiene contenido")
        return self


class CompletionRequest(_CommonRequest):
    prompt: str | list[str] = Field(min_length=1)
    logprobs: int | None = Field(default=None, ge=0, le=5)

    @field_validator("prompt")
    @classmethod
    def _single_prompt(cls, v: str | list[str]) -> str | list[str]:
        if isinstance(v, list) and len(v) != 1:
            raise ValueError("prompt admite un solo texto")
        text = v if isinstance(v, str) else v[0]
        if not text.strip():
            raise ValueError("prompt está vacío")
        return v

    def prompt_text(self) -> str:
        return self.prompt if isinstance(self.prompt, str) else self.prompt[0]


# --- Pipeline -----------------------------------------------------------------------------------


@dataclass
class _Params:
    max_tokens: int
    temperature: float
    top_p: float
    seed: int | None
    stop: list[str] | None
    tiers: tuple[Tier, ...]

    def kwargs(self) -> dict[str, Any]:
        return {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
            "stop": self.stop,
            "tiers": self.tiers,
        }


def _valid_prob(x: Any) -> bool:
    return isinstance(x, int | float) and not isinstance(x, bool) and math.isfinite(x) and 0.0 <= x <= 1.0


class _Scoring:
    """Envuelve al detector y aplica `on_detector_error`."""

    def __init__(self, scorer: Scorer | None, policy: Policy, alpha: float, info: ModelInfo) -> None:
        self.scorer = scorer
        self.fail_open = policy.on_detector_error == "fail_open"
        self.degraded = False
        self.tau: float | None = None
        if scorer is None:
            return
        try:
            scorer.check_model(info)
            tau = scorer.tau(alpha)
            if not _valid_prob(tau):
                raise ValueError("tau fuera de [0, 1]")
            self.tau = float(tau)
        except Exception as exc:  # noqa: BLE001 - cualquier falla del detector sigue on_detector_error
            self._fail(exc)

    @property
    def active(self) -> bool:
        return self.scorer is not None and not self.degraded

    def _fail(self, exc: Exception) -> None:
        logger.warning("falla del detector: %s", type(exc).__name__)
        if not self.fail_open:
            raise DetectorUnavailable from exc
        self.degraded = True
        self.tau = None

    def score(self, tokens: list[TokenEvent]) -> list[float | None]:
        if not self.active or not tokens:
            return [None] * len(tokens)
        try:
            ps = list(self.scorer.score_tokens(tokens))  # type: ignore[union-attr]
            if len(ps) != len(tokens) or not all(p is None or _valid_prob(p) for p in ps):
                raise ValueError("score_tokens devolvió una forma o valores inválidos")
            return [None if p is None else float(p) for p in ps]
        except Exception as exc:  # noqa: BLE001
            self._fail(exc)
            return [None] * len(tokens)

    def response(self, ps: list[float | None]) -> float | None:
        if not self.active:
            return None
        try:
            r = self.scorer.score_response(ps)  # type: ignore[union-attr]
            if r is not None and not _valid_prob(r):
                raise ValueError("score_response fuera de [0, 1]")
            return None if r is None else float(r)
        except Exception as exc:  # noqa: BLE001
            self._fail(exc)
            return None


def _detector_tiers(scorer: Scorer | None, caps: Capabilities) -> tuple[Tier, ...]:
    """Niveles que el backend debe generar para cubrir las features del detector, en orden de caps.

    Las features de uso existen en F1 (ponderadas) y en U (solo IDs): sin F1 alcanza con U.
    Un nivel que el backend no tiene se ignora acá; esa incompatibilidad es del detector.
    """
    needed: set[Tier] = set()
    for name in getattr(scorer, "feature_names", None) or ():
        try:
            tier = feature_tier(name)
        except ValueError:
            continue
        base = name.split("@")[0].split(":")[-1]
        if tier == Tier.F1 and Tier.F1 not in caps.tiers and base in USAGE_LAYER_FEATURES:
            tier = Tier.U
        needed.add(tier)
    return tuple(t for t in caps.tiers if t in needed)


def _min_calibrated_alpha(scorer: Scorer | None) -> float | None:
    """Menor alpha calibrado del detector (`thresholds`, duck typing); None si no se conoce."""
    try:
        values = [float(t["alpha"]) for t in getattr(scorer, "thresholds", None) or ()]
    except (TypeError, KeyError, ValueError):
        return None
    finite = [a for a in values if math.isfinite(a)]
    return min(finite) if finite else None


def _unsupported(param: str, message: str) -> APIError:
    return APIError(422, message, "invalid_request_error", "unsupported_value", param)


# Parámetros OpenAI que cambian la semántica de la respuesta y que Vigía no implementa. Con un valor
# fuera de los inocuos (ausente, null o el default de OpenAI) se rechazan: ignorarlos en silencio
# devolvería texto libre a un cliente que espera tool_calls, JSON o el prompt antepuesto.
_HARMLESS_VALUES: dict[str, tuple[Any, ...]] = {
    "tools": (None, []),
    "tool_choice": (None, "none"),
    "functions": (None, []),
    "function_call": (None, "none"),
    "response_format": (None, {"type": "text"}),
    "logit_bias": (None, {}),
    "echo": (None, False),
    "best_of": (None, 1),
    "suffix": (None, ""),
    "modalities": (None, ["text"]),  # pedir audio y recibir texto sin aviso rompe al cliente
    "audio": (None,),
    "reasoning_effort": (None,),  # ponytail: aplicarlo al prompt harmony de gpt-oss cuando haga falta
}
_UNSUPPORTED_PARAMS: dict[str, tuple[str, ...]] = {
    "chat": (
        "tools",
        "tool_choice",
        "functions",
        "function_call",
        "response_format",
        "logit_bias",
        "modalities",
        "audio",
        "reasoning_effort",
    ),
    "text": ("echo", "best_of", "suffix", "logit_bias"),
}


def _reject_unsupported(data: dict[str, Any], kind: Literal["chat", "text"]) -> None:
    for name in _UNSUPPORTED_PARAMS[kind]:
        if name in data and data[name] not in _HARMLESS_VALUES[name]:
            raise _unsupported(name, f"{name} no está soportado por Vigía.")


def _visible(t: TokenEvent) -> bool:
    return t.channel == "final" and not t.is_special


def _result_status(tokens: list[TokenEvent], text: str, finish_reason: str, degraded: bool) -> str:
    """`VigiaResult.status` de una generación que terminó (contracts.VigiaResult).

    `detector_error`: el detector falló con fail_open. Sin texto visible: `truncated_before_final` si una
    respuesta harmony cortó por longitud antes del canal final, `no_final_channel` si terminó sin
    abrirlo, `empty_generation` en el resto de los casos.
    """
    if degraded:
        return "detector_error"
    if text.strip():
        return "ok"
    if any(t.channel in _HARMONY_CHANNELS for t in tokens):
        return "truncated_before_final" if finish_reason == "length" else "no_final_channel"
    return "empty_generation"


def _collect(stream: Any, cancel: threading.Event | None) -> list[TokenEvent]:
    """Consume el stream del backend; corta entre tokens si `cancel` se marca (cliente desconectado)."""
    if cancel is None:
        return list(stream)
    tokens: list[TokenEvent] = []
    for ev in stream:
        if cancel.is_set():
            raise _ClientGone
        tokens.append(ev)
    return tokens


def _scored(t: TokenEvent, p: float | None, tau: float | None, text_len: int) -> ScoredToken:
    start = min(max(t.start, 0), text_len)
    end = min(max(t.end, start), text_len)
    flag = p is not None and tau is not None and p >= tau
    return ScoredToken(index=t.index, text=t.text, start=start, end=end, p_error=p, flag=flag)


@dataclass
class _Outcome:
    trace_id: str
    info: ModelInfo
    mode: Literal["full", "reduced"]
    tokens: list[TokenEvent]
    text: str
    result: VigiaResult
    finish_reason: Literal["stop", "length"]
    prompt_tokens: int
    abstain_message: str
    gen_s: float = 0.0
    overhead_s: float = 0.0
    rag_status: str | None = None

    @property
    def content(self) -> str:
        return self.abstain_message if self.result.action == "abstain" else self.text

    def client_result(self) -> VigiaResult:
        """Resultado para el cliente: con abstain no viaja ningún token del texto original."""
        if self.result.action == "abstain":
            return replace(self.result, tokens=[], spans=[])
        return self.result


class _Engine:
    """Un request: genera, puntúa, arma fragmentos, decide y guarda la traza."""

    def __init__(
        self,
        ctx: RequestContext,
        messages: list[dict[str, str]],
        params: _Params,
        alpha: float,
        cancel: threading.Event | None = None,
    ) -> None:
        self.ctx = ctx
        self.cancel = cancel
        self.messages = messages
        self.params = params
        self.alpha = alpha
        self.trace_id = secrets.token_hex(16)
        self.started = time.perf_counter()
        self.info = ctx.backend.model_info()
        caps = ctx.backend.capabilities()
        self.mode: Literal["full", "reduced"] = "full" if Tier.F1 in caps.tiers else "reduced"
        self.scoring = _Scoring(ctx.scorer, ctx.policy, alpha, self.info)

    def prompt_tokens(self, messages: list[dict[str, str]], stream: Any = None) -> int:
        """Tokens del prompt: los que informa el stream del backend (`TokenStream.prompt_tokens`).

        `count_prompt_tokens(messages)` (duck typing) queda para backends cuyo stream no los informa.
        """
        reported = getattr(stream, "prompt_tokens", None)
        if isinstance(reported, int) and not isinstance(reported, bool) and reported > 0:
            return reported
        count = getattr(self.ctx.backend, "count_prompt_tokens", None)
        return int(count(messages)) if callable(count) else 0

    def finalize(self, tokens: list[TokenEvent], ps: list[float | None]) -> tuple[str, VigiaResult]:
        policy = self.ctx.policy
        # score_response puede degradar al detector (fail_open): eso anula tau y los p_error por token,
        # así que va antes de marcar tokens. Una señal degradada no deja flags ni fragmentos (A-06).
        resp = self.scoring.response(ps)
        if self.scoring.degraded:
            ps = [None] * len(ps)
        visible = [(t, p) for t, p in zip(tokens, ps, strict=True) if _visible(t)]
        text = "".join(t.text for t, _ in visible)
        scored = [_scored(t, p, self.scoring.tau, len(text)) for t, p in visible]
        spans: list[Span] = []
        if any(s.flag for s in scored):
            spans = build_spans(text, scored, gap_max=policy.gap_max, min_tokens=policy.min_tokens)
        action: Action = decide(policy, resp, spans)
        result = VigiaResult(
            trace_id=self.trace_id,
            tokens=scored,
            spans=spans,
            response_p_error=resp,
            action=action,
            calibration_id=self.ctx.scorer.id if self.ctx.scorer is not None else None,
            calibrated=self.scoring.active,
            mode=self.mode,
            degraded=self.scoring.degraded,
            alpha=self.alpha,
            tau=self.scoring.tau,
        )
        return text, result

    def run(self) -> _Outcome:
        """Pipeline completo sin streaming, con rag_retry estilo FLARE.

        Si no se puede re-consultar (sin retriever, reintentos agotados, retriever caído o prompt con
        contexto que el backend rechaza), la acción pasa a `policy.rag_fallback_action`.
        """
        ctx, policy = self.ctx, self.ctx.policy
        question = next((m["content"] for m in reversed(self.messages) if m["role"] == "user"), "")
        messages = self.messages
        attempts = 0
        rag_status: str | None = None
        gen_s = overhead_s = 0.0
        stream = _open_stream(ctx.backend, messages, self.params)
        while True:
            g0 = time.perf_counter()
            try:
                tokens = _collect(stream, self.cancel)
            except BaseException:
                _close_backend(stream)  # libera el lock y los hooks del backend ya, no en el GC
                raise
            o0 = time.perf_counter()
            gen_s += o0 - g0
            ps = self.scoring.score(tokens)
            text, result = self.finalize(tokens, ps)
            overhead_s += time.perf_counter() - o0
            if result.action != "rag_retry":
                break
            if ctx.retriever is None or attempts >= policy.rag_max_retries:
                rag_status = rag_status or "unavailable"
                result.action = rag_fallback(policy)
                break
            try:
                docs = ctx.retriever.retrieve(build_rag_query(question, text, result.spans))
                retry_messages = inject_context(self.messages, docs)
                stream = _open_stream(ctx.backend, retry_messages, self.params)
            except Exception as exc:  # noqa: BLE001 - el retriever es externo; falla -> acción de respaldo
                logger.warning("falla de la re-consulta: %s", type(exc).__name__)
                rag_status = "error"
                result.action = rag_fallback(policy)
                break
            attempts += 1
            rag_status = "retried"
            messages = retry_messages
        result.rag_status = rag_status
        out = self.outcome(tokens, text, result, messages, stream)
        out.gen_s, out.overhead_s, out.rag_status = gen_s, overhead_s, rag_status
        return out

    def outcome(
        self,
        tokens: list[TokenEvent],
        text: str,
        result: VigiaResult,
        messages: list[dict[str, str]],
        stream: Any = None,
    ) -> _Outcome:
        # El backend sabe por qué paró (max_tokens, contexto del modelo, EOS, stop) y cuántos tokens
        # tuvo el prompt. Los cálculos locales quedan solo para backends que no lo informan.
        backend_finish = getattr(stream, "finish_reason", None)
        fr: Literal["stop", "length"]
        if backend_finish in ("stop", "length"):
            fr = backend_finish
        else:
            fr = "length" if len(tokens) >= self.params.max_tokens else "stop"
        if result.status == "ok":  # un estado ya asignado (cancelled) no se pisa
            result.status = _result_status(tokens, text, fr, result.degraded)
        return _Outcome(
            trace_id=self.trace_id,
            info=self.info,
            mode=self.mode,
            tokens=tokens,
            text=text,
            result=result,
            finish_reason=fr,
            prompt_tokens=self.prompt_tokens(messages, stream),
            abstain_message=self.ctx.policy.abstain_message,
        )

    def store(self, out: _Outcome, finish_reason: Literal["stop", "length", "error"] | None = None) -> None:
        if self.ctx.sink is None:
            return
        gen = Generation(
            text=out.text,
            tokens=out.tokens,
            finish_reason=finish_reason or out.finish_reason,
            prompt_tokens=out.prompt_tokens,
            model=out.info,
            mode=out.mode,
            timings={"decode_s": out.gen_s, "overhead_s": out.overhead_s},
        )
        record = TraceRecord(
            trace_id=out.trace_id,
            org_id=self.ctx.org_id,
            key_id=self.ctx.key_id,
            created_at=datetime.now(UTC).isoformat(),
            model=out.info,
            prompt_messages=self.messages,
            generation=gen,
            result=out.result,
            latency_ms=(time.perf_counter() - self.started) * 1000.0,
            overhead_ms=out.overhead_s * 1000.0,
            source=self.ctx.source,
            rag_status=out.rag_status,
        )
        try:
            self.ctx.sink(record)
        except Exception:  # noqa: BLE001 - guardar la traza no debe tumbar la respuesta ya generada
            logger.exception("no se pudo guardar la traza %s", out.trace_id)


# --- Formato de salida --------------------------------------------------------------------------


def _usage(out: _Outcome) -> dict[str, int]:
    completion = len(out.tokens)
    return {
        "prompt_tokens": out.prompt_tokens,
        "completion_tokens": completion,
        "total_tokens": out.prompt_tokens + completion,
    }


def _logprob_entry(t: ScoredToken | TokenEvent, logprob: float) -> dict[str, Any]:
    return {"token": t.text, "logprob": logprob, "bytes": list(t.text.encode("utf-8")), "top_logprobs": []}


def _chat_logprobs(out: _Outcome) -> dict[str, Any] | None:
    visible = [t for t in out.tokens if _visible(t)]
    if out.result.action == "abstain" or any(t.logprob is None for t in visible):
        return None
    return {"content": [_logprob_entry(t, float(t.logprob)) for t in visible], "refusal": None}  # type: ignore[arg-type]


def _sse(obj: dict[str, Any] | str) -> str:
    payload = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False, allow_nan=False)
    return f"data: {payload}\n\n"


class _Shape:
    """Arma respuestas y chunks para chat (`chat.completion`) o texto (`text_completion`)."""

    def __init__(self, kind: Literal["chat", "text"], model: str, with_vigia: bool, logprobs: bool) -> None:
        self.kind = kind
        self.model = model
        self.with_vigia = with_vigia
        self.logprobs = logprobs
        self.id = ("chatcmpl-" if kind == "chat" else "cmpl-") + secrets.token_hex(12)
        self.created = int(time.time())

    def full(self, out: _Outcome) -> dict[str, Any]:
        content = out.content
        if self.kind == "chat":
            choice: dict[str, Any] = {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": out.finish_reason,
                "logprobs": _chat_logprobs(out) if self.logprobs else None,
            }
            obj = "chat.completion"
        else:
            choice = {"index": 0, "text": content, "finish_reason": out.finish_reason, "logprobs": None}
            obj = "text_completion"
        if self.with_vigia:
            choice["vigia"] = out.client_result().to_dict()
        return {
            "id": self.id,
            "object": obj,
            "created": self.created,
            "model": self.model,
            "choices": [choice],
            "usage": _usage(out),
        }

    def chunk(
        self,
        text: str | None,
        vigia: dict[str, Any] | None,
        finish_reason: str | None = None,
        role: bool = False,
        logprobs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.kind == "chat":
            delta: dict[str, Any] = {}
            if role:
                delta["role"] = "assistant"
            if text is not None:
                delta["content"] = text
            choice: dict[str, Any] = {
                "index": 0,
                "delta": delta,
                "logprobs": logprobs,
                "finish_reason": finish_reason,
            }
            obj = "chat.completion.chunk"
        else:
            choice = {"index": 0, "text": text or "", "logprobs": None, "finish_reason": finish_reason}
            obj = "text_completion"
        if self.with_vigia and vigia is not None:
            choice["vigia"] = vigia
        return {
            "id": self.id,
            "object": obj,
            "created": self.created,
            "model": self.model,
            "choices": [choice],
        }

    def usage_chunk(self, out: _Outcome) -> dict[str, Any]:
        obj = "chat.completion.chunk" if self.kind == "chat" else "text_completion"
        return {
            "id": self.id,
            "object": obj,
            "created": self.created,
            "model": self.model,
            "choices": [],
            "usage": _usage(out),
        }

    def token_logprobs(self, t: TokenEvent) -> dict[str, Any] | None:
        if self.kind != "chat" or not self.logprobs or t.logprob is None:
            return None
        return {"content": [_logprob_entry(t, float(t.logprob))], "refusal": None}


def _final_vigia(result: VigiaResult) -> dict[str, Any]:
    data = result.to_dict()
    data["tokens"] = []
    return data


# --- Router -------------------------------------------------------------------------------------


def _request_id(request: Request) -> str:
    rid = getattr(request.state, "request_id", None)
    if isinstance(rid, str) and _ID_RE.match(rid):
        return rid
    header = request.headers.get(REQUEST_ID_HEADER, "")
    return header if _ID_RE.match(header) else "req_" + secrets.token_hex(8)


async def _read_body(request: Request) -> dict[str, Any]:
    raw = await request.body()
    try:  # un anidamiento muy profundo agota la recursión del parser: es JSON que no se acepta
        data = json.loads(raw)
    except (ValueError, UnicodeDecodeError, RecursionError) as exc:
        raise APIError(400, "El body no es JSON válido.", "invalid_request_error", "invalid_json") from exc
    if not isinstance(data, dict):
        raise APIError(400, "El body debe ser un objeto JSON.", "invalid_request_error", "invalid_json")
    try:  # JSON admite escapes como "\ud800" que no son Unicode válido; el tokenizer los rechaza
        json.dumps(data, ensure_ascii=False).encode("utf-8")
    except RecursionError as exc:
        raise APIError(400, "El body no es JSON válido.", "invalid_request_error", "invalid_json") from exc
    except UnicodeEncodeError as exc:
        raise APIError(
            400,
            "El body contiene texto Unicode inválido (surrogates sueltos).",
            "invalid_request_error",
            "invalid_value",
        ) from exc
    return data


def _validate(model: type[_M], data: dict[str, Any]) -> _M:
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        first = exc.errors(include_input=False, include_url=False)[0]
        param = ".".join(str(p) for p in first["loc"]) or None
        msg = str(first["msg"]).removeprefix("Value error, ")
        raise APIError(
            422,
            f"Parámetro inválido{f' en {param!r}' if param else ''}: {msg}",
            "invalid_request_error",
            "invalid_value",
            param,
        ) from exc


def create_api_router(
    get_context: Callable[[Request], Awaitable[RequestContext]],
    *,
    max_tokens_limit: int = 4096,
    default_max_tokens: int = 512,
    max_queued_generations: int = 64,
    queue_timeout_s: float = 120.0,
) -> APIRouter:
    """Router OpenAI-compatible. `max_tokens` fuera de 1..max_tokens_limit devuelve 422.

    Las generaciones de un backend comparten un `GenerationGate`: las que exceden el cupo esperan
    en el event loop hasta `queue_timeout_s`; con más de `max_queued_generations` en espera, o al
    vencer la espera, la respuesta es 503 `server_overloaded` con `Retry-After`.
    """
    if not 1 <= default_max_tokens <= max_tokens_limit:
        raise ValueError("default_max_tokens debe estar entre 1 y max_tokens_limit")
    if max_queued_generations < 0 or queue_timeout_s <= 0:
        raise ValueError("max_queued_generations >= 0 y queue_timeout_s > 0")
    router = APIRouter()

    def params_for(req: _CommonRequest, max_tokens: int | None, ctx: RequestContext) -> _Params:
        mt = default_max_tokens if max_tokens is None else max_tokens
        if mt > max_tokens_limit:
            raise APIError(
                422,
                f"max_tokens debe estar entre 1 y {max_tokens_limit}.",
                "invalid_request_error",
                "invalid_value",
                "max_tokens",
            )
        caps = ctx.backend.capabilities()
        if req.vigia is not None and req.vigia.tiers:
            unsupported = [t.value for t in req.vigia.tiers if t not in caps.tiers]
            if unsupported:
                raise APIError(
                    422,
                    f"El backend no soporta los niveles {', '.join(unsupported)}.",
                    "invalid_request_error",
                    "unsupported_tier",
                    "vigia.tiers",
                )
            tiers = tuple(dict.fromkeys(req.vigia.tiers))
        else:
            tiers = tuple(t for t in CHEAP_TIERS if t in caps.tiers)
        # Los niveles pedidos se suman a los que exige el detector: un cliente no puede apagar
        # features del detector (se imputarían con la media y cambiaría la acción).
        tiers += tuple(t for t in _detector_tiers(ctx.scorer, caps) if t not in tiers)
        return _Params(mt, req.temperature, req.top_p, req.seed, req.stop_list(), tiers)

    def alpha_for(req: _CommonRequest, ctx: RequestContext) -> float:
        if req.vigia is None or req.vigia.alpha is None:
            return ctx.policy.alpha
        alpha = req.vigia.alpha
        min_alpha = _min_calibrated_alpha(ctx.scorer)
        if min_alpha is not None and alpha < min_alpha:
            raise APIError(
                422,
                f"vigia.alpha debe ser >= {min_alpha:g}, el mínimo calibrado del detector activo.",
                "invalid_request_error",
                "invalid_value",
                "vigia.alpha",
            )
        return alpha

    def check_model(req: _CommonRequest, ctx: RequestContext) -> None:
        if req.model != ctx.backend.model_info().model_id:
            raise APIError(
                404,
                f"El modelo {req.model!r} no existe.",
                "invalid_request_error",
                "model_not_found",
                "model",
            )

    async def handle(
        request: Request,
        ctx: RequestContext,
        kind: Literal["chat", "text"],
    ) -> JSONResponse | StreamingResponse:
        request_id = _request_id(request)
        try:
            data = await _read_body(request)
            if kind == "chat":
                req: _CommonRequest = _validate(ChatCompletionRequest, data)
                _reject_unsupported(data, kind)
                assert isinstance(req, ChatCompletionRequest)  # noqa: S101 - estrechamiento de tipo
                messages = [m.as_backend() for m in req.messages]
                max_tokens = (
                    req.max_completion_tokens if req.max_completion_tokens is not None else req.max_tokens
                )
                want_logprobs = req.logprobs
                if req.top_logprobs:
                    raise _unsupported("top_logprobs", "top_logprobs no está soportado; usá logprobs=true.")
            else:
                req = _validate(CompletionRequest, data)
                _reject_unsupported(data, kind)
                assert isinstance(req, CompletionRequest)  # noqa: S101
                if req.logprobs is not None:
                    raise _unsupported("logprobs", "logprobs no está soportado en /v1/completions.")
                # ponytail: el contrato Backend solo acepta mensajes; el prompt va como mensaje de usuario.
                messages = [{"role": "user", "content": req.prompt_text()}]
                max_tokens = req.max_tokens
                want_logprobs = False
            check_model(req, ctx)
            params = params_for(req, max_tokens, ctx)
            alpha = alpha_for(req, ctx)
            shape = _Shape(kind, ctx.backend.model_info().model_id, req.vigia is not None, want_logprobs)
            include_usage = bool(req.stream_options and req.stream_options.include_usage)
            headers = {REQUEST_ID_HEADER: request_id}
            gate = generation_gate(ctx.backend, max_waiting=max_queued_generations, timeout_s=queue_timeout_s)
            await _take_slot(gate, request, ctx.closing)
            if not req.stream:
                try:
                    out = await _run_watched(request, _run_full, ctx, messages, params, alpha)
                finally:
                    gate.release()
                headers[TRACE_HEADER] = out.trace_id
                return JSONResponse(shape.full(out), headers=headers)
            try:
                live = await _run_watched(
                    request, _start_stream, ctx, messages, params, alpha, shape, include_usage, request_id
                )
            except BaseException:
                gate.release()
                raise
            if not live.holds_backend:
                gate.release()  # buffer: la generación ya terminó; un cliente lento no frena a otros
            if await request.is_disconnected():  # cortó durante el primer token: no hay a quién enviar
                try:
                    await run_in_threadpool(live.close)
                finally:
                    if live.holds_backend:
                        gate.release()
                raise _ClientGone

            async def on_close() -> None:
                try:
                    await run_in_threadpool(live.close)
                finally:
                    if live.holds_backend:
                        gate.release()

            headers[TRACE_HEADER] = live.trace_id
            headers["Cache-Control"] = "no-cache"
            return _ClosingStreamingResponse(
                live.events, on_close, media_type="text/event-stream", headers=headers
            )
        except APIError as err:
            return _error_response(err, request_id)
        except DetectorUnavailable:
            return _error_response(_detector_error(), request_id)
        except _ClientGone:  # nadie lee esta respuesta; 499 es la convención de nginx
            return _error_response(
                APIError(499, "El cliente cerró la conexión.", "invalid_request_error", "client_closed"),
                request_id,
            )
        except VllmProxyError as exc:
            _log_upstream(exc, request_id)
            return _error_response(_upstream_error(), request_id)
        except Exception as exc:  # noqa: BLE001 - nunca filtrar stack traces al cliente
            logger.exception(
                "error interno en %s (error_type=%s, request_id=%s)",
                request.url.path,
                type(exc).__name__,
                request_id,
                extra={"error_type": type(exc).__name__, "request_id": request_id},
            )
            return _error_response(
                APIError(500, "Error interno del servidor.", "server_error", "internal_error"), request_id
            )

    @router.post("/v1/chat/completions")
    async def chat_completions(request: Request, ctx: RequestContext = Depends(get_context)):  # noqa: B008
        return await handle(request, ctx, "chat")

    @router.post("/v1/completions")
    async def completions(request: Request, ctx: RequestContext = Depends(get_context)):  # noqa: B008
        return await handle(request, ctx, "text")

    @router.get("/v1/models")
    async def list_models(request: Request, ctx: RequestContext = Depends(get_context)):  # noqa: B008
        info = ctx.backend.model_info()
        return JSONResponse(
            {
                "object": "list",
                "data": [{"id": info.model_id, "object": "model", "created": _STARTED, "owned_by": "vigia"}],
            },
            headers={REQUEST_ID_HEADER: _request_id(request)},
        )

    @router.get("/v1/vigia/traces/{trace_id}")
    async def get_trace(trace_id: str, request: Request, ctx: RequestContext = Depends(get_context)):  # noqa: B008
        request_id = _request_id(request)
        found = None
        if _ID_RE.match(trace_id) and ctx.trace_lookup is not None:
            found = await run_in_threadpool(ctx.trace_lookup, trace_id)
        if found is None:
            return _error_response(
                APIError(404, "La traza no existe.", "invalid_request_error", "trace_not_found", "trace_id"),
                request_id,
            )
        return JSONResponse(found, headers={REQUEST_ID_HEADER: request_id})

    return router


# --- Espera del cupo y vigilancia del cliente ---------------------------------------------------


def _is_closing(closing: Callable[[], bool] | None) -> bool:
    return closing is not None and bool(closing())


async def _take_slot(gate: GenerationGate, request: Request, closing: Callable[[], bool] | None) -> None:
    """Toma el cupo del backend solo para un request que todavía lo necesita.

    Mientras espera en la cola, cada `_POLL_S` mira si el cliente cortó (`_ClientGone`) o si empezó el
    cierre del proceso (503 `shutting_down`). Lo vuelve a mirar al obtener el cupo: un request abandonado
    o uno que llega al cupo durante el cierre lo libera sin generar.
    """
    task = asyncio.ensure_future(gate.acquire())
    try:
        while not task.done():
            await asyncio.wait({task}, timeout=_POLL_S)
            if task.done():
                break
            if _is_closing(closing):
                raise _shutting_down_error()
            if await request.is_disconnected():
                raise _ClientGone
        try:
            task.result()
        except GenerationBusy:
            raise _busy_error(gate) from None
        if _is_closing(closing):
            raise _shutting_down_error()
        if await request.is_disconnected():
            raise _ClientGone
    except BaseException:
        if not task.done():
            task.cancel()  # acquire() saca la espera de la cola o devuelve el cupo si llegó justo
            await asyncio.wait({task})
        elif not task.cancelled() and task.exception() is None:
            gate.release()
        raise


async def _watch_disconnect(request: Request, cancel: threading.Event) -> None:
    while not cancel.is_set():
        if await request.is_disconnected():
            cancel.set()
            return
        await asyncio.sleep(_POLL_S)


async def _run_watched(request: Request, fn: Callable[..., _R], *args: Any) -> _R:
    """Corre `fn(*args, cancel=...)` en el threadpool; si el cliente corta, `cancel` se marca y la
    generación se interrumpe entre tokens (como el abort de vLLM)."""
    cancel = threading.Event()
    watcher = asyncio.ensure_future(_watch_disconnect(request, cancel))
    try:
        return await run_in_threadpool(fn, *args, cancel=cancel)
    finally:
        watcher.cancel()
        await asyncio.wait({watcher})


# --- Ejecución (corre en el threadpool) ---------------------------------------------------------


def _run_full(
    ctx: RequestContext,
    messages: list[dict[str, str]],
    params: _Params,
    alpha: float,
    *,
    cancel: threading.Event | None = None,
) -> _Outcome:
    engine = _Engine(ctx, messages, params, alpha, cancel)
    out = engine.run()
    engine.store(out)
    return out


@dataclass
class _LiveStream:
    """Stream SSE en curso. `close()` libera el backend aunque el cliente haya cortado."""

    events: Iterator[str]
    trace_id: str
    backend_stream: Any = None  # None: no retiene el backend (stream_policy=buffer)
    backend_iter: Any = None

    @property
    def holds_backend(self) -> bool:
        return self.backend_stream is not None

    def close(self) -> None:
        """Cierra los eventos (guarda la traza parcial) y después el stream del backend (lock, hooks)."""
        try:
            close_events = getattr(self.events, "close", None)
            if callable(close_events):
                close_events()
        finally:
            _close_backend(self.backend_stream, self.backend_iter)


def _close_backend(stream: Any, it: Any = None) -> None:
    """Cierra el stream del backend y su iterador si es otro objeto. Cerrar dos veces no hace nada."""
    for obj in (stream,) if it is None or it is stream else (it, stream):
        close = getattr(obj, "close", None)
        if not callable(close):
            continue
        try:
            close()
        except Exception:  # noqa: BLE001 - cerrar no debe tapar el error original ni la respuesta
            logger.exception("no se pudo cerrar el stream del backend")


def _start_stream(
    ctx: RequestContext,
    messages: list[dict[str, str]],
    params: _Params,
    alpha: float,
    shape: _Shape,
    include_usage: bool,
    request_id: str,
    *,
    cancel: threading.Event | None = None,
) -> _LiveStream:
    """Arranca el stream. Las fallas previas al primer byte (detector, backend) salen como HTTP.

    `cancel` solo corta la generación completa de `stream_policy=buffer`; en vivo, el corte del
    cliente lo detecta el envío de cada chunk.
    """
    engine = _Engine(ctx, messages, params, alpha, cancel)
    if ctx.policy.stream_policy == "buffer":
        out = engine.run()
        engine.store(out)
        return _LiveStream(_buffered_events(out, shape, include_usage), engine.trace_id)
    stream = _open_stream(ctx.backend, messages, params)
    it = iter(stream)
    try:
        first = next(it, None)
        first_ps = engine.scoring.score([first]) if first is not None else []
    except BaseException:
        _close_backend(stream, it)
        raise
    events = _live_events(engine, stream, it, first, first_ps, shape, include_usage, request_id)
    return _LiveStream(events, engine.trace_id, stream, it)


def _buffered_events(out: _Outcome, shape: _Shape, include_usage: bool) -> Iterator[str]:
    """stream_policy=buffer: todo se generó y decidió antes de emitir; abstain no filtra el original."""
    result = out.client_result()
    yield _sse(shape.chunk("", {"trace_id": out.trace_id, "tokens": []}, role=True))
    if out.result.action == "abstain":
        yield _sse(shape.chunk(out.content, {"tokens": []}))
    else:
        by_index = {t.index: t for t in out.tokens}
        for st in result.tokens:
            tok = by_index[st.index]
            yield _sse(
                shape.chunk(st.text, {"tokens": [vars(st).copy()]}, logprobs=shape.token_logprobs(tok))
            )
    yield _sse(shape.chunk(None, _final_vigia(result), finish_reason=out.finish_reason))
    if include_usage:
        yield _sse(shape.usage_chunk(out))
    yield _sse("[DONE]")


def _live_events(
    engine: _Engine,
    stream: Any,
    it: Iterator[TokenEvent],
    first: TokenEvent | None,
    first_ps: list[float | None],
    shape: _Shape,
    include_usage: bool,
    request_id: str,
) -> Iterator[str]:
    """stream_policy=annotate_only: cada token sale apenas se puntúa; la acción llega en el chunk final.

    En vivo no se puede re-consultar: `rag_retry` pasa a `policy.rag_fallback_action` con
    `rag_status="unavailable"`, igual que el playground. El texto ya viajó; el cliente que necesite
    que abstain lo oculte debe usar `stream_policy=buffer`.
    """
    tokens: list[TokenEvent] = []
    ps: list[float | None] = []
    text_len = 0
    gen_s = overhead_s = 0.0
    finished = False
    cut_status = "cancelled"  # estado de la traza parcial: cliente desconectado o falla a mitad del stream
    out: _Outcome | None = None
    try:
        yield _sse(shape.chunk("", {"trace_id": engine.trace_id, "tokens": []}, role=True))
        nxt, p = first, (first_ps[0] if first_ps else None)
        while nxt is not None:
            tokens.append(nxt)
            ps.append(p)
            if _visible(nxt):
                text_len += len(nxt.text)
                st = _scored(nxt, p, engine.scoring.tau, text_len)
                chunk = shape.chunk(
                    nxt.text, {"tokens": [vars(st).copy()]}, logprobs=shape.token_logprobs(nxt)
                )
                yield _sse(chunk)
            g0 = time.perf_counter()
            nxt = next(it, None)
            o0 = time.perf_counter()
            gen_s += o0 - g0
            p = engine.scoring.score([nxt])[0] if nxt is not None else None
            overhead_s += time.perf_counter() - o0
        o0 = time.perf_counter()
        text, result = engine.finalize(tokens, ps)
        if result.action == "rag_retry":
            result.action = rag_fallback(engine.ctx.policy)
            result.rag_status = "unavailable"
        overhead_s += time.perf_counter() - o0
        out = engine.outcome(tokens, text, result, engine.messages, stream)
        out.rag_status = result.rag_status
        yield _sse(shape.chunk(None, _final_vigia(result), finish_reason=out.finish_reason))
        if include_usage:
            yield _sse(shape.usage_chunk(out))
        yield _sse("[DONE]")
        finished = True
    except DetectorUnavailable:
        cut_status = "detector_error"
        yield _sse(_error_body(_detector_error(), request_id))
        yield _sse("[DONE]")
    except VllmProxyError as exc:
        _log_upstream(exc, request_id)
        yield _sse(_error_body(_upstream_error(), request_id))
        yield _sse("[DONE]")
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "error interno durante el stream (error_type=%s, request_id=%s)",
            type(exc).__name__,
            request_id,
            extra={"error_type": type(exc).__name__, "request_id": request_id},
        )
        err = APIError(500, "Error interno del servidor.", "server_error", "internal_error")
        yield _sse(_error_body(err, request_id))
        yield _sse("[DONE]")
    finally:
        _close_backend(stream, it)
        if out is None:  # stream cortado: error o cliente desconectado; se guarda lo emitido
            text = "".join(t.text for t in tokens if _visible(t))
            partial = VigiaResult(
                trace_id=engine.trace_id,
                tokens=[
                    _scored(t, q, None, len(text)) for t, q in zip(tokens, ps, strict=True) if _visible(t)
                ],
                spans=[],
                response_p_error=None,
                action="pass",
                calibration_id=engine.ctx.scorer.id if engine.ctx.scorer is not None else None,
                calibrated=False,
                mode=engine.mode,
                degraded=cut_status == "detector_error",  # degraded es solo la falla del detector
                status=cut_status,
                alpha=engine.alpha,
                tau=None,
            )
            out = engine.outcome(tokens, text, partial, engine.messages, stream)
        out.gen_s, out.overhead_s = gen_s, overhead_s
        engine.store(out, None if finished else "error")
