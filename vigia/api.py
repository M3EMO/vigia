"""API compatible con OpenAI con el campo `vigia` por choice y por chunk (P-13, P-14, ADR A-09).

`create_api_router(get_context)` devuelve un `APIRouter` que la plataforma monta. `get_context` es una
dependencia async `(request) -> RequestContext` que resuelve auth, tenant, backend, detector y política.

- Sin `vigia` en el body (lo que el SDK manda con `extra_body`), la respuesta es idéntica a OpenAI.
  La traza igual se guarda vía `sink` y el header `X-Vigia-Trace-Id` la identifica.
- Con `vigia` (`{}` o `{"tiers": [...], "alpha": 0.1}`), cada choice trae `vigia = VigiaResult`.
- La política del tenant se aplica siempre: `abstain` reemplaza el contenido por `abstain_message`
  y el texto original queda solo en la traza.
- Los errores salen como `{"error": {message, type, code, param, request_id}}`, sin stack trace.
"""

from __future__ import annotations

import json
import logging
import math
import re
import secrets
import time
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any, Literal, TypeVar

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
from starlette.concurrency import run_in_threadpool

from vigia.contracts import (
    Action,
    Backend,
    Generation,
    ModelInfo,
    Policy,
    ScoredToken,
    Scorer,
    Span,
    Tier,
    TokenEvent,
    VigiaResult,
)
from vigia.policy import (
    RAG_FALLBACK_ACTION,
    Retriever,
    build_rag_query,
    decide,
    inject_context,
)
from vigia.spans import build_spans

logger = logging.getLogger("vigia.api")

TRACE_HEADER = "X-Vigia-Trace-Id"
REQUEST_ID_HEADER = "X-Request-Id"
CHEAP_TIERS = (Tier.F0, Tier.F1, Tier.U)
_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_STARTED = int(time.time())
_M = TypeVar("_M", bound="BaseModel")


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


# --- Errores ------------------------------------------------------------------------------------


class APIError(Exception):
    def __init__(self, status: int, message: str, type_: str, code: str, param: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.type = type_
        self.code = code
        self.param = param


class DetectorUnavailable(Exception):
    """El detector falló con `on_detector_error=fail_closed`."""


def _detector_error() -> APIError:
    return APIError(
        503,
        "El detector de confianza no está disponible y la política exige fail_closed.",
        "service_unavailable_error",
        "detector_unavailable",
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
        _error_body(err, request_id), status_code=err.status, headers={REQUEST_ID_HEADER: request_id}
    )


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


def _visible(t: TokenEvent) -> bool:
    return t.channel == "final" and not t.is_special


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
        self, ctx: RequestContext, messages: list[dict[str, str]], params: _Params, alpha: float
    ) -> None:
        self.ctx = ctx
        self.messages = messages
        self.params = params
        self.alpha = alpha
        self.trace_id = secrets.token_hex(16)
        self.started = time.perf_counter()
        self.info = ctx.backend.model_info()
        caps = ctx.backend.capabilities()
        self.mode: Literal["full", "reduced"] = "full" if Tier.F1 in caps.tiers else "reduced"
        self.scoring = _Scoring(ctx.scorer, ctx.policy, alpha, self.info)

    def prompt_tokens(self, messages: list[dict[str, str]]) -> int:
        count = getattr(self.ctx.backend, "count_prompt_tokens", None)
        return int(count(messages)) if callable(count) else 0

    def finalize(self, tokens: list[TokenEvent], ps: list[float | None]) -> tuple[str, VigiaResult]:
        policy = self.ctx.policy
        visible = [(t, p) for t, p in zip(tokens, ps, strict=True) if _visible(t)]
        text = "".join(t.text for t, _ in visible)
        tau = self.scoring.tau
        scored = [_scored(t, p, tau, len(text)) for t, p in visible]
        resp = self.scoring.response(ps)
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
        """Pipeline completo sin streaming, con rag_retry estilo FLARE."""
        ctx, policy = self.ctx, self.ctx.policy
        question = next((m["content"] for m in reversed(self.messages) if m["role"] == "user"), "")
        messages = self.messages
        attempts = 0
        rag_status: str | None = None
        gen_s = overhead_s = 0.0
        while True:
            g0 = time.perf_counter()
            tokens = list(ctx.backend.stream(messages, **self.params.kwargs()))
            o0 = time.perf_counter()
            gen_s += o0 - g0
            ps = self.scoring.score(tokens)
            text, result = self.finalize(tokens, ps)
            overhead_s += time.perf_counter() - o0
            if result.action != "rag_retry":
                break
            if ctx.retriever is None or attempts >= policy.rag_max_retries:
                rag_status = rag_status or "unavailable"
                result.action = RAG_FALLBACK_ACTION
                break
            try:
                docs = ctx.retriever.retrieve(build_rag_query(question, text, result.spans))
            except Exception as exc:  # noqa: BLE001 - el retriever es externo; falla -> acción de respaldo
                logger.warning("falla del retriever: %s", type(exc).__name__)
                rag_status = "error"
                result.action = RAG_FALLBACK_ACTION
                break
            attempts += 1
            rag_status = "retried"
            messages = inject_context(self.messages, docs)
        out = self.outcome(tokens, text, result, messages)
        out.gen_s, out.overhead_s, out.rag_status = gen_s, overhead_s, rag_status
        return out

    def outcome(
        self, tokens: list[TokenEvent], text: str, result: VigiaResult, messages: list[dict[str, str]]
    ) -> _Outcome:
        fr: Literal["stop", "length"] = "length" if len(tokens) >= self.params.max_tokens else "stop"
        return _Outcome(
            trace_id=self.trace_id,
            info=self.info,
            mode=self.mode,
            tokens=tokens,
            text=text,
            result=result,
            finish_reason=fr,
            prompt_tokens=self.prompt_tokens(messages),
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
    try:
        data = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise APIError(400, "El body no es JSON válido.", "invalid_request_error", "invalid_json") from exc
    if not isinstance(data, dict):
        raise APIError(400, "El body debe ser un objeto JSON.", "invalid_request_error", "invalid_json")
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
) -> APIRouter:
    """Router OpenAI-compatible. `max_tokens` fuera de 1..max_tokens_limit devuelve 422."""
    if not 1 <= default_max_tokens <= max_tokens_limit:
        raise ValueError("default_max_tokens debe estar entre 1 y max_tokens_limit")
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
        return _Params(mt, req.temperature, req.top_p, req.seed, req.stop_list(), tiers)

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
                assert isinstance(req, ChatCompletionRequest)  # noqa: S101 - estrechamiento de tipo
                messages = [m.as_backend() for m in req.messages]
                max_tokens = (
                    req.max_completion_tokens if req.max_completion_tokens is not None else req.max_tokens
                )
                want_logprobs = req.logprobs
            else:
                req = _validate(CompletionRequest, data)
                assert isinstance(req, CompletionRequest)  # noqa: S101
                # ponytail: el contrato Backend solo acepta mensajes; el prompt va como mensaje de usuario.
                messages = [{"role": "user", "content": req.prompt_text()}]
                max_tokens = req.max_tokens
                want_logprobs = False
            check_model(req, ctx)
            params = params_for(req, max_tokens, ctx)
            alpha = (
                req.vigia.alpha if req.vigia is not None and req.vigia.alpha is not None else ctx.policy.alpha
            )
            shape = _Shape(kind, ctx.backend.model_info().model_id, req.vigia is not None, want_logprobs)
            include_usage = bool(req.stream_options and req.stream_options.include_usage)
            headers = {REQUEST_ID_HEADER: request_id}
            if not req.stream:
                out = await run_in_threadpool(_run_full, ctx, messages, params, alpha)
                headers[TRACE_HEADER] = out.trace_id
                return JSONResponse(shape.full(out), headers=headers)
            events, trace_id = await run_in_threadpool(
                _start_stream, ctx, messages, params, alpha, shape, include_usage, request_id
            )
            headers[TRACE_HEADER] = trace_id
            headers["Cache-Control"] = "no-cache"
            return StreamingResponse(events, media_type="text/event-stream", headers=headers)
        except APIError as err:
            return _error_response(err, request_id)
        except DetectorUnavailable:
            return _error_response(_detector_error(), request_id)
        except Exception:  # noqa: BLE001 - nunca filtrar stack traces al cliente
            logger.exception("error interno en %s (request_id=%s)", request.url.path, request_id)
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


# --- Ejecución (corre en el threadpool) ---------------------------------------------------------


def _run_full(ctx: RequestContext, messages: list[dict[str, str]], params: _Params, alpha: float) -> _Outcome:
    engine = _Engine(ctx, messages, params, alpha)
    out = engine.run()
    engine.store(out)
    return out


def _start_stream(
    ctx: RequestContext,
    messages: list[dict[str, str]],
    params: _Params,
    alpha: float,
    shape: _Shape,
    include_usage: bool,
    request_id: str,
) -> tuple[Iterator[str], str]:
    """Arranca el stream. Las fallas previas al primer byte (detector, backend) salen como HTTP."""
    engine = _Engine(ctx, messages, params, alpha)
    if ctx.policy.stream_policy == "buffer":
        out = engine.run()
        engine.store(out)
        return _buffered_events(out, shape, include_usage), engine.trace_id
    it = iter(ctx.backend.stream(messages, **params.kwargs()))
    try:
        first = next(it, None)
        first_ps = engine.scoring.score([first]) if first is not None else []
    except BaseException:
        close = getattr(it, "close", None)
        if callable(close):
            close()
        raise
    return _live_events(engine, it, first, first_ps, shape, include_usage, request_id), engine.trace_id


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
    it: Iterator[TokenEvent],
    first: TokenEvent | None,
    first_ps: list[float | None],
    shape: _Shape,
    include_usage: bool,
    request_id: str,
) -> Iterator[str]:
    """stream_policy=annotate_only: cada token sale apenas se puntúa; la acción llega en el chunk final."""
    tokens: list[TokenEvent] = []
    ps: list[float | None] = []
    text_len = 0
    gen_s = overhead_s = 0.0
    finished = False
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
        overhead_s += time.perf_counter() - o0
        out = engine.outcome(tokens, text, result, engine.messages)
        yield _sse(shape.chunk(None, _final_vigia(result), finish_reason=out.finish_reason))
        if include_usage:
            yield _sse(shape.usage_chunk(out))
        yield _sse("[DONE]")
        finished = True
    except DetectorUnavailable:
        yield _sse(_error_body(_detector_error(), request_id))
        yield _sse("[DONE]")
    except Exception:  # noqa: BLE001
        logger.exception("error interno durante el stream (request_id=%s)", request_id)
        err = APIError(500, "Error interno del servidor.", "server_error", "internal_error")
        yield _sse(_error_body(err, request_id))
        yield _sse("[DONE]")
    finally:
        close = getattr(it, "close", None)
        if callable(close):
            close()
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
                degraded=True,
                alpha=engine.alpha,
                tau=None,
            )
            out = engine.outcome(tokens, text, partial, engine.messages)
        out.gen_s, out.overhead_s = gen_s, overhead_s
        engine.store(out, None if finished else "error")
