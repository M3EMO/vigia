"""Seguridad HTTP de la plataforma: headers, CSRF, rate limiting, límite de body, errores y logs.

Uso típico en la app:

    install_security(app, max_body_bytes=1_048_576)

Orden de los middlewares (de afuera hacia adentro): `SecurityHeadersMiddleware` →
`RequestContextMiddleware` (request_id, log de acceso, 500 genérico) → `BodySizeLimitMiddleware`.
Así hasta un 500 lleva headers de seguridad y `X-Request-ID`.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import re
import secrets
import sys
import threading
import time
import traceback
from collections import OrderedDict
from collections.abc import Awaitable, Callable, MutableMapping
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "TRACE"})
SESSION_COOKIE = "__Host-vigia_session"
DEFAULT_MAX_BODY_BYTES = 1_048_576

# --- logs JSON con allowlist ----------------------------------------------------------------------

LOG_FIELDS = frozenset(
    {
        "event",
        "request_id",
        "method",
        "path",
        "status",
        "duration_ms",
        "org_id",
        "user_id",
        "key_id",
        "action",
        "scope",
        "retry_after",
        "error_type",
        "frames",
    }
)
_CTRL_RE = re.compile(r"[\x00-\x1f\x7f-\x9f  ]")


def sanitize_log_value(value: Any, *, max_len: int = 500) -> str:
    """Texto de una sola línea: escapa CR, LF y demás controles; recorta a `max_len`."""
    text = str(value)

    def esc(m: re.Match[str]) -> str:
        ch = m.group(0)
        return {"\r": "\\r", "\n": "\\n", "\t": "\\t"}.get(ch, f"\\u{ord(ch):04x}")

    text = _CTRL_RE.sub(esc, text)
    return text if len(text) <= max_len else text[:max_len] + "…"


class JsonLogFormatter(logging.Formatter):
    """Una línea JSON por evento con solo los campos de `LOG_FIELDS`."""

    def format(self, record: logging.LogRecord) -> str:
        out: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": sanitize_log_value(record.getMessage()),
        }
        for key in sorted(LOG_FIELDS):
            if key in record.__dict__:
                val = record.__dict__[key]
                out[key] = (
                    val if isinstance(val, int | float | bool) or val is None else sanitize_log_value(val)
                )
        return json.dumps(out, ensure_ascii=False)


def configure_json_logging(level: int = logging.INFO, stream: Any = None) -> logging.Logger:
    """Configura el logger `vigia` con salida JSON. Idempotente."""
    logger = logging.getLogger("vigia")
    logger.setLevel(level)
    for h in logger.handlers:
        if getattr(h, "_vigia_json", False):
            if stream is not None:
                h.setStream(stream)  # type: ignore[attr-defined]
            return logger
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JsonLogFormatter())
    handler._vigia_json = True  # type: ignore[attr-defined]
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def log_event(logger: logging.Logger, event: str, *, level: int = logging.INFO, **fields: Any) -> None:
    """Loguea un evento; descarta en silencio todo campo fuera de `LOG_FIELDS`."""
    extra = {k: v for k, v in fields.items() if k in LOG_FIELDS and k != "event"}
    extra["event"] = event
    logger.log(level, event, extra=extra)


_log = logging.getLogger("vigia.http")


# --- headers de seguridad -------------------------------------------------------------------------


def build_csp(nonce: str) -> str:
    return (
        "default-src 'self'; "
        f"script-src 'nonce-{nonce}' 'strict-dynamic'; "
        "style-src 'self'; "
        "img-src 'self' data:; "
        "object-src 'none'; "
        "base-uri 'none'; "
        "frame-ancestors 'none'; "
        "form-action 'self'"
    )


STATIC_SECURITY_HEADERS: tuple[tuple[str, str], ...] = (
    ("strict-transport-security", "max-age=63072000; includeSubDomains"),
    ("x-content-type-options", "nosniff"),
    ("referrer-policy", "no-referrer"),
    ("permissions-policy", "geolocation=(), camera=(), microphone=(), payment=(), usb=()"),
    ("cross-origin-opener-policy", "same-origin"),
    ("cross-origin-resource-policy", "same-origin"),
    ("x-frame-options", "DENY"),
)


def _is_authenticated(scope: Scope) -> bool:
    state = scope.get("state") or {}
    if state.get("authenticated"):
        return True
    for name, value in scope.get("headers") or ():
        if name == b"authorization":
            return True
        if name == b"cookie" and SESSION_COOKIE.encode() in value:
            return True
    return False


class SecurityHeadersMiddleware:
    """Agrega CSP con nonce por respuesta y los headers de `STATIC_SECURITY_HEADERS`.

    El nonce queda en `request.state.csp_nonce` antes de llegar a la ruta. Una respuesta a un
    request autenticado (sesión, `Authorization` o `request.state.authenticated`) lleva
    `Cache-Control: no-store`.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        nonce = secrets.token_urlsafe(18)
        scope.setdefault("state", {})["csp_nonce"] = nonce

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                drop = {b"content-security-policy", b"server", b"x-powered-by"}
                drop.update(name.encode() for name, _ in STATIC_SECURITY_HEADERS)
                authenticated = _is_authenticated(scope)
                if authenticated:
                    drop.update({b"cache-control", b"pragma"})
                headers = [(k, v) for k, v in message.get("headers", []) if k.lower() not in drop]
                headers.append((b"content-security-policy", build_csp(nonce).encode()))
                headers.extend((k.encode(), v.encode()) for k, v in STATIC_SECURITY_HEADERS)
                if authenticated:
                    headers.append((b"cache-control", b"no-store"))
                    headers.append((b"pragma", b"no-cache"))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_wrapper)


# --- request id, log de acceso y 500 genérico -----------------------------------------------------

_RID_RE = re.compile(r"^[A-Za-z0-9-]{8,64}$")


def _json_body(status: int, payload: dict[str, Any]) -> tuple[Message, Message]:
    body = json.dumps(payload, ensure_ascii=False).encode()
    start = {
        "type": "http.response.start",
        "status": status,
        "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
    }
    return start, {"type": "http.response.body", "body": body}


class RequestContextMiddleware:
    """Asigna `request_id`, loguea el acceso sin query string y convierte excepciones en 500 genérico.

    El 500 nunca incluye stack trace. El log de la excepción guarda tipo y frames (archivo, línea,
    función) pero no el mensaje, que puede traer datos del cliente.
    """

    def __init__(self, app: ASGIApp, *, trust_incoming_id: bool = False) -> None:
        self.app = app
        self.trust_incoming_id = trust_incoming_id

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        rid = None
        if self.trust_incoming_id:
            for name, value in scope.get("headers") or ():
                if name == b"x-request-id":
                    candidate = value.decode("latin-1")
                    rid = candidate if _RID_RE.match(candidate) else None
        rid = rid or secrets.token_hex(8)
        scope.setdefault("state", {})["request_id"] = rid
        started = False
        status = 500
        t0 = time.perf_counter()

        async def send_wrapper(message: Message) -> None:
            nonlocal started, status
            if message["type"] == "http.response.start":
                started = True
                status = message["status"]
                kept = [(k, v) for k, v in message.get("headers", []) if k.lower() != b"x-request-id"]
                message["headers"] = [*kept, (b"x-request-id", rid.encode())]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as exc:  # noqa: BLE001 - frontera: ninguna excepción sale al cliente
            frames = ";".join(
                f"{f.filename.rsplit('/', 1)[-1].rsplit(chr(92), 1)[-1]}:{f.lineno}:{f.name}"
                for f in traceback.extract_tb(exc.__traceback__)[-8:]
            )
            log_event(
                _log,
                "unhandled_error",
                level=logging.ERROR,
                request_id=rid,
                error_type=type(exc).__name__,
                frames=frames,
                path=scope.get("path", ""),
            )
            if not started:
                start, body = _json_body(
                    500, {"error": "internal_error", "message": "Error interno.", "request_id": rid}
                )
                await send_wrapper(start)
                await send(body)
        finally:
            log_event(
                _log,
                "http_request",
                request_id=rid,
                method=scope.get("method", ""),
                path=scope.get("path", ""),
                status=status,
                duration_ms=round((time.perf_counter() - t0) * 1000, 2),
            )


# --- límite de body -------------------------------------------------------------------------------


class BodyTooLarge(HTTPException):
    def __init__(self) -> None:
        super().__init__(413, "El cuerpo del request supera el tamaño máximo.")


class BodySizeLimitMiddleware:
    """Responde 413 si el body supera `max_bytes`, por Content-Length o contando lo recibido."""

    def __init__(self, app: ASGIApp, *, max_bytes: int = DEFAULT_MAX_BODY_BYTES) -> None:
        self.app = app
        self.max_bytes = int(max_bytes)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        for name, value in scope.get("headers") or ():
            if name == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    declared = -1
                if declared < 0 or declared > self.max_bytes:
                    await self._reject(scope, send, 413 if declared > self.max_bytes else 400)
                    return
        received = 0
        started = False

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise BodyTooLarge()
            return message

        async def send_wrapper(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, send_wrapper)
        except BodyTooLarge:
            if not started:
                await self._reject(scope, send, 413)

    @staticmethod
    async def _reject(scope: Scope, send: Send, status: int) -> None:
        rid = (scope.get("state") or {}).get("request_id")
        code = "payload_too_large" if status == 413 else "bad_request"
        msg = (
            "El cuerpo del request supera el tamaño máximo." if status == 413 else "Content-Length inválido."
        )
        start, body = _json_body(status, {"error": code, "message": msg, "request_id": rid})
        await send(start)
        await send(body)


# --- rate limiting --------------------------------------------------------------------------------


class RateLimiter:
    """Token bucket en memoria por clave (API key, tenant, IP de login).

    `ponytail:` el estado vive en este proceso. Con varios workers o nodos, cada uno cuenta aparte
    y el límite efectivo se multiplica por la cantidad de procesos; ahí corresponde Redis o la base.
    La memoria se acota con `max_keys` (LRU): una clave desalojada vuelve con el bucket lleno.
    """

    def __init__(
        self,
        rate: float,
        burst: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        max_keys: int = 100_000,
    ) -> None:
        if rate <= 0 or burst < 1:
            raise ValueError("rate debe ser > 0 y burst >= 1")
        self.rate = float(rate)
        self.burst = float(burst)
        self.clock = clock
        self.max_keys = int(max_keys)
        self._buckets: OrderedDict[str, tuple[float, float]] = OrderedDict()
        self._lock = threading.Lock()

    def _level(self, key: str, now: float) -> float:
        tokens, ts = self._buckets.get(key, (self.burst, now))
        return min(self.burst, tokens + (now - ts) * self.rate)

    def hit(self, key: str, cost: float = 1.0) -> float | None:
        """Consume `cost`. Devuelve None si pasó, o los segundos a esperar si no alcanzó."""
        with self._lock:
            now = self.clock()
            tokens = self._level(key, now)
            if tokens >= cost:
                self._buckets[key] = (tokens - cost, now)
                self._buckets.move_to_end(key)
                while len(self._buckets) > self.max_keys:
                    self._buckets.popitem(last=False)
                return None
            self._buckets[key] = (tokens, now)
            self._buckets.move_to_end(key)
            return (cost - tokens) / self.rate

    def blocked(self, key: str, cost: float = 1.0) -> float | None:
        """Como `hit` pero sin consumir: segundos a esperar, o None si hay saldo."""
        with self._lock:
            tokens = self._level(key, self.clock())
            return None if tokens >= cost else (cost - tokens) / self.rate

    def reset(self, key: str) -> None:
        with self._lock:
            self._buckets.pop(key, None)


class RateLimitExceeded(HTTPException):
    def __init__(self, retry_after: float, scope: str = "") -> None:
        super().__init__(
            429,
            "Demasiadas solicitudes. Reintentá más tarde.",
            headers={"Retry-After": str(max(1, math.ceil(retry_after)))},
        )
        self.retry_after = retry_after
        self.scope = scope


def enforce_rate_limit(limiter: RateLimiter | None, key: str, *, scope: str = "", cost: float = 1.0) -> None:
    """Consume del bucket `key`; lanza 429 con `Retry-After` si no alcanza."""
    if limiter is None:
        return
    wait = limiter.hit(key, cost)
    if wait is not None:
        log_event(_log, "rate_limited", level=logging.WARNING, scope=scope, retry_after=round(wait, 2))
        raise RateLimitExceeded(wait, scope)


# --- CSRF -----------------------------------------------------------------------------------------

CSRF_FORM_FIELD = "csrf_token"
CSRF_HEADER = "x-csrf-token"


def csrf_token_for(csrf_secret: str, session_id_hash: str) -> str:
    """Token CSRF firmado y atado a la sesión: HMAC-SHA256(csrf_secret, id de sesión hasheado)."""
    return hmac.new(
        csrf_secret.encode(), b"vigia-csrf|" + session_id_hash.encode(), hashlib.sha256
    ).hexdigest()


def check_fetch_site(request: Request) -> None:
    """403 si el navegador declara un request cross-site en un método no seguro."""
    if (
        request.method not in SAFE_METHODS
        and request.headers.get("sec-fetch-site", "").lower() == "cross-site"
    ):
        raise HTTPException(403, "Solicitud de otro sitio rechazada.")


async def enforce_csrf(request: Request, csrf_secret: str, session_id_hash: str) -> None:
    """Valida `Sec-Fetch-Site` y el token CSRF (header `X-CSRF-Token` o campo `csrf_token`)."""
    if request.method in SAFE_METHODS:
        return
    check_fetch_site(request)
    token = request.headers.get(CSRF_HEADER)
    ctype = request.headers.get("content-type", "")
    if not token and (
        ctype.startswith("application/x-www-form-urlencoded") or ctype.startswith("multipart/form-data")
    ):
        form = await request.form()
        value = form.get(CSRF_FORM_FIELD)
        token = value if isinstance(value, str) else None
    expected = csrf_token_for(csrf_secret, session_id_hash)
    if not token or not hmac.compare_digest(token.encode(), expected.encode()):
        log_event(_log, "csrf_rejected", level=logging.WARNING, path=request.url.path)
        raise HTTPException(403, "Token CSRF inválido o ausente.")


# --- errores genéricos ----------------------------------------------------------------------------

_ERROR_CODES = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    413: "payload_too_large",
    422: "validation_error",
    429: "rate_limited",
    503: "unavailable",
}


def request_id_of(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


async def _http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    detail = exc.detail if isinstance(exc.detail, str) else _ERROR_CODES.get(exc.status_code, "error")
    return JSONResponse(
        {
            "error": _ERROR_CODES.get(exc.status_code, "error"),
            "message": detail,
            "request_id": request_id_of(request),
        },
        status_code=exc.status_code,
        headers=getattr(exc, "headers", None),
    )


async def _validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    # Sin `input` ni `ctx`: pydantic los incluye y pueden traer el texto del cliente.
    fields = [
        {"loc": list(e.get("loc", ())), "msg": str(e.get("msg", "")), "type": str(e.get("type", ""))}
        for e in exc.errors()
    ]
    return JSONResponse(
        {
            "error": "validation_error",
            "message": "Datos inválidos.",
            "request_id": request_id_of(request),
            "fields": fields,
        },
        status_code=422,
    )


def install_error_handlers(app: FastAPI) -> None:
    """Errores HTTP y de validación en JSON con `request_id`, sin eco del input."""
    app.add_exception_handler(StarletteHTTPException, _http_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, _validation_handler)  # type: ignore[arg-type]


def install_security(
    app: FastAPI, *, max_body_bytes: int = DEFAULT_MAX_BODY_BYTES, trust_incoming_request_id: bool = False
) -> None:
    """Instala middlewares (en el orden correcto) y manejadores de error."""
    install_error_handlers(app)
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=max_body_bytes)
    app.add_middleware(RequestContextMiddleware, trust_incoming_id=trust_incoming_request_id)
    app.add_middleware(SecurityHeadersMiddleware)
