"""Servidor single-tenant del núcleo abierto (sin `vigia_ee`): API OpenAI-compatible + trazas.

`create_app(backend, detector=..., policy=..., api_key=...)` monta `vigia.api.create_api_router`
sobre una app FastAPI mínima:

- Con `api_key`, cada request a `/v1/*` exige `Authorization: Bearer <key>` (comparación en tiempo
  constante). Sin `api_key`, la app solo atiende clientes de loopback y lo advierte en el log. Un
  navegador del mismo equipo también es loopback: sin key se rechaza todo request de navegador desde
  otro sitio (`Origin` o `Sec-Fetch-Site`), todo `Host` que no sea loopback (DNS rebinding) y todo
  POST sin `Content-Type: application/json` (un POST `text/plain` no dispara el preflight CORS).
- Las trazas viven en memoria con tope LRU (1000) para `GET /v1/vigia/traces/{id}`.
- Una política con `rag_url` usa un `HttpRetriever` contra esa URL para `rag_retry` (AT-A07). La URL
  se valida contra SSRF al crear la app; `rag_allowed_hosts` habilita hosts privados o http.
- `/livez` y `/readyz` no exigen credencial y no devuelven datos de requests.
- Sin `/docs`, `/redoc` ni `/openapi.json`. Body máximo de 1 MiB (413).
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import logging
import secrets
import threading
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Collection
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from vigia.api import REQUEST_ID_HEADER, RequestContext, TraceRecord, create_api_router
from vigia.contracts import Backend, Policy, Scorer
from vigia.policy import HttpRetriever, Retriever, validate_policy

logger = logging.getLogger("vigia.serve")

MAX_TRACES = 1000
MAX_BODY_BYTES = 1_048_576
MIN_API_KEY_CHARS = 16
LOOPBACK_NAMES = frozenset({"localhost"})
_QUERY_CREDENTIALS = ("api_key", "apikey", "key", "token", "access_token")
_FOREIGN_FETCH_SITES = frozenset({"cross-site", "same-site"})
_SECURITY_HEADERS = (
    (b"x-content-type-options", b"nosniff"),
    (b"referrer-policy", b"no-referrer"),
    (b"x-frame-options", b"DENY"),
    (b"content-security-policy", b"default-src 'none'; frame-ancestors 'none'"),
)

ASGIApp = Callable[
    [dict[str, Any], Callable[[], Awaitable[dict[str, Any]]], Callable[..., Awaitable[None]]], Any
]


def is_loopback(host: str | None) -> bool:
    """True para 127.0.0.0/8, ::1 y `localhost`."""
    if not host:
        return False
    if host in LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def _host_name(authority: str) -> str:
    """Nombre de host de un header `Host` u `Origin` sin esquema: `[::1]:8000` -> `::1`."""
    if authority.startswith("["):
        return authority[1 : authority.find("]")] if "]" in authority else ""
    return authority.rsplit(":", 1)[0] if authority.count(":") == 1 else authority


def browser_origin_problem(headers: Any) -> str | None:
    """Motivo para rechazar un request sin API key que no viene de un cliente local legítimo.

    Un sitio que el usuario visita corre en un navegador de loopback: puede mandar un POST simple
    (sin preflight) o, con DNS rebinding, leer respuestas. Devuelve None si el request es aceptable.
    """
    host = headers.get("host", "")
    if not is_loopback(_host_name(host.strip().lower())):
        return "El header Host debe ser localhost, 127.0.0.1 o [::1]."
    if headers.get("sec-fetch-site", "").strip().lower() in _FOREIGN_FETCH_SITES:
        return "Sin API key no se aceptan requests de navegador desde otro sitio."
    origin = headers.get("origin")
    if origin is not None:
        scheme, sep, rest = origin.strip().lower().partition("://")
        if not sep or scheme not in ("http", "https") or not is_loopback(_host_name(rest)):
            return "Sin API key no se aceptan requests de navegador desde otro origen."
    return None


class TraceStore:
    """Trazas recientes en memoria, LRU con tope y seguro entre hilos."""

    def __init__(self, max_items: int = MAX_TRACES) -> None:
        if max_items < 1:
            raise ValueError("max_items debe ser >= 1")
        self.max_items = max_items
        self._items: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def put(self, record: TraceRecord) -> None:
        view = trace_view(record)
        with self._lock:
            self._items[record.trace_id] = view
            self._items.move_to_end(record.trace_id)
            while len(self._items) > self.max_items:
                self._items.popitem(last=False)

    def get(self, trace_id: str) -> dict[str, Any] | None:
        with self._lock:
            view = self._items.get(trace_id)
            if view is not None:
                self._items.move_to_end(trace_id)
            return view


def trace_view(record: TraceRecord) -> dict[str, Any]:
    """Lo que devuelve `GET /v1/vigia/traces/{id}`. Con `abstain` no viaja el texto original."""
    abstained = record.result.action == "abstain"
    vigia = record.result.to_dict()
    if abstained:
        vigia["tokens"], vigia["spans"] = [], []
    return {
        "id": record.trace_id,
        "object": "vigia.trace",
        "created_at": record.created_at,
        "model": record.model.model_id,
        "source": record.source,
        "finish_reason": record.generation.finish_reason,
        "response": None if abstained else record.generation.text,
        "vigia": vigia,
        "latency_ms": record.latency_ms,
        "overhead_ms": record.overhead_ms,
        "rag_status": record.rag_status,
    }


def _error_payload(status: int, message: str, type_: str, code: str, request_id: str) -> bytes:
    body = {
        "error": {"message": message, "type": type_, "code": code, "param": None, "request_id": request_id}
    }
    return json.dumps(body, ensure_ascii=False).encode("utf-8")


class _BodyLimit:
    """Lee el body entero hasta `max_bytes` y lo reenvía; si se pasa, responde 413 sin llamar a la app."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        body = bytearray()
        more = True
        while more:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body += message.get("body", b"")
            more = message.get("more_body", False)
            if len(body) > self.max_bytes:
                payload = _error_payload(
                    413,
                    f"El body supera {self.max_bytes} bytes.",
                    "invalid_request_error",
                    "body_too_large",
                    _new_request_id(),
                )
                await send(
                    {
                        "type": "http.response.start",
                        "status": 413,
                        "headers": [(b"content-type", b"application/json"), *_SECURITY_HEADERS],
                    }
                )
                await send({"type": "http.response.body", "body": payload})
                return
        replayed = False

        async def replay() -> dict[str, Any]:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


class _SecurityHeaders:
    """Headers de seguridad en toda respuesta; `no-store` salvo que la ruta fije su Cache-Control."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def wrapped(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                present = {k.lower() for k, _ in headers}
                headers += [(k, v) for k, v in _SECURITY_HEADERS if k not in present]
                if b"cache-control" not in present:
                    headers.append((b"cache-control", b"no-store"))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, wrapped)


class _AuthError(Exception):
    def __init__(self, status: int, message: str, code: str, headers: dict[str, str] | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.code = code
        self.headers = headers or {}


def _digest(value: str) -> bytes:
    return hashlib.sha256(value.encode("utf-8")).digest()


def _new_request_id() -> str:
    return "req_" + secrets.token_hex(8)


def create_app(
    backend: Backend,
    *,
    detector: Scorer | None = None,
    policy: Policy | None = None,
    api_key: str | None = None,
    max_traces: int = MAX_TRACES,
    max_body_bytes: int = MAX_BODY_BYTES,
    max_tokens_limit: int = 4096,
    default_max_tokens: int = 512,
    rag_allowed_hosts: Collection[str] = (),
    retriever: Retriever | None = None,
) -> FastAPI:
    """App FastAPI single-tenant. Falla al crearla si el detector es de otro modelo o si la política
    es inválida (por ejemplo, `rag_url` fuera de `rag_allowed_hosts` o hacia una red privada).

    Sin `retriever`, una política con `rag_url` usa `HttpRetriever(rag_url, allowed_hosts=...)`.
    """
    if api_key is not None:
        if not isinstance(api_key, str) or len(api_key) < MIN_API_KEY_CHARS:
            raise ValueError(f"la API key debe tener al menos {MIN_API_KEY_CHARS} caracteres")
        if any(c.isspace() for c in api_key):
            raise ValueError("la API key no puede tener espacios ni saltos de línea")
    else:
        logger.warning(
            "servidor sin API key: solo se aceptan clientes de loopback (127.0.0.1); "
            "usá una API key para exponerlo en la red"
        )
    info = backend.model_info()
    if detector is not None:
        detector.check_model(info)
    policy = validate_policy(policy if policy is not None else Policy(), rag_allowed_hosts=rag_allowed_hosts)
    if retriever is None and policy.rag_url is not None:
        retriever = HttpRetriever(policy.rag_url, allowed_hosts=rag_allowed_hosts)
    expected = _digest(api_key) if api_key is not None else None
    traces = TraceStore(max_traces)
    ctx = RequestContext(
        backend=backend,
        scorer=detector,
        policy=policy,
        sink=traces.put,
        trace_lookup=traces.get,
        retriever=retriever,
    )

    async def get_context(request: Request) -> RequestContext:
        if any(k in request.query_params for k in _QUERY_CREDENTIALS):
            raise _AuthError(400, "Las credenciales no se aceptan en la URL.", "credentials_in_query")
        if expected is None:
            host = request.client.host if request.client is not None else None
            if not is_loopback(host):
                raise _AuthError(403, "Sin API key solo se aceptan clientes locales.", "forbidden")
            problem = browser_origin_problem(request.headers)
            if problem is not None:
                raise _AuthError(403, problem, "forbidden")
            if request.method == "POST":
                ctype = request.headers.get("content-type", "").split(";")[0].strip().lower()
                if ctype != "application/json":
                    raise _AuthError(
                        415,
                        "Sin API key el body debe ser Content-Type: application/json.",
                        "unsupported_media_type",
                    )
            return ctx
        scheme, _, token = request.headers.get("authorization", "").partition(" ")
        valid = scheme.lower() == "bearer" and hmac.compare_digest(_digest(token.strip()), expected)
        if not valid:
            raise _AuthError(
                401,
                "Falta la API key o es inválida.",
                "invalid_api_key",
                {"WWW-Authenticate": 'Bearer realm="vigia"'},
            )
        return ctx

    app = FastAPI(title="Vigía", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.vigia_traces = traces

    @app.exception_handler(_AuthError)
    async def _auth_error(request: Request, exc: _AuthError) -> JSONResponse:
        rid = _new_request_id()
        type_ = "authentication_error" if exc.status == 401 else "invalid_request_error"
        return JSONResponse(
            json.loads(_error_payload(exc.status, exc.message, type_, exc.code, rid)),
            status_code=exc.status,
            headers={**exc.headers, REQUEST_ID_HEADER: rid},
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        rid = _new_request_id()
        message = "La ruta no existe." if exc.status_code == 404 else "Request inválido."
        code = "not_found" if exc.status_code == 404 else "http_error"
        return JSONResponse(
            json.loads(_error_payload(exc.status_code, message, "invalid_request_error", code, rid)),
            status_code=exc.status_code,
            headers={REQUEST_ID_HEADER: rid},
        )

    @app.get("/livez", include_in_schema=False)
    async def livez() -> JSONResponse:
        return JSONResponse({"status": "ok"})

    @app.get("/readyz", include_in_schema=False)
    async def readyz() -> JSONResponse:
        try:
            current = backend.model_info()
        except Exception:  # noqa: BLE001 - readiness nunca filtra el error, solo informa el estado
            logger.exception("el backend no responde")
            return JSONResponse({"status": "not_ready"}, status_code=503)
        return JSONResponse(
            {"status": "ready", "model": current.model_id, "calibrated": detector is not None}
        )

    app.include_router(
        create_api_router(
            get_context, max_tokens_limit=max_tokens_limit, default_max_tokens=default_max_tokens
        )
    )
    app.add_middleware(_BodyLimit, max_bytes=max_body_bytes)
    app.add_middleware(_SecurityHeaders)
    return app


__all__ = ["MAX_TRACES", "TraceStore", "browser_origin_problem", "create_app", "is_loopback", "trace_view"]
