"""Aplicación de la plataforma: composición de seguridad, auth, API, panel, health y CLI.

    app = create_app()                      # lee Settings.from_env()
    uvicorn vigia_ee.app:create_app --factory

CLI (`python -m vigia_ee.app <comando>`):

- `init --org NOMBRE --email EMAIL`: crea la organización y su owner. La contraseña se pide por
  `getpass` o se lee de `VIGIA_OWNER_PASSWORD_FILE`; nunca va como argumento.
- `create-key --org-id ID --name NOMBRE [--scopes ingest,read]`: imprime el token una sola vez.
- `serve [--host 127.0.0.1] [--port 8000] [--proxy-headers]`: uvicorn sin header `Server`. Ctrl+C o
  SIGTERM marcan el runtime como cerrando (las generaciones nuevas responden 503) y el cierre espera
  a lo sumo `VIGIA_SHUTDOWN_TIMEOUT` segundos (30 por defecto) antes de cancelar lo pendiente. Los
  logs de uvicorn y asyncio salen en el mismo JSON que los de Vigía, sin stack traces.

Rutas:

- API OpenAI-compatible (`vigia.api`): `/v1/chat/completions`, `/v1/completions` y `/v1/models` piden
  una API key con scope `ingest`; `/v1/vigia/traces/{id}` pide scope `read`.
- Health públicas sin datos de tenants: `/livez`, `/startupz`, `/readyz`. `/metrics` pide
  `Authorization: Bearer <VIGIA_METRICS_TOKEN>`; sin token configurado responde 404.
- Panel (`vigia_ee.panel`) montado en `/`; el panel sirve sus estáticos en `/static/{name}` con una
  ruta pública declarada (no hace falta un `StaticFiles` aparte).

Límite de body: `max_body_bytes` para todo, salvo las rutas de `UPLOAD_PATHS` (subida de datasets de
calibración), que admiten `max_upload_bytes` más el margen del multipart.
"""

from __future__ import annotations

import argparse
import getpass
import hmac
import logging
import os
import sys
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from vigia.api import RequestContext, create_api_router
from vigia_ee import audit, retention
from vigia_ee.auth import (
    AuthState,
    PasswordPolicyError,
    PepperConfigError,
    PepperRing,
    Principal,
    check_password_policy,
    create_api_key,
    install_auth,
    public_route,
    register_user,
    require_api_key,
)
from vigia_ee.db import STORE_TEXT_MODES, create_org, utcnow
from vigia_ee.metrics import CONTENT_TYPE, Metrics, MetricsMiddleware
from vigia_ee.runtime import Runtime, Settings, SettingsError, open_database
from vigia_ee.security import BodySizeLimitMiddleware, configure_json_logging, install_security

_log = logging.getLogger("vigia.app")

TRACE_ROUTE = "/v1/vigia/traces/{trace_id}"
UPLOAD_PATHS = ("/calibration/datasets",)
MULTIPART_MARGIN = 65_536  # boundaries, headers y el campo csrf_token alrededor del archivo
PURGE_EVERY_HOURS = 24
EXIT_OK, EXIT_ERROR, EXIT_USAGE = 0, 1, 2
FOREIGN_LOGGERS = ("uvicorn", "asyncio")  # loggers de terceros que pasan al formato JSON


# --- Middlewares propios ---------------------------------------------------------------------------


class PathBodyLimitMiddleware:
    """Límite de body por ruta: `upload_max` en `upload_paths` (solo POST), `default_max` en el resto.

    Va por dentro del `BodySizeLimitMiddleware` de `install_security`, que aplica el tope mayor.
    """

    def __init__(self, app: Any, *, default_max: int, upload_max: int, upload_paths: Sequence[str]) -> None:
        self.app = app
        self.paths = frozenset(upload_paths)
        self.default = BodySizeLimitMiddleware(app, max_bytes=default_max)
        self.upload = BodySizeLimitMiddleware(app, max_bytes=upload_max)

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] == "http" and scope.get("method") == "POST" and scope.get("path") in self.paths:
            await self.upload(scope, receive, send)
        else:
            await self.default(scope, receive, send)


class StatusMetricsMiddleware:
    """Cuenta 401/403 como fallos de autenticación y 429 como rate limit (sin labels)."""

    def __init__(self, app: Any, *, metrics: Metrics) -> None:
        self.app = app
        self.metrics = metrics

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_wrapper(message: Any) -> None:
            if message["type"] == "http.response.start":
                status = int(message.get("status", 0))
                if status in (401, 403):
                    self.metrics.inc_auth_failure()
                elif status == 429:
                    self.metrics.inc_rate_limited()
            await send(message)

        await self.app(scope, receive, send_wrapper)


def _runtime(request: Request) -> Runtime:
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(503, "La plataforma no terminó de arrancar.")
    return runtime


def _context_dependency(scope: str) -> Any:
    """`get_context` para `create_api_router`: exige la API key con `scope` y arma el contexto."""

    # Default con Depends (no `Annotated`): con `from __future__ import annotations` la anotación es
    # texto y FastAPI no podría resolver `scope`, que es una variable de esta clausura.
    key_principal = Depends(require_api_key(scope))

    async def get_context(request: Request, principal: Principal = key_principal) -> RequestContext:
        runtime = _runtime(request)
        if runtime.shutting_down:
            raise HTTPException(503, "La plataforma se está cerrando.", headers={"Retry-After": "5"})
        return await run_in_threadpool(runtime.context_for, principal)

    return get_context


def _api_router(settings: Settings) -> APIRouter:
    """Rutas de `vigia.api`: generación con scope `ingest`; lectura de trazas con scope `read`."""
    kwargs = {
        "max_tokens_limit": settings.max_tokens_limit,
        "default_max_tokens": settings.default_max_tokens,
    }
    ingest = create_api_router(_context_dependency("ingest"), **kwargs)
    read = create_api_router(_context_dependency("read"), **kwargs)
    router = APIRouter()
    router.routes.extend(r for r in ingest.routes if getattr(r, "path", None) != TRACE_ROUTE)
    router.routes.extend(r for r in read.routes if getattr(r, "path", None) == TRACE_ROUTE)
    return router


def _health_router() -> APIRouter:
    router = APIRouter()
    public = [Depends(public_route())]

    @router.get("/livez", dependencies=public, include_in_schema=False)
    async def livez() -> JSONResponse:
        return JSONResponse({"status": "ok"})

    @router.get("/startupz", dependencies=public, include_in_schema=False)
    async def startupz(request: Request) -> JSONResponse:
        runtime = getattr(request.app.state, "runtime", None)
        model = runtime is not None and getattr(runtime, "model_info", None) is not None
        detectors = (
            runtime is not None
            and bool(getattr(runtime, "startup_complete", False))
            and (getattr(runtime, "startup_errors", 1) == 0)
        )
        ok = model and detectors
        body = {"status": "ok" if ok else "starting", "checks": {"model": model, "detectors": detectors}}
        return JSONResponse(body, status_code=200 if ok else 503)

    @router.get("/readyz", dependencies=public, include_in_schema=False)
    async def readyz(request: Request) -> JSONResponse:
        checks = await run_in_threadpool(_runtime(request).readiness)
        ok = all(checks.values())
        return JSONResponse(
            {"status": "ok" if ok else "unavailable", "checks": checks}, status_code=200 if ok else 503
        )

    @router.get("/metrics", dependencies=public, include_in_schema=False)
    async def metrics(request: Request) -> Response:
        runtime = _runtime(request)
        token = runtime.settings.metrics_token
        if not token:
            raise HTTPException(404, "No encontrado.")
        header = request.headers.get("authorization", "")
        scheme, _, provided = header.partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(provided.strip().encode(), token.encode()):
            raise HTTPException(401, "Token de métricas inválido.", headers={"WWW-Authenticate": "Bearer"})
        text = await run_in_threadpool(runtime.metrics.render)
        return Response(text, media_type=CONTENT_TYPE)

    return router


def _panel_router(app: FastAPI, runtime: Runtime) -> APIRouter:
    """Router del panel con su página HTML de errores registrada en la app."""
    try:
        from vigia_ee.panel import PanelHTTPError, create_panel_router, panel_error_handler
    except ModuleNotFoundError as exc:
        if exc.name in ("vigia_ee.panel", "vigia_ee.panel_admin"):
            raise RuntimeError(
                "Falta el módulo del panel (vigia_ee.panel); la plataforma no puede arrancar."
            ) from exc
        raise
    app.add_exception_handler(PanelHTTPError, panel_error_handler)  # type: ignore[arg-type]
    return create_panel_router(runtime)


def _mount(app: FastAPI, router: APIRouter) -> None:
    """Agrega las rutas del router a la app tal cual (sin prefijo ni dependencias extra).

    `include_router` de FastAPI 0.14x guarda un `_IncludedRouter` opaco en `app.routes`; así las
    rutas quedan visibles para `undeclared_routes` y `viewer_mutation_violations` (AT-P02).
    """
    app.router.routes.extend(router.routes)


# --- App -------------------------------------------------------------------------------------------


def create_app(settings: Settings | None = None, *, runtime: Runtime | None = None) -> FastAPI:
    """Arma la app. Sin `runtime`, crea uno con `settings` (o `Settings.from_env()`) y lo cierra al salir."""
    owns_runtime = runtime is None
    if runtime is None:
        runtime = Runtime(settings or Settings.from_env())
    settings = runtime.settings

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        runtime.jobs.start()
        retention.schedule(runtime.jobs, every_hours=PURGE_EVERY_HOURS)
        try:
            yield
        finally:
            runtime.jobs.stop()
            if owns_runtime:
                runtime.close()

    app = FastAPI(
        title="Vigía",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.runtime = runtime
    # Orden (de afuera hacia adentro): métricas → seguridad → límite por ruta → auth y rutas.
    app.add_middleware(
        PathBodyLimitMiddleware,
        default_max=settings.max_body_bytes,
        upload_max=settings.max_upload_bytes + MULTIPART_MARGIN,
        upload_paths=UPLOAD_PATHS,
    )
    install_security(
        app, max_body_bytes=max(settings.max_body_bytes, settings.max_upload_bytes + MULTIPART_MARGIN)
    )
    app.add_middleware(StatusMetricsMiddleware, metrics=runtime.metrics)
    app.add_middleware(MetricsMiddleware, metrics=runtime.metrics)
    install_auth(app, runtime.auth)
    _mount(app, _health_router())
    _mount(app, _api_router(settings))
    _mount(app, _panel_router(app, runtime))
    return app


# --- CLI -------------------------------------------------------------------------------------------


def _read_owner_password(environ: Any) -> str:
    path = environ.get("VIGIA_OWNER_PASSWORD_FILE")
    if path:
        try:
            with open(path, encoding="utf-8") as fh:
                return fh.read().rstrip("\r\n")
        except OSError:
            raise SettingsError("No se pudo leer VIGIA_OWNER_PASSWORD_FILE.") from None
    first = getpass.getpass("Contraseña del owner (mínimo 15 caracteres): ")
    second = getpass.getpass("Repetí la contraseña: ")
    if first != second:
        raise SettingsError("Las contraseñas no coinciden.")
    return first


def _admin_state(settings: Settings) -> AuthState:
    """Base y auth sin cargar el modelo (para `init` y `create-key`)."""
    db = open_database(settings)
    try:
        return AuthState(db=db, peppers=settings.peppers or PepperRing.from_env())
    except BaseException:
        db.close()
        raise


def cmd_init(args: argparse.Namespace, settings: Settings, environ: Any) -> int:
    password = _read_owner_password(environ)
    check_password_policy(password, email=args.email)
    state = _admin_state(settings)
    try:
        with state.db.transaction():
            org_id = create_org(
                state.db, args.org, store_text=args.store_text, retention_days=args.retention_days
            )
            audit.append(state.db, org_id, None, "org_create", org_id)
            user_id = register_user(state, org_id, args.email, password, "owner")
    finally:
        state.db.close()
    print(f"Organización creada: {org_id}")
    print(f"Owner creado: {user_id} ({args.email})")
    print("El detector base (VIGIA_BASE_DETECTOR) se instala al arrancar `serve` si coincide con el modelo.")
    return EXIT_OK


def cmd_create_key(args: argparse.Namespace, settings: Settings, environ: Any) -> int:
    scopes = [s.strip() for s in args.scopes.split(",") if s.strip()]
    state = _admin_state(settings)
    try:
        if state.repo(args.org_id).get_org() is None:
            raise SettingsError("La organización no existe.")
        expires = utcnow() + timedelta(days=args.expires_days) if args.expires_days else None
        token, key_id = create_api_key(state, args.org_id, args.name, scopes, expires_at=expires)
    finally:
        state.db.close()
    print(f"API key creada: {key_id}")
    print("Token (se muestra una sola vez; guardalo en un gestor de secretos):")
    print(token)
    return EXIT_OK


def _configure_logging() -> None:
    """Logs en JSON: el logger `vigia` y también `uvicorn` y `asyncio`.

    Con `log_config=None`, uvicorn no instala handlers y sus avisos (por ejemplo "Invalid HTTP
    request received.") y los de asyncio caían en texto plano con stack traces. Comparten el handler
    de `vigia`, cuyo formateador descarta `exc_info`.
    """
    configure_json_logging()
    handler = next((h for h in logging.getLogger("vigia").handlers if getattr(h, "_vigia_json", False)), None)
    if handler is None:
        return
    for name in FOREIGN_LOGGERS:
        logger = logging.getLogger(name)
        if handler not in logger.handlers:
            logger.addHandler(handler)
        logger.propagate = False


def _server(app: FastAPI, runtime: Runtime, **kwargs: Any) -> Any:
    """`uvicorn.Server` que marca el runtime como cerrando apenas llega la señal de salida."""
    import uvicorn

    class ClosingServer(uvicorn.Server):
        def handle_exit(self, sig: int, frame: Any) -> None:
            runtime.begin_shutdown()
            super().handle_exit(sig, frame)

    return ClosingServer(uvicorn.Config(app, **kwargs))


def cmd_serve(args: argparse.Namespace, settings: Settings, environ: Any) -> int:
    _configure_logging()
    app = create_app(settings)
    server = _server(
        app,
        app.state.runtime,
        host=args.host,
        port=args.port,
        server_header=False,
        proxy_headers=args.proxy_headers,
        forwarded_allow_ips=args.forwarded_allow_ips,
        workers=1,  # el modelo y los rate limiters viven en este proceso
        log_config=None,
        timeout_graceful_shutdown=settings.shutdown_timeout_s,
    )
    server.run()
    return EXIT_OK


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("debe ser un entero") from None
    if number < 1:
        raise argparse.ArgumentTypeError("debe ser mayor que cero")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vigia-platform", description="Plataforma Vigía (panel + API).")
    sub = parser.add_subparsers(dest="command", required=True)
    p_init = sub.add_parser("init", help="crea una organización y su owner")
    p_init.add_argument("--org", required=True, help="nombre de la organización")
    p_init.add_argument("--email", required=True, help="email del owner")
    p_init.add_argument("--store-text", choices=STORE_TEXT_MODES, default="redacted")
    p_init.add_argument("--retention-days", type=_positive_int, default=30)
    p_init.set_defaults(handler=cmd_init)
    p_key = sub.add_parser("create-key", help="crea una API key e imprime el token una sola vez")
    p_key.add_argument("--org-id", required=True)
    p_key.add_argument("--name", required=True)
    p_key.add_argument("--scopes", default="ingest", help="lista separada por comas: ingest, read, admin")
    p_key.add_argument("--expires-days", type=_positive_int, default=None)
    p_key.set_defaults(handler=cmd_create_key)
    p_serve = sub.add_parser("serve", help="sirve el panel y la API con uvicorn")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=_positive_int, default=8000)
    p_serve.add_argument(
        "--proxy-headers",
        action="store_true",
        help="confiar en X-Forwarded-* (solo detrás de un proxy propio)",
    )
    p_serve.add_argument("--forwarded-allow-ips", default="127.0.0.1")
    p_serve.set_defaults(handler=cmd_serve)
    return parser


def main(argv: Sequence[str] | None = None, *, environ: Any = None) -> int:
    env = os.environ if environ is None else environ
    args = build_parser().parse_args(argv)
    try:
        settings = Settings.from_env(env)
        return int(args.handler(args, settings, env))
    except (SettingsError, PepperConfigError, PasswordPolicyError, PermissionError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except KeyboardInterrupt:
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
