"""Webhooks salientes con el formato Standard Webhooks.

- Firma: `webhook-signature: v1,<base64(HMAC-SHA256(secreto, "{id}.{timestamp}.{cuerpo}"))>`. El
  secreto es `whsec_<base64>`. `webhook-id` es el mismo en cada reintento: el receptor lo usa como
  clave de idempotencia. `webhook-timestamp` es la hora del intento, así un reintento a las 10 h
  sigue dentro de la tolerancia del receptor.
- Cada evento se encola como un job `webhook` por webhook suscrito. La entrega corre en un carril
  propio del worker de `vigia_ee.jobs` (`lane="webhook"`): un destino lento o caído no atrasa la
  purga ni la calibración. Usa el calendario de reintentos de la spec (5 s, 5 min, 30 min, 2 h,
  5 h, 10 h, 10 h). Solo un 2xx cuenta como éxito; un 3xx es fallo porque nunca se siguen redirects.
- Aislamiento entre tenants: el carril tiene `LANE_WORKERS` hilos y cada tenant tiene a lo sumo una
  entrega en curso, así el destino lento de un tenant no frena los webhooks de los demás. Cada
  entrega tiene un plazo TOTAL de `timeout` segundos (conexión, TLS, envío y headers). httpx solo
  acota cada lectura; un destino que gotea bytes la retendría sin límite. Superar el plazo es un
  fallo: cuenta para el reintento y para la deshabilitación automática. El plazo total vale para el
  cliente propio; un `http_client` inyectado conserva sus timeouts.
- SSRF: la URL se valida con `vigia.policy.check_url` al crear el webhook y antes de cada entrega
  (https obligatorio salvo allowlist; localhost, redes privadas y 169.254.169.254 bloqueados).
  `ponytail:` queda una ventana TOCTOU de DNS entre la validación y la conexión de httpx; cerrarla
  exige conectar a la IP validada con SNI propio.
- Tras `max_consecutive_failures` entregas seguidas cuyo último intento falló, el webhook se
  deshabilita y el evento queda en auditoría. Cuenta en cada intento, no solo en el último: un
  destino caído con mucho tráfico se corta en N intentos y no 28 h después. La entrega que lo
  deshabilita termina `failed`; el resto de su cola se descarta sin tocar la red.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import secrets
import threading
import time
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime
from typing import Any

import httpcore
import httpx

from vigia.policy import UnsafeURLError, check_url
from vigia_ee import audit
from vigia_ee.db import Database, TenantScopedRepository, new_id, to_iso
from vigia_ee.jobs import JobQueue, PermanentJobError

JOB_KIND = "webhook"
EVENT_TYPES = ("trace.flagged", "review.completed", "calibration.completed", "calibration.activated")
SECRET_PREFIX = "whsec_"  # noqa: S105 - prefijo público del formato, no un secreto
SIGNATURE_VERSION = "v1"
DEFAULT_TOLERANCE_S = 300
MAX_PAYLOAD_BYTES = 20_000
DELIVERY_TIMEOUT_S = 15.0
# Retrasos tras cada intento fallido; el primer intento es inmediato (Standard Webhooks, recortado).
RETRY_SCHEDULE_S = (5.0, 300.0, 1800.0, 7200.0, 18000.0, 36000.0, 36000.0)
MAX_ATTEMPTS = len(RETRY_SCHEDULE_S) + 1
DEFAULT_MAX_CONSECUTIVE_FAILURES = 5
USER_AGENT = "Vigia-Webhooks/1"
LANE_WORKERS = 4  # hilos del carril de webhooks
MAX_RUNNING_PER_ORG = 1  # entregas en curso por tenant dentro del carril

Resolver = Callable[[str], list[str]]


class WebhookVerificationError(ValueError):
    """La firma, el timestamp o los headers de un webhook entrante no validan."""


class DeliveryError(Exception):
    """El destino no respondió 2xx (o la conexión falló)."""


# --- firma ----------------------------------------------------------------------------------------


def generate_secret() -> str:
    """Secreto nuevo `whsec_<base64 de 32 bytes>`."""
    return SECRET_PREFIX + base64.b64encode(secrets.token_bytes(32)).decode()


def _secret_bytes(secret: str) -> bytes:
    if not isinstance(secret, str) or not secret.startswith(SECRET_PREFIX):
        raise ValueError("el secreto debe empezar con whsec_")
    try:
        key = base64.b64decode(secret[len(SECRET_PREFIX) :], validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("el secreto no es base64 válido") from exc
    if len(key) < 16:
        raise ValueError("el secreto debe tener al menos 16 bytes")
    return key


def _check_msg_id(msg_id: str) -> str:
    if not isinstance(msg_id, str) or not msg_id or len(msg_id) > 256 or "." in msg_id:
        raise ValueError("webhook-id inválido")
    return msg_id


def sign(secret: str, msg_id: str, timestamp: int, payload: bytes) -> str:
    """Firma `v1,<base64>` de Standard Webhooks sobre `{msg_id}.{timestamp}.{payload}`."""
    if isinstance(timestamp, bool) or not isinstance(timestamp, int):
        raise ValueError("el timestamp debe ser un entero de segundos unix")
    if not isinstance(payload, bytes | bytearray):
        raise ValueError("el payload debe ser bytes")
    content = f"{_check_msg_id(msg_id)}.{timestamp}.".encode() + bytes(payload)
    digest = hmac.new(_secret_bytes(secret), content, hashlib.sha256).digest()
    return f"{SIGNATURE_VERSION},{base64.b64encode(digest).decode()}"


def verify(
    secret: str,
    msg_id: str,
    timestamp: str | int,
    payload: bytes,
    signature: str,
    *,
    tolerance: int = DEFAULT_TOLERANCE_S,
    now: float | None = None,
) -> Any:
    """Verifica un webhook entrante y devuelve el cuerpo parseado.

    `signature` es el header completo: admite varias firmas separadas por espacio (rotación de
    secreto). Rechaza timestamps a más de `tolerance` segundos (pasado o futuro) para frenar replay.
    Lanza `WebhookVerificationError` con el motivo.
    """
    try:
        ts = int(str(timestamp).strip())
    except ValueError:
        raise WebhookVerificationError("webhook-timestamp inválido") from None
    current = time.time() if now is None else now
    if abs(current - ts) > tolerance:
        raise WebhookVerificationError("webhook-timestamp fuera de tolerancia")
    try:
        expected = sign(secret, msg_id, ts, payload).split(",", 1)[1]
    except ValueError as exc:
        raise WebhookVerificationError(str(exc)) from None
    for candidate in str(signature or "").split():
        version, _, value = candidate.partition(",")
        if version == SIGNATURE_VERSION and hmac.compare_digest(value.encode(), expected.encode()):
            try:
                return json.loads(payload)
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise WebhookVerificationError("el cuerpo no es JSON válido") from None
    raise WebhookVerificationError("firma inválida")


# --- alta, baja y estado ----------------------------------------------------------------------------


def _check_events(events: Iterable[str]) -> tuple[str, ...]:
    if isinstance(events, str):
        events = [e for e in events.split(",") if e.strip()]
    chosen = tuple(dict.fromkeys(e.strip() for e in events))
    if not chosen:
        raise ValueError("elegí al menos un evento")
    unknown = [e for e in chosen if e not in EVENT_TYPES]
    if unknown:
        raise ValueError(f"eventos desconocidos: {unknown}; válidos: {list(EVENT_TYPES)}")
    return chosen


def validate_url(url: str, *, allowed_hosts: Collection[str] = (), resolve: Resolver | None = None) -> str:
    """Valida el destino contra SSRF. Lanza `UnsafeURLError` (la API responde 422)."""
    if not isinstance(url, str) or not url.strip():
        raise UnsafeURLError("la URL es obligatoria")
    check_url(url.strip(), allowed_hosts, resolve=resolve)
    return url.strip()


def create_webhook(
    db: Database,
    org_id: str,
    url: str,
    events: Iterable[str],
    actor: Any,
    *,
    allowed_hosts: Collection[str] = (),
    resolve: Resolver | None = None,
) -> tuple[str, str]:
    """Crea un webhook y devuelve `(webhook_id, secreto)`. El secreto se muestra una sola vez."""
    clean_url = validate_url(url, allowed_hosts=allowed_hosts, resolve=resolve)
    chosen = _check_events(events)
    secret = generate_secret()
    repo = TenantScopedRepository(db, org_id)
    with db.transaction():
        webhook_id = repo.insert_webhook(clean_url, secret, chosen)
        audit.append(db, org_id, actor, "webhook_create", webhook_id, {"events": list(chosen)})
    return webhook_id, secret


def delete_webhook(db: Database, org_id: str, webhook_id: str, actor: Any) -> bool:
    """Borra un webhook del tenant. Otro tenant → False (404)."""
    repo = TenantScopedRepository(db, org_id)
    with db.transaction():
        if not repo.delete_webhook(webhook_id):
            return False
        audit.append(db, org_id, actor, "webhook_delete", webhook_id)
    return True


def set_enabled(
    db: Database,
    org_id: str,
    webhook_id: str,
    enabled: bool,
    actor: Any,
    *,
    allowed_hosts: Collection[str] = (),
    resolve: Resolver | None = None,
) -> bool:
    """Habilita o deshabilita un webhook. Habilitar vuelve a validar la URL contra SSRF."""
    repo = TenantScopedRepository(db, org_id)
    hook = repo.get_webhook(webhook_id)
    if hook is None:
        return False
    if enabled:
        validate_url(hook["url"], allowed_hosts=allowed_hosts, resolve=resolve)
    with db.transaction():
        if not repo.set_webhook_disabled(webhook_id, not enabled):
            return False
        audit.append(db, org_id, actor, "webhook_enable" if enabled else "webhook_disable", webhook_id)
    return True


# --- emisión --------------------------------------------------------------------------------------


def build_body(event_type: str, data: Mapping[str, Any], *, now: datetime) -> bytes:
    """Cuerpo `{type, timestamp, data}` en JSON compacto. Si supera 20 KB, manda solo la referencia."""
    if not isinstance(data, Mapping):
        raise ValueError("data debe ser un objeto")

    def dump(body: Mapping[str, Any]) -> bytes:
        try:
            return json.dumps(
                body, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        except (TypeError, ValueError) as exc:
            raise ValueError("data debe ser JSON serializable") from exc

    body = {"type": event_type, "timestamp": to_iso(now), "data": dict(data)}
    raw = dump(body)
    if len(raw) <= MAX_PAYLOAD_BYTES:
        return raw
    ref: dict[str, Any] = {"truncated": True}
    trace_id = data.get("trace_id")
    if isinstance(trace_id, str) and len(trace_id) <= 64:
        ref["trace_id"] = trace_id
    return dump({"type": event_type, "timestamp": body["timestamp"], "data": ref})


def emit(db: Database, jobs: JobQueue, org_id: str, event_type: str, data: Mapping[str, Any]) -> list[str]:
    """Encola una entrega por cada webhook habilitado del tenant suscrito a `event_type`.

    Devuelve los ids de job. Todas las entregas de un evento comparten `webhook-id`.
    """
    if event_type not in EVENT_TYPES:
        raise ValueError(f"evento desconocido: {event_type!r}")
    hooks = TenantScopedRepository(db, org_id).list_webhooks(event=event_type, enabled_only=True)
    if not hooks:
        return []
    body = build_body(event_type, data, now=jobs.now()).decode()
    msg_id = f"msg_{new_id()}"
    return [
        jobs.enqueue(
            JOB_KIND,
            {"webhook_id": h["id"], "msg_id": msg_id, "event_type": event_type, "body": body},
            org_id=org_id,
            max_attempts=MAX_ATTEMPTS,
        )
        for h in hooks
    ]


# --- plazo total de entrega -----------------------------------------------------------------------

_deadline = threading.local()  # `at`: time.monotonic() límite de la entrega que corre en este hilo


@contextmanager
def _delivery_deadline(seconds: float) -> Iterator[None]:
    """Fija el plazo total de la entrega que corre en este hilo."""
    previous = getattr(_deadline, "at", None)
    _deadline.at = time.monotonic() + seconds
    try:
        yield
    finally:
        _deadline.at = previous


def _bounded(timeout: float | None, expired: type[Exception]) -> float | None:
    """Timeout de una operación de red recortado al plazo total. Lanza `expired` si ya venció."""
    deadline = getattr(_deadline, "at", None)
    if deadline is None:
        return timeout
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise expired("se venció el plazo total de la entrega")
    return remaining if timeout is None else min(timeout, remaining)


class _DeadlineStream(httpcore.NetworkStream):
    """Conexión de httpcore que recorta cada lectura y escritura al plazo total del hilo."""

    def __init__(self, inner: httpcore.NetworkStream) -> None:
        self._inner = inner

    def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        return self._inner.read(max_bytes, _bounded(timeout, httpcore.ReadTimeout))

    def write(self, buffer: bytes, timeout: float | None = None) -> None:
        self._inner.write(buffer, _bounded(timeout, httpcore.WriteTimeout))

    def close(self) -> None:
        self._inner.close()

    def start_tls(
        self, ssl_context: Any, server_hostname: str | None = None, timeout: float | None = None
    ) -> httpcore.NetworkStream:
        bounded = _bounded(timeout, httpcore.ConnectTimeout)
        return _DeadlineStream(self._inner.start_tls(ssl_context, server_hostname, bounded))

    def get_extra_info(self, info: str) -> Any:
        return self._inner.get_extra_info(info)


class _DeadlineBackend(httpcore.NetworkBackend):
    """Backend de red que acota la conexión al plazo total y abre `_DeadlineStream`."""

    def __init__(self, inner: httpcore.NetworkBackend) -> None:
        self._inner = inner

    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Any = None,
    ) -> httpcore.NetworkStream:
        bounded = _bounded(timeout, httpcore.ConnectTimeout)
        return _DeadlineStream(self._inner.connect_tcp(host, port, bounded, local_address, socket_options))

    def connect_unix_socket(
        self, path: str, timeout: float | None = None, socket_options: Any = None
    ) -> httpcore.NetworkStream:
        bounded = _bounded(timeout, httpcore.ConnectTimeout)
        return _DeadlineStream(self._inner.connect_unix_socket(path, bounded, socket_options))

    def sleep(self, seconds: float) -> None:
        self._inner.sleep(seconds)


def _delivery_client(timeout: float) -> httpx.Client:
    """Cliente propio de entrega: sin redirects ni proxies del entorno y con plazo total.

    httpx no expone el backend de red, así que se envuelve el del pool. Si httpcore cambia ese
    atributo, falla al crear el handler en vez de entregar sin plazo total.
    """
    transport = httpx.HTTPTransport(trust_env=False)
    pool = getattr(transport, "_pool", None)
    backend = getattr(pool, "_network_backend", None)
    if not isinstance(backend, httpcore.NetworkBackend):
        raise RuntimeError("httpcore cambió: no se puede imponer el plazo total de entrega")
    pool._network_backend = _DeadlineBackend(backend)  # type: ignore[union-attr]
    return httpx.Client(transport=transport, timeout=timeout, follow_redirects=False, trust_env=False)


# --- entrega --------------------------------------------------------------------------------------


def _consecutive_failures(db: Database, org_id: str, webhook_id: str, limit: int, current: str) -> int:
    """Entregas seguidas cuyo último intento falló, de la más nueva hacia atrás (sin la actual).

    Cuenta las `failed` y las que esperan un reintento tras un fallo (`queued` con intentos y
    error). Una `done` (éxito o descartada) corta la racha; las que nunca se intentaron no cuentan.
    """
    rows = db.query(
        "SELECT state FROM jobs WHERE org_id = ? AND kind = ? AND id != ? "
        "AND json_extract(payload_json, '$.webhook_id') = ? "
        "AND (state IN ('done', 'failed') OR (state = 'queued' AND attempts > 0 AND error IS NOT NULL "
        "AND error NOT LIKE 'LeaseExpired:%')) "
        "ORDER BY updated_at DESC, rowid DESC LIMIT ?",
        (org_id, JOB_KIND, current, webhook_id, limit),
    )
    count = 0
    for r in rows:
        if r["state"] == "done":
            break
        count += 1
    return count


def make_delivery_handler(
    db: Database,
    jobs: JobQueue,
    *,
    http_client: httpx.Client | None = None,
    allowed_hosts: Collection[str] = (),
    resolve: Resolver | None = None,
    timeout: float = DELIVERY_TIMEOUT_S,
    max_consecutive_failures: int = DEFAULT_MAX_CONSECUTIVE_FAILURES,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Handler del job `webhook`: valida destino, firma y hace el POST sin seguir redirects."""
    if max_consecutive_failures < 1:
        raise ValueError("max_consecutive_failures debe ser al menos 1")
    if timeout <= 0:
        raise ValueError("timeout debe ser positivo")
    client = http_client or _delivery_client(timeout)

    def disable_if_needed(org_id: str, webhook_id: str, job_id: str, error: str) -> bool:
        """Deshabilita el webhook si este intento completa la racha. True si quedó deshabilitado."""
        previous = _consecutive_failures(db, org_id, webhook_id, max_consecutive_failures, job_id)
        failures = previous + 1  # la entrega actual sigue en `running`
        if failures < max_consecutive_failures:
            return False
        repo = TenantScopedRepository(db, org_id)
        with db.transaction():
            if repo.set_webhook_disabled(webhook_id, True):
                audit.append(
                    db,
                    org_id,
                    None,
                    "webhook_auto_disabled",
                    webhook_id,
                    {"consecutive_failures": failures, "last_error": error},
                )
        return True

    def handler(job: dict[str, Any]) -> dict[str, Any]:
        org_id = job["org_id"]
        payload = job["payload"]
        webhook_id = payload["webhook_id"]
        if not org_id:
            raise PermanentJobError("un webhook siempre pertenece a un tenant")
        hook = TenantScopedRepository(db, org_id).get_webhook(webhook_id)
        if hook is None:
            return {"skipped": "webhook_deleted"}
        if hook["disabled"]:
            return {"skipped": "webhook_disabled"}
        try:
            try:
                validate_url(hook["url"], allowed_hosts=allowed_hosts, resolve=resolve)
            except UnsafeURLError as exc:
                raise PermanentJobError(f"destino bloqueado: {exc}") from None
            body = payload["body"].encode()
            timestamp = int(jobs.now().timestamp())
            headers = {
                "content-type": "application/json",
                "user-agent": USER_AGENT,
                "webhook-id": payload["msg_id"],
                "webhook-timestamp": str(timestamp),
                "webhook-signature": sign(hook["secret"], payload["msg_id"], timestamp, body),
            }
            try:
                with (
                    _delivery_deadline(timeout),
                    client.stream(
                        "POST",
                        hook["url"],
                        content=body,
                        headers=headers,
                        follow_redirects=False,
                        timeout=timeout,
                    ) as response,
                ):
                    status = response.status_code
            except httpx.TimeoutException:
                raise DeliveryError(f"el destino superó el plazo de {timeout:g} s") from None
            except httpx.HTTPError as exc:
                # Sin el mensaje: puede incluir la URL con tokens del cliente.
                raise DeliveryError(f"falló la conexión ({type(exc).__name__})") from None
            if not 200 <= status < 300:
                raise DeliveryError(f"el destino respondió HTTP {status}")
        except (DeliveryError, PermanentJobError) as exc:
            if disable_if_needed(org_id, webhook_id, job["id"], str(exc)[:200]) and not isinstance(
                exc, PermanentJobError
            ):
                raise PermanentJobError(f"webhook deshabilitado tras fallos seguidos: {exc}") from None
            raise
        return {"status": status, "attempt": job["attempts"], "webhook_id": webhook_id}

    return handler


def register_delivery(
    jobs: JobQueue,
    *,
    http_client: httpx.Client | None = None,
    allowed_hosts: Collection[str] = (),
    resolve: Resolver | None = None,
    timeout: float = DELIVERY_TIMEOUT_S,
    max_consecutive_failures: int = DEFAULT_MAX_CONSECUTIVE_FAILURES,
) -> None:
    """Registra el handler `webhook` en su carril propio con el calendario de reintentos de la spec.

    El carril corre `LANE_WORKERS` hilos con a lo sumo `MAX_RUNNING_PER_ORG` entrega en curso por
    tenant: el destino lento de un tenant nunca deja sin hilo a los demás.
    """
    handler = make_delivery_handler(
        jobs.db,
        jobs,
        http_client=http_client,
        allowed_hosts=allowed_hosts,
        resolve=resolve,
        timeout=timeout,
        max_consecutive_failures=max_consecutive_failures,
    )
    jobs.configure_lane(JOB_KIND, workers=LANE_WORKERS, max_running_per_org=MAX_RUNNING_PER_ORG)
    jobs.register(JOB_KIND, handler, retry_schedule=RETRY_SCHEDULE_S, lane=JOB_KIND)
