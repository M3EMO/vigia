"""Políticas de acción por respuesta (P-11) y retriever estilo FLARE para rag_retry (P-12).

La acción se decide por prioridad `abstain > rag_retry > human_review > annotate > pass`. Los umbrales
comparan el p_error calibrado de la respuesta con `>=`. Sin p_error (sin detector calibrado o detector
degradado) la acción es `pass`: Vigía no actúa sobre una señal sin calibrar (ADR A-06).
"""

from __future__ import annotations

import ipaddress
import math
import socket
from collections.abc import Callable, Collection, Sequence
from dataclasses import fields
from typing import Any, Protocol
from urllib.parse import urlsplit

import httpx

from vigia.contracts import ACTION_PRIORITY, Action, Policy, Span, VigiaError

RAG_FALLBACK_ACTION: Action = "human_review"
MAX_ABSTAIN_MESSAGE = 2000
MAX_RAG_RETRIES = 5
MAX_GAP = 1000
MAX_MIN_TOKENS = 10_000


class PolicyError(VigiaError, ValueError):
    """La política es inválida. El mensaje nombra el campo."""


class RetrieverError(VigiaError):
    """El retriever falló, venció o respondió algo inválido."""


class UnsafeURLError(RetrieverError, ValueError):
    """La URL apunta a un destino no permitido (SSRF)."""


# --- Decisión -----------------------------------------------------------------------------------


def matched_actions(policy: Policy, response_p_error: float | None, spans: Sequence[Span]) -> list[Action]:
    """Todas las acciones cuyo umbral se cumple, en orden de prioridad. Siempre termina en `pass`."""
    out: list[Action] = []
    if response_p_error is not None:
        p = float(response_p_error)
        if policy.abstain_at is not None and p >= policy.abstain_at:
            out.append("abstain")
        if policy.rag_retry_at is not None and p >= policy.rag_retry_at:
            out.append("rag_retry")
        if policy.human_review_at is not None and p >= policy.human_review_at:
            out.append("human_review")
        if policy.annotate and spans:
            out.append("annotate")
    out.append("pass")
    return out


def decide(policy: Policy, response_p_error: float | None, spans: Sequence[Span]) -> Action:
    """Acción de mayor prioridad entre las que se cumplen."""
    matched = matched_actions(policy, response_p_error, spans)
    return min(matched, key=ACTION_PRIORITY.index)


# --- Validación ---------------------------------------------------------------------------------


def _check_threshold(name: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise PolicyError(f"{name} debe ser un número entre 0 y 1 o null")
    if not 0.0 <= float(value) <= 1.0:
        raise PolicyError(f"{name} debe estar entre 0 y 1")


def _check_int(name: str, value: Any, lo: int, hi: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
        raise PolicyError(f"{name} debe ser un entero entre {lo} y {hi}")


def validate_policy(
    policy: Policy,
    *,
    rag_allowed_hosts: Collection[str] = (),
    resolve: Callable[[str], list[str]] | None = None,
) -> Policy:
    """Valida rangos y la URL del retriever. Devuelve la misma política o lanza PolicyError."""
    a = policy.alpha
    if isinstance(a, bool) or not isinstance(a, int | float) or not math.isfinite(a) or not 0.0 < a < 1.0:
        raise PolicyError("alpha debe estar en el intervalo abierto (0, 1)")
    _check_int("gap_max", policy.gap_max, 0, MAX_GAP)
    _check_int("min_tokens", policy.min_tokens, 1, MAX_MIN_TOKENS)
    if not isinstance(policy.annotate, bool):
        raise PolicyError("annotate debe ser booleano")
    for name in ("human_review_at", "rag_retry_at", "abstain_at"):
        _check_threshold(name, getattr(policy, name))
    msg = policy.abstain_message
    if not isinstance(msg, str) or not msg.strip() or len(msg) > MAX_ABSTAIN_MESSAGE:
        raise PolicyError(f"abstain_message: texto no vacío de {MAX_ABSTAIN_MESSAGE} caracteres o menos")
    _check_int("rag_max_retries", policy.rag_max_retries, 0, MAX_RAG_RETRIES)
    if policy.on_detector_error not in ("fail_closed", "fail_open"):
        raise PolicyError("on_detector_error debe ser fail_closed o fail_open")
    if policy.stream_policy not in ("annotate_only", "buffer"):
        raise PolicyError("stream_policy debe ser annotate_only o buffer")
    _check_int("version", policy.version, 1, 2**31 - 1)
    if policy.rag_url is not None:
        if not isinstance(policy.rag_url, str):
            raise PolicyError("rag_url debe ser texto")
        try:
            check_url(policy.rag_url, rag_allowed_hosts, resolve=resolve)
        except UnsafeURLError as exc:
            raise PolicyError(f"rag_url inválida: {exc}") from exc
    if policy.rag_retry_at is not None and policy.rag_url is None:
        raise PolicyError("rag_retry_at exige rag_url")
    return policy


def policy_from_dict(
    data: dict[str, Any],
    *,
    rag_allowed_hosts: Collection[str] = (),
    resolve: Callable[[str], list[str]] | None = None,
) -> Policy:
    """Construye y valida una Policy desde JSON. Rechaza campos desconocidos."""
    if not isinstance(data, dict):
        raise PolicyError("la política debe ser un objeto JSON")
    known = {f.name for f in fields(Policy)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise PolicyError(f"campos desconocidos en la política: {', '.join(unknown)}")
    return validate_policy(Policy(**data), rag_allowed_hosts=rag_allowed_hosts, resolve=resolve)


# --- Protección SSRF ----------------------------------------------------------------------------


def _resolve_host(host: str) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except (socket.gaierror, UnicodeError) as exc:
        raise UnsafeURLError(f"no se pudo resolver el host {host!r}") from exc
    return [str(info[4][0]) for info in infos]


def _is_public_ip(addr: str) -> bool:
    ip = ipaddress.ip_address(addr.split("%")[0])
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


def _normalize_hosts(hosts: Collection[str]) -> set[str]:
    return {h.strip().lower().rstrip(".") for h in hosts if h and h.strip()}


def check_url(
    url: str,
    allowed_hosts: Collection[str] = (),
    *,
    resolve: Callable[[str], list[str]] | None = None,
) -> str:
    """Valida una URL saliente y devuelve su host normalizado.

    - Un host de `allowed_hosts` se acepta con http o https, aunque sea privado (allowlist explícita).
    - Con allowlist no vacía, cualquier otro host se rechaza.
    - Sin allowlist, la URL debe ser https y todas las IPs del host deben ser públicas: se bloquean
      localhost, loopback, redes privadas, link-local (169.254.169.254) y reservadas.
    """
    if len(url) > 2048:
        raise UnsafeURLError("la URL supera 2048 caracteres")
    try:
        parts = urlsplit(url)
        host = (parts.hostname or "").lower().rstrip(".")
        _ = parts.port  # valida el puerto
    except ValueError as exc:
        raise UnsafeURLError("URL mal formada") from exc
    if parts.scheme not in ("http", "https"):
        raise UnsafeURLError("el esquema debe ser http o https")
    if not host:
        raise UnsafeURLError("la URL no tiene host")
    if parts.username is not None or parts.password is not None:
        raise UnsafeURLError("la URL no puede llevar credenciales")
    allowed = _normalize_hosts(allowed_hosts)
    if host in allowed:
        return host
    if allowed:
        raise UnsafeURLError(f"el host {host!r} está fuera de la allowlist")
    if parts.scheme != "https":
        raise UnsafeURLError("sin allowlist, la URL debe usar https")
    if host == "localhost" or host.endswith(".localhost"):
        raise UnsafeURLError("destino localhost bloqueado")
    try:
        addrs = [str(ipaddress.ip_address(host))]
    except ValueError:
        addrs = (resolve or _resolve_host)(host)
    if not addrs:
        raise UnsafeURLError(f"el host {host!r} no resolvió a ninguna IP")
    for addr in addrs:
        try:
            public = _is_public_ip(addr)
        except ValueError as exc:
            raise UnsafeURLError(f"IP inválida para {host!r}") from exc
        if not public:
            raise UnsafeURLError(f"el host {host!r} apunta a una red privada o reservada")
    return host


# --- Retriever estilo FLARE ---------------------------------------------------------------------


class Retriever(Protocol):
    def retrieve(self, query: str) -> list[str]: ...


def mask_spans(text: str, spans: Sequence[Span], mask: str = "___") -> str:
    """Texto con cada fragmento dudoso reemplazado por `mask` (consulta estilo FLARE)."""
    out: list[str] = []
    pos = 0
    for s in sorted(spans, key=lambda s: s.start):
        if s.start < pos:
            continue
        out.append(text[pos : s.start])
        out.append(mask)
        pos = s.end
    out.append(text[pos:])
    return "".join(out)


def build_rag_query(question: str, answer: str, spans: Sequence[Span], max_chars: int = 2000) -> str:
    """Consulta para el retriever: la pregunta más la respuesta con los fragmentos dudosos enmascarados."""
    query = f"{question.strip()}\n{mask_spans(answer, spans).strip()}".strip()
    return query[:max_chars]


def inject_context(messages: list[dict[str, str]], documents: Sequence[str]) -> list[dict[str, str]]:
    """Mensajes con el contexto recuperado como primer mensaje de sistema."""
    numbered = "\n".join(f"[{i}] {d}" for i, d in enumerate(documents, start=1))
    ctx = {
        "role": "system",
        "content": "Usá solo el siguiente contexto recuperado si es relevante para responder.\n" + numbered,
    }
    return [ctx, *messages]


class HttpRetriever:
    """Retriever por HTTP: POST {"query": ...} -> {"documents": [str | {"text": str}, ...]}.

    Valida la URL contra SSRF en la construcción y en cada llamada (la resolución DNS puede cambiar).
    No sigue redirecciones. Acota cantidad y tamaño de los documentos.
    """

    def __init__(
        self,
        url: str,
        *,
        allowed_hosts: Collection[str] = (),
        timeout: float = 5.0,
        http_client: httpx.Client | None = None,
        resolve: Callable[[str], list[str]] | None = None,
        max_documents: int = 5,
        max_chars: int = 8000,
    ) -> None:
        self.url = url
        self.allowed_hosts = tuple(allowed_hosts)
        self._resolve = resolve
        check_url(url, self.allowed_hosts, resolve=resolve)
        self._client = http_client or httpx.Client(timeout=timeout, follow_redirects=False)
        self.timeout = timeout
        self.max_documents = max_documents
        self.max_chars = max_chars

    def retrieve(self, query: str) -> list[str]:
        check_url(self.url, self.allowed_hosts, resolve=self._resolve)
        try:
            resp = self._client.post(
                self.url, json={"query": query}, timeout=self.timeout, follow_redirects=False
            )
        except httpx.HTTPError as exc:
            raise RetrieverError(f"el retriever no respondió: {type(exc).__name__}") from exc
        if resp.status_code != 200:
            raise RetrieverError(f"el retriever respondió HTTP {resp.status_code}")
        try:
            payload = resp.json()
        except ValueError as exc:
            raise RetrieverError("el retriever respondió JSON inválido") from exc
        docs = payload.get("documents") if isinstance(payload, dict) else None
        if not isinstance(docs, list):
            raise RetrieverError("la respuesta del retriever no tiene una lista 'documents'")
        out: list[str] = []
        budget = self.max_chars
        for d in docs[: self.max_documents]:
            text = d.get("text") if isinstance(d, dict) else d
            if not isinstance(text, str):
                raise RetrieverError("cada documento debe ser texto o un objeto con 'text'")
            text = text[:budget]
            if text:
                out.append(text)
                budget -= len(text)
            if budget <= 0:
                break
        return out
