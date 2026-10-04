"""Tests de políticas de acción, validación y retriever con protección SSRF (P-11, P-12)."""

from __future__ import annotations

import json

import httpx
import pytest

from vigia.contracts import Policy, Span
from vigia.policy import (
    HttpRetriever,
    PolicyError,
    RetrieverError,
    UnsafeURLError,
    build_rag_query,
    check_url,
    decide,
    inject_context,
    mask_spans,
    matched_actions,
    policy_from_dict,
    validate_policy,
)

SPAN = Span(start=0, end=4, token_start=0, token_end=1, score=0.9, text="1820")


def public(_: str) -> list[str]:
    return ["93.184.216.34"]


def private(_: str) -> list[str]:
    return ["10.0.0.7"]


FULL = Policy(
    abstain_at=0.9,
    rag_retry_at=0.7,
    human_review_at=0.5,
    annotate=True,
    rag_url="https://rag.example.com/q",
)


@pytest.mark.parametrize(
    ("p", "spans", "expected"),
    [
        (0.95, [SPAN], "abstain"),
        (0.9, [SPAN], "abstain"),  # el límite cuenta (>=)
        (0.8, [SPAN], "rag_retry"),
        (0.6, [SPAN], "human_review"),
        (0.5, [], "human_review"),
        (0.3, [SPAN], "annotate"),
        (0.3, [], "pass"),
        (None, [SPAN], "pass"),  # sin p_error calibrado no se actúa (A-06)
    ],
)
def test_AT_A06_priority(p, spans, expected):
    """AT-A06: la acción respeta abstain > rag_retry > human_review > annotate > pass."""
    assert decide(FULL, p, spans) == expected


def test_AT_A06_all_matches_reported_in_priority_order():
    """AT-A06: con varias reglas cumplidas gana la de mayor prioridad y se listan todas."""
    assert matched_actions(FULL, 0.99, [SPAN]) == ["abstain", "rag_retry", "human_review", "annotate", "pass"]
    assert decide(FULL, 0.99, [SPAN]) == "abstain"


def test_disabled_thresholds_and_annotate_off():
    pol = Policy(human_review_at=None, annotate=False)
    assert decide(pol, 0.99, [SPAN]) == "pass"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("alpha", 0.0),
        ("alpha", 1.0),
        ("alpha", float("nan")),
        ("gap_max", -1),
        ("min_tokens", 0),
        ("human_review_at", 1.5),
        ("abstain_at", -0.1),
        ("abstain_message", "  "),
        ("rag_max_retries", 99),
        ("on_detector_error", "ignore"),
        ("stream_policy", "drop"),
        ("version", 0),
    ],
)
def test_validate_policy_rejects_out_of_range(field, value):
    with pytest.raises(PolicyError, match=field):
        validate_policy(Policy(**{field: value}))


def test_validate_policy_accepts_defaults_and_full():
    assert validate_policy(Policy()) == Policy()
    validate_policy(FULL, resolve=public)


def test_rag_retry_requires_url():
    with pytest.raises(PolicyError, match="rag_url"):
        validate_policy(Policy(rag_retry_at=0.5))


def test_policy_from_dict_rejects_unknown_fields():
    with pytest.raises(PolicyError, match="desconocidos"):
        policy_from_dict({"alpha": 0.1, "shell": "rm"})
    assert policy_from_dict({"alpha": 0.2}).alpha == 0.2


@pytest.mark.parametrize(
    "url",
    [
        "http://rag.example.com/q",  # sin allowlist exige https
        "https://localhost/q",
        "https://127.0.0.1/q",
        "https://169.254.169.254/latest/meta-data",
        "https://[::1]/q",
        "https://[::ffff:10.0.0.1]/q",
        "https://user:pw@rag.example.com/q",
        "ftp://rag.example.com/q",
        "https:///nohost",
    ],
)
def test_AT_A07_check_url_blocks_unsafe_targets(url):
    """AT-A07: localhost, IPs privadas, metadata y esquemas inseguros se rechazan sin allowlist."""
    with pytest.raises(UnsafeURLError):
        check_url(url, resolve=public)


def test_check_url_blocks_hostname_resolving_to_private_ip():
    with pytest.raises(UnsafeURLError, match="privada"):
        check_url("https://intranet.example.com/q", resolve=private)
    assert check_url("https://rag.example.com/q", resolve=public) == "rag.example.com"


def test_AT_A07_allowlist_is_explicit_and_exclusive():
    """AT-A07: un host en la allowlist pasa aunque sea privado; un host fuera de ella se rechaza."""
    assert check_url("http://rag.internal:8080/q", ["rag.internal"]) == "rag.internal"
    assert check_url("http://169.254.169.254/x", ["169.254.169.254"]) == "169.254.169.254"
    with pytest.raises(UnsafeURLError, match="allowlist"):
        check_url("https://other.example.com/q", ["rag.internal"], resolve=public)


def test_policy_validation_uses_allowlist():
    pol = Policy(rag_url="http://rag.internal/q", rag_retry_at=0.6)
    with pytest.raises(PolicyError, match="rag_url"):
        validate_policy(pol)
    validate_policy(pol, rag_allowed_hosts=["rag.internal"])


def _transport(calls: list[dict], status: int = 200, payload: object | None = None) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append({"url": str(request.url), "body": json.loads(request.content)})
        if payload is None:
            return httpx.Response(status, content=b"no-json")
        return httpx.Response(status, json=payload)

    return httpx.MockTransport(handler)


def test_AT_A07_http_retriever_posts_query_and_parses_documents():
    """AT-A07: el retriever hace POST con la consulta y devuelve los documentos acotados."""
    calls: list[dict] = []
    payload = {"documents": ["Rosario es una ciudad.", {"text": "Belgrano nació en 1770."}]}
    client = httpx.Client(transport=_transport(calls, payload=payload))
    r = HttpRetriever("http://rag.internal/q", allowed_hosts=["rag.internal"], http_client=client)
    docs = r.retrieve("¿Dónde nació Belgrano? ___")
    assert docs == ["Rosario es una ciudad.", "Belgrano nació en 1770."]
    assert calls == [{"url": "http://rag.internal/q", "body": {"query": "¿Dónde nació Belgrano? ___"}}]


def test_AT_A07_http_retriever_rejects_host_outside_allowlist():
    """AT-A07: un host fuera de la allowlist se rechaza antes de cualquier request."""
    calls: list[dict] = []
    client = httpx.Client(transport=_transport(calls, payload={"documents": []}))
    with pytest.raises(UnsafeURLError):
        HttpRetriever("https://evil.example.com/q", allowed_hosts=["rag.internal"], http_client=client)
    with pytest.raises(UnsafeURLError):
        HttpRetriever("http://169.254.169.254/latest", http_client=client)
    assert calls == []


@pytest.mark.parametrize(
    ("status", "payload"),
    [(500, {"documents": []}), (200, None), (200, {"docs": []}), (200, {"documents": [1]})],
)
def test_http_retriever_errors(status, payload):
    client = httpx.Client(transport=_transport([], status=status, payload=payload))
    r = HttpRetriever("http://rag.internal/q", allowed_hosts=["rag.internal"], http_client=client)
    with pytest.raises(RetrieverError):
        r.retrieve("q")


def test_http_retriever_limits_size():
    payload = {"documents": ["x" * 5000, "y" * 5000, "z"]}
    client = httpx.Client(transport=_transport([], payload=payload))
    r = HttpRetriever(
        "http://rag.internal/q", allowed_hosts=["rag.internal"], http_client=client, max_chars=6000
    )
    docs = r.retrieve("q")
    assert sum(len(d) for d in docs) == 6000


def test_flare_query_masks_doubtful_spans():
    answer = "Belgrano nació en 1820 en Rosario."
    spans = [Span(start=18, end=22, token_start=3, token_end=4, score=0.9, text="1820")]
    assert mask_spans(answer, spans) == "Belgrano nació en ___ en Rosario."
    q = build_rag_query("¿Cuándo nació Belgrano?", answer, spans)
    assert q == "¿Cuándo nació Belgrano?\nBelgrano nació en ___ en Rosario."
    msgs = inject_context([{"role": "user", "content": "hola"}], ["doc uno"])
    assert msgs[0]["role"] == "system" and "[1] doc uno" in msgs[0]["content"]
    assert msgs[1] == {"role": "user", "content": "hola"}
