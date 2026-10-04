"""Tests de vigia_ee.metrics: formato Prometheus 0.0.4, cardinalidad acotada y middleware ASGI."""

from __future__ import annotations

import asyncio
import re
import threading

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vigia_ee.metrics import (
    CONTENT_TYPE,
    DURATION_BUCKETS,
    MAX_ROUTES,
    Metrics,
    MetricsMiddleware,
    escape_label_value,
    format_value,
    status_class,
)

SAMPLE_RE = re.compile(
    r'^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>(?:[a-zA-Z_]\w*="(?:[^"\\\n]|\\.)*",?)*)\})? '
    r"(?P<value>[-+]?(?:\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|Inf|NaN))$"
)
LABEL_RE = re.compile(r'([a-zA-Z_]\w*)="((?:[^"\\\n]|\\.)*)"')
ALLOWED_LABELS = {"route", "status", "le", "action", "detector_id", "version"}
FAMILIES = (
    ("vigia_requests_total", "counter"),
    ("vigia_request_duration_seconds", "histogram"),
    ("vigia_overhead_ratio", "histogram"),
    ("vigia_tokens_scored_total", "counter"),
    ("vigia_low_confidence_tokens_total", "counter"),
    ("vigia_actions_total", "counter"),
    ("vigia_calibration_info", "gauge"),
    ("vigia_auth_failures_total", "counter"),
    ("vigia_rate_limited_total", "counter"),
    ("vigia_review_queue_depth", "gauge"),
    ("vigia_jobs_queue_depth", "gauge"),
    ("vigia_build_info", "gauge"),
)


def parse(text: str) -> list[tuple[str, dict[str, str], str]]:
    """Parser estricto del formato de texto: falla ante cualquier línea mal formada."""
    assert text.endswith("\n")
    samples = []
    for line in text.splitlines():
        if line.startswith("# HELP ") or line.startswith("# TYPE "):
            continue
        m = SAMPLE_RE.match(line)
        assert m, f"línea inválida: {line!r}"
        labels = dict(LABEL_RE.findall(m.group("labels") or ""))
        samples.append((m.group("name"), labels, m.group("value")))
    return samples


def value_of(samples, name, **labels):
    for n, lab, v in samples:
        if n == name and all(lab.get(k) == val for k, val in labels.items()):
            return v
    raise KeyError((name, labels))


def test_at_p14_render_exposes_every_family_with_help_and_type():
    text = Metrics(version="1.2.3").render()
    for name, kind in FAMILIES:
        assert f"# TYPE {name} {kind}\n" in text
        assert f"# HELP {name} " in text
    samples = parse(text)
    assert value_of(samples, "vigia_build_info", version="1.2.3") == "1"
    assert CONTENT_TYPE == "text/plain; version=0.0.4; charset=utf-8"


def test_counters_gauges_and_histograms_accumulate():
    m = Metrics(version="t")
    m.observe_request("/v1/chat/completions", 200, 0.03)
    m.observe_request("/v1/chat/completions", 201, 0.2)
    m.observe_request("/v1/chat/completions", 503, 100.0)
    m.observe_overhead(0.04)
    m.observe_overhead(float("nan"))  # se ignora
    m.add_tokens(10, 3)
    m.inc_action("annotate")
    m.inc_action("annotate")
    m.inc_auth_failure()
    m.inc_rate_limited()
    m.set_review_queue_depth(4)
    m.set_jobs_queue_depth(-1)
    s = parse(m.render())
    route = "/v1/chat/completions"
    assert value_of(s, "vigia_requests_total", route=route, status="2xx") == "2"
    assert value_of(s, "vigia_requests_total", route=route, status="5xx") == "1"
    assert value_of(s, "vigia_request_duration_seconds_bucket", route=route, le="0.05") == "1"
    assert value_of(s, "vigia_request_duration_seconds_bucket", route=route, le="0.25") == "2"
    assert value_of(s, "vigia_request_duration_seconds_bucket", route=route, le="60") == "2"
    assert value_of(s, "vigia_request_duration_seconds_bucket", route=route, le="+Inf") == "3"
    assert value_of(s, "vigia_request_duration_seconds_count", route=route) == "3"
    assert float(value_of(s, "vigia_request_duration_seconds_sum", route=route)) == pytest.approx(100.23)
    assert value_of(s, "vigia_overhead_ratio_count") == "1"
    assert value_of(s, "vigia_overhead_ratio_bucket", le="0.05") == "1"
    assert value_of(s, "vigia_tokens_scored_total") == "10"
    assert value_of(s, "vigia_low_confidence_tokens_total") == "3"
    assert value_of(s, "vigia_actions_total", action="annotate") == "2"
    assert value_of(s, "vigia_auth_failures_total") == "1"
    assert value_of(s, "vigia_rate_limited_total") == "1"
    assert value_of(s, "vigia_review_queue_depth") == "4"
    assert value_of(s, "vigia_jobs_queue_depth") == "0"
    buckets = [n for n, lab, _ in s if n == "vigia_request_duration_seconds_bucket" and lab["route"] == route]
    assert len(buckets) == len(DURATION_BUCKETS) + 1
    with pytest.raises(ValueError):
        m.add_tokens(-1)


def test_labels_are_bounded():
    m = Metrics(version="t")
    m.inc_action("acción inventada por el cliente")
    m.set_calibrations(["0123456789abcdef", "org-secreta", "x" * 16, "fedcba9876543210"])
    m.observe_request("/items/abc?token=secreto", 200, 0.01)  # path crudo: no es una plantilla válida
    for i in range(MAX_ROUTES + 5):
        m.observe_request(f"/r{i}", 200, 0.01)
    s = parse(m.render())
    assert value_of(s, "vigia_actions_total", action="other") == "1"
    detectors = sorted(lab["detector_id"] for n, lab, _ in s if n == "vigia_calibration_info")
    assert detectors == ["0123456789abcdef", "fedcba9876543210"]
    routes = {lab["route"] for n, lab, _ in s if n == "vigia_requests_total"}
    assert len(routes) <= MAX_ROUTES + 1 and "other" in routes
    assert not any("token" in r for r in routes)
    m.set_calibrations([])
    assert not any(n == "vigia_calibration_info" for n, _, _ in parse(m.render()))


def test_escaping_of_label_values_and_special_numbers():
    assert escape_label_value('a"b\\c\nd') == 'a\\"b\\\\c\\nd'
    m = Metrics(version='1.0"\n} vigia_falso 1')
    text = m.render()
    s = parse(text)
    assert value_of(s, "vigia_build_info") == "1"
    assert "vigia_falso" not in {n for n, _, _ in s}
    assert format_value(float("inf")) == "+Inf"
    assert format_value(float("-inf")) == "-Inf"
    assert format_value(float("nan")) == "NaN"
    assert format_value(3.0) == "3"
    assert format_value(0.005) == "0.005"
    codes = (200, 302, 404, 500, 99, 700)
    assert [status_class(c) for c in codes] == ["2xx", "3xx", "4xx", "5xx", "5xx", "5xx"]


def test_depth_callables_are_read_on_render_and_failures_keep_last_value():
    calls = {"n": 7}

    def jobs_depth():
        if calls["n"] < 0:
            raise RuntimeError("base caída")
        return calls["n"]

    m = Metrics(version="t", jobs_depth=jobs_depth, review_depth=lambda: 2)
    s = parse(m.render())
    assert value_of(s, "vigia_jobs_queue_depth") == "7"
    assert value_of(s, "vigia_review_queue_depth") == "2"
    calls["n"] = -1
    assert value_of(parse(m.render()), "vigia_jobs_queue_depth") == "7"


def test_thread_safety_under_concurrent_updates():
    m = Metrics(version="t")

    def work():
        for _ in range(500):
            m.observe_request("/v1/models", 200, 0.001)
            m.add_tokens(2, 1)
            m.inc_action("pass")

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    s = parse(m.render())
    assert value_of(s, "vigia_requests_total", route="/v1/models", status="2xx") == "4000"
    assert value_of(s, "vigia_tokens_scored_total") == "8000"
    assert value_of(s, "vigia_actions_total", action="pass") == "4000"


# --- middleware ASGI -------------------------------------------------------------------------------


@pytest.fixture
def app_and_metrics():
    app = FastAPI()
    metrics = Metrics(version="t")

    @app.get("/v1/vigia/traces/{trace_id}")
    def get_trace(trace_id: str):
        return {"id": trace_id}

    @app.get("/boom")
    def boom():
        raise RuntimeError("interno")

    app.add_middleware(MetricsMiddleware, metrics=metrics)
    return app, metrics


def test_at_p14_middleware_uses_route_template_and_never_raw_ids(app_and_metrics):
    app, metrics = app_and_metrics
    client = TestClient(app, raise_server_exceptions=False)
    org_like = "org_5f2a9c1e"
    for tid in ("tr_aaaaaaaa", "tr_bbbbbbbb", org_like):
        assert client.get(f"/v1/vigia/traces/{tid}").status_code == 200
    assert client.get("/no/existe/tr_cccccccc").status_code == 404
    assert client.get("/boom").status_code == 500
    text = metrics.render()
    s = parse(text)
    assert value_of(s, "vigia_requests_total", route="/v1/vigia/traces/{trace_id}", status="2xx") == "3"
    assert value_of(s, "vigia_requests_total", route="unmatched", status="4xx") == "1"
    assert value_of(s, "vigia_requests_total", route="/boom", status="5xx") == "1"
    for raw in ("tr_aaaaaaaa", "tr_bbbbbbbb", "tr_cccccccc", org_like):
        assert raw not in text
    label_names = {k for _, lab, _ in s for k in lab}
    assert label_names <= ALLOWED_LABELS
    assert not label_names & {"org_id", "trace_id", "user", "email", "request_id"}


def test_middleware_passes_through_non_http_scopes():
    seen = []

    async def inner(scope, receive, send):
        seen.append(scope["type"])

    mw = MetricsMiddleware(inner, Metrics(version="t"))
    asyncio.run(mw({"type": "lifespan"}, None, None))
    assert seen == ["lifespan"]
