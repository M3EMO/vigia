"""Runtime de la plataforma: configuración, detector y política por tenant, trazas y playground."""

import json

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from vigia_platform_support import (
    PEPPERS,
    add_user,
    constant_detector,
    install_detector,
    make_runtime,
    make_settings,
    shared_backend,
)

from vigia.api import create_api_router
from vigia.contracts import Span, VigiaResult
from vigia.policy import HttpRetriever, PolicyError
from vigia_ee import db as dbmod
from vigia_ee.auth import PepperConfigError, Principal
from vigia_ee.runtime import Runtime, Settings, SettingsError, render_prompt

MESSAGES = [{"role": "user", "content": "Escribile a juan.perez@example.com sobre el turno."}]


@pytest.fixture
def rt(tmp_path):
    runtime = make_runtime(tmp_path)
    yield runtime
    runtime.close()


@pytest.fixture
def org(rt):
    return rt.create_org("Acme")


@pytest.fixture
def owner(rt, org):
    return add_user(rt, org, "owner@acme.com.ar", "owner")


def audit_rows(rt, org_id):
    return rt.db.query(
        "SELECT action, target, detail_json FROM audit_log WHERE org_id = ? ORDER BY seq", (org_id,)
    )


def api_client(rt, principal):
    async def get_context(request: Request):
        return rt.context_for(principal)

    app = FastAPI()
    app.include_router(create_api_router(get_context, max_tokens_limit=64, default_max_tokens=8))
    return TestClient(app)


def chat(client, **extra):
    body = {"model": "tiny:mixtral", "messages": MESSAGES, "max_tokens": 6, "temperature": 0, **extra}
    resp = client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 200, resp.text
    return resp


# --- Settings -------------------------------------------------------------------------------------


def test_settings_from_env_reads_every_variable(tmp_path):
    token_file = tmp_path / "metrics.txt"
    token_file.write_text("t" * 24 + "\n", encoding="utf-8")
    env = {
        "VIGIA_DATA_DIR": str(tmp_path / "datos"),
        "VIGIA_MODEL": "tiny:gpt_oss",
        "VIGIA_MAX_TOKENS_LIMIT": "512",
        "VIGIA_DEFAULT_MAX_TOKENS": "64",
        "VIGIA_MAX_BODY_BYTES": "2048",
        "VIGIA_MAX_UPLOAD_BYTES": "4096",
        "VIGIA_METRICS_TOKEN_FILE": str(token_file),
        "VIGIA_RAG_ALLOWED_HOSTS": "rag.interno, Docs.Interno ",
        "VIGIA_WEBHOOK_ALLOWED_HOSTS": "hooks.interno",
        "VIGIA_SECURE_COOKIES": "false",
        "VIGIA_BASE_DETECTOR": str(tmp_path / "base.json"),
        "VIGIA_PEPPERS": "k9:" + "q" * 40,
    }
    s = Settings.from_env(env)
    assert s.data_dir == tmp_path / "datos" and s.db_path == tmp_path / "datos" / "vigia.db"
    assert s.model == "tiny:gpt_oss" and s.backend == "hf"
    assert (s.max_tokens_limit, s.default_max_tokens, s.max_body_bytes, s.max_upload_bytes) == (
        512,
        64,
        2048,
        4096,
    )
    assert s.metrics_token == "t" * 24
    assert s.rag_allowed_hosts == ("rag.interno", "docs.interno")
    assert s.webhook_allowed_hosts == ("hooks.interno",)
    assert s.secure_cookies is False
    assert s.base_detector_path == tmp_path / "base.json"
    assert s.peppers.current_kid == "k9"
    assert "t" * 24 not in repr(s), "los secretos no salen en el repr"


@pytest.mark.parametrize(
    ("env", "fragment"),
    [
        ({}, "VIGIA_MODEL"),
        ({"VIGIA_MODEL": "tiny:llama"}, "diminuto"),
        ({"VIGIA_MODEL": "x", "VIGIA_BACKEND": "tgi"}, "VIGIA_BACKEND"),
        ({"VIGIA_MODEL": "x", "VIGIA_BACKEND": "vllm"}, "VIGIA_VLLM_URL"),
        ({"VIGIA_MODEL": "x", "VIGIA_MAX_TOKENS_LIMIT": "mucho"}, "entero"),
        ({"VIGIA_MODEL": "x", "VIGIA_DEFAULT_MAX_TOKENS": "2048"}, "default_max_tokens"),
        ({"VIGIA_MODEL": "x", "VIGIA_SECURE_COOKIES": "quizas"}, "true o false"),
        ({"VIGIA_MODEL": "x", "VIGIA_METRICS_TOKEN": "corto"}, "16 caracteres"),
        ({"VIGIA_MODEL": "x", "VIGIA_METRICS_TOKEN_FILE": "/no/existe"}, "_FILE"),
    ],
)
def test_settings_reject_invalid_configuration(env, fragment):
    with pytest.raises(SettingsError) as exc:
        Settings.from_env(env)
    assert fragment in str(exc.value)


def test_runtime_requires_peppers(tmp_path, monkeypatch):
    monkeypatch.delenv("VIGIA_PEPPERS", raising=False)
    monkeypatch.delenv("VIGIA_PEPPERS_FILE", raising=False)
    with pytest.raises(PepperConfigError):
        Runtime(Settings(data_dir=tmp_path, model="tiny:mixtral"), backend=shared_backend())


# --- Detector por tenant --------------------------------------------------------------------------


def test_detector_for_caches_by_active_id_and_rejects_other_models(rt, org):
    assert rt.detector_for(org) is None
    first = constant_detector(rt.model_info, 0.6)
    install_detector(rt, org, first)
    cached = rt.detector_for(org)
    assert cached.id == first.id and rt.detector_for(org) is cached
    second = constant_detector(rt.model_info, 0.7)
    repo = rt.repo(org)
    repo.insert_detector(second.id, second.to_json())
    repo.activate_detector(second.id)
    assert rt.detector_for(org).id == second.id, "otro id activo invalida la cache sin tocarla a mano"
    alien = constant_detector(rt.model_info, model={**rt.model_info.fingerprint(), "n_experts": 64})
    install_detector(rt, org, alien)
    assert rt.detector_for(org) is None
    assert "otro modelo" in rt.detector_alert(org)
    assert f'detector_id="{second.id}"' not in rt.metrics.render()
    repo.activate_detector(first.id)
    assert rt.detector_for(org).id == first.id and rt.detector_alert(org) is None
    assert f'vigia_calibration_info{{detector_id="{first.id}"}} 1' in rt.metrics.render()
    assert rt.detector_for("org-inexistente-000") is None


def test_base_detector_is_installed_for_new_orgs(tmp_path):
    info = shared_backend().model_info()
    base = constant_detector(info, 0.4)
    path = tmp_path / "base.json"
    base.save(path)
    rt = make_runtime(tmp_path / "rt", base_detector_path=path)
    try:
        org = rt.create_org("Nueva")
        assert rt.detector_for(org).id == base.id
        rows = audit_rows(rt, org)
        assert [r["action"] for r in rows] == ["org_create", "detector_activate"]
        assert json.loads(rows[1]["detail_json"]) == {"source": "base"}
        legacy = dbmod.create_org(rt.db, "Creada por fuera")  # sin pasar por Runtime.create_org
        assert rt.detector_for(legacy).id == base.id
        with_candidate = dbmod.create_org(rt.db, "Con candidato")
        own = constant_detector(info, 0.55)
        rt.repo(with_candidate).insert_detector(own.id, own.to_json())
        assert rt.detector_for(with_candidate) is None, "una org con detectores no recibe el base"
    finally:
        rt.close()
    other = constant_detector(info, model={**info.fingerprint(), "model_id": "otro"})
    other.save(path)
    rt2 = make_runtime(tmp_path / "rt2", base_detector_path=path)
    try:
        assert rt2.base_detector is None
        assert rt2.detector_for(rt2.create_org("Sin base")) is None
    finally:
        rt2.close()


# --- Política -------------------------------------------------------------------------------------


def test_save_policy_merges_validates_versions_and_audits(rt, org, owner):
    assert rt.policy_for(org).version == 1
    with pytest.raises(PermissionError):
        rt.save_policy(add_user(rt, org, "rev@acme.com.ar", "reviewer"), {"alpha": 0.2})
    saved = rt.save_policy(owner, {"alpha": 0.2, "human_review_at": 0.7, "version": 99})
    assert (saved.alpha, saved.human_review_at, saved.version) == (0.2, 0.7, 2)
    again = rt.save_policy(owner, {"abstain_at": 0.95})
    assert (again.alpha, again.abstain_at, again.version) == (0.2, 0.95, 3), (
        "un cambio parcial conserva el resto"
    )
    assert rt.policy_for(org) == again
    change = [r for r in audit_rows(rt, org) if r["action"] == "policy_change"]
    assert json.loads(change[0]["detail_json"]) == {"changed": ["alpha", "human_review_at"], "version": 2}
    for bad in ({"alpha": 1.5}, {"rag_retry_at": 0.5}, {"campo": 1}, {"rag_url": "http://169.254.169.254/x"}):
        with pytest.raises(PolicyError):
            rt.save_policy(owner, bad)
    install_detector(rt, org, constant_detector(rt.model_info, alphas=(0.15, 0.3), taus=(0.4, 0.5)))
    with pytest.raises(PolicyError, match="mínimo calibrado"):
        rt.save_policy(owner, {"alpha": 0.1})
    assert rt.policy_for(org).version == 3


def test_invalid_stored_policy_falls_back_to_safe_defaults(rt, org):
    rt.repo(org).set_policy({"alpha": 7, "on_detector_error": "fail_open"})
    policy = rt.policy_for(org)
    assert policy.alpha == 0.1 and policy.on_detector_error == "fail_closed"
    assert "inválida" in rt.policy_alert(org)


# --- Contexto, trazas y revisión ------------------------------------------------------------------


def test_context_for_stores_redacted_trace_and_isolates_lookup(rt, org, owner):
    events = []
    rt.emit_event = lambda o, e, d: events.append((o, e)) or []
    resp = chat(api_client(rt, owner), vigia={})
    trace_id = resp.headers["X-Vigia-Trace-Id"]
    vigia = resp.json()["choices"][0]["vigia"]
    assert vigia["calibrated"] is False and vigia["action"] == "pass"
    stored = rt.repo(org).get_trace(trace_id)
    assert stored["source"] == "api" and stored["n_tokens"] == 6
    assert "juan.perez@example.com" not in stored["prompt_text"] and "[EMAIL]" in stored["prompt_text"]
    assert stored["prompt_text"].startswith("user: ")
    view = rt.trace_view(org, trace_id)
    assert view["id"] == trace_id and len(view["tokens"]) == 6 and "prompt_text" not in view
    other_org = rt.create_org("Otra")
    assert rt.trace_view(other_org, trace_id) is None
    assert rt.context_for(Principal(org_id=other_org, role="owner")).trace_lookup(trace_id) is None
    assert events == [], "una respuesta que pasa no emite trace.flagged"


def test_human_review_creates_span_items_emits_webhook_and_metrics(rt, org, owner):
    install_detector(rt, org, constant_detector(rt.model_info, 0.9))
    events = []
    rt.emit_event = lambda o, e, d: events.append((o, e, d)) or []
    resp = chat(api_client(rt, owner), vigia={})
    vigia = resp.json()["choices"][0]["vigia"]
    assert vigia["action"] == "human_review" and vigia["spans"]
    trace_id = vigia["trace_id"]
    items = rt.repo(org).list_review_items()
    assert [i["trace_id"] for i in items] == [trace_id] * len(vigia["spans"])
    assert sorted(i["span_idx"] for i in items) == list(range(len(vigia["spans"])))
    assert all(i["min_reviewers"] == 2 for i in items)
    assert len(events) == 1
    org_id, event, data = events[0]
    assert (org_id, event) == (org, "trace.flagged")
    assert data == {
        "trace_id": trace_id,
        "action": "human_review",
        "response_p_error": vigia["response_p_error"],
        "calibration_id": vigia["calibration_id"],
        "source": "api",
    }
    text = rt.metrics.render()
    assert 'vigia_actions_total{action="human_review"} 1' in text
    assert "vigia_tokens_scored_total 6" in text and "vigia_low_confidence_tokens_total 6" in text
    assert "vigia_review_queue_depth " + str(len(items)) in text
    usage = json.loads(rt.repo(org).get_trace(trace_id)["usage_json"])
    assert usage["layers"] == [0, 1] and sum(map(sum, usage["counts"])) == 6 * 2 * 2


def test_whole_response_review_when_no_span_reaches_tau(rt, org):
    result = VigiaResult(
        trace_id="t" * 16,
        tokens=[],
        spans=[Span(0, 3, 0, 1, 0.1)],
        response_p_error=0.9,
        action="human_review",
        calibration_id=None,
        calibrated=True,
        mode="full",
        tau=0.5,
        alpha=0.1,
    )
    repo = rt.repo(org)
    trace_id = repo.insert_trace(model_id="tiny:mixtral", response_text="abc", action="human_review")
    rt._enqueue_review(repo, trace_id, result)
    assert [(i["span_idx"], i["min_reviewers"]) for i in repo.list_review_items()] == [(None, 2)]


def test_rag_retriever_follows_policy_and_allowlist(tmp_path):
    rt = make_runtime(tmp_path, rag_allowed_hosts=("rag.interno",))
    try:
        org = rt.create_org("Acme")
        owner = add_user(rt, org, "o@acme.com.ar", "owner")
        assert rt.context_for(owner).retriever is None
        rt.save_policy(owner, {"rag_url": "http://rag.interno/buscar", "rag_retry_at": 0.8})
        retriever = rt.context_for(owner).retriever
        assert isinstance(retriever, HttpRetriever) and rt.context_for(owner).retriever is retriever
    finally:
        rt.close()


# --- Playground -----------------------------------------------------------------------------------


def test_playground_streams_tokens_and_stores_trace(rt, org, owner):
    with pytest.raises(PermissionError):
        rt.playground_stream(add_user(rt, org, "v@acme.com.ar", "viewer"), MESSAGES, max_tokens=4)
    events = list(rt.playground_stream(owner, MESSAGES, max_tokens=5))
    tokens, done = events[:-1], events[-1]
    assert [e["type"] for e in tokens] == ["token"] * 5 and done["type"] == "done"
    assert all(e["p_error"] is None and e["flag"] is False for e in tokens)
    assert done["text"] == "".join(e["text"] for e in tokens)
    assert done["calibrated"] is False and done["action"] == "pass"
    stored = rt.repo(org).get_trace(done["trace_id"])
    assert stored["source"] == "playground" and stored["n_tokens"] == 5


def test_playground_with_detector_flags_and_queues_review(rt, org, owner):
    install_detector(rt, org, constant_detector(rt.model_info, 0.9))
    events = list(rt.playground_stream(owner, MESSAGES, max_tokens=4))
    done = events[-1]
    assert all(e["flag"] and e["p_error"] == pytest.approx(0.9) for e in events[:-1])
    assert done["calibrated"] and done["action"] == "human_review" and done["spans"]
    assert done["response_p_error"] == pytest.approx(0.9)
    assert rt.repo(org).list_review_items()[0]["trace_id"] == done["trace_id"]


@pytest.mark.parametrize(
    ("messages", "max_tokens", "temperature", "fragment"),
    [
        ([], 4, 0.0, "entre 1 y"),
        ([{"role": "tool", "content": "x"}], 4, 0.0, "rol"),
        ([{"role": "user", "content": "   "}], 4, 0.0, "vacío"),
        (MESSAGES, 0, 0.0, "max_tokens"),
        (MESSAGES, 10_000, 0.0, "max_tokens"),
        (MESSAGES, 4, 3.0, "temperatura"),
        ([{"role": "user", "content": "x" * 40_000}], 4, 0.0, "supera"),
    ],
)
def test_playground_rejects_invalid_input_with_error_event(
    rt, owner, messages, max_tokens, temperature, fragment
):
    events = list(rt.playground_stream(owner, messages, max_tokens=max_tokens, temperature=temperature))
    assert len(events) == 1 and events[0]["type"] == "error" and fragment in events[0]["message"]


class BrokenScorer:
    id = "0123456789abcdef"
    thresholds = [{"alpha": 0.05, "tau": 0.5, "expected_flag_rate": 0.1, "expected_precision": None}]

    def tau(self, alpha):
        raise RuntimeError("detector roto")


def test_playground_follows_on_detector_error(rt, org, owner, monkeypatch):
    monkeypatch.setattr(rt, "detector_for", lambda org_id: BrokenScorer())
    events = list(rt.playground_stream(owner, MESSAGES, max_tokens=3))
    assert events == [
        {
            "type": "error",
            "message": "El detector de confianza no está disponible y la política exige fail_closed.",
        }
    ]
    rt.save_policy(owner, {"on_detector_error": "fail_open"})
    events = list(rt.playground_stream(owner, MESSAGES, max_tokens=3))
    done = events[-1]
    assert done["type"] == "done" and done["calibrated"] is False and done["response_p_error"] is None
    assert rt.repo(org).get_trace(done["trace_id"])["degraded"] == 1


def test_playground_internal_error_is_generic(rt, owner, monkeypatch):
    def broken_stream(*args, **kwargs):
        raise RuntimeError("ruta interna /srv/modelo")

    monkeypatch.setattr(rt.backend, "stream", broken_stream)
    events = list(rt.playground_stream(owner, MESSAGES, max_tokens=3))
    assert events == [{"type": "error", "message": "Error interno al generar la respuesta."}]


# --- Salud y ciclo de vida ------------------------------------------------------------------------


def test_readiness_checks_db_worker_and_queue(tmp_path):
    rt = make_runtime(tmp_path, max_queue_depth=2)
    try:
        assert rt.readiness() == {"db": True, "worker": False, "queue": True}
        rt.jobs.start()
        assert rt.readiness()["worker"] is True
        rt.jobs.stop()
        for _ in range(2):
            rt.jobs.enqueue("sin_handler", {})
        assert rt.readiness()["queue"] is False
    finally:
        rt.close()
        rt.close()  # idempotente


def test_render_prompt_is_readable_transcript():
    msgs = [{"role": "system", "content": "Sé breve."}, {"role": "user", "content": "Hola\nchau"}]
    assert render_prompt(msgs) == "system: Sé breve.\n\nuser: Hola\nchau"


def test_make_settings_uses_isolated_peppers(tmp_path):
    assert make_settings(tmp_path).peppers is PEPPERS
