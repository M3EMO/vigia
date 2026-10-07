"""Runtime de la plataforma: configuración, detector y política por tenant, trazas y playground."""

import base64
import json

import httpx
import numpy as np
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

from vigia.api import TraceRecord, create_api_router
from vigia.contracts import Generation, Span, VigiaResult
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
        ({"VIGIA_MODEL": "x", "VIGIA_DEFAULT_MAX_TOKENS": "2048"}, "VIGIA_DEFAULT_MAX_TOKENS"),
        ({"VIGIA_MODEL": "x", "VIGIA_SECURE_COOKIES": "quizas"}, "true o false"),
        ({"VIGIA_MODEL": "x", "VIGIA_METRICS_TOKEN": "corto"}, "16 caracteres"),
        ({"VIGIA_MODEL": "x", "VIGIA_METRICS_TOKEN_FILE": "/no/existe"}, "_FILE"),
    ],
)
def test_settings_reject_invalid_configuration(env, fragment):
    with pytest.raises(SettingsError) as exc:
        Settings.from_env(env)
    assert fragment in str(exc.value)


VLLM_ENV = {
    "VIGIA_MODEL": "x",
    "VIGIA_BACKEND": "vllm",
    "VIGIA_VLLM_URL": "http://127.0.0.1:8118/v1",
    "VIGIA_VLLM_MODEL": "m",
    "VIGIA_VLLM_N_EXPERTS": "8",
    "VIGIA_VLLM_TOP_K": "2",
    "VIGIA_VLLM_N_LAYERS": "2",
}


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({"VIGIA_SHUTDOWN_TIMEOUT": "0"}, "VIGIA_SHUTDOWN_TIMEOUT debe ser un entero entre 1 y 3600."),
        ({"VIGIA_KEY_RATE_PER_MINUTE": "-1"}, "VIGIA_KEY_RATE_PER_MINUTE debe ser un entero entre"),
        ({"VIGIA_TENANT_RATE_PER_MINUTE": "0"}, "VIGIA_TENANT_RATE_PER_MINUTE debe ser un entero entre"),
        ({"VIGIA_MAX_QUEUE_DEPTH": "0"}, "VIGIA_MAX_QUEUE_DEPTH debe ser un entero entre"),
        ({"VIGIA_REVIEW_MIN_REVIEWERS": "0"}, "VIGIA_REVIEW_MIN_REVIEWERS debe ser un entero entre"),
        ({"VIGIA_MAX_TOKENS_LIMIT": "0"}, "VIGIA_MAX_TOKENS_LIMIT debe ser un entero entre"),
        ({"VIGIA_MAX_BODY_BYTES": "1"}, "VIGIA_MAX_BODY_BYTES debe ser un entero entre"),
        ({"VIGIA_MAX_UPLOAD_BYTES": "1"}, "VIGIA_MAX_UPLOAD_BYTES debe ser un entero entre"),
        (
            {"VIGIA_DEFAULT_MAX_TOKENS": "2000", "VIGIA_MAX_TOKENS_LIMIT": "1000"},
            "VIGIA_DEFAULT_MAX_TOKENS no puede superar VIGIA_MAX_TOKENS_LIMIT.",
        ),
        ({**VLLM_ENV, "VIGIA_VLLM_URL": "notaurl"}, "VIGIA_VLLM_URL debe ser una URL http(s) con host."),
        ({**VLLM_ENV, "VIGIA_VLLM_N_EXPERTS": "1"}, "VIGIA_VLLM_N_EXPERTS debe ser un entero entre"),
        ({**VLLM_ENV, "VIGIA_VLLM_TOP_K": "9"}, "VIGIA_VLLM_TOP_K no puede superar VIGIA_VLLM_N_EXPERTS."),
    ],
)
def test_settings_range_errors_name_the_env_var(env, expected):
    """Regresión: 'Los errores de rango de configuración nombran el campo interno y no la variable
    de entorno (shutdown_timeout_s en lugar de VIGIA_SHUTDOWN_TIMEOUT)'."""
    with pytest.raises(SettingsError) as exc:
        Settings.from_env({"VIGIA_MODEL": "x", **env})
    assert expected in str(exc.value)
    for field_name in ("shutdown_timeout_s", "key_rate_per_minute", "default_max_tokens", "base_url"):
        assert field_name not in str(exc.value)


@pytest.mark.parametrize("missing", ["VIGIA_VLLM_N_EXPERTS", "VIGIA_VLLM_TOP_K", "VIGIA_VLLM_N_LAYERS"])
def test_settings_vllm_requires_served_model_shape(missing):
    """Regresión: 'Modo vLLM de la plataforma: sin forma de configurar n_experts, las features U se
    normalizan con el máximo ID de cada respuesta y el p_error de un token depende de los tokens
    futuros'. Sin la forma del modelo servido el arranque falla y nombra la variable."""
    env = {k: v for k, v in VLLM_ENV.items() if k != missing}
    with pytest.raises(SettingsError) as exc:
        Settings.from_env(env)
    assert missing in str(exc.value)


def _routed_vllm_client() -> httpx.Client:
    """vLLM falso: `routed_experts[i][l] = [(i+l+k) % 8 for k in 0..1]`, largo según max_tokens."""

    def handler(request: httpx.Request) -> httpx.Response:
        n = json.loads(request.content)["max_tokens"]
        routed = np.array(
            [[[(i + layer + k) % 8 for k in range(2)] for layer in range(2)] for i in range(n)], dtype="<u1"
        )
        content = [
            {
                "token": f"t{i}",
                "logprob": -0.5,
                "bytes": list(f"t{i}".encode()),
                "top_logprobs": [{"token": f"t{i}", "logprob": -0.5}, {"token": "z", "logprob": -1.5}],
            }
            for i in range(n)
        ]
        choice = {
            "index": 0,
            "message": {"role": "assistant", "content": "".join(c["token"] for c in content)},
            "finish_reason": "length",
            "logprobs": {"content": content},
            "routed_experts": {
                "dtype": "uint8",
                "shape": list(routed.shape),
                "data": base64.b64encode(routed.tobytes()).decode(),
            },
        }
        return httpx.Response(200, json={"choices": [choice], "usage": {"prompt_tokens": 3}})

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_vllm_backend_uses_configured_shape_and_u_features_are_causal():
    """Regresión: 'Modo vLLM de la plataforma: sin forma de configurar n_experts, las features U se
    normalizan con el máximo ID de cada respuesta y el p_error de un token depende de los tokens
    futuros'. Con VIGIA_VLLM_N_EXPERTS fijo, U del token 0 no depende del largo de la respuesta."""
    from vigia.contracts import Tier
    from vigia_ee.runtime import build_backend

    backend = build_backend(Settings.from_env(VLLM_ENV))
    info = backend.model_info()
    assert (info.n_experts, info.top_k, info.n_layers) == (8, 2, 2)
    assert not any("n_experts" in d for d in backend.capabilities().disabled)
    backend._client = _routed_vllm_client()  # el cliente HTTP propio apunta al vLLM falso
    messages = [{"role": "user", "content": "hola"}]
    first = {}
    for max_tokens in (6, 3):
        events = list(backend.stream(messages, max_tokens=max_tokens, tiers=(Tier.F0, Tier.U)))
        first[max_tokens] = {k: v for k, v in events[0].features.items() if "u_" in k}
    assert first[6] and first[6] == first[3]


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


@pytest.mark.parametrize("case", ["missing", "broken_json", "bad_hash"])
def test_unreadable_base_detector_is_a_settings_error(tmp_path, case):
    """Regresión: 'VIGIA_BASE_DETECTOR inexistente o ilegible no frena el arranque y /startupz dice
    detectors:true'. Un archivo que falta o no es un detector es un error de configuración."""
    path = tmp_path / "base.json"
    if case == "broken_json":
        path.write_text('{"roto": ', encoding="utf-8")
    elif case == "bad_hash":
        artifact = json.loads(constant_detector(shared_backend().model_info()).to_json())
        artifact["bias"] = 0.123  # el hash ya no valida
        path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(SettingsError) as exc:
        make_runtime(tmp_path / "rt", base_detector_path=path)
    message = str(exc.value)
    assert f"VIGIA_BASE_DETECTOR={path}" in message and len(message.splitlines()) == 1
    assert ("No existe" in message) if case == "missing" else ("inválido" in message)


@pytest.mark.parametrize("kind", ["empty_dir", "missing_path"])
def test_unloadable_model_is_a_settings_error(tmp_path, kind):
    """Regresión: 'VIGIA_MODEL inválido (ruta o repo inexistente) aborta el arranque con traceback
    crudo y sin nombrar la variable'."""
    from vigia_ee.runtime import build_backend

    model = tmp_path / "modelo"
    if kind == "empty_dir":
        model.mkdir()
    with pytest.raises(SettingsError) as exc:
        build_backend(make_settings(tmp_path, model=str(model)))
    message = str(exc.value)
    assert message.startswith(f"No se puede cargar el modelo (VIGIA_MODEL={model}): ")
    assert len(message.splitlines()) == 1 and len(message) < 500


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
    assert json.loads(change[0]["detail_json"]) == {
        "changed": ["alpha", "human_review_at"],
        "version": 2,
        "alpha_source": "policy",  # el alpha pasó de seguir al detector a quedar fijado
    }
    for bad in ({"alpha": 1.5}, {"rag_retry_at": 0.5}, {"campo": 1}, {"rag_url": "http://169.254.169.254/x"}):
        with pytest.raises(PolicyError):
            rt.save_policy(owner, bad)
    install_detector(rt, org, constant_detector(rt.model_info, alphas=(0.15, 0.3), taus=(0.4, 0.5)))
    with pytest.raises(PolicyError, match="mínimo calibrado"):
        rt.save_policy(owner, {"alpha": 0.1})
    assert rt.policy_for(org).version == 3


def test_effective_alpha_is_policy_if_fixed_else_detector_default(rt, org, owner):
    """alpha efectivo: el de la política si el owner lo fijó; si no, default_alpha del detector activo
    (el recomendado en calibración); sin detector, Policy.alpha = 0,1."""
    assert rt.policy_for(org).alpha == 0.1 and not rt.policy_alpha_explicit(org)
    install_detector(rt, org, constant_detector(rt.model_info, alphas=(0.15, 0.3), taus=(0.4, 0.5)))
    assert rt.policy_for(org).alpha == 0.15 and rt.context_for(owner).policy.alpha == 0.15
    # Cambiar otro campo no fija alpha: la política guardada no lleva la clave.
    rt.save_policy(owner, {"gap_max": 2})
    assert "alpha" not in rt.repo(org).get_policy() and not rt.policy_alpha_explicit(org)
    # Recalibrar mueve el alpha de una política que no lo fijó.
    install_detector(rt, org, constant_detector(rt.model_info, alphas=(0.2, 0.4), taus=(0.4, 0.5)))
    assert rt.policy_for(org).alpha == 0.2 and rt.policy_for(org).gap_max == 2
    # Fijado por el owner: no se mueve al recalibrar.
    rt.save_policy(owner, {"alpha": 0.3})
    assert rt.policy_alpha_explicit(org) and rt.repo(org).get_policy()["alpha"] == 0.3
    install_detector(rt, org, constant_detector(rt.model_info, alphas=(0.25, 0.4), taus=(0.4, 0.5)))
    assert rt.policy_for(org).alpha == 0.3
    # alpha None lo suelta: vuelve a seguir al detector.
    released = rt.save_policy(owner, {"alpha": None})
    assert released.alpha == 0.25 and not rt.policy_alpha_explicit(org)
    details = [json.loads(r["detail_json"]) for r in audit_rows(rt, org) if r["action"] == "policy_change"]
    assert [d.get("alpha_source") for d in details] == [None, "policy", "detector"]


def test_activation_checks_only_a_fixed_alpha(rt, org, owner):
    """Un alpha que sigue al detector no bloquea activar uno calibrado desde un alpha mayor."""
    high = constant_detector(rt.model_info, alphas=(0.2, 0.4), taus=(0.4, 0.5))
    rt.repo(org).insert_detector(high.id, high.to_json())
    rt.calibration.activate(owner, high.id)
    assert rt.policy_for(org).alpha == 0.2


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
    assert view["id"] == trace_id and len(view["vigia"]["tokens"]) == 6 and "prompt_text" not in view
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


def test_trace_view_hides_text_tokens_and_spans_on_abstain(rt, org, owner):
    """Regresión: 'GET /v1/vigia/traces/{id} de la plataforma devuelve el texto original y los
    tokens de una respuesta en abstain'."""
    install_detector(rt, org, constant_detector(rt.model_info, 0.9))
    rt.save_policy(owner, {"abstain_at": 0.5, "abstain_message": "NO SE"})
    rt.emit_event = lambda o, e, d: []
    resp = chat(api_client(rt, owner), vigia={})
    choice = resp.json()["choices"][0]
    assert choice["vigia"]["action"] == "abstain" and choice["message"]["content"] == "NO SE"
    trace_id = resp.headers["X-Vigia-Trace-Id"]
    stored = rt.repo(org).get_trace(trace_id)
    assert stored["response_text"] and stored["tokens"], "el panel conserva la traza completa"
    view = rt.trace_view(org, trace_id)
    assert view["vigia"]["action"] == "abstain"
    assert view["response"] is None and view["vigia"]["tokens"] == [] and view["vigia"]["spans"] == []
    via_api = api_client(rt, owner).get(f"/v1/vigia/traces/{trace_id}")
    assert via_api.status_code == 200
    body = via_api.json()
    assert body["response"] is None and body["vigia"]["tokens"] == [] and body["vigia"]["spans"] == []


def test_trace_view_has_the_core_shape(rt, org, owner):
    """Regresión: 'GET /v1/vigia/traces/{id} de la plataforma pierde rag_status y no tiene la forma
    del núcleo'. La traza por ID tiene la forma de `vigia.serve.trace_view` y el campo `vigia`
    coincide con el de la respuesta, status y rag_status incluidos (migración 3)."""
    from vigia.serve import trace_view as core_trace_view

    install_detector(rt, org, constant_detector(rt.model_info, 0.9))
    rt.emit_event = lambda o, e, d: []
    resp = chat(api_client(rt, owner), vigia={})
    choice = resp.json()["choices"][0]
    trace_id = resp.headers["X-Vigia-Trace-Id"]
    body = api_client(rt, owner).get(f"/v1/vigia/traces/{trace_id}").json()
    record = TraceRecord(
        trace_id=trace_id,
        org_id=org,
        key_id=None,
        created_at="",
        model=rt.model_info,
        prompt_messages=[],
        generation=Generation(text="", tokens=[], finish_reason="stop", prompt_tokens=0, model=rt.model_info),
        result=VigiaResult(**{**choice["vigia"], "tokens": [], "spans": []}),
        latency_ms=0.0,
        overhead_ms=0.0,
    )
    assert list(body) == list(core_trace_view(record))
    assert list(body["vigia"]) == list(choice["vigia"])
    assert body["response"] == choice["message"]["content"] and body["model"] == "tiny:mixtral"
    assert body["vigia"]["spans"] and body["vigia"]["spans"] == choice["vigia"]["spans"]
    assert body["vigia"]["tokens"] == choice["vigia"]["tokens"]
    keys = ("action", "response_p_error", "calibration_id", "calibrated", "mode", "alpha", "tau", "status")
    for key in (*keys, "rag_status"):
        assert body["vigia"][key] == choice["vigia"][key], key


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


@pytest.mark.parametrize("fallback", ["abstain", "human_review"])
def test_playground_rag_retry_uses_policy_fallback(tmp_path, fallback):
    """Regresión: 'El playground ignora policy.rag_fallback_action y siempre usa human_review'."""
    rt = make_runtime(tmp_path, rag_allowed_hosts=("rag.interno",))
    try:
        org = rt.create_org("Acme")
        owner = add_user(rt, org, "o@acme.com.ar", "owner")
        install_detector(rt, org, constant_detector(rt.model_info, 0.9))
        rt.save_policy(
            owner,
            {"rag_retry_at": 0.8, "rag_url": "http://rag.interno/ok", "rag_fallback_action": fallback},
        )
        done = list(rt.playground_stream(owner, MESSAGES, max_tokens=4))[-1]
        assert done["type"] == "done" and done["action"] == fallback
        assert rt.repo(org).get_trace(done["trace_id"])["action"] == fallback
        assert rt.trace_view(org, done["trace_id"])["rag_status"] == "unavailable"
    finally:
        rt.close()


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


def test_readiness_is_bounded_when_the_database_is_locked(tmp_path):
    """Regresión: '/readyz no tiene tope de tiempo: con la base bloqueada no responde hasta que se
    libera el lock y ocupa un hilo del threadpool por cada sonda'. Con la base tomada por otro proceso
    y el RLock en manos del hilo de jobs, /readyz responde 503 con db:false dentro del tope y las
    sondas repetidas reusan el chequeo en curso en vez de colgar un hilo cada una."""
    import sqlite3
    import threading
    import time

    from vigia_ee.app import create_app

    rt = make_runtime(tmp_path)
    holder = sqlite3.connect(rt.db.path, isolation_level=None, timeout=1)
    release = threading.Event()

    def jobs_thread():  # como el worker que espera el busy_timeout con el RLock de Database tomado
        with rt.db._lock:
            release.wait(8)

    hog = threading.Thread(target=jobs_thread, daemon=True)
    try:
        holder.execute("BEGIN EXCLUSIVE")
        hog.start()
        client = TestClient(create_app(runtime=rt))
        for _ in range(3):
            t0 = time.monotonic()
            resp = client.get("/readyz")
            assert time.monotonic() - t0 < 3.5, "la sonda tiene tope aunque la base siga bloqueada"
            assert resp.status_code == 503
            assert resp.json()["checks"]["db"] is False and resp.json()["checks"]["queue"] is False
        probes = [t for t in threading.enumerate() if t.name == "vigia-readyz-db" and t.is_alive()]
        assert len(probes) <= 1, "las sondas siguientes esperan el mismo chequeo"
    finally:
        holder.execute("ROLLBACK")
        holder.close()
        release.set()
        hog.join()
    try:
        deadline = time.monotonic() + 10
        while not rt.readiness()["db"] and time.monotonic() < deadline:
            time.sleep(0.1)
        assert rt.readiness() == {"db": True, "worker": False, "queue": True}, "se recupera solo"
    finally:
        rt.close()


def test_readiness_ignores_the_webhook_lane(tmp_path):
    """Regresión: '/readyz cuenta la cola de webhooks: el backlog de webhooks de un tenant saca el nodo
    para todos' y 'el webhook colgado de un solo tenant saca al nodo de servicio'."""
    from vigia_ee import webhooks
    from vigia_ee.app import create_app
    from vigia_ee.jobs import DEFAULT_LANE

    rt = make_runtime(tmp_path, max_queue_depth=3)
    try:
        org = rt.create_org("A")
        rt.repo(org).insert_webhook("https://hooks-a.example.com/x", "whsec_" + "A" * 43, ["trace.flagged"])
        for i in range(6):
            webhooks.emit(rt.db, rt.jobs, org, "trace.flagged", {"trace_id": f"t{i}"})
        assert rt.jobs.depth() == 6 and rt.jobs.depth(lane=DEFAULT_LANE) == 0
        assert rt.readiness()["queue"] is True
        body = TestClient(create_app(runtime=rt)).get("/readyz").json()  # sin lifespan: sin worker
        assert body["checks"] == {"db": True, "worker": False, "queue": True}
        for _ in range(3):  # el carril por defecto (purga, calibración) sí cuenta
            rt.jobs.enqueue("sin_handler", {})
        assert rt.readiness()["queue"] is False
    finally:
        rt.close()


def test_render_prompt_is_readable_transcript():
    msgs = [{"role": "system", "content": "Sé breve."}, {"role": "user", "content": "Hola\nchau"}]
    assert render_prompt(msgs) == "system: Sé breve.\n\nuser: Hola\nchau"


def test_make_settings_uses_isolated_peppers(tmp_path):
    assert make_settings(tmp_path).peppers is PEPPERS


# --- Integración de la ronda de verificación -------------------------------------------------------


class _FakeRetriever:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def retrieve(self, query: str) -> list[str]:
        self.queries.append(query)
        return ["Documento de apoyo."]


def test_api_trace_persists_rag_status_and_status(tmp_path):
    """Regresión: 'GET /v1/vigia/traces/{id} de la plataforma devuelve rag_status null'. La base
    guarda rag_status y status (migración 3) y la traza por ID coincide con la respuesta."""
    from dataclasses import replace

    rt = make_runtime(tmp_path, rag_allowed_hosts=("rag.interno",))
    try:
        org = rt.create_org("Acme")
        owner = add_user(rt, org, "o@acme.com.ar", "owner")
        install_detector(rt, org, constant_detector(rt.model_info, 0.9))
        rt.save_policy(owner, {"rag_retry_at": 0.8, "rag_url": "http://rag.interno/ok"})
        retriever = _FakeRetriever()

        async def get_context(request: Request):
            return replace(rt.context_for(owner), retriever=retriever)

        app = FastAPI()
        app.include_router(create_api_router(get_context, max_tokens_limit=64, default_max_tokens=8))
        client = TestClient(app)
        resp = chat(client, vigia={})
        vigia = resp.json()["choices"][0]["vigia"]
        assert retriever.queries and vigia["rag_status"] is not None
        trace_id = resp.headers["X-Vigia-Trace-Id"]
        stored = rt.repo(org).get_trace(trace_id)
        assert stored["rag_status"] == vigia["rag_status"] and stored["status"] == vigia["status"]
        body = client.get(f"/v1/vigia/traces/{trace_id}").json()
        assert body["rag_status"] == vigia["rag_status"] == body["vigia"]["rag_status"]
    finally:
        rt.close()


def test_store_trace_persists_a_non_ok_status(rt, org):
    record = TraceRecord(
        trace_id="estado" + "0" * 10,
        org_id=org,
        key_id=None,
        created_at="2026-10-04T12:00:00+00:00",
        model=rt.model_info,
        prompt_messages=MESSAGES,
        generation=Generation(
            text="", tokens=[], finish_reason="length", prompt_tokens=3, model=rt.model_info
        ),
        result=VigiaResult(
            trace_id="estado" + "0" * 10,
            tokens=[],
            spans=[],
            response_p_error=None,
            action="pass",
            calibration_id=None,
            calibrated=False,
            mode="full",
            status="truncated_before_final",
        ),
        latency_ms=1.0,
        overhead_ms=0.0,
    )
    rt.store_trace(org, record)
    assert rt.trace_view(org, record.trace_id)["vigia"]["status"] == "truncated_before_final"


def test_trace_view_of_a_cut_stream_is_not_calibrated(rt, org):
    """Regresión: 'GET /v1/vigia/traces/{id} de la plataforma dice calibrated=true en una traza de
    stream cortado, que el núcleo guardó con calibrated=false'. El núcleo guarda el stream cortado con
    calibration_id, tau=None y calibrated=False; la vista de la plataforma coincide con la del núcleo."""
    from vigia.contracts import ScoredToken
    from vigia.serve import trace_view as core_trace_view

    detector = constant_detector(rt.model_info, 0.9)
    install_detector(rt, org, detector)
    trace_id = "cortado" + "0" * 9
    tokens = [ScoredToken(k, "x", k, k + 1, 0.9, False) for k in range(3)]
    record = TraceRecord(
        trace_id=trace_id,
        org_id=org,
        key_id=None,
        created_at="2026-10-06T12:00:00+00:00",
        model=rt.model_info,
        prompt_messages=MESSAGES,
        generation=Generation(
            text="xxx", tokens=[], finish_reason="stop", prompt_tokens=3, model=rt.model_info
        ),
        result=VigiaResult(
            trace_id=trace_id,
            tokens=tokens,
            spans=[],
            response_p_error=None,
            action="pass",
            calibration_id=detector.id,
            calibrated=False,
            mode="full",
            status="cancelled",
            alpha=0.1,
            tau=None,
        ),
        latency_ms=1.0,
        overhead_ms=0.0,
    )
    rt.store_trace(org, record)
    view = rt.trace_view(org, trace_id)["vigia"]
    core = core_trace_view(record)["vigia"]
    assert view["calibrated"] is False and core["calibrated"] is False
    for key in ("calibration_id", "tau", "status", "degraded", "tokens"):
        assert view[key] == core[key], key


def test_context_for_wires_closing_to_shutdown(rt, org, owner):
    """Regresión: 'el 503 en cola no se activa en vigia_ee.app serve': context_for no pasaba closing."""
    assert rt.context_for(owner).closing() is False
    rt.begin_shutdown()
    assert rt.context_for(owner).closing() is True


def test_forget_org_drops_every_cached_state(rt, org, owner):
    install_detector(rt, org, constant_detector(rt.model_info, 0.9))
    rt.save_policy(owner, {"gap_max": 2})
    assert rt.detector_for(org) is not None and rt.policy_for(org).gap_max == 2
    rt.drift_for(org)
    assert rt.repo(org).delete_org()  # forget_org acompaña al borrado de la org (panel /data/delete)
    rt.forget_org(org)
    for state in (rt._detectors, rt._policies, rt._policy_alerts, rt._retrievers, rt._drift):
        assert org not in state
    assert "vigia_calibration_info{" not in rt.metrics.render()


def test_vllm_max_concurrency_is_configurable(tmp_path):
    """Regresión: 'El proxy vLLM serializa todas las generaciones': VIGIA_VLLM_MAX_CONCURRENCY llega al
    backend y el cupo de vigia.api lo usa."""
    from vigia.api import generation_gate
    from vigia_ee.runtime import build_backend

    assert Settings.from_env(VLLM_ENV).vllm_max_concurrency == 16
    settings = Settings.from_env({**VLLM_ENV, "VIGIA_VLLM_MAX_CONCURRENCY": "5"})
    backend = build_backend(settings)
    try:
        assert backend.max_concurrency == 5
        assert generation_gate(backend).slots == 5
    finally:
        backend.close()
    with pytest.raises(SettingsError, match="VIGIA_VLLM_MAX_CONCURRENCY"):
        Settings.from_env({**VLLM_ENV, "VIGIA_VLLM_MAX_CONCURRENCY": "0"})


class _PingBackend:
    """Backend diminuto compartido con un `ping` controlable (como el proxy vLLM)."""

    def __init__(self, ok) -> None:
        self._inner = shared_backend()
        self.ok = ok

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def ping(self) -> bool:
        if isinstance(self.ok, Exception):
            raise self.ok
        return self.ok


@pytest.mark.parametrize(("ok", "expected"), [(True, True), (False, False), (RuntimeError("x"), False)])
def test_readiness_checks_the_upstream_when_the_backend_pings(tmp_path, ok, expected):
    """Regresión: '/readyz no ve un vLLM caído': con ping() en el backend, readiness suma `backend`."""
    from vigia_ee.app import create_app

    rt = make_runtime(tmp_path, backend=_PingBackend(ok))
    try:
        assert rt.readiness()["backend"] is expected
        resp = TestClient(create_app(runtime=rt)).get("/readyz")
        assert resp.json()["checks"]["backend"] is expected and resp.status_code == 503  # sin worker
    finally:
        rt.close()
    plain = make_runtime(tmp_path / "plain")
    try:
        assert "backend" not in plain.readiness()  # el backend HF no tiene upstream
    finally:
        plain.close()


def _drift_detector(info):
    """Detector constante (p = 0,9) con una referencia de drift uniforme en [0,05, 0,95]."""
    from vigia.detector import Detector, artifact_sha256
    from vigia.drift import build_reference

    art = constant_detector(info, 0.9).artifact
    reference = build_reference(np.linspace(0.05, 0.95, 400), 0.5, {"out_entropy": np.linspace(0, 4, 400)})
    art["metrics"] = {"drift_reference": reference}
    art["sha256"] = artifact_sha256(art)
    return Detector(art)


def _store_scored(rt, org, det, n_traces, per_trace, *, prefix="drift", p=0.9):
    from vigia.contracts import ScoredToken

    for i in range(n_traces):
        trace_id = f"{prefix}{i:011d}"
        tokens = [ScoredToken(k, "x", k, k + 1, p, p >= 0.5) for k in range(per_trace)]
        rt.store_trace(
            org,
            TraceRecord(
                trace_id=trace_id,
                org_id=org,
                key_id=None,
                created_at=f"2026-10-04T12:{i // 60:02d}:{i % 60:02d}+00:00",
                model=rt.model_info,
                prompt_messages=MESSAGES,
                generation=Generation(
                    text="x" * per_trace,
                    tokens=[],
                    finish_reason="stop",
                    prompt_tokens=3,
                    model=rt.model_info,
                ),
                result=VigiaResult(
                    trace_id=trace_id,
                    tokens=tokens,
                    spans=[],
                    response_p_error=p,
                    action="pass",
                    calibration_id=det.id,
                    calibrated=True,
                    mode="full",
                    alpha=0.1,
                    tau=0.5,
                ),
                latency_ms=1.0,
                overhead_ms=0.1,
            ),
        )


def test_drift_for_reports_psi_against_the_active_detector_reference(rt, org):
    """P-36: PSI de las últimas trazas puntuadas con el detector activo contra su referencia."""
    old = constant_detector(rt.model_info, 0.9)
    install_detector(rt, org, old)
    report = rt.drift_for(org)
    assert report["available"] is False and "recalibrá" in report["reason"]
    det = _drift_detector(rt.model_info)
    install_detector(rt, org, det)
    _store_scored(rt, org, det, 3, 10, prefix="pocas")
    report = rt.drift_for(org)
    assert report["available"] is False and "al menos" in report["reason"] and report["n_tokens"] == 30
    _store_scored(rt, org, det, 30, 10)  # 300 tokens con p = 0,9; la referencia es uniforme
    _store_scored(rt, org, old, 5, 10, prefix="viejo")  # otro detector: no cuenta
    report = rt.drift_for(org)
    assert report["available"] and report["detector_id"] == det.id
    assert report["n_traces"] == 33 and report["n_tokens"] == 330
    assert set(report["features"]) == {"p_error", "flag"}  # estas trazas no guardan entropías
    assert report["features"]["p_error"]["level"] == "alert" and report["level"] == "alert"
    assert report["features"]["flag"]["psi"] > 0.25  # todo marcado contra 50 % en calibración
    gauges = rt.drift_gauges()
    assert gauges["p_error"] == pytest.approx(report["features"]["p_error"]["psi"])
    rt.metrics.set_drift(gauges)
    text = rt.metrics.render()
    assert 'vigia_drift_psi{feature="p_error"}' in text and org not in text


def test_settings_device_and_dtype_validation():
    """VIGIA_DEVICE y VIGIA_DTYPE se validan y nombran la variable al fallar."""
    import pytest

    from vigia_ee.runtime import Settings, SettingsError

    env = {"VIGIA_MODEL": "tiny:mixtral", "VIGIA_DEVICE": "cuda:1", "VIGIA_DTYPE": "bfloat16"}
    s = Settings.from_env(env)
    assert (s.device, s.dtype) == ("cuda:1", "bfloat16")
    for bad in ({"VIGIA_DEVICE": "gpu0"}, {"VIGIA_DTYPE": "int4"}):
        with pytest.raises(SettingsError, match=next(iter(bad))):
            Settings.from_env({"VIGIA_MODEL": "tiny:mixtral", **bad})


def test_insecure_cookies_use_a_dev_cookie_name(tmp_path):
    """Sin TLS la cookie no puede llamarse __Host- (el navegador la rechaza sin Secure)."""
    from tests.vigia_platform_support import PEPPERS
    from vigia_ee.runtime import Runtime, Settings
    from vigia_ee.security import DEV_SESSION_COOKIE, SESSION_COOKIE

    rt = Runtime(Settings(data_dir=tmp_path, model="tiny:mixtral", secure_cookies=False, peppers=PEPPERS))
    try:
        assert rt.auth.cookie_name == DEV_SESSION_COOKIE != SESSION_COOKIE
    finally:
        rt.close()
