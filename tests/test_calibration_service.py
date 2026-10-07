"""Calibración gestionada: validación del JSONL, job, activación y export de revisiones (P-23, AT-P16)."""

import json

import pytest
from vigia_platform_support import (
    add_user,
    constant_detector,
    install_detector,
    learnable_jsonl,
    make_runtime,
)

from vigia_ee import calibration as cal
from vigia_ee.calibration import (
    ActivationError,
    DatasetError,
    DatasetNotFound,
    DetectorNotFound,
    parse_dataset,
)
from vigia_ee.runtime import render_prompt


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


def audit_actions(rt, org_id):
    return [
        r["action"]
        for r in rt.db.query("SELECT action FROM audit_log WHERE org_id = ? ORDER BY seq", (org_id,))
    ]


def jsonl(*rows) -> bytes:
    return ("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n").encode()


# --- Validación del JSONL -------------------------------------------------------------------------


def test_parse_dataset_accepts_the_three_formats():
    content = (
        "﻿".encode()
        + jsonl(
            {"id": "a", "question": "¿Capital de Francia?", "answers": ["París", "Paris"]},
            {
                "id": "b",
                "messages": [
                    {"role": "developer", "content": "Sé breve."},
                    {"role": "user", "content": "Hola"},
                ],
                "response": "Hola, ¿cómo va?",
                "label": 0,
            },
            {
                "id": "c",
                "question": "¿Quién?",
                "response": "Fue Borges y Borges",
                "hallucinated_spans": ["Borges", "Borges"],
            },
            {
                "id": "d",
                "question": "¿Quién?",
                "response": "Fue Borges y Borges",
                "hallucinated_spans": [{"text": "Borges", "occurrence": 2}],
            },
            {"id": "e", "question": "¿Algo?", "response": "Todo bien", "hallucinated_spans": []},
        )
        + b"\n   \n"
    )
    items = parse_dataset(content)
    assert [i.kind for i in items] == ["answers", "label", "spans", "spans", "spans"]
    assert items[0].messages == [{"role": "user", "content": "¿Capital de Francia?"}]
    assert items[1].messages[0]["role"] == "system"
    assert items[2].spans == [(4, 10), (13, 19)]
    assert items[3].spans == [(13, 19)]
    assert items[4].spans == []


@pytest.mark.parametrize(
    ("content", "fragment"),
    [
        (b"\xff\xfe", "UTF-8"),
        (b'{"id": "a"', "Línea 1: JSON inválido"),
        (b'["no", "objeto"]', "objeto JSON"),
        (b'{"question": "q", "answers": ["x"]}', "'id'"),
        (
            b'{"id": "a", "question": "q", "messages": [{"role": "user", "content": "x"}], "answers": ["x"]}',
            "'messages' o 'question'",
        ),
        (b'{"id": "a", "question": "q"}', "'answers' o 'response'"),
        (b'{"id": "a", "question": "q", "answers": ["x"], "label": 1}', "no se combina"),
        (b'{"id": "a", "question": "q", "response": "r"}', "'label' o 'hallucinated_spans'"),
        (b'{"id": "a", "question": "q", "response": "r", "label": 2}', "0 (correcta) o 1"),
        (b'{"id": "a", "question": "q", "response": "r", "label": true}', "0 (correcta) o 1"),
        (
            b'{"id": "a", "question": "q", "response": "r", "hallucinated_spans": ["zzz"]}',
            "fragmento inválido",
        ),
        (b'{"id": "a", "question": "q", "answers": []}', "'answers'"),
        (b'{"id": "a", "question": "  ", "answers": ["x"]}', "'question'"),
        (b'{"id": "a", "messages": [{"role": "tool", "content": "x"}], "answers": ["x"]}', "rol inválido"),
        (b'{"id": "a", "question": "q", "answers": ["x"], "extra": 1}', "campos desconocidos: extra"),
        (b'{"id": "a", "id": "b", "question": "q", "answers": ["x"]}', "clave repetida"),
        (b'{"id": "a", "question": "q", "response": "r", "label": NaN}', "JSON inválido"),
        (b"\n\n", "vacío"),
        (
            b'{"id": "a", "question": "q", "answers": ["x"]}\n{"id": "a", "question": "q", "answers": ["y"]}',
            "Línea 2: el id 'a' está repetido",
        ),
    ],
)
def test_parse_dataset_rejects_with_a_clear_message(content, fragment):
    with pytest.raises(DatasetError) as exc:
        parse_dataset(content)
    assert fragment in str(exc.value)


def test_parse_dataset_rejects_deeply_nested_json_with_dataset_error():
    """Regresión: 'Subida de dataset de calibración con JSON profundamente anidado lanza
    RecursionError no capturada y devuelve HTTP 500 en vez de 422'."""
    prefix = '{"id":"a","question":"q","response":"r","label":0,"x":'
    deep = (prefix + "[" * 6000 + "]" * 6000 + "}").encode()
    content = jsonl({"id": "ok", "question": "q", "answers": ["x"]}) + deep
    with pytest.raises(DatasetError, match=r"Línea 2: JSON inválido \(anidamiento excesivo\)"):
        parse_dataset(content)


def test_parse_dataset_limits_the_number_of_items():
    content = jsonl(*({"id": str(i), "question": "q", "answers": ["x"]} for i in range(4)))
    assert len(parse_dataset(content, max_items=4)) == 4
    with pytest.raises(DatasetError, match="máximo de 3 items"):
        parse_dataset(content, max_items=3)


# --- Subida y arranque ----------------------------------------------------------------------------


def test_upload_dataset_validates_permission_size_and_audits(rt, org, owner, tmp_path):
    viewer = add_user(rt, org, "viewer@acme.com.ar", "viewer")
    reviewer = add_user(rt, org, "rev@acme.com.ar", "reviewer")
    for p in (viewer, reviewer):
        with pytest.raises(PermissionError):
            rt.calibration.upload_dataset(p, "x", learnable_jsonl(20))
    dataset_id = rt.calibration.upload_dataset(owner, "  demo  ", learnable_jsonl(20))
    row = rt.repo(org).get_dataset(dataset_id)
    assert row["name"] == "demo" and row["n_items"] == 20
    assert "dataset_upload" in audit_actions(rt, org)
    with pytest.raises(DatasetError, match="nombre"):
        rt.calibration.upload_dataset(owner, "a\x00b", learnable_jsonl(20))
    small = make_runtime(tmp_path / "small", max_upload_bytes=2048)
    try:
        org2 = small.create_org("Chica")
        p2 = add_user(small, org2, "o@chica.com.ar", "owner")
        with pytest.raises(DatasetError, match="supera el máximo"):
            small.calibration.upload_dataset(p2, "grande", learnable_jsonl(200))
    finally:
        small.close()


def test_start_checks_dataset_tenant_alphas_and_seed(rt, org, owner):
    other_org = rt.create_org("Otra")
    other = add_user(rt, other_org, "o@otra.com.ar", "owner")
    foreign = rt.calibration.upload_dataset(other, "ajeno", learnable_jsonl(20, prefix="o"))
    dataset_id = rt.calibration.upload_dataset(owner, "demo", learnable_jsonl(20))
    with pytest.raises(DatasetNotFound):
        rt.calibration.start(owner, foreign)
    with pytest.raises(DatasetNotFound):
        rt.calibration.start(owner, "no-existe-0000")
    for bad in ([0.0], [1.5], [], ["0.1"], [0.1] * 9):
        with pytest.raises(DatasetError):
            rt.calibration.start(owner, dataset_id, alphas=bad)
    with pytest.raises(DatasetError, match="semilla"):
        rt.calibration.start(owner, dataset_id, seed=-1)
    with pytest.raises(PermissionError):
        rt.calibration.start(add_user(rt, org, "v@acme.com.ar", "viewer"), dataset_id)
    job_id = rt.calibration.start(owner, dataset_id, alphas=[0.2, 0.1, 0.1], seed=7)
    job = rt.calibration.job(org, job_id)
    assert job["state"] == "queued" and job["alphas"] == [0.1, 0.2] and job["seed"] == 7
    assert rt.jobs.get(org, job_id)["max_attempts"] == 1
    assert rt.calibration.job(other_org, job_id) is None
    assert [j["id"] for j in rt.calibration.list_jobs(org)] == [job_id]
    assert rt.calibration.list_jobs(other_org) == []
    assert "calibration_start" in audit_actions(rt, org)


# --- Job de calibración ---------------------------------------------------------------------------


def test_AT_P16_job_trains_candidate_with_report(rt, org, owner, monkeypatch):
    """AT-P16: el job entrena, guarda el candidato con su reporte y emite calibration.completed."""
    events = []
    monkeypatch.setattr(rt, "emit_event", lambda o, e, d: events.append((o, e, dict(d))) or [])
    dataset_id = rt.calibration.upload_dataset(owner, "demo", learnable_jsonl(40))
    job_id = rt.calibration.start(owner, dataset_id)
    finished = rt.jobs.run_once()
    assert finished["id"] == job_id and finished["state"] == "done", finished["error"]
    job = rt.calibration.job(org, job_id)
    detector_id = job["result"]["detector_id"]
    assert job["error"] is None and job["result"]["n_responses"] == 40
    row = rt.repo(org).get_detector(detector_id)
    assert row["status"] == "candidate" and row["dataset_id"] == dataset_id
    report = json.loads(row["report_json"])
    assert report["detector_id"] == detector_id
    assert report["bootstrap"]["n_boot"] == cal.PLATFORM_N_BOOT
    assert report["detector"]["available"] and "auroc" in report["detector"]["token"]
    assert report["calibration"]["dataset_id"] == dataset_id
    assert report["calibration"]["split"]["detector_id"] == detector_id
    assert events == [
        (
            org,
            "calibration.completed",
            {"detector_id": detector_id, "dataset_id": dataset_id, "job_id": job_id},
        )
    ]
    assert rt.detector_for(org) is None, "un candidato no se activa solo"
    # Mismo dataset, semilla e instante: mismo artefacto (id = hash); el job avisa que ya existía.
    fixed = rt.now()
    monkeypatch.setattr(rt, "clock", lambda: fixed)
    rt.calibration.start(owner, dataset_id)
    first = rt.jobs.run_once()
    rt.calibration.start(owner, dataset_id)
    again = rt.jobs.run_once()
    assert first["result"]["already_existed"] is False
    assert again["state"] == "done" and again["result"]["already_existed"] is True
    assert again["result"]["detector_id"] == first["result"]["detector_id"]


@pytest.mark.parametrize(
    ("content", "fragment"),
    [
        (learnable_jsonl(10), "al menos 20 respuestas etiquetadas; el dataset aportó 10"),
        (
            jsonl(*({"id": str(i), "question": "q", "response": "la casa", "label": 0} for i in range(25))),
            "Hacen falta respuestas correctas y erróneas",
        ),
    ],
)
def test_job_fails_with_user_message_when_data_is_insufficient(rt, org, owner, content, fragment):
    dataset_id = rt.calibration.upload_dataset(owner, "poco", content)
    job_id = rt.calibration.start(owner, dataset_id)
    assert rt.jobs.run_once()["state"] == "failed"
    job = rt.calibration.job(org, job_id)
    assert job["state"] == "failed" and fragment in job["error"]
    assert job["result"] is None


def test_unexpected_job_error_is_generic_for_the_user(rt, org, owner, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("detalle interno /srv/secreto")

    monkeypatch.setattr(cal, "train_detector", boom)
    dataset_id = rt.calibration.upload_dataset(owner, "demo", learnable_jsonl(40))
    job_id = rt.calibration.start(owner, dataset_id)
    assert rt.jobs.run_once()["state"] == "failed"
    job = rt.calibration.job(org, job_id)
    assert job["error"] == cal._GENERIC_JOB_ERROR
    assert "secreto" not in json.dumps(job)


def test_answers_items_are_generated_and_labeled_computably(rt):
    """Con `answers`, Vigía genera y etiqueta por EM/F1: el gold igual a la generación vale 0."""
    question = [{"role": "user", "content": "pregunta fija"}]
    tokens = list(rt.backend.stream(question, max_tokens=cal.ANSWER_MAX_TOKENS, temperature=0.0))
    generated = "".join(t.text for t in tokens if t.channel == "final" and not t.is_special)
    items = [
        cal.CalibrationItem("ok", question, "answers", answers=[generated]),
        cal.CalibrationItem("mal", question, "answers", answers=["respuesta que nunca sale"]),
    ]
    rows, labels, groups, ids, skipped, kinds = rt.calibration._collect(items)
    by_item = {i: {lab for lab, ii in zip(labels, ids, strict=True) if ii == i} for i in set(ids)}
    assert kinds == {"answers"} and not skipped
    assert by_item == {"ok": {0}, "mal": {1}}
    assert len(set(groups)) == 1, "la misma pregunta es un solo grupo del split"
    assert all("out_entropy" in r for r in rows)


# --- Activación -----------------------------------------------------------------------------------


def test_activate_checks_tenant_model_and_policy_alpha(rt, org, owner, monkeypatch):
    events = []
    monkeypatch.setattr(rt, "emit_event", lambda o, e, d: events.append((e, dict(d))) or [])
    repo = rt.repo(org)
    with pytest.raises(DetectorNotFound):
        rt.calibration.activate(owner, "0123456789abcdef")
    other_org = rt.create_org("Otra")
    foreign = constant_detector(rt.model_info, 0.3)
    rt.repo(other_org).insert_detector(foreign.id, foreign.to_json())
    with pytest.raises(DetectorNotFound):
        rt.calibration.activate(owner, foreign.id)
    alien = constant_detector(rt.model_info, model={**rt.model_info.fingerprint(), "model_id": "otro/modelo"})
    repo.insert_detector(alien.id, alien.to_json())
    with pytest.raises(ActivationError, match="otro modelo"):
        rt.calibration.activate(owner, alien.id)
    strict = constant_detector(rt.model_info, 0.8, alphas=(0.2, 0.3), taus=(0.4, 0.5))
    repo.insert_detector(strict.id, strict.to_json())
    rt.save_policy(owner, {"alpha": 0.1})  # solo un alpha fijado por la política bloquea la activación
    with pytest.raises(ActivationError, match="alpha"):
        rt.calibration.activate(owner, strict.id)
    good = constant_detector(rt.model_info, 0.7)
    repo.insert_detector(good.id, good.to_json())
    with pytest.raises(PermissionError):
        rt.calibration.activate(add_user(rt, org, "rev@acme.com.ar", "reviewer"), good.id)
    assert rt.detector_for(org) is None
    rt.calibration.activate(owner, good.id)
    assert rt.detector_for(org).id == good.id
    assert repo.get_detector(good.id)["status"] == "active"
    assert "detector_activate" in audit_actions(rt, org)
    assert events == [("calibration.activated", {"detector_id": good.id})]


# --- Export de la cola de revisión ----------------------------------------------------------------


def test_dataset_from_reviews_exports_completed_decisions(rt, org, owner):
    repo = rt.repo(org)
    repo.update_org(store_text="full")
    r1 = add_user(rt, org, "r1@acme.com.ar", "reviewer")
    r2 = add_user(rt, org, "r2@acme.com.ar", "reviewer")
    with pytest.raises(DatasetError, match="No hay decisiones"):
        rt.calibration.dataset_from_reviews(owner, "vacío")
    prompt = render_prompt(
        [{"role": "system", "content": "Sé breve."}, {"role": "user", "content": "¿Quién?"}]
    )
    response = "Fue Borges, sí, Borges en 1899"

    def trace(text=response):
        return repo.insert_trace(
            model_id="tiny:mixtral",
            response_text=text,
            prompt_text=prompt,
            action="human_review",
            spans=[{"start": 16, "end": 22, "score": 0.9}, {"start": 0, "end": 3, "score": 0.8}],
        )

    def decide(item_id, *labels):
        for user, label in zip((r1, r2), labels, strict=False):
            assert repo.add_review_decision(item_id, user.user_id, label)

    whole, spans, conflict, unsure = trace(), trace(), trace(), trace()
    decide(repo.create_review_item(whole, min_reviewers=2), "error", "error")
    decide(repo.create_review_item(spans, span_idx=0, min_reviewers=2), "error", "uncertain")
    decide(repo.create_review_item(spans, span_idx=1, min_reviewers=1), "correct")
    decide(repo.create_review_item(conflict, min_reviewers=2), "error", "correct")
    decide(repo.create_review_item(unsure, min_reviewers=1), "uncertain")
    repo.create_review_item(trace(), min_reviewers=2)  # pendiente: no se exporta
    with pytest.raises(PermissionError):
        rt.calibration.dataset_from_reviews(r1, "x")
    dataset_id = rt.calibration.dataset_from_reviews(owner, "revisiones")
    content = bytes(repo.get_dataset(dataset_id)["content"])
    rows = {r["id"]: r for r in map(json.loads, content.decode().splitlines())}
    assert set(rows) == {whole, spans}
    assert rows[whole]["label"] == 1 and rows[whole]["messages"][0] == {
        "role": "system",
        "content": "Sé breve.",
    }
    assert rows[spans]["hallucinated_spans"] == [{"text": "Borges", "occurrence": 2}]
    items = {i.id: i for i in parse_dataset(content)}
    assert items[spans].spans == [(16, 22)]
    assert "dataset_from_reviews" in audit_actions(rt, org)


def test_dataset_from_reviews_skips_traces_without_text(rt, org, owner):
    repo = rt.repo(org)
    repo.update_org(store_text="none")
    reviewer = add_user(rt, org, "r1@acme.com.ar", "reviewer")
    trace_id = repo.insert_trace(
        model_id="tiny:mixtral", response_text="secreto", prompt_text="user: hola", action="human_review"
    )
    item = repo.create_review_item(trace_id, min_reviewers=1)
    repo.add_review_decision(item, reviewer.user_id, "error")
    with pytest.raises(DatasetError, match="con texto"):
        rt.calibration.dataset_from_reviews(owner, "nada")


def test_activated_detector_from_job_is_used_by_the_runtime(rt, org, owner):
    """El detector entrenado por el job, una vez activado, es el que puntúa los requests siguientes."""
    install_detector(rt, org, constant_detector(rt.model_info, 0.2))
    dataset_id = rt.calibration.upload_dataset(owner, "demo", learnable_jsonl(40))
    rt.calibration.start(owner, dataset_id)
    detector_id = rt.jobs.run_once()["result"]["detector_id"]
    rt.calibration.activate(owner, detector_id)
    assert rt.context_for(owner).scorer.id == detector_id


# --- Regresiones: tope de jobs, split por pregunta y etiquetas con gold_correct -------------------


def test_start_rejects_a_second_pending_calibration_of_the_same_org(rt, org, owner):
    """Regresión: 'Un admin puede encolar calibraciones sin límite: atrasa las de otros tenants y la
    purga, y pone /readyz en rojo'. Una org tiene a lo sumo una calibración en cola o en curso."""
    other_org = rt.create_org("Otra")
    other = add_user(rt, other_org, "o@otra.com.ar", "owner")
    mine = rt.calibration.upload_dataset(owner, "demo", learnable_jsonl(20))
    theirs = rt.calibration.upload_dataset(other, "ajeno", learnable_jsonl(20, prefix="o"))
    first = rt.calibration.start(owner, mine)
    for _ in range(3):
        with pytest.raises(cal.CalibrationBusy, match="en cola o en curso"):
            rt.calibration.start(owner, mine)
    assert isinstance(cal.CalibrationBusy("x"), DatasetError), "el panel lo muestra como 422"
    assert [j["id"] for j in rt.calibration.list_jobs(org)] == [first]
    assert rt.jobs.depth() == 1
    rt.calibration.start(other, theirs)  # otro tenant no queda bloqueado
    for _ in range(2):
        rt.jobs.run_once()
    assert rt.calibration.job(org, first)["state"] in ("done", "failed")
    assert rt.calibration.start(owner, mine) != first  # con el job terminado se puede volver a lanzar


def test_collect_groups_the_split_by_question_not_by_item(rt):
    """Regresión: 'El job de calibración de la plataforma agrupa el split por id de item y no por
    pregunta: la misma pregunta cae en train y en test' (AT-M04, P-05)."""
    from vigia.calibrate import split_groups

    rows = []
    for q in range(30):
        question = f"¿Cuál es la capital del país {q}?"
        rows.append({"id": f"q{q}-ok", "question": question, "response": "Lima", "label": 0})
        # Mismo texto con otro espaciado y mayúsculas: sigue siendo la misma pregunta.
        rows.append({"id": f"q{q}-mal", "question": "  " + question.upper(), "response": "Quito", "label": 1})
    items = parse_dataset(jsonl(*rows))
    _, _, groups, ids, skipped, _ = rt.calibration._collect(items)
    assert not skipped and len(set(ids)) == 60 and len(set(groups)) == 30
    split = split_groups(groups, seed=0)
    where = {gid: name for name, members in split.items() for gid in members}
    question_of = {it.id: it.messages[-1]["content"].strip().casefold() for it in items}
    splits_by_question: dict[str, set[str]] = {}
    for gid, item_id in zip(groups, ids, strict=True):
        splits_by_question.setdefault(question_of[item_id], set()).add(where[gid])
    assert all(len(s) == 1 for s in splits_by_question.values())


def test_explicit_group_field_defines_the_question():
    """Regresión del split por pregunta: el campo opcional 'group' manda sobre el hash del texto."""
    items = parse_dataset(
        jsonl(
            {"id": "a", "group": "g1", "question": "¿Capital de Perú?", "response": "Lima", "label": 0},
            {"id": "b", "group": "g1", "question": "¿Y la de Perú?", "response": "Quito", "label": 1},
            {"id": "c", "question": "¿Capital de Perú?", "response": "Lima", "label": 0},
        )
    )
    assert items[0].question_key() == items[1].question_key() != items[2].question_key()
    with pytest.raises(DatasetError, match="'group'"):
        parse_dataset(jsonl({"id": "a", "group": "", "question": "q", "answers": ["x"]}))


def test_answers_items_use_gold_correct_for_numbers_and_dates():
    """Regresión: 'El job de calibración de la plataforma etiqueta los items answers solo con EM/F1,
    sin el arreglo de gold_correct: 15 cuenta como correcta para el gold 1,5'."""
    from types import SimpleNamespace

    from vigia.contracts import Capabilities, Tier, TokenEvent

    cases = [  # (pregunta, gold, respuesta del modelo, etiqueta esperada: 1 = errónea)
        ("¿Cuántos litros?", "1,5", "15", 1),
        ("¿Cuándo terminó la guerra en Europa?", "8 de mayo de 1945", "8 de mayo de 1946", 1),
        ("¿Cuántos habitantes?", "3,2 millones", "4,2 millones", 1),
        ("¿Cuántos litros hay?", "1,5", "1,5", 0),
        ("¿Capital de Francia?", "París", "París", 0),
    ]
    replies = {q: r for q, _, r, _ in cases}

    class FakeBackend:
        def capabilities(self):
            return Capabilities(tiers=(Tier.F0, Tier.F1), streaming=True, disabled=())

        def stream(self, messages, **kwargs):
            text = replies[messages[-1]["content"]]
            event = TokenEvent(
                index=0, token_id=1, text=text, start=0, end=len(text), features={"out_nll": 1.0}
            )
            return iter([event])

    content = jsonl(*({"id": f"i{n}", "question": q, "answers": [g]} for n, (q, g, _, _) in enumerate(cases)))
    svc = cal.CalibrationService.__new__(cal.CalibrationService)
    svc.runtime = SimpleNamespace(backend=FakeBackend())
    _, labels, _, ids, skipped, _ = svc._collect(parse_dataset(content))
    assert not skipped and ids == [f"i{n}" for n in range(len(cases))]
    assert labels == [expected for _, _, _, expected in cases]


def test_collect_maps_harmony_spans_to_the_final_text():
    """Regresión (gpt-oss): con marcado harmony en la respuesta, los spans del dataset se aplicaban en
    el marco del texto con marcado y marcaban el token equivocado ('.' en vez de 'Quito')."""
    from types import SimpleNamespace

    from vigia.contracts import Capabilities, Tier, TokenEvent

    marked = (
        "<|channel|>analysis<|message|>Quito, creo.<|end|><|start|>assistant<|channel|>final"
        "<|message|>Es Quito.<|return|>"
    )
    finals = {marked: ["Es", " Quito", "."], "Otra respuesta": ["Distinto"]}

    class FakeBackend:
        def capabilities(self):
            return Capabilities(tiers=(Tier.F0, Tier.F1), streaming=True, disabled=())

        def teacher_force(self, messages, response, tiers=()):
            out, pos = [], 0
            for i, piece in enumerate(finals[response]):
                out.append(TokenEvent(i, i, piece, pos, pos + len(piece), {"out_nll": 1.0}))
                pos += len(piece)
            return out

    content = jsonl(
        {
            "id": "h",
            "question": "¿Capital de Perú?",
            "response": marked,
            "hallucinated_spans": [{"text": "Quito", "occurrence": 2}],
        },
        {"id": "x", "question": "¿Algo?", "response": "Otra respuesta", "hallucinated_spans": ["respuesta"]},
    )
    svc = cal.CalibrationService.__new__(cal.CalibrationService)
    svc.runtime = SimpleNamespace(backend=FakeBackend())
    _, labels, _, ids, skipped, _ = svc._collect(parse_dataset(content))
    assert ids == ["h", "h", "h"] and labels == [0, 1, 0]
    assert skipped == {"texto_final_distinto": 1}
