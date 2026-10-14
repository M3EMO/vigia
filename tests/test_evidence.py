"""Datasets, etiquetas computables y experimento de evidencia reanudable (AT-M09, AT-M08, P-06, P-08)."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from vigia.cli import make_tiny_backend
from vigia.contracts import Tier
from vigia.datasets import (
    DEFAULT_INSTRUCTION,
    DatasetError,
    item_to_json,
    load_mkqa_es,
    parse_jsonl,
    question_messages,
    stratified_sample,
    validate_item,
)
from vigia.detector import Detector
from vigia.evidence import (
    FEATURES_FILE,
    ROWS_FILE,
    EvidenceError,
    FeatureSet,
    RunStore,
    build_features,
    computable_labels,
    extract,
    gold_correct,
    relabel,
    run_evidence,
)
from vigia.labels import IGNORE_LABEL, FakeJudge

LETTERS = "abcdefghijklmnopqrstuvwxyz "


# --- Dataset sintético sobre el modelo diminuto ----------------------------------------------------


def ascii_greedy(backend, messages, n_tokens: int) -> str:
    """Continuación greedy restringida a letras ASCII: la respuesta que el modelo "prefiere"."""
    tok = backend.tokenizer
    allowed = sorted(set(tok(LETTERS, add_special_tokens=False)["input_ids"]))
    ids = backend._prompt_ids(messages)
    out: list[int] = []
    with torch.inference_mode():
        for _ in range(n_tokens):
            logits = backend.model(torch.tensor([ids + out])).logits[0, -1]
            out.append(allowed[int(torch.argmax(logits[allowed]))])
    return tok.decode(out)


def synthetic_items(backend, *, n_forced: int = 20, n_gold: int = 10, seed: int = 0) -> list[dict]:
    """30 items: respuestas dadas (greedy = correcta, azar = errónea) y preguntas con gold.

    Las preguntas con gold usan como alias la propia salida del modelo (8 correctas) o un alias que
    no puede coincidir (2 erróneas), así el modelo diminuto tiene exactitud por tipo verificable.
    """
    rng = random.Random(seed)  # noqa: S311 - datos de prueba reproducibles
    items = []
    for i in range(n_forced):
        question = f"Pregunta de prueba número {i}"
        msgs = question_messages(question)
        if i % 2 == 0:
            response, label = ascii_greedy(backend, msgs, 10), 0
        else:
            response = "".join(rng.choice(LETTERS[:-1]) for _ in range(10))
            label = 1
        items.append(
            validate_item(
                {"id": f"tf{i}", "question": question, "response": response, "label": label, "type": "entity"}
            )
        )
    for i in range(n_gold):
        question = f"Otra pregunta {i}"
        stream = backend.stream(question_messages(question), max_tokens=8)
        list(stream)
        own = stream.text if stream.text.strip() else "x"
        answer = own if i < n_gold - 2 else "Buenos Aires"
        items.append(
            validate_item(
                {
                    "id": f"g{i}",
                    "question": question,
                    "answers": [answer],
                    "type": "number" if i % 2 else "date",
                }
            )
        )
    return items


@pytest.fixture(scope="module")
def backend():
    return make_tiny_backend("mixtral")


@pytest.fixture(scope="module")
def items(backend):
    return synthetic_items(backend)


class CountingBackend:
    """Envuelve un backend y cuenta llamadas a stream y teacher_force."""

    def __init__(self, inner):
        self.inner = inner
        self.calls = 0

    def model_info(self):
        return self.inner.model_info()

    def capabilities(self):
        return self.inner.capabilities()

    def stream(self, *args, **kwargs):
        self.calls += 1
        return self.inner.stream(*args, **kwargs)

    def teacher_force(self, *args, **kwargs):
        self.calls += 1
        return self.inner.teacher_force(*args, **kwargs)


# --- Datasets --------------------------------------------------------------------------------------


def test_validate_item_three_label_modes():
    gold = validate_item({"id": 1, "question": "¿Capital de Francia?", "answers": "París"})
    assert gold["mode"] == "gold" and gold["id"] == "1" and gold["group"] == "1"
    assert gold["answers"] == ["París"]
    assert gold["messages"] == [{"role": "user", "content": f"¿Capital de Francia?\n\n{DEFAULT_INSTRUCTION}"}]
    messages = [{"role": "developer", "content": "Sé breve."}, {"role": "user", "content": "Hola"}]
    lab = validate_item(
        {"id": "a", "messages": messages, "response": "Hola, ¿qué tal?", "label": 1, "group": "q1"}
    )
    assert lab["mode"] == "label" and lab["label"] == 1 and lab["group"] == "q1"
    assert lab["messages"][0]["role"] == "system" and lab["question"] == "Hola"
    spans = validate_item(
        {
            "id": "s",
            "question": "¿Quién?",
            "response": "Lo pintó Goya en 1900.",
            "hallucinated_spans": ["1900", [9, 13]],
        }
    )
    assert spans["mode"] == "spans" and spans["hallucinated_spans"] == [[9, 13], [17, 21]]
    for item in (gold, lab, spans):
        assert validate_item(item_to_json(item)) == item


@pytest.mark.parametrize(
    ("obj", "fragment"),
    [
        ({"id": "x", "question": "q"}, "exactamente una"),
        (
            {"id": "x", "question": "q", "response": "r", "label": 1, "hallucinated_spans": []},
            "exactamente una",
        ),
        ({"id": "x", "question": "q", "answers": []}, "'answers'"),
        ({"id": "x", "question": "q", "response": "r", "label": True}, "'label'"),
        ({"id": "x", "question": "q", "response": "abc", "hallucinated_spans": [[2, 9]]}, "fuera"),
        ({"id": "x", "question": "q", "response": "ab ab", "hallucinated_spans": ["ab"]}, "varias veces"),
        (
            {"id": "x", "question": "q", "response": "abcd", "hallucinated_spans": [[0, 3], [2, 4]]},
            "solapados",
        ),
        ({"id": "x", "question": "q", "answers": ["a"], "extra": 1}, "desconocidos"),
        ({"id": "", "question": "q", "answers": ["a"]}, "'id'"),
        ({"id": "x", "messages": [{"role": "assistant", "content": "a"}], "answers": ["a"]}, "usuario"),
        ({"id": "x", "question": "q", "messages": [], "answers": ["a"]}, "solo uno"),
    ],
)
def test_validate_item_rejects_with_clear_message(obj, fragment):
    with pytest.raises(DatasetError, match=fragment):
        validate_item(obj)


def test_parse_jsonl_reports_line_numbers():
    ok = '{"id": "a", "question": "q", "answers": ["x"]}'
    with pytest.raises(DatasetError, match="línea 2: JSON inválido"):
        parse_jsonl(f"{ok}\n{{roto\n")
    with pytest.raises(DatasetError, match="línea 3: id repetido"):
        parse_jsonl(f"{ok}\n\n{ok}\n")
    with pytest.raises(DatasetError, match="JSON inválido"):
        parse_jsonl('{"id": "a", "question": "q", "answers": [NaN]}')
    with pytest.raises(DatasetError, match="UTF-8"):
        parse_jsonl(b"\xff\xfe")
    with pytest.raises(DatasetError, match="no tiene items"):
        parse_jsonl("\n\n")
    assert [it["id"] for it in parse_jsonl(("﻿" + ok).encode("utf-8"))] == ["a"]


def test_stratified_sample_is_proportional_and_reproducible():
    pool = [
        {"id": str(i), "type": "entity" if i < 60 else "date" if i < 90 else "number"} for i in range(100)
    ]
    a = stratified_sample(pool, 20, seed=3)
    b = stratified_sample(pool, 20, seed=3)
    assert [x["id"] for x in a] == [x["id"] for x in b]
    counts = {t: sum(1 for x in a if x["type"] == t) for t in ("entity", "date", "number")}
    assert counts == {"entity": 12, "date": 6, "number": 2}
    assert len({x["id"] for x in a}) == 20
    assert stratified_sample(pool, 20, seed=4) != a
    assert len(stratified_sample(pool, 500)) == 100


def test_load_mkqa_es_filters_and_samples(tmp_path):
    path = tmp_path / "mkqa_es.jsonl"
    rows = [
        {"id": "1", "question": "¿cuántas temporadas?", "answers": ["2.0", "2", "2"], "type": "number"},
        {"id": "2", "question": "quién canta", "answers": ["Chris Young"], "type": "entity"},
        {"id": "3", "question": "sin respuesta", "answers": ["", " "], "type": "entity"},
        {"id": "4", "question": "cuándo", "answers": ["2013"], "type": "date"},
    ]
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    items = load_mkqa_es(path)
    assert [it["id"] for it in items] == ["1", "2", "4"]
    assert items[0]["answers"] == ["2.0", "2"] and items[0]["mode"] == "gold"
    assert items[0]["messages"][0]["content"].endswith(DEFAULT_INSTRUCTION)
    assert [it["id"] for it in load_mkqa_es(path, types=["entity"])] == ["2"]
    assert len(load_mkqa_es(path, n=2, seed=0)) == 2
    with pytest.raises(DatasetError, match="no existe"):
        load_mkqa_es(tmp_path / "falta.jsonl")


# --- Etiquetas computables -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("prediction", "answers", "kind", "expected"),
    [
        ("La Ciudad de México.", ["Ciudad de Mexico"], "entity", True),  # EM con artículo, tilde y punto
        ("Fue Johannes Hubert de Mol", ["John de Mol", "Johannes Hubert de Mol"], "entity", True),  # F1
        ("Chris Young\nEs un cantante country.", ["Chris Young"], "entity", True),  # primera línea
        ("Son 11 años", ["11.0 año terrestre", "11 año terrestre"], "number_with_unit", True),
        ("Hay 2 temporadas", ["2.0", "2"], "number", True),
        ("En 2013", ["2013"], "date", True),
        ("El 29 de agosto de 2019", ["2019-08-29", "08/29/2019"], "date", True),
        ("El 11 de agosto de 2017", ["2017-08-11", "08/11/2017"], "date", True),
        ("El 8 de noviembre de 2017", ["2017-08-11", "08/11/2017"], "date", False),  # alias US no se usa
        ("Apolo 12", ["Apolo 11"], "entity", False),  # F1 0,5, pero el número difiere
        ("La misión Apolo 11", ["Apolo 11"], "entity", True),  # F1 con el mismo número
        ("Ringo Starr", ["Chris Young"], "entity", False),
        ("   ", ["Chris Young"], "entity", False),
    ],
)
def test_gold_correct(prediction, answers, kind, expected):
    assert gold_correct(prediction, answers, kind)[0] is expected


def test_computable_labels_by_mode():
    tokens = [
        {"i": 0, "id": 1, "text": "Goya", "start": 0, "end": 4, "channel": "final", "special": False},
        {"i": 1, "id": 2, "text": " en", "start": 4, "end": 7, "channel": "final", "special": False},
        {"i": 2, "id": 3, "text": " 1900", "start": 7, "end": 12, "channel": "final", "special": False},
        {"i": 3, "id": 4, "text": "<|eos|>", "start": 12, "end": 12, "channel": "control", "special": True},
    ]
    item = {"mode": "spans", "response": "Goya en 1900", "hallucinated_spans": [[8, 12]]}
    row = {"id": "s", "response": "Goya en 1900", "tokens": tokens, "item": item}
    assert computable_labels(row)[0] == [0, 0, 1, IGNORE_LABEL]
    row["item"]["response"] = "Otro texto"
    labels, info = computable_labels(row)
    assert labels == [IGNORE_LABEL] * 4 and info["rejected"] == "texto_final_distinto"
    row["item"] = {"mode": "gold", "answers": ["Goya en 1900"], "type": None}
    assert computable_labels(row)[0] == [0, 0, 0, IGNORE_LABEL]
    row["item"] = {"mode": "label", "label": 1}
    assert computable_labels(row) == ([1, 1, 1, IGNORE_LABEL], {"correct": False})


def test_computable_labels_maps_harmony_spans_to_the_final_text():
    """Regresión (gpt-oss): con marcado harmony en la respuesta del dataset, los spans se referían a la
    respuesta con marcado y los tokens al texto final; el item quedaba rechazado o mal marcado."""
    tokens = [
        {"i": 0, "id": 1, "text": "Goya", "start": 0, "end": 4, "channel": "final", "special": False},
        {"i": 1, "id": 2, "text": " en", "start": 4, "end": 7, "channel": "final", "special": False},
        {"i": 2, "id": 3, "text": " 1900", "start": 7, "end": 12, "channel": "final", "special": False},
    ]
    marked = (
        "<|channel|>analysis<|message|>Creo que 1900.<|end|><|start|>assistant<|channel|>final"
        "<|message|>Goya en 1900<|return|>"
    )
    at = marked.rindex("1900")
    item = {"mode": "spans", "response": marked, "hallucinated_spans": [[at, at + 4]]}
    row = {"id": "h", "response": "Goya en 1900", "tokens": tokens, "item": item}
    assert computable_labels(row) == ([0, 0, 1], {"correct": False})
    analysis_only = marked.index("1900")  # un span que cae en analysis no marca nada del texto final
    row["item"] = {**item, "hallucinated_spans": [[analysis_only, analysis_only + 4]]}
    assert computable_labels(row) == ([0, 0, 0], {"correct": True})


def test_parse_jsonl_rejects_deep_nesting_as_dataset_error():
    """Regresión: una línea con anidamiento profundo levantaba RecursionError, no el error de validación."""
    with pytest.raises(DatasetError, match="línea 1: JSON inválido"):
        parse_jsonl("[" * 100_000 + "]" * 100_000)


def test_run_store_resumes_a_manifest_without_router_hash(tmp_path):
    """Compatibilidad: una corrida previa a ModelInfo.router_sha256 se reanuda con el mismo modelo; un
    router distinto sí la rechaza."""
    info = {"model_id": "m", "model_type": "mixtral", "n_experts": 4, "router_sha256": None}
    manifest = {"schema_version": "1", "model_info": info, "tiers": ["F0"], "max_tokens": 8, "seed": 0}
    RunStore(tmp_path).open(manifest)
    RunStore(tmp_path).open({**manifest, "model_info": {**info, "router_sha256": "a" * 64}})
    store = tmp_path / "otra"
    RunStore(store).open({**manifest, "model_info": {**info, "router_sha256": "a" * 64}})
    with pytest.raises(EvidenceError, match="model_info"):
        RunStore(store).open({**manifest, "model_info": {**info, "router_sha256": "b" * 64}})


# --- Experimento de evidencia ----------------------------------------------------------------------


def test_at_m09_evidence_report_end_to_end(backend, items, tmp_path):
    """AT-M09 (y AT-M11 sobre el diminuto): el reporte trae AUROC con IC por nivel y los 3 baselines."""
    report = run_evidence(
        backend,
        items,
        out_dir=tmp_path,
        max_tokens=8,
        n_boot=200,
        bench_prompts=[it["messages"] for it in items[:2]],
        # En el diminuto, out_entropy sola no discrimina en todos los cal de 6 preguntas (Platt <= 0
        # en la mitad de las semillas del split estratificado); esta semilla entrena los 3 baselines.
        seed=3,
    )
    for name in (
        "rows.jsonl",
        "features.npz",
        "detector.json",
        "split.json",
        "report.json",
        "report.html",
        "summary.txt",
        "manifest.json",
    ):
        assert (tmp_path / name).is_file(), name
    auroc = report["detector"]["token"]["auroc"]
    assert auroc["value"] > 0.8 and auroc["ci"][0] <= auroc["value"] <= auroc["ci"][1]
    assert set(report["tiers"]) >= {"F0", "F1", "F0+F1"}
    assert all(report["tiers"][k]["available"] for k in ("F0", "F1", "F0+F1"))
    assert set(report["baselines"]) == {"out_entropy", "out_nll", "h_probe"}
    for entry in report["baselines"].values():
        assert entry["available"] and len(entry["token"]["auroc"]["ci"]) == 2
    assert report["bootstrap"]["n_boot"] == 200 and report["bootstrap"]["cluster"] == "pregunta"

    ev = report["evidence"]
    assert ev["n_rows"] == 30 and ev["n_generated"] == 10 and ev["n_teacher_forced"] == 20
    assert ev["tiers"] == ["F0", "F1", "H"] and ev["label_source"] == "computable"
    gen = ev["model_quality"]["generated"]
    assert gen["n"] == 10 and gen["correct"] == 8 and gen["accuracy"] == pytest.approx(0.8)
    assert gen["by_type"]["number"]["hallucination_rate"] == pytest.approx(0.2)
    assert gen["by_type"]["date"]["accuracy"] == pytest.approx(0.8)
    assert "latency" in report or "latency_error" in report

    det = Detector.load(tmp_path / "detector.json")
    assert det.id == report["detector_id"]
    det.check_model(backend.model_info())
    assert det.feature_names and not any(n.startswith("h_") for n in det.feature_names)  # F0+F1
    groups = json.loads((tmp_path / "split.json").read_text(encoding="utf-8"))["groups"]
    assert groups["test"] and not set(groups["test"]) & (set(groups["train"]) | set(groups["cal"]))

    page = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "Tabla 5. Exactitud del modelo" in page and "Baseline h_probe" in page
    summary = (tmp_path / "summary.txt").read_text(encoding="utf-8")
    assert "Exactitud del modelo: 80,0 %" in summary and "Baseline out_nll" in summary
    assert json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))["detector_id"] == det.id


class _Stop(Exception):
    pass


def test_evidence_resumes_without_duplicates(backend, items, tmp_path):
    """Un corte a mitad (y una línea final a medio escribir) se retoma sin duplicar ni recalcular."""
    run_dir = tmp_path / "corrida"

    def stop_at_12(done, total, item_id):
        if done == 12:
            raise _Stop

    counting = CountingBackend(backend)
    with pytest.raises(_Stop):
        extract(counting, items, run_dir, max_tokens=8, progress_cb=stop_at_12)
    assert counting.calls == 12
    rows_path = run_dir / ROWS_FILE
    assert len(rows_path.read_text(encoding="utf-8").splitlines()) == 12
    with open(rows_path, "a", encoding="utf-8") as fh:  # corte durante la escritura de la fila 13
        fh.write('{"id": "tf12", "group": "tf12", "tok')

    counting.calls = 0
    report = run_evidence(counting, items, out_dir=run_dir, max_tokens=8, n_boot=100)
    assert counting.calls == 18
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()]
    assert sorted(r["id"] for r in rows) == sorted(it["id"] for it in items)
    assert report["evidence"]["n_rows"] == 30

    fresh_dir = tmp_path / "sin_corte"
    run_evidence(backend, items, out_dir=fresh_dir, max_tokens=8, n_boot=100)
    resumed, fresh = FeatureSet.load(run_dir / FEATURES_FILE), FeatureSet.load(fresh_dir / FEATURES_FILE)
    order_r = np.lexsort((resumed.token_index, resumed.response_ids))
    order_f = np.lexsort((fresh.token_index, fresh.response_ids))
    assert resumed.names == fresh.names
    np.testing.assert_array_equal(resumed.labels[order_r], fresh.labels[order_f])
    np.testing.assert_allclose(resumed.X[order_r], fresh.X[order_f], equal_nan=True)

    counting.calls = 0
    run_evidence(counting, items, out_dir=run_dir, max_tokens=8, n_boot=100)
    assert counting.calls == 0  # todo hecho: solo recalibra y reporta


def test_resume_rejects_other_configuration(backend, items, tmp_path):
    extract(backend, items[:3], tmp_path, max_tokens=8)
    with pytest.raises(EvidenceError, match="max_tokens"):
        extract(backend, items[:3], tmp_path, max_tokens=9)
    with pytest.raises(EvidenceError, match="tiers"):
        extract(backend, items[:3], tmp_path, max_tokens=8, tiers=(Tier.F0,))


def test_corrupt_middle_line_is_an_error(backend, items, tmp_path):
    extract(backend, items[:3], tmp_path, max_tokens=8)
    path = tmp_path / ROWS_FILE
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join([lines[0], "{roto", lines[2]]) + "\n", encoding="utf-8")
    with pytest.raises(EvidenceError, match="dañada en el medio"):
        RunStore(tmp_path).load_rows()


def test_skipped_item_is_recorded_once(backend, items, tmp_path):
    long_item = validate_item({"id": "largo", "question": "x" * 2000, "answers": ["a"]})
    rows = extract(backend, [items[0], long_item], tmp_path, max_tokens=8)
    assert rows[1]["skipped"].startswith("InvalidRequestError") and rows[1]["tokens"] == []
    counting = CountingBackend(backend)
    extract(counting, [items[0], long_item], tmp_path, max_tokens=8)
    assert counting.calls == 0


def test_at_m08_single_class_fails_clearly(backend, items, tmp_path):
    """AT-M08: con una sola clase el experimento falla con mensaje claro, sin AUROC inventado."""
    only_wrong = [it for it in items if it["mode"] == "label" and it["label"] == 1]
    with pytest.raises(EvidenceError, match="una de cada clase"):
        run_evidence(backend, only_wrong, out_dir=tmp_path, max_tokens=8, n_boot=50)
    assert (tmp_path / ROWS_FILE).is_file() and not (tmp_path / "report.json").exists()


def test_judge_labels_spans_and_rejections(backend, items, tmp_path):
    """El juez etiqueta por fragmentos; un veredicto inválido deja la respuesta en IGNORE y se cuenta."""
    forced = [it for it in items if it["mode"] == "label"][:4]
    target = next(it for it in forced if it["response"][:3].strip())
    bad = next(it for it in forced if it is not target)

    def verdict(question, answer, evidence):
        if question == bad["question"]:
            return {"label": 1, "hallucinated_spans": ["no está en la respuesta"]}
        if question == target["question"]:
            return {"label": 1, "hallucinated_spans": [{"text": answer[:3], "occurrence": 1}]}
        return {"label": 0, "hallucinated_spans": []}

    judge = FakeJudge(verdict)
    rows = extract(backend, forced, tmp_path, max_tokens=8, judge=judge)
    by_id = {r["id"]: r for r in rows}
    first = by_id[target["id"]]
    assert first["labels"][:3] == [1, 1, 1] and set(first["labels"][3:]) == {0}
    assert first["label_info"]["computable_correct"] is (target["label"] == 0)
    assert set(by_id[bad["id"]]["labels"]) == {IGNORE_LABEL}
    assert by_id[bad["id"]]["label_info"]["rejected"] == "span_inexistente"
    assert RunStore(tmp_path).manifest()["label_source"] == "judge"

    report = relabel(tmp_path)  # vuelve a computable
    assert report["label_source"] == "computable" and report["rejected_by_reason"] == {}
    fs = build_features(tmp_path)
    assert fs.label_source == "computable" and fs.labeled_count() == len(fs.labels)
    report = relabel(tmp_path, judge=judge)
    assert report["rejected_by_reason"] == {"span_inexistente": 1}
    assert report["judge_vs_gold"]["agree"] + report["judge_vs_gold"]["disagree"] == 3
    sequential = [(r["id"], r["labels"]) for r in RunStore(tmp_path).load_rows()]
    assert relabel(tmp_path, judge=judge, workers=4) == report  # en paralelo: mismo reporte y orden
    assert [(r["id"], r["labels"]) for r in RunStore(tmp_path).load_rows()] == sequential
    assert json.loads((tmp_path / "labels_report.json").read_text(encoding="utf-8"))["n_rows"] == 4


def test_tiers_not_available_in_backend(backend, items, tmp_path):
    class NoH(CountingBackend):
        def capabilities(self):
            caps = self.inner.capabilities()
            return type(caps)(tiers=(Tier.F0, Tier.F1), streaming=True)

    with pytest.raises(EvidenceError, match="no produce los niveles"):
        extract(NoH(backend), items[:2], tmp_path, max_tokens=4)


# --- Experimento real (gate slow: Granite 1B-A400M, ~3 GB de RAM; correr solo y de a uno) ----------

GRANITE = os.environ.get("VIGIA_GRANITE_MODEL", "ibm-granite/granite-3.1-1b-a400m-instruct")


@pytest.mark.slow
def test_at_m11_at_m12_granite_evidence(tmp_path):
    """AT-M11 y AT-M12: MKQA-es con gold sobre Granite; AUROC de F0, F1, F0+F1 y baselines con IC 95 %
    y overhead p50/p95 publicado contra la meta de 5 %."""
    from vigia.cli import load_backend
    from vigia.datasets import DEFAULT_MKQA_PATH

    if not DEFAULT_MKQA_PATH.is_file():
        pytest.skip(f"falta {DEFAULT_MKQA_PATH}")
    items = load_mkqa_es(n=int(os.environ.get("VIGIA_EVIDENCE_N", "60")), seed=0)
    backend = load_backend(GRANITE)
    out = Path(os.environ.get("VIGIA_EVIDENCE_OUT", str(tmp_path)))
    report = run_evidence(
        backend, items, out_dir=out, max_tokens=16, bench_prompts=[it["messages"] for it in items[:2]]
    )
    for tier in ("F0", "F1", "F0+F1"):
        entry = report["tiers"][tier]
        assert entry["available"] and len(entry["token"]["auroc"]["ci"]) == 2
    assert report["bootstrap"]["level"] == pytest.approx(0.95)
    assert set(report["baselines"]) == {"out_entropy", "out_nll", "h_probe"}
    assert "latency" in report and {"p50", "p95"} <= set(report["latency"]["vigia"])
    assert (out / "report.html").is_file()


def test_arithmetic_items_have_exact_gold_and_are_reproducible():
    from vigia.datasets import arithmetic_items
    from vigia.labels import numeric_match

    items = arithmetic_items(30, seed=7)
    assert items == arithmetic_items(30, seed=7) and len({i["id"] for i in items}) == 30
    for it in items:
        q = it["question"].replace("¿", "").replace("?", "").split()
        a, op, b = int(q[2]), q[3], int(q[4])
        expected = {"+": a + b, "más": a + b, "-": a - b, "menos": a - b, "×": a * b, "por": a * b}[op]
        assert it["answers"] == [str(expected)] and numeric_match(str(expected), it["answers"][0])


@pytest.mark.parametrize(
    ("prediction", "answers", "kind", "expected"),
    [
        ("15", ["1,5"], "number", False),  # la normalización borraba la coma: EM falso positivo
        ("5 metros", ["1.234 metros"], "number_with_unit", False),  # F1 0,5 por la unidad
        ("12 o 15", ["12"], "arit+", False),  # el último número manda
        ("8 de mayo de 1944", ["8 de mayo de 1945"], "date", False),  # F1 0,8 con otro año
        ("15", ["1,5"], None, False),  # gold numérico sin tipo: también decide el número
        ("Es 1,5", ["1,5"], "number", True),
        ("Mide 1.234 metros", ["1.234 metros"], "number_with_unit", True),
        ("12 + 3 = 15", ["15"], "arit+", True),
        ("Terminó el 8 de mayo de 1945.", ["8 de mayo de 1945"], "date", True),
        ("Fue en 1945", ["1945"], "date", True),
        ("Apolo 12", ["Apolo 11"], "entity", False),  # texto con número: el número debe coincidir
        ("No lo sé", ["dos mil"], "number", False),  # gold no numérico: decide EM/F1
        ("dos mil", ["dos mil"], "number", True),
        # Gold sin tipo que es una fecha o un número con unidad: decide la fecha o el número.
        ("8 de mayo de 1946", ["8 de mayo de 1945"], None, False),  # F1 0,8 con otro año
        ("4,2 millones", ["3,2 millones"], None, False),  # F1 0,5 por la unidad
        ("Terminó el 8 de mayo de 1945.", ["8 de mayo de 1945", "1945-05-08"], None, True),
        ("Unos 3,2 millones", ["3,2 millones"], None, True),
        ("El 11 de agosto de 2017", ["08/11/2017"], None, False),  # gold ambiguo: sigue EM/F1
        ("08/11/2017", ["08/11/2017"], None, True),
        ("Apolo 12", ["Apolo 11"], None, False),  # gold de texto con número: debe coincidir
        ("Fue Johannes de Mol", ["Johannes Hubert de Mol"], None, True),  # gold sin números: F1 vale
    ],
)
def test_gold_correct_typed_items_ignore_em_and_f1(prediction, answers, kind, expected):
    """Regresión: "gold_correct marca como correctas respuestas numéricas y de fecha con valor
    incorrecto: EM y F1 >= 0,5 pisan la comparación tipada" y "gold_correct sin tipo sigue
    aceptando por F1 un año o un número distinto cuando el gold es una fecha o un número con
    unidad (arreglo previo incompleto)"."""
    assert gold_correct(prediction, answers, kind)[0] is expected


@pytest.mark.parametrize(
    ("prediction", "answers", "kind", "expected"),
    [
        ("mayo de 1946", ["mayo de 1945"], "date", False),
        ("junio de 1945", ["mayo de 1945"], "date", False),
        ("el año 1946", ["el año 1945"], None, False),
        ("Apolo 13", ["Apolo 11"], "entity", False),
        ("unos 300 km", ["unos 200 km"], None, False),
        ("15 km", ["unos 1,5 km"], None, False),  # la normalización borraba la coma
        ("Apolo 13", ["Apolo 11", "el Apolo once"], "entity", False),  # otro alias sin número
        # Lo correcto sigue valiendo.
        ("Fue en mayo de 1945", ["mayo de 1945"], "date", True),
        ("El 8 de mayo de 1945", ["mayo de 1945"], "date", True),
        ("mayo de 1945", ["mayo de 1945"], None, True),  # gold sin tipo de mes y año: es fecha
        ("junio de 1945", ["mayo de 1945"], None, False),
        ("el año 1945", ["el año 1945"], None, True),
        ("Fueron unos 200 km", ["unos 200 km"], None, True),
        ("El Apolo once", ["Apolo 11", "el Apolo once"], "entity", True),
    ],
)
def test_gold_correct_rejects_other_year_month_or_number(prediction, answers, kind, expected):
    """Regresión: "gold_correct acepta por F1 >= 0,5 respuestas con otro año, mes o número cuando el
    gold no es un número puro ni una fecha completa (el arreglo previo quedó incompleto)"."""
    assert gold_correct(prediction, answers, kind)[0] is expected


@pytest.mark.parametrize(
    ("prediction", "answers", "expected"),
    [
        ("En 1945, el 8 de mayo.", ["1945"], True),
        ("1945-05-08", ["1945"], True),
        ("8 de mayo de 1945", ["1945"], True),
        ("En el año 1945", ["el año 1945"], True),
        ("En 1944, el 8 de mayo.", ["1945"], False),
        ("1944-05-08", ["1945"], False),
        ("Entre 1944 y 1946", ["1945"], False),
    ],
)
def test_gold_year_only_date_compares_the_year(prediction, answers, expected):
    """Regresión: "Item de fecha con gold de solo el año marca como incorrecta una respuesta
    correcta que termina en el día ('En 1945, el 8 de mayo') o que está en formato ISO"."""
    assert gold_correct(prediction, answers, "date")[0] is expected


def test_resume_rejects_item_whose_content_changed(backend, items, tmp_path):
    """Regresión: "Al reanudar, extract reutiliza filas viejas aunque el contenido del item cambió
    (mismo id), y el reporte publica el hash del dataset nuevo"."""
    first = items[-3:]
    extract(backend, first, tmp_path, max_tokens=8)
    gold = first[-1]
    assert gold["mode"] == "gold"
    for change in ({"answers": ["otra respuesta"]}, {"group": "otro-grupo"}, {"type": "entity"}):
        edited = [*first[:-1], validate_item(item_to_json(gold) | change)]
        with pytest.raises(EvidenceError, match=rf"cambiaron de contenido \({gold['id']}\)"):
            extract(backend, edited, tmp_path, max_tokens=8)
    counting = CountingBackend(backend)
    rows = extract(counting, [*first, items[0]], tmp_path, max_tokens=8)  # agregar items sigue valiendo
    assert counting.calls == 1 and [r["id"] for r in rows] == [*(it["id"] for it in first), items[0]["id"]]


# --- Benchmark de código (modo code) ---------------------------------------------------------------

CODE_TESTS = "assert {name}(3) == 9\nassert {name}(-2) == 4\n"
CODE_RESPONSES = (
    "```python\ndef {name}(x):\n    return x * x\n```",  # pasa
    "```python\ndef {name}(x):\n    return x + x\n```",  # falla un assert
    "Claro:\n```python\ndef {name}(x):\n    return x ** 2\n```\nListo.",  # pasa
    "```python\ndef {name}(x)\n    return x * x\n```",  # error de sintaxis
    "```\ndef {name}(x):\n    return abs(x) * abs(x)\n```",  # pasa (bloque sin lenguaje)
    "No sé resolverlo.",  # sin bloque: el texto entero no define la función
)


def code_items(n: int = 6) -> list[dict]:
    items = []
    for i in range(n):
        name = f"square_{i}"
        items.append(
            validate_item(
                {
                    "id": f"code{i}",
                    "question": f"Implementá `{name}(x)` que devuelve x al cuadrado.",
                    "tests": CODE_TESTS.format(name=name),
                    "entry_point": name,
                    "type": "code",
                }
            )
        )
    return items


class _FixedStream:
    """Stream de un backend falso: eventos reales de teacher forcing sobre un texto fijo."""

    def __init__(self, events, text):
        self._it = iter(events)
        self.text = text
        self.finish_reason = "stop"

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._it)


class FixedTextBackend(CountingBackend):
    """El modelo diminuto no escribe código: `stream` devuelve una respuesta fija por pregunta."""

    def __init__(self, inner, responses):
        super().__init__(inner)
        self.responses = responses

    def stream(self, messages, *, tiers, **kwargs):
        text = self.responses.get(messages[-1]["content"])
        if text is None:
            return super().stream(messages, tiers=tiers, **kwargs)
        self.calls += 1
        return _FixedStream(self.inner.teacher_force(messages, text, tiers=tiers), text)


def fixed_code_backend(backend, items):
    responses = {}
    code = [it for it in items if it["mode"] == "code"]
    for i, it in enumerate(code):
        text = CODE_RESPONSES[i % len(CODE_RESPONSES)].format(name=it["entry_point"])
        responses[it["messages"][-1]["content"]] = text
    return FixedTextBackend(backend, responses)


def test_validate_item_code_mode():
    item = code_items(1)[0]
    assert item["mode"] == "code" and item["type"] == "code" and item["entry_point"] == "square_0"
    # El prompt de código no lleva la instrucción de respuesta corta.
    assert item["messages"] == [{"role": "user", "content": item["question"]}]
    assert DEFAULT_INSTRUCTION not in item["messages"][0]["content"]
    assert validate_item(item_to_json(item)) == item
    no_type = item_to_json(item)
    del no_type["type"]
    assert validate_item(no_type)["type"] == "code"


@pytest.mark.parametrize(
    ("extra", "fragment"),
    [
        ({"entry_point": None}, "tests' y 'entry_point' juntos"),
        ({"answers": ["9"]}, "tests' y 'entry_point' juntos"),
        ({"response": "x", "label": 0}, "tests' y 'entry_point' juntos"),
        ({"tests": "assert (\n"}, "no es código Python válido"),
        ({"entry_point": "no valido"}, "nombre de función"),
        ({"entry_point": "class"}, "nombre de función"),
        ({"tests": "x" * 60_000}, "supera"),
    ],
)
def test_validate_item_rejects_bad_code_items(extra, fragment):
    obj = item_to_json(code_items(1)[0]) | extra
    obj = {k: v for k, v in obj.items() if v is not None}
    with pytest.raises(DatasetError, match=fragment):
        validate_item(obj)


def test_other_modes_do_not_change_with_code_mode(items):
    for it in items:
        assert it["tests"] is None and it["entry_point"] is None
        assert it["mode"] in ("gold", "label")
        assert it["messages"] == question_messages(it["question"])


def test_load_humaneval_builds_spanish_code_items(tmp_path):
    import gzip

    from vigia.datasets import load_humaneval, read_jsonl, write_jsonl

    raw = [
        {
            "task_id": f"HumanEval/{i}",
            "prompt": f'def add_{i}(a: int, b: int) -> int:\n    """Suma a y b."""\n',
            "canonical_solution": "    return a + b\n",
            "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n",
            "entry_point": f"add_{i}",
        }
        for i in range(5)
    ]
    path = tmp_path / "HumanEval.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in raw)
    items = load_humaneval(path)
    assert [it["id"] for it in items] == [r["task_id"] for r in raw]
    first = items[0]
    assert first["mode"] == "code" and first["type"] == "code" and first["entry_point"] == "add_0"
    assert first["tests"].endswith("\ncheck(add_0)\n") and first["tests"].startswith(raw[0]["test"])
    question = first["messages"][0]["content"]
    assert "Implementá" in question and raw[0]["prompt"].strip() in question
    assert "SOLO con un bloque ```python```" in question
    sample = load_humaneval(path, n=3, seed=1)
    assert len(sample) == 3 and sample == load_humaneval(path, n=3, seed=1)
    out = tmp_path / "humaneval_es.jsonl"
    write_jsonl(out, items)
    assert read_jsonl(out) == items
    with pytest.raises(DatasetError, match="prepare_humaneval"):
        load_humaneval(tmp_path / "no_existe.jsonl.gz")


def test_run_evidence_without_permission_fails_before_generating(backend, items, tmp_path):
    from vigia.evidence import CodeExecNotAllowed

    counting = CountingBackend(backend)
    with pytest.raises(CodeExecNotAllowed, match="allow-code-exec"):
        run_evidence(counting, [*items[:3], *code_items(2)], out_dir=tmp_path, max_tokens=8)
    with pytest.raises(CodeExecNotAllowed):
        extract(counting, code_items(2), tmp_path, max_tokens=8)
    assert counting.calls == 0 and not (tmp_path / ROWS_FILE).exists()


def test_extract_labels_code_by_running_tests(backend, tmp_path):
    from vigia.evidence import CodeExecNotAllowed

    code = code_items(6)
    fake = fixed_code_backend(backend, code)
    rows = extract(fake, code, tmp_path, max_tokens=8, allow_code_exec=True)
    expected = ["passed", "failed", "passed", "error", "passed", "error"]
    assert [r["label_info"]["status"] for r in rows] == expected
    for row, status in zip(rows, expected, strict=True):
        assert row["source"] == "generated" and row["tokens"]
        scored = {v for v in row["labels"] if v != IGNORE_LABEL}
        assert scored == ({0} if status == "passed" else {1})
        assert row["label_info"]["correct"] is (status == "passed")
        assert isinstance(row["label_info"]["detail"], str)
    assert "AssertionError" in rows[1]["label_info"]["detail"]
    assert "SyntaxError" in rows[3]["label_info"]["detail"]

    # Re-etiquetar ejecuta código otra vez: también exige permiso.
    with pytest.raises(CodeExecNotAllowed):
        relabel(tmp_path)
    assert relabel(tmp_path, allow_code_exec=True)["n_rows"] == 6


def test_run_evidence_with_code_items_reports_by_type(backend, items, tmp_path):
    code = code_items(6)
    mixed = [*items, *code]
    fake = fixed_code_backend(backend, mixed)
    report = run_evidence(
        fake, mixed, out_dir=tmp_path, max_tokens=8, n_boot=100, seed=3, allow_code_exec=True
    )
    ev = report["evidence"]
    assert ev["code_status"] == {"error": 2, "failed": 1, "passed": 3}
    by_type = ev["model_quality"]["generated"]["by_type"]
    assert by_type["code"]["n"] == 6 and by_type["code"]["accuracy"] == pytest.approx(0.5)
    rows = {r["id"]: r for r in RunStore(tmp_path).load_rows()}
    for i, it in enumerate(code):
        scored = {v for v in rows[it["id"]]["labels"] if v != IGNORE_LABEL}
        assert scored == ({0} if i in (0, 2, 4) else {1})
    summary = (tmp_path / "summary.txt").read_text(encoding="utf-8")
    assert "code: exactitud 50,0 %" in summary
    assert "Tests de código: error 2, failed 1, passed 3" in summary
