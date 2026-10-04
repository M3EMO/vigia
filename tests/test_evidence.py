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
        ("Apolo 12", ["Apolo 11"], "entity", True),  # F1 0,5: regla documentada
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
    assert rows[1]["skipped"].startswith("ValueError") and rows[1]["tokens"] == []
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
