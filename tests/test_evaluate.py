import json
import re

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.metrics import log_loss as sk_log_loss

from vigia.calibrate import DEFAULT_ALPHAS, train_detector
from vigia.contracts import ModelInfo
from vigia.evaluate import (
    SingleClassError,
    auprc,
    aurc,
    auroc,
    brier,
    cluster_bootstrap_ci,
    ece,
    evaluate_report,
    log_loss,
    precision_at_recall,
    recall_at_fpr,
    reliability_bins,
    render_report_html,
    span_iou,
    threshold_metrics,
)

INFO = ModelInfo("<b>tiny</b>", "mixtral", 2, (0, 1), 4, 2, 100, "softmax_topk")


def _synthetic(seed=0, n_q=160, resp_per_q=2, tok=12, with_h=True):
    """Señal plantada con sesgo por respuesta (hay respuestas limpias y respuestas con errores)."""
    rng = np.random.default_rng(seed)
    n_resp = n_q * resp_per_q
    n = n_resp * tok
    groups = np.repeat([f"q{i}" for i in range(n_q)], resp_per_q * tok)
    responses = np.repeat([f"r{i}" for i in range(n_resp)], tok)
    offset = np.repeat(rng.normal(scale=2.0, size=n_resp), tok)
    a, b, h = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(2.5 * a - 1.5 * b + 0.8 * h - 3.0 + offset)))).astype(int)
    feats = {
        "out_entropy": a,
        "out_nll": 0.5 * a + rng.normal(size=n),
        "r_ent_n@0": b,
        "u_gini@0": rng.normal(size=n),
    }
    if with_h:
        feats["h_probe"] = h
    return feats, y, groups, responses


def test_auroc_and_auprc_match_sklearn_with_ties_and_weights():
    rng = np.random.default_rng(0)
    s = np.round(rng.random(400), 1)  # muchos empates
    y = (rng.random(400) < s).astype(int)
    w = rng.integers(0, 4, size=400).astype(float)
    assert auroc(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-12)
    assert auprc(y, s) == pytest.approx(average_precision_score(y, s), abs=1e-12)
    assert auroc(y, s, w) == pytest.approx(roc_auc_score(y, s, sample_weight=w), abs=1e-12)
    assert auprc(y, s, w) == pytest.approx(average_precision_score(y, s, sample_weight=w), abs=1e-12)
    p = np.clip(s, 0.01, 0.99)
    assert brier(y, p) == pytest.approx(brier_score_loss(y, p))
    assert log_loss(y, p) == pytest.approx(sk_log_loss(y, p))


def test_ece_and_reliability_hand_values():
    y, p = np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.6, 0.9])
    # Bins de igual masa {0,1; 0,4} y {0,6; 0,9}: (|0 - 0,5| + |2 - 1,5|) / 4 = 0,25.
    assert ece(y, p, n_bins=2) == pytest.approx(0.25)
    bins = reliability_bins(y, p, n_bins=2)
    assert bins[0] == {"mean_pred": pytest.approx(0.25), "frac_pos": 0.0, "count": 2, "lo": 0.1, "hi": 0.4}
    assert ece(np.array([0, 1] * 50), np.full(100, 0.5)) == pytest.approx(0.0)


def test_operating_point_metrics_hand_values():
    y = np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0])
    s = np.arange(10, 0, -1) / 10
    assert recall_at_fpr(y, s, 0.1) == pytest.approx(1 / 3)  # el 2º negativo ya da FPR 1/7 > 0,1
    assert precision_at_recall(y, s, 0.8) == pytest.approx(0.6)  # recall 1 en el 5º: 3/5
    m = threshold_metrics(y, s, 0.65)  # marca los 4 primeros: TP 2, FP 2, FN 1
    assert m["precision"] == pytest.approx(0.5) and m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(4 / 7) and m["flag_rate"] == pytest.approx(0.4)
    assert m["fnr"] == pytest.approx(1 / 3)


def test_span_iou_and_aurc_hand_values():
    flags = [1, 1, 0, 0, 0, 1]
    labs = [1, 0, 0, 0, 0, 1]
    resp = ["a", "a", "a", "b", "b", "c"]
    assert span_iou(flags, labs, resp) == pytest.approx((0.5 + 1.0 + 1.0) / 3)
    out = aurc([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.4])
    assert out["aurc"] == pytest.approx((0 + 0 + 1 / 3 + 2 / 4) / 4)
    assert out["e_aurc"] == pytest.approx(0.0)
    assert aurc([0, 1, 0, 1], [0.9, 0.1, 0.8, 0.2])["e_aurc"] > 0


def test_cluster_bootstrap_ci_contains_point_estimate():
    feats, y, groups, _ = _synthetic(seed=1, n_q=80)
    s = feats["out_entropy"]
    lo, hi = cluster_bootstrap_ci(auroc, y, s, groups, n_boot=200, seed=1)
    assert lo < auroc(y, s) < hi


def test_at_m08_single_class_fails_with_clear_message():
    """AT-M08: con etiquetas de una sola clase el reporte falla con mensaje claro."""
    feats, y, groups, responses = _synthetic(seed=2, n_q=40)
    with pytest.raises(SingleClassError, match="una sola clase"):
        evaluate_report(feats, np.zeros_like(y), groups, responses, INFO, n_boot=10)
    with pytest.raises(SingleClassError, match="una sola clase"):
        auroc(np.ones(5), np.arange(5))
    with pytest.raises(SingleClassError):
        auprc(np.zeros(5), np.arange(5))


@pytest.fixture(scope="module")
def report():
    feats, y, groups, responses = _synthetic(seed=3)
    return evaluate_report(feats, y, groups, responses, INFO, n_boot=300, seed=3)


def test_at_m09_report_has_metrics_cis_and_baselines(report):
    """AT-M09: el reporte trae las métricas listadas, IC por bootstrap de cluster y los 3 baselines."""
    assert report["bootstrap"] == {"n_boot": 300, "level": 0.95, "seed": 3, "cluster": "pregunta"}
    tok, resp = report["detector"]["token"], report["detector"]["response"]
    for key in (
        "auroc",
        "auprc",
        "ece",
        "brier",
        "log_loss",
        "recall_at_fpr",
        "precision_at_recall_0.8",
        "f1",
    ):
        lo, hi = tok[key]["ci"]
        assert lo <= tok[key]["value"] <= hi, key
    for key in ("auroc", "auprc", "ece", "brier", "span_iou", "aurc", "e_aurc"):
        assert resp[key]["ci"] is not None, key
    assert tok["auroc"]["value"] > 0.8  # señal plantada; el valor exacto depende del split (estratificado)
    assert set(report["baselines"]) == {"out_entropy", "out_nll", "h_probe"}
    for entry in report["baselines"].values():
        assert entry["available"]
        delta = entry["token"]["delta_auroc_vs_detector"]
        assert delta["ci"][0] <= delta["value"] <= delta["ci"][1]
    assert report["baselines"]["out_nll"]["token"]["delta_auroc_vs_detector"]["value"] < 0
    assert set(report["tiers"]) == {"F0", "F1", "F0+F1", "todos"}
    assert all(t["available"] for t in report["tiers"].values())
    assert report["detector_id"] == report["tiers"]["F0+F1"]["id"]
    assert 0 < report["dataset"]["prevalence_token"] < 1
    assert [c["alpha"] for c in report["crc"]] == list(DEFAULT_ALPHAS)
    # Frontera por respuesta: cal (la que eligió el alpha recomendado) y test (auditoría).
    pareto = report["curves"]["pareto"]
    rec = pareto["recommended_alpha"]
    assert rec is not None and report["alpha"] == rec  # default_alpha del detector = recomendado
    assert rec in {p["alpha"] for p in pareto["cal"]} and pareto["test"]
    for front in (pareto["cal"], pareto["test"]):
        assert all(set(p) == {"alpha", "tau", "recall", "false_alarm"} for p in front)
        assert [p["false_alarm"] for p in front] == sorted(p["false_alarm"] for p in front)
    for c in report["crc"]:
        assert 0 <= c["observed_response_recall"] <= 1 and 0 <= c["observed_response_false_alarm"] <= 1
        assert c["expected_response_recall"] is not None
    json.dumps(report, allow_nan=False)


def test_at_m09_report_html_is_self_contained_with_charts(report):
    """AT-M09: el HTML del reporte es autocontenido y usa los gráficos SVG accesibles."""
    page = render_report_html(report)
    assert page.startswith("<!doctype html>")
    assert page.count('role="img"') >= 4 and page.count("<details") >= 4
    assert "Baseline h_probe" in page and "Nivel F0+F1" in page and "IC 95 %" in page
    assert "<script" not in page
    assert not re.search(r'(?:src|href)\s*=\s*"(?:https?:)?//', page)
    assert "&lt;b&gt;tiny&lt;/b&gt;" in page and "<b>tiny</b>" not in page
    assert "Frontera en calibración (elige el alpha recomendado)" in page
    assert "Frontera en test (auditoría)" in page and "(recomendado)" in page
    assert "Detección de respuestas malas (test / cal)" in page
    with_latency = render_report_html(
        dict(
            report,
            latency={
                "baseline": {"p50": 10, "p95": 12, "p99": 15},
                "vigia": {"p50": 10.3, "p95": 12.9, "p99": 15.2},
            },
        )
    )
    assert with_latency.count('role="img"') == page.count('role="img"') + 1


def test_report_marks_missing_probe_and_uses_given_detector():
    feats, y, groups, responses = _synthetic(seed=4, n_q=100, with_h=False)
    det, split = train_detector(feats, y, groups, responses, INFO, seed=4, c_grid=(1.0,))
    rep = evaluate_report(
        feats, y, groups, responses, INFO, detector=det, split=split["groups"], n_boot=50, seed=4
    )
    assert rep["detector_id"] == det.id
    assert rep["baselines"]["h_probe"] == {
        "available": False,
        "reason": "sin features de este nivel en el dataset",
    }
    assert "No disponible" in render_report_html(rep)
    other = ModelInfo("otro", "mixtral", 2, (0, 1), 4, 2, 100, "softmax_topk")
    with pytest.raises(Exception, match="otro modelo"):
        evaluate_report(feats, y, groups, responses, other, detector=det, split=split["groups"], n_boot=10)


def test_aurc_is_invariant_to_row_order_with_tied_scores():
    """Regresión: "AURC y E-AURC dependen del orden de las filas cuando hay puntajes empatados"."""
    a, b = aurc([1.0, 0.0], [0.5, 0.5]), aurc([0.0, 1.0], [0.5, 0.5])
    assert a == pytest.approx(b) and a["aurc"] == pytest.approx(0.5)
    assert a["e_aurc"] == pytest.approx(0.5 - 0.25)
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 40).astype(float)
    p = rng.integers(0, 4, 40) / 3  # muchos empates
    ref = aurc(y, p)
    for _ in range(5):
        perm = rng.permutation(40)
        assert aurc(y[perm], p[perm]) == pytest.approx(ref)
    # sin empates el valor no cambia respecto del cálculo fila por fila
    assert aurc([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.4])["aurc"] == pytest.approx((0 + 0 + 1 / 3 + 2 / 4) / 4)


def test_e_aurc_with_integer_weights_equals_expanded_dataset():
    """Regresión: "E-AURC del bootstrap no equivale al E-AURC del dataset remuestreado: la
    referencia óptima trata una fila con peso w como un solo bloque"."""
    from vigia.evaluate import _aurc_stats, _digamma

    y = np.array([0, 1, 1, 0, 1.0])
    p = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    w = np.array([1, 3, 1, 2, 1.0])
    rep = np.repeat(np.arange(5), w.astype(int))
    via_w = {k: float(v[0]) for k, v in _aurc_stats(y, p, w[None, :]).items()}
    assert via_w == pytest.approx(aurc(y[rep], p[rep]), abs=1e-12)
    rng = np.random.default_rng(5)
    y, p = rng.integers(0, 2, 60).astype(float), rng.random(60)
    W = rng.multinomial(60, np.full(60, 1 / 60), size=20).astype(float)
    stats = _aurc_stats(y, p, W)
    for b in range(len(W)):
        rep = np.repeat(np.arange(60), W[b].astype(int))
        expanded = aurc(y[rep], p[rep])
        assert stats["e_aurc"][b] == pytest.approx(expanded["e_aurc"], abs=1e-12)
        assert stats["aurc"][b] == pytest.approx(expanded["aurc"], abs=1e-12)
    # digamma(n + 1) = H_n - gamma de Euler
    euler = 0.5772156649015329
    assert _digamma(np.array([1.0, 5.0]))[0] == pytest.approx(-euler, abs=1e-12)
    assert _digamma(np.array([5.0]))[0] == pytest.approx(1 + 1 / 2 + 1 / 3 + 1 / 4 - euler, abs=1e-12)
