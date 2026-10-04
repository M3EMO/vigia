"""Verificación matemática y estadística con implementaciones independientes (ronda 1, lente ML).

Cada test compara el código de Vigía contra un oráculo que no comparte código con él:
scikit-learn, scipy, una fórmula escrita a mano o una simulación con verdad conocida.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.optimize import minimize
from scipy.spatial.distance import jensenshannon
from scipy.special import log_softmax as sp_log_softmax
from scipy.special import softmax as sp_softmax
from sklearn import metrics as skm
from sklearn.linear_model import LogisticRegression

from vigia import evaluate as E
from vigia import features as fx
from vigia.calibrate import (
    crc_threshold,
    fit_logistic,
    fit_platt,
    split_groups,
    split_masks,
    train_detector,
)
from vigia.contracts import AGGREGATES, ModelInfo, Tier, agg_feature, layer_feature
from vigia.detector import Detector, DetectorMismatchError, artifact_sha256
from vigia.labels import (
    IGNORE_LABEL,
    date_match,
    exact_match,
    extract_numbers,
    normalize_es,
    numeric_match,
    parse_dates_es,
    parse_number_es,
    spans_to_token_labels,
    token_f1,
)

sys.path.insert(0, str(Path(__file__).parent))

INFO = ModelInfo(
    model_id="tiny/verify",
    model_type="mixtral",
    n_layers=2,
    moe_layers=(0, 1),
    n_experts=4,
    top_k=2,
    vocab_size=64,
    gating="softmax_topk",
)


# --- Métricas de evaluate contra scikit-learn ---------------------------------------------------------


def _metric_cases(n_cases: int = 120):
    rng = np.random.default_rng(0)
    for it in range(n_cases):
        n = int(rng.integers(5, 150))
        prev = float(rng.choice([0.02, 0.1, 0.5, 0.9]))  # desbalance fuerte en los dos sentidos
        y = (rng.random(n) < prev).astype(int)
        if y.min() == y.max():
            y[0] = 1 - y[0]
        s = np.round(rng.random(n) * float(rng.choice([3, 10, 1000]))) / 1000  # muchos empates
        w = rng.integers(0, 4, n).astype(float) if it % 2 else None
        if w is not None and (np.sum(w * y) == 0 or np.sum(w * (1 - y)) == 0):
            w = None
        yield y, s, w


def test_ranking_metrics_match_sklearn_with_ties_weights_and_imbalance():
    for y, s, w in _metric_cases():
        assert E.auroc(y, s, w) == pytest.approx(skm.roc_auc_score(y, s, sample_weight=w), abs=1e-12)
        assert E.auprc(y, s, w) == pytest.approx(
            skm.average_precision_score(y, s, sample_weight=w), abs=1e-12
        )
        fpr, tpr, _ = skm.roc_curve(y, s, drop_intermediate=False)
        assert E.recall_at_fpr(y, s) == pytest.approx(tpr[fpr <= 0.1 + 1e-12].max(), abs=1e-12)
        prec, rec, _ = skm.precision_recall_curve(y, s, drop_intermediate=False)
        assert E.precision_at_recall(y, s, 0.8) == pytest.approx(prec[rec >= 0.8 - 1e-12].max(), abs=1e-12)


def test_probability_metrics_match_sklearn():
    for y, s, w in _metric_cases():
        p = np.clip(s, 1e-15, 1 - 1e-15)
        assert E.brier(y, p, w) == pytest.approx(skm.brier_score_loss(y, p, sample_weight=w), abs=1e-12)
        assert E.log_loss(y, p, w) == pytest.approx(
            skm.log_loss(y, p, sample_weight=w, labels=[0, 1]), abs=1e-10
        )


def _ece_oracle(y, p, n_bins=15, w=None):
    """Bins de igual masa por rango; un empate queda entero en el bin de su primera aparición."""
    n = len(p)
    w = np.ones(n) if w is None else w
    order = sorted(range(n), key=lambda i: (p[i], i))
    first: dict[float, int] = {}
    bin_of = {}
    for rank, i in enumerate(order):
        first.setdefault(p[i], rank * n_bins // n)
        bin_of[i] = first[p[i]]
    total = 0.0
    for b in set(bin_of.values()):
        total += abs(sum(w[i] * (y[i] - p[i]) for i in range(n) if bin_of[i] == b))
    return total / w.sum()


def test_ece_matches_equal_mass_oracle():
    rng = np.random.default_rng(1)
    for it in range(120):
        n = int(rng.integers(1, 100))
        y = (rng.random(n) < 0.3).astype(float)
        p = np.round(rng.random(n), int(rng.integers(1, 4)))
        w = rng.integers(0, 3, n).astype(float) if it % 2 else None
        if w is not None and w.sum() == 0:
            w = None
        assert E.ece(y, p, w=w) == pytest.approx(_ece_oracle(y, p, w=w), abs=1e-12)


def test_ece_closed_forms():
    p = np.full(100, 0.3)
    assert E.ece(np.r_[np.ones(30), np.zeros(70)], p) == pytest.approx(0.0, abs=1e-12)
    assert E.ece(np.ones(100), p) == pytest.approx(0.7)


def test_span_iou_per_response_by_hand():
    flags = [1, 1, 0, 0, 1, 0, 0]
    labels = [1, 0, 1, 0, 0, 0, 0]
    resp = ["a", "a", "a", "b", "b", "c", "c"]
    ids, iou = E.span_iou_per_response(flags, labels, resp)
    assert ids.tolist() == ["a", "b", "c"]
    # a: inter 1, unión 3; b: inter 0, unión 1; c: ambos vacíos = 1.
    assert iou.tolist() == pytest.approx([1 / 3, 0.0, 1.0])
    assert E.span_iou(flags, labels, resp) == pytest.approx((1 / 3 + 0 + 1) / 3)


def test_report_point_values_equal_public_metrics():
    """El motor vectorizado del reporte da los mismos valores puntuales que las funciones públicas."""
    rng = np.random.default_rng(5)
    y = (rng.random(300) < 0.2).astype(float)
    p = np.clip(0.2 + 0.4 * y + 0.2 * rng.normal(size=300), 0.001, 0.999)
    lv = E._Level(y, np.arange(300) // 10)
    lv.add("x", p, p >= 0.5)
    st = {k: float(v[0]) for k, v in lv.stats(np.ones((1, 300)))["x"].items()}
    assert st["auroc"] == pytest.approx(skm.roc_auc_score(y, p), abs=1e-12)
    assert st["auprc"] == pytest.approx(skm.average_precision_score(y, p), abs=1e-12)
    assert st["brier"] == pytest.approx(skm.brier_score_loss(y, p), abs=1e-12)
    assert st["ece"] == pytest.approx(E.ece(y, p), abs=1e-12)
    flags = p >= 0.5
    assert st["recall"] == pytest.approx(skm.recall_score(y, flags), abs=1e-12)
    assert st["precision"] == pytest.approx(skm.precision_score(y, flags), abs=1e-12)
    assert st["f1"] == pytest.approx(skm.f1_score(y, flags), abs=1e-12)


def test_cluster_bootstrap_covers_true_prevalence():
    """Cobertura aproximada del IC 95 % por cluster con efecto aleatorio por pregunta."""

    def gen(rng, n_groups):
        size = rng.integers(3, 15, n_groups)
        u = rng.normal(size=n_groups)
        c = np.repeat(np.arange(n_groups), size)
        y = (rng.random(len(c)) < 1 / (1 + np.exp(-(u[c] - 1)))).astype(float)
        return y, c

    truth = gen(np.random.default_rng(999), 200_000)[0].mean()
    prevalence = lambda y, s, w: float(np.sum(w * y) / np.sum(w))  # noqa: E731
    hits = 0
    reps = 120
    for rep in range(reps):
        y, c = gen(np.random.default_rng(rep), 60)
        lo, hi = E.cluster_bootstrap_ci(prevalence, y, y, c, n_boot=300, seed=rep)
        hits += lo <= truth <= hi
    assert 0.85 <= hits / reps <= 0.99


# --- Regresión logística, Platt y CRC -----------------------------------------------------------------


def test_fit_logistic_matches_sklearn_with_weights():
    rng = np.random.default_rng(2)
    for it in range(20):
        n, d = int(rng.integers(40, 300)), int(rng.integers(1, 6))
        X = rng.normal(size=(n, d)) * rng.uniform(0.1, 5, d)
        y = (rng.random(n) < 1 / (1 + np.exp(-(X @ rng.normal(size=d) - 1.5)))).astype(float)
        if y.min() == y.max():
            continue
        c = float(rng.choice([0.01, 0.1, 1.0, 10.0]))
        sw = rng.uniform(0.2, 3, n) if it % 2 else None
        coef, bias = fit_logistic(X, y, l2=1 / c, sample_weight=sw)
        sk = LogisticRegression(C=c, tol=1e-12, max_iter=10_000).fit(X, y, sample_weight=sw)
        assert np.r_[coef, bias] == pytest.approx(np.r_[sk.coef_[0], sk.intercept_[0]], abs=1e-5)


def test_fit_platt_matches_scipy_minimum():
    rng = np.random.default_rng(3)
    s = rng.normal(size=400)
    y = (rng.random(400) < 1 / (1 + np.exp(-(2 * s - 1)))).astype(float)
    a, b = fit_platt(s, y)
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    t = np.where(y > 0.5, (n_pos + 1) / (n_pos + 2), 1 / (n_neg + 2))

    def nll(v):
        z = v[0] * s + v[1]
        return np.sum(np.logaddexp(0, z) - t * z)

    ref = minimize(nll, [1.0, 0.0], method="BFGS", options={"gtol": 1e-10}).x
    assert (a, b) == pytest.approx(tuple(ref), abs=1e-5)


def _crc_data(rng, n_resp, prev_resp, tok_prev):
    lengths = rng.integers(1, 30, n_resp)
    r = np.repeat(np.arange(n_resp), lengths)
    has = rng.random(n_resp) < prev_resp
    y = (rng.random(len(r)) < tok_prev) & has[r]
    s = rng.normal(size=len(r)) + 1.0 * y
    return 1 / (1 + np.exp(-s)), y.astype(int), r.astype(str)


def _fnr_per_response(p, y, r, tau):
    pos = y == 1
    _, inv = np.unique(r[pos], return_inverse=True)
    return float((np.bincount(inv, weights=p[pos] < tau) / np.bincount(inv)).mean())


@pytest.mark.parametrize(
    "n_cal, prev_resp, tok_prev, alpha",
    [(200, 0.5, 0.2, 0.1), (30, 0.9, 0.3, 0.1), (400, 0.05, 0.5, 0.2), (150, 0.5, 0.3, 0.05)],
)
def test_crc_fnr_guarantee_over_seeds(n_cal, prev_resp, tok_prev, alpha):
    """E[FNR por respuesta] <= alpha en 200 semillas, también con pocas respuestas erróneas."""
    values = []
    for seed in range(200):
        rng = np.random.default_rng(seed)
        pc, yc, rc = _crc_data(rng, n_cal, prev_resp, tok_prev)
        if yc.sum() == 0:
            continue
        tau = crc_threshold(pc, yc, rc, alpha).tau
        values.append(_fnr_per_response(*_crc_data(rng, 400, prev_resp, tok_prev), tau))
    se = np.std(values) / math.sqrt(len(values))
    assert np.mean(values) <= alpha + 2 * se


def test_crc_threshold_is_the_largest_feasible_candidate():
    rng = np.random.default_rng(4)
    p, y, r = _crc_data(rng, 80, 0.6, 0.3)
    alpha = 0.1
    res = crc_threshold(p, y, r, alpha)
    pos = y == 1
    _, inv, counts = np.unique(r[pos], return_inverse=True, return_counts=True)
    n = len(counts)

    def bound(tau):
        loss = np.bincount(inv, weights=p[pos] < tau) / counts
        return (loss.sum() + 1) / (n + 1)

    assert res.feasible and res.n == n
    assert bound(res.tau) <= alpha + 1e-12
    bigger = np.sort(np.unique(p[pos]))
    bigger = bigger[bigger > res.tau]
    assert all(bound(t) > alpha for t in bigger)


def test_step_tau_keeps_fnr_in_simulation():
    """`Detector.tau` usa el tau del alpha guardado inferior; en simulación la FNR sigue <= alpha."""
    alphas = (0.05, 0.1, 0.2, 0.3)
    values = []
    for seed in range(100):
        rng = np.random.default_rng(seed)
        pc, yc, rc = _crc_data(rng, 150, 0.5, 0.3)
        det = _det_with_thresholds([(a, crc_threshold(pc, yc, rc, a).tau) for a in alphas])
        values.append(_fnr_per_response(*_crc_data(rng, 1000, 0.5, 0.3), det.tau(0.15)))
    assert np.mean(values) <= 0.15 + 2 * np.std(values) / math.sqrt(len(values))


def _det_with_thresholds(pairs):
    return Detector.build(
        model=INFO.fingerprint(),
        feature_names=["out_nll"],
        mean=[0.0],
        std=[1.0],
        weights=[1.0],
        bias=0.0,
        platt={"a": 1.0, "b": 0.0},
        response_calibrator={"a": 1.0, "b": 0.0},
        thresholds=[
            {"alpha": a, "tau": t, "expected_flag_rate": 0.5, "expected_precision": None} for a, t in pairs
        ],
        default_alpha=pairs[0][0],
        metrics={},
        training={
            "dataset_sha256": "x",
            "n_tokens": 1,
            "n_responses": 1,
            "label_source": "computable",
            "judge": None,
            "created_at": "2026-10-04T00:00:00+00:00",
        },
    )


def _bimodal_crc_data(rng, n):
    """Tokens erróneos bimodales: 15 % con p ~ 0,0015 y el resto en [0,5; 0,99] (CDF cóncava)."""
    p, y, r = [], [], []
    for i in range(n):
        low = rng.random(4) < 0.15
        p += np.where(low, rng.uniform(0.001, 0.002, 4), rng.uniform(0.5, 0.99, 4)).tolist()
        p += rng.uniform(0, 0.6, 6).tolist()
        y += [1] * 4 + [0] * 6
        r += [f"r{i}"] * 10
    return np.array(p), np.array(y), np.array(r)


def test_tau_between_stored_alphas_keeps_fnr_with_concave_cdf():
    """Regresión: "Detector.tau interpola τ linealmente entre alphas guardados y rompe la garantía
    FNR <= alpha que prometen el docstring y el panel". Con la CDF cóncava del repro, el tau
    interpolado para alpha = 0,12 daba FNR ~0,149; el escalón queda <= alpha."""
    alphas = (0.05, 0.1, 0.2, 0.3)
    values, interpolated = [], []
    for seed in range(30):
        rng = np.random.default_rng(seed)
        pc, yc, rc = _bimodal_crc_data(rng, 300)
        taus = [crc_threshold(pc, yc, rc, a).tau for a in alphas]
        det = _det_with_thresholds(list(zip(alphas, taus, strict=True)))
        test = _bimodal_crc_data(rng, 1000)
        values.append(_fnr_per_response(*test, det.tau(0.12)))
        interpolated.append(_fnr_per_response(*test, float(np.interp(0.12, alphas, taus))))
    assert np.mean(interpolated) > 0.13  # el escenario sí rompe la interpolación
    assert np.mean(values) <= 0.12 + 2 * np.std(values) / math.sqrt(len(values))


def _synthetic(seed=0, n_groups=60, nan_rate=0.0):
    rng = np.random.default_rng(seed)
    feats = {"out_entropy": [], "out_nll": [], "r_ent_n@0": []}
    labels, groups, resp = [], [], []
    for g in range(n_groups):
        for k in range(2):
            n = int(rng.integers(4, 12))
            bad = rng.random() < 0.4
            yy = (rng.random(n) < 0.5) & bad
            feats["out_entropy"] += list(rng.normal(size=n) + 1.5 * yy)
            feats["out_nll"] += list(rng.normal(size=n) + 1.0 * yy)
            feats["r_ent_n@0"] += list(rng.normal(size=n))
            labels += yy.astype(int).tolist()
            groups += [f"g{g}"] * n
            resp += [f"g{g}r{k}"] * n
    feats = {k: np.asarray(v) for k, v in feats.items()}
    if nan_rate:
        for v in feats.values():
            v[rng.random(len(v)) < nan_rate] = np.nan
    return feats, np.asarray(labels), np.asarray(groups), np.asarray(resp)


def test_train_detector_splits_have_no_leakage():
    feats, y, g, r = _synthetic()
    det, rep = train_detector(feats, y, g, r, INFO, seed=3)
    sets = {k: set(v) for k, v in rep["groups"].items()}
    assert not (sets["train"] & sets["cal"] or sets["train"] & sets["test"] or sets["cal"] & sets["test"])
    assert sets["train"] | sets["cal"] | sets["test"] == set(g)
    split = split_groups(g, seed=3, labels=y)  # train_detector estratifica por clase
    assert split == rep["groups"]
    masks = split_masks(g, split)
    for name, m in masks.items():
        assert set(r[m]).isdisjoint(set(r[~m])), name  # ninguna respuesta cruza splits


def test_runtime_detector_reproduces_training_with_nan_imputation():
    """El z-score con NaN imputado en entrenamiento y la media imputada en runtime coinciden."""
    feats, y, g, r = _synthetic(nan_rate=0.15)
    det, rep = train_detector(feats, y, g, r, INFO, seed=1)
    cal = np.isin(g, rep["groups"]["cal"])
    X = det.matrix_from_features({k: v[cal] for k, v in feats.items()}, int(cal.sum()))
    assert np.isnan(X).any()
    p = det.score_matrix(X)
    cal_metrics = det.artifact["metrics"]["cal"]
    assert cal_metrics["auroc"] == pytest.approx(skm.roc_auc_score(y[cal], p), abs=1e-12)
    assert cal_metrics["brier"] == pytest.approx(skm.brier_score_loss(y[cal], p), abs=1e-12)
    # Manual: p = sigmoid(a * ((x_imputado - mean) / std) . w + b0) + b)
    art = det.artifact
    Xi = np.where(np.isnan(X), np.asarray(art["mean"]), X)
    z = ((Xi - art["mean"]) / np.asarray(art["std"])) @ np.asarray(art["weights"]) + art["bias"]
    manual = 1 / (1 + np.exp(-(art["platt"]["a"] * z + art["platt"]["b"])))
    assert p == pytest.approx(manual, abs=1e-12)


def test_detector_hash_and_tau_monotone():
    feats, y, g, r = _synthetic()
    det, _ = train_detector(feats, y, g, r, INFO, seed=2)
    art = det.artifact
    assert art["sha256"] == artifact_sha256(art)
    assert Detector.loads(det.to_json()).score_matrix(np.zeros((1, 3))) == pytest.approx(
        det.score_matrix(np.zeros((1, 3))), abs=0
    )
    for key in ("bias", "default_alpha"):
        bad = dict(art)
        bad[key] = art[key] + 1e-9 if key == "bias" else 0.2
        with pytest.raises(DetectorMismatchError):
            Detector(bad)
    grid = np.linspace(min(t["alpha"] for t in art["thresholds"]), 0.99, 200)
    taus = [det.tau(a) for a in grid]
    assert all(b >= a for a, b in zip(taus, taus[1:], strict=False))
    for t in art["thresholds"]:
        assert det.tau(t["alpha"]) == t["tau"]
    with pytest.raises(ValueError):
        det.tau(0.01)


# --- Features contra las fórmulas de docs/spec/matriz-features.md -------------------------------------


def test_output_features_match_formulas():
    rng = np.random.default_rng(6)
    z = rng.normal(size=(50, 300)) * 3
    tgt = rng.integers(0, 300, 50)
    f = fx.output_features(z, tgt)
    q = sp_softmax(z, axis=-1)
    assert f["out_entropy"] == pytest.approx(-(q * np.log(q)).sum(-1), abs=1e-10)
    assert f["out_nll"] == pytest.approx(-sp_log_softmax(z, axis=-1)[np.arange(50), tgt], abs=1e-10)
    qs = -np.sort(-q, axis=-1)
    assert f["out_margin"] == pytest.approx(qs[:, 0] - qs[:, 1], abs=1e-12)
    top = qs[:, :20] / qs[:, :20].sum(-1, keepdims=True)
    assert f["out_top20_entropy"] == pytest.approx(-(top * np.log(top)).sum(-1), abs=1e-10)


def test_router_features_match_formulas_and_bounds():
    rng = np.random.default_rng(7)
    for gating, k, kw in (
        ("softmax_topk", 2, {"renormalize": True}),
        ("softmax_topk", 3, {}),
        ("topk_softmax", 2, {}),
    ):
        z = rng.normal(size=(40, 8)) * 2
        st = fx.router_stats(z, gating, k, **kw)
        p = sp_softmax(z, axis=-1)
        assert st["r_ent_n"] == pytest.approx(-(p * np.log(p)).sum(-1) / np.log(8), abs=1e-12)
        idx = np.argsort(-z, axis=-1)[:, :k]
        assert np.array_equal(np.sort(st["idx"], -1), np.sort(idx, -1))
        assert st["r_topk_mass"] == pytest.approx(np.take_along_axis(p, idx, -1).sum(-1), abs=1e-12)
        zs = -np.sort(-z, -1)
        assert st["r_margin12"] == pytest.approx(zs[:, 0] - zs[:, 1])
        assert st["r_margin_kk1"] == pytest.approx(zs[:, k - 1] - zs[:, k])
        wn = st["w"] / st["w"].sum(-1, keepdims=True)
        assert st["r_ent_k"] == pytest.approx(-(wn * np.log(wn)).sum(-1) / np.log(k), abs=1e-12)
        for name in ("r_ent_n", "r_ent_k", "r_topk_mass"):
            assert np.all((st[name] >= -1e-12) & (st[name] <= 1 + 1e-12)), name


def test_usage_closed_forms():
    n = 8
    uniform = np.ones(n)
    one_hot = np.eye(n)[2] * 5
    u = fx.usage_stats(np.stack([uniform, one_hot]))
    assert u["u_gini"] == pytest.approx([1 - 1 / n, 0.0])
    assert u["u_ihhi"] == pytest.approx([1.0, 1 / n])
    assert u["u_ent"] == pytest.approx([1.0, 0.0])
    counts = np.array([3.0, 1.0, 0.0, 4.0])
    q = counts / counts.sum()
    hhi = (q * q).sum()
    st = fx.usage_stats(counts)
    assert st["u_gini"] == pytest.approx(1 - hhi)
    assert st["u_ihhi"] == pytest.approx(1 / hhi / 4)
    assert st["u_ent"] == pytest.approx(-(q[q > 0] * np.log(q[q > 0])).sum() / np.log(4))
    assert np.isnan(fx.usage_stats(np.zeros(4))["u_gini"])
    assert fx.route_switch(np.array([[0, 1]]), np.array([[1, 3]])).tolist() == [0.5]


def test_weighted_jsd_matches_scipy_and_is_bounded():
    rng = np.random.default_rng(8)
    p = rng.dirichlet(np.ones(30), size=(100, 2))
    js = fx.weighted_jsd(p, np.full((100, 2), 0.5))
    ref = np.array([jensenshannon(a, b) ** 2 for a, b in p]) / np.log(2)
    assert js == pytest.approx(ref, abs=1e-10)
    for k in (2, 3, 5):
        pk = rng.dirichlet(np.full(20, 0.05), size=(200, k))  # casi disjuntas: JSD cerca del tope
        w = rng.dirichlet(np.ones(k), size=200)
        v = fx.weighted_jsd(pk, w)
        assert np.all((v >= 0) & (v <= 1))
    disjoint = np.eye(4)[None, :2]
    assert fx.weighted_jsd(disjoint, np.array([[0.5, 0.5]]))[0] == pytest.approx(1.0)
    same = np.repeat(p[:1, :1], 3, axis=1)
    assert fx.weighted_jsd(same, np.ones((1, 3)))[0] == pytest.approx(0.0, abs=1e-12)


def test_sigmoid_group_route_matches_deepseek_router_with_groups():
    """Selección por grupos (n_group=2, topk_group=1) idéntica al router real de transformers."""
    import transformers as tf
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3TopkRouter

    cfg = tf.DeepseekV3Config(
        hidden_size=16, n_routed_experts=8, num_experts_per_tok=2, n_group=2, topk_group=1,
        norm_topk_prob=True, routed_scaling_factor=2.5,
    )  # fmt: skip
    torch.manual_seed(0)
    router = DeepseekV3TopkRouter(cfg)
    with torch.no_grad():
        router.weight.normal_()
        router.e_score_correction_bias.copy_(torch.linspace(-0.2, 0.3, 8))
        logits, w_ref, idx_ref = router(torch.randn(64, 16))
    idx, w = fx.route(
        logits.numpy(), "sigmoid_group", 2, renormalize=True,
        correction_bias=router.e_score_correction_bias.numpy(), n_group=2, topk_group=1, scaling=2.5,
    )  # fmt: skip
    order = np.argsort(idx_ref.numpy(), -1)
    assert np.array_equal(np.sort(idx, -1), np.take_along_axis(idx_ref.numpy(), order, -1))
    w_sorted = np.take_along_axis(w, np.argsort(idx, -1), -1)
    assert w_sorted == pytest.approx(np.take_along_axis(w_ref.numpy(), order, -1), abs=1e-6)


def test_extractor_f1_usage_match_token_by_token_oracle():
    """F1/U del extractor contra un oráculo en bucle sobre los logits del router (Mixtral)."""
    from test_extractor import ids_of, tiny

    from vigia.extractor import SignalExtractor

    model = tiny("mixtral")
    ids = ids_of(9)
    with torch.inference_mode(), SignalExtractor(model, (Tier.F1,), keep_router_logits=True) as ex:
        ex.begin_step()
        model(input_ids=ids)
        rows = ex.end_step()[0]
        z = {layer: t[0].numpy().astype(np.float64) for layer, t in ex.router_logits.items()}
    n_exp, k = 4, 2
    counts = {layer: np.zeros(n_exp) for layer in z}
    prev = {layer: None for layer in z}
    for t, sig in enumerate(rows):
        per_base: dict[str, list[float]] = {}
        for layer, zl in z.items():
            p = np.exp(zl[t] - zl[t].max())
            p /= p.sum()
            sel = list(np.argsort(-p, kind="stable")[:k])
            w = p[sel] / p[sel].sum()  # Mixtral renormaliza el top-k
            for e, wi in zip(sel, w, strict=True):
                counts[layer][e] += wi  # uso ponderado acumulado, causal
            u = counts[layer] / counts[layer].sum()
            nz = u[u > 0]
            zs = np.sort(zl[t])[::-1]
            expected = {
                "r_ent_n": -(p * np.log(p)).sum() / math.log(n_exp),
                "r_ent_k": -(w * np.log(w)).sum() / math.log(k),
                "r_topk_mass": p[sel].sum(),
                "r_margin12": zs[0] - zs[1],
                "r_margin_kk1": zs[k - 1] - zs[k],
                "u_gini": 1 - (u * u).sum(),
                "u_ihhi": 1 / (u * u).sum() / n_exp,
                "u_ent": -(nz * np.log(nz)).sum() / math.log(n_exp),
                "r_switch": math.nan if prev[layer] is None else 1 - len(set(sel) & prev[layer]) / k,
            }
            prev[layer] = set(sel)
            for base, value in expected.items():
                got = sig.features[layer_feature(base, layer)]
                assert got == pytest.approx(value, abs=1e-5, nan_ok=True), (t, layer, base)
                per_base.setdefault(base, []).append(value)
        for base, values in per_base.items():
            v = np.array(values)
            if np.isnan(v).all():
                continue
            v = v[~np.isnan(v)]
            for agg, ref in zip(AGGREGATES, (v.mean(), v.max(), v.std()), strict=True):
                assert sig.features[agg_feature(agg, base)] == pytest.approx(ref, abs=1e-5)


# --- Backend HF: stream vs teacher forcing en varias arquitecturas ------------------------------------

ARCHS = ("mixtral", "gpt_oss", "qwen2_moe", "qwen3_moe", "olmoe", "granitemoe", "phimoe", "deepseek_v3")


@pytest.mark.parametrize("arch", ARCHS)
def test_stream_and_teacher_force_are_identical(arch, monkeypatch):
    from test_extractor import CONFIGS
    from transformers import AutoModelForCausalLM

    from vigia.backends.hf import HFBackend
    from vigia.cli import _byte_tokenizer, make_tiny_backend

    if arch == "gpt_oss":
        be = make_tiny_backend("gpt_oss")
        tok = be.tokenizer
    else:
        tok = _byte_tokenizer(["<|eos|>"])
        cfg = CONFIGS[arch]()
        cfg.vocab_size = len(tok)
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(cfg).eval()
        model.generation_config.eos_token_id = None
        be = HFBackend(model, tok, model_id=f"tiny/{arch}")
    messages = [{"role": "user", "content": "¿Qué día es hoy?"}]
    text = "Sí, señor 😀 ñ"
    resp = tok(text, add_special_tokens=False)["input_ids"]
    forced = iter(resp)
    monkeypatch.setattr("vigia.backends.hf._sample", lambda *a: next(forced))
    tiers = tuple(t for t in (Tier.F0, Tier.F1, Tier.F2, Tier.H) if t in be.capabilities().tiers)
    stream = be.stream(messages, max_tokens=len(resp), tiers=tiers)
    generated = list(stream)
    teacher = be.teacher_force(messages, text, tiers=tiers)
    assert stream.text == text
    assert len(generated) == len(teacher) == len(resp)
    pos = 0
    for a, b in zip(generated, teacher, strict=True):
        assert (a.token_id, a.start, a.end, a.text, a.expert_ids) == (
            b.token_id, b.start, b.end, b.text, b.expert_ids,
        )  # fmt: skip
        assert a.start == pos and text[a.start : a.end] == a.text
        pos = a.end
        assert a.logprob == pytest.approx(b.logprob, abs=1e-5)
        assert a.features.keys() == b.features.keys()
        for key, value in a.features.items():
            assert value == pytest.approx(b.features[key], abs=1e-4, nan_ok=True), key
    assert pos == len(text)


# --- Etiquetas en español -----------------------------------------------------------------------------


def test_spanish_normalization_by_hand():
    assert normalize_es("¡El Niño, la Ñandú y los ÁRBOLES!") == "niño ñandu y arboles"
    assert normalize_es("¿Pingüino?") == "pinguino"
    assert exact_match("los Andes.", ["Cordillera", "Andes"])
    assert not exact_match("nino", "niño")  # la ñ no se pierde
    assert token_f1("la casa roja grande", "casa roja") == pytest.approx(0.8)
    assert token_f1("roja roja", "roja") == pytest.approx(2 / 3)


def test_spanish_numbers_and_dates_by_hand():
    cases = {
        "1.234": 1234.0, "1.234,5": 1234.5, "1,5": 1.5, "12.345.678": 12345678.0,
        "1,234,567": 1234567.0, "12.5": 12.5, "1234.567": 1234.567, "−3,25": -3.25,
    }  # fmt: skip
    for text, value in cases.items():
        assert parse_number_es(text) == pytest.approx(value), text
    assert extract_numbers("Entre 10-12 personas y 1.234,5 kg.") == [10.0, 12.0, 1234.5]
    assert numeric_match("El total es 1.234.", 1234)
    assert parse_dates_es("el 1º de mayo del 2000, el 12/10/1492 y 2020-02-30") == [
        __import__("datetime").date(2000, 5, 1),
        __import__("datetime").date(1492, 10, 12),
    ]
    assert date_match("Llegó el 12 de octubre de 1492.", "1492-10-12")
    assert date_match("Fue el 9 de setiembre de 1973", "1973-09-09")


def test_spans_to_token_labels_offsets():
    toks = [(0, 3), (3, 3), (3, 7), (7, 10)]
    assert spans_to_token_labels(toks, [(4, 6)]) == [0, 0, 1, 0]
    assert spans_to_token_labels(toks, [(3, 4)]) == [0, 1, 1, 0]  # el token de ancho cero cae adentro


# --- Evidencia en el modelo diminuto: reanudación y etiquetas ------------------------------------------


def test_evidence_resume_matches_single_run(tmp_path):
    from vigia import evidence as ev
    from vigia.cli import make_tiny_backend
    from vigia.datasets import validate_item

    be = make_tiny_backend("mixtral")
    items = [
        validate_item({"id": f"q{i}", "question": f"Pregunta {i}", "answers": ["París"]}) for i in range(3)
    ] + [validate_item({"id": "t1", "question": "Capital?", "response": "Lima", "label": 1})]
    full = ev.extract(be, items, tmp_path / "full", tiers=("F0", "F1"), max_tokens=5)
    part = tmp_path / "part"
    ev.extract(be, items[:2], part, tiers=("F0", "F1"), max_tokens=5)
    with open(part / ev.ROWS_FILE, "ab") as fh:  # corte a mitad de una escritura
        fh.write(b'{"id": "q2", "tok')
    resumed = ev.extract(be, items, part, tiers=("F0", "F1"), max_tokens=5)
    strip = lambda r: {k: v for k, v in r.items() if k != "elapsed_s"}  # noqa: E731
    assert [strip(r) for r in resumed] == [strip(r) for r in full]
    a, b = ev.build_features(tmp_path / "full"), ev.build_features(part)
    assert np.array_equal(a.X, b.X, equal_nan=True) and np.array_equal(a.labels, b.labels)
    for row in full:
        expected = 1 if row["id"] == "t1" else int(not row["label_info"]["correct"])
        scorable = [t for t in row["tokens"] if t["channel"] == "final" and not t["special"]]
        assert row["labels"] == [expected] * len(scorable)
        assert IGNORE_LABEL not in row["labels"]


# --- Ronda 2: pesos del motor del reporte, bootstrap, F2, causalidad y alineación ---------------------


def test_report_engine_weighted_curves_match_sklearn():
    """El motor vectorizado con pesos de bootstrap da recall@FPR, precisión@recall y AUROC de sklearn."""
    rng = np.random.default_rng(20)
    checked = 0
    for _ in range(300):
        n = int(rng.integers(5, 120))
        y = (rng.random(n) < rng.choice([0.05, 0.3, 0.8])).astype(float)
        if y.min() == y.max():
            y[0] = 1 - y[0]
        s = np.round(rng.random(n) * rng.choice([3, 10, 1000])) / 1000
        w = rng.integers(0, 4, n).astype(float)
        if np.sum(w * y) == 0 or np.sum(w * (1 - y)) == 0:
            continue
        st = E._curve_stats(E._Ranked(y, s), w[None])
        fpr, tpr, _ = skm.roc_curve(y, s, sample_weight=w, drop_intermediate=False)
        assert st["recall_at_fpr"][0] == pytest.approx(tpr[fpr <= 0.1 + 1e-12].max(), abs=1e-12)
        prec, rec, _ = skm.precision_recall_curve(y, s, sample_weight=w, drop_intermediate=False)
        for r in E.RECALL_TARGETS:
            ref = prec[rec >= r - 1e-12].max()
            assert st[f"precision_at_recall_{r}"][0] == pytest.approx(ref, abs=1e-12)
        assert st["auroc"][0] == pytest.approx(skm.roc_auc_score(y, s, sample_weight=w), abs=1e-12)
        checked += 1
    assert checked > 200


def test_weighted_aurc_equals_expanded_dataset():
    """Pesos enteros = filas duplicadas: AURC con pesos igual al AURC del dataset expandido."""
    rng = np.random.default_rng(21)
    for _ in range(100):
        n = int(rng.integers(2, 40))
        y = (rng.random(n) < 0.4).astype(float)
        p = np.round(rng.random(n), int(rng.integers(1, 3)))
        w = rng.integers(1, 4, n).astype(float)
        rep = np.repeat(np.arange(n), w.astype(int))
        got = float(E._aurc_stats(y, p, w[None])["aurc"][0])
        assert got == pytest.approx(E.aurc(y[rep], p[rep])["aurc"], abs=1e-12)


def test_report_bootstrap_auroc_ci_coverage():
    """IC 95 % del AUROC del motor del reporte (cluster = pregunta) cubre el valor poblacional."""

    def gen(rng, n_groups):
        size = rng.integers(3, 15, n_groups)
        u = rng.normal(size=n_groups)
        c = np.repeat(np.arange(n_groups), size)
        y = (rng.random(len(c)) < 1 / (1 + np.exp(-(u[c] - 1)))).astype(float)
        return y, 0.8 * y + 0.7 * u[c] + rng.normal(size=len(c)), c

    y, s, _ = gen(np.random.default_rng(12345), 60_000)
    truth = skm.roc_auc_score(y, s)
    hits, reps = 0, 100
    for rep in range(reps):
        y, s, c = gen(np.random.default_rng(rep), 60)
        lv = E._Level(y, c)
        lv.add("detector", s, s >= 0.5)
        boot = E._bootstrap({"token": lv}, 60, "detector", n_boot=300, seed=rep, level=0.95)
        lo, hi = boot["token"]["detector"]["auroc"]["ci"]
        hits += lo <= truth <= hi
    assert 0.88 <= hits / reps <= 0.995


def test_f2_features_and_aggregates_match_loops():
    import warnings

    rng = np.random.default_rng(22)
    for k in (1, 2, 3, 4):
        e = rng.normal(size=(20, k, 7))
        w = rng.random((20, k))
        cos, disp = fx.expert_cosine(e, w), fx.expert_dispersion(e, w)
        for r in range(20):
            wn = w[r] / w[r].sum()
            unit = e[r] / np.linalg.norm(e[r], axis=-1, keepdims=True)
            pairs = [(i, j) for i in range(k) for j in range(k) if i != j]
            den = sum(wn[i] * wn[j] for i, j in pairs)
            ref = sum(wn[i] * wn[j] * unit[i] @ unit[j] for i, j in pairs) / den if pairs else math.nan
            assert cos[r] == pytest.approx(ref, abs=1e-12, nan_ok=True)
            mean = (wn[:, None] * e[r]).sum(0)
            ref_disp = sum(wn[i] * np.sum((e[r, i] - mean) ** 2) for i in range(k)) / (mean @ mean)
            assert disp[r] == pytest.approx(ref_disp, rel=1e-10)
    v = rng.normal(size=(5, 40))
    v[rng.random(v.shape) < 0.3] = np.nan
    v[:, 0] = np.nan
    agg = fx.layer_aggregates(v, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        refs = {"mean": np.nanmean(v, 0), "max": np.nanmax(v, 0), "std": np.nanstd(v, 0)}
    for key, ref in refs.items():
        assert np.allclose(agg[key], ref, equal_nan=True), key
    idx = rng.integers(0, 6, size=(10, 3, 2))
    ww = rng.random((10, 3, 2))
    ref = np.zeros((10, 3, 6))
    for a, b, t in np.ndindex(10, 3, 2):
        ref[a, b, idx[a, b, t]] += ww[a, b, t]
    assert np.allclose(fx.usage_counts(idx, 6, ww), ref)
    z = rng.normal(size=(20, 100)) * 3
    assert fx.top_n_entropy(sp_log_softmax(z, -1)) == pytest.approx(
        fx.output_features(z, np.zeros(20, int))["out_top20_entropy"], abs=1e-10
    )


def _tiny_hf(arch):
    from test_extractor import CONFIGS
    from transformers import AutoModelForCausalLM

    from vigia.backends.hf import HFBackend
    from vigia.cli import _byte_tokenizer, make_tiny_backend

    if arch == "gpt_oss":
        return make_tiny_backend("gpt_oss")
    tok = _byte_tokenizer(["<|eos|>"])
    cfg = CONFIGS[arch]()
    cfg.vocab_size = len(tok)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg).eval()
    model.generation_config.eos_token_id = None
    return HFBackend(model, tok, model_id=f"tiny/{arch}")


@pytest.mark.parametrize("arch", ("mixtral", "qwen2_moe", "deepseek_v3", "phimoe", "gpt_oss"))
def test_features_are_causal_under_longer_continuation(arch):
    """Toda feature es causal: agregar tokens después no cambia los valores de los anteriores."""
    be = _tiny_hf(arch)
    tiers = tuple(t for t in (Tier.F0, Tier.F1, Tier.F2, Tier.H) if t in be.capabilities().tiers)
    msgs = [{"role": "user", "content": "hola"}]
    short = be.teacher_force(msgs, "Buenos días", tiers=tiers)
    long = be.teacher_force(msgs, "Buenos días a todos, ñandú", tiers=tiers)
    for a, b in zip(short, long, strict=False):
        assert (a.start, a.end, a.text) == (b.start, b.end, b.text)
        for key, value in a.features.items():
            assert value == pytest.approx(b.features[key], abs=1e-5, nan_ok=True), key


def _sentencepiece_tokenizer():
    """Tokenizer estilo Llama/Mixtral: Metaspace, byte fallback <0xNN> en el vocabulario y Strip."""
    import json

    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    unk = {"unk_token": "<unk>"}  # kwargs aparte: S106 confunde "token" con una credencial
    base = Tokenizer(models.BPE(**unk))
    base.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="first")
    specials = ["<unk>", "<s>", "</s>"]
    trainer = trainers.BpeTrainer(vocab_size=300, special_tokens=specials, show_progress=False)
    base.train_from_iterator(["La capital de Peru es Lima y la de Chile es Santiago."] * 30, trainer=trainer)
    state = json.loads(base.to_str())
    vocab = dict(state["model"]["vocab"])
    merges = [tuple(m.split(" ")) if isinstance(m, str) else tuple(m) for m in state["model"]["merges"]]
    for i in range(256):
        vocab.setdefault(f"<0x{i:02X}>", len(vocab))
    tk = Tokenizer(models.BPE(vocab=vocab, merges=merges, byte_fallback=True, **unk))
    tk.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="first")
    tk.decoder = decoders.Sequence(
        [decoders.Replace("▁", " "), decoders.ByteFallback(), decoders.Fuse(), decoders.Strip(" ", 1, 0)]
    )
    tk.add_special_tokens(specials)
    return PreTrainedTokenizerFast(tokenizer_object=tk, **unk, **{"eos_token": "</s>"})


def test_sentencepiece_byte_fallback_offsets_stream_equals_teacher(monkeypatch):
    from test_extractor import CONFIGS
    from transformers import AutoModelForCausalLM

    from vigia.backends.hf import HFBackend

    tok = _sentencepiece_tokenizer()
    cfg = CONFIGS["mixtral"]()
    cfg.vocab_size = len(tok)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg).eval()
    model.generation_config.eos_token_id = None
    be = HFBackend(model, tok, model_id="tiny/sp")
    msgs = [{"role": "user", "content": "Capital?"}]
    for text in ("Es Lima, en Perú 😀 ñ.", "Perú: ¿sí?", "Lima"):
        ids = tok(text, add_special_tokens=False)["input_ids"]
        teacher = be.teacher_force(msgs, text)
        assert "".join(e.text for e in teacher if e.channel == "final" and not e.is_special) == text
        assert all(text[e.start : e.end] == e.text for e in teacher)
        forced = iter(ids)
        monkeypatch.setattr("vigia.backends.hf._sample", lambda *a, it=forced: next(it))
        stream = be.stream(msgs, max_tokens=len(ids))
        generated = list(stream)
        assert stream.text == text
        assert [(e.start, e.end, e.text) for e in generated] == [(e.start, e.end, e.text) for e in teacher]


def test_vllm_token_offsets_assign_each_char_to_its_last_byte():
    from vigia.backends.vllm_proxy import token_offsets

    rng = np.random.default_rng(23)
    s = "Sí, señor 😀 ñandú — 東京 ok"
    b = s.encode()
    char_end = np.cumsum([len(c.encode()) for c in s])
    for _ in range(300):
        cuts = sorted(set(rng.integers(1, len(b), rng.integers(0, 12)).tolist()))
        parts = [b[i:j] for i, j in zip([0, *cuts], [*cuts, len(b)], strict=True)]
        text, offs = token_offsets([p.decode("utf-8", "replace") for p in parts], [list(p) for p in parts])
        assert text == s
        pos = acc = 0
        for part, (start, end) in zip(parts, offs, strict=True):
            acc += len(part)
            assert start == pos and end == int(np.searchsorted(char_end, acc, side="right"))
            pos = end
        assert pos == len(s)


def test_crc_guarantee_through_full_training_pipeline():
    """train_detector (z-score, LR, Platt, CRC) + tau interpolado: FNR por respuesta en test <= alpha."""
    import warnings

    def synth(rng, n_groups):
        f = {"out_entropy": [], "out_nll": []}
        lab, g, r = [], [], []
        for q in range(n_groups):
            for k in range(2):
                n = int(rng.integers(3, 20))
                yy = (rng.random(n) < 0.4) & (rng.random() < 0.3)
                f["out_entropy"] += list(rng.normal(size=n) + 1.0 * yy)
                f["out_nll"] += list(rng.normal(size=n) + 0.5 * yy)
                lab += yy.astype(int).tolist()
                g += [f"q{q}"] * n
                r += [f"q{q}r{k}"] * n
        return {k: np.asarray(v) for k, v in f.items()}, np.asarray(lab), np.asarray(g), np.asarray(r)

    alphas = (0.05, 0.1, 0.15, 0.2, 0.3)
    fnr: dict[float, list[float]] = {a: [] for a in alphas}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for seed in range(40):
            feats, y, g, r = synth(np.random.default_rng(seed), 150)
            det, _ = train_detector(feats, y, g, r, INFO, seed=seed, c_grid=(1.0,))
            ft, yt, _, rt = synth(np.random.default_rng(10_000 + seed), 300)
            p = det.score_matrix(det.matrix_from_features(ft, len(yt)))
            for a in alphas:
                fnr[a].append(_fnr_per_response(p, yt, rt, det.tau(a)))
    for a, values in fnr.items():
        assert np.mean(values) <= a + 2 * np.std(values) / math.sqrt(len(values)), a


# --- Ronda 3: métricas públicas, modo reducido de vLLM, español y tabla CRC del reporte ---------------


def test_public_metrics_match_sklearn_and_brute_force():
    """AUROC, AUPRC, Brier, log loss, recall@FPR, precisión@recall y métricas con tau contra oráculos."""
    import warnings

    rng = np.random.default_rng(30)
    checked = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for it in range(300):
            n = int(rng.integers(5, 150))
            y = (rng.random(n) < rng.choice([0.02, 0.1, 0.5, 0.9])).astype(int)
            if y.min() == y.max():
                continue
            s = np.clip(
                np.round(rng.random(n) * rng.choice([3, 10, 1000])) / 1000 + 0.3 * y * rng.random(), 0, 1
            )
            w = rng.integers(0, 4, n).astype(float) if it % 2 else None
            if w is not None and (np.sum(w * y) == 0 or np.sum(w * (1 - y)) == 0):
                continue
            assert E.auroc(y, s, w) == pytest.approx(skm.roc_auc_score(y, s, sample_weight=w), abs=1e-12)
            assert E.auprc(y, s, w) == pytest.approx(
                skm.average_precision_score(y, s, sample_weight=w), abs=1e-12
            )
            assert E.brier(y, s, w) == pytest.approx(skm.brier_score_loss(y, s, sample_weight=w), abs=1e-12)
            ll = skm.log_loss(y, np.clip(s, 1e-15, 1 - 1e-15), sample_weight=w, labels=[0, 1])
            assert E.log_loss(y, s, w) == pytest.approx(ll, abs=1e-7)
            best_rec = max(
                ((s >= t) & (y == 1)).sum() / (y == 1).sum()
                for t in np.r_[np.unique(s), np.inf]
                if ((s >= t) & (y == 0)).sum() / (y == 0).sum() <= 0.1 + 1e-12
            )
            assert E.recall_at_fpr(y, s) == pytest.approx(best_rec, abs=1e-12)
            best_prec = max(
                (((s >= t) & (y == 1)).sum() / (s >= t).sum() for t in np.unique(s)
                 if ((s >= t) & (y == 1)).sum() / (y == 1).sum() >= 0.8 - 1e-12),
                default=0.0,
            )  # fmt: skip
            assert E.precision_at_recall(y, s, 0.8) == pytest.approx(best_prec, abs=1e-12)
            tau = float(rng.random())
            flags = s >= tau
            tm = E.threshold_metrics(y, s, tau)
            assert tm["recall"] == pytest.approx(skm.recall_score(y, flags), abs=1e-12)
            assert tm["f1"] == pytest.approx(skm.f1_score(y, flags, zero_division=0), abs=1e-12)
            assert tm["flag_rate"] == pytest.approx(flags.mean(), abs=1e-12)
            if flags.any():
                assert tm["precision"] == pytest.approx(skm.precision_score(y, flags), abs=1e-12)
            checked += 1
    assert checked > 200


def test_vllm_reduced_f0_equals_hf_output_features():
    """out_nll, out_margin y out_top20_entropy del modo reducido (top-20 de vLLM) = los del backend HF."""
    from vigia.backends.vllm_proxy import top_logprob_features

    rng = np.random.default_rng(31)
    for _ in range(200):
        z = rng.normal(size=300) * 3
        t = int(rng.integers(300))
        hf = fx.output_features(z[None], np.array([t]))
        lp = sp_log_softmax(z)
        order = np.argsort(-lp)[:20]
        top = [float(lp[i]) for i in order] + ([] if t in order else [float(lp[t])])
        got = top_logprob_features(float(lp[t]), top)
        for key in ("out_nll", "out_margin", "out_top20_entropy"):
            assert got[key] == pytest.approx(float(hf[key][0]), abs=1e-9), key
        assert math.isnan(got["out_entropy"])


def test_vllm_routed_usage_matches_token_loop():
    """Features U del modo reducido contra conteos acumulados por token escritos a mano."""
    from vigia.backends.vllm_proxy import routed_usage_features

    rng = np.random.default_rng(32)
    n_exp, k = 8, 2
    routed = np.stack([np.stack([rng.choice(n_exp, k, replace=False) for _ in range(3)]) for _ in range(12)])
    feats = routed_usage_features(routed, n_exp)
    for t in range(12):
        for layer in range(3):
            c = np.bincount(routed[: t + 1, layer].ravel(), minlength=n_exp).astype(float)
            u = c / c.sum()
            nz = u[u > 0]
            assert feats[t][f"u_gini@{layer}"] == pytest.approx(1 - np.sum(u * u), abs=1e-12)
            assert feats[t][f"u_ihhi@{layer}"] == pytest.approx(1 / np.sum(u * u) / n_exp, abs=1e-12)
            assert feats[t][f"u_ent@{layer}"] == pytest.approx(
                -np.sum(nz * np.log(nz)) / np.log(n_exp), abs=1e-12
            )
            if t == 0:
                assert math.isnan(feats[t][f"r_switch@{layer}"])
            else:
                shared = len(set(routed[t, layer]) & set(routed[t - 1, layer]))
                assert feats[t][f"r_switch@{layer}"] == pytest.approx(1 - shared / k, abs=1e-12)


def test_spanish_number_and_date_edge_cases():
    import datetime as dt

    cases = {
        "1.234,56": 1234.56, "1,234.56": 1234.56, "0.500": 0.5, "0,050": 0.05, "-1.234": -1234.0,
        "+3": 3.0, "−7": -7.0, "10,000": 10.0, "1.000,0": 1000.0, "123.4567": 123.4567,
        "007": 7.0, "1,000,000": 1_000_000.0, "3,14159": 3.14159,
    }  # fmt: skip
    for text, value in cases.items():
        assert parse_number_es(text) == pytest.approx(value), text
    assert extract_numbers("Tiene 1.234 habitantes.") == [1234.0]
    assert extract_numbers("cerca de 1.000.") == [1000.0]
    assert extract_numbers("1.234.567,89 pesos y el 50% y 1.5%") == [1234567.89, 50.0, 1.5]
    assert extract_numbers("123-456=-333") == [123.0, 456.0, -333.0]
    assert parse_dates_es("12 DE OCTUBRE DE 1492.") == [dt.date(1492, 10, 12)]
    assert parse_dates_es("1° de mayo de 2000 y 1o de mayo de 2000") == [dt.date(2000, 5, 1)] * 2
    assert parse_dates_es("29 de febrero de 1900") == []
    assert parse_dates_es("29 de febrero de 2000") == [dt.date(2000, 2, 29)]
    assert parse_dates_es("9.7.1816") == [dt.date(1816, 7, 9)]
    assert normalize_es("¿Qué es «El Quijote»?") == "que es quijote"
    assert normalize_es("O’Higgins") == "ohiggins"


def test_report_crc_table_matches_independent_fnr():
    """La tabla CRC del reporte recalculada a mano sobre el split de test."""
    import warnings

    feats, y, g, r = _synthetic(seed=7, n_groups=80)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        det, rep = train_detector(feats, y, g, r, INFO, seed=7, c_grid=(1.0,))
        report = E.evaluate_report(feats, y, g, r, INFO, detector=det, split=rep["groups"], n_boot=20, seed=7)
    test = np.isin(g, rep["groups"]["test"])
    yt, rt = y[test], r[test]
    p = det.score_matrix(det.matrix_from_features({k: v[test] for k, v in feats.items()}, int(test.sum())))
    assert len(report["crc"]) == len(det.thresholds)
    for row, th in zip(report["crc"], det.thresholds, strict=True):
        flagged = p >= th["tau"]
        assert row["tau"] == th["tau"]
        assert row["observed_fnr_token"] == pytest.approx(1 - flagged[yt == 1].mean(), abs=1e-12)
        assert row["observed_fnr_response"] == pytest.approx(
            _fnr_per_response(p, yt, rt, th["tau"]), abs=1e-12
        )
        assert row["flag_rate"] == pytest.approx(flagged.mean(), abs=1e-12)
    assert report["dataset"]["prevalence_token"] == pytest.approx(yt.mean(), abs=1e-12)
