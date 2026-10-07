import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from vigia.calibrate import (
    DEFAULT_ALPHAS,
    CalibrationError,
    _response_max,
    build_dataset,
    crc_threshold,
    crc_thresholds,
    fit_logistic,
    recommend_alpha,
    response_rates,
    split_groups,
    split_masks,
    train_detector,
)
from vigia.contracts import ModelInfo
from vigia.evaluate import auroc, ece
from vigia.labels import IGNORE_LABEL

INFO = ModelInfo("tiny-mixtral", "mixtral", 2, (0, 1), 4, 2, 100, "softmax_topk")


def _synthetic(seed=0, n_q=400, resp_per_q=2, tok_per_resp=15):
    """Features con señal plantada: logit verdadero = 3 a - 2 b - 1, más una feature ruido."""
    rng = np.random.default_rng(seed)
    n = n_q * resp_per_q * tok_per_resp
    groups = np.repeat([f"q{i}" for i in range(n_q)], resp_per_q * tok_per_resp)
    responses = np.repeat([f"r{i}" for i in range(n_q * resp_per_q)], tok_per_resp)
    a, b, noise = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(3.0 * a - 2.0 * b - 1.0)))).astype(int)
    return {"out_entropy": a, "r_ent_n@0": b, "u_gini@0": noise}, y, groups, responses


def test_fit_logistic_matches_sklearn():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(600, 4))
    y = (rng.random(600) < 1 / (1 + np.exp(-(X @ [1.0, -2.0, 0.5, 0.0] + 0.3)))).astype(int)
    for c in (0.05, 1.0):
        coef, bias = fit_logistic(X, y, l2=1.0 / c)
        ref = LogisticRegression(C=c, tol=1e-12, max_iter=10_000).fit(X, y)
        np.testing.assert_allclose(coef, ref.coef_[0], atol=1e-5)
        assert bias == pytest.approx(ref.intercept_[0], abs=1e-5)


def test_at_m03_planted_signal_gives_high_auroc_and_low_ece():
    """AT-M03: features sintéticas con señal conocida (semilla fija): AUROC > 0,9 y ECE < 0,05 tras Platt."""
    feats, y, groups, responses = _synthetic(seed=7)
    det, report = train_detector(feats, y, groups, responses, INFO, seed=7)
    masks = split_masks(groups, report["groups"])
    test = masks["test"]
    X = det.matrix_from_features({k: v[test] for k, v in feats.items()}, int(test.sum()))
    p = det.score_matrix(X)
    assert auroc(y[test], p) > 0.9
    assert ece(y[test], p) < 0.05
    assert det.artifact["metrics"]["test"]["auroc"] > 0.9
    assert det.artifact["training"]["label_source"] == "computable"


def test_at_m04_repeated_questions_never_cross_splits():
    """AT-M04: con preguntas repetidas, ningún id de pregunta aparece en dos splits."""
    feats, y, groups, responses = _synthetic(seed=1, n_q=120, resp_per_q=4, tok_per_resp=6)
    _, report = train_detector(feats, y, groups, responses, INFO, c_grid=(1.0,))
    sets = {k: set(v) for k, v in report["groups"].items()}
    assert (
        not (sets["train"] & sets["cal"])
        and not (sets["train"] & sets["test"])
        and not (sets["cal"] & sets["test"])
    )
    assert sets["train"] | sets["cal"] | sets["test"] == set(groups)
    masks = split_masks(groups, report["groups"])
    for split_mask in masks.values():
        resp_in = set(responses[split_mask])
        resp_out = set(responses[~split_mask])
        assert not resp_in & resp_out  # una respuesta nunca queda partida entre splits
    assert sum(report["counts"][k]["n_tokens"] for k in ("train", "cal", "test")) == len(y)


def test_at_m04_leaky_split_and_mixed_response_are_rejected():
    """AT-M04: un split con fuga o una respuesta en dos preguntas se rechazan."""
    feats, y, groups, responses = _synthetic(seed=2, n_q=30)
    leaky = split_groups(groups, seed=0)
    leaky["cal"] = leaky["cal"] + [leaky["train"][0]]
    with pytest.raises(CalibrationError, match="fuga"):
        train_detector(feats, y, groups, responses, INFO, split=leaky)
    bad_groups = groups.copy()
    bad_groups[0] = "otra"
    with pytest.raises(ValueError, match="dos ids de pregunta"):
        build_dataset(feats, y, bad_groups, responses)


def _crc_round(rng, n_resp):
    """Respuestas intercambiables: cada una con 1 a 6 tokens alucinados y un sesgo propio."""
    k = rng.integers(1, 7, size=n_resp)
    shift = np.repeat(rng.normal(scale=0.5, size=n_resp), k)
    p = 1 / (1 + np.exp(-(rng.normal(size=k.sum()) + 1.0 + shift)))
    return p, np.ones(k.sum(), dtype=int), np.repeat(np.arange(n_resp), k)


def test_at_m05_crc_controls_fnr_over_200_seeds():
    """AT-M05: CRC con alpha = 0,1: la FNR media en test sobre 200 semillas es <= 0,1 + tolerancia."""
    alpha, fnrs = 0.1, []
    for seed in range(200):
        rng = np.random.default_rng(seed)
        res = crc_threshold(*_crc_round(rng, 300), alpha)
        assert res.feasible
        p, _, resp = _crc_round(rng, 300)
        missed = np.bincount(resp, weights=p < res.tau) / np.bincount(resp)
        fnrs.append(missed.mean())
    mean = float(np.mean(fnrs))
    se = float(np.std(fnrs) / np.sqrt(len(fnrs)))
    assert mean <= alpha + 3 * se + 0.002
    assert mean >= alpha - 2 / 301 - 3 * se - 0.01  # no es trivialmente conservador


def test_crc_threshold_matches_hand_computation():
    # 19 respuestas, un token alucinado cada una, puntajes 0,05 ... 0,95. Con alpha 0,1:
    # condición: (peso con p < tau) + 1 <= 0,1 * 20 = 2 -> como máximo 1 respuesta sin marcar -> tau = 0,10.
    p = np.arange(1, 20) / 20
    res = crc_threshold(p, np.ones(19, dtype=int), np.arange(19), 0.1)
    assert res.feasible and res.n == 19 and res.tau == pytest.approx(0.10)
    # Una respuesta con dos tokens pesa 1/2 cada uno.
    p2 = np.r_[0.01, 0.02, np.arange(2, 20) / 20]
    resp2 = np.r_[0, 0, np.arange(1, 19)]
    assert crc_threshold(p2, np.ones(20, dtype=int), resp2, 0.1).tau == pytest.approx(0.10)
    # Los tokens correctos no cambian el umbral.
    p3 = np.r_[p, [0.99, 0.5]]
    y3 = np.r_[np.ones(19, dtype=int), [0, 0]]
    assert crc_threshold(p3, y3, np.r_[np.arange(19), [0, 1]], 0.1).tau == pytest.approx(0.10)


def test_crc_infeasible_with_too_few_responses():
    res = crc_threshold(np.array([0.9, 0.8, 0.7, 0.6, 0.5]), np.ones(5, dtype=int), np.arange(5), 0.1)
    assert not res.feasible and res.tau == 0.0


def test_artifact_thresholds_are_monotone_and_report_flag_rate():
    """AT-M05: el artefacto guarda tau, tasa de marcado y precisión esperadas por alpha."""
    feats, y, groups, responses = _synthetic(seed=4, n_q=200)
    det, report = train_detector(feats, y, groups, responses, INFO, c_grid=(1.0,))
    th = det.thresholds
    assert [t["alpha"] for t in th] == list(DEFAULT_ALPHAS)
    assert all(a["tau"] <= b["tau"] for a, b in zip(th, th[1:], strict=False))
    assert all(a["expected_flag_rate"] >= b["expected_flag_rate"] for a, b in zip(th, th[1:], strict=False))
    assert all(r["feasible"] for r in report["crc"])
    # Con 15 tokens por respuesta y prevalencia ~0,4 todas las respuestas de cal tienen errores: sin
    # respuestas buenas la falsa alarma es null, no hay recomendación y default_alpha cae a 0,1.
    assert all(t["expected_response_false_alarm"] is None for t in th)
    assert det.recommended_alpha is None and det.default_alpha == 0.1 and det.pareto() == []


def test_ignore_labels_dropped_and_bad_features_reported():
    feats, y, groups, responses = _synthetic(seed=5, n_q=100)
    y = y.copy()
    y[:10] = IGNORE_LABEL
    feats = dict(feats, **{"e_jsd": np.full(len(y), np.nan), "r_switch@0": np.ones(len(y))})
    det, report = train_detector(feats, y, groups, responses, INFO, c_grid=(1.0,))
    assert report["dropped_features"] == {
        "e_jsd": "sin valores finitos en train",
        "r_switch@0": "constante en train",
    }
    assert "e_jsd" not in det.feature_names
    assert sum(report["counts"][k]["n_tokens"] for k in ("train", "cal", "test")) == len(y) - 10


def test_invalid_inputs_fail_clearly():
    feats, y, groups, responses = _synthetic(seed=6, n_q=40)
    with pytest.raises(ValueError, match="feature desconocida"):
        build_dataset({"nope": feats["out_entropy"]}, y, groups, responses)
    with pytest.raises(ValueError, match="labels fuera"):
        build_dataset(feats, np.where(y == 1, 2, 0), groups, responses)
    with pytest.raises(ValueError, match="forma"):
        build_dataset({"out_entropy": feats["out_entropy"][:-1]}, y, groups, responses)
    with pytest.raises(CalibrationError, match="una sola clase"):
        train_detector(feats, np.zeros_like(y), groups, responses, INFO)
    with pytest.raises(ValueError):
        train_detector(feats, y, groups, responses, INFO, default_alpha=0.01)


def _answer_level_dataset(seed=0, n_q=24, tok_per_resp=5):
    """Como el e2e de calibración: una respuesta por pregunta, la mitad correcta (todos sus tokens
    en 0) y la mitad errónea (todos en 1), con señal moderada en una feature."""
    rng = np.random.default_rng(seed)
    groups = np.repeat([f"q{rng.integers(1 << 30):08x}{i}" for i in range(n_q)], tok_per_resp)
    responses = np.repeat([f"r{i}" for i in range(n_q)], tok_per_resp)
    y = np.repeat(np.arange(n_q) % 2, tok_per_resp)
    a = rng.normal(size=len(y)) + 1.5 * y
    return {"out_entropy": a, "r_ent_n@0": rng.normal(size=len(y))}, y, groups, responses


def test_stratified_split_gives_both_classes_to_every_split():
    """Regresión: "El split de calibración por pregunta no se estratifica: un dataset balanceado de
    24 respuestas falla con la semilla por defecto". Con 12 preguntas correctas y 12 erróneas, cada
    split (train, cal y test) recibe las dos clases con cualquier semilla, y el split sigue sin fuga."""
    for data_seed in range(5):
        _, y, groups, _ = _answer_level_dataset(seed=data_seed)
        cls = dict(zip(groups, y, strict=True))
        unstratified_failures = 0
        for seed in range(60):
            split = split_groups(groups, seed=seed, labels=y)
            sizes = {k: len(v) for k, v in split.items()}
            assert sizes == {k: len(v) for k, v in split_groups(groups, seed=seed).items()}
            assert sorted(sum(split.values(), [])) == sorted(set(groups))
            for name, members in split.items():
                assert {cls[m] for m in members} == {0, 1}, (data_seed, seed, name)
            old = split_groups(groups, seed=seed)
            unstratified_failures += any(len({cls[m] for m in old[k]}) < 2 for k in ("cal", "test"))
        assert unstratified_failures > 0  # sin estratificar, alguna semilla deja un split de una clase


def test_balanced_answer_level_dataset_calibrates_with_default_seed():
    """Regresión: "El split de calibración por pregunta no se estratifica: un dataset balanceado de
    24 respuestas falla con la semilla por defecto". train_detector usa el split estratificado."""
    feats, y, groups, responses = _answer_level_dataset(seed=0)
    det, report = train_detector(feats, y, groups, responses, INFO, c_grid=(1.0,))
    for name in ("train", "cal", "test"):
        assert 0 < report["counts"][name]["positive_rate"] < 1, name
    assert det.thresholds


def test_single_class_split_message_suggests_another_seed():
    """Regresión del mismo defecto: si un split explícito queda con una clase, el mensaje dice que
    el dataset tiene las dos y propone otra semilla o más preguntas."""
    feats, y, groups, responses = _answer_level_dataset(seed=0)
    neg = sorted({g for g, c in zip(groups, y, strict=True) if c == 0})
    pos = sorted({g for g, c in zip(groups, y, strict=True) if c == 1})
    split = {"cal": neg[:5], "test": pos[:3] + neg[5:7], "train": neg[7:] + pos[3:]}
    with pytest.raises(CalibrationError, match="otra semilla"):
        train_detector(feats, y, groups, responses, INFO, split=split, c_grid=(1.0,))


# --- grilla fina y alpha recomendado ----------------------------------------------------------------


def test_default_alpha_grid_is_fine_and_without_float_noise():
    assert len(DEFAULT_ALPHAS) == 25 and DEFAULT_ALPHAS[0] == 0.02 and DEFAULT_ALPHAS[-1] == 0.5
    assert all(repr(a) == f"{a:.2f}".rstrip("0") for a in DEFAULT_ALPHAS)  # 0.06, no 0.06000000000000001
    assert all(round(b - a, 10) == 0.02 for a, b in zip(DEFAULT_ALPHAS, DEFAULT_ALPHAS[1:], strict=False))
    assert 0.1 in DEFAULT_ALPHAS and 0.14 in DEFAULT_ALPHAS and 0.32 in DEFAULT_ALPHAS


def test_response_rates_and_recommendation_hand_values():
    # Respuestas: r0 mala (máx 0,9), r1 mala (máx 0,4), r2 buena (máx 0,6), r3 buena (máx 0,2).
    p = np.array([0.9, 0.1, 0.4, 0.3, 0.6, 0.2])
    y = np.array([1, 0, 1, 0, 0, 0])
    resp = np.array(["r0", "r0", "r1", "r1", "r2", "r3"])
    m, lab = _response_max(p, y, resp)
    assert response_rates(m, lab, 0.5) == (0.5, 0.5)
    assert response_rates(m, lab, 0.3) == (1.0, 0.5)
    assert response_rates(m, lab, 0.95) == (0.0, 0.0)
    assert response_rates(m[lab == 1], lab[lab == 1], 0.5) == (0.5, None)
    entries, _ = crc_thresholds(p, y, resp, [0.6, 0.8])
    for e in entries:
        assert (e["expected_response_recall"], e["expected_response_false_alarm"]) == response_rates(
            m, lab, e["tau"]
        )

    def th(alpha, recall, false_alarm):
        return {
            "alpha": alpha,
            "expected_response_recall": recall,
            "expected_response_false_alarm": false_alarm,
        }

    # Youden: 0,99-0,60 = 0,39; 0,96-0,20 = 0,76; 0,85-0,15 = 0,70 -> 0,14 (el codo medido con Granite).
    assert recommend_alpha([th(0.1, 0.99, 0.6), th(0.14, 0.96, 0.2), th(0.32, 0.85, 0.15)]) == 0.14
    assert recommend_alpha([th(0.2, 0.9, 0.1), th(0.1, 0.9, 0.1), th(0.3, 0.5, 0.0)]) == 0.1  # empate: menor
    assert recommend_alpha([th(0.1, 1.0, None)]) is None
    assert recommend_alpha([{"alpha": 0.1, "tau": 0.5}]) is None  # umbrales de un artefacto viejo


def _mixed_dataset(seed=11, n_q=200, resp_per_q=2, tok=10):
    """Respuestas limpias y con errores (sesgo por respuesta), para que haya falsas alarmas medibles."""
    rng = np.random.default_rng(seed)
    n_resp = n_q * resp_per_q
    n = n_resp * tok
    groups = np.repeat([f"q{i}" for i in range(n_q)], resp_per_q * tok)
    responses = np.repeat([f"r{i}" for i in range(n_resp)], tok)
    offset = np.repeat(rng.normal(scale=2.0, size=n_resp), tok)
    a, b = rng.normal(size=n), rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(2.5 * a - 1.5 * b - 4.0 + offset)))).astype(int)
    return {"out_entropy": a, "r_ent_n@0": b}, y, groups, responses


def test_recommended_alpha_is_chosen_only_with_calibration_data():
    """Cambiar las etiquetas de test no cambia el alpha recomendado ni los umbrales: test no elige."""
    feats, y, groups, responses = _mixed_dataset()
    det, report = train_detector(feats, y, groups, responses, INFO, c_grid=(1.0,))
    assert det.recommended_alpha is not None and det.recommended_alpha in DEFAULT_ALPHAS
    assert det.default_alpha == det.recommended_alpha == report["recommended_alpha"]
    th = det.thresholds
    assert det.recommended_alpha == recommend_alpha(th)
    best = max(t["expected_response_recall"] - t["expected_response_false_alarm"] for t in th)
    winner = next(t for t in th if t["alpha"] == det.recommended_alpha)
    assert winner["expected_response_recall"] - winner["expected_response_false_alarm"] == pytest.approx(best)
    assert all(0 <= t["expected_response_false_alarm"] <= 1 for t in th)
    assert det.pareto() and det.recommended_alpha in {p["alpha"] for p in det.pareto()}

    test_mask = split_masks(groups, report["groups"])["test"]
    flipped = y.copy()
    flipped[test_mask] = 1 - flipped[test_mask]
    det2, report2 = train_detector(
        feats, flipped, groups, responses, INFO, c_grid=(1.0,), split=report["groups"]
    )
    assert det2.recommended_alpha == det.recommended_alpha and det2.default_alpha == det.default_alpha
    assert det2.thresholds == th
    assert det2.artifact["metrics"]["test"] != det.artifact["metrics"]["test"]  # test sí cambió


def test_explicit_default_alpha_is_kept_and_recommendation_still_stored():
    feats, y, groups, responses = _mixed_dataset(seed=12, n_q=120)
    det, _ = train_detector(feats, y, groups, responses, INFO, c_grid=(1.0,), default_alpha=0.3)
    assert det.default_alpha == 0.3 and det.recommended_alpha is not None
