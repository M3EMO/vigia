"""Monitor de drift (P-36): PSI con valores cerrados, niveles, reporte y referencia del artefacto."""

from __future__ import annotations

import math

import numpy as np
import pytest

from vigia.calibrate import train_detector
from vigia.contracts import ModelInfo
from vigia.detector import Detector
from vigia.drift import (
    ALERT_PSI,
    DRIFT_FEATURES,
    EPS,
    WARN_PSI,
    build_reference,
    drift_report,
    level,
    psi,
    psi_from_summary,
    summarize,
    worst_level,
)

INFO = ModelInfo("tiny-drift", "mixtral", 2, (0, 1), 4, 2, 100, "softmax_topk")


def test_psi_identical_distributions_is_zero():
    values = np.arange(10, dtype=float)
    assert psi(values, values) == pytest.approx(0.0, abs=1e-12)


def test_psi_two_bins_closed_value():
    """Esperado [0,0,1,1] (bins 0,5/0,5), actual [0,1,1,1] (0,25/0,75): PSI = 0,25·ln 3."""
    value = psi([0, 0, 1, 1], [0, 1, 1, 1], bins=2)
    assert value == pytest.approx(0.25 * math.log(3), rel=1e-12)


def test_psi_all_mass_in_one_bin_uses_eps():
    """Diez bins de 0,1; todo el actual cae en el primero: (1-0,1)ln(10) + 9(EPS-0,1)ln(EPS/0,1)."""
    expected = (1 - 0.1) * math.log(1 / 0.1) + 9 * (EPS - 0.1) * math.log(EPS / 0.1)
    assert psi(np.arange(10), np.zeros(50)) == pytest.approx(expected, rel=1e-12)


def test_psi_is_symmetric_in_the_bin_fractions():
    a = psi([0, 0, 1, 1], [0, 1, 1, 1], bins=2)
    b = psi([0, 1, 1, 1], [0, 0, 1, 1], bins=2)
    # Los bins salen del esperado; acá coinciden ({0}, {1}), así que el PSI es simétrico.
    assert a == pytest.approx(b, rel=1e-12)


def test_summarize_quantile_edges_and_discrete_values():
    s = summarize(np.linspace(0.0, 1.0, 101), bins=4)
    assert s["edges"] == pytest.approx([0.25, 0.5, 0.75])
    assert s["n"] == 101 and sum(s["fractions"]) == pytest.approx(1.0)
    flags = summarize([0, 0, 0, 1], bins=10)  # pocos valores distintos: un bin por valor
    assert flags["edges"] == [0.5] and flags["fractions"] == [0.75, 0.25]
    const = summarize([3.0, 3.0], bins=10)
    assert const["edges"] == [] and const["fractions"] == [1.0]
    assert psi_from_summary(const, [9.0, 9.0]) == 0.0  # una referencia constante no mide nada


def test_summarize_and_psi_reject_bad_input():
    with pytest.raises(ValueError):
        summarize([float("nan")])
    with pytest.raises(ValueError):
        summarize([1.0, 2.0], bins=1)
    with pytest.raises(ValueError):
        psi([1.0, 2.0], [float("nan")])


def test_levels_follow_the_usual_thresholds():
    assert level(0.0) == "ok" and level(WARN_PSI - 1e-9) == "ok"
    assert level(WARN_PSI) == "warn" and level(ALERT_PSI - 1e-9) == "warn"
    assert level(ALERT_PSI) == "alert" and level(3.0) == "alert"
    assert worst_level(["ok", "alert", "warn"]) == "alert" and worst_level([]) is None


def test_drift_report_accepts_raw_or_summarized_reference_and_skips_missing():
    rng = np.random.default_rng(0)
    ref = rng.uniform(0, 1, 2000)
    same = rng.uniform(0, 1, 2000)
    shifted = rng.uniform(0.5, 1.5, 2000)
    report = drift_report(
        {"p_error": ref, "out_entropy": summarize(ref)},
        {"p_error": same, "out_entropy": shifted, "flag": [1.0, 0.0]},
    )
    assert set(report) == {"p_error", "out_entropy"}  # flag sin referencia no aparece
    assert report["p_error"]["level"] == "ok" and report["p_error"]["n"] == 2000
    assert report["out_entropy"]["level"] == "alert" and report["out_entropy"]["psi"] > ALERT_PSI


def test_build_reference_keeps_tau_and_only_drift_features():
    ref = build_reference(
        [0.1, 0.2, 0.6, 0.9],
        0.5,
        {"out_entropy": [1.0, 2.0, 3.0, 4.0], "out_nll": [1.0] * 4, "agg_mean:r_ent_n": [float("nan")] * 4},
    )
    assert set(ref) == {"p_error", "flag", "out_entropy"}
    assert ref["flag"]["tau"] == 0.5 and ref["flag"]["fractions"] == [0.5, 0.5]


def test_train_detector_stores_a_valid_drift_reference():
    """El artefacto guarda la referencia de calibración y sigue validando (hash incluido)."""
    rng = np.random.default_rng(1)
    n_resp, per = 60, 8
    y = (rng.uniform(size=n_resp * per) < 0.3).astype(int)
    feats = {
        "out_entropy": rng.normal(size=y.size) + 1.5 * y,
        "agg_mean:r_ent_n": rng.uniform(size=y.size) + 0.3 * y,
    }
    responses = np.repeat([f"r{i}" for i in range(n_resp)], per)
    detector, _ = train_detector(feats, y, responses, responses, INFO, seed=0)
    ref = detector.drift_reference
    assert set(ref) == set(DRIFT_FEATURES) == {"p_error", "flag", "agg_mean:r_ent_n", "out_entropy"}
    assert ref["flag"]["tau"] == pytest.approx(detector.tau(detector.default_alpha))
    reloaded = Detector.loads(detector.to_json())
    assert reloaded.drift_reference == ref and reloaded.sha256 == detector.sha256
