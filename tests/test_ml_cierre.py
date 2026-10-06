"""Regresiones de la ronda de cierre del grupo ML (evaluate y labels)."""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from vigia import evaluate as E
from vigia.calibrate import train_detector
from vigia.contracts import ModelInfo
from vigia.evidence import gold_correct
from vigia.labels import extract_numbers, parse_date_parts_es, parse_dates_es, parse_number_es

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


def _synthetic(n_q: int = 300, tok: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = np.repeat(rng.random(n_q) < 0.4, tok).astype(int)
    groups = np.repeat([f"q{i}" for i in range(n_q)], tok)
    resp = np.repeat([f"r{i}" for i in range(n_q)], tok)
    feats = {"out_entropy": y * 1.5 + rng.normal(size=len(y)), "out_nll": y + rng.normal(size=len(y))}
    return feats, y, groups, resp


# --- evaluate_report con alphas calibrados menores a 0,05 ----------------------------------------------


def test_evaluate_report_accepts_detector_calibrated_below_default_alphas():
    """evaluate_report no puede evaluar un detector calibrado con alphas menores a 0,05: los niveles
    y baselines internos usan DEFAULT_ALPHAS y tau(alpha) lanza ValueError."""
    feats, y, groups, resp = _synthetic()
    det, split = train_detector(feats, y, groups, resp, INFO, alphas=(0.01, 0.02), default_alpha=0.02)
    rep = E.evaluate_report(feats, y, groups, resp, INFO, detector=det, split=split["groups"], n_boot=20)
    assert rep["alpha"] == 0.02
    assert rep["detector"]["tau"] == det.tau(0.02)
    comparators = [rep["tiers"]["F0"], rep["baselines"]["out_entropy"], rep["baselines"]["out_nll"]]
    assert all(c["available"] and c["alpha"] == 0.02 for c in comparators)
    # Un alpha explícito fuera de lo calibrado del detector sigue fallando con mensaje claro.
    with pytest.raises(ValueError, match="mínimo calibrado"):
        E.evaluate_report(
            feats, y, groups, resp, INFO, detector=det, split=split["groups"], alpha=0.005, n_boot=5
        )


def test_evaluate_report_explicit_small_alpha_without_detector():
    """Mismo defecto sin detector: un alpha explícito < 0,05 se calibra en todos los comparadores."""
    feats, y, groups, resp = _synthetic(seed=1)
    rep = E.evaluate_report(feats, y, groups, resp, INFO, alpha=0.02, n_boot=10)
    assert rep["alpha"] == 0.02
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        E.evaluate_report(feats, y, groups, resp, INFO, alpha=1.5, n_boot=5)


def test_cli_evaluate_with_alphas_below_default(tmp_path, capsys):
    """evaluate_report no puede evaluar un detector calibrado con alphas menores a 0,05 (vía CLI:
    'vigia calibrate --alphas 0.01,0.02 --default-alpha 0.02' y luego 'vigia evaluate')."""
    from test_evidence import synthetic_items

    from vigia.cli import main, make_tiny_backend
    from vigia.datasets import write_jsonl

    data = tmp_path / "items.jsonl"
    write_jsonl(data, synthetic_items(make_tiny_backend("mixtral")))
    run = tmp_path / "run"
    argv = ["extract", "--model", "tiny:mixtral", "--data", str(data), "--out", str(run), "--max-tokens", "8"]
    assert main([*argv, "--quiet"]) == 0
    assert (
        main(["calibrate", "--features", str(run), "--alphas", "0.01,0.02", "--default-alpha", "0.02"]) == 0
    )
    code = main(["evaluate", "--features", str(run), "--n-boot", "30"])
    err = capsys.readouterr().err
    assert code == 0, err
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    assert report["alpha"] == 0.02


# --- números con espacio de miles ---------------------------------------------------------------------


def test_spaced_thousands_follow_rae_norm():
    """gold_correct lee mal los números con espacio de miles (norma RAE '12 345'): el gold '12 345'
    vale 12, la respuesta exacta queda incorrecta y '12' queda correcta."""
    assert gold_correct("12 345", ["12 345"], "number")[0]
    assert not gold_correct("Son 12 habitantes.", ["12 345"], "number")[0]
    assert gold_correct("12 345", ["12345"], "number")[0]
    assert not gold_correct("Tiene 12 habitantes", ["12 345 habitantes"])[0]
    assert gold_correct("Tiene 12 345 habitantes", ["12 345 habitantes"])[0]
    nbsp, thin = " ", " "
    assert parse_number_es("12 345") == 12345.0
    assert parse_number_es(f"1{nbsp}234{thin}567,89") == pytest.approx(1234567.89)
    assert parse_number_es("-12 345.5") == -12345.5
    assert parse_number_es("12 3456") is None
    assert extract_numbers("Tiene 12 3456 y 1 000 000 de 7") == [12.0, 3456.0, 1_000_000.0, 7.0]
    assert extract_numbers("En 1945 200 personas") == [1945.0, 200.0]
    assert extract_numbers("12 345") == [12345.0]


# --- fechas con ordinal RAE y "del año" ---------------------------------------------------------------


def test_spanish_full_dates_with_ordinal_dot_and_del_anio():
    """parse_dates_es no reconoce formas españolas comunes de fecha completa: '1.º de mayo de 1945'
    (ordinal RAE) y '12 de octubre del año 1492'."""
    cases = [
        ("Fue el 1.º de mayo de 1945.", "1945-05-01", dt.date(1945, 5, 1)),
        ("Fue el 1.° de mayo de 1945.", "1945-05-01", dt.date(1945, 5, 1)),
        ("Colón llegó el 12 de octubre del año 1492.", "1492-10-12", dt.date(1492, 10, 12)),
        ("12 DE OCTUBRE DEL AÑO 1492", "1492-10-12", dt.date(1492, 10, 12)),
    ]
    for text, gold, date in cases:
        assert parse_dates_es(text) == [date], text
        assert gold_correct(text, [gold], "date")[0], text
    assert not gold_correct("Fue el 2.º de mayo de 1945", ["1945-05-01"], "date")[0]
    assert parse_date_parts_es("en octubre del año 1492") == [(1492, 10, None)]
    assert gold_correct("Fue en octubre del año 1492", ["octubre de 1492"], "date")[0]


# --- ECE con pesos = ECE del dataset con filas repetidas ---------------------------------------------


def test_weighted_ece_equals_expanded_dataset():
    """ECE con pesos (y por lo tanto la réplica bootstrap de ECE) no equivale al ECE del dataset con
    filas repetidas: los bins de igual masa ignoran los pesos."""
    y, p, w = np.array([1.0, 0.0]), np.array([0.7, 0.4]), np.array([2.0, 1.0])
    assert E.ece(y, p, n_bins=2, w=w) == pytest.approx(E.ece(y[[0, 0, 1]], p[[0, 0, 1]], n_bins=2))
    rng = np.random.default_rng(0)
    for _ in range(300):
        n = int(rng.integers(1, 40))
        y = rng.integers(0, 2, n).astype(float)
        p = np.round(rng.random(n), int(rng.integers(1, 3)))
        w = rng.integers(0, 4, n).astype(float)
        if w.sum() == 0:
            continue
        rep = np.repeat(np.arange(n), w.astype(int))
        for n_bins in (2, 5, 15):
            assert E.ece(y, p, n_bins=n_bins, w=w) == pytest.approx(
                E.ece(y[rep], p[rep], n_bins=n_bins), abs=1e-12
            )


def test_bootstrap_ece_replicas_equal_resampled_datasets():
    """Réplica bootstrap del ECE del reporte = ECE recalculado sobre cada dataset remuestreado."""
    rng = np.random.default_rng(1)
    G = 40
    c = np.repeat(np.arange(G), rng.integers(3, 9, G))
    p = np.round(rng.beta(2, 5, len(c)), 2)
    y = (rng.random(len(c)) < p).astype(float)
    lv = E._Level(y, c)
    lv.add("d", p, p >= 0.5)
    W = np.vstack([np.bincount(r, minlength=G)[c] for r in np.random.default_rng(4).integers(0, G, (25, G))])
    got = lv.stats(W.astype(float))["d"]["ece"]
    for b in range(len(W)):
        idx = np.repeat(np.arange(len(c)), W[b])
        assert got[b] == pytest.approx(E.ece(y[idx], p[idx]), abs=1e-12)
