import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from vigia.calibrate import train_detector
from vigia.contracts import DetectorMismatchError, ModelInfo, TokenEvent
from vigia.detector import Detector, artifact_sha256

INFO = ModelInfo("tiny-mixtral", "mixtral", 2, (0, 1), 4, 2, 100, "softmax_topk")
ROOT = Path(__file__).resolve().parents[1]


def _synthetic(seed=0, n_q=150, per_q=20):
    rng = np.random.default_rng(seed)
    n = n_q * per_q
    groups = np.repeat(np.arange(n_q), per_q)
    responses = np.repeat(np.arange(n_q * 2), per_q // 2)
    a, b = rng.normal(size=n), rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(2.0 * a - 1.0 * b - 1.0)))).astype(int)
    return {"out_entropy": a, "r_ent_n@0": b}, y, groups, responses


def _hand_detector(**overrides):
    params = dict(
        model=INFO.fingerprint(),
        feature_names=["r_ent_n@0", "out_entropy"],
        mean=[0.5, 2.0],
        std=[0.25, 1.0],
        weights=[-1.0, 2.0],
        bias=0.5,
        platt={"a": 1.5, "b": -0.25},
        response_calibrator={"a": 2.0, "b": 0.1},
        thresholds=[
            {"alpha": 0.1, "tau": 0.2, "expected_flag_rate": 0.4, "expected_precision": 0.5},
            {"alpha": 0.3, "tau": 0.6, "expected_flag_rate": 0.1, "expected_precision": None},
        ],
        default_alpha=0.1,
        metrics={"cal": {"auroc": 0.9}},
        training={
            "dataset_sha256": "0" * 64,
            "n_tokens": 10,
            "n_responses": 2,
            "label_source": "computable",
            "judge": None,
            "created_at": "2026-10-04T00:00:00+00:00",
        },
    )
    params.update(overrides)
    return Detector.build(**params)


def _sigmoid(z):
    return 1 / (1 + math.exp(-z))


def _token(i, feats, **kw):
    return TokenEvent(index=i, token_id=i, text="x", start=i, end=i + 1, features=feats, **kw)


@pytest.fixture(scope="module")
def trained():
    feats, y, groups, responses = _synthetic()
    det, _ = train_detector(feats, y, groups, responses, INFO, c_grid=(1.0,))
    return det, feats


def test_at_m01_json_roundtrip_reproduces_scores_and_hash(trained, tmp_path):
    """AT-M01: guardar y cargar el JSON reproduce los scores con tolerancia 1e-9 y el hash valida."""
    det, feats = trained
    n = len(feats["out_entropy"])
    X = det.matrix_from_features(feats, n)
    path = tmp_path / "det.json"
    det.save(path)
    loaded = Detector.load(path)
    assert loaded.id == det.id == det.sha256[:16]
    assert artifact_sha256(json.loads(path.read_text(encoding="utf-8"))) == det.sha256
    np.testing.assert_allclose(loaded.score_matrix(X), det.score_matrix(X), rtol=0, atol=1e-9)
    tokens = [
        _token(i, {"out_entropy": float(feats["out_entropy"][i]), "r_ent_n@0": float(feats["r_ent_n@0"][i])})
        for i in range(50)
    ]
    again = Detector.loads(det.to_json())
    for p, q in zip(det.score_tokens(tokens), again.score_tokens(tokens), strict=True):
        assert abs(p - q) <= 1e-9
    assert Detector.load(det.artifact).id == det.id


def test_at_m02_tampered_artifact_is_rejected(trained):
    """AT-M02: un JSON con hash alterado falla con error explícito."""
    det, _ = trained
    art = det.artifact
    art["weights"][0] += 1e-6
    with pytest.raises(DetectorMismatchError, match="hash"):
        Detector.load(art)
    art = det.artifact
    art["sha256"] = "f" * 64
    with pytest.raises(DetectorMismatchError, match="hash"):
        Detector.load(art)


def test_at_m02_detector_for_other_model_is_rejected(trained):
    """AT-M02: un detector calibrado para otro modelo falla con error explícito."""
    det, _ = trained
    det.check_model(INFO)
    other = ModelInfo("tiny-mixtral", "mixtral", 2, (0, 1), 8, 2, 100, "softmax_topk")
    with pytest.raises(DetectorMismatchError, match="n_experts"):
        det.check_model(other)
    quantized = ModelInfo(
        "tiny-mixtral", "mixtral", 2, (0, 1), 4, 2, 100, "softmax_topk", quantization="mxfp4"
    )
    with pytest.raises(DetectorMismatchError, match="quantization"):
        det.check_model(quantized)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda a: a.update(extra=1), "sobrantes"),
        (lambda a: a.pop("platt"), "faltantes"),
        (lambda a: a.update(schema_version="9"), "schema_version"),
        (lambda a: a["std"].__setitem__(0, 0.0), "std"),
        (lambda a: a["platt"].__setitem__("a", -1.0), "platt"),
        (lambda a: a["thresholds"].reverse(), "crecientes"),
        (lambda a: a.update(feature_names=["aaa_bad", "out_entropy"]), "feature"),
        (lambda a: a["training"].pop("judge"), "training"),
        (lambda a: a["model"].update(router_sha256="abc"), "router_sha256"),
        (lambda a: a["model"].update(revision="x"), "model"),
        (lambda a: a["metrics"].update(drift_reference=[]), "drift_reference"),
        (
            lambda a: a["metrics"].update(
                drift_reference={"p_error": {"edges": [0.5, 0.2], "fractions": [0.3, 0.3, 0.4], "n": 3}}
            ),
            "creciente",
        ),
        (
            lambda a: a["metrics"].update(
                drift_reference={"p_error": {"edges": [0.5], "fractions": [0.3, 0.3], "n": 3}}
            ),
            "suman 1",
        ),
        (
            lambda a: a["metrics"].update(
                drift_reference={"flag": {"edges": [0.5], "fractions": [0.5, 0.5], "n": 0}}
            ),
            "entero positivo",
        ),
    ],
)
def test_at_m02_structural_errors_are_explicit(mutate, message):
    """AT-M02: un artefacto mal formado, aun con hash recalculado, falla con mensaje claro."""
    art = _hand_detector().artifact
    mutate(art)
    art["sha256"] = artifact_sha256(art)
    with pytest.raises(DetectorMismatchError, match=message):
        Detector.load(art)


def test_load_rejects_non_json_constants_and_unreadable_files(tmp_path):
    text = _hand_detector().to_json().replace('"bias":0.5', '"bias":NaN')
    with pytest.raises(DetectorMismatchError):
        Detector.loads(text)
    with pytest.raises(DetectorMismatchError):
        Detector.loads("{no es json")
    with pytest.raises(DetectorMismatchError):
        Detector.load(tmp_path / "no-existe.json")


def test_feature_names_are_sorted_and_scores_match_hand_formula():
    det = _hand_detector()
    assert det.feature_names == ("out_entropy", "r_ent_n@0")
    tok = _token(0, {"out_entropy": 3.0, "r_ent_n@0": 0.75})
    z = 2.0 * (3.0 - 2.0) / 1.0 + -1.0 * (0.75 - 0.5) / 0.25 + 0.5
    assert det.score_tokens([tok])[0] == pytest.approx(_sigmoid(1.5 * z - 0.25), abs=1e-12)


def test_score_tokens_skips_special_and_non_final_and_imputes_missing():
    det = _hand_detector()
    tokens = [
        _token(0, {"out_entropy": 2.0, "r_ent_n@0": 0.5}),
        _token(1, {"out_entropy": 9.0}, channel="analysis"),
        _token(2, {}, is_special=True),
        _token(3, {"out_entropy": 2.0, "r_ent_n@0": float("nan"), "u_gini@0": 7.0}),
        _token(4, {}),
    ]
    scores = det.score_tokens(tokens)
    assert scores[1] is None and scores[2] is None
    at_mean = _sigmoid(1.5 * 0.5 - 0.25)  # todas las features en la media: z = bias
    assert scores[0] == pytest.approx(at_mean)
    assert scores[3] == pytest.approx(at_mean)
    assert scores[4] == pytest.approx(at_mean)
    assert det.imputed_counts == {"r_ent_n@0": 2, "out_entropy": 1}
    assert det.score_tokens([]) == []


def test_score_response_uses_max_and_response_calibrator():
    det = _hand_detector()
    m = 0.7
    expected = _sigmoid(2.0 * math.log(m / (1 - m)) + 0.1)
    assert det.score_response([0.1, None, m, 0.3]) == pytest.approx(expected)
    assert det.score_response([None, None]) is None
    with pytest.raises(ValueError):
        det.score_response([1.5])


def test_tau_steps_and_refuses_unsupported_alpha():
    """Regresión: "Detector.tau interpola τ linealmente entre alphas guardados y rompe la garantía
    FNR <= alpha": un alpha entre dos guardados usa el tau del alpha guardado inferior."""
    det = _hand_detector()
    assert det.tau(0.1) == pytest.approx(0.2)
    assert det.tau(0.2) == pytest.approx(0.2)  # entre 0,1 y 0,3 guardados: tau de 0,1
    assert det.tau(0.2999) == pytest.approx(0.2)
    assert det.tau(0.3) == pytest.approx(0.6)
    assert det.tau(0.5) == pytest.approx(0.6)
    with pytest.raises(ValueError, match="mínimo"):
        det.tau(0.05)
    with pytest.raises(ValueError):
        det.tau(1.0)


def test_runtime_does_not_import_sklearn_or_scipy():
    code = "import sys, vigia.detector; print('sklearn' in sys.modules, 'scipy' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)  # noqa: S603
    assert out.stdout.split() == ["False", "False"]


def test_tau_accepts_numpy_scalars():
    assert _hand_detector().tau(np.float32(0.3)) == pytest.approx(0.6, abs=1e-6)
    assert _hand_detector().tau(np.float32(0.2)) == pytest.approx(0.2, abs=1e-6)


def test_artifact_keys_accept_optional_router_hash_and_drift_reference():
    """Las claves nuevas son opcionales: el artefacto sin ellas y con ellas valida."""
    art = _hand_detector().artifact
    art["model"]["router_sha256"] = "a" * 64
    art["metrics"]["drift_reference"] = {
        "p_error": {"edges": [0.2, 0.5], "fractions": [0.5, 0.25, 0.25], "n": 4},
        "flag": {"edges": [0.5], "fractions": [0.75, 0.25], "n": 4, "tau": 0.5},
    }
    art["sha256"] = artifact_sha256(art)
    det = Detector.load(art)
    assert det.model["router_sha256"] == "a" * 64
    assert det.drift_reference["flag"]["tau"] == 0.5
    assert _hand_detector().drift_reference is None


def test_check_model_compares_router_hash_only_when_both_have_it():
    import dataclasses

    with_hash = dataclasses.replace(INFO, router_sha256="b" * 64)
    art = _hand_detector().artifact
    art["model"]["router_sha256"] = "a" * 64
    art["sha256"] = artifact_sha256(art)
    det = Detector.load(art)
    det.check_model(INFO)  # el modelo no conoce su hash (proxy vLLM): no se compara
    with pytest.raises(DetectorMismatchError, match="router"):
        det.check_model(with_hash)
    _hand_detector().check_model(with_hash)  # detector previo al campo: sigue valiendo


# --- frontera detección / falsas alarmas y alpha recomendado ----------------------------------------


@pytest.mark.parametrize("path", sorted((ROOT / "detectors").glob("*.json")), ids=lambda p: p.name)
def test_shipped_detectors_without_new_keys_still_load_and_validate_hash(path):
    """Los artefactos previos a recommended_alpha y a las métricas por respuesta siguen cargando: el
    hash valida porque las claves nuevas son opcionales y no se agregan al cargar."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert "recommended_alpha" not in raw
    assert all("expected_response_recall" not in t for t in raw["thresholds"])
    det = Detector.load(path)
    assert det.sha256 == raw["sha256"] == artifact_sha256(raw)
    assert det.recommended_alpha is None and det.pareto() == []
    assert det.tau(det.default_alpha) == pytest.approx(
        next(t["tau"] for t in raw["thresholds"] if t["alpha"] == det.default_alpha)
    )
    assert "recommended_alpha" not in det.artifact  # cargar no reescribe el artefacto


def _with_response_metrics(rows):
    """rows: (alpha, tau, recall, false_alarm) -> thresholds con métricas por respuesta."""
    return [
        {
            "alpha": a,
            "tau": tau,
            "expected_flag_rate": 0.5 - tau / 4,
            "expected_precision": 0.5,
            "expected_response_recall": r,
            "expected_response_false_alarm": f,
        }
        for a, tau, r, f in rows
    ]


def test_pareto_keeps_only_non_dominated_points_and_smaller_alpha_on_ties():
    rows = [
        (0.02, 0.10, 1.00, 0.90),
        (0.04, 0.20, 0.99, 0.60),
        (0.06, 0.20, 0.99, 0.60),  # mismo tau que 0,04: queda el alpha menor
        (0.08, 0.30, 0.95, 0.65),  # dominado por 0,04 (más falsas alarmas y menos detección)
        (0.10, 0.40, 0.96, 0.20),
        (0.12, 0.50, 0.90, 0.20),  # dominado por 0,10 (misma falsa alarma, menos detección)
        (0.14, 0.60, 0.85, 0.15),
    ]
    det = _hand_detector(thresholds=_with_response_metrics(rows), recommended_alpha=0.1, default_alpha=0.1)
    front = det.pareto()
    assert [p["alpha"] for p in front] == [0.14, 0.1, 0.04, 0.02]
    assert set(front[0]) == {"alpha", "tau", "recall", "false_alarm"}
    points = [{"recall": r, "false_alarm": f} for _, _, r, f in rows]
    for p in front:  # ningún punto de la grilla domina a uno de la frontera
        assert not any(
            q["recall"] >= p["recall"]
            and q["false_alarm"] <= p["false_alarm"]
            and q != {"recall": p["recall"], "false_alarm": p["false_alarm"]}
            for q in points
        )
    assert det.recommended_alpha == 0.1
    # tau(alpha) sigue siendo escalón: el alpha guardado inferior.
    assert det.tau(0.11) == 0.4 and det.tau(0.5) == 0.6


def test_pareto_ignores_thresholds_without_false_alarm():
    rows = [(0.1, 0.2, 1.0, None), (0.3, 0.6, 0.9, None)]
    det = _hand_detector(thresholds=_with_response_metrics(rows))
    assert det.pareto() == [] and det.recommended_alpha is None


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda a: a.update(recommended_alpha=0.2), "recommended_alpha"),
        (lambda a: a.update(recommended_alpha="0.1"), "recommended_alpha"),
        (lambda a: a["thresholds"][0].update(expected_response_recall=1.5), "expected_response_recall"),
        (lambda a: a["thresholds"][0].update(expected_response_false_alarm=-0.1), "false_alarm"),
        (lambda a: a["thresholds"][0].update(otra=1), "opcionalmente"),
    ],
)
def test_new_optional_keys_are_validated(mutate, message):
    art = _hand_detector().artifact
    mutate(art)
    art["sha256"] = artifact_sha256(art)
    with pytest.raises(DetectorMismatchError, match=message):
        Detector.load(art)


def test_recommended_alpha_is_covered_by_the_hash():
    art = _hand_detector(recommended_alpha=0.1).artifact
    assert art["recommended_alpha"] == 0.1
    art["recommended_alpha"] = 0.3
    with pytest.raises(DetectorMismatchError, match="hash"):
        Detector.load(art)
