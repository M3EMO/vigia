"""Detector de runtime: regresión logística con z-score, Platt y umbrales CRC (ADR A-04, A-05).

El artefacto es un JSON propio con hash SHA-256 de su forma canónica. El runtime solo usa
numpy: nada de scikit-learn ni pickle. `id` = primeros 16 hex del hash = `calibration_id`.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import numbers
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from vigia.contracts import (
    DETECTOR_SCHEMA_VERSION,
    DetectorMismatchError,
    ModelInfo,
    TokenEvent,
    feature_tier,
)

_FINGERPRINT_KEYS = ("model_id", "model_type", "n_experts", "top_k", "quantization")
_TRAINING_KEYS = ("dataset_sha256", "n_tokens", "n_responses", "label_source", "judge", "created_at")
_ARTIFACT_KEYS = (
    "schema_version",
    "model",
    "feature_names",
    "mean",
    "std",
    "weights",
    "bias",
    "platt",
    "response_calibrator",
    "thresholds",
    "default_alpha",
    "metrics",
    "training",
    "sha256",
)
_LOGIT_CLIP = 1e-12


def canonical_json(obj: Any) -> str:
    """JSON canónico: claves ordenadas, sin espacios, UTF-8, sin NaN ni infinitos."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def artifact_sha256(artifact: Mapping[str, Any]) -> str:
    """SHA-256 de la forma canónica del artefacto sin el campo `sha256`."""
    body = {k: v for k, v in artifact.items() if k != "sha256"}
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


def sigmoid(z: np.ndarray | float) -> np.ndarray:
    """Sigmoide numéricamente estable."""
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def logit(p: np.ndarray | float) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), _LOGIT_CLIP, 1.0 - _LOGIT_CLIP)
    return np.log(p) - np.log1p(-p)


# --- Validación del artefacto -----------------------------------------------------------------------


def _fail(message: str) -> DetectorMismatchError:
    return DetectorMismatchError(f"artefacto de detector inválido: {message}")


def _is_number(x: Any) -> bool:
    return isinstance(x, numbers.Real) and not isinstance(x, bool) and math.isfinite(x)


def _number_list(artifact: Mapping[str, Any], key: str, n: int) -> list[float]:
    values = artifact[key]
    if not isinstance(values, list) or len(values) != n or not all(_is_number(v) for v in values):
        raise _fail(f"`{key}` debe ser una lista de {n} números finitos")
    return [float(v) for v in values]


def _ab(artifact: Mapping[str, Any], key: str) -> tuple[float, float]:
    cal = artifact[key]
    if not isinstance(cal, dict) or set(cal) != {"a", "b"} or not all(_is_number(cal[k]) for k in ("a", "b")):
        raise _fail(f"`{key}` debe ser {{a, b}} con números finitos")
    if cal["a"] <= 0:
        raise _fail(f"`{key}.a` debe ser positivo (un calibrador no invierte el ranking)")
    return float(cal["a"]), float(cal["b"])


def _validate_thresholds(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list) or not raw:
        raise _fail("`thresholds` debe ser una lista no vacía")
    keys = {"alpha", "tau", "expected_flag_rate", "expected_precision"}
    out = []
    for entry in raw:
        if not isinstance(entry, dict) or set(entry) != keys:
            raise _fail(f"cada umbral lleva exactamente {sorted(keys)}")
        alpha, tau, rate, precision = (
            entry[k] for k in ("alpha", "tau", "expected_flag_rate", "expected_precision")
        )
        if not (_is_number(alpha) and 0 < alpha < 1):
            raise _fail(f"alpha fuera de (0, 1): {alpha!r}")
        if not (_is_number(tau) and 0 <= tau <= 1):
            raise _fail(f"tau fuera de [0, 1]: {tau!r}")
        if not (_is_number(rate) and 0 <= rate <= 1):
            raise _fail(f"expected_flag_rate fuera de [0, 1]: {rate!r}")
        if precision is not None and not (_is_number(precision) and 0 <= precision <= 1):
            raise _fail(f"expected_precision fuera de [0, 1]: {precision!r}")
        out.append(entry)
    alphas = [e["alpha"] for e in out]
    taus = [e["tau"] for e in out]
    if any(b <= a for a, b in zip(alphas, alphas[1:], strict=False)):
        raise _fail("los alphas de `thresholds` deben ser estrictamente crecientes")
    if any(b < a for a, b in zip(taus, taus[1:], strict=False)):
        raise _fail("tau debe crecer con alpha (riesgo monótono)")
    return out


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Valida forma, tipos, consistencia y hash. Lanza DetectorMismatchError ante cualquier falla."""
    if not isinstance(artifact, Mapping):
        raise _fail("se esperaba un objeto JSON")
    if set(artifact) != set(_ARTIFACT_KEYS):
        missing = sorted(set(_ARTIFACT_KEYS) - set(artifact))
        extra = sorted(set(artifact) - set(_ARTIFACT_KEYS))
        raise _fail(f"claves faltantes {missing}, sobrantes {extra}")
    if artifact["schema_version"] != DETECTOR_SCHEMA_VERSION:
        raise _fail(f"schema_version {artifact['schema_version']!r}, se esperaba {DETECTOR_SCHEMA_VERSION!r}")
    sha = artifact["sha256"]
    if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise _fail("`sha256` debe ser 64 caracteres hex en minúscula")
    try:
        expected = artifact_sha256(artifact)
    except (TypeError, ValueError) as exc:
        raise _fail(f"no se puede serializar en forma canónica ({exc})") from None
    if not hmac.compare_digest(expected, sha):
        raise DetectorMismatchError("el hash SHA-256 del detector no valida: el artefacto fue alterado")

    model = artifact["model"]
    if not isinstance(model, dict) or set(model) != set(_FINGERPRINT_KEYS):
        raise _fail(f"`model` debe tener exactamente {list(_FINGERPRINT_KEYS)}")
    names = artifact["feature_names"]
    if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
        raise _fail("`feature_names` debe ser una lista no vacía de strings")
    if names != sorted(set(names)):
        raise _fail("`feature_names` debe estar ordenada y sin repetidos")
    for name in names:
        try:
            feature_tier(name)
        except ValueError:
            raise _fail(f"feature desconocida: {name}") from None
    n = len(names)
    _number_list(artifact, "mean", n)
    if any(s <= 0 for s in _number_list(artifact, "std", n)):
        raise _fail("`std` debe ser positivo")
    _number_list(artifact, "weights", n)
    if not _is_number(artifact["bias"]):
        raise _fail("`bias` debe ser un número finito")
    _ab(artifact, "platt")
    _ab(artifact, "response_calibrator")
    thresholds = _validate_thresholds(artifact["thresholds"])
    default_alpha = artifact["default_alpha"]
    if not (_is_number(default_alpha) and thresholds[0]["alpha"] <= default_alpha < 1):
        raise _fail("`default_alpha` debe estar en [min alpha guardado, 1)")
    if not isinstance(artifact["metrics"], dict):
        raise _fail("`metrics` debe ser un objeto")
    training = artifact["training"]
    if not isinstance(training, dict) or set(training) != set(_TRAINING_KEYS):
        raise _fail(f"`training` debe tener exactamente {list(_TRAINING_KEYS)}")
    for key in ("n_tokens", "n_responses"):
        if isinstance(training[key], bool) or not isinstance(training[key], int) or training[key] < 0:
            raise _fail(f"`training.{key}` debe ser un entero no negativo")
    for key in ("dataset_sha256", "label_source", "created_at"):
        if not isinstance(training[key], str):
            raise _fail(f"`training.{key}` debe ser string")


def _reject_constant(name: str) -> Any:
    raise DetectorMismatchError(f"artefacto de detector inválido: constante no JSON {name}")


# --- Detector ---------------------------------------------------------------------------------------


class Detector:
    """Implementa `contracts.Scorer` sobre un artefacto validado."""

    def __init__(self, artifact: Mapping[str, Any]):
        validate_artifact(artifact)
        self._artifact: dict[str, Any] = json.loads(canonical_json(artifact))
        a = self._artifact
        self.sha256: str = a["sha256"]
        self.id: str = self.sha256[:16]
        self.feature_names: tuple[str, ...] = tuple(a["feature_names"])
        self._index = {name: i for i, name in enumerate(self.feature_names)}
        self._mean = np.asarray(a["mean"], dtype=np.float64)
        self._std = np.asarray(a["std"], dtype=np.float64)
        self._weights = np.asarray(a["weights"], dtype=np.float64)
        self._bias = float(a["bias"])
        self._platt = (float(a["platt"]["a"]), float(a["platt"]["b"]))
        self._resp = (float(a["response_calibrator"]["a"]), float(a["response_calibrator"]["b"]))
        self._alphas = np.asarray([t["alpha"] for t in a["thresholds"]], dtype=np.float64)
        self._taus = np.asarray([t["tau"] for t in a["thresholds"]], dtype=np.float64)
        self.default_alpha: float = float(a["default_alpha"])
        self.imputed_counts: Counter[str] = Counter()  # features faltantes imputadas con la media

    # -- construcción y serialización --

    @classmethod
    def build(
        cls,
        *,
        model: Mapping[str, Any],
        feature_names: Sequence[str],
        mean: Sequence[float],
        std: Sequence[float],
        weights: Sequence[float],
        bias: float,
        platt: Mapping[str, float],
        response_calibrator: Mapping[str, float],
        thresholds: Sequence[Mapping[str, Any]],
        default_alpha: float,
        metrics: Mapping[str, Any],
        training: Mapping[str, Any],
    ) -> Detector:
        """Arma el artefacto, calcula su hash y lo valida. Las features se reordenan por nombre."""
        order = sorted(range(len(feature_names)), key=lambda i: feature_names[i])

        def pick(values: Sequence[float]) -> list[float]:
            if len(values) != len(feature_names):
                raise ValueError("mean/std/weights deben tener una entrada por feature")
            return [float(values[i]) for i in order]

        artifact: dict[str, Any] = {
            "schema_version": DETECTOR_SCHEMA_VERSION,
            "model": dict(model),
            "feature_names": [feature_names[i] for i in order],
            "mean": pick(mean),
            "std": pick(std),
            "weights": pick(weights),
            "bias": float(bias),
            "platt": {"a": float(platt["a"]), "b": float(platt["b"])},
            "response_calibrator": {
                "a": float(response_calibrator["a"]),
                "b": float(response_calibrator["b"]),
            },
            "thresholds": [dict(t) for t in sorted(thresholds, key=lambda t: t["alpha"])],
            "default_alpha": float(default_alpha),
            "metrics": json.loads(canonical_json(dict(metrics))),
            "training": json.loads(canonical_json(dict(training))),
        }
        artifact["sha256"] = artifact_sha256(artifact)
        return cls(artifact)

    @classmethod
    def load(cls, source: str | os.PathLike[str] | Mapping[str, Any]) -> Detector:
        """Carga desde una ruta a un JSON o desde un dict ya parseado. Valida hash y forma."""
        if isinstance(source, Mapping):
            return cls(source)
        try:
            text = Path(source).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise DetectorMismatchError(f"no se puede leer el detector: {exc}") from None
        return cls.loads(text)

    @classmethod
    def loads(cls, text: str) -> Detector:
        """Carga desde el texto JSON del artefacto (por ejemplo, la columna `artifact_json`)."""
        try:
            data = json.loads(text, parse_constant=_reject_constant)
        except json.JSONDecodeError as exc:
            raise DetectorMismatchError(
                f"artefacto de detector inválido: JSON ilegible ({exc.msg})"
            ) from None
        return cls(data)

    @property
    def artifact(self) -> dict[str, Any]:
        """Copia profunda del artefacto."""
        return json.loads(canonical_json(self._artifact))

    @property
    def thresholds(self) -> list[dict[str, Any]]:
        return [dict(t) for t in self._artifact["thresholds"]]

    @property
    def model(self) -> dict[str, Any]:
        return dict(self._artifact["model"])

    def to_json(self, *, indent: int | None = None) -> str:
        if indent is None:
            return canonical_json(self._artifact)
        return json.dumps(self._artifact, sort_keys=True, indent=indent, ensure_ascii=False, allow_nan=False)

    def save(self, path: str | os.PathLike[str]) -> None:
        Path(path).write_text(self.to_json(indent=2), encoding="utf-8")

    # -- puntaje --

    def _impute(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.array(X, dtype=np.float64, copy=True)
        if X.ndim != 2 or X.shape[1] != len(self.feature_names):
            raise ValueError(f"se esperaba una matriz (n, {len(self.feature_names)})")
        missing = ~np.isfinite(X)
        if missing.any():
            X[missing] = np.broadcast_to(self._mean, X.shape)[missing]
        return X, missing

    def raw_logit(self, X: np.ndarray) -> np.ndarray:
        """Logit de la regresión antes de Platt. Columnas en el orden de `feature_names`."""
        X, _ = self._impute(X)
        return ((X - self._mean) / self._std) @ self._weights + self._bias

    def score_matrix(self, X: np.ndarray) -> np.ndarray:
        """p_error calibrado por fila. NaN o infinito se imputan con la media de entrenamiento."""
        a, b = self._platt
        return sigmoid(a * self.raw_logit(X) + b)

    def matrix_from_features(
        self, features: Mapping[str, np.ndarray | Sequence[float]], n: int
    ) -> np.ndarray:
        """Arma la matriz (n, d) desde columnas por nombre; una columna ausente queda en NaN."""
        X = np.full((n, len(self.feature_names)), np.nan)
        for j, name in enumerate(self.feature_names):
            if name in features:
                col = np.asarray(features[name], dtype=np.float64)
                if col.shape != (n,):
                    raise ValueError(f"la feature {name} no tiene {n} valores")
                X[:, j] = col
        return X

    def score_tokens(self, tokens: list[TokenEvent]) -> list[float | None]:
        """p_error por token; None para especiales y canales que no son `final`."""
        scorable = [i for i, t in enumerate(tokens) if t.channel == "final" and not t.is_special]
        out: list[float | None] = [None] * len(tokens)
        if not scorable:
            return out
        X = np.full((len(scorable), len(self.feature_names)), np.nan)
        for row, i in enumerate(scorable):
            for name, value in tokens[i].features.items():
                j = self._index.get(name)
                if j is not None and value is not None:
                    X[row, j] = float(value)
        missing = ~np.isfinite(X)
        if missing.any():
            for j in np.nonzero(missing.any(axis=0))[0]:
                self.imputed_counts[self.feature_names[j]] += int(missing[:, j].sum())
        for row, p in zip(scorable, self.score_matrix(X), strict=True):
            out[row] = float(p)
        return out

    def score_response(self, token_p_errors: list[float | None]) -> float | None:
        """Máximo de los p_error por token pasado por el calibrador de respuesta."""
        values = [p for p in token_p_errors if p is not None]
        if not values:
            return None
        if not all(math.isfinite(p) and 0.0 <= p <= 1.0 for p in values):
            raise ValueError("los p_error por token deben estar en [0, 1]")
        a, b = self._resp
        return float(sigmoid(a * logit(max(values)) + b))

    def tau(self, alpha: float) -> float:
        """Umbral por token para FNR <= alpha: el tau del mayor alpha guardado que no supera alpha.

        No interpola: la FNR es la CDF de los p de los tokens erróneos, que puede ser cóncava, y un
        tau interpolado puede superar alpha. El escalón conserva FNR <= alpha guardado <= alpha.
        Un alpha menor que el mínimo guardado no tiene garantía y se rechaza. Un alpha mayor que
        el máximo usa el tau del máximo (más conservador).
        """
        if not (_is_number(alpha) and 0 < alpha < 1):
            raise ValueError(f"alpha debe estar en (0, 1), llegó {alpha!r}")
        if alpha < self._alphas[0]:
            raise ValueError(f"alpha {alpha} es menor que el mínimo calibrado ({self._alphas[0]})")
        return float(self._taus[np.searchsorted(self._alphas, alpha, side="right") - 1])

    def check_model(self, info: ModelInfo) -> None:
        """Lanza DetectorMismatchError si el detector fue calibrado para otro modelo."""
        expected = self._artifact["model"]
        actual = info.fingerprint()
        diffs = [k for k in _FINGERPRINT_KEYS if expected.get(k) != actual.get(k)]
        if diffs:
            detail = ", ".join(f"{k}: {expected.get(k)!r} != {actual.get(k)!r}" for k in diffs)
            raise DetectorMismatchError(f"el detector {self.id} fue calibrado para otro modelo ({detail})")
