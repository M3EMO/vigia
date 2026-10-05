"""Monitor de drift (P-36): PSI del score, la tasa de marcado y features clave del router.

Un cliente que ajusta su modelo, o un tráfico que cambia de dominio, mueve la distribución de las
señales lejos de la del set de calibración. El detector deja de estar calibrado sin aviso. El PSI
(population stability index) compara las dos distribuciones por bins:

    PSI = sum_i (a_i - e_i) * ln(a_i / e_i)

con `e_i` la fracción de la referencia y `a_i` la del tráfico actual en el bin i. Los bins salen de
los cuantiles de la referencia; una referencia con pocos valores distintos (la tasa de marcado es
0/1) usa un bin por valor. Una fracción vacía se lleva a `EPS` para que el logaritmo exista.

Umbrales usuales: PSI < 0,1 estable (`ok`); 0,1 a 0,25 cambio moderado (`warn`); >= 0,25 cambio
fuerte (`alert`), recalibrar.

El artefacto del detector guarda la referencia ya resumida (`summarize`) en
`metrics.drift_reference`, así el runtime no necesita los datos de calibración.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

DEFAULT_BINS = 10
EPS = 1e-4
WARN_PSI = 0.1
ALERT_PSI = 0.25
DRIFT_FEATURES = ("p_error", "flag", "agg_mean:r_ent_n", "out_entropy")
LEVELS = ("ok", "warn", "alert")


def _finite(values: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).ravel()
    return arr[np.isfinite(arr)]


def summarize(values: Sequence[float] | np.ndarray, bins: int = DEFAULT_BINS) -> dict[str, Any]:
    """Resumen de una distribución de referencia: `{edges, fractions, n}`.

    `edges` son los bordes interiores (estrictamente crecientes); el bin i es `(edges[i-1], edges[i]]`
    y los extremos quedan abiertos. Con a lo sumo `bins` valores distintos, cada valor tiene su bin
    (bordes en los puntos medios); si no, los bordes son los cuantiles `1/bins … (bins-1)/bins`.
    """
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 2:
        raise ValueError("bins debe ser un entero >= 2")
    v = _finite(values)
    if v.size == 0:
        raise ValueError("la referencia no tiene valores finitos")
    distinct = np.unique(v)
    if distinct.size <= bins:
        edges = (distinct[:-1] + distinct[1:]) / 2.0
    else:
        edges = np.unique(np.quantile(v, np.linspace(0.0, 1.0, bins + 1)[1:-1]))
    counts = np.bincount(np.searchsorted(edges, v, side="left"), minlength=edges.size + 1)
    return {
        "edges": [float(e) for e in edges],
        "fractions": [float(c) / v.size for c in counts],
        "n": int(v.size),
    }


def psi_from_summary(summary: Mapping[str, Any], actual: Sequence[float] | np.ndarray) -> float:
    """PSI del tráfico `actual` contra una referencia ya resumida con `summarize`."""
    a = _finite(actual)
    if a.size == 0:
        raise ValueError("el tráfico actual no tiene valores finitos")
    edges = np.asarray(summary["edges"], dtype=np.float64)
    expected = np.asarray(summary["fractions"], dtype=np.float64)
    if expected.size != edges.size + 1:
        raise ValueError("fractions debe tener len(edges) + 1 valores")
    counts = np.bincount(np.searchsorted(edges, a, side="left"), minlength=edges.size + 1)
    observed = np.maximum(counts / a.size, EPS)
    expected = np.maximum(expected, EPS)
    return float(np.sum((observed - expected) * np.log(observed / expected)))


def psi(
    expected_scores: Sequence[float] | np.ndarray,
    actual_scores: Sequence[float] | np.ndarray,
    bins: int = DEFAULT_BINS,
) -> float:
    """PSI de `actual_scores` contra `expected_scores`, con bins por cuantiles del esperado."""
    return psi_from_summary(summarize(expected_scores, bins), actual_scores)


def level(value: float) -> str:
    """`ok` (< 0,1), `warn` (>= 0,1) o `alert` (>= 0,25)."""
    if not math.isfinite(value):
        raise ValueError("PSI no finito")
    if value >= ALERT_PSI:
        return "alert"
    if value >= WARN_PSI:
        return "warn"
    return "ok"


def worst_level(levels: Sequence[str]) -> str | None:
    """El nivel más grave de la lista (None si está vacía)."""
    present = [lv for lv in levels if lv in LEVELS]
    return max(present, key=LEVELS.index) if present else None


def drift_report(
    reference: Mapping[str, Any],
    current: Mapping[str, Sequence[float] | np.ndarray],
    features: Sequence[str] = DRIFT_FEATURES,
    *,
    bins: int = DEFAULT_BINS,
) -> dict[str, dict[str, Any]]:
    """`{feature: {psi, level, n}}` para cada feature con referencia y con datos actuales.

    `reference[feature]` es un resumen de `summarize` (con `edges`) o los valores crudos de la
    referencia. `n` es la cantidad de valores actuales usados. Una feature sin referencia o sin
    valores actuales finitos no aparece.
    """
    out: dict[str, dict[str, Any]] = {}
    for name in features:
        ref = reference.get(name)
        values = current.get(name)
        if ref is None or values is None:
            continue
        actual = _finite(values)
        if actual.size == 0:
            continue
        summary = ref if isinstance(ref, Mapping) and "edges" in ref else summarize(ref, bins)
        value = psi_from_summary(summary, actual)
        out[name] = {"psi": value, "level": level(value), "n": int(actual.size)}
    return out


def build_reference(
    p_error: Sequence[float] | np.ndarray,
    tau: float,
    features: Mapping[str, Sequence[float] | np.ndarray] | None = None,
    *,
    bins: int = DEFAULT_BINS,
) -> dict[str, dict[str, Any]]:
    """Referencia que guarda el artefacto (`metrics.drift_reference`) a partir del set de calibración.

    `p_error` son los P(error) calibrados de los tokens de calibración y `tau` el umbral del alpha por
    defecto: `flag` es `p_error >= tau` y su resumen guarda `tau`, así el runtime marca el tráfico con
    el mismo umbral aunque la política del tenant use otro alpha. `features` suma las columnas crudas
    de `DRIFT_FEATURES` que existan (sin valores finitos, se omiten).
    """
    p = _finite(p_error)
    if p.size == 0:
        raise ValueError("p_error no tiene valores finitos")
    flag = summarize((p >= tau).astype(np.float64), bins)
    flag["tau"] = float(tau)
    ref: dict[str, dict[str, Any]] = {"p_error": summarize(p, bins), "flag": flag}
    for name, values in (features or {}).items():
        if name in DRIFT_FEATURES and name not in ref and _finite(values).size:
            ref[name] = summarize(values, bins)
    return ref
