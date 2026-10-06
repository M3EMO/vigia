"""Métricas, bootstrap por cluster y reporte de evaluación (P-08).

Todas las métricas son numpy puro y aceptan pesos por muestra. El bootstrap remuestrea preguntas
(clusters) con reemplazo: cada réplica es un vector de pesos = cuántas veces salió la pregunta de
cada token. Así el mismo código calcula el valor puntual (pesos 1) y las réplicas, y la comparación
entre detectores es pareada (las mismas réplicas para todos).
"""

from __future__ import annotations

import datetime as dt
import html
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from vigia import charts
from vigia.calibrate import (
    DEFAULT_ALPHAS,
    DEFAULT_C_GRID,
    CalibrationError,
    build_dataset,
    split_groups,
    split_masks,
    train_detector,
)
from vigia.contracts import ModelInfo, Tier, feature_tier
from vigia.detector import Detector

REPORT_SCHEMA_VERSION = "1"
N_BINS = 15
FPR_TARGET = 0.1
RECALL_TARGETS = (0.8, 0.9)
_EPS = 1e-15
_CHUNK_CELLS = 2_000_000  # réplicas x muestras por bloque de bootstrap


class SingleClassError(ValueError):
    """Las etiquetas tienen una sola clase: AUROC y AUPRC no están definidos."""


def _arrays(y: Any, s: Any, w: Any = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    if y.ndim != 1 or y.shape != s.shape or y.size == 0:
        raise ValueError("etiquetas y puntajes deben ser vectores no vacíos del mismo largo")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("las etiquetas deben ser 0 o 1")
    if not np.all(np.isfinite(s)):
        raise ValueError("los puntajes deben ser finitos")
    w = np.ones_like(y) if w is None else np.asarray(w, dtype=np.float64)
    if w.shape != y.shape or np.any(w < 0):
        raise ValueError("los pesos deben ser no negativos y tener el largo de las etiquetas")
    return y, s, w


def _require_two_classes(y: np.ndarray, w: np.ndarray | None = None, what: str = "las etiquetas") -> None:
    w = np.ones_like(y) if w is None else w
    pos, neg = float(np.sum(w * y)), float(np.sum(w * (1 - y)))
    if pos == 0 or neg == 0:
        only = "todas 1 (erróneas)" if neg == 0 else "todas 0 (correctas)"
        raise SingleClassError(f"{what} tienen una sola clase ({only}): AUROC/AUPRC no están definidos")


# --- motor de curvas con pesos ----------------------------------------------------------------------


class _Ranked:
    """Orden por puntaje descendente y fin de cada grupo de empates, calculado una sola vez."""

    def __init__(self, y: np.ndarray, s: np.ndarray):
        self.order = np.argsort(-s, kind="stable")
        self.y = y[self.order]
        ss = s[self.order]
        self.thresholds = ss
        self.ends = np.r_[np.nonzero(np.diff(ss))[0], len(ss) - 1]

    def counts(self, W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """TP y FP acumulados en cada umbral distinto, por réplica: (B, k)."""
        Ws = W[:, self.order]
        tp = np.cumsum(Ws * self.y, axis=1)[:, self.ends]
        fp = np.cumsum(Ws * (1.0 - self.y), axis=1)[:, self.ends]
        return tp, fp


def _curve_stats(ranked: _Ranked, W: np.ndarray) -> dict[str, np.ndarray]:
    tp, fp = ranked.counts(W)
    P, N = tp[:, -1:], fp[:, -1:]
    valid = (P[:, 0] > 0) & (N[:, 0] > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        tpr = np.hstack([np.zeros_like(P), tp / P])
        fpr = np.hstack([np.zeros_like(N), fp / N])
        flagged = tp + fp
        precision = np.where(flagged > 0, tp / np.where(flagged > 0, flagged, 1.0), 0.0)
        out = {
            "auroc": np.sum(np.diff(fpr, axis=1) * (tpr[:, 1:] + tpr[:, :-1]) / 2.0, axis=1),
            "auprc": np.sum(np.diff(tpr, axis=1) * precision, axis=1),
            "recall_at_fpr": np.max(np.where(fpr <= FPR_TARGET + 1e-12, tpr, 0.0), axis=1),
        }
        for r in RECALL_TARGETS:
            out[f"precision_at_recall_{r}"] = np.max(
                np.where(tpr[:, 1:] >= r - 1e-12, precision, 0.0), axis=1
            )
    for v in out.values():
        v[~valid] = np.nan
    return out


class _MassBins:
    """Bins de igual masa con pesos, calculados por réplica.

    Con los puntajes en orden ascendente, una fila cae en el bin floor(W_antes * n_bins / W), donde
    W_antes es el peso de las filas anteriores y W el peso total. Con pesos 1 es floor(i * n_bins /
    n). Los puntajes empatados quedan en el bin de su primera aparición (un empate nunca se parte).
    Una fila con peso k cuenta como k filas: el ECE de una réplica bootstrap es el ECE del dataset
    con las filas repetidas (los bins se recalculan en cada réplica).
    """

    def __init__(self, p: np.ndarray):
        self.order = np.argsort(p, kind="stable")
        ps = p[self.order]
        starts = np.r_[0, np.nonzero(np.diff(ps))[0] + 1]
        self.first = np.repeat(starts, np.diff(np.r_[starts, len(ps)]))

    def sorted_bins(self, W: np.ndarray, n_bins: int) -> np.ndarray:
        """Bin de cada fila en orden ascendente de puntaje, por réplica: (B, n)."""
        Ws = W[:, self.order]
        before = np.hstack([np.zeros((len(Ws), 1)), np.cumsum(Ws, axis=1)[:, :-1]])
        total = Ws.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            b = np.floor(before[:, self.first] * n_bins / total)
        return np.clip(np.nan_to_num(b), 0, n_bins - 1).astype(np.int64)

    def ece(self, y: np.ndarray, p: np.ndarray, W: np.ndarray, n_bins: int) -> np.ndarray:
        bins = self.sorted_bins(W, n_bins)
        gap = W[:, self.order] * (y - p)[self.order]
        B = len(W)
        flat = (bins + n_bins * np.arange(B)[:, None]).ravel()
        per_bin = np.bincount(flat, weights=gap.ravel(), minlength=B * n_bins).reshape(B, n_bins)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.sum(np.abs(per_bin), axis=1) / W.sum(axis=1)


def _bin_index(p: np.ndarray, n_bins: int, w: np.ndarray | None = None) -> np.ndarray:
    """Bin de igual masa (con pesos, ver `_MassBins`) de cada puntaje, en el orden original."""
    w = np.ones(len(p)) if w is None else w
    mass = _MassBins(p)
    idx = np.empty(len(p), dtype=np.int64)
    idx[mass.order] = mass.sorted_bins(w[None, :], n_bins)[0]
    return idx


def _prob_stats(
    y: np.ndarray, p: np.ndarray, flags: np.ndarray, mass: _MassBins | None, W: np.ndarray
) -> dict[str, np.ndarray]:
    """Métricas de probabilidad y de umbral por réplica; sin `mass` no calcula ECE."""
    sw = W.sum(axis=1)
    pc = np.clip(p, _EPS, 1 - _EPS)
    f = flags.astype(np.float64)
    tp, fp, fn = W @ (f * y), W @ (f * (1 - y)), W @ ((1 - f) * y)
    out = {} if mass is None else {"ece": mass.ece(y, p, W, N_BINS)}
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        return out | {
            "brier": W @ ((p - y) ** 2) / sw,
            "log_loss": W @ (-(y * np.log(pc) + (1 - y) * np.log(1 - pc))) / sw,
            "precision": precision,
            "recall": recall,
            "fnr": 1.0 - recall,
            "f1": 2 * tp / (2 * tp + fp + fn),
            "flag_rate": W @ f / sw,
            "prevalence": W @ y / sw,
        }


def _digamma(x: np.ndarray) -> np.ndarray:
    """Función digamma para x > 0, numpy puro: recurrencia hasta x + 6 y serie asintótica."""
    x = np.asarray(x, dtype=float)
    shift = sum(1.0 / (x + i) for i in range(6))
    z = x + 6.0
    z2 = 1.0 / (z * z)
    series = z2 * (
        1 / 12 - z2 * (1 / 120 - z2 * (1 / 252 - z2 * (1 / 240 - z2 * (1 / 132 - z2 * 691 / 32760))))
    )
    return np.log(z) - 0.5 / z - series - shift


def _aurc_stats(y: np.ndarray, p: np.ndarray, W: np.ndarray) -> dict[str, np.ndarray]:
    """AURC: riesgo medio al aceptar respuestas de menor a mayor p_error (menor es mejor).

    Un bloque de p_error empatados se acepta entero: todas sus filas llevan el riesgo acumulado al
    final del bloque, así el valor no depende del orden de las filas (igual que AUROC). La
    referencia óptima acepta primero todas las correctas, una por una. Una fila con peso k cuenta
    como k filas: con C correctas y E erróneas, la k-ésima errónea tiene riesgo k / (C + k), y
    sum_{k=1..E} k / (C + k) = E - C * (psi(C + E + 1) - psi(C + 1)). La forma cerrada hace que el
    E-AURC de una réplica bootstrap sea igual al del dataset con las filas repetidas.
    """
    order = np.argsort(p, kind="stable")
    Ws, ys, ks = W[:, order], y[order], p[order]
    cum_w = np.cumsum(Ws, axis=1)
    cum_err = np.cumsum(Ws * ys, axis=1)
    ends = np.r_[np.nonzero(np.diff(ks))[0], len(ks) - 1]
    block_end = np.repeat(ends, np.diff(np.r_[-1, ends]))
    cum_w, cum_err = cum_w[:, block_end], cum_err[:, block_end]
    with np.errstate(divide="ignore", invalid="ignore"):
        risk = np.where(cum_w > 0, cum_err / np.where(cum_w > 0, cum_w, 1.0), 0.0)
        n = W.sum(axis=1)
        aurc = np.sum(Ws * risk, axis=1) / n
        errors = W @ y
        correct = n - errors
        optimal = (errors - correct * (_digamma(n + 1.0) - _digamma(correct + 1.0))) / n
    return {"aurc": aurc, "e_aurc": aurc - optimal}


# --- métricas públicas (valor puntual) --------------------------------------------------------------


def auroc(y: Any, s: Any, w: Any = None) -> float:
    """Área bajo la ROC con empates (equivale al rango medio)."""
    y, s, w = _arrays(y, s, w)
    _require_two_classes(y, w)
    return float(_curve_stats(_Ranked(y, s), w[None, :])["auroc"][0])


def auprc(y: Any, s: Any, w: Any = None) -> float:
    """Precisión promedio (área bajo PR escalonada). Comparar siempre con la prevalencia."""
    y, s, w = _arrays(y, s, w)
    _require_two_classes(y, w)
    return float(_curve_stats(_Ranked(y, s), w[None, :])["auprc"][0])


def recall_at_fpr(y: Any, s: Any, fpr_target: float = FPR_TARGET) -> float:
    y, s, _ = _arrays(y, s)
    _require_two_classes(y)
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.max(tpr[fpr <= fpr_target + 1e-12]))


def precision_at_recall(y: Any, s: Any, recall_target: float = 0.8) -> float:
    y, s, _ = _arrays(y, s)
    _require_two_classes(y)
    recall, precision, _ = pr_curve(y, s)
    ok = recall >= recall_target - 1e-12
    return float(np.max(precision[ok])) if ok.any() else 0.0


def roc_curve(y: Any, s: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(fpr, tpr, umbrales) en cada puntaje distinto, empezando en (0, 0)."""
    y, s, w = _arrays(y, s)
    _require_two_classes(y)
    ranked = _Ranked(y, s)
    tp, fp = ranked.counts(w[None, :])
    tpr = np.r_[0.0, tp[0] / tp[0, -1]]
    fpr = np.r_[0.0, fp[0] / fp[0, -1]]
    return fpr, tpr, np.r_[np.inf, ranked.thresholds[ranked.ends]]


def pr_curve(y: Any, s: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(recall, precisión, umbrales) en cada puntaje distinto."""
    y, s, w = _arrays(y, s)
    _require_two_classes(y)
    ranked = _Ranked(y, s)
    tp, fp = ranked.counts(w[None, :])
    return tp[0] / tp[0, -1], tp[0] / (tp[0] + fp[0]), ranked.thresholds[ranked.ends]


def ece(y: Any, p: Any, n_bins: int = N_BINS, w: Any = None) -> float:
    """Error de calibración esperado con bins de igual masa (con pesos: masa de los pesos)."""
    y, p, w = _arrays(y, p, w)
    return float(_MassBins(p).ece(y, p, w[None, :], n_bins)[0])


def reliability_bins(y: Any, p: Any, n_bins: int = N_BINS) -> list[dict[str, float]]:
    """Por bin de igual masa: puntaje medio, frecuencia observada, conteo y rango [lo, hi]."""
    y, p, _ = _arrays(y, p)
    idx = _bin_index(p, n_bins)
    out = []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            out.append(
                {
                    "mean_pred": float(p[m].mean()),
                    "frac_pos": float(y[m].mean()),
                    "count": int(m.sum()),
                    "lo": float(p[m].min()),
                    "hi": float(p[m].max()),
                }
            )
    return out


def brier(y: Any, p: Any, w: Any = None) -> float:
    y, p, w = _arrays(y, p, w)
    return float(np.sum(w * (p - y) ** 2) / np.sum(w))


def log_loss(y: Any, p: Any, w: Any = None) -> float:
    y, p, w = _arrays(y, p, w)
    pc = np.clip(p, _EPS, 1 - _EPS)
    return float(np.sum(w * -(y * np.log(pc) + (1 - y) * np.log(1 - pc))) / np.sum(w))


def threshold_metrics(y: Any, p: Any, tau: float) -> dict[str, float | None]:
    """Precisión, recall, F1, FNR y tasa de marcado con la regla `p >= tau`."""
    y, p, w = _arrays(y, p)
    stats = _prob_stats(y, p, p >= tau, None, w[None, :])
    return {k: _num(stats[k][0]) for k in ("precision", "recall", "f1", "fnr", "flag_rate")}


def span_iou_per_response(flags: Any, labels: Any, response_ids: Any) -> tuple[np.ndarray, np.ndarray]:
    """IoU entre tokens marcados y tokens alucinados por respuesta (ambos vacíos = 1)."""
    f = np.asarray(flags).astype(bool)
    y = np.asarray(labels).astype(bool)
    resp = np.asarray(response_ids).astype(str)
    uniq, inv = np.unique(resp, return_inverse=True)
    inter = np.bincount(inv, weights=(f & y), minlength=len(uniq))
    union = np.bincount(inv, weights=(f | y), minlength=len(uniq))
    iou = np.where(union > 0, inter / np.where(union > 0, union, 1.0), 1.0)
    return uniq, iou


def span_iou(flags: Any, labels: Any, response_ids: Any) -> float:
    return float(span_iou_per_response(flags, labels, response_ids)[1].mean())


def aurc(y: Any, p: Any) -> dict[str, float]:
    """AURC y E-AURC por respuesta (y = 1 si la respuesta es errónea)."""
    y, p, w = _arrays(y, p)
    return {k: float(v[0]) for k, v in _aurc_stats(y, p, w[None, :]).items()}


# --- bootstrap --------------------------------------------------------------------------------------


def _num(x: Any) -> float | None:
    x = float(x)
    return x if np.isfinite(x) else None


def _cluster_index(clusters: np.ndarray) -> tuple[np.ndarray, int]:
    _, inv = np.unique(np.asarray(clusters).astype(str), return_inverse=True)
    return inv, int(inv.max()) + 1


def cluster_bootstrap_ci(
    metric: Any, y: Any, s: Any, clusters: Any, *, n_boot: int = 2000, seed: int = 0, level: float = 0.95
) -> tuple[float, float]:
    """IC percentil de `metric(y, s, w)` remuestreando clusters con reemplazo.

    Función general (un llamado por réplica). El reporte usa un motor vectorizado equivalente.
    """
    y, s, _ = _arrays(y, s)
    inv, g = _cluster_index(clusters)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        counts = np.bincount(rng.integers(0, g, size=g), minlength=g).astype(np.float64)
        try:
            values.append(metric(y, s, counts[inv]))
        except SingleClassError:
            continue
    if not values:
        raise SingleClassError("ninguna réplica bootstrap tiene las dos clases")
    tail = (1 - level) / 2 * 100
    return float(np.percentile(values, tail)), float(np.percentile(values, 100 - tail))


class _Level:
    """Arrays de un nivel (token o respuesta) para todas las entradas del reporte."""

    def __init__(self, y: np.ndarray, cluster_idx: np.ndarray):
        self.y = y.astype(np.float64)
        self.cluster_idx = cluster_idx
        self.entries: dict[str, dict[str, Any]] = {}

    def add(
        self,
        name: str,
        p: np.ndarray,
        flags: np.ndarray,
        *,
        per_item: Mapping[str, np.ndarray] | None = None,
        with_aurc: bool = False,
    ) -> None:
        """`per_item`: valores por muestra que se promedian con los pesos (ej. IoU por respuesta)."""
        self.entries[name] = {
            "p": p,
            "flags": flags,
            "ranked": _Ranked(self.y, p) if len(np.unique(self.y)) == 2 else None,
            "mass": _MassBins(p),
            "per_item": dict(per_item or {}),
            "with_aurc": with_aurc,
        }

    def stats(self, W: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
        out = {}
        for name, e in self.entries.items():
            st = _prob_stats(self.y, e["p"], e["flags"], e["mass"], W)
            if e["ranked"] is not None:
                st.update(_curve_stats(e["ranked"], W))
            if e["with_aurc"]:
                st.update(_aurc_stats(self.y, e["p"], W))
            for key, values in e["per_item"].items():
                st[key] = W @ values / W.sum(axis=1)
            out[name] = st
        return out


def _bootstrap(
    levels: Mapping[str, _Level], n_groups: int, reference: str, *, n_boot: int, seed: int, level: float
) -> dict[str, dict[str, dict[str, Any]]]:
    """Valor puntual, IC percentil y diferencia pareada contra `reference` para cada nivel y entrada."""
    point = {lv: L.stats(np.ones((1, len(L.y)))) for lv, L in levels.items()}
    samples: dict[str, dict[str, dict[str, list[np.ndarray]]]] = {
        lv: {name: {} for name in L.entries} for lv, L in levels.items()
    }
    rng = np.random.default_rng(seed)
    n_max = max(len(L.y) for L in levels.values())
    chunk = max(1, min(n_boot, _CHUNK_CELLS // max(1, n_max)))
    done = 0
    while done < n_boot:
        b = min(chunk, n_boot - done)
        draws = rng.integers(0, n_groups, size=(b, n_groups)) + np.arange(b)[:, None] * n_groups
        counts = np.bincount(draws.ravel(), minlength=b * n_groups).reshape(b, n_groups).astype(np.float64)
        for lv, L in levels.items():
            for name, st in L.stats(counts[:, L.cluster_idx]).items():
                for key, values in st.items():
                    samples[lv][name].setdefault(key, []).append(values)
        done += b
    tail = (1 - level) / 2 * 100
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for lv, L in levels.items():
        out[lv] = {}
        for name in L.entries:
            metrics = {}
            for key, chunks in samples[lv][name].items():
                values = np.concatenate(chunks)
                ok = np.isfinite(values)
                ci = (
                    [_num(np.percentile(values[ok], tail)), _num(np.percentile(values[ok], 100 - tail))]
                    if ok.any()
                    else None
                )
                metrics[key] = {
                    "value": _num(point[lv][name][key][0]),
                    "ci": ci,
                    "n_boot_valid": int(ok.sum()),
                }
            if name != reference and reference in L.entries:
                for key in ("auroc", "auprc"):
                    if key in samples[lv][name] and key in samples[lv][reference]:
                        delta = np.concatenate(samples[lv][name][key]) - np.concatenate(
                            samples[lv][reference][key]
                        )
                        ok = np.isfinite(delta)
                        metrics[f"delta_{key}_vs_detector"] = {
                            "value": _num(point[lv][name][key][0] - point[lv][reference][key][0]),
                            "ci": [
                                _num(np.percentile(delta[ok], tail)),
                                _num(np.percentile(delta[ok], 100 - tail)),
                            ]
                            if ok.any()
                            else None,
                            "p_delta_ge_0": _num(np.mean(delta[ok] >= 0)) if ok.any() else None,
                        }
            out[lv][name] = metrics
    return out


# --- reporte ----------------------------------------------------------------------------------------

TIERS = {
    "F0": (Tier.F0,),
    "F1": (Tier.F1,),
    "F0+F1": (Tier.F0, Tier.F1),
}


def _downsample(*arrays: np.ndarray, max_points: int = 200) -> list[list[float]]:
    n = len(arrays[0])
    idx = np.unique(np.linspace(0, n - 1, min(n, max_points)).round().astype(int)) if n else np.array([], int)
    return [[float(v) for v in a[idx]] for a in arrays]


def _response_scores(det: Detector, p: np.ndarray, y: np.ndarray, resp: np.ndarray, groups: np.ndarray):
    """Por respuesta: p_error calibrado, etiqueta (max), máximo por token y grupo."""
    uniq, inv = np.unique(resp, return_inverse=True)
    m = np.full(len(uniq), -np.inf)
    lab = np.zeros(len(uniq))
    np.maximum.at(m, inv, p)
    np.maximum.at(lab, inv, y)
    grp = np.empty(len(uniq), dtype=object)
    grp[inv] = groups
    p_resp = np.array([det.score_response([float(v)]) for v in m])
    return uniq, p_resp, lab, m, grp.astype(str)


def _feature_sets(names: Sequence[str]) -> dict[str, list[str]]:
    tiers = {n: feature_tier(n) for n in names}
    sets = {key: [n for n in names if tiers[n] in allowed] for key, allowed in TIERS.items()}
    sets["todos"] = list(names)
    return sets


def evaluate_report(
    features: Mapping[str, Sequence[float] | np.ndarray],
    labels: Sequence[int] | np.ndarray,
    groups: Sequence[Any] | np.ndarray,
    response_ids: Sequence[Any] | np.ndarray,
    model_info: ModelInfo,
    *,
    detector: Detector | None = None,
    split: Mapping[str, Sequence[str]] | None = None,
    alpha: float | None = None,
    n_boot: int = 2000,
    seed: int = 0,
    level: float = 0.95,
    cal_frac: float = 0.2,
    test_frac: float = 0.2,
    c_grid: Sequence[float] = DEFAULT_C_GRID,
) -> dict[str, Any]:
    """Reporte canónico sobre el split de test con IC por bootstrap de cluster (pregunta).

    Entrena un detector por nivel (F0, F1, F0+F1, todos) y tres baselines (`out_entropy`,
    `out_nll` y una sonda lineal sobre `h_*`) con el mismo split. `detector` es el detector a
    reportar; si falta, se usa el de nivel F0+F1. Falla con SingleClassError si las etiquetas
    tienen una sola clase.
    """
    if n_boot < 1 or not 0 < level < 1:
        raise ValueError("n_boot >= 1 y level en (0, 1)")
    data = build_dataset(features, labels, groups, response_ids)
    _require_two_classes(data.y.astype(np.float64), what="las etiquetas del dataset")
    if split is None:
        split = split_groups(data.groups, cal_frac=cal_frac, test_frac=test_frac, seed=seed, labels=data.y)
    masks = split_masks(data.groups, split)
    test = data.subset(masks["test"])
    if len(test.y) == 0:
        raise CalibrationError("el split de test está vacío")
    _require_two_classes(test.y.astype(np.float64), what="las etiquetas de test")
    if detector is not None:
        detector.check_model(model_info)
        alpha = detector.default_alpha if alpha is None else float(alpha)
        detector.tau(alpha)  # un alpha fuera de lo calibrado falla antes de entrenar comparadores
    elif alpha is not None:
        alpha = float(alpha)
        if not 0 < alpha < 1:
            raise ValueError(f"alpha debe estar en (0, 1), llegó {alpha!r}")
    # Los comparadores se calibran también en el alpha reportado y en los del detector: con un
    # detector calibrado en alphas < min(DEFAULT_ALPHAS), tau(alpha) no existiría en ellos.
    comparator_alphas = set(DEFAULT_ALPHAS)
    if detector is not None:
        comparator_alphas |= {float(t["alpha"]) for t in detector.thresholds}
    if alpha is not None:
        comparator_alphas.add(alpha)

    # Entrenamiento por nivel y baselines (se reutiliza el detector si el set de features coincide).
    cache: dict[tuple[str, ...], Detector | str] = {}

    def train(names: Sequence[str]) -> Detector | str:
        key = tuple(sorted(names))
        if not key:
            return "sin features de este nivel en el dataset"
        if key not in cache:
            try:
                cache[key] = train_detector(
                    {n: features[n] for n in key},
                    labels,
                    groups,
                    response_ids,
                    model_info,
                    split=split,
                    seed=seed,
                    c_grid=c_grid,
                    alphas=sorted(comparator_alphas),
                )[0]
            except CalibrationError as exc:
                cache[key] = f"no se pudo entrenar: {exc}"
        return cache[key]

    sets = _feature_sets(data.names)
    tiers = {key: train(names) for key, names in sets.items()}
    baselines = {
        "out_entropy": train([n for n in data.names if n == "out_entropy"]),
        "out_nll": train([n for n in data.names if n == "out_nll"]),
        "h_probe": train([n for n in data.names if feature_tier(n) is Tier.H]),
    }
    main = (
        detector
        if detector is not None
        else tiers["F0+F1"]
        if isinstance(tiers["F0+F1"], Detector)
        else tiers["todos"]
    )
    if not isinstance(main, Detector):
        raise CalibrationError(f"no hay detector para reportar: {main}")
    alpha = main.default_alpha if alpha is None else float(alpha)
    tau = main.tau(alpha)

    entries: dict[str, Detector] = {"detector": main}
    entries.update({f"tier:{k}": v for k, v in tiers.items() if isinstance(v, Detector)})
    entries.update({f"baseline:{k}": v for k, v in baselines.items() if isinstance(v, Detector)})

    test_features = {n: test.X[:, j] for j, n in enumerate(data.names)}
    g_inv, n_groups = _cluster_index(test.groups)
    group_of = dict(zip(test.groups, g_inv, strict=True))
    token_level = _Level(test.y, g_inv)
    resp_level: _Level | None = None
    per_entry: dict[str, dict[str, Any]] = {}
    resp_labels = None
    for name, det in entries.items():
        p = det.score_matrix(det.matrix_from_features(test_features, len(test.y)))
        t = det.tau(alpha)
        flags = p >= t
        token_level.add(name, p, flags)
        uniq, p_resp, lab, m, grp = _response_scores(det, p, test.y, test.responses, test.groups)
        _, iou = span_iou_per_response(flags, test.y, test.responses)
        if resp_level is None:
            resp_level = _Level(lab, np.array([group_of[gid] for gid in grp]))
            resp_labels = lab
        resp_level.add(name, p_resp, m >= t, per_item={"span_iou": iou}, with_aurc=True)
        missing = [f for f in det.feature_names if f not in features]
        per_entry[name] = {
            "id": det.id,
            "tau": t,
            "n_features": len(det.feature_names),
            "missing_features": missing,
            "p": p,
        }

    assert resp_level is not None and resp_labels is not None
    levels = {"token": token_level}
    response_single_class = len(np.unique(resp_labels)) < 2
    levels["response"] = resp_level
    boot = _bootstrap(levels, n_groups, "detector", n_boot=n_boot, seed=seed, level=level)

    def entry_report(name: str) -> dict[str, Any]:
        info = per_entry[name]
        return {
            "available": True,
            "id": info["id"],
            "n_features": info["n_features"],
            "missing_features": info["missing_features"],
            "alpha": alpha,
            "tau": info["tau"],
            "token": boot["token"][name],
            "response": boot["response"][name],
        }

    def unavailable(reason: str) -> dict[str, Any]:
        return {"available": False, "reason": reason}

    # Curvas del detector principal y ROC de todas las entradas.
    p_main = per_entry["detector"]["p"]
    rec, prec, _ = pr_curve(test.y, p_main)
    roc_series = []
    for name in entries:
        f_, t_, _ = roc_curve(test.y, per_entry[name]["p"])
        xs, ys = _downsample(f_, t_)
        roc_series.append({"name": name, "fpr": xs, "tpr": ys, "auc": boot["token"][name]["auroc"]["value"]})
    edges = np.linspace(0.0, 1.0, 21)
    op = threshold_metrics(test.y, p_main, tau)
    rec_ds, prec_ds = _downsample(rec, prec)
    curves = {
        "roc": roc_series,
        "pr": {
            "recall": rec_ds,
            "precision": prec_ds,
            "ap": boot["token"]["detector"]["auprc"]["value"],
            "threshold_point": {"recall": op["recall"], "precision": op["precision"], "tau": tau},
        },
        "reliability": reliability_bins(test.y, p_main),
        "histogram": {
            "edges": edges.tolist(),
            "correct": np.histogram(p_main[test.y == 0], bins=edges)[0].tolist(),
            "error": np.histogram(p_main[test.y == 1], bins=edges)[0].tolist(),
        },
    }

    # Validación de CRC en test: FNR por respuesta (respuestas con errores) y por token.
    crc = []
    pos = test.y == 1
    for th in main.thresholds:
        flagged = p_main >= th["tau"]
        uniq, inv = np.unique(test.responses[pos], return_inverse=True)
        missed = np.bincount(inv, weights=~flagged[pos], minlength=len(uniq)) / np.bincount(
            inv, minlength=len(uniq)
        )
        crc.append(
            {
                "alpha": th["alpha"],
                "tau": th["tau"],
                "observed_fnr_response": float(missed.mean()),
                "observed_fnr_token": float(1 - flagged[pos].mean()),
                "flag_rate": float(flagged.mean()),
                "precision": _num(test.y[flagged].mean()) if flagged.any() else None,
                "expected_flag_rate": th["expected_flag_rate"],
                "expected_precision": th["expected_precision"],
            }
        )

    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "created_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "model": model_info.fingerprint(),
        "detector_id": main.id,
        "alpha": alpha,
        "tau": tau,
        "bootstrap": {"n_boot": n_boot, "level": level, "seed": seed, "cluster": "pregunta"},
        "dataset": {
            "n_tokens_test": int(len(test.y)),
            "n_responses_test": int(len(resp_labels)),
            "n_groups_test": n_groups,
            "prevalence_token": float(test.y.mean()),
            "prevalence_response": float(resp_labels.mean()),
            "n_groups": {k: len(v) for k, v in split.items()},
        },
        "detector": entry_report("detector"),
        "tiers": {
            k: entry_report(f"tier:{k}") if isinstance(v, Detector) else unavailable(v)
            for k, v in tiers.items()
        },
        "baselines": {
            k: entry_report(f"baseline:{k}") if isinstance(v, Detector) else unavailable(v)
            for k, v in baselines.items()
        },
        "crc": crc,
        "curves": curves,
    }
    if response_single_class:
        note = (
            "las respuestas de test tienen una sola clase: "
            "las métricas de ranking por respuesta no están definidas"
        )
        report["dataset"]["response_note"] = note
    return report


# --- HTML -------------------------------------------------------------------------------------------

_PAGE_CSS = (
    ":root{--vg-page-bg:#FFFFFF;--vg-page-ink:#1F2328;--vg-page-line:#D1D9E0}"
    '@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--vg-page-bg:#0D1117;'
    "--vg-page-ink:#E6EDF3;--vg-page-line:#3D444D}}"
    ':root[data-theme="dark"]{--vg-page-bg:#0D1117;--vg-page-ink:#E6EDF3;--vg-page-line:#3D444D}'
    "body{background:var(--vg-page-bg);color:var(--vg-page-ink);font:15px/1.5 system-ui,sans-serif;"
    "margin:0 auto;max-width:1100px;padding:16px}"
    "table.vg-metrics{border-collapse:collapse;margin:8px 0 24px;display:block;overflow-x:auto}"
    "table.vg-metrics th,table.vg-metrics td{border-bottom:1px solid var(--vg-page-line);padding:4px 10px;"
    "text-align:right;white-space:nowrap}"
    "table.vg-metrics th[scope=row],table.vg-metrics thead th:first-child{text-align:left}"
    ".vg-grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}"
    "figure{margin:0}"
    ".vg-warning{border-left:4px solid #9A6700;padding:4px 12px}"
)


def _e(x: Any) -> str:
    return html.escape(str(x), quote=True)


def _cell(metric: Mapping[str, Any] | None, digits: int = 3, pct: bool = False) -> str:
    if not metric:
        return "s/d"
    f = (lambda v: charts.fmt_pct(v, 1)) if pct else (lambda v: charts.fmt_num(v, digits))
    text = f(metric.get("value"))
    ci = metric.get("ci")
    if ci and ci[0] is not None and ci[1] is not None:
        text += f" [{f(ci[0])}; {f(ci[1])}]"
    return text


def _metrics_table(
    caption: str, columns: Sequence[tuple[str, str, str, bool]], rows: Sequence[tuple[str, Mapping[str, Any]]]
) -> str:
    """columns: (encabezado, nivel, métrica, porcentaje). rows: (nombre, entrada del reporte)."""
    head = '<th scope="col">Detector</th>' + "".join(f'<th scope="col">{_e(c[0])}</th>' for c in columns)
    body = []
    for name, entry in rows:
        if not entry.get("available"):
            body.append(
                f'<tr><th scope="row">{_e(name)}</th><td colspan="{len(columns)}">'
                f"No disponible: {_e(entry.get('reason', ''))}</td></tr>"
            )
            continue
        cells = "".join(
            f"<td>{_e(_cell(entry.get(lv, {}).get(m), pct=pct))}</td>" for _, lv, m, pct in columns
        )
        body.append(f'<tr><th scope="row">{_e(name)}</th>{cells}</tr>')
    return (
        f'<table class="vg-metrics"><caption>{_e(caption)}</caption><thead><tr>{head}</tr></thead>'
        f"<tbody>{''.join(body)}</tbody></table>"
    )


# Tasa de marcado esperada desde la que el umbral CRC deja de ser útil: marca casi todo el texto.
SATURATED_FLAG_RATE = 0.95


def saturation_warning(
    alpha: float, expected_flag_rate: Any, n_cal_responses: int | None = None
) -> str | None:
    """Aviso cuando la garantía CRC de `alpha` obliga a marcar casi todo (tasa esperada >= 0,95).

    Con pocas respuestas de calibración, `(n/(n+1)) R_n(tau) + 1/(n+1) <= alpha` solo se cumple con
    tau = 0: la garantía es correcta, pero el detector marca el 100 % aunque el AUROC sea alto.
    Devuelve None si la tasa es menor o falta.
    """
    if not isinstance(expected_flag_rate, int | float) or not expected_flag_rate >= SATURATED_FLAG_RATE:
        return None
    who = (
        f"Con {charts.fmt_num(n_cal_responses, 0)} respuestas de calibración"
        if isinstance(n_cal_responses, int) and n_cal_responses > 0
        else "Con estos datos de calibración"
    )
    return (
        f"{who}, la garantía de alpha {charts.fmt_num(alpha, 2)} obliga a marcar casi todo "
        f"({charts.fmt_pct(expected_flag_rate)} de los tokens). Sumá datos de calibración o subí alpha."
    )


def render_report_html(report: Mapping[str, Any]) -> str:
    """HTML autocontenido (sin recursos externos) con tablas, IC y los gráficos de `charts`."""
    ds, model = report["dataset"], report["model"]
    rows = [("Detector reportado", report["detector"])]
    rows += [(f"Nivel {k}", v) for k, v in report["tiers"].items()]
    rows += [(f"Baseline {k}", v) for k, v in report["baselines"].items()]
    discrimination = _metrics_table(
        "Tabla 1. Discriminación (IC 95 % por bootstrap de pregunta)",
        [
            ("AUROC token", "token", "auroc", False),
            ("Δ AUROC vs detector", "token", "delta_auroc_vs_detector", False),
            ("AUPRC token", "token", "auprc", False),
            ("Recall con FPR 0,1", "token", "recall_at_fpr", False),
            ("Precisión con recall 0,8", "token", "precision_at_recall_0.8", False),
            ("AUROC respuesta", "response", "auroc", False),
            ("AUPRC respuesta", "response", "auprc", False),
        ],
        rows,
    )
    calibration = _metrics_table(
        "Tabla 2. Calibración por token y por respuesta",
        [
            ("ECE token (15 bins)", "token", "ece", False),
            ("Brier token", "token", "brier", False),
            ("Log loss token", "token", "log_loss", False),
            ("ECE respuesta", "response", "ece", False),
            ("Brier respuesta", "response", "brier", False),
        ],
        rows,
    )
    operation = _metrics_table(
        f"Tabla 3. Operación con alpha = {charts.fmt_num(report['alpha'], 2)}",
        [
            ("Tasa de marcado", "token", "flag_rate", True),
            ("Precisión", "token", "precision", False),
            ("Recall", "token", "recall", False),
            ("F1", "token", "f1", False),
            ("FNR token", "token", "fnr", False),
            ("IoU de fragmentos", "response", "span_iou", False),
            ("AURC", "response", "aurc", False),
            ("E-AURC", "response", "e_aurc", False),
        ],
        rows,
    )
    crc_rows = "".join(
        f'<tr><th scope="row">{_e(charts.fmt_num(c["alpha"], 2))}</th><td>{_e(charts.fmt_num(c["tau"]))}</td>'
        f"<td>{_e(charts.fmt_num(c['observed_fnr_response']))}</td><td>{_e(charts.fmt_num(c['observed_fnr_token']))}</td>"
        f"<td>{_e(charts.fmt_pct(c['flag_rate']))}</td><td>{_e(charts.fmt_pct(c['expected_flag_rate']))}</td>"
        f"<td>{_e(charts.fmt_num(c['precision']))}</td></tr>"
        for c in report["crc"]
    )
    crc_table = (
        '<table class="vg-metrics"><caption>Tabla 4. Validación del umbral CRC en test</caption><thead><tr>'
        '<th scope="col">alpha</th><th scope="col">tau</th><th scope="col">FNR por respuesta</th>'
        '<th scope="col">FNR por token</th><th scope="col">Tasa de marcado</th>'
        '<th scope="col">Tasa esperada</th><th scope="col">Precisión</th></tr></thead>'
        f"<tbody>{crc_rows}</tbody></table>"
    )
    for c in report["crc"]:
        warning = saturation_warning(c["alpha"], c.get("expected_flag_rate"))
        if warning is not None:
            crc_table += f'<p class="vg-warning" role="note"><strong>Aviso:</strong> {_e(warning)}</p>'

    cv = report["curves"]
    labels = {"detector": "Detector"} | {f"baseline:{k}": f"Baseline {k}" for k in report["baselines"]}
    roc_curves = [dict(c, name=labels[c["name"]]) for c in cv["roc"] if c["name"] in labels]
    tok_ece = report["detector"]["token"]["ece"]["value"]
    pr_curves = [{"name": "Detector", **{k: cv["pr"][k] for k in ("recall", "precision", "ap")}}]
    tp = cv["pr"]["threshold_point"]
    figures = [
        charts.figure(
            charts.reliability_svg(cv["reliability"], tok_ece),
            charts.reliability_table(cv["reliability"], tok_ece),
        ),
        charts.figure(charts.roc_svg(roc_curves), charts.roc_table(roc_curves)),
        charts.figure(
            charts.pr_svg(pr_curves, ds["prevalence_token"], threshold_point=tp),
            charts.pr_table(pr_curves, ds["prevalence_token"], threshold_point=tp),
        ),
        charts.figure(
            charts.score_histogram_svg(
                cv["histogram"]["edges"],
                cv["histogram"]["correct"],
                cv["histogram"]["error"],
                tau=report["tau"],
            ),
            charts.score_histogram_table(
                cv["histogram"]["edges"],
                cv["histogram"]["correct"],
                cv["histogram"]["error"],
                tau=report["tau"],
            ),
        ),
    ]
    latency = report.get("latency")
    if isinstance(latency, Mapping) and "baseline" in latency and "vigia" in latency:
        target = latency.get("target", 0.05)
        figures.append(
            charts.figure(
                charts.latency_svg(latency["baseline"], latency["vigia"], target=target),
                charts.latency_table(latency["baseline"], latency["vigia"], target=target),
            )
        )
    note = ds.get("response_note")
    header = (
        f"<p>Modelo: <strong>{_e(model.get('model_id'))}</strong> ({_e(model.get('model_type'))}, "
        f"cuantización {_e(model.get('quantization'))}). Detector <code>{_e(report['detector_id'])}</code>. "
        f"Fecha {_e(report['created_at'])}.</p>"
        f"<p>Test: {ds['n_tokens_test']} tokens, {ds['n_responses_test']} respuestas, "
        f"{ds['n_groups_test']} preguntas. "
        f"Prevalencia por token {_e(charts.fmt_pct(ds['prevalence_token']))}; por respuesta "
        f"{_e(charts.fmt_pct(ds['prevalence_response']))}. Umbral τ = {_e(charts.fmt_num(report['tau']))} "
        f"con alpha = {_e(charts.fmt_num(report['alpha'], 2))}. Bootstrap: "
        f"{report['bootstrap']['n_boot']} réplicas por "
        f"pregunta, percentil {_e(charts.fmt_pct(report['bootstrap']['level'], 0))}.</p>"
        + (f'<p role="note">{_e(note)}</p>' if note else "")
    )
    return (
        '<!doctype html><html lang="es"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>Reporte de evaluación de Vigía</title>"
        f"<style>{charts.CHART_CSS}{_PAGE_CSS}</style></head><body>"
        f"<main><h1>Reporte de evaluación</h1>{header}"
        f"<h2>Discriminación</h2>{discrimination}<h2>Calibración</h2>{calibration}"
        f"<h2>Operación</h2>{operation}{crc_table}"
        f'<h2>Gráficos</h2><div class="vg-grid2">{"".join(figures)}</div></main></body></html>'
    )


__all__ = [
    "SingleClassError",
    "aurc",
    "auprc",
    "auroc",
    "brier",
    "cluster_bootstrap_ci",
    "ece",
    "evaluate_report",
    "log_loss",
    "pr_curve",
    "precision_at_recall",
    "recall_at_fpr",
    "reliability_bins",
    "render_report_html",
    "roc_curve",
    "span_iou",
    "span_iou_per_response",
    "threshold_metrics",
]
