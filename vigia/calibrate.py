"""Entrenamiento y calibración del detector (P-05).

Pipeline: split train/cal/test agrupado por pregunta, regresión logística L2 con z-score
(Newton/IRLS en numpy), Platt sobre calibración, umbrales por conformal risk control (CRC)
sobre la FNR por token y calibrador aparte para el puntaje de respuesta.

CRC (Angelopoulos et al. 2022): para cada respuesta i con tokens alucinados Y_i, la pérdida es
la fracción de Y_i que queda sin marcar, `L_i(tau) = |{t in Y_i : p_t < tau}| / |Y_i|`, acotada
por B = 1 y monótona. `tau_hat` es el mayor tau con `(n/(n+1)) R_n(tau) + B/(n+1) <= alpha`.
Garantía: la FNR esperada de una respuesta nueva con errores es <= alpha (intercambiabilidad).
Las respuestas sin tokens alucinados tienen pérdida 0 y no entran en n.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from vigia.contracts import ModelInfo, feature_tier
from vigia.detector import Detector, logit, sigmoid
from vigia.labels import IGNORE_LABEL

DEFAULT_ALPHAS = (0.05, 0.1, 0.2, 0.3)
DEFAULT_C_GRID = (0.01, 0.1, 1.0, 10.0)
SPLITS = ("train", "cal", "test")


class CalibrationError(ValueError):
    """Los datos no alcanzan para entrenar o calibrar (una clase, splits vacíos, sin señal)."""


# --- Validación de entradas -------------------------------------------------------------------------


@dataclass
class Dataset:
    """Tokens etiquetados con su pregunta (grupo) y su respuesta. Filas IGNORE ya filtradas."""

    names: list[str]
    X: np.ndarray  # (n, d), NaN = feature faltante
    y: np.ndarray  # (n,) 0/1
    groups: np.ndarray  # (n,) str
    responses: np.ndarray  # (n,) str

    def subset(self, mask: np.ndarray) -> Dataset:
        return Dataset(self.names, self.X[mask], self.y[mask], self.groups[mask], self.responses[mask])


def build_dataset(
    features: Mapping[str, Sequence[float] | np.ndarray],
    labels: Sequence[int] | np.ndarray,
    groups: Sequence[Any] | np.ndarray,
    response_ids: Sequence[Any] | np.ndarray,
) -> Dataset:
    """Valida longitudes, nombres de features y etiquetas; descarta filas IGNORE_LABEL."""
    if not features:
        raise ValueError("no hay features")
    y = np.asarray(labels)
    n = y.shape[0]
    if y.ndim != 1 or n == 0:
        raise ValueError("labels debe ser un vector no vacío")
    if not np.issubdtype(y.dtype, np.integer):
        if not np.all(np.isfinite(y.astype(np.float64))) or np.any(y != np.round(y)):
            raise ValueError("labels debe contener enteros 0, 1 o IGNORE_LABEL")
        y = y.astype(np.int64)
    bad = ~np.isin(y, (0, 1, IGNORE_LABEL))
    if bad.any():
        raise ValueError(f"labels fuera de {{0, 1, {IGNORE_LABEL}}}: {np.unique(y[bad])[:5].tolist()}")
    g = np.asarray(groups).astype(str)
    r = np.asarray(response_ids).astype(str)
    if g.shape != (n,) or r.shape != (n,):
        raise ValueError("groups y response_ids deben tener un valor por token")
    names = sorted(features)
    for name in names:
        feature_tier(name)  # ValueError con el nombre si es desconocida
    X = np.empty((n, len(names)))
    for j, name in enumerate(names):
        col = np.asarray(features[name], dtype=np.float64)
        if col.shape != (n,):
            raise ValueError(f"la feature {name} tiene forma {col.shape}, se esperaba ({n},)")
        X[:, j] = col
    X[~np.isfinite(X)] = np.nan
    # Cada respuesta pertenece a una sola pregunta: si no, el split agrupado tendría fuga.
    pairs = np.unique(np.stack([r, g], axis=1), axis=0)
    if len(np.unique(pairs[:, 0])) != len(pairs):
        raise ValueError("una misma respuesta aparece con dos ids de pregunta distintos")
    keep = y != IGNORE_LABEL
    return Dataset(names, X[keep], y[keep].astype(np.int64), g[keep], r[keep])


def dataset_sha256(data: Dataset) -> str:
    """Hash estable de features, etiquetas, grupos y respuestas."""
    h = hashlib.sha256()
    h.update("\x1f".join(data.names).encode("utf-8"))
    h.update(np.ascontiguousarray(data.X, dtype="<f8").tobytes())
    h.update(np.ascontiguousarray(data.y, dtype="<i8").tobytes())
    h.update("\x1f".join(data.groups.tolist()).encode("utf-8"))
    h.update("\x1f".join(data.responses.tolist()).encode("utf-8"))
    return h.hexdigest()


# --- Split agrupado ---------------------------------------------------------------------------------


def split_groups(
    groups: Sequence[Any] | np.ndarray,
    *,
    cal_frac: float = 0.2,
    test_frac: float = 0.2,
    seed: int = 0,
    labels: Sequence[int] | np.ndarray | None = None,
) -> dict[str, list[str]]:
    """Asigna cada id de pregunta a un solo split. Devuelve {split: [ids]} (ids como str).

    Con `labels` (una etiqueta por token, alineada con `groups`) el split se estratifica por la
    clase mayoritaria de cada pregunta: cada estrato se permuta aparte y aporta a test y cal en
    proporción a su tamaño, con al menos una pregunta por split si el estrato tiene tres o más.
    Así train, cal y test reciben las dos clases mientras el dataset las tenga.
    """
    if not (0 < cal_frac < 1 and 0 <= test_frac < 1 and cal_frac + test_frac < 1):
        raise ValueError("cal_frac en (0, 1), test_frac en [0, 1) y su suma < 1")
    g = np.asarray(groups).astype(str)
    unique = np.unique(g)
    n = len(unique)
    n_test = max(1, round(n * test_frac)) if test_frac > 0 else 0
    n_cal = max(1, round(n * cal_frac))
    if n - n_test - n_cal < 1:
        raise CalibrationError(f"{n} preguntas no alcanzan para tres splits")
    rng = np.random.default_rng(seed)
    if labels is None:
        order = unique[rng.permutation(n)]
        return {
            "test": sorted(order[:n_test].tolist()),
            "cal": sorted(order[n_test : n_test + n_cal].tolist()),
            "train": sorted(order[n_test + n_cal :].tolist()),
        }
    strata = [s[rng.permutation(len(s))] for s in _majority_strata(g, labels, unique)]
    sizes = np.array([len(s) for s in strata])
    test_q = _allocate(n_test, sizes, sizes)
    cal_q = _allocate(n_cal, sizes, sizes - test_q)
    out: dict[str, list[str]] = {"test": [], "cal": [], "train": []}
    for s, t, c in zip(strata, test_q, cal_q, strict=True):
        out["test"] += s[:t].tolist()
        out["cal"] += s[t : t + c].tolist()
        out["train"] += s[t + c :].tolist()
    return {k: sorted(v) for k, v in out.items()}


def _majority_strata(
    g: np.ndarray, labels: Sequence[int] | np.ndarray, unique: np.ndarray
) -> list[np.ndarray]:
    """Ids de pregunta por clase mayoritaria de sus tokens (0 o 1; IGNORE_LABEL no cuenta)."""
    y = np.asarray(labels)
    if y.shape != g.shape:
        raise ValueError("labels debe tener una etiqueta por token, alineada con groups")
    valid = np.isin(y, (0, 1))
    _, inverse = np.unique(g, return_inverse=True)
    pos = np.bincount(inverse[valid], weights=(y[valid] == 1), minlength=len(unique))
    tot = np.bincount(inverse[valid], minlength=len(unique))
    majority = pos * 2 > tot  # empate o sin etiquetas: clase 0
    return [unique[~majority], unique[majority]]


def _allocate(total: int, sizes: np.ndarray, room: np.ndarray) -> np.ndarray:
    """Reparte `total` preguntas entre estratos en proporción a `sizes` (mayor resto), sin superar
    `room`. Un estrato con tres o más preguntas recibe al menos una si el total lo permite."""
    quota = np.zeros(len(sizes), dtype=np.int64)
    if total == 0 or sizes.sum() == 0:
        return quota
    share = total * sizes / sizes.sum()
    quota = np.minimum(np.floor(share).astype(np.int64), room)
    for i in np.argsort(-(share - np.floor(share)), kind="stable"):
        if quota.sum() >= total:
            break
        if quota[i] < room[i]:
            quota[i] += 1
    for i in range(len(sizes)):
        if quota[i] == 0 and sizes[i] >= 3 and room[i] > 0:
            donor = int(np.argmax(quota))
            if quota[donor] > 1:
                quota[donor] -= 1
                quota[i] += 1
    while quota.sum() < total and (room - quota).max() > 0:  # algún estrato sin lugar: completar
        quota[int(np.argmax(room - quota))] += 1
    return quota


def split_masks(groups: np.ndarray, split: Mapping[str, Sequence[str]]) -> dict[str, np.ndarray]:
    """Máscaras booleanas por split. Falla si un id está en dos splits o en ninguno."""
    g = np.asarray(groups).astype(str)
    sets = {name: set(map(str, split.get(name, ()))) for name in SPLITS}
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1 :]:
            if sets[a] & sets[b]:
                raise CalibrationError(f"fuga entre splits: preguntas en {a} y {b}")
    unassigned = set(np.unique(g)) - set().union(*sets.values())
    if unassigned:
        raise CalibrationError(f"{len(unassigned)} preguntas sin split asignado")
    return {name: np.isin(g, list(sets[name])) for name in SPLITS}


# --- Regresión logística (Newton/IRLS) --------------------------------------------------------------


def fit_logistic(
    X: np.ndarray,
    y: np.ndarray,
    *,
    l2: float = 1.0,
    sample_weight: np.ndarray | None = None,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> tuple[np.ndarray, float]:
    """Minimiza `sum w * logloss + l2/2 * ||coef||^2` (sesgo sin penalizar) con Newton.

    Equivale a scikit-learn `LogisticRegression(C=1/l2)`. `y` puede ser blando en [0, 1].
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, d = X.shape
    w = np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    Xa = np.hstack([X, np.ones((n, 1))])
    reg = np.full(d + 1, float(l2))
    reg[-1] = 0.0
    beta = np.zeros(d + 1)

    def objective(b: np.ndarray) -> float:
        z = Xa @ b
        return float(np.sum(w * (np.logaddexp(0.0, z) - y * z)) + 0.5 * np.sum(reg * b * b))

    f = objective(beta)
    for _ in range(max_iter):
        p = sigmoid(Xa @ beta)
        grad = Xa.T @ (w * (p - y)) + reg * beta
        hess = (Xa.T * (w * p * (1.0 - p))) @ Xa + np.diag(reg + 1e-10)
        step = np.linalg.solve(hess, grad)
        t = 1.0
        while True:
            candidate = beta - t * step
            f_new = objective(candidate)
            if f_new <= f - 1e-4 * t * float(grad @ step) or t < 1e-10:
                break
            t *= 0.5
        beta, f_prev, f = candidate, f, f_new
        if np.max(np.abs(t * step)) < tol or abs(f_prev - f) <= tol * max(1.0, abs(f)):
            break
    return beta[:-1], float(beta[-1])


def fit_platt(scores: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Platt 1999 con objetivos suavizados: p = sigmoid(a * score + b)."""
    y = np.asarray(y, dtype=np.float64)
    n_pos, n_neg = float(y.sum()), float(len(y) - y.sum())
    if n_pos == 0 or n_neg == 0:
        raise CalibrationError("Platt necesita las dos clases en calibración")
    soft = np.where(y > 0.5, (n_pos + 1) / (n_pos + 2), 1.0 / (n_neg + 2))
    coef, bias = fit_logistic(np.asarray(scores, dtype=np.float64)[:, None], soft, l2=0.0)
    return float(coef[0]), bias


def _standardize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    Z = (X - mean) / std
    Z[np.isnan(Z)] = 0.0  # imputación con la media = 0 en z-score
    return Z


def _select_c(data: Dataset, c_grid: Sequence[float], seed: int) -> tuple[float, dict[str, float]]:
    """Elige C por AUROC medio en validación cruzada de 3 folds agrupada por pregunta."""
    from vigia.evaluate import SingleClassError, auroc  # import diferido: evaluate depende de calibrate

    if len(c_grid) == 1:
        return float(c_grid[0]), {}
    unique = np.unique(data.groups)
    fold_of = dict(zip(unique, np.random.default_rng(seed + 1).permutation(len(unique)) % 3, strict=True))
    folds = np.array([fold_of[g] for g in data.groups])
    scores: dict[str, float] = {}
    for c in c_grid:
        aucs = []
        for k in range(3):
            tr, va = folds != k, folds == k
            if va.sum() == 0 or len(np.unique(data.y[tr])) < 2:
                continue
            mean = np.nanmean(data.X[tr], axis=0)
            std = np.nanstd(data.X[tr], axis=0)
            std[~(std > 0)] = 1.0
            coef, bias = fit_logistic(_standardize(data.X[tr], mean, std), data.y[tr], l2=1.0 / c)
            try:
                aucs.append(auroc(data.y[va], _standardize(data.X[va], mean, std) @ coef + bias))
            except SingleClassError:
                continue
        if aucs:
            scores[repr(float(c))] = float(np.mean(aucs))
    if not scores:
        return 1.0, {}
    best = max(scores, key=lambda k: scores[k])  # empate: el primero (C más chico del grid)
    return float(best), scores


# --- Conformal risk control -------------------------------------------------------------------------


@dataclass(frozen=True)
class CRCResult:
    alpha: float
    tau: float
    n: int  # respuestas con al menos un token alucinado
    feasible: bool  # False: n demasiado chico para alpha; tau = 0 marca todo


def crc_threshold(p: np.ndarray, y: np.ndarray, response_ids: np.ndarray, alpha: float) -> CRCResult:
    """Mayor tau con `(n/(n+1)) R_n(tau) + 1/(n+1) <= alpha` para la FNR por respuesta."""
    if not 0 < alpha < 1:
        raise ValueError("alpha debe estar en (0, 1)")
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y)
    pos = y == 1
    resp_pos = np.asarray(response_ids).astype(str)[pos]
    if resp_pos.size == 0:
        raise CalibrationError("CRC necesita tokens alucinados en calibración")
    _, inverse, counts = np.unique(resp_pos, return_inverse=True, return_counts=True)
    n = len(counts)
    weight = 1.0 / counts[inverse]  # cada respuesta suma 1 a la pérdida total
    order = np.argsort(p[pos], kind="stable")
    sp, sw = p[pos][order], weight[order]
    uniq, first = np.unique(sp, return_index=True)
    missed_below = np.concatenate([[0.0], np.cumsum(sw)])[first]  # peso con p < candidato
    ok = missed_below + 1.0 <= alpha * (n + 1) + 1e-12
    if not ok.any():
        return CRCResult(alpha, 0.0, n, False)
    return CRCResult(alpha, float(uniq[np.nonzero(ok)[0][-1]]), n, True)


def crc_thresholds(
    p: np.ndarray, y: np.ndarray, response_ids: np.ndarray, alphas: Sequence[float]
) -> tuple[list[dict[str, Any]], list[CRCResult]]:
    """Entradas `thresholds` del artefacto con tasa de marcado y precisión esperadas."""
    entries, results = [], []
    for alpha in sorted(alphas):
        res = crc_threshold(p, y, response_ids, alpha)
        flagged = p >= res.tau
        precision = float(y[flagged].mean()) if flagged.any() else None
        entries.append(
            {
                "alpha": float(alpha),
                "tau": res.tau,
                "expected_flag_rate": float(flagged.mean()),
                "expected_precision": precision,
            }
        )
        results.append(res)
    return entries, results


def _response_max(p: np.ndarray, y: np.ndarray, responses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, inverse = np.unique(responses, return_inverse=True)
    m = np.full(inverse.max() + 1, -np.inf)
    lab = np.zeros(inverse.max() + 1, dtype=np.int64)
    np.maximum.at(m, inverse, p)
    np.maximum.at(lab, inverse, y)
    return m, lab


# --- Entrenamiento completo -------------------------------------------------------------------------


def train_detector(
    features: Mapping[str, Sequence[float] | np.ndarray],
    labels: Sequence[int] | np.ndarray,
    groups: Sequence[Any] | np.ndarray,
    response_ids: Sequence[Any] | np.ndarray,
    model_info: ModelInfo,
    *,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    default_alpha: float = 0.1,
    cal_frac: float = 0.2,
    test_frac: float = 0.2,
    seed: int = 0,
    c_grid: Sequence[float] = DEFAULT_C_GRID,
    split: Mapping[str, Sequence[str]] | None = None,
    label_source: str = "computable",
    judge: str | Mapping[str, Any] | None = None,
    dataset_sha: str | None = None,
    created_at: str | None = None,
) -> tuple[Detector, dict[str, Any]]:
    """Entrena, calibra y fija umbrales. Devuelve (Detector, split_report).

    `features` mapea nombre canónico -> valor por token (NaN = faltante). `groups` es el id de
    pregunta y define el split sin fuga. `split` reutiliza un split previo (`split_report["groups"]`).
    Features sin valores finitos o constantes en train se descartan y se informan.
    """
    from vigia.evaluate import SingleClassError, auroc, brier, ece  # import diferido (ver _select_c)

    if not alphas or not all(0 < a < 1 for a in alphas):
        raise ValueError("alphas debe ser una lista de valores en (0, 1)")
    if not min(alphas) <= default_alpha < 1:
        raise ValueError("default_alpha debe ser >= al menor alpha calibrado")
    data = build_dataset(features, labels, groups, response_ids)
    if split is None:
        split = split_groups(data.groups, cal_frac=cal_frac, test_frac=test_frac, seed=seed, labels=data.y)
    masks = split_masks(data.groups, split)
    train, cal = data.subset(masks["train"]), data.subset(masks["cal"])
    if len(np.unique(data.y)) < 2:
        raise CalibrationError("las etiquetas tienen una sola clase: hacen falta tokens correctos y erróneos")
    for name, part in (("train", train), ("cal", cal)):
        if len(np.unique(part.y)) < 2:
            n_groups = len(np.unique(part.groups))
            raise CalibrationError(
                f"el split {name} quedó con una sola clase ({n_groups} preguntas), aunque el dataset "
                "tiene tokens correctos y erróneos: probá otra semilla o sumá preguntas de la clase "
                "minoritaria"
            )

    # Features utilizables: con valores finitos y varianza en train.
    finite = np.isfinite(train.X)
    dropped: dict[str, str] = {}
    keep = []
    for j, name in enumerate(data.names):
        if not finite[:, j].any():
            dropped[name] = "sin valores finitos en train"
        elif np.nanstd(train.X[:, j]) <= 0:
            dropped[name] = "constante en train"
        else:
            keep.append(j)
    if not keep:
        raise CalibrationError("ninguna feature utilizable en train")
    names = [data.names[j] for j in keep]
    Xtr, Xcal = train.X[:, keep], cal.X[:, keep]
    mean = np.nanmean(Xtr, axis=0)
    std = np.nanstd(Xtr, axis=0)

    c, cv_scores = _select_c(Dataset(names, Xtr, train.y, train.groups, train.responses), c_grid, seed)
    coef, bias = fit_logistic(_standardize(Xtr, mean, std), train.y, l2=1.0 / c)

    z_cal = _standardize(Xcal, mean, std) @ coef + bias
    a, b = fit_platt(z_cal, cal.y)
    if a <= 0:
        raise CalibrationError("el detector no discrimina en calibración (pendiente de Platt <= 0)")
    p_cal = sigmoid(a * z_cal + b)

    thresholds, crc_results = crc_thresholds(p_cal, cal.y, cal.responses, alphas)

    m_cal, y_resp_cal = _response_max(p_cal, cal.y, cal.responses)
    resp_note = "platt"
    if len(np.unique(y_resp_cal)) < 2:
        ra, rb, resp_note = 1.0, 0.0, "identidad: una sola clase de respuesta en calibración"
    else:
        ra, rb = fit_platt(logit(m_cal), y_resp_cal)
        if ra <= 0:
            ra, rb, resp_note = 1.0, 0.0, "identidad: el máximo no ordena respuestas en calibración"

    def summary(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
        out: dict[str, Any] = {"n_tokens": int(len(y)), "prevalence": float(y.mean())}
        try:
            out["auroc"] = auroc(y, p)
        except SingleClassError:
            out["auroc"] = None
        out["ece"] = ece(y, p)
        out["brier"] = brier(y, p)
        return out

    metrics: dict[str, Any] = {"c": c, "cal": summary(cal.y, p_cal)}
    test = data.subset(masks["test"])
    if len(test.y):
        z_test = _standardize(test.X[:, keep], mean, std) @ coef + bias
        metrics["test"] = summary(test.y, sigmoid(a * z_test + b))

    training = {
        "dataset_sha256": dataset_sha or dataset_sha256(data),
        "n_tokens": int(len(train.y) + len(cal.y)),
        "n_responses": int(len(np.unique(np.concatenate([train.responses, cal.responses])))),
        "label_source": label_source,
        "judge": judge,
        "created_at": created_at or dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
    }
    detector = Detector.build(
        model=model_info.fingerprint(),
        feature_names=names,
        mean=mean.tolist(),
        std=std.tolist(),
        weights=coef.tolist(),
        bias=bias,
        platt={"a": a, "b": b},
        response_calibrator={"a": ra, "b": rb},
        thresholds=thresholds,
        default_alpha=default_alpha,
        metrics=metrics,
        training=training,
    )

    def count(part: Dataset) -> dict[str, Any]:
        return {
            "n_groups": int(len(np.unique(part.groups))),
            "n_responses": int(len(np.unique(part.responses))),
            "n_tokens": int(len(part.y)),
            "positive_rate": float(part.y.mean()) if len(part.y) else None,
        }

    split_report = {
        "seed": seed,
        "groups": {k: list(map(str, split.get(k, ()))) for k in SPLITS},
        "counts": {k: count(data.subset(masks[k])) for k in SPLITS},
        "c_selected": c,
        "cv_auroc": cv_scores,
        "dropped_features": dropped,
        "response_calibrator": resp_note,
        "crc": [
            {"alpha": r.alpha, "tau": r.tau, "n_responses": r.n, "feasible": r.feasible} for r in crc_results
        ],
        "detector_id": detector.id,
    }
    return detector, split_report
