"""Fórmulas puras de las features por token (docs/spec/matriz-features.md).

Todas las funciones son puras y vectorizadas con numpy: operan sobre el último eje y aceptan
dimensiones previas arbitrarias. No guardan estado. El extractor (`vigia.extractor`) y el modo
reducido de vLLM las reutilizan. Las entropías se miden en nats; las "normalizadas" se dividen por
el logaritmo del soporte y caen en [0, 1].
"""

from __future__ import annotations

from typing import Literal

import numpy as np

Gating = Literal["softmax_topk", "topk_softmax", "sigmoid_group", "sigmoid_topk", "sparsemixer"]
TOP_N_OUTPUT = 20  # vLLM entrega como mucho top-20 logprobs
_TINY = 1e-300  # p·log(max(p, _TINY)) = 0 exacto para p = 0


# --- Primitivas ---------------------------------------------------------------------------------


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    m = np.max(z, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    e = np.exp(z - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def log_softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    m = np.max(z, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    shifted = z - m
    return shifted - np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))


def entropy(p: np.ndarray, axis: int = -1) -> np.ndarray:
    """Entropía de Shannon en nats; 0 log 0 = 0. NaN en la entrada se propaga."""
    p = np.asarray(p, dtype=np.float64)
    return -(p * np.log(np.maximum(p, _TINY))).sum(axis=axis)


def normalized_entropy(p: np.ndarray, support: int, axis: int = -1) -> np.ndarray:
    """Entropía dividida por log(support). Con support = 1 la entropía es 0 por definición."""
    h = entropy(p, axis=axis)
    if support <= 1:
        return np.zeros_like(h)
    return h / np.log(support)


def _normalize(w: np.ndarray, axis: int = -1) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64)
    s = np.sum(w, axis=axis, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(s > 0, w / np.where(s > 0, s, 1.0), np.nan)


def _topk_indices(x: np.ndarray, k: int) -> np.ndarray:
    """Índices de los k mayores del último eje, en orden descendente (estable ante empates)."""
    return np.argsort(-x, axis=-1, kind="stable")[..., :k]


# --- F0: distribución de salida -----------------------------------------------------------------


def _top_n(x: np.ndarray, n: int) -> np.ndarray:
    """Los n mayores del último eje en orden descendente, con partición O(V) (V puede ser 200k)."""
    v = x.shape[-1]
    n = min(n, v)
    top = np.partition(x, v - n, axis=-1)[..., v - n :] if n < v else x
    return -np.sort(-top, axis=-1)


def top_n_entropy(log_probs: np.ndarray, n: int = TOP_N_OUTPUT) -> np.ndarray:
    """Entropía de los n tokens más probables, renormalizados (lo único que expone vLLM)."""
    top = _top_n(np.asarray(log_probs, dtype=np.float64), n)
    shifted = top - top[..., :1]
    e = np.exp(shifted)
    se = e.sum(axis=-1)
    return np.log(se) - (e * shifted).sum(axis=-1) / se


def output_features(logits: np.ndarray, targets: np.ndarray) -> dict[str, np.ndarray]:
    """Features F0 por fila: `logits` (..., V) y `targets` (...) con el token emitido."""
    z = np.asarray(logits, dtype=np.float64)
    top = _top_n(z, TOP_N_OUTPUT)
    shifted = z - top[..., :1]
    e = np.exp(shifted)
    se = e.sum(axis=-1)
    log_se = np.log(se)
    targets = np.asarray(targets, dtype=np.int64)
    flat = shifted.reshape(-1, shifted.shape[-1])
    picked = flat[np.arange(flat.shape[0]), targets.reshape(-1)].reshape(targets.shape)
    top_shifted = top - top[..., :1]
    e_top = np.exp(top_shifted)
    se_top = e_top.sum(axis=-1)
    margin = (e_top[..., 0] - e_top[..., 1]) / se if z.shape[-1] > 1 else e_top[..., 0] / se
    return {
        "out_entropy": log_se - (e * shifted).sum(axis=-1) / se,
        "out_nll": log_se - picked,
        "out_margin": margin,
        "out_top20_entropy": np.log(se_top) - (e_top * top_shifted).sum(axis=-1) / se_top,
    }


# --- F1: logits del router ----------------------------------------------------------------------


def router_entropy_n(z: np.ndarray) -> np.ndarray:
    """`r_ent_n`: entropía de softmax(z) sobre los N expertos dividida por log N (ADR A-02)."""
    z = np.asarray(z)
    return normalized_entropy(softmax(z), z.shape[-1])


def topk_weight_entropy(w: np.ndarray) -> np.ndarray:
    """`r_ent_k`: entropía normalizada de los pesos top-k reales del modelo (renormalizados a 1)."""
    w = np.asarray(w)
    return normalized_entropy(_normalize(w), w.shape[-1])


def topk_mass(p: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """`r_topk_mass`: masa de softmax(z) sobre los expertos elegidos."""
    return np.sum(np.take_along_axis(np.asarray(p), np.asarray(idx), axis=-1), axis=-1)


def logit_margins(z: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """(`r_margin12`, `r_margin_kk1`): z_(1) − z_(2) y z_(k) − z_(k+1). NaN si no existen."""
    z = np.asarray(z, dtype=np.float64)
    s = -np.sort(-z, axis=-1)
    n = z.shape[-1]
    nan = np.full(z.shape[:-1], np.nan)
    m12 = s[..., 0] - s[..., 1] if n >= 2 else nan
    mkk1 = s[..., k - 1] - s[..., k] if n > k else nan
    return m12, mkk1


def _sparsemixer(z: np.ndarray, jitter_eps: float) -> tuple[np.ndarray, np.ndarray]:
    """Selección top-2 de Phi-MoE en inferencia (réplica de `sparsemixer` con training=False)."""
    scores = np.asarray(z, dtype=np.float64)
    thr = np.max(scores, axis=-1, keepdims=True)
    sel1 = np.argmax(scores, axis=-1)[..., None]
    factor = np.maximum(np.abs(scores), thr)
    masked = np.where((thr - scores) / factor > 2 * jitter_eps, -np.inf, scores)
    w1 = np.take_along_axis(softmax(masked), sel1, axis=-1)
    masked_scores = scores.copy()
    np.put_along_axis(masked_scores, sel1, -np.inf, axis=-1)
    thr2 = np.max(masked_scores, axis=-1, keepdims=True)
    sel2 = np.argmax(masked_scores, axis=-1)[..., None]
    factor2 = np.maximum(np.abs(scores), thr2)
    masked2 = np.where((thr2 - scores) / factor2 > 2 * jitter_eps, -np.inf, masked_scores)
    w2 = np.take_along_axis(softmax(masked2), sel2, axis=-1)
    return np.concatenate([sel1, sel2], axis=-1), np.concatenate([w1, w2], axis=-1)


def route(
    z: np.ndarray,
    gating: Gating,
    k: int,
    *,
    renormalize: bool = False,
    correction_bias: np.ndarray | None = None,
    n_group: int = 1,
    topk_group: int = 1,
    scaling: float = 1.0,
    jitter_eps: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruye (índices top-k, pesos reales) a partir de los logits del router.

    - `softmax_topk` (Mixtral, Qwen, OLMoE): softmax sobre N, top-k, renormaliza opcional.
    - `topk_softmax` (gpt-oss, Granite): top-k sobre los logits, softmax sobre los k.
    - `sigmoid_group` (DeepSeek-V3): sigmoide, + sesgo de corrección para elegir, top-k por grupos,
      renormaliza opcional y escala. `sigmoid_topk` es el caso sin grupos.
    - `sparsemixer` (Phi-MoE): top-2 secuencial con umbral por jitter.
    """
    z = np.asarray(z, dtype=np.float64)
    n = z.shape[-1]
    if not 1 <= k <= n:
        raise ValueError(f"top_k={k} fuera de rango para {n} expertos")
    if gating == "softmax_topk":
        p = softmax(z)
        idx = _topk_indices(p, k)
        w = np.take_along_axis(p, idx, axis=-1)
        if renormalize:
            w = w / np.sum(w, axis=-1, keepdims=True)
        return idx, w
    if gating == "topk_softmax":
        idx = _topk_indices(z, k)
        return idx, softmax(np.take_along_axis(z, idx, axis=-1))
    if gating in ("sigmoid_group", "sigmoid_topk"):
        s = 1.0 / (1.0 + np.exp(-z))
        choice = s + (0.0 if correction_bias is None else np.asarray(correction_bias, dtype=np.float64))
        if gating == "sigmoid_group" and n_group > 1:
            if n % n_group:
                raise ValueError(f"{n} expertos no se dividen en {n_group} grupos")
            grouped = choice.reshape(*choice.shape[:-1], n_group, n // n_group)
            group_scores = np.sum(-np.sort(-grouped, axis=-1)[..., :2], axis=-1)
            keep = np.zeros_like(group_scores, dtype=bool)
            np.put_along_axis(keep, _topk_indices(group_scores, topk_group), True, axis=-1)
            mask = np.repeat(keep, n // n_group, axis=-1)
            choice = np.where(mask, choice, -np.inf)
        idx = _topk_indices(choice, k)
        w = np.take_along_axis(s, idx, axis=-1)
        if renormalize:
            w = w / (np.sum(w, axis=-1, keepdims=True) + 1e-20)
        return idx, w * scaling
    if gating == "sparsemixer":
        if k != 2:
            raise ValueError("sparsemixer solo define top-2")
        return _sparsemixer(z, jitter_eps)
    raise ValueError(f"gating desconocido: {gating}")


def router_stats(z: np.ndarray, gating: Gating, k: int, **route_kwargs) -> dict[str, np.ndarray]:
    """Features F1 de una capa para cada fila de `z` (..., N), más `idx` y `w` de la selección.

    Equivale a componer `router_entropy_n`, `route`, `topk_weight_entropy`, `topk_mass` y
    `logit_margins`, con un solo ordenamiento: softmax es monótona, así que el top-k de p y el de
    z coinciden en `softmax_topk` y `topk_softmax`.
    """
    z = np.asarray(z, dtype=np.float64)
    lead, n = z.shape[:-1], z.shape[-1]
    if not 1 <= k <= n:
        raise ValueError(f"top_k={k} fuera de rango para {n} expertos")
    z = z.reshape(-1, n)
    rows = np.arange(z.shape[0])[:, None]
    order = np.argsort(-z, axis=-1, kind="stable")
    zs = z[rows, order]
    shifted = z - zs[:, :1]
    e = np.exp(shifted)
    se = e.sum(axis=-1, keepdims=True)
    p = e / se
    if gating in ("softmax_topk", "topk_softmax"):
        idx = order[:, :k]
        mass_w = p[rows, idx]
        if gating == "topk_softmax":
            ew = np.exp(zs[:, :k] - zs[:, :1])
            w = ew / ew.sum(axis=-1, keepdims=True)
        elif route_kwargs.get("renormalize"):
            w = mass_w / mass_w.sum(axis=-1, keepdims=True)
        else:
            w = mass_w
    else:
        idx, w = route(z, gating, k, **route_kwargs)
        mass_w = p[rows, idx]
    wn = w / w.sum(axis=-1, keepdims=True)
    m = z.shape[0]
    out = {
        "r_ent_n": (np.log(se[:, 0]) - (p * shifted).sum(axis=-1)) / np.log(n) if n > 1 else np.zeros(m),
        "r_ent_k": entropy(wn) / np.log(k) if k > 1 else np.zeros(m),
        "r_topk_mass": mass_w.sum(axis=-1),
        "r_margin12": zs[:, 0] - zs[:, 1] if n >= 2 else np.full(m, np.nan),
        "r_margin_kk1": zs[:, k - 1] - zs[:, k] if n > k else np.full(m, np.nan),
    }
    out = {name: v.reshape(lead) for name, v in out.items()}
    out["idx"] = idx.reshape(*lead, k)
    out["w"] = w.reshape(*lead, k)
    return out


# --- F1/U: uso acumulado de expertos ------------------------------------------------------------


def usage_counts(idx: np.ndarray, n_experts: int, weights: np.ndarray | None = None) -> np.ndarray:
    """Vector de uso (..., N) de una selección `idx` (..., k). Sin pesos cuenta 1 por experto (U).

    N se fija explícito: nunca se infiere del mayor índice visto (bug del repo de InnerExpert).
    """
    idx = np.asarray(idx, dtype=np.int64)
    if idx.size and (idx.min() < 0 or idx.max() >= n_experts):
        raise ValueError(f"índice de experto fuera de [0, {n_experts})")
    flat_idx = idx.reshape(-1, idx.shape[-1])
    n_rows = flat_idx.shape[0]
    bins = (np.arange(n_rows)[:, None] * n_experts + flat_idx).ravel()
    w = None
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        w = (w if w.shape == idx.shape else np.broadcast_to(w, idx.shape)).ravel()
    # bincount suma repetidos igual que np.add.at, con un solo recorrido en C.
    flat_out = np.bincount(bins, weights=w, minlength=n_rows * n_experts).astype(np.float64, copy=False)
    return flat_out.reshape(*idx.shape[:-1], n_experts)


def gini_impurity(u: np.ndarray) -> np.ndarray:
    """`u_gini`: 1 − Σ u_i² sobre el uso normalizado (Eq. 10 de InnerExpert)."""
    u = _normalize(u)
    return 1.0 - np.sum(u * u, axis=-1)


def inverse_hhi(u: np.ndarray) -> np.ndarray:
    """`u_ihhi`: (1 / Σ u_i²) / N. Uniforme → 1; un solo experto → 1/N."""
    u = _normalize(u)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 / np.sum(u * u, axis=-1) / u.shape[-1]


def usage_entropy(u: np.ndarray) -> np.ndarray:
    """`u_ent`: entropía del uso normalizado dividida por log N."""
    u = _normalize(u)
    out = normalized_entropy(np.nan_to_num(u), u.shape[-1])
    return np.where(np.isnan(u).any(axis=-1), np.nan, out)


def usage_stats(counts: np.ndarray) -> dict[str, np.ndarray]:
    """`u_gini`, `u_ihhi`, `u_ent` del uso acumulado (..., N). Uso nulo → NaN.

    Equivale a `gini_impurity`, `inverse_hhi` y `usage_entropy`, normalizando una sola vez.
    """
    counts = np.asarray(counts, dtype=np.float64)
    n = counts.shape[-1]
    total = counts.sum(axis=-1, keepdims=True)
    u = np.divide(counts, total, out=np.full_like(counts, np.nan), where=total > 0)  # NaN se propaga
    sq = (u * u).sum(axis=-1)
    ent = entropy(u)
    return {"u_gini": 1.0 - sq, "u_ihhi": 1.0 / sq / n, "u_ent": ent / np.log(n) if n > 1 else ent * 0.0}


def route_switch(prev_idx: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """`r_switch`: 1 − |S(t) ∩ S(t−1)| / k."""
    prev_idx = np.asarray(prev_idx)
    idx = np.asarray(idx)
    shared = (idx[..., :, None] == prev_idx[..., None, :]).any(axis=-1).sum(axis=-1)
    return 1.0 - shared / idx.shape[-1]


# --- F2: salidas por experto --------------------------------------------------------------------


def expert_cosine(e: np.ndarray, w: np.ndarray) -> np.ndarray:
    """`e_cos`: media ponderada del coseno entre pares distintos de expertos elegidos.

    `e` (..., k, d) son las salidas sin ponderar y `w` (..., k) los pesos del router.
    Fórmula: Σ_{i≠j} w_i w_j cos(e_i, e_j) / Σ_{i≠j} w_i w_j. Con k = 1 da NaN.
    """
    e = np.asarray(e, dtype=np.float64)
    w = _normalize(w)
    norms = np.linalg.norm(e, axis=-1, keepdims=True)
    unit = e / np.where(norms > 0, norms, 1.0)
    cos = unit @ np.swapaxes(unit, -1, -2)
    ww = w[..., :, None] * w[..., None, :]
    off = 1.0 - np.eye(e.shape[-2])
    den = np.sum(ww * off, axis=(-1, -2))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, np.sum(ww * off * cos, axis=(-1, -2)) / np.where(den > 0, den, 1.0), np.nan)


def expert_dispersion(e: np.ndarray, w: np.ndarray) -> np.ndarray:
    """`e_disp`: Σ w_i ‖e_i − ē‖² / ‖ē‖² con ē = Σ w_i e_i (pesos renormalizados). ‖ē‖ = 0 → NaN."""
    e = np.asarray(e, dtype=np.float64)
    w = _normalize(w)
    mean = np.sum(w[..., None] * e, axis=-2)
    spread = np.sum(w * np.sum((e - mean[..., None, :]) ** 2, axis=-1), axis=-1)
    den = np.sum(mean * mean, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, spread / np.where(den > 0, den, 1.0), np.nan)


def weighted_jsd(p: np.ndarray, w: np.ndarray) -> np.ndarray:
    """`e_jsd`: Jensen-Shannon generalizada de k distribuciones ponderadas, dividida por log k.

    JSD_w = H(Σ w_i P_i) − Σ w_i H(P_i), con pesos renormalizados. Cae en [0, 1]
    (docs/spec/mf-moe-revision.md, objeción 1). Con k = 1 da 0.
    """
    p = np.asarray(p, dtype=np.float64)
    w = _normalize(w)
    k = p.shape[-2]
    mix = np.sum(w[..., None] * p, axis=-2)
    jsd = entropy(mix) - np.sum(w * entropy(p), axis=-1)
    if k <= 1:
        return np.zeros_like(jsd)
    return np.clip(jsd / np.log(k), 0.0, 1.0)


# --- Agregados entre capas ----------------------------------------------------------------------


def layer_aggregates(values: np.ndarray, axis: int = 0) -> dict[str, np.ndarray]:
    """`mean`, `max`, `std` (poblacional) entre capas ignorando NaN. Todo NaN → NaN."""
    v = np.asarray(values, dtype=np.float64)
    if not np.isnan(v).any():
        mean = v.mean(axis=axis)
        dev = v - np.expand_dims(mean, axis)
        # Misma cuenta que np.std (poblacional), sin recalcular la media.
        return {"mean": mean, "max": v.max(axis=axis), "std": np.sqrt((dev * dev).mean(axis=axis))}
    valid = ~np.isnan(v)
    count = np.sum(valid, axis=axis)
    filled = np.where(valid, v, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.sum(filled, axis=axis) / count
        var = np.sum(np.where(valid, (v - np.expand_dims(mean, axis)) ** 2, 0.0), axis=axis) / count
        mx = np.max(np.where(valid, v, -np.inf), axis=axis)
    none = count == 0
    return {
        "mean": np.where(none, np.nan, mean),
        "max": np.where(none, np.nan, mx),
        "std": np.where(none, np.nan, np.sqrt(var)),
    }
