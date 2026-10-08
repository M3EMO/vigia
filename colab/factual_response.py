"""Señales por respuesta en factual: ¿la entropía de salida aporta algo que el largo no explique?

Uso: python colab/factual_response.py FEATURES.npz ROWS.jsonl
Excluye aritmética (gpt-oss casi no falla ahí y el tipo de pregunta se confunde con el error).
Compara señales sin entrenar y detectores por respuesta con validación cruzada, global y por tipo.
"""

import json
import sys

import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from vigia.evaluate import auroc

TYPES = ("date", "entity", "number")


def main(features: str, rows_path: str) -> None:
    d = np.load(features, allow_pickle=False)
    names = [str(n) for n in d["names"]]
    keep = d["labels"] != -100
    x, r = d["X"][keep], d["response_ids"].astype(str)[keep]
    rows = {o["id"]: o for o in map(json.loads, open(rows_path, encoding="utf-8"))}
    ids = [
        i
        for i in dict.fromkeys(r)
        if rows[i]["label_info"].get("correct") is not None
        and not str(rows[i].get("type", "")).startswith("arit")
    ]
    lab = np.array([0 if rows[i]["label_info"]["correct"] else 1 for i in ids])
    typ = np.array([rows[i].get("type") for i in ids])
    idx = {i: np.where(r == i)[0] for i in ids}

    def col(n: str) -> np.ndarray:
        return x[:, names.index(n)]

    def agg(v: np.ndarray, how) -> np.ndarray:
        return np.array([how(v[idx[i]]) for i in ids])

    def row(name: str, s: np.ndarray) -> None:
        per = "  ".join(f"{auroc(lab[typ == t], s[typ == t]):8.3f}" for t in TYPES)
        print(f"{name:40s} {auroc(lab, s):.3f}  {per}")

    oe = col("out_entropy")
    length = np.array([len(idx[i]) for i in ids], float)
    print(f"respuestas sin aritmética: {len(ids)}, fallan {lab.mean():.0%}")
    print(f"{'señal':40s} global  " + "  ".join(f"{t:>8s}" for t in TYPES))
    row("entropía salida, media", agg(oe, np.nanmean))
    row("entropía salida, máximo", agg(oe, np.nanmax))
    row("entropía salida, primer token", agg(oe, lambda a: a[0]))
    row("entropía salida, suma", agg(oe, np.nansum))
    row("NLL salida, suma (=-log P respuesta)", agg(col("out_nll"), np.nansum))
    row("margen top1-top2, mínimo (negado)", -agg(col("out_margin"), np.nanmin))
    row("entropía del router, media", agg(col("agg_mean:r_ent_n"), np.nanmean))
    row("largo de la respuesta", length)

    def feats(cols: list[int]) -> np.ndarray:
        m = np.array([np.nanmean(x[idx[i]][:, cols], 0) for i in ids])
        mx = np.array([np.nanmax(x[idx[i]][:, cols], 0) for i in ids])
        return np.nan_to_num(np.column_stack([m, mx]))

    out = feats([names.index(n) for n in ("out_entropy", "out_nll", "out_margin", "out_top20_entropy")])
    router = feats([j for j, n in enumerate(names) if n.startswith(("r_", "u_", "agg_"))])
    base = np.column_stack([*(np.asarray(typ == t, float) for t in sorted(set(typ))), np.log(length)])
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    for name, z in (
        ("CV tipo + largo", base),
        ("CV tipo + largo + salida", np.column_stack([base, out])),
        ("CV tipo + largo + router", np.column_stack([base, router])),
        ("CV tipo + largo + salida + router", np.column_stack([base, out, router])),
    ):
        z = (z - z.mean(0)) / np.where(z.std(0) > 0, z.std(0), 1)
        clf = LogisticRegressionCV(Cs=[1e-3, 1e-2, 1e-1, 1], max_iter=3000)
        row(name, cross_val_predict(clf, z, lab, cv=cv, method="predict_proba")[:, 1])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
