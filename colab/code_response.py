"""Análisis por respuesta de un benchmark de código (`vigia evidence --humaneval`).

Uso: python colab/code_response.py /content/runs/gptoss20b_code
En código, una solución que falla suele tener el error en pocas líneas, así que la etiqueta por
token diluye la señal. Este script mide si las señales agregadas por respuesta predicen que la
solución falla sus tests, y controla el largo de la respuesta (las que fallan suelen ser más largas).
"""

import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from vigia.evaluate import auroc

FEATURES = ("agg_mean:r_ent_n", "agg_mean:r_topk_mass", "agg_mean:u_ihhi", "out_entropy", "out_nll")


def main(run: str) -> None:
    d = np.load(f"{run}/features.npz", allow_pickle=False)
    x, names, y, r = d["X"], [str(n) for n in d["names"]], d["labels"], d["response_ids"].astype(str)
    keep = y != -100
    x, y, r = x[keep], y[keep], r[keep]
    ids = sorted(set(r))
    lab = np.array([int(y[r == i].max()) for i in ids])
    lens = np.array([(r == i).sum() for i in ids])
    print(f"respuestas {len(ids)}: fallan {lab.sum()}, pasan {(lab == 0).sum()}")
    rng = np.random.default_rng(0)

    def ci(score: np.ndarray) -> tuple[float, float]:
        vals = []
        for _ in range(1000):
            s = rng.integers(0, len(ids), len(ids))
            if lab[s].min() != lab[s].max():
                vals.append(auroc(lab[s], score[s]))
        lo, hi = np.percentile(vals, [2.5, 97.5])
        return float(lo), float(hi)

    scores = {}
    for f in FEATURES:
        if f not in names:
            continue
        j = names.index(f)
        for how, fn in (("media", np.nanmean), ("max", np.nanmax)):
            sc = np.array([fn(x[r == i, j]) for i in ids])
            scores[f"{f} ({how})"] = sc
            lo, hi = ci(sc)
            print(f"{f:22s} {how:5s} AUROC {auroc(lab, sc):.3f} [{lo:.3f}; {hi:.3f}]")
    print(f"largo de la respuesta   AUROC {auroc(lab, lens):.3f}")

    cv = StratifiedKFold(5, shuffle=True, random_state=0)

    def cv_auc(cols: list[np.ndarray]) -> float:
        z = np.column_stack(cols)
        z = (z - z.mean(0)) / np.where(z.std(0) > 0, z.std(0), 1)
        p = cross_val_predict(LogisticRegression(), z, lab, cv=cv, method="predict_proba")[:, 1]
        return auroc(lab, p)

    length = np.log(lens)
    router = scores.get("agg_mean:r_ent_n (media)")
    output = scores.get("out_entropy (max)")
    print(f"CV AUROC largo {cv_auc([length]):.3f}")
    if router is not None:
        print(f"CV AUROC largo+router {cv_auc([length, router]):.3f}")
    if output is not None:
        print(f"CV AUROC largo+salida {cv_auc([length, output]):.3f}")
    if router is not None and output is not None:
        print(f"CV AUROC largo+router+salida {cv_auc([length, router, output]):.3f}")

    # Detector por respuesta con todas las señales del router (media por respuesta), L2 elegida
    # por CV interna. Responde si el 0,41 por token era la unidad equivocada o falta de datos.
    from sklearn.linear_model import LogisticRegressionCV

    cols = [j for j, n in enumerate(names) if n.startswith(("r_", "u_", "agg_"))]
    z = np.array([np.nanmean(x[r == i][:, cols], axis=0) for i in ids])
    z = np.column_stack([np.nan_to_num(z), length])
    z = (z - z.mean(0)) / np.where(z.std(0) > 0, z.std(0), 1)
    clf = LogisticRegressionCV(Cs=[1e-3, 1e-2, 1e-1, 1], cv=5, max_iter=2000)
    p = cross_val_predict(clf, z, lab, cv=cv, method="predict_proba")[:, 1]
    print(f"CV AUROC detector por respuesta (router completo + largo, {len(cols)} señales) {auroc(lab, p):.3f}")


if __name__ == "__main__":
    main(sys.argv[1])
