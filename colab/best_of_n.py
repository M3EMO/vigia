"""Experimento "mejor de N" en código: ¿la respuesta que elige Vigía pasa los tests más seguido?

Uso:
  python colab/best_of_n.py subset RUN data/code_es.jsonl SUBSET.jsonl
      Escribe las tareas del split de test de RUN (la corrida greedy de código).
  python colab/best_of_n.py analyze RUN SAMPLE_DIR [SAMPLE_DIR ...]
      Compara la elección de Vigía contra el azar, la respuesta más corta y el techo (alguna pasa).

El puntaje de riesgo se entrena solo con las tareas de train+cal de RUN (respuestas greedy) y se aplica
a las muestras de las tareas de test. Vigía nunca ve los tests: los tests solo califican la elección.
"""

import json
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

SIGNALS = ("out_entropy", "agg_mean:r_ent_n")


def responses(run: str) -> dict[str, tuple[int, np.ndarray]]:
    """id -> (falla 0/1, [log largo, peor bloque de 5 líneas de cada señal, media de cada señal])."""
    d = np.load(f"{run}/features.npz", allow_pickle=False)
    names = [str(n) for n in d["names"]]
    keep = d["labels"] != -100
    x, r, ti = d["X"][keep], d["response_ids"].astype(str)[keep], d["token_index"][keep]
    cols = [names.index(s) for s in SIGNALS]
    out = {}
    for line in open(f"{run}/rows.jsonl", encoding="utf-8"):
        o = json.loads(line)
        m = r == o["id"]
        if o.get("skipped") or o["label_info"].get("correct") is None or not m.any():
            continue
        resp = o["response"]
        start = {t["i"]: t["start"] for t in o["tokens"]}
        blk = np.array([resp.count("\n", 0, start.get(int(t), 0)) // 5 for t in ti[m]])
        feats = [np.log(m.sum())]
        for c in cols:
            v = x[m, c]
            means = [np.nanmean(v[blk == k]) for k in set(blk) if (blk == k).sum() >= 3]
            feats += [np.nanmax(means) if means else np.nanmean(v), np.nanmean(v)]
        out[o["id"]] = (0 if o["label_info"]["correct"] else 1, np.nan_to_num(np.array(feats)))
    return out


def test_groups(run: str) -> set[str]:
    split = json.load(open(f"{run}/split.json", encoding="utf-8"))
    return set(split.get("groups", split)["test"])


def subset(run: str, data: str, out: str) -> None:
    test = test_groups(run)
    lines = [ln for ln in open(data, encoding="utf-8") if json.loads(ln)["id"] in test]
    open(out, "w", encoding="utf-8").writelines(lines)
    print(f"{len(lines)} tareas de test -> {out}")


def analyze(run: str, sample_dirs: list[str]) -> None:
    test = test_groups(run)
    train = {k: v for k, v in responses(run).items() if k not in test}
    z = np.array([v[1] for v in train.values()])
    mu, sd = z.mean(0), np.where(z.std(0) > 0, z.std(0), 1)
    clf = LogisticRegression(max_iter=2000).fit((z - mu) / sd, [v[0] for v in train.values()])
    samples = [responses(s) for s in sample_dirs]
    tasks = sorted(set.intersection(*(set(s) for s in samples)) & test)
    print(f"puntaje: {len(train)} tareas greedy; prueba: {len(tasks)} tareas x {len(samples)} muestras")

    # por tarea: pasa (0/1) de cada muestra, riesgo de Vigía y largo
    ok = np.array([[1 - s[t][0] for s in samples] for t in tasks])
    risk = np.array(
        [clf.predict_proba((np.array([s[t][1] for s in samples]) - mu) / sd)[:, 1] for t in tasks]
    )
    length = np.array([[s[t][1][0] for s in samples] for t in tasks])
    rows = np.arange(len(tasks))
    picks = {
        "azar (esperado)": ok.mean(1),
        "primera muestra": ok[:, 0].astype(float),
        "la más corta": ok[rows, length.argmin(1)].astype(float),
        "Vigía (menor riesgo)": ok[rows, risk.argmin(1)].astype(float),
        "techo (alguna pasa)": ok.max(1).astype(float),
    }
    rng = np.random.default_rng(0)
    boots = rng.integers(0, len(tasks), (2000, len(tasks)))
    for name, v in picks.items():
        print(f"{name:22s} pasan {v.mean():.1%}")
    for base in ("azar (esperado)", "la más corta"):
        diff = picks["Vigía (menor riesgo)"] - picks[base]
        lo, hi = np.percentile(diff[boots].mean(1), [2.5, 97.5])
        print(f"Vigía - {base}: {diff.mean():+.1%} [IC 95 %: {lo:+.1%}; {hi:+.1%}]")


if __name__ == "__main__":
    if sys.argv[1] == "subset":
        subset(*sys.argv[2:5])
    else:
        analyze(sys.argv[2], sys.argv[3:])
