"""Entropía semántica + router en factual: ¿suman, y conviene la cascada?

Uso:
  python colab/semantic_entropy.py subset ROWS.jsonl SPLIT.json DATA.jsonl OUT.jsonl
      Escribe las preguntas de test sin aritmética (la aritmética casi no falla).
  python colab/semantic_entropy.py analyze FEATURES.npz ROWS.jsonl SPLIT.json SAMPLE_ROWS.jsonl [...]
      FEATURES/ROWS son de la corrida greedy; cada SAMPLE_ROWS es una corrida muestreada (otra semilla).

La etiqueta es si la respuesta greedy es incorrecta. El puntaje del router se entrena por respuesta
con las preguntas de train+cal (sin aritmética) y se evalúa en test. La entropía semántica es la
entropía de la distribución de clusters de significado entre las muestras (versión discreta).
"""

import json
import re
import sys

import numpy as np
from sklearn.linear_model import LogisticRegressionCV

from vigia.evaluate import auroc
from vigia.labels import extract_numbers, normalize_es, token_f1


def same_meaning(a: str, b: str) -> bool:
    # ponytail: equivalencia por números o solapamiento de palabras, suficiente para respuestas
    # cortas de MKQA; para respuestas largas, cambiar por implicación bidireccional con un NLI.
    na, nb = extract_numbers(a), extract_numbers(b)
    if na or nb:
        return bool(na) and na == nb
    return normalize_es(a) == normalize_es(b) or token_f1(a, b) >= 0.5


def _first_number(text: str) -> int | None:
    # Dígitos del primer número sin separadores: "560 000", "560,000" y "560.000" dan 560000.
    m = re.search(r"\d[\d.,   ]*\d|\d", text)
    return int(re.sub(r"\D", "", m.group())) if m else None


def same_meaning_tolerant(a: str, b: str) -> bool:
    """Como `same_meaning`, pero el primer número va sin separadores, con 2 % de tolerancia (años: exactos).

    Corrige que el parser rioplatense lea "560,000" (miles en inglés, como escribe gpt-oss) como 560.
    """
    xa, xb = _first_number(a), _first_number(b)
    if xa is not None or xb is not None:
        if xa is None or xb is None:
            return False
        if 1000 <= xa <= 2100 or 1000 <= xb <= 2100:  # parece un año: tiene que coincidir exacto
            return xa == xb
        return abs(xa - xb) <= 0.02 * max(xa, xb, 1)
    return normalize_es(a) == normalize_es(b) or token_f1(a, b) >= 0.5


def semantic_entropy(answers: list[str], same=same_meaning) -> float:
    clusters: list[list[str]] = []
    for a in answers:
        for c in clusters:
            if same(a, c[0]):
                c.append(a)
                break
        else:
            clusters.append([a])
    p = np.array([len(c) for c in clusters], float) / len(answers)
    return float(-(p * np.log(p)).sum())


def _rows(path: str) -> dict[str, dict]:
    return {o["id"]: o for o in map(json.loads, open(path, encoding="utf-8"))}


def _test_ids(rows: dict[str, dict], split_path: str) -> tuple[set[str], set[str]]:
    split = json.load(open(split_path, encoding="utf-8"))
    groups = split.get("groups", split)
    ok = {i for i, o in rows.items() if not str(o.get("type", "")).startswith("arit")}
    return set(groups["test"]) & ok, (set(groups["train"]) | set(groups["cal"])) & ok


def subset(rows_path: str, split_path: str, data: str, out: str) -> None:
    test, _ = _test_ids(_rows(rows_path), split_path)
    lines = [ln for ln in open(data, encoding="utf-8") if json.loads(ln)["id"] in test]
    open(out, "w", encoding="utf-8").writelines(lines)
    print(f"{len(lines)} preguntas de test sin aritmética -> {out}")


def analyze(features: str, rows_path: str, split_path: str, sample_paths: list[str]) -> None:
    rows = _rows(rows_path)
    test, train = _test_ids(rows, split_path)
    d = np.load(features, allow_pickle=False)
    names = [str(n) for n in d["names"]]
    keep = d["labels"] != -100
    x, r = d["X"][keep], d["response_ids"].astype(str)[keep]
    cols = [j for j, n in enumerate(names) if n.startswith(("r_", "u_", "agg_"))]
    types = sorted({str(o.get("type")) for o in rows.values()})

    def feats(i: str) -> np.ndarray:
        v = x[r == i][:, cols]
        t = [float(rows[i].get("type") == k) for k in types]
        return np.nan_to_num(np.concatenate([np.nanmean(v, 0), np.nanmax(v, 0), t, [np.log(len(v))]]))

    def usable(ids: set[str]) -> list[str]:
        return sorted(i for i in ids if rows[i]["label_info"].get("correct") is not None and (r == i).any())

    tr, te = usable(train), usable(test)
    samples = [_rows(p) for p in sample_paths]
    te = [i for i in te if all(i in s for s in samples)]
    z_tr = np.array([feats(i) for i in tr])
    mu, sd = z_tr.mean(0), np.where(z_tr.std(0) > 0, z_tr.std(0), 1)
    clf = LogisticRegressionCV(Cs=[1e-3, 1e-2, 1e-1, 1], max_iter=3000)
    clf.fit((z_tr - mu) / sd, [0 if rows[i]["label_info"]["correct"] else 1 for i in tr])

    lab = np.array([0 if rows[i]["label_info"]["correct"] else 1 for i in te])
    router = clf.predict_proba((np.array([feats(i) for i in te]) - mu) / sd)[:, 1]
    answers = [[s[i].get("response") or "" for s in samples] for i in te]
    se = np.array([semantic_entropy(a) for a in answers])
    se_tol = np.array([semantic_entropy(a, same_meaning_tolerant) for a in answers])
    length = np.array([(r == i).sum() for i in te], float)

    def rank(v: np.ndarray) -> np.ndarray:
        return np.argsort(np.argsort(v, kind="stable"), kind="stable") / (len(v) - 1)

    both = (rank(router) + rank(se)) / 2
    both_tol = (rank(router) + rank(se_tol)) / 2
    print(f"train {len(tr)} | test {len(te)} preguntas, fallan {lab.mean():.0%}, {len(samples)} muestras")
    rng = np.random.default_rng(0)
    boots = [b for b in rng.integers(0, len(te), (2000, len(te))) if lab[b].min() != lab[b].max()]
    for name, s in (
        ("largo", length),
        ("router", router),
        ("entropía semántica", se),
        ("semántica, números tolerantes", se_tol),
        ("router + semántica", both),
        ("router + semántica tolerante", both_tol),
    ):
        lo, hi = np.percentile([auroc(lab[b], s[b]) for b in boots], [2.5, 97.5])
        print(f"{name:30s} AUROC {auroc(lab, s):.3f} [{lo:.3f}; {hi:.3f}]")
    n = len(samples)
    for tag, comb in (("estricta", both), ("tolerante", both_tol)):
        diff = [auroc(lab[b], comb[b]) - auroc(lab[b], router[b]) for b in boots]
        lo, hi = np.percentile(diff, [2.5, 97.5])
        print(f"router + semántica {tag} - router: {np.mean(diff):+.3f} [IC 95 %: {lo:+.3f}; {hi:+.3f}]")
        # Cascada: la entropía semántica (n generaciones más) solo corre en la fracción más dudosa
        # según el router; las demás preguntas conservan el puntaje del router, debajo de las dudosas.
        for frac in (0.25, 0.5):
            s = np.where(rank(router) >= 1 - frac, 1 + comb, rank(router))
            print(
                f"  cascada {tag} {frac:.0%} más dudoso: AUROC {auroc(lab, s):.3f}, costo x{1 + n * frac:.2f}"
            )


if __name__ == "__main__":
    if sys.argv[1] == "subset":
        subset(*sys.argv[2:6])
    else:
        analyze(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:])
