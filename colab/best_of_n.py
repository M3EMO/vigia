"""Experimento "mejor de N" en código: ¿la respuesta que elige Vigía pasa los tests más seguido?

Uso:
  python colab/best_of_n.py subset RUN data/code_es.jsonl SUBSET.jsonl
      Escribe las tareas del split de test de RUN (la corrida greedy de código).
  python colab/best_of_n.py analyze RUN SAMPLE_DIR [SAMPLE_DIR ...]
      Compara la elección de Vigía contra el azar, la respuesta más corta y el techo (alguna pasa).

El puntaje entrenado usa solo las tareas de train+cal de RUN (respuestas greedy) y se aplica a las
muestras de las tareas de test. Vigía nunca ve los tests ocultos: solo califican la elección. Los
ejemplos visibles del enunciado (doctest de HumanEval, assert mostrado de MBPP) sí se pueden usar,
porque el usuario también los ve. Ejecuta código del modelo: correr solo en una VM descartable.
"""

import doctest
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from sklearn.linear_model import LogisticRegression

from vigia.codeexec import extract_code, run_tests

SIGNALS = ("out_entropy", "agg_mean:r_ent_n")


def code_mask(response: str, starts: list[int]) -> np.ndarray:
    """True para los tokens en líneas de código: sin comentarios, docstrings, cercos ``` ni líneas vacías.

    En Granite (n=164), la entropía de salida media sobre estos tokens dio AUROC 0,79 contra 0,58 con
    todos: el 63 % de los tokens son prosa (docstrings copiados, comentarios) que diluye la señal.
    """
    kind, in_doc = [], False
    for line in response.split("\n"):
        s = line.strip()
        quotes = s.count('"""') + s.count("'''")
        if in_doc or quotes:
            kind.append(False)
            in_doc ^= quotes % 2 == 1
        else:
            kind.append(bool(s) and not s.startswith(("#", "```")))
    return np.array([kind[response.count("\n", 0, st)] for st in starts], dtype=bool)


def visible_tests(question: str) -> str:
    """Asserts a partir de los ejemplos que el enunciado muestra: `>>>` de doctest y `assert` explícitos."""
    out = []
    for block in re.findall(r"```python\n(.*?)```", question, re.S):
        out += [ln.strip() for ln in block.splitlines() if ln.strip().startswith("assert ")]
        for ex in doctest.DocTestParser().get_examples(block):
            src, want = ex.source.strip(), ex.want.strip()
            if want and "\n" not in src:
                out.append(f"assert ({src}) == ({want})")
    valid = []
    for line in out:
        try:
            compile(line, "<ejemplo>", "exec")
            valid.append(line)
        except SyntaxError:
            continue
    return "\n".join(valid)


def responses(run: str) -> dict[str, dict]:
    """id -> {fail, feats, code_entropy, code, question}.

    feats = [log largo, por cada señal: peor bloque de 5 líneas, media, media sobre tokens de código].
    """
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
        starts = [start.get(int(t), 0) for t in ti[m]]
        blk = np.array([resp.count("\n", 0, s) // 5 for s in starts])
        code = code_mask(resp, starts)
        feats = [np.log(m.sum())]
        for c in cols:
            v = x[m, c]
            means = [np.nanmean(v[blk == k]) for k in set(blk) if (blk == k).sum() >= 3]
            feats += [
                np.nanmax(means) if means else np.nanmean(v),
                np.nanmean(v),
                np.nanmean(v[code]) if code.sum() >= 3 else np.nanmean(v),
            ]
        feats = np.nan_to_num(np.array(feats))
        out[o["id"]] = {
            "fail": 0 if o["label_info"]["correct"] else 1,
            "feats": feats,
            "code_entropy": feats[3],  # out_entropy media sobre tokens de código
            "code": extract_code(resp),
            "question": o["item"]["question"],
        }
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
    train = [v for k, v in responses(run).items() if k not in test]
    z = np.array([v["feats"] for v in train])
    mu, sd = z.mean(0), np.where(z.std(0) > 0, z.std(0), 1)
    clf = LogisticRegression(max_iter=2000).fit((z - mu) / sd, [v["fail"] for v in train])
    samples = [responses(s) for s in sample_dirs]
    tasks = sorted(set.intersection(*(set(s) for s in samples)) & test)
    n = len(samples)
    print(f"puntaje: {len(train)} tareas greedy; prueba: {len(tasks)} tareas x {n} muestras")

    # matrices tarea x muestra
    ok = np.array([[1 - s[t]["fail"] for s in samples] for t in tasks])
    risk = np.array(
        [clf.predict_proba((np.array([s[t]["feats"] for s in samples]) - mu) / sd)[:, 1] for t in tasks]
    )
    ent = np.array([[s[t]["code_entropy"] for s in samples] for t in tasks])
    length = np.array([[s[t]["feats"][0] for s in samples] for t in tasks])
    vis = [visible_tests(samples[0][t]["question"]) for t in tasks]
    jobs = [(a, k) for a in range(len(tasks)) for k in range(n) if vis[a]]
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(lambda j: run_tests(samples[j[1]][tasks[j[0]]]["code"], vis[j[0]]).passed, jobs))
    ex_ok = np.ones_like(ok, dtype=bool)  # sin ejemplos visibles, ninguna muestra queda descartada
    for (a, k), passed in zip(jobs, res, strict=True):
        ex_ok[a, k] = passed
    ex_ok[~ex_ok.any(1)] = True  # si ninguna pasa sus ejemplos, se elige entre todas
    print(f"tareas con ejemplos visibles: {sum(map(bool, vis))}/{len(tasks)}")

    rows = np.arange(len(tasks))
    rng = np.random.default_rng(0)

    def pick(score: np.ndarray, allowed: np.ndarray | None = None) -> np.ndarray:
        s = np.where(allowed, score, np.inf) if allowed is not None else score
        return ok[rows, s.argmin(1)].astype(float)

    picks = {
        "azar (esperado)": ok.mean(1),
        "la más corta": pick(length),
        "menor entropía en código": pick(ent),
        "Vigía (puntaje entrenado)": pick(risk),
        "ejemplos + azar": np.array([ok[a][ex_ok[a]].mean() for a in rows]),
        "ejemplos + la más corta": pick(length, ex_ok),
        "ejemplos + menor entropía": pick(ent, ex_ok),
        "ejemplos + Vigía": pick(risk, ex_ok),
        "techo (alguna pasa)": ok.max(1).astype(float),
    }
    for name, v in picks.items():
        print(f"{name:26s} pasan {v.mean():.1%}")
    # En MBPP el ejemplo visible es además uno de los tests ocultos: separar las fuentes.
    for src in ("HumanEval", "MBPP"):
        k = np.array([t.startswith(src) for t in tasks])
        if k.any():
            cols = ", ".join(f"{n} {v[k].mean():.1%}" for n, v in picks.items())
            print(f"{src} (n={k.sum()}): {cols}")
    boots = rng.integers(0, len(tasks), (2000, len(tasks)))
    for a, b in (
        ("menor entropía en código", "azar (esperado)"),
        ("menor entropía en código", "la más corta"),
        ("Vigía (puntaje entrenado)", "la más corta"),
        ("ejemplos + azar", "azar (esperado)"),
        ("ejemplos + menor entropía", "ejemplos + la más corta"),
    ):
        diff = picks[a] - picks[b]
        lo, hi = np.percentile(diff[boots].mean(1), [2.5, 97.5])
        print(f"{a} - {b}: {diff.mean():+.1%} [IC 95 %: {lo:+.1%}; {hi:+.1%}]")


if __name__ == "__main__":
    if sys.argv[1] == "subset":
        subset(*sys.argv[2:5])
    else:
        analyze(sys.argv[2], sys.argv[3:])
