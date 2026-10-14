"""Comparación con InnerExpert: entrena en RealtimeQA 2024-2025 y evalúa fuera de distribución.

Uso (local, después de `vigia label` con el juez): python colab/innerexpert_eval.py RUNS_DIR
Cada conjunto de test (rtqa_2026 y los 4 OOS) se evalúa con el reporte canónico de Vigía: el split
de train y cal sale de RealtimeQA 2024-2025 (cal = 10 % de las preguntas, como el paper) y el de test
es el conjunto entero. El grupo es la pregunta: se deriva del id, sin el sufijo -base o -ev.
"""

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

from vigia.evidence import FeatureSet, build_features, summary_text

TRAIN = ("rtqa_2024", "rtqa_2025")
TESTS = ("rtqa_2026", "oos_squad", "oos_truthfulqa", "oos_nq_open", "oos_freshqa")


def _load(root: Path, name: str) -> FeatureSet:
    fs = build_features(root / name)
    question = np.array([r.rsplit("-", 1)[0] for r in fs.response_ids])
    return dataclasses.replace(fs, groups=np.char.add(f"{name}:", question))


def _concat(parts: list[FeatureSet]) -> FeatureSet:
    names = sorted({n for p in parts for n in p.names})
    blocks = []
    for p in parts:
        col = {n: j for j, n in enumerate(p.names)}
        X = np.full((len(p.labels), len(names)), np.nan, dtype=np.float32)
        for j, n in enumerate(names):
            if n in col:
                X[:, j] = p.X[:, col[n]]
        blocks.append(X)
    cat = lambda attr: np.concatenate([getattr(p, attr) for p in parts])  # noqa: E731
    return dataclasses.replace(
        parts[0], names=names, X=np.concatenate(blocks), labels=cat("labels"), groups=cat("groups"),
        response_ids=cat("response_ids"), token_index=cat("token_index"),
    )  # fmt: skip


def main(root: str) -> None:
    from vigia.evaluate import evaluate_report

    base = Path(root)
    train = _concat([_load(base, n) for n in TRAIN])
    qs = np.unique(train.groups)
    rng = np.random.default_rng(0)
    cal = set(rng.choice(qs, size=max(1, len(qs) // 10), replace=False))
    out = {}
    for name in TESTS:
        test = _load(base, name)
        fs = _concat([train, test])
        split = {
            "train": [q for q in qs if q not in cal],
            "cal": sorted(cal),
            "test": np.unique(test.groups).tolist(),
        }
        rep = evaluate_report(
            fs.features(), fs.labels, fs.groups, fs.response_ids, fs.model_info, split=split, n_boot=1000
        )
        out[name] = rep
        print(f"== {name}\n" + summary_text({"model": rep.get("model", {}), **rep}), flush=True)
    (base / "comparacion.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1])
