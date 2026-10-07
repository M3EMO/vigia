"""Frontera de Pareto de alpha para una corrida de `vigia evidence`.

Uso: python colab/pareto.py /content/runs/gptoss20b_v2
Elige el umbral con CRC sobre el split de calibración y lo evalúa sobre el split de test, a nivel
respuesta: una respuesta queda marcada si algún token supera tau. Imprime una línea por alpha y la
frontera no dominada.
"""

import json
import sys

import numpy as np

from vigia.calibrate import crc_threshold
from vigia.detector import Detector


def main(run: str) -> None:
    d = np.load(f"{run}/features.npz", allow_pickle=False)
    x, names, y = d["X"], [str(n) for n in d["names"]], d["labels"]
    g, r = d["groups"].astype(str), d["response_ids"].astype(str)
    det = Detector.load(f"{run}/detector.json")
    split = json.load(open(f"{run}/split.json", encoding="utf-8"))
    groups = split.get("groups", split)
    idx = [names.index(f) for f in det.feature_names]

    def part(name: str):
        keep = set(groups[name])
        m = np.array([v in keep for v in g]) & (y != -100)
        return det.score_matrix(x[m][:, idx]), y[m], r[m]

    pc, yc, rc = part("cal")
    pt, yt, rt = part("test")

    def response_rates(p, yy, rr, tau):
        resp: dict[str, list] = {}
        for rid, s, lab in zip(rr, p, yy, strict=True):
            a = resp.setdefault(rid, [False, False])
            a[0] |= bool(s >= tau)
            a[1] |= bool(lab == 1)
        bad = [v for v in resp.values() if v[1]]
        good = [v for v in resp.values() if not v[1]]
        rec = sum(v[0] for v in bad) / len(bad) if bad else float("nan")
        fa = sum(v[0] for v in good) / len(good) if good else float("nan")
        return rec, fa, len(bad), len(good)

    rows = []
    for a in [round(0.02 * k, 2) for k in range(1, 26)]:
        try:
            tau = crc_threshold(pc, yc, rc, a).tau
        except Exception as exc:  # noqa: BLE001 - se informa y sigue
            print(f"alpha {a:.2f}: error {exc}")
            continue
        rec_c, fa_c, _, _ = response_rates(pc, yc, rc, tau)
        rec_t, fa_t, nb, ng = response_rates(pt, yt, rt, tau)
        rows.append((a, tau, rec_c, fa_c, rec_t, fa_t))
        print(
            f"alpha {a:.2f} tau {tau:.3f} | cal: detecta {rec_c:.0%} falsas {fa_c:.0%} "
            f"| test: detecta {rec_t:.0%} falsas {fa_t:.0%} (malas {nb}, buenas {ng})"
        )
    best = max(rows, key=lambda t: (t[2] - t[3], -t[0]))
    print(
        f"RECOMENDADO (elegido en cal): alpha {best[0]:.2f} -> test detecta {best[4]:.0%}, "
        f"falsas {best[5]:.0%}"
    )
    front = [
        t
        for t in rows
        if not any(o[4] >= t[4] and o[5] <= t[5] and (o[4] > t[4] or o[5] < t[5]) for o in rows)
    ]
    print("PARETO (test):", [(t[0], f"{t[4]:.0%}", f"{t[5]:.0%}") for t in front])


if __name__ == "__main__":
    main(sys.argv[1])
