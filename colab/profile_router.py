"""Perfil de la latencia del router en GPU: ¿cuesta el hook o la reducción en CPU?

Uso (en una VM con GPU): python colab/profile_router.py [n_prompts] [tokens] [corridas]
Variantes, intercaladas por prompt y corrida, con longitud fija (ignore_eos):
  base       sin hooks
  F0+F1      el extractor actual
  sin_red    F0+F1 con la reducción del router reemplazada por ceros (aísla el costo de CPU)
  logits_mod F0+F1 leyendo los logits que el router del modelo ya calculó (sin recalcular)
Imprime la mediana de ms por token de decodificación y el overhead contra base.
"""

import os
import statistics
import sys

import numpy as np
import torch

from vigia.cli import load_backend
from vigia.contracts import Tier
from vigia.datasets import read_jsonl
from vigia.extractor import SignalExtractor

ORIG_ENTER = SignalExtractor.__enter__
ORIG_STEP = SignalExtractor._router_step


def _no_reduction(self, step, valid, scored, seq):
    m = int(np.count_nonzero(scored))
    return np.zeros((m, len(self._router_names))), [[[0] * self.top_k] * len(self.blocks)] * m


def _enter_router_output(self):
    # Hook de salida sobre el router del modelo: toma los logits que ya calculó (bf16) en lugar de
    # recalcularlos en float32 desde el residual.
    for j, b in enumerate(self.blocks):

        def hook(module, args, output, j=j):
            logits = output[0] if isinstance(output, tuple) else output
            step = self._current()
            step.shape = (1, logits.shape[0])
            step.router[j] = logits.detach().float()
            step.fired.add(("router", j))

        self._handles.append(b.router.register_forward_hook(hook))
    return self


VARIANTS = {
    "base": ((), None, None),
    "F0+F1": ((Tier.F0, Tier.F1), None, None),
    "sin_red": ((Tier.F0, Tier.F1), None, _no_reduction),
    "logits_mod": ((Tier.F0, Tier.F1), _enter_router_output, None),
}


def main(n_prompts: int, tokens: int, runs: int) -> None:
    model = os.environ.get("VIGIA_MODEL", "openai/gpt-oss-20b")  # "tiny:gpt_oss" para probar en CPU
    device = os.environ.get("VIGIA_DEVICE", "cuda")
    backend = load_backend(model, device=device, dtype="auto", reasoning_effort="low")
    items = read_jsonl(os.environ.get("VIGIA_DATA", "data/evidence_v2.jsonl"))[:n_prompts]
    times: dict[str, list[float]] = {k: [] for k in VARIANTS}
    order = list(VARIANTS)
    for r in range(runs + 1):  # la corrida 0 es calentamiento
        for p, it in enumerate(items):
            rot = order[(r + p) % len(order) :] + order[: (r + p) % len(order)]
            for name in rot:
                tiers, enter, step = VARIANTS[name]
                SignalExtractor.__enter__ = enter or ORIG_ENTER
                SignalExtractor._router_step = step or ORIG_STEP
                try:
                    s = backend.stream(it["messages"], max_tokens=tokens, tiers=tiers, ignore_eos=True)
                    for _ in s:
                        pass
                    if device == "cuda":
                        torch.cuda.synchronize()
                finally:
                    SignalExtractor.__enter__ = ORIG_ENTER
                    SignalExtractor._router_step = ORIG_STEP
                t = s.timings
                if r:
                    times[name].append(1000 * t["decode_s"] / max(1, s.completion_tokens - 1))
    base = statistics.median(times["base"])
    for name, v in times.items():
        med = statistics.median(v)
        print(f"{name:11s} {med:7.2f} ms/token  overhead {100 * (med / base - 1):+.2f} %  (n={len(v)})")


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:]] + [6, 128, 3][len(sys.argv) - 1 :]
    main(*args[:3])
