"""Huella por dominio: ¿cada dominio usa los expertos de un MoE de forma distinta?

Uso (VM con GPU, después de colab/innerexpert_prep.py /content/ie):
  python colab/fingerprint.py OUT_PREFIX [n_por_dominio]
Lee texto de siete dominios, pasa cada texto por OLMoE-1B-7B (solo forward, sin generar) y cuenta,
por texto y por capa, cuántas veces el router eligió cada experto (top-8 de 64). Guarda los conteos en
OUT_PREFIX.npz y el resumen en OUT_PREFIX.json.

La comparación justa usa mitades: en cada réplica, cada dominio se parte al azar en dos mitades de
textos. "Dentro" es la JSD entre las dos mitades de un dominio; "entre" es la JSD entre la primera
mitad de un dominio y la de otro. Las dos tienen el mismo tamaño de muestra, así que el ruido de
muestreo pesa igual. La premisa del MoE fractal predice entre >> dentro.
"""

import json
import random
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "allenai/OLMoE-1B-7B-0924-Instruct"


def _jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def _user(item: dict) -> str:
    return item["messages"][-1]["content"] if "messages" in item else item["question"]


def domains() -> dict[str, list[str]]:
    ev2 = _jsonl("data/evidence_v2.jsonl")
    ie = lambda name: [_user(it) for it in _jsonl(f"/content/ie/{name}.jsonl") if it["id"].endswith("-ev")]  # noqa: E731
    return {
        "código": [it["question"] for it in _jsonl("data/code_es.jsonl")],
        "aritmética": [it["question"] for it in ev2 if str(it.get("type", "")).startswith("arit")],
        "factual_es": [it["question"] for it in _jsonl("data/mkqa_es.jsonl")],
        "noticias": ie("rtqa_2026"),
        "wikipedia": ie("oos_squad"),
        "mitos": ie("oos_truthfulqa"),
        "actualidad": ie("oos_freshqa"),
    }


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    """JSD en bits entre dos distribuciones (vectores que suman 1): cae en [0, 1]."""
    m = (p + q) / 2

    def kl(a: np.ndarray) -> float:
        nz = a > 0
        return float(np.sum(a[nz] * np.log2(a[nz] / m[nz])))

    return (kl(p) + kl(q)) / 2


def usage(counts: np.ndarray) -> np.ndarray:
    """(textos, capas, expertos) -> distribución por capa (capas, expertos)."""
    tot = counts.sum(axis=0).astype(np.float64)
    return tot / tot.sum(axis=1, keepdims=True)


def layer_mean_jsd(a: np.ndarray, b: np.ndarray) -> float:
    pa, pb = usage(a), usage(b)
    return float(np.mean([jsd(pa[layer], pb[layer]) for layer in range(pa.shape[0])]))


def summarize(counts: dict[str, np.ndarray], reps: int = 20, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    names = list(counts)
    within = {n: [] for n in names}
    between = {(a, b): [] for i, a in enumerate(names) for b in names[i + 1 :]}
    for _ in range(reps):
        halves = {}
        for n in names:
            idx = rng.permutation(len(counts[n]))
            h = len(idx) // 2
            halves[n] = (counts[n][idx[:h]], counts[n][idx[h : 2 * h]])
            within[n].append(layer_mean_jsd(*halves[n]))
        for a, b in between:
            between[(a, b)].append(layer_mean_jsd(halves[a][0], halves[b][0]))
    w = {n: float(np.mean(v)) for n, v in within.items()}
    bt = {f"{a} | {b}": float(np.mean(v)) for (a, b), v in between.items()}
    return {
        "dentro": w,
        "entre": bt,
        "dentro_medio": float(np.mean(list(w.values()))),
        "entre_medio": float(np.mean(list(bt.values()))),
        "entre_minimo": min(bt.items(), key=lambda kv: kv[1]),
    }


def main(out: str, n: int = 200) -> None:
    random.seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    layers = model.model.layers
    n_exp = model.config.num_experts
    current: list[np.ndarray] = []

    def hook(layer_idx: int):
        def fn(module, args, output):
            top_k_index = output[2]  # OlmoeTopKRouter devuelve (logits, pesos top-k, índices top-k)
            current[layer_idx] += np.bincount(top_k_index.reshape(-1).cpu().numpy(), minlength=n_exp)

        return fn

    for i, layer in enumerate(layers):
        layer.mlp.gate.register_forward_hook(hook(i))
    counts = {}
    for name, texts in domains().items():
        sample = random.sample(texts, min(n, len(texts)))
        rows = []
        for text in sample:
            current[:] = [np.zeros(n_exp, dtype=np.int64) for _ in layers]
            ids = tok(text, return_tensors="pt", truncation=True, max_length=512).to("cuda")
            with torch.no_grad():
                model(**ids)
            rows.append(np.stack(current))
        counts[name] = np.stack(rows)
        print(f"{name}: {len(sample)} textos, {int(counts[name][:, 0].sum()) // 8} tokens", flush=True)
    np.savez_compressed(f"{out}.npz", **counts)
    res = summarize(counts)
    json.dump(res, open(f"{out}.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], *(int(a) for a in sys.argv[2:3]))
