"""Diferencias pareadas router contra entropía con los datos que quedan en esta máquina.

Uso: python colab/paired_deltas.py
- Granite (600 preguntas) y OLMoE (protocolo de InnerExpert): lee el delta pareado que el reporte
  canónico ya guarda (`delta_auroc_vs_detector`, bootstrap por pregunta) y lo invierte.
- vLLM (600 preguntas, gpt-oss): reentrena dentro de MKQA (sin aritmética) con un split nuevo del
  estrato (semilla 0, 60/20/20), como el control de §4.10, y da el delta pareado F0+U contra F0 y
  contra la NLL. Comprueba que reproduce el 0,784 de §4.10 (con el split global guardado da otro
  número, 0,895: el estrato es chico y el resultado depende del split).
- Latencia vLLM: mediana pareada por (corrida, prompt) del cociente con/sin la opción, con IC por
  bootstrap sobre prompts. Las dos mediciones vienen de dos sesiones de servidor, sin intercalar.
Las corridas factuales de gpt-oss (T4 y 2.600 preguntas) no tienen features locales.
"""

import json
from pathlib import Path

import numpy as np

from vigia.evaluate import evaluate_report
from vigia.evidence import FeatureSet

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / ".cache/runs"


def flip(d: dict) -> str:
    """delta(baseline - detector) -> detector - baseline, con su IC."""
    lo, hi = d["ci"]
    return f"{-d['value']:+.3f} [{-hi:+.3f}; {-lo:+.3f}]"


def saved_delta(name: str, rep: dict, level: str = "token") -> None:
    ent = rep["baselines"]["out_entropy"][level]
    det = rep["detector"][level]["auroc"]["value"]
    n = rep["bootstrap"]["n_boot"]
    print(
        f"{name:28s} {level:8s} F0+F1 {det:.3f} entropía {ent['auroc']['value']:.3f} "
        f"F0+F1-entropía {flip(ent['delta_auroc_vs_detector'])} (n_boot {n})"
    )


def vllm_stratum() -> None:
    run = RUNS / "lat_vllm/vllm_v1"
    fs = FeatureSet.load(run / "features.npz")
    rows = [json.loads(line) for line in open(run / "rows.jsonl", encoding="utf-8")]
    arit = {r["group"] for r in rows if str(r.get("type", "")).startswith("arit")}
    m = np.array([g not in arit for g in fs.groups])
    feats = {k: v[m] for k, v in fs.features().items()}
    rep = evaluate_report(
        feats, fs.labels[m], fs.groups[m], fs.response_ids[m], fs.model_info, n_boot=1000
    )
    det = rep["detector"]["token"]["auroc"]
    f0 = rep["tiers"]["F0"]["token"]
    nll = rep["baselines"]["out_nll"]["token"]
    print(
        f"vLLM MKQA (sin aritmética)    token    F0+U {det['value']:.3f} {np.round(det['ci'], 3).tolist()} "
        f"F0 {f0['auroc']['value']:.3f} F0+U-F0 {flip(f0['delta_auroc_vs_detector'])} "
        f"F0+U-NLL {flip(nll['delta_auroc_vs_detector'])} "
        f"(test {rep['dataset']['n_groups_test']} preguntas, n_boot 1000)"
    )
    assert abs(det["value"] - 0.784) < 0.005, "no reproduce el 0,784 de §4.10"


def vllm_latency(n_boot: int = 10000, n_prompts: int = 8) -> None:
    """`values` viene en orden (corrida, prompt), con los mismos 8 prompts en las dos sesiones de
    servidor (`colab/vllm_mode.py latency`). Se parea por (corrida, prompt), como el estimador de
    §4.8, y el bootstrap remuestrea prompts. Las sesiones no se intercalan: la deriva entre ellas
    queda dentro del delta."""
    a = np.array(json.loads((RUNS / "lat_vllm/lat_flag.json").read_text())["values"])
    b = np.array(json.loads((RUNS / "lat_vllm/lat_noflag.json").read_text())["values"])
    ratio = (a / b - 1).reshape(-1, n_prompts)  # (corrida, prompt)
    rng = np.random.default_rng(0)
    d = np.array([np.median(ratio[:, rng.integers(0, n_prompts, n_prompts)]) for _ in range(n_boot)])
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(
        f"vLLM latencia p50 con/sin expertos: {np.median(a):.2f} / {np.median(b):.2f} ms; "
        f"mediana pareada de cocientes {np.median(ratio):+.2%} [{lo:+.2%}; {hi:+.2%}] "
        f"({ratio.size} unidades, bootstrap sobre {n_prompts} prompts; sesiones de servidor separadas)"
    )
    print("  cociente medio por prompt:", np.round(ratio.mean(0) * 100, 2).tolist())


def main() -> None:
    granite = json.loads((ROOT / "docs/evidencia/evidence_v1-report.json").read_text())
    saved_delta("Granite 600 (test)", granite)
    saved_delta("Granite 600 (test)", granite, "response")
    ie = json.loads((RUNS / "innerexpert/runs/comparacion.json").read_text())
    for name, rep in ie.items():
        saved_delta(f"OLMoE {name}", rep)
    vllm_stratum()
    vllm_latency()


if __name__ == "__main__":
    main()
