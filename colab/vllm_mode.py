"""Modo reducido real: Vigía como proxy de vLLM con `--enable-return-routed-experts` (gpt-oss-20b).

Uso (en la VM, con `vllm serve` ya escuchando en :8000):
  python colab/vllm_mode.py smoke            una pregunta: forma de la respuesta y de routed_experts
  python colab/vllm_mode.py latency OUT.json ms por token con streaming y longitud fija
  python colab/vllm_mode.py evidence DATA OUT run_evidence completo con el backend proxy (F0 parcial + U)
  python colab/vllm_mode.py concurrency OUT.json [plain,proxy] [1,4,16,32]
                                             N requests simultáneos (hasta 256 tokens): vLLM directo
                                             (plain) contra el proxy de Vigía (logprobs + expertos)

gpt-oss usa `reasoning_effort=low`, igual que las corridas con el backend HF.
"""

import json
import math
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

from vigia.backends.vllm_proxy import VllmProxyBackend, parse_routed_experts
from vigia.datasets import read_jsonl
from vigia.harmony import has_markup

URL = "http://127.0.0.1:8000"
MODEL = "openai/gpt-oss-20b"


class LowEffortProxy(VllmProxyBackend):
    # ponytail: la opción vive solo en el experimento; si el modo vLLM queda, pasa a VllmProxyBackend.
    def _request(self, body):
        return super()._request({**body, "reasoning_effort": "low"})


def backend() -> LowEffortProxy:
    return LowEffortProxy(URL, MODEL, n_experts=32, top_k=4, n_layers=24, model_type="gpt_oss", timeout=300)


def smoke() -> None:
    q = read_jsonl("data/evidence_v1.jsonl")[0]
    raw = httpx.post(
        f"{URL}/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": q["messages"],
            "max_tokens": 256,
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": 20,
            "return_routed_experts": True,
            "reasoning_effort": "low",
        },
        timeout=300,
    ).json()
    ch = raw["choices"][0]
    msg = ch.get("message") or {}
    print("claves choice:", sorted(ch), "| claves message:", sorted(msg))
    print("content:", repr((msg.get("content") or "")[:120]))
    print("reasoning:", repr((msg.get("reasoning_content") or msg.get("reasoning") or "")[:120]))
    print("tokens con logprobs:", len((ch.get("logprobs") or {}).get("content") or []))
    re_ = ch.get("routed_experts") or raw.get("routed_experts")
    shape = parse_routed_experts(re_).shape if re_ is not None else None
    print("routed_experts:", type(re_).__name__, shape, "| token_ids:", len(ch.get("token_ids") or []))
    s = backend().stream(q["messages"], max_tokens=256, tiers=backend().capabilities().tiers)
    events = list(s)
    print("proxy: eventos", len(events), "| texto", repr(s.text[:120]), "| fin", s.finish_reason)
    print("features del primer evento:", sorted(events[0].features)[:8] if events else None)
    # El texto puntuado tiene que ser el canal final, igual a message.content y sin marcado harmony.
    assert not has_markup(s.text) and s.text == (msg.get("content") or ""), "el proxy no aisló el canal final"


def latency(out: str, n_prompts: int = 8, tokens: int = 256, runs: int = 3) -> None:
    qs = read_jsonl("data/evidence_v2.jsonl")[:n_prompts]
    per_tok = []
    for r in range(runs + 1):
        for q in qs:
            t_first = t_last = None
            n = 0
            body = {
                "model": MODEL,
                "messages": q["messages"],
                "max_tokens": tokens,
                "temperature": 0,
                "ignore_eos": True,
                "stream": True,
                "reasoning_effort": "low",
            }
            with httpx.stream("POST", f"{URL}/v1/chat/completions", json=body, timeout=300) as resp:
                for line in resp.iter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        now = time.perf_counter()
                        t_first = t_first or now
                        t_last = now
                        n += 1
            if r and n > 1:
                per_tok.append(1000 * (t_last - t_first) / (n - 1))
    res = {"ms_per_token_p50": statistics.median(per_tok), "n": len(per_tok), "values": per_tok}
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "values"}))


def _plain(q: dict, tokens: int) -> tuple[float, int]:
    body = {"model": MODEL, "messages": q["messages"], "max_tokens": tokens, "temperature": 0,
            "reasoning_effort": "low"}  # fmt: skip
    t0 = time.perf_counter()
    r = httpx.post(f"{URL}/v1/chat/completions", json=body, timeout=900)
    r.raise_for_status()
    return time.perf_counter() - t0, r.json()["usage"]["completion_tokens"]


def concurrency(
    out: str, variants: str = "plain,proxy", levels: str = "1,4,16,32", tokens: int = 256
) -> None:
    # Sin ignore_eos: vLLM 0.31 rompe el parser harmony de gpt-oss si se genera después de <|return|>
    # en una respuesta no streaming. Las tareas de código dan respuestas largas y de largo parecido.
    qs = read_jsonl("data/code_es.jsonl")
    proxy = backend()

    def via_proxy(q: dict) -> tuple[float, int]:
        t0 = time.perf_counter()
        n = len(list(proxy.stream(q["messages"], max_tokens=tokens, tiers=proxy.capabilities().tiers)))
        return time.perf_counter() - t0, n

    run = {"plain": lambda q: _plain(q, tokens), "proxy": via_proxy}
    res = {}
    for v in variants.split(","):
        for c in (int(x) for x in levels.split(",")):
            lat, per_tok, toks, wall = [], [], 0, 0.0
            for r in range(3):  # la ronda 0 calienta este nivel
                batch = [qs[(r * c + i) % len(qs)] for i in range(c)]
                t0 = time.perf_counter()
                with ThreadPoolExecutor(c) as ex:
                    outs = list(ex.map(run[v], batch))
                if r:
                    wall += time.perf_counter() - t0
                    lat += [t for t, _ in outs]
                    per_tok += [1000 * t / max(n, 1) for t, n in outs]
                    toks += sum(n for _, n in outs)
            lat.sort()
            p95 = lat[math.ceil(0.95 * len(lat)) - 1]  # rango más cercano: con n chico es el máximo
            res[f"{v}@{c}"] = {
                "req_s_p50": statistics.median(lat),
                "req_s_p95": p95,
                "ms_per_tok_p50": statistics.median(per_tok),
                "tok_per_s": toks / wall,
            }
            print(f"{v}@{c}", json.dumps(res[f"{v}@{c}"]), flush=True)
    json.dump(res, open(out, "w"), indent=1)


def evidence(data: str, out: str) -> None:
    from vigia.evidence import run_evidence, summary_text

    b = backend()
    report = run_evidence(
        b, read_jsonl(data), out_dir=out, max_tokens=256, tiers=b.capabilities().tiers, seed=0, n_boot=1000
    )
    print(summary_text(report))


if __name__ == "__main__":
    {"smoke": smoke, "latency": latency, "evidence": evidence, "concurrency": concurrency}[sys.argv[1]](
        *sys.argv[2:]
    )
