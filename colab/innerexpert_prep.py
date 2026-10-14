"""Datasets de InnerExpert (arXiv 2608.17687) en el formato de Vigía, con su protocolo.

Uso (en la VM, con red): python colab/innerexpert_prep.py OUT_DIR
Escribe un JSONL por conjunto: rtqa_2024, rtqa_2025 (train), rtqa_2026 (enero a junio, test) y
oos_{squad,truthfulqa,nq_open,freshqa} (200 preguntas cada uno, `df.sample(n=200, random_state=42)`).
Cada pregunta da dos items, como en el paper: sin evidencia (`-base`) y con evidencia (`-ev`).
Prompts copiados de `moeuncert` (incluido el "andvery" sin espacio del system prompt de RealtimeQA).
Diferencia: la evidencia se corta en 10.000 caracteres para no pasar el contexto de 4.096 tokens de
OLMoE; el paper trunca el prompt entero por la derecha y puede perder la pregunta.
"""

import json
import os
import re
import sys
from html import unescape

import pandas as pd
import requests

RTQA_SYS = (
    "You are a helpful assistant who provides accurate and"
    "very concise answers to questions about recent"
    "events. Today is "
)
OOS_SYS = "You are a helpful assistant who provides accurate and very concise answers to questions. Today is "
MAX_EV = 10_000


def _user(q: str, ev: str | None) -> str:
    return f"Evidence: {ev[:MAX_EV]}\n\nQuestion: {q}" if ev else q


def _items(qid: str, q: str, answer: str, ev: str, sys_prompt: str, kind: str) -> list[dict]:
    # Sin evidencia nativa, el paper usa la respuesta de referencia como evidencia (filtra la respuesta).
    ev_prompt = ev or answer
    judge_ev = f"{ev[:MAX_EV]}\n\nCorrect answer: {answer}" if ev else None  # None: Vigía usa `answers`
    out = []
    for suffix, user in (("base", _user(q, None)), ("ev", _user(q, ev_prompt))):
        out.append(
            {
                "id": f"{qid}-{suffix}",
                "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}],
                "answers": [a.strip() for a in answer.split(" | ") if a.strip()],
                "evidence": judge_ev,
                "type": f"{kind}-{suffix}",
            }
        )
    return out


def _strip_html(text: str) -> str:
    return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", text or ""))).strip()


def rtqa(year: int, months: range | None = None) -> list[dict]:
    api = f"https://api.github.com/repos/realtimeqa/realtimeqa_public/contents/past/{year}"
    files = [
        f
        for f in requests.get(api, timeout=60).json()
        if f["name"].replace("_public", "").endswith("_qa.jsonl")
    ]
    if months is not None:
        files = [f for f in files if any(f["name"].startswith(f"{year}{m:02d}") for m in months)]
    items = []
    for f in sorted(files, key=lambda f: f["name"]):
        for line in requests.get(f["download_url"], timeout=60).text.splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("answer"):
                answer = r["choices"][int(r["answer"][0])]
            else:
                continue  # el paper adivina la respuesta por la evidencia; sin oro no hay etiqueta fiable
            date = pd.to_datetime(r["question_date"]).strftime("%B %d, %Y")
            ev = _strip_html(r.get("evidence", ""))
            sys_prompt = RTQA_SYS + date + "."
            items += _items(
                f"rtqa-{r['question_id']}", r["question_sentence"], answer, ev, sys_prompt, f"rtqa_{year}"
            )
    return items


def oos(name: str) -> list[dict]:
    if name == "squad":
        df = pd.read_parquet("hf://datasets/rajpurkar/squad/plain_text/validation-00000-of-00001.parquet")
        df = pd.DataFrame(
            {
                "q": df["question"],
                "a": df["answers"].map(lambda a: " | ".join(a["text"])),
                "ev": df["context"],
            }
        )
    elif name == "truthfulqa":
        df = pd.read_csv("hf://datasets/domenicrosati/TruthfulQA/train.csv")
        df = pd.DataFrame({"q": df["Question"], "a": df["Best Answer"], "ev": ""})
    elif name == "nq_open":
        df = pd.read_parquet(
            "hf://datasets/google-research-datasets/nq_open/nq_open/validation-00000-of-00001.parquet"
        )
        df = pd.DataFrame({"q": df["question"], "a": df["answer"].map(" | ".join), "ev": ""})
    else:  # freshqa, versión del 11 de noviembre de 2025
        sheet = "1X6oTXzU1L9PWc2uim1eVzdX8V7y7_4crWfdJhVV08L4"
        df = pd.read_csv(f"https://docs.google.com/spreadsheets/d/{sheet}/export?format=csv", skiprows=2)
        col = next((c for c in ("more_info", "evidence_sources", "evidence") if c in df.columns), None)
        df = pd.DataFrame({"q": df["question"], "a": df["answer_0"], "ev": df[col] if col else ""})
    df = df.sample(n=200, random_state=42).reset_index(drop=True).fillna("")
    sys_prompt = OOS_SYS + "January 1, 2024."
    return [
        it
        for i, r in df.iterrows()
        for it in _items(f"{name}-{i}", str(r.q), str(r.a), str(r.ev), sys_prompt, name)
    ]


def main(out: str) -> None:
    os.makedirs(out, exist_ok=True)
    sets = {"rtqa_2026": rtqa(2026, range(1, 7)), "rtqa_2025": rtqa(2025), "rtqa_2024": rtqa(2024)}
    sets |= {f"oos_{n}": oos(n) for n in ("squad", "truthfulqa", "nq_open", "freshqa")}
    for name, items in sets.items():
        with open(f"{out}/{name}.jsonl", "w", encoding="utf-8") as fh:
            fh.writelines(json.dumps(it, ensure_ascii=False) + "\n" for it in items)
        print(f"{name}: {len(items) // 2} preguntas, {len(items)} items", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
