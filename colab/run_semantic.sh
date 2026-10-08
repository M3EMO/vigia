#!/usr/bin/env bash
# Latencia sin H y entropía semántica (gpt-oss-20b, L4). Requiere los artefactos de la corrida
# factual en /content/se_in/: rows.jsonl y split.json de gptoss20b_v2.
# Uso (en la VM): bash colab/run_semantic.sh
set -u
cd /content/vigia
RUNS=/content/runs
mkdir -p "$RUNS"
MODEL="--model openai/gpt-oss-20b --device cuda --dtype auto --reasoning-effort low"

echo "== latencia $(date -u)"
python -m vigia.cli bench $MODEL --data data/evidence_v2.jsonl --n-prompts 8 --tiers F0 F0,F1 F0,F1,H \
  --max-tokens 256 --warmup 1 --runs 5 --out "$RUNS/bench_l4.json" > "$RUNS/bench.log" 2>&1
echo "latencia terminó con código $? $(date -u)"

python colab/semantic_entropy.py subset /content/se_in/rows.jsonl /content/se_in/split.json \
  data/evidence_v2.jsonl "$RUNS/se_tasks.jsonl"
for k in 1 2 3 4 5; do
  echo "== muestra $k $(date -u)"
  python -m vigia.cli extract $MODEL --data "$RUNS/se_tasks.jsonl" --tiers F0 --max-tokens 256 \
    --temperature 0.8 --seed $k --out "$RUNS/se_s$k" > "$RUNS/se_s$k.log" 2>&1
done
echo "muestras terminadas $(date -u)"
