#!/usr/bin/env bash
# Corre en Colab (GPU) los benchmarks ampliados de gpt-oss-20b, uno detrás del otro.
# Uso: nohup bash colab/run_benchmarks.sh > /content/runs/all.log 2>&1 &
# Cada corrida es reanudable: si se corta, volver a lanzar el script retoma donde quedó.
set -u
cd /content/vigia
mkdir -p /content/runs
MODEL=openai/gpt-oss-20b
COMMON="--model $MODEL --device cuda --dtype auto --reasoning-effort low --n-boot 2000"

echo "== código (HumanEval 164) $(date -u)"
python -m vigia.cli evidence $COMMON --humaneval 164 --humaneval-path data/humaneval_es.jsonl \
  --allow-code-exec --out /content/runs/gptoss20b_code --max-tokens 768 --tiers F0,F1 \
  > /content/runs/code.log 2>&1
echo "código terminó con código $? $(date -u)"

echo "== factual ampliado (2600 preguntas) $(date -u)"
python -m vigia.cli evidence $COMMON --data data/evidence_v2.jsonl --out /content/runs/gptoss20b_v2 \
  --max-tokens 256 --tiers F0,F1,H --bench 4 --bench-runs 5 > /content/runs/v2.log 2>&1
echo "factual terminó con código $? $(date -u)"
