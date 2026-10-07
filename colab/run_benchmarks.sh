#!/usr/bin/env bash
# Corre en Colab (GPU) los benchmarks ampliados de gpt-oss-20b, uno detrás del otro.
# Uso: RUNS=/content/drive/MyDrive/vigia_runs bash colab/run_benchmarks.sh
# Cada corrida es reanudable: si se corta, volver a lanzar el script retoma donde quedó.
# Con RUNS en Drive, los resultados sobreviven a un reinicio de la VM.
set -u
cd /content/vigia
RUNS=${RUNS:-/content/runs}
mkdir -p "$RUNS"
MODEL=openai/gpt-oss-20b
COMMON="--model $MODEL --device cuda --dtype auto --reasoning-effort low --n-boot 2000"

echo "== código (HumanEval 164 + MBPP 970) $(date -u)"
python -m vigia.cli evidence $COMMON --humaneval 1134 --humaneval-path data/code_es.jsonl \
  --allow-code-exec --out "$RUNS/gptoss20b_code_all" --max-tokens 768 --tiers F0,F1 \
  > "$RUNS/code.log" 2>&1
echo "código terminó con código $? $(date -u)"
python colab/code_response.py "$RUNS/gptoss20b_code_all" > "$RUNS/code_response.txt" 2>&1

echo "== factual ampliado (2600 preguntas) $(date -u)"
python -m vigia.cli evidence $COMMON --data data/evidence_v2.jsonl --out "$RUNS/gptoss20b_v2" \
  --max-tokens 256 --tiers F0,F1,H --bench 4 --bench-runs 5 > "$RUNS/v2.log" 2>&1
echo "factual terminó con código $? $(date -u)"
python colab/pareto.py "$RUNS/gptoss20b_v2" > "$RUNS/pareto_v2.txt" 2>&1
python colab/pareto.py "$RUNS/gptoss20b_code_all" > "$RUNS/pareto_code.txt" 2>&1
