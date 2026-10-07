#!/usr/bin/env bash
# Mejor de N en código (gpt-oss-20b): 5 muestras por tarea del split de test de la corrida greedy.
# Requiere que run_benchmarks.sh haya terminado la parte de código. Reanudable como las demás corridas.
# Uso: RUNS=/content/drive/MyDrive/vigia_runs bash colab/run_best_of_n.sh
set -u
cd /content/vigia
RUNS=${RUNS:-/content/runs}
RUN="$RUNS/gptoss20b_code_all"
python colab/best_of_n.py subset "$RUN" data/code_es.jsonl "$RUNS/bon_tasks.jsonl"
for k in 1 2 3 4 5; do
  echo "== muestra $k $(date -u)"
  python -m vigia.cli extract --model openai/gpt-oss-20b --device cuda --dtype auto --reasoning-effort low \
    --data "$RUNS/bon_tasks.jsonl" --tiers F0,F1 --max-tokens 768 --temperature 0.8 --seed $k \
    --allow-code-exec --out "$RUNS/bon_s$k" > "$RUNS/bon_s$k.log" 2>&1
done
python colab/best_of_n.py analyze "$RUN" "$RUNS"/bon_s{1,2,3,4,5} > "$RUNS/best_of_n.txt" 2>&1
cat "$RUNS/best_of_n.txt"
