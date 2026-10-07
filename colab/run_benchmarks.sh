#!/usr/bin/env bash
# Corre en Colab (GPU) los benchmarks de gpt-oss-20b, uno detrás del otro: código (HumanEval + MBPP),
# mejor de N sobre las tareas de test de código y las 2.600 preguntas factuales.
# Uso: RUNS=/content/drive/MyDrive/vigia_runs bash colab/run_benchmarks.sh
# Cada corrida es reanudable: si se corta, volver a lanzar el script retoma donde quedó.
# Se trabaja en el disco local de la VM y se copia a RUNS (Drive) cada 5 minutos: escribir directo
# en Drive cortó la corrida del 2026-10-07 (Drive perdió un .tmp.npz en pleno os.replace).
set -u
cd /content/vigia
DEST=${RUNS:-}
export RUNS=/content/runs
mkdir -p "$RUNS"

sync_to_drive() {
  # Primero una foto de cada rows.jsonl y después los chunks: así cada fila copiada tiene su chunk
  # (append escribe el chunk antes que la fila). load_rows repara una última línea cortada.
  for d in "$RUNS"/*/; do
    [ -f "$d/rows.jsonl" ] && cp "$d/rows.jsonl" "$d/.rows.snap"
  done
  cp -ru "$RUNS"/. "$DEST"/
  for d in "$RUNS"/*/; do
    n=$(basename "$d")
    [ -f "$d/.rows.snap" ] && cp "$d/.rows.snap" "$DEST/$n/rows.jsonl"
  done
}

if [ -n "$DEST" ]; then
  mkdir -p "$DEST"
  cp -rn "$DEST"/. "$RUNS"/ 2>/dev/null  # retoma lo que ya estaba en Drive
  (while true; do sleep 300; sync_to_drive 2>/dev/null; done) &
  SYNC_PID=$!
  trap 'kill $SYNC_PID 2>/dev/null; sync_to_drive' EXIT
fi

MODEL=openai/gpt-oss-20b
COMMON="--model $MODEL --device cuda --dtype auto --reasoning-effort low --n-boot 2000"

echo "== código (HumanEval 164 + MBPP 970) $(date -u)"
python -m vigia.cli evidence $COMMON --humaneval 1134 --humaneval-path data/code_es.jsonl \
  --allow-code-exec --out "$RUNS/gptoss20b_code_all" --max-tokens 768 --tiers F0,F1 \
  > "$RUNS/code.log" 2>&1
echo "código terminó con código $? $(date -u)"
python colab/code_response.py "$RUNS/gptoss20b_code_all" > "$RUNS/code_response.txt" 2>&1
python colab/pareto.py "$RUNS/gptoss20b_code_all" > "$RUNS/pareto_code.txt" 2>&1
[ -n "$DEST" ] && sync_to_drive

echo "== mejor de N $(date -u)"
bash colab/run_best_of_n.sh
[ -n "$DEST" ] && sync_to_drive

echo "== factual ampliado (2600 preguntas) $(date -u)"
python -m vigia.cli evidence $COMMON --data data/evidence_v2.jsonl --out "$RUNS/gptoss20b_v2" \
  --max-tokens 256 --tiers F0,F1,H --bench 4 --bench-runs 5 > "$RUNS/v2.log" 2>&1
echo "factual terminó con código $? $(date -u)"
python colab/pareto.py "$RUNS/gptoss20b_v2" > "$RUNS/pareto_v2.txt" 2>&1
