#!/usr/bin/env bash
# Corre los 7 conjuntos de InnerExpert en VMs L4 separadas (una muerte de VM pierde solo un conjunto).
# Uso: bash colab/cli/innerexpert_chain.sh [conjuntos...]   (desde la raíz del repo)
SETS=${*:-"rtqa_2026 oos_squad oos_truthfulqa oos_nq_open oos_freshqa rtqa_2025 rtqa_2024"}
for DS in $SETS; do
  JOB=$(mktemp --suffix=.py); sed "s/__DS__/$DS/" colab/cli/innerexpert_vm.py > "$JOB"
  bash colab/cli/job_runner.sh ie "$JOB" "IE_FIN|FALLA" /content/ie_stage.log ".cache/runs/innerexpert/$DS" "/content/ie_$DS.tgz"
  echo "$(date -u +%T) $DS runner=$?"
done
echo CADENA_FIN
