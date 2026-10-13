#!/usr/bin/env bash
# Corre un job en una VM L4 de Colab; si Colab retira la VM, crea otra y lo vuelve a lanzar.
# Uso: job_runner.sh NOMBRE JOB.py MARCADOR_LOG LOG_REMOTO DIR_LOCAL [archivos remotos a bajar...]
CLI=$HOME/.local/bin/colab; C="timeout 180 $CLI"
ZIP=$(cd "$(dirname "$0")/.." && pwd)/vigia-colab.zip
S=$1; JOB=$2; MARK=$3; RLOG=$4; D=$5; shift 5
mkdir -p $D; LOG=$D/runner.log
log() { echo "$(date -u +%T) $*" >> $LOG; }
start() {
  [ -f $D/exec.pid ] && kill $(cat $D/exec.pid) 2>/dev/null  # nunca pkill -f: el patrón aparece en la línea de comandos del propio runner
  $C stop -s $S >/dev/null 2>&1
  $C new -s $S --gpu L4 >> $LOG 2>&1
  $C upload -s $S $ZIP /content/vigia-colab.zip >> $LOG 2>&1
  nohup $CLI exec -s $S -f $JOB --timeout 30000 > $D/exec.log 2>&1 &
  echo $! > $D/exec.pid
  log "lanzado"
}
start
for i in $(seq 1 200); do
  sleep 120
  if $C status -s $S 2>&1 | grep -q "not found"; then log "VM retirada: relanzo"; start; continue; fi
  $C download -s $S $RLOG $D/$(basename $RLOG) >/dev/null 2>&1
  log "$(tail -1 $D/$(basename $RLOG) 2>/dev/null | cut -c1-100)"
  if grep -qE "$MARK" $D/$(basename $RLOG) 2>/dev/null; then
    for f in "$@"; do $C download -s $S $f $D/$(basename $f) >/dev/null 2>&1; done
    $C stop -s $S >> $LOG 2>&1; log "FIN"; exit 0
  fi
  # Un cliente con traceback y sin EXIT puede quedar colgado ("Connection was lost"): cuenta como caído.
  if ! kill -0 $(cat $D/exec.pid) 2>/dev/null || { grep -q "Traceback" $D/exec.log && ! grep -q "EXIT" $D/exec.log; }; then
    # Sin EXIT se cayó el cliente (p. ej. ReadTimeout) y la VM sin cliente muere a los ~20 min: relanzar.
    if ! grep -q "EXIT" $D/exec.log; then log "cliente caído: relanzo"; start; continue; fi
    # Con EXIT el job terminó sin marcador: relanzar repetiría la falla.
    for f in "$@"; do $C download -s $S $f $D/$(basename $f) >/dev/null 2>&1; done
    $C stop -s $S >> $LOG 2>&1; log "FALLA: exec terminó sin marcador: $(tail -c 150 $D/exec.log | tr '\n' ' ')"; exit 1
  fi
done
