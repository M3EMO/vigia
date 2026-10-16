#!/usr/bin/env bash
# Corre un job en una VM L4 de Colab; si Colab retira la VM, crea otra y lo vuelve a lanzar.
# Uso: job_runner.sh NOMBRE JOB.py MARCADOR_LOG LOG_REMOTO DIR_LOCAL [archivos remotos a bajar...]
CLI=$HOME/.local/bin/colab; C="timeout 180 $CLI"
ZIP=${ZIP:-$(cd "$(dirname "$0")/.." && pwd)/vigia-colab.zip}  # otro proyecto: ZIP=ruta bash job_runner.sh ...
S=$1; JOB=$2; MARK=$3; RLOG=$4; D=$5; shift 5
EARLY=${EARLY:-}  # regex: archivos que se bajan apenas existen (p. ej. '\.safetensors$'), sin esperar al final
# Ruta remota de un checkpoint para retomar (p. ej. /content/ckpt.pt): se baja cuando cambia su marcador
# `<ruta>.step` y se sube a cada VM nueva antes de lanzar el job.
RESUME=${RESUME:-}; CK=$D/$(basename "${RESUME:-ckpt}")
mkdir -p $D; LOG=$D/runner.log
log() { echo "$(date -u +%T) $*" >> $LOG; }
start() {
  [ -f $D/exec.pid ] && kill $(cat $D/exec.pid) 2>/dev/null  # nunca pkill -f: el patrón aparece en la línea de comandos del propio runner
  $C stop -s $S >/dev/null 2>&1
  $C new -s $S --gpu L4 >> $LOG 2>&1
  $C upload -s $S $ZIP /content/vigia-colab.zip >> $LOG 2>&1
  if [ -n "$RESUME" ] && [ -s $CK ]; then
    timeout 900 $CLI upload -s $S $CK $RESUME >> $LOG 2>&1 && log "subido checkpoint $(cat $CK.step 2>/dev/null)"
  fi
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
  if [ -n "$RESUME" ]; then
    timeout 120 $CLI download -s $S $RESUME.step $CK.step.new >/dev/null 2>&1
    if [ -s $CK.step.new ] && ! cmp -s $CK.step.new $CK.step; then
      timeout 900 $CLI download -s $S $RESUME $CK.tmp >/dev/null 2>&1 && mv $CK.tmp $CK && mv $CK.step.new $CK.step \
        && log "bajado checkpoint $(cat $CK.step)"
    fi
  fi
  # Un retiro de la VM borra lo que no se bajó; el job escribe estos archivos de forma atómica.
  if [ -n "$EARLY" ]; then for f in "$@"; do
    [[ $f =~ $EARLY ]] && [ ! -s $D/$(basename $f) ] && timeout 900 $CLI download -s $S $f $D/$(basename $f) >/dev/null 2>&1 && log "bajado $(basename $f)"
  done; fi
  if grep -qE "$MARK" $D/$(basename $RLOG) 2>/dev/null; then
    for f in "$@"; do timeout 900 $CLI download -s $S $f $D/$(basename $f) >/dev/null 2>&1; done  # checkpoints de cientos de MB
    $C stop -s $S >> $LOG 2>&1; log "FIN"; exit 0
  fi
  # Un cliente con traceback y sin EXIT puede quedar colgado ("Connection was lost"): cuenta como caído.
  if ! kill -0 $(cat $D/exec.pid) 2>/dev/null || { grep -q "Traceback" $D/exec.log && ! grep -q "EXIT" $D/exec.log; }; then
    # Sin EXIT se cayó el cliente (p. ej. ReadTimeout) y la VM sin cliente muere a los ~20 min: relanzar.
    if ! grep -q "EXIT" $D/exec.log; then log "cliente caído: relanzo"; start; continue; fi
    # Con EXIT el job terminó sin marcador: relanzar repetiría la falla.
    for f in "$@"; do timeout 900 $CLI download -s $S $f $D/$(basename $f) >/dev/null 2>&1; done  # checkpoints de cientos de MB
    $C stop -s $S >> $LOG 2>&1; log "FALLA: exec terminó sin marcador: $(tail -c 150 $D/exec.log | tr '\n' ' ')"; exit 1
  fi
done
