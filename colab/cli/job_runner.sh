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
# Colab rechaza subir un archivo de 100 MB o más (400 o 500; 36 MB pasa, 2026-10-16): el checkpoint va en
# partes de 16 MB (con 32 MB hubo 408 por tiempo de espera) con reintentos y se junta en la VM. Si falla, el runner corta: arrancar de cero pisaría
# después el checkpoint bueno.
upload_ckpt() {
  local p n ok ka
  rm -f $CK.part.*
  split -b 16M -d -a 3 $CK $CK.part.
  # Sin cliente conectado, Colab retira la VM a los ~20 min, y subir 1 GB puede tardar más (2026-10-16:
  # retirada en la parte 50 de 64). Un exec corto que se repite la mantiene viva; la subida va por otro canal.
  printf 'import time\ntime.sleep(45)\n' > $D/keepalive.py
  ( while true; do timeout 120 $CLI exec -s $S -f $D/keepalive.py > /dev/null 2>&1; sleep 5; done ) &
  ka=$!
  for p in $CK.part.*; do
    ok=0
    for n in 1 2 3 4 5; do timeout 600 $CLI upload -s $S $p /content/$(basename $p) >> $LOG 2>&1 && { ok=1; break; }; sleep 10; done
    [ $ok = 1 ] || break
  done
  pkill -P $ka; kill $ka; wait $ka 2> /dev/null
  [ $ok = 1 ] || return 1
  sleep 50  # el último exec de la guardia puede seguir ocupando el kernel
  printf 'import os, subprocess\nsubprocess.run("cat /content/%s.part.* > %s && rm /content/%s.part.*", shell=True, check=True)\nprint("JOIN_OK", os.path.getsize("%s"))\n' \
    "$(basename $CK)" "$RESUME" "$(basename $CK)" "$RESUME" > $D/join_ckpt.py
  timeout 600 $CLI exec -s $S -f $D/join_ckpt.py > $D/join_ckpt.out 2>&1
  grep -q "JOIN_OK $(stat -c %s $CK)" $D/join_ckpt.out || return 1  # el tamaño junto tiene que coincidir
  rm -f $CK.part.*
  log "subido checkpoint $(cat $CK.step 2>/dev/null)"
}
# Bajar el checkpoint entero con `colab download` lo carga en RAM: con 3,6 GB el runner usó 10 GB y el OOM killer
# lo mató en cada cierre de dominio (fase 2 de mf-moe, 2026-10-19). Si el job escribe `<ruta>.partes` ("n bytes")
# y `<ruta>.part.NNN`, se baja por partes; partir con otro exec no sirve, porque queda en cola detrás del job.
download_ckpt() {
  local n size i ok
  rm -f $CK.partes $CK.dl.*
  timeout 120 $CLI download -s $S $RESUME.partes $CK.partes >/dev/null 2>&1
  if [ ! -s $CK.partes ]; then  # checkpoint chico, sin partes: de una vez
    timeout ${CKPT_TIMEOUT:-900} $CLI download -s $S $RESUME $CK.tmp >/dev/null 2>&1 && mv $CK.tmp $CK
    return
  fi
  read -r n size < $CK.partes
  for i in $(seq -f %03g 0 $((n - 1))); do
    ok=0
    for _ in 1 2 3 4 5; do timeout 600 $CLI download -s $S $RESUME.part.$i $CK.dl.$i >/dev/null 2>&1 && { ok=1; break; }; sleep 10; done
    [ $ok = 1 ] || { rm -f $CK.dl.*; return 1; }
  done
  cat $CK.dl.* > $CK.tmp && rm -f $CK.dl.*
  [ "$(stat -c %s $CK.tmp)" = "$size" ] || { rm -f $CK.tmp; return 1; }  # el tamaño junto tiene que coincidir
  mv $CK.tmp $CK
}
start() {
  [ -f $D/exec.pid ] && kill $(cat $D/exec.pid) 2>/dev/null  # nunca pkill -f: el patrón aparece en la línea de comandos del propio runner
  $C stop -s $S >/dev/null 2>&1
  $C new -s $S --gpu L4 >> $LOG 2>&1
  $C upload -s $S $ZIP /content/vigia-colab.zip >> $LOG 2>&1
  if [ -n "$RESUME" ] && [ -s $CK ]; then upload_ckpt || { log "FALLA: no se pudo subir el checkpoint"; exit 1; }; fi
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
      download_ckpt && mv $CK.step.new $CK.step && log "bajado checkpoint $(cat $CK.step)"
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
    # El job pudo escribir el marcador después de la última bajada del log (carrera vista el 2026-10-16).
    $C download -s $S $RLOG $D/$(basename $RLOG) >/dev/null 2>&1
    if grep -qE "$MARK" $D/$(basename $RLOG) 2>/dev/null; then $C stop -s $S >> $LOG 2>&1; log "FIN"; exit 0; fi
    $C stop -s $S >> $LOG 2>&1; log "FALLA: exec terminó sin marcador: $(tail -c 150 $D/exec.log | tr '\n' ' ')"; exit 1
  fi
done
