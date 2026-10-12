#!/usr/bin/env bash
# Vigilante de la corrida de entropía semántica: respalda las filas cada 3 min y, si Colab retira
# la VM, crea otra, restaura el respaldo y retoma (extract saltea los ids ya escritos).
CLI=/home/map12/.local/bin/colab
C="timeout 180 $CLI"  # la CLI a veces se cuelga tras "Connection was lost": ninguna llamada espera sin límite
T=/mnt/c/Users/map12/AppData/Local/Temp
V=/mnt/c/Users/map12/Desktop/Proyectos/vigia
B=$V/.cache/runs/se_backup
S=sew
LOG=$B/watchdog.log
mkdir -p $B
log() { echo "$(date -u +%T) $*" >> $LOG; }

start_vm() {
  pkill -f "exec -s $S" 2>/dev/null
  $C stop -s $S >> $LOG 2>&1
  $C new -s $S --gpu L4 >> $LOG 2>&1
  $C upload -s $S $V/colab/vigia-colab.zip /content/vigia-colab.zip >> $LOG 2>&1
  $C exec -s $S -f $T/se_restore.py --timeout 60 >> $LOG 2>&1 || $C exec -s $S -f $T/se_restore.py --timeout 60 >> $LOG 2>&1
  $C upload -s $S $V/.cache/runs/gptoss_l4/gptoss20b_v2/rows.jsonl /content/se_in/rows.jsonl >> $LOG 2>&1
  $C upload -s $S $V/.cache/runs/gptoss_l4/gptoss20b_v2_split.json /content/se_in/split.json >> $LOG 2>&1
  for k in 1 2 3 4 5; do
    if [ -s $B/se_s$k/rows.jsonl ]; then
      $C upload -s $S $B/se_s$k/manifest.json /content/runs/se_s$k/manifest.json >> $LOG 2>&1
      $C upload -s $S $B/se_s$k/rows.jsonl /content/runs/se_s$k/rows.jsonl >> $LOG 2>&1
    fi
  done
  timeout 1000 $CLI exec -s $S -f $T/se_setup.py --timeout 900 >> $LOG 2>&1
  nohup $CLI exec -s $S -f $T/se_launch.py --timeout 40000 > /tmp/sew_client.log 2>&1 &
  log "VM lanzada"
}

backup() {
  for k in 1 2 3 4 5; do
    mkdir -p $B/se_s$k
    if $C download -s $S /content/runs/se_s$k/rows.jsonl /tmp/rows_$k.jsonl > /dev/null 2>&1; then
      new=$(wc -l < /tmp/rows_$k.jsonl); old=$(wc -l < $B/se_s$k/rows.jsonl 2>/dev/null || echo 0)
      [ "$new" -ge "$old" ] && cp /tmp/rows_$k.jsonl $B/se_s$k/rows.jsonl
      $C download -s $S /content/runs/se_s$k/manifest.json $B/se_s$k/manifest.json > /dev/null 2>&1
    fi
  done
  $C download -s $S /content/master.log $B/master.log > /dev/null 2>&1
}

start_vm
while true; do
  sleep 180
  if $C status -s $S 2>&1 | grep -q "not found"; then
    log "VM retirada por Colab: relanzo"; start_vm; continue
  fi
  backup
  log "respaldo: $(for k in 1 2 3 4 5; do printf '%s ' $(wc -l < $B/se_s$k/rows.jsonl 2>/dev/null || echo 0); done)"
  if grep -q "latencia terminó" $B/master.log 2>/dev/null; then
    $C download -s $S /content/runs/bench_l4.json $B/bench_l4.json >> $LOG 2>&1
    $C stop -s $S >> $LOG 2>&1
    log "FIN"; exit 0
  fi
done
