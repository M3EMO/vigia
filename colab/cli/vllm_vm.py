import subprocess
cmd = r'''
set -u
L=/content/vllm_stage.log
rm -rf /content/vigia && mkdir -p /content/vigia && cd /content/vigia && unzip -q -o /content/vigia-colab.zip
pip install -q "vllm>=0.21" > /content/pip_vllm.log 2>&1 || { echo "FALLA pip vllm" >> $L; exit 1; }
pip install -q --no-deps -e /content/vigia > /dev/null 2>&1
python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__)" >> $L 2>&1
serve() {
  [ -f /content/vllm.pid ] && kill $(cat /content/vllm.pid) 2>/dev/null && sleep 15  # nunca pkill -f: el patrón está en este mismo script
  nohup vllm serve openai/gpt-oss-20b $1 --max-model-len 4096 --gpu-memory-utilization 0.90 > /content/vllm_server.log 2>&1 &
  echo $! > /content/vllm.pid
  for i in $(seq 1 120); do curl -sf http://127.0.0.1:8000/health > /dev/null && return 0; sleep 10; done
  echo "FALLA vllm no arrancó ($1)" >> $L; return 1
}
serve --enable-return-routed-experts || exit 1
echo "== humo" >> $L
python colab/vllm_mode.py smoke >> $L 2>&1 || { echo "FALLA humo" >> $L; exit 1; }
grep -q "^proxy: eventos [1-9]" $L || { echo "FALLA proxy sin eventos" >> $L; exit 1; }
echo "== latencia con routed experts" >> $L
python colab/vllm_mode.py latency /content/lat_flag.json >> $L 2>&1
echo "== evidencia 600" >> $L
python colab/vllm_mode.py evidence data/evidence_v1.jsonl /content/vllm_v1 > /content/vllm_evidence.log 2>&1
tail -30 /content/vllm_evidence.log >> $L
tar czf /content/vllm_v1.tgz -C /content vllm_v1  # filas para el control por estrato
serve "" || exit 1
echo "== latencia sin routed experts" >> $L
python colab/vllm_mode.py latency /content/lat_noflag.json >> $L 2>&1
echo "ETAPAS_FIN" >> $L
'''
r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
print(r.stdout[-1500:], r.stderr[-1500:], "EXIT", r.returncode)
