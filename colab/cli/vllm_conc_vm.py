import subprocess
cmd = r'''
set -u
L=/content/conc_stage.log
rm -rf /content/vigia && mkdir -p /content/vigia && cd /content/vigia && unzip -q -o /content/vigia-colab.zip
pip install -q "vllm==0.31.0" > /content/pip_vllm.log 2>&1 || { echo "FALLA pip vllm" >> $L; exit 1; }
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
echo "== concurrencia con routed experts" >> $L
python colab/vllm_mode.py concurrency /content/conc_flag.json plain,proxy 1,4,16,32 >> $L 2>&1 || { echo "FALLA concurrencia" >> $L; exit 1; }
serve "" || exit 1
echo "== concurrencia sin routed experts" >> $L
python colab/vllm_mode.py concurrency /content/conc_noflag.json plain 1,4,16,32 >> $L 2>&1 || { echo "FALLA concurrencia" >> $L; exit 1; }
echo "ETAPAS_FIN" >> $L
'''
r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
print(r.stdout[-1500:], r.stderr[-1500:], "EXIT", r.returncode)
