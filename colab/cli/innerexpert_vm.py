import subprocess

# Plantilla: el encadenador reemplaza __DS__ por el conjunto (rtqa_2026, oos_squad, ...) antes de subirla.
cmd = r"""
set -u
DS=__DS__
L=/content/ie_stage.log
rm -rf /content/vigia && mkdir -p /content/vigia && cd /content/vigia && unzip -q -o /content/vigia-colab.zip
pip install -q "transformers==5.18.0" accelerate safetensors scikit-learn pandas requests > /dev/null 2>&1
pip install -q -e /content/vigia > /dev/null 2>&1
echo "== prep" >> $L
python colab/innerexpert_prep.py /content/ie >> $L 2>&1 || { echo "FALLA prep" >> $L; exit 1; }
echo "== extract $DS" >> $L
python -m vigia.cli extract --model allenai/OLMoE-1B-7B-0924-Instruct --device cuda --dtype bfloat16 \
  --data /content/ie/$DS.jsonl --out /content/ie_runs/$DS --tiers F0,F1 --max-tokens 65 \
  >> $L 2>&1 || { echo "FALLA extract" >> $L; exit 1; }  # el progreso por item queda en $L
cp /content/ie/$DS.jsonl /content/ie_runs/$DS/items.jsonl
tar czf /content/ie_$DS.tgz -C /content/ie_runs $DS
echo "IE_FIN" >> $L
"""
r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
print(r.stdout[-1500:], r.stderr[-1500:], "EXIT", r.returncode)
