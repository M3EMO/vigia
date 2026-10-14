import subprocess

cmd = r"""
set -u
L=/content/huella_stage.log
rm -rf /content/vigia && mkdir -p /content/vigia && cd /content/vigia && unzip -q -o /content/vigia-colab.zip
pip install -q "transformers==5.18.0" accelerate safetensors pandas requests > /dev/null 2>&1
echo "== prep" >> $L
python colab/innerexpert_prep.py /content/ie >> $L 2>&1 || { echo "FALLA prep" >> $L; exit 1; }
echo "== huella" >> $L
python colab/fingerprint.py /content/huella 200 >> $L 2>&1 || { echo "FALLA huella" >> $L; exit 1; }
echo "HUELLA_FIN" >> $L
"""
r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
print(r.stdout[-1500:], r.stderr[-1500:], "EXIT", r.returncode)
