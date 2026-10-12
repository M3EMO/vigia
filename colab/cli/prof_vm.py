import subprocess
cmd = r'''
set -e
rm -rf /content/vigia && mkdir -p /content/vigia && cd /content/vigia && unzip -q -o /content/vigia-colab.zip
pip install -q "transformers==5.18.0" accelerate safetensors kernels "triton>=3.4" scikit-learn > /dev/null 2>&1
pip install -q -e /content/vigia > /dev/null 2>&1
python colab/profile_router.py 6 128 3 > /content/profile.tmp 2>/content/profile.err; mv /content/profile.tmp /content/profile.txt; echo PERFIL_FIN >> /content/profile.txt
'''
r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
print(r.stdout[-1500:], r.stderr[-1500:], "EXIT", r.returncode)
