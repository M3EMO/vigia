import subprocess
cmd = r'''
set -e
rm -rf /content/vigia && mkdir -p /content/vigia && cd /content/vigia && unzip -q -o /content/vigia-colab.zip
pip install -q "transformers==5.18.0" accelerate safetensors kernels "triton>=3.4" scikit-learn > /dev/null 2>&1
pip install -q -e /content/vigia > /dev/null 2>&1
python -c "import torch; print(torch.cuda.get_device_name(0))"
ls /content/se_in
'''
r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
print(r.stdout[-1500:], r.stderr[-1500:], "EXIT", r.returncode)
