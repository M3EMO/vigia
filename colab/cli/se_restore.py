import os
for k in range(1, 6):
    os.makedirs(f"/content/runs/se_s{k}", exist_ok=True)
os.makedirs("/content/se_in", exist_ok=True)
print("ok")
