import subprocess, threading, time

def mem():
    # Registro de memoria cada minuto: si la VM muere, el último registro descargado muestra por qué.
    while True:
        out = subprocess.run("date -u +%T; free -m | sed -n 2p; nvidia-smi --query-gpu=memory.used --format=csv,noheader",
                             shell=True, capture_output=True, text=True).stdout.replace("\n", " ")
        with open("/content/mem.log", "a") as f:
            f.write(out + "\n")
        time.sleep(60)

threading.Thread(target=mem, daemon=True).start()
with open("/content/master.log", "w") as log:
    subprocess.run(["bash", "colab/run_semantic.sh"], cwd="/content/vigia", stdout=log, stderr=subprocess.STDOUT)
