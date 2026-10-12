---
name: colab-cli-wsl
description: "Correr GPU larga en Colab sin navegador: CLI oficial en WSL Ubuntu, exec bloqueante mantiene viva la VM"
metadata:
  node_type: memory
  type: reference
  originSessionId: 983d4a16-7fab-466c-9791-cbed900c30cc
  modified: 2026-10-08T19:44:35.173Z
---

Colab Pay As You Go en el navegador cortó 3 corridas largas (2026-10-07): la VM muere si la pestaña o la PC se duermen, y escribir directo en Drive perdió un `.tmp.npz`.

Lo que funcionó (2026-10-08, corrida de 11 h sin cortes):
- CLI oficial `colab` (google-colab-cli 0.7.4) instalada en WSL Ubuntu (`~/.local/bin/colab`); Windows no está soportado.
- Desde Git Bash: `MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -- bash -lc '...'` (sin eso, las rutas /mnt/c se rompen).
- `colab new -s NOMBRE --gpu L4`, `colab upload`, `colab exec -f script.py`.
- Lanzar el trabajo largo como `subprocess.run` bloqueante dentro de `colab exec`. Medido 2026-10-08: el kernel BUSY NO alcanza; sin un cliente conectado, Colab retira la VM a los ~20 min (3 VMs murieron así, con memoria sana). La corrida de 15 h sobrevivió porque un `colab exec` quedó colgado y conectado. Mantener un cliente vivo toda la corrida: `nohup colab exec ... --timeout 40000 &` dentro de WSL, y la PC despierta.
- Con el kernel ocupado no se puede `exec`; el progreso se lee con `colab download` de los logs.
- `colab drivemount` pide autorización OAuth interactiva: correrlo en una pestaña de terminal del usuario (run_in_terminal), no con stdin redirigido.
- Escribir en el disco local de la VM y copiar a Drive cada 5 min (ver `colab/run_benchmarks.sh` de [[vigia-proyecto]]).
- Otra sesión de Claude puede tener su propia VM ("moe") en la misma cuenta: no tocar sesiones ajenas.
