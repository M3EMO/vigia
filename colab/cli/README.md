# Corridas en Colab con la CLI oficial (sin navegador)

Scripts que se usaron entre el 2026-10-08 y el 2026-10-12. Corren dentro de WSL Ubuntu, porque
`google-colab-cli` no soporta Windows. Las rutas `/mnt/c/Users/map12/...` son de la PC original:
ajustarlas en la PC nueva.

| Script | Qué hace |
|---|---|
| `job_runner.sh` | Corre un job en una L4; si Colab retira la VM, crea otra y relanza. Uso en la cabecera. |
| `se_watchdog.sh` | Vigilante de la corrida de entropía semántica: respalda filas cada 3 min y retoma. |
| `se_setup.py`, `se_restore.py`, `se_launch.py` | Preparación, restauración y lanzamiento de esa corrida. |
| `prof_vm.py` | Job pendiente: perfil del router en GPU (`colab/profile_router.py`). |
| `vllm_vm.py` | Job pendiente: modo vLLM real (`colab/vllm_mode.py`): humo, latencia y evidencia 600. |

Problema abierto: Colab retira las VMs creadas por la CLI a los ~20-45 min, de día. Dos corridas
largas sobrevivieron (15 h y 4,5 h), las dos de noche. Hipótesis sin confirmar: una pestaña de Colab
abierta en el navegador desplaza a la VM de la CLI. Por eso los jobs tienen que ser reanudables.
