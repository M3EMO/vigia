# Traspaso a otra PC (2026-10-12)

## Instalar

```bash
git clone https://github.com/M3EMO/vigia.git
cd vigia
uv venv --python 3.12 .venv
uv pip install -e ".[hf,serve,ee,dev]"
.venv/Scripts/python.exe -m pytest -q   # en Linux/macOS: .venv/bin/python
```

Los datasets ya vienen en `.cache/data/` (licencias en `.cache/data/LICENCIAS.md`).

## Corridas (no van en git)

Las corridas pesan unos 700 MB y viajan en `vigia-runs-2026-10-12.zip`. Descomprimirlo en la raíz del
repo: crea `.cache/runs/`. Lo que hay adentro:

- `gptoss_l4/`: gpt-oss-20b en L4. Factual 2.600 (`v2_f0f1.npz` sin nivel H, `gptoss20b_v2/rows.jsonl`),
  código 1.134 y mejor de N (`vigia_raw_code.tgz`), detectores, splits y resúmenes.
- `se_backup/`: las 5 muestras de entropía semántica y la latencia sin H.
- `evidence_v1/`, `granite_code/`: corridas de Granite en CPU.

Sin el zip, todo se puede regenerar en Colab con `colab/run_benchmarks.sh` (unas 11 h de L4).

## Estado

- Informe técnico al día hasta §4.9 y §5 (`docs/informe-tecnico.md`).
- Latencia F0+F1 en L4: +4,7 % después del commit `fbf073a` (meta 5 % cumplida, poco margen).
- Pendiente, a pedido del usuario ("probá ambos, elegimos después"):
  1. Perfil del router en GPU (`colab/profile_router.py`, job `colab/cli/prof_vm.py`).
  2. Modo vLLM real (`colab/vllm_mode.py`, job `colab/cli/vllm_vm.py`): humo, latencia con y sin
     `--enable-return-routed-experts`, evidencia sobre las 600 preguntas.
  Los dos se cortaron porque Colab retiró las VMs a los ~20 min (ver `colab/cli/README.md`).
- Orden de fases del usuario: Vigía → OS Harness → agente kernel. MF-MoE (fractal) es investigación.

## Colab desde la PC nueva

La CLI oficial (`google-colab-cli`) solo corre en Linux/macOS: en Windows, dentro de WSL Ubuntu.
`uv tool install google-colab-cli`, después cualquier comando pide el login de Google una vez.
Notas de uso medidas: `docs/claude/memoria/colab-cli-wsl.md`.

## Memoria de Claude

Copias de la memoria de la sesión original en `docs/claude/memoria/`. En la PC nueva, Claude Code
guarda la memoria en `~/.claude/projects/<ruta-del-proyecto>/memory/`: copiar esos archivos ahí y
agregarlos al índice `MEMORY.md` para que la próxima sesión arranque con el contexto.
