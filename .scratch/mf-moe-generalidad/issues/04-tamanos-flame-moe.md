# Tamaños con FLAME-MoE

Type: task
Status: bloqueado (2026-11-16): FLAME-MoE no carga; el tamaño de §4.15 pasa al ticket 14 (Granite 3B)
Blocked by: 02

## Question

¿El gradiente y la comparación de mapas cambian con el tamaño? Suite de 38M/100M a 1,7B/10,3B, con 64 expertos y 2 compartidos. Revisar la licencia: la tarjeta dice Apache-2.0 y también «research purposes only». Verificar que los modelos chicos conozcan bastantes definiciones: si no, no hay daño que medir.

Paper: https://arxiv.org/abs/2505.20225

## Viabilidad (2026-11-06, sin gasto de unidades)

- FLAME-MoE no carga en transformers: los 3 repos leídos traen .distcp sin config.json.
- FLAME-290M-1.3B es el único tamaño comparable con Granite.
- Este ticket queda condicionado a encontrar un conversor.

Detalle: nota del vault "Viabilidad de ampliar mf-moe: OLMoE, FLAME-MoE, UniEdit y frecuencia de corpus".

## Revisión (2026-11-16, sin gasto de unidades)

- El repo de FLAME-MoE-290M-1.3B tiene 10 carpetas `iter_*` de Megatron, 182 GB en total, sin config.json ni safetensors.
- Hugging Face marca 10 archivos como inseguros. Los checkpoints de Megatron se cargan con pickle.
- El README de GitHub describe el entrenamiento y la evaluación. No publica un conversor a transformers.
- El PR 2 del repo 419M solo agrega metadatos.
- Decisión: no ejecuto pickles marcados como inseguros, y escribir un conversor propio no cambia ninguna decisión ahora.
- La pregunta de tamaño para §4.15 pasa al ticket 14, con Granite-3.1-3B-A800M.
- La pregunta original sobre el gradiente y los mapas de §4.13 sigue abierta.
