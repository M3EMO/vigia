# B20 en OLMoE-1B-7B

Type: task
Status: open
Blocked by: 01

## Question

¿Mapas, gradiente y fiabilidad de B20 se repiten en un MoE de 64 expertos con datos abiertos? Primero medir memoria y tiempo con una corrida piloto: 7B de parámetros no entran enteros en fp32 en una L4.

OLMoE: https://lacuna.tiptreesystems.com/work/olmoe-open-mixture-of-experts-language-models-6sr7/version/art_699789bfd8074140ab09d875380361c3

## Viabilidad (2026-11-06, sin gasto de unidades)

- OLMoE-1B-7B-0125 es compatible con ajustes, pero 12 capas entrenadas ocupan 84,3 GB sin activaciones.
- 1 capa ocupa 19,7 GB y 4 capas 37,3 GB, así que la comparación con Granite pierde fuerza.
- Antes de descargar OLMoE, una corrida en Granite mide si C0 pasa con 1 y 4 capas finales entrenables.
- Si C0 falla en Granite, este ticket no sirve en L4.
- La VRAM de L4 sale de una fuente secundaria (~22 GB): falta confirmarla con nvidia-smi.

Detalle: nota del vault "Viabilidad de ampliar mf-moe: OLMoE, FLAME-MoE, UniEdit y frecuencia de corpus".
