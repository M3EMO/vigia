# Tamaños con FLAME-MoE

Type: task
Status: open
Blocked by: 02

## Question

¿El gradiente y la comparación de mapas cambian con el tamaño? Suite de 38M/100M a 1,7B/10,3B, con 64 expertos y 2 compartidos. Revisar la licencia: la tarjeta dice Apache-2.0 y también «research purposes only». Verificar que los modelos chicos conozcan bastantes definiciones: si no, no hay daño que medir.

Paper: https://arxiv.org/abs/2505.20225

## Viabilidad (2026-11-06, sin gasto de unidades)

- FLAME-MoE no carga en transformers: los 3 repos leídos traen .distcp sin config.json.
- FLAME-290M-1.3B es el único tamaño comparable con Granite.
- Este ticket queda condicionado a encontrar un conversor.

Detalle: nota del vault "Viabilidad de ampliar mf-moe: OLMoE, FLAME-MoE, UniEdit y frecuencia de corpus".
