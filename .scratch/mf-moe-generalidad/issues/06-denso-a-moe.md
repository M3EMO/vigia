# Denso convertido a MoE

Type: research
Status: open
Blocked by: 02, 04

## Question

¿Un denso con datos abiertos, convertido sin reentrenar (ExpertWeaver, ToMoE), se comporta como un MoE entrenado como MoE? Decidir si esta rama cambia alguna decisión antes de costearla.

ExpertWeaver: https://lacuna.tiptreesystems.com/work/expertweaver-unlocking-the-inherent-moe-in-dense-llms/version/art_b2a3ee8dd4d44ae8aeaaa1b04cb52c01
ToMoE: https://lacuna.tiptreesystems.com/work/tomoe-converting-dense-large-language-models-to-mixture/version/art_a28cc1f70fb9449a8598e55599f9e9f2

## Viabilidad (2026-11-06, sin gasto de unidades)

- Decisión: se descarta por ahora por costo.
- ToMoE tiene código, pero entrena una hipered de 20.000 pasos en su ejemplo.
- DIVE reentrena con 5B tokens, y no se halló código de ExpertWeaver.

Detalle: nota del vault "Viabilidad de ampliar mf-moe: OLMoE, FLAME-MoE, UniEdit y frecuencia de corpus".
