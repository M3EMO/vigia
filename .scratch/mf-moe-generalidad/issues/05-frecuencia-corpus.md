# Frecuencia en el corpus abierto

Type: research
Status: open
Blocked by: 02

## Question

El nivel por concepto pesa cerca de la mitad de la varianza y nadie lo explica. Con el corpus de OLMoE o DCLM, ¿la frecuencia de cada concepto en el entrenamiento lo predice? Cuesta CPU y descarga parcial, no GPU.

## Viabilidad (2026-11-06, sin gasto de unidades)

- Prueba local sobre B20: rho entre nivel de daño y frecuencia SemCor -0,118, un proxy débil.
- Infini-gram da conteos exactos sin GPU; falta probarlo en vivo.
- Los datos de Granite no son públicos, así que el conteo sería un proxy para Granite y casi directo para OLMoE.

Detalle: nota del vault "Viabilidad de ampliar mf-moe: OLMoE, FLAME-MoE, UniEdit y frecuencia de corpus".
