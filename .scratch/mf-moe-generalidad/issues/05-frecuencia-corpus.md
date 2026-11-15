# Frecuencia en el corpus abierto

Type: research
Status: closed
Blocked by: 02

## Question

El nivel por concepto pesa cerca de la mitad de la varianza y nadie lo explica. Con el corpus de OLMoE o DCLM, ¿la frecuencia de cada concepto en el entrenamiento lo predice? Cuesta CPU y descarga parcial, no GPU.

## Viabilidad (2026-11-06, sin gasto de unidades)

- Prueba local sobre B20: rho entre nivel de daño y frecuencia SemCor -0,118, un proxy débil.
- Infini-gram da conteos exactos sin GPU; falta probarlo en vivo.
- Los datos de Granite no son públicos, así que el conteo sería un proxy para Granite y casi directo para OLMoE.

Detalle: nota del vault "Viabilidad de ampliar mf-moe: OLMoE, FLAME-MoE, UniEdit y frecuencia de corpus".

## Resultado con proxy (2026-11-06, sin gasto de unidades)

Conteos exactos de `v4_dclm-baseline_llama` en Infini-gram para los 600 conceptos de B20, sumando minúscula y mayúscula inicial. DCLM es un proxy, porque los datos de Granite no son públicos.

Regla fijada antes de ver los conteos: la frecuencia entra como covariable si |ρ parcial| ≥ 0,15 y el IC90 excluye 0. El umbral se fijó conociendo ρ = −0,118 de SemCor.

- ρ parcial global: −0,140 [−0,202; −0,074]. La regla no se cumple.
- ρ parcial dentro de tema: −0,038 [−0,107; +0,027].
- Controles: pérdida previa, tokens del sujeto, SemCor y polisemia.
- Sensibilidad post hoc con 377 conceptos de un sentido: global −0,172 [−0,255; −0,092], dentro de tema −0,038 [−0,126; +0,042].

Decisión: la frecuencia no entra como covariable de los mapas. El paper declara que la frecuencia de palabra no explica el nivel de daño en este proxy.

La señal global desaparece dentro de cada tema, así que refleja sobre todo diferencias entre temas. El ticket sigue abierto solo para una prueba directa con OLMoE, si el ticket 02 corre.

Límites: el conteo es por palabra y no por sentido; el análisis no controla colocaciones; el servicio limitó las consultas, pero los conteos quedaron completos.

## Cierre (2026-11-14, decisión del usuario)

- La frecuencia no entra como covariable de los mapas: el ρ parcial dio −0,140, y la regla pedía |ρ| ≥ 0,15.
- La prueba directa con OLMoE dependía del ticket 02, y el ticket 02 está cerrado.
- El paper declara el resultado del proxy DCLM con sus límites.
