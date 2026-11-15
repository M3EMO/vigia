# B29: ¿la puerta de duda crea el nivel 2 también en otro banco?

Type: task
Status: G1 falla por la regla (2026-11-14); la diferencia pareada contra el azar aparece; regla pareada adoptada (2026-11-15)
Blocked by: 09, 10

## Question

B27 y B27c corrieron en un solo banco: 206 hechos reales que Granite conoce en las dos direcciones.
¿El nivel 2 por pesos, y el enlace de la sonda con el nivel 2, se repiten con otros hechos?
El usuario eligió esta réplica el 2026-11-14, con un tope de 2 unidades compartido con el ticket 12.

## Banco nuevo (2026-11-14, CPU, sin gasto de unidades)

- Fuente: los hechos de UniEdit de `b23_hechos.json`, fuera de los 461 candidatos de B25 y sin valores compartidos con ellos.
- Uno por valor. Se sortean 1.200 y quedan 687 con 9 donantes de filtro y 4 de medida distintos.
- Filtro: Granite prefiere el valor real contra los 9 donantes, solo en la ida. Pasan 348 (51 %).
- Banco: 256 de esos 348, sorteados. Los 8 tipos gruesos aparecen, de 17 a 47 hechos cada uno.
- Diferencia con B27: el banco de B27 exigía las dos direcciones. Este incluye hechos más dudosos.
- Código: `colab/b29_replica.py banco`. Salida: `colab/data/b29_banco.json`.

## Prerregistro B29 (2026-11-14, antes de ver datos de GPU)

Diseño (`python -m colab.b29_replica main`):

- Brazos oráculo, azar y sonda, con la puerta de B27: w = 0,25 + 1,5 · rango / (n − 1).
- 8 conjuntos de 32 hechos (tag «b29»). Solo la dosis 10, donde B27 y B27c midieron sus reglas principales.
- La sonda es la ridge de B27c sobre las capas 8, 12, 16 y 20. Entrena con los 461 candidatos y los otros conjuntos del banco.
- La medida N2 y su regla son las de B26 y B27.

Reglas:

- Integridad: la ida base gana contra los 4 donantes de medida en ≥ 80 %, y la sonda sigue a la pérdida base con Spearman ≥ 0,5.
- G1: N2 del oráculo presente, y N2 del azar no presente.
- P1: N2 de la sonda presente.
- P2: la sonda logra ≥ 0,75 del efecto del oráculo.

Qué decide cada resultado:

- G1 pasa: el nivel 2 por pesos se repite en otro banco. El paper lo afirma fuera del banco de B27.
- G1 falla: el nivel 2 queda limitado al banco de B27, y el paper lo dice.
- P1 y P2 pasan: el enlace entre la sonda (nivel 3) y el nivel 2 se repite.
- P1 o P2 fallan: el enlace queda limitado al banco de B27.

Predicciones: integridad 0,7. G1 0,8. P1 0,6. P2 0,45.
Costo: cerca de 0,35 u. Timer de 15 minutos, cerca de 0,48 u.

## Resultado B29 (2026-11-14, sesión b29rep, 10 minutos, cerca de 0,32 u)

Integridad: pasa. La ida base gana en 86,7 %, y la sonda sigue a la pérdida base con Spearman 0,527.

| Brazo | N2 en la dosis 10 | Regla |
|---|---|---|
| oráculo | −0,166 [−0,288; −0,033] | indeterminado |
| azar | +0,114 [−0,019; +0,237] | ausente |
| sonda | −0,081 [−0,224; +0,073] | indeterminado |

- G1 falla: el oráculo no llega a −0,2. P1 y P2 fallan: la sonda logra 0,49 del oráculo.
- Según el prerregistro: el nivel 2 por pesos queda limitado al banco de B27, y el paper lo dice.
- Brier: integridad 0,09. G1 0,64. P1 0,36. P2 0,2025.

Informativo, fuera del prerregistro:

- Sin puerta, este banco ya da N2 positivo: el azar marca +0,114. En B27 el azar marcó +0,038.
- La diferencia pareada contra el azar sí aparece: oráculo −0,280 [−0,380; −0,186], sonda −0,194 [−0,293; −0,093].
- En B27, la misma diferencia dio −0,366 para el oráculo y −0,309 para la sonda.
- Con la diferencia pareada, la sonda logra 0,69 del oráculo.
- La regla de B27 suponía un azar cerca de 0. Este banco muestra que ese supuesto no siempre vale.
- Aquí se aprende más lento: la pérdida de lo nuevo en la dosis 10 da 0,63, contra 0,33 en B27.
- Dos causas posibles: conjuntos de 32 hechos en vez de 25 (menos exposiciones por hecho) y hechos más dudosos.
- Análisis: el scratchpad de la sesión, `b29_analisis.py`. Salida en `b29_replica_gpu/analisis.out`.

## Decisión del usuario (2026-11-15): regla pareada para las réplicas

- El usuario adoptó la regla pareada para las réplicas futuras.
- La regla principal compara cada brazo contra la línea de base del mismo banco, conjunto por conjunto.
- La diferencia es presente si su IC90 queda entero de un lado y la media pasa el mínimo fijado antes de correr.
- B29 no cambia de resultado: su regla G1 falla, y la diferencia pareada queda como dato informativo.
- B31 (ticket 12) es la primera corrida con esta regla.

## Comments
