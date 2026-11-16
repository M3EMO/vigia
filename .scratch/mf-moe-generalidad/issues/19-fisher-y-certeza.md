# B40: ¿la Fisher de EWC sigue a la certeza de cada hecho?

Type: task
Status: B40 corrido (2026-11-16): R1 y R2 fallan; la lectura de la puerta implícita es falsa
Blocked by: 18

Pedido del usuario (2026-11-16): «Dale, medí la correlación entre la Fisher y la certeza».

Pregunta: ¿el gradiente de sobrescribir un hecho seguro cae más sobre los pesos que EWC frena?

## Por qué esta corrida cambia decisiones

- B38 y B39 leyeron a EWC como una puerta de certeza implícita. Esa lectura no está medida.
- Si la lectura se confirma, el informe la afirma como resultado. Si falla, el informe la saca y deja abierta la causa del N2 de EWC.

## Diseño (`colab/b40_fisher.py`)

- El modelo, las capas entrenables y la Fisher son los de B38: 255 candidatos fuera del banco, media 1 y el 10 % más alto.
- Para cada hecho i del banco (206), se calcula el gradiente g de la pérdida del valor nuevo, como en la sobrescritura.
- r_i = Σ F · g² sobre las entradas guardadas, dividido por Σ g² sobre todos los pesos entrenables.
- r_i mide qué fracción del cambio pedido por el hecho cae sobre pesos protegidos, pesada por su Fisher.
- Se guarda además r_i con el gradiente del valor viejo, como dato descriptivo.
- La certeza es −pérdida base del valor viejo, como en B27c.

## Reglas

- I0 (integridad): la Fisher tiene el mismo número de entradas que en B38.
- R1 (principal): el Spearman entre certeza y r_i, sobre los 206 hechos, tiene IC90 sobre 0 y media de 0,20 o más.
- R2: sobre los hechos sobrescritos de los conjuntos 1 a 7, el Spearman entre r_i y la protección que da EWC tiene IC90 sobre 0 y media de 0,15 o más.
  - La protección es la subida de B25 menos la subida de B38 (ewc), en la dosis 10.
- Los IC90 salen de un bootstrap de 10.000 remuestreos de hechos.

## Decisiones

- R1 y R2 pasan: la lectura de la puerta implícita queda medida. §1 y §4.15 la afirman.
- R1 pasa y R2 falla: la Fisher sigue a la certeza, pero no explica qué hechos protege EWC.
- R1 falla: la lectura es falsa. §4.15 la saca, y la causa del N2 de EWC queda abierta en §7.

## Predicciones (Brier al cierre)

- R1 pasa: 0,60.
- R2 pasa: 0,50.

## Costo y tope

- Son 255 gradientes para la Fisher y 412 gradientes por hecho, sin entrenamiento.
- Estimado: unos 8 minutos de L4, cerca de 0,25 u. Tope por timer: 20 minutos.

## Puntos débiles conocidos

- r_i usa un solo gradiente, en el punto de partida. Durante el ajuste, el gradiente cambia.
- La Fisher sale de hechos verdaderos, y la certeza también mira hechos verdaderos. Parte de la relación puede venir de esa fuente común.
- Los hechos de un mismo conjunto no son independientes en R2.

## Resultados de B40 (2026-11-16, unos 8 minutos de L4, ~0,25 u)

Análisis: `.cache/runs/bloques/b40_fisher_gpu/analisis.out` en mf-moe.
- I0 falla según la regla: la Fisher tiene 96.270.619 entradas, contra 96.310.832 en B38. La diferencia es 0,04 %.
- B39 ya difería de B38 en 0,02 %. Los gradientes del MoE en GPU no son deterministas, y el corte del 10 % se mueve apenas.
- La pérdida base coincide con B25, con error 0. La diferencia de 0,04 % no cambia la lectura de R1 ni de R2.
- R1 falla: el Spearman entre certeza y r (valor nuevo) vale −0,020 [−0,134; +0,092].
- R2 falla: el Spearman entre r y la protección de EWC vale +0,027 [−0,101; +0,159], en 175 hechos.
- Decisión según el prerregistro: la lectura de la puerta implícita es falsa. §4.15 la saca, y §7 deja abierta la causa.

Datos descriptivos:
- Certeza contra r del valor viejo: +0,101 [−0,015; +0,218].
- Certeza contra la norma del gradiente del valor nuevo: +0,166 [+0,056; +0,272]. Sobrescribir un hecho seguro pide un gradiente más grande.
- Certeza contra la protección de EWC, por hecho: +0,152 [+0,020; +0,282]. Es el N2 de EWC visto hecho por hecho.

Brier de B40: R1 0,360 y R2 0,250.
