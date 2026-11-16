# B40: ¿la Fisher de EWC sigue a la certeza de cada hecho?

Type: task
Status: prerregistrado (2026-11-16), sin correr
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
