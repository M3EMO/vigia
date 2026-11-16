# B39: EWC con la puerta de certeza

Type: task
Status: B39 corrido (2026-11-16): C0, S1 y S2 fallan; D1 pasa. Las selectividades no se suman; la protección sí queda
Blocked by: 17

Pedido del usuario (2026-11-16): «Dale, escribí el prerregistro de EWC con la puerta».

Pregunta: ¿la puerta de certeza suma selectividad sobre EWC, sin perder la protección del resto?

## Por qué esta corrida cambia decisiones

- B38 mostró que EWC corta 61 % del daño al resto y crea N2 −0,171 en la dosis 10. Esa selectividad vale 0,52 del oráculo.
- La puerta de certeza da la selectividad más fuerte (sonda −0,271, oráculo −0,328), pero no reduce el daño al resto.
- Si las dos se suman, el informe tiene una receta: protección y selectividad juntas.
- Si la puerta no suma sobre EWC, EWC alcanza en este banco, y la §4.15 lo dice.

## Base común

- El banco (206 hechos), los conjuntos (tag «b25»), las semillas, los lotes, la tasa y las dosis 10 y 40 son los de B25 y B38.
- EWC usa λ = 1, la Fisher y su corte del 10 % de B38, sin elegir de nuevo.
- Las reglas usan los conjuntos 1 a 7, los mismos de B38.

## Brazos nuevos (`colab/b39_combinada.py`)

- ewc+oraculo: EWC con los pesos del oráculo de B27 (0,25 a 1,75 por rango de duda, media 1).
- ewc+sonda: EWC con los pesos de la sonda de B27c.
- ewc+azar: EWC con los pesos permutados de B27. Es el control sin información.
- Los pesos se leen de `colab/data/b39_pesos.json`, copiados de los JSON de B27 y B27c.

## Referencias que no corren de nuevo

- comun: la ida de B25.
- ewc: el brazo ewc de B38.
- oraculo, azar y sonda sin EWC: los brazos de B27 y B27c.
- Con pesos iguales a 1, la pérdida pesada es la pérdida media del modelo. Un test lo verifica.

## Medidas (conjuntos 1 a 7, dosis 10 y 40)

- N2, D y A son los de B38.
- Los IC90 salen de un bootstrap de los 7 conjuntos, con `colab.b20_analisis.ci`.

## Reglas

- I0 (integridad): la pérdida base coincide con la de B25, con error 0.
- L0 (por brazo, dosis 40): A vale 0,5 o menos.
- C0 (control positivo, dosis 10): N2(ewc+oraculo) − N2(ewc) tiene IC90 bajo 0 y media de −0,15 o menos.
- S1 (principal, dosis 10): N2(ewc+sonda) − N2(ewc) tiene IC90 bajo 0 y media de −0,10 o menos.
- S2 (dosis 10): N2(ewc+sonda) − N2(sonda) tiene IC90 bajo 0 y media de −0,05 o menos.
- D1 (dosis 40): la reducción relativa del daño de ewc+sonda contra comun tiene IC90 sobre 0 y media de 0,40 o más.
- K (descriptivo): N2(ewc+azar) − N2(ewc), en las dosis 10 y 40.
- N2 de cada brazo en la dosis 40, como dato de persistencia.

## Decisiones

- C0 falla: la puerta no suma sobre EWC. La §4.15 dice que EWC ya captura la selectividad en este banco.
- C0, S1 y D1 pasan: EWC con la sonda es la receta del informe. El paso siguiente mide la dosis 160 (B39b).
- S1 pasa y S2 falla: la combinación iguala a la sonda en selectividad y suma la protección de EWC.
- D1 falla: los pesos de la puerta deshacen la protección de EWC. La §4.15 informa ese costo.

## Predicciones (Brier al cierre)

- C0 pasa: 0,65.
- S1 pasa: 0,55.
- S2 pasa: 0,35.
- D1 pasa: 0,70.

## Costo y tope

- Son 3 brazos, 7 conjuntos y 40 pasos, más 255 gradientes para la Fisher.
- Estimado: unos 15 minutos de L4, cerca de 0,5 u. Tope por timer: 35 minutos.
- La dosis 160 queda para B39b, solo si S1 pasa.

## Puntos débiles conocidos

- λ = 1 se eligió en B38 sin pesos. Con pesos, el mejor λ puede ser otro.
- Los pesos multiplican el gradiente de cada hecho. Eso cambia la tasa efectiva de cada hecho frente a la penalidad.
- Las referencias de B27 y B27c corrieron en otras VM. El entrenamiento es determinista entre VM, y I0 controla solo la base.
- La sonda se ajustó sobre el modelo de partida. EWC usa la Fisher del mismo modelo. Las dos señales pueden solaparse.

## Resultados de B39 (2026-11-16, unos 15 minutos de L4, ~0,5 u)

Análisis: `.cache/runs/bloques/b39_combinada_gpu/analisis.out` en mf-moe.
- I0 pasa, con error 0. L0 pasa en los tres brazos.
- Chequeo en CPU: con pesos 1, B39 repite exactamente el brazo ewc de B38 (diferencia 0). El camino de código es el mismo.
- C0 falla por poco: el oráculo suma −0,137 [−0,245; −0,024] sobre EWC en la dosis 10. El corte era −0,15.
- S1 falla: la sonda suma −0,064 [−0,160; +0,028] sobre EWC en la dosis 10.
- S2 falla: la combinación contra la sonda sola da +0,040 [−0,014; +0,089].
- D1 pasa: la combinación corta el daño al resto 49 % [45 %; 55 %] en la dosis 40.
- K: los pesos al azar no suman sobre EWC (+0,042 [−0,057; +0,137]). El control funciona.
- Decisión según el prerregistro: la puerta no suma sobre EWC. La §4.15 dice que EWC ya captura parte de la selectividad en este banco.
- S1 falló, así que B39b (dosis 160) no corre.

| Brazo (conjuntos 1 a 7) | N2, dosis 10 | N2, dosis 40 | Daño al resto, dosis 40 |
|---|---|---|---|
| comun (B25) | −0,019 | −0,031 | +2,74 |
| ewc (B38) | −0,171 | −0,109 | +1,04 |
| sonda (B27c) | −0,276 | −0,184 | +3,13 |
| oraculo (B27) | −0,339 | −0,288 | +2,64 |
| ewc+sonda | −0,235 | −0,193 | +1,39 |
| ewc+oraculo | −0,308 | −0,360 | +1,42 |
| ewc+azar | −0,129 | −0,089 | +1,15 |

Lectura (no prerregistrada):
- Las dos selectividades se solapan: EWC y la sonda leen la misma certeza del modelo de partida.
- La combinación conserva la selectividad de la sonda en la dosis 40 (−0,193 contra −0,184) y corta el daño a menos de la mitad (1,39 contra 3,13).
- Con el oráculo, la combinación supera a los dos en la dosis 40: −0,360, contra −0,288 del oráculo solo.
- En la dosis 10, los brazos combinados aprenden más lento: los hechos nuevos quedan en 1,1, contra 0,26 de ewc solo.
- En la dosis 10, los brazos combinados también dañan más (+2,5). La causa probable es la interacción entre los pesos desiguales y la penalidad.

Brier de B39: C0 0,423, S1 0,303, S2 0,122 y D1 0,090. La media es 0,234.
