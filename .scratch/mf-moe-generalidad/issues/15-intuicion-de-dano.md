# B37: una intuición de daño elige qué repasar (nivel 4 del video)

Type: task
Status: B37 corrido (2026-11-16): C0 falla; cualquier repaso protege igual, y la elección no importa
Blocked by: 13

Pedido del usuario (2026-11-16): «Dale», sobre el orden B37, regla heredada y saliencia.

Pregunta: ¿una intuición de daño, aprendida de ediciones pasadas, protege mejor que el azar al elegir qué hechos repasar?

## Base exploratoria (CPU, 2026-11-16, sin prerregistro)

- El previo es la fragilidad media de cada hecho en 331 corridas guardadas de B27 a B34.
- Cada conjunto usa solo filas de otros conjuntos.
- Contra la ida de B25 en la dosis 40, el previo tiene Spearman 0,571 con el daño real.
- El previo acierta 41 % de las 20 víctimas reales de cada conjunto. El azar acierta 11 %.
- Las 20 víctimas reales cargan el 24,5 % del daño total.
- La intuición mejora con la experiencia: el Spearman vale 0,31 con 1 fila, 0,52 con 10 y 0,57 con 100.

## Diseño (`colab/b37_repaso.py`)

- El banco, los conjuntos (tag «b25»), las semillas y los lotes son los de B25. Solo corre la ida, en las dosis 10 y 40.
- Cada edición suma 20 hechos viejos de repaso, con su valor real. Los lotes mezclan los 45 ítems.
- Brazos:
  - azar: 20 hechos no sobrescritos, al azar, con semilla fija por conjunto;
  - guiado: los 20 hechos con más fragilidad prevista por el previo (`colab/data/b37_previo.json`);
  - oraculo: los 20 hechos que más subieron en B25 sin repaso. Es el techo, porque mira el resultado.
- Con 0 hechos de repaso, el brazo repite la ida de B25. Un test lo verifica.
- La medida D es la subida media de la pérdida de los hechos no sobrescritos, incluidos los repasados.

## Reglas (por conjunto, IC90 por bootstrap de 8 conjuntos)

- I0 (integridad): la pérdida base coincide con la de B25, con error 0.
- C0 (control positivo, dosis 40): la reducción relativa del oráculo, (D_azar − D_oraculo) / D_azar, tiene IC90 sobre 0 y media de 0,03 o más.
- L4 (principal, dosis 40): la reducción relativa del guiado, (D_azar − D_guiado) / D_azar, cumple la misma regla.
- F (descriptivo): la fracción del techo, reducción del guiado sobre reducción del oráculo.
- A (aprendizaje): la pérdida de los hechos nuevos con guiado y con azar. Una diferencia mayor que 10 % queda marcada.
- Las mismas medidas se informan en la dosis 10.

## Decisiones

- C0 falla: repasar 20 hechos no cambia bastante el daño total. Este diseño no puede medir el nivel 4.
- C0 y L4 pasan: la intuición de daño protege mejor que el azar. El nivel 4 entra al lazo, y el informe lo suma.
- C0 pasa y L4 falla: el previo predice víctimas, pero esa predicción no alcanza para proteger mejor.

## Predicciones (Brier al cierre)

- C0 pasa: 0,75.
- L4 pasa: 0,55.

## Costo y tope

- Son 3 brazos, 8 conjuntos y 40 pasos. Eso da unos 17 minutos de L4, ~0,55 u.
- Tope por timer: 35 minutos.

## Puntos débiles conocidos

- El previo sale del mismo banco y del mismo modelo. Las filas de experiencia comparten hechos y no son independientes.
- El repaso diluye los lotes: los hechos nuevos ven menos pasos que en B25. Por eso B25 no sirve como brazo sin repaso.
- El oráculo usa la dosis 40 de B25 sin repaso. Con repaso, las víctimas reales pueden cambiar.

## Resultados de B37 (2026-11-16, 22 minutos de L4, ~0,7 u)

Análisis: `.cache/runs/bloques/b37_repaso_gpu/analisis.out` en mf-moe.
- I0 pasa, con error 0.
- C0 falla: el oráculo reduce el daño −0,053 [−0,250; +0,137] respecto del azar, en la dosis 40.
- L4 falla: el guiado reduce +0,023 [−0,108; +0,164].
- A queda marcada: la pérdida de los hechos nuevos en la dosis 40 es 0,041 con guiado y 0,021 con azar. La diferencia absoluta es chica.
- Decisión según el prerregistro: este diseño no puede medir el nivel 4.

Lectura de los hechos no repasados (exploratoria, la medida D tenía una falla):
- La medida D incluía a los hechos repasados. Esos hechos bajan su pérdida entre 1,55 y 1,92, y eso tapa el resto.
- Hechos no repasados, dosis 40: azar +1,098, guiado +1,048 y oráculo +1,102.
- B25 sin repaso, dosis 40: el resto sube +2,720, con los hechos nuevos ya aprendidos (0,019).
- Repasar 20 hechos cualesquiera corta el daño al resto cerca de 60 %, con un aprendizaje parecido.
- El repaso actúa como un ancla global. La identidad de los hechos repasados casi no cambia el daño de los demás.
- En la dosis 10 el repaso frena el aprendizaje: los hechos nuevos quedan en 0,93, contra 0,19 en B25.

Consecuencias:
- El nivel 4 no tiene palanca aquí: la intuición predice víctimas (Spearman 0,57), pero proteger no necesita esa predicción.
- La saliencia sale gratis de estos datos: cualquier hecho elegido para el repaso queda protegido, sin costo para los demás.
- Eso responde la parte de saliencia del nivel 5 sin otra corrida. La lectura es exploratoria.

Brier de B37: C0 0,563 y L4 0,303.
