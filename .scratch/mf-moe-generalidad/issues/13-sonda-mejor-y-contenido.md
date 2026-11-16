# B33 y B34: una sonda mejor, la dosis 160 y fuentes con contenido

Type: task
Status: B33 y B34 corridos (2026-11-16); S0 falla, D1 pasa; K1 y K3 pasan, con C1
Blocked by: 12

Pedido del usuario (2026-11-16): «Hace todo eso en colab». Eso incluye la sonda mejor, la dosis 160, las fuentes con contenido, otro modelo y FLAME-MoE.
Este ticket cubre los tres primeros puntos. El ticket 14 cubre otro modelo y el tamaño.

## B33: una sonda mejor y la caída en la dosis 160

Pregunta: ¿una sonda mejor acerca la puerta al oráculo? ¿La caída de la sonda en la dosis 160 viene de la calidad de la señal?

Hechos previos (B27 y B27c, mismo banco, mismos conjuntos y lotes):
- La sonda de B27c tiene Spearman 0,616 con la pérdida real.
- En la dosis 160, N2 de la sonda vale −0,023 [−0,063; +0,016]. N2 del oráculo vale −0,243 [−0,359; −0,134].

Diseño (`colab/b33_sonda2.py`):
- El banco, los conjuntos (tag «b25»), las semillas, los lotes y las dosis 10, 40 y 160 son los de B27c.
- La puerta es la de B27c: pesos de 0,25 a 1,75 por rango de duda.
- La sonda2 elige entre cuatro juegos de rasgos del modelo de partida:
  - b27c: el último token de la pregunta en las capas 8, 12, 16 y 20;
  - pares: el último token en todas las capas pares;
  - pares+media: además, el promedio de los tokens de la pregunta;
  - pares+media+extra: lo mismo, con 3.000 hechos de UniEdit más para entrenar.
- La elección usa validación cruzada en 5 pliegues sobre los candidatos que no están en el banco. La elección nunca mira el banco.
- El brazo oraculo_ruido usa la pérdida real con ruido gaussiano fijo. El ruido deja el Spearman en 0,616, el de la sonda de B27c.

Reglas (diferencia pareada por conjunto contra el azar de B27, IC90 por bootstrap de 8 conjuntos):
- I0 (integridad): la pérdida base coincide con la de B27, con error 0.
- R0 (control): el Spearman del ruido queda a menos de 0,01 de 0,616.
- S0 (señal): la sonda2 es mejor si su Spearman con la pérdida real llega a 0,666 o más.
- S1 (dosis 10): la sonda2 crea el nivel 2 si la diferencia contra el azar tiene IC90 bajo 0 y media de −0,15 o menos.
- D1 (dosis 160, principal): el ruido mantiene el efecto si la diferencia contra el azar tiene IC90 bajo 0 y media de −0,10 o menos.
- D2 (dosis 160): la misma regla para la sonda2.

Decisiones:
- D1 falla: una señal con Spearman 0,62 se apaga en la dosis 160. La caída de B27c viene de la calidad de la señal.
- D1 pasa: el ruido del mismo tamaño no se apaga. Los errores de la sonda tienen estructura, y el paso siguiente es buscarla.
- D2 pasa: el límite de §4.15 sobre la dosis 160 sale del informe.
- S0 falla: la sonda de B27c queda como la mejor lectura barata.

Predicciones (Brier al cierre):
- S0 pasa: 0,55.
- S1 pasa: 0,80.
- D1 pasa: 0,40.
- D2 pasa: 0,35.

## B34: fuentes con contenido y un umbral sin etiquetas

Pregunta: ¿el umbral funciona cuando las fuentes cambian el contenido? ¿El modelo puede contar fuentes sin etiquetas?

Diseño (`colab/b30_fuentes.py`, brazos nuevos):
- contenido: cada fuente antepone a su frase una oración de contexto distinta. Las 10 oraciones cambian el origen, la época y el método.
- umbral_sim: una frase cuenta como fuente nueva si su coseno con todas las fuentes ya contadas es menor que el corte.
  - El coseno usa el estado medio de los tokens en la capa del medio del modelo de partida.
  - El corte es el percentil 5 del coseno entre una frase y la misma frase con «Note: » adelante.
  - Por eso, un cambio trivial casi nunca cuenta como fuente nueva. El corte no mira ningún resultado.
- Brazos (dosis 10 y 40, banco de B27, tag «b25»): contenido|peso, contenido|umbral, fuentes|umbral_sim y contenido|umbral_sim.
- una|umbral_sim es igual a una|umbral por construcción: una frase repetida tiene coseno 1. Un test lo verifica.

Reglas (frases H de B30, que ningún brazo entrena):
- C0 (control positivo): en contenido|peso, el tercio dudoso adopta en 80 % o más en la dosis 40.
- C1 (chequeo de la manipulación): de 10 fuentes, el umbral_sim cuenta al menos 1,0 fuente más con contenido que con forma.
- K1: en contenido|umbral, el tercio seguro adopta en 80 % o más en la dosis 40.
- K2: en contenido|peso, el tercio seguro resiste en 50 % o más en la dosis 10.
- K3 (principal): en contenido|umbral_sim, el seguro adopta en 80 % o más en la dosis 40, y el dudoso adopta en 80 % o más en la dosis 10.
- K4 (descriptivo): en fuentes|umbral_sim, la adopción del seguro en la dosis 40 y la fracción de seguros abiertos.

Decisiones:
- K3 y C1 pasan: el modelo de partida puede contar fuentes independientes sin etiquetas. La afirmación 3 de §4.15 deja de depender de etiquetas.
- K3 pasa y C1 falla: el umbral funciona, pero la similitud no separa forma de contenido.
- K1 falla: el umbral con etiquetas no alcanza cuando cambia el contenido. La afirmación 3 queda limitada a la forma.
- K2 pasa: la diversidad de contenido sola protege al hecho seguro. Eso contradice B30 en la forma.

Predicciones (Brier al cierre):
- C1 pasa: 0,55.
- K1 pasa: 0,85.
- K2 pasa: 0,15.
- K3 pasa: 0,50.

## Costo y tope

- B33: unos 45 minutos de L4, 1,4 u. Tope por timer: 60 minutos.
- B34: unos 25 minutos de L4, 0,8 u. Tope por timer: 35 minutos.
- Las dos corren en paralelo, en dos VM.

## Puntos débiles conocidos

- Las 10 oraciones de contexto son plantillas mías. El contenido cambia, pero no viene de fuentes reales.
- El umbral_sim usa el modelo de partida y no el modelo que cambia. B28 mostró que la lectura externa y fija es la útil.
- La sonda2 elige entre cuatro opciones con 5 pliegues. La elección tiene algo de ruido.

## Resultados de B33 (2026-11-16, 36 minutos de L4, ~1,1 u)

Análisis: `.cache/runs/bloques/b33_sonda2_gpu/analisis.out` en mf-moe.
- I0 pasa: la pérdida base coincide con B27, con error 0. R0 pasa: el ruido tiene Spearman 0,6162.
- La validación cruzada eligió pares+media+extra: 0,495 contra 0,429 de b27c, sobre los candidatos.
- S0 falla: sobre el banco, la sonda2 tiene Spearman 0,571, menos que el 0,616 de B27c.
- Causa probable: los hechos extra no tienen filtro, y el banco solo tiene hechos que Granite conoce. La elección midió otra distribución.
- S1 pasa: en la dosis 10, la sonda2 contra el azar da −0,235 [−0,373; −0,123], con 0,64 del oráculo.
- D1 pasa: en la dosis 160, el ruido contra el azar da −0,120 [−0,219; −0,013], con 0,47 del oráculo.
- D2 falla: en la dosis 160, la sonda2 da −0,006 [−0,129; +0,129].
- Decisión según el prerregistro: los errores de la sonda tienen estructura. El ruido del mismo tamaño no se apaga como la sonda.

Chequeos exploratorios (CPU, sin prerregistro):
- El ruido menos la sonda de B27c, por conjunto: −0,010 en la dosis 10, −0,028 en la 40 y −0,082 [−0,155; −0,019] en la 160.
- La diferencia aparece solo en la dosis alta.
- Las dos señales dan el mismo rango medio de duda a los tercios: seguros 0,296 contra 0,291.
- El error de la sonda no sigue al largo del valor (−0,019), ni al tipo de hecho, ni a la fragilidad del hecho bajo el azar (+0,023).
- La estructura queda sin encontrar.

Brier de B33: S0 0,303, S1 0,040, D1 0,360 y D2 0,123. La media es 0,206.

## Resultados de B34 (2026-11-16, 25 minutos de L4, ~0,8 u)

Análisis: `.cache/runs/bloques/b34_contenido_gpu/analisis.out` en mf-moe.
- I0 pasa, con error 0. Las frases H1 y H2 sirven.
- C0 pasa: con contenido y pesos, el dudoso adopta en 100 % en la dosis 40.
- C1 pasa: el umbral por similitud cuenta 10,0 de 10 fuentes con contenido y 2,0 de 10 con forma.
- K1 pasa: con contenido y umbral por etiqueta, el seguro adopta en 98,5 % en la dosis 40.
- K2 falla: con contenido y pesos, el seguro resiste solo 0,209 en la dosis 10. La diversidad sola no protege.
- K3 pasa: con contenido y umbral por similitud, el seguro adopta en 98,5 % en la dosis 40, y el dudoso en 100 % en la dosis 10.
- K4: con forma y umbral por similitud, el seguro adopta solo en 3 % en la dosis 40. Ningún seguro se abre.
- El daño al resto queda igual que en B30: 2,31 con contenido y umbral, contra 2,30 con una voz y umbral.
- Decisión según el prerregistro: el modelo de partida puede contar fuentes sin etiquetas. La afirmación 3 de §4.15 deja de depender de etiquetas.

Punto débil nuevo:
- El corte vale 0,99997. Los estados medios de todas las frases tienen coseno cerca de 1.
- La separación vive en el quinto decimal. Esa similitud mide la diferencia del texto, no la independencia de las fuentes.

Brier de B34: C1 0,203, K1 0,023, K2 0,023 y K3 0,250. La media es 0,124.
