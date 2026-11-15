# B24: ¿el olvido es direccional? (candidato 1)

Origen: Alien, nota del vault «Paper candidato de Alien: ¿el olvido es direccional?».
El conteo inverso sobre UniEdit dio pocos hechos conocidos en las dos direcciones, así que B24 usa un banco ficticio.

## Literatura y Lacuna (2026-11-11)

Berglund et al. (Reversal Curse, ICLR 2024) muestran que un hecho enseñado solo en la ida no se recupera en la vuelta.
«The Illusion of Latent Generalization» (Re-Align, ICLR 2026) halla que la ida y la vuelta viven en entradas separadas.
Ese trabajo mide distancias y sondas lineales. No mide qué pasa con una dirección cuando se sobrescribe la otra.
BAKE (arXiv 2310.10322) y la representación bilineal (ICLR 2026) miden si una edición llega al reverso del hecho editado.
Consistency-Aware Editing (arXiv 2601.08840) borra entidades con paráfrasis, sin separar la dirección inversa.
No encontré en Lacuna un trabajo que mida el olvido de la dirección no entrenada. B24 es la prueba causal de las dos entradas.

## Diseño preregistrado

- Banco: 256 hechos ficticios, uno a uno. Cabezas y valores son nombres inventados de dos palabras, de unos 6,5 tokens.
- 8 propiedades asimétricas: founder, director, author, inventor, architect, composer, designer, discoverer.
- Ida: «The {propiedad} of {cabeza} is» + « {valor}.». Vuelta: «{valor} is the {propiedad} of» + « {cabeza}.».
- Cada hecho compite con 4 donantes de igual propiedad en cada dirección. El azar gana con p = 0,2.
- Etapa 1: Granite aprende las 512 oraciones, capas 6 a 23, AdamW 5e-5, lotes de 16. Para con pérdida media < 0,25 o a los 2000 pasos.
- Etapa 2: 8 conjuntos disjuntos de 32 hechos cubren el banco. Cada conjunto corre dos brazos desde la etapa 1.
- Brazo ida: enseña un valor nuevo inventado en la ida. Brazo vuelta: enseña una cabeza nueva inventada en la vuelta.
- La etapa 2 no usa anclas y para como B23: pérdida media < 0,5, con tope de 300 pasos.
- Medida: subida de la pérdida del hecho viejo en cada dirección, contra el final de la etapa 1.
- Daño específico de una dirección: subida media de los 32 sobrescritos menos subida media de los 224 no tocados.
- Unidad de réplica: el conjunto (8). Intervalo: IC90 bootstrap con `b20_analisis.ci`.

## Reglas

- Puerta E1, dentro del script: la etapa 1 debe ganar en ≥ 90 % de los hechos en cada dirección. Si no, corta antes de la etapa 2.
- Aprendizaje: cada brazo baja la pérdida nueva de 0,5 en ≥ 90 % de sus corridas.
- P_ida y P_vuelta (controles positivos): el daño específico de la dirección entrenada tiene IC90 > 0 y punto ≥ 0,5.
- P_vuelta además prueba que la medida ve daño en la vuelta. Sin él, una vuelta intacta no prueba nada.
- D_ida (regla principal): R = daño específico de la vuelta / daño específico de la ida, en el brazo ida.
- D_ida es direccional si la cota superior del IC90 de R queda debajo de 0,25. Es compartido si la cota inferior supera 0,5.
- Entre esos bordes, D_ida es parcial. D_vuelta usa la misma regla con las direcciones cambiadas, y es informativo.
- C (maldición de la inversión): en el brazo ida, la vuelta del hecho nuevo gana con IC90 superior < 0,5.
- Informativo: fracción de sobrescritos que todavía ganan en la dirección no entrenada.

## Decisión que cambia

- Direccional: el candidato 1 sigue como paper. Un borrado o una edición medidos en la ida dejan el hecho vivo en la vuelta.
- Direccional: el paso que sigue es repetirlo con hechos reales de preentrenamiento, con los 544 del conteo inverso.
- Compartido: las dos entradas no se sostienen por su efecto. El candidato 1 muere en esta forma y queda como nota.
- Parcial: el informe da la magnitud. El paso siguiente se decide con el número.
- Si C falla, Granite invierte los hechos nuevos, y la lectura de D cambia: la dirección no entrenada también se aprende.

## Costo

Estimado: 1,3 unidades, con tope de 75 minutos. Arranque cerca de 15 minutos, etapa 1 hasta 14 y 16 corridas de 40 a 110 s.
Shortcut: la etapa 1 no se guarda entre VMs. Una VM nueva la repite con la misma semilla.
Análisis: `b24_analisis.py SALIDA.json`, escrito antes de ver datos.

## Resultado (2026-11-11, sesión b24inv, 10 minutos, cerca de 0,35 unidades)

Puerta E1: la etapa 1 paró a los 250 pasos con pérdida 0,16. Gana en 95 % de la ida y 97 % de la vuelta.
Los dos brazos aprenden en 8 de 8 corridas, siempre en 10 pasos, el primer control de la pérdida.
P_ida: daño específico de la ida +1,40 [+1,31; +1,50]: gana. P_vuelta: daño específico de la vuelta +1,40 [+1,27; +1,50]: gana.
D_ida: R = +0,095 [+0,064; +0,125]: direccional. La vuelta pierde cerca de un décimo de lo que pierde la ida.
D_vuelta: R = +0,041 [+0,012; +0,071]: direccional también, con las direcciones cambiadas.
En aciertos, tras el brazo ida, la vuelta vieja todavía gana en 90 % de los sobrescritos y la ida en 55 %.
C: la otra cara del hecho nuevo gana en 20 %, el azar exacto. Granite no invierte lo nuevo.
El daño compartido es chico pero no es cero: el IC90 de la vuelta en el brazo ida no toca 0.
Decisión según el prerregistro: el candidato 1 sigue. El paso que sigue es repetirlo con hechos reales de preentrenamiento.

Puntos débiles que este resultado no cubre:
- Dosis: la etapa 2 paró siempre a los 10 pasos. No sé si el cociente se mantiene con más sobrescritura.
- Origen: los dos sentidos se enseñaron como dos oraciones de ajuste fino. En el preentrenamiento podrían guardarse juntos.
- Hechos reales: los 544 del conteo inverso incluyen cerca de 192 por azar. Esos hechos no tienen vuelta que perder y empujan R hacia direccional.
- Profundidad: la etapa 1 enseñó poco, unas 8 pasadas. Un hecho más fijado podría compartir más.

## Variante de dosis preregistrada (2026-11-11, antes de ver datos)

Pregunta: ¿la vuelta sobrevive si la ida se sobrescribe con mucha más presión?
La literatura de edición mide el reverso tras una sola edición. No encontré una curva de dosis del reverso.

Diseño:
- Mismo banco, misma etapa 1, mismas semillas y mismos conjuntos que B24.
- La etapa 2 no para por pérdida. Cada brazo corre 160 pasos y se mide en los pasos 10, 40 y 160.
- La medida intercalada no cambia el entrenamiento. Un test lo prueba contra la corrida que para por pérdida.
- La dosis 10 es la de B24: la etapa 2 de B24 paró siempre en el paso 10.
- El paso 160 es 16 veces la dosis de B24, unas 80 pasadas por los 32 hechos nuevos.

Reglas:
- Réplica (control): en la dosis 10, el R_ida medio cae en el IC90 de B24, [+0,064; +0,125]. Si no, busco el error y no leo el resto.
- Aprendizaje: cada brazo y cada dosis baja la pérdida nueva de 0,5 en ≥ 90 % de las corridas.
- P_ida en la dosis 160: el daño específico de la ida tiene IC90 > 0 y punto ≥ 0,5. Sin P, D160 no se lee.
- D160 (regla principal): R_ida en la dosis 160, con los bordes de B24 (direccional < 0,25, compartido > 0,5).
- S160 (guarda): tras el brazo ida, la vuelta vieja gana en los sobrescritos con cota inferior del IC90 ≥ 0,75.
- Informativo: R_ida(160) − R_ida(10) por conjunto, D_vuelta y C en cada dosis, y la deriva del resto.

Decisión que cambia:
- D160 direccional y S160 en pie: la debilidad de dosis se cierra. La réplica con hechos reales usa una sola dosis.
- D160 parcial o compartido, o S160 caída: la separación depende de la dosis. El paper informa la curva.
- En ese caso, la réplica con hechos reales corre las tres dosis.
- D160 direccional con S160 caída: la afirmación queda en la escala de pérdida y el paper lo dice.

Costo: cerca de 1,2 unidades, con tope de 70 minutos (2,3 unidades).
El arranque toma cerca de 6 minutos, la etapa 1 cerca de 3, y 16 corridas de cerca de 100 s.
Análisis: `b24_dosis_analisis.py SALIDA.json`, escrito y probado sobre una corrida chica antes de ver datos.

## Resultado de la variante de dosis (2026-11-11, sesión b24dos, 38 minutos, cerca de 1,25 unidades)

Réplica: la etapa 1 y la dosis 10 repiten B24 en todos los números. R_ida en la dosis 10 vale +0,095 [+0,064; +0,125].
Aprendizaje: los dos brazos aprenden en 8 de 8 corridas en cada dosis. P_ida y P_vuelta ganan en las tres dosis.
R_ida: +0,095 en la dosis 10, +0,109 en la 40 y +0,135 [+0,091; +0,183] en la 160.
D160 (regla principal): direccional. La cota superior, 0,183, queda debajo de 0,25.
R_vuelta: +0,041, +0,088 y +0,084 [+0,035; +0,133]. Direccional en las tres dosis.
Tendencia: R_ida sube +0,040 [+0,002; +0,078] entre las dosis 10 y 160. La subida es chica pero no es cero.
S160 (guarda): falla. Tras el brazo ida, la vuelta vieja gana en 0,605 [0,559; 0,652] de los sobrescritos.
C: se cumple en las tres dosis. La otra cara del hecho nuevo gana en 0,15 o menos.

Lectura de S160 (informativo, agregado tras ver el conjunto 0):
- Los no tocados también caen. Su vuelta gana en 0,93 en la dosis 10 y en 0,70 en la 160.
- La deriva general explica la mayor parte de la caída de S160. La caída específica de la vuelta es 0,093 en aciertos.
- En la dosis 160, el resto del banco sube 3,2 nats en la ida. El ajuste largo daña todo el banco.
- En aciertos, el cociente del brazo ida sube de 0,12 a 0,44. La dirección propia toca el piso, y ese piso comprime el cociente.
- El brazo vuelta queda direccional también en aciertos: 0,08, 0,07 y 0,14.

Decisión según el prerregistro:
- D160 direccional con S160 caída: la afirmación queda en la escala de pérdida, y el paper lo dice.
- S160 caída: el paper informa la curva de dosis. La réplica con hechos reales corre las tres dosis.

Puntos débiles nuevos:
- La dosis 160 destruye el banco entero. Una edición real usa mucho menos presión.
- La escala de aciertos tiene piso, así que el cociente en aciertos no se lee limpio en la dosis 160.
- La asimetría entre brazos sigue sin explicación: la ida comparte más daño que la vuelta.

## B25: réplica con hechos reales, preregistrada (2026-11-11, antes de ver datos)

Pregunta: ¿la vuelta sobrevive cuando las dos direcciones vienen del preentrenamiento y no de un ajuste fino?

Literatura y Lacuna:
- Berglund et al. miden la maldición de la inversión con padres de celebridades reales. No sobrescriben nada.
- «Dynamics Reveals Structure» (Chang, Mucsányi y Oh, 2026) predice que una actualización local no llega a la relación inversa.
- Ese trabajo es teórico y no mide el olvido de hechos reales. B25 es la prueba causal.

Banco:
- Parte de los 544 hechos del conteo inverso de B23: pasan F0 en la ida y en la vuelta y no son piso.
- Quedan 461 al dejar un hecho por valor. Un valor compartido haría que un brazo vuelta toque a otro hecho.
- Quedan 379 con donantes suficientes. El script `colab.b25_reales candidatos` arma la lista.
- Filtro estricto en GPU: cada hecho gana contra 9 donantes nuevos en la ida y contra 9 en la vuelta.
- Con el filtro, un hecho desconocido pasa las dos direcciones por azar con p ≈ 0,01.
- La medida usa otros 4 donantes por dirección, como B24. Esos donantes no repiten los del filtro.
- Los valores y cabezas nuevos son nombres inventados, distintos de todo UniEdit.

Etapa 2: la de B24 en modo dosis, sin etapa 1. Granite parte tal como viene. Mide en los pasos 10, 40 y 160.
Los conjuntos son 8. Cada uno tiene n // 8 hechos, y el sobrante queda siempre en el resto.

Reglas:
- Puerta F: el filtro estricto deja ≥ 128 hechos. Si no, el script corta antes de entrenar.
- Puerta E0: la línea base gana en ≥ 80 % de los hechos en cada dirección con los donantes de la medida.
- Aprendizaje: cada brazo y cada dosis baja la pérdida nueva de 0,5 en ≥ 90 % de las corridas.
- P_ida en la dosis 10: daño específico con IC90 > 0 y punto ≥ 0,5.
- Si P_ida falla en la dosis 10, la regla principal pasa a la primera dosis donde P_ida gana.
- D10 (regla principal): R_ida en la dosis 10, con los bordes de B24 (direccional < 0,25, compartido > 0,5).
- D160 (segunda regla) y S160 (guarda): las mismas de la variante de dosis.
- Informativo: tendencia de R_ida, D_vuelta, C, deriva del resto y aciertos de los no tocados.

Decisión que cambia:
- D10 direccional: la separación vale también para hechos de preentrenamiento. El candidato 1 sigue como paper.
- D10 compartido: la separación es un artefacto del ajuste fino. El candidato 1 se achica a hechos aprendidos por ajuste.
- D10 parcial: el informe compara la magnitud con B24. El paso siguiente se decide con el número.
- D160 y S160 deciden la escala de la afirmación, como en la variante de dosis.

Costo: cerca de 1,3 unidades, con tope de 70 minutos (2,3 unidades).
El arranque toma cerca de 6 minutos, el filtro cerca de 1, y 16 corridas de cerca de 2 minutos.
Antes de gastar, una vista previa en CPU cuenta cuántos hechos pasan el filtro. Esa vista solo hace inferencia.
Análisis: `b25_analisis.py SALIDA.json`, escrito antes de ver datos.

## Resultado de B25 (2026-11-11, sesión b25real, 37 minutos, cerca de 1,2 unidades)

Puerta F: el filtro estricto deja 206 de 379 hechos. Biomedicina aporta 50 y obra cultural aporta 15.
Puerta E0: la línea base gana en 0,90 de la ida y en 0,94 de la vuelta.
Aprendizaje: los dos brazos aprenden en 8 de 8 corridas en cada dosis. P_ida y P_vuelta ganan en las tres dosis.
D10 (regla principal): parcial. R_ida vale +0,328 [+0,210; +0,425]. En B24 valía +0,095 [+0,064; +0,125].
D_vuelta en la dosis 10: parcial, +0,434 [+0,222; +0,624].
R_ida sube con la dosis: +0,533 en la 40 y +0,610 [+0,554; +0,655] en la 160.
D160 (segunda regla): compartido. S160 falla: la vuelta vieja gana en 0,38 de los sobrescritos.
Tendencia: R_ida sube +0,282 [+0,172; +0,415] entre las dosis 10 y 160. En B24 subía +0,040.
C: se cumple. La otra cara del hecho nuevo gana en 0,29 en la dosis 10, un poco sobre el azar de 0,2.

Informativo, calculado después del análisis preregistrado:
- En aciertos, la dosis 10 casi no daña la vuelta. Los no tocados ganan en 0,78 y los sobrescritos en 0,74.
- La caída específica en aciertos es +0,045 [+0,007; +0,083] en la dosis 10. En la dosis 40 sube a +0,196.
- La deriva es mayor que en B24. El resto sube 1,9 nats en la ida en la dosis 10. En B24 subía 0,86.
- La línea de aciertos de los no tocados faltó en `b25_analisis.py`. La copia base no tenía el agregado de la variante de dosis.

Decisión según el prerregistro:
- D10 parcial: el informe compara la magnitud con B24. El paso siguiente se decide con el número.
- Los hechos del preentrenamiento comparten cerca de tres veces más daño que los del ajuste fino.
- Con dosis alta, el daño de los hechos reales pasa a compartido. La separación en hechos reales es un efecto de dosis baja.

Puntos débiles:
- Origen y entidad van juntos. Un hecho real tiene una cabeza conocida, y un hecho de B24 tiene un nombre inventado.
- Por eso no sé si la diferencia viene del preentrenamiento o de la riqueza de la entidad.
- La deriva general es grande. El ajuste de 18 capas con LR 5e-5 daña los hechos reales en bloque.
- Los tipos tienen tamaños distintos. No separé el resultado por tipo, porque cada conjunto tiene pocos hechos por tipo.

## B26: celda 2×2 que separa origen y entidad, preregistrada (2026-11-13, antes de ver datos)

Pregunta: ¿B25 comparte más daño porque el hecho viene del preentrenamiento o porque la entidad es real?
B24 juntó origen de ajuste con entidad inventada. B25 juntó origen de preentrenamiento con entidad real.
B26 junta origen de ajuste con entidad real. La cuarta celda no existe, porque un hecho inventado no está en el preentrenamiento.

Banco:
- 256 hechos falsos. Las cabezas son organizaciones y obras reales de UniEdit, con forma de nombre propio.
- Los valores son personas reales de UniEdit. Las propiedades son las 8 de B24.
- Ejemplo de forma: «The founder of {organización} is {persona}.» y su vuelta.
- Los valores y cabezas nuevos son nombres inventados, como en B24.
- Etapa 1, etapa 2, semillas, donantes y medida: los de B24. Dosis 10 y 40.
- Elegí la dosis 40 como principal porque ahí B24 y B25 quedan más lejos: 0,109 contra 0,533.

Reglas:
- Puerta E1, dentro del script: la etapa 1 gana en ≥ 90 % en cada dirección.
- Aprendizaje y P_ida: los de B24, en cada dosis. Sin P_ida, la regla de esa dosis no se lee.
- D40c (regla principal): el borde es el punto medio entre B24 y B25 en la dosis 40, 0,321.
- D40c da «origen» si la cota superior del IC90 de R_ida queda debajo del borde.
- D40c da «entidad» si la cota inferior supera el borde. Si no, da «mezcla».
- D10c (segunda regla): la misma, con el borde de la dosis 10, 0,212.

Decisión que cambia:
- Origen: la direccionalidad depende de cómo se aprendió el hecho. El paper afirma eso.
- Entidad: una entidad real comparte daño entre direcciones aunque el hecho venga del ajuste. El paper afirma eso.
- Mezcla: el informe da la magnitud de cada factor con las tres celdas.

Punto débil conocido: un hecho falso sobre una organización conocida choca con lo que Granite ya sabe de ella.
Costo: cerca de 0,6 unidades, con tope de 60 minutos (1,9 unidades).
Análisis: `b26_analisis.py SALIDA.json`, escrito y probado sobre una corrida chica antes de ver datos.

## Nivel 2 (certeza contra plasticidad), preregistrado sobre B26 (2026-11-13, antes de ver datos)

Origen: el video «niveles de aprendizaje». Un cerebro cambia menos una creencia segura que una dudosa.
Ya vi la correlación en B24 y B25: rho cerca de 0 (exploratorio, en la nota del vault). Por eso la regla se lee sobre B26.

Medida, por conjunto y por brazo, en los hechos sobrescritos:
- Certeza = menos la pérdida base del objetivo real, en la dirección entrenada. Más certeza es menos pérdida.
- Corregido antes de ver datos: la salida no guarda las pérdidas base de los donantes, así que no hay margen.
- Plasticidad = subida de la pérdida del objetivo real en la dirección entrenada, menos la subida media de los no tocados.
- Estadístico = Spearman entre certeza y plasticidad. La unidad es el conjunto, con IC90 bootstrap.

Regla N2 (principal: brazo ida, dosis 10):
- Presente si la cota superior del IC90 queda debajo de 0 y la media es ≤ −0,2. Un hecho seguro cambia menos.
- Ausente si la cota inferior supera −0,1. Granite cambia igual un hecho seguro y uno dudoso.
- Si no, indeterminado.
- Informativo: brazo vuelta, dosis 40, y la misma medida para el daño colateral de la otra dirección.

Decisión que cambia:
- Ausente: el nivel 2 falta en Granite, ahora con dato preregistrado. La «plasticidad con puerta de duda» tiene un hueco que llenar.
- Presente: Granite ya protege lo seguro. La puerta de duda tendría que probar que mejora algo que ya existe.
Costo: 0 unidades. Usa la salida de B26.

## Predicciones antes de B26 (2026-11-13, nivel 4: intuición con retroalimentación)

Claude escribe estas probabilidades antes de ver datos. Se puntúan con Brier cuando llega el resultado.
- D40c: origen 0,35, entidad 0,25, mezcla 0,40.
- Razón: la correlación por entidad de B25 apoya la entidad, pero el ajuste fino de B24 separó fuerte.
- Puerta E1 pasa: 0,85. Los nombres reales son más largos que los inventados.
- N2 (brazo ida, dosis 10): ausente 0,65, indeterminado 0,30, presente 0,05.
- Razón de N2: B24 y B25 dieron rho cerca de 0, y el piso de pérdida empuja rho hacia arriba.

## B26, primer intento (2026-11-13, sesión b26ent, 5 minutos, cerca de 0,15 unidades)

La puerta E1 cortó. La etapa 1 paró a los 200 pasos con pérdida 0,239. Ganó en 0,94 de la ida y en 0,87 de la vuelta.
Causa probable: los nombres reales de organizaciones y obras son largos. La pérdida media baja antes de que la vuelta quede aprendida.
Corrección, escrita antes del segundo intento: la etapa 1 de B26 para con pérdida < 0,16, la misma con la que terminó B24.
Así las dos celdas parten de la misma profundidad. Las reglas D40c, D10c y N2 no cambian.
Puntaje de la predicción de E1: dije 0,85 y no pasó. Brier 0,72.

## Resultado de B26 (2026-11-13, sesión b26ent2, 18 minutos, cerca de 0,6 unidades)

Puerta E1: pasa. La etapa 1 paró a los 300 pasos con pérdida 0,114. Gana en 0,99 de la ida y en 0,96 de la vuelta.
Aprendizaje, P_ida y P_vuelta: ganan en las dos dosis.
D40c (regla principal): origen. R_ida vale +0,130 [+0,089; +0,165]. B24 daba 0,109 y B25 daba 0,533.
D10c (segunda regla): origen. R_ida vale +0,051 [+0,028; +0,073].
El brazo vuelta queda direccional en las dos dosis: 0,037 y 0,055.
En aciertos, la vuelta vieja gana en 0,94 de los sobrescritos en la dosis 10. Los no tocados ganan en 0,95.

N2 (principal: brazo ida, dosis 10): ausente. Spearman +0,005 [−0,096; +0,103].
Granite cambia igual un hecho seguro y uno dudoso. Las otras tres combinaciones también dan ausente.
Informativo: la certeza de la otra dirección la protege un poco del daño colateral, −0,217 [−0,276; −0,160].

Decisión según el prerregistro:
- Origen: la direccionalidad depende de cómo se aprendió el hecho, no de la entidad. El paper afirma eso.
- Lo aprendido por ajuste fino queda en entradas separadas, aun con entidades reales. Lo del preentrenamiento comparte más.
- N2 ausente: el nivel 2 falta en Granite. La puerta de duda (ticket 09) tiene un hueco que llenar.

Puntaje de las predicciones (Brier, menos es mejor):
- D40c: 0,645. Le di 0,35 a origen, y fue origen.
- N2: 0,215. Le di 0,65 a ausente, y fue ausente.
- E1 del primer intento: 0,72.

Puntos débiles:
- La etapa 1 de B26 terminó más profunda que la de B24: pérdida 0,114 contra 0,16.
- Aun así, B26 quedó direccional. La profundidad no explica la diferencia con B25.
- El origen sigue mezclado con la frecuencia: un hecho de preentrenamiento se vio muchas veces y en muchas formas.
