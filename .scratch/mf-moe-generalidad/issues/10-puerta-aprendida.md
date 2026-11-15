# B28: la puerta aprendida, ¿el modelo puede llevar su propia puerta de certeza?

Type: grilling
Status: opciones A y D cerradas en Granite (B28d, B28e y B28f, 2026-11-14); tope de 2 u agotado
Blocked by: 09

## Question

En B27c, una sonda externa lee la certeza del estado oculto, y esa lectura pesa la pérdida de cada ejemplo.
¿Qué forma de «aprendida» queremos probar, en qué modelo, con qué meta y con qué tope?
El usuario decide. El agente no responde estas preguntas.

## Lo que ya sabemos (ticket 09)

- Sin puerta, Granite cambia igual un hecho seguro y uno dudoso: N2 +0,003.
- El efecto de la puerta crece con su señal: azar 0, salida −0,16, sonda −0,27, oráculo −0,33.
- La sonda de B27c no ve el valor y logra 0,83 del oráculo. Es un nivel 3 que el modelo tiene y no usa.
- Ninguna puerta hace que un hecho seguro rechace una sola voz. Con pesos desde 0,02, el 85 % adopta lo nuevo.
- Con la sonda, el efecto se apaga en la dosis 160. Con el oráculo no se apaga, y falta explicarlo.
- El router de Vigía no lee la certeza de un hecho, en base ni en instruct.

## Antecedentes (Lacuna, 2026-11-13)

- [Learning to Reweight Examples](https://lacuna.tiptreesystems.com/work/learning-to-reweight-examples-for-robust-deep-learning/version/art_478e761cfa7b44c787c53e8a59b65c73) (ICML 2018) aprende el peso de cada ejemplo por metagradiente.
- Ese trabajo necesita un conjunto limpio de validación.
- El cluster [Bi-Level Optimization for Meta-Learned Sample Weighting](https://lacuna.tiptreesystems.com/direction/bi-level-optimization-for-meta-learned-sample-weighting-18962) junta 348 papers de pesos por ejemplo aprendidos.
- [Learning Loss for Active Learning](https://lacuna.tiptreesystems.com/work/learning-loss-for-active-learning/version/art_f47e459b21dc4f1c868ddf5c01e145de) (CVPR 2019) entrena una cabeza que predice la pérdida propia.
- Esa cabeza elige qué etiquetar, pero no regula cuánto aprende el modelo.
- [Titans](https://lacuna.tiptreesystems.com/work/titans-learning-to-memorize-at-test-time/version/art_543a483e95234b199fbe3cc56b3e78ee) (NeurIPS 2025) pesa la actualización de una memoria por su sorpresa, dentro de la arquitectura.
- No encontré un trabajo que use una cabeza de autoconocimiento del modelo como puerta de su propio ajuste de hechos.

## Opciones para «aprendida»

- A, cabeza heredada: una cabeza chica sobre el estado predice la pérdida propia, entrenada como pérdida auxiliar.
- A es B27c con la sonda dentro del modelo. El modelo la lleva consigo. Costo cercano a 1 unidad en Granite.
- B, peso metaaprendido: una red chica aprende el peso de cada ejemplo por metagradiente, como Learning to Reweight.
- B necesita un conjunto limpio y una optimización en dos niveles. Costo de 2 a 4 unidades en Granite.
- C, arquitectura propia: un modelo chico desde cero, de 10 a 30 M, con memoria pesada por sorpresa.
- C prueba si el nivel 2 aparece sin puerta externa. Es la más cara, con un costo a medir con un piloto.
- D, puerta en línea: el modelo usa su cabeza heredada para regular su paso durante un ajuste continuo, hecho por hecho.
- D necesita A primero.

## Preguntas para el usuario

1. ¿Cuál de las formas A, B, C o D cuenta como «el modelo aprende su propia puerta»?
2. ¿Medimos en Granite, que es barato y ya tiene banco, o en un modelo propio desde cero?
3. ¿La meta es el olvido selectivo (N2), o la conducta del video: rechazar una voz y aceptar cien?
4. Si la meta es rechazar una voz, ¿probamos también un mecanismo distinto de los pesos, como un umbral o fuentes independientes?
5. ¿Qué tope de unidades tiene este ticket?

## Puntos débiles a resolver antes de preregistrar

- La persistencia del oráculo en la dosis 160 no tiene explicación. Un reanálisis gratis de B27 puede mirarlo.
- La dosis repite la misma oración. Para la meta del video falta un banco con fuentes independientes (candidato N3).
- Todo corre en un modelo y un banco. Un resultado positivo necesita una réplica en otro banco.

## Respuestas del usuario (2026-11-14)

- Pregunta 1: la opción A.
- Pregunta 2: Granite.
- Pregunta 5: un tope de 2 unidades.
- Las preguntas 3 y 4 siguen abiertas. La opción A mide N2, así que la meta del video queda para después.

## Reanálisis gratis de la dosis 160 (B27, B27b y B27c)

- En azar, señal y sonda, el peso deja de importar en d160: rho(subida, peso) baja a +0,04, +0,10 y +0,04.
- Con el oráculo, el peso sigue importando en d160: +0,24. Con el oráculo fuerte: +0,22.
- En azar, la celda «peso bajo y hecho seguro» no se protege en d160: subida 7,39 contra 7,40.
- En la sonda, esa celda es la más protegida en d160: 7,03 contra 7,31 a 7,58.
- Lectura: la protección en dosis altas vive en la cola, con el peso más bajo sobre el hecho más seguro.
- Una señal con rho 0,6 diluye esa cola. Una puerta aprendida necesita una señal mejor para durar en d160.
- Script: el scratchpad de la sesión, `d160/re.py` y `d160/inter.py`.

## Prerregistro B28 (opción A, 2026-11-14)

Pregunta: ¿entrenar la cabeza de certeza dentro del modelo mejora su lectura de la duda en hechos no vistos?

Diseño (`colab/b28_cabeza.py`):

- La cabeza es lineal sobre el estado del último token de la pregunta, en las capas 8, 12, 16 y 20.
- La cabeza predice la NLL propia del valor real. El objetivo sale de la misma pasada, sin gradiente.
- Entrenan 204 candidatos fuera del banco. Validan 51, que eligen el punto de parada. El banco (206) solo prueba.
- Brazo congelada: solo la cabeza aprende, con AdamW, lr 1e-4, wd 1 y 300 pasos.
- Brazo conjunta: la cabeza y las capas 6-23 aprenden juntas, con lr 5e-5 en el tronco.
- La conjunta suma un KL contra el modelo de partida, con beta 1, sobre las oraciones de entrenamiento.
- Referencias sin decisión: la ridge antes y después del ajuste conjunto, con los mismos 255 hechos.

Piloto gratis (CPU, estados de VH): la cabeza congelada logra 0,48 en el banco y la ridge 0,54.
Elegí lr y wd mirando el banco del piloto. Ese sesgo afecta igual a los dos brazos.

Reglas de la fase 1:

- C1 (control): rho_banco(congelada) ≥ 0,40.
- I1 (integridad): Spearman de la NLL del banco antes y después ≥ 0,90, y cambio medio ≤ 0,15 nats.
- D1: Δ = rho_banco(conjunta) − rho_banco(congelada), con IC90 por bootstrap pareado de hechos.
- D1 «hereda» si Δ ≥ 0,05 y lo > 0. «No hereda» si hi < 0,05. Si no, «indeterminado».

Fase 2, solo si C1 pasa, I1 pasa y D1 da «hereda»:

- B27 sobre el modelo conjunto, con los brazos cabeza y oráculo, en los 8 conjuntos y las 3 dosis.
- H1: N2 de la cabeza en d10 presente, con una fracción del oráculo ≥ 0,83, el valor de la sonda.
- H2: N2 de la cabeza en d160 ≤ −0,10. La sonda dio −0,023.

Qué decide cada resultado:

- «Hereda» con H2: la puerta en línea (D) pasa a ser el siguiente candidato, con la cabeza heredada.
- «Hereda» sin H2: la herencia mejora la lectura, pero no alcanza para las dosis altas.
- «No hereda»: el modelo ya tiene la información, y meterla adentro no suma. A queda igual a B27c.
- «Indeterminado»: no corre la fase 2. El usuario decide si replica con más hechos.
- C1 falla: el entrenamiento de la cabeza falla. Se arregla en CPU y no se concluye nada.

Predicciones (Brier al cerrar): C1 pasa 0,85. D1 hereda 0,25, no hereda 0,45, indeterminado 0,30.
Dado «hereda»: H1 0,50 y H2 0,35.

Costo: la fase 1 sola cuesta cerca de 0,25 u. Con la fase 2, cerca de 1,5 u. Un timer corta a los 60 min, cerca de 1,9 u.

## Resultado B28 (2026-11-14, cerca de 0,2 u)

- C1 pasa: la cabeza congelada lee el banco con rho 0,41.
- I1 falla: el ajuste conjunto movió el banco 0,66 nats, con Spearman 0,56 contra el original.
- La cabeza conjunta da 0,10 en el banco, y D1 registra «no hereda» (Δ −0,315).
- D1 no cuenta, porque el tronco se dañó en los primeros 25 pasos, con picos de pérdida de 91.
- La ridge después del ajuste baja de 0,53 a 0,22: el tronco se volvió menos legible.
- Brier: C1 0,0225. D1 sin puntaje, porque I1 anula la comparación.

## Diagnóstico gratis (CPU, solo inferencia)

- Granite tiene dropout de atención de 0,1. `def_losses` y `prompt_states` dejan el modelo en modo train.
- Por eso la pasada de la cabeza corría con dropout, y la evaluación corría sin dropout.
- Con dropout, el estado de la pregunta cambia mucho: coseno mediano 0,90 y mínimo 0,25 contra la referencia.
- Sin dropout, el lote de oraciones enteras coincide con la referencia sin lote: coseno 1,000.
- El KL comparaba la referencia sin dropout con el modelo con dropout, y empujaba el tronco desde el paso 1.
- Además, la cabeza conjunta arrancaba al azar. [Kumar et al. (ICLR 2022)](https://lacuna.tiptreesystems.com/work/fine-tuning-can-distort-pretrained-features-and-underperform-5jw3/version/art_32d79521d80244638bc9ebe732727738) muestran que eso deforma los rasgos.
- B24 a B27 no tienen este problema: sus mediciones corren sin dropout, y su ajuste es igual en todos los brazos.

## Prerregistro B28b (2026-11-14)

Cambios contra B28, sin otros cambios:

- La pasada de la cabeza corre sin dropout en los dos brazos.
- La cabeza conjunta arranca de la cabeza congelada ya entrenada (LP-FT).

Las reglas C1, I1, D1, H1 y H2 y la fase 2 quedan iguales.
Predicciones: C1 pasa 0,90. I1 pasa 0,65. D1 hereda 0,20, no hereda 0,50, indeterminado 0,30.
Dado «hereda»: H1 0,50 y H2 0,35.
Tope: un timer de 50 min, cerca de 1,6 u. Con B28, el total queda por debajo de 2 u.

## Resultado B28b (2026-11-14, cerca de 0,2 u)

- C1 pasa: la cabeza congelada llega a 0,48 en el banco, igual que el piloto, y aprende de forma estable.
- I1 falla otra vez: el banco se movió 0,98 nats, con Spearman 0,40.
- La cabeza conjunta oscila, con picos de pérdida de 7,2, y da 0,34 en el banco. D1 no cuenta.
- El dropout explicaba la falla de la congelada, no la de la conjunta.
- Lectura: con el lr de edición (5e-5), el gradiente de la cabeza reescribe el conocimiento que la cabeza intenta leer.
- El KL solo cubre las oraciones de entrenamiento, y no protege los hechos del banco.
- Brier: C1 0,01, I1 0,4225. D1 sin puntaje.

## Prerregistro B28c (2026-11-14)

- Único cambio contra B28b: el tronco de la conjunta aprende con lr 5e-6, diez veces menos.
- Las reglas C1, I1, D1, H1 y H2 y la fase 2 quedan iguales.
- Predicciones: I1 pasa 0,60. D1 hereda 0,15, no hereda 0,55, indeterminado 0,30.
- Si I1 vuelve a fallar, la opción A queda sin medir en esta forma, y no corro otra variante sin el usuario.
- Tope: un timer de 48 min, cerca de 1,5 u. El total queda por debajo de 2 u.

## Resultado B28c (2026-11-14, cerca de 0,3 u)

- C1 falla: la cabeza congelada da 0,32 en el banco.
- I1 falla por poco: el banco se movió 0,161 nats (límite 0,15), con Spearman 0,909 (pasa).
- D1 indeterminado: Δ +0,07, IC90 [−0,0005; 0,14]. Conjunta 0,39 y congelada 0,32.
- Desde el paso 175, la conjunta queda por encima de la congelada en el banco: 0,41 a 0,44 contra 0,33 a 0,39.
- Brier: C1 0,81, I1 0,36, D1 0,815 (tres clases).

## Lectura de las tres corridas

- La cabeza congelada tiene el mismo código y las mismas semillas en B28b y B28c.
- Aun así dio 0,48 en B28b y 0,32 en B28c. El instrumento varía cerca de 0,16 entre corridas iguales.
- Ese ruido supera la ganancia de 0,05 que D1 busca. Con este diseño, D1 no se puede resolver.
- La ridge sobre los mismos rasgos da 0,53 sin ruido de optimización. La cabeza con AdamW y parada por val (51 hechos) agrega el ruido.
- En las tres corridas, la ridge después del ajuste conjunto lee peor que antes: 0,22, 0,36 y 0,38 contra 0,53.
- Lectura tentativa: el tronco se adapta a su cabeza, pero no se vuelve más legible en general.
- Con el lr de edición, el ajuste conjunto reescribe el conocimiento. Con un lr diez veces menor, casi lo conserva.
- Gasto de la opción A: cerca de 0,7 u del tope de 2.

## Acciones posibles (decide el usuario)

- Medir la cabeza con ridge cerrada en vez de AdamW, para quitar el ruido de optimización.
- Repetir la congelada 3 veces en una VM, para medir el ruido del instrumento (cerca de 0,2 u).
- Ampliar el conjunto de entrenamiento con más candidatos, para bajar el ruido de la parada por val.
- Probar la opción D, la puerta en línea, con la sonda ridge congelada en lugar de una cabeza heredada.
- Cerrar la opción A con este resultado y pasar a otra pregunta del mapa.

## Prerregistro B28d: medir con ridge cerrada (2026-11-14)

El usuario eligió la acción 1: medir las cabezas con ridge cerrada.

Diseño (`python -m colab.b28_cabeza d`):

- La medida es la ridge cerrada de B27c: entrena con los 255 candidatos fuera del banco y lee el banco.
- La ridge es determinista: dio 0,534 en B28, B28b y B28c.
- El tronco aprende contra una ridge por episodio, como [R2D2](https://lacuna.tiptreesystems.com/work/meta-learning-with-differentiable-closed-form-solvers/version/art_62af322aa9614aee80b00e9721e3b295) (Bertinetto et al., ICLR 2019).
- Cada lote de 16 se parte en 12 de soporte y 4 de consulta. El error de la consulta entrena el tronco.
- El alpha del episodio es el que la ridge eligió por LOO antes del ajuste.
- No hay cabeza que aprender, así que no hay cabeza aleatoria ni parada por val.
- El tronco usa lr 5e-6, KL con beta 1, sin dropout y 300 pasos.
- Cada 50 pasos mido cuánto se movió el banco. Uso el último punto que cumple I1.
- Esa elección usa las pérdidas del banco solo por integridad, nunca por la lectura.
- Corren dos réplicas del ajuste, con semillas de episodio distintas, desde el mismo modelo.

Reglas:

- C1: ridge antes ≥ 0,45.
- I1 por réplica: algún punto con Spearman ≥ 0,90 y cambio medio ≤ 0,15 nats.
- Δ por réplica: ridge después − ridge antes en el banco, cada modelo con su propia pérdida, con IC90 por bootstrap pareado.
- D1 «hereda» si las dos réplicas dan Δ ≥ 0,05 con lo > 0. «No hereda» si las dos dan hi < 0,05.
- D1 «sin medida» si alguna réplica falla I1. En otro caso, «indeterminado».
- La fase 2 corre solo con C1 y D1 «hereda»: un brazo «cabeza» con la ridge después de la réplica 0.
- La referencia de la fase 2 es el oráculo de B27 (−0,328 en d10), válida porque I1 se cumple.
- H1: N2 d10 presente con fracción ≥ 0,83. H2: N2 d160 ≤ −0,10.

Qué decide cada resultado:

- «Hereda»: entrenar con la herramienta vuelve al modelo más legible para sí mismo. La fase 2 dice si eso sirve como puerta.
- «No hereda»: el modelo no gana legibilidad a una fuerza que conserva el conocimiento. La opción A se cierra.
- «Indeterminado»: el ruido entre réplicas queda medido, y el usuario decide si vale más.
- «Sin medida»: el ajuste no conserva el conocimiento ni con lr 5e-6. La opción A se cierra en Granite.

Predicciones: C1 0,97. I1 en las dos réplicas 0,45. Dado I1: hereda 0,15, no hereda 0,45, indeterminado 0,40.
Costo: cerca de 0,35 u sin fase 2 y cerca de 1 u con ella. Timer de 40 min, cerca de 1,3 u, el resto del tope.

## Resultado B28d (2026-11-14, cerca de 0,3 u)

- C1 pasa: la ridge antes da 0,5325, casi igual a las corridas anteriores (0,534). El instrumento es estable.
- I1 falla en las dos réplicas desde el paso 50: el banco se movió 0,26 y 0,26 nats, con Spearman 0,83 y 0,84.
- El cambio no crece después: entre los pasos 50 y 300 queda entre 0,22 y 0,26 nats.
- El ajuste del episodio baja de 3,0 a 0,2: el tronco sí aprende a volverse legible para la ridge del lote.
- D1 da «sin medida». Por la regla prerregistrada, la opción A se cierra en Granite.
- El cambio no es ruido: en B27 las pérdidas base del banco se repitieron entre VMs con error menor a 0,01.
- Brier: C1 0,0009, I1 0,2025.
- Gasto total de la opción A: cerca de 1,0 u del tope de 2.

## Lectura de la opción A (B28 a B28d)

- Entrenar al modelo para leer su propia certeza reescribe el conocimiento que intenta leer.
- Pasó con una cabeza aleatoria, con LP-FT, con lr diez veces menor y con una ridge cerrada por episodio.
- El KL sobre las oraciones de entrenamiento no protege los hechos que no ve.
- La lectura externa no tiene ese costo: la ridge sobre el modelo congelado da 0,53 sin tocar nada.
- Para la tesis del informe, es un efecto observador: medir la certeza desde adentro cambia lo medido.
- Límite: probé un modelo, un banco, 255 hechos de entrenamiento y un solo tipo de preservación (KL).

## Control gratis: ¿el optimizador explica el cambio del banco? (2026-11-14, CPU)

- AdamW mueve cada parámetro cerca de lr por paso, aunque el gradiente sea casi cero.
- Sumé un ruido de signo aleatorio a los 963 M de parámetros entrenables, sin entrenar, y medí el banco.

| Amplitud | Pasos de AdamW equivalentes | Cambio medio | Spearman |
|---|---|---|---|
| 2,5e-5 | 5 | 0,020 | 0,999 |
| 1e-4 | 20 | 0,030 | 0,998 |
| 2,5e-4 | 50 | 0,053 | 0,994 |

- Con el desplazamiento máximo de 50 pasos, el ruido mueve el banco 0,053 nats. B28d lo movió 0,26, con Spearman 0,83.
- El cambio de B28d no viene del ruido del optimizador. Viene de la dirección del objetivo de lectura.
- La lectura del efecto observador resiste este control.
- En B28d el KL valió entre 0,05 y 0,10. El ajuste valió entre 0,2 y 3. El KL con beta 1 pesó poco.
- Script: el scratchpad de la sesión, `b28e/ruido.py`.

## Respuesta del usuario (2026-11-14)

- El usuario eligió las acciones 2, 3 y 4: la puerta en línea, una preservación más fuerte y otra pregunta del mapa.
- El tope de 2 unidades sigue. Quedan cerca de 1,0 u para B28e y B28f.

## Prerregistro B28e: la puerta en línea (opción D, 2026-11-14)

Diseño (`python -m colab.b27_puerta l`):

- Es B27c con un cambio: tras cada paso, la sonda relee el estado actual de los hechos que se sobrescriben.
- La sonda es la ridge de B27c, ajustada una vez sobre el modelo de partida, sin los hechos de cada conjunto.
- El peso es el rango de la lectura entre el banco. Los hechos no sobrescritos guardan su lectura base.
- En el paso 0, el peso es igual al de B27c. Los lotes y las semillas son los de B27c. La comparación es pareada.
- Dosis 10 y 160. La dosis 40 sale para entrar en el tope.
- Antecedente: [Titans](https://lacuna.tiptreesystems.com/work/titans-learning-to-memorize-at-test-time/version/art_543a483e95234b199fbe3cc56b3e78ee) pesa cada actualización de una memoria por su sorpresa en línea.

Reglas (la medida N2 y su regla son las de B26 y B27):

- Integridad: la pérdida base repite la de B27c con error < 0,01, y la sonda da Spearman 0,616 con la pérdida base.
- L2 (dosis 10): N2 de sonda_linea presente.
- L1 (dosis 160): N2 de sonda_linea ≤ −0,10 con la cota alta < 0. Entonces la puerta en línea dura.
- Informativo: la diferencia pareada por conjunto contra la sonda fija de B27c, con IC90 por bootstrap sobre conjuntos.
- Informativo: el Spearman entre el peso en la dosis 160 y la certeza base.

Qué decide cada resultado:

- L1 pasa: la puerta en línea dura donde la fija se apaga. El paper la suma, y el paso siguiente es la meta del video.
- L1 falla y L2 pasa: releer no agrega nada en dosis altas. Una puerta durable necesita mejor señal. La opción D se cierra.
- L2 falla: releer el modelo que cambia rompe la puerta aun con una voz. La opción D se cierra.

Mecanismo que espero: un hecho aprendido parece seguro y pierde peso. Eso iguala el aprendizaje y borra la cola protegida.
Predicciones: integridad 0,95. L2 pasa 0,55. L1 pasa 0,15.
Costo: cerca de 0,55 u. Timer de 21 minutos, cerca de 0,67 u.

## Prerregistro B28f: KL más fuerte con hechos ancla (2026-11-14)

Diseño (`python -m colab.b28_cabeza f`):

- Es B28d con tres cambios: beta 10, hechos ancla en el KL y 150 pasos.
- Beta 10 lleva el KL a la escala del ajuste.
- Las anclas son 512 hechos de `b23_hechos.json` fuera de los 461 candidatos y del banco. La lectura no los entrena.
- Cada lote suma 16 anclas al KL. El ajuste de la ridge usa solo los 16 hechos de entrenamiento.
- La fase 2 no corre: sale del tope de 2 unidades. Si D1 da «hereda», la decide el usuario.
- Antecedente: [LoRA Learns Less and Forgets Less](https://lacuna.tiptreesystems.com/work/lora-learns-less-and-forgets-less/version/art_4ecc350e107f466d97d2f19a3fddce56) (TMLR 2024) mide cuánto olvida un ajuste según cuánto restringe.

Reglas: C1, I1 y D1 de B28d, sin cambios.

Qué decide cada resultado:

- I1 falla en una réplica: el efecto observador resiste un KL diez veces más fuerte con anclas. El informe suma este control.
- I1 pasa y D1 «hereda»: la opción A se reabre. La fase 2 cuesta cerca de 0,6 u y necesita un tope nuevo.
- I1 pasa y D1 «no hereda»: con una fuerza que conserva el conocimiento, el modelo no se vuelve más legible. La opción A se cierra.
- I1 pasa y D1 «indeterminado»: el usuario decide si vale más.

Predicciones: C1 0,97. I1 en las dos réplicas 0,35. Dado I1: hereda 0,10, no hereda 0,60, indeterminado 0,30.
Costo: cerca de 0,25 u. Timer de 10 minutos, cerca de 0,32 u.

## Resultado B28f (2026-11-14, sesión b28fanc, 9 minutos, cerca de 0,28 u)

| Réplica | Paso guardado | Cambio del banco | Spearman | Ridge después | Δ [IC90] |
|---|---|---|---|---|---|
| 0 | 50 | 0,137 | 0,949 | 0,446 | −0,087 [−0,145; −0,028] |
| 1 | 150 | 0,149 | 0,937 | 0,378 | −0,155 [−0,220; −0,091] |

- C1 pasa: la ridge antes da 0,5325, igual que en B28d.
- I1 pasa en las dos réplicas, cerca del borde: el banco se movió entre 0,13 y 0,16 nats.
- D1 da «no hereda». Las dos réplicas leen peor después del ajuste, con la cota alta debajo de 0.
- El ajuste del episodio queda entre 0,12 y 0,58. El término beta · KL queda entre 0,27 y 0,60, en la misma escala.
- Según el prerregistro: con una fuerza que conserva el conocimiento, el modelo no se vuelve más legible. La opción A se cierra.
- Lectura: el tronco aprende a ajustar la ridge de cada lote, pero esa ganancia no llega a la ridge sobre los 255 hechos.
- El banco se movió 2,5 veces el piso de ruido del optimizador (0,053), aun con KL fuerte y anclas.
- Brier: C1 0,0009. I1 0,4225. D1 dado I1, 0,26.

## Resultado B28e (2026-11-14, sesión b28elin, 22 minutos, cerca de 0,70 u)

Integridad: la pérdida base repite la de B27c con error 0,0. La sonda da Spearman 0,616. El peso del paso 0 es el de B27c.

| Dosis | N2 en línea | N2 fija (B27c) | Diferencia pareada | rho(peso, pérdida base) | Pérdida de lo nuevo, en línea contra fija |
|---|---|---|---|---|---|
| 10 | −0,248 [−0,325; −0,168] presente | −0,271 | +0,023 [−0,012; +0,058] | 0,43 | 0,232 contra 0,308 |
| 160 | +0,004 [−0,122; +0,132] indeterminado | −0,023 | +0,027 [−0,076; +0,133] | 0,31 | 0,001 contra 0,006 |

- L2 pasa: con una voz, la puerta en línea crea el nivel 2, igual que la fija.
- L1 falla: en la dosis 160, la puerta en línea no dura.
- Según el prerregistro: releer no agrega nada en dosis altas. Una puerta durable necesita mejor señal. La opción D se cierra.
- El mecanismo esperado aparece: la relectura diluye la señal. rho(peso, pérdida base) baja de 0,61 a 0,43 y 0,31.
- La puerta en línea aprende lo nuevo más rápido: 0,232 contra 0,308 en la dosis 10. Eso iguala el aprendizaje.
- Brier: integridad 0,0025. L2 0,2025. L1 0,0225.
- Análisis: el scratchpad de la sesión, `b28e/analisis.py`. Salida en `b28e_linea_gpu/analisis.out`.

## Lectura de B28e y B28f

- Opción A: con un KL que conserva el conocimiento, el modelo entrenado para leerse se vuelve menos legible (B28f).
- Sin ese KL, el entrenamiento reescribe el conocimiento que lee (B28 a B28d). El ruido del optimizador no lo explica.
- Opción D: releer la certeza del modelo que cambia no hace durar la puerta. La relectura iguala el aprendizaje.
- En los dos casos, la lectura útil es la externa y fija: la ridge sobre el modelo de partida.
- Gasto del ticket: cerca de 1,98 u del tope de 2. El tope queda agotado.
- Límite: un modelo, un banco y una sola forma de releer (cada paso, con la sonda del modelo de partida).

## Respuestas del usuario a las preguntas 3 y 4 (2026-11-14)

- Pregunta 3: la meta es la conducta del video, rechazar una voz y aceptar cien.
- Pregunta 4: probamos fuentes independientes y un umbral de fuentes. El diseño sigue en el ticket 12 (B30).
- El usuario también pidió la réplica en otro banco (ticket 11, B29) y cerrar los tickets 03, 05 y 06.
- Tope: 2 unidades para los tickets 11 y 12 juntos.

## Comments
