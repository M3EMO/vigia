# B27 (propuesta): plasticidad con puerta de duda

Origen: el video «niveles de aprendizaje», nivel 2. Una creencia segura cambia menos que una dudosa.
Estado: B27c corrido (2026-11-13): la sonda del estado (nivel 3, sin etiqueta) crea el nivel 2 con 0,83 del efecto del oráculo.

## Pregunta
¿La duda de Vigía puede decidir cuánto cambia cada hecho durante un ajuste?
La meta es la conducta del video: el café cerrado se actualiza; la gravedad no se reescribe con una sola voz.

## Literatura y Lacuna (2026-11-13)
- UCB (Ebrahimi et al., ICLR 2020) escala el paso de cada peso por su incertidumbre bayesiana.
- EWC y SI protegen los pesos importantes. Son las versiones baratas que el video llama débiles.
- «Fine-Tuning with Uncertainty-Aware Priors» (Rudner et al., AISTATS 2025) usa priors inciertos en el ajuste fino.
- «Adaptive Chameleon or Stubborn Sloth» (ICLR 2024) mide conflictos en el contexto, no en los pesos.
- No encontré un trabajo que use la duda del router de un MoE como puerta de plasticidad por ejemplo.

## Diseño tentativo
- Señal: el puntaje F1 de Vigía sobre la respuesta vieja de cada hecho, antes del ajuste. Más duda es más plasticidad.
- Puerta: cada ejemplo pesa en la pérdida según su duda. Un brazo de control usa pesos al azar con la misma media.
- Control positivo: una puerta con la certeza real (la pérdida base) debe dar rho negativo en la medida N2.
- Banco: los 206 hechos reales de B25, que tienen certeza variada.
- Evidencia débil contra fuerte: la dosis 10 hace de una voz, y la dosis 160 hace de cien voces.

## Reglas tentativas (se preregistran después de N2)
- Éxito: la puerta da rho de N2 ≤ −0,2 con IC90 debajo de 0, y el azar no.
- Costo aceptable: los hechos dudosos siguen aprendiendo lo nuevo en ≥ 90 % de las corridas.
- Riesgo: la puerta puede bajar la plasticidad en general. Por eso el control al azar tiene la misma media.

## Decisión que cambia
- Funciona: Vigía pasa de detector a regulador del aprendizaje. El paper une el candidato 1 con Vigía.
- No funciona: Vigía queda como detector. La puerta de duda se descarta en esta forma.

## Costo estimado
Cerca de 1 unidad. La puntuación de Vigía en 206 hechos cuesta segundos en una L4.

## Punto débil encontrado antes del prerregistro (2026-11-13)
- El detector guardado de Vigía se calibró en Granite instruct. Nuestros experimentos usan Granite base.
- El detector guardado también usa features de salida, como `out_nll`. Con teacher forcing, `out_nll` es la pérdida del objetivo.
- Entonces el detector completo copiaría a la puerta oráculo. La puerta de B27 debe usar solo router y uso (F1 y U).
- La señal de B27 es el detector guardado con las features de salida en NaN. El detector las imputa con su media.
- Shortcut: no es un detector entrenado solo con router. Si la puerta V0 falla por poco, conviene reentrenarlo.

## Puerta V0 (preregistrada antes de ver datos, 2026-11-13)
Pregunta: ¿la duda del router sigue la certeza real en Granite base?
- Datos: los 461 candidatos de B25, con teacher forcing de la ida «The {p} of {h} is» + « {v}.».
- Certeza real: la NLL media de los tokens del valor.
- Duda del router: el máximo por token del detector sin features de salida, sobre los tokens del valor.
- Referencia: el máximo de `out_entropy` sobre los mismos tokens, que es la señal barata de salida.
- Pasa: Spearman(duda del router, NLL) ≥ 0,3 con IC90 por bootstrap sobre hechos que no baja de 0,1.
- Informativo: AUROC de cada señal para estar en el banco estricto, entre los 379 candidatos con donantes.
- Costo: inferencia en CPU con bf16, sin entrenamiento, unos minutos. No gasta unidades.
- Pasa: B27 usa la duda del router como puerta, y suma un brazo con la puerta de `out_entropy`.
- No pasa: B27 no usa a Vigía en esta forma. Las opciones son reentrenar el detector en base o usar la puerta de salida.

## Corrección tras leer el video completo (2026-11-13)
- El diseño tentativo decía que la dosis 160 hace de cien voces. El video dice otra cosa.
- Cien voces que repiten la misma fuente valen poco. Muchas fuentes independientes valen mucho.
- La dosis repite la misma oración, así que es el caso de una sola fuente repetida.
- Candidato N3, sin preregistrar: a igual cantidad de pasos, ¿las paráfrasis cambian más un hecho que la repetición?
- N3 mediría la otra mitad del nivel 2: la fuerza de la evidencia, además de la certeza previa.

## Resultado de V0 (2026-11-13, CPU local, 20 minutos, sin unidades)
- V0 no pasa. Spearman(duda del router, NLL) vale +0,079 [+0,001; +0,155], con 461 hechos.
- El puntaje del router queda saturado en el modelo base: la mediana es 0,997 y el percentil 5 es 0,960.
- El detector completo da +0,372 [+0,299; +0,444], pero usa `out_nll`, que es casi la NLL misma.
- La entropía de salida da +0,227 [+0,151; +0,298].
- AUROC para estar fuera del banco estricto, entre 379: NLL 0,699, router 0,426, entropía de salida 0,443.
- Lectura: el detector de instruct no transfiere a base en esta tarea. Falta saber si el router de base guarda la certeza.
- Archivos: `v0.py`, `v0_rows.json` y `v0_res.json` en el scratchpad de la sesión.

## Puerta V1 (preregistrada antes de ver datos, 2026-11-13)
Pregunta: ¿el router de Granite base guarda la certeza de un hecho, aunque el detector de instruct no la lea?
- Features: las F1 y U por capa del extractor de Vigía, promediadas sobre los tokens del valor. Sin features de salida.
- Sonda: regresión logística estandarizada, C = 0,1, con validación cruzada de 5 particiones estratificadas por hecho.
- Etiqueta: estar fuera del banco estricto, entre los 379 candidatos con donantes.
- Pasa: AUROC fuera de partición ≥ 0,65, con IC90 por bootstrap que no baja de 0,55.
- Informativo: Spearman entre la probabilidad de la sonda y la NLL, y la misma sonda con features de salida.
- Pasa: B27 suma un brazo con la sonda del router, ajustada en particiones que no incluyen al hecho.
- No pasa: B27 corre solo con la puerta oráculo, la de salida y el azar. Vigía queda como detector.
- Orden nuevo de B27: la puerta oráculo es el control positivo y corre primero. Sin efecto con el oráculo, ninguna puerta real sirve.

## Resultado de V1 (2026-11-13, CPU local, 20 minutos, sin unidades)
- V1 no pasa. La sonda solo con router da AUROC 0,591 [0,541; 0,637], con 379 hechos y 216 features.
- El router de base guarda algo de la certeza, porque el IC90 queda sobre 0,5. La señal es débil.
- Spearman entre la sonda del router y la NLL: +0,136.
- La misma sonda con 4 features de salida sube a 0,67 [0,623; 0,716]. La NLL sola da 0,699.
- Decisión según el prerregistro: B27 corre con la puerta oráculo, la puerta de salida y el azar. Vigía queda como detector.
- Pregunta abierta: la puerta de salida usa la entropía, que en V0 dio rho +0,227. Conviene preregistrar cuál señal de salida usar.

## B27 con la puerta oráculo, preregistrado (2026-11-13, antes de escribir el código y de ver datos)
Pregunta: ¿pesar la pérdida de cada ejemplo por su duda real crea el nivel 2 en Granite?
La puerta oráculo es el techo de cualquier señal real. Si el oráculo falla, ninguna señal sin etiqueta sirve.

Diseño:
- Banco: los 206 hechos de B25, leídos del JSON de B25. No se vuelve a filtrar.
- Conjuntos, semillas de lotes y dosis: los de B25 (tag «b25», dosis 10, 40 y 160). Solo el brazo ida.
- Duda real: la pérdida base de la ida, medida en la GPU antes de entrenar.
- Peso del brazo oráculo: w = 0,25 + 1,5 · rango(duda) / (n − 1). La media vale 1 sobre el banco.
- Brazo azar: los mismos pesos, permutados entre hechos con una semilla fija.
- Brazo uniforme: solo el conjunto 0, como chequeo de réplica contra B25. El resto de la referencia uniforme es B25.
- Pérdida con pesos: la entropía cruzada por token, por el peso de su ejemplo, sumada y dividida por los tokens válidos.
- Con todos los pesos en 1, esa pérdida es la de B25. Un test lo verifica en CPU.
- Medida nueva: la pérdida del valor nuevo por hecho sobrescrito. Aprende si el valor nuevo pierde menos que el viejo.

Medida N2: Spearman por conjunto entre la certeza (menos la pérdida base) y la subida de la ida en los sobrescritos.
Se promedia sobre 8 conjuntos, con IC90 por bootstrap sobre conjuntos. Regla de N2: presente si la cota alta < 0 y la media ≤ −0,2.
Ausente si la cota baja > −0,1. Indeterminado en otro caso.

Reglas:
- G1 (principal, dosis 10): N2 presente en el oráculo, y no presente en el azar. Entonces la puerta crea el nivel 2.
- G2 (cien voces, dosis 160): la media de N2 del oráculo sube hacia 0 respecto de la dosis 10.
- G2 también pide que el tercio más seguro aprenda lo nuevo en ≥ 90 % de los hechos en la dosis 160.
- G3 (informativo): la subida media del resto en el oráculo contra el azar, en cada dosis.
- Integridad: la pérdida base coincide con la de B25 con error medio < 0,01.
- Integridad: el uniforme del conjunto 0 en la dosis 10 repite la subida de B25 con error medio < 0,01.

Decisión que cambia:
- G1 falla: pesar la pérdida por certeza no crea el nivel 2 en este régimen. La familia B27 se cierra.
- G1 pasa y G2 pasa: el paso siguiente es una señal sin etiqueta, preregistrada antes de correr.
- G1 pasa y G2 falla: la puerta vuelve rígidos a los hechos seguros. El paso siguiente es otra forma de peso.

Predicciones (nivel 4, Brier al final):
- G1 pasa: 0,75. El peso actúa directo sobre el gradiente de cada ejemplo, aun con Adam.
- G2 pasa, dado G1: 0,5. Con pesos de 0,25, 160 pasos pueden no alcanzar para el tercio seguro.
- Integridad completa: 0,85.

Costo: dos brazos completos por 8 conjuntos y una corrida extra, como B25. Cerca de 1,2 unidades.

## Resultado de B27 con la puerta oráculo (2026-11-13, sesión b27pta, 37 minutos, cerca de 1,2 unidades)
Integridad: la pérdida base y el uniforme del conjunto 0 repiten B25 con error 0,0. La réplica es exacta.
- El script marcó la integridad como fallida por un error: `0.0 or 1` vale 1. Lo corregí y la integridad pasa.

| Dosis | N2 oráculo | N2 azar | N2 uniforme (B25) |
|---|---|---|---|
| 10 | −0,328 [−0,382; −0,273] presente | +0,038 [−0,093; +0,185] ausente | +0,003 ausente |
| 40 | −0,261 [−0,352; −0,174] presente | +0,005 indeterminado | −0,020 indeterminado |
| 160 | −0,243 [−0,359; −0,134] presente | +0,015 ausente | +0,093 ausente |

- G1 pasa: la puerta oráculo crea el nivel 2 en la medida N2, y el azar no.
- G2 pasa: N2 sube hacia 0 en la dosis 160, y el tercio seguro aprende lo nuevo en 100 %.
- G3 (informativo): el resto sube 3,73 nats con el oráculo y 4,11 con el azar en la dosis 160. La puerta baja el daño colateral.

Informativo, calculado después del análisis preregistrado:
- En la dosis 10, la respuesta vieja del tercio seguro sube 2,19 nats con el oráculo y 2,92 con el azar.
- Aun así, lo nuevo le gana a lo viejo por 2,78 nats en el tercio seguro. El 97 % lo adopta en la dosis 10.
- Lectura: la puerta cambia cuánto se borra lo viejo, pero no hace que un hecho seguro rechace una sola voz.
- La regla G2 resultó floja: casi todo aprende ya en la dosis 10, así que «cien voces convencen» pasa sin probar nada.

Decisión según el prerregistro:
- G1 y G2 pasan: el paso siguiente es una señal sin etiqueta, preregistrada antes de correr.
- Punto débil para el paso siguiente: medir también si un hecho seguro resiste, con el margen entre lo nuevo y lo viejo.
- La conducta del video, «la gravedad no cambia con una voz», no apareció. Pesos de 0,25 no alcanzan contra este ajuste.

Puntaje de las predicciones (Brier):
- G1: 0,0625. Le di 0,75, y pasó.
- G2: 0,25. Le di 0,5, y pasó.
- Integridad: 0,0225. Le di 0,85, y pasó.

## B27b: señal sin etiqueta y oráculo fuerte, preregistrado (2026-11-13, antes de escribir el código y de ver datos)
Dos preguntas, en una sola corrida:
- ¿Una señal sin etiqueta, sacada de la salida del modelo, crea el nivel 2 como el oráculo?
- ¿Un peso más extremo hace que un hecho seguro rechace una sola voz y acepte cien?

Señal sin etiqueta: la probabilidad máxima del siguiente token al final de «The {p} of {h} is».
- Usa solo la pregunta. No conoce el valor viejo ni el nuevo.
- Más probabilidad es más certeza. La duda es menos esa probabilidad.

Brazos nuevos, con banco, conjuntos, lotes y dosis de B27:
- senal: w = 0,25 + 1,5 · rango(duda de la señal) / (n − 1), la misma forma que el oráculo de B27.
- oraculo_fuerte: la duda real, con w = 0,02 + 1,96 · rango / (n − 1). La media sigue en 1.
- Referencias: los brazos oráculo y azar de B27. Comparten semillas y lotes, así que la comparación es pareada.

Medidas nuevas:
- Spearman entre la señal y la pérdida base, sobre los 206 hechos. Es informativo y muestra cuánto sigue la señal a la certeza.
- Margen por hecho sobrescrito: la pérdida del valor nuevo menos la pérdida actual del viejo. Positivo es que lo viejo resiste.

Reglas:
- S1 (principal, dosis 10): N2 del brazo senal presente, con la regla de N2 de B26.
- S1 informa también la fracción del efecto del oráculo: N2 de senal dividido N2 del oráculo de B27.
- R1 (una voz contra cien, oraculo_fuerte): el tercio más seguro resiste en ≥ 50 % de los hechos en la dosis 10.
- R1 también pide que el mismo tercio adopte lo nuevo en ≥ 90 % de los hechos en la dosis 160.
- R1b (costo): en oraculo_fuerte, el tercio más dudoso adopta lo nuevo en ≥ 90 % en la dosis 10.
- Integridad: la pérdida base coincide con la de B27 con error medio < 0,01.

Decisión que cambia:
- S1 pasa: la confianza propia de la salida alcanza para el nivel 2. El paper suma ese resultado, y Vigía queda como detector.
- S1 falla: el nivel 2 necesita una señal de certeza mejor. El paso siguiente es una sonda entrenada sobre el estado oculto (nivel 3).
- R1 pasa: pesar la pérdida reproduce la conducta del video. R1 falla: pesar la pérdida solo modula el olvido.
- Aviso: la dosis repite la misma oración. R1 mide repetición, no fuentes independientes.

Predicciones (Brier al final):
- S1 pasa: 0,45. La entropía de salida siguió a la pérdida con rho de solo +0,227 en V0.
- R1 pasa: 0,4. Con 10 pasos y peso 0,02, el paso efectivo del tercio seguro es chico, pero este ajuste es fuerte.
- R1b pasa: 0,9.

Costo: dos brazos por 8 conjuntos, como B27. Cerca de 1,2 unidades.

## Candidatos siguientes, sin preregistrar (2026-11-13, a pedido del usuario)
- V0i: repetir V0 con Granite instruct, el modelo donde se calibró Vigía. Separa el problema del modelo del problema de la tarea.
- V0i necesita bajar Granite instruct (cerca de 2,7 GB) a la PC, o armar un paquete de Vigía para Colab.
- Vigía antes y después de una edición: ¿el router marca más duda en las respuestas sobre hechos reescritos?
- Esa prueba mide si el nivel 3 detecta cambios del nivel 1.
- Nivel 3 alimenta al nivel 2: una sonda sobre el estado oculto predice la pérdida, y esa predicción pesa la pérdida.
- Modelo chico desde cero (10 a 30 M) con memoria pesada por sorpresa, al estilo Titans, sobre hechos sintéticos.
- La prueba sería si el modelo aprende solo la conducta del nivel 2: fuentes repetidas contra fuentes independientes.
- Costo de referencia: el piloto midió cerca de 14,5 unidades por corrida desde cero a 80-100 M.

## V0i: V0 y V1 sobre Granite instruct, preregistrado (2026-11-13, antes de ver datos)
Pregunta: ¿el fallo de V0 vino del cambio de modelo, o la duda del router no lee la certeza de un hecho?
- Modelo: Granite 3.1 1B-A400M instruct, donde se calibró el detector de Vigía. Corre en una L4 de Colab.
- Todo lo demás es igual a V0 y V1: los 461 candidatos de B25, las mismas indicaciones sin plantilla de chat y las mismas medidas.
- V0i pasa: Spearman(duda del router, NLL) ≥ 0,3, con IC90 que no baja de 0,1.
- V1i pasa: la sonda solo con router da AUROC ≥ 0,65, con IC90 que no baja de 0,55.
- La etiqueta del banco viene de Granite base. En instruct, saber un hecho puede cambiar, y eso agrega ruido.
- Las indicaciones sin plantilla de chat se alejan de los datos de calibración de Vigía. Es un límite conocido.

Decisión que cambia:
- V0i pasa: Vigía puede regular en instruct. El paso siguiente es preregistrar B27 sobre instruct con la puerta de Vigía.
- V0i falla y V1i pasa: el router de instruct guarda la certeza, pero hace falta una sonda propia de hechos.
- Los dos fallan: el router no lee la certeza de un hecho ni en su modelo. Vigía queda como detector, y el paper lo dice.

Predicciones: V0i pasa 0,3; V1i pasa 0,35.
Costo: una sesión L4 corta, cerca de 0,15 unidades.

## Resultado de V0i (2026-11-13, sesión v0ipta, cerca de 8 minutos, cerca de 0,15 unidades)
- V0i no pasa. En instruct, la duda del router da Spearman +0,059 [−0,019; +0,134] con la NLL. En base daba +0,079.
- V1i no pasa. La sonda solo con router da AUROC 0,598 [0,550; 0,645]. En base daba 0,591.
- El detector completo da +0,421, y la entropía de salida da +0,201. La NLL separa el banco con AUROC 0,671.
- Lectura: el modelo no explicaba el fallo de V0. El router no lee la certeza de un hecho, ni en el modelo donde se calibró Vigía.
- Decisión según el prerregistro: Vigía queda como detector. El paper dice que la duda del router no sirve como puerta de certeza.
- Límite: las indicaciones no usan la plantilla de chat, y Vigía se calibró con respuestas de chat.
- Brier: V0i 0,09 (le di 0,3 a pasar); V1i 0,1225 (le di 0,35).

## Resultado de B27b (2026-11-13, sesión b27bsg, 35 minutos, cerca de 1,2 unidades)
Integridad: la pérdida base repite la de B27 con error 0,0. La señal sigue a la certeza con Spearman +0,317.

| Dosis | N2 senal | N2 oraculo_fuerte | N2 oráculo (B27) | N2 azar (B27) |
|---|---|---|---|---|
| 10 | −0,160 [−0,228; −0,101] indeterminado | −0,389 [−0,460; −0,320] presente | −0,328 presente | +0,038 ausente |
| 40 | −0,114 indeterminado | −0,200 presente | −0,261 presente | +0,005 indeterminado |
| 160 | +0,056 ausente | −0,217 presente | −0,243 presente | +0,015 ausente |

- S1 falla: la señal sin etiqueta da N2 indeterminado. Logra 0,49 del efecto del oráculo, cerca de su Spearman con la certeza.
- R1 falla: con pesos de 0,02, el tercio seguro igual adopta lo nuevo en 85 % de los hechos en la dosis 10.
- R1b pasa: el tercio dudoso adopta lo nuevo en 100 % en la dosis 10.

Decisión según el prerregistro:
- S1 falla: el nivel 2 necesita una señal de certeza mejor. El paso siguiente es una sonda sobre el estado oculto.
- R1 falla: pesar la pérdida solo modula cuánto se olvida. No logra que un hecho seguro rechace una voz.
- El efecto de la puerta crece con la calidad de la señal: azar 0, señal propia la mitad, oráculo entero.

Brier: S1 0,2025 (le di 0,45 a pasar); R1 0,16 (le di 0,4); R1b 0,01 (le di 0,9).

## VH: sonda sobre el estado oculto, preregistrada (2026-11-13, antes de ver datos)
Pregunta: ¿el estado oculto de Granite base, leído solo con la pregunta, predice la certeza mejor que la confianza de salida?
- Datos: los 461 candidatos de B25, en CPU con bf16. Sin entrenamiento del modelo y sin unidades.
- Features: el estado oculto del último token de «The {p} of {h} is», en las capas 8, 12, 16 y 20, concatenado.
- Ese estado es causal: no ve el valor. La sonda queda sin etiqueta al usarla.
- Objetivo: la NLL media del valor real con teacher forcing.
- Sonda: regresión ridge con alpha elegido por validación interna, fuera de partición con 5 particiones.
- Referencia en la misma pasada: la confianza de salida de B27b, la probabilidad máxima del primer token.
- Pasa: Spearman(predicción, NLL) fuera de partición ≥ 0,45, con IC90 que no baja de 0,35.
- El umbral pide superar con claridad a la confianza de salida, que dio +0,317 sobre el banco.

Decisión que cambia:
- Pasa: preregistrar B27c con la puerta de la sonda, ajustada sin el conjunto que se sobrescribe.
- Falla: en Granite, una señal propia da a lo sumo la mitad del efecto del oráculo. El paper informa ese techo.

Predicción: pasa 0,55. Las sondas lineales sobre el estado suelen leer bien si el modelo sabe un hecho.

## Resultado de VH (2026-11-13, CPU local, 42 minutos, sin unidades)
- VH pasa. La sonda sobre el estado da Spearman +0,493 [+0,429; +0,556] con la NLL, sobre 461 candidatos.
- La confianza de salida da +0,173 [+0,098; +0,248] en los mismos 461.
- Informativo: dentro del banco de 206, la sonda da +0,622 y la confianza de salida +0,326.
- Decisión según el prerregistro: preregistrar B27c con la puerta de la sonda.
- Brier: 0,2025 (le di 0,55 a pasar, y pasó).

## B27c: la sonda del estado como puerta, preregistrado (2026-11-13, antes de escribir el código y de ver datos)
Pregunta: ¿una sonda de autoconocimiento (nivel 3), sin etiqueta al usarla, crea el nivel 2 casi como el oráculo?
- La sonda y el objetivo se calculan en la GPU, con el modelo de la corrida, sobre los 461 candidatos.
- Features: el estado del último token de la pregunta, en las capas 8, 12, 16 y 20. Objetivo: la NLL del valor real.
- Ajuste cruzado: la predicción de cada conjunto sale de una sonda ajustada sin los hechos de ese conjunto.
- Los 6 hechos que nunca se sobrescriben forman un noveno grupo, con el mismo trato.
- Peso: w = 0,25 + 1,5 · rango(predicción) / (n − 1), la misma forma que el oráculo y la señal.
- Un solo brazo nuevo: sonda. Referencias pareadas: oráculo y azar de B27, y senal de B27b.

Reglas:
- P1 (principal, dosis 10): N2 del brazo sonda presente.
- P2: la sonda logra ≥ 0,75 del efecto del oráculo en la dosis 10.
- Integridad: la pérdida base repite la de B27 con error < 0,01.
- Integridad: la predicción cruzada sigue a la pérdida base del banco con Spearman ≥ 0,5.

Decisión que cambia:
- P1 pasa: un nivel 3 sin etiqueta alimenta al nivel 2. El paper suma ese resultado como enlace entre niveles.
- Con P1, el paso siguiente es probar si el modelo puede aprender su propia puerta.
- P1 falla: el ajuste cruzado o la calidad de la señal no alcanzan. El nivel 2 por pesos queda en la mitad del oráculo.

Predicciones: P1 pasa 0,7; P2 pasa 0,5.
Costo: un brazo por 8 conjuntos. Cerca de 0,6 unidades.

## Resultado de B27c (2026-11-13, sesión b27csd, 19 minutos, cerca de 0,6 unidades)
Integridad: la pérdida base repite la de B27 con error 0,0. La predicción cruzada sigue a la pérdida base con Spearman +0,616.

| Dosis | N2 sonda | N2 oráculo (B27) | N2 señal de salida (B27b) | N2 azar (B27) |
|---|---|---|---|---|
| 10 | −0,271 [−0,330; −0,216] presente | −0,328 presente | −0,160 indeterminado | +0,038 ausente |
| 40 | −0,161 [−0,284; −0,064] indeterminado | −0,261 presente | −0,114 indeterminado | +0,005 indeterminado |
| 160 | −0,023 [−0,063; +0,016] ausente | −0,243 presente | +0,056 ausente | +0,015 ausente |

- P1 pasa: la puerta de la sonda crea el nivel 2 en la dosis 10.
- P2 pasa: la sonda logra 0,826 del efecto del oráculo.
- Decisión según el prerregistro: un nivel 3 sin etiqueta alimenta al nivel 2. El paper suma ese resultado.
- El paso siguiente es probar si el modelo puede aprender su propia puerta.

Informativo:
- El efecto crece con la calidad de la señal: azar 0, salida −0,16, sonda −0,27, oráculo −0,33.
- Con la sonda, el efecto se apaga con la dosis: −0,27, −0,16 y −0,02. Con el oráculo se mantiene.
- Ese patrón se parece al del video: una voz cambia poco lo seguro, y cien voces lo cambian todo.
- No sé por qué el oráculo mantiene el efecto en la dosis 160. Falta explicarlo antes de leer el patrón como conducta.

Brier: P1 0,09 (le di 0,7 a pasar); P2 0,25 (le di 0,5).
