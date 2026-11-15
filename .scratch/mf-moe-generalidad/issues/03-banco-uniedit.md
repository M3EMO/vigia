# Banco con UniEdit

Type: research
Status: closed
Blocked by: 01

## Question

¿UniEdit (311.000 casos de Wikidata, 25 dominios, con pruebas de localidad) sirve para armar el banco de conflicto con otra ontología? Revisar licencia y formato. Decidir cómo se traduce a hechos con conflicto real, y si sus dominios forman temas y supertemas.

Datos: https://huggingface.co/datasets/qizhou/UniEdit

## Viabilidad (2026-11-06, sin gasto de unidades)

- UniEdit tiene 311K filas, licencia MIT, 5 sectores y 25 dominios, sin par viejo/nuevo.
- El conflicto se construye con otra fila de igual property.id y mismo dominio.
- El puente con B20 mide también head_entity.description en cada corrida.
- Falta ver filas reales, el rendimiento del filtro y cuántos hechos conoce Granite.

Detalle: nota del vault "Viabilidad de ampliar mf-moe: OLMoE, FLAME-MoE, UniEdit y frecuencia de corpus".

## Filas reales (2026-11-08, sin gasto de unidades)

Muestra: primeros 512 KB de los 50 archivos (25 dominios × train/test), 2.172 registros, sin aleatorizar.

- La vista de datasets-server falla (500 al convertir a Arrow). Se leen los JSON crudos: un objeto por archivo, 3,76 GB en total.
- Cada registro tiene `edit`, `generality` y `locality`, con un caso de cada uno. Total estimado: 312K filas.
- `target` es el valor real de Wikidata, con alias entre paréntesis. El banco debe usar la etiqueta sola.
- Pasan `wikibase-item` con `single_value` 59,8 % (1.299 de 2.172).
- Tienen donante de igual propiedad y dominio en esta muestra 404 de 1.299. Es una cota inferior.
- Propiedades dominantes: country 5,7 %, subclass of 3,5 %, part of 2,8 %.
- El prompt mide 18 palabras de mediana y trae la descripción de la cabeza.
- El valor aparece literal en 2 prompts. Comparte una palabra con el prompt en 251 (19 %).
- Coherencia de dominio despareja: biology y mathematics tienen léxico propio; agronomy y sports science casi no.
- Locality: property of edit 363, tail of edit 349, none 323, head of edit 264.

Puntos débiles para el diseño:

1. El dominio puede no formar tema. El gradiente por tema de B20 venía de WordNet. Hay que medir la coherencia antes de preregistrar.
2. La descripción del prompt puede contradecir al donante. Conviene entrenar con plantilla corta.
3. Falta medir cuántos hechos conoce Granite. Sin hecho conocido no hay conflicto.

Siguiente paso: medir en CPU la pérdida del valor real contra la del donante, sin entrenar.

## Cuántos hechos conoce Granite (2026-11-08, CPU, sin gasto de unidades)

Reglas fijadas antes de ver pérdidas. 755 hechos con 4 donantes, Granite en bf16. Salida: `mf-moe/.cache/runs/bloques/uniedit_conoce/`.

- Conocidos 42,9 %. Piso con cabeza ajena 17,4 %. Exceso +25,6 puntos, IC90 [+22,8; +28,2].
- Banco viable: sí. El exceso supera 15 puntos y los 25 dominios tienen 412 o más hechos reales estimados, contra 124 necesarios.
- Dominio como tema: no. kNN normalizado 0,050 contra 0,554 de los temas de B20.
- El prompt largo de UniEdit sube el acierto por pares de 64,6 % a 79,1 %. Confirma el aporte de la descripción: se entrena con plantilla corta.
- El exceso cambia por propiedad: ubicación administrativa 0,44; subclass of 0,41; part of 0,38; country 0,24; employer 0,11; occupation 0,00.

Decisión: UniEdit sirve como fuente de hechos conocidos, pero el dominio no sirve como tema. El tema necesita otra definición antes de preregistrar.

Límites: la muestra no es aleatoria, exigir 4 donantes favorece propiedades frecuentes, y 10 temas contra 25 dominios no es la misma tarea de kNN.

## P31 y relación como tema (2026-11-09, CPU, sin gasto de unidades)

Reglas fijadas antes de calcular kNN. Tarea: 10 clases balanceadas, 50 sorteos, k=5, Granite en bf16. WordNet usa igual n y k. Salida: `mf-moe/.cache/runs/bloques/uniedit_tema/`.

| Definición | n | kNN Granite | kNN WordNet | Control léxico | Clases con ≥ 124 | Veredicto |
|---|---|---|---|---|---|---|
| `P31`, nombre | 13 | 0,379 [0,320; 0,427] | 0,445 | 0,269 | 8 de 10 | válido |
| `property.id`, prompt | 18 | 0,910 [0,889; 0,929] | 0,468 | 0,804 | 9 de 10 | válido |

Las permutaciones dan cerca de 0. El léxico de WordNet da 0,133.

Por la regla previa, `P31` gana porque usa la misma sonda que WordNet.

Puntos débiles (análisis exploratorio, posterior a la regla):

1. La ganancia de `P31` sobre su control léxico es 0,110. La de WordNet es 0,312. `P31` tiene un 35 % de esa ganancia.
2. Cuatro clases llevan el resultado: proteína 0,81, persona 0,78, taxón 0,70, empresa 0,58.
3. Las clases de títulos se mezclan: edición 0,06, obra literaria 0,21, sencillo 0,25, película 0,31, episodio 0,33.
4. Tres de las 8 clases con hechos suficientes se estiman con menos de 10 hechos. Con 10 hechos o más quedan 5 clases, y la regla pide 6.
5. La relación se mide con otra sonda que WordNet. Su kNN no compara el mismo objeto.
6. La muestra sale del comienzo de cada archivo.

Decisión: `P31` queda como definición elegida, pero sin 10 temas separables. El modelo distingue unos 4 grupos: persona, biología, organización y obra. El preregistro necesita grupos gruesos definidos antes de medir, más filas de ventanas al azar, y el gradiente de daño real como prueba final.

## Ocho tipos gruesos como tema (2026-11-09, CPU, sin gasto de unidades)

Datos: 505 ventanas al azar, 10.177 registros, 6.069 pasan el filtro. Se excluyen las cabezas de la muestra anterior. Reglas fijadas antes de medir. Salida: `mf-moe/.cache/runs/bloques/uniedit_tipos/` y `uniedit_ventanas/`.

| Condición | Resultado | Umbral | Cumple |
|---|---|---|---|
| kNN de tipos, 8 clases, n=60 | 0,583 [0,540; 0,620] | ≥ 0,288 (mitad de WordNet 0,576) | sí |
| Ganancia sobre el control léxico | 0,220 | ≥ 0,122 (mitad de 0,243) | sí |
| Tipos con acierto ≥ 0,40 | 8 de 8 | ≥ 6 | sí |
| Tipos con ≥ 124 hechos reales estimados | 8 de 8 | ≥ 6 | sí |

Decisión: los 8 tipos gruesos forman tema válido para el banco.

| Tipo | Acierto kNN | Exceso | IC90 | Conocidos y no piso | Reales estimados |
|---|---|---|---|---|---|
| persona | 0,88 | +0,12 | [+0,03; +0,20] | 8 de 60 | 3.927 |
| organización | 0,43 | +0,17 | [+0,10; +0,25] | 10 de 60 | 1.554 |
| biomedicina | 0,86 | +0,27 | [+0,17; +0,37] | 16 de 60 | 1.789 |
| publicación | 0,50 | +0,08 | [+0,02; +0,15] | 6 de 60 | 874 |
| obra cultural | 0,48 | +0,12 | [+0,02; +0,22] | 10 de 60 | 2.328 |
| lugar y estructura | 0,62 | +0,20 | [+0,12; +0,28] | 12 de 60 | 722 |
| derecho y gobierno | 0,80 | +0,17 | [+0,03; +0,30] | 17 de 60 | 621 |
| software y datos | 0,52 | +0,18 | [+0,08; +0,28] | 13 de 60 | 469 |

Exceso global +0,162, IC90 [+0,129; +0,196]. Esta muestra al azar corrige a la baja el +25,6 de la muestra inicial.

Puntos débiles para el diseño:

1. Tipo y relación están confundidos. Biomedicina usa part of y subclass of. Derecho usa country y jurisdicción. Country aparece en tres tipos y permite cruzar tipo con relación.
2. La selección es cara: solo 10 % a 28 % de los hechos son conocidos y no piso. Armar el banco exige filtrar hechos en el dataset completo.
3. La pureza es baja. Con 36,9 % conocidos y 20,6 % de piso, cerca de 56 % de los conocidos son reales. Conviene un filtro más estricto.
4. B18 sortea 64 hechos por conjunto. Seis conjuntos casi independientes piden unos 444 hechos por tipo, no 124. Software queda justo, con 469 estimados.
5. B20 usa árbol de supertemas. Los 8 tipos todavía no tienen árbol.
6. La agrupación se definió por criterio propio sobre etiquetas. Otra agrupación puede dar otro resultado.
7. kNN solo filtra candidatos. Falta medir el gradiente de daño por tipo.

Siguiente paso: diseñar el banco y costearlo, con investigación en Lacuna antes de gastar unidades.

## Diseño del banco B23 (2026-11-09, borrador de preregistro)

Investigación: nota del vault "Banco UniEdit: filtro de hechos conocidos y predictores de daño".

### Pregunta

¿El daño del conflicto sigue al tipo de la entidad, a la relación, o a ninguno? La respuesta decide si el gradiente de B20 generaliza fuera de WordNet.

### Hechos

- Cada hecho usa la plantilla "The {propiedad} of {cabeza} is" y la respuesta " {valor}.".
- El conflicto cambia el valor por el de otro hecho de igual propiedad.
- Las cabezas son disjuntas entre entrenamiento, evaluación y anclas.
- Cada tipo aporta 60 hechos de evaluación fijos.
- Las 512 anclas salen de los otros tipos, en partes iguales.
- Un tipo entra si tiene 444 hechos seleccionados o más. El banco necesita 6 tipos o más.

### Filtro de hechos conocidos

- F1: el valor real gana a 4 donantes en la plantilla corta y en el prompt de UniEdit.
- Prueba retenida: una formulación con 3 ejemplos de la misma relación.
- Se adopta F1 si su exceso retenido sobre el piso llega a 0,40.
- Si F1 no llega, se usa F0: conocido y no piso.
- Control de la prueba retenida: 60 hechos no conocidos. Si exceden igual que los conocidos, la prueba no informa.
- Las formulaciones nuevas se miden solo en los hechos conocidos y en ese control.

Resultado de la etapa 0 (2026-11-09, 444 hechos con ejemplos):

| Grupo | n | Gana con cabeza real | Gana con cabeza ajena | Exceso | IC90 |
|---|---|---|---|---|---|
| F1 | 146 | 0,849 | 0,452 | +0,397 | [+0,322; +0,473] |
| F0 | 85 | 0,824 | 0,176 | +0,647 | [+0,565; +0,730] |
| Conocidos | 165 | 0,830 | 0,436 | +0,394 | [+0,321; +0,461] |
| Control | 60 | 0,233 | 0,117 | +0,117 | [+0,017; +0,217] |

- La regla elige F0, porque F1 no llega a 0,40.
- El exceso de F0 está inflado en parte: su selección y su piso retenido usan la misma cabeza ajena.
- F1 deja 45,2 % de hechos que ganan también con una cabeza ajena. Esos hechos miden plausibilidad, no conocimiento.
- F0 aprueba 85 de 444 hechos. Los 8 tipos superan 444 estimados, y publicación es el más justo: 4 de 51 aprobados.
- Resultados y scripts: `.cache/runs/bloques/uniedit_filtro/` en mf-moe.

### Cruce de tipo y relación

El muestreo natural deja solo 1,4 % de pares en "mismo tipo, otra relación". Por eso cada conjunto reparte al azar las propiedades del tipo en dos mitades. El entrenamiento usa una mitad. La otra mitad queda reservada para esa celda.

### Análisis

Modelo principal, con efecto fijo por hecho y por conjunto: subida = tipo + relación + tipo × relación + hecho + conjunto. Este modelo estima el efecto total y decide las etapas.

Modelo de mediación: el modelo principal más el coseno. El coseno no entra en el modelo principal porque es un mediador del tipo. Controlarlo borraría parte del efecto buscado.

- El coseno es la media de los 10 mayores cosenos contra el entrenamiento, en la capa 8.
- El intervalo sale de un bootstrap por conjunto.
- Efecto mínimo de interés: 0,10 nats, un tercio del efecto de B20 (0,324).
- Medida secundaria: fracción de hechos donde el valor real deja de ganar a sus donantes.
- Control del análisis: datos sintéticos con efecto plantado y con efecto nulo, en CPU, antes de gastar GPU.

### Corridas y decisiones

| Etapa | Dónde | Corridas | Costo | Decisión que cambia |
|---|---|---|---|---|
| 0. Filtro | CPU | 480 hechos, 2 formulaciones | 0 unidades | Elige F1 o F0 |
| 1. Selección | GPU | Hasta 444 por tipo | tope 1 unidad | Fija los tipos que entran |
| 2. Gradiente, primera tanda | GPU | 8 tipos × 3 conjuntos + 4 repeticiones | tope 2 unidades | Decide si sigue la segunda tanda |
| 3. Gradiente, segunda tanda | GPU | 8 tipos × 3 conjuntos | tope 1 unidad | Separa tipo de relación |
| 4. Consolidación | GPU | 8 tipos × 2 conjuntos × `cons24`, `conf24` y `azar24` + 8 controles | tope 2 unidades | Replica la consolidación fuera de WordNet y compara conflicto contra uso |

Tope total: 6 unidades. Tiempos de referencia: B20 tardó 44 s por corrida y B21 tardó 53 s.

Reglas de decisión:

1. La puerta C0 de B18 corre en la primera corrida. Exige conflicto específico.
2. La segunda tanda no corre en tres casos: tipo llega a 0,15 con IC90 > 0; tipo y relación quedan bajo 0,05; o tipo queda bajo 0,05 y relación llega a 0,15 con IC90 > 0. En los demás casos corre. Control sintético: 800 simulaciones, 100 % de acuerdo con 48 conjuntos fijos.
3. Tipo gana si su IC90 supera 0 y su punto llega a 0,10. Entonces la etapa 4 usa tipo como tema.
4. Relación gana si solo ella cumple esa regla. Entonces la etapa 4 usa relación como tema.
5. Si ninguno cumple, el paper limita el gradiente a WordNet y la etapa 4 no corre.
6. Las reglas 2 a 5 usan el efecto total del modelo principal.
7. Si el coseno absorbe más de la mitad del efecto total de tipo, el paper lo reporta como efecto de representación.

### Brazo de conflicto en la etapa 4

- `conf24` protege, por capa, los 24 expertos con el producto interno más negativo entre el gradiente medio nuevo y el viejo.
- El gradiente se mide una vez, antes de entrenar, con los mismos datos viejos que usa `cons24`.
- La comparación con `cons24` es pareada por conjunto.
- `conf24` gana si su IC90 supera 0 y su punto llega a 0,04, la mitad del efecto `H_cons24` de B21.
- Si no gana, el paper recomienda el uso, que es la señal más barata.
- El brazo agrega 16 corridas. La etapa 4 suma 56 corridas, cerca de 1,6 unidades.

Investigación: nota del vault "Proteger expertos MoE por conflicto de gradiente entre hechos nuevos y viejos".

### Código

Módulo nuevo `colab/b23_uniedit.py`, sin editar `b18.run_set`. Reutiliza `b17.batch` y `b17.def_losses`. La etapa 4 reutiliza los brazos de `b21_contencion`.

### Abierto

- El mapa no confirma un tope propio de unidades. El saldo estimado es de unas 580.
- La velocidad de selección en GPU no está medida. La etapa 1 mide 2.000 secuencias y corta a los 30 minutos.
- El árbol de supertemas no entra: ninguna decisión de estas etapas lo necesita.

## Etapa 1: resultado y regla de interacción (2026-11-09)

La selección en GPU terminó con 8 tipos dentro: 446 a 459 hechos por tipo. Corrió 547 s a 229 secuencias por segundo y costó unas 0,7 unidades.

Los 480 hechos evaluados cubren 151 propiedades, y 124 aparecen en un solo tipo. La celda otro tipo y misma relación tiene entre 2 y 123 hechos; `persona` tiene de 2 a 4.

El poder simulado con las mitades reales alcanza. Con 24 conjuntos: tipo 0,10 gana 53 %, tipo 0,20 gana 100 %, relación 0,15 gana 98 %, nulo gana 0 %. La codificación centrada no mejora; queda la codificación de referencia.

Regla nueva: la interacción gana con IC90 > 0 y punto >= 0,10. Una victoria indica daño que exige mismo tipo y misma relación, y habilita la etapa 4. Con 24 conjuntos: interacción 0,15 gana 98 %, nulo gana 0 %, tipo 0,20 solo gana 0 %. La simulación no incluye un choque por conjunto propio de la interacción.

## Etapa 2: resultado del gradiente (2026-11-09)

La etapa corrió 28 de 28 corridas en 21 min y costó unas 0,7 unidades. La puerta C0 pasó en la primera corrida.

Modelo principal, 24 conjuntos, subida en nats, IC90 por bootstrap de conjuntos:

| Efecto | Punto | IC90 | Regla |
|---|---|---|---|
| Tipo | +0,479 | [+0,420; +0,552] | gana |
| Relación | +0,382 | [+0,350; +0,423] | gana |
| Interacción | -0,056 | [-0,169; +0,042] | no gana |

El coseno absorbe 18 % del tipo y 3 % de la relación. La regla de absorción exige más de 50 %, y no se cumple.

Comparación por mitad sorteada, con efectos fijos de conjunto y de hecho por tipo entrenado: relación fuera del tipo +0,336 [+0,268; +0,410]; diferencia dentro del tipo +0,182 [+0,027; +0,314]. La interacción queda sin resolver entre ambos modelos.

La victoria media cae de 0,963 a 0,781 en otro tipo y otra relación, y a 0,540 en mismo tipo y misma relación. La pérdida verdadera baja -0,60 en otro tipo y otra relación, porque el entrenamiento adapta el modelo a la plantilla.

El ruido por hecho medido con repeticiones va de 0,46 a 0,68; el supuesto era 0,50. La subida de los entrenados tiene media 0,71 y mínimo 0,02. Los pasos medios son 22,9. El tipo varía por conjunto entrenado: TO-OO va de -0,15 a +0,80.

La regla de parada aplica: tipo >= 0,15 con IC90 > 0. La segunda tanda no corre. Etapa 4 habilitada.

## Etapa 4: lanzamiento (2026-11-09)

Corrección del diseño: `cons24` deja 24 expertos entrenables y congela 8 por capa. `conf24` congela los 8 de producto interno más negativo, con igual capacidad.

Corrección de datos: las anclas vienen agrupadas por tipo, y las primeras 64 eran de un solo tipo. Lo viejo de los brazos ahora se sortea entre todas las anclas. Los lotes de entrenamiento no cambian.

Commit 554c71a en la rama `b23-uniedit`. Corte automático a los 59 minutos.

## Etapa 4: resultado (2026-11-09)

Corrieron 56 de 56 corridas en 51 min, unas 1,65 unidades. Los 8 controles repiten la etapa 2 con correlación 1,000. Los tres brazos aprenden en 100 % de las corridas y la fuga es 0.

Reducción del daño frente a la corrida sin congelar, unidad = tipo (8), IC90 por bootstrap:

| Regla | Punto | IC90 | Resultado |
|---|---|---|---|
| H_cons24 | +0,102 | [+0,086; +0,118] | gana; B21 dio +0,080 |
| D_cons (cons24 - azar24) | +0,043 | [+0,024; +0,061] | no gana: punto bajo 0,05 |
| D_conf (conf24 - cons24) | -0,034 | [-0,053; -0,017] | no gana: conf24 daña más |
| conf24 - azar24 | +0,009 | [-0,004; +0,023] | no gana |

`conf24` comparte 9,6 % de sus congelados con `cons24`; al azar se esperaría 25 %. Solo 15,1 % de los productos internos son negativos. Por la regla del ticket, el paper recomienda el uso.

## Robustez: fuerza del hecho (2026-11-10, CPU, sin costo)

La idea viene de Alien (cuarteto MoE, olvido, memorización). Con la pérdida previa como fuerza del hecho, el efecto de tipo no varía con la fuerza: corr +0,03, IC90 [−0,06; +0,12]. El de relación tampoco: corr −0,01, IC90 [−0,10; +0,09]. Detalle en la nota de vault de Alien.

## Etapa 4b preregistrada: prec24, el cuarteto de localización (2026-11-10)

Origen: Alien propone el cuarteto MoE, olvido, memorización y localización por gradiente (átomos 19, 39, 47 y 179).
En B21, prec24 (Fisher diagonal de las anclas) le ganó a cons24 por +0,035 [+0,027; +0,043]. B23 no lo corrió.
Literatura: en poda de expertos, los criterios de gradiente y activación localizan mejor que la frecuencia de uso (Finding Fantastic Experts, arXiv 2504.05586).

Diseño:
- Brazo prec24 en los conjuntos 0 y 1 de los 8 tipos: 16 corridas, con los mismos hechos entrenados que cons24.
- Un control sin congelar (biomedicina, s0) comprueba que la sesión repite la etapa 2.
- Regla: D_prec = prec24 − cons24 gana con IC90 > 0 y punto ≥ 0,02, y prec24 aprende en ≥ 90 % de las corridas.
- El umbral 0,02 es un quinto de lo que cons24 reduce (+0,102).
- Potencia simulada con la dispersión real de la etapa 4: efecto 0,035 gana 88 %, efecto nulo gana 4 %.
- Control negativo del análisis: cons24 renombrado como prec24 da D_prec = 0 y no gana.

Decisión que cambia:
- Si D_prec gana, el paper recomienda proteger por sensibilidad, replicado en WordNet y UniEdit.
- Si no gana, el paper recomienda proteger por uso, y prec24 queda como resultado secundario de WordNet.

Costo estimado: 1 unidad, con tope de 45 minutos. Análisis: `b23_etapa4.py CONSOLIDACION GRADIENTE PREC24`.

## Resultado de la etapa 4b: prec24 (2026-11-10, 0,7 unidades)

Control biomedicina s0: corr 1,000 con la etapa 2. prec24 aprende en 16 de 16 corridas, con fuga 0.
D_prec = prec24 − cons24 = +0,015 [+0,005; +0,025]: no gana, porque el punto queda debajo de 0,02.
prec24 reduce +0,117 contra +0,102 de cons24. Va a favor en 6 de 8 tipos.
prec24 y cons24 comparten 58 % de los expertos congelados (azar: 25 %), y eso explica la diferencia chica.
La dirección repite la de B21 (+0,035), con la mitad del tamaño.
Para el paper (opción A del ticket 07): la regla recomienda proteger por uso. La sensibilidad entra como secundario con la misma dirección en WordNet y UniEdit.
Costo total de B23: cerca de 3,75 de 6 unidades.

## Etapa 4c preregistrada: candidato 2 y conteo del candidato 1 (2026-11-10)

Pregunta del candidato 2: ¿los expertos que cons24 congela se pueden guardar en pocos bits sin perder la protección?
Literatura: MoQE (arXiv 2310.02410) muestra que los pesos de expertos toleran pocos bits.
"8-bit Quantization Improves Continual Learning" (arXiv 2512.18934) cuantiza el modelo entero.
Nadie cuantiza solo los expertos que protegen lo viejo mientras el resto entrena.

Diseño del candidato 2:
- cuant4 y cuant2 congelan las mismas filas que cons24 (regla «viejo», 0,75).
- Antes de entrenar, esas filas pasan por RTN asimétrico con escala y cero por grupo de 128 entradas.
- El router de esas filas queda en fp32 y congelado. La fuga se mide contra los pesos cuantizados.
- Cada corrida guarda la evaluación tras cuantizar y antes de entrenar: el daño de cuantizar solo.
- Conjuntos 0 y 1 de los 8 tipos: 32 corridas, con los mismos hechos entrenados que cons24.
- Un control sin congelar (biomedicina, s0) comprueba que la sesión repite la etapa 2.

Reglas:
- N_cuant = cuant − cons24 se sostiene si la cota inferior del IC90 supera −0,02, con aprendizaje en ≥ 90 % de las corridas.
- El margen −0,02 es el mismo que D_prec: un quinto de lo que cons24 reduce.
- S_cuant2: el daño de cuantizar solo tiene IC90 > 0. Es el control de sensibilidad de la medida.
- Control positivo: H_cons24 = +0,102 [+0,086; +0,118] de la etapa 4.
- Control negativo del análisis: prec24 renombrado como cuant4 con daño fijo 0,03 da daño +0,030 y N = +0,015.

Decisión que cambia:
- Si N_cuant4 se sostiene, el candidato 2 afirma que la protección sobrevive a 4 bits. Eso libera memoria en los expertos congelados.
- Si N_cuant4 no se sostiene, el candidato 2 queda como resultado negativo dentro del paper A.
- cuant2 ubica el borde: si también se sostiene, la afirmación sube a 2 bits.
- Si S_cuant2 no detecta daño, la medida no ve la cuantización y N_cuant4 no prueba nada.

Conteo del candidato 1, en paralelo y solo de inferencia:
- Plantilla inversa: «{valor} is the {propiedad} of» + « {cabeza}.».
- La cabeza real compite con 9 cabezas de igual propiedad y otro valor. Con 9, el azar pasa con p ≈ 0,09.
- Un hecho es piso si su cabeza también gana con un valor ajeno de esa propiedad.
- Entre 291 y 384 hechos por tipo tienen 9 cabezas donantes.
- Puerta del candidato 1: al menos 6 tipos con 124 hechos (64 entrenados y 60 evaluados) conocidos en las dos direcciones.
- Si la puerta no pasa, el candidato 1 necesita otro banco de hechos antes de gastar unidades.

Costo estimado: 1,1 unidades, con tope de 70 minutos. VRAM estimada: 17 GB más 3,6 GB, en una L4 de 22,5 GB.
Análisis: `b23_etapa4.py CONSOLIDACION GRADIENTE CUANT`.

## Resultado del conteo inverso (2026-11-10)

La primera sesión corrió el conteo y el entrenamiento juntos. El conteo ocupó 5,4 GB y el entrenamiento murió por OOM en su primera corrida.
El conteo terminó. El entrenamiento se relanzó solo, en la sesión b23cuant2.

Hechos conocidos en las dos direcciones, por tipo: biomedicina 103, derecho_gobierno 114, organizacion 74, persona 76.
Los otros cuatro tipos quedan entre 33 y 57. La puerta pide 6 tipos con 124: pasan 0. La puerta no pasa.
El descuento del azar empeora el cuadro. De los 544 hechos «ambos», cerca de 192 se esperan por azar.
En lugar_estructura, obra_cultural, publicacion y software_datos, casi todo lo que pasa es azar.
Las relaciones que se conocen al revés son pocas: part of 72, country 57, subclass of 41. Ninguna otra llega a 40.
Consecuencia: el candidato 1 no puede usar este banco por tipo. Necesita otro banco antes de gastar unidades.

## Resultado de la etapa 4c: candidato 2 (2026-11-10, cerca de 1,6 unidades con la sesión fallida)

Control biomedicina s0: corr 1,000 con la etapa 2. cuant4 y cuant2 aprenden en 16 de 16 corridas, con fuga 0.
Daño de cuantizar solo: cuant4 +0,028 [+0,026; +0,029] y cuant2 +1,634 [+1,617; +1,650]. S_cuant2 detecta: la medida ve la cuantización.
N_cuant4 = −0,016 [−0,023; −0,008]: no se sostiene. La cota inferior queda 0,003 debajo del margen −0,02.
N_cuant2 = −0,489 [−0,527; −0,463]: no se sostiene. Con 2 bits, congelar daña más que no congelar (−0,387).
cuant4 todavía reduce el daño +0,086 [+0,073; +0,099] contra no congelar. Le gana a azar24 por +0,027 [+0,013; +0,041].
El entrenamiento repara 43 % del daño de cuantizar en cuant4 y 70 % en cuant2.
Sin el daño de cuantizar, cuant4 reduce +0,114, más que cons24 (+0,102).
Decisión según el prerregistro: el candidato 2 queda como resultado negativo dentro del paper A.
La lectura: 4 bits con RTN conservan 84 % de la protección de cons24, y el costo es el daño de cuantizar.
El cuantizador es RTN, el más simple. GPTQ o AWQ dañan menos, y eso no se midió.

## Etapa 4d preregistrada: candidato 2 con GPTQ (2026-11-11)

Pregunta: ¿N_cuant4 falló por el cuantizador o por guardar en 4 bits?
Literatura: GPTQ (Frantar et al., arXiv 2210.17323) corrige el error de cada columna con la inversa del hessiano de las entradas.
El banco de PTQ para MoE (arXiv 2406.08155, en Lacuna) aplica GPTQ por experto con el ruteo de la calibración.
Ese banco señala un riesgo: los expertos poco usados reciben pocos tokens de calibración.

Diseño:
- gptq4 congela las mismas filas que cons24 y cuant4, con la misma grilla de 4 bits y grupos de 128.
- La calibración sigue a GPTQ: 128 tramos de 2048 tokens del texto de entrenamiento de WikiText-2, con 1 % de amortiguación.
- La calibración no usa los hechos evaluados ni las anclas: eso ajustaría el cuantizador a la prueba o a lo que el brazo repasa.
- Cada hessiano sale del ruteo de Granite: el experto ve solo los tokens que el router le manda.
- Una entrada sin tokens queda con RTN. GPTQ la pondría en cero, y el modelo se evalúa con otro texto.
- Shortcut: todas las capas se calibran con el Granite sin cuantizar. GPTQ propaga el error de una capa a la siguiente.
- Shortcut: down_proj se calibra con la salida de gate_up_proj sin cuantizar.
- GPTQ corre una vez por sesión sobre los 32 expertos de las capas 6 a 23. Cada corrida copia las filas que congela.
- Cada corrida mide también el daño de RTN sobre las mismas filas: es el control pareado.
- Conjuntos 0 y 1 de los 8 tipos: 16 corridas, más el control biomedicina s0.

Controles:
- Control positivo en CPU: con H = I, GPTQ da lo mismo que RTN. Con entradas correlacionadas, GPTQ reconstruye mejor.
- Control positivo en la sesión: el error de reconstrucción sobre la calibración debe bajar contra RTN. Si no baja, el script corta antes de entrenar.
- Puerta de la primera corrida: si GPTQ no daña menos que RTN en la misma corrida, el script corta. Así no se gastan 16 corridas sin pregunta.
- El daño RTN pareado debe repetir el daño de cuant4 de la etapa 4c.

Reglas:
- N_gptq4 = gptq4 − cons24 se sostiene con la regla de N_cuant4: cota inferior del IC90 > −0,02 y aprendizaje en ≥ 90 % de las corridas.
- G_gptq4 = daño RTN pareado − daño GPTQ gana si su IC90 supera 0.
- gptq4 − cuant4 es informativo y no decide nada.

Predicción: cuant4 dio N = +0,012 − daño. Si la suma se mantiene, N_gptq4 se sostiene con un daño menor a 0,024.
Eso pide que GPTQ baje el daño de RTN al menos 13 %.

Decisión que cambia:
- Si G_gptq4 gana y N_gptq4 se sostiene, el candidato 2 afirma que la protección sobrevive a 4 bits con un cuantizador estándar.
- Si G_gptq4 gana y N_gptq4 no se sostiene, el resultado negativo vale para RTN y para GPTQ.
- Si G_gptq4 no gana, N_gptq4 no separa al cuantizador del ancho de bits. El resultado negativo queda como está.

Costo estimado: 1 unidad, con tope de 50 minutos. 17 corridas de unos 40 s, calibración de unos 3 minutos y unos 15 de arranque.
Análisis: `b23_etapa4.py CONSOLIDACION GRADIENTE CUANT GPTQ`.

## Resultado de la etapa 4d: candidato 2 con GPTQ (2026-11-11, cerca de 0,9 unidades)

La sesión duró 27 minutos. Ningún corte se activó.
Calibración: GPTQ deja 62 % del error de reconstrucción de RTN, sin expertos sin tokens.
Control biomedicina s0: corr 1,000 con la etapa 2. gptq4 aprende en 16 de 16 corridas, con fuga 0.
El daño RTN pareado repite el daño de cuant4 de la etapa 4c con diferencia máxima 0,00000.
Daño de cuantizar solo: gptq4 +0,022 [+0,021; +0,023] contra RTN +0,028. GPTQ baja el daño 20 %.
G_gptq4 = +0,0057 [+0,0043; +0,0072]: gana. GPTQ daña menos que RTN en las mismas filas.
N_gptq4 = −0,013 [−0,022; −0,004]: no se sostiene. La cota inferior queda 0,002 debajo del margen −0,02.
gptq4 − cuant4 = +0,003 [−0,003; +0,009]: la protección final no se distingue de la de RTN.
gptq4 reduce el daño +0,089 contra no congelar. Sin el daño de cuantizar, reduciría +0,111.
La predicción del punto se cumplió (−0,013). El IC90 salió más ancho que el supuesto, y por eso falla.
Decisión según el prerregistro: el resultado negativo vale para RTN y para GPTQ.
La lectura: 4 bits conservan cerca de 87 % de la protección de cons24, y el costo sigue al daño de cuantizar.
Abierto, sin lanzar: calibrar con texto del dominio de los hechos. Eso cambia la pregunta y pide su propio prerregistro.

## Cierre (2026-11-14, decisión del usuario)

- UniEdit sirve como banco con otra ontología: licencia MIT, hechos de Wikidata y conflicto con otra fila de igual propiedad.
- B23 midió el gradiente fuera de WordNet: tipo +0,479 y relación +0,382, los dos con IC90 sobre 0.
- El dominio de UniEdit no sirve como tema. Los 8 tipos gruesos sí sirven.
- Queda sin lanzar una pregunta distinta: calibrar GPTQ con texto del dominio. Necesita su propio prerregistro.
