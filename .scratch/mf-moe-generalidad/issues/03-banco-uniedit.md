# Banco con UniEdit

Type: research
Status: open
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
