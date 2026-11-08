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
