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
