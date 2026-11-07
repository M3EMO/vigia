# B20 en OLMoE-1B-7B

Type: task
Status: open
Blocked by: 01

## Question

¿Mapas, gradiente y fiabilidad de B20 se repiten en un MoE de 64 expertos con datos abiertos? Primero medir memoria y tiempo con una corrida piloto: 7B de parámetros no entran enteros en fp32 en una L4.

OLMoE: https://lacuna.tiptreesystems.com/work/olmoe-open-mixture-of-experts-language-models-6sr7/version/art_699789bfd8074140ab09d875380361c3

## Viabilidad (2026-11-06, sin gasto de unidades)

- OLMoE-1B-7B-0125 es compatible con ajustes, pero 12 capas entrenadas ocupan 84,3 GB sin activaciones.
- 1 capa ocupa 19,7 GB y 4 capas 37,3 GB, así que la comparación con Granite pierde fuerza.
- Antes de descargar OLMoE, una corrida en Granite mide si C0 pasa con 1 y 4 capas finales entrenables.
- Si C0 falla en Granite, este ticket no sirve en L4.
- La VRAM de L4 sale de una fuente secundaria (~22 GB): falta confirmarla con nvidia-smi.

Detalle: nota del vault "Viabilidad de ampliar mf-moe: OLMoE, FLAME-MoE, UniEdit y frecuencia de corpus".

## Preregistro de B22: profundidad en Granite (2026-11-06)

### Pregunta

¿El banco de B20 sobrevive con 1 o 4 capas finales entrenables? OLMoE solo entra en una L4 con muy pocas capas. Si el banco no sobrevive en Granite, comparar OLMoE con Granite no dice nada.

### Literatura

- Hase (Lacuna, art_e098c4244ab64f78ae744785b4318813) halla que la capa pesa mucho en el éxito de edición, sin medir daño.
- No hallé un trabajo que mida este banco por profundidad. Solo leí resúmenes.

### Diseño

- Conjunto 0 de B20, 10 temas, brazos `p4` y `p1`, más un `p18` en fish: 21 corridas.
- Congela con `requires_grad` en una sola carga, así que el congelado sigue en fp32 y no en bf16.
- Control: `p18` debe repetir B20 con ρ cercano a 1.
- Registra VRAM total y pico de memoria, para corregir la estimación de OLMoE.

### Reglas

Una profundidad es válida si cumple todas:

- Aprende en ≥ 90 % de las corridas.
- Conserva el conflicto (C0) en ≥ 80 % de las corridas.
- Tiene gradiente tema − fuera con IC90 > 0 y ≥ 0,05 nats, con ≥ 3 temas.
- Tiene ρ medio con B20 ≥ 0,55, el parecido entre dos conjuntos del mismo tema en B20 (0,553). El techo medido con otro orden de lotes es ~0,65.

Un brazo se abandona si sus primeras 3 corridas no aprenden.

### Decisión

- `p1` válida: se corre el piloto de OLMoE con 1 capa en L4.
- `p1` falla y `p4` es válida: OLMoE en L4 no sirve, y solo sigue un ticket nuevo con menos memoria o una GPU mayor.
- Ambas fallan: este ticket se cierra en esta forma, y la generalidad sigue por UniEdit en Granite.

### Auditoría de gasto

- Cada brazo decide una rama de arriba, y `p18` valida el código por 1 corrida.
- Costo: tope de 1,5 h más carga, menos de 3,2 unidades. Se espera menos si `p1` se abandona.

### Límites

- Granite no es OLMoE: el router, la cantidad de expertos y el dropout siguen distintos.
- Un conjunto por tema no separa conjunto de profundidad.
- Las 18 capas de B20 no se comparan con repetición independiente.
