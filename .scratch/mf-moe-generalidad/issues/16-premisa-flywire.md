# FW1: ¿las sinapsis por par crecen con la importancia de los socios? (FlyWire 783)

Type: task
Status: FW1 corrido (2026-11-16): M1 falla (0,15 a 0,17); el brazo de la mosca en B38 agrega capacidad
Blocked by: 15

Pedido del usuario (2026-11-16): investigar las sinapsis dinámicas por importancia y usar el conectoma de la mosca.

## Lo ya publicado (no se repite)

- Las sinapsis totales de una neurona siguen al grado (R 0,93). La fuerza media por socio sigue poco al grado (R 0,25 a 0,30). Fuente: Dorkenwald 2024.
- Las aristas de más de 10 sinapsis reaparecen en otro cerebro en más de 90 %. Fuente: Schlegel 2024.
- El detalle está en la nota del vault `research/sinapsis-dinámicas-e-importancia-biología-redes-artificiales.md`.

## Datos

- `proofread_connections_783.feather` (Zenodo 10676866, md5 f48f972d262323a102aed49af1396b8a).
- `Supplemental_file1_neuron_annotations.tsv` (flywire_annotations), columna `side`.
- Los dos viven en `mf-moe/.cache/flywire/`, fuera de git.

## Diseño (`colab/fw_premisa.py`)

- Una arista suma las filas de un par en todos los neurópilos. Las autapsis salen. El umbral es 5 sinapsis.
- El grado se cuenta en el grafo con umbral, en el cerebro entero.
- M1: Spearman del peso de cada arista contra el grado de salida del emisor y contra el grado de entrada del receptor.
- M1 tiene un control: 20 permutaciones de los pesos dan el desvío del Spearman sin relación.
- M2: la media de sinapsis por arista dentro del club rico (grado total mayor que 37 en los dos extremos) contra el resto.
- M3: el Spearman entre las dos direcciones de cada par recíproco.
- M4: M1 a M3 se repiten en las aristas izquierda-izquierda y en las derecha-derecha.
- Se informa además el grafo con umbral 1.

## Regla y decisión

- M1 pasa si un mismo Spearman de M1 llega a 0,2 o más en los dos hemisferios.
- M1 pasa: en la mosca, la importancia también vive en la fuerza por par. El brazo de la mosca en B38 protege pesos.
- M1 falla: la importancia vive en el número de socios. El brazo de la mosca en B38 agrega capacidad.

## Predicciones (Brier al cierre)

- M1 pasa: 0,30.
- M2, razón mayor que 1,2 en los dos hemisferios: 0,55.
- M3, Spearman mayor que 0,3 en los dos hemisferios: 0,60.

## Puntos débiles conocidos

- El grado se cuenta con el mismo umbral que filtra las aristas. Una arista fuerte ayuda a su propio grado.
- Faltan muchas sinapsis: solo 44,7 % de las postsinapsis caen en neuronas revisadas.
- Las aristas de una misma neurona no son independientes. El control por permutación no corrige eso; la réplica por hemisferio sí ayuda.
- El análisis no usa tipos celulares. Un efecto puede venir de pocas clases grandes.

## Costo

- Corre en CPU, con unos 2 GB de RAM. No gasta unidades de Colab.

## Resultados de FW1 (2026-11-16, CPU, unos 3 minutos)

Salida: `.cache/runs/bloques/fw1/fw1.json` en mf-moe. El md5 del archivo coincide.
- Integridad: el grafo con umbral 5 tiene 2.700.513 aristas, el mismo número que Dorkenwald 2024.
- M1 falla. El peso por arista sube poco con el grado de los socios:
  - emisor: 0,170 a la izquierda y 0,150 a la derecha;
  - receptor: 0,149 a la izquierda y 0,145 a la derecha.
- El control por permutación da un desvío de 0,001. El efecto es real, pero chico.
- Con umbral 1, M1 baja a 0,04 (emisor) y 0,10 (receptor). Parte del efecto con umbral 5 viene del umbral mismo.
- M2 pasa la predicción: las aristas del club rico tienen 1,44 veces más sinapsis a la izquierda y 1,39 a la derecha.
- El club rico cubre la mitad de las aristas (48 % y 52 %). Con umbral 1, casi todo es club rico, y M2 pierde sentido.
- M3 no llega a 0,3: las dos direcciones de un par recíproco correlacionan 0,21 a la izquierda y 0,19 a la derecha.
- Decisión según el prerregistro: la importancia de una neurona vive sobre todo en su número de socios. El brazo de la mosca en B38 agrega capacidad.

Brier de FW1: M1 0,090, M2 0,203 y M3 0,360. La media es 0,218.
