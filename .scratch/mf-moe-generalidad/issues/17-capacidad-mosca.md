# B38: capacidad a la mosca contra protección de pesos (EWC) al sobrescribir hechos

Type: task
Status: B38 corrido (2026-11-16): E1 pasa y ewc crea N2; F1 y F2 fallan, la mosca daña todo
Blocked by: 16

Pedido del usuario (2026-11-16): «Dale, escribí el prerregistro de B38».

Pregunta: ¿una capacidad nueva y esparsa, como el cuerpo fungiforme de la mosca, protege el resto del conocimiento mejor que el ajuste común y que EWC?

## Por qué esta corrida cambia decisiones

- FW1 (ticket 16) mostró que en la mosca la importancia vive sobre todo en el número de socios. El prerregistro de FW1 eligió el brazo de capacidad.
- El informe no compara la puerta de certeza con EWC. Un revisor lo va a pedir. El brazo ewc cierra ese hueco con cualquier resultado.
- B37 mostró que repasar 20 hechos cualesquiera corta cerca de 60 % del daño al resto. B38 dice si la capacidad nueva llega a ese nivel sin repaso.

## Base común

- El banco, los conjuntos (tag «b25»), las semillas, los lotes, la tasa de 5e-5 y las dosis 10 y 40 son los de B25.
- El brazo comun es la ida de B25 sin cambios. B38 no lo vuelve a correr: lo lee de `b25_reales.json`.
- El conjunto 0 sirve solo para elegir un hiperparámetro por brazo. Las reglas usan los conjuntos 1 a 7.

## Brazos nuevos (`colab/b38_capacidad.py`)

- ewc: el ajuste de B25 con la penalidad λ/2 · Σ F_i (θ_i − θ*_i)² sobre las capas entrenables (6 a 23).
  - F es la Fisher empírica: el cuadrado medio del gradiente de la pérdida de cada hecho verdadero, uno por uno.
  - Los hechos son los 255 de `b25_candidatos.json` que no están en el banco de 206 hechos. F se normaliza a media 1.
  - λ se elige en el conjunto 0 entre 1, 100 y 10.000.
- mosca: el modelo base queda congelado entero. Una expansión nueva se suma al residuo después del bloque MoE de la capa 16.
  - Entrada: la norma RMS del estado, por una proyección fija, aleatoria y binaria a 40 veces el ancho del estado.
  - El ancho se lee de la configuración del modelo. Si vale 1.024, la expansión tiene 40.960 unidades.
  - Cada unidad mira 40 dimensiones al azar, cerca de 4 %, como las garras de una célula de Kenyon.
  - Solo quedan activas las 5 % de unidades más altas de cada token (ganador por k). La proyección de entrada nunca cambia.
  - Salida: una matriz de 40.960 por 1.024, con inicio en cero. Es lo único entrenable, como las sinapsis de Kenyon a las neuronas de salida.
  - La tasa de la salida se elige en el conjunto 0 entre 1e-4, 1e-3 y 1e-2.
- mosca_densa: la misma expansión con todas las unidades activas (ReLU sin ganador por k), con la tasa elegida para mosca.
- Con la salida en cero, los brazos mosca y mosca_densa dan la pérdida base de B25. Un test lo verifica.

## Regla de elección en el conjunto 0 (por brazo)

- Se descartan los valores cuya pérdida de los hechos nuevos en la dosis 40 supere 0,5.
- Entre los que quedan, gana el de menor daño al resto en la dosis 40.
- Si ninguno queda, gana el de menor pérdida de los hechos nuevos, y el brazo se informa como sin aprendizaje.

## Medidas (conjuntos 1 a 7, dosis 10 y 40)

- D: la subida media de la pérdida de ida de los hechos no sobrescritos.
- A: la pérdida media de los hechos nuevos.
- N2: el Spearman entre la certeza previa y la subida de cada hecho sobrescrito, como en B27c.
- Los IC90 salen de un bootstrap de los 7 conjuntos, con `colab.b20_analisis.ci`.

## Reglas (dosis 40)

- I0 (integridad): la pérdida base coincide con la de B25, con error 0.
- L0 (control de aprendizaje, por brazo): A vale 0,5 o menos. Un brazo que falla L0 no entra en F1, F2 ni E1.
- F1 (principal): la reducción relativa del daño, (D_comun − D_mosca) / D_comun, tiene IC90 sobre 0 y media de 0,30 o más.
- F2 (esparsidad): (D_densa − D_mosca) / D_densa tiene IC90 sobre 0 y media de 0,10 o más.
- E1 (EWC): (D_comun − D_ewc) / D_comun tiene IC90 sobre 0 y media de 0,10 o más.
- N (descriptivo): N2 de cada brazo y su diferencia pareada con comun, en las dosis 10 y 40.

## Decisiones

- F1 pasa: la capacidad esparsa protege el resto. El informe la suma como alternativa a la puerta, y el paso siguiente la combina con la puerta.
- F1 falla con L0: la regla de la mosca no alcanza para sobrescribir hechos en un modelo de lenguaje.
- F2 pasa: la esparsidad es el ingrediente activo, no solo la capacidad nueva.
- E1 pasa o falla: el informe suma EWC como línea de base en §4.15.
- Si ewc no crea N2, la selectividad por certeza queda como aporte propio de la puerta.

## Predicciones (Brier al cierre)

- L0 pasa para mosca: 0,75.
- F1 pasa: 0,55.
- F2 pasa: 0,50.
- E1 pasa: 0,60.
- ewc crea N2 (IC90 bajo 0 y media de −0,15 o menos en la dosis 10): 0,15.

## Costo y tope

- Elección en el conjunto 0: 6 corridas de 40 pasos.
- Conjuntos 1 a 7: 3 brazos, 21 corridas de 40 pasos, más 255 gradientes para F.
- Estimado: unos 30 minutos de L4, cerca de 1 u. Tope por timer: 50 minutos.

## Puntos débiles conocidos

- La capa 16 y los tamaños de la expansión se fijan a priori, sin barrido.
- La elección de λ y de la tasa usa un solo conjunto y tres valores.
- comun entrena 18 capas y mosca entrena solo la salida nueva. La comparación mezcla capacidad nueva con congelamiento; mosca_densa controla solo la esparsidad.
- La mosca aprende con una regla local y dopamina. Aquí la salida aprende por gradiente.
- La capacidad no depende de la importancia de cada hecho. Una variante con más unidades para hechos importantes queda para después.
- La regla de FlyModel solo suma asociaciones. Sobrescribir un valor viejo es otra tarea, y mosca puede fallar L0 por eso.

## Resultados de B38 (2026-11-16, 20 minutos de L4, ~0,65 u)

Análisis: `.cache/runs/bloques/b38_capacidad_gpu/analisis.out` en mf-moe.
- I0 pasa, con error 0.
- Elección en el conjunto 0: λ = 1 para ewc y tasa 1e-2 para mosca. Los dos aprenden.
- Con λ = 10.000, ewc no aprende (4,15). Con λ = 100 aprende, con más daño que con λ = 1.
- L0: comun, ewc y mosca aprenden. mosca_densa falla L0 (1,36).
- E1 pasa: ewc corta el daño al resto 61 % [57 %; 66 %] en la dosis 40, de +2,74 a +1,04.
- ewc crea N2 en la dosis 10: −0,171 [−0,230; −0,110]. Contra comun, la diferencia vale −0,152 [−0,235; −0,069].
- Esa selectividad vale 0,52 del oráculo de B27 y se apaga en la dosis 40: −0,109 [−0,254; +0,027].
- F1 falla: mosca daña el resto +10,19 contra +2,74 de comun. La reducción relativa vale −2,81 [−3,15; −2,46].
- F2 falla: mosca daña 16 % más que mosca_densa en la dosis 40. En la dosis 10, mosca daña 5 % menos.
- mosca y mosca_densa dan N2 positivo (+0,24 y +0,22): con la base congelada, los hechos seguros suben más.

Decisiones según el prerregistro:
- F1 falla con L0: la regla de la mosca, así implementada, no alcanza para sobrescribir hechos en un modelo de lenguaje.
- E1 pasa: el informe suma EWC como línea de base en §4.15.
- ewc crea N2: la selectividad por certeza no es un aporte exclusivo de la puerta. El informe lo dice en §1 y §4.15.

Lectura (no prerregistrada):
- La Fisher de hechos conocidos marca los pesos de los hechos seguros. Por eso EWC actúa como una puerta de certeza implícita.
- La expansión de la mosca escribe en todos los tokens. Las preguntas comparten la plantilla, y la expansión no tiene una entrada selectiva.
- El paso siguiente con más valor es combinar EWC con la puerta de certeza.

Brier de B38: L0 0,062, F1 0,303, F2 0,250, E1 0,160 y ewc N2 0,722. La media es 0,299.
