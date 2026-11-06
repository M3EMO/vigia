# B21: contención contra consolidación en Granite

Type: task
Status: claimed
Blocked by:

## Question

En el banco de B20, con la misma capacidad entrenable, ¿qué reduce más el daño: elegir los expertos que usa lo nuevo (contención) o congelar los que usa lo viejo (consolidación)? ¿Cuánto daño pasa por el tronco, fuera de los expertos?

## Preregistro (2026-11-06)

`colab/b21_contencion.py`, `colab/b21_analisis.py` de mf-moe. Banco, temas y evaluación de B20. Conjuntos 0 y 1, con orden 0.

### Medición previa, sin gasto (CPU, Granite real, uso en los tokens de definición)

Esta medición ya responde parte de la pregunta, y cambió el diseño.
- Los temas casi no separan expertos: Jaccard del top-8 entre temas 0,644, con azar 0,143. Coseno de uso entre temas 0,95.
- Uso dentro del top-8 de los hechos nuevos, con azar 0,25: tema 0,471; supertema 0,469; otro 0,460.
- Los hechos nuevos y las anclas eligen casi los mismos expertos: Jaccard 0,767. Por eso `cont8` entrena expertos populares, no propios del tema.
- Mejor separación posible, elegida en la misma muestra: top-8 específico por definiciones / uso general, tema 0,304 y otro 0,215 (cociente 1,41).
- Los hechos nuevos usan solo 7,7 % los 8 expertos menos usados por las anclas. `cons8` puede no aprender.
- Lectura: con este ruteo, los temas comparten expertos, y una contención por tema selectiva no tiene dónde actuar. Una real tendría que cambiar el ruteo, como LEGO.

### Literatura

- LEGO (SCALE @ ICML 2026): aísla un dominio nuevo con expertos clonados, presión de ruteo y anclas. Un solo cambio de dominio. Actualizar el tronco fue necesario.
- Hiratani (NeurIPS 2024): lo peor son entradas parecidas con salidas distintas. Las aproximaciones diagonales de Fisher fallan con tareas muy parecidas.
- §4.12 de vigia: consolidación por uso −31 %, al azar −16 %, en idiomas en fila. El diagnóstico atribuye 69 % a la atención y 96 % a los expertos, con solapamiento.
- La búsqueda no encontró contención por tema en este tipo de banco; eso no prueba novedad.

### Brazos

Cada brazo se compara con el control de B20 del mismo conjunto. Congelar un experto congela también su fila del router.
- `cont8`: entrenan los 8 más usados por los hechos nuevos.
- `esp8`: entrenan los 8 más específicos (uso de 64 conceptos del tema / uso de 64 anclas). Es la contención más justa sin cambiar el ruteo.
- `cons8`: entrenan los 8 menos usados por las anclas.
- `azar8`: entrenan 8 al azar.
- `cont8_tronco`: como `cont8`, y además congela lo que no es experto.
- `cons24`: se congelan los 8 más usados por las anclas (regla de §4.12).
- `azar24`: se congelan 8 al azar.
- `prec24`: como `cons24`, ordenando por Fisher diagonal de las anclas. Viene de la analogía con energía libre. Secundario.
- Repetición del control en 5 temas: valida reutilizar B20.

### Reglas

Reducción > 0 significa menos daño. Unidad de réplica: el tema. Cada regla pide IC90 sin 0, |media| ≥ 0,05 nats, Holm < 0,05 entre las 6, y que los brazos aprendan (hechos nuevos < 0,5 en ≥ 90 % de las corridas).
- C_mask: `fuga` = 0 en todas las corridas.
- C_pos: `cont8_tronco` reduce el daño fuera del tema, IC90 > 0 y ≥ 0,05. Si falla, un nulo de contención no se interpreta.
- H_esp: `esp8` reduce el daño fuera del tema.
- D_esp: `esp8` − `azar8` en daño total. Dice si elegir por especificidad suma algo a reservar capacidad.
- D_pop: `cont8` − `azar8` en daño total. Dice si importa entrenar expertos populares.
- H_cons24: `cons24` reduce el daño total.
- D_cons: `cons24` − `azar24` en daño total. Repite en este banco la pregunta uso contra azar de §4.12.
- D_tronco: `cont8_tronco` − `cont8` fuera del tema. Dice cuánto pasa por el tronco.
- Secundario: `prec24` − `cons24`, sin Holm. Se informa daño por relación, pasos, pérdida final y `uso_entrenable`.

### Decisión

- H_esp y D_esp se cumplen: elegir expertos por tema contiene algo, y el próximo bloque mide contención en secuencia.
- Fallan, con C_pos cumplido: la contención por tema con este ruteo queda descartada. Solo sigue una versión que cambie el ruteo (LEGO), si un ticket nuevo justifica su costo.
- D_tronco se cumple: congelar expertos deja pasar daño por el tronco, y el paper lo informa como límite de cualquier regla por expertos.
- H_cons24 y D_cons se cumplen: la consolidación por uso se sostiene en un segundo banco. Si solo se cumple H_cons24, el efecto es sobre todo reservar capacidad.
- Nulo significa "menos de ~25 % del daño": la simulación da potencia 36 % con 0,05 nats, 83 % con 0,07 y 95 % con 0,085, suponiendo desvío 0,05 por tema.

### Auditoría de gasto

- Cada brazo cambia una decisión escrita arriba. `cont24` salió porque ninguna decisión dependía de él.
- Un brazo se abandona si sus primeras 3 corridas no aprenden, porque ese hecho ya es su resultado.
- Costo: 165 corridas. Con 46 a 90 s por corrida, 2,1 a 4,1 h. Tope duro de 3 h, ~5,8 unidades. El orden por conjunto y tema deja pares completos si el tope corta.
- Puerta: `b18.gate` en la primera corrida.

### Límites

- Es un solo entrenamiento sobre conocimiento previo, no idiomas en fila: no se compara con el −31 % de §4.12.
- El uso se midió en tokens de definición, y el gradiente también pasa por los del sujeto.
- Menos capacidad puede pedir más pasos, y más pasos dañan: por eso se informan los pasos.
- El top-8 por uso sobre los hechos nuevos refleja el texto de los donantes lejanos, no el tema.
- `esp8` elige con 64 conceptos del grupo de entrenamiento del tema; pueden coincidir con los entrenados.

## Resultado (2026-11-06, L4, 3,1 h, ~6,0 unidades)

Corrieron 144 de 165 corridas: conjunto 0 completo, 7 temas completos del conjunto 1 y 1 tema parcial. El tope de 3 h cortó el resto. La fuga es 0 y todos los brazos aprenden. Los 5 controles repetidos dan ρ = 1,000 con B20, con la misma semilla: esto valida la reutilización y no mide ruido.

| Regla | Media | IC90 | Holm | Veredicto |
|---|---|---|---|---|
| C_pos | `cont8_tronco` fuera del tema, sin Holm | ≥ 0,05 e IC90 > 0 | — | Se cumple |
| H_esp | +0,143 | [+0,117; +0,173] | 0,0001 | Se cumple |
| D_esp | −0,027 | [−0,070; +0,015] | 0,67 | Falla |
| D_pop | +0,020 | [−0,010; +0,051] | 0,67 | Falla |
| H_cons24 | +0,080 | [+0,042; +0,119] | 0,029 | Se cumple |
| D_cons | +0,071 | [+0,061; +0,081] | < 0,0001 | Se cumple |
| D_tronco | +0,156 | [+0,133; +0,176] | < 0,0001 | Se cumple |

Secundario: `prec24` − `cons24` da +0,035 [+0,027; +0,043]. Solo con el conjunto 0, las seis reglas dan el mismo veredicto.

### Decisión aplicada

- La contención por tema con este ruteo queda descartada: `esp8` no supera al azar, y H_esp refleja capacidad reservada.
- Solo sigue una versión que cambie el ruteo (LEGO), si un ticket nuevo justifica su costo.
- La consolidación por uso se sostiene en un segundo banco: H_cons24 y D_cons se cumplen.
- El paper informa el tronco como límite de cualquier regla por expertos.

### Exploratorio, no preregistrado

- Con 8 expertos entrenables, las cuatro reglas reducen 46-75 % del daño en el conjunto 0; `cons8`, `azar8` y `cont8` quedan entre 69 % y 75 %.
- `cons24` reduce 33 %, `azar24` 1 %, `prec24` 48 %. `cont8_tronco` reduce 139 %, así que el daño medio pasa a negativo.
- `esp8` − `azar8` da −0,139 [−0,207; −0,075] en el tema, −0,103 en el supertema y −0,003 [−0,042; +0,036] en otros supertemas.
- Elegir por especificidad concentra daño en el tema sin reducir el daño lejano. Esta lectura es post hoc.
- El efecto de `esp8` varía por tema: de −0,175 en mammal a +0,111 en vegetable. Los primeros 4 temas exageraron la señal.
- Ajustando por `nuevos_final` y pasos, D_cons queda en +0,054 a +0,058, y `prec24` − `cons24` en +0,035. Es exploratorio porque `nuevos_final` es mediador.

### Límites

- Es un entrenamiento único, no la secuencia de §4.12.
- Los pasos varían: 29-30 en brazos de 24, 31-42 en brazos de 8, control 24 en los temas repetidos.
- El daño usa teacher forcing, sobre definiciones, y no se registró ‖Δθ‖.
- El tema con conjunto 1 parcial no tiene los mismos conjuntos en todos los brazos. El análisis solo con conjunto 0 evita ese desbalance.
