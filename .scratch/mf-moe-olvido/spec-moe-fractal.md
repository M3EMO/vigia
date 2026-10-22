# Spec del MoE fractal: requisitos, medidas y riesgos (desarrollo guiado por spec)

Fecha: 2026-10-22. Pedido del usuario: "un mapa completo y detallado, con spec driven development".
Este documento es la fuente de verdad de qué tiene que cumplir cada corrida antes de gastar Colab.
Los tickets del mapa (`map.md`) guardan las decisiones; este documento junta los requisitos que salen de ellas.

## Cómo se usa

- Cada requisito tiene un ID, un porqué, una medida o un test, un criterio de aceptación y un estado.
- Los estados son tres: **hecho** (test o medida en el código), **parcial** y **falta**.
- Un brazo nuevo entra a la cola de Colab solo si todos sus requisitos **obligatorios** están hechos.
- El orden de trabajo es fijo: requisito escrito, después test que falla, después código, después piloto.
- Un hallazgo que contradice un requisito frena la cola y abre un comentario en el ticket que corresponde.
- Las predicciones se escriben antes de correr (sección H) y no se cambian después de ver los datos.

## H. Hipótesis y predicciones (preregistradas)

| ID | Hipótesis | Predicción medible | Estado del dato |
|---|---|---|---|
| H1 | Dividir expertos (MoE plano) olvida menos que un MoE fijo con repetición | olvido medio de `division` < `control`, IC 95 % pareado sobre 0 | refutada: empate con 3 pares (+0,018, IC [−0,005; +0,051]) |
| H2 | Fusionar expertos ayuda | `division+fusion_alineada` < `division` | refutada en fase 1 y fase 2 (empeora) |
| H3 | La duda elige mejor qué dividir que el azar | `division+padre_congelado` < `azar+padre_congelado` | a favor, chico (+0,033 con 2 pares); falta la semilla 2 |
| H4 | Proteger lo usado reduce el olvido | `division+consolidacion_uso` < `control` | a favor (+0,164 con 2 pares) |
| H5 | La división aporta algo sobre la consolidación | `division+consolidacion_uso` < `control+consolidacion_uso` | en cola (etapa A) |
| H6 | Elegir por uso importa | `control+consolidacion_uso` < `control+consolidacion_azar` | en cola (etapa A) |
| H7 | El árbol fractal olvida menos, o aprende mejor con el mismo olvido, que el plano con la misma consolidación | `arbol+consolidacion` contra `control+consolidacion_uso` | piloto 8 en cola |
| H8 | El árbol se vuelve más autosimilar que el plano | `ruteo.fractal.autosimilitud` del árbol > la del plano | piloto 8 en cola |
| H9 | La duda elige dónde abrir rama mejor que el azar | `arbol+consolidacion` < `arbol_azar+consolidacion` | depende del piloto 8 |

## V. Validez del experimento

| ID | Requisito | Porqué | Medida o test | Aceptación | Estado |
|---|---|---|---|---|---|
| V1 | Control con los mismos parámetros entrenables que el árbol | una hoja de bajo rango tiene ~57 mil parámetros y un experto entero ~1,5 M; sin este control, "olvida menos" puede ser "aprende menos" | brazo `control+lora_dominio`: un delta de bajo rango nuevo por dominio sobre los expertos más usados, sin árbol | mismo número de parámetros entrenables (±10 %) que `arbol+consolidacion` | **falta, obligatorio para H7** |
| V2 | Mismo cómputo por token | las compuertas suman cómputo | FLOPs por token y segundos por paso en el resultado | diferencia < 5 % o reportada | parcial (segundos en pilotos) |
| V3 | Poder estadístico | el ruido entre semillas del control es ±0,01 nats | bootstrap pareado | 3 pares alcanzan para efectos ≥ 0,05; efectos menores (H3, H9) piden 5 semillas | parcial |
| V4 | Más de un orden de dominios | hoy todas las semillas usan finés → turco → indonesio → suajili; el olvido depende del orden | 2 órdenes más para los brazos que ganen | el signo del efecto se mantiene en los 3 órdenes | **falta** |
| V5 | Comparaciones múltiples | hay ~11 comparaciones | corrección de Holm sobre los p del bootstrap | las conclusiones del paper sobreviven a Holm; el resto se marca exploratorio | **falta** |
| V6 | Mismos datos en todas las corridas | | sha256 en `result.json` y chequeo en `analysis2` | idénticos | hecho |
| V7 | Evaluación sin fuga | | split por documento (`prepare_phase2`) | ningún documento en train y eval | hecho |
| V8 | Hiperparámetros justos para cada brazo | el lr (6e-5) se calibró para expertos enteros; los deltas U·V pueden pedir otro | barrido chico del lr de los deltas (×1, ×3, ×10) en un piloto | el árbol usa el mejor de los tres, elegido por pérdida aprendida, no por olvido | **falta** |
| V9 | Retomes registrados | retomar pone los momentos de AdamW en cero y cambia la trayectoria | cantidad de retomes por corrida (runner.log) | los brazos no difieren en retomes de forma sistemática, o se reporta | parcial |
| V10 | Todo resultado se publica | evita reportar solo lo que gana | tabla completa en §4.12 del paper | todas las corridas en la tabla | hecho (fase 1), en curso (fase 2) |

## M. Medidas

| ID | Medida | Para qué | Estado |
|---|---|---|---|
| M1 | Olvido medio (nats) de los 3 primeros idiomas | medida primaria | hecho |
| M2 | Pérdida aprendida media | plasticidad: ¿aprende cada idioma nuevo? | hecho |
| M3 | Pérdida general final (FineWeb-Edu) | ¿daña lo que el modelo ya sabía? | hecho |
| M4 | Frontera estabilidad-plasticidad | un brazo que olvida menos pero aprende peor no gana solo por M1; se grafica M1 contra M2 | **falta (figura)** |
| M5 | Transferencia hacia adelante | ¿aprende más rápido el idioma nuevo? pérdida a los 50 y 100 pasos | parcial (el log la tiene, falta el análisis) |
| M6 | Evaluación externa | la pérdida no es todo; una tarea (p. ej. Belebele por idioma) confirma | falta, opcional |
| M7 | Especialización y fractal | `ruteo.fractal` (familia/idioma/documento), exceso sobre nulo multinomial | hecho (fase 2b en adelante) |
| M8 | Costo | parámetros entrenables, VRAM, segundos por paso, unidades de Colab | parcial |

## R. Ruteo

| ID | Requisito | Medida o test | Aceptación | Estado |
|---|---|---|---|---|
| R1 | Sin colapso del enrutador | `ruteo.entropia_uso`, `ruteo.hojas_muertas` | hojas muertas ≤ 25 % por capa | hecho (medida); regla de corte en el piloto 8 |
| R2 | Compuertas balanceadas | sesgo por compuerta; test `test_gate_balancing_pushes_a_lopsided_gate_toward_both_daughters` | ninguna compuerta > 95 % a una hija en dos revisiones seguidas | hecho (mecanismo); falta registrar el reparto por revisión |
| R3 | Sin inferencia dispar | `ruteo.train_vs_eval`, `ruteo.bf16_vs_fp32` | ≥ 0,99 | hecho |
| R4 | Deriva del ruteo de lo viejo medida | `ruteo.deriva` (Jaccard top-k al cerrar el dominio contra el final) | se reporta; la consolidación sin deriva baja es sospechosa | hecho |
| R5 | El balanceo no aleja tokens de los expertos congelados | uso de los congelados en los dominios viejos, al cerrar y al final | la caída de uso < 20 % | **falta** |
| R6 | Árbol sin mitosis = Granite | test `test_without_mitosis_the_tree_routes_like_granite` | logits iguales | hecho |
| R7 | Mitosis sin cambio de salida | test `test_mitosis_and_nested_mitosis_keep_the_output...` | logits iguales | hecho |

## O. Optimización

| ID | Requisito | Test | Estado |
|---|---|---|---|
| O1 | Filas congeladas no se mueven (gradiente y momentos en cero, sin weight decay) | `test_training_moves_only_the_low_rank_delta_and_the_gates`, tests de consolidación | hecho |
| O2 | Deltas nacen en cero (U = 0) | test de mitosis | hecho |
| O3 | lr de los deltas calibrado | ver V8 | **falta** |
| O4 | Actualizaciones en fp32, multiplicación en bf16 | `_Bf16Weights` | hecho |
| O5 | Recalentamiento del lr por dominio y al retomar | `run2` | hecho |
| O6 | Retome exacto de la estructura | `test_resume_mid_domain...[arbol+consolidacion]` | hecho |

## D. Dónde vive el olvido

| ID | Hallazgo o requisito | Estado |
|---|---|---|
| D1 | Devolver los expertos recupera 96 % del olvido; la atención sola 69 %; el router solo 6 %; las normas 0 % | hecho (diagnóstico, ticket 12) |
| D2 | Proteger la atención (etapa B: lr de la atención × 0,1 desde el segundo dominio) | falta (spec en ticket 14) |
| D3 | Repetir el diagnóstico sobre el brazo ganador: ¿dónde olvida lo que queda? | **falta** |

## F. Fractal

| ID | Requisito | Estado |
|---|---|---|
| F1 | Definición operativa: autosimilitud por niveles de la jerarquía de los datos, contra un nulo multinomial | hecho (`expert_tree.fractal`) |
| F2 | Nivel documento en el árbol natural (inferencia, ~2 unidades) | falta |
| F3 | Forma del árbol: profundidad de las hojas, factor de ramificación, mitosis por nivel | **falta (agregar al reporte)** |
| F4 | Árbol natural con estonio y DeepSeek-V2-Lite (descarga a CPU) | falta |
| F5 | Árbol aprendido por gradiente (candidato 3) | falta, después del piloto 8 |
| F6 | Árbol natural como punto de partida (principio 1 de la síntesis) | falta: hoy las raíces son los 32 expertos sin agrupar |

## C. Mecanismos (en orden)

1. Etapa A: consolidación sin división, por uso y al azar (en cola, `fase2b.txt`).
2. Piloto 8: árbol + consolidación, con reglas de corte (en cola, unidad `mfmoe-etapac`).
3. V1 y V8: control con los mismos parámetros y barrido del lr de los deltas (antes de las 3 semillas del árbol).
4. Etapa C: `arbol+consolidacion` y `arbol_azar+consolidacion`, 3 semillas.
5. Etapa B: atención lenta sobre el ganador.
6. Candidato 3: árbol aprendido.
7. Pendientes de diseño: cuándo dividir guiado por datos (desacuerdo de gradientes entre mitades) y fusión solo aceptada.

## L. Comparación con la literatura

| Método conocido | Relación con lo nuestro | Estado |
|---|---|---|
| Repetición de datos viejos | todos los brazos la usan (5 % general) | hecho |
| Aislamiento de parámetros (PackNet, congelar lo importante) | la consolidación por uso es una variante | hay que citarlo; H5 decide si la división suma algo |
| Adaptadores por tarea (LoRA por dominio, O-LoRA) | las hojas de bajo rango se parecen; V1 es justamente este control | falta |
| EWC y regularización | no corrido en fase 2 | opcional |
| MoE con adaptadores para aprendizaje continuo | precedente directo | citar |
| Fractal Generative Models (Li et al. 2025) | misma regla en cada escala, módulos más chicos abajo | nota en el vault |

## I. Infraestructura

| ID | Requisito | Estado |
|---|---|---|
| I1 | Checkpoint al cerrar cada dominio, bajado en partes | hecho |
| I2 | Cola con retome y supervisor | hecho |
| I3 | A lo sumo 2 sesiones de Colab a la vez | hecho (lanzadores) |
| I4 | Presupuesto antes de cada lote (saldo 207 unidades el 2026-10-22) | parcial |
| I5 | El reporte de ruteo nunca tira una corrida | hecho (`try`, error en el resultado) |
| I6 | Código de cada corrida identificable | parcial: el zip no guarda el commit; **agregar el hash al resultado** |

## P. Paper

- §4.12 del informe técnico: fase 1 hecha; fase 2 con la tabla completa al cerrar las 3 semillas.
- Figuras: frontera estabilidad-plasticidad (M4), árbol natural con niveles, autosimilitud por brazo.
- Los resultados negativos (H1, H2) entran con el mismo detalle que los positivos.

## Camino crítico

1. Cerrar la semilla 2 de la fase 2 y escribir §4.12.
2. Leer el piloto 8 contra las reglas de corte.
3. Hacer V1, V8, I6 y F3 antes de las 3 semillas del árbol.
4. Correr la etapa A y la etapa C; decidir H5 a H9.
5. Hacer V4 (órdenes) y V5 (Holm) solo sobre los brazos que ganen.
