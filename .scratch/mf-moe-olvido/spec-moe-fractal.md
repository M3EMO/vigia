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
| H2 | Fusionar expertos ayuda | `division+fusion_alineada` < `division` | refutada en fase 1 y fase 2: −0,078, peor en las 3 semillas, p 0,042 (Holm 0,21) |
| H3 | La duda elige mejor qué dividir que el azar | `division+padre_congelado` < `azar+padre_congelado` | a favor pero no significativo: +0,026 en las 3 semillas, p 0,13, Holm 0,53 |
| H4 | Proteger lo usado reduce el olvido | `division+consolidacion_uso` < `control` | confirmada: +0,169 [0,145; 0,182], p 0,005, Holm 0,029; aprende 0,025 peor (p 0,008) |
| H5 | La división aporta algo sobre la consolidación | `division+consolidacion_uso` < `control+consolidacion_uso` | refutada: con división olvida más en las 3 semillas (−0,056, p 0,08, Holm 0,40) |
| H6 | Elegir por uso importa | `control+consolidacion_uso` < `control+consolidacion_azar` | a favor en las 3 semillas (+0,110, p 0,025, Holm 0,17); el azar solo ya baja 16 % |
| H7 | El árbol fractal olvida menos, o aprende mejor con el mismo olvido, que el plano con la misma consolidación | `arbol+consolidacion` contra `control+consolidacion_uso` | refutada con 3 pares: empate con la consolidación plana (−0,001, p 0,94) y aprende 0,024 peor; contra el control de bajo rango, +0,018 (p 0,36) y aprende 0,013 mejor (p 0,036) |
| H8 | El árbol se vuelve más autosimilar que el plano | `ruteo.fractal.autosimilitud` del árbol > la del plano | refutada: autosimilitud del árbol 0,025 contra 0,052 del plano, en las 3 semillas |
| H9 | La duda elige dónde abrir rama mejor que el azar | `arbol+consolidacion` < `arbol_azar+consolidacion` | solo s0 por saldo: duda 0,495 contra azar 0,528 (s1 y s2 en pausa, 2026-10-24) |
| H10 | La recursión (lo fractal) suma algo sobre un árbol de un solo nivel | `arbol+consolidacion` < `arbol_un_nivel+consolidacion` (las hijas no pueden volver a dividirse), con la misma cantidad de mitosis | a favor sin significancia: +0,035 [+0,010; +0,082], las 3 semillas a favor de la recursión, p 0,28 |
| H11 | Los territorios de expertos son autosimilares | `territorios`: el perfil de escalas se repite en los niveles 1, 2, 4, 8 y en los expertos solos, y se aparta del nulo gaussiano | refutada: dimensión ~13 (Granite) y ~20 (OLMoE), mitad del nulo gaussiano, pero igual a una partición al azar (dimensión ±2, extensión ±3 %) en todos los niveles; el perfil no se repite |

### Cómo se lee el conjunto (eficiencia de buscar una arquitectura fractal)

Eficiencia = olvido evitado por parámetro nuevo entrenable (`parametros_entrenables_expertos`) y por segundo de
paso (`segundos_por_paso`), con la pérdida aprendida dentro de 0,02 nats del control.

| H11 (¿hay fractal en los datos?) | H7 y H10 (¿construirlo ayuda?) | Conclusión |
|---|---|---|
| sí | sí | la arquitectura fractal es eficiente: copia una estructura que existe y la protege con hijos baratos |
| sí | no | el fractal describe al modelo pero no sirve como regla de diseño: el MoE plano ya lo aprende solo |
| no | sí | la ventaja viene de otra cosa (proteger, hijos de bajo rango), no de la autosimilitud |
| no | no | la idea fractal pierde su base; queda la consolidación como resultado |

### Sistema candidato en lugar de dividir y fusionar (2026-10-23)

"Consolidar, proteger el ruteo y brotar barato". Cada pieza tiene su evidencia o su prueba:

| Pieza | Evidencia hoy | La decide |
|---|---|---|
| Repetición de datos viejos (5 %) | base de todos los brazos | — |
| Consolidación por uso al cerrar cada dominio, sobre un MoE fijo | sin división: −31 %, Holm 0,006; dividir resta (H5) y el uso suma sobre el azar (H6) | hecho |
| Proteger el ruteo de lo congelado | R5: el plano pierde 19 % del tráfico viejo; el árbol, 1 % (piloto 8b) | H7 (árbol contra plano) |
| Capacidad nueva de bajo rango, no expertos enteros | V1: aprende 0,01-0,03 peor con ~26 veces menos parámetros nuevos (piloto 8) | H7 con V1 |
| Recursión (lo fractal) | dentro del árbol suma 0,035 (3 de 3 semillas, p 0,28); el árbol no le gana al plano (H7) | hecho |
| Proteger la atención (69 % del olvido en el diagnóstico) | sin datos | etapa B, después del circuito |
| Fusión y división por calendario | refutadas (H1, H2) | se descartan |

## V. Validez del experimento

| ID | Requisito | Porqué | Medida o test | Aceptación | Estado |
|---|---|---|---|---|---|
| V1 | Control con los mismos parámetros entrenables que el árbol | una hoja de bajo rango tiene ~57 mil parámetros y un experto entero ~1,5 M; sin este control, "olvida menos" puede ser "aprende menos" | brazo `control_bajo_rango+consolidacion`: los 16 expertos nuevos son copias fijas + U·V (rango 16), ruteo plano; `parametros_entrenables_expertos` en el resultado | mismos parámetros nuevos que el árbol; el conteo final se compara | hecho (mf-moe 54f1f51; test `test_low_rank_control_matches_the_tree...`) |
| V2 | Mismo cómputo por token | las compuertas suman cómputo | `segundos_por_paso` en el resultado | diferencia < 5 % o reportada | hecho (54f1f51) |
| V3 | Poder estadístico | el ruido entre semillas del control es ±0,01 nats | bootstrap pareado y t pareada | con 2-3 pares el IC del bootstrap es casi el rango de las diferencias y ningún p sobrevive a Holm (consolidación contra control: p_t 0,07, Holm 0,44 con 2 pares). Las comparaciones clave pasan a 5 semillas | decidido; las semillas 3 y 4 entran al plan de presupuesto |
| V4 | Más de un orden de dominios | hoy todas las semillas usan finés → turco → indonesio → suajili; el olvido depende del orden | `ORDERS` en `phase2.py` (f2, f2o1, f2o2), elegido por el tercer campo de la cola | el signo del efecto se mantiene en los 3 órdenes | mecanismo hecho (54f1f51); corridas postergadas: H10 tomó su presupuesto (decisión del usuario 2026-10-22). Límite declarado en el paper: un solo orden de idiomas |
| V5 | Comparaciones múltiples | hay ~11 comparaciones | t pareada y Holm sobre la métrica primaria en `analysis2` | las conclusiones del paper sobreviven a Holm; el resto se marca exploratorio | hecho (54f1f51) |
| V6 | Mismos datos en todas las corridas | | sha256 en `result.json` y chequeo en `analysis2` | idénticos | hecho |
| V7 | Evaluación sin fuga | | split por documento (`prepare_phase2`) | ningún documento en train y eval | hecho |
| V8 | Hiperparámetros justos para cada brazo | el lr (6e-5) se calibró para expertos enteros; los deltas U·V pueden pedir otro | `Cfg2.lowrank_lr_mult` (grupo propio en AdamW); el piloto 8b barrió ×1, ×3 y ×10 | el árbol usa el mejor de los tres, elegido por pérdida aprendida, no por olvido | hecho: empate (2,971/2,864 contra 2,971/2,866 y 2,973/2,865); queda ×1 |
| V9 | Retomes registrados | retomar pone los momentos de AdamW en cero y cambia la trayectoria | `retomes` en el resultado | los brazos no difieren en retomes de forma sistemática, o se reporta | hecho (54f1f51) |
| V10 | Todo resultado se publica | evita reportar solo lo que gana | tabla completa en §4.12 del paper | todas las corridas en la tabla | hecho (fase 1), en curso (fase 2) |

## M. Medidas

| ID | Medida | Para qué | Estado |
|---|---|---|---|
| M1 | Olvido medio (nats) de los 3 primeros idiomas | medida primaria | hecho |
| M2 | Pérdida aprendida media | plasticidad: ¿aprende cada idioma nuevo? | hecho |
| M3 | Pérdida general final (FineWeb-Edu) | ¿daña lo que el modelo ya sabía? | hecho |
| M4 | Frontera estabilidad-plasticidad | un brazo que olvida menos pero aprende peor no gana solo por M1; se grafica M1 contra M2 | **falta (figura)** |
| M5 | Transferencia hacia adelante | ¿aprende más rápido el idioma nuevo? `curva`: pérdida media cada 50 pasos | medida hecha (54f1f51); falta el análisis |
| M6 | Evaluación externa | la pérdida no es todo; una tarea (p. ej. Belebele por idioma) confirma | falta, opcional |
| M7 | Especialización y fractal | `ruteo.fractal` (familia/idioma/documento), exceso sobre nulo multinomial | hecho (fase 2b en adelante) |
| M8 | Costo | parámetros entrenables, VRAM, segundos por paso, unidades de Colab | parcial |

## R. Ruteo

| ID | Requisito | Medida o test | Aceptación | Estado |
|---|---|---|---|---|
| R1 | Sin colapso del enrutador | `ruteo.entropia_uso`, `ruteo.hojas_muertas` | hojas muertas ≤ 25 % por capa | hecho (medida); regla de corte en el piloto 8 |
| R2 | Compuertas balanceadas | sesgo por compuerta (test); `ruteo.forma.compuerta_reparto_max` al final | ninguna compuerta > 95 % a una hija | hecho; regla de corte del piloto 8 |
| R3 | Sin inferencia dispar | `ruteo.train_vs_eval`, `ruteo.bf16_vs_fp32` | ≥ 0,99 | hecho |
| R4 | Deriva del ruteo de lo viejo medida | `ruteo.deriva` (Jaccard top-k al cerrar el dominio contra el final) | se reporta; la consolidación sin deriva baja es sospechosa | hecho |
| R5 | El balanceo no aleja tokens de los expertos congelados | `ruteo.uso_congelados`: parte de los tokens de cada dominio viejo que va a congelados, al cerrarlo y al final | la caída de uso < 20 % | hecho (54f1f51); piloto 8: −19 % en los controles, al límite |
| R6 | Árbol sin mitosis = Granite | test `test_without_mitosis_the_tree_routes_like_granite` | logits iguales | hecho |
| R7 | Mitosis sin cambio de salida | test `test_mitosis_and_nested_mitosis_keep_the_output...` | logits iguales | hecho |

## O. Optimización

| ID | Requisito | Test | Estado |
|---|---|---|---|
| O1 | Filas congeladas no se mueven (gradiente y momentos en cero, sin weight decay) | `test_training_moves_only_the_low_rank_delta_and_the_gates`, tests de consolidación | hecho |
| O2 | Deltas nacen en cero (U = 0) | test de mitosis | hecho |
| O3 | lr de los deltas calibrado | ver V8 | hecho (×1) |
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
| F2 | Nivel documento en el árbol natural | código hecho (758e2d6); corre en la VM del piloto 8 |
| F3 | Forma del árbol: hojas por profundidad y reparto máximo de compuerta | hecho (`ruteo.forma`, 54f1f51) |
| F4 | Árbol natural con estonio y DeepSeek-V2-Lite (descarga a CPU) | estonio hecho (piloto 8); DeepSeek no entra en la L4: postergado |
| F5 | Árbol aprendido por gradiente (candidato 3) | falta, después del circuito |
| F6 | Árbol natural como punto de partida (principio 1 de la síntesis) | falta: hoy las raíces son los 32 expertos sin agrupar |
| F7 | La figura de cada experto: `territory_shape` mide la nube de estados que el router le manda (dimensión intrínseca, de correlación y lineal, perfil de escalas, nulo gaussiano). Autosimilitud de territorios: el mismo perfil en cada nivel del dendrograma de expertos | medida hecha (mf-moe 758e2d6, probada con plano, Sierpinski y gaussiana); corre sobre Granite y OLMoE en la VM del piloto 8 |

## C. Mecanismos (en orden)

1. Etapa A: consolidación sin división, por uso y al azar (en curso, `fase2b.txt`).
2. Piloto 8 y 8b: árbol + consolidación, con reglas de corte (hecho: el 8 encontró el NaN de la compuerta, el 8b pasó).
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
| I6 | Código de cada corrida identificable | hecho: `colab/build_zip.sh` arma el zip desde HEAD con `COMMIT`; `result.json` guarda `commit` |
| I7 | Checkpoints que sobreviven al retiro de la VM | hecho: copia a Drive desde la VM (rclone, scope drive.file, autorizado por el usuario 2026-10-23); la PC ya no sube 3,5 GB si Drive tiene uno igual o más nuevo |
| I8 | Un job trabado no queda horas gastando | hecho: el runner relanza si el log remoto no cambia en 75 min (vigia 69b02f9) |
| I9 | Reintento de corridas fallidas antes del resto de la lista | hecho a mano (`.cache/jobs/reintentar.sh`): corta la cola al terminar su corrida y la cola relee la lista desde arriba |

## Registro de cambios de la spec

- 2026-10-22: creación; huecos V1, V2, V4, V5, V8, V9, I6, R5, F3, M5 resueltos; H10 reemplaza a V4 (usuario).
- 2026-10-22: F7 (figura del territorio) y H11; tabla de lectura de la eficiencia fractal.
- 2026-10-26: circuito cerrado. H7, H8 y H11 refutadas; H10 a favor sin significancia. Fila de lectura "no, no": queda la consolidación por uso sobre un MoE fijo.
- 2026-10-23: fase 2 completa (H1-H4); primero se cierra el circuito (usuario); sistema candidato; sección T.
- 2026-10-24: etapa A completa (H5 refutada, H6 a favor); T1 y T4 verificados; H9 recortada a s0 por saldo.
- 2026-10-23: T4 roto en PRODUCTION y arreglado (duda total, compuertas acotadas); T1 con dos fallas arregladas;
  I7-I9 (Drive, detector de cuelgues, reintentos); presupuesto rehecho.

## T. Lo que nunca corrió en GPU (se verifica en su primera corrida real)

| ID | Pieza | Riesgo | Cómo se verifica |
|---|---|---|---|
| T1 | Drive desde la VM (`drive_sync`, `phase2_vm.py`, runner) | dos fallas encontradas y arregladas: carpeta común por `$5` (vigia 81b798a) y un cuelgue de 3,5 h en el bloque de Drive antes de entrenar (timeouts, mf-moe 24f64ca). Primera retoma real desde Drive: control_bajo_rango s0 (2:0) | `runner.log` dice "checkpoint en Drive" en un relanzo; `fase2.log` dice "checkpoint ... desde Drive"; `rclone lsf` muestra la carpeta de la corrida Verificado 2026-10-24: uso s0 perdió la VM a las 22:20 UTC y retomó 1:0 desde Drive en 30 s, sin subir nada; control_bajo_rango s0 retomó 2:0 desde Drive. |
| T2 | `arbol_un_nivel+consolidacion` (H10) | solo probado en CPU | `ruteo.forma`: ninguna hoja por debajo de la profundidad 1 |
| T3 | `arbol_azar+consolidacion` (H9) | solo probado en CPU | sin NaN y con las reglas de corte del piloto 8b |
| T4 | Árbol con la configuración PRODUCTION (48 hojas, 4 idiomas, clones silenciosos) | el piloto 8b usó 2 idiomas y otra configuración | arbol s0 rompió R1 y R2 (cadenas de 11 niveles, 31 % de hojas muertas, compuerta al 100 %); arreglado en mf-moe 48056ac (duda total y compuertas acotadas); se verifica otra vez en la nueva arbol s0 |
| T5 | Territorios con nulo de partición (H11) | código probado solo en CPU | `territorios` trae `extension_azar` y `correlacion_azar` |

## P. Paper

- §4.12 del informe técnico: fase 1 hecha; fase 2 con la tabla completa al cerrar las 3 semillas.
- Figuras: frontera estabilidad-plasticidad (M4), árbol natural con niveles, autosimilitud por brazo.
- Los resultados negativos (H1, H2) entran con el mismo detalle que los positivos.

## Presupuesto (saldo 203 unidades el 2026-10-22; ~5,7 unidades por corrida completa)

| Bloque | Corridas | Unidades | Estado |
|---|---|---|---|
| Fase 2, semilla 2 restante | 4 | ~23 | en curso |
| Pilotos 8 y 8b (barrido V8, control V1, reglas de corte) | 9 cortas | ~5 | hecho |
| Etapa A, semillas 0-2 | 6 (+3 repetidas por retiros) | ~45 | en curso |
| Etapa C: árbol, árbol al azar, control de bajo rango y árbol de un nivel (H10), semillas 0-2 | 12 | ~68 | después del piloto 8 |
| V3: semillas 3 y 4 de control, division+consolidacion_uso, control+consolidacion_uso y árbol | 8 | ~46 | después de la etapa C |
| V4: órdenes f2o1 y f2o2 | 0 | 0 | postergado (H10 tomó su lugar) |

Total ~174 unidades: entra en el saldo de 203. Un piloto que falla libera las 68 de la etapa C.

Actualización 2026-10-23 20:40 UTC: saldo 94 unidades. Las pérdidas de la noche y la tarde (retiros de VM,
subida a ~70 KB/s, un cuelgue de 3,5 h, tapa cerrada cuatro veces, arbol s0 repetida por el arreglo de T4)
costaron ~50 unidades. Quedan ~13 corridas (~75 unidades): consolidación por uso s0 y s1; control de bajo rango
s0 (desde 2:0), s1 y s2; árbol, árbol de un nivel y árbol al azar, semillas 0-2. Orden de corte si falta saldo:
primero arbol_azar (H9), después V3 entera.

Decisión del usuario (2026-10-23): primero se cierra el circuito (etapa A, etapa C y H11, ~98 unidades); la
robustez de V3 (semillas 3 y 4) se decide después según el saldo. Si el árbol no gana, V3 se concentra en la
consolidación. H11 corre sola al terminar la cola fase2b (unidad `mfmoe-h11`, zip caf49a8).

## Camino crítico

1. Cerrar la semilla 2 de la fase 2 y escribir §4.12.
2. Leer el piloto 8 contra las reglas de corte.
3. V1, V2, V4 (mecanismo), V5, V8 (mecanismo), V9, I6, R5, F3 y M5 (medida): hechos el 2026-10-22 (mf-moe 54f1f51).
4. Correr la etapa A y la etapa C; decidir H5 a H9.
5. Hacer V4 (órdenes) y V5 (Holm) solo sobre los brazos que ganen.
