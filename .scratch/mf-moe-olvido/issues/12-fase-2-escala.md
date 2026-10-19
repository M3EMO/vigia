# Fase 2: ¿el MoE fractal funciona a escala, con un MoE preentrenado?

Type: grilling
Status: resolved
Blocked by: 11

## Question

La fase 1 (MoE chico desde cero, 12 tareas de dígitos) muestra que la fusión ingenua destruye lo recién
aprendido y que la división sola casi empata con el control. Objeción del usuario (2026-10-17): el resultado
puede depender de la escala y del preentrenamiento, porque en un MoE preentrenado los expertos ya están
especializados. ¿Qué modelo, qué datos, qué condiciones y qué presupuesto prueban esa objeción en una L4?
Ninguna corrida entra a Colab hasta cerrar este ticket.

Research: nota del vault «mf-moe fase 2: probar la escala con un MoE preentrenado en una L4» (2026-10-17).

## Propuesta

- Modelo: Granite-3.1-1B-A400M-base (1,3 B totales, 0,4 B activos, 32 expertos × top-8 por capa, 24 capas,
  Apache-2.0, HF nativo `granitemoe`). Es el único MoE preentrenado que entra en una L4 con ajuste completo
  de pesos: AdamW de 8 bits, unos 8 a 13 GB. OLMoE-1B-7B solo entra con LoRA, y LoRA cambia el método.
- Datos: 4 dominios de texto en fila (matemática, código, biomédico, legal), unos 20 M de tokens por dominio;
  repetición de 5 % con texto general (FineWeb-Edu), que hace de dominio 0. Licencias por verificar.
- Olvido: loss held-out de 1 M de tokens por dominio después de cada dominio; olvido = loss final menos loss
  justo después de aprender el dominio. Es un oráculo computable, sin juez.
- Condiciones, todas con los mismos parámetros finales y top-8 fijo:
  - A, control: crece al tamaño final antes de empezar (32 → 48 expertos por clonado), con repetición.
  - B, división sola: 32 → 48 expertos con el disparador por duda, con repetición.
  - C, división + fusión alineada por permutación, con repetición.
  - D, opcional: división + congelar el padre (protege lo viejo, como Lifelong-MoE).
- Semillas: 3 por condición, pareadas con el mismo orden de datos.
- Precondición: un piloto de 1 h mide tokens por hora, VRAM y si el control olvida de forma medible. Si el
  control casi no olvida, las condiciones empatan por techo: se cambian dominios o tasa de aprendizaje.

## Costo estimado (sin medir)

| Parte | Horas de L4 | Unidades |
|---|---|---|
| Piloto | unas 8 | unas 12 |
| A, B y C con 3 semillas (9 corridas de ~6 h) | unas 54 | unas 83 |
| Retiros de VM (+15 %) | unas 9 | unas 14 |
| D con 3 semillas (opcional) | unas 18 | unas 28 |
| Total sin D / con D | 71 / 89 | 109 / 137 |

Presupuesto: la fase 1 todavía necesita unas 330 unidades de las ~420 disponibles dentro del techo de 550;
quedan unas 90 para la fase 2. Hace falta subir el techo, recortar la fase 1 o achicar la fase 2.

## Objeciones abiertas

- Cambiar de banco (dígitos → texto) y de escala a la vez mezcla dos explicaciones posibles.
- Ninguna evidencia publicada muestra que dividir con clon + ruido mejore la retención; lo que funciona agrega
  expertos y protege los viejos (Lifelong-MoE, LEGO). Drop-Upcycling sugiere reinicializar parte del clon.
- El throughput (~15-20 M tokens/h) y el layout de los tensores de expertos de Granite no están medidos.

## Decisiones abiertas para el usuario

1. ¿Modelo: Granite-1B con ajuste completo, u OLMoE con LoRA?
2. ¿Presupuesto: subir el techo, recortar la fase 1 o achicar la fase 2?
3. ¿Entra la condición D (congelar el padre)?

## Answer

Resuelto con el usuario el 2026-10-17.

- Modelo: Granite-3.1-1B-A400M-base con ajuste completo de pesos (AdamW de 8 bits).
- Presupuesto: sale de recortar las preguntas 5 y 6 de la fase 1 (fijo, Lifelong-MoE y sin repetición, 15
  corridas, unas 114 unidades); esas corridas quedan reservadas para una VM inexistente y la cola las saltea.
- La condición D (división con el padre congelado) entra.
- Antes de Colab: código del adaptador de Granite (división, fusión alineada, congelado) con tests en CPU
  sobre un modelo diminuto armado desde la config; datos y licencias verificados; piloto de 1 h que mida
  tokens por hora, VRAM y olvido del control.

## Comments

- 2026-10-17, datos elegidos (abiertos y sin registro previo): matemática `open-web-math/open-web-math` (ODC-By 1.0); código `common-pile/stackv2` (licencias abiertas); legal `common-pile/caselaw_access_project` (CC0 o dominio público); biomédico un subconjunto de abstracts de PubMed con CC0 (p. ej. `slinusc/PubMedAbstractsSubset`); repetición general `HuggingFaceFW/fineweb-edu` (ODC-By). Descartados: The Stack (pide aceptar términos con cuenta) y Pile of Law (licencia poco clara). El piloto verifica que los cinco se lean en streaming desde la VM.
- 2026-10-17, adaptador de Granite hecho (mf-moe fe77d60): router con buffers `bias` en cero y `load` de uso, y registro de la señal; división, fusión alineada y AdamW sirven sin cambios. Falta: datos en streaming, bucle por tokens con loss por dominio, condiciones A a D, retome, y el humo en CPU.
- 2026-10-17 (usuario), memoria: se entrenan las 12 capas superiores (expertos, router y atención) en fp32 con AdamW común; las 12 inferiores y los embeddings quedan congelados en bf16. Las capas entrenables crecen de 32 a 48 expertos. Cuenta: ajuste completo con AdamW fp32 y 48 expertos pide ~30 GB; con AdamW de 8 bits a 48, ~22,5 GB; a 40, ~19 GB, pero dividir, fusionar y alinear exigen reordenar momentos cuantizados por bloques. Con 12 capas, ~14-17 GB y el código ya probado sirve sin cambios.
- 2026-10-17 (usuario), retome: el checkpoint guarda solo los pesos entrenables en fp32 (~3,6 GB, ~20 min de subida a 3 MB/s); al retomar, AdamW arranca con momentos en cero y vuelve a calentar el learning rate (re-warming, Ibrahim et al. 2024). La parte congelada sale del modelo preentrenado. Con momentos, el checkpoint pesaría ~10,8 GB (~3,5 h de subida).
- 2026-10-17, datos: un job de preparación tokeniza los cinco datasets una sola vez (uint16, secuencias de 1.024 tokens, `.npy` sin pickle) con evaluación separada del entrenamiento; el archivo se baja una vez y se sube a cada VM, así todas las corridas ven los mismos datos.
- 2026-10-17, piloto 2, primer intento: la preparación de datos funcionó en 4 min (cinco dominios, hashes en `datos.json`; stackv2 dio 106 documentos para 5,2 M tokens, documentos muy largos). El control se quedó sin memoria en el primer `opt.step`: 19,5 GB ocupados, y AdamW foreach pidió temporales del tamaño de los momentos. La cuenta de activaciones con lote 8 (~8 GB en 12 capas, más ~3 GB de logits) tampoco entraba junto a los momentos. Arreglo (mf-moe 0ddcbdd): AdamW fused y micro-lotes de 2 con acumulación; el paso es el mismo que con el lote entero (test con tolerancia 1e-4). El piloto vuelve a correr con eso.
- 2026-10-17, piloto 2, resultados (micro-lotes de 2, AdamW fused; 500 pasos por dominio en el control, 50 en las otras dos):
  - Memoria: el control a 48 expertos llega a 22,78 GB (96 % de la L4); sin crecer, 16,86 GB.
  - Ritmo: 4.856 tokens/s con evaluaciones; una corrida de 4 dominios × 20 M tokens dura ~5 h; 12 corridas, ~85 unidades.
  - Olvido medible: el control olvida 0,061 nats de matemática al aprender código.
  - Hallazgo 1, crecimiento: clonar los 16 expertos más usados con la fila del router idéntica sube la pérdida inicial (+0,27 nats en matemática, +0,25 en general): padre y clon empatan en logit, el top-8 elige a los dos y desplaza a otro experto. El control arranca dañado y en 1.000 pasos no se recupera (general 2,256 contra 2,085 del preentrenado).
  - Hallazgo 2, dropout: la config de Granite trae `attention_dropout` 0,1. La pérdida de entrenamiento queda ~1 nat sobre la de evaluación, y 50 pasos de matemática suben la pérdida de matemática (1,688 → 1,727).
  - Hallazgo 3, el piloto no probó la división: revisa cada 250 pasos y las otras dos condiciones dieron 50 por dominio. La fusión alineada repitió los números de la división.
- 2026-10-17, spec del piloto 3 (`colab/pilot3_phase2.py`, ~40 min, ~1 unidad): (1) pérdida del preentrenado contra el control crecido con fila clonada y con fila en cero; (2) pérdida en modo entrenamiento con dropout 0,1, y 100 pasos de matemática con dropout 0,1 y 0; (3) división y fusión alineada con revisión cada 10 pasos; (4) memoria del control a 48 con micro-lotes de 2 y de 1. Reglas de decisión: dropout 0 si con 0,1 los 100 pasos no bajan la pérdida de matemática; fila en cero si su daño inicial queda bajo 0,05 nats y el de la fila clonada no; micro-lote 1 si con 2 la memoria supera 22 GB y con 1 baja; si la división o la fusión fallan en la GPU, se arreglan antes de encolar.
- 2026-10-17, piloto 3, resultados: (1) crecer con la fila del router clonada daña +0,271 nats en matemática y +0,253 en general; con la fila en cero, +0,001 y +0,001 → el control usa fila en cero (regla del spec). (2) Con dropout 0,1 la pérdida en modo entrenamiento es 3,99 contra 1,69 sin dropout → dropout 0. (3) 100 pasos de matemática suben la pérdida de matemática con dropout 0,1 (1,688 → 1,763) y también con 0 (→ 1,709): el modelo no aprende el dominio que entrena. (4) La división no ocurrió ni con revisión cada 10 pasos: ninguna capa supera z 3,5 en el modelo preentrenado. (5) El control a 48 con micro-lote 2 usa 22,88 GB.
- 2026-10-17, investigación del lote (pedido del usuario): Zhang et al. 2024 (ICLR 2025, arXiv 2410.21676) ajustan el lote crítico a B* ≈ 22,91 · D^0,47 secuencias de 512 tokens (D en millones de tokens; la unidad del paper es ambigua). Con 20 M tokens por dominio da ~48 k tokens (~47 secuencias de 1.024); con 80 M en total, ~92 k (~90 secuencias). Ibrahim et al. 2024 (TMLR, arXiv 2403.08763) re-entrenan con lote de 1.104 × 2.048 = 2,26 M tokens, pero sobre cientos de miles de millones de tokens; con 20 M tokens por dominio, ese lote daría ~9 pasos. Nuestro lote de 8 × 1.024 = 8 k tokens queda ~6 veces bajo el lote crítico estimado, y con Adam cada paso tiene tamaño completo aunque el gradiente sea casi ruido. Para Adam el lr escala ~ con la raíz del lote.
- 2026-10-17, spec del piloto 4 (`colab/pilot4_phase2.py`, ~75 min, ~2 unidades): (1) pérdida del preentrenado en la evaluación y en las últimas 200 secuencias de entrenamiento de cada dominio; (2) 3.200 secuencias de matemática sin crecer con (lote, lr) = (8, 2e-5), (64, 2e-5), (64, 6e-5), (128, 8e-5); (3) división y fusión alineada con umbral -1e9 y revisión cada 10 pasos, con el z más alto de cada revisión. Reglas: se elige la combinación con menor pérdida de matemática en la evaluación y en la cola; si ninguna baja la pérdida bajo la inicial, se frena y se re-especifica; si la evaluación y la cola difieren más de 0,1 nats en un dominio, la evaluación de ese dominio se rehace con secuencias mezcladas. El umbral de la división se re-especifica con los z medidos.
- 2026-10-18, piloto 4, resultados: (1) la evaluación de código daba 0,18 nats menos que la cola del entrenamiento (las primeras 1.000 secuencias venían de ~20 documentos) → la preparación reparte documentos enteros a la evaluación a lo largo de todo el stream (mf-moe 81a05b0). (2) Ninguna combinación de lote y lr aprende matemática en 3,3 M tokens; evaluación (inicio 1,688): lote 8 lr 2e-5 → 1,741; 64, 2e-5 → 1,702; 64, 6e-5 → 1,749; 128, 8e-5 → 1,752. El lote grande reduce el daño pero no produce aprendizaje: Granite ya aprendió estos dominios en 10 billones de tokens. Sin aprendizaje, el olvido medido sería deriva. Regla del spec: se frena y se re-especifica. (3) La división corre en la GPU, pero forzada cada 10 pasos destruye el modelo: 5 divisiones por capa suben matemática de 1,69 a 4,24 y los hijos se llevan el 42 % del ruteo; la fila del hijo en la dirección de la duda (fase 1) choca con un router preentrenado sin balanceo. El z más alto de la primera revisión fue 4,9, así que z 3,5 sí puede dispararse.
- 2026-10-18 (usuario), dominios: cuatro idiomas fuera de los 12 de Granite 3.1, de FineWeb-2 (ODC-By): finés (`fin_Latn`), turco (`tur_Latn`), indonesio (`ind_Latn`) y suajili (`swh_Latn`). General sigue en FineWeb-Edu. Cada idioma deja mucho por aprender y produce olvido fuerte (Ibrahim et al. 2024, cambio inglés → alemán).
- 2026-10-18, spec del piloto 5 (`colab/pilot5_phase2.py`, ~70 min, ~2 unidades): (1) pérdida del preentrenado en cada idioma; (2) daño y uso de 16 divisiones por capa sin entrenar con la fila del hijo `mix` 1 (fase 1), 0,1 y 0 (clon); (3) finés y después turco sin crecer, 1.600 secuencias cada uno, lote 64, lr 2e-5 y 6e-5; (4) división con el umbral real (z 3,5, revisión cada 5 pasos) con `mix` 0,1 y 1. Reglas: se elige el lr con menor pérdida aprendida si la pérdida del idioma baja al menos 0,1 nats; si ningún lr la baja, se frena; se elige el `mix` más alto cuyo daño estático en general quede bajo 0,05 nats; el olvido de finés tras turco tiene que superar 0,02 nats para seguir; el umbral se re-especifica con los z medidos.
- 2026-10-18, piloto 5, resultados (parciales; faltan las dos divisiones con la fila vieja): (1) pérdida inicial: suajili 3,27, finés 3,20, turco 2,95, indonesio 2,42, general 2,14 (la evaluación general cambió de documentos). (2) 16 divisiones por capa sin entrenar dañan siempre: general 2,14 → 5,40 con `mix` 1, 3,14 con 0,1 y 2,67 con el clon puro; ninguna fila pasa la regla de 0,05. Causa: el router de Granite elige 8 por logit y no balancea durante el ajuste; un hijo elegido desde el primer paso desplaza a otros expertos. (3) Finés y después turco, lote 64: con lr 2e-5 finés baja 3,20 → 3,01 y olvida 0,26; con 6e-5 baja a 2,97 y olvida 0,62 (general +0,06). Regla: lr 6e-5. Hay aprendizaje y olvido de sobra para medir.
- 2026-10-18, decisión de diseño (balanceo de nacidos): el router de Granite elige con un sesgo sobre la softmax, como el router de la fase 1, pero solo los expertos nacidos en la corrida tienen balanceo; los 32 de Granite conservan su ruteo. Cada experto nuevo (hijo o clon del control) nace con sesgo -1 y el balanceo lo recluta de a poco. Control y fractal crecen con el mismo mecanismo; la diferencia es cuándo y dónde. Código: `RecordingGraniteRouter.u/n_born`, `split(silent=True)`, `Cfg2.balance_u/load_decay/silent`.
- 2026-10-18, spec del piloto 6 (`colab/pilot6_phase2.py`, ~45 min, ~1,2 unidades): finés y turco, 1.600 secuencias cada uno, lote 64, lr 6e-5, u 2,5e-3 y decay 0,99 (recluta en ~1/4 de dominio en este piloto comprimido): (1) control con 16 clones silenciosos por capa; (2) división con padre congelado e hijo silencioso, `mix` 0 y 0,1, z 3,5, revisión cada 5 pasos. Referencia: sin crecer del piloto 5. Reglas: el crecimiento silencioso sirve si la pérdida general al final queda a menos de 0,05 nats de la referencia sin crecer (2,20) y el uso de los nacidos supera 0; se elige el `mix` con menor pérdida aprendida entre los que cumplen; si la división no se dispara con z 3,5, el umbral pasa al percentil que dé ~16 divisiones por capa en la corrida completa.
- 2026-10-18 (usuario), quinto brazo: `azar+padre_congelado` divide con el mismo calendario que `division+padre_congelado`, pero elige al padre al azar. Si D gana al control y empata con este brazo, el mérito es del congelado y no de la duda (antecedente: Lifelong-MoE congela expertos viejos y agrega nuevos). Diseño: 5 brazos × 3 semillas = 15 corridas.
- 2026-10-18, calendario de la división: en los idiomas ninguna revisión superó z 3,5 (piloto 5: z máximo 2,5-3,1). Por la regla del spec, todas las condiciones con división usan un calendario fijo: una división por capa en cada revisión (umbral -1e9) hasta 48 expertos, con revisiones espaciadas para que las 16 divisiones se repartan entre los 4 dominios. La duda elige qué experto se divide; el brazo al azar lo elige al azar.
- 2026-10-18, piloto 6, resultados (finés y turco, 1.600 secuencias, lote 64, lr 6e-5; referencia sin crecer: finés 2,971, olvido 0,625, general final 2,200): control con 16 clones silenciosos por capa: finés 3,017, olvido 0,635, general 2,258, clones con 34 % del ruteo (su parte justa). División silenciosa forzada (10 por capa) con padre congelado: `mix` 0 → finés 2,975, olvido 0,620, general 2,283, hijos con 20 %; `mix` 0,1 → 2,974, 0,625, 2,282. El reclutamiento funciona y el daño general baja de +0,5 a +0,06-0,08, pero no pasa la regla de 0,05. Se acepta igual: es el mecanismo de menor daño de los probados y es el mismo para todos los brazos que crecen, así que no sesga la comparación. `mix` 0,1 (pérdida aprendida apenas menor y la duda conserva un papel en la fila del hijo). Congelar 10 de 42 padres no cambió el olvido en este piloto corto. Además el control falló una vez por memoria: calibraba con el lote entero de 64; las calibraciones ahora usan 8 secuencias, sin cabeza y sin grafo.
- 2026-10-18, SPEC DE LAS 15 CORRIDAS (`colab/fase2.txt`, `colab/phase2_vm.py`, `mf_moe.phase2.PRODUCTION`):
  - Datos: finés, turco, indonesio y suajili de FineWeb-2 (~10 M tokens cada uno, 152 pasos de 64 secuencias) en ese orden; general FineWeb-Edu como repetición (5 %, 3 de 64). Evaluación: 500 secuencias por dominio, documentos enteros repartidos en el stream. Cada VM tokeniza igual; result.json guarda los sha256 y el análisis comprueba que coincidan.
  - Modelo: Granite-3.1-1B-A400M, capas 12-23 entrenables en fp32, dropout 0, micro-lote 1, AdamW fused, lr 6e-5 con recalentamiento de 15 pasos en cada dominio.
  - Brazos (5 × semillas 0, 1, 2): control (16 clones silenciosos por capa al inicio), division, division+fusion_alineada, division+padre_congelado, azar+padre_congelado. División por calendario: una por capa cada 38 pasos (16 en total, 4 por dominio); `mix` 0,1; hijos silenciosos; balanceo de nacidos u 4e-4, decay 0,99.
  - Métrica primaria: olvido medio (pérdida final − pérdida al terminar cada dominio, promedio de los 3 primeros dominios). Secundarias: pérdida aprendida media, pérdida general final, uso de los nacidos. Comparación pareada por semilla contra el control, bootstrap IC95; veredicto con 3 pares. Pregunta de la duda: division+padre_congelado contra azar+padre_congelado, pareada por semilla.
  - Costo: ~3,5 h por corrida (40 M tokens a ~4.200 tokens/s, evaluaciones y preparación de datos), ~52 h y ~80 unidades en total. Saldo: 431 unidades. Corre en una VM (g2) mientras la fase 1 termina en la otra (fa); al terminar la fase 1, la segunda VM se suma a la lista.
  - Retome: checkpoint de 3,6 GB al cerrar cada dominio; bajada con tope de 40 min (CKPT_TIMEOUT).
- 2026-10-19 (usuario), SPEC DEL DIAGNÓSTICO "¿dónde vive el olvido?" (`colab/diag_phase2.py`, ~30 min, ~1 unidad; corre en la VM de la fase 1 mientras la fase 2 sigue en g2). Motivo: congelar padres no cambió el olvido (piloto 6), y en todos los brazos la atención de las capas 12-23 entrena; si el olvido vive fuera de los expertos, los cinco brazos empatan por construcción. Protocolo: sin crecer, lote 64, lr 6e-5, 1.600 secuencias; entrena finés (θ1) y turco (θ2); desde θ2 devuelve a θ1 un grupo de las capas entrenables (expertos, router, atención, normas) y también "todo menos el grupo"; mide finés, turco y general (200 secuencias), la deriva relativa de cada grupo y la fracción del top-8 de los tokens de finés que se conserva por capa. Recuperación = (finés(θ2) − finés(restaurado)) / (finés(θ2) − finés(θ1)). Reglas: si restaurar solo los expertos recupera ≥ 50 % del olvido, la fase 2 sigue como está; si los expertos recuperan < 50 % y la atención o el router recuperan más, se frena la fase 2 y se re-especifica con plasticidad guiada por duda (proteger expertos y filas del router de lo viejo, atención con lr bajo o congelada, gradiente de los tokens con duda hacia expertos nuevos). Nota: el segundo dominio arranca con AdamW nuevo (en la fase 2 el optimizador sigue entre dominios); alcanza para ubicar el olvido.
- 2026-10-19, falla de la fase 2: la primera corrida (control, semilla 0) se relanzó 7 veces sin pasar del primer dominio. Causa: `colab download` carga el checkpoint entero en RAM; con 3,6 GB el runner llegó a 10 GB y el OOM killer lo mató en cada cierre de dominio. Arreglo (vigia 26c3527): la VM parte el checkpoint y el runner lo baja en partes de 64 MB, con reintentos y control de tamaño; probado con un `colab` falso. Además la PC se reinició y mató todos los servicios. Dato parcial útil: con ~10 M tokens el control baja el finés de 3,20 a 2,55, y el turco sube de 2,95 a 3,76 (interferencia fuerte). La fase 2 queda en pausa hasta el diagnóstico.
- 2026-10-19, DIAGNÓSTICO, resultados (finés θ1 2,955 → θ2 3,669 tras turco; olvido 0,714). Recuperación del olvido de finés al devolver a θ1: solo expertos 96 % (turco 2,752 → 2,969); todo menos el router 97 %; todo menos la atención 98 %; todo menos los expertos 72 %; solo atención 69 %; solo router 6 %; solo normas 0 %. Deriva relativa: router 6,4 %, expertos 1,7 %, atención 1,3 %, normas 0,003 %. Ruteo del finés: cada capa entrenable conserva 84-96 % de su top-8. Lectura: el olvido vive sobre todo en los pesos de los expertos; la atención también pesa y los dos efectos no se suman; el router casi no participa aunque se movió más. Regla del spec: los expertos recuperan ≥ 50 % → la fase 2 sigue como está. Consecuencia de diseño: proteger expertos puede funcionar, pero hay que proteger los que usa lo viejo; el brazo con padre congelado protege 4 por dominio y capa elegidos por la duda del dominio nuevo, que no son los que guardan lo viejo (explica el piloto 6).
- 2026-10-19, arreglo de la bajada (vigia 1d1c7ec, mf-moe 3752e41): el job escribe el checkpoint también en partes de 64 MB con `<ruta>.partes`; el runner las baja sin exec (un exec queda en cola detrás del job). Probado con un `colab` falso. La fase 2 se relanzó en g2.
