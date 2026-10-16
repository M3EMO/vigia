# Condiciones, controles y criterio de éxito

Type: grilling
Status: resolved
Blocked by: 03, 05, 06, 07, 08

## Question

¿Qué condiciones corren (fijo, fijo con repetición, crecimiento estilo Lifelong-MoE, fractal completo, ablaciones), con cuántas semillas, qué métricas (exactitud media, BWT, FLOPs) y qué prueba estadística decide si el fractal gana? Cerrar este ticket cierra la spec.

## Answer

Resuelto con el usuario el 2026-10-15 y 2026-10-16 (grilling).

- Criterio principal: el fractal gana si su BWT supera al del fijo con repetición, en pares con la misma semilla y escenario.
- Prueba: una variante principal fijada de antemano (duda del router + fusión por similitud de salidas); bootstrap pareado con IC 95 % de la diferencia de BWT sobre sus 6 pares. Las otras variantes son exploratorias.
- Factorial: 1 base (Qwen2-MoE) × 2 escenarios × 14 condiciones × 3 semillas = 84 corridas, unas 410 unidades.
- Condiciones: fijo; fijo con repetición; Lifelong-MoE; fractal 3 disparadores × 3 fusiones; ablaciones de la principal sin fusión y sin repetición.
- Métricas secundarias: exactitud media final, olvido, transferencia hacia adelante (razón de pasos contra la tarea sola), pasos totales y expertos finales por capa.
- La evaluación reserva sus entradas: el entrenamiento nunca ve un ejemplo de evaluación.

## Comments

- 2026-10-15 (usuario): la métrica incluye la transferencia hacia adelante, además del BWT. La pregunta es si las tareas aprendidas aceleran las siguientes (p. ej. `sumamult` reutiliza suma y multiplicación). La base son los pasos hasta 88 % de cada tarea entrenada sola (calibraciones 2 a 5, una semilla); falta decidir cómo se resume y si entra al criterio de éxito o queda como medida secundaria.
- 2026-10-15 (usuario): la evaluación reserva sus entradas; el entrenamiento nunca ve un ejemplo de evaluación (`mf_moe.train.sample`). Antes, `bin` y `modmult` habían visto casi toda su evaluación.
- 2026-10-15: costo estimado del factorial ~940 unidades (85.000 pasos por corrida, ~3,5 h de L4, 1,6 unidades/h), sobre el techo de 550. Se mide cuántas corridas caben en paralelo en una L4 antes de recortar.
- 2026-10-16 (usuario), criterio principal: el fractal gana si su BWT supera al del fijo con repetición, en pares con la misma semilla, base y escenario.
- 2026-10-16 (usuario), prueba: una variante principal fijada de antemano, con disparador por duda; decide un bootstrap pareado con IC 95 % de la diferencia de BWT (fractal menos fijo con repetición) sobre los 12 pares; las demás variantes son exploratorias, sin corrección por multiplicidad.
- 2026-10-16 (usuario), variante principal: disparador por duda del router + fusión por similitud de salidas (estilo HC-SMoE).
- 2026-10-16 (usuario), presupuesto: una sola base (Qwen2-MoE). El factorial queda en 1 base × 2 escenarios × 14 condiciones × 3 semillas = 84 corridas, unas 410 unidades (≈ 4,9 por corrida). El MoE propio no aportaba velocidad (perfil en L4). Cada variante queda con 6 pares para el bootstrap, no 12.
- 2026-10-16 (usuario), condiciones (14): fijo; fijo con repetición; crecimiento estilo Lifelong-MoE (agrega expertos al empezar cada tarea); fractal 3 disparadores (duda, pérdida, ambas) × 3 fusiones (salidas, pesos, ambas) = 9; ablaciones de la principal: sin fusión y sin repetición.
- 2026-10-16 (usuario), métricas secundarias: exactitud media final, olvido, transferencia (pasos de la tarea en fila / pasos de la tarea sola), pasos totales y expertos finales por capa.
- 2026-10-16 (usuario), Lifelong-MoE fiel al paper: copia del experto más usado (Net2WiderNet), congela expertos y filas del router viejos al crecer, destila con λ = 0,1 contra el modelo del final de la tarea anterior, sin repetición ni fusión. Calendario: un experto por capa al empezar las tareas 2, 4, 6, 8, 10 y 12 (6 → 12). Entre dos crecimientos sigue entrenando el experto más nuevo. Código: `mf_moe/lifelong.py`.
- 2026-10-16, piloto en L4 (4 tareas: suma, digmult, binsuma, ordsuma; semilla 0, con nombre): fijo BWT −0,92 / exactitud 0,23; fijo con repetición −0,52 / 0,51; Lifelong-MoE −0,91 / 0,23; fractal duda+salidas −0,42 / 0,50 (60 divisiones, 28 fusiones, capas de 8 a 12); fractal pérdida+pesos −0,32 / 0,44; fractal ambas+ambas −0,48 / 0,37 (casi no crece). Sin errores de GPU. Entrenar en fila costó 2,3 a 3,3 veces los pasos de las tareas solas.
- 2026-10-16, recorte por presupuesto (decisión del agente, con delegación del usuario): saldo 571,8 unidades, unas 488 dentro del techo. Una corrida completa ronda 220.000 pasos, 8,7 h y 14,5 unidades, así que entran unas 33. Factorial de 30 corridas: fase 1 principal contra fijo con repetición (3 semillas × 2 escenarios = 12, la prueba principal intacta); fase 2 fijo y Lifelong-MoE (12); fase 3 las dos ablaciones con nombre (6). Las 8 variantes exploratorias salen. Lista: `mf-moe/colab/factorial.txt`; colas `colab/queue.sh` en dos VMs.
- 2026-10-16, primer par del factorial (semilla 0, con nombre): fijo con repetición BWT −0,68, exactitud 0,29, 96.500 pasos; fractal principal BWT −0,42, exactitud 0,10, 185.750 pasos. La fusión al cierre de cada tarea daña la tarea recién aprendida (diagonal de R hasta 0,01); el BWT clásico, medido después de la fusión, esconde ese daño. **Desvío del preregistro** (decisión del agente con delegación del usuario): el criterio principal pasa a `bwt_aprendido`, medido desde la exactitud antes de fusionar; las corridas viejas usan la cota max(R[j][j], 0,88), que favorece al fractal. Con esa medida, el primer par da −0,80 contra −0,68: pierde el fractal. Se agrega la condición «fusión con recuperación» y se adelanta la ablación sin fusión; las 36 corridas pendientes corren en una cola compartida (unas 380 unidades).
