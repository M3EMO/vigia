# Condiciones, controles y criterio de éxito

Type: grilling
Status: open
Blocked by: 03, 05, 06, 07, 08

## Question

¿Qué condiciones corren (fijo, fijo con repetición, crecimiento estilo Lifelong-MoE, fractal completo, ablaciones), con cuántas semillas, qué métricas (exactitud media, BWT, FLOPs) y qué prueba estadística decide si el fractal gana? Cerrar este ticket cierra la spec.

## Comments

- 2026-10-15 (usuario): la métrica incluye la transferencia hacia adelante, además del BWT. La pregunta es si las tareas aprendidas aceleran las siguientes (p. ej. `sumamult` reutiliza suma y multiplicación). La base son los pasos hasta 88 % de cada tarea entrenada sola (calibraciones 2 a 5, una semilla); falta decidir cómo se resume y si entra al criterio de éxito o queda como medida secundaria.
- 2026-10-15 (usuario): la evaluación reserva sus entradas; el entrenamiento nunca ve un ejemplo de evaluación (`mf_moe.train.sample`). Antes, `bin` y `modmult` habían visto casi toda su evaluación.
- 2026-10-15: costo estimado del factorial ~940 unidades (85.000 pasos por corrida, ~3,5 h de L4, 1,6 unidades/h), sobre el techo de 550. Se mide cuántas corridas caben en paralelo en una L4 antes de recortar.
