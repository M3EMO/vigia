# Medir el olvido de referencia y la velocidad real en una L4

Type: task
Status: claimed
Blocked by: 03, 04

## Question

Antes de cerrar el diseño: entrenar el MoE fijo de la arquitectura elegida en dos tareas seguidas en una L4. Registrar tokens por segundo, unidades gastadas y cuánto cae la exactitud de la primera tarea. Si no hay olvido medible, las tareas o la escala cambian.

Ampliado el 2026-10-14 (enmienda del ticket de tareas): además, calibrar la dificultad con una prueba de capacidad. Con 6 expertos ruteados, el MoE fijo NO debe dominar todas las tareas juntas al 88 %; con 12, sí. Ajustar largos de números y lista de tareas (10-12, con compuestas) hasta cumplir eso, y fijar el largo del tramo sin mejora que dispara la alarma de estancamiento.

## Comments

- 2026-10-14, corrida 1 (L4, MoE fijo de 12 expertos, `mf_moe.reference`, ~65 min, ~1,7 unidades; datos en `~/Proyectos/mf-moe/.cache/runs/ref_1/`):
  - Olvido total: la suma (7+7 dígitos) llega a 98,5 % en 1.000 pasos y cae a 0 % después de entrenar multiplicación; igual sin nombre (93,9 % → 0 %).
  - Velocidad: 20.000-24.000 tokens/s; 1.000 pasos ≈ 2 min con lotes de 128.
  - Multiplicación 4×4: 0 % después de 5.250 pasos (alarma).
  - Defecto propio: la alarma miraba solo la peor tarea; en el conjunto, «mult» en 0 % cortó las dos pruebas de capacidad a los 5.250 pasos (~440 por tarea). Las pruebas de capacidad de la corrida 1 no sirven. Corregido: la alarma salta solo si ninguna tarea mejora 1 punto (con prueba de regresión).
- 2026-10-14, corrida 2 en curso: aprendibilidad de cada tarea sola (12 expertos, paciencia de 10.000 pasos), para calibrar largos antes de repetir la capacidad.
