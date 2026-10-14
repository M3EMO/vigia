# Medir el olvido de referencia y la velocidad real en una L4

Type: task
Status: claimed
Blocked by: 03, 04

## Question

Antes de cerrar el diseño: entrenar el MoE fijo de la arquitectura elegida en dos tareas seguidas en una L4. Registrar tokens por segundo, unidades gastadas y cuánto cae la exactitud de la primera tarea. Si no hay olvido medible, las tareas o la escala cambian.

Ampliado el 2026-10-14 (enmienda del ticket de tareas): además, calibrar la dificultad con una prueba de capacidad. Con 6 expertos ruteados, el MoE fijo NO debe dominar todas las tareas juntas al 88 %; con 12, sí. Ajustar largos de números y lista de tareas (10-12, con compuestas) hasta cumplir eso, y fijar el largo del tramo sin mejora que dispara la alarma de estancamiento.
