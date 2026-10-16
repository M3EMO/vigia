# Medir el olvido de referencia y la velocidad real en una L4

Type: task
Status: resolved
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
- 2026-10-14, corrida 2 (aprendibilidad, cada tarea sola, 12 expertos, paciencia 10.000 pasos; ~5 h, ~8 unidades; datos en `~/Proyectos/mf-moe/.cache/runs/ref_2_aprendibilidad/`):
  - Aprenden al 88 %: inv 12 (250 pasos), ord 10 (250), suma 7+7 (1.000), ordsuma 6+6 (1.750), invsuma 8+6 (2.250), bin 5 (11.750).
  - No aprenden: mult 4×4 (0 %), mod 9÷2 (10 %), sumamult 5-5-2 (0 %), binsuma 5+4 (6 %), modmult 5-5-3 (0 %), digmult 5×3 (11 %).
  - Decisión del usuario: respuesta con dígitos invertidos (desde la unidad) en esas seis tareas. Advertencia registrada: invertir ayuda donde el cálculo avanza desde la unidad (mult, sumamult, binsuma); en mod, modmult y digmult la respuesta es corta y casi no cambia.
- 2026-10-14, corrida 3 en curso (calibración): las seis invertidas con sus largos, más respaldo con números más cortos para mod (6÷1), modmult (3-3-2) y digmult (3×2).
- 2026-10-15, corrida 3 (respuesta invertida): ninguna de las seis aprende (0 % a 10,3 %); con números cortos, mod 6÷1 59,5 % y modmult 3-3-2 12,2 %; digmult 3×2 cortada en 46 %. Research al vault: «mf-moe: por qué no aprenden mult y mod» (dependencias largas; invertir solo la respuesta no alcanza).
- 2026-10-15, corrida 4 (decisión del usuario: formatos con evidencia + pasos intermedios; alarma también por dígito; checkpoint por variante): mult 5×2 desde la unidad 88,3 % (12.000 pasos), sumamult 4-4-2 88,6 % (36.000), binsuma 4+3 89,4 % (2.500), digmult 5×1 88,1 % (1.750); mod 6 %7 15,3 % (azar) y modmult 4-2 %7 22,5 % (falla solo el resto).
- 2026-10-15, corrida 5 (decisión del usuario: restos parciales): mod 94,5 % (250 pasos), modmult 89,6 % (15.000). Las doce tareas aprenden solas. Ojo: corridas 2 a 5 sin evaluación reservada; bin y modmult vieron casi toda su evaluación, así que la prueba de capacidad (ya con reserva) confirma si generalizan.
- 2026-10-15: varias corridas en paralelo en una L4 no rinden (1 proceso ya usa 94 % de la GPU; 2 y 4 dan 8 % menos en total). Expertos multiplicando en bf16 con pesos fp32: 183 → 110 ms por paso (en verificación con una corrida de mult).
- 2026-10-16, prueba de capacidad (reparto del lote según los pasos de cada tarea sola; evaluación reservada): las dos corridas en L4 en paralelo. La de 6 expertos, a los 122.000 pasos, domina las difíciles (mult 98 %, modmult 99 %, sumamult 99 %) y oscila en las fáciles (inv 83 → 53 %, digmult 27 → 74 %); con doce tareas que oscilan, la alarma quizás nunca salta. Decisión del usuario: la de 6 corre hasta que termine la de 12, y ahí se corta.
- 2026-10-16, resultado de la capacidad (decisión del agente, con delegación del usuario): la de 6 expertos llegó a 129.750 pasos con 6 de 12 tareas sobre 88 %; la de 12, a 151.750 pasos con 3 de 12. Las dos fallan en las mismas tareas (ord, ordsuma, binsuma, bin: 1 % a 60 %) y oscilan en las fáciles. Duplicar los expertos no ayuda: la falla viene de la interferencia entre tareas en el entrenamiento conjunto, no de la capacidad. La prueba no muestra presión de capacidad entre 6 y 12 expertos; se cortaron las dos. El factorial sigue válido, porque compara condiciones con los mismos parámetros finales. Ticket cerrado.
