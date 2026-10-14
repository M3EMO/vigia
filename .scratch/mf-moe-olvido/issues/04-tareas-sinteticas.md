# Tareas sintéticas de la fase 1

Type: grilling
Status: resolved

## Question

¿Qué 4 a 6 tareas aprende el modelo, en qué orden, con cuántos ejemplos y con qué formato? Las tareas tienen que tener respuesta exacta y una dificultad que haga visible el olvido en un MoE fijo.

## Answer

Resuelto con el usuario el 2026-10-14 (grilling).

- Seis tareas sobre dígitos, que interfieren entre sí: suma («4821+377=»), multiplicación («48×37=»), inversión («inv 83920»), orden ascendente («ord 83920»), paso a binario («57→b») y resto («57%7=»).
- Dos escenarios, cada uno con el experimento completo: con nombre de tarea al inicio del ejemplo y sin nombre (la tarea se deduce del formato). Con las dos bases son cuatro experimentos.
- Orden: un orden al azar distinto por semilla, el mismo para todas las condiciones de esa semilla.
- Duración (decisión final del usuario, 2026-10-14, tras dos correcciones): cada tarea entrena hasta dominarla al 88 % de exactitud en validación, sin tope de pasos. Riesgo abierto: una tarea que no llega al 88 % consumiría cómputo sin fin; el ticket de referencia mide si alguna tarea se estanca.
- Alarma operativa (aceptada por el usuario el mismo día): si la exactitud de validación de una tarea no mejora durante un tramo largo, la corrida se pausa y se avisa al usuario, que decide seguir o cortar. No es un tope del diseño. El largo del tramo se calibra en el ticket de referencia.
- Datos: generación al vuelo (cada lote es nuevo); evaluación con 1.000 ejemplos fijos por tarea, apartados del entrenamiento, respuesta exacta con decodificación greedy.
- Pendiente para el ticket de referencia: rangos de longitud de los números, calibrados para que el MoE fijo aprenda cada tarea y muestre olvido.

## Comments

- 2026-10-14, enmienda con el usuario (al empezar el código): el escenario sin nombre necesita una pista. Un símbolo como «+» equivale a un nombre, y una entrada idéntica para dos tareas las vuelve contradictorias (mide sobrescritura, no olvido).
  - Decisión: pista implícita en los datos. Todas las tareas usan el mismo formato, y cada una usa números de un largo propio (y, en las compuestas, otra cantidad de números). Los dos escenarios comparten exactamente los mismos datos; solo cambia si el ejemplo empieza con el nombre de la tarea.
  - Dificultad: el usuario pidió subirla por tres vías a la vez, para que haya presión de capacidad (si 12 expertos alcanzan sobrado, dividir no puede ayudar):
    1. Números más largos, calibrados (por ejemplo, multiplicación de 4 por 4 dígitos).
    2. Tareas compuestas de varios pasos, por ejemplo (a+b)×c, invertir y sumar, binario de una suma, resto de un producto.
    3. Más tareas: de 6 a 10-12 en total (las seis base más las compuestas).
  - La lista final y los largos se fijan con la prueba de capacidad del ticket "Medir el olvido de referencia y la velocidad real en una L4".
- 2026-10-14, cambio de formato decidido por el usuario tras la aprendibilidad: mult, mod, sumamult, binsuma, modmult y digmult escriben la respuesta desde la unidad (`REVERSED_ANSWER` en `mf_moe/tasks.py`). Las otras seis siguen en orden normal.
