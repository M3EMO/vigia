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
- Datos: generación al vuelo (cada lote es nuevo); evaluación con 1.000 ejemplos fijos por tarea, apartados del entrenamiento, respuesta exacta con decodificación greedy.
- Pendiente para el ticket de referencia: rangos de longitud de los números, calibrados para que el MoE fijo aprenda cada tarea y muestre olvido.
