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
- Duración (corregida por el usuario el mismo día; primero había elegido pasos fijos): cada tarea entrena hasta 88 % de exactitud en validación, con un tope de pasos. El tope sale del ticket "Medir el olvido de referencia y la velocidad real en una L4". Una tarea que llega al tope sin el 88 % se registra como no aprendida.
- Datos: generación al vuelo (cada lote es nuevo); evaluación con 1.000 ejemplos fijos por tarea, apartados del entrenamiento, respuesta exacta con decodificación greedy.
- Pendiente para el ticket de referencia: rangos de longitud de los números, calibrados para que el MoE fijo aprenda cada tarea y muestre olvido.
