# Cuándo y cómo se fusionan expertos

Type: grilling
Status: resolved
Blocked by: 01

## Question

¿Qué criterio decide que dos expertos son redundantes (similitud de salidas, de pesos o covarianza de uso; objeción 4), cuándo se evalúa y cómo se combinan sus pesos y su fila del router?

## Answer

Resuelto con el usuario el 2026-10-14 (grilling).

- Criterio, tres variantes como condiciones separadas (cada una usa un solo criterio, como pide la objeción 4): similitud de salidas (salida media sobre un set fijo de tokens, estilo HC-SMoE), similitud de pesos (coseno) y las dos a la vez (fusiona solo si ambas pasan). Cruzadas con los tres disparadores de la división dan las 9 variantes del fractal.
- Momento: al terminar cada tarea, como una fase de consolidación. Nunca baja de 6 expertos ruteados por capa.
- Umbral relativo, simétrico con la división: se fusiona cada par de la capa cuya similitud supera en 2 desvíos a la media de todos los pares de esa capa (pares disjuntos, en orden de similitud).
- Operador: promedio de pesos ponderado por la frecuencia de uso de cada experto (HC-SMoE). La fila del router y el sesgo de balanceo quedan los del experto más usado; el router redirige al resto.
- La covarianza de uso quedó descartada: según el ticket de precedentes, los criterios basados en el router pierden 21-24 puntos en HC-SMoE.
