# Cómo funciona la repetición de datos viejos

Type: grilling
Status: resolved

## Question

¿Qué proporción de datos viejos entra en cada lote, cuántos ejemplos guarda el buffer y cómo se eligen? La misma regla vale para toda condición que use repetición.

## Answer

Resuelto con el usuario el 2026-10-14 (grilling).

- Fuente: buffer finito. Al terminar cada tarea se guardan 1.000 ejemplos al azar de esa tarea; la repetición usa solo esos, nunca regenera tareas viejas. Imita el límite real de memoria y mantiene comparable la fase 2 con texto.
- Proporción: el 10 % de cada lote sale del buffer, repartido parejo entre las tareas viejas. La primera tarea no tiene repetición.
- La misma regla vale para toda condición que use repetición (fijo con repetición, fractal y sus ablaciones).
- Riesgo a vigilar: con 1.000 ejemplos por tarea, el modelo puede memorizar el buffer; la evaluación usa los 1.000 ejemplos apartados, distintos del buffer.
