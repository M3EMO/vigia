# Spec del factorial, etapa 2: qué se corre después del primer par

Type: grilling
Status: open
Blocked by: 09

## Question

El primer par del factorial mostró que la fusión daña la tarea recién aprendida. ¿Qué preguntas responde la
etapa 2, con qué condiciones, en qué orden, con qué costo y con qué reglas de corte? Ninguna corrida nueva
entra a Colab hasta cerrar este ticket (pedido del usuario, 2026-10-16).

## Lo medido hasta ahora

- Banco: doce tareas aprenden solas al 88 % (ticket 05). La prueba de capacidad no muestra presión entre
  6 y 12 expertos: los dos fallan las mismas tareas por interferencia.
- Primer par (semilla 0, con nombre): fractal principal contra fijo con repetición. BWT desde lo aprendido
  −0,80 contra −0,68; exactitud final 10 % contra 29 %; pasos 185.750 contra 96.500. La fusión al cerrar
  cada tarea baja la tarea recién aprendida hasta 1 %; el BWT clásico escondía ese daño.
- Control fijo con repetición, semilla 0 sin nombre: BWT −0,77, exactitud 22 %. Promedio del control:
  BWT −0,73, exactitud 25 %, transferencia 1,56.
- En curso (terminan antes de cerrar este ticket): sin fusión y con recuperación, semilla 0, con nombre.
  Señal temprana: sin fusión aprende rápido (sumamult 12.000 pasos contra 26.750 del control); la
  recuperación cuesta 250 a 1.750 pasos por tarea.
- Causa probable del daño: la fusión promedia neuronas sin alinearlas; dos expertos entrenados por
  separado ordenan sus neuronas distinto. MC-SMoE alinea por permutación antes de promediar.

## Costos medidos (L4, 1,54 unidades por hora por VM, 2 VMs a la vez)

| Tipo de corrida | Horas | Unidades |
|---|---|---|
| Fijo y fijo con repetición | 3,5 a 4 | 5,5 a 6 |
| Lifelong-MoE (piloto: similar al fijo) | unas 4,5 | unas 7 |
| Fractal con fusión (con o sin recuperación) | unas 8,5 | unas 13 |
| Fractal sin fusión (estimado por el avance) | unas 5 | unas 8 |

Saldo: 532,9 unidades (2026-10-16 20:04 UTC). Techo del esfuerzo: 550 desde un saldo de 634, así que el
piso es 84 y quedan unas 440 disponibles después de las dos corridas en curso.

## Preguntas, en orden de prioridad

1. **Preregistrada:** ¿el fractal principal (duda + fusión por salidas) olvida menos que el fijo con
   repetición? Métrica: BWT desde lo aprendido; secundaria: exactitud final. Respuesta esperada: no.
2. **¿La división sola ayuda?** Ablación sin fusión contra fijo con repetición.
3. **¿La fusión reparada ayuda?** Fusión con recuperación contra fijo con repetición.
4. **¿La fusión bien hecha ayuda?** Fusión alineada por permutación (estilo MC-SMoE) contra fijo con
   repetición. Código nuevo: necesita tests en CPU y un humo corto en GPU antes de encolar.
5. **Contexto:** fijo sin repetición y Lifelong-MoE, para ubicar el resultado entre los métodos conocidos.
6. **Ablación sin repetición** del fractal: ¿la división aporta sin el buffer?

## Diseño propuesto

Cada pregunta usa 3 semillas × 2 escenarios = 6 corridas y 6 pares contra el control (que se corre una
sola vez y se comparte). Prueba: bootstrap pareado, IC 95 % de la diferencia; gana si el IC queda entero
sobre 0. Las preguntas 2 a 6 son exploratorias: se reportan sin corrección por multiplicidad.

| Pregunta | Corridas que faltan | Unidades |
|---|---|---|
| 1. Principal (5) + control (4) | 9 | unas 87 |
| 2. Sin fusión | 5 | unas 40 |
| 3. Con recuperación | 5 | unas 65 |
| 4. Fusión alineada (+ humo en GPU) | 6 | unas 81 |
| 5. Fijo y Lifelong-MoE | 12 | unas 75 |
| 6. Sin repetición (solo con nombre) | 3 | unas 39 |
| Total | 40 | unas 387 |

Margen: unas 50 unidades sobre las 440 disponibles.

## Orden y reglas de corte

- Orden: control y preguntas 2 y 3 en las semillas 0 y 1 primero; después la pregunta 4 (tras el humo);
  después la semilla 2 de todo; al final las preguntas 5 y 6.
- Corte temprano: si una variante pierde contra el control en BWT desde lo aprendido Y en exactitud en los
  4 pares de las semillas 0 y 1, sus corridas restantes salen y su presupuesto pasa a la pregunta siguiente.
- Antes de encolar: tests en CPU de todo código nuevo, humo en GPU de cada condición nueva, cola con
  reservas y supervisor como servicio de systemd, subidas de checkpoint en partes de 16 MB.

## Decisiones abiertas para el usuario

1. ¿Entra la pregunta 4 (fusión alineada)?
2. ¿Se mantiene la regla de corte temprano?
3. ¿Las preguntas 5 y 6 corren, o el presupuesto se guarda para repetir con otra semilla las que den
   señal?
