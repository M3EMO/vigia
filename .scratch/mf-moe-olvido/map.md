# Mapa: ¿el MoE fractal olvida menos?

Label: wayfinder:map · Creado: 2026-10-14

## Destination

Un resultado medido, dentro del paper "MoE fractal y el router como sensor de duda": un MoE de 80-100 M
de parámetros con división, repetición y fusión olvida menos que controles con los mismos parámetros, o
no. El primer hito es una spec cerrada y lista para correr; después se construye, se corre y se escribe.

## Notes

- Dominio: aprendizaje continuo con MoE; investigación, no producto Vigía.
- Este mapa SÍ lleva la ejecución adentro (decisión del usuario): spec → código → corridas → paper.
- Fase 1 con tareas sintéticas de respuesta exacta; fase 2 con dominios de texto solo si la fase 1 da señal.
- Tamaño del modelo: 80-100 M de parámetros (decisión del usuario, 2026-10-14).
- Presupuesto máximo de Colab para todo el esfuerzo: 300 unidades (decisión del usuario, 2026-10-14).
- Research al vault global, nunca al repo (`vault/research/`, tag `vigia`). Evaluación previa:
  nota "MF-MoE: evaluación de entrenar un MoE chico para medir el olvido (2026-10-14)".
- Objeciones abiertas que el diseño debe responder: `docs/spec/mf-moe-revision.md` (3, 4, 5 y 7).
- Corridas largas en Colab: `colab/cli/job_runner.sh`; la PC tiene que quedar despierta y con la tapa abierta.
- Los tickets grilling se resuelven con el usuario, una pregunta por vez; el agente nunca responde por él.

## Decisions so far

- Destino: spec + experimento + resultado en el paper (usuario, 2026-10-14).
- Tareas: sintéticas primero, texto después si hay señal (usuario, 2026-10-14).
- Presupuesto: hasta 300 unidades de Colab (usuario, 2026-10-14).
- [Cómo crecen y se fusionan expertos en los trabajos publicados](issues/01-precedentes.md) — nadie dispara con la duda del router (eso es lo nuevo); fusión por similitud de salidas (HC-SMoE); el fijo arranca con los parámetros finales; fijo+repetición es el baseline fuerte; nadie reporta BWT.

## Not yet specified

- Fase 2 con texto: qué dominios, qué datos, qué tokenizer y cuánto cómputo; depende del resultado de la fase 1.
- Construcción: módulos de división, fusión y repetición, y el bucle de entrenamiento secuencial; se tiquetea al cerrar la spec.
- Corridas: orden, reintentos y cómo se guardan los resultados para que sobrevivan a una VM retirada.
- Medida secundaria: si la huella topológica (JSD de uso de expertos) cambia durante el aprendizaje continuo.
- Escritura: cómo entra el resultado al paper y qué figuras lo cuentan.

## Out of scope

- Cambios al producto Vigía: este mapa solo usa su sensor de duda.
- Modelos de más de 100 M de parámetros y la pérdida de anclaje o Procrustes de la revisión v2.
- Trampa de recompensa y complacencia: el diseño fractal no ataca su causa (análisis del 2026-10-12).
