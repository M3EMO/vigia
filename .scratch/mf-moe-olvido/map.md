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
- Presupuesto máximo de Colab para todo el esfuerzo: 550 unidades (el usuario lo subió de 300 el 2026-10-14, al elegir el factorial completo). Saldo de la cuenta ese día: 634.
- Research al vault global, nunca al repo (`vault/research/`, tag `vigia`). Evaluación previa:
  nota "MF-MoE: evaluación de entrenar un MoE chico para medir el olvido (2026-10-14)".
- Objeciones abiertas que el diseño debe responder: `docs/spec/mf-moe-revision.md` (3, 4, 5 y 7).
- Almacenamiento (usuario, 2026-10-14): checkpoints y resultados pesados del entrenamiento van al Google Drive del usuario, no al disco local; en local quedan solo resúmenes.
- Corridas largas en Colab: `colab/cli/job_runner.sh`; la PC tiene que quedar despierta y con la tapa abierta.
- Los tickets grilling se resuelven con el usuario, una pregunta por vez; el agente nunca responde por él.

## Decisions so far

- Destino: spec + experimento + resultado en el paper (usuario, 2026-10-14).
- Tareas: sintéticas primero, texto después si hay señal (usuario, 2026-10-14).
- Presupuesto: hasta 550 unidades de Colab (usuario, 2026-10-14; antes 300).
- Diseño: 30 corridas en una base (Qwen2-MoE): principal contra fijo con repetición en 3 semillas × 2 escenarios, fijo y Lifelong-MoE igual, y las dos ablaciones con nombre (2026-10-16; el piloto midió ~14,5 unidades por corrida y las 84 corridas no entraban).
- [Cómo crecen y se fusionan expertos en los trabajos publicados](issues/01-precedentes.md) — nadie dispara con la duda del router (eso es lo nuevo); fusión por similitud de salidas (HC-SMoE); el fijo arranca con los parámetros finales; fijo+repetición es el baseline fuerte; nadie reporta BWT.
- [Qué enseñan los papers de DeepSeek sobre expertos finos, compartidos y balanceo](issues/10-deepseek-moe.md) — los expertos finos son una división fija (el control fijo usa la granularidad final del fractal); el compartido no es un ancla demostrada; balancear por lote en tareas en fila puede causar olvido; con sigmoide la entropía cambia de sentido.
- [Arquitectura base del MoE de 80-100 M](issues/03-arquitectura-base.md) — dos bases con experimento completo cada una (Qwen2-MoE y MoE propio); d = 512, 8 capas, 1 compartido + ruteados finos, top-3; fractal 6 → 12 expertos (52,6 → 90,3 M), fijo con 12; sesgo de balanceo con ventana larga; bytes, contexto 512.
- [Tareas sintéticas de la fase 1](issues/04-tareas-sinteticas.md) — 10-12 tareas de dígitos (seis base más compuestas, con números largos calibrados por presión de capacidad); escenarios con y sin nombre, con pista implícita por largo de los números; un orden por semilla; cada tarea entrena hasta dominarla al 88 %, sin tope de pasos (una alarma pausa y avisa si una tarea se estanca); datos al vuelo y evaluación exacta sobre 1.000 ejemplos fijos.
- [Dónde vive el código del experimento](issues/02-donde-vive-el-codigo.md) — repo aparte `mf-moe`, privado hasta el paper, Apache 2.0, con Vigía como dependencia.
- [Cómo funciona la repetición de datos viejos](issues/08-repeticion.md) — buffer finito de 1.000 ejemplos por tarea; 10 % de cada lote, parejo entre tareas viejas; misma regla en toda condición con repetición.
- [Qué dispara la división de un experto](issues/06-disparador-de-division.md) — tres disparadores (duda del router, pérdida, combinación); z-score > 2 sobre la capa cada 500 pasos; hijo clonado con fila de alta duda; router global.
- [Cuándo y cómo se fusionan expertos](issues/07-criterio-de-fusion.md) — tres criterios como condiciones (salidas, pesos, ambos); al terminar cada tarea; z-score > 2 sobre los pares de la capa; promedio por frecuencia; nunca menos de 6 expertos.
- [Condiciones, controles y criterio de éxito](issues/09-condiciones-y-criterio.md) — gana el fractal si su BWT supera al del fijo con repetición; variante principal duda + fusión por salidas, bootstrap pareado IC 95 % sobre 6 pares; 14 condiciones; 84 corridas en una base.
- [Olvido de referencia, velocidad y capacidad](issues/05-olvido-de-referencia.md) — las doce tareas aprenden solas tras cinco calibraciones; la capacidad no muestra presión entre 6 y 12 expertos (las dos fallan las mismas tareas por interferencia).

## Not yet specified

- [Fase 2: ¿el fractal funciona a escala?](issues/12-fase-2-escala.md) — resuelto: Granite-1B-A400M con ajuste completo, 4 dominios de texto, condiciones A a D con 3 semillas; presupuesto sacado de las preguntas 5 y 6 de la fase 1.

- [Spec del factorial, etapa 2](issues/11-spec-factorial-v2.md) — resuelto: 40 corridas en 6 preguntas (principal, sin fusión, con recuperación, fusión alineada, contexto, sin repetición), unas 387 unidades, corte temprano con 4 pares.

- Fase 2 con texto: qué dominios, qué datos, qué tokenizer y cuánto cómputo; depende del resultado de la fase 1.
- Adaptador de Vigía para el MoE propio (registrar su `model_type` y el atributo del router).
- Construcción: módulos de división, fusión y repetición, y el bucle de entrenamiento secuencial; se tiquetea al cerrar la spec.
- Corridas: orden, reintentos y cómo escribe cada VM en Google Drive (hoy `colab drivemount` pide login interactivo por VM).
- [MoE fractal jerárquico](issues/13-fractal-jerarquico.md) — abierto: prototipo con router en árbol y mitosis con partición, con tests; falta elegir datos (aritmética o idiomas emparentados) y correr. Retoma el pedido del 2026-10-14 (micro-router jerárquico por padre).
- [Fractal bien hecho](issues/14-fractal-bien-hecho.md) — abierto: etapa A en la cola (consolidación sin división, por uso y al azar, 3 semillas); etapas B (atención lenta) y C (árbol desde el árbol natural, mitosis guiada por datos, fusión solo aceptada) en spec.
- [Spec del MoE fractal](spec-moe-fractal.md) — requisitos con ID, medida, aceptación y estado (hipótesis H1-H9, validez, medidas, ruteo, optimización, olvido, fractal, literatura, infraestructura); un brazo entra a Colab solo con sus requisitos obligatorios hechos.
- Experimento aparte (pedido del usuario, 2026-10-14): aprender las reglas de división y fusión por búsqueda de orden cero (idea tomada de DUST, qlabs.sh/research/dust), no el entrenamiento de los pesos; va después de la fase 1.
- Medida secundaria: si la huella topológica (JSD de uso de expertos) cambia durante el aprendizaje continuo.
- Escritura: cómo entra el resultado al paper y qué figuras lo cuentan.

## Out of scope

- Cambios al producto Vigía: este mapa solo usa su sensor de duda.
- Modelos de más de 100 M de parámetros y la pérdida de anclaje o Procrustes de la revisión v2.
- Trampa de recompensa y complacencia: el diseño fractal no ataca su causa (análisis del 2026-10-12).
