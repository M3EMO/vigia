# Arquitectura base del MoE de 80-100 M

Type: grilling
Status: resolved
Blocked by: 10

## Question

¿Qué forma tiene el MoE base? Hay que fijar capas, ancho, expertos iniciales, top-k, tokenizer (de caracteres para tareas sintéticas), contexto y si se parte de `OlmoeForCausalLM` para reusar el sensor de Vigía. Hay que definir también qué significa 'mismos parámetros' cuando un modelo crece: totales al final, activos por token o FLOPs.

## Answer

Resuelto con el usuario el 2026-10-14 (grilling, una pregunta por vez).

- Dos bases, cada una con el experimento completo (todas las condiciones y semillas): `Qwen2MoeForCausalLM` de transformers y un MoE propio mínimo escrito para el experimento. El MoE propio necesita un adaptador de Vigía para el sensor de duda.
- Forma, igual en las dos bases: d = 512, 8 capas, 8 cabezas de atención, 1 experto compartido de 512 + expertos ruteados finos de 512 (0,25 de una FFN de 2.048), top-3 ruteados + el compartido.
- Crecimiento: el fractal arranca con 6 expertos ruteados por capa (52,6 M parámetros totales) y puede dividir hasta 12 (90,3 M; 33,7 M activos por token). El control fijo arranca con 12.
- "Mismos parámetros" = mismos totales al final (12 expertos) y mismos activos por token (top-3 + compartido, FLOPs iguales). Cifras calculadas con Qwen2MoeConfig en meta, vocabulario 128; con 256 bytes suben unos 0,07 M.
- El compartido nunca se divide ni se fusiona; se mide su deriva (ticket de DeepSeek).
- Router softmax en la fase 1, para que la entropía conserve su sentido; la entropía se calcula sobre el puntaje sin sesgo.
- Balanceo: sesgo por experto sin gradiente estilo DeepSeek-V3, u = 0,001, con la carga contada sobre una ventana de varias tareas, nunca por lote. Requiere código propio en las dos bases.
- Tokenizer de bytes (vocabulario 256), contexto 512. La fase 2 con texto define su propio tokenizer.
- Corrección registrada: en la pregunta se dijo "6 expertos ≈ 62 M"; el valor real es 52,6 M. El usuario siguió adelante con la cifra corregida.
