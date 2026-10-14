# Arquitectura base del MoE de 80-100 M

Type: grilling
Status: open

## Question

¿Qué forma tiene el MoE base? Hay que fijar capas, ancho, expertos iniciales, top-k, tokenizer (de caracteres para tareas sintéticas), contexto y si se parte de `OlmoeForCausalLM` para reusar el sensor de Vigía. Hay que definir también qué significa 'mismos parámetros' cuando un modelo crece: totales al final, activos por token o FLOPs.
