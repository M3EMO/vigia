# Gotchas

- La laptop de build tiene 8 GB de RAM y no tiene GPU. gpt-oss-20b no corre local.
- Granite 1B-A400M en bf16 genera a ~0,5 tok/s en esta CPU. Para calibrar conviene teacher forcing
  (una pasada sobre prompt + respuesta) en vez de generar.
- En transformers 5 el router de Granite devuelve los logits en el índice 2 de la tupla; el resto en el 0.
  Vigía no lee la tupla: recalcula logits con un pre-hook sobre el bloque MoE (ADR A-01).
- `output_router_logits=True` rompe `generate()` con cache KV. No usarlo.
- Con MXFP4, transformers no llama al módulo router. Un hook sobre el router no dispara.
