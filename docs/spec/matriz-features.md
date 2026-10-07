# Matriz canónica de features

Prioridad: **P0** bloquea el release. **P1** entra en v1. **P2** queda documentado y pendiente.
"Debilidad corregida" indica qué falla del mercado o del paper mejora Vigía.

## 1. Señales por token (contrato de features)

Notación: `z ∈ R^N` son los logits del router de una capa MoE, `p = softmax(z)`, `S_k` son los k
expertos elegidos, `q` es la distribución de salida sobre el vocabulario V.

| Nivel | Nombre | Fórmula | Costo | Prioridad |
|---|---|---|---|---|
| F0 | `out_entropy` | `−Σ_v q(v) log q(v)` sobre V completo | gratis | P0 |
| F0 | `out_nll` | `−log q(token emitido)` | gratis | P0 |
| F0 | `out_margin` | `q_(1) − q_(2)` | gratis | P0 |
| F0 | `out_top20_entropy` | entropía de los 20 mayores, renormalizada (lo único que da vLLM) | gratis | P0 |
| F1 | `r_ent_n@l` | `−Σ_i p_i log p_i / log N` | ~0 | P0 |
| F1 | `r_ent_k@l` | entropía normalizada de los pesos top-k reales del modelo | ~0 | P0 |
| F1 | `r_topk_mass@l` | `Σ_{i∈S_k} p_i` | ~0 | P0 |
| F1 | `r_margin12@l` | `z_(1) − z_(2)` | ~0 | P0 |
| F1 | `r_margin_kk1@l` | `z_(k) − z_(k+1)` (frontera de selección) | ~0 | P0 |
| F1/U | `u_gini@l` | Gini del uso ponderado acumulado (causal, hasta t) | ~0 | P0 |
| F1/U | `u_ihhi@l` | `1 / Σ_i u_i²` normalizado por N | ~0 | P0 |
| F1/U | `u_ent@l` | entropía normalizada del uso acumulado | ~0 | P0 |
| F1/U | `r_switch@l` | `1 − |S_k(t) ∩ S_k(t−1)| / k` | ~0 | P0 |
| F1 | `agg_{mean,max,std}:<f>` | agregados entre capas de cada feature F1 | ~0 | P0 |
| F2 | `e_cos@l` | similitud coseno media ponderada entre salidas de expertos elegidos | alto, opt-in | P1 |
| F2 | `e_disp@l` | `Σ w_i ‖e_i − ē‖² / ‖ē‖²` | alto, opt-in | P1 |
| F2 | `e_jsd` | Jensen-Shannon normalizada entre las distribuciones *logit-lens* de los expertos elegidos de la última capa MoE (comité MF-MoE corregido) | muy alto, opt-in | P1 |
| H | `h_probe` | sonda lineal sobre el residual stream en la capa ~0,95·L (baseline obligatorio y feature opcional) | bajo | P1 |

Reglas del contrato:

- Toda feature es causal: el valor en t no cambia al agregar tokens posteriores (test).
- Las capas densas y los expertos compartidos quedan fuera de F1 (DeepSeek, Qwen2-MoE).
- Con MXFP4 o kernels fusionados, F2 se reporta como capacidad deshabilitada, sin fallar.
- El hook solo guarda los logits del router del paso (`detach()` bajo `inference_mode`). La reducción
  a escalares ocurre en `end_step`, por paso, vectorizada entre capas. Nunca se acumulan logits entre pasos.

## 2. Producto

| ID | Feature | Detalle verificable | Debilidad corregida | Prioridad |
|---|---|---|---|---|
| P-01 | Extractor multi-arquitectura | gpt_oss, mixtral, qwen2_moe, qwen3_moe, olmoe, granitemoe, deepseek_v3, phimoe por registro declarativo | Ningún producto usa el router MoE | P0 |
| P-02 | Backend HF con streaming | genera token a token con hooks, separa prefill de decode, respeta `attention_mask` | `output_router_logits` falla con cache KV | P0 |
| P-03 | Canal `final` de gpt-oss | parseo de harmony; `analysis` nunca se muestra ni se puntúa | — | P0 |
| P-04 | Detector JSON | LR + z-score + Platt + τ por alpha, hash SHA-256, runtime numpy | El repo del paper usa pickle | P0 |
| P-05 | Calibración | splits agrupados por pregunta, Platt (A-04; isotónica pendiente, ver P-41), CRC sobre FNR; aviso si la garantía obliga a marcar ≥ 95 % de los tokens; grilla fina de alpha (0,02 a 0,50) con detección y falsas alarmas por respuesta en cal y recommended_alpha elegido solo con cal | Umbral F1 no transfiere | P0 |
| P-06 | Etiquetas computables | EM y F1 de tokens con normalización española; aritmética | CI sin LLM | P0 |
| P-07 | Juez LLM | endpoint OpenAI-compatible, schema `{label, hallucinated_spans}`, spans substring exacto, mapeo por offsets | Juez con sesgo a falso positivo | P1 |
| P-08 | Reporte de evaluación | AUROC, AUPRC, ECE, Brier, recall@FPR 0,1, IoU de span; bootstrap por cluster; 3 baselines | Competencia no publica métricas en español | P0 |
| P-09 | Benchmark de overhead | A/B hooks on/off, warmup, semilla, p50/p95/p99, por nivel | Meta < 5 % sin medición | P0 |
| P-10 | Fragmentos | subtokens → palabras → fragmentos contiguos con `gap_max`, `min_tokens`, score max | — | P0 |
| P-11 | Políticas | acciones pass/annotate/human_review/rag_retry/abstain por umbral de respuesta | El color solo induce sobreconfianza | P0 |
| P-12 | rag_retry estilo FLARE | consulta a un retriever con allowlist, contexto inyectado, `max_retries` | — | P1 |
| P-13 | API OpenAI-compatible | `/v1/chat/completions`, `/v1/completions`, `/v1/models`, SSE, campo `vigia` | — | P0 |
| P-14 | Trazas por ID | `GET /v1/vigia/traces/{id}` con header `X-Vigia-Trace-Id` | LangChain descarta campos extra | P0 |
| P-15 | Modo reducido vLLM | proxy con `routed_experts` (uint8/uint16/int16) + logprobs top-20; nivel U | vLLM no da logits del router | P1 |
| P-16 | CLI | `vigia extract | label | calibrate | evaluate | bench | serve` | — | P0 |
| P-17 | Multi-tenant | organizaciones aisladas; 404 entre tenants; IDs aleatorios | — | P0 |
| P-18 | Identidad | login con scrypt, sesión server-side `__Host-`, CSRF, roles owner/admin/reviewer/viewer | — | P0 |
| P-19 | API keys | `vg_live_<kid>_<secret><crc>`, HMAC con pepper, scopes, expiración, revocación, mostrada una vez | — | P0 |
| P-20 | Panel: Resumen | KPIs, tendencia de flags, latencia p50/p95 contra meta | — | P0 |
| P-21 | Panel: Generaciones y Detalle | lista filtrable (querystring sin PII), detalle con fragmentos codificados 3 veces y aviso "texto sin marcar no está verificado" | Galileo usa logprobs crudos | P0 |
| P-22 | Panel: Cola de revisión | estados pendiente/reservado/necesita otro/completado; teclado 1/2/3, Ctrl+Enter; etiquetas realimentan calibración | — | P0 |
| P-23 | Panel: Calibración | subir JSONL, job de entrenamiento, reporte con 7 gráficos, activar versión | — | P0 |
| P-24 | Panel: Políticas | alpha con input numérico, tasa de marcado y precisión esperadas, acciones; aviso si el alpha elegido marca casi todo; frontera detección de respuestas malas / falsas alarmas en buenas medida en calibración (gráfico + tabla), alpha recomendado (máximo de detección - falsas alarmas en cal, grilla 0,02 a 0,50) con botón 'Usar el alpha recomendado'; sin alpha fijado por el owner rige el default_alpha del detector activo (= recomendado); aviso para recalibrar si el artefacto es previo | — | P0 |
| P-25 | Panel: Modelo | heatmap capa × experto del uso | Diferenciador frente al mercado | P1 |
| P-26 | Panel: Playground | prompt → respuesta en streaming con fragmentos marcados | Demo de 2 minutos del plan | P0 |
| P-27 | Panel: Configuración | retención, `store_text`, webhooks, keys, usuarios | — | P0 |
| P-28 | Auditoría | append-only con cadena de hashes; export por rango | — | P0 |
| P-29 | PII argentina | DNI, CUIT/CUIL (módulo 11), CBU, tarjeta (Luhn), email, teléfono, IP | Competencia no cubre formatos locales | P0 |
| P-30 | Webhooks | Standard Webhooks (HMAC v1, reintentos, idempotencia, SSRF bloqueado) | — | P1 |
| P-31 | Retención | purga por tenant, borrado por request, export JSON del tenant | Ley 25.326 art. 14 y 16 | P0 |
| P-32 | Seguridad HTTP | CSP con nonce, HSTS, nosniff, frame-ancestors none, errores genéricos con request_id | — | P0 |
| P-33 | Rate limiting | por key, tenant e IP de login; 429 + Retry-After | — | P0 |
| P-34 | Observabilidad | Prometheus sin labels no acotados; logs JSON con allowlist | — | P0 |
| P-35 | Despliegue | Dockerfile no-root, filesystem de solo lectura, secretos por archivo, sin red saliente | Air-gapped | P1 |
| P-36 | Drift | PSI sobre score, tasa de marcado y features clave; alerta en el panel | — | P1 |
| P-37 | Export | JSONL con `schema_version`, CSV con neutralización de fórmulas | — | P1 |
| P-38 | Export OTLP / Langfuse | spans OpenInference + `vigia.*` resumidos | — | P2 |
| P-39 | Plugin vLLM con logits completos | captura en `select_experts` con buffer preasignado | — | P2 (necesita GPU Linux) |
| P-40 | MFA TOTP, SSO | owner y admin | — | P2 |
| P-41 | Calibración isotónica | isotónica en lugar de Platt con > 1000 respuestas de calibración | — | P2 (exige cambio de contrato del artefacto, ADR A-04) |

## 3. Fuera del producto

- La mitosis y la fusión de expertos de MF-MoE quedan como protocolo de investigación.
  Ver `docs/spec/mf-moe-revision.md`.
- El juez local gpt-oss-120b por defecto queda descartado: necesita ~80 GB de GPU.
