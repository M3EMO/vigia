# Arquitectura de Vigía

Vigía marca en tiempo real los tokens dudosos de un modelo MoE autoalojado. Vigía usa señales que
el modelo ya calcula: logits de salida, logits del router y uso de expertos. Un detector liviano
convierte esas señales en un P(error) calibrado por token.

Fuente de cada decisión: notas `vigia-*-2026-10-04.md` del vault global
(`~/.claude/orchestration/vault/research/`). El índice está en `docs/research/README.md`.

## 1. Decisiones de arquitectura (ADR cortos)

| ID | Decisión | Motivo |
|---|---|---|
| A-01 | El hook de captura es un `forward_pre_hook` sobre el **bloque MoE**. El hook recalcula `F.linear(x, W_router, b)` en float32. | Con MXFP4, transformers no llama al módulo router y un hook sobre el router no dispara (crítica K1). |
| A-02 | La entropía del router usa el softmax sobre los **N** expertos. La entropía sobre los k elegidos es otra feature. | gpt-oss y Granite aplican softmax después del top-k. El paper usa N y su código usa k. |
| A-03 | Las señales se agrupan en niveles de costo: F0, F1, U, F2 y H. El nivel por defecto es F0+F1. | El set completo de InnerExpert cuesta ~3x la inferencia. La meta de < 5 % solo es posible con señales baratas. |
| A-04 | El detector es una regresión logística con z-score más Platt. El artefacto es un JSON propio con hash. | El runtime corre sin scikit-learn y sin pickle. Pickle ejecuta código y no carga entre versiones. |
| A-05 | El umbral sale de *conformal risk control* sobre la tasa de falsos negativos (FNR) por token. El usuario elige alpha. | El umbral óptimo por F1 no transfiere entre datasets (XGB cae a F1 0,276 en Gemma). |
| A-06 | Sin detector calibrado, la API devuelve `score: null` y el panel no resalta. | Una entropía cruda no es un P(error). Resaltar sin calibrar induce sobreconfianza. |
| A-07 | La topología CPU/MVP es **un proceso**: la plataforma hospeda el modelo con el backend HF. | Las señales del router solo existen dentro del proceso del modelo. Un proxy no las calcula. |
| A-08 | La topología GPU con vLLM es **modo reducido**: Vigía actúa como proxy y usa `routed_experts` más logprobs top-20. | vLLM nativo entrega solo IDs de expertos al final del request. El plugin con logits completos queda pendiente de GPU. |
| A-09 | La API es compatible con OpenAI. Vigía agrega el campo `vigia` por choice y por chunk. Sin `extra_body.vigia`, la respuesta es idéntica a OpenAI. | El SDK de OpenAI conserva campos extra en `model_extra`. LangChain los descarta, así que existe `GET /v1/vigia/traces/{id}`. |
| A-10 | El almacenamiento es SQLite (stdlib) con WAL y un repositorio con tenant obligatorio. | Un despliegue autoalojado de un nodo no necesita Postgres. `ponytail:` Postgres con RLS cuando haya más de un nodo. |
| A-11 | El panel es server-rendered: FastAPI + Jinja2 con autoescape, formularios HTML y un JS propio chico. No usa htmx ni CDN. | La CSP estricta queda sin `unsafe-inline` ni `unsafe-eval`. htmx 2 trae `allowEval` y cache de historial con PII (crítica H2). |
| A-12 | Los gráficos son SVG generados por funciones Python puras, con `role=img` y tabla de datos alternativa. | Sin dependencias de gráficos. Los mismos SVG sirven para el reporte HTML y el panel. |
| A-13 | `vigia/` es el núcleo Apache 2.0. `vigia_ee/` es la plataforma comercial. Un test falla si `vigia/` importa `vigia_ee`. | Modelo open-core del plan. |
| A-14 | Las contraseñas usan `hashlib.scrypt` (stdlib, N=2^17, r=8, p=1). Las API keys usan HMAC-SHA256 con pepper. | scrypt cumple OWASP sin dependencias. Argon2 por request habilita DoS de memoria. |
| A-15 | Los otros tenants reciben siempre 404, nunca 403. | Un 403 confirma que el recurso existe (crítica K6). |
| A-16 | El juez de calibración es una interfaz sobre un endpoint compatible con OpenAI, sin modelo por defecto. Un juez externo exige opt-in del owner y queda en auditoría. | Ley 25.326 art. 12 restringe transferencias. Los tests usan etiquetas computables y un juez falso determinista. |

## 2. Topología

```
                ┌──────────────────────── proceso vigia-platform ─────────────────────────┐
cliente SDK ──► │ /v1/chat/completions ─► auth(API key→tenant) ─► límites ─► Backend HF    │
OpenAI          │                                                     │ hooks pre-bloque MoE│
                │                                                     ▼                     │
navegador ────► │ panel (Jinja)        detector del tenant ◄── features F0/F1/(F2/H) por token│
                │   │                     │ P(error) calibrado                               │
                │   ▼                     ▼                                                  │
                │ SQLite (WAL) ◄── trazas, spans, cola de revisión, auditoría, jobs          │
                │   ▲                                                                       │
                │ workers de jobs (un hilo por carril): calibración y purga; webhooks aparte │
                └─────────────────────────────────────────────────────────────────────────┘
modo reducido GPU: cliente ─► vigia-platform (proxy) ─► vLLM --enable-return-routed-experts
```

## 3. Flujo de información de un request

1. `POST /v1/chat/completions` llega con `Authorization: Bearer vg_...`.
2. `auth` valida la key con HMAC en tiempo constante y resuelve tenant, scopes y key_id.
3. `ratelimit` aplica cuotas por key y por tenant. Un exceso devuelve 429 con `Retry-After`.
4. `limits` valida tamaño del body (413), mensajes y `max_tokens` (422).
5. El backend aplica la plantilla de chat y genera token a token con los hooks activos.
6. El extractor reduce cada paso a un vector de features por token. Nunca guarda tensores por experto.
7. El detector activo del tenant calcula `p_error` calibrado y el flag `p_error ≥ τ(alpha)`.
8. `spans` agrega subtokens a palabras y palabras contiguas marcadas a fragmentos (score = max).
9. `policy` elige la acción por prioridad: `abstain > rag_retry > human_review > annotate > pass`.
10. La respuesta sale en formato OpenAI más el campo `vigia`. En streaming, cada chunk lleva sus tokens.
11. La traza se guarda según `store_text` del tenant (`none | redacted | full`, default `redacted`).
12. Las métricas Prometheus se actualizan. Los webhooks y la cola de revisión reciben sus eventos.

Si el detector falla, `on_detector_error` decide: `fail_closed` (default) devuelve 503 y
`fail_open` responde con `score: null` y `degraded: true`.

## 4. Módulos y autoría

Cada módulo tiene un solo dueño. Un agente solo edita los archivos de su fila. Los contratos
compartidos viven en `vigia/contracts.py` y `vigia_ee/schema.sql`; cambiarlos exige pasar por el
orquestador.

| Módulo | Archivos | Responsabilidad | Depende de |
|---|---|---|---|
| contratos | `vigia/contracts.py`, `vigia_ee/schema.sql` | Tipos, nombres de features, artefacto, esquema de datos | — |
| señales | `vigia/features.py`, `vigia/adapters.py`, `vigia/extractor.py` | Fórmulas puras, registro por arquitectura, hooks | contratos |
| backend HF | `vigia/backends/hf.py`, `vigia/harmony.py` | Carga, generación con streaming, canal `final` de gpt-oss | señales |
| backend vLLM | `vigia/backends/vllm_proxy.py` | Modo reducido con IDs y logprobs | contratos |
| ML | `vigia/detector.py`, `vigia/calibrate.py`, `vigia/labels.py` | Artefacto, entrenamiento, CRC, etiquetas computables y juez | contratos |
| evaluación | `vigia/evaluate.py`, `vigia/charts.py`, `vigia/bench.py` | Métricas, bootstrap por cluster, SVG, benchmark de latencia | ML |
| servicio | `vigia/spans.py`, `vigia/policy.py`, `vigia/api.py`, `vigia/cli.py`, `vigia/serve.py` | Agregación, políticas, API OpenAI, CLI, servidor single-tenant | todos los de núcleo |
| evidencia | `vigia/evidence.py`, `vigia/datasets.py` | Experimento reanudable, datasets JSONL y MKQA-es | ML, evaluación, backend HF |
| plataforma base | `vigia_ee/db.py`, `vigia_ee/auth.py`, `vigia_ee/security.py`, `vigia_ee/audit.py`, `vigia_ee/pii.py` | Datos, identidad, RBAC, headers, auditoría, PII | contratos |
| plataforma ops | `vigia_ee/jobs.py`, `vigia_ee/webhooks.py`, `vigia_ee/metrics.py`, `vigia_ee/retention.py` | Cola de jobs, webhooks, Prometheus, purga | plataforma base |
| panel | `vigia_ee/panel.py`, `vigia_ee/templates/`, `vigia_ee/static/` | Páginas, accesibilidad, atajos | plataforma base, evaluación |
| app | `vigia_ee/app.py`, `vigia_ee/runtime.py`, `vigia_ee/calibration.py` | Composición, health, runtime por tenant, calibración gestionada | todos |
| panel admin | `vigia_ee/panel_admin.py` | Cola, calibración, políticas, configuración, auditoría | plataforma base |

## 5. Health checks

| Ruta | Chequea | Uso |
|---|---|---|
| `/livez` | Solo que el proceso responde | liveness, nunca dependencias |
| `/startupz` | Modelo cargado y arranque de detectores terminado; informa `detector_errors` sin fallar | startup probe; un MoE carga lento y un detector roto de una org no reinicia el pod |
| `/readyz` | Base escribible, todos los hilos de jobs vivos, cola del carril por defecto bajo el tope y, en modo vLLM, upstream que responde (`ping`) | readiness |
| `/metrics` | Prometheus, puerto o ruta interna con token | observabilidad |

Ninguna ruta de health devuelve datos de tenants.

## 6. Circuito cerrado de verificación

`make check` (o `python -m vigia.selfcheck`) corre en orden: ruff, pytest rápido (CPU, modelos
diminutos), test de import core→ee, test de CSP, e2e del panel con Playwright y el benchmark de
overhead sobre el modelo diminuto. El experimento de evidencia con Granite corre aparte con
`pytest -m slow` porque usa 3 GB de RAM.

## 7. Controles de deriva

- `ModelInfo.router_sha256` hashea los pesos de todos los routers. Si el detector y el modelo lo
  tienen y difieren, `check_model` rechaza el detector. Un detector viejo sin el campo sigue válido.
- `vigia/drift.py` calcula PSI entre la referencia guardada en el detector y las últimas trazas.
  El panel lo muestra en Rendimiento y `/metrics` expone `vigia_drift_psi{feature}`.

## 8. Hardware

`VIGIA_DEVICE` (cpu, cuda, cuda:N, mps) y `VIGIA_DTYPE` (auto, float32, bfloat16, float16) eligen
dónde corre el backend HF. Los pesos se cargan solo desde safetensors y sin código remoto.
