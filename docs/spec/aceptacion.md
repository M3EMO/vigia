# Plan de tests de aceptación

Cada test de aceptación (AT) tiene un oráculo computable. Ningún AT depende de un LLM que se
evalúa a sí mismo. La columna "Gate" indica dónde corre: `fast` (pytest por defecto, CPU,
modelos diminutos), `slow` (Granite real), `e2e` (plataforma + navegador) o `gpu` (pendiente de
hardware).

Estado de release: todos los AT `fast`, `e2e` y `slow` pasan. Los AT `gpu` quedan documentados como
pendientes con su procedimiento.

## Señales (P-01, P-02, P-03)

| AT | Dado | Entonces | Gate |
|---|---|---|---|
| AT-S01 | Un modelo diminuto de cada arquitectura soportada | los logits recalculados por el pre-hook son `allclose` a los del router, y el top-k reconstruido coincide con el del modelo | fast |
| AT-S02 | Un MLP parcheado que no llama al router (MXFP4 simulado) | el pre-hook igual captura logits | fast |
| AT-S03 | Un router uniforme / one-hot sintético | `r_ent_n = 1 / 0`, `u_ihhi` y `u_gini` dan sus valores cerrados | fast |
| AT-S04 | Una secuencia y la misma secuencia con tokens extra al final | las features de los primeros t tokens no cambian (causalidad) | fast |
| AT-S05 | Un batch con padding a izquierda | las posiciones de padding no generan features | fast |
| AT-S06 | El extractor sale del context manager, aun con excepción | no queda ningún hook registrado | fast |
| AT-S07 | Un `model_type` no soportado o un modelo denso | error claro que lista las arquitecturas soportadas | fast |
| AT-S08 | Generación con streaming de N tokens | se emiten exactamente N eventos con offsets que reconstruyen el texto | fast |
| AT-S09 | Texto con tildes, ñ, emojis y bytes UTF-8 partidos entre tokens | los offsets en code points cubren el texto sin huecos ni solapes | fast |
| AT-S10 | Una secuencia harmony con canales `analysis` y `final` | solo los tokens de `final` reciben score; los especiales tienen `score=null` | fast |
| AT-S11 | F2 activado sobre un modelo diminuto | la salida del modelo con el backend instrumentado es igual a la salida eager | fast |

## Detector y calibración (P-04 a P-09)

| AT | Dado | Entonces | Gate |
|---|---|---|---|
| AT-M01 | Un detector entrenado | guardar y cargar el JSON reproduce los scores con tolerancia 1e-9 y el hash valida | fast |
| AT-M02 | Un JSON con hash alterado o para otro modelo | la carga falla con error explícito | fast |
| AT-M03 | Features sintéticas con señal conocida (semilla fija) | AUROC del detector > 0,9 y ECE < 0,05 tras Platt | fast |
| AT-M04 | Un dataset con preguntas repetidas | ningún id de pregunta aparece en dos splits | fast |
| AT-M05 | CRC con alpha = 0,1 sobre datos sintéticos intercambiables | la FNR observada en test es ≤ 0,1 + tolerancia de muestreo en 200 semillas (media) | fast |
| AT-M06 | Respuestas y gold en español con artículos, tildes y puntuación | EM y F1 de tokens coinciden con valores calculados a mano | fast |
| AT-M07 | Un juez falso que devuelve spans inexistentes o JSON inválido | el validador rechaza el item y lo cuenta en el reporte | fast |
| AT-M08 | Etiquetas de una sola clase | el reporte falla con mensaje claro, no con AUROC inventado | fast |
| AT-M09 | Un reporte de evaluación | contiene las métricas listadas, IC por bootstrap de cluster y los 3 baselines | fast |
| AT-M10 | Benchmark sobre modelo diminuto | reporta p50/p95/p99 de hooks on/off y el ratio; el overhead F1 es < 25 % en el diminuto | fast |
| AT-M11 | **Experimento de evidencia** con Granite 1B-A400M y QA en español con gold | el reporte se genera de punta a punta y publica AUROC por token de F0, F1, F0+F1 y baselines con IC 95 % | slow |
| AT-M12 | Overhead F1 en Granite (CPU) | se mide y se publica p50/p95; la meta < 5 % se compara sin ocultar el resultado | slow |
| AT-M13 | gpt-oss-20b en GPU | AUROC y overhead medidos con el mismo procedimiento | gpu |

## Servicio y API (P-10 a P-16)

| AT | Dado | Entonces | Gate |
|---|---|---|---|
| AT-A01 | Un request sin `extra_body.vigia` | la respuesta valida contra el esquema de OpenAI y no tiene campos extra | fast |
| AT-A02 | Un request con `vigia` | cada choice trae `vigia.tokens`, `vigia.spans`, `calibration_id` y `action` | fast |
| AT-A03 | Streaming SSE | cada chunk trae sus tokens, el chunk final trae spans y acción, y termina en `[DONE]` | fast |
| AT-A04 | El cliente oficial `openai` con `base_url` local | el SDK parsea la respuesta y expone `model_extra["vigia"]` | fast |
| AT-A05 | Scores sintéticos con huecos y subtokens | los fragmentos respetan `gap_max`, `min_tokens` y el score max por palabra | fast |
| AT-A06 | Una respuesta que supera cada umbral | la acción elegida respeta la prioridad abstain > rag_retry > human_review > annotate > pass | fast |
| AT-A07 | rag_retry con retriever falso | se re-consulta una vez con contexto; un host fuera de la allowlist se rechaza | fast |
| AT-A08 | El detector lanza una excepción | `fail_closed` → 503; `fail_open` → `score:null`, `degraded:true` | fast |
| AT-A09 | Un servidor vLLM falso con `routed_experts` uint8/uint16/int16 | el modo reducido produce features U y marca `mode:"reduced"` | fast |
| AT-A10 | `vigia extract → calibrate → evaluate` por CLI sobre el modelo diminuto | cada comando sale con código 0 y deja sus artefactos | fast |

## Plataforma (P-17 a P-37)

| AT | Dado | Entonces | Gate |
|---|---|---|---|
| AT-P01 | Dos tenants con datos | cada endpoint de API y panel devuelve 404 a los recursos del otro tenant (barrido automático de rutas) | fast |
| AT-P02 | Cada ruta del router | declara permiso; un viewer no puede mutar; deny-by-default (test por introspección) | fast |
| AT-P03 | Una API key | se muestra una vez, se guarda como HMAC, una key revocada o vencida devuelve 401, una key en query string se rechaza | fast |
| AT-P04 | Login | contraseña < 15 caracteres rechazada, sesión regenerada, cookie `__Host-` Secure HttpOnly SameSite=Strict, logout invalida en servidor, inactividad expira | fast |
| AT-P05 | Un POST del panel sin token CSRF o con `Sec-Fetch-Site: cross-site` | 403 | fast |
| AT-P06 | Ráfaga de requests | 429 con `Retry-After`; login bloqueado por IP tras N fallos | fast |
| AT-P07 | Body enorme / max_tokens enorme | 413 / 422 | fast |
| AT-P08 | Texto con DNI, CUIT válido, CBU, tarjeta Luhn, email, teléfono | `store_text=redacted` guarda los marcadores y nunca el dato | fast |
| AT-P09 | La cadena de auditoría | alterar una fila rompe la verificación | fast |
| AT-P10 | Retención de N días | el job purga lo vencido y nada más; export JSON del tenant incluye todo lo suyo y nada ajeno | fast |
| AT-P11 | Webhook | firma Standard Webhooks verificable, reintento ante 500, destino localhost o 169.254.169.254 bloqueado | fast |
| AT-P12 | Cualquier página del panel | CSP sin `unsafe-inline` ni `unsafe-eval`, headers de seguridad presentes, sin recursos externos | fast |
| AT-P13 | Texto del modelo con `<script>`, bidi y control | se renderiza escapado; CSV neutraliza `=`, `+`, `-`, `@` | fast |
| AT-P14 | `/livez`, `/readyz`, `/startupz`, `/metrics` | responden según su contrato y sin datos de tenants | fast |
| AT-P15 | Revisión en la cola | dos revisores, mínimo 2, decisiones ocultas hasta completar, etiquetas exportables a calibración | fast |
| AT-P16 | Subir un JSONL de calibración y activar | el job entrena, el reporte aparece y el nuevo detector puntúa los requests siguientes | fast |
| AT-P17 | Errores internos | respuesta genérica con `request_id`, sin stack trace | fast |
| AT-P18 | `vigia/` importa algo de `vigia_ee` | el test falla | fast |

## Panel en navegador real

| AT | Dado | Entonces | Gate |
|---|---|---|---|
| AT-E01 | Owner crea org, invita revisor, crea key, usa el playground | todo el flujo funciona sin errores de consola | e2e |
| AT-E02 | Revisor completa la cola solo con teclado | la cola avanza y las etiquetas quedan guardadas | e2e |
| AT-E03 | Viewer intenta mutar | los controles no aparecen y el POST directo devuelve 403 | e2e |
| AT-E04 | axe-core sobre cada página | cero violaciones serias o críticas de WCAG 2.2 AA | e2e |
| AT-E05 | Ancho 375 px | sin scroll horizontal; el detalle se apila | e2e |
| AT-E06 | Contraste de cada nivel de resaltado, claro y oscuro | ≥ 4,5:1 | fast |
| AT-E07 | JavaScript desactivado | las páginas y formularios principales siguen funcionando | e2e |
