# Operación de la plataforma Vigía

Esta guía cubre la plataforma (`vigia_ee`, comando `python -m vigia_ee.app` o `vigia-platform`).
El servidor single-tenant del núcleo (`vigia serve`) se configura por argumentos y no lee estas
variables. La fuente de verdad es `Settings.from_env` en `vigia_ee/runtime.py`.

Advertencia: el `Dockerfile` y el `compose.yaml` no se construyeron en la máquina de desarrollo.
Esa máquina no tiene Docker. Los comandos fuera de Docker de esta guía sí se probaron con
`tiny:mixtral` el 2026-10-04.

## 1. Variables de entorno

Reglas comunes:

- Un entero acepta guion bajo como separador (`1_048_576`).
- Un booleano acepta `1/true/yes/on/si/sí` o `0/false/no/off`.
- Una variable vacía equivale a no definida.
- Un secreto con variante `*_FILE` lee el archivo y recorta espacios. Si existen las dos, gana `*_FILE`.
- Un valor inválido corta el arranque con exit code 2 y un mensaje que nombra la variable.

### 1.1 Modelo y backend

| Variable | Default | Obligatoria | Descripción |
|---|---|---|---|
| `VIGIA_MODEL` | — | sí | Id de Hugging Face, ruta local, `tiny:mixtral` o `tiny:gpt_oss`. El texto exacto entra en la huella del detector (ver §9). |
| `VIGIA_BACKEND` | `vllm`; `hf` con `tiny:*` | no | `vllm` actúa como proxy en modo reducido (default de producción). `hf` hospeda el modelo en el proceso y da la mejor detección (§13). |
| `VIGIA_BASE_DETECTOR` | — | no | Ruta a un `detector.json`. Se instala y activa en cada org sin detectores, si coincide con el modelo. |
| `VIGIA_VLLM_URL` | — | con `vllm` | URL http(s) del servidor vLLM, sin `/v1`. |
| `VIGIA_VLLM_MODEL` | — | con `vllm` | Nombre del modelo en vLLM. |
| `VIGIA_VLLM_API_KEY` / `_FILE` | — | no | Key que Vigía manda a vLLM. |
| `VIGIA_VLLM_N_EXPERTS` | — | con `vllm` | Expertos por capa del modelo servido (2–4096). Sale de `config.json`. |
| `VIGIA_VLLM_TOP_K` | — | con `vllm` | Expertos elegidos por token (1–4096, ≤ `N_EXPERTS`). |
| `VIGIA_VLLM_N_LAYERS` | — | con `vllm` | Capas MoE del modelo servido (1–1024). |
| `VIGIA_VLLM_MAX_CONCURRENCY` | `16` | no | Generaciones simultáneas hacia vLLM (1–4096). |

### 1.2 Datos y secretos

| Variable | Default | Obligatoria | Descripción |
|---|---|---|---|
| `VIGIA_DATA_DIR` | `var` (relativo al directorio actual) | no | Carpeta de datos. La imagen usa `/var/lib/vigia`. |
| `VIGIA_DB_PATH` | `$VIGIA_DATA_DIR/vigia.db` | no | Ruta de la base SQLite. |
| `VIGIA_PEPPERS_FILE` | — (imagen: `/run/secrets/vigia_peppers`) | una de las dos | Archivo con peppers `kid:secreto`, uno por línea o separados por coma. El último es el vigente. Cada secreto tiene 32 caracteres o más. |
| `VIGIA_PEPPERS` | — | una de las dos | Igual que el archivo, en la variable. Usala solo en desarrollo. |
| `VIGIA_OWNER_PASSWORD_FILE` | — (imagen: `/run/secrets/vigia_owner_password`) | no | Solo la usa `init`. Sin ella, `init` pide la contraseña con `getpass`. |
| `VIGIA_METRICS_TOKEN` / `_FILE` | — | no | Bearer de `/metrics`, 16 caracteres o más. Sin token, `/metrics` responde 404. |

### 1.3 Límites y seguridad

| Variable | Default | Obligatoria | Descripción |
|---|---|---|---|
| `VIGIA_MAX_TOKENS_LIMIT` | `1024` | no | Tope de `max_tokens` por request (1–32768). Más que eso da 422. |
| `VIGIA_DEFAULT_MAX_TOKENS` | `256` | no | `max_tokens` cuando el request no lo trae (1–32768, ≤ límite). |
| `VIGIA_MAX_BODY_BYTES` | `1048576` | no | Tope del body (1 KiB–64 MiB). Más que eso da 413. |
| `VIGIA_MAX_UPLOAD_BYTES` | `20971520` | no | Tope del JSONL de calibración en `/calibration/datasets` (1 KiB–256 MiB), más 64 KiB de margen multipart. |
| `VIGIA_KEY_RATE_PER_MINUTE` | `120` | no | Requests por minuto por API key; la ráfaga es igual al valor. |
| `VIGIA_TENANT_RATE_PER_MINUTE` | `600` | no | Requests por minuto por organización. |
| `VIGIA_SECURE_COOKIES` | `true` | no | Cookie de sesión `Secure`. No la apagues en producción: ver §11. |
| `VIGIA_DEVICE` | `cpu` | no | Dispositivo del backend HF: `cpu`, `cuda`, `cuda:N` o `mps`. |
| `VIGIA_DTYPE` | `auto` | no | Dtype de los pesos: `auto`, `float32`, `bfloat16` o `float16`. Cambiarlo cambia `router_sha256` y obliga a recalibrar. |
| `VIGIA_RAG_ALLOWED_HOSTS` | vacío | no | Hosts, separados por coma, que habilitan http y redes privadas para `rag_url`. |
| `VIGIA_WEBHOOK_ALLOWED_HOSTS` | vacío | no | Hosts que habilitan http y redes privadas para webhooks. |

### 1.4 Operación

| Variable | Default | Obligatoria | Descripción |
|---|---|---|---|
| `VIGIA_MAX_QUEUE_DEPTH` | `1000` | no | `/readyz` falla si la cola de jobs (purga y calibración) alcanza este valor. |
| `VIGIA_REVIEW_MIN_REVIEWERS` | `2` | no | Revisores necesarios para cerrar un item de la cola (1–10). |
| `VIGIA_SHUTDOWN_TIMEOUT` | `30` | no | Segundos que el cierre espera los requests en curso (1–3600). |

### 1.5 Variables del núcleo y de terceros

| Variable | Dónde | Descripción |
|---|---|---|
| `VIGIA_JUDGE_API_KEY` | `vigia label` y `vigia evidence` | Key del juez OpenAI-compatible. La plataforma no la lee. |
| `HF_HOME` | transformers | Cache de modelos. La imagen usa `/opt/vigia/hf`. |
| `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE` | transformers | `1` en la imagen: nada sale a la red. |

## 2. Arranque y argumentos de `serve`

```bash
python -m vigia_ee.app serve [--host 127.0.0.1] [--port 8000] [--proxy-headers] [--forwarded-allow-ips 127.0.0.1]
```

- `serve` corre un solo worker. El modelo y los rate limiters viven en ese proceso.
- El modelo carga antes de abrir el puerto. Un MoE real tarda minutos.
- Detrás de un proxy TLS, usá `--proxy-headers --forwarded-allow-ips <IP del proxy>`. Sin eso,
  todos los logins comparten la IP del proxy y el bloqueo por IP los frena juntos.
- SIGTERM marca el proceso como cerrando. Las generaciones nuevas reciben 503 con `Retry-After`.
  El cierre espera hasta `VIGIA_SHUTDOWN_TIMEOUT` y después cancela lo pendiente.

## 3. Health checks

| Ruta | 200 cuando | 503 cuando | Uso |
|---|---|---|---|
| `/livez` | el proceso responde | nunca | liveness |
| `/startupz` | el modelo cargó y todos los detectores de las orgs cargaron sin error | falta el modelo o algún detector falló al cargar | startup probe |
| `/readyz` | la base acepta escrituras, el worker de jobs está vivo y la cola está bajo `VIGIA_MAX_QUEUE_DEPTH` | falla cualquiera de los tres | readiness |
| `/metrics` | token correcto | — (401 sin token válido; 404 sin `VIGIA_METRICS_TOKEN`) | Prometheus |

Las cuatro rutas no devuelven datos de tenants. `/readyz` responde
`{"status": ..., "checks": {"db", "worker", "queue"}}`. `/startupz` responde
`{"checks": {"model", "detectors"}}`.

Un `/startupz` en 503 con `detectors: false` persiste hasta reiniciar. Buscá el evento
`detector_warm_error` en el log: nombra la org con el detector roto.

Configuración sugerida en Kubernetes: startup probe sobre `/startupz` con `failureThreshold` alto
(un MoE grande tarda en cargar), liveness sobre `/livez` y readiness sobre `/readyz`.

## 4. Métricas Prometheus

`GET /metrics` con `Authorization: Bearer <VIGIA_METRICS_TOKEN>`. Formato de texto 0.0.4. Ningún
label lleva org, usuario, email ni id de traza.

| Métrica | Tipo | Labels | Significado |
|---|---|---|---|
| `vigia_requests_total` | counter | `route` (plantilla), `status` (`2xx`…`5xx`) | Requests HTTP |
| `vigia_request_duration_seconds` | histogram | `route` | Duración de requests |
| `vigia_overhead_ratio` | histogram | — | Overhead de Vigía sobre la generación base (0,05 = 5 %) |
| `vigia_tokens_scored_total` | counter | — | Tokens puntuados por un detector calibrado |
| `vigia_low_confidence_tokens_total` | counter | — | Tokens con `p_error ≥ τ` |
| `vigia_actions_total` | counter | `action` | Acción de política por respuesta |
| `vigia_calibration_info` | gauge | `detector_id` | Detectores activos (valor 1) |
| `vigia_auth_failures_total` | counter | — | Respuestas 401 y 403 |
| `vigia_rate_limited_total` | counter | — | Respuestas 429 |
| `vigia_review_queue_depth` | gauge | — | Items pendientes de revisión |
| `vigia_jobs_queue_depth` | gauge | — | Jobs pendientes o en curso (todos los carriles) |
| `vigia_build_info` | gauge | `version` | Versión instalada (valor 1) |

Más de 200 plantillas de ruta caen en `route="other"`. Un request sin ruta cuenta como `unmatched`.
Si la base está bloqueada, `/metrics` espera como máximo 1 s y repite el último valor de las colas.

Alertas sugeridas: `rate(vigia_requests_total{status="5xx"}[5m]) > 0`, crecimiento de
`vigia_auth_failures_total`, `vigia_review_queue_depth` alto y `vigia_overhead_ratio` p95 > 0,05.

## 5. Logs

- Formato: una línea JSON por evento en stderr. Uvicorn y asyncio usan el mismo formato.
- Campos permitidos (allowlist de `vigia_ee/security.py`): `ts`, `level`, `logger`, `msg`,
  `event`, `request_id`, `method`, `path`, `status`, `duration_ms`, `org_id`, `user_id`, `key_id`,
  `action`, `scope`, `retry_after`, `error_type`, `frames`. Todo otro campo se descarta.
- El log nunca lleva prompts, respuestas, tokens, contraseñas ni cookies. El path va sin query string.
- Un error interno guarda tipo y frames (archivo, línea, función), sin el mensaje.
- Cada respuesta lleva `X-Request-ID`; el mismo valor aparece en el log.
- La plataforma no rota archivos. Configurá rotación en el driver de logs de Docker o en journald.

Eventos útiles: `http_request`, `login_failed`, `api_key_rejected`, `csrf_rejected`,
`rate_limited`, `unhandled_error`, `detector_warm_error`, `detector_rejected`,
`base_detector_rejected`, `policy_invalid`, `retention_purge_error`, `shutdown_started`.

La auditoría de negocio (logins, keys, roles, políticas, calibraciones, exports, borrados) no va al
log: va a la tabla `audit_log`, con cadena de hashes. El panel la exporta en `/audit/export`.

## 6. Backups y restore

La base es un archivo SQLite en modo WAL (`vigia.db` más `vigia.db-wal` y `vigia.db-shm`). Copiar
solo `vigia.db` con el proceso andando puede perder transacciones. Usá la API de backup de SQLite:
es consistente con el proceso en marcha.

```bash
# Fuera de Docker
python -c "import sqlite3, sys; src = sqlite3.connect(sys.argv[1]); dst = sqlite3.connect(sys.argv[2]); src.backup(dst); print(dst.execute('PRAGMA integrity_check').fetchone()[0]); dst.close(); src.close()" \
  var/vigia.db backups/vigia-$(date -u +%Y%m%dT%H%M%SZ).db

# Con compose (el contenedor de un solo uso comparte el volumen de datos)
mkdir -p backups && sudo chown 10001:10001 backups
docker compose run --rm -v "$PWD/backups:/backups" vigia python -c "import sqlite3, sys; src = sqlite3.connect(sys.argv[1]); dst = sqlite3.connect(sys.argv[2]); src.backup(dst); print(dst.execute('PRAGMA integrity_check').fetchone()[0]); dst.close(); src.close()" \
  /var/lib/vigia/vigia.db /backups/vigia-$(date -u +%Y%m%dT%H%M%SZ).db
```

El comando imprime `ok` si la copia pasa `integrity_check`. Verificá también la cadena de
auditoría de cada org sobre la copia:

```bash
python -c "
import sys
from vigia_ee.audit import verify_chain
from vigia_ee.db import Database, list_org_ids
db = Database(sys.argv[1])
for org in list_org_ids(db):
    print(org, verify_chain(db, org))
db.close()
" backups/vigia-AAAAMMDDTHHMMSSZ.db
```

Cifrá los backups y guardalos fuera del nodo. Contienen texto de clientes según `store_text`.

Restore:

1. Detené la plataforma (`docker compose stop vigia`).
2. Reemplazá `vigia.db` por la copia y borrá `vigia.db-wal` y `vigia.db-shm` viejos.
3. Dejá el archivo con dueño `10001:10001` y permisos `0640`.
4. Arrancá. Si la copia tiene un esquema más viejo, las migraciones la actualizan solas.
5. Verificá `/startupz`, `/readyz` y la cadena de auditoría.

Un restore revive lo que se borró después del backup: trazas suprimidas a pedido y orgs borradas.
Llevá aparte (fuera de la base) el registro de supresiones y re-aplicalas tras cada restore. La
ventana de retención de los backups debe figurar en tu política de datos.

## 7. Rotación de secretos

### 7.1 Peppers de API keys

Cada key guarda el `kid` del pepper con el que se creó. El último pepper del archivo firma las keys
nuevas; los anteriores siguen validando sus keys.

1. Agregá una línea `k2:<secreto de 32+ caracteres>` al final del archivo de peppers.
2. Reiniciá. Las keys nuevas usan `k2`; las viejas siguen funcionando.
3. Listá las keys vigentes que todavía usan `k1`:

   ```bash
   python -c "import sqlite3, sys; db = sqlite3.connect(sys.argv[1]); [print(r) for r in db.execute('SELECT org_id, id, name FROM api_keys WHERE pepper_kid = ? AND revoked_at IS NULL', (sys.argv[2],))]" var/vigia.db k1
   ```

4. Cada org crea una key nueva, la despliega y revoca la vieja en Configuración.
5. Cuando la consulta no devuelve filas, quitá `k1` del archivo y reiniciá. Una key con `k1` que
   quedara viva pasa a responder 401.

### 7.2 API keys

La revocación es inmediata, sin cache. La rotación sin corte es manual:

1. Creá la key nueva (panel → Configuración, o `create-key`). El token se muestra una sola vez.
2. Desplegá la key nueva en el cliente.
3. Revocá la vieja en el panel. La columna "último uso" confirma que nadie la usa.

`create-key --expires-days N` crea keys con vencimiento. Una key vencida responde 401.

### 7.3 Otros secretos

- Token de métricas: cambiá el archivo, reiniciá y actualizá Prometheus.
- Contraseñas: cada usuario la cambia en `/account`. Cambiarla cierra todas sus sesiones.
- Secretos de webhooks: se regeneran al recrear el webhook. El receptor acepta varias firmas
  durante la transición (formato Standard Webhooks).

## 8. Retención y borrado

| Control | Dónde | Detalle |
|---|---|---|
| `store_text` | Configuración (owner) | `none` no guarda texto, `redacted` (default) guarda marcadores de PII, `full` guarda todo. |
| `retention_days` | Configuración (owner) | 1 a 3650 días (default 30 al crear la org con `init`). |
| Purga automática | worker de jobs | Corre al arrancar y cada 24 h. Borra trazas vencidas con sus tokens, fragmentos e items de revisión. Cada corrida queda en auditoría (`retention_purge`). |
| Supresión puntual | Generaciones → borrar (owner) | Borra una traza y sus dependientes. Queda en auditoría (`trace_delete`). |
| Export del tenant | Datos → exportar (admin) | JSON con usuarios, keys (sin HMAC), trazas, revisión, detectores, datasets y webhooks. |
| Borrado del tenant | Datos → borrar (owner) | Borra todo el tenant. Pide escribir el nombre exacto de la org. |

La auditoría no se purga: es evidencia y no guarda texto de clientes. SQLite deja páginas libres
con datos borrados hasta que las reusa. Si necesitás borrado físico inmediato, corré `VACUUM` con
la plataforma detenida. Los backups conservan los datos hasta que vencen.

## 9. Actualización y migraciones

1. Hacé un backup (§6).
2. Construí o descargá la imagen nueva.
3. `docker compose up -d`. Al abrir la base, Vigía aplica las migraciones pendientes, cada una
   atómica. No existen migraciones hacia atrás: volver de versión exige el backup.
4. Verificá `/startupz` y `/readyz`.

Un detector solo vale para el modelo con el que se calibró. La huella compara `model_id`,
`model_type`, `n_experts`, `top_k`, `quantization` y, si existe, el hash del router. `model_id` es
el texto de `VIGIA_MODEL`: calibrar con `ibm-granite/granite-3.1-1b-a400m-instruct` y servir con
`/models/granite` no coincide. Un detector que no coincide no se usa: la API devuelve
`p_error: null` y el panel muestra el motivo.

## 10. Dimensionamiento

| Modelo | RAM del proceso | Fuente |
|---|---|---|
| `tiny:mixtral`, `tiny:gpt_oss` | < 1 GB, casi todo torch | estimado |
| Granite 3.1 1B-A400M (bf16, CPU) | ~3 GB | medido en la corrida del informe técnico |
| gpt-oss-20b (MXFP4) | ~16 GB de VRAM en GPU; en CPU transformers descuantiza a bf16 (~42 GB) | estimado, no probado |
| gpt-oss-120b (MXFP4) | ~80 GB de VRAM | estimado, no probado |
| Mixtral 8x7B (bf16) | ~95 GB | estimado: 47 B parámetros × 2 bytes |

Regla para estimar: parámetros totales × bytes por parámetro, más 10–20 % de activaciones y cache.
En un MoE pesan los parámetros totales, no los activos.

Otros recursos:

- CPU: la plataforma con backend HF genera de a una respuesta por vez. Las demás esperan en cola
  hasta 120 s (máximo 64 en espera) y después reciben 503 `server_overloaded`.
- Disco: cada traza guarda una fila por token. Con `store_text=none` las filas no llevan texto.
- `/tmp`: 256 MB alcanzan para la plataforma.

## 11. Acceso por navegador y TLS

La cookie de sesión es `__Host-vigia_session` con `Secure`, `HttpOnly` y `SameSite=Strict`. El
navegador solo la guarda en HTTPS o en `http://127.0.0.1` / `http://localhost`.

- Acceso local: `http://127.0.0.1:8000` funciona sin TLS (compose publica solo en loopback).
- Acceso remoto: poné un proxy TLS (nginx, Caddy, Traefik) delante y corré `serve` con
  `--proxy-headers`. La app ya manda HSTS.
- `VIGIA_SECURE_COOKIES=false` emite la cookie `vigia_session_dev` sin `Secure` y sin el prefijo
  `__Host-`. Sirve solo para desarrollo en un host sin TLS. En producción dejala en `true`.

## 12. Modelo sin red

La imagen corre con `HF_HUB_OFFLINE=1`: transformers solo lee el cache de `HF_HOME`
(`/opt/vigia/hf`). Dos opciones para poner el modelo ahí:

1. **En la imagen:** `docker build --build-arg PREFETCH_MODEL=ibm-granite/granite-3.1-1b-a400m-instruct .`
   La descarga ocurre durante el build, con red.
2. **Montado:** descargá el cache en una máquina con red y montalo en solo lectura.

   ```bash
   HF_HOME=./models/hf python -c "from huggingface_hub import snapshot_download; snapshot_download('ibm-granite/granite-3.1-1b-a400m-instruct')"
   ```

En los dos casos, `VIGIA_MODEL` es el id del repo, no la ruta. Así coincide con el detector.
Para comprobar el arranque sin red: `docker run --network none ...` debe pasar `/startupz`.
Sin probar: si alguna versión de transformers intenta escribir en un cache de solo lectura, montalo
con escritura.

## 13. Modo vLLM reducido

El modo vLLM es el default con un modelo real (decisión 2026-10-13). Medido en gpt-oss-20b sobre una
L4 (informe §4.10): dentro de MKQA da AUROC 0,784 contra 0,837 del backend HF, y devolver los
expertos no agrega latencia, ni con 32 requests simultáneos. El backend HF suma +4,7 % por token.
En una L4, el throughput sube de 279 a 491 tokens/s entre 16 y 32 requests: con GPU de sobra,
conviene subir `VIGIA_VLLM_MAX_CONCURRENCY` sobre el default de 16. Elegí HF con `VIGIA_BACKEND=hf`
cuando la detección pesa más que la latencia.

Con `VIGIA_BACKEND=vllm`, Vigía no carga el modelo. Hace de proxy hacia un vLLM lanzado con
`--enable-return-routed-experts` (ADR A-08). El ejemplo comentado está en `compose.yaml`.

Qué cambia frente al backend HF:

- vLLM no expone los logits del router. No hay F1.
- F0 queda parcial: `out_nll`, `out_margin` y `out_top20_entropy` (logprobs top-20).
  `out_entropy` necesita el vocabulario completo y queda vacía.
- Se agrega el nivel U: uso acumulado de expertos y cambio de ruta, a partir de los IDs elegidos.
- Las respuestas marcan `mode: "reduced"`.
- vLLM entrega los IDs al final del request. El streaming SSE llega de una vez al terminar.
- Un detector calibrado con el backend HF no sirve: cambian la huella y las features. Calibrá con
  datos generados por el mismo vLLM.
- `VIGIA_VLLM_N_EXPERTS`, `_TOP_K` y `_N_LAYERS` fijan la normalización de U. Sin ellos, el
  `p_error` de un token dependería de tokens futuros. Copialos del `config.json` del modelo.
- Una falla de vLLM responde 502 `upstream_unavailable` con `Retry-After`.
- Probado contra vLLM 0.31 real en una L4. vLLM 0.31 entrega `routed_experts` como `.npy` en base64,
  con las filas del prompt; con gpt-oss, Vigía puntúa solo el canal `final`.

Seguridad de vLLM: su `--api-key` protege solo rutas bajo `/v1`. Rutas como `/invocations`,
`/pause` o `/update_weights` quedan sin auth. Nunca publiques vLLM: dejalo en una red interna que
solo alcance Vigía.

## 14. Troubleshooting

Los mensajes salen en stderr con el prefijo `error:` y exit code 2.

| Mensaje | Causa | Qué hacer |
|---|---|---|
| `Falta VIGIA_MODEL (id o ruta del modelo HF, o 'tiny:mixtral' para la demo).` | Variable sin definir | Definí `VIGIA_MODEL`. |
| `Falta VIGIA_PEPPERS_FILE o VIGIA_PEPPERS.` | Sin peppers | Creá el archivo (ver README) y apuntá la variable. |
| `No se pudo leer VIGIA_PEPPERS_FILE.` (también `VIGIA_METRICS_TOKEN_FILE`, `VIGIA_VLLM_API_KEY_FILE`, `VIGIA_OWNER_PASSWORD_FILE`) | Ruta inexistente o sin permiso de lectura | En compose sin swarm, el uid 10001 debe poder leer el archivo del host (`chmod 0444`). |
| `Pepper mal formado: se espera kid:secreto.` | Línea sin `:` o kid con caracteres fuera de `A-Za-z0-9-` | Corregí el archivo. |
| `El pepper k1 tiene menos de 32 caracteres.` | Secreto corto | Generá uno con `secrets.token_urlsafe(48)`. |
| `VIGIA_METRICS_TOKEN debe tener al menos 16 caracteres.` | Token corto | Generá uno más largo. |
| `VIGIA_X debe ser un entero.` / `VIGIA_X debe ser un entero entre A y B.` | Valor fuera de rango | Ver tablas de §1. |
| `VIGIA_DEFAULT_MAX_TOKENS no puede superar VIGIA_MAX_TOKENS_LIMIT.` | Default mayor que el tope | Bajá el default o subí el tope. |
| `VIGIA_SECURE_COOKIES debe ser true o false.` | Booleano inválido | Usá `true` o `false`. |
| `VIGIA_BACKEND debe ser 'hf' o 'vllm'.` | Backend desconocido | Corregí el valor. |
| `Con VIGIA_BACKEND=vllm hacen falta VIGIA_VLLM_URL y VIGIA_VLLM_MODEL.` | Falta configuración de vLLM | Definí las dos. |
| `VIGIA_VLLM_URL debe ser una URL http(s) con host.` | URL mal formada | Usá `http://host:puerto`. |
| `Con VIGIA_BACKEND=vllm hacen falta VIGIA_VLLM_N_EXPERTS, ... (config.json del modelo servido).` | Falta la forma del modelo | Copiala del `config.json`. |
| `VIGIA_VLLM_TOP_K no puede superar VIGIA_VLLM_N_EXPERTS.` | Forma inconsistente | Revisá el `config.json`. |
| `Modelo diminuto desconocido; opciones: tiny:mixtral, tiny:gpt_oss.` | `tiny:` con otra arquitectura | Usá una de las dos. |
| `No se puede cargar el modelo (VIGIA_MODEL=...): ...` | Repo inexistente, ruta sin tokenizer o modelo fuera del cache con `HF_HUB_OFFLINE=1` | Revisá el id y el cache (§12). |
| `No se puede crear la carpeta de datos (VIGIA_DATA_DIR=...): ...` | Sin permiso sobre la carpeta padre | Creá la carpeta o cambiá la ruta. |
| `No se puede abrir la base (...): la base es de solo lectura` | El volumen pertenece a root o se montó `:ro` | `chown -R 10001:10001` sobre el volumen. |
| `No se puede abrir la base (...): el archivo no es una base SQLite` | Ruta apunta a otro archivo | Revisá `VIGIA_DB_PATH`. |
| `La base no acepta escrituras (...): otra conexión tiene la base bloqueada` | Otro proceso con una transacción abierta | Cerrá el otro proceso. No corras dos `serve` sobre la misma base. |
| `No existe el detector base (VIGIA_BASE_DETECTOR=...).` / `El detector base es inválido (...)` | Ruta o artefacto incorrectos | Revisá el archivo; el hash del artefacto debe validar. |
| `La organización no existe.` | `create-key` con un `--org-id` incorrecto | Copiá el id que imprimió `init`. |
| `La contraseña debe tener al menos 15 caracteres.` / `La contraseña es demasiado común...` / `La contraseña no puede contener tu email.` | Política NIST de contraseñas | Usá una frase larga distinta del email. |
| `Las contraseñas no coinciden.` | `getpass` con dos valores distintos | Repetí, o usá `VIGIA_OWNER_PASSWORD_FILE`. |

Síntomas sin mensaje de arranque:

| Síntoma | Causa probable | Qué hacer |
|---|---|---|
| El login vuelve siempre a `/login` | La cookie `Secure` no se guarda por http a un host remoto | Usá TLS o `127.0.0.1` (§11). |
| `p_error: null` y `calibrated: false` | Sin detector activo, o detector de otro modelo | Mirá el banner del panel y el evento `detector_rejected`. |
| Log `base_detector_rejected` | `VIGIA_BASE_DETECTOR` se calibró para otro `VIGIA_MODEL` | Recalibrá con el mismo texto de modelo (§9). |
| 503 `server_overloaded` | Más de 64 generaciones en cola o espera > 120 s | Bajá la carga o pasá a vLLM. |
| 429 con `Retry-After` | Rate limit por key o por org | Subí `VIGIA_KEY_RATE_PER_MINUTE` o `VIGIA_TENANT_RATE_PER_MINUTE`. |
| 503 `detector_unavailable` en `/v1/*` | El detector falló y la política es `fail_closed` | Revisá el detector; `fail_open` responde con `degraded: true`. |
| `/readyz` con `queue: false` | Purga o calibración atrasadas | Mirá `vigia_jobs_queue_depth` y el log del worker. |
| `/startupz` en 503 con `detectors: false` | Un detector de alguna org no cargó | Buscá `detector_warm_error` (§3). |
