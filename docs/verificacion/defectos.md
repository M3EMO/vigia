# Log de defectos de la verificación adversarial

Generado del ciclo `vigia-verify-loop` (2026-10-04/05). Cada defecto pasó por un reparador del módulo dueño, que agregó un test de regresión o lo rechazó con motivo.

## Ronda 1

Defectos nuevos: 33.

| Sev. | Lente | Archivo | Defecto |
|---|---|---|---|
| critical | e2e | `vigia_ee/panel.py` | Si el cliente se desconecta durante un streaming, el lock del modelo queda tomado y se bloquea toda generación de todos los tenants |
| critical | api-edge | `vigia/api.py` | Un cliente que corta un stream deja tomado el lock del backend y bloquea a todos los tenants hasta un GC no determinista |
| critical | api-edge | `vigia/api.py` | 40 o más streams concurrentes agotan el threadpool y dejan el proceso en deadlock permanente (también /readyz) |
| high | a11y-visual | `vigia_ee/templates/policies.html` | Tablas angostas de alphas sin foco: violación serious scrollable-region-focusable a 375 px y a 320 px (reflow) |
| high | api-edge | `vigia/api.py` | vigia.tiers del cliente puede excluir features del detector: p_error se imputa con la media, la acción cambia y sigue calibrated=true (bypass de la política) |
| high | api-edge | `vigia/api.py` | policy.rag_fallback_action se ignora: siempre cae a human_review aunque el tenant eligió abstain |
| high | ml-correctness | `vigia/evidence.py` | gold_correct marca como correctas respuestas numéricas y de fecha con valor incorrecto: EM y F1 >= 0,5 pisan la comparación tipada |
| high | ops-resilience | `vigia_ee/jobs.py` | Los jobs huérfanos con lease vigente no se recuperan al reiniciar y quedan en 'running' para siempre |
| high | ops-resilience | `vigia_ee/webhooks.py` | El webhook caído de un solo tenant satura el único worker: atrasa la purga de los demás tenants y pone /readyz en rojo para toda la plataforma |
| medium | a11y-visual | `vigia_ee/static/app.css` | overflow-wrap:anywhere global en body parte palabras y números dentro de las tablas |
| medium | a11y-visual | `vigia_ee/static/app.css` | Los gráficos SVG escalan al ancho del contenedor: texto de 6,7 px en el reporte del detector, de 21 px en Rendimiento y de 58 px en el heatmap |
| medium | a11y-visual | `vigia_ee/static/app.css` | El placeholder de los inputs no cumple 4,5:1 en tema oscuro (3,75:1) |
| medium | api-edge | `vigia/api.py` | Streaming annotate_only con rag_retry devuelve action 'rag_retry' sin re-consultar y la guarda así en la traza |
| medium | api-edge | `vigia/api.py` | vigia.rag_status nunca se completa en la respuesta al cliente (siempre null) |
| medium | api-edge | `vigia/api.py` | Prompts que el backend rechaza (más largos que el contexto o con surrogates sueltos) devuelven 500 internal_error |
| medium | api-edge | `vigia/api.py` | finish_reason es 'stop' cuando la generación se corta por el contexto del modelo |
| medium | api-edge | `vigia_ee/runtime.py` | GET /v1/vigia/traces/{id} de la plataforma devuelve el texto original y los tokens de una respuesta en abstain |
| medium | api-edge | `vigia/api.py` | Un vigia.alpha por debajo del mínimo calibrado se trata como falla del detector: 503 con fail_closed o degradado silencioso con fail_open |
| medium | ml-correctness | `vigia/evidence.py` | Al reanudar, extract reutiliza filas viejas aunque el contenido del item cambió (mismo id), y el reporte publica el hash del dataset nuevo |
| medium | ops-resilience | `vigia_ee/app.py` | Un error de almacenamiento al arrancar (carpeta inválida, base de solo lectura o corrupta) sale como traceback crudo en vez de un mensaje claro |
| medium | ops-resilience | `vigia/backends/hf.py` | En streaming SSE, el modelo genera con autograd prendido porque torch.inference_mode no acompaña al generador entre hilos |
| low | a11y-visual | `vigia/charts.py` | Leyenda del heatmap superpuesta y fuera del viewBox con pocos expertos (por ejemplo Mixtral con 8) |
| low | a11y-visual | `vigia/charts.py` | Las marcas de los ejes Y no son números redondos (85,0 / 63,7 / 42,5 / 21,2; 3390,2 / 2542,6 / 1695,1 / 847,5) |
| low | a11y-visual | `vigia_ee/templates/detail.html` | Celda de token vacía en la tabla de señales para tokens de control (th sin nombre accesible) |
| low | a11y-visual | `vigia_ee/static/app.js` | El foco se pierde al body al generar en el playground con teclado |
| low | a11y-visual | `vigia_ee/static/app.css` | La tarjeta 'Editar la política' tiene un hueco superior de unos 42 px: el h2 conserva su margin-top |
| low | a11y-visual | `vigia_ee/static/app.css` | Los controles del filtro de Generaciones y Auditoría tienen alturas distintas y desalinean los labels |
| low | security | `vigia_ee/calibration.py` | Subida de dataset de calibración con JSON profundamente anidado lanza RecursionError no capturada y devuelve HTTP 500 en vez de 422 |
| low | isolation-rbac | `vigia_ee/panel_admin.py` | Borrar la organización (POST /data/delete) deja el detector del tenant borrado en la cache del runtime y en /metrics |
| low | api-edge | `vigia_ee/security.py` | Los errores de auth, body límite y 405 en /v1 de la plataforma no tienen la forma de error de OpenAI |
| low | api-edge | `vigia/api.py` | top_logprobs se acepta pero siempre devuelve []; logprobs en /v1/completions se acepta y devuelve null |
| low | ml-correctness | `vigia/labels.py` | parse_number_es lee '0.500' como 500: un cero a la izquierda se toma como grupo de miles |
| low | ml-correctness | `vigia/evaluate.py` | AURC y E-AURC dependen del orden de las filas cuando hay puntajes empatados |

## Ronda 2

Defectos nuevos: 40.

| Sev. | Lente | Archivo | Defecto |
|---|---|---|---|
| high | a11y-visual | `vigia_ee/static/app.css` | El foco de teclado queda oculto bajo la barra superior sticky al retroceder con Shift+Tab (WCAG 2.4.11 AA) |
| high | isolation-rbac | `vigia_ee/runtime.py` | /readyz cuenta la cola de webhooks: el backlog de webhooks de un tenant saca el nodo para todos (el arreglo previo no está conectado) |
| high | isolation-rbac | `vigia_ee/webhooks.py` | Carril de webhooks FIFO compartido y sin plazo total: el destino lento de un tenant frena los webhooks de los demás |
| high | ml-correctness | `vigia/backends/hf.py` | Con gpt-oss (harmony), teacher_force y los spans del dataset usan marcos de offsets distintos: el span marca el token equivocado o no queda ningún token puntuable |
| medium | e2e | `vigia_ee/templates/error.html` | La página de error del panel ofrece 'Reintentar' hacia la ruta POST-only que falló; el enlace da 405 en JSON |
| medium | a11y-visual | `vigia_ee/static/app.css` | El texto de los gráficos SVG baja de 10 px: 9,7 px en el resumen de escritorio y 6,4 px a 375 px (el arreglo previo quedó incompleto) |
| medium | a11y-visual | `vigia/charts.py` | Las tablas 'Ver datos' de los gráficos no tienen contenedor con scroll y provocan scroll horizontal de la página a 375 px y 320 px |
| medium | a11y-visual | `vigia/charts.py` | En la curva precisión-recall la etiqueta 'Base: prevalencia' se pisa con 'τ = …' cuando el punto del umbral cae en recall 1 |
| medium | a11y-visual | `vigia_ee/static/app.css` | En modo de colores forzados los tres niveles de resaltado se ven idénticos y las muestras de la leyenda pierden su texto |
| medium | isolation-rbac | `vigia_ee/calibration.py` | Un admin puede encolar calibraciones sin límite: atrasa las de otros tenants y la purga, y pone /readyz en rojo |
| medium | isolation-rbac | `vigia_ee/panel.py` | El playground no aplica la cuota por tenant: una org genera sin límite con el modelo compartido |
| medium | api-edge | `vigia/api.py` | usage.prompt_tokens siempre es 0 con los backends reales (HF y vLLM) |
| medium | api-edge | `vigia/backends/vllm_proxy.py` | Errores de vLLM salen como 500 internal_error: un 400 de contexto excedido no es 400 |
| medium | api-edge | `vigia/backends/vllm_proxy.py` | finish_reason de vLLM se ignora: se infiere por conteo y sale invertido |
| medium | api-edge | `vigia/api.py` | El proxy vLLM serializa todas las generaciones del proceso (cupo 1) |
| medium | api-edge | `vigia/api.py` | Parámetros OpenAI nullable con null devuelven 422 (temperature, top_p, n, stream) |
| medium | ml-correctness | `vigia_ee/calibration.py` | El job de calibración de la plataforma agrupa el split por id de item y no por pregunta: la misma pregunta cae en train y en test |
| medium | ml-correctness | `vigia_ee/calibration.py` | El job de calibración de la plataforma etiqueta los items 'answers' solo con EM/F1, sin el arreglo de gold_correct: '15' cuenta como correcta para el gold '1,5' |
| medium | ml-correctness | `vigia/evidence.py` | gold_correct sin tipo sigue aceptando por F1 un año o un número distinto cuando el gold es una fecha o un número con unidad (arreglo previo incompleto) |
| medium | ops-resilience | `vigia_ee/runtime.py` | /readyz cuenta la cola de webhooks: el webhook colgado de un solo tenant saca al nodo de servicio |
| medium | ops-resilience | `vigia_ee/runtime.py` | VIGIA_MODEL inválido (ruta o repo inexistente) aborta el arranque con traceback crudo y sin nombrar la variable |
| low | e2e | `vigia_ee/panel.py` | 'Salir' con la sesión vencida muestra 'Sesión vencida' (401) en /logout en lugar de llevar al login y borrar la cookie |
| low | a11y-visual | `vigia_ee/static/app.css` | El formulario del playground se estira y cambia sus espacios cuando crece la tarjeta de respuesta |
| low | a11y-visual | `vigia_ee/static/app.css` | Tabla de usuarios en Configuración: el texto de las celdas queda arriba y los controles más abajo, con alturas distintas entre select y botones |
| low | a11y-visual | `vigia_ee/templates/overview.html` | Estado vacío del resumen: los botones 'Crear una API key' y 'Probar en el playground' quedan pegados sin espacio |
| low | a11y-visual | `vigia/charts.py` | Los números de los gráficos y de sus tablas no usan separador de miles, a diferencia del resto del panel |
| low | a11y-visual | `vigia/charts.py` | La tendencia de marcado con un solo día dibuja su único punto encima del eje Y |
| low | a11y-visual | `vigia_ee/templates/calibration_job.html` | El título de la página del job de calibración siempre dice 'Calibración en curso', aunque el job terminó o falló |
| low | a11y-visual | `vigia_ee/static/app.css` | El checkbox de confirmación de borrado en el detalle se achica a 13 px cuando su label ocupa dos líneas |
| low | api-edge | `vigia/api.py` | JSON profundamente anidado en /v1/* devuelve 500 en lugar de 400 |
| low | api-edge | `vigia/api.py` | fail_open con falla de score_response deja tokens con p_error y flag=true aunque tau=null y degraded=true |
| low | api-edge | `vigia_ee/runtime.py` | El playground ignora policy.rag_fallback_action y siempre usa human_review |
| low | api-edge | `vigia/serve.py` | vigia serve acepta una política con rag_retry pero nunca re-consulta |
| low | api-edge | `vigia_ee/runtime.py` | GET /v1/vigia/traces/{id} de la plataforma pierde rag_status y no tiene la forma del núcleo |
| low | api-edge | `vigia/api.py` | Parámetros OpenAI no soportados se ignoran en silencio (tools, response_format, echo, best_of, logit_bias) |
| low | ml-correctness | `vigia/evaluate.py` | E-AURC del bootstrap no equivale al E-AURC del dataset remuestreado: la referencia óptima trata una fila con peso w como un solo bloque |
| low | ops-resilience | `vigia/api.py` | usage.prompt_tokens siempre es 0 en la plataforma con el backend HF |
| low | ops-resilience | `vigia_ee/app.py` | Ctrl+C espera sin tope a todas las generaciones en cola y hasta arranca generaciones nuevas después de la señal |
| low | ops-resilience | `vigia_ee/app.py` | Los logs de uvicorn y asyncio salen en texto plano, fuera del formato JSON, con stack traces |
| low | ops-resilience | `vigia_ee/runtime.py` | VIGIA_BASE_DETECTOR inexistente o ilegible no frena el arranque y /startupz dice detectors:true |

## Ronda 3

Defectos nuevos: 29.

| Sev. | Lente | Archivo | Defecto |
|---|---|---|---|
| high | isolation-rbac | `vigia_ee/auth.py` | Un admin deshabilitado o degradado completa un POST que empezó antes del cambio y crea una API key admin que sigue funcionando |
| medium | e2e | `vigia/calibrate.py` | El split de calibración por pregunta no se estratifica: un dataset balanceado de 24 respuestas falla con la semilla por defecto |
| medium | e2e | `vigia_ee/static/app.js` | El deslizador del umbral de vista recarga la página en cada flecha del teclado y el foco cae al body |
| medium | a11y-visual | `vigia_ee/static/app.js` | El deslizador del umbral de vista envía el formulario con cada flecha del teclado; recarga la página y el foco cae a body |
| medium | a11y-visual | `vigia_ee/templates/account.html` | Las ayudas de formulario no están asociadas a sus campos con aria-describedby (contraseña de 15 caracteres, umbrales, URL RAG, retención, formato JSONL) |
| medium | api-edge | `vigia/serve.py` | vigia serve sin API key acepta POST cross-site text/plain y cualquier Host: una página web dispara generaciones y, con DNS rebinding, lee respuestas y trazas |
| medium | api-edge | `vigia_ee/runtime.py` | GET /v1/vigia/traces/{id} de la plataforma sigue devolviendo rag_status null: el arreglo previo depende de una columna que no existe |
| medium | api-edge | `vigia/api.py` | Un request sin streaming que el cliente abandona se genera completo igual, incluso si estaba en la cola del GenerationGate |
| medium | api-edge | `vigia_ee/runtime.py` | Modo vLLM de la plataforma: sin forma de configurar n_experts, las features U se normalizan con el máximo ID de cada respuesta y el p_error de un token depende de los tokens futuros |
| medium | ml-correctness | `vigia/detector.py` | Detector.tau interpola τ linealmente entre alphas guardados y rompe la garantía FNR <= alpha que prometen el docstring y el panel |
| medium | ml-correctness | `vigia/evidence.py` | gold_correct acepta por F1 >= 0,5 respuestas con otro año, mes o número cuando el gold no es un número puro ni una fecha completa (el arreglo previo quedó incompleto) |
| medium | ops-resilience | `vigia/api.py` | Las generaciones en cola cuando llega CTRL+C/SIGTERM igual arrancan y terminan cortadas: 500 en texto plano sin request_id o stream truncado |
| medium | ops-resilience | `vigia_ee/metrics.py` | /metrics se cuelga todo el tiempo que la base está bloqueada: el scrape de Prometheus vence justo durante el incidente |
| low | a11y-visual | `vigia_ee/panel.py` | Resumen y Rendimiento muestran 'Overhead p95' distinto para los mismos requests (2,4 % contra 0,0 %) |
| low | a11y-visual | `vigia_ee/panel_admin.py` | La leyenda y la tabla del ROC del detector muestran ids internos 'tier:F0', 'tier:todos' |
| low | a11y-visual | `vigia/charts.py` | El histograma de la curva de confiabilidad es un bloque gris uniforme: con bins de igual masa todas las barras miden lo mismo |
| low | a11y-visual | `vigia/charts.py` | Dos series del gráfico ROC en tema claro tienen contraste no textual menor que 3:1 (#56B4E9 y #E69F00 sobre blanco) |
| low | a11y-visual | `vigia_ee/panel.py` | La página de error 404 de un GET ofrece 'Reintentar' como acción principal |
| low | a11y-visual | `vigia_ee/static/app.css` | El botón Aplicar del umbral de vista mide 28 px junto a un input de 40 px |
| low | a11y-visual | `vigia_ee/static/app.css` | La barra lateral y el shell terminan 6 px antes del borde inferior en páginas cortas |
| low | a11y-visual | `vigia_ee/static/app.css` | El bloque de código del estado vacío del Resumen indenta solo la primera línea |
| low | a11y-visual | `vigia_ee/static/app.js` | El resultado del playground con JS muestra la acción como texto plano y sin JS como badge |
| low | a11y-visual | `vigia_ee/templates/model.html` | /model tiene dos regiones con el mismo nombre accesible (axe landmark-unique) |
| low | isolation-rbac | `vigia_ee/audit.py` | El seq global de la auditoría muestra a cada tenant cuántos eventos registraron las otras organizaciones |
| low | api-edge | `vigia/api.py` | VigiaResult.status nunca se setea: siempre 'ok' aunque el stream se cortó, el detector falló (fail_open) o la generación quedó vacía |
| low | api-edge | `vigia/api.py` | vLLM caído, 5xx o timeout sale como 500 'Error interno del servidor' sin Retry-After, el log no dice la causa y /readyz sigue ok |
| low | api-edge | `vigia/api.py` | Parámetros OpenAI que cambian la respuesta siguen ignorándose en silencio: modalities ['audio'], audio y reasoning_effort |
| low | ml-correctness | `vigia/evidence.py` | Item de fecha con gold de solo el año marca como incorrecta una respuesta correcta que termina en el día ('En 1945, el 8 de mayo') o que está en formato ISO |
| low | ops-resilience | `vigia_ee/runtime.py` | Los errores de rango de configuración nombran el campo interno y no la variable de entorno (shutdown_timeout_s en lugar de VIGIA_SHUTDOWN_TIMEOUT) |

## No arreglados (con motivo)

- R1 `panel`: Si el cliente se desconecta durante un streaming... (lado API, vigia/api.py _live_events): queda fuera de alcance porque vigia/api.py pertenece al grupo servicio. Otro reparador ya lo está arreglando en paralelo: el archivo ya tiene una _ClosingStreamingResponse en la línea 297. No toqué ese archivo.
- R1 `panel`: Si el cliente se desconecta durante un streaming... (sugerencia de agregar a /readyz un chequeo del lock del backend tomado por más de N s): queda fuera de alcance porque vigia_ee/runtime.py y vigia_ee/app.py pertenecen al grupo app. Con el cierre explícito, el lock ya no queda tomado tras una desconexión del panel.
- R1 `panel`: Borrar la organización... (limpieza completa del estado en memoria): invalidate_detector arregla la cache de detectores y /metrics. _policies, _policy_alerts, _retrievers y _base_checked conservan la org borrada. Ese estado ya no es alcanzable, pero ocupa memoria. Hace falta Runtime.forget_org en runtime.py, que es del grupo app. Ver contract_requests.
- R1 `core-ml`: Ninguno de los 6 defectos queda sin arreglar. Primero reproduje los 6: repro_defects.py D1, D2 y D4, m8.py y el código de heatmap_svg. Los 10 tests nuevos de regresión fallan contra HEAD y pasan con el arreglo. Lo comprobé en una copia del paquete con los 4 módulos tomados de HEAD.
- R1 `core-ml`: Fuera de alcance, sin arreglar: normalize_es sigue juntando dígitos separados por '.' o ','. Es la regla del script de MLQA, y los items tipados ya no dependen de EM. Un item de texto con gold no numérico, por ejemplo 'versión 1.5', todavía puede coincidir por EM con '15'.
- R1 `core-ml`: Fuera de alcance, previo a este cambio: en score_histogram_svg el rótulo '0 %' del eje Y se pisa unos píxeles con el rótulo '0,0' del eje X en el origen. Las marcas del eje X del histograma redondean 0,25 como '0,2'. Ninguno de los dos figura en los defectos asignados.
- R1 `core-ml`: Suite rápida: fallan 2 tests de otro grupo, test_app.py::test_cli_storage_errors_exit_2_with_clear_message[init-missing_parent] y [init-not_sqlite]. El error es 'OSError: pytest: reading from stdin while output is captured', que salta cuando getpass pide la password de init. Vienen de vigia_ee/app.py y tests/test_app.py, que otro reparador tiene modificados en paralelo. No tocan ningún archivo de core-ml. El resto de la suite pasa, y también pasan tests/test_verify_ml.py (36) y tests/test_cli.py.
- R1 `ee-app`: Parte del defecto de JSON anidado en vigia/datasets.py parse_jsonl (misma causa raíz: solo captura ValueError): ese archivo pertenece al grupo 'evidencia' y otro reparador lo está editando. Solo se alcanza por CLI u operador, no por HTTP. Va en contract_requests.
- R1 `ee-app`: Parte del defecto de almacenamiento sobre Database.__init__, que no cierra la conexión si falla un PRAGMA o la migración: vigia_ee/db.py pertenece al grupo 'plataforma base'. El síntoma visible (traceback) ya quedó resuelto desde open_database. Lo único pendiente es que la conexión se libera por GC y no de forma explícita. Va en contract_requests.
- R1 `ee-base`: La suite rápida completa termina con 2 fallos ajenos a este cambio: tests/test_app.py::test_cli_storage_errors_exit_2_with_clear_message[init-missing_parent] e [init-not_sqlite]. Ese test es nuevo y lo agregó otro reparador en paralelo sobre el CLI de app.py. El CLI init lee la contraseña por stdin y no toca security ni auth. El resto de la suite pasa. ruff marca un I001 en tests/test_webhooks.py, que también es de otro grupo. Mis archivos pasan ruff check y ruff format.
- R1 `ee-ops`: Parte del defecto de saturación por webhooks queda pendiente: /readyz todavía cuenta la cola de webhooks. El motivo es que necesita un contrato, porque Runtime.readiness() vive en vigia_ee/runtime.py, que no es de este grupo. Un destino roto ya no satura /readyz, porque se desactiva tras N intentos y su cola se vacía. Un destino lento pero que responde 2xx todavía puede llevar la cola total por encima de max_queue_depth. Ver contract_requests.
- R1 `ee-ops`: Riesgo que queda, fuera de alcance: dentro del carril de webhooks no hay reparto justo entre tenants. Un tenant con un destino lento que responde 2xx (hasta 15 s por intento) todavía atrasa los webhooks de otros tenants. Ya no atrasa la purga ni la calibración. Arreglarlo exige un claim por turnos entre orgs o un pool por tenant.
- R1 `ee-ops`: Observación del entorno, no es un defecto del código: durante la corrida de la suite completa en paralelo, /readyz dio un 503 transitorio con worker:false. La causa probable es un bloqueo del lock de SQLite o del disco por carga de la máquina (Granite y otras suites corriendo al mismo tiempo). Con la máquina libre, una sonda de 25 s x3 no reprodujo ningún falso negativo de is_alive().
- R1 `core-signals`: Ninguno de los defectos asignados queda sin arreglo. Aparte, 'ruff check .' reporta un error I001 (imports desordenados) en tests/test_webhooks.py:3. Ese archivo es de otro grupo y lo modifica otro reparador en paralelo, así que no lo toqué. Mis archivos (vigia/backends/hf.py, tests/test_backend_hf.py) pasan ruff.
- R2 `ee-app`: GET /v1/vigia/traces/{id} pierde rag_status: falta la parte de persistencia. Guardar rag_status y status exige columnas en vigia_ee/schema.sql, una migración y un parámetro nuevo en TenantScopedRepository.insert_trace (vigia_ee/db.py). Esos archivos son de otro grupo y el diff exacto va en contract_requests. La forma ya está alineada y trace_view lee las columnas apenas existan.
- R2 `ee-app`: Ctrl+C: el corte inmediato de los requests de la API que ya esperan el cupo de GenerationGate no está. Al recibir la señal, esos requests ya pasaron la dependencia de contexto. Hoy los cancela el tope de timeout_graceful_shutdown. Rechazarlos al instante exige una API pública en vigia/api.py (GenerationGate.close), que es de otro grupo y va como pedido opcional en contract_requests. Una generación no-stream que ya corre en un hilo termina igual: la limita max_tokens_limit.
- R2 `core-service`: La parte vLLM de 'usage.prompt_tokens siempre es 0' necesita que VllmProxyBackend.stream devuelva un objeto con prompt_tokens. vigia/backends/vllm_proxy.py es de otro grupo. Su reparador ya lo está haciendo en paralelo (clase de stream con prompt_tokens leído de usage), y api.py ya lo consume. No pude probarlo contra el vLLM falso desde mi grupo.
- R2 `core-service`: La parte configurable de 'El proxy vLLM serializa todas las generaciones' necesita VIGIA_VLLM_MAX_CONCURRENCY. Eso vive en VllmProxyBackend.max_concurrency y en Settings de vigia_ee/runtime.py, que son de otros grupos. Va en contract_requests. Sin ese cambio, el default de 16 en api.py ya resuelve el defecto.
- R2 `core-signals`: Errores de vLLM salen como 500 internal_error (parte 5xx o caída): sigue en 500 internal_error. Es un contrato pendiente. Ningún error de contracts.py se mapea hoy a 502/503 en vigia/api.py, y esos archivos no son de este grupo. vllm_proxy ya separa el caso: 5xx, 401, 404 y errores de transporte quedan como VllmProxyError y no como InvalidRequestError. Ver contract_requests.
- R2 `core-signals`: Con gpt-oss (harmony), marcos de offsets distintos (parte respuesta CON marcado harmony en el dataset): vigia_ee/calibration._collect y vigia/evidence.computable_labels siguen pasando los spans en el marco del string con marcado. Son archivos de otros grupos. El helper vigia.harmony.final_view ya existe y está probado: con él, el repro marca 'Quito' en los 3 casos (plain, marked y marked con analysis). Falta adoptarlo en esos dos archivos; ver contract_requests.
- R3 `core-ml`: Parte del primer defecto ('Ese dataset calibra con la semilla que trae el formulario' para el JSONL original con 'answers' que nunca coinciden): no es un defecto del split. Ese dataset no tiene señal (mismo texto generado para toda pregunta), y el rechazo por 'pendiente de Platt <= 0' es el comportamiento correcto; se arreglo la causa raiz (estratificacion + mensaje) y se cambio el dataset e2e por uno con señal.
- R3 `core-ml`: Fallo transitorio tests/test_db.py::test_review_flow_with_two_reviewers en la ultima corrida completa: fuera de alcance (vigia_ee/db.py y tests/test_db.py modificados por otro reparador en paralelo); al correrlo solo despues no se listo ningun fallo.
- R3 `ee-base`: Fallos de la suite rápida fuera de ee-base: test_evaluate::test_at_m09_report_has_metrics_cis_and_baselines (0.8486 > 0.85) y los de test_evidence (Apolo 12 y el end-to-end). Esos tests son del núcleo vigia/, que otros reparadores tienen en edición. No los causan mis cambios.
- R3 `ee-base`: Fallos inestables o en edición por otros grupos: test_api::test_abandoned_request_in_queue_is_not_generated, test_panel::test_panel_errors_render_as_html_with_request_id, test_panel_r3::test_model_page_regions_have_unique_names y test_panel_admin::test_policies_inline_errors. Los cuatro fallaron en la corrida completa y pasaron al correrlos de nuevo por separado. Archivos panel*/api que tocan otros grupos.
- R3 `ee-base`: Fuera de alcance: schema.sql cita docs/spec/seguridad.md y ese archivo no existe. No lo arreglé porque no estaba en la lista de defectos.
- R3 `ee-base`: Aviso de entorno: al principio usé por error la carpeta compartida scratchpad/live. Esa carpeta es de otro agente y está sin actividad desde las 13:37. Ahí sobrescribí env.sh, h.py, seed.py, slowpost.py, slowapi.py, seqleak.py, peppers.txt y ownerpw.txt. También creé las orgs Alfa SA y Beta SRL con el usuario admin-b/reviewer-b/viewer-b en su data/vigia.db, que además migró a la versión 2. Después usé scratchpad/eebase_r3_live con el puerto 8137. Ya maté mi servidor (PID 44406) y el puerto quedó libre.
- R3 `core-service`: Parte de /readyz del defecto 'vLLM caido, 5xx o timeout...': /readyz de la plataforma vive en vigia_ee/app.py y vigia_ee/runtime.py, y el chequeo del upstream vive en vigia/backends/vllm_proxy.py. Ninguno es del grupo core-service, asi que necesita contrato. vigia serve solo usa el backend HF y no tiene caso vLLM.
- R3 `core-service`: Persistencia de status en la plataforma (defecto VigiaResult.status): la tabla traces no tiene columna status y runtime.trace_view lee trace.get('status') or 'ok'. Exige cambiar schema.sql, db.py y runtime.py, que son contrato u otros grupos. La respuesta de la API y la traza de vigia serve ya llevan el status correcto.
- R3 `core-service`: Cableado de closing en la plataforma (defecto CTRL+C/SIGTERM): falta una linea en vigia_ee/runtime.py (otro grupo). Sin ese cambio, el 503 en cola no se activa en 'vigia_ee.app serve'. Por eso no se repitio en vivo el inflight2.py contra la plataforma.
- R3 `ee-app`: GET /v1/vigia/traces/{id} de la plataforma sigue devolviendo rag_status null: el arreglo previo depende de una columna que no existe. NECESITA CONTRATO. Leí el código y el defecto se confirma: schema.sql no tiene columnas rag_status ni status en traces, y TenantScopedRepository.insert_trace (vigia_ee/db.py) no las acepta. Runtime.trace_view ya lee trace.get('rag_status') y trace.get('status') de SELECT *, así que funciona en cuanto existan las columnas. El arreglo exige una migración en db.py (MIGRATIONS es de 'plataforma base', no de este grupo) y un parámetro nuevo en insert_trace. Si runtime pasa hoy rag_status=... a insert_trace, se rompe con TypeError. Por eso el cambio de runtime va en contract_requests y se aplica junto con el de db.py.
- R3 `ee-ops`: La lentitud de /readyz durante el lock (34,8 s) aparece en el repro pero no forma parte de este defecto. La lectura de readyz vive en vigia_ee/app.py y runtime.py, que pertenecen a otro grupo. No toqué esos archivos.

## Pedidos de contrato de los reparadores

- R1 `panel`: vigia_ee/runtime.py (grupo app): agregar Runtime.forget_org para limpiar todo el estado en memoria de una org borrada. Diff: después de invalidate_detector, agregar
+    def forget_org(self, org_id: str) -> None:
+        """Saca de memoria todo el estado de una org borrada (detector, política, retriever)."""
+        with self._lock:
+            self._detectors.pop(org_id, None)
+            self._detector_alerts.pop(org_id, None)
+            self._policies.pop(org_id, None)
+            self._policy_alerts.pop(org_id, None)
+            self._retrievers.pop(org_id, None)
+            self._base_checked.discard(org_id)
+            active = [d.id for _, d in self._detectors.values() if d is not None]
+        self.metrics.set_calibrations(active)
Cuando se integre, vigia_ee/panel_admin.py data_delete cambia `runtime.invalidate_detector(p.org_id)` por `runtime.forget_org(p.org_id)`. El test test_data_delete_forgets_the_tenant_detector_in_runtime_and_metrics sigue válido.
- R1 `ee-app`: vigia/datasets.py (grupo evidencia), en parse_jsonl, cerca de la línea 229: capturar también RecursionError. Diff: después del bloque `except ValueError as exc: ...` agregar `except RecursionError:\n            raise DatasetError(f"Línea {n}: JSON inválido (anidamiento excesivo).") from None`. Adaptar el nombre de la variable de línea y la clase de error a las que usa ese archivo. Repro: una línea con '['*6000 + ']'*6000 levanta RecursionError en lugar del error de validación.
- R1 `ee-app`: vigia_ee/db.py (grupo plataforma base), en Database.__init__: cerrar la conexión si falla un PRAGMA o la migración. Diff: reemplazar el cuerpo desde `conn = sqlite3.connect(...)` por `conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)\n        try:\n            conn.row_factory = sqlite3.Row\n            conn.execute("PRAGMA foreign_keys=ON")\n            conn.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")\n            if self.path != ":memory:":\n                conn.execute("PRAGMA journal_mode=WAL")\n                conn.execute("PRAGMA synchronous=NORMAL")\n            self._conn = conn\n            self.migrate()\n        except BaseException:\n            conn.close()\n            raise`. Motivo: con un archivo que no es SQLite, el PRAGMA falla y la conexión queda abierta hasta el GC. En Windows eso puede retener el archivo.
- R1 `core-service`: vigia_ee/runtime.py (grupo app), arreglo de 'policy.rag_fallback_action se ignora' en el playground: en el import de vigia.policy reemplazar `RAG_FALLBACK_ACTION,` por `rag_fallback,`. En Runtime._playground (~línea 890) reemplazar `action, rag_status = RAG_FALLBACK_ACTION, "unavailable"` por `action, rag_status = rag_fallback(policy), "unavailable"`.
- R1 `core-service`: vigia_ee/panel.py (grupo panel), deadlock con el lock del HF: el playground usa runtime.backend fuera del cupo de la API y puede volver a bloquear hilos del threadpool esperando el lock. Agregar `from vigia.api import GenerationBusy, generation_gate`. En playground_submit, antes de `out = await run_in_threadpool(run_playground, runtime, p, inp)`, poner `gate = generation_gate(runtime.backend)` y luego `try: await gate.acquire()` / `except GenerationBusy: return render(... status=503 ...)`. Envolver la llamada en `try: ... finally: gate.release()`. En la ruta de streaming del playground (~línea 1130), adquirir el mismo gate antes de crear el stream y llamar `gate.release()` en el callback de cierre de su ClosingStreamingResponse.
- R1 `core-service`: vigia/backends/vllm_proxy.py (grupo backend vLLM): ahora el cupo de generación por backend vale 1 salvo que el backend declare otro. vLLM admite concurrencia, así que conviene agregar a VllmProxyBackend el atributo de clase `max_concurrency: int = 16` (o hacerlo configurable). Si no, el modo reducido serializa los requests.
- R1 `core-service`: vigia/backends/hf.py (grupo backend HF): en HFBackend._prompt_ids, lanzar `InvalidRequestError(f"el prompt ({len(ids)} tokens) excede el contexto del modelo ({self._max_positions})", code="context_length_exceeded")` en lugar de ValueError, con `from vigia.contracts import InvalidRequestError`. Hacer lo mismo con 'el prompt quedó vacío' usando code='invalid_prompt'. La API ya mapea InvalidRequestError a 400 con su code; hoy responde 400 invalid_prompt genérico.
- R1 `core-service`: docs/spec/api-modulos.md (orquestador): documentar lo nuevo de vigia.api: GenerationGate, generation_gate(backend, *, max_waiting=64, timeout_s=120.0), GenerationBusy, los kwargs de create_api_router max_queued_generations=64 y queue_timeout_s=120.0, y el duck typing backend.max_concurrency (default 1). Documentar en vigia.policy: rag_fallback(policy) y RAG_FALLBACK_ACTIONS. Documentar en vigia.contracts: InvalidRequestError(VigiaError, ValueError) con atributo code. Documentar en la API: 503 server_overloaded con Retry-After, 400 invalid_prompt / invalid_value / context_length_exceeded, 422 unsupported_value para top_logprobs>0 y logprobs en completions, y 422 vigia.alpha cuando alpha es menor que el mínimo calibrado.
- R1 `ee-base`: docs/spec/api-modulos.md, línea 100 (doc de install_security): el formato documentado quedó desactualizado. Diff: '- ... + handlers JSON {error, message, request_id} para HTTPException y 422 sin eco de input' -> '+ ... + handlers JSON sin eco de input: bajo /v1 el sobre de OpenAI {error:{message,type,code,param,request_id}} (error_payload/is_openai_path); en el panel {error, message, request_id} (+fields en 422). CodedHTTPException(status, detail, code) fija error.code'. Además, en la línea 102 hay que agregar 'error_payload(path, status, code, message, request_id, *, param=None) ; is_openai_path(path) ; CodedHTTPException' a la lista pública de vigia_ee.security.
- R1 `ee-ops`: vigia_ee/runtime.py (readiness): /readyz debe medir solo el carril por defecto, para que la cola de webhooks de un tenant no saque el nodo de rotación. La métrica vigia_jobs_queue_depth sigue con jobs.depth() sin lane, que cuenta todo. Diff:
--- a/vigia_ee/runtime.py
+++ b/vigia_ee/runtime.py
@@
-from vigia_ee.jobs import JobQueue
+from vigia_ee.jobs import DEFAULT_LANE, JobQueue
@@ def readiness(self) -> dict[str, bool]:
-            checks["queue"] = self.jobs.depth() < self.settings.max_queue_depth
+            checks["queue"] = self.jobs.depth(lane=DEFAULT_LANE) < self.settings.max_queue_depth
- R1 `ee-ops`: docs/spec/api-modulos.md: hay que sincronizar la API nueva de vigia_ee.jobs y vigia_ee.webhooks. (a) Sección jobs: agregar DEFAULT_LANE='default' y MAX_RECOVER_INTERVAL_S=60. Cambian estas firmas: .register(kind, handler, *, retry_schedule=None, lane='default'), .registered_kinds(lane=None), .lanes(), .depth(*, include_scheduled=False, lane=None), .run_once(lane=None). Agregar el atributo .recover_interval. .start() arranca un hilo por carril ('vigia-jobs' y 'vigia-jobs-<lane>') y recupera huérfanos al arrancar y cada recover_interval; nunca reencola un job propio en curso. .is_alive() exige que todos los hilos vivan y laten. (b) Sección webhooks: register_delivery registra 'webhook' con lane='webhook'. La desactivación ocurre tras N entregas seguidas cuyo último intento falló, contadas en cada intento. La entrega que desactiva el webhook termina failed. (c) arquitectura.md línea 43: 'worker de jobs (hilo)' pasa a ser un hilo por carril (default: calibración y purga; webhook: webhooks).
- R2 `ee-app`: Persistir rag_status y status de la traza (para GET /v1/vigia/traces/{id}). (1) vigia_ee/schema.sql, tabla traces: agregar `  rag_status TEXT,                     -- None | retried | error | unavailable` y `  status TEXT NOT NULL DEFAULT 'ok',    -- VigiaResult.status` antes de usage_json. (2) vigia_ee/db.py: agregar a MIGRATIONS la versión 2 con `ALTER TABLE traces ADD COLUMN rag_status TEXT; ALTER TABLE traces ADD COLUMN status TEXT NOT NULL DEFAULT 'ok';` para las bases existentes. (3) vigia_ee/db.py: TenantScopedRepository.insert_trace(..., rag_status: str | None = None, status: str = 'ok') guarda las dos columnas. (4) Después, en vigia_ee/runtime.py Runtime.store_trace (es mi grupo, lo aplica el orquestador o la ronda siguiente): pasar `rag_status=record.rag_status or res.rag_status, status=res.status` a repo.insert_trace. Runtime.trace_view ya lee trace.get('rag_status') y trace.get('status').
- R2 `ee-app`: Opcional, para el cierre ordenado: vigia/api.py GenerationGate.close() pública. Marca el gate como cerrado, rechaza los acquire nuevos con GenerationBusy y despierta a los waiters con GenerationBusy (fut.set_exception vía call_soon_threadsafe). Así Runtime.begin_shutdown() llama generation_gate(self.backend).close() y los requests de la API en espera responden 503 al instante, sin esperar el tope de timeout_graceful_shutdown.
- R2 `ee-app`: Docs (no son contrato, pero no son de mi grupo): en docs/spec/api-modulos.md, sección vigia_ee.runtime, agregar a los campos extra de Settings `shutdown_timeout_s=30 (VIGIA_SHUTDOWN_TIMEOUT, tope del cierre ordenado)`. Agregar a los métodos extra de Runtime `shutting_down` y `begin_shutdown()`. Documentar que trace_view(org_id, trace_id) tiene la misma forma que vigia.serve.trace_view. En vigia_ee.calibration: CalibrationBusy(DatasetError), MAX_PENDING_JOBS=1, campo opcional 'group' del JSONL y n_questions en report['calibration'].
- R2 `ee-ops`: docs/spec/api-modulos.md (sección vigia_ee.jobs, no es de mi grupo): agregar la API nueva, que es aditiva. Diff: después de la línea '.start() (recupera huerfanos, hilo daemon 'vigia-jobs', idempotente) ; ...' agregar '  .configure_lane(lane, *, workers=1, max_running_per_org=None) (hilos del carril y tope de jobs en curso por tenant en este proceso; los jobs de sistema cuentan como un tenant)'. En la sección vigia_ee.webhooks, agregar 'LANE_WORKERS=4; MAX_RUNNING_PER_ORG=1' a la línea de constantes. Agregar a la línea 'entrega:' el texto 'plazo TOTAL de timeout s por entrega (conexión+TLS+envío+headers); vencerlo es fallo'.
- R2 `ee-ops`: Nota sin cambio de contrato: _delivery_client envuelve transport._pool._network_backend, un atributo privado de httpx 0.28/httpcore 1.0. httpx no expone el backend de red. Si una actualización de httpx/httpcore cambia ese atributo, make_delivery_handler lanza RuntimeError al arrancar y nunca entrega sin plazo. Conviene fijar 'httpx>=0.27,<0.29' en pyproject.toml, que no es de mi grupo.
- R2 `core-service`: vigia/backends/vllm_proxy.py (grupo backend vLLM): hacer configurable el cupo del proxy. Diff: en VllmProxyBackend.__init__ agregar el kwarg `max_concurrency: int = 16`; después de validar n_experts agregar `if isinstance(max_concurrency, bool) or not isinstance(max_concurrency, int) or max_concurrency < 1: raise ValueError("max_concurrency debe ser un entero >= 1")` y `self.max_concurrency = max_concurrency`. vigia.api.generation_gate ya lee backend.max_concurrency por duck typing.
- R2 `core-service`: vigia_ee/runtime.py (grupo app): agregar a Settings el campo `vllm_max_concurrency: int = 16`. En Settings.from_env agregar `vllm_max_concurrency=_env_int(env, "VIGIA_VLLM_MAX_CONCURRENCY", 16)` y validar `>= 1` en __post_init__ con SettingsError. En la línea ~231 cambiar `VllmProxyBackend(settings.vllm_url, settings.vllm_model, api_key=settings.vllm_api_key)` por `VllmProxyBackend(settings.vllm_url, settings.vllm_model, api_key=settings.vllm_api_key, max_concurrency=settings.vllm_max_concurrency)`. Depende del pedido anterior.
- R2 `core-service`: docs/spec/api-modulos.md (orquestador). En vigia.api, reemplazar la línea 'opcional (duck typing): si el backend tiene count_prompt_tokens(messages) -> int, se usa para usage.prompt_tokens; si no, 0.' por 'usage.prompt_tokens = stream.prompt_tokens (TokenStream) si es > 0; si no, backend.count_prompt_tokens(messages) (duck typing); si no, 0. generation_gate: backend.max_concurrency, o REMOTE_MAX_CONCURRENCY=16 si backend.mode == "reduced", o 1. 422 unsupported_value para tools/tool_choice/functions/function_call/response_format/logit_bias (chat) y echo/best_of/suffix/logit_bias (completions) con valor no inocuo; temperature/top_p/n/stream/logprobs null = default.' En vigia.serve, agregar a create_app los parámetros `rag_allowed_hosts: Collection[str] = (), retriever: Retriever | None = None` (valida la política y arma HttpRetriever si hay rag_url). En la CLI serve, agregar `--rag-allowed-hosts`.
- R2 `core-signals`: contracts.py + api.py (502 para vLLM caído o 5xx). En vigia/contracts.py, después de InvalidRequestError:
+class BackendUnavailableError(VigiaError):
+    """El servidor de generación (vLLM) no respondió o falló (5xx, timeout, auth). La API responde 502."""
En vigia/api.py: importar BackendUnavailableError desde vigia.contracts. En _open_stream:
     except (InvalidRequestError, ValueError) as exc:
         raise _invalid_prompt(exc) from exc
+    except BackendUnavailableError as exc:
+        logger.warning("backend de generación no disponible: %s", exc)
+        raise APIError(502, "El servidor de generación no está disponible.", "server_error", "backend_unavailable") from exc
Cuando eso exista, el grupo core-signals hace en vllm_proxy.py: `class VllmProxyError(BackendUnavailableError)`. VllmProxyError ya se usa solo para fallas de vLLM y respuestas ilegibles; los rechazos del cliente ya son InvalidRequestError.
- R2 `core-signals`: vigia_ee/calibration.py (_collect): agregar `from vigia.harmony import final_view` y reemplazar la rama de spans:
                     else:
-                        token_labels = spans_to_token_labels(tokens, item.spans)
+                        final_text, spans = final_view(item.response, item.spans)
+                        if "".join(t.text for t in tokens if _visible(t)) != final_text:
+                            skipped["texto_final_distinto"] += 1
+                            continue
+                        token_labels = spans_to_token_labels(tokens, spans)
Con esto el item 'marked' del repro marca 'Quito' y no el '.' final. Un item cuyo texto final difiere de la respuesta se descarta con un motivo visible.
- R2 `core-signals`: vigia/evidence.py (computable_labels): agregar `from vigia.harmony import final_view` y cambiar:
-    if _final_text(meta) != item["response"]:
+    response, spans = final_view(item["response"], [tuple(s) for s in item["hallucinated_spans"]])
+    if _final_text(meta) != response:
         # Los offsets del dataset se refieren a la respuesta dada; si el texto final difiere, no valen.
         return [IGNORE_LABEL] * len(meta), {"correct": None, "rejected": "texto_final_distinto"}
-    spans = [tuple(s) for s in item["hallucinated_spans"]]
     return spans_to_token_labels(toks, spans), {"correct": not spans}
final_view devuelve la respuesta y los spans sin cambios cuando no hay marcado harmony, así que los items no-harmony no cambian.
- R3 `core-ml`: docs/spec/api-modulos.md (vigia.detector), linea 137: '-  .tau(alpha: float) -> float      # interpola; alpha < min guardado -> ValueError; alpha > max -> tau del max' => '+  .tau(alpha: float) -> float      # escalon: tau del mayor alpha guardado <= alpha (no interpola: conserva FNR <= alpha); alpha < min guardado -> ValueError; alpha > max -> tau del max'
- R3 `core-ml`: docs/spec/api-modulos.md (vigia.calibrate), linea 147: '-split_groups(groups, *, cal_frac=0.2, test_frac=0.2, seed=0) -> dict[str, list[str]]   # {'train','cal','test'}' => '+split_groups(groups, *, cal_frac=0.2, test_frac=0.2, seed=0, labels=None) -> dict[str, list[str]]   # {'train','cal','test'}; con labels (una por token) estratifica por la clase mayoritaria de cada pregunta; train_detector y evaluate_report la pasan'
- R3 `core-ml`: docs/spec/api-modulos.md (vigia.labels), despues de la linea 'parse_dates_es(text) -> list[date] ; date_match(...)': '+parse_date_parts_es(text) -> list[(año, mes|None, dia|None)]   # menciones completas, mes+año y año suelto de 3-4 digitos, en orden, sin superposicion' (y opcional en vigia.evidence: '  gold_correct(...)  # EM/F1 solo contra alias cuyos numeros aparecen en la prediccion; items date aceptan gold de mes+año o año solo')
- R3 `core-ml`: vigia_ee/templates/policies.html linea 27 (grupo panel): '-  <p class="vg-hint">Un alpha entre dos valores guardados interpola τ.</p>' => '+  <p class="vg-hint">Un alpha entre dos valores guardados usa el τ del alpha guardado inferior: así la FNR sigue garantizada.</p>'
- R3 `core-service`: vigia_ee/runtime.py (grupo app), cierre durante la cola. En Runtime.context_for agregar el callable closing al RequestContext:
--- a/vigia_ee/runtime.py
+++ b/vigia_ee/runtime.py
@@ def context_for(
             retriever=self._retriever_for(org_id, policy),
             source=source,
+            closing=lambda: self.shutting_down,
         )
Test sugerido en tests/test_runtime.py: assert runtime.context_for(p).closing() is False; runtime.begin_shutdown(); assert runtime.context_for(p).closing() is True.
- R3 `core-service`: vigia_ee/schema.sql + vigia_ee/db.py (contrato y plataforma base) + vigia_ee/runtime.py, persistencia de VigiaResult.status:
--- a/vigia_ee/schema.sql (tabla traces, junto a degraded)
+  status TEXT NOT NULL DEFAULT 'ok',
--- a/vigia_ee/db.py
 MIGRATIONS += ((N, "ALTER TABLE traces ADD COLUMN status TEXT NOT NULL DEFAULT 'ok'"),)
 def insert_trace(..., degraded=False, finish_reason=None, usage=None, created_at=None,
+                 status: str = 'ok')  # y agregar la columna al INSERT y a get_trace/export_all
--- a/vigia_ee/runtime.py (store_trace -> repo.insert_trace)
                 degraded=res.degraded,
+                status=res.status,
                 finish_reason=gen.finish_reason,
runtime.trace_view ya lee trace.get('status'), asi que no cambia.
- R3 `core-service`: vigia/backends/vllm_proxy.py (grupo backend vLLM) + vigia_ee/runtime.py, readiness del upstream:
--- a/vigia/backends/vllm_proxy.py (clase VllmProxyBackend)
+    def ping(self) -> bool:
+        """GET {base_url}/v1/models con timeout 2 s; cachea el resultado 5 s. Nunca lanza."""
+        now = time.monotonic()
+        if now - self._ping_at < 5.0:
+            return self._ping_ok
+        try:
+            headers = {'Authorization': f'Bearer {self._api_key}'} if self._api_key else {}
+            ok = self._client.get(f'{self.base_url}/v1/models', headers=headers, timeout=2.0).status_code == 200
+        except httpx.HTTPError:
+            ok = False
+        self._ping_at, self._ping_ok = now, ok
+        return ok
(en __init__: self._ping_at = float('-inf'); self._ping_ok = False)
--- a/vigia_ee/runtime.py (Runtime.readiness, antes del return)
+        ping = getattr(self.backend, 'ping', None)
+        if callable(ping):
+            try:
+                checks['backend'] = bool(ping())
+            except Exception as exc:  # noqa: BLE001
+                log_event(_log, 'readyz_backend_error', level=logging.WARNING, error_type=type(exc).__name__)
+                checks['backend'] = False
Con esto, /readyz devuelve 503 cuando vLLM no responde (app.py ya hace all(checks.values())).
- R3 `core-service`: docs/spec/api-modulos.md (orquestador), solo documentacion. En vigia.api, RequestContext suma closing: Callable[[], bool] | None = None. Hay errores nuevos: 503 shutting_down y 502 upstream_unavailable, los dos con Retry-After, y 499 en el log para un cliente que corto. En vigia.serve se exporta browser_origin_problem(headers). Sin key, serve devuelve 403 ante un Host u Origin no loopback o ante Sec-Fetch-Site cross-site/same-site, y 415 ante un POST que no es application/json.
- R3 `ee-app`: vigia_ee/db.py (dueño: plataforma base), para el defecto 'GET /v1/vigia/traces/{id} ... rag_status null'. (1) Cambiar MIGRATIONS a: `MIGRATIONS: tuple[tuple[int, str], ...] = ((1, SCHEMA_PATH.read_text(encoding="utf-8")), (2, "ALTER TABLE traces ADD COLUMN rag_status TEXT; ALTER TABLE traces ADD COLUMN status TEXT NOT NULL DEFAULT 'ok';"),)`. No hay que agregar las columnas también en schema.sql: la migración 1 corre en una base nueva y la 2 fallaría con 'duplicate column'. (2) Agregar a la firma de insert_trace `rag_status: str | None = None, status: str = "ok",`. Agregar `rag_status, status` a la lista de columnas del INSERT INTO traces, con dos '?' más (23 en total), y agregar `rag_status, status,` a la tupla después de `_json(dict(usage)) if usage is not None else None,`. (3) tests/test_db.py:53 debe pasar de `== 1` a `== 2`. Después yo aplico el cambio de vigia_ee/runtime.py en Runtime.store_trace: agregar `rag_status=res.rag_status, status=res.status,` en la llamada repo.insert_trace(...), y agregar un test en tests/test_runtime.py. El test hace POST con rag_retry contra un retriever falso y comprueba que trace_view(org, id)['rag_status'] == 'retried'.
- R3 `ee-app`: docs/spec/api-modulos.md (spec de contrato), en la sección vigia_ee.runtime 'Campos extra, todos con default'. Agregar shutdown_timeout_s=30 (VIGIA_SHUTDOWN_TIMEOUT). Agregar vllm_n_experts, vllm_top_k y vllm_n_layers (VIGIA_VLLM_N_EXPERTS / VIGIA_VLLM_TOP_K / VIGIA_VLLM_N_LAYERS), con default None y obligatorios con backend=vllm porque fijan la normalización de U. Agregar que todo SettingsError nombra la variable VIGIA_* que el operador corrige.

## Ronda 4 (cierre)

Defectos nuevos: 13.

| Sev. | Lente | Archivo | Defecto |
|---|---|---|---|
| high | ops-resilience | `vigia_ee/auth.py` | Las dependencias async de autenticación consultan SQLite en el event loop: con la base bloqueada, un solo request congela todo el proceso, /livez incluido |
| medium | ml-correctness | `vigia/evaluate.py` | evaluate_report no puede evaluar un detector calibrado con alphas menores a 0,05: los niveles y baselines internos usan DEFAULT_ALPHAS y tau(alpha) lanza ValueError |
| medium | ml-correctness | `vigia/labels.py` | gold_correct lee mal los números con espacio de miles (norma RAE '12 345'): el gold '12 345' vale 12, la respuesta exacta queda incorrecta y '12' queda correcta |
| medium | ops-resilience | `vigia_ee/runtime.py` | /readyz no tiene tope de tiempo: con la base bloqueada no responde hasta que se libera el lock y ocupa un hilo del threadpool por cada sonda |
| low | a11y-visual | `vigia_ee/panel.py` | Gráficos con 'Ver datos' abierto: la región del SVG y la región de la tabla tienen el mismo nombre accesible (axe landmark-unique) |
| low | a11y-visual | `vigia_ee/templates/audit.html` | Auditoría: la columna Objeto mide 142 px y parte cada id de 32 caracteres en 3 líneas |
| low | a11y-visual | `vigia_ee/static/app.css` | Rendimiento: el valor del KPI 'Tokens por segundo (p50)' queda 21 px más abajo que los demás |
| low | a11y-visual | `vigia_ee/templates/performance.html` | Rendimiento: la nota que define el overhead queda pegada a la tarjeta siguiente (0 px) y parece pertenecer al gráfico |
| low | api-edge | `vigia/backends/hf.py` | Una temperature positiva muy chica (1e-40, 1e-300, 5e-324) devuelve 500 internal_error en chat y completions, con y sin streaming |
| low | ml-correctness | `vigia/labels.py` | parse_dates_es no reconoce formas españolas comunes de fecha completa: '1.º de mayo de 1945' (ordinal RAE) y '12 de octubre del año 1492' |
| low | ml-correctness | `vigia/evaluate.py` | ECE con pesos (y por lo tanto la réplica bootstrap de ECE) no equivale al ECE del dataset con filas repetidas: los bins de igual masa ignoran los pesos |
| low | ml-correctness | `vigia/calibrate.py` | La spec P-05 promete calibración isotónica con más de 1000 respuestas y el entrenamiento solo usa Platt |
| low | ops-resilience | `vigia_ee/app.py` | Un cierre ordenado con Ctrl+C (SIGINT) termina 'serve' con código de salida 1, como si fuera un error |

## Ronda 5 (cierre)

Defectos nuevos: 14.

| Sev. | Lente | Archivo | Defecto |
|---|---|---|---|
| high | ops-resilience | `vigia/api.py` | Fuga de hilos nativos y memoria bajo carga: cada hilo nuevo del threadpool de anyio que genera con torch deja vivo un equipo OpenMP de unos 18 MB |
| medium | e2e | `vigia_ee/security.py` | Los errores que no salen de un handler del panel (404 y 405 de ruteo, 413 del middleware de tamaño) llegan al navegador como JSON crudo y en inglés, no como la página de error del panel |
| medium | a11y-visual | `vigia_ee/templates/audit.html` | Auditoría: entre 640 y 1080 px de ancho la tabla desborda su contenedor, la columna Detalle queda fuera de vista y axe marca scrollable-region-focusable (serious) |
| medium | a11y-visual | `vigia_ee/static/app.js` | Los atajos de una sola tecla (j, k, ?, 1, 2, 3) actúan en toda la página y no se pueden desactivar (WCAG 2.1.4 Character Key Shortcuts, nivel A) |
| medium | api-edge | `vigia/backends/vllm_proxy.py` | Modo vLLM: con stop strings la respuesta incluye el stop y el resto del token, porque el texto se arma con logprobs e ignora message.content |
| medium | api-edge | `vigia/backends/hf.py` | El texto del usuario con tokens especiales de la plantilla se tokeniza como control: un mensaje de user puede inyectar un turno system |
| medium | ml-correctness | `vigia/evidence.py` | gold_correct marca incorrecta la respuesta exacta y acepta respuestas erróneas cuando el gold sin tipo tiene varios números (rangos, marcadores, horas, fracciones) |
| medium | ops-resilience | `vigia_ee/db.py` | Con la base bloqueada, un request de la API tarda unos 57 s en devolver 503 en lugar de unos 5 s: los workers de jobs retienen el RLock de Database durante cada busy_timeout |
| low | isolation-rbac | `vigia_ee/auth.py` | Bajo carga, un admin degradado a viewer sigue creando API keys admin válidas después de que la degradación quedó confirmada. El arreglo de R3 solo cubre el body lento. |
| low | api-edge | `vigia_ee/runtime.py` | GET /v1/vigia/traces/{id} de la plataforma dice calibrated=true en una traza de stream cortado, que el núcleo guardó con calibrated=false |
| low | api-edge | `vigia/backends/vllm_proxy.py` | Modo vLLM: un prompt que excede el contexto responde code invalid_prompt; el backend HF responde context_length_exceeded |
| low | ml-correctness | `vigia/labels.py` | parse_dates_es no reconoce el ordinal '1ro'/'1ero'/'1er' y pierde el día: una respuesta de fecha correcta queda marcada como error |
| low | ml-correctness | `vigia/calibrate.py` | split_groups no reserva lugar en train para un estrato de 3 o más preguntas: con 5 preguntas y fracciones 0,3/0,3, train recibe siempre la única pregunta de la otra clase |
| low | ops-resilience | `vigia_ee/app.py` | Un cierre ordenado por SIGTERM (STOPSIGNAL del Dockerfile) o Ctrl+Break no sale con código 0: uvicorn vuelve a lanzar la señal y el proceso muere por el manejador por defecto |
