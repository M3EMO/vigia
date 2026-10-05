# Seguridad de Vigía

Este documento resume el modelo de amenazas, los controles implementados y los riesgos aceptados.
Cada control nombra el archivo que lo implementa y el test que lo cubre. La fuente de cada
requisito es la nota `vigia-seguridad-operacion-2026-10-04.md` del vault global. La operación
segura (secretos, TLS, backups) está en [`docs/operacion.md`](../operacion.md).

## 1. Modelo de amenazas

Activos:

- Texto de prompts y respuestas de clientes (puede traer PII, datos de salud o datos legales).
- Scores por token, detectores calibrados y etiquetas de revisión.
- Credenciales: API keys, contraseñas, sesiones, peppers, secretos de webhooks.
- Auditoría: evidencia de quién hizo qué.

Actores:

| Actor | Capacidad supuesta | Objetivo |
|---|---|---|
| Cliente de la API de otro tenant | Tiene una API key válida de su org | Leer o alterar datos de otra org |
| Usuario del panel con rol bajo | Tiene sesión de viewer o reviewer | Escalar privilegios, mutar configuración |
| Atacante web | Controla una página que visita un usuario logueado | CSRF, XSS por texto del modelo, clickjacking |
| Atacante de red interna | Alcanza el puerto de Vigía | Fuerza bruta de login o keys, DoS, SSRF vía webhooks o RAG |
| Atacante con copia de la base | Lee el archivo SQLite o un backup | Recuperar keys o contraseñas |
| Autor de datos de calibración | Sube JSONL o texto que llega a revisión | Envenenar el umbral o inyectar instrucciones a un juez |

Fuera de alcance: un atacante con root en el nodo, con acceso a los secretos montados o con
control del modelo servido. Vigía es autoalojado: el perímetro de red, el TLS y el cifrado del
disco son del operador.

## 2. Controles implementados

### 2.1 Identidad y sesiones

| Control | Implementación | Test |
|---|---|---|
| API key `vg_live_<kid>_<secreto><crc32>`; la base guarda solo HMAC-SHA256 con pepper | `vigia_ee/auth.py` (`generate_api_key`, `key_hmac`) | `test_auth.py::test_at_p03_api_key_shown_once_and_stored_as_hmac` |
| Checksum rechaza keys mal formadas sin consultar la base | `auth.py::parse_api_key` | `test_auth.py::test_api_key_format_and_checksum_detected_without_db` |
| Key revocada o vencida: 401 inmediato, sin cache | `auth.py::verify_api_key` | `test_auth.py::test_at_p03_revoked_or_expired_key_returns_401` |
| Key en query string: 400 | `auth.py::reject_query_credentials` | `test_auth.py::test_at_p03_key_in_query_string_is_rejected` |
| Scopes `ingest`, `read`, `admin` | `auth.py::Principal.has_scope` | `test_auth.py::test_api_key_scopes_wrong_pepper_and_last_used`, `test_app.py::test_trace_route_requires_read_scope` |
| Rotación de pepper por `kid` | `auth.py::PepperRing` | `test_auth.py::test_pepper_rotation_keeps_old_keys_valid` |
| Contraseñas scrypt N=2^17, r=8, p=1; rehash al login | `auth.py::hash_password`, `needs_rehash` | `test_auth.py::test_password_hash_default_params_format_and_verify`, `test_login_rehashes_when_params_change` |
| Política NIST: 15–128 caracteres, blocklist, sin email | `auth.py::check_password_policy` | `test_auth.py::test_at_p04_password_policy_nist` |
| Login con email inexistente tarda igual (hash descartable) | `auth.py::login`, `_dummy_hash` | `test_auth.py::test_wrong_password_and_unknown_email_fail_the_same_way` |
| Bloqueo de cuenta: 10 fallos, 15 minutos; bloqueo por IP | `auth.py::login` | `test_auth.py::test_account_locks_after_10_failures_for_15_minutes`, `test_at_p06_login_blocked_by_ip_after_n_failures` |
| Cookie `__Host-` Secure, HttpOnly, SameSite=Strict | `auth.py::set_session_cookie` | `test_auth.py::test_at_p04_login_sets_host_cookie_with_strict_attributes` |
| Sesión regenerada al login; logout invalida en servidor | `auth.py::SessionManager` | `test_auth.py::test_at_p04_session_is_regenerated_on_login`, `test_at_p04_logout_invalidates_session_on_server` |
| Timeouts: 30 min de inactividad, 8 h absoluto | `auth.py::SessionManager.resolve` | `test_auth.py::test_at_p04_idle_and_absolute_timeouts` |
| Credenciales se re-validan al terminar de leer el body | `auth.py::_await_body` | `test_auth.py::test_disabled_user_cannot_finish_a_slow_post_started_before`, `test_revoked_key_cannot_finish_a_slow_request_started_before` |

### 2.2 Autorización y aislamiento

| Control | Implementación | Test |
|---|---|---|
| Toda ruta declara permiso; deny-by-default | `auth.py::PERMISSIONS`, `undeclared_routes` | `test_auth.py::test_at_p02_every_route_declares_a_permission`, `test_app.py::test_AT_P02_every_route_declares_permission_and_viewer_cannot_mutate` |
| Matriz de roles owner > admin > reviewer > viewer | `auth.py::has_permission`, `can_assign_role` | `test_verify_isolation.py::test_permission_matrix_matches_the_documented_roles`, `test_auth.py::test_role_assignment_rules` |
| Cambio de rol o deshabilitar usuario cierra sus sesiones | `auth.py::change_role` | `test_verify_isolation.py::test_role_change_and_disable_invalidate_sessions` |
| Repositorio con `org_id` obligatorio en cada consulta | `vigia_ee/db.py::TenantScopedRepository` | `test_db.py::test_repository_requires_org_id`, `test_at_p01_other_tenant_resources_are_invisible` |
| Recurso de otro tenant: siempre 404 (barrido de rutas) | `db.py`, `panel.py::not_found` | `test_isolation.py::test_AT_P01_every_route_with_ids_returns_404_for_other_tenant`, `test_verify_isolation.py::test_head_options_and_id_variants_do_not_distinguish_foreign_ids` |
| IDs aleatorios, no secuenciales | `db.py::new_id` | `test_db.py::test_new_id_is_random_and_not_sequential` |
| Exports, métricas y health sin datos de otros tenants | `retention.py::export_org`, `metrics.py` | `test_verify_isolation.py::test_exports_contain_only_own_tenant`, `test_metrics_and_health_carry_no_tenant_data` |
| Jobs y webhooks nunca cruzan tenants | `jobs.py`, `webhooks.py` | `test_verify_isolation.py::test_jobs_and_webhooks_never_cross_tenants` |
| El núcleo abierto no importa la plataforma | `vigia/` | `test_core_boundary.py::test_AT_P18_core_never_imports_platform` |

### 2.3 Web y API

| Control | Implementación | Test |
|---|---|---|
| CSRF firmado y atado a la sesión; `Sec-Fetch-Site: cross-site` rechazado | `vigia_ee/security.py::enforce_csrf` | `test_security.py::test_at_p05_csrf_token_required_and_bound_to_session`, `test_at_p05_cross_site_fetch_is_rejected_even_with_token` |
| CSP con nonce por respuesta, sin `unsafe-inline` ni `unsafe-eval`; HSTS, nosniff, frame-ancestors none | `security.py::SecurityHeadersMiddleware` | `test_security.py::test_at_p12_security_headers_and_strict_csp`, `test_panel.py::test_AT_P12_rendered_pages_have_strict_csp_and_nonced_scripts` |
| Panel sin recursos externos, sin `eval` ni `innerHTML` | `vigia_ee/templates/`, `vigia_ee/static/` | `test_panel.py::test_AT_P12_templates_have_no_inline_styles_handlers_or_external_resources`, `test_AT_P12_static_js_has_no_eval_or_inner_html` |
| Texto del modelo escapado; bidi y controles neutralizados | Jinja2 autoescape, `vigia_ee/pii.py::safe_display` | `test_panel.py::test_AT_P13_model_text_with_script_bidi_and_control_is_escaped`, `test_pii.py::test_at_p13_safe_display_strips_bidi_and_control_characters` |
| CSV sin inyección de fórmulas | `pii.py::neutralize_csv_cell` | `test_pii.py::test_at_p13_csv_neutralizes_formula_prefixes`, `test_panel_admin.py::test_AT_P13_audit_csv_export_neutralizes_formulas` |
| `Cache-Control: no-store` en respuestas autenticadas | `security.py::SecurityHeadersMiddleware` | `test_security.py::test_cache_control_no_store_only_for_authenticated`, `test_verify_isolation.py::test_exports_and_api_traces_are_never_cacheable` |
| Error 500 genérico con `request_id`, sin stack trace | `security.py::RequestContextMiddleware` | `test_security.py::test_at_p17_internal_error_is_generic_with_request_id`, `test_app.py::test_AT_P17_internal_errors_are_generic_with_request_id` |
| Error de validación sin eco del input | `security.py::_validation_handler` | `test_security.py::test_validation_error_does_not_echo_input` |
| Fail-closed: error del backend de auth da 503, nunca 200 | `auth.py::_resolve_session`, `_resolve_key` | `test_auth.py::test_fail_closed_when_auth_missing_or_backend_errors` |
| Sin `/docs`, `/openapi.json` ni header `Server` | `vigia_ee/app.py::create_app`, `cmd_serve` | `test_app.py::test_cli_serve_hides_server_header_and_binds_localhost` |
| `vigia serve` sin key: solo loopback, rechaza Origin cruzado y DNS rebinding | `vigia/serve.py::browser_origin_problem` | `test_serve.py::test_without_key_rejects_cross_site_browser_and_foreign_host` |

### 2.4 Consumo de recursos

| Control | Implementación | Test |
|---|---|---|
| Body sobre el tope: 413; dataset con tope propio | `security.py::BodySizeLimitMiddleware`, `app.py::PathBodyLimitMiddleware` | `test_security.py::test_at_p07_body_over_limit_returns_413`, `test_app.py::test_body_limit_is_strict_except_on_dataset_upload` |
| `max_tokens` fuera de rango: 422 | `vigia/api.py` | `test_api.py::test_input_validation_is_422_openai_style` |
| Rate limit por key y por tenant: 429 con `Retry-After` | `security.py::RateLimiter`, `auth.py::_key_principal` | `test_auth.py::test_api_key_rate_limits_per_key_and_tenant`, `test_security.py::test_at_p06_burst_returns_429_with_retry_after` |
| Memoria del rate limiter acotada (LRU) | `security.py::RateLimiter` | `test_security.py::test_rate_limiter_bounds_memory` |
| Cola de generación acotada: 503 `server_overloaded` | `api.py::GenerationGate` | `test_api.py::test_many_concurrent_streams_do_not_exhaust_threadpool` |
| Body que nunca termina: 408 a los 30 s | `auth.py::_await_body` | `test_auth.py::test_body_that_never_finishes_is_cut_with_408` |
| `vigia.tiers` del cliente no puede quitar features del detector | `api.py` | `test_api.py::test_requested_tiers_cannot_drop_detector_features` |

### 2.5 Datos, auditoría y salidas

| Control | Implementación | Test |
|---|---|---|
| `store_text=none` no guarda texto; `redacted` guarda marcadores de PII | `db.py::insert_trace`, `pii.py::redact` | `test_db.py::test_store_text_none_stores_no_text`, `test_at_p08_store_text_redacted_keeps_markers_and_never_the_data` |
| PII argentina: DNI con contexto, CUIT/CUIL módulo 11, CBU, tarjeta Luhn, email, teléfono, IP | `pii.py` | `test_pii.py::test_at_p08_redact_replaces_every_argentine_identifier_with_marker`, `test_cuit_module_11_rejects_invalid_check_digit` |
| Retención por tenant con purga; supresión puntual; export | `vigia_ee/retention.py` | `test_retention.py::test_at_p10_purge_org_removes_only_expired_traces_of_that_tenant`, `test_delete_trace_request_is_scoped_and_audited`, `test_at_p10_export_contains_everything_of_the_tenant_and_nothing_foreign` |
| Export del tenant sin secretos | `db.py::export_all` | `test_verify_isolation.py::test_tenant_export_carries_no_secrets` |
| Auditoría append-only (triggers) con cadena de hashes por org | `vigia_ee/audit.py`, `schema.sql` | `test_audit.py::test_at_p09_triggers_block_update_and_delete`, `test_at_p09_altering_a_row_breaks_verification` |
| La auditoría rechaza claves de secretos en el detalle | `audit.py::_check_detail` | `test_audit.py::test_detail_rejects_secret_keys_and_bad_action` |
| Logs JSON con allowlist, sin query string, CR/LF escapados | `security.py::JsonLogFormatter` | `test_security.py::test_json_logger_uses_allowlist_and_single_lines`, `test_access_log_omits_query_string`, `test_sanitize_log_value_escapes_crlf_and_controls` |
| `abstain` no filtra el texto original al cliente | `api.py` | `test_api.py::test_AT_A03_buffer_abstain_never_leaks_original_text`, `test_runtime.py::test_trace_view_hides_text_tokens_and_spans_on_abstain` |
| Detector: JSON con hash SHA-256, sin pickle; otro modelo se rechaza | `vigia/detector.py` | `test_detector.py::test_at_m02_tampered_artifact_is_rejected`, `test_at_m02_detector_for_other_model_is_rejected` |
| `/metrics` con token en tiempo constante; sin token, 404 | `app.py::_health_router` | `test_app.py::test_AT_P14_metrics_require_token_and_have_no_tenant_labels`, `test_AT_P14_metrics_disabled_without_token` |

### 2.6 Salidas de red (SSRF)

| Control | Implementación | Test |
|---|---|---|
| Webhooks: https obligatorio, localhost, redes privadas y 169.254.169.254 bloqueados, revalidación antes de cada entrega, sin seguir redirects | `vigia_ee/webhooks.py`, `vigia/policy.py::check_url` | `test_webhooks.py::test_at_p11_create_blocks_unsafe_destinations`, `test_at_p11_delivery_revalidates_destination_and_blocks_ssrf`, `test_redirects_are_not_followed` |
| Hostname que resuelve a IP privada: bloqueado | `policy.py::check_url` | `test_webhooks.py::test_create_blocks_hostname_resolving_to_private_ip`, `test_policy.py::test_check_url_blocks_hostname_resolving_to_private_ip` |
| Firma Standard Webhooks (HMAC v1) verificable | `webhooks.py` | `test_webhooks.py::test_at_p11_signature_matches_standard_webhooks_reference_vector` |
| RAG solo hacia hosts permitidos | `policy.py::HttpRetriever` | `test_policy.py::test_AT_A07_allowlist_is_explicit_and_exclusive`, `test_AT_A07_http_retriever_rejects_host_outside_allowlist` |

### 2.7 Despliegue

| Control | Implementación | Test |
|---|---|---|
| Usuario no-root, filesystem de solo lectura, sin capabilities, `no-new-privileges` | `Dockerfile`, `compose.yaml` | Sin test automático: Docker no está disponible en la máquina de desarrollo |
| Secretos por archivo (`*_FILE`) | `runtime.py::_env_secret`, `auth.py::PepperRing.from_env`, `app.py::_read_owner_password` | `test_runtime.py::test_settings_from_env_reads_every_variable`, `test_auth.py::test_pepper_ring_parsing_and_env`, `test_app.py::test_cli_init_reads_password_from_file_and_creates_owner` |
| Sin red saliente en runtime (`HF_HUB_OFFLINE=1`) | `Dockerfile` | Sin test automático |

## 3. Canales laterales y riesgos aceptados

| Riesgo | Por qué se acepta | Mitigación actual | Cambio que lo cerraría |
|---|---|---|---|
| **Email único global.** `users.email` es único en toda la plataforma porque el login no pide organización. Un admin de la org A que invita un email ya registrado en la org B recibe "No se pudo crear una cuenta con ese email." y así sabe que existe. | El login por email sin org es más simple para los usuarios. | El mensaje no nombra a la otra org. El alta queda en auditoría y requiere rol admin. | Login con organización o emails únicos por org. |
| **Rate limit por proceso.** Los buckets viven en memoria (`security.py::RateLimiter`). Un reinicio los vacía. Con varios procesos o nodos, el límite efectivo se multiplica. | La plataforma corre un solo proceso (`serve` fuerza `workers=1`). | El bloqueo de cuenta por 10 fallos sí persiste en la base. | Buckets en la base o en Redis al pasar a varios nodos. |
| **TOCTOU de DNS en webhooks y RAG.** `check_url` resuelve el host y valida las IPs; después httpx resuelve de nuevo al conectar. Un DNS con TTL 0 puede devolver una IP privada en la segunda resolución. | Cerrarlo exige conectar a la IP validada con SNI propio. | Validación al crear y antes de cada entrega; https obligatorio fuera de la allowlist; sin redirects. | Transporte httpx que conecte a la IP ya validada. Mientras tanto, filtrá la salida a nivel de red. |
| **Scores como canal lateral (OWASP LLM10).** Los `p_error` por token se parecen a logprobs y ayudan a destilar el detector. | Los scores son el producto. | Solo con API key válida; rate limit por key y por tenant. | Redondeo opcional de scores, cuotas diarias. |
| **Truncar el final de la auditoría.** Borrar las últimas filas no rompe la cadena. | Requiere acceso de escritura al archivo de la base. | Triggers append-only contra UPDATE y DELETE desde la app. | Exportar periódicamente el último hash fuera del nodo (`/audit/export`). |
| **Sin RLS.** SQLite no tiene row-level security. El aislamiento vive solo en la aplicación. | Un nodo, una base, un repositorio con `org_id` obligatorio. | Barridos automáticos de rutas y de exports por tenant. | Postgres con RLS `FORCE` al pasar a varios nodos (ADR A-10). |
| **Sin cifrado en reposo propio.** La base y los backups guardan texto según `store_text`. | Cifrado de disco es responsabilidad del operador. | `store_text=redacted` por defecto; `none` disponible. | — |
| **Sin MFA.** Login solo con contraseña. | Pendiente (P-40, P2). | Política NIST de 15 caracteres, bloqueo de cuenta e IP. | TOTP para owner y admin. |
| **Carga del checkpoint.** El backend HF usa `from_pretrained` sin forzar `use_safetensors`; `trust_remote_code` queda en su default (desactivado). | Los modelos soportados se publican en safetensors. | Modelo pre-descargado y offline en la imagen. | Forzar `use_safetensors=True` y fijar el hash del snapshot. |
| **Redacción de PII incompleta.** Las regex no cubren nombres, direcciones ni fechas. | Sin NER en el runtime (sin dependencias pesadas). | `store_text=none` para datos sensibles. | Detector NER opcional. |

## 4. Respuesta a incidentes

1. Revocá las API keys comprometidas en el panel. La revocación es inmediata.
2. Rotá el pepper si se filtró la base o el archivo de peppers (`docs/operacion.md` §7.1).
3. Exportá la auditoría del período (`/audit/export`) y verificá la cadena.
4. Buscá en el log los eventos `login_failed`, `api_key_rejected`, `csrf_rejected` y
   `rate_limited` por `request_id`.
5. El responsable del tratamiento notifica según su marco legal. Ver
   [`docs/cumplimiento.md`](../cumplimiento.md).
