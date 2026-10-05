# API pública por módulo

Referencia para agentes. Generada de los reportes de la ola 1; la fuente de verdad es el código.

```text
vigia.spans:
  word_bounds(text: str) -> list[tuple[int, int]]
  words(text: str, tokens: Sequence[ScoredToken]) -> list[Word]   # Word(start, end, token_start, token_end, score: float|None, flag: bool)
  build_spans(text: str, tokens: Sequence[ScoredToken], *, gap_max: int = 1, min_tokens: int = 1) -> list[Span]

vigia.policy:
  RAG_FALLBACK_ACTION: Action = "human_review"; RAG_FALLBACK_ACTIONS = ("human_review", "abstain")
  rag_fallback(policy) -> Action   # policy.rag_fallback_action si es válida; si no, RAG_FALLBACK_ACTION (API y playground)
  class PolicyError(VigiaError, ValueError); class RetrieverError(VigiaError); class UnsafeURLError(RetrieverError, ValueError)
  matched_actions(policy: Policy, response_p_error: float | None, spans: Sequence[Span]) -> list[Action]
  decide(policy: Policy, response_p_error: float | None, spans: Sequence[Span]) -> Action
  validate_policy(policy: Policy, *, rag_allowed_hosts: Collection[str] = (), resolve: Callable[[str], list[str]] | None = None) -> Policy
  policy_from_dict(data: dict[str, Any], *, rag_allowed_hosts: Collection[str] = (), resolve: Callable[[str], list[str]] | None = None) -> Policy
  check_url(url: str, allowed_hosts: Collection[str] = (), *, resolve: Callable[[str], list[str]] | None = None) -> str
  class Retriever(Protocol): def retrieve(self, query: str) -> list[str]
  class HttpRetriever(url: str, *, allowed_hosts: Collection[str] = (), timeout: float = 5.0, http_client: httpx.Client | None = None, resolve=None, max_documents: int = 5, max_chars: int = 8000); .retrieve(query) -> list[str]
  mask_spans(text: str, spans: Sequence[Span], mask: str = "___") -> str
  build_rag_query(question: str, answer: str, spans: Sequence[Span], max_chars: int = 2000) -> str
  inject_context(messages: list[dict[str, str]], documents: Sequence[str]) -> list[dict[str, str]]

vigia.api:
  TRACE_HEADER = "X-Vigia-Trace-Id"; REQUEST_ID_HEADER = "X-Request-Id"
  @dataclass TraceRecord(trace_id: str, org_id: str | None, key_id: str | None, created_at: str, model: ModelInfo, prompt_messages: list[dict[str, str]], generation: Generation, result: VigiaResult, latency_ms: float, overhead_ms: float, source: str = "api", rag_status: str | None = None)
  @dataclass RequestContext(backend: Backend, scorer: Scorer | None, policy: Policy, org_id: str | None = None, key_id: str | None = None, sink: Callable[[TraceRecord], None] | None = None, trace_lookup: Callable[[str], dict[str, Any] | None] | None = None, retriever: Retriever | None = None, source: Literal["api", "playground"] = "api", closing: Callable[[], bool] | None = None)
    closing: True desde que el proceso empezó a cerrarse; la plataforma lo llena con runtime.shutting_down (context_for)
  create_api_router(get_context: Callable[[Request], Awaitable[RequestContext]], *, max_tokens_limit: int = 4096, default_max_tokens: int = 512, max_queued_generations: int = 64, queue_timeout_s: float = 120.0) -> fastapi.APIRouter
    rutas: POST /v1/chat/completions, POST /v1/completions, GET /v1/models, GET /v1/vigia/traces/{trace_id}
    usage.prompt_tokens = stream.prompt_tokens (TokenStream) si es > 0; si no, backend.count_prompt_tokens(messages) (duck typing); si no, 0.
  REMOTE_MAX_CONCURRENCY = 16; RETRY_AFTER_S = 5
  class GenerationBusy(Exception)   # cola llena o espera vencida
  class GenerationGate(slots, *, max_waiting, timeout_s): async .acquire() (FIFO en el event loop; GenerationBusy) ; .release() (seguro desde cualquier hilo) ; .waiting ; .slots
  generation_gate(backend, *, max_waiting=64, timeout_s=120.0) -> GenerationGate   # uno por backend, compartido por la API y el playground del panel; max_waiting/timeout_s los fija el primer llamador
    cupo: backend.max_concurrency (duck typing, entero >= 1), o REMOTE_MAX_CONCURRENCY si backend.mode == "reduced", o 1
  Errores ({"error": {message, type, code, param, request_id}}, sin stack):
    400 invalid_request_error: invalid_prompt | invalid_value | context_length_exceeded (InvalidRequestError.code del backend)
    422 unsupported_value: top_logprobs > 0; logprobs en /v1/completions; tools/tool_choice/functions/function_call/response_format/logit_bias (chat) y echo/best_of/suffix/logit_bias (completions) con valor no inocuo; temperature/top_p/n/stream/logprobs null = default
    422 en vigia.alpha si alpha es menor que el mínimo calibrado del detector
    499 client_closed (solo en el log: el cliente cortó mientras esperaba el cupo o generaba)
    502 upstream_unavailable con Retry-After: el backend lanzó BackendUnavailableError (p. ej. VllmProxyError)
    503 server_overloaded con Retry-After (cola del GenerationGate llena o espera vencida); 503 shutting_down con Retry-After (closing() es True); 503 detector_unavailable (fail_closed)

vigia.backends.vllm_proxy:
  class VllmProxyError(BackendUnavailableError)   # 5xx, 401, 404, transporte o respuesta ilegible -> la API responde 502
  CLIENT_REJECT_STATUS = {400, 413, 422} -> InvalidRequestError (400 en la API); DEFAULT_MAX_CONCURRENCY = 16; PING_TIMEOUT_S = 2.0; PING_CACHE_S = 5.0
  class VllmProxyBackend(base_url: str, model: str, api_key: str | None = None, http_client: httpx.Client | None = None, *, n_experts: int | None = None, top_k: int = 0, n_layers: int = 0, model_type: str = "unknown", timeout: float = 120.0, max_concurrency: int = 16)
    .mode = "reduced"; .max_concurrency (cupo de generation_gate; ValueError si no es entero >= 1); .model_info() -> ModelInfo (router_sha256=None); .capabilities() -> Capabilities(tiers=(F0, U), streaming=False)
    .stream(...) -> VllmTokenStream (hace el request al llamar; .prompt_tokens leído de usage) ; .ping() -> bool (GET /v1/models, timeout 2 s, cache 5 s, nunca lanza; /readyz lo usa) ; .close()
  parse_routed_experts(raw: Any) -> np.ndarray  # int64 [len, capas, top_k]
  routed_usage_features(routed: np.ndarray, n_experts: int) -> list[dict[str, float]]
  usage_features(counts: np.ndarray, n_experts: int) -> tuple[float, float, float]
  top_logprob_features(logprob: float | None, top: list[float]) -> dict[str, float]
  token_offsets(pieces: list[str], raw_bytes: list[list[int] | None]) -> tuple[str, list[tuple[int, int]]]
  nan_usage(n_layers: int) -> dict[str, float]
```

```text
## vigia.contracts (cambios posteriores a la ola 1)
class InvalidRequestError(VigiaError, ValueError)(message, *, code="invalid_prompt")   # .code; la API responde 400 con ese code
class BackendUnavailableError(VigiaError)   # el servidor de generación no respondió o falló; la API responde 502 upstream_unavailable
ModelInfo.router_sha256: str | None = None   # SHA-256 de los routers MoE (vigia.adapters.router_sha256); None = desconocido
ModelInfo.fingerprint() -> {model_id, model_type, n_experts, top_k, quantization} + router_sha256 SOLO si no es None
  Regla de compatibilidad: un detector sin router_sha256 (previo al campo o de un backend que no lo conoce) sigue valiendo;
  Detector.check_model compara router_sha256 solo si el artefacto y el modelo lo tienen, y si difieren lanza DetectorMismatchError.
Scorer.tau: escalón (tau del mayor alpha guardado <= alpha), no interpola.

## vigia_ee.db
utcnow() -> datetime ; to_iso(dt) -> str (UTC 'YYYY-MM-DDTHH:MM:SS.ffffffZ', se ordena como texto) ; parse_iso(s) -> datetime ; new_id(nbytes=16) -> str (hex de secrets)
STORE_TEXT_MODES, ROLES, REVIEW_LABELS, REVIEW_STATES, MIGRATIONS: tuple[tuple[int,str],...] (1 = schema.sql; 2 = audit_log.org_seq; 3 = traces.status TEXT NOT NULL DEFAULT 'ok' y traces.rag_status TEXT)
class Database(path: str|Path, *, busy_timeout_ms=5000)  # ':memory:' sirve en tests; si un PRAGMA o una migración fallan, cierra la conexión y relanza
  .migrate()->int ; .schema_version()->int ; .transaction() (context manager, BEGIN IMMEDIATE; una anidada se une a la exterior)
  .execute(sql, params=())->int (filas afectadas) ; .insert(sql, params)->int (rowid) ; .executemany(sql, rows) ; .query(sql, params)->list[dict] ; .query_one(sql, params)->dict|None ; .close()
create_org(db, name, *, store_text='redacted', retention_days=30, policy=None) -> str
list_org_ids(db) -> list[str]   # para jobs de sistema (purga)
lookup_user_by_email(db, email) / lookup_api_key(db, key_id) / lookup_session(db, id_hash) -> dict|None   # SIN tenant, solo para resolver el principal
class TenantScopedRepository(db, org_id)   # toda lectura/escritura de datos de cliente; otro tenant -> None/False/[]
  org: get_org() ; update_org(**{name,store_text,retention_days,external_judge_allowed})->bool ; get_policy()->dict|None ; set_policy(dict)->bool ; delete_org()->bool
  users: create_user(email, password_hash, role)->id (ValueError si el email existe) ; get_user(id) ; list_users() (sin password_hash) ; update_user(id, *, role=None, disabled=None, password_hash=None)->bool ; record_login_failure(id, *, max_failures, lock_until)->bool ; reset_login_failures(id) ; delete_user(id)->bool
  sessions: create_session(id_hash, user_id, csrf_secret, *, created_at, expires_at) ; touch_session(id_hash, now) ; delete_session(id_hash)->bool ; delete_user_sessions(user_id)->int
  api_keys: insert_api_key(key_id, *, name, hmac_hex, pepper_kid, scopes, created_by=None, expires_at=None) ; get_api_key(id) ; list_api_keys() (sin hmac) ; revoke_api_key(id, *, now=None)->bool ; touch_api_key(id, now)
  traces: insert_trace(*, model_id, response_text, action, prompt_text=None, tokens=(), spans=(), trace_id=None, n_tokens=None, api_key_id=None, source='api', mode='full', response_p_error=None, calibration_id=None, alpha=None, tau=None, latency_ms=None, overhead_ms=None, degraded=False, finish_reason=None, usage=None, created_at=None, status='ok', rag_status=None) -> trace_id   # status/rag_status = los de VigiaResult
      # recibe TEXTO CRUDO; aplica store_text del org aca (none|redacted|full) y re-mapea offsets de tokens/spans al texto redactado. tokens/spans: dicts (idx|index,text,start,end,p_error,flag,out_entropy,r_ent_mean / start,end,score) o ScoredToken/Span de contracts. trace_id opcional (^[A-Za-z0-9_-]{8,64}$).
    get_trace(id)->dict con 'tokens' y 'spans' | None ; list_traces(*, limit=50, cursor=None, action=None, source=None, since=None, until=None, min_p_error=None, degraded=None) -> (rows, next_cursor|None) (ValueError si cursor invalido) ; delete_trace(id)->bool ; purge_before(ts: datetime)->int
    recent_token_signals(calibration_id, *, limit_traces=500)->{n_traces, tokens: [{p_error, flag, r_ent_mean, out_entropy}]}   # monitor de drift: últimas trazas no degradadas de ese detector, tokens con p_error
  review: create_review_item(trace_id, *, span_idx=None, min_reviewers=1)->id|None ; get_review_item(id) ; list_review_items(*, state=None, limit=100) ; reserve_review_item(item_id, user_id, *, until, now=None)->bool ; add_review_decision(item_id, user_id, label, *, note=None, now=None)->id|None (avanza a needs_other/completed segun min_reviewers) ; list_review_decisions(item_id)->list|None
  detectors: insert_detector(detector_id, artifact_json, *, report_json='{}', dataset_id=None, created_by=None) (status candidate; ValueError si existe) ; get_detector(id) ; list_detectors() (sin artifact_json) ; set_detector_report(id, report_json)->bool ; activate_detector(id)->bool (retira el activo anterior y setea orgs.active_detector_id) ; get_active_detector()->dict|None
  datasets: insert_dataset(name, content: bytes, *, n_items)->id ; get_dataset(id) ; list_datasets() (sin content) ; delete_dataset(id)->bool
  webhooks: insert_webhook(url, secret, events)->id (SSRF lo valida webhooks.py) ; get_webhook(id) (con secret) ; list_webhooks(*, event=None, enabled_only=False, with_secret=False) ; set_webhook_disabled(id, bool)->bool ; delete_webhook(id)->bool
  export_all()->dict JSON-serializable (schema_version, exported_at, org, users, api_keys, traces+tokens+spans, review_items, review_decisions, detectors, datasets, webhooks, jobs, audit_log; sin password_hash/hmac/secret/sesiones)
  # jobs.py usa Database directo (tabla jobs es infraestructura; filtrar org_id a mano).

## vigia_ee.auth
SCRYPT_N=2**17 ; MIN_PASSWORD_LEN=15 ; MAX_PASSWORD_LEN=128 ; SCOPES=('ingest','read','admin') ; ROLE_RANK ; PERMISSIONS: dict[perm,min_role] = account:self(viewer, logout/propia clave), traces:read, metrics:read, model:read, review:read, policy:read, calibration:read (viewer); review:write, playground:use (reviewer); policy:write, calibration:write, keys:read, keys:write, users:read, users:write, audit:read, data:export, settings:read (admin); settings:write, data:delete (owner)
hash_password(pw, *, n=SCRYPT_N)->'scrypt$N$r$p$salt$hash' ; verify_password(pw, stored)->bool ; needs_rehash(stored, *, n=SCRYPT_N)->bool ; check_password_policy(pw, *, email=None) (PasswordPolicyError)
class PepperRing(peppers, current_kid): .parse('kid1:secret,kid2:secret') (ultimo = vigente, secreto >= 32 chars) ; .from_env(environ=None) (VIGIA_PEPPERS_FILE preferido, luego VIGIA_PEPPERS; PepperConfigError)
generate_api_key()->(token, key_id, secret) ; parse_api_key(token)->(key_id, secret) (InvalidApiKey, valida checksum sin base) ; key_hmac(pepper, key_id, secret)->hex
@dataclass(frozen) Principal(org_id, role=None, user_id=None, key_id=None, scopes=frozenset()) ; .can(perm) ; .has_scope(scope) ('admin' incluye todo)
has_permission(role, perm)->bool (deny-by-default) ; can_assign_role(actor_role, target_role)->bool
class SessionManager(db, *, idle=30min, absolute=8h, clock=utcnow, touch_every=60s): create(user_id, org_id, *, previous_id=None)->raw_id ; resolve(raw)->SessionInfo|None ; destroy(raw)->bool ; purge_expired()->int
SessionInfo(id_hash, user_id, org_id, role, email, csrf_secret).csrf_token
set_session_cookie(response, raw_id) ; clear_session_cookie(response)   # cookie __Host-vigia_session
@dataclass AuthState(db, peppers, sessions=None, key_limiter=None, tenant_limiter=None, login_ip_limiter=<10 fallos/15 min>, scrypt_n=SCRYPT_N, clock=utcnow) ; .repo(org_id)
install_auth(app, state)   # app.state.vigia_auth = state; si falta, las dependencias devuelven 503 (fail-closed)
login(state, email, password, *, ip, previous_session_id=None)->LoginResult(session_id, principal, csrf_token)  # BLOQUEANTE (scrypt): llamar con run_in_threadpool. Lanza LoginFailed (mensaje generico) o RateLimitExceeded (429). Bloquea la cuenta 15 min tras 10 fallos; audita login/login_failed/account_locked.
logout(state, raw_session_id)->bool
register_user(state, org_id, email, password, role, *, actor=None)->user_id (actor None = bootstrap; PermissionError/ValueError/PasswordPolicyError)
change_role(state, org_id, user_id, new_role, *, actor)->bool (invalida sesiones del usuario)
set_password(state, org_id, user_id, new_password, *, actor)->bool (invalida sesiones)
create_api_key(state, org_id, name, scopes, *, actor=None, expires_at=None)->(token, key_id)  # token solo aca
revoke_api_key(state, org_id, key_id, *, actor=None)->bool ; verify_api_key(state, token)->Principal|None
DEPENDENCIAS (cada ruta DEBE usar una; asi declara su permiso):
  require_session(perm)            -> panel: sesion + CSRF en POST/PUT/PATCH/DELETE (header X-CSRF-Token o campo de form csrf_token) + permiso. Deja request.state.principal, .session, .csrf_token (para templates), .authenticated
  require_api_key(scope)           -> API: Authorization: Bearer vg_live_...; 400 si hay credencial en query; 401 con WWW-Authenticate; 403 sin scope; aplica key_limiter y tenant_limiter (429)
  require_session_or_key(perm, scope) -> Bearer si hay header Authorization, si no sesion
  public_route()                   -> declaracion explicita de ruta publica (health, login, estaticos)
  Uso: p: Annotated[Principal, Depends(require_session("policy:write"))]  o  @app.get("/livez", dependencies=[Depends(public_route())])
  Un nombre de permiso/scope desconocido lanza ValueError al definir la ruta.
INTROSPECCION (AT-P02, para el test de app): route_permission(route)->PermissionSpec|None ; undeclared_routes(app, *, exempt_paths=())->list['METHODS /path'] (mounts/docs solo pasan si estan exentos) ; viewer_mutation_violations(app)->list[str] ; PermissionSpec(kind, permission, scope)

## vigia_ee.security
SESSION_COOKIE='__Host-vigia_session' ; SAFE_METHODS
install_security(app, *, max_body_bytes=1_048_576, trust_incoming_request_id=False)  # orden: SecurityHeaders -> RequestContext (request_id, log acceso sin query, 500 generico) -> BodySizeLimit; + handlers JSON sin eco de input: bajo /v1 el sobre de OpenAI {error:{message,type,code,param,request_id}} (error_payload/is_openai_path); en el panel {error, message, request_id} (+fields en 422). CodedHTTPException(status, detail, code) fija error.code
SecurityHeadersMiddleware (nonce en request.state.csp_nonce; Cache-Control no-store si hay sesion, Authorization o request.state.authenticated) ; build_csp(nonce)->str ; STATIC_SECURITY_HEADERS
RequestContextMiddleware (request.state.request_id, header X-Request-ID) ; BodySizeLimitMiddleware(max_bytes) ; install_error_handlers(app) ; request_id_of(request)
error_payload(path, status, code, message, request_id, *, param=None) -> dict ; is_openai_path(path) -> bool ; class CodedHTTPException(HTTPException)(status_code, detail, code, *, headers=None)
class RateLimiter(rate, burst, *, clock=time.monotonic, max_keys=100_000): hit(key, cost=1)->None|segundos ; blocked(key)->None|segundos ; reset(key)
RateLimitExceeded(retry_after, scope='') (HTTPException 429 con Retry-After) ; enforce_rate_limit(limiter|None, key, *, scope='', cost=1)
csrf_token_for(csrf_secret, session_id_hash)->str ; check_fetch_site(request) ; async enforce_csrf(request, csrf_secret, session_id_hash)
LOG_FIELDS (allowlist) ; sanitize_log_value(v, *, max_len=500) ; JsonLogFormatter ; configure_json_logging(level=INFO, stream=None)->Logger('vigia') ; log_event(logger, event, *, level=INFO, **fields) (descarta campos fuera de la allowlist)

## vigia_ee.audit
GENESIS ; append(db, org_id|None, actor, action, target=None, detail=None, *, now=None)->dict (actor: Principal, dict con user_id/key_id, o None; detail rechaza claves password/token/secret/...) ; verify_chain(db, org_id)->ChainCheck(ok, checked, broken_at) ; verify_rows(rows, *, expected_prev=None)->ChainCheck ; export_range(db, org_id, *, start=None, end=None)->list[dict] (con 'detail' parseado) ; row_hash(row) ; canonical_json(v)

## vigia_ee.pii
redact(text)->(texto_redactado, conteos: dict[kind,int]) ; redact_with_map(text)->Redaction(text, counts, replacements: tuple[Replacement(start,end,new_start,new_end,kind)]) ; map_offset(replacements, pos)->int ; find_pii(text)->list[(start,end,kind)]
kinds/marcadores: [EMAIL] [CBU] [TARJETA] [CUIT] [IP] [DNI] [TELEFONO]
cuit_is_valid(s) ; cbu_is_valid(s) ; luhn_is_valid(s)
safe_display(text)->str (quita bidi, controles salvo \n\t, invisibles y tags; NO escapa HTML: eso es autoescape de Jinja)
neutralize_csv_cell(v) ; csv_dumps(rows, header=None)->str
```

```text
## vigia.detector
canonical_json(obj) -> str            # json sort_keys, separators (',',':'), ensure_ascii=False, allow_nan=False
artifact_sha256(artifact: Mapping) -> str   # sha256 de canonical_json(artifact sin 'sha256')
sigmoid(z) -> np.ndarray ; logit(p) -> np.ndarray
validate_artifact(artifact: Mapping) -> None   # DetectorMismatchError ante cualquier falla
class Detector:   # implementa contracts.Scorer
  id: str (= sha256[:16]); sha256: str; feature_names: tuple[str, ...]; default_alpha: float; imputed_counts: Counter[str]
  Detector(artifact: Mapping)                          # valida
  Detector.build(*, model, feature_names, mean, std, weights, bias, platt={a,b}, response_calibrator={a,b}, thresholds=[{alpha,tau,expected_flag_rate,expected_precision}], default_alpha, metrics, training={dataset_sha256,n_tokens,n_responses,label_source,judge,created_at}) -> Detector
  Detector.load(source: str | PathLike | Mapping) -> Detector
  Detector.loads(text: str) -> Detector               # para la columna detectors.artifact_json
  .artifact -> dict (copia) ; .thresholds -> list[dict] ; .model -> dict
  .to_json(*, indent=None) -> str (canónico si indent=None) ; .save(path) -> None
  .raw_logit(X) -> np.ndarray ; .score_matrix(X: (n,d) en orden feature_names, NaN imputado) -> np.ndarray
  .matrix_from_features(features: Mapping[str, array], n: int) -> np.ndarray
  .score_tokens(tokens: list[TokenEvent]) -> list[float | None]
  .score_response(token_p_errors: list[float | None]) -> float | None
  .tau(alpha: float) -> float      # escalón: tau del mayor alpha guardado <= alpha (no interpola: conserva FNR <= alpha); alpha < min guardado -> ValueError; alpha > max -> tau del max
  .check_model(info: ModelInfo) -> None   # DetectorMismatchError; router_sha256 se compara solo si los dos lo tienen
  .drift_reference -> dict | None  # metrics.drift_reference (vigia.drift); None en artefactos previos
DRIFT_REFERENCE_KEY = "drift_reference"

Artefacto JSON (claves exactas): schema_version, model (= ModelInfo.fingerprint(): 5 claves + router_sha256 opcional, 64 hex), feature_names (ordenadas), mean, std, weights, bias, platt{a,b}, response_calibrator{a,b}, thresholds[{alpha,tau,expected_flag_rate,expected_precision|null}], default_alpha, metrics, training{dataset_sha256,n_tokens,n_responses,label_source,judge,created_at}, sha256. metrics.drift_reference es opcional y se valida si existe: {feature: {edges (estrictamente crecientes), fractions (len(edges)+1, suman 1), n >= 1, tau? (solo flag)}}. Token: p = sigmoid(platt.a * (((x-mean)/std)·weights + bias) + platt.b). Respuesta: sigmoid(rc.a * logit(max p) + rc.b). Flag: p >= tau.

## vigia.calibrate
DEFAULT_ALPHAS = (0.05, 0.1, 0.2, 0.3); DEFAULT_C_GRID = (0.01, 0.1, 1.0, 10.0)
class CalibrationError(ValueError)
build_dataset(features, labels, groups, response_ids) -> Dataset   # descarta labels == IGNORE_LABEL (-100)
dataset_sha256(data: Dataset) -> str
split_groups(groups, *, cal_frac=0.2, test_frac=0.2, seed=0, labels=None) -> dict[str, list[str]]   # {'train','cal','test'}; con labels (una por token) estratifica por la clase mayoritaria de cada pregunta; train_detector y evaluate_report la pasan
split_masks(groups, split: Mapping[str, Sequence[str]]) -> dict[str, np.ndarray]       # CalibrationError si hay fuga
fit_logistic(X, y, *, l2=1.0, sample_weight=None, max_iter=100, tol=1e-10) -> tuple[np.ndarray, float]
fit_platt(scores, y) -> tuple[float, float]
crc_threshold(p, y, response_ids, alpha) -> CRCResult(alpha, tau, n, feasible)
crc_thresholds(p, y, response_ids, alphas) -> tuple[list[dict], list[CRCResult]]
train_detector(features: Mapping[str, array], labels, groups, response_ids, model_info: ModelInfo, *, alphas=DEFAULT_ALPHAS, default_alpha=0.1, cal_frac=0.2, test_frac=0.2, seed=0, c_grid=DEFAULT_C_GRID, split=None, label_source='computable', judge=None, dataset_sha=None, created_at=None) -> tuple[Detector, dict]
  el artefacto guarda metrics.drift_reference = vigia.drift.build_reference(p_cal, tau(default_alpha), features crudas de DRIFT_FEATURES en cal)
  split_report = {seed, groups{train,cal,test: [str]}, counts{split: {n_groups,n_responses,n_tokens,positive_rate}}, c_selected, cv_auroc, dropped_features{name: motivo}, response_calibrator, crc[{alpha,tau,n_responses,feasible}], detector_id}

## vigia.labels
IGNORE_LABEL = -100; ARTICLES_ES; MONTHS_ES; JUDGE_SCHEMA; JUDGE_SYSTEM_PROMPT; JUDGE_PROMPT_SHA256; REJECTION_REASONS
normalize_es(text, *, strip_accents=True) -> str
exact_match(prediction, gold: str | Sequence[str], *, strip_accents=True) -> bool
token_f1(prediction, gold: str | Sequence[str], *, strip_accents=True) -> float
parse_number_es(token) -> float | None ; extract_numbers(text) -> list[float] ; numeric_match(prediction, gold, *, abs_tol=1e-9) -> bool
parse_dates_es(text) -> list[date] ; date_match(prediction, gold: date | str) -> bool
parse_date_parts_es(text) -> list[(año, mes|None, dia|None)]   # menciones completas, mes+año y año suelto de 3-4 dígitos, en orden, sin superposición
spans_to_token_labels(tokens: Sequence[TokenEvent | (start,end)], spans: Sequence[(start,end)]) -> list[int]
label_tokens_from_answer(tokens, answer_span_offsets: Sequence[(start,end)] | None, correct: bool) -> list[int]
class InvalidJudgment(VigiaError) (.reason in REJECTION_REASONS); class JudgeError(VigiaError)
JudgeVerdict(label: int, spans: tuple[(start,end)], span_texts: tuple[str])
validate_judgment(raw: str | Mapping, answer: str) -> JudgeVerdict
class Judge(Protocol): model: str; judge(question, answer, evidence=None) -> str | Mapping
class OpenAICompatibleJudge(base_url, api_key, model, *, timeout=120.0, max_retries=2, client: httpx.Client | None = None, allow_insecure_http=False); .judge(question, answer, evidence=None) -> str
class FakeJudge(responses: Mapping[answer, str | dict] | Callable[[q, a, ev], str | dict]); .judge(...) -> str
JudgeReport(model, prompt_sha256, verdicts: dict[id, JudgeVerdict], rejected: dict[id, reason], counts: Counter); .n_items; .to_dict()
label_with_judge(judge: Judge, items: Iterable[{id, question, answer, evidence?}]) -> JudgeReport

## vigia.evaluate
class SingleClassError(ValueError)
auroc(y, s, w=None) -> float ; auprc(y, s, w=None) -> float ; recall_at_fpr(y, s, fpr_target=0.1) -> float ; precision_at_recall(y, s, recall_target=0.8) -> float
roc_curve(y, s) -> (fpr, tpr, thr) ; pr_curve(y, s) -> (recall, precision, thr)
ece(y, p, n_bins=15, w=None) -> float ; reliability_bins(y, p, n_bins=15) -> list[{mean_pred, frac_pos, count, lo, hi}]
brier(y, p, w=None) -> float ; log_loss(y, p, w=None) -> float ; threshold_metrics(y, p, tau) -> {precision, recall, f1, fnr, flag_rate}
span_iou_per_response(flags, labels, response_ids) -> (ids, iou) ; span_iou(flags, labels, response_ids) -> float ; aurc(y, p) -> {aurc, e_aurc}
cluster_bootstrap_ci(metric, y, s, clusters, *, n_boot=2000, seed=0, level=0.95) -> (lo, hi)
evaluate_report(features, labels, groups, response_ids, model_info, *, detector: Detector | None = None, split=None, alpha=None, n_boot=2000, seed=0, level=0.95, cal_frac=0.2, test_frac=0.2, c_grid=DEFAULT_C_GRID) -> dict
  report = {schema_version, created_at, model, detector_id, alpha, tau, bootstrap{n_boot,level,seed,cluster}, dataset{n_tokens_test,n_responses_test,n_groups_test,prevalence_token,prevalence_response,n_groups,response_note?}, detector, tiers{F0,F1,'F0+F1',todos}, baselines{out_entropy,out_nll,h_probe}, crc[...], curves{roc[{name,fpr,tpr,auc}], pr{recall,precision,ap,threshold_point}, reliability[...], histogram{edges,correct,error}}}
  entrada = {available, id, n_features, missing_features, alpha, tau, token{metric: {value, ci:[lo,hi], n_boot_valid}}, response{...}} o {available: False, reason}
  métricas token/response: auroc, auprc, recall_at_fpr, precision_at_recall_0.8, precision_at_recall_0.9, ece, brier, log_loss, precision, recall, fnr, f1, flag_rate, prevalence, delta_auroc_vs_detector, delta_auprc_vs_detector (+ response: span_iou, aurc, e_aurc)
render_report_html(report: Mapping) -> str   # usa report['latency'] = {baseline:{p50,p95,p99}, vigia:{...}, target?} si existe

## vigia.charts
OKABE_ITO; CATEGORY_LIGHT; CATEGORY_DARK; CIVIDIS; DASHES; CHART_CSS (servir como CSS estático o con nonce)
fmt_num(x, digits=3) -> str ; fmt_pct(x, digits=1) -> str ; cividis(t) -> str
data_table(caption, headers, rows) -> str ; figure(svg, table_html) -> str
reliability_svg(bins, ece, *, title=...) / reliability_table(...)
roc_svg(curves[{name,fpr,tpr,auc}], *, title=...) / roc_table(...)
pr_svg(curves[{name,recall,precision,ap}], prevalence, *, threshold_point={recall,precision,tau}|None, title=...) / pr_table(...)
score_histogram_svg(edges, counts_correct, counts_error, *, tau=None, title=...) / score_histogram_table(...)
latency_svg(baseline{p50,p95,p99}, vigia{p50,p95,p99}, *, target=0.05, unit='ms', title=...) / latency_table(...)
flag_trend_svg(labels, rates, *, recalibrations: Sequence[int]=(), title=...) / flag_trend_table(...)
heatmap_svg(matrix, row_labels, col_labels, *, value_label='uso', title=...) / heatmap_table(...)
```

```text
vigia.features (numpy puro, ops sobre el último eje):
  softmax(z, axis=-1); log_softmax(z, axis=-1); entropy(p, axis=-1); normalized_entropy(p, support, axis=-1)
  top_n_entropy(log_probs, n=20); output_features(logits[..., V], targets[...]) -> dict{out_entropy, out_nll, out_margin, out_top20_entropy}
  router_entropy_n(z); topk_weight_entropy(w); topk_mass(p, idx); logit_margins(z, k) -> (r_margin12, r_margin_kk1)
  route(z, gating, k, *, renormalize=False, correction_bias=None, n_group=1, topk_group=1, scaling=1.0, jitter_eps=0.01) -> (idx, w)
  router_stats(z, gating, k, **route_kwargs) -> dict{r_ent_n, r_ent_k, r_topk_mass, r_margin12, r_margin_kk1, idx, w}
  usage_counts(idx, n_experts, weights=None) -> (..., N); gini_impurity(u); inverse_hhi(u); usage_entropy(u); usage_stats(counts) -> dict{u_gini, u_ihhi, u_ent}
  route_switch(prev_idx, idx); expert_cosine(e[...,k,d], w[...,k]); expert_dispersion(e, w); weighted_jsd(p[...,k,V], w[...,k])
  layer_aggregates(values, axis=0) -> dict{mean, max, std}   (NaN-aware)
  Gating = Literal['softmax_topk','topk_softmax','sigmoid_group','sigmoid_topk','sparsemixer']

vigia.adapters:
  RouterSpec(model_type, block_attr, router_attr, experts_attr, gating, num_experts_key, top_k_key='num_experts_per_tok', renormalize=False, shared_experts_attr=None)
  register(spec) -> None; supported_model_types() -> tuple[str, ...]; get_spec(model_type) -> RouterSpec
  MoEBlock(layer: int, block, router, experts, spec)
  find_moe_blocks(model) -> list[MoEBlock]   # UnsupportedModelError con la lista de soportadas
  decoder_layers(model) -> nn.ModuleList; final_norm(model) -> nn.Module | None; top_k(model) -> int
  routing_params(block: MoEBlock) -> dict   # kwargs de features.route leídos del router real
  model_info(model, model_id: str) -> contracts.ModelInfo   # incluye router_sha256
  router_sha256(blocks: list[MoEBlock]) -> str | None   # SHA-256 de weight, bias y e_score_correction_bias de cada router en orden de capa (float32 LE, con capa/nombre/forma); None si un tensor está en 'meta'. Otro dtype de carga cambia el hash
  Registrados: mixtral, gpt_oss, qwen2_moe, qwen3_moe, olmoe, granitemoe, deepseek_v3, phimoe

vigia.extractor:
  TokenSignals(features: dict[str, float], expert_ids: list[list[int]] | None)
  SignalExtractor(model, tiers=(Tier.F0, Tier.F1), probe_layer=None, *, keep_router_logits=False)  # context manager
    .begin_step(attention_mask=None, score_mask=None) -> None      # máscaras (B, T)
    .end_step(logits=None, targets=None) -> list[list[TokenSignals]]  # logits (B,T',V) y targets (B,T') alineados a la derecha; HookNotFiredError si un hook no disparó
    .reset() -> None; atributos: blocks, layers, n_experts, top_k, tiers (frozenset[Tier]), disabled (list[str]), probe_layer, router_logits (dict capa -> Tensor (B,T,N) si keep_router_logits)
  expert_outputs(experts, hidden[n,H], idx[n,k]) -> Tensor[n,k,H]
  f2_unsupported_reason(blocks) -> str | None
  resolve_probe_layer(probe_layer: int | None, n_layers: int) -> int   # default round(0.95*L), rango [1, L]

vigia.backends.hf:
  HFBackend(model_id_or_model, tokenizer=None, device='cpu', dtype=None, *, model_id=None, probe_layer=None)
    .model_info() -> ModelInfo; .capabilities() -> Capabilities
    .stream(messages, *, max_tokens, temperature=0.0, top_p=1.0, seed=None, stop=None, tiers=(Tier.F0, Tier.F1)) -> GenerationStream
    .generate(messages, **kwargs_de_stream) -> contracts.Generation
    .teacher_force(messages, response_text, tiers=(Tier.F0, Tier.F1)) -> list[TokenEvent]
    prompt rechazado al validar: InvalidRequestError(code='context_length_exceeded' si no entra en el contexto, 'invalid_prompt' si queda vacío); sigue siendo ValueError
  GenerationStream(Iterator[TokenEvent]): .finish_reason ('stop'|'length'), .text, .prompt_tokens, .completion_tokens, .timings {prefill_s, decode_s, total_s}, .close()
  IncrementalDecoder(tokenizer): .push(token_id) -> str, .flush() -> str, .pending

vigia.harmony:
  HarmonyParser().feed(text: str, is_special: bool) -> str   # canal: final|analysis|commentary|header|control|none; atributos channel, role, ended_with
  is_scorable(event) -> bool   # channel == 'final' and not is_special
  final_view(response, spans=(), *, prefix='<|start|>assistant') -> (texto_final, spans en ese texto)   # sin marcado harmony devuelve todo sin cambios; lo usan evidence.computable_labels y calibration._collect
  SPECIAL_TOKENS, STOP_TOKENS, SCORED_CHANNEL='final', HEADER='header', CONTROL='control', NO_CHANNEL='none'
```

## Ola 2

```text
## vigia_ee.jobs
JOB_STATES; MAX_PAYLOAD_BYTES=256KB; DEFAULT_LEASE=15min; DEFAULT_LANE='default'; MAX_RECOVER_INTERVAL_S=60; class PermanentJobError(Exception) (marca failed sin reintentar); format_error(exc)->'Tipo: mensaje' (sin stack, max 500)
class JobQueue(db, *, clock=None, lease=15min, poll_interval=1.0, backoff_base=5.0, backoff_max=3600.0)
  .db ; .now()->datetime
  .register(kind, handler(job_dict)->dict|None, *, retry_schedule: Sequence[float]|None=None, lane='default')  # sin schedule: backoff exponencial con jitter determinista (sha256(job_id:intento), hasta +25%)
  .registered_kinds(lane=None)->tuple ; .lanes()->tuple ; .recover_interval (segundos, atributo)
  .configure_lane(lane, *, workers=1, max_running_per_org=None)  # hilos del carril y tope de jobs en curso por tenant en este proceso; los jobs de sistema cuentan como un tenant
  .enqueue(kind, payload: Mapping, *, org_id=None, run_after=None, max_attempts=5)->id  # kind ^[a-z][a-z0-9_.-]{0,31}$, payload JSON sin NaN
  .get(org_id, job_id)->dict|None (otro tenant o job de sistema -> None) ; .get_system(job_id) ; .list(org_id, kind=None, limit=50) (max 200) ; .list_system(kind=None, *, states=JOB_STATES)
  job dict = columnas de jobs con payload y result ya parseados (sin payload_json/result_json)
  .depth(*, include_scheduled=False, lane=None)->int (queued vencidos + running; lane=None cuenta todos los carriles) ; .purge_finished(before, *, org_id)->int
  .retry_delay(job_id, attempts, schedule=None)->float ; .recover_orphans()->int (running con lease vencido -> queued, o failed si agoto intentos)
  .run_once(lane=None)->job final|None (solo reclama kinds registrados; claim en BEGIN IMMEDIATE) ; .run_until_idle(max_jobs=1000)->int
  .start() (hilos por carril: 'vigia-jobs' y 'vigia-jobs-<lane>'; recupera huérfanos al arrancar y cada recover_interval; nunca reencola un job propio en curso; idempotente) ; .is_alive() exige que todos los hilos vivan y latan ; .stop(timeout=5.0)->bool ; .is_alive(max_age=None)->bool (para /readyz; un job largo cuenta como vivo hasta el lease) ; .last_beat (monotonic)

## vigia_ee.webhooks
JOB_KIND='webhook'; EVENT_TYPES=('trace.flagged','review.completed','calibration.activated'); RETRY_SCHEDULE_S=(5,300,1800,7200,18000,36000,36000); MAX_ATTEMPTS=8; MAX_PAYLOAD_BYTES=20000; DELIVERY_TIMEOUT_S=15.0; DEFAULT_TOLERANCE_S=300; DEFAULT_MAX_CONSECUTIVE_FAILURES=5; LANE_WORKERS=4; MAX_RUNNING_PER_ORG=1
class WebhookVerificationError(ValueError); class DeliveryError(Exception)
generate_secret()->'whsec_<b64 32 bytes>' ; sign(secret, msg_id, timestamp:int, payload:bytes)->'v1,<b64>'
verify(secret, msg_id, timestamp, payload, signature_header, *, tolerance=300, now=None)->cuerpo parseado (acepta varias firmas separadas por espacio; lanza WebhookVerificationError)
validate_url(url, *, allowed_hosts=(), resolve=None)->url (UnsafeURLError de vigia.policy -> la API responde 422)
create_webhook(db, org_id, url, events, actor, *, allowed_hosts=(), resolve=None)->(webhook_id, secret)  # valida SSRF y eventos, audita 'webhook_create'; el secreto se muestra una vez
delete_webhook(db, org_id, webhook_id, actor)->bool ; set_enabled(db, org_id, webhook_id, enabled, actor, *, allowed_hosts=(), resolve=None)->bool (habilitar re-valida la URL; audita webhook_enable/webhook_disable)
build_body(event_type, data, *, now)->bytes ({type,timestamp,data}; >20KB -> data={trace_id?, truncated:true})
emit(db, jobs, org_id, event_type, data)->list[job_id] (un job por webhook habilitado suscrito; mismo webhook-id para el evento)
make_delivery_handler(db, jobs, *, http_client=None, allowed_hosts=(), resolve=None, timeout=15.0, max_consecutive_failures=5)
register_delivery(jobs, *, http_client=None, allowed_hosts=(), resolve=None, timeout=15.0, max_consecutive_failures=5)  # registra 'webhook' con RETRY_SCHEDULE_S en lane='webhook' (LANE_WORKERS hilos, MAX_RUNNING_PER_ORG por tenant)
  entrega: re-valida SSRF (bloqueado -> failed permanente), POST sin redirects con webhook-id/webhook-timestamp(hora del intento)/webhook-signature, solo 2xx es exito, no lee el cuerpo de la respuesta; plazo TOTAL de timeout s por entrega (conexión+TLS+envío+headers), vencerlo es fallo
  desactivación: tras N entregas seguidas cuyo último intento falló (contadas en cada intento) deshabilita y audita 'webhook_auto_disabled'; la entrega que lo desactiva termina failed

## vigia_ee.metrics
CONTENT_TYPE='text/plain; version=0.0.4; charset=utf-8'; DURATION_BUCKETS; OVERHEAD_BUCKETS; MAX_ROUTES=200; ACTIONS
escape_label_value(v) ; escape_help(t) ; format_value(x) ; status_class(code)->'2xx'..'5xx'
class Metrics(*, version=None, jobs_depth: Callable[[],int]|None=None, review_depth: Callable[[],int]|None=None, drift: Callable[[], Mapping[str,float]]|None=None)   # las fuentes corren en un hilo con tope GAUGE_TIMEOUT_S
  .observe_request(route_template|None, status, duration_s) ; .observe_overhead(ratio) ; .add_tokens(scored, low_confidence=0) ; .inc_action(action) ; .set_calibrations(detector_ids) (solo 16 hex) ; .inc_auth_failure() ; .inc_rate_limited() ; .set_review_queue_depth(n) ; .set_jobs_queue_depth(n) ; .set_drift({feature: psi}) (solo DRIFT_FEATURES, finitos >= 0) ; .render()->str
  familias: vigia_requests_total{route,status}, vigia_request_duration_seconds{route}, vigia_overhead_ratio, vigia_tokens_scored_total, vigia_low_confidence_tokens_total, vigia_actions_total{action}, vigia_calibration_info{detector_id}, vigia_auth_failures_total, vigia_rate_limited_total, vigia_review_queue_depth, vigia_jobs_queue_depth, vigia_drift_psi{feature} (máximo entre tenants; feature en DRIFT_FEATURES), vigia_build_info{version}
class MetricsMiddleware(app, metrics)  # ASGI puro; app.add_middleware(MetricsMiddleware, metrics=m); usa scope['route'].path, sin ruta -> 'unmatched'

## vigia_ee.retention
PURGE_KIND='purge'; SYSTEM_JOBS_RETENTION=30 dias
purge_org(db, org_id, *, now=None, jobs=None)->{traces, jobs, cutoff} (audita 'retention_purge' siempre; LookupError si no existe)
purge_all(db, jobs=None, *, now=None)->{orgs, traces, jobs} (un tenant que falla no frena al resto; al final lanza RuntimeError)
schedule(jobs, every_hours=24)->job_id (registra 'purge', idempotente; el handler encola la siguiente antes de purgar)
delete_trace_request(repo, trace_id, actor)->bool (audita 'trace_delete'; otro tenant -> False sin auditar)
export_org(repo, actor)->dict (repo.export_all + audita 'data_export' con conteos)
```

```text
vigia.datasets:
  DEFAULT_INSTRUCTION = "Respondé solo con la respuesta, en pocas palabras, sin explicar."; DEFAULT_MKQA_PATH = .cache/data/mkqa_es.jsonl; LABEL_MODES = ("gold","label","spans")
  class DatasetError(VigiaError, ValueError)   # el mensaje incluye la línea y el campo
  validate_item(obj, *, where="item") -> dict   # normalizado: {id, group, messages, question, mode, answers, response, label, hallucinated_spans ([[s,e]] ordenados), type, evidence}
    entrada: {id, messages|question, exactamente uno de answers | response+label | response+hallucinated_spans ([s,e] o substring único), group?, type?, evidence?}
  parse_jsonl(source: str|bytes|Iterable[str], *, max_items=100_000) -> list[dict]   # ids únicos, rechaza NaN, UTF-8 con BOM; JSON anidado en exceso (RecursionError) -> DatasetError con la línea
  read_jsonl(path, *, max_items) -> list[dict] ; write_jsonl(path, items) ; item_to_json(item) -> dict (inverso de validate_item)
  question_messages(question, instruction=DEFAULT_INSTRUCTION) -> [{"role":"user","content": f"{q}\n\n{instr}"}]
  stratified_sample(items, n, *, key="type", seed=0) -> list   # asignación proporcional por resto mayor
  load_mkqa_es(path=DEFAULT_MKQA_PATH, *, n=None, seed=0, types=None) -> list[item]
  items_sha256(items) -> str

vigia.evidence:
  DEFAULT_TIERS=(F0,F1,H); file names ROWS_FILE, FEATURES_FILE, MANIFEST_FILE, DETECTOR_FILE, SPLIT_FILE, REPORT_JSON, REPORT_HTML, SUMMARY_FILE, BENCH_FILE
  class EvidenceError(VigiaError)
  gold_correct(prediction, answers, item_type=None) -> (bool, {exact_match, token_f1, numeric, date})
  computable_labels(row) / judge_labels(row, judge) / label_row(row, judge=None) -> (labels per token, info)   # modo spans: harmony.final_view lleva los spans al texto final; si el texto final difiere -> rejected 'texto_final_distinto'
  class RunStore(out_dir): .open(manifest) (error if the configuration differs; model_info.router_sha256 is compared only if both manifests have it), .load_rows() (repairs a cut-off last line; a damaged middle line -> error), .append(row, X, names) (atomic chunk + fsync), .load_chunk(id), .rewrite_rows(rows), .manifest()
  extract(backend, items, out_dir, *, tiers=DEFAULT_TIERS, max_tokens=24, seed=0, judge=None, progress_cb=None) -> list[row]   # resumable
  @dataclass FeatureSet(names, X float32, labels, groups, response_ids, token_index, model_info, label_source, judge); .features(names=None) -> {name: float64 col}; .save(path); FeatureSet.load(path); .labeled_count()
  build_features(out_dir, rows=None) -> FeatureSet (writes features.npz)
  relabel(out_dir, *, judge=None) -> dict (rewrites rows, manifest and features.npz; writes labels_report.json with judge_vs_gold)
  detector_feature_names(names) -> F0+F1 names
  calibrate_features(fs, *, seed=0, alphas=DEFAULT_ALPHAS, default_alpha=0.1, feature_names=None) -> (Detector, split_report)
  save_calibration(out_dir, detector, split_report) -> (detector.json, split.json)
  evaluate_features(fs, *, detector=None, split=None, n_boot=2000, seed=0, alpha=None) -> report
  model_quality(rows) -> {generated:{n,correct,accuracy,hallucination_rate,by_type}, teacher_forced, all}
  run_evidence(backend, items, *, out_dir, max_tokens=24, tiers=DEFAULT_TIERS, seed=0, judge=None, n_boot=2000, alpha=None, bench_prompts=None, bench_runs=3, bench_warmup=1, progress_cb=None) -> report
    report = evaluate_report(...) + evidence{n_items,n_rows,n_generated,n_teacher_forced,skipped,labels_rejected,tiers,detector_features,max_tokens,seed,dataset_sha256,label_source,judge,model_quality,generation{tokens,seconds,tokens_per_s}} + split + (latency{baseline,vigia,target} y bench | latency_error)
  report_html(report) -> str (render_report_html + Table 5: model accuracy by type); summary_text(report) -> str; write_report(out_dir, report) -> paths; json_safe(obj); model_info_from_dict(d)

vigia.serve:
  create_app(backend, *, detector=None, policy=None, api_key=None, max_traces=1000, max_body_bytes=1_048_576, max_tokens_limit=4096, default_max_tokens=512, rag_allowed_hosts: Collection[str] = (), retriever: Retriever | None = None) -> FastAPI
    valida la política al crear (rag_url fuera de rag_allowed_hosts o a red privada -> error); sin retriever y con rag_url arma HttpRetriever
    sin key: 403 ante un Host u Origin no loopback o ante Sec-Fetch-Site cross-site/same-site; 415 ante un POST que no es application/json
  browser_origin_problem(headers) -> str | None   # motivo para rechazar un request sin key que no viene de un cliente local legítimo
    Bearer compared with hmac.compare_digest over sha256 (key >= 16 characters, no whitespace); without a key only loopback is accepted (403) and a warning is logged; credentials in the query -> 400; /livez, /readyz (503 if the backend fails); no docs/openapi; nosniff/no-store/CSP default-src 'none' headers; detector.check_model at creation; app.state.vigia_traces
  class TraceStore(max_items=1000): put(TraceRecord), get(id) -> dict|None (thread-safe LRU)
  trace_view(record) -> {id, object:"vigia.trace", created_at, model, source, finish_reason, response, vigia, latency_ms, overhead_ms, rag_status}   # on abstain: response=None, tokens/spans=[]
  is_loopback(host) -> bool

vigia.cli:
  main(argv=None) -> int (0 ok, 1 error with "error: ..." on stderr, 2 usage, 130 interrupted); build_parser(); `python -m vigia.cli`
  subcommands: extract, label (--judge-url/--judge-model; key only from VIGIA_JUDGE_API_KEY), calibrate (--tiers F0,F1), evaluate (needs split.json if a detector is used), bench, evidence (--bench N), serve (--host 127.0.0.1, --port, --detector, --policy, --api-key-env, --rag-allowed-hosts; refuses a non-loopback host without a key), models
  make_tiny_backend(arch="mixtral"|"gpt_oss", *, seed=0) -> HFBackend   # deterministic for a given seed, offline, byte-level tokenizer, model_id "tiny:<arch>"; gpt_oss uses a harmony template that opens the final channel; does not touch the global RNG
  load_backend(model, *, device="cpu", dtype="auto") -> HFBackend   # "tiny:<arch>" or an HF id/path
  parse_tiers("F0,F1,H") -> tuple[Tier, ...]; TINY_ARCHS; JUDGE_KEY_ENV; class CLIError(VigiaError)
```

```text
vigia_ee.runtime:
  SettingsError(ValueError)
  @dataclass Settings
    Campos del contrato: data_dir, db_path, model, backend, vllm_url, vllm_model, max_tokens_limit=1024, default_max_tokens=256, max_body_bytes=1_048_576, max_upload_bytes=20_971_520, metrics_token, rag_allowed_hosts, webhook_allowed_hosts, secure_cookies=True, base_detector_path.
    Campos extra, todos con default:
      vllm_api_key (VIGIA_VLLM_API_KEY[_FILE])
      peppers: PepperRing|None (None = PepperRing.from_env)
      key_rate_per_minute=120, tenant_rate_per_minute=600
      max_queue_depth=1000 (tope de /readyz)
      review_min_reviewers=2
      shutdown_timeout_s=30 (VIGIA_SHUTDOWN_TIMEOUT, tope del cierre ordenado)
      vllm_n_experts, vllm_top_k, vllm_n_layers (VIGIA_VLLM_N_EXPERTS / _TOP_K / _N_LAYERS): default None, obligatorios con backend=vllm porque fijan la normalización de U
      vllm_max_concurrency=16 (VIGIA_VLLM_MAX_CONCURRENCY, 1..4096): cupo de generaciones simultáneas hacia vLLM (VllmProxyBackend.max_concurrency)
    Settings.from_env(environ=None) exige VIGIA_MODEL. Los secretos aceptan *_FILE. Todo SettingsError nombra la variable VIGIA_* que el operador corrige.
  DRIFT_TRACES=500 ; DRIFT_MIN_TOKENS=200 ; DRIFT_CACHE_S=60
  build_backend(settings)
  make_tiny_backend(arch): delega en vigia.cli.make_tiny_backend
  render_prompt(messages) -> 'rol: contenido' separados por línea vacía (formato de traces.prompt_text)
  class Runtime(settings, *, backend=None, clock=None)
    Atributos del contrato: settings, db, auth (rate limiters por key y tenant), backend, model_info, jobs (con el handler webhook registrado), metrics, calibration.
    Atributos extra: base_detector, startup_complete, startup_errors, started_at.
    Métodos del contrato: repo, detector_for, policy_for, save_policy, context_for, playground_stream, close.
    save_policy hace merge sobre la política vigente, sube version, audita {changed, version} y rechaza un alpha menor que el mínimo del detector activo.
    Métodos extra (el panel puede usarlos):
      now()
      create_org(name, *, store_text, retention_days): audita org_create e instala el detector base
      ensure_base_detector(org_id)
      detector_alert(org_id) / policy_alert(org_id) -> str|None: texto para un banner del panel
      invalidate_detector(org_id)
      forget_org(org_id): saca de memoria detector, alertas, política, retriever, drift y la marca del detector base de una org borrada (panel /data/delete)
      emit_event(org_id, event_type, data): nunca lanza
      store_trace(org_id, TraceRecord) -> trace_id   # guarda también status y rag_status de VigiaResult
      trace_view(org_id, trace_id)   # misma forma que vigia.serve.trace_view: {id, object, created_at, model, source, finish_reason, response, vigia, latency_ms, overhead_ms, rag_status}
      shutting_down (property) ; begin_shutdown(): los requests en cola y el playground responden 503/corte; context_for pasa closing=lambda: shutting_down
      readiness() -> {db, worker, queue[, backend]}   # queue mide solo DEFAULT_LANE; backend aparece si el backend tiene ping() (proxy vLLM)
      drift_for(org_id, *, n_traces=DRIFT_TRACES, max_age=0.0) -> {available, reason, detector_id, n_traces, n_tokens, features: {feature: {psi, level, n}}, level}
        # PSI de las últimas trazas puntuadas con el detector activo contra detector.drift_reference; flag se recalcula con el tau de la referencia
      drift_gauges() -> {feature: psi máximo entre tenants}   # fuente de vigia_drift_psi; cachea DRIFT_CACHE_S por tenant

vigia_ee.calibration:
  JOB_KIND='calibrate', MAX_ITEMS=5000, MIN_RESPONSES=20, PLATFORM_N_BOOT=500, ANSWER_MAX_TOKENS=64, MAX_PENDING_JOBS=1
  Excepciones:
    DatasetError(ValueError)
    DatasetNotFound(DatasetError, LookupError)
    CalibrationBusy(DatasetError)   # la org ya tiene MAX_PENDING_JOBS calibraciones en cola o en curso
    DetectorNotFound(LookupError)
    ActivationError(ValueError)
    CalibrationJobError(PermanentJobError)
  CalibrationItem, parse_dataset(content: bytes, *, max_items=5000)   # campo opcional 'group' (define la pregunta del split)
  _collect: items con spans y marcado harmony pasan por vigia.harmony.final_view; si el texto final no coincide, skipped['texto_final_distinto']
  class CalibrationService(runtime)
    upload_dataset(principal, name, content) -> dataset_id
    start(principal, dataset_id, *, alphas=None, seed=0) -> job_id (max_attempts=1)
    job(org_id, job_id) / list_jobs(org_id, limit=20) -> dicts {id, state, created_at, updated_at, dataset_id, alphas, seed, result, error}
      error es el mensaje para el usuario, o uno genérico ante una falla interna
    activate(principal, detector_id): verifica modelo y alpha de la política, audita, invalida la cache y emite calibration.activated
    dataset_from_reviews(principal, name) -> dataset_id
  El reporte guardado es evaluate_report + report['calibration'] (job_id, dataset_id, dataset_name, n_items, n_responses, n_questions, n_tokens, positive_rate, skipped, split).

vigia_ee.app:
  TRACE_ROUTE
  UPLOAD_PATHS = ('/calibration/datasets',): solo esas rutas POST admiten max_upload_bytes + MULTIPART_MARGIN (65536); el resto usa max_body_bytes.
  Middlewares: PathBodyLimitMiddleware, StatusMetricsMiddleware (401/403 -> inc_auth_failure, 429 -> inc_rate_limited), UniqueRequestIdMiddleware.
  create_app(settings=None, *, runtime=None) -> FastAPI (docs/redoc/openapi desactivados). El lifespan arranca y detiene jobs y retention.schedule(24 h). Cierra el runtime solo si lo creó.
  build_parser() / main(argv=None, *, environ=None) con los subcomandos:
    init --org --email [--store-text --retention-days]: contraseña por getpass o VIGIA_OWNER_PASSWORD_FILE
    create-key --org-id --name [--scopes --expires-days]
    serve [--host 127.0.0.1 --port 8000 --proxy-headers --forwarded-allow-ips]: server_header=False, workers=1
```

```text
vigia_ee.panel:
- install_panel(app, runtime) incluye el router y registra panel_error_handler para PanelHTTPError. app.py ya lo usa.
- create_panel_router(runtime) -> APIRouter. Las rutas van en la raíz, sin prefijo.
- PanelHTTPError(HTTPException): un error de página se muestra en HTML con role=alert y request_id. Una página GET sin sesión redirige con 303 a /login?next=...
- page(permission): envuelve require_session y conserva PERMISSION_ATTR para la introspección.
- render, ENV, helpers fmt_* (es-AR, UTC-3), summary_stats, trace_window, latency_percentiles, usage_matrix, fragments_for, segments_for, parse_filters.

Rutas públicas:
- GET /login, POST /login
- GET /static/{app.css|app.js|favicon.svg|charts.css}. charts.css se genera desde vigia.charts.CHART_CSS.

Rutas con sesión:
- POST /logout, GET /account, POST /account/password
- GET / (resumen), GET /generations, GET /generations/{id}
- POST /generations/{id}/review, POST /generations/{id}/delete
- GET /playground, POST /playground (sin JS), POST /playground/stream (SSE, header X-CSRF-Token)
  Las dos toman vigia.api.generation_gate(runtime.backend), el mismo cupo que la API; lleno -> 503 con Retry-After
  (HTML con role=alert, o JSON {error: 'server_overloaded', message, retry_after}). El stream devuelve el cupo al cerrar.
- GET /model, GET /performance (latencia + sección "Deriva del detector": runtime.drift_for, tabla de PSI por señal con estado en texto)
- ClosingStreamingResponse(content, *, on_close=None, **kw): cierra el iterador sync y llama on_close aunque el cliente corte.

vigia_ee.panel_admin.register(router, runtime) agrega:
- GET /review, POST /review/next, POST /review/dataset
- GET /review/{id}, POST /review/{id}/reserve, POST /review/{id}/decide
- GET /calibration, POST /calibration/datasets (multipart), POST /calibration/start
- GET /calibration/jobs/{id}, GET /calibration/jobs/{id}/status (JSON)
- GET /calibration/detectors/{id}, POST /calibration/detectors/{id}/activate
- GET /policies, POST /policies
- GET /settings, POST /settings/org
- POST /settings/users, POST /settings/users/{id}/role, POST /settings/users/{id}/disable
- POST /settings/keys, POST /settings/keys/{id}/revoke
- POST /settings/webhooks, POST /settings/webhooks/{id}/delete, POST /settings/webhooks/{id}/test
- GET /audit, GET /audit/export?formato=json|csv&desde&hasta
- GET /data, GET /data/export, POST /data/delete

Del Runtime consumo: settings, db, auth, repo, detector_for, policy_for, save_policy, playground_stream, calibration.*, model_info, backend.capabilities(), jobs.list/enqueue/now, emit_event e invalidate_detector (a través de calibration), forget_org (POST /data/delete) y drift_for (GET /performance).
```

```text
vigia.bench:
  DEFAULT_TIERS_LIST = ((), (Tier.F0,), (Tier.F0, Tier.F1)); TARGET = 0.05; BENCH_SCHEMA_VERSION = "1"; BASELINE_LABEL = "sin_hooks"
  tier_label(tiers) -> str  # () -> "sin_hooks"; si no, los niveles unidos con "+", ej. "F0+F1"
  run_overhead_bench(backend: Backend | Callable[[], Backend], prompts: list[list[messages]], *, tiers_list=DEFAULT_TIERS_LIST, max_tokens=32 (>=2), warmup=3, runs=5, seed=0 (>=0), threads=None, ignore_eos=True, n_boot=1000, level=0.95, target=0.05, clock=time.perf_counter) -> dict
    La línea base () siempre se mide y siempre va primero. Las configuraciones se miden intercaladas con un orden que rota. El GC queda apagado mientras se mide. torch.set_num_threads se restaura al terminar. Un backend que crea la fábrica se cierra siempre, también ante un error.
    Devuelve {schema_version, config{model_id, model_type, tiers_list, n_prompts, max_tokens, warmup, runs, seed, threads, temperature, ignore_eos{requested, supported, applied, note}, order, metric, overhead, bootstrap{n_boot, level, seed, unit}, n_units, n_units_dropped, baseline_hooks}, baseline{p50, p95, p99, tpot_ms, ttft_ms, tok_s, n_requests, n_token_latencies}, tiers{<etiqueta>: {...las mismas claves..., overhead_ratio, ci:[lo, hi], ttft_overhead_ratio, ttft_ci, meets_target}}, vigia{la última configuración + label}, target, hardware{cpu, machine, threads, torch, python, platform}}.
    p50, p95 y p99 son la latencia por token en ms: el intervalo entre eventos medido con reloj de pared, con hooks y end_step incluidos. overhead_ratio es la mediana, sobre las unidades pareadas (corrida, prompt), del cociente p50(nivel)/p50(base), menos 1. El IC es un bootstrap con 1000 réplicas sobre las unidades. El dict entra tal cual en report['latency'] de render_report_html, que usa baseline, vigia y target. Es compatible con las llamadas que ya hacen vigia/cli.py y vigia/evidence.py.
vigia.extractor: la API pública no cambia. Cambian los internos: _Step.router ahora es dict[capa j -> logits Tensor (n, N) float32]; el uso acumulado es un ndarray (B, L, N). router_logits (con keep_router_logits) ya no es un clone: es una vista que vive solo durante el paso.
vigia.features: las firmas no cambian. usage_counts usa np.bincount, que da el mismo resultado y es más rápido.
vigia.backends.vllm_proxy: las firmas no cambian. top_logprob_features, usage_features y routed_usage_features delegan en vigia.features. usage_features ahora lanza ValueError si counts.shape != (n_experts,).
```

```text
vigia.drift (P-36, monitor de drift; numpy puro):
  DEFAULT_BINS=10; EPS=1e-4; WARN_PSI=0.1; ALERT_PSI=0.25; DRIFT_FEATURES=("p_error", "flag", "agg_mean:r_ent_n", "out_entropy"); LEVELS=("ok","warn","alert")
  summarize(values, bins=10) -> {edges, fractions, n}   # bordes interiores por cuantiles; con <= bins valores distintos, un bin por valor (puntos medios); bin i = (edges[i-1], edges[i]]
  psi_from_summary(summary, actual) -> float   # sum (a-e)·ln(a/e) con fracciones llevadas a EPS
  psi(expected_scores, actual_scores, bins=10) -> float   # = psi_from_summary(summarize(expected, bins), actual)
  level(psi) -> 'ok' (< 0.1) | 'warn' (>= 0.1) | 'alert' (>= 0.25) ; worst_level(levels) -> str | None
  drift_report(reference, current, features=DRIFT_FEATURES, *, bins=10) -> {feature: {psi, level, n}}   # reference[f]: resumen de summarize o valores crudos; sin referencia o sin datos actuales, la feature no aparece
  build_reference(p_error, tau, features=None, *, bins=10) -> {p_error, flag (+tau), [agg_mean:r_ent_n], [out_entropy]}   # lo guarda train_detector en metrics.drift_reference
```
