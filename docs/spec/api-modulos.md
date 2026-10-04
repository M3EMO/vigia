# API pública por módulo

Referencia para agentes. Generada de los reportes de la ola 1; la fuente de verdad es el código.

```text
vigia.spans:
  word_bounds(text: str) -> list[tuple[int, int]]
  words(text: str, tokens: Sequence[ScoredToken]) -> list[Word]   # Word(start, end, token_start, token_end, score: float|None, flag: bool)
  build_spans(text: str, tokens: Sequence[ScoredToken], *, gap_max: int = 1, min_tokens: int = 1) -> list[Span]

vigia.policy:
  RAG_FALLBACK_ACTION: Action = "human_review"
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
  @dataclass RequestContext(backend: Backend, scorer: Scorer | None, policy: Policy, org_id: str | None = None, key_id: str | None = None, sink: Callable[[TraceRecord], None] | None = None, trace_lookup: Callable[[str], dict[str, Any] | None] | None = None, retriever: Retriever | None = None, source: Literal["api", "playground"] = "api")
  create_api_router(get_context: Callable[[Request], Awaitable[RequestContext]], *, max_tokens_limit: int = 4096, default_max_tokens: int = 512) -> fastapi.APIRouter
    rutas: POST /v1/chat/completions, POST /v1/completions, GET /v1/models, GET /v1/vigia/traces/{trace_id}
    opcional (duck typing): si el backend tiene count_prompt_tokens(messages) -> int, se usa para usage.prompt_tokens; si no, 0.

vigia.backends.vllm_proxy:
  class VllmProxyError(VigiaError)
  class VllmProxyBackend(base_url: str, model: str, api_key: str | None = None, http_client: httpx.Client | None = None, *, n_experts: int | None = None, top_k: int = 0, n_layers: int = 0, model_type: str = "unknown", timeout: float = 120.0)
    .mode = "reduced"; .model_info() -> ModelInfo; .capabilities() -> Capabilities(tiers=(F0, U), streaming=False); .stream(...) -> Iterator[TokenEvent]; .close()
  parse_routed_experts(raw: Any) -> np.ndarray  # int64 [len, capas, top_k]
  routed_usage_features(routed: np.ndarray, n_experts: int) -> list[dict[str, float]]
  usage_features(counts: np.ndarray, n_experts: int) -> tuple[float, float, float]
  top_logprob_features(logprob: float | None, top: list[float]) -> dict[str, float]
  token_offsets(pieces: list[str], raw_bytes: list[list[int] | None]) -> tuple[str, list[tuple[int, int]]]
  nan_usage(n_layers: int) -> dict[str, float]
```

```text
## vigia_ee.db
utcnow() -> datetime ; to_iso(dt) -> str (UTC 'YYYY-MM-DDTHH:MM:SS.ffffffZ', se ordena como texto) ; parse_iso(s) -> datetime ; new_id(nbytes=16) -> str (hex de secrets)
STORE_TEXT_MODES, ROLES, REVIEW_LABELS, REVIEW_STATES, MIGRATIONS: tuple[tuple[int,str],...] (1 = schema.sql)
class Database(path: str|Path, *, busy_timeout_ms=5000)  # ':memory:' sirve en tests
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
  traces: insert_trace(*, model_id, response_text, action, prompt_text=None, tokens=(), spans=(), trace_id=None, n_tokens=None, api_key_id=None, source='api', mode='full', response_p_error=None, calibration_id=None, alpha=None, tau=None, latency_ms=None, overhead_ms=None, degraded=False, finish_reason=None, usage=None, created_at=None) -> trace_id
      # recibe TEXTO CRUDO; aplica store_text del org aca (none|redacted|full) y re-mapea offsets de tokens/spans al texto redactado. tokens/spans: dicts (idx|index,text,start,end,p_error,flag,out_entropy,r_ent_mean / start,end,score) o ScoredToken/Span de contracts. trace_id opcional (^[A-Za-z0-9_-]{8,64}$).
    get_trace(id)->dict con 'tokens' y 'spans' | None ; list_traces(*, limit=50, cursor=None, action=None, source=None, since=None, until=None, min_p_error=None, degraded=None) -> (rows, next_cursor|None) (ValueError si cursor invalido) ; delete_trace(id)->bool ; purge_before(ts: datetime)->int
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
install_security(app, *, max_body_bytes=1_048_576, trust_incoming_request_id=False)  # orden: SecurityHeaders -> RequestContext (request_id, log acceso sin query, 500 generico) -> BodySizeLimit; + handlers JSON {error, message, request_id} para HTTPException y 422 sin eco de input
SecurityHeadersMiddleware (nonce en request.state.csp_nonce; Cache-Control no-store si hay sesion, Authorization o request.state.authenticated) ; build_csp(nonce)->str ; STATIC_SECURITY_HEADERS
RequestContextMiddleware (request.state.request_id, header X-Request-ID) ; BodySizeLimitMiddleware(max_bytes) ; install_error_handlers(app) ; request_id_of(request)
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
  .tau(alpha: float) -> float      # interpola; alpha < min guardado -> ValueError; alpha > max -> tau del max
  .check_model(info: ModelInfo) -> None   # DetectorMismatchError

Artefacto JSON (claves exactas): schema_version, model (= ModelInfo.fingerprint()), feature_names (ordenadas), mean, std, weights, bias, platt{a,b}, response_calibrator{a,b}, thresholds[{alpha,tau,expected_flag_rate,expected_precision|null}], default_alpha, metrics, training{dataset_sha256,n_tokens,n_responses,label_source,judge,created_at}, sha256. Token: p = sigmoid(platt.a * (((x-mean)/std)·weights + bias) + platt.b). Respuesta: sigmoid(rc.a * logit(max p) + rc.b). Flag: p >= tau.

## vigia.calibrate
DEFAULT_ALPHAS = (0.05, 0.1, 0.2, 0.3); DEFAULT_C_GRID = (0.01, 0.1, 1.0, 10.0)
class CalibrationError(ValueError)
build_dataset(features, labels, groups, response_ids) -> Dataset   # descarta labels == IGNORE_LABEL (-100)
dataset_sha256(data: Dataset) -> str
split_groups(groups, *, cal_frac=0.2, test_frac=0.2, seed=0) -> dict[str, list[str]]   # {'train','cal','test'}
split_masks(groups, split: Mapping[str, Sequence[str]]) -> dict[str, np.ndarray]       # CalibrationError si hay fuga
fit_logistic(X, y, *, l2=1.0, sample_weight=None, max_iter=100, tol=1e-10) -> tuple[np.ndarray, float]
fit_platt(scores, y) -> tuple[float, float]
crc_threshold(p, y, response_ids, alpha) -> CRCResult(alpha, tau, n, feasible)
crc_thresholds(p, y, response_ids, alphas) -> tuple[list[dict], list[CRCResult]]
train_detector(features: Mapping[str, array], labels, groups, response_ids, model_info: ModelInfo, *, alphas=DEFAULT_ALPHAS, default_alpha=0.1, cal_frac=0.2, test_frac=0.2, seed=0, c_grid=DEFAULT_C_GRID, split=None, label_source='computable', judge=None, dataset_sha=None, created_at=None) -> tuple[Detector, dict]
  split_report = {seed, groups{train,cal,test: [str]}, counts{split: {n_groups,n_responses,n_tokens,positive_rate}}, c_selected, cv_auroc, dropped_features{name: motivo}, response_calibrator, crc[{alpha,tau,n_responses,feasible}], detector_id}

## vigia.labels
IGNORE_LABEL = -100; ARTICLES_ES; MONTHS_ES; JUDGE_SCHEMA; JUDGE_SYSTEM_PROMPT; JUDGE_PROMPT_SHA256; REJECTION_REASONS
normalize_es(text, *, strip_accents=True) -> str
exact_match(prediction, gold: str | Sequence[str], *, strip_accents=True) -> bool
token_f1(prediction, gold: str | Sequence[str], *, strip_accents=True) -> float
parse_number_es(token) -> float | None ; extract_numbers(text) -> list[float] ; numeric_match(prediction, gold, *, abs_tol=1e-9) -> bool
parse_dates_es(text) -> list[date] ; date_match(prediction, gold: date | str) -> bool
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
  model_info(model, model_id: str) -> contracts.ModelInfo
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
  GenerationStream(Iterator[TokenEvent]): .finish_reason ('stop'|'length'), .text, .prompt_tokens, .completion_tokens, .timings {prefill_s, decode_s, total_s}, .close()
  IncrementalDecoder(tokenizer): .push(token_id) -> str, .flush() -> str, .pending

vigia.harmony:
  HarmonyParser().feed(text: str, is_special: bool) -> str   # canal: final|analysis|commentary|header|control|none; atributos channel, role, ended_with
  is_scorable(event) -> bool   # channel == 'final' and not is_special
  SPECIAL_TOKENS, STOP_TOKENS, SCORED_CHANNEL='final', HEADER='header', CONTROL='control', NO_CHANNEL='none'
```
