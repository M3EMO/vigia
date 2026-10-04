-- Esquema de la plataforma Vigía (SQLite, WAL). Contrato compartido entre agentes.
-- Reglas: IDs aleatorios (secrets.token_hex / uuid4), nunca secuenciales expuestos.
-- Toda tabla con datos de cliente lleva org_id y toda consulta filtra por org_id (TenantScopedRepository).
-- Fechas en UTC, ISO-8601 (texto). Booleans como INTEGER 0/1.

CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS orgs (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  created_at TEXT NOT NULL,
  store_text TEXT NOT NULL DEFAULT 'redacted' CHECK (store_text IN ('none','redacted','full')),
  retention_days INTEGER NOT NULL DEFAULT 30 CHECK (retention_days BETWEEN 1 AND 3650),
  policy_json TEXT NOT NULL DEFAULT '{}',
  active_detector_id TEXT,
  external_judge_allowed INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS users (
  id TEXT PRIMARY KEY,
  org_id TEXT NOT NULL REFERENCES orgs(id) ON DELETE CASCADE,
  email TEXT NOT NULL UNIQUE COLLATE NOCASE,
  password_hash TEXT NOT NULL,
  role TEXT NOT NULL CHECK (role IN ('owner','admin','reviewer','viewer')),
  created_at TEXT NOT NULL,
  disabled INTEGER NOT NULL DEFAULT 0,
  failed_logins INTEGER NOT NULL DEFAULT 0,
  locked_until TEXT
);
CREATE INDEX IF NOT EXISTS ix_users_org ON users(org_id);

CREATE TABLE IF NOT EXISTS sessions (
  id_hash TEXT PRIMARY KEY,            -- sha256 del id de sesión; el id crudo solo vive en la cookie
  user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  org_id TEXT NOT NULL REFERENCES orgs(id) ON DELETE CASCADE,
  csrf_secret TEXT NOT NULL,
  created_at TEXT NOT NULL,
  last_seen_at TEXT NOT NULL,
  expires_at TEXT NOT NULL             -- tope absoluto
);

CREATE TABLE IF NOT EXISTS api_keys (
  id TEXT PRIMARY KEY,                 -- key_id público (parte del token vg_live_<id>_...)
  org_id TEXT NOT NULL REFERENCES orgs(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  hmac TEXT NOT NULL,                  -- HMAC-SHA256(pepper[pepper_kid], id || secret), hex
  pepper_kid TEXT NOT NULL,
  scopes TEXT NOT NULL,                -- csv: ingest,read,admin
  created_by TEXT,
  created_at TEXT NOT NULL,
  expires_at TEXT,
  last_used_at TEXT,
  revoked_at TEXT
);
CREATE INDEX IF NOT EXISTS ix_keys_org ON api_keys(org_id);

CREATE TABLE IF NOT EXISTS detectors (
  id TEXT NOT NULL,                    -- sha256 corto (16 hex) del artefacto; único por tenant
  org_id TEXT NOT NULL REFERENCES orgs(id) ON DELETE CASCADE,
  created_at TEXT NOT NULL,
  created_by TEXT,
  status TEXT NOT NULL CHECK (status IN ('candidate','active','retired')),
  artifact_json TEXT NOT NULL,
  report_json TEXT NOT NULL DEFAULT '{}',
  dataset_id TEXT,
  PRIMARY KEY (org_id, id)             -- un PK global filtraría que otro tenant tiene el mismo artefacto
);

CREATE TABLE IF NOT EXISTS datasets (
  id TEXT PRIMARY KEY,
  org_id TEXT NOT NULL REFERENCES orgs(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  created_at TEXT NOT NULL,
  n_items INTEGER NOT NULL,
  sha256 TEXT NOT NULL,
  content BLOB NOT NULL                -- JSONL tal como se subió (validado)
);

CREATE TABLE IF NOT EXISTS traces (
  id TEXT PRIMARY KEY,
  org_id TEXT NOT NULL REFERENCES orgs(id) ON DELETE CASCADE,
  created_at TEXT NOT NULL,
  api_key_id TEXT,
  source TEXT NOT NULL DEFAULT 'api' CHECK (source IN ('api','playground')),
  model_id TEXT NOT NULL,
  mode TEXT NOT NULL DEFAULT 'full',
  prompt_text TEXT,                    -- según store_text: NULL | redactado | completo
  response_text TEXT,
  text_sha256 TEXT NOT NULL,
  n_tokens INTEGER NOT NULL,
  response_p_error REAL,
  action TEXT NOT NULL,
  calibration_id TEXT,
  alpha REAL,
  tau REAL,
  latency_ms REAL,
  overhead_ms REAL,
  degraded INTEGER NOT NULL DEFAULT 0,
  finish_reason TEXT,
  usage_json TEXT                      -- {"layers":[...], "counts":[[...]]} uso capa x experto
);
CREATE INDEX IF NOT EXISTS ix_traces_org_time ON traces(org_id, created_at);

CREATE TABLE IF NOT EXISTS trace_tokens (
  trace_id TEXT NOT NULL REFERENCES traces(id) ON DELETE CASCADE,
  org_id TEXT NOT NULL,
  idx INTEGER NOT NULL,
  text TEXT,                           -- NULL si store_text = none
  start INTEGER NOT NULL,
  end INTEGER NOT NULL,
  p_error REAL,
  flag INTEGER NOT NULL DEFAULT 0,
  out_entropy REAL,
  r_ent_mean REAL,
  PRIMARY KEY (trace_id, idx)
);

CREATE TABLE IF NOT EXISTS spans (
  trace_id TEXT NOT NULL REFERENCES traces(id) ON DELETE CASCADE,
  org_id TEXT NOT NULL,
  idx INTEGER NOT NULL,
  start INTEGER NOT NULL,
  end INTEGER NOT NULL,
  score REAL NOT NULL,
  PRIMARY KEY (trace_id, idx)
);

CREATE TABLE IF NOT EXISTS review_items (
  id TEXT PRIMARY KEY,
  org_id TEXT NOT NULL REFERENCES orgs(id) ON DELETE CASCADE,
  trace_id TEXT NOT NULL REFERENCES traces(id) ON DELETE CASCADE,
  span_idx INTEGER,                    -- NULL = revisar la respuesta completa
  state TEXT NOT NULL CHECK (state IN ('pending','reserved','needs_other','completed')),
  reserved_by TEXT,
  reserved_until TEXT,
  min_reviewers INTEGER NOT NULL DEFAULT 1,
  created_at TEXT NOT NULL,
  completed_at TEXT
);
CREATE INDEX IF NOT EXISTS ix_review_org_state ON review_items(org_id, state);

CREATE TABLE IF NOT EXISTS review_decisions (
  id TEXT PRIMARY KEY,
  item_id TEXT NOT NULL REFERENCES review_items(id) ON DELETE CASCADE,
  org_id TEXT NOT NULL,
  user_id TEXT NOT NULL,
  label TEXT NOT NULL CHECK (label IN ('correct','error','uncertain')),
  note TEXT,
  created_at TEXT NOT NULL,
  UNIQUE (item_id, user_id)
);

CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY,
  org_id TEXT,
  kind TEXT NOT NULL,                  -- calibrate | webhook | purge | rag
  payload_json TEXT NOT NULL,
  state TEXT NOT NULL CHECK (state IN ('queued','running','done','failed')),
  attempts INTEGER NOT NULL DEFAULT 0,
  max_attempts INTEGER NOT NULL DEFAULT 5,
  run_after TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  error TEXT,
  result_json TEXT
);
CREATE INDEX IF NOT EXISTS ix_jobs_state ON jobs(state, run_after);

CREATE TABLE IF NOT EXISTS webhooks (
  id TEXT PRIMARY KEY,
  org_id TEXT NOT NULL REFERENCES orgs(id) ON DELETE CASCADE,
  url TEXT NOT NULL,
  secret TEXT NOT NULL,                -- whsec_<base64>
  events TEXT NOT NULL,                -- csv: trace.flagged,review.completed,calibration.activated
  created_at TEXT NOT NULL,
  disabled INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS audit_log (
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  org_id TEXT,
  ts TEXT NOT NULL,
  actor_user_id TEXT,
  actor_key_id TEXT,
  action TEXT NOT NULL,                -- login, login_failed, role_change, key_create, key_revoke, policy_change, ...
  target TEXT,
  detail_json TEXT NOT NULL DEFAULT '{}',
  prev_hash TEXT NOT NULL,
  hash TEXT NOT NULL                   -- sha256(prev_hash || canonical_json(fila sin hash))
);
CREATE INDEX IF NOT EXISTS ix_audit_org ON audit_log(org_id, seq);
-- La migración 2 (vigia_ee/db.py) agrega org_seq, la posición del evento dentro de su organización.
-- seq es global e interno: nunca se muestra al tenant porque deja inferir la actividad de otras orgs.

-- users.email es único global porque el login no pide organización. Un alta con un email ajeno
-- devuelve un error genérico; el canal lateral queda documentado en docs/spec/seguridad.md.

CREATE INDEX IF NOT EXISTS ix_trace_tokens_org ON trace_tokens(org_id);
CREATE INDEX IF NOT EXISTS ix_spans_org ON spans(org_id);
CREATE INDEX IF NOT EXISTS ix_sessions_user ON sessions(user_id);

CREATE TRIGGER IF NOT EXISTS audit_log_no_update BEFORE UPDATE ON audit_log
BEGIN SELECT RAISE(ABORT, 'audit_log es append-only'); END;
CREATE TRIGGER IF NOT EXISTS audit_log_no_delete BEFORE DELETE ON audit_log
BEGIN SELECT RAISE(ABORT, 'audit_log es append-only'); END;
