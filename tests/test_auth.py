"""Tests de vigia_ee.auth: contraseñas, API keys, sesiones, CSRF, RBAC y dependencias FastAPI."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Annotated

import pytest
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.testclient import TestClient

from vigia_ee import auth
from vigia_ee.auth import (
    AuthState,
    LoginFailed,
    PepperRing,
    Principal,
    clear_session_cookie,
    install_auth,
    login,
    logout,
    public_route,
    require_api_key,
    require_session,
    require_session_or_key,
    set_session_cookie,
)
from vigia_ee.db import Database, create_org
from vigia_ee.security import SESSION_COOKIE, RateLimiter, RateLimitExceeded, install_security

N = 2**14  # scrypt barato para tests; el default de producción se prueba aparte
PEPPERS = PepperRing.parse("k1:" + "p" * 40)
PW = "una frase larga y propia para entrar"

SelfP = Annotated[Principal, Depends(require_session("account:self"))]
ReadP = Annotated[Principal, Depends(require_session("traces:read"))]
PolicyP = Annotated[Principal, Depends(require_session("policy:write"))]
TraceP = Annotated[Principal, Depends(require_session_or_key("traces:read", "read"))]
IngestP = Annotated[Principal, Depends(require_api_key("ingest"))]


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 10, 4, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "auth.db")
    yield database
    database.close()


@pytest.fixture
def state(db, clock):
    return AuthState(db=db, peppers=PEPPERS, scrypt_n=N, clock=clock, login_ip_limiter=None)


@pytest.fixture
def orgs(db):
    return create_org(db, "A"), create_org(db, "B")


def make_app(state: AuthState) -> FastAPI:
    app = FastAPI(openapi_url=None)
    install_security(app)
    install_auth(app, state)

    @app.post("/login", dependencies=[Depends(public_route())])
    async def do_login(request: Request, response: Response):
        form = await request.form()
        try:
            res = await run_in_threadpool(
                login,
                state,
                str(form.get("email")),
                str(form.get("password")),
                ip=request.client.host,
                previous_session_id=request.cookies.get(SESSION_COOKIE),
            )
        except LoginFailed as exc:
            raise HTTPException(401, str(exc)) from None
        set_session_cookie(response, res.session_id)
        return {"csrf": res.csrf_token}

    @app.post("/logout")
    async def do_logout(request: Request, response: Response, p: SelfP):
        logout(state, request.cookies.get(SESSION_COOKIE))
        clear_session_cookie(response)
        return {"ok": True}

    @app.get("/me")
    def me(request: Request, p: ReadP):
        return {"role": p.role, "org": p.org_id, "csrf": request.state.csrf_token}

    @app.post("/policy")
    def policy(p: PolicyP):
        return {"ok": True}

    @app.get("/api/traces/{trace_id}")
    def get_trace(trace_id: str, p: TraceP):
        trace = state.repo(p.org_id).get_trace(trace_id)
        if trace is None:
            raise HTTPException(404, "Traza inexistente.")
        return {"id": trace["id"]}

    @app.post("/api/ingest")
    def ingest(p: IngestP):
        return {"org": p.org_id, "key": p.key_id}

    return app


def client_for(state):
    return TestClient(make_app(state), base_url="https://testserver")


def add_user(state, org, email="ana@example.com", role="owner", password=PW):
    return auth.register_user(state, org, email, password, role)


def do_login(client, email="ana@example.com", password=PW):
    return client.post("/login", data={"email": email, "password": password})


# --- contraseñas ----------------------------------------------------------------------------------


def test_password_hash_default_params_format_and_verify():
    stored = auth.hash_password(PW)  # N=2**17, r=8, p=1 (único hash caro del archivo)
    algo, n, r, p, salt, dk = stored.split("$")
    assert (algo, n, r, p) == ("scrypt", str(2**17), "8", "1")
    assert len(auth.base64.b64decode(salt)) == 16 and len(auth.base64.b64decode(dk)) == 32
    assert auth.verify_password(PW, stored)
    assert not auth.verify_password(PW + "x", stored)
    assert not auth.needs_rehash(stored)


def test_password_verify_rejects_malformed_and_detects_rehash():
    stored = auth.hash_password(PW, n=N)
    assert auth.needs_rehash(stored)
    assert not auth.needs_rehash(stored, n=N)
    for bad in (
        "",
        "scrypt$x",
        "bcrypt$16384$8$1$AA==$AA==",
        "scrypt$1000$8$1$AA==$AA==",
        "scrypt$1073741824$8$1$AA==$AA==",
    ):
        assert not auth.verify_password(PW, bad)
    assert auth.verify_password("ﬁ" + PW, auth.hash_password("fi" + PW, n=N))  # normalización NFKC


def test_at_p04_password_policy_nist():
    """AT-P04: < 15 caracteres se rechaza; 64 y 128 se aceptan; sin reglas de composición."""
    with pytest.raises(auth.PasswordPolicyError):
        auth.check_password_policy("corta de 14 ch")
    auth.check_password_policy("solo minusculas sin numeros")
    auth.check_password_policy("tiene 15 chars!")
    auth.check_password_policy(("frase larga de prueba " * 3)[:64])
    auth.check_password_policy(("palabras sueltas al azar " * 6)[:128])
    with pytest.raises(auth.PasswordPolicyError):
        auth.check_password_policy("y" * 129)
    for common in ("password123456789", "123456789012345", "Contraseña2024!!!", "aaaaaaaaaaaaaaaaaaaa"):
        with pytest.raises(auth.PasswordPolicyError):
            auth.check_password_policy(common)
    with pytest.raises(auth.PasswordPolicyError):
        auth.check_password_policy("anaperez es mi nombre largo", email="anaperez@example.com")


# --- API keys -------------------------------------------------------------------------------------


def test_api_key_format_and_checksum_detected_without_db():
    token, kid, secret = auth.generate_api_key()
    assert token.startswith(f"vg_live_{kid}_{secret}") and len(token) == len("vg_live_") + 16 + 1 + 40 + 6
    assert auth.parse_api_key(token) == (kid, secret)
    flipped = token[:-1] + ("A" if token[-1] != "A" else "B")
    with pytest.raises(auth.InvalidApiKey):
        auth.parse_api_key(flipped)
    with pytest.raises(auth.InvalidApiKey):
        auth.parse_api_key("vg_live_corta")


def test_at_p03_api_key_shown_once_and_stored_as_hmac(state, orgs):
    """AT-P03: la key solo sale al crearla; la base guarda HMAC y nunca el secreto."""
    token, kid = auth.create_api_key(state, orgs[0], "ingesta", ["ingest"])
    _, secret = auth.parse_api_key(token)
    row = state.db.query_one("SELECT * FROM api_keys WHERE id = ?", (kid,))
    assert row["hmac"] == auth.key_hmac(PEPPERS.peppers["k1"], kid, secret)
    assert all(secret not in str(v) and token != v for v in row.values())
    assert "hmac" not in state.repo(orgs[0]).get_api_key(kid)
    p = auth.verify_api_key(state, token)
    assert p == Principal(org_id=orgs[0], key_id=kid, scopes=frozenset({"ingest"}))


def test_at_p03_revoked_or_expired_key_returns_401(state, orgs, clock):
    """AT-P03: key revocada o vencida → 401 en el request siguiente."""
    client = client_for(state)
    token, kid = auth.create_api_key(state, orgs[0], "k", ["ingest"])
    h = {"Authorization": f"Bearer {token}"}
    assert client.post("/api/ingest", headers=h).status_code == 200
    assert auth.revoke_api_key(state, orgs[0], kid)
    r = client.post("/api/ingest", headers=h)
    assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"

    token2, _ = auth.create_api_key(
        state, orgs[0], "k2", ["ingest"], expires_at=clock.now + timedelta(hours=1)
    )
    h2 = {"Authorization": f"Bearer {token2}"}
    assert client.post("/api/ingest", headers=h2).status_code == 200
    clock.now += timedelta(hours=1)
    assert client.post("/api/ingest", headers=h2).status_code == 401


def test_at_p03_key_in_query_string_is_rejected(state, orgs):
    """AT-P03: una key en la query string → 400, aunque sea válida."""
    client = client_for(state)
    token, _ = auth.create_api_key(state, orgs[0], "k", ["ingest"])
    assert client.post(f"/api/ingest?api_key={token}").status_code == 400
    assert (
        client.post(f"/api/ingest?x={token}", headers={"Authorization": f"Bearer {token}"}).status_code == 400
    )


def test_api_key_scopes_wrong_pepper_and_last_used(state, orgs, clock):
    client = client_for(state)
    token, kid = auth.create_api_key(state, orgs[0], "lectura", ["read"])
    assert client.post("/api/ingest", headers={"Authorization": f"Bearer {token}"}).status_code == 403
    assert state.repo(orgs[0]).get_api_key(kid)["last_used_at"] is not None
    admin, _ = auth.create_api_key(state, orgs[0], "admin", ["admin"])
    assert client.post("/api/ingest", headers={"Authorization": f"Bearer {admin}"}).status_code == 200
    other = AuthState(db=state.db, peppers=PepperRing.parse("k1:" + "q" * 40), scrypt_n=N, clock=clock)
    assert auth.verify_api_key(other, token) is None
    assert client.post("/api/ingest", headers={"Authorization": "Basic abc"}).status_code == 401
    assert client.post("/api/ingest").status_code == 401
    with pytest.raises(ValueError):
        auth.create_api_key(state, orgs[0], "k", ["superpoder"])


def test_pepper_rotation_keeps_old_keys_valid(state, orgs, clock):
    old_token, _ = auth.create_api_key(state, orgs[0], "vieja", ["read"])
    rotated = AuthState(
        db=state.db, peppers=PepperRing.parse(f"k1:{'p' * 40},k2:{'n' * 40}"), scrypt_n=N, clock=clock
    )
    assert rotated.peppers.current_kid == "k2"
    new_token, new_kid = auth.create_api_key(rotated, orgs[0], "nueva", ["read"])
    assert (
        state.db.query_one("SELECT pepper_kid FROM api_keys WHERE id = ?", (new_kid,))["pepper_kid"] == "k2"
    )
    assert auth.verify_api_key(rotated, old_token) is not None
    assert auth.verify_api_key(rotated, new_token) is not None


def test_pepper_ring_parsing_and_env(tmp_path):
    for bad in ("", "sin-separador", "k1:corto", f"k1:{'a' * 40},k1:{'b' * 40}", f"k 1:{'a' * 40}"):
        with pytest.raises(auth.PepperConfigError):
            PepperRing.parse(bad)
    f = tmp_path / "peppers"
    f.write_text(f"a1:{'x' * 40}\nb2:{'y' * 40}\n", encoding="utf-8")
    ring = PepperRing.from_env({"VIGIA_PEPPERS_FILE": str(f), "VIGIA_PEPPERS": f"z:{'z' * 40}"})
    assert ring.current_kid == "b2" and set(ring.peppers) == {"a1", "b2"}
    assert PepperRing.from_env({"VIGIA_PEPPERS": f"z:{'z' * 40}"}).current_kid == "z"
    with pytest.raises(auth.PepperConfigError):
        PepperRing.from_env({})
    with pytest.raises(auth.PepperConfigError):
        PepperRing.from_env({"VIGIA_PEPPERS_FILE": str(tmp_path / "no-existe")})


# --- login y sesiones -----------------------------------------------------------------------------


def test_at_p04_login_sets_host_cookie_with_strict_attributes(state, orgs):
    """AT-P04: cookie __Host- con Secure, HttpOnly, SameSite=Strict, Path=/ y sin Domain."""
    add_user(state, orgs[0])
    r = do_login(client_for(state))
    assert r.status_code == 200
    cookie = r.headers["set-cookie"]
    attrs = [a.strip().lower() for a in cookie.split(";")]
    assert cookie.startswith(f"{SESSION_COOKIE}=")
    assert "secure" in attrs and "httponly" in attrs and "samesite=strict" in attrs and "path=/" in attrs
    assert not any(a.startswith("domain") for a in attrs)
    raw = cookie.split(";")[0].split("=", 1)[1]
    assert len(raw) == 43
    stored = state.db.query("SELECT id_hash FROM sessions")
    assert stored == [{"id_hash": auth._hash_session_id(raw)}]  # la base nunca guarda el id crudo


def test_at_p04_session_is_regenerated_on_login(state, orgs):
    """AT-P04: el login emite un id nuevo e invalida el anterior."""
    add_user(state, orgs[0])
    client = client_for(state)
    do_login(client)
    first = client.cookies.get(SESSION_COOKIE)
    do_login(client)
    second = client.cookies.get(SESSION_COOKIE)
    assert first != second
    assert state.sessions.resolve(first) is None
    assert state.sessions.resolve(second) is not None
    assert len(state.db.query("SELECT * FROM sessions")) == 1


def test_at_p04_logout_invalidates_session_on_server(state, orgs):
    """AT-P04: tras logout, reusar la cookie vieja da 401."""
    add_user(state, orgs[0])
    client = client_for(state)
    csrf = do_login(client).json()["csrf"]
    raw = client.cookies.get(SESSION_COOKIE)
    assert client.get("/me").status_code == 200
    assert client.post("/logout", headers={"X-CSRF-Token": csrf}).status_code == 200
    replay = TestClient(make_app(state), base_url="https://testserver", cookies={SESSION_COOKIE: raw})
    assert replay.get("/me").status_code == 401
    assert state.db.query("SELECT * FROM sessions") == []


def test_at_p04_idle_and_absolute_timeouts(state, orgs, clock):
    """AT-P04: 30 min de inactividad expira la sesión; el tope absoluto es 8 h aun con actividad."""
    add_user(state, orgs[0])
    client = client_for(state)
    do_login(client)
    clock.now += timedelta(minutes=29)
    assert client.get("/me").status_code == 200
    clock.now += timedelta(minutes=31)
    assert client.get("/me").status_code == 401

    do_login(client)
    statuses = []
    for _ in range(25):  # actividad cada 20 min durante 8 h 20 min
        clock.now += timedelta(minutes=20)
        statuses.append(client.get("/me").status_code)
    assert statuses[:23] == [200] * 23 and statuses[-1] == 401


def test_wrong_password_and_unknown_email_fail_the_same_way(state, orgs):
    add_user(state, orgs[0])
    client = client_for(state)
    a = do_login(client, password="otra frase que no es la clave")  # noqa: S106
    b = do_login(client, email="nadie@example.com")
    assert a.status_code == b.status_code == 401 and a.json()["message"] == b.json()["message"]
    assert "set-cookie" not in a.headers


def test_account_locks_after_10_failures_for_15_minutes(state, orgs, clock):
    uid = add_user(state, orgs[0])
    for _ in range(auth.MAX_LOGIN_FAILURES):
        with pytest.raises(LoginFailed):
            login(state, "ana@example.com", "clave equivocada larguísima", ip="1.2.3.4")
    assert state.repo(orgs[0]).get_user(uid)["locked_until"] is not None
    with pytest.raises(LoginFailed):
        login(state, "ana@example.com", PW, ip="1.2.3.4")  # correcta pero bloqueada
    clock.now += timedelta(minutes=16)
    assert login(state, "ana@example.com", PW, ip="1.2.3.4").principal.user_id == uid
    actions = [r["action"] for r in state.db.query("SELECT action FROM audit_log ORDER BY seq")]
    assert actions.count("account_locked") == 1 and actions[-1] == "login"


def test_at_p06_login_blocked_by_ip_after_n_failures(state, orgs):
    """AT-P06: tras N fallos desde una IP, el login devuelve 429 con Retry-After."""
    add_user(state, orgs[0])
    state.login_ip_limiter = RateLimiter(rate=1 / 60, burst=3)
    client = client_for(state)
    bad = "clave equivocada larguísima"
    codes = [do_login(client, password=bad).status_code for _ in range(3)]
    assert codes == [401, 401, 401]
    r = do_login(client)  # aun con la clave correcta
    assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1
    with pytest.raises(RateLimitExceeded):
        login(state, "ana@example.com", PW, ip="testclient")
    assert login(state, "ana@example.com", PW, ip="10.0.0.9").principal.role == "owner"


def test_disabled_user_cannot_login_and_loses_sessions(state, orgs):
    uid = add_user(state, orgs[0])
    client = client_for(state)
    do_login(client)
    state.repo(orgs[0]).update_user(uid, disabled=True)
    assert client.get("/me").status_code == 401
    assert do_login(client).status_code == 401


def test_login_rehashes_when_params_change(state, orgs):
    uid = add_user(state, orgs[0])
    weaker = AuthState(db=state.db, peppers=PEPPERS, scrypt_n=2**15, clock=state.clock, login_ip_limiter=None)
    login(weaker, "ana@example.com", PW, ip="x")
    stored = state.db.query_one("SELECT password_hash FROM users WHERE id = ?", (uid,))["password_hash"]
    assert stored.startswith(f"scrypt${2**15}$")


# --- CSRF en rutas del panel ----------------------------------------------------------------------


def test_at_p05_panel_post_requires_csrf_token_of_its_own_session(state, orgs):
    """AT-P05: POST del panel sin token, con token de otra sesión o cross-site → 403."""
    add_user(state, orgs[0], "ana@example.com", "owner")
    add_user(state, orgs[0], "beto@example.com", "owner")
    ana, beto = client_for(state), client_for(state)
    ana_csrf = do_login(ana).json()["csrf"]
    beto_csrf = do_login(beto, "beto@example.com").json()["csrf"]
    assert ana.get("/me").json()["csrf"] == ana_csrf
    assert ana.post("/policy").status_code == 403
    assert ana.post("/policy", headers={"X-CSRF-Token": beto_csrf}).status_code == 403
    assert ana.post("/policy", data={"csrf_token": beto_csrf}).status_code == 403
    assert (
        ana.post("/policy", headers={"X-CSRF-Token": ana_csrf, "Sec-Fetch-Site": "cross-site"}).status_code
        == 403
    )
    assert ana.post("/policy", headers={"X-CSRF-Token": ana_csrf}).status_code == 200
    assert ana.post("/policy", data={"csrf_token": ana_csrf}).status_code == 200


def test_at_p05_bearer_routes_are_exempt_from_csrf(state, orgs):
    """AT-P05: una ruta con Authorization Bearer no pide token CSRF."""
    token, _ = auth.create_api_key(state, orgs[0], "k", ["ingest"])
    r = client_for(state).post(
        "/api/ingest", headers={"Authorization": f"Bearer {token}", "Sec-Fetch-Site": "none"}
    )
    assert r.status_code == 200


# --- RBAC y declaración de permisos ---------------------------------------------------------------


def test_has_permission_is_deny_by_default():
    assert auth.has_permission("owner", "settings:write")
    assert not auth.has_permission("admin", "settings:write")
    assert auth.has_permission("reviewer", "review:write")
    assert not auth.has_permission("viewer", "review:write")
    assert not auth.has_permission("owner", "permiso:inventado")
    assert not auth.has_permission("superadmin", "traces:read")
    assert not auth.has_permission(None, "traces:read")
    for perm, minimum in auth.PERMISSIONS.items():
        allowed = {r for r in auth.ROLE_RANK if auth.has_permission(r, perm)}
        assert allowed == {r for r, rank in auth.ROLE_RANK.items() if rank >= auth.ROLE_RANK[minimum]}


def test_role_assignment_rules(state, orgs):
    assert auth.can_assign_role("owner", "owner")
    assert auth.can_assign_role("admin", "reviewer")
    assert not auth.can_assign_role("admin", "owner")
    assert not auth.can_assign_role("reviewer", "viewer")
    admin = Principal(org_id=orgs[0], role="admin", user_id="x")
    with pytest.raises(PermissionError):
        auth.register_user(state, orgs[0], "nuevo@example.com", PW, "owner", actor=admin)
    with pytest.raises(PermissionError):
        auth.register_user(state, orgs[1], "nuevo@example.com", PW, "viewer", actor=admin)
    uid = auth.register_user(state, orgs[0], "nuevo@example.com", PW, "viewer", actor=admin)
    owner_id = add_user(state, orgs[0], "duena@example.com", "owner")
    with pytest.raises(PermissionError):
        auth.change_role(state, orgs[0], owner_id, "viewer", actor=admin)
    assert auth.change_role(state, orgs[0], uid, "reviewer", actor=admin)
    assert not auth.change_role(state, orgs[1], uid, "admin", actor=Principal(org_id=orgs[1], role="owner"))
    with pytest.raises(ValueError):
        auth.register_user(state, orgs[0], "no-es-email", PW, "viewer")


def test_at_p02_viewer_cannot_mutate_and_role_change_kills_sessions(state, orgs):
    """AT-P02: un viewer recibe 403 al mutar; cambiar el rol invalida sus sesiones."""
    uid = add_user(state, orgs[0], "vera@example.com", "viewer")
    client = client_for(state)
    csrf = do_login(client, "vera@example.com").json()["csrf"]
    assert client.get("/me").json()["role"] == "viewer"
    assert client.post("/policy", headers={"X-CSRF-Token": csrf}).status_code == 403
    owner = Principal(org_id=orgs[0], role="owner", user_id="o")
    assert auth.change_role(state, orgs[0], uid, "admin", actor=owner)
    assert client.get("/me").status_code == 401  # debe volver a entrar con el rol nuevo
    csrf = do_login(client, "vera@example.com").json()["csrf"]
    assert client.post("/policy", headers={"X-CSRF-Token": csrf}).status_code == 200


def test_at_p02_every_route_declares_a_permission(state):
    """AT-P02: introspección del router: toda ruta declara permiso y ninguna mutación es de viewer."""
    app = make_app(state)
    assert auth.undeclared_routes(app) == []
    assert auth.viewer_mutation_violations(app) == []
    assert auth.route_permission(next(r for r in app.routes if r.path == "/policy")) == auth.PermissionSpec(
        "session", permission="policy:write"
    )

    @app.get("/olvidada")
    def forgotten():
        return {}

    @app.post("/peligrosa")
    def dangerous(p: ReadP):
        return {}

    assert auth.undeclared_routes(app) == ["GET /olvidada"]
    assert auth.viewer_mutation_violations(app) == ["POST /peligrosa"]
    docs_app = FastAPI()
    assert "GET,HEAD /docs" in auth.undeclared_routes(docs_app)
    assert (
        auth.undeclared_routes(
            docs_app, exempt_paths=["/docs", "/docs/oauth2-redirect", "/redoc", "/openapi.json"]
        )
        == []
    )


def test_unknown_permission_or_scope_fails_at_definition():
    with pytest.raises(ValueError):
        require_session("traces:borrar_todo")
    with pytest.raises(ValueError):
        require_api_key("root")


# --- aislamiento y fail-closed --------------------------------------------------------------------


def test_at_p01_key_of_other_tenant_gets_404(state, orgs):
    """AT-P01: con key o sesión del tenant B, la traza del tenant A devuelve 404 (no 403)."""
    trace_a = state.repo(orgs[0]).insert_trace(model_id="m", response_text="hola", action="pass")
    token_a, _ = auth.create_api_key(state, orgs[0], "a", ["read"])
    token_b, _ = auth.create_api_key(state, orgs[1], "b", ["read"])
    client = client_for(state)
    assert (
        client.get(f"/api/traces/{trace_a}", headers={"Authorization": f"Bearer {token_a}"}).status_code
        == 200
    )
    r = client.get(f"/api/traces/{trace_a}", headers={"Authorization": f"Bearer {token_b}"})
    assert r.status_code == 404
    add_user(state, orgs[1], "beto@example.com", "owner")
    do_login(client, "beto@example.com")
    assert client.get(f"/api/traces/{trace_a}").status_code == 404


def test_fail_closed_when_auth_missing_or_backend_errors(state, orgs):
    app = FastAPI()
    install_security(app)

    @app.get("/x")
    def x(p: TraceP):
        return {}

    assert TestClient(app, base_url="https://testserver").get("/x").status_code == 503

    token, _ = auth.create_api_key(state, orgs[0], "k", ["read"])
    client = client_for(state)
    state.db.close()  # la base cae: nunca 200
    assert client.get("/api/traces/abc", headers={"Authorization": f"Bearer {token}"}).status_code == 503
    client.cookies.set(SESSION_COOKIE, "a" * 43)
    assert client.get("/me").status_code == 503


def test_api_key_rate_limits_per_key_and_tenant(state, orgs):
    state.key_limiter = RateLimiter(rate=0.01, burst=2)
    state.tenant_limiter = RateLimiter(rate=0.01, burst=3)
    t1, _ = auth.create_api_key(state, orgs[0], "k1", ["ingest"])
    t2, _ = auth.create_api_key(state, orgs[0], "k2", ["ingest"])
    client = client_for(state)
    post = lambda t: client.post("/api/ingest", headers={"Authorization": f"Bearer {t}"})  # noqa: E731
    assert [post(t1).status_code for _ in range(3)] == [200, 200, 429]
    assert post(t2).status_code == 200  # tercera del tenant
    r = post(t2)
    assert r.status_code == 429 and "retry-after" in r.headers  # el tenant agotó su cupo


def test_set_password_and_audit_events(state, orgs):
    uid = add_user(state, orgs[0])
    me = Principal(org_id=orgs[0], role="viewer", user_id=uid)
    with pytest.raises(auth.PasswordPolicyError):
        auth.set_password(state, orgs[0], uid, "corta", actor=me)
    raw = state.sessions.create(uid, orgs[0])
    assert auth.set_password(state, orgs[0], uid, "otra frase nueva y bastante larga", actor=me)
    assert state.sessions.resolve(raw) is None
    other = Principal(org_id=orgs[0], role="viewer", user_id="otro")
    with pytest.raises(PermissionError):
        auth.set_password(state, orgs[0], uid, "otra frase nueva y bastante larga", actor=other)
    actions = {r["action"] for r in state.db.query("SELECT action FROM audit_log")}
    assert {"user_create", "password_change"} <= actions
    assert all("frase" not in r["detail_json"] for r in state.db.query("SELECT detail_json FROM audit_log"))


def test_purge_expired_sessions(state, orgs, clock):
    uid = add_user(state, orgs[0])
    state.sessions.create(uid, orgs[0])
    clock.now += timedelta(hours=9)
    assert state.sessions.purge_expired() == 1
