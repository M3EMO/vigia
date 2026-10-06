"""Tests de vigia_ee.auth: contraseñas, API keys, sesiones, CSRF, RBAC y dependencias FastAPI."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Annotated
from urllib.parse import urlencode

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
    admin_id = add_user(state, orgs[0], "ada@example.com", "admin")
    admin = Principal(org_id=orgs[0], role="admin", user_id=admin_id)
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
    owner_id = add_user(state, orgs[0], "olga@example.com", "owner")
    owner = Principal(org_id=orgs[0], role="owner", user_id=owner_id)
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


# --- credenciales revocadas mientras el body llega lento ------------------------------------------


async def _slow_post(app, path, headers, body, *, between=None, stall=False):
    """POST ASGI en dos tramos: `between()` corre antes del segundo; con `stall`, nunca llega."""
    chunks = [body[:5], body[5:]]
    sent: list[dict] = []
    calls = 0

    async def receive():
        nonlocal calls
        i, calls = calls, calls + 1
        if i == 0:
            return {"type": "http.request", "body": chunks[0], "more_body": True}
        if i == 1 and not stall:
            if between is not None:
                between()
            return {"type": "http.request", "body": chunks[1], "more_body": False}
        await asyncio.Event().wait()  # el cliente no manda nada más

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "https",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()]
        + [(b"host", b"testserver"), (b"content-length", str(len(body)).encode())],
        "client": ("127.0.0.1", 50000),
        "server": ("testserver", 443),
    }
    await app(scope, receive, send)
    return next(m["status"] for m in sent if m["type"] == "http.response.start")


def test_disabled_user_cannot_finish_a_slow_post_started_before(state, orgs):
    """Un admin deshabilitado o degradado completa un POST que empezó antes del cambio y crea una API key
    admin que sigue funcionando: la sesión se vuelve a resolver con el body completo."""
    uid = add_user(state, orgs[0], "ada@example.com", "admin")
    client = client_for(state)
    csrf = do_login(client, "ada@example.com").json()["csrf"]
    cookie = f"{SESSION_COOKIE}={client.cookies.get(SESSION_COOKIE)}"
    headers = {"Cookie": cookie, "Content-Type": "application/x-www-form-urlencoded"}
    body = urlencode({"csrf_token": csrf, "scopes": "admin"}).encode()
    app = make_app(state)
    assert asyncio.run(_slow_post(app, "/policy", headers, body)) == 200

    def disable():
        state.repo(orgs[0]).update_user(uid, disabled=True)

    assert asyncio.run(_slow_post(app, "/policy", headers, body, between=disable)) == 401


def test_revoked_key_cannot_finish_a_slow_request_started_before(state, orgs):
    """Variante API: una key revocada mientras su request está a medio body recibe 401."""
    token, kid = auth.create_api_key(state, orgs[0], "efímera", ["ingest"])
    app = make_app(state)

    @app.post("/api/echo")
    async def echo(request: Request, p: IngestP):
        return {"n": len(await request.json())}

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    body = b'{"messages": ["hola", "chau"]}'
    assert asyncio.run(_slow_post(app, "/api/echo", headers, body)) == 200

    def revoke():
        auth.revoke_api_key(state, orgs[0], kid)

    assert asyncio.run(_slow_post(app, "/api/echo", headers, body, between=revoke)) == 401


def test_body_that_never_finishes_is_cut_with_408(state, orgs, monkeypatch):
    """Un body que no termina de llegar no retiene la conexión: 408 al vencer el tope."""
    monkeypatch.setattr(auth, "BODY_READ_TIMEOUT", timedelta(milliseconds=50))
    token, _ = auth.create_api_key(state, orgs[0], "k", ["ingest"])
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    status = asyncio.run(_slow_post(make_app(state), "/api/ingest", headers, b'{"a": 1}', stall=True))
    assert status == 408


# --- base bloqueada: la autenticación no congela el event loop ------------------------------------


@pytest.mark.parametrize("kind", ["api_key", "session"])
def test_locked_db_auth_does_not_freeze_event_loop(tmp_path, clock, kind):
    """Las dependencias async de autenticación consultan SQLite en el event loop: con la base
    bloqueada, un solo request congela todo el proceso, /livez incluido.

    Otra conexión toma `BEGIN EXCLUSIVE`. Un request autenticado queda esperando el `busy_timeout`
    en su hilo, y `/livez` del mismo proceso responde enseguida (arquitectura.md §5).
    """
    import sqlite3
    import time

    import httpx

    busy = 2.0
    database = Database(tmp_path / "locked.db", busy_timeout_ms=int(busy * 1000))
    st = AuthState(db=database, peppers=PEPPERS, scrypt_n=N, clock=clock, login_ip_limiter=None)
    org = create_org(database, "A")
    app = make_app(st)

    @app.get("/livez", dependencies=[Depends(public_route())])
    async def livez():
        return {"ok": True}

    if kind == "api_key":
        token, _ = auth.create_api_key(st, org, "k", ["ingest"])  # nunca usada: el uso escribe
        method, path, headers = "POST", "/api/ingest", {"Authorization": f"Bearer {token}"}
        cookies = {}
    else:
        add_user(st, org)
        tc = client_for(st)
        do_login(tc)
        cookies = {SESSION_COOKIE: tc.cookies.get(SESSION_COOKIE)}
        clock.now += timedelta(minutes=5)  # vence el intervalo de touch: resolver escribe
        method, path, headers = "GET", "/me", {}

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        base = "https://testserver"
        async with httpx.AsyncClient(transport=transport, base_url=base, cookies=cookies) as c:
            t0 = time.perf_counter()
            authed = asyncio.create_task(c.request(method, path, headers=headers))
            await asyncio.sleep(0.2)  # con el loop bloqueado, este sleep dura todo el busy_timeout
            live = await c.get("/livez")
            livez_s = time.perf_counter() - t0 - 0.2
            resp = await authed
        return live.status_code, livez_s, resp.status_code

    lock = sqlite3.connect(tmp_path / "locked.db", isolation_level=None)
    try:
        lock.execute("BEGIN EXCLUSIVE")
        live_status, livez_s, auth_status = asyncio.run(scenario())
    finally:
        lock.execute("ROLLBACK")
        lock.close()
        database.close()
    assert live_status == 200
    assert livez_s < busy / 4, f"/livez tardó {livez_s:.2f} s con la base bloqueada"
    assert auth_status == 503  # fail-closed en su propio hilo


# --- acciones en cola con un principal viejo (TOCTOU) ---------------------------------------------

STALE_TITLE = (
    "Bajo carga, un admin degradado a viewer sigue creando API keys admin válidas después de que la "
    "degradación quedó confirmada"
)


def _audit_actions(db, org):
    rows = db.query("SELECT action FROM audit_log WHERE org_id = ? ORDER BY seq", (org,))
    return [r["action"] for r in rows]


def _session_principal_for(state, email):
    """Principal como lo arma `_session_principal`: con el hash de una sesión real."""
    result = auth.login(state, email, PW, ip="203.0.113.9")
    info = state.sessions.resolve(result.session_id)
    return Principal(org_id=info.org_id, role=info.role, user_id=info.user_id, session_id_hash=info.id_hash)


def test_a_demoted_admin_with_a_stale_principal_cannot_create_or_revoke_keys_nor_invite(state, orgs):
    """Regresión de «Bajo carga, un admin degradado a viewer sigue creando API keys admin válidas después
    de que la degradación quedó confirmada». El principal se resolvió como admin; el owner lo degrada
    antes de que corra el handler: la mutación debe fallar dentro de su transacción."""
    org = orgs[0]
    owner_id = add_user(state, org, "olga@example.com", "owner")
    admin_id = add_user(state, org, "ada@example.com", "admin")
    owner = Principal(org_id=org, role="owner", user_id=owner_id)
    stale_admin = _session_principal_for(state, "ada@example.com")
    token, kid = auth.create_api_key(state, org, "previa", ["ingest"], actor=stale_admin)
    assert auth.change_role(state, org, admin_id, "viewer", actor=owner)

    keys_before = {k["id"] for k in state.repo(org).list_api_keys()}
    with pytest.raises(PermissionError):
        auth.create_api_key(state, org, "tardía", ["admin"], actor=stale_admin)
    with pytest.raises(PermissionError):
        auth.revoke_api_key(state, org, kid, actor=stale_admin)
    with pytest.raises(PermissionError):
        auth.register_user(state, org, "nuevo@example.com", PW, "admin", actor=stale_admin)
    with pytest.raises(PermissionError):
        auth.change_role(state, org, owner_id, "viewer", actor=stale_admin)
    assert {k["id"] for k in state.repo(org).list_api_keys()} == keys_before
    assert auth.verify_api_key(state, token) is not None  # la key previa sigue vigente (no se revocó)
    assert auth.lookup_user_by_email(state.db, "nuevo@example.com") is None
    actions = _audit_actions(state.db, org)
    assert actions[-1] == "role_change", actions


def test_require_current_rejects_a_closed_session_a_disabled_account_and_a_revoked_key(state, orgs):
    """Mismo defecto: además del rol, una sesión cerrada, una cuenta deshabilitada o una key revocada
    tampoco confirman la acción en cola."""
    org = orgs[0]
    admin_id = add_user(state, org, "ada@example.com", "admin")
    stale = _session_principal_for(state, "ada@example.com")
    assert auth.require_current(state, stale, "keys:write").role == "admin"
    state.repo(org).delete_user_sessions(admin_id)
    with pytest.raises(auth.StaleActor):
        auth.require_current(state, stale, "keys:write")
    with pytest.raises(PermissionError):  # el cambio de la propia contraseña tampoco con la sesión cerrada
        auth.set_password(state, org, admin_id, PW + " nueva", actor=stale)
    fresh = _session_principal_for(state, "ada@example.com")
    state.repo(org).update_user(admin_id, disabled=True)
    with pytest.raises(auth.StaleActor):
        auth.require_current(state, fresh, None)
    _token, kid = auth.create_api_key(state, org, "k", ["admin"])
    key_actor = Principal(org_id=org, key_id=kid, scopes=frozenset({"admin"}))
    assert auth.require_current(state, key_actor) is key_actor
    auth.revoke_api_key(state, org, kid)
    with pytest.raises(auth.StaleActor):
        auth.require_current(state, key_actor)
    assert auth.require_current(state, None) is None


def test_panel_key_creation_fails_when_the_role_drops_between_session_resolution_and_handler(
    tmp_path, monkeypatch
):
    """Regresión de «Bajo carga, un admin degradado a viewer sigue creando API keys admin válidas después
    de que la degradación quedó confirmada». La degradación se intercala justo en el hueco: después de
    resolver la sesión (dentro de enforce_csrf) y antes del handler de POST /settings/keys."""
    import _panel_support as ps

    runtime = ps.make_runtime(tmp_path)
    try:
        org = runtime.create_org("Acme")
        owner_email = ps.add_user(runtime, org, "owner")
        admin_email = ps.add_user(runtime, org, "admin")
        app = ps.make_app(runtime)
        client = ps.login(app, admin_email)
        csrf = ps.csrf_of(client, "/settings")
        owner = ps.principal_for(runtime, owner_email)
        admin_id = ps.principal_for(runtime, admin_email).user_id
        real_enforce = auth.enforce_csrf

        async def demote_then_check(request, secret, id_hash):
            await real_enforce(request, secret, id_hash)
            await run_in_threadpool(auth.change_role, runtime.auth, org, admin_id, "viewer", actor=owner)

        monkeypatch.setattr(auth, "enforce_csrf", demote_then_check)
        resp = client.post(
            "/settings/keys",
            data={"csrf_token": csrf, "name": "tardía", "scopes": ["admin"]},
            follow_redirects=False,
        )
        assert resp.status_code in (401, 403), resp.status_code
        assert [k["name"] for k in runtime.repo(org).list_api_keys()] == []
        assert _audit_actions(runtime.db, org)[-1] == "role_change"
    finally:
        runtime.close()


def test_auth_backend_error_503_carries_retry_after(state, monkeypatch):
    """Regresión de «Con la base bloqueada, un request de la API tarda unos 57 s en devolver 503 …»: el
    503 de auth_backend_error no traía Retry-After."""

    def boom(*_a, **_k):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(auth, "verify_api_key", boom)
    monkeypatch.setattr(state.sessions, "resolve", boom)
    with pytest.raises(HTTPException) as key_err:
        auth._resolve_key(state, "vg_live_x")
    with pytest.raises(HTTPException) as session_err:
        auth._resolve_session(state, "x" * 43)
    for err in (key_err.value, session_err.value):
        assert err.status_code == 503 and err.headers["Retry-After"].isdigit()
