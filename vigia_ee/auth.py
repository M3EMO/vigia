"""Identidad de la plataforma: contraseñas, API keys, sesiones, RBAC y dependencias FastAPI.

Cómo se usa desde la app y el panel:

    state = AuthState(db=db, peppers=PepperRing.from_env())
    install_auth(app, state)

    @app.get("/v1/vigia/traces/{trace_id}")
    def get_trace(trace_id: str, p: Principal = Depends(require_session_or_key("traces:read", "read"))): ...

    @app.post("/panel/policy")
    def save_policy(p: Principal = Depends(require_session("policy:write"))): ...

    @app.get("/livez", dependencies=[Depends(public_route())])
    def livez(): ...

Cada ruta declara su permiso usando una de esas dependencias; `undeclared_routes(app)` lo verifica
por introspección (AT-P02). Las rutas con sesión validan CSRF en métodos no seguros; las rutas con
`Authorization: Bearer` no usan cookies y quedan exentas.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
import re
import secrets
import unicodedata
import zlib
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.routing import APIRoute

from vigia_ee import audit
from vigia_ee.db import (
    Database,
    TenantScopedRepository,
    lookup_api_key,
    lookup_session,
    lookup_user_by_email,
    parse_iso,
    to_iso,
    utcnow,
)
from vigia_ee.security import (
    SESSION_COOKIE,
    RateLimiter,
    RateLimitExceeded,
    csrf_token_for,
    enforce_csrf,
    enforce_rate_limit,
    log_event,
)

_log = logging.getLogger("vigia.auth")


class AuthError(Exception):
    """Error base de identidad."""


class PasswordPolicyError(AuthError, ValueError):
    pass


class PepperConfigError(AuthError):
    pass


class InvalidApiKey(AuthError):
    pass


class LoginFailed(AuthError):
    """Credenciales inválidas, cuenta bloqueada o deshabilitada. El mensaje no distingue el caso."""

    def __init__(self) -> None:
        super().__init__("Email o contraseña incorrectos.")


# --- contraseñas (scrypt, NIST SP 800-63B rev 4) --------------------------------------------------

SCRYPT_N = 2**17
SCRYPT_R = 8
SCRYPT_P = 1
SCRYPT_SALT_BYTES = 16
SCRYPT_DKLEN = 32
MIN_PASSWORD_LEN = 15
MAX_PASSWORD_LEN = 128

COMMON_PASSWORDS = frozenset(
    {
        "123456789012345",
        "1234567890123456",
        "12345678901234567890",
        "111111111111111",
        "000000000000000",
        "qwertyuiopasdfg",
        "qwertyuiopasdfghjkl",
        "qwertyuiop123456",
        "1q2w3e4r5t6y7u8i",
        "1qaz2wsx3edc4rfv",
        "zaq12wsxcde34rfv",
        "abcdefghijklmnop",
        "passwordpassword",
        "iloveyouiloveyou",
        "contraseñacontraseña",
        "correcthorsebatterystaple",
    }
)
# Palabras base: una contraseña que es solo esto más dígitos o símbolos se rechaza.
COMMON_BASES = frozenset(
    {
        "password",
        "passw",
        "contrasena",
        "contraseña",
        "clave",
        "qwerty",
        "qwertyuiop",
        "asdfgh",
        "zxcvbn",
        "admin",
        "administrator",
        "administrador",
        "letmein",
        "welcome",
        "bienvenido",
        "iloveyou",
        "teamo",
        "abc",
        "abcdef",
        "secret",
        "secreto",
        "vigia",
        "vigía",
        "changeme",
        "default",
        "test",
        "prueba",
        "usuario",
        "user",
        "root",
        "dragon",
        "monkey",
        "football",
        "futbol",
        "boca",
        "river",
        "argentina",
    }
)


def _normalize_password(password: str) -> str:
    return unicodedata.normalize("NFKC", password)


def check_password_policy(password: str, *, email: str | None = None) -> None:
    """Política NIST: 15 a 128 caracteres, sin reglas de composición, con blocklist y contexto."""
    if not isinstance(password, str):
        raise PasswordPolicyError("La contraseña debe ser texto.")
    pw = _normalize_password(password)
    if len(pw) < MIN_PASSWORD_LEN:
        raise PasswordPolicyError(f"La contraseña debe tener al menos {MIN_PASSWORD_LEN} caracteres.")
    if len(pw) > MAX_PASSWORD_LEN:
        raise PasswordPolicyError(f"La contraseña admite hasta {MAX_PASSWORD_LEN} caracteres.")
    low = pw.lower()
    letters = "".join(ch for ch in low if ch.isalpha())
    if low in COMMON_PASSWORDS or letters in COMMON_BASES or len(set(low)) < 4:
        raise PasswordPolicyError("La contraseña es demasiado común. Probá con una frase más larga.")
    if email:
        local = email.split("@", 1)[0].lower()
        if len(local) >= 4 and local in low:
            raise PasswordPolicyError("La contraseña no puede contener tu email.")


def _scrypt(password: str, salt: bytes, n: int, r: int, p: int) -> bytes:
    return hashlib.scrypt(
        _normalize_password(password).encode("utf-8"),
        salt=salt,
        n=n,
        r=r,
        p=p,
        maxmem=128 * r * (n + p + 2) + 1024 * 1024,
        dklen=SCRYPT_DKLEN,
    )


def hash_password(password: str, *, n: int = SCRYPT_N) -> str:
    """Hash `scrypt$N$r$p$salt$hash` (base64). No aplica la política: eso es `check_password_policy`."""
    salt = secrets.token_bytes(SCRYPT_SALT_BYTES)
    dk = _scrypt(password, salt, n, SCRYPT_R, SCRYPT_P)
    b64 = base64.b64encode
    return f"scrypt${n}${SCRYPT_R}${SCRYPT_P}${b64(salt).decode()}${b64(dk).decode()}"


def _parse_hash(stored: str) -> tuple[int, int, int, bytes, bytes] | None:
    try:
        algo, n_s, r_s, p_s, salt_s, dk_s = stored.split("$")
        n, r, p = int(n_s), int(r_s), int(p_s)
        if algo != "scrypt" or n < 2**10 or n > 2**20 or n & (n - 1) or not 1 <= r <= 32 or not 1 <= p <= 16:
            return None
        return n, r, p, base64.b64decode(salt_s, validate=True), base64.b64decode(dk_s, validate=True)
    except (ValueError, AttributeError):
        return None


def verify_password(password: str, stored: str) -> bool:
    """Compara en tiempo constante. Un hash mal formado devuelve False."""
    parsed = _parse_hash(stored) if isinstance(stored, str) else None
    if parsed is None or not isinstance(password, str) or len(password) > 4 * MAX_PASSWORD_LEN:
        return False
    n, r, p, salt, dk = parsed
    return hmac.compare_digest(_scrypt(password, salt, n, r, p), dk)


def needs_rehash(stored: str, *, n: int = SCRYPT_N) -> bool:
    parsed = _parse_hash(stored)
    return parsed is None or parsed[:3] != (n, SCRYPT_R, SCRYPT_P)


_DUMMY_HASHES: dict[int, str] = {}


def _dummy_hash(n: int) -> str:
    """Hash descartable para igualar el tiempo de login cuando el email no existe."""
    if n not in _DUMMY_HASHES:
        _DUMMY_HASHES[n] = hash_password(secrets.token_urlsafe(24), n=n)
    return _DUMMY_HASHES[n]


# --- peppers y API keys ---------------------------------------------------------------------------

API_KEY_PREFIX = "vg_live_"
SCOPES = ("ingest", "read", "admin")
_B62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_SECRET_LEN = 40  # ~238 bits
_API_KEY_RE = re.compile(r"^vg_live_([0-9a-f]{16})_([0-9A-Za-z]{40})([0-9A-Za-z]{6})$")
_PEPPER_KID_RE = re.compile(r"^[A-Za-z0-9-]{1,32}$")
MIN_PEPPER_LEN = 32
KEY_TOUCH_INTERVAL = timedelta(seconds=60)


@dataclass(frozen=True)
class PepperRing:
    """Peppers HMAC por `kid`. El último de la lista es el vigente para keys nuevas."""

    peppers: Mapping[str, bytes]
    current_kid: str

    @classmethod
    def parse(cls, spec: str) -> PepperRing:
        """Formato `kid1:secreto,kid2:secreto` (comas o saltos de línea)."""
        peppers: dict[str, bytes] = {}
        current = None
        for raw in re.split(r"[,\n]", spec or ""):
            entry = raw.strip()
            if not entry:
                continue
            kid, sep, secret = entry.partition(":")
            if not sep or not _PEPPER_KID_RE.match(kid):
                raise PepperConfigError("Pepper mal formado: se espera kid:secreto.")
            if len(secret) < MIN_PEPPER_LEN:
                raise PepperConfigError(f"El pepper {kid} tiene menos de {MIN_PEPPER_LEN} caracteres.")
            if kid in peppers:
                raise PepperConfigError(f"El kid {kid} está repetido.")
            peppers[kid] = secret.encode()
            current = kid
        if current is None:
            raise PepperConfigError("No hay peppers configurados.")
        return cls(peppers, current)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> PepperRing:
        """Lee `VIGIA_PEPPERS_FILE` (preferido, para /run/secrets) o `VIGIA_PEPPERS`."""
        env = os.environ if environ is None else environ
        path = env.get("VIGIA_PEPPERS_FILE")
        if path:
            try:
                with open(path, encoding="utf-8") as fh:
                    return cls.parse(fh.read())
            except OSError:
                raise PepperConfigError("No se pudo leer VIGIA_PEPPERS_FILE.") from None
        if env.get("VIGIA_PEPPERS"):
            return cls.parse(env["VIGIA_PEPPERS"])
        raise PepperConfigError("Falta VIGIA_PEPPERS_FILE o VIGIA_PEPPERS.")


def _b62(value: int, width: int) -> str:
    out = []
    for _ in range(width):
        value, rem = divmod(value, 62)
        out.append(_B62[rem])
    return "".join(reversed(out))


def _checksum(body: str) -> str:
    return _b62(zlib.crc32(body.encode()), 6)


def generate_api_key() -> tuple[str, str, str]:
    """Devuelve `(token, key_id, secret)` con formato `vg_live_<kid>_<secret><crc32 base62>`."""
    kid = secrets.token_hex(8)
    secret = "".join(secrets.choice(_B62) for _ in range(_SECRET_LEN))
    body = f"{API_KEY_PREFIX}{kid}_{secret}"
    return body + _checksum(body), kid, secret


def parse_api_key(token: str) -> tuple[str, str]:
    """Valida formato y checksum sin tocar la base. Devuelve `(key_id, secret)`."""
    if not isinstance(token, str) or len(token) > 128:
        raise InvalidApiKey("API key mal formada.")
    m = _API_KEY_RE.match(token)
    if not m:
        raise InvalidApiKey("API key mal formada.")
    kid, secret, crc = m.groups()
    if not hmac.compare_digest(_checksum(token[: -len(crc)]), crc):
        raise InvalidApiKey("Checksum de API key inválido.")
    return kid, secret


def key_hmac(pepper: bytes, key_id: str, secret: str) -> str:
    """HMAC-SHA256(pepper, key_id || secret) en hex: lo único que se guarda de la key."""
    return hmac.new(pepper, (key_id + secret).encode(), hashlib.sha256).hexdigest()


# --- principal y RBAC -----------------------------------------------------------------------------

ROLE_RANK = {"viewer": 0, "reviewer": 1, "admin": 2, "owner": 3}

# Permiso -> rol mínimo. Un permiso que no está acá se niega siempre (deny-by-default).
PERMISSIONS: dict[str, str] = {
    "account:self": "viewer",  # logout y cambio de la propia contraseña
    "traces:read": "viewer",
    "metrics:read": "viewer",
    "model:read": "viewer",
    "review:read": "viewer",
    "policy:read": "viewer",
    "calibration:read": "viewer",
    "review:write": "reviewer",
    "playground:use": "reviewer",
    "policy:write": "admin",
    "calibration:write": "admin",
    "keys:read": "admin",
    "keys:write": "admin",
    "users:read": "admin",
    "users:write": "admin",
    "audit:read": "admin",
    "data:export": "admin",
    "settings:read": "admin",
    "settings:write": "owner",
    "data:delete": "owner",
}
# Permisos que un viewer puede usar en métodos que mutan: solo acciones sobre su propia cuenta.
SELF_SERVICE_PERMISSIONS = frozenset({"account:self"})
MUTATING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def has_permission(role: str | None, permission: str) -> bool:
    """True solo si el rol existe, el permiso existe y el rol alcanza el mínimo."""
    minimum = PERMISSIONS.get(permission)
    if minimum is None or role not in ROLE_RANK:
        return False
    return ROLE_RANK[role] >= ROLE_RANK[minimum]


def can_assign_role(actor_role: str | None, target_role: str) -> bool:
    """Owner asigna cualquier rol; admin asigna admin, reviewer o viewer; el resto, ninguno."""
    if target_role not in ROLE_RANK or actor_role not in ("owner", "admin"):
        return False
    return actor_role == "owner" or target_role != "owner"


@dataclass(frozen=True)
class Principal:
    """Quién hace el request. Sesión: `role` y `user_id`. API key: `key_id` y `scopes`."""

    org_id: str
    role: str | None = None
    user_id: str | None = None
    key_id: str | None = None
    scopes: frozenset[str] = frozenset()

    def can(self, permission: str) -> bool:
        return has_permission(self.role, permission)

    def has_scope(self, scope: str) -> bool:
        """El scope `admin` incluye `ingest` y `read`."""
        return scope in SCOPES and (scope in self.scopes or "admin" in self.scopes)


# --- sesiones -------------------------------------------------------------------------------------

IDLE_TIMEOUT = timedelta(minutes=30)
ABSOLUTE_TIMEOUT = timedelta(hours=8)
SESSION_TOUCH_INTERVAL = timedelta(seconds=60)
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{43}$")


@dataclass(frozen=True)
class SessionInfo:
    id_hash: str
    user_id: str
    org_id: str
    role: str
    email: str
    csrf_secret: str

    @property
    def csrf_token(self) -> str:
        return csrf_token_for(self.csrf_secret, self.id_hash)


def _hash_session_id(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


class SessionManager:
    """Sesiones server-side: la base guarda sha256(id); el id crudo solo vive en la cookie."""

    def __init__(
        self,
        db: Database,
        *,
        idle: timedelta = IDLE_TIMEOUT,
        absolute: timedelta = ABSOLUTE_TIMEOUT,
        clock: Callable[[], datetime] = utcnow,
        touch_every: timedelta = SESSION_TOUCH_INTERVAL,
    ) -> None:
        self.db = db
        self.idle = idle
        self.absolute = absolute
        self.clock = clock
        self.touch_every = touch_every

    def create(self, user_id: str, org_id: str, *, previous_id: str | None = None) -> str:
        """Crea una sesión nueva e invalida la anterior (regeneración al login). Devuelve el id crudo."""
        if previous_id:
            self.destroy(previous_id)
        raw = secrets.token_urlsafe(32)
        now = self.clock()
        TenantScopedRepository(self.db, org_id).create_session(
            _hash_session_id(raw),
            user_id,
            secrets.token_hex(32),
            created_at=now,
            expires_at=now + self.absolute,
        )
        return raw

    def resolve(self, raw: str | None) -> SessionInfo | None:
        """Sesión vigente o None. Borra la sesión vencida por inactividad o por tope absoluto."""
        if not isinstance(raw, str) or not _SESSION_ID_RE.match(raw):
            return None
        id_hash = _hash_session_id(raw)
        row = lookup_session(self.db, id_hash)
        if row is None:
            return None
        now = self.clock()
        repo = TenantScopedRepository(self.db, row["org_id"])
        last_seen = parse_iso(row["last_seen_at"])
        if (
            row["user_disabled"]
            or row["user_org_id"] != row["org_id"]
            or parse_iso(row["expires_at"]) <= now
            or last_seen + self.idle <= now
        ):
            repo.delete_session(id_hash)
            return None
        if now - last_seen >= self.touch_every:
            repo.touch_session(id_hash, now)
        return SessionInfo(
            id_hash, row["user_id"], row["org_id"], row["role"], row["email"], row["csrf_secret"]
        )

    def destroy(self, raw: str | None) -> bool:
        if not isinstance(raw, str) or not _SESSION_ID_RE.match(raw):
            return False
        id_hash = _hash_session_id(raw)
        row = lookup_session(self.db, id_hash)
        return bool(row) and TenantScopedRepository(self.db, row["org_id"]).delete_session(id_hash)

    def purge_expired(self) -> int:
        """Borra sesiones vencidas de todos los tenants (mantenimiento del sistema)."""
        now = self.clock()
        return self.db.execute(
            "DELETE FROM sessions WHERE expires_at <= ? OR last_seen_at <= ?",
            (to_iso(now), to_iso(now - self.idle)),
        )


def set_session_cookie(response: Response, raw_id: str, *, max_age: timedelta = ABSOLUTE_TIMEOUT) -> None:
    """Cookie `__Host-vigia_session`: Secure, HttpOnly, SameSite=Strict, Path=/ y sin Domain."""
    response.set_cookie(
        SESSION_COOKIE,
        raw_id,
        max_age=int(max_age.total_seconds()),
        path="/",
        secure=True,
        httponly=True,
        samesite="strict",
    )


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(SESSION_COOKIE, path="/", secure=True, httponly=True, samesite="strict")


# --- estado de autenticación de la app ------------------------------------------------------------

MAX_LOGIN_FAILURES = 10
LOCKOUT = timedelta(minutes=15)


def _default_login_limiter() -> RateLimiter:
    # 10 fallos por IP, que se recuperan en 15 minutos.
    return RateLimiter(rate=MAX_LOGIN_FAILURES / LOCKOUT.total_seconds(), burst=MAX_LOGIN_FAILURES)


@dataclass
class AuthState:
    """Lo que las dependencias necesitan; vive en `app.state.vigia_auth` (ver `install_auth`)."""

    db: Database
    peppers: PepperRing
    sessions: SessionManager | None = None
    key_limiter: RateLimiter | None = None
    tenant_limiter: RateLimiter | None = None
    login_ip_limiter: RateLimiter | None = field(default_factory=_default_login_limiter)
    scrypt_n: int = SCRYPT_N
    clock: Callable[[], datetime] = utcnow

    def __post_init__(self) -> None:
        if self.sessions is None:
            self.sessions = SessionManager(self.db, clock=self.clock)

    def repo(self, org_id: str) -> TenantScopedRepository:
        return TenantScopedRepository(self.db, org_id)


def install_auth(app: FastAPI, state: AuthState) -> None:
    app.state.vigia_auth = state


# --- operaciones de identidad ---------------------------------------------------------------------

_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+-]{1,64}@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}$")


@dataclass(frozen=True)
class LoginResult:
    session_id: str  # va solo a la cookie
    principal: Principal
    csrf_token: str


def login(
    state: AuthState, email: str, password: str, *, ip: str, previous_session_id: str | None = None
) -> LoginResult:
    """Verifica credenciales y crea una sesión nueva. Bloquea.

    Lanza `RateLimitExceeded` (429) si la IP acumuló demasiados fallos y `LoginFailed` ante
    cualquier otro fallo. Tras `MAX_LOGIN_FAILURES` fallos seguidos, la cuenta queda bloqueada
    `LOCKOUT`. Es CPU-intensiva (scrypt): desde una ruta async, correrla en un threadpool.
    """
    limiter = state.login_ip_limiter
    ip_key = f"login-ip:{ip}"
    if limiter is not None:
        wait = limiter.blocked(ip_key)
        if wait is not None:
            raise RateLimitExceeded(wait, "login_ip")
    user = None
    if isinstance(email, str) and len(email) <= 254:
        user = lookup_user_by_email(state.db, email)
    if user is None:
        verify_password(password, _dummy_hash(state.scrypt_n))
        if limiter is not None:
            limiter.hit(ip_key)
        raise LoginFailed()
    now = state.clock()
    repo = state.repo(user["org_id"])
    locked = bool(user["locked_until"]) and parse_iso(user["locked_until"]) > now
    ok = verify_password(password, user["password_hash"])
    actor = {"user_id": user["id"]}
    if not ok or locked or user["disabled"]:
        reason = "locked" if locked else "disabled" if user["disabled"] else "bad_password"
        if not ok and not locked:
            just_locked = repo.record_login_failure(
                user["id"], max_failures=MAX_LOGIN_FAILURES, lock_until=now + LOCKOUT
            )
            if just_locked:
                audit.append(state.db, user["org_id"], actor, "account_locked", user["id"], now=now)
        audit.append(state.db, user["org_id"], actor, "login_failed", user["id"], {"reason": reason}, now=now)
        if limiter is not None:
            limiter.hit(ip_key)
        log_event(_log, "login_failed", level=logging.WARNING, org_id=user["org_id"], user_id=user["id"])
        raise LoginFailed()
    repo.reset_login_failures(user["id"])
    if needs_rehash(user["password_hash"], n=state.scrypt_n):
        repo.update_user(user["id"], password_hash=hash_password(password, n=state.scrypt_n))
    raw = state.sessions.create(user["id"], user["org_id"], previous_id=previous_session_id)
    audit.append(state.db, user["org_id"], actor, "login", user["id"], now=now)
    info = state.sessions.resolve(raw)
    principal = Principal(org_id=user["org_id"], role=user["role"], user_id=user["id"])
    return LoginResult(raw, principal, info.csrf_token if info else "")


def logout(state: AuthState, raw_session_id: str | None) -> bool:
    """Invalida la sesión en el servidor."""
    info = state.sessions.resolve(raw_session_id)
    destroyed = state.sessions.destroy(raw_session_id)
    if info is not None:
        audit.append(state.db, info.org_id, {"user_id": info.user_id}, "logout", info.user_id)
    return destroyed


def register_user(
    state: AuthState, org_id: str, email: str, password: str, role: str, *, actor: Principal | None = None
) -> str:
    """Crea un usuario. Sin `actor` es el alta inicial (bootstrap); con actor se aplica RBAC."""
    if not isinstance(email, str) or len(email) > 254 or not _EMAIL_RE.match(email.strip()):
        raise ValueError("Email inválido.")
    if role not in ROLE_RANK:
        raise ValueError("Rol inválido.")
    if actor is not None and (
        actor.org_id != org_id or not actor.can("users:write") or not can_assign_role(actor.role, role)
    ):
        raise PermissionError("No podés asignar ese rol.")
    check_password_policy(password, email=email)
    user_id = state.repo(org_id).create_user(email, hash_password(password, n=state.scrypt_n), role)
    audit.append(state.db, org_id, actor, "user_create", user_id, {"role": role})
    return user_id


def change_role(state: AuthState, org_id: str, user_id: str, new_role: str, *, actor: Principal) -> bool:
    """Cambia el rol e invalida las sesiones del usuario (regeneración al cambiar privilegios)."""
    repo = state.repo(org_id)
    target = repo.get_user(user_id)
    if target is None or actor.org_id != org_id:
        return False
    if not actor.can("users:write") or not can_assign_role(actor.role, new_role):
        raise PermissionError("No podés asignar ese rol.")
    if target["role"] == "owner" and actor.role != "owner":
        raise PermissionError("Solo un owner puede cambiar a otro owner.")
    repo.update_user(user_id, role=new_role)
    repo.delete_user_sessions(user_id)
    audit.append(state.db, org_id, actor, "role_change", user_id, {"from": target["role"], "to": new_role})
    return True


def set_password(state: AuthState, org_id: str, user_id: str, new_password: str, *, actor: Principal) -> bool:
    """Cambia la contraseña (propia, o de otro con `users:write`) y cierra todas sus sesiones."""
    repo = state.repo(org_id)
    target = repo.get_user(user_id)
    if target is None or actor.org_id != org_id:
        return False
    if actor.user_id != user_id and not actor.can("users:write"):
        raise PermissionError("No podés cambiar esa contraseña.")
    check_password_policy(new_password, email=target["email"])
    repo.update_user(user_id, password_hash=hash_password(new_password, n=state.scrypt_n))
    repo.delete_user_sessions(user_id)
    audit.append(state.db, org_id, actor, "password_change", user_id)
    return True


def create_api_key(
    state: AuthState,
    org_id: str,
    name: str,
    scopes: Iterable[str],
    *,
    actor: Principal | None = None,
    expires_at: datetime | None = None,
) -> tuple[str, str]:
    """Crea una key y devuelve `(token, key_id)`. El token se muestra una vez y nunca se guarda."""
    scope_list = sorted(set(scopes))
    if not scope_list or any(s not in SCOPES for s in scope_list):
        raise ValueError(f"Scopes inválidos; opciones: {SCOPES}")
    if not isinstance(name, str) or not name.strip() or len(name) > 100:
        raise ValueError("Nombre de key inválido.")
    if actor is not None and (actor.org_id != org_id or not actor.can("keys:write")):
        raise PermissionError("No podés crear keys.")
    token, kid, secret = generate_api_key()
    pepper_kid = state.peppers.current_kid
    state.repo(org_id).insert_api_key(
        kid,
        name=name.strip(),
        hmac_hex=key_hmac(state.peppers.peppers[pepper_kid], kid, secret),
        pepper_kid=pepper_kid,
        scopes=scope_list,
        created_by=actor.user_id if actor else None,
        expires_at=expires_at,
    )
    audit.append(state.db, org_id, actor, "key_create", kid, {"scopes": scope_list, "name": name.strip()})
    return token, kid


def revoke_api_key(state: AuthState, org_id: str, key_id: str, *, actor: Principal | None = None) -> bool:
    """Revoca de inmediato (sin cache). False si la key no es de este tenant o ya estaba revocada."""
    if actor is not None and (actor.org_id != org_id or not actor.can("keys:write")):
        raise PermissionError("No podés revocar keys.")
    revoked = state.repo(org_id).revoke_api_key(key_id, now=state.clock())
    if revoked:
        audit.append(state.db, org_id, actor, "key_revoke", key_id)
    return revoked


_DUMMY_PEPPER = secrets.token_bytes(32)


def verify_api_key(state: AuthState, token: str) -> Principal | None:
    """Principal de una key válida, vigente y no revocada; None en cualquier otro caso."""
    try:
        kid, secret = parse_api_key(token)
    except InvalidApiKey:
        return None
    row = lookup_api_key(state.db, kid)
    pepper = state.peppers.peppers.get(row["pepper_kid"]) if row else None
    candidate = key_hmac(pepper or _DUMMY_PEPPER, kid, secret)
    if row is None or pepper is None or not hmac.compare_digest(candidate, row["hmac"]):
        return None
    now = state.clock()
    if row["revoked_at"] or (row["expires_at"] and parse_iso(row["expires_at"]) <= now):
        return None
    if not row["last_used_at"] or now - parse_iso(row["last_used_at"]) >= KEY_TOUCH_INTERVAL:
        state.repo(row["org_id"]).touch_api_key(kid, now)
    return Principal(org_id=row["org_id"], key_id=kid, scopes=frozenset(row["scopes"].split(",")))


# --- dependencias FastAPI y declaración de permisos -----------------------------------------------


@dataclass(frozen=True)
class PermissionSpec:
    kind: str  # session | api_key | session_or_key | public
    permission: str | None = None
    scope: str | None = None


PERMISSION_ATTR = "vigia_permission"
_QUERY_CREDENTIAL_NAMES = frozenset(
    {"api_key", "apikey", "api-key", "access_token", "authorization", "bearer"}
)


def _auth_state(request: Request) -> AuthState:
    state = getattr(request.app.state, "vigia_auth", None)
    if not isinstance(state, AuthState):
        raise HTTPException(503, "Autenticación no configurada.")
    return state


def reject_query_credentials(request: Request) -> None:
    """400 si el request trae una credencial en la URL (terminaría en logs de proxies)."""
    for name, value in request.query_params.multi_items():
        if name.lower() in _QUERY_CREDENTIAL_NAMES or API_KEY_PREFIX in value:
            raise HTTPException(400, "No envíes credenciales en la URL; usá el header Authorization.")


def bearer_token(request: Request) -> str | None:
    header = request.headers.get("authorization")
    if header is None:
        return None
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(401, "Header Authorization inválido.", headers={"WWW-Authenticate": "Bearer"})
    return token.strip()


def _bind(request: Request, principal: Principal) -> Principal:
    request.state.principal = principal
    request.state.authenticated = True
    return principal


async def _session_principal(request: Request, permission: str) -> Principal:
    state = _auth_state(request)
    raw = request.cookies.get(SESSION_COOKIE)
    try:
        info = state.sessions.resolve(raw) if raw else None
    except Exception:  # noqa: BLE001 - fail-closed: un error interno nunca autentica
        log_event(_log, "auth_backend_error", level=logging.ERROR)
        raise HTTPException(503, "Autenticación no disponible.") from None
    if info is None:
        raise HTTPException(401, "Sesión inválida o vencida.")
    request.state.authenticated = True
    request.state.session = info
    request.state.csrf_token = info.csrf_token
    await enforce_csrf(request, info.csrf_secret, info.id_hash)
    if not has_permission(info.role, permission):
        raise HTTPException(403, "Tu rol no tiene permiso para esta acción.")
    return _bind(request, Principal(org_id=info.org_id, role=info.role, user_id=info.user_id))


async def _key_principal(request: Request, scope: str) -> Principal:
    state = _auth_state(request)
    reject_query_credentials(request)
    token = bearer_token(request)
    if token is None:
        raise HTTPException(401, "Falta la API key.", headers={"WWW-Authenticate": "Bearer"})
    try:
        principal = verify_api_key(state, token)
    except Exception:  # noqa: BLE001 - fail-closed
        log_event(_log, "auth_backend_error", level=logging.ERROR)
        raise HTTPException(503, "Autenticación no disponible.") from None
    if principal is None:
        log_event(_log, "api_key_rejected", level=logging.WARNING)
        raise HTTPException(
            401, "API key inválida, vencida o revocada.", headers={"WWW-Authenticate": "Bearer"}
        )
    request.state.authenticated = True
    if not principal.has_scope(scope):
        raise HTTPException(403, "La API key no tiene el scope requerido.")
    enforce_rate_limit(state.key_limiter, f"key:{principal.key_id}", scope="key")
    enforce_rate_limit(state.tenant_limiter, f"org:{principal.org_id}", scope="tenant")
    return _bind(request, principal)


def _mark(dependency: Callable[..., Any], spec: PermissionSpec) -> Callable[..., Any]:
    setattr(dependency, PERMISSION_ATTR, spec)
    return dependency


def _check_names(permission: str | None, scope: str | None) -> None:
    if permission is not None and permission not in PERMISSIONS:
        raise ValueError(f"permiso desconocido: {permission}")
    if scope is not None and scope not in SCOPES:
        raise ValueError(f"scope desconocido: {scope}")


def require_session(permission: str) -> Callable[..., Any]:
    """Dependencia para rutas del panel: sesión vigente, CSRF en métodos no seguros y permiso."""
    _check_names(permission, None)

    async def dependency(request: Request) -> Principal:
        return await _session_principal(request, permission)

    return _mark(dependency, PermissionSpec("session", permission=permission))


def require_api_key(scope: str) -> Callable[..., Any]:
    """Dependencia para la API: `Authorization: Bearer vg_live_...`, scope y rate limit."""
    _check_names(None, scope)

    async def dependency(request: Request) -> Principal:
        return await _key_principal(request, scope)

    return _mark(dependency, PermissionSpec("api_key", scope=scope))


def require_session_or_key(permission: str, scope: str) -> Callable[..., Any]:
    """Acepta API key (si hay header Authorization) o sesión del panel."""
    _check_names(permission, scope)

    async def dependency(request: Request) -> Principal:
        if request.headers.get("authorization") is not None:
            return await _key_principal(request, scope)
        return await _session_principal(request, permission)

    return _mark(dependency, PermissionSpec("session_or_key", permission=permission, scope=scope))


def public_route() -> Callable[..., Any]:
    """Declara explícitamente que una ruta no requiere identidad (health, login, estáticos)."""

    async def dependency() -> None:
        return None

    return _mark(dependency, PermissionSpec("public"))


def route_permission(route: Any) -> PermissionSpec | None:
    """Permiso declarado por una ruta (buscando en todo su árbol de dependencias)."""
    dependant = getattr(route, "dependant", None)
    stack = [dependant] if dependant is not None else []
    while stack:
        dep = stack.pop()
        spec = getattr(dep.call, PERMISSION_ATTR, None) if dep.call is not None else None
        if isinstance(spec, PermissionSpec):
            return spec
        stack.extend(dep.dependencies)
    return None


def _label(route: Any) -> str:
    methods = ",".join(sorted(getattr(route, "methods", None) or ["*"]))
    return f"{methods} {getattr(route, 'path', '?')}"


def _walk_routes(routes: Iterable[Any]) -> Iterator[Any]:
    """Aplana los routers incluidos (FastAPI >= 0.140 guarda `_IncludedRouter` perezosos en app.routes)."""
    for route in routes:
        inner = getattr(route, "original_router", None)
        if inner is not None:
            yield from _walk_routes(inner.routes)
        else:
            yield route


def undeclared_routes(app: FastAPI, *, exempt_paths: Iterable[str] = ()) -> list[str]:
    """Rutas sin permiso declarado. Una ruta que no es de FastAPI (mount, docs) solo pasa si está exenta."""
    exempt = set(exempt_paths)
    out = []
    for route in _walk_routes(app.routes):
        if isinstance(route, APIRoute):
            if route_permission(route) is None:
                out.append(_label(route))
        elif getattr(route, "path", None) not in exempt:
            out.append(_label(route))
    return out


def viewer_mutation_violations(app: FastAPI) -> list[str]:
    """Rutas que mutan y que un viewer podría usar (salvo acciones sobre su propia cuenta)."""
    out = []
    for route in _walk_routes(app.routes):
        if not isinstance(route, APIRoute) or not (route.methods & MUTATING_METHODS):
            continue
        spec = route_permission(route)
        if spec is None:
            out.append(_label(route))
        elif (
            spec.kind in ("session", "session_or_key")
            and spec.permission not in SELF_SERVICE_PERMISSIONS
            and has_permission("viewer", spec.permission or "")
        ):
            out.append(_label(route))
    return out
