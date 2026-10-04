"""Soporte de la suite e2e: plataforma real en un subproceso y helpers de Playwright (sync API).

La plataforma corre con `python -m vigia_ee.app serve` sobre el modelo diminuto. Variables útiles:

- VIGIA_E2E_PORT: puerto fijo (por defecto, uno libre).
- VIGIA_E2E_DIR: carpeta de datos fija (por defecto, una temporal de pytest).
- VIGIA_E2E_MODEL: modelo diminuto (por defecto `tiny:mixtral`).
- VIGIA_E2E_STARTUP_S: espera máxima de `/startupz` (por defecto 900 s; el import de torch es lento).
"""

from __future__ import annotations

import json
import os
import random
import re
import secrets
import socket
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
from playwright.sync_api import BrowserContext, Page, expect

REPO = Path(__file__).resolve().parents[2]
OWNER_PASSWORD = "Clave-Larga-Del-Owner-2026"  # noqa: S105 - credencial de prueba local
OWNER_B_PASSWORD = "Otra-Clave-Larga-Org-B-2026"  # noqa: S105 - credencial de prueba local
TEMP_PASSWORD_RE = re.compile(r"\b[A-Za-z0-9]{5}-[A-Za-z0-9]{5}-[A-Za-z0-9]{5}-[A-Za-z0-9]{5}\b")
API_KEY_RE = re.compile(r"vg_live_[A-Za-z0-9_]+")
ORG_RE = re.compile(r"Organizaci\S+ creada: ([0-9a-f]{32})")
USER_RE = re.compile(r"Owner creado: ([0-9a-f]{32})")


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@dataclass
class Platform:
    base_url: str
    data_dir: Path
    env: dict[str, str]
    org_a: str
    org_b: str
    owner_email: str = "duena@acme.test"
    owner_b_email: str = "duena@otra.test"
    api_key: str = ""
    accounts: dict[str, str] = field(default_factory=dict)
    cache: dict[str, Any] = field(default_factory=dict)

    @property
    def db_path(self) -> Path:
        return self.data_dir / "vigia.db"

    def url(self, path: str) -> str:
        return self.base_url + path

    # -- acceso directo a la base (solo para preparar estados que el reloj no permite esperar) --

    def sql(self, query: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        con = sqlite3.connect(self.db_path, timeout=30)
        con.row_factory = sqlite3.Row
        try:
            rows = con.execute(query, params).fetchall()
            con.commit()
            return rows
        finally:
            con.close()

    # -- API --

    def chat(
        self,
        prompt: str,
        *,
        key: str | None = None,
        max_tokens: int = 16,
        vigia: bool = True,
        timeout: float = 300.0,
    ) -> dict:
        body: dict[str, Any] = {
            "model": self.env["VIGIA_MODEL"],
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        if vigia:
            body["vigia"] = {}
        resp = httpx.post(
            self.url("/v1/chat/completions"),
            headers={"Authorization": f"Bearer {key or self.api_key}"},
            json=body,
            timeout=timeout,
        )
        assert resp.status_code == 200, resp.text[:500]
        return resp.json()


def _run_cli(args: list[str], env: dict[str, str]) -> str:
    proc = subprocess.run(  # noqa: S603 - argumentos fijos del propio repo
        [sys.executable, "-m", "vigia_ee.app", *args],
        cwd=REPO,
        env=env,
        capture_output=True,
        timeout=600,
        check=False,
    )
    out = proc.stdout.decode("utf-8", "replace")
    err = proc.stderr.decode("utf-8", "replace")
    assert proc.returncode == 0, f"vigia_ee.app {args[0]} falló: {err or out}"
    return out


def start_platform(data_dir: Path, port: int) -> tuple[Platform, subprocess.Popen[bytes], Any]:
    data_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(
        {
            "VIGIA_DATA_DIR": str(data_dir),
            "VIGIA_MODEL": os.environ.get("VIGIA_E2E_MODEL", "tiny:mixtral"),
            "VIGIA_PEPPERS": "k1:" + secrets.token_urlsafe(32),
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
        }
    )
    for name in ("VIGIA_PEPPERS_FILE", "VIGIA_DB_PATH", "VIGIA_BASE_DETECTOR", "VIGIA_BACKEND"):
        env.pop(name, None)
    orgs = []
    for org, email, password in (
        ("Acme E2E", "duena@acme.test", OWNER_PASSWORD),
        ("Otra Org", "duena@otra.test", OWNER_B_PASSWORD),
    ):
        pw_file = data_dir / f"pw-{email.split('@')[1]}.txt"
        pw_file.write_text(password, encoding="utf-8")
        out = _run_cli(
            ["init", "--org", org, "--email", email], {**env, "VIGIA_OWNER_PASSWORD_FILE": str(pw_file)}
        )
        pw_file.unlink()
        match = ORG_RE.search(out)
        assert match, out
        orgs.append(match.group(1))
    out = _run_cli(["create-key", "--org-id", orgs[0], "--name", "e2e-cli", "--scopes", "ingest,read"], env)
    key = API_KEY_RE.search(out)
    assert key, out
    log = open(data_dir / "server.log", "wb")  # noqa: SIM115 - se cierra en stop_platform
    proc = subprocess.Popen(  # noqa: S603 - argumentos fijos del propio repo
        [sys.executable, "-m", "vigia_ee.app", "serve", "--host", "127.0.0.1", "--port", str(port)],
        cwd=REPO,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + float(os.environ.get("VIGIA_E2E_STARTUP_S", "900"))
    while True:
        if proc.poll() is not None:
            log.close()
            tail = (data_dir / "server.log").read_text("utf-8", "replace")[-3000:]
            raise RuntimeError(f"la plataforma terminó al arrancar (código {proc.returncode}):\n{tail}")
        try:
            if httpx.get(base + "/startupz", timeout=5).status_code == 200:
                break
        except httpx.HTTPError:
            pass
        if time.monotonic() > deadline:
            stop_platform(proc, log)
            raise RuntimeError("la plataforma no respondió /startupz a tiempo")
        time.sleep(1)
    platform = Platform(base_url=base, data_dir=data_dir, env=env, org_a=orgs[0], org_b=orgs[1])
    platform.api_key = key.group(0)
    return platform, proc, log


def create_org(platform: Platform, name: str, email: str, password: str) -> tuple[str, str]:
    """Crea con la CLI (sobre la base de la plataforma en marcha) una org con owner y una API key.

    Devuelve `(org_id, api_key)`. La contraseña viaja por archivo, como en producción."""
    slug = re.sub(r"[^a-z0-9]+", "-", email.lower())
    pw_file = platform.data_dir / f"pw-{slug}.txt"
    pw_file.write_text(password, encoding="utf-8")
    try:
        out = _run_cli(
            ["init", "--org", name, "--email", email],
            {**platform.env, "VIGIA_OWNER_PASSWORD_FILE": str(pw_file)},
        )
    finally:
        pw_file.unlink(missing_ok=True)
    match = ORG_RE.search(out)
    assert match, out
    args = ["create-key", "--org-id", match.group(1), "--name", "e2e", "--scopes", "ingest,read"]
    out = _run_cli(args, platform.env)
    key = API_KEY_RE.search(out)
    assert key, out
    return match.group(1), key.group(0)


def stop_platform(proc: subprocess.Popen[bytes], log: Any) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)
    log.close()


# --- navegador -------------------------------------------------------------------------------------


class ConsoleLog:
    """Junta errores de consola, excepciones de página y respuestas 5xx de una página."""

    def __init__(self, page: Page) -> None:
        self.errors: list[str] = []
        page.on("console", self._console)
        page.on("pageerror", lambda exc: self.errors.append(f"pageerror: {exc}"))
        page.on(
            "response",
            lambda r: self.errors.append(f"HTTP {r.status} {r.url}") if r.status >= 500 else None,
        )

    def _console(self, msg: Any) -> None:
        if msg.type == "error":
            self.errors.append(f"console: {msg.text}")


def login(page: Page, platform: Platform, email: str, password: str, *, expect_ok: bool = True) -> None:
    page.goto(platform.url("/login"))
    page.fill("#email", email)
    page.fill("#password", password)
    page.click("form[action='/login'] button[type=submit]")
    page.wait_for_load_state()
    if expect_ok:
        assert "/login" not in page.url, f"login falló para {email}: {page.url}"


def csrf_token(page: Page) -> str:
    token = page.locator("input[name=csrf_token]").first.get_attribute("value")
    assert token, "la página no trae token CSRF"
    return token


FETCH_JS = """async ({url, method, form, json, files, headers}) => {
    const opts = {method, credentials: "same-origin", headers: headers || {}};
    if (form) {
        const body = files ? new FormData() : new URLSearchParams();
        for (const [k, v] of Object.entries(form)) { body.append(k, v); }
        for (const [k, f] of Object.entries(files || {})) {
            body.append(k, new Blob([f.text], {type: f.type}), f.name);
        }
        opts.body = body;
    }
    if (json !== null) { opts.body = json; opts.headers["Content-Type"] = "application/json"; }
    const r = await fetch(url, opts);
    return {status: r.status, url: r.url, redirected: r.redirected};
}"""


def browser_fetch(
    page: Page,
    url: str,
    *,
    method: str = "POST",
    form: dict[str, Any] | None = None,
    json: str | None = None,
    files: dict[str, dict[str, str]] | None = None,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Request desde la página con `fetch` (cookie de sesión real del navegador; sigue redirecciones).

    No se usa `page.request`: el APIRequestContext de Playwright no manda la cookie `__Host-` Secure
    a http://127.0.0.1, mientras que Chromium sí la manda en navegación y en fetch."""
    return page.evaluate(
        FETCH_JS,
        {"url": url, "method": method, "form": form, "json": json, "files": files, "headers": headers},
    )


def post_form(page: Page, path: str, data: dict[str, Any], platform: Platform) -> dict[str, Any]:
    """POST de formulario con la sesión y el token CSRF de la página."""
    return browser_fetch(page, platform.url(path), form={"csrf_token": csrf_token(page), **data})


def invite_user(page: Page, platform: Platform, email: str, role: str) -> str:
    """Invita desde Configuración y devuelve la contraseña temporal que muestra el panel."""
    page.goto(platform.url("/settings"))
    page.fill("#invite-email", email)
    page.select_option("#invite-role", role)
    page.click("form[action='/settings/users'] button[type=submit]")
    page.wait_for_load_state()
    match = TEMP_PASSWORD_RE.search(page.content())
    assert match, f"el panel no mostró la contraseña temporal de {email}"
    return match.group(0)


def create_key_in_panel(page: Page, platform: Platform, name: str, scopes: tuple[str, ...]) -> str:
    page.goto(platform.url("/settings"))
    page.fill("#key-name", name)
    for scope in ("ingest", "read", "admin"):
        box = page.locator(f"form[action='/settings/keys'] input[name=scopes][value={scope}]")
        if scope in scopes:
            box.check()
        else:
            box.uncheck()
    page.click("form[action='/settings/keys'] button[type=submit]")
    page.wait_for_load_state()
    match = API_KEY_RE.search(page.content())
    assert match, "el panel no mostró el token de la API key"
    return match.group(0)


def ensure_accounts(platform: Platform, context: BrowserContext) -> dict[str, str]:
    """Crea una vez por sesión, desde el panel del owner, las cuentas compartidas por los tests."""
    wanted = {
        "admin@acme.test": "admin",
        "revisora1@acme.test": "reviewer",
        "revisor2@acme.test": "reviewer",
        "lectora@acme.test": "viewer",
    }
    missing = [e for e in wanted if e not in platform.accounts]
    if missing:
        page = context.new_page()
        login(page, platform, platform.owner_email, OWNER_PASSWORD)
        for email in missing:
            platform.accounts[email] = invite_user(page, platform, email, wanted[email])
        page.close()
    return platform.accounts


def calibration_dataset(platform: Platform, n: int = 24) -> bytes:
    """JSONL de 24 respuestas balanceadas: la mitad con `answers` = lo que el modelo diminuto genera
    (correcta) y la otra mitad con `response` = texto al azar y `label: 1` (errónea, por teacher
    forcing).

    Las erróneas no pueden ser `answers` con un gold que nunca sale: el modelo genera el mismo texto
    para todas las preguntas, las dos clases tienen features idénticas y, con un split estratificado,
    Platt rechaza siempre el detector (pendiente <= 0). El texto al azar sí deja señal (NLL alta).
    """
    rng = random.Random(0)  # noqa: S311 - datos de prueba reproducibles
    lines = []
    for i in range(n):
        question = f"pregunta de calibración número {i}"
        item: dict[str, Any] = {"id": f"e2e-{i}", "question": question}
        if i % 2 == 0:
            out = platform.chat(question, max_tokens=64, vigia=False)
            item["answers"] = [out["choices"][0]["message"]["content"]]
        else:
            item["response"] = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz ") for _ in range(48))
            item["label"] = 1
        lines.append(json.dumps(item, ensure_ascii=False))
    return ("\n".join(lines) + "\n").encode("utf-8")


def run_calibration_flow(platform: Platform, context: BrowserContext) -> dict[str, Any]:
    """Flujo de calibración por el panel: subir JSONL, esperar el job, ver el reporte y activar."""
    if "calibration" in platform.cache:
        return platform.cache["calibration"]
    path = platform.data_dir / "calibracion-e2e.jsonl"
    path.write_bytes(calibration_dataset(platform))
    page = context.new_page()
    console = ConsoleLog(page)
    login(page, platform, platform.owner_email, OWNER_PASSWORD)
    page.goto(platform.url("/calibration"))
    page.set_input_files("#file", str(path))
    page.fill("form[action='/calibration/datasets'] #ds-name", "dataset e2e")
    page.click("form[action='/calibration/datasets'] button[type=submit]")
    page.wait_for_load_state()
    assert "ok=dataset_uploaded" in page.url, page.locator(".vg-alert").all_inner_texts()
    # La semilla 0 (la del formulario) se prueba primero; si el job falla, se reintenta con otra
    # semilla como haría una persona, y el fallo queda en `seed_failures` para el test que lo exige.
    seed_failures: list[str] = []
    report_link = page.get_by_role("link", name="Ver el reporte del detector")
    job_id = ""
    for seed in (0, 1, 2, 3):
        page.goto(platform.url("/calibration"))
        option = page.locator("#dataset_id option", has_text="dataset e2e").first
        page.select_option("#dataset_id", option.get_attribute("value"))
        page.fill("#seed", str(seed))
        page.click("form[action='/calibration/start'] button[type=submit]")
        page.wait_for_load_state()
        assert "/calibration/jobs/" in page.url, page.url
        job_id = page.url.split("/calibration/jobs/")[1].split("?")[0]
        # El JS consulta el estado cada 3 s y muestra el enlace al reporte cuando termina.
        status = page.locator("[data-job-status]")
        expect(status).to_have_attribute("aria-busy", "false", timeout=600_000)
        if report_link.count():
            break
        seed_failures.append(f"semilla {seed}: {status.inner_text()}")
    expect(report_link).to_be_visible()
    report_link.click()
    page.wait_for_load_state()
    detector_id = page.url.rstrip("/").split("/")[-1]
    expect(page.locator("h1")).to_contain_text(detector_id)
    report_text = page.locator("main").inner_text()
    page.check("form[action$='/activate'] input[name=confirm]")
    page.click("form[action$='/activate'] button[type=submit]")
    page.wait_for_load_state()
    assert "ok=detector_activated" in page.url, page.locator(".vg-alert").all_inner_texts()
    info = {
        "job_id": job_id,
        "detector_id": detector_id,
        "report_text": report_text,
        "console": list(console.errors),
        "seed_failures": seed_failures,
    }
    page.close()
    platform.cache["calibration"] = info
    return info


def h_overflow(page: Page) -> int:
    """Píxeles de scroll horizontal del documento (0 = sin scroll horizontal)."""
    return int(
        page.evaluate(
            "() => Math.max(document.documentElement.scrollWidth, document.body.scrollWidth)"
            " - document.documentElement.clientWidth"
        )
    )


def iso_ago(minutes: int) -> str:
    return (datetime.now(UTC) - timedelta(minutes=minutes)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def enqueue_for_review(page: Page, platform: Platform, trace_id: str, min_reviewers: int = 2) -> None:
    resp = post_form(
        page,
        f"/generations/{trace_id}/review",
        {"target": "response", "min_reviewers": str(min_reviewers)},
        platform,
    )
    assert resp["redirected"] and "ok=review_created" in resp["url"], resp
