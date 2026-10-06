"""Ejecución de código generado contra tests: etiqueta computable para benchmarks de código.

ADVERTENCIA: esto NO es un sandbox de seguridad. El código del modelo corre con los permisos del
usuario que lanza Vigía: puede leer y escribir archivos, abrir conexiones de red y lanzar procesos.
El aislamiento cubre solo accidentes comunes:

- un subprocess aparte con `python -I -S -X utf8` (sin site-packages, sin variables PYTHON*),
- un directorio temporal propio como directorio de trabajo, borrado al terminar,
- stdin cerrado y un entorno mínimo (solo PATH, y SYSTEMROOT en Windows),
- un timeout duro que mata el árbol de procesos completo,
- salida acotada y un marcador con nonce para que `sys.exit(0)` u `os._exit(0)` no pasen por éxito.

Corré esto solo en una VM o un contenedor descartable (por ejemplo, un entorno de Colab), nunca en
una máquina con credenciales o datos que importen.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

CodeStatus = Literal["passed", "failed", "error", "timeout", "no_code"]
CODE_STATUSES: tuple[str, ...] = ("passed", "failed", "error", "timeout", "no_code")
MAX_OUTPUT_CHARS = 64_000
MAX_DETAIL_CHARS = 200
_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+.-]*)[^\n]*\n(.*?)```", re.DOTALL)
_OPEN_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+.-]*)[^\n]*\n")
_PY_LANGS = frozenset({"python", "py", "python3"})

_RUNNER = r'''
import io, json, os, sys, traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_MAX = 64000


class _Sink(io.TextIOBase):
    """Salida del código evaluado, acotada: lo que pasa del tope se descarta."""

    def __init__(self):
        self.size = 0

    def writable(self):
        return True

    def write(self, s):
        self.size += len(s)
        if self.size <= _MAX:
            sys.__stdout__.write(s)
        return len(s)


def _where(exc):
    frames = [f for f in traceback.extract_tb(exc.__traceback__) if f.filename in ("solution.py", "tests.py")]
    return f" ({frames[-1].filename}:{frames[-1].lineno})" if frames else ""


def _main():
    path = os.path.join(_HERE, "nonce.txt")
    with open(path, encoding="ascii") as fh:
        nonce = fh.read().strip()
    os.remove(path)
    out = sys.__stdout__

    def report(status, detail):
        out.write("\n" + nonce + " " + json.dumps({"status": status, "detail": detail}) + "\n")
        out.flush()

    with open(os.path.join(_HERE, "solution.py"), encoding="utf-8") as fh:
        code = fh.read()
    with open(os.path.join(_HERE, "tests.py"), encoding="utf-8") as fh:
        tests = fh.read()
    sys.stdout = sys.stderr = _Sink()
    try:
        compiled = compile(code, "solution.py", "exec")
    except (SyntaxError, ValueError) as exc:
        line = getattr(exc, "lineno", None)
        return report("error", f"{type(exc).__name__} en solution.py" + (f":{line}" if line else ""))
    namespace = {"__name__": "__solution__", "__builtins__": __builtins__}
    try:
        exec(compiled, namespace)
    except BaseException as exc:
        return report("error", f"{type(exc).__name__} al cargar el código{_where(exc)}")
    try:
        compiled_tests = compile(tests, "tests.py", "exec")
    except (SyntaxError, ValueError) as exc:
        return report("error", f"{type(exc).__name__} en los tests")
    try:
        exec(compiled_tests, namespace)
    except BaseException as exc:
        return report("failed", f"{type(exc).__name__} en los tests{_where(exc)}")
    return report("passed", "")


_main()
'''


@dataclass(frozen=True)
class CodeResult:
    """Resultado de correr los tests: `passed` es True solo con `status == "passed"`.

    `detail` es corto y no copia la salida del programa: nombre de la excepción y dónde ocurrió.
    """

    passed: bool
    status: CodeStatus
    detail: str


def extract_code(response: str) -> str:
    """Código de una respuesta: el último bloque ```python``` (o ```py```, ```python3```).

    Sin bloque Python, toma el último bloque ``` de cualquier lenguaje. Un bloque abierto sin
    cierre (respuesta cortada por max_tokens) aporta lo que sigue a la apertura. Sin ningún bloque,
    devuelve el texto entero. No adivina: si la respuesta trae solo el cuerpo de la función, devuelve
    ese cuerpo y los tests fallarán.
    """
    if not isinstance(response, str):
        raise TypeError("extract_code espera un str")
    blocks = [(m.group(1).lower(), m.group(2)) for m in _FENCE.finditer(response)]
    python_blocks = [body for lang, body in blocks if lang in _PY_LANGS]
    if python_blocks:
        return python_blocks[-1].strip("\n")
    if blocks:
        return blocks[-1][1].strip("\n")
    opened = list(_OPEN_FENCE.finditer(response))
    if opened:
        return response[opened[-1].end() :].strip("\n")
    return response.strip("\n")


def _child_env() -> dict[str, str]:
    env = {"PATH": os.environ.get("PATH", os.defpath)}
    if os.name == "nt":
        env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", r"C:\Windows")
    return env


def _kill_tree(proc: subprocess.Popen[bytes]) -> None:
    """Mata el proceso y todos sus descendientes (taskkill /T en Windows, grupo de sesión en POSIX)."""
    if os.name == "nt":
        if proc.poll() is not None:
            return  # sin el padre, taskkill no recorre el árbol
        taskkill = Path(os.environ.get("SYSTEMROOT", r"C:\Windows")) / "System32" / "taskkill.exe"
        subprocess.run(  # noqa: S603 - ruta fija del sistema, pid propio
            [str(taskkill), "/F", "/T", "/PID", str(proc.pid)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if proc.poll() is None:
            proc.kill()
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()


def _parse_marker(stdout: str, nonce: str) -> dict[str, str] | None:
    for line in reversed(stdout.splitlines()):
        if line.startswith(nonce + " "):
            try:
                data = json.loads(line[len(nonce) + 1 :])
            except ValueError:
                return None
            if isinstance(data, dict) and data.get("status") in CODE_STATUSES:
                return {"status": str(data["status"]), "detail": str(data.get("detail", ""))}
            return None
    return None


def run_tests(
    code: str, tests: str, *, timeout_s: float = 10.0, python: str = sys.executable
) -> CodeResult:
    """Corre `code` y después `tests` en el mismo espacio de nombres, en un subprocess aparte.

    Estados: `passed` (los tests terminan sin excepción), `failed` (una excepción en los tests,
    típicamente AssertionError), `error` (el código no compila o falla al cargarse, o el proceso
    termina sin informar), `timeout` (superó `timeout_s`; se mata el árbol de procesos) y
    `no_code` (no hay código). NO es un sandbox: ver el docstring del módulo.
    """
    if not isinstance(code, str) or not isinstance(tests, str):
        raise TypeError("code y tests deben ser str")
    if timeout_s <= 0:
        raise ValueError("timeout_s debe ser > 0")
    if not code.strip():
        return CodeResult(False, "no_code", "la respuesta no tiene código")
    nonce = secrets.token_hex(16)
    with tempfile.TemporaryDirectory(prefix="vigia-code-", ignore_cleanup_errors=True) as tmp:
        root = Path(tmp)
        (root / "_vigia_runner.py").write_text(_RUNNER, encoding="utf-8")
        (root / "solution.py").write_text(code, encoding="utf-8")
        (root / "tests.py").write_text(tests, encoding="utf-8")
        (root / "nonce.txt").write_text(nonce, encoding="ascii")
        kwargs: dict[str, object] = {}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        else:
            kwargs["start_new_session"] = True
        proc = subprocess.Popen(  # noqa: S603 - intérprete y script propios; ver advertencia del módulo
            [python, "-I", "-S", "-X", "utf8", "_vigia_runner.py"],
            cwd=root,
            env=_child_env(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            **kwargs,  # type: ignore[arg-type]
        )
        try:
            raw, _ = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            _kill_tree(proc)
            try:
                proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            return CodeResult(False, "timeout", f"superó {timeout_s:g} s")
        finally:
            _kill_tree(proc)  # también nietos que sigan vivos tras una salida normal
        stdout = raw[-MAX_OUTPUT_CHARS * 4 :].decode("utf-8", errors="replace")
    marker = _parse_marker(stdout, nonce)
    if marker is None:
        return CodeResult(False, "error", f"el proceso terminó con código {proc.returncode} sin resultado")
    status = marker["status"]
    detail = marker["detail"][:MAX_DETAIL_CHARS]
    if status == "passed" and proc.returncode != 0:
        return CodeResult(False, "error", f"el proceso terminó con código {proc.returncode}")
    return CodeResult(status == "passed", status, detail)  # type: ignore[arg-type]


__all__ = ["CODE_STATUSES", "CodeResult", "CodeStatus", "extract_code", "run_tests"]
