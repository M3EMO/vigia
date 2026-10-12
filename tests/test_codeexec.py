"""Ejecución de código generado contra tests (vigia.codeexec): extracción, estados y aislamiento."""

from __future__ import annotations

import time

import pytest

from vigia.codeexec import CodeResult, extract_code, run_tests

TESTS = "assert double(2) == 4\nassert double(-3) == -6\n"
GOOD = "def double(x):\n    return 2 * x\n"


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("Acá va:\n```python\ndef f():\n    return 1\n```\nListo.", "def f():\n    return 1"),
        ("```\ndef f():\n    return 2\n```", "def f():\n    return 2"),
        ("```py\nx = 1\n```", "x = 1"),
        # El último bloque Python gana sobre bloques anteriores.
        ("```python\nx = 1\n```\ntexto\n```python\nx = 2\n```", "x = 2"),
        # Un bloque Python gana sobre un bloque de otro lenguaje posterior.
        ("```python\nx = 1\n```\n```text\nsalida\n```", "x = 1"),
        # Sin bloque Python, el último bloque de cualquier lenguaje.
        ("```\na = 1\n```\n```\nb = 2\n```", "b = 2"),
        # Sin bloque, el texto entero.
        ("def f():\n    return 3\n", "def f():\n    return 3"),
        # Bloque abierto sin cierre (respuesta cortada): lo que sigue a la apertura.
        ("```python\ndef f():\n    return", "def f():\n    return"),
        # Solo el cuerpo: no adivina la firma.
        ("    return x * 2", "    return x * 2"),
        ("", ""),
    ],
)
def test_extract_code_formats(response, expected):
    assert extract_code(response) == expected


def test_run_tests_passed():
    assert run_tests(GOOD, TESTS) == CodeResult(True, "passed", "")


def test_run_tests_failed_assertion():
    result = run_tests("def double(x):\n    return x + 2\n", TESTS)
    assert not result.passed and result.status == "failed"
    assert result.detail.startswith("AssertionError") and "tests.py:2" in result.detail


def test_run_tests_syntax_error():
    result = run_tests("def double(x)\n    return 2 * x\n", TESTS)
    assert not result.passed and result.status == "error" and "SyntaxError" in result.detail


def test_run_tests_error_while_loading_code():
    result = run_tests("raise RuntimeError('secreto')\n", TESTS)
    assert result.status == "error" and "RuntimeError" in result.detail
    assert "secreto" not in result.detail  # detail no copia mensajes ni salida del programa


def test_run_tests_timeout_kills_the_process():
    t0 = time.monotonic()
    result = run_tests("while True:\n    pass\n", TESTS, timeout_s=2)
    assert not result.passed and result.status == "timeout"
    assert time.monotonic() - t0 < 10


@pytest.mark.parametrize("code", ["", "   \n\n"])
def test_run_tests_no_code(code):
    assert run_tests(code, TESTS) == CodeResult(False, "no_code", "la respuesta no tiene código")


@pytest.mark.parametrize("exit_call", ["import sys; sys.exit(0)", "import os; os._exit(0)"])
def test_early_exit_is_not_a_pass(exit_call):
    result = run_tests(GOOD + exit_call + "\n", TESTS)
    assert not result.passed and result.status == "error"


def test_subprocess_does_not_see_parent_secrets(monkeypatch, tmp_path):
    monkeypatch.setenv("VIGIA_TEST_SECRET", "s3cr3t")
    monkeypatch.chdir(tmp_path)
    code = "import os\ndef env():\n    return dict(os.environ)\n"
    tests = (
        "import os, sys\n"
        "assert 'VIGIA_TEST_SECRET' not in env()\n"
        # LC_CTYPE lo agrega el propio Python en Linux (coerción de locale, PEP 538), no se hereda.
        "assert {k.upper() for k in env()} <= {'PATH', 'SYSTEMROOT', 'LC_CTYPE'}\n"
        f"assert os.path.realpath(os.getcwd()) != os.path.realpath({str(tmp_path)!r})\n"
        "assert sys.flags.isolated and sys.flags.no_site\n"
    )
    result = run_tests(code, tests)
    assert result.passed, result


def test_stdin_is_closed():
    result = run_tests("def read():\n    return input()\n", "read()\n")
    assert result.status == "failed" and "EOFError" in result.detail


def test_large_output_does_not_break_the_result():
    result = run_tests("print('x' * 2_000_000)\n" + GOOD, TESTS)
    assert result.passed
