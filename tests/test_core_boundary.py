"""Frontera open-core: el núcleo `vigia/` nunca importa la plataforma `vigia_ee` (ADR A-13, AT-P18)."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "vigia"
PLATFORM = "vigia_ee"
DYNAMIC_IMPORTERS = {"import_module", "__import__", "find_spec"}


def _is_platform(name: str | None) -> bool:
    return bool(name) and (name == PLATFORM or name.startswith(PLATFORM + "."))


def platform_imports(source: str, filename: str = "<memoria>") -> list[str]:
    """Imports de `vigia_ee` en un archivo: estáticos, `from` e imports dinámicos con literal."""
    found = []
    for node in ast.walk(ast.parse(source, filename=filename)):
        if isinstance(node, ast.Import):
            found += [f"{filename}:{node.lineno} import {a.name}" for a in node.names if _is_platform(a.name)]
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and _is_platform(node.module):
                found.append(f"{filename}:{node.lineno} from {node.module}")
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name in DYNAMIC_IMPORTERS and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and _is_platform(arg.value):
                    found.append(f"{filename}:{node.lineno} {name}({arg.value!r})")
    return found


def test_AT_P18_core_never_imports_platform():
    """AT-P18: ningún archivo de vigia/ importa vigia_ee (análisis AST de todo el árbol)."""
    files = sorted(CORE.rglob("*.py"))
    assert len(files) > 10, "el barrido no encontró el núcleo"
    violations = []
    for path in files:
        violations += platform_imports(path.read_text(encoding="utf-8"), str(path.relative_to(ROOT)))
    assert violations == []


def test_AT_P18_checker_detects_every_import_form():
    """AT-P18: el chequeo falla ante cada forma de importar la plataforma desde el núcleo."""
    bad = {
        "import vigia_ee": 1,
        "import vigia_ee.db as d": 1,
        "from vigia_ee import auth": 1,
        "from vigia_ee.runtime import Runtime": 1,
        "import importlib\nimportlib.import_module('vigia_ee.app')": 1,
        "__import__('vigia_ee')": 1,
        "def f():\n    from vigia_ee.panel import install_panel\n": 1,
    }
    for source, expected in bad.items():
        assert len(platform_imports(source)) == expected, source
    ok = "import vigia\nfrom vigia.api import create_api_router\nfrom . import contracts\nimport vigia_eex\n"
    assert platform_imports(ok) == []
