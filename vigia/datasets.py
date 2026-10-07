"""Datasets de calibración y evaluación: JSONL validado, MKQA-es y HumanEval con gold computable.

Formato de un item (una línea JSON por item, el mismo que acepta la plataforma):

- `id`: texto no vacío y único dentro del archivo.
- `messages` (lista de `{role, content}`) o `question` (texto). Con `question`, el prompt agrega la
  instrucción de respuesta corta en español (`DEFAULT_INSTRUCTION`).
- Exactamente una fuente de etiqueta:
  - `answers`: lista de alias gold. El modelo genera y la respuesta se compara con el gold.
  - `response` + `label`: respuesta dada (teacher forcing) con etiqueta 0 (correcta) o 1 (errónea).
  - `response` + `hallucinated_spans`: respuesta dada con fragmentos erróneos, como pares
    `[start, end]` en code points o como substrings que aparecen una sola vez.
  - `tests` + `entry_point` (modo `code`): el modelo genera código y la etiqueta sale de correr
    `tests` (código Python con asserts que llama a `entry_point`) en un subprocess
    (`vigia.codeexec`). Con `question`, el prompt es la pregunta sola, sin la instrucción de
    respuesta corta. `type` vale `code` si falta. Correr estos items exige permiso explícito.
- Opcionales: `group` (id de pregunta para el split; por defecto `id`), `type` (tipo de pregunta),
  `evidence` (texto de referencia para el juez).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import keyword
import os
import random
import re
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from vigia.contracts import VigiaError

DEFAULT_INSTRUCTION = "Respondé solo con la respuesta, en pocas palabras, sin explicar."
DEFAULT_MKQA_PATH = Path(".cache") / "data" / "mkqa_es.jsonl"
HUMANEVAL_URL = "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"
DEFAULT_HUMANEVAL_RAW = Path(".cache") / "data" / "HumanEval.jsonl.gz"
DEFAULT_HUMANEVAL_PATH = Path(".cache") / "data" / "humaneval_es.jsonl"
MAX_HUMANEVAL_BYTES = 5_000_000
MAX_ITEMS = 100_000
MAX_LINE_BYTES = 1_000_000
MAX_TEXT_CHARS = 50_000
MAX_ID_CHARS = 128
MAX_MESSAGES = 64
MAX_ANSWERS = 64
ROLES = ("system", "developer", "user", "assistant")
LABEL_MODES = ("gold", "label", "spans", "code")
_KNOWN_KEYS = frozenset(
    {
        "id",
        "messages",
        "question",
        "answers",
        "response",
        "label",
        "hallucinated_spans",
        "group",
        "type",
        "evidence",
        "tests",
        "entry_point",
    }
)


class DatasetError(VigiaError, ValueError):
    """El dataset no cumple el formato. El mensaje indica la línea y el campo."""


def _text(value: Any, field: str, where: str, *, max_chars: int = MAX_TEXT_CHARS) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetError(f"{where}: '{field}' debe ser un texto no vacío")
    if len(value) > max_chars:
        raise DatasetError(f"{where}: '{field}' supera {max_chars} caracteres")
    return value


def question_messages(question: str, instruction: str = DEFAULT_INSTRUCTION) -> list[dict[str, str]]:
    """Prompt de QA corto en español: la pregunta y la instrucción en un solo mensaje de usuario."""
    return [{"role": "user", "content": f"{question.strip()}\n\n{instruction}"}]


def code_messages(question: str) -> list[dict[str, str]]:
    """Prompt de un item de código: la pregunta sola (ya dice cómo responder)."""
    return [{"role": "user", "content": question.strip()}]


def _default_messages(question: str, mode: str) -> list[dict[str, str]]:
    return code_messages(question) if mode == "code" else question_messages(question)


def _tests(value: Any, where: str) -> str:
    tests = _text(value, "tests", where)
    try:
        compile(tests, "<tests>", "exec")  # solo compila: no ejecuta nada
    except (SyntaxError, ValueError, RecursionError, MemoryError) as exc:
        raise DatasetError(f"{where}: 'tests' no es código Python válido ({type(exc).__name__})") from None
    return tests


def _entry_point(value: Any, where: str) -> str:
    name = _text(value, "entry_point", where, max_chars=MAX_ID_CHARS).strip()
    if not name.isidentifier() or keyword.iskeyword(name):
        raise DatasetError(f"{where}: 'entry_point' debe ser un nombre de función de Python")
    return name


def _messages(value: Any, where: str) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise DatasetError(f"{where}: 'messages' debe ser una lista no vacía")
    if len(value) > MAX_MESSAGES:
        raise DatasetError(f"{where}: 'messages' admite hasta {MAX_MESSAGES} mensajes")
    out = []
    for n, m in enumerate(value):
        if not isinstance(m, Mapping) or m.get("role") not in ROLES or not isinstance(m.get("content"), str):
            raise DatasetError(
                f"{where}: messages[{n}] debe tener 'role' en {list(ROLES)} y 'content' de tipo texto"
            )
        if len(m["content"]) > MAX_TEXT_CHARS:
            raise DatasetError(f"{where}: messages[{n}].content supera {MAX_TEXT_CHARS} caracteres")
        role = "system" if m["role"] == "developer" else m["role"]
        out.append({"role": role, "content": m["content"]})
    if not any(m["role"] == "user" and m["content"].strip() for m in out):
        raise DatasetError(f"{where}: 'messages' necesita al menos un mensaje de usuario con texto")
    return out


def _spans(value: Any, response: str, where: str) -> list[list[int]]:
    if not isinstance(value, list):
        raise DatasetError(f"{where}: 'hallucinated_spans' debe ser una lista")
    out: list[list[int]] = []
    for n, s in enumerate(value):
        if isinstance(s, str):
            if not s:
                raise DatasetError(f"{where}: hallucinated_spans[{n}] está vacío")
            first = response.find(s)
            if first < 0:
                raise DatasetError(f"{where}: hallucinated_spans[{n}] no aparece en 'response'")
            if response.find(s, first + 1) >= 0:
                raise DatasetError(
                    f"{where}: hallucinated_spans[{n}] aparece varias veces; usá offsets [start, end]"
                )
            start, end = first, first + len(s)
        elif (
            isinstance(s, list)
            and len(s) == 2
            and all(isinstance(x, int) and not isinstance(x, bool) for x in s)
        ):
            start, end = s
        else:
            raise DatasetError(f"{where}: hallucinated_spans[{n}] debe ser [start, end] o un substring")
        if not 0 <= start < end <= len(response):
            raise DatasetError(
                f"{where}: hallucinated_spans[{n}] = [{start}, {end}] está fuera de 'response'"
            )
        out.append([start, end])
    out.sort()
    for a, b in zip(out, out[1:], strict=False):
        if b[0] < a[1]:
            raise DatasetError(f"{where}: 'hallucinated_spans' tiene fragmentos solapados")
    return out


def validate_item(obj: Any, *, where: str = "item") -> dict[str, Any]:
    """Valida un item y lo devuelve normalizado (ver docstring del módulo).

    El item normalizado tiene siempre: id, group, messages, question, mode (gold | label | spans |
    code), answers, response, label, hallucinated_spans, tests, entry_point, type y evidence (None
    si no aplica).
    """
    if not isinstance(obj, Mapping):
        raise DatasetError(f"{where}: cada línea debe ser un objeto JSON")
    unknown = sorted(set(obj) - _KNOWN_KEYS)
    if unknown:
        raise DatasetError(f"{where}: campos desconocidos {unknown}")
    item_id = obj.get("id")
    if isinstance(item_id, int) and not isinstance(item_id, bool):
        item_id = str(item_id)
    item_id = _text(item_id, "id", where, max_chars=MAX_ID_CHARS)
    where = f"{where} (id {item_id!r})"
    has_messages, has_question = "messages" in obj, "question" in obj
    if has_messages == has_question:
        raise DatasetError(f"{where}: hace falta 'messages' o 'question', y solo uno")
    has_answers, has_response = "answers" in obj, "response" in obj
    has_label, has_spans = "label" in obj, "hallucinated_spans" in obj
    has_tests, has_entry = "tests" in obj, "entry_point" in obj
    is_code = has_tests or has_entry
    if has_question:
        question = _text(obj["question"], "question", where)
        messages = _default_messages(question, "code" if is_code else "gold")
    else:
        messages = _messages(obj["messages"], where)
        question = next(
            m["content"] for m in reversed(messages) if m["role"] == "user" and m["content"].strip()
        )

    answers = response = label = spans = tests = entry_point = None
    if is_code:
        if not (has_tests and has_entry) or has_answers or has_response or has_label or has_spans:
            raise DatasetError(
                f"{where}: un item de código lleva 'tests' y 'entry_point' juntos, "
                "sin 'answers', 'response', 'label' ni 'hallucinated_spans'"
            )
        tests = _tests(obj["tests"], where)
        entry_point = _entry_point(obj["entry_point"], where)
        mode = "code"
    elif has_answers and not (has_response or has_label or has_spans):
        raw = obj["answers"]
        raw = [raw] if isinstance(raw, str) else raw
        if not isinstance(raw, list) or not raw or len(raw) > MAX_ANSWERS:
            raise DatasetError(f"{where}: 'answers' debe ser una lista de 1 a {MAX_ANSWERS} textos")
        answers = list(dict.fromkeys(_text(a, "answers", where).strip() for a in raw))
        mode = "gold"
    elif has_response and not has_answers and has_label != has_spans:
        response = _text(obj["response"], "response", where)
        if has_label:
            label = obj["label"]
            if isinstance(label, bool) or label not in (0, 1):
                raise DatasetError(f"{where}: 'label' debe ser 0 (correcta) o 1 (errónea)")
            label = int(label)
            mode = "label"
        else:
            spans = _spans(obj["hallucinated_spans"], response, where)
            mode = "spans"
    else:
        raise DatasetError(
            f"{where}: la etiqueta debe ser 'answers', o 'response' con 'label', "
            "o 'response' con 'hallucinated_spans' (exactamente una)"
        )

    group = obj.get("group", item_id)
    if isinstance(group, int) and not isinstance(group, bool):
        group = str(group)
    group = _text(group, "group", where, max_chars=MAX_ID_CHARS)
    item_type = obj.get("type", "code" if mode == "code" else None)
    if item_type is not None:
        item_type = _text(item_type, "type", where, max_chars=64)
    evidence = obj.get("evidence")
    if evidence is not None:
        evidence = _text(evidence, "evidence", where)
    return {
        "id": item_id,
        "group": group,
        "messages": messages,
        "question": question,
        "mode": mode,
        "answers": answers,
        "response": response,
        "label": label,
        "hallucinated_spans": spans,
        "tests": tests,
        "entry_point": entry_point,
        "type": item_type,
        "evidence": evidence,
    }


def _reject_constant(name: str) -> Any:
    raise ValueError(f"constante JSON no permitida: {name}")


def parse_jsonl(source: str | bytes | Iterable[str], *, max_items: int = MAX_ITEMS) -> list[dict[str, Any]]:
    """Parsea y valida un JSONL completo. Falla en la primera línea inválida con su número."""
    if isinstance(source, bytes):
        try:
            source = source.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise DatasetError("el archivo no está en UTF-8") from exc
    lines = source.splitlines() if isinstance(source, str) else source
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for n, line in enumerate(lines, start=1):
        if n == 1:
            line = line.lstrip("﻿")
        if not line.strip():
            continue
        if len(line.encode("utf-8")) > MAX_LINE_BYTES:
            raise DatasetError(f"línea {n}: supera {MAX_LINE_BYTES} bytes")
        try:
            obj = json.loads(line, parse_constant=_reject_constant)
        except ValueError as exc:
            raise DatasetError(f"línea {n}: JSON inválido") from exc
        except RecursionError:
            raise DatasetError(f"línea {n}: JSON inválido (anidamiento excesivo)") from None
        item = validate_item(obj, where=f"línea {n}")
        if item["id"] in seen:
            raise DatasetError(f"línea {n}: id repetido {item['id']!r}")
        seen.add(item["id"])
        items.append(item)
        if len(items) > max_items:
            raise DatasetError(f"el dataset supera {max_items} items")
    if not items:
        raise DatasetError("el dataset no tiene items")
    return items


def read_jsonl(path: str | Path, *, max_items: int = MAX_ITEMS) -> list[dict[str, Any]]:
    """Lee y valida un archivo JSONL (ver `parse_jsonl`)."""
    if not Path(path).is_file():
        raise DatasetError(f"no existe el dataset {path}")
    with open(path, encoding="utf-8-sig") as fh:
        return parse_jsonl(fh, max_items=max_items)


def item_to_json(item: Mapping[str, Any]) -> dict[str, Any]:
    """Item normalizado -> objeto JSONL del formato público (inverso de `validate_item`)."""
    out: dict[str, Any] = {"id": item["id"]}
    if item["messages"] == _default_messages(item["question"], item["mode"]):
        out["question"] = item["question"]
    else:
        out["messages"] = item["messages"]
    if item["group"] != item["id"]:
        out["group"] = item["group"]
    if item["mode"] == "gold":
        out["answers"] = item["answers"]
    elif item["mode"] == "code":
        out["tests"] = item["tests"]
        out["entry_point"] = item["entry_point"]
    else:
        out["response"] = item["response"]
        if item["mode"] == "label":
            out["label"] = item["label"]
        else:
            out["hallucinated_spans"] = item["hallucinated_spans"]
    for key in ("type", "evidence"):
        if item.get(key) is not None:
            out[key] = item[key]
    return out


def write_jsonl(path: str | Path, items: Iterable[Mapping[str, Any]]) -> None:
    """Escribe items normalizados en formato JSONL público."""
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for item in items:
            fh.write(json.dumps(item_to_json(item), ensure_ascii=False) + "\n")


def items_sha256(items: Sequence[Mapping[str, Any]]) -> str:
    """Hash estable del contenido normalizado de los items (para el manifiesto de una corrida)."""
    h = hashlib.sha256()
    for item in items:
        h.update(json.dumps(item_to_json(item), ensure_ascii=False, sort_keys=True).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def stratified_sample(
    items: Sequence[Mapping[str, Any]], n: int, *, key: str = "type", seed: int = 0
) -> list[dict[str, Any]]:
    """Muestra `n` items con asignación proporcional por `key` (resto mayor) y semilla fija.

    El orden de salida es aleatorio pero reproducible. Si `n` supera el total, devuelve todos.
    """
    if n < 1:
        raise ValueError("n debe ser >= 1")
    rng = random.Random(seed)  # noqa: S311 - muestreo reproducible, no criptográfico
    strata: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        strata.setdefault(str(item.get(key)), []).append(dict(item))
    total = sum(len(v) for v in strata.values())
    n = min(n, total)
    names = sorted(strata)
    quotas = {s: n * len(strata[s]) / total for s in names}
    alloc = {s: int(quotas[s]) for s in names}
    left = n - sum(alloc.values())
    for s in sorted(names, key=lambda s: (-(quotas[s] - alloc[s]), s))[:left]:
        alloc[s] += 1
    chosen: list[dict[str, Any]] = []
    for s in names:
        pool = strata[s]
        chosen.extend(rng.sample(pool, alloc[s]))
    rng.shuffle(chosen)
    return chosen


def load_mkqa_es(
    path: str | Path = DEFAULT_MKQA_PATH,
    *,
    n: int | None = None,
    seed: int = 0,
    types: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """MKQA-es convertido (`{id, question, answers, type}` por línea) como items validados.

    Descarta items sin alias útiles. Con `n`, muestrea estratificado por `type` con `seed`.
    """
    path = Path(path)
    if not path.is_file():
        raise DatasetError(f"no existe {path}: convertí MKQA-es a JSONL antes de usarlo")
    wanted = set(types) if types else None
    items: list[dict[str, Any]] = []
    with open(path, encoding="utf-8-sig") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line, parse_constant=_reject_constant)
            except ValueError as exc:
                raise DatasetError(f"{path.name} línea {line_no}: JSON inválido") from exc
            except RecursionError:
                raise DatasetError(
                    f"{path.name} línea {line_no}: JSON inválido (anidamiento excesivo)"
                ) from None
            if not isinstance(raw, Mapping):
                raise DatasetError(f"{path.name} línea {line_no}: se esperaba un objeto")
            if wanted is not None and raw.get("type") not in wanted:
                continue
            answers = [a for a in raw.get("answers") or [] if isinstance(a, str) and a.strip()]
            if not answers:
                continue
            items.append(
                validate_item(
                    {
                        "id": raw.get("id"),
                        "question": raw.get("question"),
                        "answers": answers,
                        "type": raw.get("type"),
                    },
                    where=f"{path.name} línea {line_no}",
                )
            )
    if not items:
        raise DatasetError(f"{path} no tiene items utilizables")
    return stratified_sample(items, n, seed=seed) if n is not None else items


def humaneval_question(prompt: str, entry_point: str) -> str:
    """Pregunta en español que pide implementar la función del prompt de HumanEval."""
    return (
        f"Implementá en Python la función `{entry_point}` según su firma y su docstring:\n\n"
        f"```python\n{prompt.strip()}\n```\n\n"
        "Respondé SOLO con un bloque ```python``` que contenga la función completa, con los imports "
        "que necesite, sin explicaciones."
    )


def humaneval_item(raw: Mapping[str, Any], *, where: str = "HumanEval") -> dict[str, Any]:
    """Item de código validado a partir de una línea oficial de HumanEval.

    Usa `task_id`, `prompt`, `test` (define `check(candidate)`) y `entry_point`. Los tests son
    `test` más la llamada `check(<entry_point>)`.
    """
    if not isinstance(raw, Mapping):
        raise DatasetError(f"{where}: se esperaba un objeto")
    fields = {k: raw.get(k) for k in ("task_id", "prompt", "test", "entry_point")}
    missing = [k for k, v in fields.items() if not isinstance(v, str) or not v.strip()]
    if missing:
        raise DatasetError(f"{where}: faltan campos de HumanEval {missing}")
    entry = _entry_point(fields["entry_point"], where)
    return validate_item(
        {
            "id": fields["task_id"],
            "question": humaneval_question(fields["prompt"], entry),
            "tests": fields["test"] + f"\ncheck({entry})\n",
            "entry_point": entry,
            "type": "code",
        },
        where=where,
    )


def mbpp_item(raw: Mapping[str, Any], *, where: str = "MBPP") -> dict[str, Any]:
    """Item de código a partir de una línea de MBPP (`text`, `test_list`, `test_setup_code`).

    La función a implementar es la que llaman los asserts. La pregunta muestra el primer assert
    como ejemplo de firma, igual que el protocolo original de MBPP.
    """
    if not isinstance(raw, Mapping):
        raise DatasetError(f"{where}: se esperaba un objeto")
    text, tests = raw.get("text"), raw.get("test_list")
    if not isinstance(text, str) or not text.strip() or not isinstance(tests, list) or not tests:
        raise DatasetError(f"{where}: faltan 'text' o 'test_list'")
    match = re.match(r"\s*assert\s+(?:not\s+)?\(?\s*(?:set\()?\s*([A-Za-z_]\w*)\s*\(", str(tests[0]))
    if not match:
        raise DatasetError(f"{where}: no se encuentra la función en el primer assert")
    entry = _entry_point(match.group(1), where)
    return validate_item(
        {
            "id": f"MBPP/{raw.get('task_id')}",
            "question": (
                f"Implementá en Python la función `{entry}`. {text.strip()}\n\n"
                f"Tiene que pasar este test:\n```python\n{tests[0]}\n```\n\n"
                "Respondé SOLO con un bloque ```python``` que contenga la función completa, con los "
                "imports que necesite, sin explicaciones."
            ),
            "tests": "\n".join([str(raw.get("test_setup_code") or ""), *map(str, tests)]) + "\n",
            "entry_point": entry,
            "type": "code",
        },
        where=where,
    )


def load_mbpp(path: str | Path) -> list[dict[str, Any]]:
    """Lee el mbpp.jsonl oficial (licencia CC BY 4.0) y arma items de código. Omite líneas inválidas."""
    items = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            try:
                items.append(mbpp_item(json.loads(line), where=f"MBPP línea {line_no}"))
            except DatasetError:
                continue  # ponytail: unas pocas tareas tienen asserts raros; se omiten
    return items


def load_humaneval(
    path: str | Path = DEFAULT_HUMANEVAL_RAW, n: int | None = None, seed: int = 0
) -> list[dict[str, Any]]:
    """Lee el HumanEval.jsonl.gz oficial (o el .jsonl sin comprimir) y arma items de código.

    Con `n`, muestrea `n` items con `seed` (orden reproducible). Sin `n`, respeta el orden original.
    """
    path = Path(path)
    if not path.is_file():
        raise DatasetError(f"no existe {path}: descargalo con prepare_humaneval()")
    opener = gzip.open if path.suffix == ".gz" else open
    items: list[dict[str, Any]] = []
    with opener(path, "rt", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            if len(line) > MAX_LINE_BYTES:
                raise DatasetError(f"{path.name} línea {line_no}: supera {MAX_LINE_BYTES} caracteres")
            try:
                raw = json.loads(line, parse_constant=_reject_constant)
            except ValueError as exc:
                raise DatasetError(f"{path.name} línea {line_no}: JSON inválido") from exc
            items.append(humaneval_item(raw, where=f"{path.name} línea {line_no}"))
    if not items:
        raise DatasetError(f"{path} no tiene items")
    if len({it["id"] for it in items}) != len(items):
        raise DatasetError(f"{path} tiene task_id repetidos")
    return stratified_sample(items, n, seed=seed) if n is not None else items


def download_humaneval(
    dest: str | Path = DEFAULT_HUMANEVAL_RAW, *, url: str = HUMANEVAL_URL, timeout: float = 60.0
) -> Path:
    """Descarga HumanEval.jsonl.gz (licencia MIT) si no existe. Escritura atómica con tope de tamaño."""
    dest = Path(dest)
    if dest.is_file():
        return dest
    if not url.startswith("https://"):
        raise DatasetError("la URL de HumanEval debe ser https")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 - https fijo
        data = resp.read(MAX_HUMANEVAL_BYTES + 1)
    if len(data) > MAX_HUMANEVAL_BYTES:
        raise DatasetError(f"la descarga de HumanEval supera {MAX_HUMANEVAL_BYTES} bytes")
    tmp.write_bytes(data)
    os.replace(tmp, dest)
    return dest


def prepare_humaneval(
    raw: str | Path = DEFAULT_HUMANEVAL_RAW, out: str | Path = DEFAULT_HUMANEVAL_PATH
) -> Path:
    """Descarga HumanEval si falta y lo convierte a JSONL de items de código en español."""
    items = load_humaneval(download_humaneval(raw))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    write_jsonl(tmp, items)
    os.replace(tmp, out)
    return out


__all__ = [
    "DEFAULT_HUMANEVAL_PATH",
    "DEFAULT_HUMANEVAL_RAW",
    "DEFAULT_INSTRUCTION",
    "DEFAULT_MKQA_PATH",
    "HUMANEVAL_URL",
    "LABEL_MODES",
    "DatasetError",
    "code_messages",
    "download_humaneval",
    "humaneval_item",
    "item_to_json",
    "items_sha256",
    "load_humaneval",
    "load_mbpp",
    "mbpp_item",
    "load_mkqa_es",
    "prepare_humaneval",
    "parse_jsonl",
    "question_messages",
    "read_jsonl",
    "stratified_sample",
    "validate_item",
    "write_jsonl",
]


def arithmetic_items(n: int, *, seed: int = 0) -> list[dict[str, Any]]:
    """Items de aritmética con respuesta exacta computable (oráculo sin LLM).

    Sirven como set de control: el modelo acierta una parte y falla otra, así que las dos clases
    existen aun con modelos chicos. Mezcla sumas, restas y productos de 2 y 3 cifras.
    """
    if n < 1:
        raise ValueError("n debe ser >= 1")
    rng = random.Random(seed)  # noqa: S311 - generación reproducible, no criptográfica
    ops = (
        ("+", "más", lambda a, b: a + b),
        ("-", "menos", lambda a, b: a - b),
        ("×", "por", lambda a, b: a * b),
    )
    items: list[dict[str, Any]] = []
    for i in range(n):
        symbol, word, fn = ops[i % len(ops)]
        hi = 99 if symbol == "×" else 999
        a, b = rng.randint(11, hi), rng.randint(11, hi)
        question = f"¿Cuánto es {a} {symbol} {b}?" if i % 2 else f"¿Cuánto da {a} {word} {b}?"
        items.append(
            validate_item(
                {
                    "id": f"arit-{seed}-{i}",
                    "question": question,
                    "answers": [str(fn(a, b))],
                    "type": f"arit{symbol}",
                },
                where=f"aritmética {i}",
            )
        )
    return items
