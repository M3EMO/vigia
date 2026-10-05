"""Datasets de calibración y evaluación: JSONL validado y MKQA-es con gold computable.

Formato de un item (una línea JSON por item, el mismo que acepta la plataforma):

- `id`: texto no vacío y único dentro del archivo.
- `messages` (lista de `{role, content}`) o `question` (texto). Con `question`, el prompt agrega la
  instrucción de respuesta corta en español (`DEFAULT_INSTRUCTION`).
- Exactamente una fuente de etiqueta:
  - `answers`: lista de alias gold. El modelo genera y la respuesta se compara con el gold.
  - `response` + `label`: respuesta dada (teacher forcing) con etiqueta 0 (correcta) o 1 (errónea).
  - `response` + `hallucinated_spans`: respuesta dada con fragmentos erróneos, como pares
    `[start, end]` en code points o como substrings que aparecen una sola vez.
- Opcionales: `group` (id de pregunta para el split; por defecto `id`), `type` (tipo de pregunta),
  `evidence` (texto de referencia para el juez).
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from vigia.contracts import VigiaError

DEFAULT_INSTRUCTION = "Respondé solo con la respuesta, en pocas palabras, sin explicar."
DEFAULT_MKQA_PATH = Path(".cache") / "data" / "mkqa_es.jsonl"
MAX_ITEMS = 100_000
MAX_LINE_BYTES = 1_000_000
MAX_TEXT_CHARS = 50_000
MAX_ID_CHARS = 128
MAX_MESSAGES = 64
MAX_ANSWERS = 64
ROLES = ("system", "developer", "user", "assistant")
LABEL_MODES = ("gold", "label", "spans")
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

    El item normalizado tiene siempre: id, group, messages, question, mode (gold | label | spans),
    answers, response, label, hallucinated_spans, type y evidence (None si no aplica).
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
    if has_question:
        question = _text(obj["question"], "question", where)
        messages = question_messages(question)
    else:
        messages = _messages(obj["messages"], where)
        question = next(
            m["content"] for m in reversed(messages) if m["role"] == "user" and m["content"].strip()
        )

    has_answers, has_response = "answers" in obj, "response" in obj
    has_label, has_spans = "label" in obj, "hallucinated_spans" in obj
    answers = response = label = spans = None
    if has_answers and not (has_response or has_label or has_spans):
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
    item_type = obj.get("type")
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
    if item["messages"] == question_messages(item["question"]):
        out["question"] = item["question"]
    else:
        out["messages"] = item["messages"]
    if item["group"] != item["id"]:
        out["group"] = item["group"]
    if item["mode"] == "gold":
        out["answers"] = item["answers"]
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


__all__ = [
    "DEFAULT_INSTRUCTION",
    "DEFAULT_MKQA_PATH",
    "LABEL_MODES",
    "DatasetError",
    "item_to_json",
    "items_sha256",
    "load_mkqa_es",
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
