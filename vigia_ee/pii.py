"""Redacción de PII argentina, texto seguro para mostrar y celdas CSV neutralizadas.

`redact` corre sobre el texto completo (un DNI puede quedar partido entre tokens) y devuelve un
mapa de reemplazos; `map_offset` re-mapea offsets de tokens del texto original al redactado.

Detectores, en orden de prioridad (un hallazgo no se solapa con uno anterior):

| Marcador | Patrón | Validación |
|---|---|---|
| [EMAIL] | usuario@dominio.tld | formato |
| [CBU] | 22 dígitos | dos dígitos verificadores |
| [TARJETA] | 13 a 19 dígitos, con espacios o guiones | Luhn |
| [CUIT] | XX-XXXXXXXX-X (CUIT o CUIL) | prefijo conocido + módulo 11 |
| [IP] | IPv6 o IPv4 | `ipaddress` / rangos de octetos |
| [DNI] | 7-8 dígitos o dd.ddd.ddd | requiere contexto ("DNI", "documento") antes |
| [TELEFONO] | +código internacional, característica/11/15, o "tel:" + número | cantidad de dígitos |

El DNI exige contexto porque un número suelto de 8 dígitos (montos, poblaciones) no es un DNI
identificable y redactarlo rompe el texto. Un regex no cubre nombres ni fechas (HIPAA Safe Harbor).
"""

from __future__ import annotations

import bisect
import csv
import io
import ipaddress
import re
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

# --- validadores ----------------------------------------------------------------------------------

CUIT_PREFIXES = frozenset({"20", "23", "24", "27", "30", "33", "34"})
_CUIT_WEIGHTS = (5, 4, 3, 2, 7, 6, 5, 4, 3, 2)
_CBU_W1 = (7, 1, 3, 9, 7, 1, 3)
_CBU_W2 = (3, 9, 7, 1, 3, 9, 7, 1, 3, 9, 7, 1, 3)


def _digits(s: str) -> str:
    return "".join(ch for ch in s if "0" <= ch <= "9")


def cuit_is_valid(value: str) -> bool:
    """CUIT/CUIL de 11 dígitos con prefijo conocido y dígito verificador módulo 11."""
    d = _digits(value)
    if len(d) != 11 or d[:2] not in CUIT_PREFIXES:
        return False
    rest = 11 - sum(int(c) * w for c, w in zip(d[:10], _CUIT_WEIGHTS, strict=True)) % 11
    check = 0 if rest == 11 else rest
    return check != 10 and check == int(d[10])


def cbu_is_valid(value: str) -> bool:
    """CBU de 22 dígitos: bloque 1 (8 dígitos) y bloque 2 (14 dígitos), cada uno con su verificador."""
    d = _digits(value)
    if len(d) != 22:
        return False

    def check(block: str, weights: tuple[int, ...]) -> bool:
        total = sum(int(c) * w for c, w in zip(block[:-1], weights, strict=True))
        return (10 - total % 10) % 10 == int(block[-1])

    return check(d[:8], _CBU_W1) and check(d[8:], _CBU_W2)


def luhn_is_valid(value: str) -> bool:
    d = _digits(value)
    if not 13 <= len(d) <= 19:
        return False
    total = 0
    for i, ch in enumerate(reversed(d)):
        n = int(ch)
        if i % 2 == 1:
            n = n * 2 - 9 if n > 4 else n * 2
        total += n
    return total % 10 == 0


def _ipv6_is_valid(value: str) -> bool:
    if not any(c.isalnum() for c in value):
        return False
    try:
        ipaddress.IPv6Address(value)
    except ValueError:
        return False
    return True


# --- detectores -----------------------------------------------------------------------------------

_OCTET = r"(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])"


@dataclass(frozen=True)
class _Detector:
    kind: str
    pattern: re.Pattern[str]
    group: int = 0
    validate: Callable[[str], bool] = field(default=lambda _s: True)


DETECTORS: tuple[_Detector, ...] = (
    _Detector("EMAIL", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")),
    _Detector("CBU", re.compile(r"(?<![0-9])[0-9]{22}(?![0-9])"), validate=cbu_is_valid),
    _Detector(
        "TARJETA",
        re.compile(r"(?<![0-9])(?:[0-9][ -]?){12,18}[0-9](?![0-9])"),
        validate=luhn_is_valid,
    ),
    _Detector(
        "CUIT",
        re.compile(r"(?<![0-9])[0-9]{2}[-. ]?[0-9]{8}[-. ]?[0-9](?![0-9])"),
        validate=cuit_is_valid,
    ),
    _Detector(
        "IP",
        re.compile(r"(?<![0-9A-Za-z:])(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4}(?![0-9A-Za-z:])"),
        validate=_ipv6_is_valid,
    ),
    _Detector("IP", re.compile(rf"(?<![0-9.]){_OCTET}(?:\.{_OCTET}){{3}}(?![0-9]|\.[0-9])")),
    _Detector(
        "DNI",
        re.compile(
            r"(?i)(?:\bd\.?\s?n\.?\s?i\b\.?|\bdocumento\b|\bdoc\.)[^0-9\n]{0,25}?"
            r"(?<![0-9.])([0-9]{1,2}\.[0-9]{3}\.[0-9]{3}|[0-9]{7,8})(?![0-9])"
        ),
        group=1,
    ),
    _Detector(
        "TELEFONO",
        re.compile(r"(?<![0-9A-Za-z+])\+[0-9]{1,3}(?:[ .-]?\(?[0-9]{1,4}\)?){2,6}(?![0-9])"),
        validate=lambda s: 8 <= len(_digits(s)) <= 15,
    ),
    _Detector(
        "TELEFONO",
        re.compile(
            r"(?<![0-9A-Za-z])(?:\(?0[0-9]{2,4}\)?[ .-]?(?:15[ .-]?)?|(?:11|15)[ .-]?)"
            r"[0-9]{3,4}[ .-]?[0-9]{4}(?![0-9])"
        ),
        validate=lambda s: 10 <= len(_digits(s)) <= 13,
    ),
    _Detector(
        "TELEFONO",
        re.compile(
            r"(?i)\b(?:tel(?:[ée]fono)?|cel(?:ular)?|whatsapp|m[óo]vil)\b\.?:?\s*"
            r"(?<![0-9])([0-9]{4}[ .-]?[0-9]{4})(?![0-9])"
        ),
        group=1,
    ),
)

KINDS = tuple(dict.fromkeys(d.kind for d in DETECTORS))


@dataclass(frozen=True)
class Replacement:
    """Un hallazgo: `[start, end)` en el texto original y `[new_start, new_end)` en el redactado."""

    start: int
    end: int
    new_start: int
    new_end: int
    kind: str


@dataclass(frozen=True)
class Redaction:
    text: str
    counts: dict[str, int]
    replacements: tuple[Replacement, ...]


def find_pii(text: str) -> list[tuple[int, int, str]]:
    """Hallazgos `(start, end, kind)` ordenados y sin solapes."""
    starts: list[int] = []
    found: list[tuple[int, int, str]] = []
    for det in DETECTORS:
        for m in det.pattern.finditer(text):
            s, e = m.span(det.group)
            if not det.validate(m.group(det.group)):
                continue
            i = bisect.bisect_left(starts, s)
            if i > 0 and found[i - 1][1] > s:
                continue
            if i < len(found) and found[i][0] < e:
                continue
            starts.insert(i, s)
            found.insert(i, (s, e, det.kind))
    return found


def redact_with_map(text: str) -> Redaction:
    """Reemplaza cada hallazgo por `[KIND]` y devuelve texto, conteos y mapa de offsets."""
    parts: list[str] = []
    counts: dict[str, int] = {}
    reps: list[Replacement] = []
    cursor = 0
    out_len = 0
    for s, e, kind in find_pii(text):
        chunk = text[cursor:s]
        parts.append(chunk)
        out_len += len(chunk)
        marker = f"[{kind}]"
        parts.append(marker)
        reps.append(Replacement(s, e, out_len, out_len + len(marker), kind))
        out_len += len(marker)
        counts[kind] = counts.get(kind, 0) + 1
        cursor = e
    parts.append(text[cursor:])
    return Redaction("".join(parts), counts, tuple(reps))


def redact(text: str) -> tuple[str, dict[str, int]]:
    """Texto con marcadores tipo `[CUIT]` y conteo por tipo de dato."""
    r = redact_with_map(text)
    return r.text, r.counts


def map_offset(replacements: Sequence[Replacement], pos: int) -> int:
    """Re-mapea un offset del texto original al redactado.

    Un offset dentro de un hallazgo cae al final del marcador. Así, si un DNI ocupa tres tokens,
    el primero se queda con el marcador completo y los siguientes quedan vacíos: los textos de los
    tokens re-mapeados concatenan exactamente al texto redactado, sin huecos ni solapes.
    """
    shift = 0
    for r in replacements:
        if pos <= r.start:
            break
        if pos < r.end:
            return r.new_end
        shift = r.new_end - r.end
    return pos + shift


# --- texto seguro para mostrar --------------------------------------------------------------------

_BIDI = frozenset("؜‎‏‪‫‬‭‮⁦⁧⁨⁩")
_INVISIBLE = frozenset("​⁠﻿")


def safe_display(text: str) -> str:
    """Quita controles bidi, caracteres de control (salvo \\n y \\t), invisibles y tags Unicode.

    No escapa HTML: eso lo hace el autoescape de Jinja al renderizar. Conserva ZWJ (emojis).
    """
    out = []
    for ch in text:
        if ch in _BIDI or ch in _INVISIBLE or "\U000e0000" <= ch <= "\U000e007f":
            continue
        if ch not in "\n\t" and unicodedata.category(ch) == "Cc":
            continue
        out.append(ch)
    return "".join(out)


# --- CSV ------------------------------------------------------------------------------------------

CSV_DANGEROUS_PREFIXES = ("=", "+", "-", "@", "\t", "\r", "＝", "＋", "－", "＠")


def neutralize_csv_cell(value: Any) -> Any:
    """Prefija con `'` un texto que una planilla interpretaría como fórmula. Números sin cambio."""
    if isinstance(value, str) and value.lstrip(" ").startswith(CSV_DANGEROUS_PREFIXES):
        return "'" + value
    return value


def csv_dumps(rows: Iterable[Sequence[Any]], header: Sequence[str] | None = None) -> str:
    """CSV con todas las celdas neutralizadas (incluido el encabezado)."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\r\n")
    if header is not None:
        writer.writerow([neutralize_csv_cell(h) for h in header])
    for row in rows:
        writer.writerow([neutralize_csv_cell(v) for v in row])
    return buf.getvalue()
