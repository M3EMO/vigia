"""Experimento de evidencia: generar, etiquetar, calibrar y evaluar de punta a punta (AT-M11).

Pipeline de `run_evidence`:

1. Extracción. Cada item con `answers` se genera con `backend.stream` (greedy). Cada item con
   `response` pasa por `backend.teacher_force`. Cada respuesta deja una fila en `rows.jsonl` y su
   matriz de features en `chunks/`. La corrida es reanudable: una fila escrita no se recalcula y una
   línea final cortada se descarta. El manifiesto (`manifest.json`) impide mezclar corridas con otro
   modelo, otros niveles, otra semilla u otra fuente de etiquetas. Cada fila guarda el hash de su
   item: si un id ya escrito cambió de contenido, la reanudación falla en lugar de reusar la fila.
2. Etiquetas. Gold: un item tipado (número, fecha, aritmética; sin tipo, el que implica su gold)
   con gold legible como número o fecha es correcto solo si coincide el último número o la última
   fecha. Los demás items son
   correctos por EM, F1 de tokens >= 0,5, número o fecha (`vigia.labels`). Todos los tokens
   `final` de la respuesta llevan la etiqueta de la respuesta. `response + label`: igual.
   `hallucinated_spans`: etiqueta por solapamiento. Con un juez, el juez etiqueta cada respuesta y
   el gold queda como auditoría.
3. Matriz `features.npz`, detector F0+F1 (`detector.json`), reporte con IC por bootstrap de
   pregunta (`report.json`, `report.html`), benchmark de latencia opcional y `summary.txt`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import html
import json
import os
import re
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from vigia import charts
from vigia.calibrate import DEFAULT_ALPHAS, train_detector
from vigia.contracts import Backend, ModelInfo, Tier, TokenEvent, VigiaError, feature_tier
from vigia.datasets import DEFAULT_INSTRUCTION, items_sha256
from vigia.detector import Detector
from vigia.evaluate import SingleClassError, evaluate_report, render_report_html
from vigia.harmony import final_view
from vigia.labels import (
    IGNORE_LABEL,
    JUDGE_PROMPT_SHA256,
    DateParts,
    Judge,
    exact_match,
    extract_numbers,
    label_tokens_from_answer,
    label_with_judge,
    numeric_match,
    parse_date_parts_es,
    parse_dates_es,
    parse_number_es,
    spans_to_token_labels,
    token_f1,
)

RUN_SCHEMA_VERSION = "2"  # 2: cada fila guarda `item_sha256`
ROWS_FILE = "rows.jsonl"
FEATURES_FILE = "features.npz"
MANIFEST_FILE = "manifest.json"
CHUNKS_DIR = "chunks"
DETECTOR_FILE = "detector.json"
SPLIT_FILE = "split.json"
REPORT_JSON = "report.json"
REPORT_HTML = "report.html"
SUMMARY_FILE = "summary.txt"
BENCH_FILE = "bench.json"
DEFAULT_TIERS = (Tier.F0, Tier.F1, Tier.H)
GOLD_F1_THRESHOLD = 0.5
DETECTOR_TIERS = (Tier.F0, Tier.F1)
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_LEADING_NUMBER = re.compile(r"^\s*[-+−]?\d")
_NUMERIC_DMY = re.compile(r"\b\d{1,2}[/.-]\d{1,2}[/.-]\d{4}\b")
TYPED_ITEM_TYPES = ("number", "number_with_unit", "date")

ProgressCallback = Callable[[int, int, str], None]


class EvidenceError(VigiaError):
    """La corrida no puede seguir: manifiesto incompatible, datos insuficientes o archivo dañado."""


# --- Etiquetas computables -------------------------------------------------------------------------


def _gold_numbers(answers: Sequence[str], item_type: str | None) -> list[float]:
    values: list[float] = []
    for alias in answers:
        value = parse_number_es(alias.strip())
        if value is None and item_type in ("number", "number_with_unit") and _LEADING_NUMBER.match(alias):
            nums = extract_numbers(alias)
            value = nums[0] if nums else None
        if value is not None and value not in values:
            values.append(value)
    return values


def _gold_date_parts(answers: Sequence[str], *, partial: bool) -> list[DateParts]:
    """Fechas del gold: alias con una sola mención de fecha (`parse_date_parts_es`).

    Sin `partial`, solo fechas completas. Con `partial` (items `date`) también mes y año
    ("mayo de 1945") y año solo ("1945", "el año 1945"). Los formatos 08/11/2017 del gold son
    ambiguos (mes/día) y no se usan.
    """
    out: list[DateParts] = []
    for alias in (a.strip() for a in answers):
        if _NUMERIC_DMY.search(alias) and not _ISO_DATE.match(alias):
            continue
        mentions = parse_date_parts_es(alias)
        if len(mentions) == 1 and (partial or mentions[0][2] is not None) and mentions[0] not in out:
            out.append(mentions[0])
    return out


def _date_parts_match(prediction: str, gold: DateParts) -> bool:
    """La última mención de la predicción con al menos el detalle del gold coincide en sus partes.

    Gold de año solo: cuenta cualquier mención (fecha completa, mes y año o año suelto). Gold de
    mes y año: menciones con mes. Gold completo: fechas completas (igual que `date_match`).
    """
    year, month, day = gold
    eligible = [
        m for m in parse_date_parts_es(prediction) if (month is None or m[1]) and (day is None or m[2])
    ]
    if not eligible:
        return False
    last = eligible[-1]
    return last[0] == year and (month is None or last[1] == month) and (day is None or last[2] == day)


def _implicit_type(answers: Sequence[str]) -> str | None:
    """Tipo de un gold sin tipo: `date` si todos los alias son fechas legibles y no ambiguas con al
    menos mes y año, `number_with_unit` si todos empiezan con un número, tienen un solo número y
    ninguno tiene una fecha, o None.

    Un alias con varios números ("1810-1816", "2 a 1", "15:30", "3/4 de taza") queda sin tipo:
    la comparación del último número de la predicción con el primero del alias no sirve ahí, y
    decide `_text_match`, que exige todos los números del alias en el mismo orden.
    """
    aliases = [a.strip() for a in answers]
    if not aliases:
        return None
    dated = [a for a in aliases if any(p[1] for p in _gold_date_parts([a], partial=True))]
    if len(dated) == len(aliases):
        return "date"
    if all(
        _LEADING_NUMBER.match(a)
        and len(extract_numbers(a)) == 1
        and not _NUMERIC_DMY.search(a)
        and not parse_dates_es(a)
        for a in aliases
    ):
        return "number_with_unit"
    return None


def _is_typed(item_type: str | None) -> bool:
    """Item cuyo gold es un número o una fecha: tipo numérico o de fecha (dado o implícito) o aritmética."""
    return item_type in TYPED_ITEM_TYPES or (item_type or "").startswith("arit")


def _in_order(needle: Sequence[float], haystack: Sequence[float]) -> bool:
    """True si `needle` es una subsecuencia de `haystack` (mismo orden, no necesariamente contigua)."""
    rest = iter(haystack)
    return all(any(v == h for h in rest) for v in needle)


def _text_match(candidate: str, answers: Sequence[str]) -> bool:
    """EM o F1 de tokens >= 0,5 contra algún alias cuyos números estén todos en la predicción.

    El F1 trata "1945" y "1946" como palabras intercambiables: un alias con números solo cuenta si
    la predicción contiene esos números en el mismo orden, con repeticiones ("2 a 1" no acepta
    "1 a 2" y "1 a 1" no acepta "1 a 0"). Un alias sin números no cuenta si la predicción da
    números y ninguno coincide con los del gold (otro alias dice el número correcto).
    """
    predicted = extract_numbers(candidate)
    gold_numbers = {v for a in answers for v in extract_numbers(a)}
    contradicts = bool(predicted and gold_numbers and not set(predicted) & gold_numbers)
    for alias in answers:
        numbers = extract_numbers(alias)
        if (numbers and not _in_order(numbers, predicted)) or (not numbers and contradicts):
            continue
        if exact_match(candidate, [alias]) or token_f1(candidate, [alias]) >= GOLD_F1_THRESHOLD:
            return True
    return False


def gold_correct(
    prediction: str, answers: Sequence[str], item_type: str | None = None
) -> tuple[bool, dict[str, Any]]:
    """¿La predicción coincide con el gold? Evalúa el texto completo y su primera línea no vacía.

    Item tipado (`number`, `number_with_unit`, `date`, `arit*`) con gold legible como número o
    fecha: correcta solo si el último número coincide con un alias numérico (para
    `number_with_unit`, el número inicial del alias) o la última fecha coincide con un alias de
    fecha. Un item `date` acepta también gold parcial: mes y año ("mayo de 1945") o año solo
    ("1945"); se compara la última mención de la predicción con ese detalle, y un año solo se
    compara con el año de una fecha completa ("8 de mayo de 1945", "1945-05-08"). EM y F1 no
    deciden: la normalización borra la coma de "1,5" y el F1 premia la unidad o el mes aunque el
    número o el año difieran. Un item sin tipo recibe el tipo implícito de su gold
    (`_implicit_type`): fecha si todos los alias son fechas con mes y año, número con unidad si
    todos empiezan con un número y tienen uno solo. Resto de los items: correcta si hay EM o F1
    de tokens >= 0,5 contra un alias cuyos números aparecen en orden en la predicción
    (`_text_match`), coincidencia numérica o coincidencia de fecha completa.
    """
    text = prediction.strip()
    first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    candidates = list(dict.fromkeys(c for c in (text, first) if c))
    detail: dict[str, Any] = {"exact_match": False, "token_f1": 0.0, "numeric": False, "date": False}
    if not candidates:
        return False, detail
    detail["exact_match"] = any(exact_match(c, answers) for c in candidates)
    detail["token_f1"] = max(token_f1(c, answers) for c in candidates)
    kind = item_type or _implicit_type(answers)
    dates = _gold_date_parts(answers, partial=kind == "date")
    numbers = [] if kind == "date" and dates else _gold_numbers(answers, kind)
    detail["numeric"] = any(numeric_match(c, v) for c in candidates for v in numbers)
    detail["date"] = any(_date_parts_match(c, d) for c in candidates for d in dates)
    typed_decision = _is_typed(kind) and bool(numbers or dates)
    correct = (detail["numeric"] or detail["date"]) or (
        not typed_decision and any(_text_match(c, answers) for c in candidates)
    )
    return bool(correct), detail


def _tokens_for_labels(meta: Sequence[Mapping[str, Any]]) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(start=t["start"], end=t["end"], channel=t["channel"], is_special=t["special"])
        for t in meta
    ]


def _final_text(meta: Sequence[Mapping[str, Any]]) -> str:
    return "".join(t["text"] for t in meta if t["channel"] == "final" and not t["special"])


def computable_labels(row: Mapping[str, Any]) -> tuple[list[int], dict[str, Any]]:
    """Etiquetas por token de una fila según la fuente del item (gold, label o spans)."""
    item, meta = row["item"], row["tokens"]
    toks = _tokens_for_labels(meta)
    mode = item["mode"]
    if mode == "gold":
        correct, detail = gold_correct(row["response"], item["answers"], item.get("type"))
        return label_tokens_from_answer(toks, None, correct), {"correct": correct, "match": detail}
    if mode == "label":
        correct = item["label"] == 0
        return label_tokens_from_answer(toks, None, correct), {"correct": correct}
    # Con marcado harmony, los offsets del dataset se refieren a la respuesta con marcado y los de los
    # tokens al texto final: final_view lleva los spans al marco del texto final (sin marcado, no cambia).
    response, spans = final_view(item["response"], [tuple(s) for s in item["hallucinated_spans"]])
    if _final_text(meta) != response:
        # Los offsets del dataset se refieren a la respuesta dada; si el texto final difiere, no valen.
        return [IGNORE_LABEL] * len(meta), {"correct": None, "rejected": "texto_final_distinto"}
    return spans_to_token_labels(toks, spans), {"correct": not spans}


def judge_labels(row: Mapping[str, Any], judge: Judge) -> tuple[list[int], dict[str, Any]]:
    """Etiquetas del juez para una fila. Un veredicto rechazado deja todos los tokens en IGNORE."""
    item, meta = row["item"], row["tokens"]
    evidence = item.get("evidence")
    if evidence is None and item.get("answers"):
        evidence = "Respuestas correctas de referencia: " + "; ".join(item["answers"])
    report = label_with_judge(
        judge,
        [{"id": row["id"], "question": item["question"], "answer": row["response"], "evidence": evidence}],
    )
    if row["id"] in report.rejected:
        return [IGNORE_LABEL] * len(meta), {"correct": None, "rejected": report.rejected[row["id"]]}
    verdict = report.verdicts[row["id"]]
    labels = spans_to_token_labels(_tokens_for_labels(meta), list(verdict.spans))
    return labels, {"correct": verdict.label == 0, "judge_label": verdict.label}


def label_row(row: Mapping[str, Any], judge: Judge | None = None) -> tuple[list[int], dict[str, Any]]:
    """Etiquetas de una fila: computables, o del juez con la etiqueta computable como auditoría."""
    if not row["tokens"]:
        return [], {"correct": computable_labels(row)[1].get("correct")}
    labels, info = computable_labels(row)
    if judge is None:
        return labels, info
    j_labels, j_info = judge_labels(row, judge)
    j_info["computable_correct"] = info.get("correct")
    return j_labels, j_info


# --- Almacenamiento de la corrida ------------------------------------------------------------------


def _model_info_to_dict(info: ModelInfo) -> dict[str, Any]:
    data = dataclasses.asdict(info)
    data["moe_layers"] = list(info.moe_layers)
    return data


def model_info_from_dict(data: Mapping[str, Any]) -> ModelInfo:
    fields = {f.name for f in dataclasses.fields(ModelInfo)}
    values = {k: v for k, v in data.items() if k in fields}
    values["moe_layers"] = tuple(values.get("moe_layers", ()))
    return ModelInfo(**values)


def _same_model_info(a: Any, b: Any) -> bool:
    """Igualdad de `model_info` entre manifiestos. `router_sha256` se compara solo si los dos lo tienen:
    una corrida previa a ese campo se puede reanudar con el mismo modelo."""
    if not isinstance(a, Mapping) or not isinstance(b, Mapping):
        return a == b
    if a.get("router_sha256") is None or b.get("router_sha256") is None:
        a = {k: v for k, v in a.items() if k != "router_sha256"}
        b = {k: v for k, v in b.items() if k != "router_sha256"}
    return dict(a) == dict(b)


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def _chunk_name(item_id: str) -> str:
    return hashlib.sha256(item_id.encode("utf-8")).hexdigest()[:32] + ".npz"


class RunStore:
    """Archivos de una corrida en `out_dir`: manifiesto, filas (append-only) y chunks de features."""

    def __init__(self, out_dir: str | Path) -> None:
        self.dir = Path(out_dir)
        self.rows_path = self.dir / ROWS_FILE
        self.manifest_path = self.dir / MANIFEST_FILE
        self.chunks = self.dir / CHUNKS_DIR

    def manifest(self) -> dict[str, Any] | None:
        if not self.manifest_path.is_file():
            return None
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def open(self, manifest: Mapping[str, Any]) -> None:
        """Crea la corrida o verifica que la existente sea compatible con `manifest`."""
        self.dir.mkdir(parents=True, exist_ok=True)
        self.chunks.mkdir(exist_ok=True)
        current = self.manifest()
        if current is None:
            if self.rows_path.exists() and self.rows_path.stat().st_size > 0:
                raise EvidenceError(f"{self.rows_path} existe sin {MANIFEST_FILE}: usá otro directorio")
            _atomic_write_text(self.manifest_path, json.dumps(dict(manifest), ensure_ascii=False, indent=2))
            return
        keys = ("schema_version", "tiers", "max_tokens", "seed", "label_source", "judge")
        diff = [k for k in keys if current.get(k) != manifest.get(k)]
        if not _same_model_info(current.get("model_info"), manifest.get("model_info")):
            diff.insert(1, "model_info")
        if diff:
            raise EvidenceError(
                f"la corrida en {self.dir} se hizo con otra configuración ({', '.join(diff)}); "
                "usá otro directorio de salida para no mezclar resultados"
            )

    def load_rows(self) -> list[dict[str, Any]]:
        """Filas escritas. Repara una última línea cortada (corte durante la escritura)."""
        if not self.rows_path.is_file():
            return []
        data = self.rows_path.read_bytes()
        rows: list[dict[str, Any]] = []
        pos = good_end = 0
        while pos < len(data):
            nl = data.find(b"\n", pos)
            if nl < 0:
                break  # segmento final sin salto de línea: escritura cortada, se descarta
            line = data[pos:nl]
            if line.strip():
                try:
                    rows.append(json.loads(line.decode("utf-8")))
                except (ValueError, UnicodeDecodeError):
                    if data[nl + 1 :].strip():
                        raise EvidenceError(f"{self.rows_path} tiene una línea dañada en el medio") from None
                    break  # última línea dañada: se descarta
            pos = good_end = nl + 1
        if good_end < len(data):
            with open(self.rows_path, "r+b") as fh:
                fh.truncate(good_end)
        ids = [r["id"] for r in rows]
        if len(ids) != len(set(ids)):
            raise EvidenceError(f"{self.rows_path} tiene ids repetidos")
        return rows

    def append(self, row: Mapping[str, Any], X: np.ndarray, names: Sequence[str]) -> None:
        """Escribe el chunk de features (atómico) y después la fila con fsync."""
        if len(X):
            _atomic_savez(self.chunks / _chunk_name(row["id"]), X=X, names=np.asarray(list(names), dtype=str))
        line = json.dumps(dict(row), ensure_ascii=False, allow_nan=False) + "\n"
        with open(self.rows_path, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())

    def load_chunk(self, item_id: str) -> tuple[np.ndarray, list[str]]:
        path = self.chunks / _chunk_name(item_id)
        if not path.is_file():
            raise EvidenceError(f"falta el chunk de features de {item_id!r} ({path.name})")
        with np.load(path, allow_pickle=False) as z:
            return z["X"], z["names"].tolist()

    def rewrite_rows(self, rows: Iterable[Mapping[str, Any]]) -> None:
        text = "".join(json.dumps(dict(r), ensure_ascii=False, allow_nan=False) + "\n" for r in rows)
        _atomic_write_text(self.rows_path, text)


# --- Extracción ------------------------------------------------------------------------------------


def _token_meta(events: Sequence[TokenEvent]) -> list[dict[str, Any]]:
    return [
        {
            "i": e.index,
            "id": e.token_id,
            "text": e.text,
            "start": e.start,
            "end": e.end,
            "channel": e.channel,
            "special": e.is_special,
        }
        for e in events
    ]


def _feature_matrix(events: Sequence[TokenEvent]) -> tuple[np.ndarray, list[str]]:
    names = sorted({k for e in events for k in e.features})
    col = {n: j for j, n in enumerate(names)}
    X = np.full((len(events), len(names)), np.nan, dtype=np.float32)
    for i, e in enumerate(events):
        for k, v in e.features.items():
            X[i, col[k]] = v
    return X, names


def _check_tiers(backend: Backend, tiers: Sequence[Tier | str]) -> tuple[Tier, ...]:
    wanted = tuple(dict.fromkeys(Tier(t) for t in tiers))
    if not wanted:
        raise ValueError("hace falta al menos un nivel de features")
    available = backend.capabilities().tiers
    missing = [t.value for t in wanted if t not in available]
    if missing:
        raise EvidenceError(
            f"el backend no produce los niveles {missing}; disponibles: {[t.value for t in available]}"
        )
    return wanted


def build_manifest(
    backend: Backend,
    items: Sequence[Mapping[str, Any]],
    *,
    tiers: Sequence[Tier],
    max_tokens: int,
    seed: int,
    judge: Judge | None,
) -> dict[str, Any]:
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "model_info": _model_info_to_dict(backend.model_info()),
        "tiers": [t.value for t in tiers],
        "max_tokens": max_tokens,
        "seed": seed,
        "label_source": "judge" if judge is not None else "computable",
        "judge": None if judge is None else {"model": str(judge.model), "prompt_sha256": JUDGE_PROMPT_SHA256},
        "instruction": DEFAULT_INSTRUCTION,
        "dataset_sha256": items_sha256(items),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _run_item(
    backend: Backend, item: Mapping[str, Any], *, tiers: tuple[Tier, ...], max_tokens: int, seed: int
) -> tuple[list[TokenEvent], dict[str, Any]]:
    t0 = time.perf_counter()
    if item["mode"] == "gold":
        stream = backend.stream(
            item["messages"], max_tokens=max_tokens, temperature=0.0, seed=seed, tiers=tiers
        )
        try:
            events = list(stream)
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()
        text = getattr(stream, "text", None)
        info = {
            "source": "generated",
            "response": text if isinstance(text, str) else _final_text(_token_meta(events)),
            "finish_reason": getattr(stream, "finish_reason", None),
        }
    else:
        teacher_force = getattr(backend, "teacher_force", None)
        if not callable(teacher_force):
            raise EvidenceError("el backend no soporta teacher forcing y el dataset trae respuestas dadas")
        events = list(teacher_force(item["messages"], item["response"], tiers=tiers))
        info = {
            "source": "teacher_forced",
            "response": _final_text(_token_meta(events)),
            "finish_reason": None,
        }
    info["elapsed_s"] = time.perf_counter() - t0
    return events, info


def extract(
    backend: Backend,
    items: Sequence[Mapping[str, Any]],
    out_dir: str | Path,
    *,
    tiers: Sequence[Tier | str] = DEFAULT_TIERS,
    max_tokens: int = 24,
    seed: int = 0,
    judge: Judge | None = None,
    progress_cb: ProgressCallback | None = None,
) -> list[dict[str, Any]]:
    """Genera o fuerza cada item, lo etiqueta y lo guarda. Reanudable: saltea los ids ya escritos.

    Un id ya escrito con otro contenido (respuestas, etiqueta, spans, grupo, mensajes...) lanza
    `EvidenceError`: su fila se etiquetó con datos que ya no son los del dataset.

    Devuelve las filas de los items pedidos, en el orden de `items`. Un item que el backend rechaza
    (por ejemplo, prompt más largo que el contexto) queda como fila con `skipped` y sin tokens.
    """
    if not isinstance(max_tokens, int) or max_tokens < 1:
        raise ValueError("max_tokens debe ser un entero >= 1")
    ids = [it["id"] for it in items]
    if len(ids) != len(set(ids)):
        raise ValueError("los items tienen ids repetidos")
    wanted = _check_tiers(backend, tiers)
    store = RunStore(out_dir)
    store.open(build_manifest(backend, items, tiers=wanted, max_tokens=max_tokens, seed=seed, judge=judge))
    done = {r["id"]: r for r in store.load_rows()}
    hashes = {it["id"]: items_sha256([it]) for it in items}
    changed = [i for i in ids if i in done and done[i].get("item_sha256") != hashes[i]]
    if changed:
        shown = ", ".join(changed[:5]) + (" …" if len(changed) > 5 else "")
        raise EvidenceError(
            f"la corrida en {store.dir} tiene filas de items que cambiaron de contenido ({shown}); "
            "usá otro directorio de salida para no mezclar etiquetas viejas con el dataset nuevo"
        )
    total = len(items)
    count = sum(1 for i in ids if i in done)
    for item in items:
        if item["id"] in done:
            continue
        base = {
            "id": item["id"],
            "item_sha256": hashes[item["id"]],
            "group": item["group"],
            "type": item.get("type"),
            "mode": item["mode"],
            "item": {
                k: item.get(k)
                for k in (
                    "question",
                    "answers",
                    "label",
                    "hallucinated_spans",
                    "type",
                    "evidence",
                    "response",
                    "mode",
                )
            },
        }
        try:
            events, info = _run_item(backend, item, tiers=wanted, max_tokens=max_tokens, seed=seed)
        except (ValueError, VigiaError) as exc:
            if isinstance(exc, EvidenceError):
                raise
            row = base | {
                "source": None,
                "response": "",
                "finish_reason": None,
                "elapsed_s": 0.0,
                "tokens": [],
                "labels": [],
                "label_info": {"correct": None},
                "skipped": f"{type(exc).__name__}: {exc}",
            }
            store.append(row, np.zeros((0, 0), dtype=np.float32), [])
        else:
            meta = _token_meta(events)
            row = base | info | {"tokens": meta, "skipped": None}
            row["labels"], row["label_info"] = label_row(row, judge)
            X, names = _feature_matrix(events)
            store.append(row, X, names)
        done[item["id"]] = row
        count += 1
        if progress_cb is not None:
            progress_cb(count, total, item["id"])
    return [done[i] for i in ids]


# --- Matriz de features ----------------------------------------------------------------------------


@dataclass
class FeatureSet:
    """Tokens de una corrida: matriz de features, etiqueta, pregunta (grupo) y respuesta."""

    names: list[str]
    X: np.ndarray  # (n, d) float32, NaN = faltante
    labels: np.ndarray  # (n,) int64 con IGNORE_LABEL
    groups: np.ndarray  # (n,) str
    response_ids: np.ndarray  # (n,) str
    token_index: np.ndarray  # (n,) índice del token dentro de su respuesta
    model_info: ModelInfo
    label_source: str = "computable"
    judge: dict[str, Any] | None = None

    def features(self, names: Sequence[str] | None = None) -> dict[str, np.ndarray]:
        cols = {n: j for j, n in enumerate(self.names)}
        return {
            n: self.X[:, cols[n]].astype(np.float64) for n in (names if names is not None else self.names)
        }

    def save(self, path: str | Path) -> None:
        _atomic_savez(
            Path(path),
            X=self.X,
            names=np.asarray(self.names, dtype=str),
            labels=self.labels.astype(np.int64),
            groups=self.groups.astype(str),
            response_ids=self.response_ids.astype(str),
            token_index=self.token_index.astype(np.int64),
            meta=np.asarray(
                json.dumps(
                    {
                        "model_info": _model_info_to_dict(self.model_info),
                        "label_source": self.label_source,
                        "judge": self.judge,
                    },
                    ensure_ascii=False,
                )
            ),
        )

    @classmethod
    def load(cls, path: str | Path) -> FeatureSet:
        path = Path(path)
        if not path.is_file():
            raise EvidenceError(f"no existe {path}")
        try:
            with np.load(path, allow_pickle=False) as z:
                meta = json.loads(str(z["meta"]))
                return cls(
                    names=z["names"].tolist(),
                    X=z["X"],
                    labels=z["labels"],
                    groups=z["groups"],
                    response_ids=z["response_ids"],
                    token_index=z["token_index"],
                    model_info=model_info_from_dict(meta["model_info"]),
                    label_source=meta.get("label_source", "computable"),
                    judge=meta.get("judge"),
                )
        except (KeyError, ValueError, OSError) as exc:
            raise EvidenceError(f"{path} no es una matriz de features de Vigía: {exc}") from exc

    def labeled_count(self) -> int:
        return int(np.sum(self.labels != IGNORE_LABEL))


def build_features(out_dir: str | Path, rows: Sequence[Mapping[str, Any]] | None = None) -> FeatureSet:
    """Arma `features.npz` con los chunks de las filas (todas las de la corrida si `rows` es None)."""
    store = RunStore(out_dir)
    manifest = store.manifest()
    if manifest is None:
        raise EvidenceError(f"{out_dir} no tiene {MANIFEST_FILE}: corré extract primero")
    rows = store.load_rows() if rows is None else rows
    chunks = [(r, *store.load_chunk(r["id"])) for r in rows if r["tokens"]]
    if not chunks:
        raise EvidenceError("ninguna respuesta tiene tokens: no hay features para guardar")
    names = sorted({n for _, _, ns in chunks for n in ns})
    col = {n: j for j, n in enumerate(names)}
    n_total = sum(len(X) for _, X, _ in chunks)
    X_all = np.full((n_total, len(names)), np.nan, dtype=np.float32)
    labels, groups, resp, tok_idx = [], [], [], []
    pos = 0
    for row, X, ns in chunks:
        if len(X) != len(row["tokens"]) or len(row["labels"]) != len(row["tokens"]):
            raise EvidenceError(f"la fila {row['id']!r} no coincide con su chunk de features")
        X_all[pos : pos + len(X), [col[n] for n in ns]] = X
        pos += len(X)
        labels.extend(row["labels"])
        groups.extend([row["group"]] * len(X))
        resp.extend([row["id"]] * len(X))
        tok_idx.extend(range(len(X)))
    fs = FeatureSet(
        names=names,
        X=X_all,
        labels=np.asarray(labels, dtype=np.int64),
        groups=np.asarray(groups, dtype=str),
        response_ids=np.asarray(resp, dtype=str),
        token_index=np.asarray(tok_idx, dtype=np.int64),
        model_info=model_info_from_dict(manifest["model_info"]),
        label_source=manifest.get("label_source", "computable"),
        judge=manifest.get("judge"),
    )
    fs.save(store.dir / FEATURES_FILE)
    return fs


def relabel(out_dir: str | Path, *, judge: Judge | None = None) -> dict[str, Any]:
    """Vuelve a etiquetar todas las filas (computable o juez), reescribe filas y `features.npz`."""
    store = RunStore(out_dir)
    manifest = store.manifest()
    if manifest is None:
        raise EvidenceError(f"{out_dir} no tiene {MANIFEST_FILE}: corré extract primero")
    rows = store.load_rows()
    rejected: Counter[str] = Counter()
    agree = disagree = 0
    for row in rows:
        if row.get("skipped"):
            continue
        row["labels"], row["label_info"] = label_row(row, judge)
        if row["label_info"].get("rejected"):
            rejected[row["label_info"]["rejected"]] += 1
        cc, jc = row["label_info"].get("computable_correct"), row["label_info"].get("correct")
        if judge is not None and cc is not None and jc is not None:
            agree, disagree = agree + (cc == jc), disagree + (cc != jc)
    store.rewrite_rows(rows)
    manifest["label_source"] = "judge" if judge is not None else "computable"
    manifest["judge"] = (
        None if judge is None else {"model": str(judge.model), "prompt_sha256": JUDGE_PROMPT_SHA256}
    )
    _atomic_write_text(store.manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2))
    fs = build_features(out_dir, rows)
    report = {
        "label_source": manifest["label_source"],
        "judge": manifest["judge"],
        "n_rows": len(rows),
        "n_tokens_labeled": fs.labeled_count(),
        "token_error_rate": _rate(fs.labels),
        "rejected_by_reason": dict(sorted(rejected.items())),
        "judge_vs_gold": None
        if judge is None
        else {
            "agree": agree,
            "disagree": disagree,
            "agreement": agree / (agree + disagree) if agree + disagree else None,
        },
    }
    _atomic_write_text(store.dir / "labels_report.json", json.dumps(report, ensure_ascii=False, indent=2))
    return report


def _rate(labels: np.ndarray) -> float | None:
    keep = labels != IGNORE_LABEL
    return float(labels[keep].mean()) if keep.any() else None


# --- Calibración y evaluación ----------------------------------------------------------------------


def detector_feature_names(names: Sequence[str]) -> list[str]:
    """Features del detector por defecto: niveles F0 y F1 (ADR A-03). Sin ellas, todas."""
    cheap = [n for n in names if feature_tier(n) in DETECTOR_TIERS]
    return cheap or list(names)


def calibrate_features(
    fs: FeatureSet,
    *,
    seed: int = 0,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    default_alpha: float = 0.1,
    feature_names: Sequence[str] | None = None,
) -> tuple[Detector, dict[str, Any]]:
    """Entrena el detector (F0+F1 por defecto) con split agrupado por pregunta."""
    names = list(feature_names) if feature_names is not None else detector_feature_names(fs.names)
    unknown = sorted(set(names) - set(fs.names))
    if unknown:
        raise EvidenceError(f"features inexistentes en la matriz: {unknown[:5]}")
    _require_labels(fs)
    return train_detector(
        fs.features(names),
        fs.labels,
        fs.groups,
        fs.response_ids,
        fs.model_info,
        alphas=alphas,
        default_alpha=default_alpha,
        seed=seed,
        label_source=fs.label_source,
        judge=fs.judge,
    )


def save_calibration(
    out_dir: str | Path, detector: Detector, split_report: Mapping[str, Any]
) -> tuple[Path, Path]:
    """Escribe `detector.json` y `split.json` (el split que evaluate necesita para no tener fuga)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    detector.save(out / DETECTOR_FILE)
    _atomic_write_text(out / SPLIT_FILE, json.dumps(dict(split_report), ensure_ascii=False, indent=2))
    return out / DETECTOR_FILE, out / SPLIT_FILE


def _require_labels(fs: FeatureSet) -> None:
    y = fs.labels[fs.labels != IGNORE_LABEL]
    if y.size == 0:
        raise EvidenceError("no hay tokens etiquetados")
    if len(np.unique(y)) < 2:
        kind = "correctas" if y[0] == 0 else "erróneas"
        raise EvidenceError(
            f"todas las respuestas etiquetadas son {kind}: hace falta al menos una de cada clase "
            "para calibrar y evaluar (no se inventa un AUROC)"
        )


def evaluate_features(
    fs: FeatureSet,
    *,
    detector: Detector | None = None,
    split: Mapping[str, Sequence[str]] | None = None,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float | None = None,
) -> dict[str, Any]:
    """Reporte canónico (`vigia.evaluate.evaluate_report`) sobre todas las features de la corrida."""
    _require_labels(fs)
    try:
        return evaluate_report(
            fs.features(),
            fs.labels,
            fs.groups,
            fs.response_ids,
            fs.model_info,
            detector=detector,
            split=split,
            n_boot=n_boot,
            seed=seed,
            alpha=alpha,
        )
    except SingleClassError as exc:
        raise EvidenceError(f"no se puede evaluar: {exc}") from exc


# --- Resumen del modelo ----------------------------------------------------------------------------


def model_quality(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Exactitud del modelo y tasa de alucinación por tipo de pregunta (respuestas con etiqueta)."""

    def summarize(group: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        labeled = [r for r in group if r["label_info"].get("correct") is not None]
        n_ok = sum(1 for r in labeled if r["label_info"]["correct"])
        n = len(labeled)
        return {
            "n": n,
            "correct": n_ok,
            "accuracy": n_ok / n if n else None,
            "hallucination_rate": (n - n_ok) / n if n else None,
        }

    usable = [r for r in rows if not r.get("skipped")]
    generated = [r for r in usable if r.get("source") == "generated"]
    by_type: dict[str, list[Mapping[str, Any]]] = {}
    for r in generated:
        by_type.setdefault(str(r.get("type") or "sin_tipo"), []).append(r)
    return {
        "generated": summarize(generated)
        | {"by_type": {k: summarize(v) for k, v in sorted(by_type.items())}},
        "teacher_forced": summarize([r for r in usable if r.get("source") == "teacher_forced"]),
        "all": summarize(usable),
    }


def _latency_from_bench(bench: Any, depth: int = 0) -> dict[str, Any] | None:
    """Busca `{baseline: {p50,p95,p99}, vigia: {...}}` en el resultado del benchmark."""

    def is_lat(x: Any) -> bool:
        return isinstance(x, Mapping) and all(
            isinstance(x.get(k), int | float) and not isinstance(x.get(k), bool)
            for k in ("p50", "p95", "p99")
        )

    if isinstance(bench, Mapping):
        if is_lat(bench.get("baseline")) and is_lat(bench.get("vigia")):
            out = {"baseline": dict(bench["baseline"]), "vigia": dict(bench["vigia"])}
            if isinstance(bench.get("target"), int | float):
                out["target"] = bench["target"]
            return out
        children: Iterable[Any] = bench.values()
    elif isinstance(bench, list | tuple):
        children = bench
    else:
        return None
    if depth >= 3:
        return None
    for child in children:
        found = _latency_from_bench(child, depth + 1)
        if found is not None:
            return found
    return None


def _run_bench(
    backend: Backend,
    prompts: Sequence[Any],
    *,
    tiers: tuple[Tier, ...],
    max_tokens: int,
    runs: int,
    warmup: int,
    seed: int,
) -> dict[str, Any]:
    try:
        from vigia.bench import run_overhead_bench  # import perezoso: módulo opcional del benchmark

        result = run_overhead_bench(
            backend,
            list(prompts),
            tiers_list=[tiers],
            max_tokens=max_tokens,
            warmup=warmup,
            runs=runs,
            seed=seed,
        )
    except Exception as exc:  # noqa: BLE001 - el benchmark es opcional; la falla se reporta, no corta
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {"result": result}


# --- Reporte ---------------------------------------------------------------------------------------


def _quality_html(report: Mapping[str, Any]) -> str:
    ev = report.get("evidence") or {}
    quality = ev.get("model_quality") or {}
    gen = quality.get("generated") or {}
    rows = [("Todas las generadas", gen)] + [(k, v) for k, v in (gen.get("by_type") or {}).items()]
    body = "".join(
        f'<tr><th scope="row">{html.escape(str(name))}</th><td>{int(s.get("n", 0))}</td>'
        f"<td>{html.escape(charts.fmt_pct(s.get('accuracy')))}</td>"
        f"<td>{html.escape(charts.fmt_pct(s.get('hallucination_rate')))}</td></tr>"
        for name, s in rows
    )
    table = (
        '<table class="vg-metrics"><caption>Tabla 5. Exactitud del modelo y tasa de alucinación por tipo '
        'de pregunta (gold computable)</caption><thead><tr><th scope="col">Tipo</th>'
        '<th scope="col">Respuestas</th><th scope="col">Exactitud</th>'
        f'<th scope="col">Tasa de alucinación</th></tr></thead><tbody>{body}</tbody></table>'
    )
    skipped = ev.get("skipped") or {}
    notes = []
    if skipped:
        notes.append(f"Respuestas salteadas: {sum(skipped.values())}.")
    if report.get("latency_error"):
        notes.append(f"Latencia no medida: {report['latency_error']}")
    notes_html = "".join(f'<p role="note">{html.escape(n)}</p>' for n in notes)
    return (
        f'<section aria-labelledby="vg-quality"><h2 id="vg-quality">Modelo</h2>{table}{notes_html}</section>'
    )


def report_html(report: Mapping[str, Any]) -> str:
    """HTML del reporte canónico más la sección de exactitud del modelo por tipo."""
    page = render_report_html(report)
    return page.replace("</main>", _quality_html(report) + "</main>", 1)


def summary_text(report: Mapping[str, Any]) -> str:
    """Resumen legible del reporte (también va a `summary.txt`)."""
    ev = report.get("evidence") or {}
    model = report["model"]
    lines = [
        "Vigía: experimento de evidencia" if ev else "Vigía: reporte de evaluación",
        f"Modelo: {model.get('model_id')} ({model.get('model_type')})",
    ]
    if ev:
        skipped = sum((ev.get("skipped") or {}).values())
        lines += [
            f"Niveles: {', '.join(ev.get('tiers', []))}; max_tokens {ev.get('max_tokens')}; "
            f"semilla {ev.get('seed')}",
            f"Respuestas: {ev.get('n_rows', 0)} (generadas {ev.get('n_generated', 0)}, teacher forcing "
            f"{ev.get('n_teacher_forced', 0)}, salteadas {skipped})",
            f"Etiquetas: {ev.get('label_source')}",
        ]
    gen = (ev.get("model_quality") or {}).get("generated") or {}
    if gen.get("n"):
        lines.append(
            f"Exactitud del modelo: {charts.fmt_pct(gen['accuracy'])} sobre {gen['n']} respuestas generadas"
        )
        for name, s in (gen.get("by_type") or {}).items():
            lines.append(
                f"  {name}: exactitud {charts.fmt_pct(s['accuracy'])}, alucinación "
                f"{charts.fmt_pct(s['hallucination_rate'])} (n={s['n']})"
            )
    ds = report["dataset"]
    lines.append(
        f"Test: {ds['n_tokens_test']} tokens, {ds['n_responses_test']} respuestas, "
        f"{ds['n_groups_test']} preguntas; prevalencia por token {charts.fmt_pct(ds['prevalence_token'])}"
    )
    lines.append(f"AUROC por token (IC {charts.fmt_pct(report['bootstrap']['level'], 0)} por pregunta):")
    entries = [("Detector", report["detector"])]
    entries += [(f"Nivel {k}", v) for k, v in report["tiers"].items()]
    entries += [(f"Baseline {k}", v) for k, v in report["baselines"].items()]
    for name, entry in entries:
        if not entry.get("available"):
            lines.append(f"  {name}: no disponible ({entry.get('reason')})")
            continue
        m = entry["token"]["auroc"]
        lo, hi = m.get("ci") or (None, None)
        lines.append(
            f"  {name}: {charts.fmt_num(m.get('value'))} [{charts.fmt_num(lo)}; {charts.fmt_num(hi)}]"
        )
    lat = report.get("latency")
    if lat:
        b, v = lat["baseline"]["p50"], lat["vigia"]["p50"]
        over = (v - b) / b if b else None
        lines.append(
            f"Latencia p50: base {charts.fmt_num(b, 2)}, con Vigía {charts.fmt_num(v, 2)}; "
            f"overhead {charts.fmt_pct(over)} (meta {charts.fmt_pct(lat.get('target', 0.05), 0)})"
        )
    elif report.get("latency_error"):
        lines.append(f"Latencia: no medida ({report['latency_error']})")
    return "\n".join(lines) + "\n"


def write_report(out_dir: str | Path, report: Mapping[str, Any]) -> dict[str, Path]:
    """Escribe report.json, report.html y summary.txt. Devuelve las rutas."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"json": out / REPORT_JSON, "html": out / REPORT_HTML, "summary": out / SUMMARY_FILE}
    _atomic_write_text(paths["json"], json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    _atomic_write_text(paths["html"], report_html(report))
    _atomic_write_text(paths["summary"], summary_text(report))
    return paths


def run_evidence(
    backend: Backend,
    items: Sequence[Mapping[str, Any]],
    *,
    out_dir: str | Path,
    max_tokens: int = 24,
    tiers: Sequence[Tier | str] = DEFAULT_TIERS,
    seed: int = 0,
    judge: Judge | None = None,
    n_boot: int = 2000,
    alpha: float | None = None,
    bench_prompts: Sequence[Any] | None = None,
    bench_runs: int = 3,
    bench_warmup: int = 1,
    progress_cb: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Experimento completo y reanudable. Devuelve el reporte y deja los artefactos en `out_dir`.

    `bench_prompts` (lista de mensajes) activa el benchmark de overhead con `vigia.bench`; si el
    benchmark falla, el reporte lo dice en `latency_error` y el resto sigue.
    """
    wanted = _check_tiers(backend, tiers)
    rows = extract(
        backend,
        items,
        out_dir,
        tiers=wanted,
        max_tokens=max_tokens,
        seed=seed,
        judge=judge,
        progress_cb=progress_cb,
    )
    fs = build_features(out_dir, rows)
    detector, split_report = calibrate_features(fs, seed=seed)
    out = Path(out_dir)
    save_calibration(out, detector, split_report)
    report = evaluate_features(
        fs, detector=detector, split=split_report["groups"], n_boot=n_boot, seed=seed, alpha=alpha
    )

    skipped = Counter(str(r["skipped"]).split(":")[0] for r in rows if r.get("skipped"))
    rejected = Counter(r["label_info"]["rejected"] for r in rows if r["label_info"].get("rejected"))
    gen_rows = [r for r in rows if r.get("source") == "generated"]
    gen_tokens = sum(len(r["tokens"]) for r in gen_rows)
    gen_s = sum(float(r.get("elapsed_s") or 0.0) for r in gen_rows)
    report["evidence"] = {
        "n_items": len(items),
        "n_rows": len(rows),
        "n_generated": len(gen_rows),
        "n_teacher_forced": sum(1 for r in rows if r.get("source") == "teacher_forced"),
        "skipped": dict(sorted(skipped.items())),
        "labels_rejected": dict(sorted(rejected.items())),
        "tiers": [t.value for t in wanted],
        "detector_features": len(detector.feature_names),
        "max_tokens": max_tokens,
        "seed": seed,
        "dataset_sha256": items_sha256(items),
        "label_source": fs.label_source,
        "judge": fs.judge,
        "model_quality": model_quality(rows),
        "generation": {
            "tokens": gen_tokens,
            "seconds": gen_s,
            "tokens_per_s": gen_tokens / gen_s if gen_s > 0 else None,
        },
    }
    report["split"] = split_report
    if bench_prompts:
        bench_tiers = tuple(t for t in wanted if t is not Tier.H) or wanted
        bench = _run_bench(
            backend,
            bench_prompts,
            tiers=bench_tiers,
            max_tokens=max_tokens,
            runs=bench_runs,
            warmup=bench_warmup,
            seed=seed,
        )
        if "error" in bench:
            report["latency_error"] = bench["error"]
        else:
            report["bench"] = bench["result"]
            latency = _latency_from_bench(bench["result"])
            if latency is None:
                report["latency_error"] = "el benchmark no devolvió p50/p95/p99 de base y de Vigía"
            else:
                report["latency"] = latency
        _atomic_write_text(out / BENCH_FILE, json.dumps(json_safe(bench), ensure_ascii=False, indent=2))
    report = json_safe(report)
    write_report(out, report)
    return report


def json_safe(obj: Any) -> Any:
    """Copia serializable en JSON estricto: numpy -> Python, NaN/inf -> None, Tier -> str."""
    if isinstance(obj, Mapping):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, Tier):
        return obj.value
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if obj is None or isinstance(obj, str | int | float | bool):
        return obj
    return str(obj)


__all__ = [
    "DEFAULT_TIERS",
    "EvidenceError",
    "FeatureSet",
    "RunStore",
    "build_features",
    "calibrate_features",
    "computable_labels",
    "detector_feature_names",
    "evaluate_features",
    "extract",
    "gold_correct",
    "json_safe",
    "judge_labels",
    "label_row",
    "model_info_from_dict",
    "model_quality",
    "relabel",
    "report_html",
    "run_evidence",
    "save_calibration",
    "summary_text",
    "write_report",
]
