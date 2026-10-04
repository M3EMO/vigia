"""Calibración gestionada por tenant: datasets JSONL, job de entrenamiento y activación (P-23, AT-P16).

Formato JSONL (una línea por item, UTF-8, máximo 20 MB y 5000 items):

    {"id": "q1", "question": "...", "answers": ["gold", "alias"]}
    {"id": "q2", "messages": [{"role": "user", "content": "..."}], "response": "...", "label": 1}
    {"id": "q3", "question": "...", "response": "...", "hallucinated_spans": ["fragmento exacto"]}

- `answers`: Vigía genera la respuesta con el modelo del tenant y la etiqueta de forma computable
  (exact match o F1 de tokens >= 0,5 contra cualquier alias).
- `response` + `label`: respuesta dada; `label` 1 = errónea, 0 = correcta. Todos sus tokens heredan
  la etiqueta. Las features salen por teacher forcing (una sola pasada).
- `response` + `hallucinated_spans`: cada fragmento es un substring exacto de la respuesta (string
  o `{"text", "occurrence"}`, como el juez de `vigia.labels`). Los tokens que tocan un fragmento
  valen 1 y el resto 0. Una lista vacía marca la respuesta completa como correcta.

El job `calibrate` corre en el worker de `vigia_ee.jobs`. Entrena con `vigia.calibrate`, evalúa con
`vigia.evaluate` (bootstrap de 500 réplicas), guarda el detector como candidato con su reporte y
emite `calibration.completed`. Un error para el usuario se guarda como `CalibrationJobError`.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from vigia.calibrate import DEFAULT_ALPHAS, CalibrationError, train_detector
from vigia.contracts import DetectorMismatchError, Tier, TokenEvent, feature_tier
from vigia.detector import Detector
from vigia.evaluate import SingleClassError, evaluate_report
from vigia.labels import (
    IGNORE_LABEL,
    InvalidJudgment,
    exact_match,
    label_tokens_from_answer,
    spans_to_token_labels,
    token_f1,
    validate_judgment,
)
from vigia_ee import audit
from vigia_ee.db import MAX_PAGE, to_iso
from vigia_ee.jobs import PermanentJobError

if TYPE_CHECKING:
    from vigia_ee.auth import Principal
    from vigia_ee.runtime import Runtime

_log = logging.getLogger("vigia.calibration")

JOB_KIND = "calibrate"
MAX_ITEMS = 5000
MIN_RESPONSES = 20
PLATFORM_N_BOOT = 500
ANSWER_MAX_TOKENS = 64
ANSWER_F1_THRESHOLD = 0.5
MAX_ID_LEN = 128
MAX_NAME_LEN = 200
MAX_MESSAGES = 64
MAX_ANSWERS = 32
MAX_ALPHAS = 8
CHEAP_TIERS = (Tier.F0, Tier.F1, Tier.U)
_ITEM_KEYS = frozenset({"id", "messages", "question", "answers", "response", "label", "hallucinated_spans"})
_ROLES = {"system": "system", "developer": "system", "user": "user", "assistant": "assistant"}
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_GENERIC_JOB_ERROR = "Error interno durante la calibración. Revisá los logs del servidor con el id del job."
_USER_ERROR_PREFIX = "CalibrationJobError: "


class DatasetError(ValueError):
    """El dataset es inválido. El mensaje es para el usuario."""


class DatasetNotFound(DatasetError, LookupError):
    """El dataset no existe en este tenant (la capa HTTP responde 404)."""


class DetectorNotFound(LookupError):
    """El detector no existe en este tenant (la capa HTTP responde 404)."""


class ActivationError(ValueError):
    """El detector existe pero no se puede activar. El mensaje es para el usuario."""


class CalibrationJobError(PermanentJobError):
    """El job falla sin reintentar con un mensaje para el usuario."""


# --- Validación del JSONL -------------------------------------------------------------------------


@dataclass
class CalibrationItem:
    id: str
    messages: list[dict[str, str]]
    kind: str  # answers | label | spans
    answers: list[str] = field(default_factory=list)
    response: str = ""
    label: int | None = None
    spans: list[tuple[int, int]] = field(default_factory=list)


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"clave repetida {key!r}")
        out[key] = value
    return out


def _reject_constant(name: str) -> Any:
    raise ValueError(f"constante no válida en JSON: {name}")


def _text(value: Any, what: str, line: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetError(f"Línea {line}: {what} debe ser un texto no vacío.")
    return value


def _messages(raw: Any, line: int) -> list[dict[str, str]]:
    if not isinstance(raw, list) or not 1 <= len(raw) <= MAX_MESSAGES:
        raise DatasetError(f"Línea {line}: 'messages' debe ser una lista de 1 a {MAX_MESSAGES} mensajes.")
    out = []
    for m in raw:
        if not isinstance(m, dict) or set(m) != {"role", "content"}:
            raise DatasetError(f"Línea {line}: cada mensaje lleva exactamente 'role' y 'content'.")
        role = m["role"]
        if role not in _ROLES:
            raise DatasetError(f"Línea {line}: rol inválido; opciones: system, user, assistant.")
        if not isinstance(m["content"], str):
            raise DatasetError(f"Línea {line}: 'content' debe ser texto.")
        out.append({"role": _ROLES[role], "content": m["content"]})
    if not any(m["content"].strip() for m in out):
        raise DatasetError(f"Línea {line}: los mensajes no tienen contenido.")
    return out


def _parse_item(data: Any, line: int) -> CalibrationItem:
    if not isinstance(data, dict):
        raise DatasetError(f"Línea {line}: cada línea debe ser un objeto JSON.")
    unknown = sorted(set(data) - _ITEM_KEYS)
    if unknown:
        raise DatasetError(f"Línea {line}: campos desconocidos: {', '.join(unknown)}.")
    item_id = data.get("id")
    if not isinstance(item_id, str) or not item_id.strip() or len(item_id) > MAX_ID_LEN:
        raise DatasetError(f"Línea {line}: 'id' debe ser un texto de 1 a {MAX_ID_LEN} caracteres.")
    if ("messages" in data) == ("question" in data):
        raise DatasetError(f"Línea {line}: usá 'messages' o 'question' (exactamente uno).")
    if "messages" in data:
        messages = _messages(data["messages"], line)
    else:
        messages = [{"role": "user", "content": _text(data["question"], "'question'", line)}]
    has_answers, has_response = "answers" in data, "response" in data
    if has_answers == has_response:
        raise DatasetError(f"Línea {line}: usá 'answers' o 'response' (exactamente uno).")
    if has_answers:
        if "label" in data or "hallucinated_spans" in data:
            raise DatasetError(f"Línea {line}: 'answers' no se combina con 'label' ni 'hallucinated_spans'.")
        answers = data["answers"]
        if not isinstance(answers, list) or not 1 <= len(answers) <= MAX_ANSWERS:
            raise DatasetError(f"Línea {line}: 'answers' debe ser una lista de 1 a {MAX_ANSWERS} textos.")
        clean = [_text(a, "cada respuesta de 'answers'", line) for a in answers]
        return CalibrationItem(item_id, messages, "answers", answers=clean)
    response = _text(data["response"], "'response'", line)
    if ("label" in data) == ("hallucinated_spans" in data):
        raise DatasetError(
            f"Línea {line}: con 'response' va 'label' o 'hallucinated_spans' (exactamente uno)."
        )
    if "label" in data:
        label = data["label"]
        if isinstance(label, bool) or label not in (0, 1):
            raise DatasetError(f"Línea {line}: 'label' debe ser 0 (correcta) o 1 (errónea).")
        return CalibrationItem(item_id, messages, "label", response=response, label=int(label))
    spans = data["hallucinated_spans"]
    if not isinstance(spans, list):
        raise DatasetError(f"Línea {line}: 'hallucinated_spans' debe ser una lista.")
    try:
        verdict = validate_judgment({"label": 1 if spans else 0, "hallucinated_spans": spans}, response)
    except InvalidJudgment as exc:
        raise DatasetError(f"Línea {line}: fragmento inválido ({exc.reason}): {exc}") from None
    return CalibrationItem(item_id, messages, "spans", response=response, spans=list(verdict.spans))


def parse_dataset(content: bytes, *, max_items: int = MAX_ITEMS) -> list[CalibrationItem]:
    """Valida el JSONL completo. Lanza DatasetError con la línea y el motivo."""
    if not isinstance(content, bytes):
        raise DatasetError("El contenido del dataset debe ser binario.")
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        raise DatasetError("El archivo no es UTF-8 válido.") from None
    text = text.removeprefix("﻿")
    items: list[CalibrationItem] = []
    seen: set[str] = set()
    for n, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            continue
        if len(items) >= max_items:
            raise DatasetError(f"El dataset supera el máximo de {max_items} items.")
        try:
            data = json.loads(raw, object_pairs_hook=_reject_duplicates, parse_constant=_reject_constant)
        except ValueError as exc:
            reason = exc.msg if isinstance(exc, json.JSONDecodeError) else str(exc)
            raise DatasetError(f"Línea {n}: JSON inválido ({reason}).") from None
        item = _parse_item(data, n)
        if item.id in seen:
            raise DatasetError(f"Línea {n}: el id {item.id[:40]!r} está repetido.")
        seen.add(item.id)
        items.append(item)
    if not items:
        raise DatasetError("El dataset está vacío.")
    return items


def _check_name(name: Any) -> str:
    if not isinstance(name, str) or not name.strip() or len(name.strip()) > MAX_NAME_LEN:
        raise DatasetError(f"El nombre debe tener entre 1 y {MAX_NAME_LEN} caracteres.")
    if _CONTROL_RE.search(name):
        raise DatasetError("El nombre no puede tener caracteres de control.")
    return name.strip()


def _check_alphas(alphas: Sequence[float] | None) -> list[float]:
    if alphas is None:
        return list(DEFAULT_ALPHAS)
    if isinstance(alphas, str | bytes) or not isinstance(alphas, Iterable):
        raise DatasetError("Los alphas deben ser una lista de números.")
    values = list(alphas)
    if not 1 <= len(values) <= MAX_ALPHAS:
        raise DatasetError(f"Elegí entre 1 y {MAX_ALPHAS} valores de alpha.")
    out = []
    for a in values:
        if isinstance(a, bool) or not isinstance(a, int | float) or not math.isfinite(a) or not 0 < a < 1:
            raise DatasetError("Cada alpha debe estar en el intervalo abierto (0, 1).")
        out.append(float(a))
    return sorted(set(out))


def _finite(obj: Any) -> Any:
    """Copia JSON-segura: NaN e infinitos pasan a None; numpy pasa a tipos nativos."""
    if isinstance(obj, Mapping):
        return {str(k): _finite(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_finite(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _visible(t: TokenEvent) -> bool:
    return t.channel == "final" and not t.is_special


def _require(principal: Principal, permission: str) -> None:
    if not principal.can(permission):
        raise PermissionError("Tu rol no tiene permiso para esta acción.")


def _parse_transcript(prompt_text: str) -> list[dict[str, str]]:
    """Inversa de `vigia_ee.runtime.render_prompt` (bloques `rol: contenido` separados por línea vacía)."""
    blocks = re.split(r"\n\n(?=(?:system|user|assistant): )", prompt_text)
    out = []
    for block in blocks:
        role, sep, content = block.partition(": ")
        if not sep or role not in ("system", "user", "assistant"):
            return [{"role": "user", "content": prompt_text}]
        out.append({"role": role, "content": content})
    return out or [{"role": "user", "content": prompt_text}]


# --- Servicio -------------------------------------------------------------------------------------


class CalibrationService:
    """Datasets, jobs de calibración y activación de detectores de un `Runtime`."""

    def __init__(self, runtime: Runtime) -> None:
        self.runtime = runtime
        runtime.jobs.register(JOB_KIND, self._handle_job)

    # -- datasets --

    def upload_dataset(self, principal: Principal, name: str, content: bytes) -> str:
        """Valida y guarda un JSONL. Devuelve el id del dataset o lanza DatasetError."""
        _require(principal, "calibration:write")
        clean_name = _check_name(name)
        limit = self.runtime.settings.max_upload_bytes
        if not isinstance(content, bytes):
            raise DatasetError("El contenido del dataset debe ser binario.")
        if len(content) > limit:
            raise DatasetError(f"El archivo supera el máximo de {limit // (1024 * 1024)} MB.")
        items = parse_dataset(content)
        repo = self.runtime.repo(principal.org_id)
        dataset_id = repo.insert_dataset(clean_name, content, n_items=len(items))
        audit.append(
            self.runtime.db,
            principal.org_id,
            principal,
            "dataset_upload",
            dataset_id,
            {"n_items": len(items), "bytes": len(content)},
        )
        return dataset_id

    def dataset_from_reviews(self, principal: Principal, name: str) -> str:
        """Exporta las decisiones completadas de la cola de revisión como dataset JSONL etiquetado.

        `error` vale 1 y `correct` vale 0; `uncertain` se omite. Un item con decisiones que se
        contradicen se omite. Una decisión sobre la respuesta completa da `response` + `label`; las
        decisiones por fragmento dan `hallucinated_spans` con la ocurrencia exacta.
        """
        _require(principal, "calibration:write")
        clean_name = _check_name(name)
        org_id = principal.org_id
        repo = self.runtime.repo(org_id)
        by_trace: dict[str, dict[str, Any]] = {}
        for item in self._completed_review_items(org_id):
            decisions = repo.list_review_decisions(item["id"]) or []
            labels = {d["label"] for d in decisions if d["label"] in ("correct", "error")}
            if len(labels) != 1:
                continue  # sin decisiones útiles o revisores en desacuerdo
            entry = by_trace.setdefault(item["trace_id"], {"whole": None, "spans": {}})
            if item["span_idx"] is None:
                entry["whole"] = labels.pop()
            else:
                entry["spans"][int(item["span_idx"])] = labels.pop()
        lines = []
        for trace_id, entry in by_trace.items():
            trace = repo.get_trace(trace_id)
            if trace is None or not trace.get("response_text") or not trace.get("prompt_text"):
                continue  # store_text=none: no hay texto para recalibrar
            response = trace["response_text"]
            row: dict[str, Any] = {
                "id": trace_id,
                "messages": _parse_transcript(trace["prompt_text"]),
                "response": response,
            }
            if entry["whole"] is not None:
                row["label"] = 1 if entry["whole"] == "error" else 0
            else:
                spans_by_idx = {s["idx"]: s for s in trace["spans"]}
                marked = []
                for idx, label in sorted(entry["spans"].items()):
                    span = spans_by_idx.get(idx)
                    if span is None or label != "error":
                        continue
                    text = response[span["start"] : span["end"]]
                    if not text.strip():
                        continue
                    occurrence = response.count(text, 0, span["start"]) + 1
                    marked.append({"text": text, "occurrence": occurrence})
                row["hallucinated_spans"] = marked
            lines.append(json.dumps(row, ensure_ascii=False))
        if not lines:
            raise DatasetError("No hay decisiones de revisión completadas con texto para exportar.")
        content = ("\n".join(lines) + "\n").encode("utf-8")
        if len(content) > self.runtime.settings.max_upload_bytes:
            raise DatasetError("Las revisiones exportadas superan el tamaño máximo de un dataset.")
        items = parse_dataset(content)
        dataset_id = repo.insert_dataset(clean_name, content, n_items=len(items))
        audit.append(
            self.runtime.db, org_id, principal, "dataset_from_reviews", dataset_id, {"n_items": len(items)}
        )
        return dataset_id

    def _completed_review_items(self, org_id: str) -> list[dict[str, Any]]:
        # `TenantScopedRepository.list_review_items` corta en MAX_PAGE sin paginar; acá hacen falta
        # todos los completados. La consulta filtra por org_id igual que el repositorio.
        return self.runtime.db.query(
            "SELECT id, trace_id, span_idx FROM review_items WHERE org_id = ? AND state = 'completed' "
            "ORDER BY created_at, id",
            (org_id,),
        )

    # -- jobs --

    def start(
        self, principal: Principal, dataset_id: str, *, alphas: Sequence[float] | None = None, seed: int = 0
    ) -> str:
        """Encola el job de calibración. Devuelve el id del job."""
        _require(principal, "calibration:write")
        if (
            not isinstance(dataset_id, str)
            or self.runtime.repo(principal.org_id).get_dataset(dataset_id) is None
        ):
            raise DatasetNotFound("El dataset no existe.")
        clean_alphas = _check_alphas(alphas)
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**31:
            raise DatasetError("La semilla debe ser un entero entre 0 y 2147483647.")
        job_id = self.runtime.jobs.enqueue(
            JOB_KIND,
            {
                "dataset_id": dataset_id,
                "alphas": clean_alphas,
                "seed": seed,
                "created_by": principal.user_id,
            },
            org_id=principal.org_id,
            max_attempts=1,  # entrenar es determinista: reintentar no cambia el resultado
        )
        audit.append(
            self.runtime.db,
            principal.org_id,
            principal,
            "calibration_start",
            job_id,
            {"dataset_id": dataset_id, "alphas": clean_alphas, "seed": seed},
        )
        return job_id

    def job(self, org_id: str, job_id: str) -> dict[str, Any] | None:
        """Estado de un job de calibración del tenant, o None (otro tenant u otro tipo de job)."""
        if not isinstance(job_id, str):
            return None
        job = self.runtime.jobs.get(org_id, job_id)
        if job is None or job["kind"] != JOB_KIND:
            return None
        return self._public(job)

    def list_jobs(self, org_id: str, limit: int = 20) -> list[dict[str, Any]]:
        return [self._public(j) for j in self.runtime.jobs.list(org_id, JOB_KIND, limit=limit)]

    @staticmethod
    def _public(job: Mapping[str, Any]) -> dict[str, Any]:
        payload = job.get("payload") or {}
        error = job.get("error")
        message = None
        if job["state"] == "failed" and error:
            message = (
                error[len(_USER_ERROR_PREFIX) :]
                if error.startswith(_USER_ERROR_PREFIX)
                else _GENERIC_JOB_ERROR
            )
        return {
            "id": job["id"],
            "state": job["state"],
            "created_at": job["created_at"],
            "updated_at": job["updated_at"],
            "dataset_id": payload.get("dataset_id"),
            "alphas": payload.get("alphas"),
            "seed": payload.get("seed"),
            "result": job.get("result") if job["state"] == "done" else None,
            "error": message,
        }

    # -- activación --

    def activate(self, principal: Principal, detector_id: str) -> None:
        """Activa un detector candidato del tenant. Audita, invalida la cache y emite el webhook."""
        _require(principal, "calibration:write")
        org_id = principal.org_id
        repo = self.runtime.repo(org_id)
        row = repo.get_detector(detector_id) if isinstance(detector_id, str) else None
        if row is None:
            raise DetectorNotFound("El detector no existe.")
        try:
            detector = Detector.loads(row["artifact_json"])
            detector.check_model(self.runtime.model_info)
        except DetectorMismatchError as exc:
            raise ActivationError(f"No se puede activar: {exc}") from None
        policy = self.runtime.policy_for(org_id)
        min_alpha = min(t["alpha"] for t in detector.thresholds)
        if policy.alpha < min_alpha:
            raise ActivationError(
                f"La política usa alpha {policy.alpha:g} y el detector solo tiene umbrales desde "
                f"{min_alpha:g}. Subí el alpha de la política o recalibrá con un alpha menor."
            )
        with self.runtime.db.transaction():
            repo.activate_detector(detector_id)
            audit.append(self.runtime.db, org_id, principal, "detector_activate", detector_id)
        self.runtime.invalidate_detector(org_id)
        self.runtime.emit_event(org_id, "calibration.activated", {"detector_id": detector_id})

    # -- handler del job --

    def _handle_job(self, job: dict[str, Any]) -> dict[str, Any]:
        org_id = job.get("org_id")
        if not org_id:
            raise CalibrationJobError("Un job de calibración siempre pertenece a una organización.")
        payload = job["payload"]
        repo = self.runtime.repo(org_id)
        dataset = repo.get_dataset(payload["dataset_id"])
        if dataset is None:
            raise CalibrationJobError("El dataset ya no existe.")
        try:
            items = parse_dataset(bytes(dataset["content"]))
        except DatasetError as exc:
            raise CalibrationJobError(str(exc)) from None
        rows, labels, groups, skipped, kinds = self._collect(items)
        n_responses = len(set(groups))
        if n_responses < MIN_RESPONSES:
            raise CalibrationJobError(
                f"Hacen falta al menos {MIN_RESPONSES} respuestas etiquetadas; el dataset aportó "
                f"{n_responses}{_skipped_note(skipped)}."
            )
        if len(set(labels)) < 2:
            only = "erróneos" if labels[0] == 1 else "correctos"
            raise CalibrationJobError(
                f"Todos los tokens quedaron {only}. Hacen falta respuestas correctas y erróneas."
            )
        names = sorted({n for r in rows for n in r if _canonical(n)})
        features = {n: np.array([r.get(n, np.nan) for r in rows], dtype=np.float64) for n in names}
        y = np.asarray(labels, dtype=np.int64)
        g = np.asarray(groups, dtype=object)
        alphas = [float(a) for a in payload.get("alphas") or DEFAULT_ALPHAS]
        default_alpha = 0.1 if min(alphas) <= 0.1 else min(alphas)
        seed = int(payload.get("seed", 0))
        info = self.runtime.model_info
        try:
            detector, split_report = train_detector(
                features,
                y,
                g,
                g,
                info,
                alphas=alphas,
                default_alpha=default_alpha,
                seed=seed,
                label_source="computable" if kinds == {"answers"} else "dataset",
                dataset_sha=dataset["sha256"],
                created_at=to_iso(self.runtime.now()),
            )
            report = evaluate_report(
                features,
                y,
                g,
                g,
                info,
                detector=detector,
                split=split_report["groups"],
                n_boot=PLATFORM_N_BOOT,
                seed=seed,
            )
        except (CalibrationError, SingleClassError) as exc:
            raise CalibrationJobError(f"No se pudo calibrar: {exc}") from None
        summary = {
            "job_id": job["id"],
            "dataset_id": dataset["id"],
            "dataset_name": dataset["name"],
            "n_items": len(items),
            "n_responses": n_responses,
            "n_tokens": len(labels),
            "positive_rate": float(y.mean()),
            "skipped": dict(skipped),
            "split": split_report,
        }
        report["calibration"] = summary
        report_json = json.dumps(_finite(report), ensure_ascii=False, allow_nan=False)
        created_by = payload.get("created_by")
        try:
            repo.insert_detector(
                detector.id,
                detector.to_json(),
                report_json=report_json,
                dataset_id=dataset["id"],
                created_by=created_by if isinstance(created_by, str) else None,
            )
            existed = False
        except ValueError:
            # Mismo dataset y semilla dan el mismo artefacto (id = hash): se actualiza el reporte.
            repo.set_detector_report(detector.id, report_json)
            existed = True
        audit.append(
            self.runtime.db,
            org_id,
            None,
            "calibration_completed",
            detector.id,
            {"job_id": job["id"], "dataset_id": dataset["id"], "n_responses": n_responses},
        )
        self.runtime.emit_event(
            org_id,
            "calibration.completed",
            {"detector_id": detector.id, "dataset_id": dataset["id"], "job_id": job["id"]},
        )
        return {
            "detector_id": detector.id,
            "already_existed": existed,
            "n_items": len(items),
            "n_responses": n_responses,
            "n_tokens": len(labels),
            "skipped": dict(skipped),
        }

    def _collect(
        self, items: list[CalibrationItem]
    ) -> tuple[list[dict[str, float]], list[int], list[str], Counter[str], set[str]]:
        backend = self.runtime.backend
        caps = backend.capabilities()
        tiers = tuple(t for t in CHEAP_TIERS if t in caps.tiers)
        teacher_force = getattr(backend, "teacher_force", None)
        rows: list[dict[str, float]] = []
        labels: list[int] = []
        groups: list[str] = []
        skipped: Counter[str] = Counter()
        kinds: set[str] = set()
        for item in items:
            try:
                if item.kind == "answers":
                    tokens = list(
                        backend.stream(
                            item.messages, max_tokens=ANSWER_MAX_TOKENS, temperature=0.0, tiers=tiers
                        )
                    )
                    text = "".join(t.text for t in tokens if _visible(t))
                    correct = (
                        exact_match(text, item.answers) or token_f1(text, item.answers) >= ANSWER_F1_THRESHOLD
                    )
                    token_labels = label_tokens_from_answer(tokens, None, correct)
                else:
                    if not callable(teacher_force):
                        raise CalibrationJobError(
                            "Este backend no puede puntuar respuestas dadas (teacher forcing). "
                            "Usá items con 'answers'."
                        )
                    tokens = teacher_force(item.messages, item.response, tiers=tiers)
                    if item.kind == "label":
                        token_labels = label_tokens_from_answer(tokens, None, item.label == 0)
                    else:
                        token_labels = spans_to_token_labels(tokens, item.spans)
            except ValueError:
                skipped["contexto_o_entrada_invalida"] += 1
                continue
            used = False
            for tok, lab in zip(tokens, token_labels, strict=True):
                if lab == IGNORE_LABEL:
                    continue
                rows.append(
                    {
                        k: float(v)
                        for k, v in tok.features.items()
                        if v is not None and isinstance(v, int | float)
                    }
                )
                labels.append(int(lab))
                groups.append(item.id)
                used = True
            if used:
                kinds.add(item.kind)
            else:
                skipped["sin_tokens_puntuables"] += 1
        return rows, labels, groups, skipped, kinds


def _canonical(name: str) -> bool:
    try:
        feature_tier(name)
    except ValueError:
        return False
    return True


def _skipped_note(skipped: Counter[str]) -> str:
    if not skipped:
        return ""
    total = sum(skipped.values())
    return f" ({total} items descartados por entrada inválida o sin tokens puntuables)"


__all__ = [
    "JOB_KIND",
    "MAX_ITEMS",
    "MAX_PAGE",
    "MIN_RESPONSES",
    "ActivationError",
    "CalibrationItem",
    "CalibrationJobError",
    "CalibrationService",
    "DatasetError",
    "DatasetNotFound",
    "DetectorNotFound",
    "parse_dataset",
]
