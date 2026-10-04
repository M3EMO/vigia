"""Contratos compartidos de Vigía.

Este módulo es la frontera entre agentes: tipos, nombres de features, artefacto del detector y
formato de salida. Cambiar algo acá exige actualizar `docs/spec/arquitectura.md`.
Solo depende de la stdlib para que cualquier módulo lo importe sin costo.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Literal, Protocol

SCHEMA_VERSION = "1"
DETECTOR_SCHEMA_VERSION = "1"

# --- Niveles y nombres de features -------------------------------------------------------------


class Tier(StrEnum):
    F0 = "F0"  # logits de salida (vocabulario)
    F1 = "F1"  # logits del router + uso de expertos
    U = "U"  # solo IDs de expertos (vLLM / SGLang nativo)
    F2 = "F2"  # salidas por experto (opt-in, caro)
    H = "H"  # residual stream (sonda lineal)


F0_FEATURES = ("out_entropy", "out_nll", "out_margin", "out_top20_entropy")
# Features F1 por capa MoE; el nombre final es f"{base}@{layer}".
F1_LAYER_FEATURES = ("r_ent_n", "r_ent_k", "r_topk_mass", "r_margin12", "r_margin_kk1")
# Features de uso por capa; existen en F1 (ponderadas por el router) y en U (sin ponderar).
USAGE_LAYER_FEATURES = ("u_gini", "u_ihhi", "u_ent", "r_switch")
F2_LAYER_FEATURES = ("e_cos", "e_disp")
F2_GLOBAL_FEATURES = ("e_jsd",)
AGGREGATES = ("mean", "max", "std")


def layer_feature(base: str, layer: int) -> str:
    return f"{base}@{layer}"


def agg_feature(agg: str, base: str) -> str:
    return f"agg_{agg}:{base}"


def feature_tier(name: str) -> Tier:
    """Nivel de una feature a partir de su nombre canónico."""
    base = name.split("@")[0].split(":")[-1]
    if base in F0_FEATURES:
        return Tier.F0
    if base in F1_LAYER_FEATURES or base in USAGE_LAYER_FEATURES:
        return Tier.F1
    if base in F2_LAYER_FEATURES or base in F2_GLOBAL_FEATURES:
        return Tier.F2
    if base.startswith("h_"):
        return Tier.H
    raise ValueError(f"feature desconocida: {name}")


# --- Información del modelo y capacidades ------------------------------------------------------


@dataclass(frozen=True)
class ModelInfo:
    model_id: str  # id de HF o ruta local
    model_type: str  # config.model_type (gpt_oss, mixtral, ...)
    n_layers: int  # capas del decoder
    moe_layers: tuple[int, ...]  # índices de las capas con router
    n_experts: int
    top_k: int
    vocab_size: int
    gating: Literal["softmax_topk", "topk_softmax", "sigmoid_group", "sigmoid_topk", "sparsemixer"]
    dtype: str = "float32"
    quantization: str = "none"  # none | mxfp4 | bnb4 | ...
    revision: str = ""  # commit del checkpoint, si se conoce

    def fingerprint(self) -> dict[str, Any]:
        """Campos que invalidan una calibración si cambian."""
        return {k: getattr(self, k) for k in ("model_id", "model_type", "n_experts", "top_k", "quantization")}


@dataclass(frozen=True)
class Capabilities:
    tiers: tuple[Tier, ...]  # niveles que este backend puede producir
    streaming: bool
    disabled: tuple[str, ...] = ()  # motivos legibles, ej. "F2: expertos MXFP4"
    mode: Literal["full", "reduced"] = "full"  # reduced = sin logits del router (solo IDs)


# --- Eventos por token --------------------------------------------------------------------------


@dataclass
class TokenEvent:
    """Un token generado con sus features. Lo emite un backend; lo consume detector/spans/api."""

    index: int  # posición dentro de la respuesta (0 = primer token generado)
    token_id: int
    text: str  # texto decodificado que aporta este token (puede ser "" si cierra bytes parciales)
    start: int  # offset en code points dentro del texto de la respuesta
    end: int
    features: dict[str, float]  # nombre canónico -> valor; NaN si no aplica
    # final | analysis | commentary (harmony) | header | control | none. Solo `final` no especial se puntúa
    # (regla única: vigia.harmony.is_scorable). Modelos sin harmony: contenido = final, especiales = control.
    channel: str = "final"
    is_special: bool = False
    logprob: float | None = None
    expert_ids: list[list[int]] | None = None  # [capa_moe][k], para el heatmap capa x experto


@dataclass
class Generation:
    """Resultado completo de un backend para un prompt."""

    text: str
    tokens: list[TokenEvent]
    finish_reason: Literal["stop", "length", "abstain", "error"]
    prompt_tokens: int
    model: ModelInfo
    mode: Literal["full", "reduced"] = "full"
    timings: dict[str, float] = field(default_factory=dict)  # prefill_s, decode_s, overhead_s


class TokenStream(Protocol):
    """Lo que devuelve Backend.stream: un iterador de TokenEvent con datos del request al terminar."""

    finish_reason: Literal["stop", "length"] | None  # None hasta que el iterador se agota
    text: str  # texto de la respuesta (solo canal final)
    prompt_tokens: int
    completion_tokens: int
    timings: dict[str, float]  # prefill_s, decode_s, total_s

    def __iter__(self) -> Iterator[TokenEvent]: ...

    def __next__(self) -> TokenEvent: ...

    def close(self) -> None:
        """Libera el backend (lock, hooks). Obligatorio si el consumidor corta antes del final."""
        ...


class Backend(Protocol):
    """Contrato de todo backend de generación (HF, proxy vLLM)."""

    def model_info(self) -> ModelInfo: ...

    def capabilities(self) -> Capabilities: ...

    def stream(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int | None = None,
        stop: list[str] | None = None,
        tiers: tuple[Tier, ...] = (Tier.F0, Tier.F1),
    ) -> TokenStream: ...


class Scorer(Protocol):
    """Contrato del detector visto desde el servicio (lo implementa `vigia.detector.Detector`)."""

    id: str  # sha256 corto del artefacto = calibration_id

    def score_tokens(self, tokens: list[TokenEvent]) -> list[float | None]:
        """p_error calibrado por token; None para especiales y canales que no son `final`."""
        ...

    def score_response(self, token_p_errors: list[float | None]) -> float | None:
        """p_error de la respuesta (agregación max + calibrador de respuesta)."""
        ...

    def tau(self, alpha: float) -> float:
        """Umbral por token que garantiza FNR <= alpha (CRC). Interpola entre los alphas guardados."""
        ...

    def check_model(self, info: ModelInfo) -> None:
        """Lanza DetectorMismatchError si el detector fue calibrado para otro modelo."""
        ...


# --- Fragmentos, políticas y salida ------------------------------------------------------------

Action = Literal["pass", "annotate", "human_review", "rag_retry", "abstain"]
ACTION_PRIORITY: tuple[Action, ...] = ("abstain", "rag_retry", "human_review", "annotate", "pass")


@dataclass
class Span:
    start: int  # offsets en code points dentro del texto de la respuesta
    end: int
    token_start: int  # índices de TokenEvent, [token_start, token_end)
    token_end: int
    score: float  # max p_error de los tokens del fragmento
    text: str = ""


@dataclass
class Policy:
    """Política por tenant. Los umbrales de acción comparan el p_error de la respuesta."""

    alpha: float = 0.1  # FNR objetivo por token; elige tau del detector
    gap_max: int = 1  # palabras no marcadas que se toleran dentro de un fragmento
    min_tokens: int = 1
    annotate: bool = True
    human_review_at: float | None = 0.5
    rag_retry_at: float | None = None
    abstain_at: float | None = None
    abstain_message: str = "No tengo una respuesta confiable para esta consulta."
    rag_url: str | None = None
    rag_max_retries: int = 1
    rag_fallback_action: Literal["human_review", "abstain"] = "human_review"
    on_detector_error: Literal["fail_closed", "fail_open"] = "fail_closed"
    stream_policy: Literal["annotate_only", "buffer"] = "annotate_only"
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScoredToken:
    index: int
    text: str
    start: int
    end: int
    p_error: float | None  # None: sin detector calibrado, token especial o canal no puntuado
    flag: bool


@dataclass
class VigiaResult:
    """Lo que viaja en el campo `vigia` de la respuesta OpenAI-compatible."""

    trace_id: str
    tokens: list[ScoredToken]
    spans: list[Span]
    response_p_error: float | None
    action: Action
    calibration_id: str | None  # sha256 corto del detector; None si no hay calibración
    calibrated: bool
    mode: Literal["full", "reduced"]
    degraded: bool = False
    # ok | no_final_channel | truncated_before_final | empty_generation | cancelled | detector_error
    status: str = "ok"
    rag_status: str | None = None  # None | retried | error | unavailable
    alpha: float | None = None
    tau: float | None = None
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --- Errores ------------------------------------------------------------------------------------


class VigiaError(Exception):
    """Error base de Vigía."""


class UnsupportedModelError(VigiaError):
    pass


class HookNotFiredError(VigiaError):
    pass


class DetectorMismatchError(VigiaError):
    """El detector fue calibrado para otro modelo o su hash no valida."""


class InvalidRequestError(VigiaError, ValueError):
    """El backend rechaza la entrada del cliente (prompt fuera del contexto, texto inválido).

    La API lo responde como 400 `invalid_request_error` con este mensaje y `code`; no es un error
    interno. El mensaje no debe incluir datos internos (rutas, pesos, configuración).
    """

    def __init__(self, message: str, *, code: str = "invalid_prompt") -> None:
        super().__init__(message)
        self.code = code
