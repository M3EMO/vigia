"""Backend de modo reducido sobre vLLM (ADR A-08, P-15).

Vigía actúa como proxy de un vLLM lanzado con `--enable-return-routed-experts`. vLLM no expone los
logits del router: solo los IDs de expertos elegidos (`routed_experts`, forma [gen_len, capas, top_k])
al final del request, más logprobs top-20. Con eso Vigía calcula:

- F0 parcial: `out_nll`, `out_margin` (probabilidades reales del top-2) y `out_top20_entropy`.
  `out_entropy` necesita el vocabulario completo y queda en NaN.
- U: `u_gini`, `u_ihhi`, `u_ent` y `r_switch` por capa sobre el uso acumulado sin ponderar, más sus
  agregados `agg_{mean,max,std}` entre capas.

ponytail: las fórmulas U y F0 de este módulo son locales porque `vigia.features` se escribe en paralelo.
Unificarlas con `vigia.features` cuando exista, con un test de igualdad numérica.
"""

from __future__ import annotations

import base64
import bisect
import math
from collections.abc import Iterator
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx
import numpy as np

from vigia.contracts import (
    AGGREGATES,
    USAGE_LAYER_FEATURES,
    Capabilities,
    ModelInfo,
    Tier,
    TokenEvent,
    VigiaError,
    agg_feature,
    layer_feature,
)

NAN = float("nan")
ROUTED_DTYPES = {"uint8": np.dtype("<u1"), "uint16": np.dtype("<u2"), "int16": np.dtype("<i2")}


class VllmProxyError(VigiaError):
    """vLLM falló o devolvió una respuesta que Vigía no puede interpretar."""


# --- Fórmulas locales ---------------------------------------------------------------------------


def top_logprob_features(logprob: float | None, top: list[float]) -> dict[str, float]:
    """F0 parcial desde el logprob emitido y los logprobs top-k (k <= 20)."""
    feats = {"out_entropy": NAN, "out_nll": NAN, "out_margin": NAN, "out_top20_entropy": NAN}
    if logprob is not None and math.isfinite(logprob):
        feats["out_nll"] = -float(logprob)
    probs = np.exp(np.asarray(sorted(top, reverse=True), dtype=np.float64)) if top else np.zeros(0)
    if probs.size >= 2:
        feats["out_margin"] = float(probs[0] - probs[1])
    total = float(probs.sum())
    if probs.size and total > 0:
        q = probs / total
        q = q[q > 0]
        feats["out_top20_entropy"] = float(-(q * np.log(q)).sum())
    return feats


def usage_features(counts: np.ndarray, n_experts: int) -> tuple[float, float, float]:
    """(u_gini, u_ihhi, u_ent) de un vector de conteos acumulados de una capa.

    u_gini es la impureza de Gini 1 - Σ u_i²; u_ihhi es (1 / Σ u_i²) / N; u_ent es la entropía / log N.
    """
    total = float(counts.sum())
    if total <= 0:
        return NAN, NAN, NAN
    u = counts.astype(np.float64) / total
    s2 = float((u * u).sum())
    nz = u[u > 0]
    ent = float(-(nz * np.log(nz)).sum())
    u_ent = ent / math.log(n_experts) if n_experts > 1 else 0.0
    return 1.0 - s2, (1.0 / s2) / n_experts, u_ent


def routed_usage_features(routed: np.ndarray, n_experts: int) -> list[dict[str, float]]:
    """Features U causales por token desde IDs [T, L, k]. El valor en t usa solo los tokens <= t."""
    t_len, n_layers, k = routed.shape
    counts = np.zeros((n_layers, n_experts), dtype=np.int64)
    out: list[dict[str, float]] = []
    prev: list[set[int]] | None = None
    for t in range(t_len):
        feats: dict[str, float] = {}
        per_base: dict[str, list[float]] = {b: [] for b in USAGE_LAYER_FEATURES}
        current: list[set[int]] = []
        for layer in range(n_layers):
            ids = routed[t, layer]
            np.add.at(counts[layer], ids, 1)
            gini, ihhi, ent = usage_features(counts[layer], n_experts)
            sel = {int(i) for i in ids}
            switch = NAN if prev is None else 1.0 - len(sel & prev[layer]) / k
            current.append(sel)
            for base, val in zip(USAGE_LAYER_FEATURES, (gini, ihhi, ent, switch), strict=True):
                feats[layer_feature(base, layer)] = val
                per_base[base].append(val)
        for base, vals in per_base.items():
            arr = np.asarray(vals, dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            for agg in AGGREGATES:
                if arr.size == 0:
                    v = NAN
                elif agg == "mean":
                    v = float(arr.mean())
                elif agg == "max":
                    v = float(arr.max())
                else:
                    v = float(arr.std())
                feats[agg_feature(agg, base)] = v
        prev = current
        out.append(feats)
    return out


def nan_usage(n_layers: int) -> dict[str, float]:
    """Features U en NaN para una posición sin IDs de expertos."""
    feats = {layer_feature(b, layer): NAN for layer in range(n_layers) for b in USAGE_LAYER_FEATURES}
    feats.update({agg_feature(a, b): NAN for b in USAGE_LAYER_FEATURES for a in AGGREGATES})
    return feats


# --- Parseo de la respuesta de vLLM -------------------------------------------------------------


def parse_routed_experts(raw: Any) -> np.ndarray:
    """Convierte `routed_experts` a int64 [len, capas, top_k].

    Acepta listas anidadas o un objeto {"dtype": uint8|uint16|int16, "shape": [...], "data": base64}.
    """
    if isinstance(raw, dict):
        dtype_name = raw.get("dtype")
        if dtype_name not in ROUTED_DTYPES:
            raise VllmProxyError(f"dtype de routed_experts no soportado: {dtype_name!r}")
        shape = raw.get("shape")
        valid_shape = isinstance(shape, list) and len(shape) == 3
        if not (valid_shape and all(isinstance(s, int) and s >= 0 for s in shape)):
            raise VllmProxyError("routed_experts.shape debe ser [len, capas, top_k]")
        data = raw.get("data")
        if not isinstance(data, str):
            raise VllmProxyError("routed_experts.data debe ser base64")
        try:
            buf = base64.b64decode(data, validate=True)
        except ValueError as exc:
            raise VllmProxyError("routed_experts.data no es base64 válido") from exc
        dt = ROUTED_DTYPES[dtype_name]
        if len(buf) != int(np.prod(shape)) * dt.itemsize:
            raise VllmProxyError("el tamaño de routed_experts.data no coincide con shape y dtype")
        arr = np.frombuffer(buf, dtype=dt).reshape(shape).astype(np.int64)
    elif isinstance(raw, list):
        try:
            arr = np.asarray(raw, dtype=np.int64)
        except (ValueError, TypeError, OverflowError) as exc:
            raise VllmProxyError("routed_experts debe ser una lista rectangular de enteros") from exc
        if arr.size == 0:
            arr = arr.reshape(0, 0, 0)
    else:
        raise VllmProxyError("routed_experts tiene un formato desconocido")
    if arr.ndim != 3:
        raise VllmProxyError("routed_experts debe tener forma [len, capas, top_k]")
    if arr.size and arr.min() < 0:
        raise VllmProxyError("routed_experts contiene IDs negativos")
    return arr


def token_offsets(pieces: list[str], raw_bytes: list[list[int] | None]) -> tuple[str, list[tuple[int, int]]]:
    """Texto de la respuesta y offsets [start, end) en code points por token, sin huecos ni solapes.

    Con bytes UTF-8 válidos, un carácter pertenece al token que aporta su último byte; un token que
    solo trae bytes parciales queda con ancho cero. Sin bytes, se usa el texto de cada token.
    """
    if pieces and all(b is not None for b in raw_bytes):
        byte_lists = [bytes(b or []) for b in raw_bytes]
        try:
            text = b"".join(byte_lists).decode("utf-8")
        except UnicodeDecodeError:
            text = None
        if text is not None:
            char_byte_end: list[int] = []
            acc = 0
            for ch in text:
                acc += len(ch.encode("utf-8"))
                char_byte_end.append(acc)
            offsets = []
            b0 = 0
            for bl in byte_lists:
                b1 = b0 + len(bl)
                # caracteres cuyo último byte cae en (b0, b1]
                start = bisect.bisect_right(char_byte_end, b0)
                end = bisect.bisect_right(char_byte_end, b1)
                offsets.append((start, end))
                b0 = b1
            return text, offsets
    offsets = []
    pos = 0
    for p in pieces:
        offsets.append((pos, pos + len(p)))
        pos += len(p)
    return "".join(pieces), offsets


# --- Backend ------------------------------------------------------------------------------------


class VllmProxyBackend:
    """Cumple `contracts.Backend` en modo reducido contra un servidor vLLM OpenAI-compatible.

    `n_experts` debe coincidir con el modelo servido. Sin ese dato, las features U se normalizan con
    el máximo ID observado + 1 en cada respuesta y la capacidad lo declara en `disabled`.
    """

    mode: Literal["reduced"] = "reduced"

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        http_client: httpx.Client | None = None,
        *,
        n_experts: int | None = None,
        top_k: int = 0,
        n_layers: int = 0,
        model_type: str = "unknown",
        timeout: float = 120.0,
    ) -> None:
        parts = urlsplit(base_url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError("base_url debe ser una URL http(s) con host")
        if n_experts is not None and n_experts < 2:
            raise ValueError("n_experts debe ser >= 2")
        self.base_url = base_url.rstrip("/")
        if self.base_url.endswith("/v1"):
            self.base_url = self.base_url[:-3]
        self.model = model
        self._api_key = api_key
        self._client = http_client or httpx.Client(timeout=timeout)
        self._owns_client = http_client is None
        self.timeout = timeout
        self.n_experts = n_experts
        self.top_k = top_k
        self.n_layers = n_layers
        self.model_type = model_type

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            model_id=self.model,
            model_type=self.model_type,
            n_layers=self.n_layers,
            moe_layers=tuple(range(self.n_layers)),
            n_experts=self.n_experts or 0,
            top_k=self.top_k,
            vocab_size=0,
            gating="softmax_topk",
        )

    def capabilities(self) -> Capabilities:
        disabled = [
            "F0: out_entropy necesita el vocabulario completo; vLLM entrega logprobs top-20",
            "F1: vLLM no expone los logits del router",
            "streaming: routed_experts llega al terminar el request",
        ]
        if self.n_experts is None:
            disabled.append("U: n_experts desconocido; la normalización usa el máximo ID observado")
        return Capabilities(tiers=(Tier.F0, Tier.U), streaming=False, disabled=tuple(disabled))

    def _request(self, body: dict[str, Any]) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        try:
            resp = self._client.post(
                f"{self.base_url}/v1/chat/completions", json=body, headers=headers, timeout=self.timeout
            )
        except httpx.HTTPError as exc:
            raise VllmProxyError(f"vLLM no respondió: {type(exc).__name__}") from exc
        if resp.status_code != 200:
            raise VllmProxyError(f"vLLM respondió HTTP {resp.status_code}")
        try:
            data = resp.json()
        except ValueError as exc:
            raise VllmProxyError("vLLM respondió JSON inválido") from exc
        if not isinstance(data, dict) or not isinstance(data.get("choices"), list) or not data["choices"]:
            raise VllmProxyError("la respuesta de vLLM no tiene choices")
        return data

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
    ) -> Iterator[TokenEvent]:
        want_u = Tier.U in tiers
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "logprobs": True,
            "top_logprobs": 20,
            "stream": False,
        }
        if want_u:
            body["return_routed_experts"] = True
        if seed is not None:
            body["seed"] = seed
        if stop:
            body["stop"] = stop
        data = self._request(body)
        choice = data["choices"][0]
        if not isinstance(choice, dict):
            raise VllmProxyError("choice inválido en la respuesta de vLLM")
        content = ((choice.get("logprobs") or {}).get("content")) or []
        if not isinstance(content, list):
            raise VllmProxyError("logprobs.content debe ser una lista")
        if not content and ((choice.get("message") or {}).get("content") or ""):
            raise VllmProxyError("vLLM no devolvió logprobs; Vigía los necesita para puntuar")

        pieces: list[str] = []
        raw_bytes: list[list[int] | None] = []
        logprobs: list[float | None] = []
        tops: list[list[float]] = []
        for entry in content:
            if not isinstance(entry, dict) or not isinstance(entry.get("token"), str):
                raise VllmProxyError("entrada de logprobs inválida")
            pieces.append(entry["token"])
            b = entry.get("bytes")
            valid_bytes = isinstance(b, list) and all(isinstance(x, int) and 0 <= x < 256 for x in b)
            raw_bytes.append(b if valid_bytes else None)
            lp = entry.get("logprob")
            logprobs.append(float(lp) if isinstance(lp, int | float) else None)
            tops.append(
                [
                    float(t["logprob"])
                    for t in entry.get("top_logprobs") or []
                    if isinstance(t, dict) and isinstance(t.get("logprob"), int | float)
                ]
            )
        text, offsets = token_offsets(pieces, raw_bytes)

        usage: list[dict[str, float]] | None = None
        routed: np.ndarray | None = None
        if want_u:
            raw = choice.get("routed_experts", data.get("routed_experts"))
            if raw is None:
                raise VllmProxyError(
                    "vLLM no devolvió routed_experts; lanzalo con --enable-return-routed-experts"
                )
            routed = parse_routed_experts(raw)
            n_tok = len(pieces)
            if routed.shape[0] not in (n_tok, n_tok - 1) or (routed.shape[0] and routed.shape[2] == 0):
                raise VllmProxyError(
                    f"routed_experts tiene {routed.shape[0]} posiciones para {n_tok} tokens generados"
                )
            n_exp = self.n_experts or (int(routed.max()) + 1 if routed.size else 2)
            if routed.size and int(routed.max()) >= n_exp:
                raise VllmProxyError("routed_experts contiene IDs >= n_experts")
            usage = routed_usage_features(routed, max(n_exp, 2))

        for i in range(len(pieces)):
            feats = top_logprob_features(logprobs[i], tops[i])
            expert_ids = None
            if usage is not None and routed is not None:
                if i < len(usage):
                    feats.update(usage[i])
                    expert_ids = routed[i].tolist()
                else:  # vLLM no devolvió la posición final: U no aplica
                    feats.update(nan_usage(routed.shape[1]))
            start, end = offsets[i]
            yield TokenEvent(
                index=i,
                token_id=-1,  # vLLM no expone IDs de token en chat logprobs
                text=text[start:end],
                start=start,
                end=end,
                features=feats,
                logprob=logprobs[i],
                expert_ids=expert_ids,
            )
