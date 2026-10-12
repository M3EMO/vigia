"""Extractor de señales por token con hooks sobre los bloques MoE (ADR A-01).

Protocolo de un paso (prefill o un token de decodificación):

    with SignalExtractor(model, tiers=(Tier.F0, Tier.F1)) as ex:
        ex.begin_step(attention_mask=mask, score_mask=score)   # opcional
        out = model(input_ids=ids, attention_mask=mask, ...)
        rows = ex.end_step(out.logits, targets)               # list[list[TokenSignals]] por fila

Alineación: la posición p de un forward produce las features del token muestreado desde los
logits de p (`targets[b, p]`). Así F0 y F1 describen el mismo cómputo que eligió el token.

- F1: un `forward_pre_hook` sobre cada bloque MoE recalcula `F.linear(x, W_router, b)` en float32,
  así que funciona aunque el bloque no llame al router (MXFP4). El hook solo guarda esos logits
  del paso (tensor `(n_tokens, N)` float32, sin grafo).
- Regla de reducción: la reducción ocurre en `end_step`, por paso; nunca se acumulan logits entre
  pasos. `end_step` apila las capas `(L, n, N)`, se queda con las filas que hacen falta (puntuadas,
  la válida previa a cada una y la última válida), convierte a numpy una sola vez y reduce todas
  las capas y tokens del paso con una sola llamada vectorizada a `vigia.features`. Entre pasos solo
  persiste el estado causal chico: uso acumulado `(B, L, N)` y la última selección `(L, k)` por fila.
  La reducción usa numpy y no torch: con tensores de pocas decenas de elementos manda el costo de
  despacho por operación, y torch tarda ~3x más que numpy en las mismas fórmulas (medido en CPU).
- U/F1: el uso acumulado y `r_switch` son causales por fila: solo cuentan posiciones puntuadas
  (tokens generados) hasta t. `r_switch` compara con la posición válida anterior.
- F2 (opt-in): un forward hook sobre el módulo de expertos recalcula la salida de cada experto
  elegido sin tocar la salida del modelo. Con expertos cuantizados queda deshabilitado.
- H: residual stream tras `probe_layer` capas del decoder (convención `hidden_states[i]` de HF).
- Las posiciones con `attention_mask == 0` no generan features ni actualizan estado (AT-S05).
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from vigia import features as fx
from vigia.adapters import MoEBlock, decoder_layers, final_norm, find_moe_blocks, routing_params, top_k
from vigia.contracts import (
    AGGREGATES,
    F1_LAYER_FEATURES,
    F2_LAYER_FEATURES,
    USAGE_LAYER_FEATURES,
    HookNotFiredError,
    Tier,
    agg_feature,
    layer_feature,
)

_ROUTER_TIERS = frozenset({Tier.F1, Tier.U})
_F2_CHUNK = 64  # filas por lote al recalcular expertos
_LENS_CHUNK = 8  # filas por lote en el logit lens (k × |V| floats por fila)


@dataclass
class TokenSignals:
    features: dict[str, float]
    expert_ids: list[list[int]] | None = None  # [capa_moe][k]


@dataclass
class _Step:
    valid: np.ndarray | None = None  # (B, T) bool
    scored: np.ndarray | None = None  # (B, T) bool
    shape: tuple[int, int] | None = None  # (B, T) visto por los hooks
    router: dict[int, torch.Tensor] = field(default_factory=dict)  # capa j -> logits (n, N) float32
    experts: dict[int, dict[str, np.ndarray]] = field(default_factory=dict)
    hidden: tuple[np.ndarray, np.ndarray] | None = None  # (filas planas, valores (n, d))
    fired: set[tuple[str, int]] = field(default_factory=set)

    def score_rows(self, n: int) -> torch.Tensor | None:
        if self.scored is None or self.scored.size != n:
            return None
        return torch.from_numpy(np.flatnonzero(self.scored.reshape(-1)))


def _inference() -> Any:
    """`inference_mode` solo si no está activo: entrar al contexto cuesta ~100 µs por hook."""
    return nullcontext() if torch.is_inference_mode_enabled() else torch.inference_mode()


def _is_float_3d(t: Any) -> bool:
    return isinstance(t, torch.Tensor) and t.dim() == 3 and t.is_floating_point()


def f2_unsupported_reason(blocks: list[MoEBlock]) -> str | None:
    """Motivo legible si no se pueden recalcular las salidas por experto; None si se puede."""
    for b in blocks:
        ex = b.experts
        if ex is None:
            return f"F2: el bloque de la capa {b.layer} no tiene módulo de expertos"
        up = getattr(ex, "gate_up_proj" if getattr(ex, "has_gate", False) else "up_proj", None)
        flags = all(hasattr(ex, a) for a in ("has_gate", "has_bias", "is_transposed"))
        if not (flags and _is_float_3d(up) and _is_float_3d(getattr(ex, "down_proj", None))):
            return f"F2: expertos cuantizados o con kernel fusionado ({type(ex).__name__})"
    return None


def expert_outputs(experts: nn.Module, hidden: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Salidas sin ponderar de los expertos elegidos: (n, H) y (n, k) → (n, k, H) en float32.

    Replica el forward eager de los módulos `*Experts` de transformers v5 (pesos 3D, gate opcional,
    bias opcional, layout transpuesto) experto por experto, sin materializar pesos por token.
    """
    n, k = idx.shape
    x = hidden.float()
    out = torch.zeros((n, k, x.shape[-1]), dtype=torch.float32, device=x.device)
    gated = experts.has_gate
    up_w = experts.gate_up_proj if gated else experts.up_proj
    up_b = (experts.gate_up_proj_bias if gated else experts.up_proj_bias) if experts.has_bias else None
    down_b = experts.down_proj_bias if experts.has_bias else None

    def linear(h: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None) -> torch.Tensor:
        y = h @ w.float() if experts.is_transposed else F.linear(h, w.float())
        return y if b is None else y + b.float()

    for e in torch.unique(idx).tolist():
        rows, slots = torch.where(idx == e)
        proj = linear(x[rows], up_w[e], None if up_b is None else up_b[e])
        proj = experts._apply_gate(proj) if gated else experts.act_fn(proj)
        out[rows, slots] = linear(proj, experts.down_proj[e], None if down_b is None else down_b[e])
    return out


def resolve_probe_layer(probe_layer: int | None, n_layers: int) -> int:
    """Capa de la sonda H: por defecto round(0.95·L). Debe caer en [1, L]."""
    p = round(0.95 * n_layers) if probe_layer is None else int(probe_layer)
    if not 1 <= p <= n_layers:
        raise ValueError(f"probe_layer={p} fuera de [1, {n_layers}]")
    return p


def _same_route_kwargs(a: dict[str, Any], b: dict[str, Any]) -> bool:
    if a.keys() != b.keys():
        return False
    for key, va in a.items():
        vb = b[key]
        if isinstance(va, np.ndarray) or isinstance(vb, np.ndarray):
            if va is None or vb is None or not np.array_equal(va, vb):
                return False
        elif va != vb:
            return False
    return True


def _as_bool(mask: Any) -> np.ndarray | None:
    if mask is None:
        return None
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError("las máscaras deben tener forma (batch, tiempo)")
    return arr.astype(bool)


class SignalExtractor:
    """Context manager que instala los hooks y produce `TokenSignals` por token puntuado."""

    def __init__(
        self,
        model: nn.Module,
        tiers: tuple[Tier | str, ...] = (Tier.F0, Tier.F1),
        probe_layer: int | None = None,
        *,
        keep_router_logits: bool = False,
    ) -> None:
        self.model = model
        self.blocks = find_moe_blocks(model)
        self.layers = tuple(b.layer for b in self.blocks)
        self.n_experts = int(self.blocks[0].router.weight.shape[0])
        self.top_k = top_k(model)
        self._route_kw = [routing_params(b) for b in self.blocks]
        # Una sola llamada a features.router_stats para todas las capas si comparten el ruteo.
        self._shared_route = len({b.spec.gating for b in self.blocks}) == 1 and all(
            _same_route_kwargs(kw, self._route_kw[0]) for kw in self._route_kw
        )
        self.disabled: list[str] = []
        wanted = {Tier(t) for t in tiers}
        if Tier.F2 in wanted:
            reason = f2_unsupported_reason(self.blocks)
            if reason:
                self.disabled.append(reason)
                wanted.discard(Tier.F2)
        self.tiers = frozenset(wanted)
        self._weighted = Tier.F1 in self.tiers  # F1 pondera el uso con el router; U cuenta IDs
        bases = F1_LAYER_FEATURES + USAGE_LAYER_FEATURES if self._weighted else USAGE_LAYER_FEATURES
        self._router_names = [layer_feature(base, layer) for base in bases for layer in self.layers] + [
            agg_feature(agg, base) for base in bases for agg in AGGREGATES
        ]
        self._f2_names = [layer_feature(base, b.layer) for b in self.blocks for base in F2_LAYER_FEATURES]
        self._lens: tuple[nn.Module, nn.Module] | None = None
        if Tier.F2 in self.tiers:
            norm, head = final_norm(model), model.get_output_embeddings()
            if norm is None or head is None:
                self.disabled.append("F2: e_jsd sin norma final o cabeza de salida")
            else:
                self._lens = (norm, head)
        self.probe_layer = (
            resolve_probe_layer(probe_layer, len(decoder_layers(model))) if Tier.H in self.tiers else None
        )
        self.keep_router_logits = keep_router_logits
        self.router_logits: dict[int, torch.Tensor] = {}  # capa -> (B, T, N) del último paso
        self._counts: np.ndarray | None = None  # (B, L, N): uso acumulado por fila
        self._prev_sel: list[np.ndarray | None] = []  # por fila: selección (L, k) de la última válida
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._router_w: dict[int, tuple[torch.Tensor, torch.Tensor | None]] = {}
        self._step: _Step | None = None
        self.reset()

    # --- ciclo de vida ----------------------------------------------------------------------

    def __enter__(self) -> SignalExtractor:
        try:
            if self.tiers & _ROUTER_TIERS:
                for j, b in enumerate(self.blocks):
                    self._handles.append(
                        b.block.register_forward_pre_hook(partial(self._router_hook, j), with_kwargs=True)
                    )
            if Tier.F2 in self.tiers:
                for j, b in enumerate(self.blocks):
                    self._handles.append(
                        b.experts.register_forward_hook(partial(self._experts_hook, j), with_kwargs=True)
                    )
            if self.probe_layer is not None:
                layer = decoder_layers(self.model)[self.probe_layer - 1]
                self._handles.append(layer.register_forward_hook(self._probe_hook))
        except BaseException:
            self._remove_hooks()
            raise
        return self

    def __exit__(self, *exc: object) -> None:
        self._remove_hooks()
        self._router_w.clear()  # los pesos pueden cambiar entre contextos (ajuste fino)

    def _remove_hooks(self) -> None:
        try:
            for h in self._handles:
                h.remove()
        finally:
            self._handles.clear()
            self._step = None

    def reset(self) -> None:
        """Borra el estado causal (uso acumulado, selección previa). Llamar entre secuencias."""
        self._counts = None
        self._prev_sel = []

    # --- hooks ------------------------------------------------------------------------------

    def _current(self) -> _Step:
        if self._step is None:
            self._step = _Step()
        return self._step

    def _router_hook(self, j: int, module: nn.Module, args: tuple, kwargs: dict) -> None:
        """Solo recalcula y guarda los logits del router del paso; la reducción es de `end_step`."""
        x = args[0] if args else next(v for v in kwargs.values() if isinstance(v, torch.Tensor))
        block = self.blocks[j]
        with _inference():
            h = x.detach().reshape(-1, x.shape[-1]).float()
            w = self._router_w.get(j)
            if w is None or w[0].device != h.device:
                # Pesos en float32 una vez por contexto: convertirlos en cada token costaba un kernel
                # y una copia por capa y paso.
                bias = getattr(block.router, "bias", None)
                w = self._router_w[j] = (
                    block.router.weight.detach().float(),
                    None if bias is None else bias.detach().float(),
                )
            logits = F.linear(h, *w)
        step = self._current()
        step.shape = tuple(x.shape[:-1]) if x.dim() == 3 else (1, h.shape[0])
        step.router[j] = logits
        step.fired.add(("router", j))
        if self.keep_router_logits:
            self.router_logits[block.layer] = logits.reshape(*step.shape, -1)

    def _experts_hook(self, j: int, module: nn.Module, args: tuple, kwargs: dict, output: Any) -> None:
        hidden = args[0] if args else kwargs["hidden_states"]
        idx = args[1] if len(args) > 1 else kwargs.get("top_k_index", kwargs.get("router_indices"))
        w = args[2] if len(args) > 2 else kwargs.get("top_k_weights", kwargs.get("routing_weights"))
        step = self._current()
        n = hidden.shape[0]
        rows = step.score_rows(n)
        rows = torch.arange(n) if rows is None else rows
        result = {name: np.full(n, np.nan) for name in (*F2_LAYER_FEATURES, "e_jsd")}
        last = j == len(self.blocks) - 1
        with _inference():
            for start in range(0, len(rows), _F2_CHUNK):
                sel = rows[start : start + _F2_CHUNK].to(hidden.device)
                e = expert_outputs(module, hidden.detach()[sel], idx.detach()[sel])
                w_np = w.detach()[sel].float().cpu().numpy()
                e_np = e.cpu().numpy()
                r = sel.cpu().numpy()
                result["e_cos"][r] = fx.expert_cosine(e_np, w_np)
                result["e_disp"][r] = fx.expert_dispersion(e_np, w_np)
                if last and self._lens is not None:
                    result["e_jsd"][r] = self._jsd(e, w_np)
        step.experts[j] = result
        step.fired.add(("experts", j))

    def _jsd(self, e: torch.Tensor, w: np.ndarray) -> np.ndarray:
        """JSD entre las distribuciones logit-lens (norma final + lm_head) de los expertos elegidos."""
        norm, head = self._lens  # type: ignore[misc]
        scale = float(getattr(self.model.config, "logits_scaling", 1.0) or 1.0)
        norm_dtype = next(norm.parameters(), e).dtype
        out = np.empty(e.shape[0])
        for start in range(0, e.shape[0], _LENS_CHUNK):
            chunk = e[start : start + _LENS_CHUNK]
            normed = norm(chunk.to(norm_dtype)).float()
            bias = None if getattr(head, "bias", None) is None else head.bias.float()
            logits = F.linear(normed, head.weight.float(), bias) / scale
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            out[start : start + _LENS_CHUNK] = fx.weighted_jsd(probs, w[start : start + _LENS_CHUNK])
        return out

    def _probe_hook(self, module: nn.Module, args: tuple, output: Any) -> None:
        hs = output[0] if isinstance(output, tuple) else output
        flat = hs.detach().reshape(-1, hs.shape[-1])
        step = self._current()
        rows = step.score_rows(flat.shape[0])
        rows = torch.arange(flat.shape[0]) if rows is None else rows
        with _inference():
            step.hidden = (rows.numpy(), flat[rows.to(flat.device)].float().cpu().numpy())
        step.fired.add(("probe", 0))

    # --- protocolo por paso -----------------------------------------------------------------

    def begin_step(self, attention_mask: Any = None, score_mask: Any = None) -> None:
        """Declara las máscaras (B, T) del próximo forward. Sin máscaras, todo es válido y puntuado."""
        valid = _as_bool(attention_mask)
        scored = _as_bool(score_mask)
        if valid is not None and scored is not None and valid.shape != scored.shape:
            raise ValueError("attention_mask y score_mask deben tener la misma forma")
        if scored is not None and valid is not None:
            scored = scored & valid
        elif scored is None:
            scored = valid
        self.router_logits.clear()
        self._step = _Step(valid=valid, scored=scored)

    def _expected_hooks(self) -> set[tuple[str, int]]:
        expected: set[tuple[str, int]] = set()
        if self.tiers & _ROUTER_TIERS:
            expected |= {("router", j) for j in range(len(self.blocks))}
        if Tier.F2 in self.tiers:
            expected |= {("experts", j) for j in range(len(self.blocks))}
        if self.probe_layer is not None:
            expected.add(("probe", 0))
        return expected

    def end_step(self, logits: torch.Tensor | None = None, targets: Any = None) -> list[list[TokenSignals]]:
        """Reduce el último forward a `TokenSignals` por fila, en orden de posición.

        `logits` (B, T', V) y `targets` (B, T') cubren las últimas T' posiciones del paso (como
        `logits_to_keep` de HF). Hacen falta si el nivel F0 está activo.
        """
        step = self._step or _Step()
        self._step = None
        missing = self._expected_hooks() - step.fired
        if missing:
            where = sorted(
                {self.blocks[j].layer if kind != "probe" else self.probe_layer for kind, j in missing}
            )
            raise HookNotFiredError(
                f"los hooks no dispararon en las capas {where}: el forward no pasó por los bloques "
                "instrumentados (¿kernel fusionado, torch.compile o forward fuera del contexto?)"
            )
        shape = step.shape
        for m in (step.scored, step.valid):
            if m is not None:
                if shape is not None and tuple(m.shape) != tuple(shape):
                    raise ValueError(f"máscara {tuple(m.shape)} no coincide con el forward {shape}")
                shape = tuple(m.shape)
        if shape is None:
            if logits is None:
                raise ValueError("no hubo forward ni logits para este paso")
            shape = tuple(logits.shape[:2])
        bsz, seq = shape
        valid = step.valid if step.valid is not None else np.ones(shape, dtype=bool)
        scored = step.scored if step.scored is not None else valid
        if self._counts is None:
            self._counts = np.zeros((bsz, len(self.blocks), self.n_experts))
            self._prev_sel = [None] * bsz
        elif self._counts.shape[0] != bsz:
            raise ValueError("el tamaño de batch cambió: llamar a reset() entre secuencias")

        bs, ps = np.nonzero(scored)  # posiciones puntuadas, por fila y en orden de posición
        flat = bs * seq + ps
        names: list[str] = []
        columns: list[np.ndarray] = []
        expert_ids: list[list[list[int]]] | None = None
        if Tier.F0 in self.tiers:
            f0 = self._f0(logits, targets, bs, ps, seq)
            names += list(f0)
            columns.append(np.array(list(f0.values())).T.reshape(len(flat), len(f0)))
        if self.tiers & _ROUTER_TIERS:
            values, expert_ids = self._router_step(step, valid, scored, seq)
            names += self._router_names
            columns.append(values)
        if step.experts:
            last = step.experts[len(self.blocks) - 1]
            f2 = [step.experts[j][base][flat] for j in range(len(self.blocks)) for base in F2_LAYER_FEATURES]
            names += [*self._f2_names, "e_jsd"]
            columns.append(np.column_stack([*f2, last["e_jsd"][flat]]).reshape(len(flat), len(f2) + 1))
        if step.hidden is not None:
            rows, vecs = step.hidden
            pos = np.searchsorted(rows, flat)
            if len(flat) and (pos.max() >= len(rows) or not np.array_equal(rows[pos], flat)):
                raise ValueError("la sonda H no capturó todas las posiciones puntuadas")
            names += [f"h_{i}" for i in range(vecs.shape[1])]
            columns.append(vecs[pos])
        # Una sola conversión a float de Python para todo el paso.
        matrix = np.concatenate(columns, axis=1) if columns else np.empty((len(flat), 0))
        out: list[list[TokenSignals]] = [[] for _ in range(bsz)]
        for i, (b, row) in enumerate(zip(bs.tolist(), matrix.tolist(), strict=True)):
            ids = None if expert_ids is None else expert_ids[i]
            out[b].append(TokenSignals(dict(zip(names, row, strict=True)), ids))
        return out

    # --- reducciones ------------------------------------------------------------------------

    @staticmethod
    def _output_features(rows: torch.Tensor, targets: np.ndarray) -> dict[str, np.ndarray]:
        """`features.output_features` en el dispositivo de los logits: copia 4 números por fila, no V.

        Copiar el vocabulario entero a la CPU en cada token era el mayor costo de latencia de F0
        (201.088 logits en gpt-oss). float64 en el dispositivo mantiene la paridad con la versión numpy.
        """
        z = rows.detach().to(torch.float64)
        tgt = torch.as_tensor(targets, dtype=torch.long, device=z.device).reshape(-1)
        top = torch.topk(z, min(fx.TOP_N_OUTPUT, z.shape[-1]), dim=-1).values
        shifted = z - top[:, :1]
        e = shifted.exp()
        se = e.sum(-1)
        log_se = se.log()
        top_shifted = top - top[:, :1]
        e_top = top_shifted.exp()
        se_top = e_top.sum(-1)
        margin = (e_top[:, 0] - e_top[:, 1]) / se if z.shape[-1] > 1 else e_top[:, 0] / se
        out = (
            torch.stack(
                [
                    log_se - (e * shifted).sum(-1) / se,
                    log_se - shifted.gather(1, tgt[:, None])[:, 0],
                    margin,
                    se_top.log() - (e_top * top_shifted).sum(-1) / se_top,
                ]
            )
            .cpu()
            .numpy()
        )
        return dict(zip(("out_entropy", "out_nll", "out_margin", "out_top20_entropy"), out, strict=True))

    def _f0(
        self, logits: torch.Tensor | None, targets: Any, bs: np.ndarray, ps: np.ndarray, seq: int
    ) -> dict[str, np.ndarray]:
        if logits is None or targets is None:
            raise ValueError("el nivel F0 necesita logits y targets en end_step")
        tgt = targets.detach().cpu().numpy() if isinstance(targets, torch.Tensor) else np.asarray(targets)
        kept = logits.shape[1]
        if tgt.shape != tuple(logits.shape[:2]):
            raise ValueError("targets debe tener la forma (B, T') de logits")
        offset = seq - kept
        if offset < 0:
            raise ValueError("logits cubre más posiciones que el paso")
        if len(ps) and ps.min() < offset:
            raise ValueError("hay posiciones puntuadas sin logits (logits_to_keep demasiado chico)")
        lg = logits.detach()
        if offset == 0 and len(bs) == lg.shape[0] * lg.shape[1]:  # todas puntuadas (decodificación)
            return self._output_features(lg.reshape(len(bs), -1), tgt.reshape(-1))
        rows = lg[torch.as_tensor(bs, device=lg.device), torch.as_tensor(ps - offset, device=lg.device)]
        return self._output_features(rows, tgt[bs, ps - offset])

    def _route(self, z: np.ndarray) -> dict[str, np.ndarray]:
        """Features F1, `idx` y `w` de logits `(L, m, N)`; una sola llamada si las capas comparten ruteo."""
        if self._shared_route:
            return fx.router_stats(z, self.blocks[0].spec.gating, self.top_k, **self._route_kw[0])
        per = [
            fx.router_stats(z[j], b.spec.gating, self.top_k, **self._route_kw[j])
            for j, b in enumerate(self.blocks)
        ]
        return {key: np.stack([s[key] for s in per]) for key in per[0]}

    def _router_plan(
        self, valid: np.ndarray, scored: np.ndarray, seq: int
    ) -> tuple[np.ndarray | None, list[tuple[Any, np.ndarray, int | None]]]:
        """Filas del paso que hay que reducir y, por fila del batch, dónde está cada cosa.

        Devuelve `rows` (índices planos `b * seq + p` en orden, o None si son todas) y por fila
        `(cols, prev_cols, last_col)`: columnas de las posiciones puntuadas, columna de la posición
        válida anterior a cada una (-1: usar la selección del paso previo) y columna de la última
        posición válida. Las columnas indexan la matriz reducida.
        """
        bsz = valid.shape[0]
        if valid.size and valid.all() and scored.all():  # decodificación o pasada sin máscaras
            plans = []
            for b in range(bsz):
                prev_cols = np.arange(b * seq - 1, (b + 1) * seq - 1)
                prev_cols[0] = -1
                plans.append((slice(b * seq, (b + 1) * seq), prev_cols, (b + 1) * seq - 1))
            return None, plans
        raw = []
        need: list[np.ndarray] = []
        for b in range(bsz):
            pv, psc = np.flatnonzero(valid[b]), np.flatnonzero(scored[b])
            prev = np.searchsorted(pv, psc) - 1  # índice en pv de la posición válida anterior
            prev_pos = np.where(prev >= 0, pv[np.maximum(prev, 0)], -1) if len(pv) else prev
            last = int(pv[-1]) if len(pv) else None
            raw.append((psc, prev_pos, last))
            need += [b * seq + psc, b * seq + prev_pos[prev_pos >= 0]]
            if last is not None:
                need.append(np.array([b * seq + last]))
        rows = np.unique(np.concatenate(need))
        plans = []
        for b, (psc, prev_pos, last) in enumerate(raw):
            prev_cols = np.where(prev_pos >= 0, np.searchsorted(rows, b * seq + prev_pos), -1)
            last_col = None if last is None else int(np.searchsorted(rows, b * seq + last))
            plans.append((np.searchsorted(rows, b * seq + psc), prev_cols, last_col))
        return rows, plans

    def _router_step(
        self, step: _Step, valid: np.ndarray, scored: np.ndarray, seq: int
    ) -> tuple[np.ndarray, list[list[list[int]]]]:
        """Features F1/U de todas las posiciones puntuadas del paso, vectorizadas entre capas.

        Devuelve la matriz `(m, columnas)` en el orden de `self._router_names` y los IDs de expertos
        `[token][capa][k]`. Actualiza el uso acumulado y la última selección de cada fila.
        """
        width = len(self._router_names)
        rows, plans = self._router_plan(valid, scored, seq)
        if rows is not None and rows.size == 0:
            return np.empty((0, width)), []
        stacked = torch.stack([step.router[j] for j in range(len(self.blocks))])  # (L, n, N)
        if rows is not None and rows.size < stacked.shape[1]:  # prefill: solo las filas necesarias
            stacked = stacked[:, torch.from_numpy(rows).to(stacked.device)]
        stats = self._route(stacked.cpu().numpy())  # única conversión a numpy del paso
        all_idx = stats["idx"]  # (L, filas, k)

        blocks: list[np.ndarray] = []
        expert_ids: list[list[list[int]]] = []
        for b, (cols, prev_cols, last_col) in enumerate(plans):
            idx = all_idx[:, cols]  # (L, m, k)
            m = idx.shape[1]
            if m:
                no_prev = prev_cols < 0
                prev_idx = all_idx[:, np.maximum(prev_cols, 0)]
                carried = self._prev_sel[b]
                if carried is not None:
                    prev_idx[:, no_prev] = carried[:, None, :]
                    no_prev[:] = False
                switch = fx.route_switch(prev_idx, idx)  # (L, m)
                switch[:, no_prev] = np.nan
                inc = fx.usage_counts(idx, self.n_experts, stats["w"][:, cols] if self._weighted else None)
                # cumsum desde el acumulado previo: mismo orden de sumas que paso a paso (causal).
                cum = np.concatenate([self._counts[b][:, None], inc], axis=1).cumsum(axis=1)[:, 1:]
                self._counts[b] = cum[:, -1]
                usage = fx.usage_stats(cum)
                per_layer = [stats[name][:, cols] for name in F1_LAYER_FEATURES] if self._weighted else []
                per_layer += [usage["u_gini"], usage["u_ihhi"], usage["u_ent"], switch]
                values = np.array(per_layer)  # (bases, L, m)
                aggs = fx.layer_aggregates(values, axis=1)  # cada una (bases, m)
                agg_values = np.array([aggs[a] for a in AGGREGATES])  # (aggs, bases, m)
                by_layer = values.transpose(2, 0, 1).reshape(m, -1)  # base mayor, capa menor
                by_agg = agg_values.transpose(2, 1, 0).reshape(m, -1)  # base mayor, agregado menor
                blocks.append(np.concatenate([by_layer, by_agg], axis=1))
                expert_ids += idx.transpose(1, 0, 2).tolist()
            if last_col is not None:
                self._prev_sel[b] = all_idx[:, last_col]
        return (np.concatenate(blocks, axis=0) if blocks else np.empty((0, width))), expert_ids
