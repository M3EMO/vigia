"""Benchmark de overhead de Vigía por nivel de señales (P-09, AT-M10, AT-M12).

Procedimiento (nota `vigia-serving-integracion`, sección "Medición de overhead"):

- A/B en el mismo proceso y con el mismo modelo. La línea base es el mismo bucle de generación del
  backend con `tiers=()`: con niveles vacíos el extractor no registra ningún hook. El benchmark lo
  verifica contando los hooks del modelo durante una generación de la línea base.
- Las configuraciones se intercalan dentro de cada corrida: para cada prompt se miden todas, con
  un orden que rota entre prompts y corridas para cancelar deriva térmica y efectos de caché.
- Warmup explícito, semilla fija, decodificación greedy, `stop=None` y `max_tokens` fijo.
- `ignore_eos`: el contrato `Backend.stream` no lo define. Si el backend acepta el parámetro
  `ignore_eos`, se pasa; si no (caso de `HFBackend`), la longitud puede variar y por eso toda
  latencia se mide por token generado. El reporte declara cuál de los dos casos midió.
- `torch.set_num_threads` se fija si se pide y se restaura al terminar. El GC queda apagado durante
  las mediciones.

Métricas por request, con reloj de pared:

- TTFT (ms): desde la llamada a `stream` hasta el primer evento. Incluye el prefill y su reducción.
- Latencia por token (ms): el intervalo entre eventos consecutivos. Cada intervalo incluye un forward
  de decodificación con sus hooks, `end_step` y el armado del evento.
- TPOT (ms): `(fin − primer evento) / (tokens − 1)`, la media por request.
- tok/s: tokens generados / tiempo total del request.

`p50`, `p95` y `p99` son percentiles de la latencia por token sobre todos los tokens de una
configuración. El overhead de un nivel se estima pareado: en cada unidad `(corrida, prompt)`, donde
todas las configuraciones se midieron intercaladas, se divide la mediana de la latencia por token del
nivel por la de la base; el overhead es la mediana de esos cocientes − 1. Las medianas resisten los
picos de una CPU compartida. El IC sale de un bootstrap sobre las unidades. El overhead del TTFT usa
el mismo esquema con el TTFT de cada request.
"""

from __future__ import annotations

import gc
import inspect
import math
import platform
import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch

from vigia.contracts import Backend, Tier

DEFAULT_TIERS_LIST: tuple[tuple[Tier, ...], ...] = ((), (Tier.F0,), (Tier.F0, Tier.F1))
TARGET = 0.05  # meta del plan: < 5 % de latencia agregada
BENCH_SCHEMA_VERSION = "1"
BASELINE_LABEL = "sin_hooks"
_PERCENTILES = (50, 95, 99)


def tier_label(tiers: Sequence[Tier | str]) -> str:
    """Nombre de una configuración: `sin_hooks` para `()`; si no, los niveles unidos con `+`."""
    return "+".join(Tier(t).value for t in tiers) if tiers else BASELINE_LABEL


def _hook_count(model: Any) -> int | None:
    modules = getattr(model, "modules", None)
    if not callable(modules):
        return None
    return sum(len(m._forward_pre_hooks) + len(m._forward_hooks) for m in modules())


def _is_message(m: Any) -> bool:
    return isinstance(m, dict) and isinstance(m.get("role"), str) and isinstance(m.get("content"), str)


def _validate_prompts(prompts: Any) -> list[list[dict[str, str]]]:
    if not isinstance(prompts, Sequence) or isinstance(prompts, str) or not prompts:
        raise ValueError("prompts debe ser una lista no vacía de listas de mensajes")
    for messages in prompts:
        if not isinstance(messages, list) or not messages:
            raise ValueError("cada prompt debe ser una lista no vacía de mensajes")
        if not all(_is_message(m) for m in messages):
            raise ValueError("cada mensaje debe ser un dict con 'role' y 'content' de tipo str")
    return list(prompts)


def _configs(tiers_list: Any, available: Sequence[Tier]) -> list[tuple[Tier, ...]]:
    """Configuraciones sin duplicados, con la línea base `()` primero."""
    if not isinstance(tiers_list, Sequence) or isinstance(tiers_list, str):
        raise ValueError("tiers_list debe ser una secuencia de tuplas de niveles")
    configs: list[tuple[Tier, ...]] = [()]
    for tiers in tiers_list:
        if isinstance(tiers, str) or not isinstance(tiers, Sequence):
            raise ValueError(f"configuración de niveles inválida: {tiers!r}")
        try:
            tup = tuple(dict.fromkeys(Tier(t) for t in tiers))
        except ValueError as exc:
            raise ValueError(f"nivel desconocido en {tiers!r}") from exc
        missing = [t.value for t in tup if t not in available]
        if missing:
            raise ValueError(f"el backend no ofrece los niveles {missing}")
        if tup not in configs:
            configs.append(tup)
    if len(configs) < 2:
        raise ValueError("tiers_list necesita al menos una configuración con niveles además de la base")
    return configs


def _check_int(name: str, value: Any, minimum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} debe ser un entero >= {minimum}")
    return value


def _ci(ratios: np.ndarray, level: float) -> list[float]:
    tail = (1.0 - level) / 2.0
    lo, hi = np.quantile(ratios, [tail, 1.0 - tail])
    return [float(lo), float(hi)]


def _unit_ratio(
    base: list[np.ndarray], other: list[np.ndarray], rng: np.random.Generator, n_boot: int, level: float
) -> tuple[float, list[float]]:
    """Overhead pareado por unidad y su IC.

    Por unidad: `p50(latencia por token del nivel) / p50(latencia por token base)`. El overhead es la
    mediana de esos cocientes − 1. Un pico de carga que cae sobre un solo request afecta una unidad
    y la mediana lo descarta; el bootstrap remuestrea unidades.
    """
    b = np.array([np.median(g) for g in base])
    o = np.array([np.median(g) for g in other])
    ratios = np.divide(o, b, out=np.full_like(o, np.nan), where=b > 0)
    ratios = ratios[np.isfinite(ratios)]
    if ratios.size == 0:
        raise ValueError("la latencia por token de la línea base es cero: no se puede medir el overhead")
    picks = rng.integers(0, ratios.size, size=(n_boot, ratios.size))
    return float(np.median(ratios) - 1.0), _ci(np.median(ratios[picks], axis=1) - 1.0, level)


def _paired_ratio(
    base: np.ndarray, other: np.ndarray, rng: np.random.Generator, n_boot: int, level: float
) -> tuple[float, list[float]]:
    """Mediana de los cocientes por unidad de un valor por request (TTFT) − 1, con IC bootstrap."""
    return _unit_ratio([np.array([v]) for v in base], [np.array([v]) for v in other], rng, n_boot, level)


def _summary(gaps: list[np.ndarray], ttft: np.ndarray, tpot: np.ndarray, tok_s: np.ndarray) -> dict[str, Any]:
    pooled = np.concatenate(gaps)
    out: dict[str, Any] = {f"p{q}": float(np.percentile(pooled, q)) for q in _PERCENTILES}
    out.update(
        tpot_ms=float(np.median(tpot)),
        ttft_ms=float(np.median(ttft)),
        tok_s=float(np.median(tok_s)),
        n_requests=len(gaps),
        n_token_latencies=int(pooled.size),
    )
    return out


def _hardware(threads: int) -> dict[str, Any]:
    return {
        "cpu": platform.processor() or platform.machine() or "desconocida",
        "machine": platform.machine(),
        "threads": threads,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def run_overhead_bench(
    backend: Backend | Callable[[], Backend],
    prompts: Sequence[list[dict[str, str]]],
    *,
    tiers_list: Sequence[Sequence[Tier | str]] = DEFAULT_TIERS_LIST,
    max_tokens: int = 32,
    warmup: int = 3,
    runs: int = 5,
    seed: int = 0,
    threads: int | None = None,
    ignore_eos: bool = True,
    n_boot: int = 1000,
    level: float = 0.95,
    target: float = TARGET,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Mide TTFT, latencia por token, TPOT y tok/s por nivel de señales contra la generación sin hooks.

    `backend` es un backend (`contracts.Backend`) o una fábrica sin argumentos que lo crea; si la
    fábrica lo crea, el benchmark lo cierra al final. Devuelve un dict con `config`, `baseline`,
    `tiers` (por etiqueta, con `overhead_ratio` y `ci`), `vigia` (la última configuración de
    `tiers_list`), `target` y `hardware`. El dict sirve tal cual como `report['latency']` de
    `vigia.evaluate.render_report_html` (usa `baseline`, `vigia` y `target`).
    """
    prompts = _validate_prompts(prompts)
    _check_int("max_tokens", max_tokens, 2)
    _check_int("warmup", warmup, 0)
    _check_int("runs", runs, 1)
    _check_int("seed", seed, 0)
    _check_int("n_boot", n_boot, 100)
    if threads is not None:
        _check_int("threads", threads, 1)
    if not 0.0 < float(level) < 1.0:
        raise ValueError("level debe estar en (0, 1)")
    if not (math.isfinite(float(target)) and float(target) > 0):
        raise ValueError("target debe ser un número positivo")
    if runs * len(prompts) < 2:
        raise ValueError("hacen falta al menos 2 mediciones por configuración (runs × prompts)")

    owned = not hasattr(backend, "stream")
    if owned and not callable(backend):
        raise ValueError("backend debe ser un Backend o una fábrica sin argumentos")
    be: Backend = backend() if owned else backend  # type: ignore[operator, assignment]
    try:
        return _run(
            be,
            prompts,
            tiers_list,
            max_tokens,
            warmup,
            runs,
            seed,
            threads,
            ignore_eos,
            n_boot,
            level,
            float(target),
            clock,
        )
    finally:
        close = getattr(be, "close", None)
        if owned and callable(close):
            close()


def _run(
    be: Backend,
    prompts: list[list[dict[str, str]]],
    tiers_list: Sequence[Sequence[Tier | str]],
    max_tokens: int,
    warmup: int,
    runs: int,
    seed: int,
    threads: int | None,
    ignore_eos: bool,
    n_boot: int,
    level: float,
    target: float,
    clock: Callable[[], float],
) -> dict[str, Any]:
    configs = _configs(tiers_list, be.capabilities().tiers)
    accepts_ignore_eos = "ignore_eos" in inspect.signature(be.stream).parameters
    stream_kw: dict[str, Any] = {"max_tokens": max_tokens, "temperature": 0.0, "seed": seed, "stop": None}
    if ignore_eos and accepts_ignore_eos:
        stream_kw["ignore_eos"] = True

    Measurement = tuple[float, np.ndarray, float, int]

    def measure(messages: list[dict[str, str]], tiers: tuple[Tier, ...]) -> Measurement:
        """(TTFT s, intervalos entre eventos s, tiempo total s, tokens generados)."""
        stamps: list[float] = []
        t0 = clock()
        stream = be.stream(messages, tiers=tiers, **stream_kw)
        try:
            for _ in stream:
                stamps.append(clock())
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()
        end = clock()
        n = int(getattr(stream, "completion_tokens", 0) or len(stamps))
        ttft = (stamps[0] if stamps else end) - t0
        return ttft, np.diff(np.asarray(stamps)), end - t0, n

    model = getattr(be, "model", None)
    prev_threads = torch.get_num_threads()
    gc_was_enabled = gc.isenabled()
    n_cfg, n_units = len(configs), runs * len(prompts)
    results: list[list[Measurement | None]] = [[None] * n_units for _ in configs]
    baseline_hooks: int | None = None
    try:
        if threads is not None:
            torch.set_num_threads(threads)
        torch.manual_seed(seed)
        for w in range(warmup):
            for tiers in configs:
                measure(prompts[w % len(prompts)], tiers)
        before = _hook_count(model)
        if before is not None:
            stream = be.stream(prompts[0], tiers=(), **stream_kw)
            try:
                next(iter(stream), None)  # el contexto de hooks del backend está abierto acá
                baseline_hooks = (_hook_count(model) or 0) - before
            finally:
                close = getattr(stream, "close", None)
                if callable(close):
                    close()
            if baseline_hooks:
                raise RuntimeError(f"la línea base instaló {baseline_hooks} hooks: el A/B no es válido")
        gc.collect()
        gc.disable()
        for r in range(runs):
            for i, messages in enumerate(prompts):
                u = r * len(prompts) + i
                shift = u % n_cfg
                for c in [*range(shift, n_cfg), *range(shift)]:
                    results[c][u] = measure(messages, configs[c])
    finally:
        if gc_was_enabled:
            gc.enable()
        if threads is not None:
            torch.set_num_threads(prev_threads)

    # Unidades válidas: (corrida, prompt) con al menos 2 eventos en todas las configuraciones.
    valid = [u for u in range(n_units) if all(len(results[c][u][1]) >= 1 for c in range(n_cfg))]  # type: ignore[index]
    if len(valid) < 2:
        raise ValueError(
            "menos de 2 mediciones con 2 tokens o más: subí max_tokens o revisá el EOS del modelo"
        )

    def series(c: int) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        rows = [results[c][u] for u in valid]
        gaps = [row[1] * 1e3 for row in rows]  # type: ignore[index]
        ttft = np.array([row[0] * 1e3 for row in rows])  # type: ignore[index]
        tpot = np.array([g.sum() / len(g) for g in gaps])
        tok_s = np.array([row[3] / row[2] if row[2] > 0 else np.nan for row in rows])  # type: ignore[index]
        return gaps, ttft, tpot, tok_s

    base = series(0)
    if np.median(np.concatenate(base[0])) <= 0:
        raise ValueError("el backend no emite los tokens de a uno: la latencia por token no es medible")
    rng = np.random.default_rng(seed)
    tiers_out: dict[str, dict[str, Any]] = {}
    for c in range(1, n_cfg):
        cur = series(c)
        entry = _summary(*cur)
        entry["overhead_ratio"], entry["ci"] = _unit_ratio(base[0], cur[0], rng, n_boot, level)
        entry["ttft_overhead_ratio"], entry["ttft_ci"] = _paired_ratio(base[1], cur[1], rng, n_boot, level)
        entry["meets_target"] = bool(entry["overhead_ratio"] <= target)
        tiers_out[tier_label(configs[c])] = entry
    used_threads = threads if threads is not None else prev_threads
    vigia_label = tier_label(configs[-1])
    applied = "ignore_eos" in stream_kw
    info = be.model_info()
    return {
        "schema_version": BENCH_SCHEMA_VERSION,
        "config": {
            "model_id": info.model_id,
            "model_type": info.model_type,
            "tiers_list": [tier_label(t) for t in configs],
            "n_prompts": len(prompts),
            "max_tokens": max_tokens,
            "warmup": warmup,
            "runs": runs,
            "seed": seed,
            "threads": used_threads,
            "temperature": 0.0,
            "ignore_eos": {
                "requested": bool(ignore_eos),
                "supported": accepts_ignore_eos,
                "applied": applied,
                "note": "longitud fija por ignore_eos"
                if applied
                else "el backend no acepta ignore_eos: la longitud puede variar; se mide por token generado",
            },
            "order": "intercalado; el orden de las configuraciones rota por prompt y corrida",
            "metric": "ms por token: intervalo entre eventos, reloj de pared, con hooks y end_step",
            "overhead": "mediana por unidad de p50(ms por token del nivel) / p50(ms por token sin hooks) - 1",
            "bootstrap": {"n_boot": n_boot, "level": level, "seed": seed, "unit": "(corrida, prompt)"},
            "n_units": len(valid),
            "n_units_dropped": n_units - len(valid),
            "baseline_hooks": baseline_hooks,
        },
        "baseline": _summary(*base),
        "tiers": tiers_out,
        "vigia": {**tiers_out[vigia_label], "label": vigia_label},
        "target": target,
        "hardware": _hardware(used_threads),
    }
