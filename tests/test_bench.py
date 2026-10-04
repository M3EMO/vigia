"""Benchmark de overhead por nivel (P-09, AT-M10)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch
import transformers as tf
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

from vigia import charts
from vigia.backends.hf import HFBackend
from vigia.bench import BASELINE_LABEL, run_overhead_bench, tier_label
from vigia.contracts import Capabilities, ModelInfo, Tier, TokenEvent
from vigia.evaluate import evaluate_report, render_report_html

INFO = ModelInfo("fake/moe", "mixtral", 2, (0, 1), 4, 2, 64, "softmax_topk")
PROMPTS = [[{"role": "user", "content": "hola"}], [{"role": "user", "content": "chau"}]]


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class FakeStream:
    def __init__(self, clock: FakeClock, n: int, prefill_ms: float, token_ms: float) -> None:
        self.clock, self.n, self.prefill, self.token = clock, n, prefill_ms, token_ms
        self.i = 0
        self.completion_tokens = n
        self.closed = False

    def __iter__(self) -> FakeStream:
        return self

    def __next__(self) -> TokenEvent:
        if self.i >= self.n:
            raise StopIteration
        self.clock.now += (self.prefill if self.i == 0 else self.token) / 1e3
        self.i += 1
        return TokenEvent(self.i - 1, 1, "x", self.i - 1, self.i, {})

    def close(self) -> None:
        self.closed = True


class FakeBackend:
    """Backend determinista: cada configuración cuesta un factor fijo sobre la línea base."""

    FACTORS = {(): 1.0, (Tier.F0,): 1.02, (Tier.F0, Tier.F1): 1.10}

    def __init__(self, clock: FakeClock, n: int = 8) -> None:
        self.clock, self.n = clock, n
        self.calls: list[tuple[Tier, ...]] = []
        self.closed = False

    def model_info(self) -> ModelInfo:
        return INFO

    def capabilities(self) -> Capabilities:
        return Capabilities(tiers=(Tier.F0, Tier.F1, Tier.U), streaming=True)

    def stream(self, messages, *, max_tokens, temperature=0.0, top_p=1.0, seed=None, stop=None, tiers=()):
        assert stop is None and temperature == 0.0 and seed == 0
        tiers = tuple(tiers)
        self.calls.append(tiers)
        f = self.FACTORS[tiers]
        return FakeStream(self.clock, min(self.n, max_tokens), 20.0 * f, 10.0 * f)

    def close(self) -> None:
        self.closed = True


class FakeBackendIgnoreEos(FakeBackend):
    def __init__(self, clock: FakeClock) -> None:
        super().__init__(clock)
        self.ignore_eos_seen: list[bool] = []

    def stream(self, messages, *, ignore_eos=False, **kw):
        self.ignore_eos_seen.append(ignore_eos)
        return super().stream(messages, **kw)


def test_at_m10_fake_backend_exact_ratios_and_shape():
    """AT-M10: reporta p50/p95/p99 con y sin hooks y el ratio por nivel (oráculo exacto)."""
    clock = FakeClock()
    be = FakeBackend(clock)
    out = run_overhead_bench(be, PROMPTS, max_tokens=8, warmup=2, runs=3, clock=clock)
    assert set(out) >= {"config", "baseline", "tiers", "vigia", "target", "hardware"}
    for key in ("p50", "p95", "p99", "tpot_ms", "ttft_ms", "tok_s"):
        assert math.isfinite(out["baseline"][key])
    assert out["baseline"]["p50"] == pytest.approx(10.0)
    assert out["baseline"]["ttft_ms"] == pytest.approx(20.0)
    assert out["baseline"]["tok_s"] == pytest.approx(8 / 0.090)
    assert set(out["tiers"]) == {"F0", "F0+F1"}
    for label, factor in (("F0", 1.02), ("F0+F1", 1.10)):
        entry = out["tiers"][label]
        assert entry["overhead_ratio"] == pytest.approx(factor - 1)
        assert entry["ci"] == pytest.approx([factor - 1, factor - 1])
        assert entry["ttft_overhead_ratio"] == pytest.approx(factor - 1)
        assert entry["p99"] == pytest.approx(10.0 * factor)
    assert out["tiers"]["F0"]["meets_target"] and not out["tiers"]["F0+F1"]["meets_target"]
    assert out["vigia"]["label"] == "F0+F1" and out["target"] == 0.05
    cfg = out["config"]
    assert cfg["tiers_list"] == [BASELINE_LABEL, "F0", "F0+F1"]
    assert cfg["runs"] == 3 and cfg["warmup"] == 2 and cfg["n_units"] == 6
    assert cfg["ignore_eos"] == {
        "requested": True,
        "supported": False,
        "applied": False,
        "note": cfg["ignore_eos"]["note"],
    }
    assert {"cpu", "threads", "torch"} <= set(out["hardware"])
    json.dumps(out, allow_nan=False)
    assert not be.closed  # un backend ajeno no se cierra


def test_configs_are_interleaved_with_rotating_order():
    clock = FakeClock()
    be = FakeBackend(clock)
    run_overhead_bench(be, PROMPTS, max_tokens=4, warmup=1, runs=3, clock=clock)
    measured = be.calls[3:]  # 3 llamadas de warmup; no hay modelo para contar hooks
    assert len(measured) == 3 * 3 * 2
    blocks = [measured[i : i + 3] for i in range(0, len(measured), 3)]
    assert all(set(b) == {(), (Tier.F0,), (Tier.F0, Tier.F1)} for b in blocks)
    assert len({tuple(b) for b in blocks}) == 3  # el orden rota entre unidades


def test_ignore_eos_is_passed_when_backend_accepts_it_and_factory_is_closed():
    clock = FakeClock()
    created: list[FakeBackendIgnoreEos] = []

    def factory() -> FakeBackendIgnoreEos:
        created.append(FakeBackendIgnoreEos(clock))
        return created[0]

    out = run_overhead_bench(factory, PROMPTS, max_tokens=4, warmup=0, runs=1, clock=clock)
    assert out["config"]["ignore_eos"]["applied"] is True
    assert all(created[0].ignore_eos_seen) and created[0].closed


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"prompts": []}, "prompts"),
        ({"prompts": [[{"role": "user"}]]}, "mensaje"),
        ({"max_tokens": 1}, "max_tokens"),
        ({"runs": 0}, "runs"),
        ({"tiers_list": [(Tier.F2,)]}, "F2"),
        ({"tiers_list": [()]}, "al menos una"),
        ({"tiers_list": [("Z",)]}, "desconocido"),
        ({"prompts": PROMPTS[:1], "runs": 1}, "al menos 2"),
        ({"threads": 0}, "threads"),
        ({"level": 1.0}, "level"),
    ],
)
def test_invalid_arguments_fail_clearly(kwargs, match):
    clock = FakeClock()
    args = {"prompts": PROMPTS, "max_tokens": 4, "warmup": 0, "runs": 1, "clock": clock, **kwargs}
    prompts = args.pop("prompts")
    with pytest.raises(ValueError, match=match):
        run_overhead_bench(FakeBackend(clock), prompts, **args)


def test_backend_created_by_factory_is_closed_even_on_error():
    clock = FakeClock()
    created: list[FakeBackend] = []

    def factory() -> FakeBackend:
        created.append(FakeBackend(clock))
        return created[0]

    with pytest.raises(ValueError, match="F2"):
        run_overhead_bench(factory, PROMPTS, tiers_list=[(Tier.F2,)], max_tokens=4, clock=clock)
    assert created[0].closed


def test_single_token_generations_are_rejected():
    clock = FakeClock()
    with pytest.raises(ValueError, match="2 tokens"):
        run_overhead_bench(FakeBackend(clock, n=1), PROMPTS, max_tokens=4, warmup=0, runs=1, clock=clock)


def test_tier_label():
    assert tier_label(()) == BASELINE_LABEL
    assert tier_label((Tier.F0, "F1")) == "F0+F1"


def test_output_feeds_render_report_html():
    """El dict del benchmark sirve tal cual como report['latency'] del reporte HTML."""
    clock = FakeClock()
    bench = run_overhead_bench(FakeBackend(clock), PROMPTS, max_tokens=6, warmup=0, runs=2, clock=clock)
    assert "+10" in charts.latency_svg(bench["baseline"], bench["vigia"], target=bench["target"])
    rng = np.random.default_rng(0)
    n_resp, tok = 80, 6
    n = n_resp * tok
    groups = np.repeat([f"q{i}" for i in range(n_resp // 2)], 2 * tok)
    responses = np.repeat([f"r{i}" for i in range(n_resp)], tok)
    a = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(2.0 * a - 1.0)))).astype(int)
    feats = {"out_entropy": a, "out_nll": a + rng.normal(size=n), "r_ent_n@0": rng.normal(size=n)}
    report = evaluate_report(feats, y, groups, responses, INFO, n_boot=100, seed=0)
    plain = render_report_html(report)
    page = render_report_html(dict(report, latency=bench))
    assert page.count('role="img"') == plain.count('role="img"') + 1
    assert "Latencia por token" in page


# --- Backend HF real sobre un MoE diminuto -------------------------------------------------------


def word_tokenizer(vocab_size: int = 512) -> PreTrainedTokenizerFast:
    """Tokenizer por palabras sin entrenamiento: cada `w<i>` es un token y no hay EOS."""
    unk = "<unk>"
    vocab = {unk: 0, **{f"w{i}": i for i in range(1, vocab_size)}}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token=unk))
    tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return PreTrainedTokenizerFast(tokenizer_object=tk, unk_token=unk)


def heavy_tiny_moe() -> torch.nn.Module:
    """MoE diminuto "más pesado" del AT-M10: hidden 256, 4 capas, 8 expertos, top-2, vocab 512."""
    torch.manual_seed(0)
    cfg = tf.MixtralConfig(
        vocab_size=512,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_local_experts=8,
        num_experts_per_tok=2,
        max_position_embeddings=512,
        eos_token_id=None,
        bos_token_id=None,
        pad_token_id=None,
    )
    model = AutoModelForCausalLM.from_config(cfg).eval()
    model.generation_config.eos_token_id = None
    return model


@pytest.fixture(scope="module")
def heavy_backend() -> HFBackend:
    return HFBackend(heavy_tiny_moe(), word_tokenizer(), model_id="tiny/mixtral-heavy")


def _hooks(model: torch.nn.Module) -> int:
    return sum(len(m._forward_pre_hooks) + len(m._forward_hooks) for m in model.modules())


def test_hf_backend_without_tiers_installs_no_hooks(heavy_backend):
    """La línea base del A/B es el mismo bucle con tiers=() y sin ningún hook registrado."""
    before = _hooks(heavy_backend.model)
    for tiers, expect_more in (((), False), ((Tier.F0, Tier.F1), True)):
        stream = heavy_backend.stream(PROMPTS[0], max_tokens=3, tiers=tiers)
        try:
            event = next(stream)
            assert (_hooks(heavy_backend.model) > before) is expect_more
            assert (event.features == {}) is not expect_more
        finally:
            stream.close()
    assert _hooks(heavy_backend.model) == before


def test_at_m10_overhead_f0_f1_under_25_percent_on_tiny_moe(heavy_backend):
    """AT-M10: en el MoE diminuto pesado el overhead F0+F1 por token es < 25 %."""
    prompts = [
        [{"role": "user", "content": " ".join(f"w{(i * 37 + j) % 500 + 1}" for j in range(30))}]
        for i in range(3)
    ]
    threads_before = torch.get_num_threads()
    out = run_overhead_bench(
        heavy_backend, prompts, tiers_list=[(Tier.F0, Tier.F1)], max_tokens=16, warmup=2, runs=6, threads=2
    )
    assert torch.get_num_threads() == threads_before  # el benchmark restaura los hilos
    assert out["config"]["baseline_hooks"] == 0 and out["hardware"]["threads"] == 2
    assert out["config"]["ignore_eos"]["applied"] is True  # HFBackend acepta ignore_eos: longitud fija
    assert out["baseline"]["n_token_latencies"] == 18 * 15  # sin EOS: 16 tokens por request
    assert set(out["tiers"]) == {"F0+F1"}
    for entry in (out["baseline"], out["vigia"]):
        assert 0 < entry["p50"] <= entry["p95"] <= entry["p99"]
    f0f1 = out["tiers"]["F0+F1"]
    lo, hi = f0f1["ci"]
    assert lo <= f0f1["overhead_ratio"] <= hi
    assert f0f1["overhead_ratio"] < 0.25, json.dumps(out["tiers"], indent=1)
