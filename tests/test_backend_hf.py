"""Backend HF: decodificación con cache KV, offsets UTF-8 y paradas (AT-S08, AT-S09)."""

import math

import pytest
import torch
import transformers as tf
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

from vigia.backends.hf import GenerationStream, HFBackend, IncrementalDecoder, _Emitter
from vigia.contracts import Backend, Tier

CORPUS = ["La niñez, el año y la acción. Buenos días señor, ¿cómo está? Canción de cuna en el jardín."] * 20
SPECIALS = ["<|eos|>", "<|user|>"]
MESSAGES = [{"role": "user", "content": "¿Qué día es hoy?"}]


def byte_tokenizer(specials=SPECIALS, vocab_size=300):
    """BPE a nivel byte entrenado en memoria: parte tildes y emojis entre tokens."""
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=list(specials),
        show_progress=False,
    )
    tk.train_from_iterator(CORPUS, trainer=trainer)
    tok = PreTrainedTokenizerFast(tokenizer_object=tk)
    tok.add_special_tokens({"additional_special_tokens": list(specials)})
    return tok


@pytest.fixture(scope="module")
def tok():
    return byte_tokenizer()


def tiny_mixtral(vocab: int, eos: int | None = None):
    torch.manual_seed(0)
    cfg = tf.MixtralConfig(
        vocab_size=vocab,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_local_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=512,
        eos_token_id=eos,
        bos_token_id=None,
        pad_token_id=None,
    )
    model = AutoModelForCausalLM.from_config(cfg).eval()
    model.generation_config.eos_token_id = eos
    return model


@pytest.fixture(scope="module")
def backend(tok):
    return HFBackend(tiny_mixtral(len(tok)), tok, model_id="tiny/mixtral")


def assert_offsets_cover(events, text):
    """Offsets contiguos sin huecos ni solapes que reconstruyen el texto."""
    pos = 0
    for ev in events:
        if ev.channel != "final" or ev.is_special:
            assert ev.start == ev.end == pos
            continue
        assert ev.start == pos and ev.end == pos + len(ev.text)
        assert text[ev.start : ev.end] == ev.text
        pos = ev.end
    assert pos == len(text)
    assert "".join(e.text for e in events if e.channel == "final" and not e.is_special) == text


def test_backend_satisfies_contract(backend):
    for name in Backend.__dict__.keys() & {"model_info", "capabilities", "stream"}:
        assert callable(getattr(backend, name))
    info = backend.model_info()
    assert info.model_type == "mixtral" and info.moe_layers == (0, 1) and info.model_id == "tiny/mixtral"
    caps = backend.capabilities()
    assert caps.streaming and Tier.F2 in caps.tiers and caps.disabled == ()


def test_greedy_loop_matches_generate(backend, tok):
    """El bucle propio con cache KV produce los mismos tokens que `model.generate` greedy."""
    events = list(backend.stream(MESSAGES, max_tokens=12))
    prompt = backend._prompt_ids(MESSAGES)
    with torch.inference_mode():
        ref = backend.model.generate(
            torch.tensor([prompt]), max_new_tokens=12, do_sample=False, pad_token_id=0
        )
    assert [e.token_id for e in events] == ref[0, len(prompt) :].tolist()


def test_at_s08_stream_emits_n_events_with_offsets(backend):
    """AT-S08: N tokens en streaming → exactamente N eventos cuyos offsets reconstruyen el texto."""
    stream = backend.stream(MESSAGES, max_tokens=17)
    assert isinstance(stream, GenerationStream)
    events = list(stream)
    assert len(events) == 17 == stream.completion_tokens
    assert [e.index for e in events] == list(range(17))
    assert stream.finish_reason == "length"
    assert_offsets_cover(events, stream.text)
    assert set(stream.timings) >= {"prefill_s", "decode_s"}
    ev = events[3]
    assert {"out_entropy", "r_ent_n@0", "agg_mean:u_gini"} <= set(ev.features)
    assert ev.logprob == pytest.approx(-ev.features["out_nll"], abs=1e-5)
    assert len(ev.expert_ids) == 2 and len(ev.expert_ids[0]) == 2


def test_at_s09_teacher_force_utf8_offsets(backend):
    """AT-S09: tildes, ñ, emojis y bytes partidos entre tokens: offsets en code points sin huecos."""
    text = "La niñez 😀 tuvo acción, ¿sí? Ñandú 🇦🇷 güero"
    events = backend.teacher_force(MESSAGES, text)
    assert len(events) == len(backend.tokenizer(text, add_special_tokens=False)["input_ids"])
    assert_offsets_cover(events, text)
    assert any(e.text == "" for e in events)  # al menos un token aporta solo bytes parciales
    assert all("�" not in e.text for e in events)
    assert all("out_nll" in e.features and e.logprob is not None for e in events)


def test_at_s09_generated_bytes_are_never_split(backend):
    """AT-S09: en generación libre a nivel byte, ningún evento emite un carácter a medias."""
    stream = backend.stream(MESSAGES, max_tokens=40, temperature=1.0, seed=3)
    events = list(stream)
    assert_offsets_cover(events, stream.text)
    assert len(events) == 40


def test_teacher_force_matches_stream_features(backend, monkeypatch):
    """Teacher forcing de una respuesta da las mismas features que generarla token a token."""
    text = "Buenos días, señor 😀"
    resp = backend.tokenizer(text, add_special_tokens=False)["input_ids"]
    forced = iter(resp)
    monkeypatch.setattr("vigia.backends.hf._sample", lambda *a: next(forced))
    tiers = (Tier.F0, Tier.F1, Tier.F2, Tier.H)
    stream = backend.stream(MESSAGES, max_tokens=len(resp), tiers=tiers)
    generated = list(stream)
    teacher = backend.teacher_force(MESSAGES, text, tiers=tiers)
    assert stream.text == text and [e.token_id for e in generated] == resp
    for a, b in zip(generated, teacher, strict=True):
        assert (a.start, a.end, a.text, a.expert_ids) == (b.start, b.end, b.text, b.expert_ids)
        assert a.features.keys() == b.features.keys()
        for k in a.features:
            assert a.features[k] == pytest.approx(b.features[k], abs=1e-4, nan_ok=True)


def test_seed_makes_sampling_deterministic(backend):
    a = [e.token_id for e in backend.stream(MESSAGES, max_tokens=10, temperature=0.9, top_p=0.9, seed=7)]
    b = [e.token_id for e in backend.stream(MESSAGES, max_tokens=10, temperature=0.9, top_p=0.9, seed=7)]
    c = [e.token_id for e in backend.stream(MESSAGES, max_tokens=10, temperature=0.9, top_p=0.9, seed=8)]
    assert a == b and a != c


def test_stop_string_truncates_text(backend):
    ref = backend.generate(MESSAGES, max_tokens=20)
    text = ref.text
    stop = text[6:9]
    cut = text.find(stop)
    gen = backend.generate(MESSAGES, max_tokens=20, stop=[stop])
    assert gen.finish_reason == "stop"
    assert gen.text == text[:cut]
    assert_offsets_cover(gen.tokens, gen.text)


def test_eos_stops_without_emitting(tok):
    base = HFBackend(tiny_mixtral(len(tok)), tok)
    ref = [e.token_id for e in base.stream(MESSAGES, max_tokens=6)]
    eos = ref[3]
    if eos in ref[:3]:
        pytest.skip("el token elegido como EOS aparece antes")
    with_eos = HFBackend(tiny_mixtral(len(tok), eos=eos), tok)
    stream = with_eos.stream(MESSAGES, max_tokens=6)
    events = list(stream)
    assert [e.token_id for e in events] == ref[:3]
    assert stream.finish_reason == "stop"


def test_emitter_holds_back_partial_stop_and_bytes(tok):
    em = _Emitter(tok, set(), harmony=False, stop=["FIN"])
    ids = tok("hola F", add_special_tokens=False)["input_ids"]
    released = []
    for t in ids:
        released += em.push(t, None, None)
    assert "".join(e.text for e in released) == "hola "  # la "F" queda retenida
    for t in tok("IN chau", add_special_tokens=False)["input_ids"]:
        released += em.push(t, None, None)
        if em.stopped:
            break
    assert em.stopped and em.text == "hola "
    assert "".join(e.text for e in released) == "hola "


def test_incremental_decoder_never_emits_partial_char(tok):
    """El emoji no está en el corpus: queda partido en 4 tokens de un byte."""
    dec = IncrementalDecoder(tok)
    ids = tok("ñ😀", add_special_tokens=False)["input_ids"]
    deltas = [dec.push(t) for t in ids]
    assert "".join(deltas) == "ñ😀"
    assert all("�" not in d for d in deltas)
    assert deltas.count("") >= 3 and not dec.pending
    assert dec.push(ids[-1]) == ""  # un byte de continuación suelto queda pendiente
    assert dec.pending and dec.flush() == "�" and not dec.pending


def test_chat_template_is_used_when_present(tok):
    templ = byte_tokenizer()
    templ.chat_template = "{% for m in messages %}<|user|>{{ m.content }}{% endfor %}<|eos|>"
    be = HFBackend(tiny_mixtral(len(templ)), templ)
    ids = be._prompt_ids(MESSAGES)
    assert ids[0] == templ.convert_tokens_to_ids("<|user|>") and ids[-1] == templ.convert_tokens_to_ids(
        "<|eos|>"
    )
    plain = HFBackend(tiny_mixtral(len(tok)), tok)
    assert tok.decode(plain._prompt_ids(MESSAGES)).startswith("user: ")


def test_validation_errors(backend):
    with pytest.raises(ValueError):
        backend.stream(MESSAGES, max_tokens=0)
    with pytest.raises(ValueError):
        backend.stream([{"role": "user"}], max_tokens=3)
    with pytest.raises(ValueError):
        backend.stream(MESSAGES, max_tokens=3, top_p=0.0)
    with pytest.raises(ValueError):
        backend.stream(MESSAGES, max_tokens=3, stop=[""])
    with pytest.raises(ValueError):
        HFBackend(tiny_mixtral(10))


def test_f2_and_h_tiers_in_stream(backend):
    events = list(backend.stream(MESSAGES, max_tokens=3, tiers=(Tier.F0, Tier.F1, Tier.F2, Tier.H)))
    feats = events[-1].features
    assert 0.0 <= feats["e_jsd"] <= 1.0 and not math.isnan(feats["e_cos@0"])
    assert "h_31" in feats


def test_disabled_tier_is_rejected(tok):
    be = HFBackend(tiny_mixtral(len(tok)), tok)
    be._f2_reason = "F2: expertos MXFP4"
    assert Tier.F2 not in be.capabilities().tiers and be.capabilities().disabled == ("F2: expertos MXFP4",)
    with pytest.raises(ValueError, match="F2"):
        be.stream(MESSAGES, max_tokens=2, tiers=(Tier.F2,))


def test_closing_stream_releases_hooks_and_lock(backend):
    stream = backend.stream(MESSAGES, max_tokens=50)
    next(stream)
    stream.close()
    assert sum(len(m._forward_pre_hooks) for m in backend.model.modules()) == 0
    assert not backend._lock.locked()


def test_stream_steps_run_without_autograd_from_any_thread(backend):
    """En streaming SSE, el modelo genera con autograd prendido porque torch.inference_mode no
    acompaña al generador entre hilos.

    StreamingResponse avanza el iterador desde el threadpool: cada `next()` puede caer en otro hilo.
    Todo forward de generación tiene que correr sin grafo de autograd, sea cual sea el hilo.
    """
    import threading

    seen = []

    def spy(_module, _args, output):
        seen.append((threading.current_thread().name, torch.is_grad_enabled(), output.logits.requires_grad))

    handle = backend.model.register_forward_hook(spy)
    try:
        stream = backend.stream(MESSAGES, max_tokens=6, ignore_eos=True)
        next(stream)
        for i in range(3):
            worker = threading.Thread(target=lambda: next(stream, None), name=f"worker{i}")
            worker.start()
            worker.join()
        stream.close()
    finally:
        handle.remove()
    assert any(name.startswith("worker") for name, _, _ in seen)
    assert [(name, grad, req) for name, grad, req in seen if grad or req] == []
    assert not backend._lock.locked()
