"""Canales harmony de gpt-oss (AT-S10)."""

import pytest
import torch
import transformers as tf
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

from vigia.backends.hf import HFBackend
from vigia.harmony import CONTROL, HEADER, SPECIAL_TOKENS, HarmonyParser, is_scorable

TEMPLATE = (
    "{% for m in messages %}<|start|>{{ m.role }}<|message|>{{ m.content }}<|end|>{% endfor %}"
    "{% if add_generation_prompt %}<|start|>assistant{% endif %}"
)
RESPONSE = (
    "<|channel|>analysis<|message|>El usuario pregunta la capital.<|end|>"
    "<|start|>assistant<|channel|>final<|message|>La capital es Córdoba, sí 😀.<|return|>"
)
FINAL_TEXT = "La capital es Córdoba, sí 😀."
MESSAGES = [{"role": "user", "content": "¿Cuál es la capital?"}]


def harmony_tokenizer():
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=320,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=list(SPECIAL_TOKENS),
        show_progress=False,
    )
    tk.train_from_iterator(
        ["assistant analysis final commentary La capital es Córdoba, sí."] * 20, trainer=trainer
    )
    tok = PreTrainedTokenizerFast(tokenizer_object=tk)
    tok.add_special_tokens({"additional_special_tokens": list(SPECIAL_TOKENS)})
    tok.chat_template = TEMPLATE
    return tok


@pytest.fixture(scope="module")
def gpt_oss_backend():
    tok = harmony_tokenizer()
    torch.manual_seed(0)
    cfg = tf.GptOssConfig(
        vocab_size=len(tok),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_local_experts=4,
        num_experts_per_tok=2,
        eos_token_id=None,
        bos_token_id=None,
        pad_token_id=None,
    )
    model = AutoModelForCausalLM.from_config(cfg).eval()
    model.generation_config.eos_token_id = None
    return HFBackend(model, tok, model_id="tiny/gpt-oss")


def feed_all(parser, pieces):
    return [parser.feed(text, text in SPECIAL_TOKENS) for text in pieces]


def test_parser_assigns_channels():
    p = HarmonyParser()
    pieces = [
        "<|start|>",
        "assistant",
        "<|channel|>",
        "analysis",
        "<|message|>",
        "pienso",
        "<|end|>",
        "<|start|>",
        "assistant",
        "<|channel|>",
        "final",
        "<|message|>",
        "Hola",
        " mundo",
        "<|return|>",
    ]
    got = feed_all(p, pieces)
    assert got == [
        CONTROL,
        HEADER,
        CONTROL,
        HEADER,
        CONTROL,
        "analysis",
        CONTROL,
        CONTROL,
        HEADER,
        CONTROL,
        HEADER,
        CONTROL,
        "final",
        "final",
        CONTROL,
    ]
    assert p.ended_with == "<|return|>" and p.role == "assistant"


def test_parser_commentary_with_recipient_and_constrain():
    p = HarmonyParser()
    pieces = [
        "<|start|>",
        "assistant",
        "<|channel|>",
        "commentary to=functions.clima ",
        "<|constrain|>",
        "json",
        "<|message|>",
        '{"ciudad": "Rosario"}',
        "<|call|>",
        "texto suelto",
    ]
    got = feed_all(p, pieces)
    assert got[5] == HEADER and got[7] == "commentary" and got[-1] == "none"
    assert p.ended_with == "<|call|>"


def test_at_s10_teacher_force_only_final_is_scorable(gpt_oss_backend):
    """AT-S10: con canales analysis y final, solo los tokens de final se puntúan; especiales no."""
    events = gpt_oss_backend.teacher_force(MESSAGES, RESPONSE)
    specials = [e for e in events if e.is_special]
    assert [e.text for e in specials] == [
        "<|channel|>",
        "<|message|>",
        "<|end|>",
        "<|start|>",
        "<|channel|>",
        "<|message|>",
        "<|return|>",
    ]
    assert all(e.channel == CONTROL and not is_scorable(e) and e.start == e.end for e in specials)
    scorable = [e for e in events if is_scorable(e)]
    assert "".join(e.text for e in scorable) == FINAL_TEXT
    analysis = [e for e in events if e.channel == "analysis"]
    assert "".join(e.text for e in analysis) == "El usuario pregunta la capital."
    assert all(e.start == e.end for e in analysis)  # analysis nunca entra en el texto de la respuesta
    headers = "".join(e.text for e in events if e.channel == HEADER)
    assert headers == "analysisassistantfinal"
    pos = 0
    for e in scorable:
        assert e.start == pos
        pos = e.end
    assert pos == len(FINAL_TEXT)


def test_at_s10_stream_text_is_final_channel_and_return_stops(gpt_oss_backend, monkeypatch):
    """AT-S10: en streaming el texto de la respuesta es solo `final` y `<|return|>` termina."""
    tok = gpt_oss_backend.tokenizer
    ids = tok(RESPONSE, add_special_tokens=False)["input_ids"]
    forced = iter(ids)
    monkeypatch.setattr("vigia.backends.hf._sample", lambda *a: next(forced))
    stream = gpt_oss_backend.stream(MESSAGES, max_tokens=len(ids) + 5)
    events = list(stream)
    assert stream.text == FINAL_TEXT and stream.finish_reason == "stop"
    assert len(events) == len(ids) - 1  # <|return|> corta y no se emite
    assert "".join(e.text for e in events if is_scorable(e)) == FINAL_TEXT


# --- Marco de offsets de teacher_force con harmony ---------------------------------------------
# Regresión: "Con gpt-oss (harmony), teacher_force y los spans del dataset usan marcos de offsets
# distintos: el span marca el token equivocado o no queda ningún token puntuable".


def test_teacher_force_plain_response_is_final_channel_with_real_template(gpt_oss_backend):
    """Regresión (marcos de offsets con harmony): con la plantilla real de gpt-oss, que termina en
    `<|start|>assistant`, una respuesta en texto plano se puntúa como canal `final`.

    Antes todos sus tokens quedaban en `header` y no había ningún token puntuable.
    """
    from vigia.labels import IGNORE_LABEL, spans_to_token_labels

    tok = gpt_oss_backend.tokenizer
    events = gpt_oss_backend.teacher_force(MESSAGES, FINAL_TEXT)
    scorable = [e for e in events if is_scorable(e)]
    assert len(scorable) == len(tok(FINAL_TEXT, add_special_tokens=False)["input_ids"])
    assert "".join(e.text for e in scorable) == FINAL_TEXT
    header = [e for e in events if not is_scorable(e)]  # <|channel|>final<|message|>, ancho cero
    assert [e.channel for e in header][0] == CONTROL and all(e.start == e.end == 0 for e in header)
    assert all(FINAL_TEXT[e.start : e.end] == e.text for e in scorable)
    start = FINAL_TEXT.index("Córdoba")
    labels = spans_to_token_labels(events, [(start, start + len("Córdoba"))])
    flagged = "".join(e.text for e, lab in zip(events, labels, strict=True) if lab == 1)
    assert sum(lab != IGNORE_LABEL for lab in labels) == len(scorable) and flagged.strip() == "Córdoba"


def test_teacher_force_plain_response_matches_generation_of_final_channel(gpt_oss_backend, monkeypatch):
    """Regresión (marcos de offsets con harmony): las features de la respuesta plana forzada son las
    mismas que cuando el modelo genera `<|channel|>final<|message|>` y luego esa respuesta."""
    tok = gpt_oss_backend.tokenizer
    header = tok("<|channel|>final<|message|>", add_special_tokens=False)["input_ids"]
    body = tok(FINAL_TEXT, add_special_tokens=False)["input_ids"]
    forced = iter(header + body)
    monkeypatch.setattr("vigia.backends.hf._sample", lambda *a: next(forced))
    stream = gpt_oss_backend.stream(MESSAGES, max_tokens=len(header) + len(body))
    generated = list(stream)
    teacher = gpt_oss_backend.teacher_force(MESSAGES, FINAL_TEXT)
    assert stream.text == FINAL_TEXT and len(generated) == len(teacher) == len(header) + len(body)
    for a, b in zip(generated, teacher, strict=True):
        key_a = (a.token_id, a.start, a.end, a.text, a.channel, a.expert_ids)
        assert key_a == (b.token_id, b.start, b.end, b.text, b.channel, b.expert_ids)
        assert a.logprob == pytest.approx(b.logprob, abs=1e-5)
        for key, value in a.features.items():
            assert value == pytest.approx(b.features[key], abs=1e-4, nan_ok=True), key


def test_final_view_maps_marked_spans_to_token_offsets(gpt_oss_backend):
    """Regresión (marcos de offsets con harmony): un span sobre la respuesta con marcado se lleva al
    texto final y marca los tokens de 'Córdoba', no otro token."""
    from vigia.harmony import final_view
    from vigia.labels import spans_to_token_labels

    start = RESPONSE.index("Córdoba")
    in_analysis = RESPONSE.index("capital")
    text, spans = final_view(RESPONSE, [(start, start + len("Córdoba")), (in_analysis, in_analysis + 7)])
    assert text == FINAL_TEXT
    assert spans == [(FINAL_TEXT.index("Córdoba"), FINAL_TEXT.index("Córdoba") + len("Córdoba"))]
    events = gpt_oss_backend.teacher_force(MESSAGES, RESPONSE)
    assert "".join(e.text for e in events if is_scorable(e)) == text
    labels = spans_to_token_labels(events, spans)
    flagged = "".join(e.text for e, lab in zip(events, labels, strict=True) if lab == 1)
    assert flagged.strip() == "Córdoba"


def test_final_view_without_markup_and_partial_spans():
    from vigia.harmony import final_view

    assert final_view("Hola mundo", [(5, 10)]) == ("Hola mundo", [(5, 10)])
    marked = "<|channel|>final<|message|>Hola mundo<|return|>"
    # un span que arranca en el marcado se recorta al contenido de final
    assert final_view(marked, [(0, 31)]) == ("Hola mundo", [(0, 4)])
    assert final_view(marked, [(0, 5)]) == ("Hola mundo", [])


def test_final_header_from_each_parser_state():
    p = HarmonyParser()
    assert p.final_header() == "<|start|>assistant<|channel|>final<|message|>"
    feed_all(p, ["<|start|>", "assistant"])
    assert p.state == "role" and p.final_header() == "<|channel|>final<|message|>"
    feed_all(p, ["<|channel|>", "analysis", "<|message|>"])
    assert p.final_header() == "<|end|><|start|>assistant<|channel|>final<|message|>"
    feed_all(p, ["<|end|>", "<|start|>", "assistant", "<|channel|>", "final", "<|message|>"])
    assert p.final_header() == ""
