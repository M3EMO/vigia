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
