"""Tests de agregación subtoken -> palabra -> fragmento (P-10)."""

from __future__ import annotations

import pytest

from vigia.contracts import ScoredToken
from vigia.spans import build_spans, word_bounds, words


def scored(pieces: list[str], scores: list[float | None], tau: float = 0.5) -> tuple[str, list[ScoredToken]]:
    out: list[ScoredToken] = []
    pos = 0
    for i, (piece, p) in enumerate(zip(pieces, scores, strict=True)):
        flag = p is not None and p >= tau
        out.append(ScoredToken(index=i, text=piece, start=pos, end=pos + len(piece), p_error=p, flag=flag))
        pos += len(piece)
    return "".join(pieces), out


def test_word_bounds_unicode_and_inner_punctuation():
    text = "La niñez, año 3.14 e-mail 🙂 ¿acción?"
    got = [text[s:e] for s, e in word_bounds(text)]
    assert got == ["La", "niñez", "año", "3.14", "e-mail", "🙂", "acción"]


def test_AT_A05_subtokens_word_score_is_max_and_span_starts_at_letter():
    """AT-A05: el score de palabra es el max de sus subtokens y el fragmento no incluye el espacio."""
    text, toks = scored(
        ["La", " capital", " es", " Bue", "nos", " Aires", "."],
        [0.1, 0.1, 0.2, 0.9, 0.3, 0.2, 0.95],
    )
    ws = words(text, toks)
    buenos = next(w for w in ws if text[w.start : w.end] == "Buenos")
    assert buenos.score == pytest.approx(0.9)
    assert buenos.flag
    assert (buenos.token_start, buenos.token_end) == (3, 5)
    spans = build_spans(text, toks, gap_max=0, min_tokens=1)
    assert len(spans) == 1
    sp = spans[0]
    assert sp.text == "Buenos"
    assert text[sp.start] == "B"
    assert (sp.token_start, sp.token_end) == (3, 5)
    assert sp.score == pytest.approx(0.9)


def test_AT_A05_gap_max_merges_or_splits():
    """AT-A05: dos palabras marcadas separadas por una palabra limpia se unen solo si gap_max >= 1."""
    text, toks = scored(["Nació", " en", " 1820", " en", " Rosario"], [0.1, 0.1, 0.8, 0.1, 0.7])
    merged = build_spans(text, toks, gap_max=1)
    assert [s.text for s in merged] == ["1820 en Rosario"]
    assert merged[0].score == pytest.approx(0.8)
    assert (merged[0].token_start, merged[0].token_end) == (2, 5)
    split = build_spans(text, toks, gap_max=0)
    assert [s.text for s in split] == ["1820", "Rosario"]


def test_AT_A05_gap_counts_words_not_tokens():
    """AT-A05: el hueco se mide en palabras: una palabra limpia de dos subtokens es un hueco de 1."""
    text, toks = scored(["A", " gran", "dísimo", " B"], [0.9, 0.1, 0.1, 0.9])
    assert [s.text for s in build_spans(text, toks, gap_max=1)] == ["A grandísimo B"]
    assert len(build_spans(text, toks, gap_max=0)) == 2


def test_AT_A05_min_tokens_drops_short_fragments():
    """AT-A05: un fragmento con menos de min_tokens tokens se descarta."""
    text, toks = scored(["El", " año", " 18", "20", " y", " Juan"], [0.1, 0.1, 0.9, 0.6, 0.1, 0.9])
    spans = build_spans(text, toks, gap_max=0, min_tokens=2)
    assert [s.text for s in spans] == ["1820"]
    assert build_spans(text, toks, gap_max=0, min_tokens=3) == []


def test_unscored_tokens_never_flag():
    text, toks = scored(["Hola", " mundo"], [None, None])
    assert build_spans(text, toks) == []
    assert all(w.score is None for w in words(text, toks))


def test_punctuation_only_tokens_do_not_create_spans():
    text, toks = scored(["Hola", "!"], [0.1, 0.99])
    assert build_spans(text, toks) == []


def test_zero_width_token_is_ignored():
    toks = [
        ScoredToken(index=0, text="", start=0, end=0, p_error=0.99, flag=True),
        ScoredToken(index=1, text="ñu", start=0, end=2, p_error=0.2, flag=False),
    ]
    assert build_spans("ñu", toks) == []


def test_invalid_parameters():
    text, toks = scored(["a"], [0.9])
    with pytest.raises(ValueError, match="gap_max"):
        build_spans(text, toks, gap_max=-1)
    with pytest.raises(ValueError, match="min_tokens"):
        build_spans(text, toks, min_tokens=0)
