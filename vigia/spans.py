"""Agregación de tokens puntuados a palabras y de palabras marcadas a fragmentos (P-10).

Reglas:
- Una palabra es una corrida máxima de caracteres que no son espacio ni puntuación. La puntuación
  entre dos caracteres alfanuméricos queda dentro de la palabra ("3.14", "e-mail", "don't").
- Los offsets son code points sobre el texto de la respuesta, igual que `TokenEvent.start/end`.
- El score de una palabra es el máximo `p_error` de los tokens que la tocan. La palabra queda marcada
  si algún token que la toca está marcado (`flag`), es decir, si su máximo supera tau.
- Un fragmento une palabras marcadas separadas por hasta `gap_max` palabras no marcadas. Un fragmento
  con menos de `min_tokens` tokens se descarta. El score del fragmento es el máximo de sus tokens.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

from vigia.contracts import ScoredToken, Span


@dataclass(frozen=True)
class Word:
    start: int
    end: int
    token_start: int  # índice de TokenEvent; -1 si ningún token toca la palabra
    token_end: int
    score: float | None
    flag: bool


def _is_boundary(ch: str) -> bool:
    return ch.isspace() or unicodedata.category(ch).startswith("P")


def word_bounds(text: str) -> list[tuple[int, int]]:
    """Límites [start, end) de cada palabra del texto, en code points."""
    bounds: list[tuple[int, int]] = []
    start: int | None = None
    n = len(text)
    for i, ch in enumerate(text):
        inner_punct = (
            start is not None
            and not ch.isspace()
            and unicodedata.category(ch).startswith("P")
            and i + 1 < n
            and text[i - 1].isalnum()
            and text[i + 1].isalnum()
        )
        if _is_boundary(ch) and not inner_punct:
            if start is not None:
                bounds.append((start, i))
                start = None
        elif start is None:
            start = i
    if start is not None:
        bounds.append((start, n))
    return bounds


def words(text: str, tokens: Sequence[ScoredToken]) -> list[Word]:
    """Palabras del texto con el score máximo de los tokens que las tocan."""
    ordered = sorted((t for t in tokens if t.end > t.start), key=lambda t: t.start)
    out: list[Word] = []
    j = 0  # primer token que todavía puede tocar la palabra actual
    for ws, we in word_bounds(text):
        while j < len(ordered) and ordered[j].end <= ws:
            j += 1
        touching: list[ScoredToken] = []
        k = j
        while k < len(ordered) and ordered[k].start < we:
            if ordered[k].end > ws:
                touching.append(ordered[k])
            k += 1
        scores = [t.p_error for t in touching if t.p_error is not None]
        out.append(
            Word(
                start=ws,
                end=we,
                token_start=min((t.index for t in touching), default=-1),
                token_end=max((t.index for t in touching), default=-2) + 1,
                score=max(scores) if scores else None,
                flag=any(t.flag for t in touching),
            )
        )
    return out


def build_spans(
    text: str, tokens: Sequence[ScoredToken], *, gap_max: int = 1, min_tokens: int = 1
) -> list[Span]:
    """Fragmentos de palabras contiguas marcadas, con tolerancia `gap_max` y largo mínimo `min_tokens`."""
    if gap_max < 0:
        raise ValueError("gap_max debe ser >= 0")
    if min_tokens < 1:
        raise ValueError("min_tokens debe ser >= 1")
    ws = words(text, tokens)
    groups: list[list[Word]] = []
    current: list[Word] = []
    gap = 0
    for w in ws:
        if w.flag:
            if current and gap > gap_max:
                groups.append(current)
                current = []
            current.append(w)
            gap = 0
        elif current:
            gap += 1
    if current:
        groups.append(current)

    by_index = {t.index: t for t in tokens}
    spans: list[Span] = []
    for group in groups:
        t0 = min(w.token_start for w in group)
        t1 = max(w.token_end for w in group)
        if t1 - t0 < min_tokens:
            continue
        scores = [
            by_index[i].p_error for i in range(t0, t1) if i in by_index and by_index[i].p_error is not None
        ]
        start, end = group[0].start, group[-1].end
        spans.append(
            Span(
                start=start,
                end=end,
                token_start=t0,
                token_end=t1,
                score=float(max(scores)),
                text=text[start:end],
            )
        )
    return spans
