"""Canales del formato harmony de gpt-oss, token a token.

Una máquina de estados sobre el texto de los tokens especiales (no usa regex sobre el texto
generado). Un mensaje tiene la forma:

    <|start|>{rol}<|channel|>{canal}[ to={destino}][<|constrain|>{tipo}]<|message|>{contenido}<|end|>

y termina en `<|end|>`, `<|return|>` (fin de la respuesta) o `<|call|>` (llamada a herramienta).
Cada token recibe un canal: `final`, `analysis` o `commentary` para el contenido; `header` para el
rol, el nombre del canal y el destino; `control` para los tokens especiales; `none` fuera de un
mensaje. Solo el contenido de `final` se muestra y se puntúa (P-03, AT-S10).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Protocol

START, END, MESSAGE, CHANNEL, CONSTRAIN, RETURN, CALL = (
    "<|start|>",
    "<|end|>",
    "<|message|>",
    "<|channel|>",
    "<|constrain|>",
    "<|return|>",
    "<|call|>",
)
SPECIAL_TOKENS = (START, END, MESSAGE, CHANNEL, CONSTRAIN, RETURN, CALL)
STOP_TOKENS = (RETURN, CALL)
SCORED_CHANNEL = "final"
HEADER = "header"
CONTROL = "control"
NO_CHANNEL = "none"


class _Event(Protocol):
    channel: str
    is_special: bool


def is_scorable(event: _Event) -> bool:
    """Regla única: solo se puntúa y se muestra el contenido no especial del canal `final`."""
    return event.channel == SCORED_CHANNEL and not event.is_special


class HarmonyParser:
    """Asigna canal a cada token. `feed` procesa un token y devuelve su canal."""

    def __init__(self) -> None:
        self._state = "idle"  # idle | role | channel | other_header | content
        self._channel_text = ""
        self.channel: str = NO_CHANNEL  # canal del mensaje abierto
        self.role = ""
        self.ended_with: str | None = None  # último terminador visto (end, return o call)

    @property
    def state(self) -> str:
        """Estado actual: idle, role, channel, other_header o content."""
        return self._state

    def final_header(self) -> str:
        """Marcado que lleva al parser al contenido del canal `final` desde el estado actual.

        Vacío si ya está en `final`. Sirve para puntuar una respuesta en texto plano después de un
        prompt que termina en `<|start|>assistant`, como la plantilla real de gpt-oss.
        """
        if self._state == "content" and self.channel == SCORED_CHANNEL:
            return ""
        to_final = CHANNEL + SCORED_CHANNEL + MESSAGE
        if self._state == "role":
            return to_final
        if self._state == "idle":
            return START + "assistant" + to_final
        if self._state == "content":
            return END + START + "assistant" + to_final
        return ""  # a mitad de un encabezado: no hay un marcado seguro

    def feed(self, text: str, is_special: bool) -> str:
        if is_special:
            self._control(text)
            return CONTROL
        if self._state == "content":
            return self.channel
        if self._state == "role":
            self.role += text
            return HEADER
        if self._state == "channel":
            self._channel_text += text
            return HEADER
        if self._state == "other_header":
            return HEADER
        return NO_CHANNEL

    def _control(self, text: str) -> None:
        if text == START:
            self._state, self.role, self._channel_text, self.channel = "role", "", "", NO_CHANNEL
            self.ended_with = None
        elif text == CHANNEL:
            self._state, self._channel_text = "channel", ""
        elif text == CONSTRAIN:
            self._state = "other_header"
        elif text == MESSAGE:
            words = self._channel_text.split()
            self.channel = words[0] if words else NO_CHANNEL
            self._state = "content"
        elif text in (END, RETURN, CALL):
            self._state, self.channel, self.ended_with = "idle", NO_CHANNEL, text
        # Otros especiales (BOS, padding) no cambian el estado.


def has_markup(text: str) -> bool:
    """True si el texto trae algún token especial de harmony."""
    return any(tok in text for tok in SPECIAL_TOKENS)


def final_view(
    response: str, spans: Sequence[tuple[int, int]] = (), *, prefix: str = START + "assistant"
) -> tuple[str, list[tuple[int, int]]]:
    """Texto del canal `final` de una respuesta con marcado harmony y sus spans en ese texto.

    Los offsets de `spans` (code points) se refieren a `response` con el marcado; los de los tokens
    de un backend se refieren al texto final. Este helper los lleva al mismo marco. Un span se
    recorta al contenido de `final`; si no toca `final` (por ejemplo, cae en `analysis`), se
    descarta. `prefix` es el marcado con que termina el prompt (por defecto `<|start|>assistant`).
    Sin marcado, devuelve la respuesta y los spans sin cambios.
    """
    if not has_markup(response):
        return response, [(int(a), int(b)) for a, b in spans]
    parser = HarmonyParser()
    for start, end, special in _segments(prefix):
        parser.feed(prefix[start:end], special)
    final_chars: list[int] = []  # posición en `response` de cada carácter del texto final
    for start, end, special in _segments(response):
        if parser.feed(response[start:end], special) == SCORED_CHANNEL and not special:
            final_chars.extend(range(start, end))
    text = "".join(response[i] for i in final_chars)
    mapped: list[tuple[int, int]] = []
    for a, b in spans:
        inside = [k for k, i in enumerate(final_chars) if a <= i < b]
        if inside:
            mapped.append((inside[0], inside[-1] + 1))
    return text, mapped


def _segments(text: str) -> Iterator[tuple[int, int, bool]]:
    """Corta un string en tramos [start, end): cada token especial de harmony y el texto entre ellos."""
    pos = 0
    while pos < len(text):
        hits = [(i, tok) for tok in SPECIAL_TOKENS if (i := text.find(tok, pos)) >= 0]
        start, tok = min(hits) if hits else (len(text), "")
        if start > pos:
            yield pos, start, False
        if tok:
            yield start, start + len(tok), True
        pos = start + len(tok)
