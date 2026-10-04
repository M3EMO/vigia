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
