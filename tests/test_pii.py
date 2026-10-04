"""Tests de vigia_ee.pii: redacción argentina, re-mapeo de offsets, texto seguro y CSV."""

from __future__ import annotations

import csv
import io

import jinja2
import pytest

from vigia_ee import pii

VALID_CUIT = "20-12345678-6"
INVALID_CUIT = "20-12345678-7"
VALID_CBU = "0110599520000001234569"
VALID_CARD = "4111 1111 1111 1111"
INVALID_CARD = "4111 1111 1111 1112"

SAMPLE = (
    f"Soy Juan. Mi DNI es 30.123.456 y mi CUIT {VALID_CUIT}. "
    f"CBU {VALID_CBU}, tarjeta {VALID_CARD}. Mail juan.perez@example.com.ar. "
    "Tel +54 9 11 4567-8901, fijo (011) 4567-8901. IP 192.168.0.1 y 2001:db8::1."
)
SECRETS_IN_SAMPLE = (
    "30.123.456",
    VALID_CUIT,
    VALID_CBU,
    VALID_CARD,
    "juan.perez@example.com.ar",
    "4567-8901",
    "192.168.0.1",
    "2001:db8::1",
)


def test_at_p08_redact_replaces_every_argentine_identifier_with_marker():
    """AT-P08: DNI, CUIT válido, CBU, tarjeta Luhn, email, teléfono e IP quedan como marcadores."""
    text, counts = pii.redact(SAMPLE)
    for secret in SECRETS_IN_SAMPLE:
        assert secret not in text
    for marker in ("[DNI]", "[CUIT]", "[CBU]", "[TARJETA]", "[EMAIL]", "[TELEFONO]", "[IP]"):
        assert marker in text
    assert counts == {"DNI": 1, "CUIT": 1, "CBU": 1, "TARJETA": 1, "EMAIL": 1, "TELEFONO": 2, "IP": 2}
    assert text.startswith("Soy Juan. Mi DNI es [DNI] y mi CUIT [CUIT].")


def test_cuit_module_11_rejects_invalid_check_digit():
    assert pii.cuit_is_valid(VALID_CUIT)
    assert pii.cuit_is_valid("20123456786")
    assert not pii.cuit_is_valid(INVALID_CUIT)
    assert not pii.cuit_is_valid("99-12345678-6")  # prefijo inexistente
    text, counts = pii.redact(f"CUIT {INVALID_CUIT}")
    assert INVALID_CUIT in text and counts == {}


def test_cbu_requires_both_check_digits():
    assert pii.cbu_is_valid(VALID_CBU)
    bad_block1 = VALID_CBU[:7] + str((int(VALID_CBU[7]) + 1) % 10) + VALID_CBU[8:]
    bad_block2 = VALID_CBU[:21] + str((int(VALID_CBU[21]) + 1) % 10)
    assert not pii.cbu_is_valid(bad_block1)
    assert not pii.cbu_is_valid(bad_block2)
    assert pii.redact(f"cuenta {bad_block2}")[1] == {}


def test_card_requires_luhn():
    assert pii.luhn_is_valid(VALID_CARD)
    assert pii.luhn_is_valid("5500-0000-0000-0004")
    assert not pii.luhn_is_valid(INVALID_CARD)
    assert pii.redact(f"tarjeta {INVALID_CARD}")[1] == {}
    assert pii.redact("pago con 5500-0000-0000-0004 ayer")[0] == "pago con [TARJETA] ayer"


def test_dni_needs_context_to_avoid_redacting_amounts_and_years():
    assert pii.redact("DNI 12345678")[0] == "DNI [DNI]"
    assert pii.redact("D.N.I. Nº 7.654.321")[0] == "D.N.I. Nº [DNI]"
    assert pii.redact("documento: 30123456")[0] == "documento: [DNI]"
    plain = "La población es 45808747 y la guerra fue 1939-1945."
    assert pii.redact(plain) == (plain, {})


@pytest.mark.parametrize(
    "phone",
    [
        "+54 9 11 4567-8901",
        "+1 (415) 555-0100",
        "(011) 4567-8901",
        "011 4567 8901",
        "15-4567-8901",
        "0351 456-7890",
    ],
)
def test_phone_formats_are_redacted(phone):
    assert pii.redact(f"llamá al {phone} hoy")[0] == "llamá al [TELEFONO] hoy"


def test_phone_with_context_word_and_without():
    assert pii.redact("cel: 4567 8901")[0] == "cel: [TELEFONO]"
    assert pii.redact("código 4567 8901")[1] == {}


def test_ip_addresses_valid_only():
    assert pii.redact("desde 10.0.0.255")[0] == "desde [IP]"
    assert pii.redact("host fe80::1ff:fe23:4567:890a listo")[0] == "host [IP] listo"
    assert pii.redact("versión 999.1.1.1 y hora 12:30:45")[1] == {}


def test_email_redacted():
    assert pii.redact("escribí a ana+test@sub.example.org.")[0] == "escribí a [EMAIL]."


def test_at_p08_offsets_remap_tokens_without_gaps_or_overlaps():
    """AT-P08: un DNI partido entre tokens queda redactado y los tokens re-mapeados cubren el texto."""
    red = pii.redact_with_map(SAMPLE)
    # Tokens de 3 caracteres: cada dato sensible queda partido en varios tokens.
    bounds = [(i, min(i + 3, len(SAMPLE))) for i in range(0, len(SAMPLE), 3)]
    mapped = [(pii.map_offset(red.replacements, s), pii.map_offset(red.replacements, e)) for s, e in bounds]
    pieces = [red.text[s:e] for s, e in mapped]
    assert "".join(pieces) == red.text
    for (s1, e1), (s2, _e2) in zip(mapped, mapped[1:], strict=False):
        assert s1 <= e1 == s2
    joined = "|".join(pieces)
    for secret in SECRETS_IN_SAMPLE:
        assert secret not in joined


def test_map_offset_outside_and_inside_regions():
    red = pii.redact_with_map("ab DNI 12345678 cd")
    assert red.text == "ab DNI [DNI] cd"
    rep = red.replacements[0]
    assert pii.map_offset(red.replacements, 0) == 0
    assert pii.map_offset(red.replacements, rep.start) == rep.new_start
    assert pii.map_offset(red.replacements, rep.start + 3) == rep.new_end
    assert pii.map_offset(red.replacements, rep.end) == rep.new_end
    assert pii.map_offset(red.replacements, len("ab DNI 12345678 cd")) == len(red.text)


def test_at_p13_safe_display_strips_bidi_and_control_characters():
    """AT-P13: bidi (U+202E y familia), controles y tags invisibles no llegan al render."""
    raw = "pago‮FDP.exe⁦x⁩\x00\x07\x1b[31m​ok\U000e0041\nlínea\tsig 👩‍💻"
    clean = pii.safe_display(raw)
    for bad in ("‮", "⁦", "⁩", "\x00", "\x07", "\x1b", "​", "\U000e0041"):
        assert bad not in clean
    assert "\n" in clean and "\t" in clean
    assert "👩‍💻" in clean  # ZWJ de emojis se conserva


def test_at_p13_script_is_escaped_when_rendered_with_autoescape():
    """AT-P13: `<script>` del modelo sale escapado (safe_display + autoescape de Jinja)."""
    env = jinja2.Environment(autoescape=True)  # el panel usa autoescape siempre
    html = env.from_string("<p>{{ t }}</p>").render(t=pii.safe_display("<script>alert(1)</script>‮"))
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


@pytest.mark.parametrize("cell", ["=1+1", "+cmd", "-2+3", "@SUM(A1)", "\tx", "\rx", " =HYPERLINK()", "＝1"])
def test_at_p13_csv_neutralizes_formula_prefixes(cell):
    """AT-P13: celdas que empiezan con =, +, -, @, tab o CR salen prefijadas con '."""
    assert pii.neutralize_csv_cell(cell) == "'" + cell


def test_csv_keeps_safe_values_and_numbers():
    assert pii.neutralize_csv_cell("hola") == "hola"
    assert pii.neutralize_csv_cell(-5) == -5
    out = pii.csv_dumps([["=cmd|' /C calc'!A0", 3.5, "ok"]], header=["texto", "score", "@x"])
    rows = list(csv.reader(io.StringIO(out)))
    assert rows[0] == ["texto", "score", "'@x"]
    assert rows[1] == ["'=cmd|' /C calc'!A0", "3.5", "ok"]
