import datetime as dt
import json

import httpx
import pytest

from vigia import labels as L
from vigia.contracts import TokenEvent
from vigia.labels import (
    IGNORE_LABEL,
    FakeJudge,
    InvalidJudgment,
    JudgeError,
    OpenAICompatibleJudge,
    date_match,
    exact_match,
    label_tokens_from_answer,
    label_with_judge,
    normalize_es,
    numeric_match,
    parse_dates_es,
    parse_number_es,
    spans_to_token_labels,
    token_f1,
    validate_judgment,
)

# --- AT-M06: etiquetas computables ------------------------------------------------------------------


def test_at_m06_normalization_spanish():
    """AT-M06: normalización con artículos, tildes, ñ y puntuación Unicode."""
    assert normalize_es("¡El Río de la Plata!") == "rio de plata"
    assert normalize_es("  Los   años «dorados»… ") == "años dorados"
    assert normalize_es("Perón", strip_accents=False) == "perón"
    assert normalize_es("PINGÜINO") == "pinguino"
    assert normalize_es("Unas Islas") == "islas"


def test_at_m06_exact_match_hand_values():
    """AT-M06: EM coincide con los valores calculados a mano."""
    assert exact_match("La Ciudad de Buenos Aires.", "ciudad de buenos aires") is True
    assert exact_match("Buenos Aires", "la Ciudad Autónoma de Buenos Aires") is False
    assert exact_match("San Martín", ["José de San Martín", "San Martin"]) is True
    assert exact_match("Perón", "Peron") is True
    assert exact_match("Perón", "Peron", strip_accents=False) is False


def test_at_m06_token_f1_hand_values():
    """AT-M06: F1 de tokens coincide con los valores calculados a mano."""
    # pred [buenos, aires]; gold [ciudad, autonoma, de, buenos, aires]: P = 1, R = 2/5 -> F1 = 4/7.
    assert token_f1("Buenos Aires", "la Ciudad Autónoma de Buenos Aires") == pytest.approx(4 / 7)
    # pred [andes, y, alpes]; gold [andes]: P = 1/3, R = 1 -> F1 = 1/2.
    assert token_f1("los Andes y los Alpes", "los Andes") == pytest.approx(0.5)
    # Multiconjunto: común = 1 rojo + 1 azul = 2; P = R = 2/3.
    assert token_f1("rojo rojo azul", "rojo azul azul") == pytest.approx(2 / 3)
    assert token_f1("el año 1810", "Año 1810.") == 1.0
    assert token_f1("Mendoza", ["Córdoba", "mendoza"]) == 1.0
    assert token_f1("El", "la") == 1.0  # ambas vacías tras normalizar
    assert token_f1("Rosario", "") == 0.0


def test_at_m06_numbers_spanish_convention():
    """AT-M06: aritmética con números en convención española."""
    assert parse_number_es("1.234,5") == 1234.5
    assert parse_number_es("1,5") == 1.5
    assert parse_number_es("1.234") == 1234
    assert parse_number_es("3.14") == 3.14
    assert parse_number_es("1,234,567") == 1234567
    assert parse_number_es("−7") == -7
    assert parse_number_es("doce") is None
    assert numeric_match("El resultado es 42.", 42)
    assert not numeric_match("2 + 2 = 5", 4)
    assert numeric_match("2 + 2 = 4", "4")
    assert numeric_match("Hay 1.500 habitantes", "1500")
    assert not numeric_match("No sé", 3)
    with pytest.raises(ValueError):
        numeric_match("3", "tres")


def test_at_m06_dates_spanish():
    """AT-M06: fechas en español con formatos de texto, ISO y día/mes/año."""
    assert parse_dates_es("Nació el 12 de octubre de 1492.") == [dt.date(1492, 10, 12)]
    assert parse_dates_es("El 1º de mayo del 2000 y el 9 de Setiembre de 1999") == [
        dt.date(2000, 5, 1),
        dt.date(1999, 9, 9),
    ]
    assert parse_dates_es("25/05/1810 o 1816-07-09") == [dt.date(1810, 5, 25), dt.date(1816, 7, 9)]
    assert parse_dates_es("el 31 de febrero de 2020") == []
    assert date_match("La Revolución fue el 25 de Mayo de 1810.", "1810-05-25")
    assert not date_match("Fue el 9 de julio de 1816.", dt.date(1810, 5, 25))
    with pytest.raises(ValueError):
        date_match("x", "mayo")


# --- etiquetas por token ----------------------------------------------------------------------------


def _tokens(text, pieces, extra=None):
    out, pos = [], 0
    for i, piece in enumerate(pieces):
        kw = (extra or {}).get(i, {})
        out.append(
            TokenEvent(index=i, token_id=i, text=piece, start=pos, end=pos + len(piece), features={}, **kw)
        )
        pos += len(piece)
    assert "".join(pieces) == text
    return out


def test_spans_to_token_labels_by_char_overlap():
    text = "Buenos Aires es la capital"
    toks = _tokens(text, ["Buen", "os", " Aires", " es", " la", " capital"], {5: {"is_special": True}})
    assert spans_to_token_labels(toks, [(0, 12)]) == [1, 1, 1, 0, 0, IGNORE_LABEL]
    assert spans_to_token_labels(toks, [(7, 8)]) == [0, 0, 1, 0, 0, IGNORE_LABEL]
    assert spans_to_token_labels([(0, 3), (3, 3), (3, 6)], [(3, 6)]) == [0, 1, 1]
    with pytest.raises(ValueError):
        spans_to_token_labels(toks, [(5, 5)])


def test_label_tokens_from_answer_masks_outside_and_non_final():
    text = "La respuesta es 1810."
    toks = _tokens(text, ["La", " respuesta", " es", " 18", "10", "."], {0: {"channel": "analysis"}})
    assert label_tokens_from_answer(toks, [(16, 20)], correct=False) == [IGNORE_LABEL] * 3 + [
        1,
        1,
        IGNORE_LABEL,
    ]
    assert label_tokens_from_answer(toks, [(16, 20)], correct=True) == [IGNORE_LABEL] * 3 + [
        0,
        0,
        IGNORE_LABEL,
    ]
    assert label_tokens_from_answer(toks, None, correct=False) == [IGNORE_LABEL, 1, 1, 1, 1, 1]


# --- AT-M07: validador del juez ---------------------------------------------------------------------

ANSWER = "La capital es Córdoba y fue fundada en 1573. Córdoba es grande."


def test_validate_judgment_maps_spans_and_occurrences():
    v = validate_judgment('{"label": 1, "hallucinated_spans": ["Córdoba"]}', ANSWER)
    assert v.label == 1 and v.spans == ((14, 21),)
    v = validate_judgment({"label": 1, "hallucinated_spans": ["Córdoba", "Córdoba"]}, ANSWER)
    assert v.spans == ((14, 21), (45, 52))
    v = validate_judgment(
        {"label": 1, "hallucinated_spans": [{"text": "Córdoba", "occurrence": 2}, "1573"]}, ANSWER
    )
    assert v.spans == ((45, 52), (39, 43))
    assert validate_judgment({"label": 0, "hallucinated_spans": []}, ANSWER).spans == ()


@pytest.mark.parametrize(
    "raw, reason",
    [
        ("no es json", "json_invalido"),
        ('```json\n{"label": 0, "hallucinated_spans": []}\n```', "json_invalido"),
        ('{"label": NaN, "hallucinated_spans": []}', "json_invalido"),
        ("[1, 2]", "schema"),
        ('{"label": 1, "label": 0, "hallucinated_spans": []}', "schema"),
        ({"label": 1, "hallucinated_spans": ["Córdoba"], "extra": 1}, "schema"),
        ({"label": 1, "hallucinated_spans": "Córdoba"}, "schema"),
        ({"label": 1, "hallucinated_spans": [3]}, "schema"),
        ({"label": 1, "hallucinated_spans": [{"text": "Córdoba"}]}, "schema"),
        ({"label": "1", "hallucinated_spans": ["Córdoba"]}, "label_invalido"),
        ({"label": True, "hallucinated_spans": ["Córdoba"]}, "label_invalido"),
        ({"label": 2, "hallucinated_spans": ["Córdoba"]}, "label_invalido"),
        ({"label": 1, "hallucinated_spans": ["Rosario"]}, "span_inexistente"),
        ({"label": 1, "hallucinated_spans": ["córdoba"]}, "span_inexistente"),
        ({"label": 1, "hallucinated_spans": ["Córdoba  y"]}, "span_inexistente"),
        ({"label": 1, "hallucinated_spans": ["  "]}, "span_vacio"),
        ({"label": 1, "hallucinated_spans": [{"text": "Córdoba", "occurrence": 3}]}, "ocurrencia_invalida"),
        ({"label": 1, "hallucinated_spans": [{"text": "Córdoba", "occurrence": 0}]}, "ocurrencia_invalida"),
        ({"label": 1, "hallucinated_spans": ["Córdoba"] * 3}, "ocurrencia_invalida"),
        ({"label": 0, "hallucinated_spans": ["Córdoba"]}, "inconsistente"),
        ({"label": 1, "hallucinated_spans": []}, "inconsistente"),
    ],
)
def test_at_m07_invalid_judgments_are_rejected(raw, reason):
    """AT-M07: JSON inválido o spans inexistentes se rechazan con un motivo contable."""
    with pytest.raises(InvalidJudgment) as exc:
        validate_judgment(raw, ANSWER)
    assert exc.value.reason == reason


def test_at_m07_fake_judge_report_counts_rejections():
    """AT-M07: un juez falso con spans inexistentes o JSON inválido: el item se rechaza y se cuenta."""
    judge = FakeJudge(
        {
            "Mendoza es la capital.": {"label": 1, "hallucinated_spans": ["Mendoza"]},
            "Lima es la capital.": {"label": 1, "hallucinated_spans": ["Bogotá"]},
            "Quito es la capital.": "esto no es JSON",
            "Santiago es la capital.": {"label": 0, "hallucinated_spans": []},
            "Asunción es la capital.": '{"label": 1}',
        }
    )
    items = [
        {"id": "a", "question": "¿Capital?", "answer": "Mendoza es la capital."},
        {"id": "b", "question": "¿Capital?", "answer": "Lima es la capital."},
        {"id": "c", "question": "¿Capital?", "answer": "Quito es la capital."},
        {"id": "d", "question": "¿Capital?", "answer": "Santiago es la capital.", "evidence": "Chile"},
        {"id": "e", "question": "¿Capital?", "answer": "Asunción es la capital."},
        {"id": "f", "question": "¿Capital?", "answer": "Montevideo es la capital."},
        {"id": "g", "question": "¿Capital?"},
    ]
    report = label_with_judge(judge, items)
    assert set(report.verdicts) == {"a", "d"}
    assert report.verdicts["a"].spans == ((0, 7),)
    assert report.rejected == {
        "b": "span_inexistente",
        "c": "json_invalido",
        "e": "schema",
        "f": "transporte",
        "g": "item_invalido",
    }
    summary = report.to_dict()
    assert summary["n_items"] == 7 and summary["n_accepted"] == 2 and summary["n_rejected"] == 5
    assert summary["rejected_by_reason"]["span_inexistente"] == 1
    assert summary["prompt_sha256"] == L.JUDGE_PROMPT_SHA256
    with pytest.raises(ValueError, match="repetido"):
        label_with_judge(judge, [items[0], items[0]])


# --- juez OpenAI-compatible -------------------------------------------------------------------------


def _chat(content):
    return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": content}}]})


def test_openai_compatible_judge_request_and_parse(monkeypatch):
    seen = []

    def handler(request):
        seen.append(request)
        return _chat('{"label": 1, "hallucinated_spans": ["Córdoba"]}')

    judge = OpenAICompatibleJudge(
        "https://juez.example/v1/",
        "sk-secreta",
        "glm-test",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    report = label_with_judge(
        judge, [{"id": "1", "question": "¿Capital?", "answer": ANSWER, "evidence": "Ev"}]
    )
    assert report.verdicts["1"].spans == ((14, 21),)
    req = seen[0]
    assert str(req.url) == "https://juez.example/v1/chat/completions"
    assert req.headers["authorization"] == "Bearer sk-secreta"
    body = json.loads(req.content)
    assert body["model"] == "glm-test" and body["temperature"] == 0
    assert body["response_format"] == {"type": "json_object"}
    assert ANSWER in body["messages"][1]["content"] and "Ev" in body["messages"][1]["content"]
    assert "sk-secreta" not in repr(judge)


def test_openai_compatible_judge_retries_and_errors(monkeypatch):
    monkeypatch.setattr(L.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def flaky(request):
        calls["n"] += 1
        return httpx.Response(503) if calls["n"] == 1 else _chat('{"label": 0, "hallucinated_spans": []}')

    judge = OpenAICompatibleJudge(
        "http://localhost:8000/v1", "k", "m", client=httpx.Client(transport=httpx.MockTransport(flaky))
    )
    assert json.loads(judge.judge("q", "a")) == {"label": 0, "hallucinated_spans": []}
    assert calls["n"] == 2

    def denied(request):
        return httpx.Response(401, json={"error": "bad key k"})

    judge = OpenAICompatibleJudge(
        "http://127.0.0.1:1/v1", "k", "m", client=httpx.Client(transport=httpx.MockTransport(denied))
    )
    with pytest.raises(JudgeError, match="401"):
        judge.judge("q", "a")

    def garbage(request):
        return httpx.Response(200, json={"nope": 1})

    judge = OpenAICompatibleJudge(
        "http://127.0.0.1:1/v1", "k", "m", client=httpx.Client(transport=httpx.MockTransport(garbage))
    )
    with pytest.raises(JudgeError, match="formato"):
        judge.judge("q", "a")

    def down(request):
        raise httpx.ConnectError("sin red")

    judge = OpenAICompatibleJudge(
        "http://localhost/v1",
        "k",
        "m",
        max_retries=1,
        client=httpx.Client(transport=httpx.MockTransport(down)),
    )
    with pytest.raises(JudgeError, match="2 intentos"):
        judge.judge("q", "a")


def test_openai_compatible_judge_validates_configuration():
    with pytest.raises(ValueError, match="https"):
        OpenAICompatibleJudge("http://juez.example/v1", "k", "m")
    with pytest.raises(ValueError):
        OpenAICompatibleJudge("ftp://juez.example", "k", "m")
    with pytest.raises(ValueError, match="modelo"):
        OpenAICompatibleJudge("https://juez.example", "k", "")
    with pytest.raises(ValueError):
        OpenAICompatibleJudge("https://juez.example", "k\nX-Inyectado: 1", "m")
    OpenAICompatibleJudge("http://juez.interno/v1", "k", "m", allow_insecure_http=True)


def test_parse_number_es_leading_zero_is_decimal():
    """Regresión: "parse_number_es lee '0.500' como 500: un cero a la izquierda se toma como grupo
    de miles"."""
    assert parse_number_es("0.500") == 0.5
    assert parse_number_es("-0.250") == -0.25
    assert numeric_match("La probabilidad es 0.500", 0.5)
    assert parse_number_es("1.234") == 1234 and parse_number_es("12.345") == 12345
    assert parse_number_es("999.000") == 999000


def test_parse_date_parts_es_reads_partial_dates():
    """Regresión de "Item de fecha con gold de solo el año marca como incorrecta una respuesta
    correcta..." y "gold_correct acepta por F1 >= 0,5 respuestas con otro año, mes o número...":
    fechas completas, mes y año, y año suelto, sin superponerse."""
    from vigia.labels import parse_date_parts_es

    assert parse_date_parts_es("En 1945, el 8 de mayo.") == [(1945, None, None)]
    assert parse_date_parts_es("1945-05-08") == [(1945, 5, 8)]
    assert parse_date_parts_es("Del 8 de mayo de 1945 a septiembre del 1946") == [
        (1945, 5, 8),
        (1946, 9, None),
    ]
    assert parse_date_parts_es("Setiembre de 1810") == [(1810, 9, None)]
    assert parse_date_parts_es("1.945 personas, 12345 y 19,45") == []
    assert parse_date_parts_es("unos 300 km") == [(300, None, None)]
