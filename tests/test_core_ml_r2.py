"""Regresiones de la ronda 2 de cierre del grupo core-ml (evidence, labels y calibrate)."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from vigia.calibrate import split_groups
from vigia.evidence import gold_correct
from vigia.labels import parse_date_parts_es, parse_dates_es

MULTI_NUMBER_GOLD = [
    ("1810-1816", ["1810-1816"], True),
    ("1810", ["1810-1816"], False),
    ("Fue entre 1810 y 1816.", ["1810-1816"], False),
    ("2 a 1", ["2 a 1"], True),
    ("El partido terminó 2 a 1.", ["2 a 1"], True),
    ("El partido terminó 3 a 2.", ["2 a 1"], False),
    ("El partido terminó 1 a 2.", ["2 a 1"], False),
    ("Terminó 1 a 0.", ["1 a 1"], False),
    ("15:30", ["15:30"], True),
    ("A las 15:30 hs.", ["15:30"], True),
    ("A las 15", ["15:30"], False),
    ("1/2", ["1/2"], True),
    ("2/1", ["1/2"], False),
    ("3 de cada 4", ["3 de cada 4"], True),
    ("4 de cada 5", ["3 de cada 4"], False),
    ("3 tazas", ["3/4 de taza"], False),
    ("3/4 de taza", ["3/4 de taza"], True),
]


@pytest.mark.parametrize(("prediction", "answers", "expected"), MULTI_NUMBER_GOLD)
def test_untyped_gold_with_several_numbers(prediction, answers, expected):
    """Regresión: "gold_correct marca incorrecta la respuesta exacta y acepta respuestas erróneas
    cuando el gold sin tipo tiene varios números (rangos, marcadores, horas, fracciones)"."""
    assert gold_correct(prediction, answers)[0] is expected


def test_untyped_gold_with_one_number_keeps_numeric_typing():
    """El arreglo anterior no cambia el gold sin tipo de un solo número con unidad."""
    assert gold_correct("Mide 5 km.", ["5 km"])[0]
    assert not gold_correct("Mide 6 km.", ["5 km"])[0]
    assert gold_correct("Tiene 12 345 habitantes", ["12 345 habitantes"])[0]


@pytest.mark.parametrize("ordinal", ["1ro", "1ero", "1er", "1º", "1.º"])
def test_spanish_ordinal_first_day(ordinal):
    """Regresión: "parse_dates_es no reconoce el ordinal '1ro'/'1ero'/'1er' y pierde el día: una
    respuesta de fecha correcta queda marcada como error"."""
    text = f"El {ordinal} de mayo de 1886."
    assert parse_dates_es(text) == [dt.date(1886, 5, 1)]
    assert parse_date_parts_es(text) == [(1886, 5, 1)]
    assert gold_correct(text, ["1886-05-01"], "date")[0]
    assert gold_correct(text, ["1 de mayo de 1886"])[0]
    assert not gold_correct(f"El {ordinal} de mayo de 1887.", ["1886-05-01"], "date")[0]


@pytest.mark.parametrize("seed", range(8))
def test_stratified_split_reserves_train_for_big_stratum(seed):
    """Regresión: "split_groups no reserva lugar en train para un estrato de 3 o más preguntas: con
    5 preguntas y fracciones 0,3/0,3, train recibe siempre la única pregunta de la otra clase"."""
    groups = np.array(list("abcde"))
    labels = np.array([0, 0, 0, 0, 1])
    split = split_groups(groups, cal_frac=0.3, test_frac=0.3, seed=seed, labels=labels)
    assert [len(split[k]) for k in ("test", "cal", "train")] == [2, 2, 1]
    for name in ("test", "cal", "train"):
        assert set(split[name]) & set("abcd"), name


def test_stratified_split_every_big_stratum_reaches_every_split():
    """Barrido del mismo defecto: si un split tiene lugar para todos los estratos de 3 o más
    preguntas, recibe al menos una de cada uno, y los tamaños de test y cal no cambian."""
    for n in range(3, 40):
        for n_pos in range(n + 1):
            for cal_frac, test_frac in ((0.2, 0.2), (0.3, 0.3), (0.4, 0.4), (0.1, 0.5)):
                groups = np.array([f"q{i:02d}" for i in range(n)])
                labels = np.array([1] * n_pos + [0] * (n - n_pos))
                if n - max(1, round(n * test_frac)) - max(1, round(n * cal_frac)) < 1:
                    continue
                split = split_groups(groups, cal_frac=cal_frac, test_frac=test_frac, seed=n, labels=labels)
                assert len(split["test"]) == max(1, round(n * test_frac))
                assert len(split["cal"]) == max(1, round(n * cal_frac))
                pos = set(groups[:n_pos])
                strata = [s for s in (pos, set(groups) - pos) if len(s) >= 3]
                for name, ids in split.items():
                    if len(ids) >= len(strata):
                        assert all(set(ids) & s for s in strata), (n, n_pos, cal_frac, test_frac, name)
