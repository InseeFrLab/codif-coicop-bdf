"""Tests for the text preprocessing of src.preprocessing.data_preparation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.preprocessing.data_preparation import normalize_text, preprocess_text

STOPWORDS_PATH = Path(__file__).resolve().parent.parent / "data" / "text" / "stopwords.json"


@pytest.fixture(scope="module")
def stopwords() -> list[str]:
    with open(STOPWORDS_PATH, encoding="utf-8") as f:
        return json.load(f)


def _pre(texts: list[str], stopwords) -> pd.Series:
    df = pd.DataFrame({"t": texts})
    return preprocess_text(df, "t", stopwords)["t"].reindex(df.index)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # € and ° are dropped, not transliterated to "eur" / "deg".
        ("PROSECCO 11° 75CL 7.99€", "prosecco cl"),
        ("Œufs frais", "oeufs frais"),
        ("Crème brûlée", "creme brulee"),
    ],
)
def test_preprocess_expected(raw, expected, stopwords):
    assert _pre([raw], stopwords).iloc[0] == expected


def test_training_and_production_text_match(stopwords):
    """Raw text (training) and l_pr_product (production) give the same result."""
    raw = [
        "FANTA EXOTIQUE 33CL 6 X 0.95€/UNITE",
        "BIERE BLONDE LEFFE 6, 6° 12 X 25CL",
        "CAPSULES CAFE N°10 – INTENSE",
        "Œufs de poule bœuf",
        "Straße",
        "E85 15.00L X 0.848€/L",
    ]
    l_pr_product = normalize_text(pd.Series(raw)).tolist()
    pd.testing.assert_series_equal(_pre(raw, stopwords), _pre(l_pr_product, stopwords))


def test_non_string_values_are_kept(stopwords):
    assert _pre([123, None, "lait"], stopwords).tolist()[2] == "lait"
