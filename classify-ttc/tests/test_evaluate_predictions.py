"""Tests for the HTML report of src.evaluation.evaluate_predictions."""

from __future__ import annotations

import pandas as pd

from src.evaluation.evaluate_predictions import (
    evaluate_predictions,
    format_html_report,
    run_evaluate_predictions,
    write_text_output,
)


def _predictions() -> pd.DataFrame:
    # Level-4 model (as train-basic on code8) evaluated against 5-level labels,
    # plus one technical code shorter than level 4.
    return pd.DataFrame(
        {
            "product": ["riz", "pates", "sucre", "sel", "carte"],
            "code": ["01.1.1.1.1", "01.1.1.3.1", "01.1.8.1.1", "01.1.9.1.1", "98.3"],
            "predicted_code": ["01.1.1.1", "01.1.1.1", "01.1.8.1", "01.1.8.1", "98.3"],
            "predicted_code_top2": ["01.1.1.3", "01.1.1.3", "01.1.1.1", "01.1.9.1", "98.4"],
            "source": ["ddc", "ddc", "app", "app", "app"],
        }
    )


def test_html_report_contains_tables():
    df = _predictions()
    results = evaluate_predictions(df.copy(), categorical_column="source", max_k=2)
    results["prediction_path"] = "s3://bucket/predictions.parquet"
    page = format_html_report(results, df, meta={"model": "runs:/abc/model"})

    assert page.startswith("<!doctype html>")
    assert "runs:/abc/model" in page
    assert "Accuracy par niveau COICOP" in page
    assert "Par source" in page
    # Finest evaluable level is 4 (the model predicts no level 5).
    assert "Accuracy top-1 par code (level4)" in page
    # Most frequent confusions at level 4, e.g. 01.1.1.3 predicted as 01.1.1.1.
    assert "<td>01.1.1.3</td><td>01.1.1.1</td><td>1</td>" in page


def test_level5_not_counted_for_level4_model():
    results = evaluate_predictions(_predictions(), max_k=2)
    assert results["levels"][5]["N"] == 0
    assert results["levels"][4]["N"] == 4  # 98.3 has no level 4
    assert results["levels"][4]["top-1"] == 0.5
    assert results["levels"][4]["top-2"] == 1.0


def test_run_evaluate_writes_html_locally(tmp_path):
    pred_path = tmp_path / "predictions.parquet"
    _predictions().to_parquet(pred_path)
    html_path = tmp_path / "out" / "report.html"

    _, report = run_evaluate_predictions(
        pred_path, max_k=2, html_output=html_path, report_meta={"train": "x"}
    )

    assert "ACCURACY BY COICOP LEVEL" in report
    assert "<td>x</td>" in html_path.read_text(encoding="utf-8")


def test_write_text_output_local(tmp_path):
    path = tmp_path / "a" / "b.txt"
    write_text_output("bonjour", path)
    assert path.read_text(encoding="utf-8") == "bonjour"
