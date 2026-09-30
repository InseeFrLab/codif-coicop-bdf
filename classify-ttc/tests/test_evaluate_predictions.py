"""Tests for the HTML report of src.evaluation.evaluate_predictions."""

from __future__ import annotations

import pandas as pd

from src.evaluation.evaluate_predictions import (
    evaluate_predictions,
    evaluate_truncation,
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


def test_canonical_truth_with_mapping(tmp_path):
    # Linear hierarchy: 01.1.1.3 is pruned to its parent 01.1.1.
    mapping = pd.DataFrame({"code": ["01.1.1.3"], "code_parent_equivalent": ["01.1.1"]})
    mapping_path = tmp_path / "mapping_lvl4.parquet"
    mapping.to_parquet(mapping_path)
    preds = pd.DataFrame(
        {
            "product": ["pates", "riz", "inconnu"],
            "code": ["01.1.1.3.1", "01.1.1.1.1", None],
            # Raw level 4 differs from the truth's level 4, but both prune to 01.1.1.
            "predicted_code": ["01.1.1", "01.1.1.1", "01.1.8.1"],
            "predicted_level4": ["stale", "stale", "stale"],
        }
    )
    pred_path = tmp_path / "predictions.parquet"
    preds.to_parquet(pred_path)

    results, report = run_evaluate_predictions(pred_path, max_k=1, mapping_path=mapping_path)

    assert results["n_samples"] == 2  # row without truth excluded
    assert results["rule"] == "truncate"
    assert set(results["levels"]) == {1, 2, 3, 4}
    # Same N at every level; the pruned 01.1.1 is still judged at level 4.
    assert all(r["N"] == 2 for r in results["levels"].values())
    assert results["levels"][4]["top-1"] == 1.0
    assert "code_lvl4" in report


def test_truncation_rule_matches_codif_common():
    """A prediction finer than the truth is not credited beyond the truth's depth."""
    df = pd.DataFrame({
        "code_lvl4": ["01.4", "01.1.1.1"],
        "predicted_code": ["01.4.3.1", "01.1.1.1"],
    })
    results = evaluate_truncation(df, max_k=1)
    assert results["levels"][2]["top-1"] == 1.0
    assert results["levels"][4]["top-1"] == 0.5
