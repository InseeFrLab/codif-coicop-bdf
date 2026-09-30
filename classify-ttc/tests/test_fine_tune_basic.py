"""Tests for src.train.fine_tune_basic_classifier (model and data mocked)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from src import train

MODULE_DIR = Path(__file__).resolve().parent.parent


def _run(tmp_path, monkeypatch, df: pd.DataFrame, **kwargs):
    monkeypatch.chdir(MODULE_DIR)  # stopwords path is CWD-relative
    classifier = MagicMock()
    classifier.fine_tune.return_value = {
        "train_samples": 1, "val_samples": 1, "dropped_samples": 0,
    }
    with (
        patch.object(train.BasicCOICOPClassifier, "load", return_value=classifier) as load,
        patch("src.preprocessing.data_preparation.read_parquet", return_value=df),
        patch("src.predict._resolve_mlflow_path", side_effect=lambda p: Path("/resolved")) as resolve,
    ):
        train.fine_tune_basic_classifier(
            data_path="data.parquet", output_dir=str(tmp_path), **kwargs
        )
    tuned_df = classifier.fine_tune.call_args.kwargs["df"]
    return tuned_df, load, resolve


def test_code_level_truncates_codes(tmp_path, monkeypatch):
    df = pd.DataFrame(
        {"product": ["riz", "pates", "carte"], "code": ["01.1.1.1.3", "01.1.1.3.1", "98.3"]}
    )
    tuned, _, _ = _run(
        tmp_path, monkeypatch, df,
        model_path="runs:/abc/model", code_column="code", code_level=4,
    )
    assert tuned["code"].tolist() == ["01.1.1.1", "01.1.1.3", "98.3"]


def test_preprocess_uses_text_column(tmp_path, monkeypatch):
    df = pd.DataFrame({"l_pr_product": ["Crème BRÛLÉE 2X"], "code": ["01.1.1.1"]})
    tuned, _, _ = _run(
        tmp_path, monkeypatch, df,
        model_path="runs:/abc/model", code_column="code",
        text_column="l_pr_product", preprocess=True,
    )
    assert tuned["l_pr_product"].tolist() == ["creme brulee"]


def test_mlflow_uri_is_resolved(tmp_path, monkeypatch):
    df = pd.DataFrame({"product": ["riz"], "code8": ["01.1.1.1"]})
    _, load, resolve = _run(tmp_path, monkeypatch, df, model_path="runs:/abc/model")
    resolve.assert_called_once_with("runs:/abc/model")
    load.assert_called_once_with(Path("/resolved"))


def test_train_basic_code_level_truncates_codes(tmp_path, monkeypatch):
    monkeypatch.chdir(MODULE_DIR)
    df = pd.DataFrame(
        {"l_pr_product": ["riz", "pates", "carte"], "code": ["01.1.1.1.3", "01.1.1.3.1", "98.3"]}
    )
    with (
        patch.object(train, "BasicCOICOPClassifier") as cls,
        patch("src.preprocessing.data_preparation.read_parquet", return_value=df),
    ):
        cls.return_value.train.return_value = {"num_classes": 3}
        train.train_basic_classifier(
            data_path="annotations_full.parquet", output_dir=str(tmp_path),
            text_column="l_pr_product", code_column="code", code_level=4,
        )
    trained = cls.return_value.train.call_args.kwargs["df"]
    assert trained["code"].tolist() == ["01.1.1.1", "01.1.1.3", "98.3"]


def test_fine_tune_drops_single_sample_classes():
    """Stratified split needs 2 samples per class: singletons are left out."""
    from src.classifiers.basic_classifier import BasicCOICOPClassifier

    clf = BasicCOICOPClassifier()
    clf._is_trained = True
    clf.label_to_idx = {"01.1.1.1": 0, "01.1.1.3": 1, "01.1.8.1": 2}
    clf.classifier = MagicMock()
    df = pd.DataFrame({
        "product": [f"riz {i}" for i in range(5)] + [f"pates {i}" for i in range(5)]
        + ["sucre", "inconnu"],
        "code8": ["01.1.1.1"] * 5 + ["01.1.1.3"] * 5 + ["01.1.8.1", "99.9.9.9"],
    })
    metrics = clf.fine_tune(df)
    assert metrics["dropped_samples"] == 2  # unknown label + singleton class
    assert metrics["total_samples"] == 10
    clf.classifier.train.assert_called_once()
