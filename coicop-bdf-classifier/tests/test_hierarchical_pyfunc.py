"""Tests for HierarchicalCOICOPPyfuncWrapper."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.tracking.mlflow_utils import HierarchicalCOICOPPyfuncWrapper


LEVEL_NAMES = ["level1", "level2", "level3", "level4", "level5"]


@pytest.fixture
def stopwords_file(tmp_path):
    """Write a small stopwords JSON file and return its path."""
    sw = ["de", "la", "le", "les", "du", "des"]
    path = tmp_path / "stopwords.json"
    path.write_text(json.dumps(sw), encoding="utf-8")
    return str(path)


@pytest.fixture
def mock_context(tmp_path, stopwords_file):
    """Mock MLflow PythonModelContext with artifacts dict."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    ctx = MagicMock()
    ctx.artifacts = {
        "model_dir": str(model_dir),
        "stopwords": stopwords_file,
    }
    return ctx


def _make_predict_result(n: int, top_k: int = 1) -> dict:
    """Build a realistic HierarchicalCOICOPClassifier.predict() return value."""
    if top_k == 1:
        all_levels = {}
        for lvl in LEVEL_NAMES:
            all_levels[lvl] = {
                "predictions": [f"{lvl}_code"] * n,
                "confidence": [0.9] * n,
            }
        return {
            "final_code": ["01.1.2.3.4"] * n,
            "final_level": ["level5"] * n,
            "final_confidence": [0.85] * n,
            "combined_confidence": [0.59] * n,
            "all_levels": all_levels,
        }

    # top_k > 1: predictions/confidence are lists of lists
    all_levels = {}
    for lvl in LEVEL_NAMES:
        all_levels[lvl] = {
            "predictions": [[f"{lvl}_code_k{k+1}" for k in range(top_k)] for _ in range(n)],
            "confidence": [[round(0.9 - k * 0.1, 2) for k in range(top_k)] for _ in range(n)],
        }
    return {
        "final_code": ["01.1.2.3.4"] * n,
        "final_level": ["level5"] * n,
        "final_confidence": [0.85] * n,
        "combined_confidence": [0.59] * n,
        "all_levels": all_levels,
    }


@pytest.fixture
def sample_input():
    return pd.DataFrame({"text": ["pain complet bio", "eau minerale"]})


@patch("src.tracking.mlflow_utils.HierarchicalCOICOPPyfuncWrapper.load_context")
def _build_wrapper(stopwords_file, mock_load_context):
    """Helper: create a wrapper with mocked internals, bypassing load_context."""
    wrapper = HierarchicalCOICOPPyfuncWrapper()
    wrapper.classifier = MagicMock()
    wrapper.stopwords = json.loads(open(stopwords_file, encoding="utf-8").read())
    wrapper._preprocess_text = MagicMock(side_effect=lambda df, col, sw: df)
    return wrapper


# ---------- Tests ----------


class TestLoadContext:
    @patch("src.classifiers.hierarchical_classifier.HierarchicalCOICOPClassifier")
    @patch("src.preprocessing.data_preparation.preprocess_text")
    def test_load_context(self, mock_preprocess, mock_cls, mock_context):
        """load_context loads the classifier and stopwords."""
        mock_cls.load.return_value = MagicMock()

        wrapper = HierarchicalCOICOPPyfuncWrapper()
        wrapper.load_context(mock_context)

        mock_cls.load.assert_called_once_with(mock_context.artifacts["model_dir"])
        assert wrapper.stopwords is not None
        assert isinstance(wrapper.stopwords, list)


class TestPredictBasic:
    def test_predict_basic(self, sample_input, stopwords_file):
        """top_k=1 returns expected columns."""
        wrapper = _build_wrapper(stopwords_file)
        n = len(sample_input)
        wrapper.classifier.predict.return_value = _make_predict_result(n, top_k=1)

        result = wrapper.predict(None, sample_input, params=None)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == n

        # Core columns
        assert "predicted_code" in result.columns
        assert "confidence" in result.columns
        assert "combined_confidence" in result.columns

        # Per-level columns
        for lvl in LEVEL_NAMES:
            assert f"predicted_{lvl}" in result.columns
            assert f"confidence_{lvl}" in result.columns

        # No top-K alternative columns
        assert "predicted_level1_top2" not in result.columns

    def test_predict_values(self, sample_input, stopwords_file):
        """Verify actual values in output."""
        wrapper = _build_wrapper(stopwords_file)
        n = len(sample_input)
        wrapper.classifier.predict.return_value = _make_predict_result(n, top_k=1)

        result = wrapper.predict(None, sample_input, params=None)

        assert list(result["predicted_code"]) == ["01.1.2.3.4"] * n
        assert list(result["confidence"]) == [0.85] * n
        assert list(result["combined_confidence"]) == [0.59] * n


class TestPredictTopK:
    def test_predict_top_k(self, sample_input, stopwords_file):
        """top_k=3 adds alternative columns."""
        wrapper = _build_wrapper(stopwords_file)
        n = len(sample_input)
        top_k = 3
        wrapper.classifier.predict.return_value = _make_predict_result(n, top_k=top_k)

        result = wrapper.predict(None, sample_input, params={"top_k": top_k})

        assert isinstance(result, pd.DataFrame)
        assert len(result) == n

        # Core columns still present
        assert "predicted_code" in result.columns
        assert "confidence" in result.columns
        assert "combined_confidence" in result.columns

        # Per-level top-1 columns
        for lvl in LEVEL_NAMES:
            assert f"predicted_{lvl}" in result.columns
            assert f"confidence_{lvl}" in result.columns

        # Alternative columns for top-2 and top-3
        for lvl in LEVEL_NAMES:
            for rank in range(2, top_k + 1):
                assert f"predicted_{lvl}_top{rank}" in result.columns
                assert f"confidence_{lvl}_top{rank}" in result.columns


class TestPreprocessCalled:
    def test_predict_calls_preprocess(self, sample_input, stopwords_file):
        """preprocess_text is called on input before prediction."""
        wrapper = _build_wrapper(stopwords_file)
        n = len(sample_input)
        wrapper.classifier.predict.return_value = _make_predict_result(n, top_k=1)

        wrapper.predict(None, sample_input, params=None)

        wrapper._preprocess_text.assert_called_once()
        call_args = wrapper._preprocess_text.call_args
        pd.testing.assert_frame_equal(call_args[0][0], sample_input)
        assert call_args[0][1] == "text"
        assert call_args[0][2] == wrapper.stopwords


class TestDefaultTopK:
    def test_predict_default_top_k(self, sample_input, stopwords_file):
        """top_k defaults to 1 when params is None."""
        wrapper = _build_wrapper(stopwords_file)
        n = len(sample_input)
        wrapper.classifier.predict.return_value = _make_predict_result(n, top_k=1)

        wrapper.predict(None, sample_input, params=None)

        call_kwargs = wrapper.classifier.predict.call_args[1]
        assert call_kwargs["top_k"] == 1

    def test_predict_empty_params(self, sample_input, stopwords_file):
        """top_k defaults to 1 when params is empty dict."""
        wrapper = _build_wrapper(stopwords_file)
        n = len(sample_input)
        wrapper.classifier.predict.return_value = _make_predict_result(n, top_k=1)

        wrapper.predict(None, sample_input, params={})

        call_kwargs = wrapper.classifier.predict.call_args[1]
        assert call_kwargs["top_k"] == 1
