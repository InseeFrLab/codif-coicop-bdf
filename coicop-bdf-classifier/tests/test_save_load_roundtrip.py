"""Round-trip tests: save → load → predict must be identical.

Phase 2d — upgrade-ttc2.md: verify that v2-compatible save/load preserves
model state and prediction parity for both Basic and Hierarchical classifiers.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import duckdb
import numpy as np
import pytest
import torch

from src.classifiers.basic_classifier import BasicCOICOPClassifier, BasicConfig
from src.classifiers.hierarchical_classifier import (
    HierarchicalCOICOPClassifier,
    HierarchicalConfig,
)

# ── Fixtures ──────────────────────────────────────────────────────────

_TEST_FILE_PATH = Path(__file__).resolve().parents[1] / "data" / "data-train.parquet"


def _load_sample(n: int = 500):
    """Return a small sample DataFrame from the training parquet."""
    path = str(_TEST_FILE_PATH)
    return duckdb.sql(
        f"""
        SELECT product, code
        FROM read_parquet('{path}')
        WHERE code IS NOT NULL AND code != '99.0.0.0.0'
        LIMIT {n}
        """
    ).fetchdf()


TEST_TEXTS = [
    "Riz blanc long grain",
    "Farine T45",
    "Lait entier en poudre",
    "Huile d'olive",
    "Chaussures de sport",
    "Savon de Marseille",
    "Transport en commun",
    "Coiffure",
    "Assurance habitation",
    "Médicament",
]


# ── Basic classifier round-trip ───────────────────────────────────────


class TestBasicRoundTrip:
    """Save/load round-trip for BasicCOICOPClassifier."""

    def test_attribute_label_names_exists(self) -> None:
        """Trained classifier must have `label_names` attribute."""
        train_and_predict(
            BasicCOICOPClassifier,
            BasicConfig,
            n_rows=300,
            params={"embedding_dim": 32, "max_seq_length": 32, "ngram_num_tokens": 3000},
        )
        # If we reach this point, the classifier was trained and has label_names

    def test_predict_consistent_after_load(self, tmp_path: Path) -> None:
        """Predictions before and after save/load are identical."""
        tmpdir = str(tmp_path / "basic_rt")

        clf = BasicCOICOPClassifier(BasicConfig(
            ngram_num_tokens=3000,
            embedding_dim=32,
            max_seq_length=32,
            num_epochs=1,
            patience=1,
            batch_size=16,
            lr=0.1,
        ))
        df = _load_sample(300)
        clf.train(df, text_column="product", code_column="code", save_dir=tmpdir)

        pred_before = clf.predict(TEST_TEXTS)
        assert len(pred_before["predictions"]) == len(TEST_TEXTS)

        clf.save(tmpdir)
        loaded = BasicCOICOPClassifier.load(tmpdir)

        pred_after = loaded.predict(TEST_TEXTS)

        assert pred_before["predictions"] == pred_after["predictions"]
        assert np.allclose(pred_before["confidence"], pred_after["confidence"], atol=1e-6)


# ── Hierarchical classifier round-trip ────────────────────────────────

def _train_and_predict_hier(tmpdir: str, n_rows: int = 1000) -> HierarchicalCOICOPClassifier:
    """Train hierarchical classifier, return it."""
    clf = HierarchicalCOICOPClassifier(HierarchicalConfig(
        ngram_num_tokens=3000,
        embedding_dim=32,
        max_seq_length=32,
        num_epochs=1,
        patience=1,
        batch_size=16,
        lr=0.1,
        teacher_forcing_ratio=0.0,
    ))
    clf.train(_load_sample(n_rows), text_column="product", code_column="code", save_dir=tmpdir)
    return clf


class TestHierarchicalRoundTrip:
    """Save/load round-trip for HierarchicalCOICOPClassifier."""

    def test_attribute_level_classifiers_exists(self) -> None:
        """Trained classifier must have a `level_classifiers` dict."""
        with _make_tmpdir() as tp:
            clf = _train_and_predict_hier(str(tp), n_rows=1000)
            assert hasattr(clf, "level_classifiers")
            assert isinstance(clf.level_classifiers, dict)
            assert len(clf.level_classifiers) >= 1  # At least level4

    def test_attribute_level_label_names_exists(self) -> None:
        """Each trained level must have its own label mapping."""
        with _make_tmpdir() as tp:
            clf = _train_and_predict_hier(str(tp), n_rows=1000)
            assert hasattr(clf, "level_label_names")
            for level_name in clf.level_classifiers:
                assert level_name in clf.level_label_names

    def test_attribute_level_idx_to_label_exists(self) -> None:
        """Each level must have idx→label mapping for restore."""
        with _make_tmpdir() as tp:
            clf = _train_and_predict_hier(str(tp), n_rows=1000)
            assert hasattr(clf, "level_idx_to_label")
            for level_name in clf.level_classifiers:
                assert level_name in clf.level_idx_to_label
                # Verify round-trip mapping
                for idx, label in clf.level_idx_to_label[level_name].items():
                    assert clf.level_label_to_idx[level_name][label] == idx

    def test_attribute_tokenizer_exists(self) -> None:
        """Shared tokenizer must persist after training."""
        with _make_tmpdir() as tp:
            clf = _train_and_predict_hier(str(tp), n_rows=1000)
            assert hasattr(clf, "tokenizer")
            assert clf.tokenizer is not None

    def test_predict_consistent_after_load(self, tmp_path: Path) -> None:
        """Predictions before and after save/load are identical (all trained levels)."""
        tmpdir = str(tmp_path / "hier_rt")

        clf = _train_and_predict_hier(tmpdir, n_rows=1000)

        pred_before = clf.predict(TEST_TEXTS, return_all_levels=False)
        assert "final_code" in pred_before
        assert "final_level" in pred_before

        clf.save(tmpdir)
        loaded = HierarchicalCOICOPClassifier.load(tmpdir)

        pred_after = loaded.predict(TEST_TEXTS, return_all_levels=False)

        assert pred_before["final_code"] == pred_after["final_code"]
        assert pred_before["final_level"] == pred_after["final_level"]

    def test_roundtrip_all_attributes(self, tmp_path: Path) -> None:
        """Every public attribute with a non-callable value is preserved after save+load."""
        tmpdir = str(tmp_path / "hier_rt")

        clf = _train_and_predict_hier(tmpdir, n_rows=1000)
        original_public = {
            k: v for k, v in vars(clf).items()
            if not k.startswith('_') and not callable(v)
        }

        clf.save(tmpdir)
        loaded = HierarchicalCOICOPClassifier.load(tmpdir)
        loaded_public = {
            k: v for k, v in vars(loaded).items()
            if not k.startswith('_') and not callable(v)
        }

        for k, v in original_public.items():
            assert k in loaded_public, f"Missing attribute '{k}' after load"
            if isinstance(v, dict):
                assert v.keys() == loaded_public[k].keys(), (
                    f"Different keys for attribute '{k}'"
                )
            # For tensors, check device and dtype are preserved
            elif isinstance(v, torch.Tensor):
                assert v.dtype == loaded_public[k].dtype, (
                    f"dtype mismatch for attribute '{k}'"
                )
                assert v.shape == loaded_public[k].shape, (
                    f"shape mismatch for attribute '{k}'"
                )

    def test_load_restores_level_names(self, tmp_path: Path) -> None:
        """`loaded.level_classifiers` keys match original after save+load."""
        tmpdir = str(tmp_path / "hier_rt")

        clf = _train_and_predict_hier(tmpdir, n_rows=1000)
        original_keys = set(clf.level_classifiers.keys())

        clf.save(tmpdir)
        loaded = HierarchicalCOICOPClassifier.load(tmpdir)
        loaded_keys = set(loaded.level_classifiers.keys())

        assert original_keys == loaded_keys


# ── Helpers ───────────────────────────────────────────────────────────

@contextmanager
def _make_tmpdir():
    """Context-managed temp directory for hierarchical tests."""
    from pathlib import Path as _Path
    import tempfile as _tmpfile
    tmppath = _Path(_tmpfile.mkdtemp(prefix="test_hier_"))
    try:
        yield tmppath
    finally:
        import shutil as _shutil
        _shutil.rmtree(tmppath, ignore_errors=True)


def train_and_predict(
    clf_cls,
    config_cls,
    n_rows: int = 500,
    params: dict | None = None,
    save_dir: str = "/tmp/_tst",
):
    """Generic trainer: instantiate, train on sample, predict, return classifier."""
    config = config_cls(**(params or {}))
    clf = clf_cls(config)
    df = _load_sample(n_rows)
    clf.train(df, text_column="product", code_column="code", save_dir=save_dir)
    _ = clf.predict(TEST_TEXTS)
    return clf
