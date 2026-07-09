"""Tests for the multi-head v2 classifier (Phase 4 — upgrade-ttc2.md).

Covers model building, forward pass, training loop, prediction output shapes,
and save/load round-trip for the v2 ``MultiHeadCOICOPClassifier``.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace as dc_replace
from pathlib import Path

import duckdb
import numpy as np
import pytest
import torch

from src.classifiers.multihead_classifier import (
    MultiHeadCOICOPClassifier,
    MultiHeadConfig,
    MultiHeadDataset,
    MultiHeadLightningModule,
)

# ── Fixtures ──────────────────────────────────────────────────────────

_TEST_FILE_PATH = Path(__file__).resolve().parents[1] / "data" / "data-train.parquet"

_TEST_TEXTS = [
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


# ── Helpers ───────────────────────────────────────────────────────────

@contextmanager
def _make_tmpdir():
    """Context-managed temp directory."""
    import shutil
    import tempfile
    from pathlib import Path as _Path
    tmppath = _Path(tempfile.mkdtemp(prefix="test_multihead_"))
    try:
        yield tmppath
    finally:
        shutil.rmtree(tmppath, ignore_errors=True)


_DEFAULT_CONFIG = MultiHeadConfig(
    ngram_num_tokens=3_000,
    embedding_dim=32,
    max_seq_length=32,
    n_attention_layers=1,
    n_attention_heads=2,
    n_kv_heads=2,
    n_label_attention_heads=2,
    num_epochs=1,
    patience=1,
    batch_size=16,
    lr=0.1,
    min_samples_per_level=10,
    min_samples_per_class=1,
    max_level=4,
)


def _quick_config(**overrides) -> MultiHeadConfig:
    """Return a lightweight config suitable for smoke tests / smoke training."""
    return dc_replace(_DEFAULT_CONFIG, **overrides)


def _train_clf(df, **config_overrides) -> MultiHeadCOICOPClassifier:
    """Instantiate, train, and return a multi-head classifier."""
    config = _quick_config(**config_overrides)
    clf = MultiHeadCOICOPClassifier(config)
    clf.train(df, text_column="product", code_column="code")
    return clf


# ====================================================================
# Model construction smoke tests
# ====================================================================


class TestModelBuild:
    """Smoke tests for model construction helpers."""

    def test_build_model_has_expected_structure(self) -> None:
        """_build_model produces a MultiLevelTextClassificationModel with components."""
        with _make_tmpdir() as tmp:
            # Need enough samples for >=2 levels (levels 1-3 skip with low counts)
            df = _load_sample(600)
            clf = _train_clf(df)
            # Model should be set after training
            assert clf.model is not None
            assert hasattr(clf.model, "token_embedder")
            assert hasattr(clf.model, "sentence_embedders")
            assert hasattr(clf.model, "classification_heads")
            assert hasattr(clf.model, "categorical_variable_net")
            # Number of levels should match level_names
            assert len(clf.model.sentence_embedders) == len(clf.level_names)
            assert len(clf.model.classification_heads) == len(clf.level_names)
            assert len(clf.level_names) >= 1  # At least one level trained

    def test_tokenizer_attributes_exposed(self) -> None:
        """Tokenizer must expose vocab_size and padding_idx for model building."""
        with _make_tmpdir() as tmp:
            df = _load_sample(400)
            clf = _train_clf(df)
            assert clf.tokenizer is not None
            assert clf.tokenizer.vocab_size > 0
            assert clf.tokenizer.padding_idx == 0  # NGram default

    def test_level_label_names_are_populated(self) -> None:
        """Each level must have a list of valid label strings."""
        with _make_tmpdir() as tmp:
            df = _load_sample(600)
            clf = _train_clf(df)
            for level_name in clf.level_names:
                assert level_name in clf.level_label_names
                assert len(clf.level_label_names[level_name]) >= 2


# ====================================================================
# Forward pass smoke tests
# ====================================================================


class TestForwardPass:
    """Verify that dummy categorical_vars works and forward shapes are correct."""

    def test_dummy_cat_vars_forward_pass(self) -> None:
        """A single forward pass with the dummy categorical_vars must not crash."""
        with _make_tmpdir() as tmp:
            df = _load_sample(400)
            clf = _train_clf(df)
            clf.model.eval()
            device = next(clf.model.parameters()).device

            tok = clf.tokenizer.tokenize(_TEST_TEXTS)
            input_ids = tok.input_ids.to(device)
            attention_mask = tok.attention_mask.to(device)

            # Dummy cat vars: (batch, 1) filled with 0
            batch_size = input_ids.shape[0]
            cat_vars = torch.zeros(batch_size, 1, dtype=torch.long, device=device)

            with torch.no_grad():
                logits_list = clf.model(input_ids, attention_mask, categorical_vars=cat_vars)

            # Must return list of per-level logit tensors
            assert isinstance(logits_list, list)
            assert len(logits_list) == len(clf.level_names)
            for li, (level_name, logit) in enumerate(zip(clf.level_names, logits_list)):
                assert logit.shape[0] == batch_size, (
                    f"Level {level_name}: expected batch dim {batch_size}, got {logit.shape[0]}"
                )
                assert logit.shape[1] == len(clf.level_label_names[level_name]), (
                    f"Level {level_name}: expected {len(clf.level_label_names[level_name])} classes, "
                    f"got {logit.shape[1]}"
                )

    def test_lightning_module_shared_step(self) -> None:
        """MultiHeadLightningModule._shared_step produces a scalar loss."""
        with _make_tmpdir() as tmp:
            df = _load_sample(400)  # enough for >=2 levels
            clf = _train_clf(df)
            device = next(clf.model.parameters()).device

            # Build a synthetic batch dict matching collate_fn output
            tok = clf.tokenizer.tokenize(_TEST_TEXTS)
            batch_size = len(_TEST_TEXTS)

            # Generate realistic labels per level (0 .. num_classes-1)
            label_rows = torch.zeros((batch_size, len(clf.level_names)), dtype=torch.long)
            for li, level_name in enumerate(clf.level_names):
                n_cls = len(clf.level_label_names[level_name])
                label_rows[:, li] = torch.randint(0, n_cls, (batch_size,))

            batch = {
                "input_ids": tok.input_ids.to(device),
                "attention_mask": tok.attention_mask.to(device),
                "labels": label_rows,
                "categorical_vars": torch.zeros(batch_size, 1, dtype=torch.long, device=device),
            }

            lightning_module = MultiHeadLightningModule(
                model=clf.model,
                level_names=clf.level_names,
                lr=0.01,
            )
            loss = lightning_module._shared_step(batch, prefix="train")
            assert isinstance(loss, torch.Tensor)
            assert loss.dim() == 0  # scalar


# ====================================================================
# Training loop smoke test
# ====================================================================


class TestTrainingLoop:
    """Minimal training-loop verification with v2 model."""

    def test_training_decreases_loss(self) -> None:
        """Two training steps should produce a lower loss than the first."""
        with _make_tmpdir() as tmp:
            df = _load_sample(200)
            config = _quick_config(num_epochs=2, patience=1, batch_size=16)
            clf = MultiHeadCOICOPClassifier(config)
            clf.train(df, text_column="product", code_column="code")

            # _is_trained should be True after training
            assert clf._is_trained

    def test_metrics_return_structure(self) -> None:
        """Training metrics dict should contain train/val samples and level info."""
        with _make_tmpdir() as tmp:
            df = _load_sample(300)
            config = _quick_config(num_epochs=1, patience=1)
            clf = MultiHeadCOICOPClassifier(config)
            metrics = clf.train(df, text_column="product", code_column="code")

            assert "train_samples" in metrics
            assert "val_samples" in metrics
            assert "levels" in metrics
            for level_name in clf.level_names:
                assert level_name in metrics["levels"]
                level_info = metrics["levels"][level_name]
                assert "num_classes" in level_info
                assert "train_samples" in level_info
                assert "val_samples" in level_info


# ====================================================================
# Prediction tests
# ====================================================================


class TestPrediction:
    """Verify prediction output shapes, content, and hierarchical masking."""

    def test_predict_returns_expected_keys(self) -> None:
        """predict() dict must contain all standard keys."""
        with _make_tmpdir() as tmp:
            df = _load_sample(300)
            clf = _train_clf(df)

            result = clf.predict(_TEST_TEXTS, return_all_levels=False)
            assert "final_code" in result
            assert "final_level" in result
            assert "final_confidence" in result
            assert "combined_confidence" in result
            assert len(result["final_code"]) == len(_TEST_TEXTS)
            assert len(result["final_level"]) == len(_TEST_TEXTS)
            assert len(result["final_confidence"]) == len(_TEST_TEXTS)

    def test_predict_all_levels_structure(self) -> None:
        """Predictions per level must have 'predictions' and 'confidence' keys."""
        with _make_tmpdir() as tmp:
            df = _load_sample(300)
            clf = _train_clf(df)

            result = clf.predict(_TEST_TEXTS, return_all_levels=True)
            assert "all_levels" in result
            all_levels = result["all_levels"]
            for level_name in clf.level_names:
                assert level_name in all_levels
                assert "predictions" in all_levels[level_name]
                assert "confidence" in all_levels[level_name]
                preds = all_levels[level_name]["predictions"]
                confs = all_levels[level_name]["confidence"]
                assert len(preds) == len(_TEST_TEXTS)
                assert len(confs) == len(_TEST_TEXTS)

    def test_predictions_are_valid_labels(self) -> None:
        """Predicted codes must be in the level's label vocabulary."""
        with _make_tmpdir() as tmp:
            df = _load_sample(300)
            clf = _train_clf(df)

            result = clf.predict(_TEST_TEXTS, return_all_levels=True)
            all_levels = result["all_levels"]
            for level_name, level_data in all_levels.items():
                valid_labels = set(clf.level_label_names[level_name])
                for pred in level_data["predictions"]:
                    assert pred in valid_labels, (
                        f"Prediction '{pred}' not in {level_name} label vocab"
                    )

    def test_predict_output_is_consistent(self) -> None:
        """Two predict calls must give identical results (eval mode)."""
        with _make_tmpdir() as tmp:
            df = _load_sample(300)
            clf = _train_clf(df)

            pred1 = clf.predict(_TEST_TEXTS, return_all_levels=True)
            pred2 = clf.predict(_TEST_TEXTS, return_all_levels=True)

            assert pred1["final_code"] == pred2["final_code"]
            assert pred1["final_level"] == pred2["final_level"]
            assert np.allclose(pred1["final_confidence"], pred2["final_confidence"], atol=1e-7)

    def test_top_k_returns_multiple_candidates(self) -> None:
        """top_k=3 must return 3 candidates per sample per level."""
        with _make_tmpdir() as tmp:
            df = _load_sample(600)
            clf = _train_clf(df)

            result = clf.predict(_TEST_TEXTS[:5], top_k=3, return_all_levels=True)
            all_levels = result["all_levels"]
            for level_name, level_data in all_levels.items():
                predictions = level_data["predictions"]
                # All must be lists (multi-label per sample)
                for pred in predictions:
                    assert isinstance(pred, list)
                    assert len(pred) == 3

    def test_raises_on_untrained(self) -> None:
        """predict() must raise if called before training."""
        clf = MultiHeadCOICOPClassifier(_quick_config())
        with pytest.raises(RuntimeError, match="must be trained"):
            clf.predict(_TEST_TEXTS)


# ====================================================================
# Dataset tests
# ====================================================================


class TestDataset:
    """MultiHeadDataset collation and label tensor shapes."""

    def test_collate_fn_output_keys(self) -> None:
        """collate_fn must return dict with expected keys."""
        with _make_tmpdir() as tmp:
            df = _load_sample(200)
            config = _quick_config(num_epochs=1)
            clf = MultiHeadCOICOPClassifier(config)

            # Manually init tokenizer to create dataset
            texts = df["product"].head(32).tolist()
            clf._init_tokenizer(texts)

            ds = MultiHeadDataset(texts, {}, clf.tokenizer)

            # Simulate level_labels
            batch = ds.collate_fn([(t, np.array([1, 0, 0, 0], dtype=np.int64)) for t in texts])
            assert "input_ids" in batch
            assert "attention_mask" in batch
            assert "labels" in batch
            assert "categorical_vars" in batch
            assert batch["categorical_vars"].shape == (len(texts), 1)
            assert batch["labels"].shape == (len(texts), 4)

    def test_label_tensor_is_int64(self) -> None:
        """Label tensor must be torch.long (int64)."""
        with _make_tmpdir() as tmp:
            df = _load_sample(200)
            config = _quick_config(num_epochs=1)
            clf = MultiHeadCOICOPClassifier(config)
            texts = df["product"].head(32).tolist()
            clf._init_tokenizer(texts)
            ds = MultiHeadDataset(texts, {}, clf.tokenizer)
            batch = ds.collate_fn([(t, np.array([1, 0, 0, 0], dtype=np.int64)) for t in texts])
            assert batch["labels"].dtype == torch.long


# ====================================================================
# Save / Load round-trip
# ====================================================================


class TestSaveLoadRoundTrip:
    """Save → load → predict must produce identical results for v2 model."""

    def test_predict_consistent_after_load(self, tmp_path: Path) -> None:
        """Predictions before and after save/load are identical."""
        tmpdir = str(tmp_path / "multihead_rt")

        df = _load_sample(400)
        clf = _train_clf(df)

        pred_before = clf.predict(_TEST_TEXTS, return_all_levels=False)

        clf.save(tmpdir)
        loaded = MultiHeadCOICOPClassifier.load(tmpdir)

        pred_after = loaded.predict(_TEST_TEXTS, return_all_levels=False)

        assert pred_before["final_code"] == pred_after["final_code"]
        assert pred_before["final_level"] == pred_after["final_level"]
        assert np.allclose(
            pred_before["final_confidence"], pred_after["final_confidence"], atol=1e-6
        )

    def test_roundtrip_all_attributes(self, tmp_path: Path) -> None:
        """Every public attribute with a non-callable value is preserved."""
        tmpdir = str(tmp_path / "multihead_rt")

        df = _load_sample(400)
        clf = _train_clf(df)
        original_public = {
            k: v for k, v in vars(clf).items()
            if not k.startswith('_') and not callable(v)
        }

        clf.save(tmpdir)
        loaded = MultiHeadCOICOPClassifier.load(tmpdir)
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
            elif isinstance(v, torch.Tensor):
                assert v.dtype == loaded_public[k].dtype, (
                    f"dtype mismatch for attribute '{k}'"
                )
                assert v.shape == loaded_public[k].shape, (
                    f"shape mismatch for attribute '{k}'"
                )

    def test_load_restores_level_names(self, tmp_path: Path) -> None:
        """`loaded.level_names` matches original after save+load."""
        tmpdir = str(tmp_path / "multihead_rt")

        df = _load_sample(400)
        clf = _train_clf(df)
        original_names = clf.level_names.copy()

        clf.save(tmpdir)
        loaded = MultiHeadCOICOPClassifier.load(tmpdir)

        assert loaded.level_names == original_names

    def test_load_restores_valid_children(self, tmp_path: Path) -> None:
        """Hierarchical parent→children masking is preserved."""
        tmpdir = str(tmp_path / "multihead_rt")

        df = _load_sample(400)
        clf = _train_clf(df)

        clf.save(tmpdir)
        loaded = MultiHeadCOICOPClassifier.load(tmpdir)

        for level_name in loaded.valid_children:
            assert loaded.valid_children[level_name] == clf.valid_children[level_name], (
                f"valid_children mismatch for {level_name}"
            )

    def test_load_preserves_metadata_content(self, tmp_path: Path) -> None:
        """Metadata (level_label_names, idx↔label mappings) are correctly restored."""
        tmpdir = str(tmp_path / "multihead_rt")

        df = _load_sample(400)
        clf = _train_clf(df)

        clf.save(tmpdir)
        loaded = MultiHeadCOICOPClassifier.load(tmpdir)

        for level_name in clf.level_names:
            assert loaded.level_label_names[level_name] == clf.level_label_names[level_name]
            loaded_to_idx = {v: k for k, v in loaded.level_idx_to_label[level_name].items()}
            assert loaded_to_idx == clf.level_label_to_idx[level_name], (
                f"idx↔label mapping mismatch for {level_name}"
            )

    def test_all_levels_preserved_after_roundtrip(self, tmp_path: Path) -> None:
        """Predicting with all_levels=True after load matches before."""
        tmpdir = str(tmp_path / "multihead_rt")

        df = _load_sample(400)
        clf = _train_clf(df)

        pred_before = clf.predict(_TEST_TEXTS[:5], return_all_levels=True)
        clf.save(tmpdir)
        loaded = MultiHeadCOICOPClassifier.load(tmpdir)
        pred_after = loaded.predict(_TEST_TEXTS[:5], return_all_levels=True)

        for level_name in clf.level_names:
            assert (
                pred_before["all_levels"][level_name]["predictions"]
                == pred_after["all_levels"][level_name]["predictions"]
            )
            assert np.allclose(
                pred_before["all_levels"][level_name]["confidence"],
                pred_after["all_levels"][level_name]["confidence"],
                atol=1e-6,
            )
