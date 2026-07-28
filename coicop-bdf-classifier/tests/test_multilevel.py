"""Tests for the interpretable multi-level COICOP classifier.

Covers both head variants (mean pooling / label attention), both trainable
tokenizers (n-gram / WordPiece), the locally-implemented summed loss, the
absence of parent->child masking, explainability, and save/load fidelity.
"""

from __future__ import annotations

from dataclasses import fields as dc_fields
from dataclasses import replace as dc_replace
from pathlib import Path

import duckdb
import numpy as np
import pytest
import torch

from src.classifiers.multilevel_classifier import (
    MultilevelCOICOPClassifier,
    MultilevelConfig,
    MultilevelDataset,
    MultilevelLightningModule,
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


def _load_sample(n: int = 800):
    """Return a small sample DataFrame from the training parquet."""
    return duckdb.sql(
        f"""
        SELECT product, code
        FROM read_parquet('{_TEST_FILE_PATH}')
        WHERE code IS NOT NULL AND code != '99.0.0.0.0'
        LIMIT {n}
        """
    ).fetchdf()


_DEFAULT_CONFIG = MultilevelConfig(
    embedding_dim=32,
    max_seq_length=32,
    ngram_num_tokens=3_000,
    wordpiece_vocab_size=500,
    n_attention_layers=0,
    n_label_attention_heads=2,
    num_epochs=1,
    patience=1,
    batch_size=16,
    lr=0.05,
    min_samples_per_level=10,
    min_samples_per_class=1,
    max_level=4,
)


def _quick_config(**overrides) -> MultilevelConfig:
    """Return a lightweight config suitable for smoke training."""
    return dc_replace(_DEFAULT_CONFIG, **overrides)


def _train_clf(df=None, **overrides) -> MultilevelCOICOPClassifier:
    """Instantiate, train and return a classifier on a small sample."""
    df = _load_sample() if df is None else df
    clf = MultilevelCOICOPClassifier(_quick_config(**overrides))
    clf.train(df, text_column="product", code_column="code")
    return clf


def _fitted_shell(**overrides) -> MultilevelCOICOPClassifier:
    """A classifier with levels and tokenizer set up but no training run.

    Lets the model-construction tests stay fast by skipping ``fit``.
    """
    clf = MultilevelCOICOPClassifier(_quick_config(**overrides))
    clf.level_names = ["level1", "level2"]
    clf.level_label_names = {
        "level1": ["01", "02", "03"],
        "level2": ["01.1", "01.2", "02.1", "03.1"],
    }
    clf._init_tokenizer(_TEST_TEXTS * 10)
    return clf


# ====================================================================
# Config validation
# ====================================================================


class TestConfig:
    """Config rejects combinations that would fail deep inside the library."""

    def test_rejects_unknown_head_type(self) -> None:
        with pytest.raises(ValueError, match="head_type"):
            MultilevelConfig(head_type="nonsense")

    def test_rejects_unknown_tokenizer_type(self) -> None:
        with pytest.raises(ValueError, match="tokenizer_type"):
            MultilevelConfig(tokenizer_type="nonsense")

    def test_rejects_indivisible_label_attention_heads(self) -> None:
        with pytest.raises(ValueError, match="divisible"):
            MultilevelConfig(
                head_type="label-attention", embedding_dim=32, n_label_attention_heads=5
            )


# ====================================================================
# Model construction
# ====================================================================


class TestModelBuild:
    """The shared encoder and one independent head per level."""

    def test_mean_pooling_has_no_transformer(self) -> None:
        """n_attention_layers=0 is the FastText-style backbone: no blocks."""
        model = _fitted_shell()._build_model()
        assert not hasattr(model.token_embedder, "transformer")

    def test_transformer_path_still_reachable(self) -> None:
        model = _fitted_shell(n_attention_layers=1, n_attention_heads=2, n_kv_heads=2)._build_model()
        assert len(model.token_embedder.transformer["h"]) == 1

    @pytest.mark.parametrize("head_type", ["mean", "label-attention"])
    def test_one_head_per_level(self, head_type: str) -> None:
        clf = _fitted_shell(head_type=head_type)
        model = clf._build_model()
        assert len(model.sentence_embedders) == len(clf.level_names)
        assert len(model.classification_heads) == len(clf.level_names)
        assert model.num_classes == [3, 4]

    def test_no_categorical_net(self) -> None:
        """This architecture uses no categorical features."""
        assert _fitted_shell()._build_model().categorical_variable_net is None


class TestForwardPass:
    """Forward returns per-level logits, and attention only where it exists."""

    @staticmethod
    def _batch(batch_size: int = 3, seq_len: int = 32):
        return (
            torch.randint(1, 100, (batch_size, seq_len)),
            torch.ones(batch_size, seq_len, dtype=torch.long),
        )

    @pytest.mark.parametrize("head_type", ["mean", "label-attention"])
    def test_logits_shapes(self, head_type: str) -> None:
        model = _fitted_shell(head_type=head_type)._build_model()
        logits = model(*self._batch())
        assert [tuple(x.shape) for x in logits] == [(3, 3), (3, 4)]

    def test_mean_pooling_has_no_attention_matrix(self) -> None:
        model = _fitted_shell(head_type="mean")._build_model()
        _, attentions = model(*self._batch(), return_attention=True)
        assert attentions == [None, None]

    def test_label_attention_returns_matrix(self) -> None:
        """(batch, n_head, n_classes, seq_len), recoverable despite the parent
        implementation discarding it."""
        model = _fitted_shell(head_type="label-attention")._build_model()
        _, attentions = model(*self._batch(), return_attention=True)
        assert [tuple(a.shape) for a in attentions] == [(3, 2, 3, 32), (3, 2, 4, 32)]


# ====================================================================
# Loss
# ====================================================================


class TestLoss:
    """The locally-implemented summed cross-entropy.

    Both cases below are ones the library's ``MultiLevelCrossEntropyLoss``
    mishandles: it calls ``output.squeeze()`` (corrupting batch size 1) and
    averages over all levels regardless of ``-100`` masking (yielding NaN when
    a level has no supervision in the batch).
    """

    @staticmethod
    def _module():
        clf = _fitted_shell()
        return MultilevelLightningModule(clf._build_model(), clf.level_names)

    @staticmethod
    def _batch(labels):
        labels = torch.tensor(labels)
        n = labels.shape[0]
        return {
            "input_ids": torch.randint(1, 100, (n, 32)),
            "attention_mask": torch.ones(n, 32, dtype=torch.long),
            "labels": labels,
        }

    def test_batch_size_one(self) -> None:
        loss = self._module()._shared_step(self._batch([[0, 2]]), "train")
        assert loss.ndim == 0 and torch.isfinite(loss)

    def test_fully_masked_level_contributes_zero_not_nan(self) -> None:
        batch = self._batch([[0, -100], [1, -100], [2, -100], [0, -100]])
        loss = self._module()._shared_step(batch, "train")
        assert torch.isfinite(loss)

    def test_all_levels_masked_gives_zero(self) -> None:
        loss = self._module()._shared_step(self._batch([[-100, -100]]), "train")
        assert torch.isfinite(loss) and float(loss) == 0.0

    def test_loss_is_sum_over_levels(self) -> None:
        """L = L_level1 + L_level2, per the reference approach."""
        module = self._module()
        batch = self._batch([[0, 1], [1, 2], [2, 3]])
        total = module._shared_step(batch, "train")

        logits = module.model(batch["input_ids"], batch["attention_mask"])
        expected = sum(
            torch.nn.functional.cross_entropy(
                logits[i], batch["labels"][:, i], ignore_index=-100
            )
            for i in range(2)
        )
        assert float(total) == pytest.approx(float(expected), abs=1e-6)


# ====================================================================
# Dataset
# ====================================================================


class TestDataset:
    def test_collate_emits_no_categorical_vars(self) -> None:
        clf = _fitted_shell()
        ds = MultilevelDataset(
            _TEST_TEXTS,
            {"level1": np.zeros(len(_TEST_TEXTS), dtype=np.int64),
             "level2": np.ones(len(_TEST_TEXTS), dtype=np.int64)},
            clf.tokenizer,
        )
        batch = ds.collate_fn([ds[i] for i in range(4)])
        assert set(batch) == {"input_ids", "attention_mask", "labels"}
        assert batch["labels"].shape == (4, 2)


# ====================================================================
# Training + prediction
# ====================================================================


class TestTraining:
    def test_trains_and_reports_levels(self) -> None:
        clf = _train_clf()
        assert clf._is_trained
        assert len(clf.level_names) >= 1
        for level_name in clf.level_names:
            assert len(clf.level_label_names[level_name]) >= 2

    def test_tokenizer_fitted_on_train_split_only(self) -> None:
        """Validation text must not leak into the vocabulary."""
        clf = _train_clf()
        assert clf.tokenizer is not None


class TestPrediction:
    def test_top1_shapes(self) -> None:
        clf = _train_clf()
        result = clf.predict(_TEST_TEXTS)
        for key in ("final_code", "final_level", "final_confidence",
                    "combined_confidence", "levels_consistent"):
            assert len(result[key]) == len(_TEST_TEXTS)
        assert all(0.0 <= c <= 1.0 for c in result["final_confidence"])

    def test_topk_shapes(self) -> None:
        clf = _train_clf()
        result = clf.predict(_TEST_TEXTS, top_k=3)
        for level_name in clf.level_names:
            level = result["all_levels"][level_name]
            n_classes = len(clf.level_label_names[level_name])
            expected_k = min(3, n_classes)
            assert len(level["predictions"][0]) == expected_k
            # Confidences are sorted descending
            assert level["confidence"][0] == sorted(level["confidence"][0], reverse=True)

    def test_final_code_is_deepest_level(self) -> None:
        clf = _train_clf()
        result = clf.predict(_TEST_TEXTS)
        deepest = clf.level_names[-1]
        assert result["final_level"] == [deepest] * len(_TEST_TEXTS)
        assert result["final_code"] == result["all_levels"][deepest]["predictions"]

    def test_no_hierarchical_masking(self) -> None:
        """Each head's softmax is independent: a level's prediction is NOT
        constrained to be a child of the level above.

        This is the defining difference from the multi-head classifier. The
        assertion is that every class stays reachable — probabilities are a
        plain softmax over the full label set, never zeroed by a parent mask.
        """
        clf = _train_clf()
        probs = clf._level_probs(_TEST_TEXTS)
        for level_name in clf.level_names[1:]:
            level_probs = probs[level_name]
            # A masked implementation would zero out non-children entirely.
            assert (level_probs > 0).all()
            assert np.allclose(level_probs.sum(axis=1), 1.0, atol=1e-5)

    def test_levels_consistent_flags_prefix_violations(self) -> None:
        """levels_consistent reports, rather than suppresses, disagreement."""
        clf = _train_clf()
        result = clf.predict(_TEST_TEXTS)

        for i in range(len(_TEST_TEXTS)):
            codes = [result["all_levels"][lv]["predictions"][i] for lv in clf.level_names]
            expected = all(
                child.startswith(parent) for parent, child in zip(codes, codes[1:])
            )
            assert result["levels_consistent"][i] == expected

    def test_confidence_threshold_stops_early(self) -> None:
        clf = _train_clf()
        strict = clf.predict(_TEST_TEXTS, confidence_threshold=0.99)
        loose = clf.predict(_TEST_TEXTS)
        # A threshold can only shorten the chain, never lengthen it.
        depth = {name: i for i, name in enumerate(clf.level_names)}
        for s, ll in zip(strict["final_level"], loose["final_level"]):
            if s:
                assert depth[s] <= depth[ll]

    def test_predict_before_training_raises(self) -> None:
        with pytest.raises(RuntimeError, match="trained"):
            MultilevelCOICOPClassifier(_quick_config()).predict(_TEST_TEXTS)


# ====================================================================
# Predictor output schema
# ====================================================================


class TestPredictorColumns:
    """The predictor must emit the columns downstream consumers key off."""

    @staticmethod
    def _predictions(tmp_path: Path, top_k: int = 3):
        import pandas as pd

        from src.predict import MultilevelCOICOPPredictor

        clf = _train_clf()
        clf.save(tmp_path / "model")
        predictor = MultilevelCOICOPPredictor(tmp_path / "model")
        df = pd.DataFrame({"product": _TEST_TEXTS})
        return clf, predictor.predict_dataframe(df, text_column="product", top_k=top_k)

    def test_emits_per_level_columns_for_topk_accuracy(self, tmp_path: Path) -> None:
        """src.evaluation.topk_accuracy keys off predicted_level{N}[_top{rank}]."""
        clf, result = self._predictions(tmp_path)
        for level_name in clf.level_names:
            assert f"predicted_{level_name}" in result.columns
            assert f"confidence_{level_name}" in result.columns
            assert f"predicted_{level_name}_top2" in result.columns

    def test_emits_predicted_code_topk(self, tmp_path: Path) -> None:
        """evaluate-report and decide-coicop read predicted_code_top{rank};
        without them top-k accuracy silently collapses to top-1."""
        _, result = self._predictions(tmp_path)
        assert {"predicted_code_top2", "predicted_code_top3"} <= set(result.columns)
        assert {"confidence_top2", "confidence_top3"} <= set(result.columns)

    def test_code_alternatives_are_distinct_and_ranked(self, tmp_path: Path) -> None:
        _, result = self._predictions(tmp_path)
        for _, row in result.iterrows():
            codes = [
                row["predicted_code"],
                row["predicted_code_top2"],
                row["predicted_code_top3"],
            ]
            assert len(set(codes)) == 3
            assert row["confidence"] >= row["confidence_top2"] >= row["confidence_top3"]

    def test_emits_levels_consistent(self, tmp_path: Path) -> None:
        _, result = self._predictions(tmp_path)
        assert "levels_consistent" in result.columns
        assert result["levels_consistent"].dtype == bool


# ====================================================================
# Explainability
# ====================================================================


class TestExplain:
    def test_mean_pooling_attribution_is_exact(self) -> None:
        """Mean pooling makes the head additive over tokens, so contributions
        sum *exactly* to ``logit - bias`` — the completeness property that
        Integrated Gradients only approximates.

        Completeness holds over all tokens. Word scores alone fall short by the
        tokens that map to no word (the n-gram tokenizer's trailing
        whole-sequence token), so those are added back here.
        """
        clf = _train_clf(head_type="mean")
        texts = _TEST_TEXTS[:4]
        # top_words high enough to retain every word, or the sum is partial
        explanations = clf.explain(texts, top_words=1000)

        tok = clf.tokenizer.tokenize(
            texts, return_word_ids=True, return_offsets_mapping=True
        )
        with torch.no_grad():
            logits = clf.model(tok.input_ids, tok.attention_mask)
            token_embeddings = clf.model.token_embedder(
                tok.input_ids, tok.attention_mask
            )["token_embeddings"]
        n_tokens = tok.attention_mask.sum(dim=1).clamp(min=1)

        def unmapped_positions(i: int) -> list[int]:
            return [
                pos
                for pos, wid in enumerate(tok.word_ids[i])
                if int(tok.attention_mask[i, pos]) == 1
                and (wid is None or (isinstance(wid, float) and np.isnan(wid)))
            ]

        for i in range(len(texts)):
            for li, level_name in enumerate(clf.level_names):
                predicted = int(logits[li][i].argmax())
                head = clf.model.classification_heads[li].net
                expected = float(logits[li][i][predicted]) - float(head.bias[predicted])

                scores = (token_embeddings[i] @ head.weight[predicted]) / n_tokens[i]
                residual = float(sum(scores[pos] for pos in unmapped_positions(i)))
                actual = sum(score for _, score in explanations[i][level_name])

                assert actual + residual == pytest.approx(expected, abs=1e-4)

    def test_label_attention_explains_every_level(self) -> None:
        clf = _train_clf(head_type="label-attention")
        explanations = clf.explain(_TEST_TEXTS[:4], top_words=5)
        assert len(explanations) == 4
        for per_level in explanations:
            assert set(per_level) == set(clf.level_names)
            for words in per_level.values():
                assert len(words) <= 5
                assert all(isinstance(w, str) and isinstance(s, float) for w, s in words)

    def test_words_come_from_the_input_text(self) -> None:
        clf = _train_clf()
        text = "Riz blanc long grain"
        for words in clf.explain([text], top_words=10)[0].values():
            for word, _ in words:
                assert word.lower() in text.lower()

    def test_sorted_by_absolute_score(self) -> None:
        clf = _train_clf()
        for words in clf.explain(_TEST_TEXTS[:2], top_words=10)[0].values():
            scores = [abs(s) for _, s in words]
            assert scores == sorted(scores, reverse=True)

    def test_explain_before_training_raises(self) -> None:
        with pytest.raises(RuntimeError, match="trained"):
            MultilevelCOICOPClassifier(_quick_config()).explain(_TEST_TEXTS)


# ====================================================================
# Save / load
# ====================================================================


class TestSaveLoad:
    @pytest.mark.parametrize("head_type", ["mean", "label-attention"])
    def test_predictions_survive_roundtrip(self, head_type: str, tmp_path: Path) -> None:
        clf = _train_clf(head_type=head_type)
        before = clf.predict(_TEST_TEXTS, top_k=3)

        clf.save(tmp_path / "model")
        after = MultilevelCOICOPClassifier.load(tmp_path / "model").predict(
            _TEST_TEXTS, top_k=3
        )

        assert after["final_code"] == before["final_code"]
        assert after["levels_consistent"] == before["levels_consistent"]
        assert np.allclose(
            after["final_confidence"], before["final_confidence"], atol=1e-6
        )

    def test_wordpiece_tokenizer_survives_roundtrip(self, tmp_path: Path) -> None:
        clf = _train_clf(tokenizer_type="wordpiece")
        before = clf.predict(_TEST_TEXTS)

        clf.save(tmp_path / "model")
        after = MultilevelCOICOPClassifier.load(tmp_path / "model").predict(_TEST_TEXTS)

        assert after["final_code"] == before["final_code"]

    def test_every_config_field_persists(self, tmp_path: Path) -> None:
        """The sibling classifiers drop fields from their metadata, so config
        silently resets on load. Guard against that here."""
        clf = _train_clf(head_type="label-attention", num_workers=0, pin_memory=False)
        clf.save(tmp_path / "model")
        loaded = MultilevelCOICOPClassifier.load(tmp_path / "model")

        for f in dc_fields(MultilevelConfig):
            assert getattr(loaded.config, f.name) == getattr(clf.config, f.name), (
                f"config field {f.name!r} did not survive the round trip"
            )

    def test_label_mappings_persist(self, tmp_path: Path) -> None:
        clf = _train_clf()
        clf.save(tmp_path / "model")
        loaded = MultilevelCOICOPClassifier.load(tmp_path / "model")

        assert loaded.level_names == clf.level_names
        assert loaded.level_label_names == clf.level_label_names
        assert loaded.level_idx_to_label == clf.level_idx_to_label
        # Keys must come back as ints, not the strings they were written as
        for mapping in loaded.level_idx_to_label.values():
            assert all(isinstance(k, int) for k in mapping)

    def test_explain_works_after_load(self, tmp_path: Path) -> None:
        clf = _train_clf(head_type="label-attention")
        clf.save(tmp_path / "model")
        loaded = MultilevelCOICOPClassifier.load(tmp_path / "model")
        explanations = loaded.explain(_TEST_TEXTS[:2], top_words=5)
        assert set(explanations[0]) == set(clf.level_names)
