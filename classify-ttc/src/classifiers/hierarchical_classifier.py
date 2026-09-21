"""Hierarchical multi-level COICOP classifier with n-gram support.

This module implements a 5-level hierarchical classifier where:
- Each level has its own classifier
- Level N receives parent predictions from level N-1 as categorical features
- Uses NGramTokenizer (character n-grams) for text representation
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from torchTextClassifiers import ModelConfig, TrainingConfig, torchTextClassifiers
from torchTextClassifiers.tokenizers import NGramTokenizer

if TYPE_CHECKING:
    import pandas as pd

from ..preprocessing.data_preparation import COICOP_LEVELS

logger = logging.getLogger(__name__)


@dataclass
class HierarchicalConfig:
    """Configuration for hierarchical COICOP classifier."""

    # N-gram tokenizer settings
    ngram_min_n: int = 3
    ngram_max_n: int = 6
    ngram_num_tokens: int = 100000
    ngram_min_count: int = 1  # Minimum count for n-gram to be included
    ngram_len_word_ngrams: int = 1  # Length of word n-grams (1 = unigrams)

    # Tokenizer override (None = use NGram, else HuggingFace pretrained name)
    tokenizer_name: str | None = None

    # Model settings
    embedding_dim: int = 128
    max_seq_length: int = 64

    # Training settings
    batch_size: int = 32
    lr: float = 2e-5
    num_epochs: int = 20
    patience: int = 5

    # Hierarchical settings
    max_level: int = 5  # Number of COICOP levels to train/predict (1-5)
    min_samples_per_level: int = 50
    min_samples_per_class: int = 2
    use_parent_features: bool = True
    parent_embedding_dim: int = 32
    confidence_buckets: int = 10

    # Teacher forcing ratio (proportion using ground truth during training)
    teacher_forcing_ratio: float = 0.8

    # DataLoader settings
    num_workers: int = 0  # 0 is safest on Windows (spawn); increase on Linux
    pin_memory: bool = True  # Pinned memory for faster CPU→GPU transfer

    # Batch size for inference (larger than training since no gradients stored)
    predict_batch_size: int = 512


class HierarchicalCOICOPClassifier:
    """5-level hierarchical classifier with parent prediction features.

    Architecture:
    - Level 1: Text-only classification (13 classes: 01-13)
    - Levels 2-5: Text + parent_code embedding + confidence bucket

    The parent code and confidence are passed as categorical features
    using the CategoricalVariableNet from torchTextClassifiers.
    """

    def __init__(self, config: HierarchicalConfig | None = None):
        """Initialize the hierarchical classifier.

        Args:
            config: Configuration settings. Uses defaults if None.
        """
        self.config = config or HierarchicalConfig()

        # Level classifiers: {level_name: torchTextClassifiers}
        self.level_classifiers: dict[str, torchTextClassifiers] = {}

        # Shared tokenizer (trained once on full corpus)
        self.tokenizer: NGramTokenizer | None = None

        # Label mappings per level
        self.level_label_names: dict[str, list[str]] = {}
        self.level_label_to_idx: dict[str, dict[str, int]] = {}
        self.level_idx_to_label: dict[str, dict[int, str]] = {}

        # Parent code mappings for categorical features
        # Maps level_name -> {parent_code: index}
        self.parent_code_to_idx: dict[str, dict[str, int]] = {}

        self._is_trained = False

    def _batched_predict(self, classifier, X, top_k=1):
        """Predict in batches to avoid OOM on large datasets."""
        batch_size = self.config.predict_batch_size
        all_preds = []
        all_confs = []
        for start in range(0, len(X), batch_size):
            batch = X[start:start + batch_size]
            result = classifier.predict(batch, top_k=top_k)
            all_preds.append(result["prediction"].numpy())
            all_confs.append(result["confidence"].numpy())
        return {
            "prediction": np.concatenate(all_preds),
            "confidence": np.concatenate(all_confs),
        }

    def _init_tokenizer(self, texts: list[str]) -> None:
        """Initialize tokenizer (HuggingFace pretrained or NGram).

        Args:
            texts: All training texts for vocabulary building (used by NGram only).
        """
        if self.config.tokenizer_name is not None:
            from torchTextClassifiers.tokenizers import HuggingFaceTokenizer

            logger.info(f"Loading HuggingFace tokenizer: {self.config.tokenizer_name}...")
            self.tokenizer = HuggingFaceTokenizer.load_from_pretrained(
                self.config.tokenizer_name,
                output_dim=self.config.max_seq_length,
            )
        else:
            logger.info(
                f"Training NGramTokenizer (n={self.config.ngram_min_n}-{self.config.ngram_max_n}, "
                f"vocab_size={self.config.ngram_num_tokens})..."
            )
            self.tokenizer = NGramTokenizer(
                min_count=self.config.ngram_min_count,
                min_n=self.config.ngram_min_n,
                max_n=self.config.ngram_max_n,
                num_tokens=self.config.ngram_num_tokens,
                len_word_ngrams=self.config.ngram_len_word_ngrams,
                training_text=texts,
                output_dim=self.config.max_seq_length,
            )
        logger.info("Tokenizer ready.")

    def _discretize_confidence(self, confidence: np.ndarray) -> np.ndarray:
        """Convert continuous confidence to discrete buckets.

        Args:
            confidence: Array of confidence values in [0, 1].

        Returns:
            Array of bucket indices in [0, num_buckets-1].
        """
        buckets = np.clip(
            (confidence * self.config.confidence_buckets).astype(int),
            0,
            self.config.confidence_buckets - 1,
        )
        return buckets

    def _extract_parent_code(self, code: str) -> str | None:
        """Extract parent code from a COICOP code.

        Args:
            code: Full COICOP code (e.g., "01.1.2.3")

        Returns:
            Parent code or None for level 1.
        """
        parts = code.split(".")
        if len(parts) <= 1:
            return None
        return ".".join(parts[:-1])

    def _get_level_codes(self, df: pd.DataFrame, level_name: str) -> list[str]:
        """Get unique codes for a specific level.

        Args:
            df: DataFrame with level columns.
            level_name: Column name (e.g., 'level2').

        Returns:
            Sorted list of unique codes.
        """
        codes = df[level_name].dropna().unique().tolist()
        return sorted(codes)

    def _create_level_classifier(
        self,
        level_name: str,
        num_classes: int,
        parent_vocab_size: int | None = None,
    ) -> torchTextClassifiers:
        """Create a classifier for a specific level.

        Args:
            level_name: Level name for logging.
            num_classes: Number of output classes.
            parent_vocab_size: Size of parent code vocabulary (None for level 1).

        Returns:
            Configured torchTextClassifiers instance.
        """
        if parent_vocab_size is not None and self.config.use_parent_features:
            # Two categorical variables: parent_code and confidence_bucket
            categorical_vocabulary_sizes = [
                parent_vocab_size,
                self.config.confidence_buckets,
            ]
            categorical_embedding_dims = [
                self.config.parent_embedding_dim,
                8,  # Small embedding for confidence buckets
            ]
            logger.info(
                f"  Creating classifier with categorical features: "
                f"parent_vocab={parent_vocab_size}, conf_buckets={self.config.confidence_buckets}"
            )
        else:
            categorical_vocabulary_sizes = None
            categorical_embedding_dims = None
            logger.info("  Creating classifier without categorical features")

        model_config = ModelConfig(
            embedding_dim=self.config.embedding_dim,
            num_classes=num_classes,
            categorical_vocabulary_sizes=categorical_vocabulary_sizes,
            categorical_embedding_dims=categorical_embedding_dims,
        )

        return torchTextClassifiers(
            tokenizer=self.tokenizer,
            model_config=model_config,
        )

    def _prepare_input_with_features(
        self,
        texts: list[str],
        parent_code_indices: np.ndarray | None,
        confidence_buckets: np.ndarray | None,
    ) -> np.ndarray:
        """Prepare input array with text and categorical features.

        For torchTextClassifiers with categorical features, the input format is:
        X = [[text, cat1_idx, cat2_idx], ...]

        Args:
            texts: List of text strings.
            parent_code_indices: Indices of parent codes (or None).
            confidence_buckets: Discretized confidence buckets (or None).

        Returns:
            Input array suitable for the classifier.
        """
        if parent_code_indices is None or confidence_buckets is None:
            # Level 1: text only
            return np.array(texts)

        # Levels 2-5: text + parent features
        n = len(texts)
        X = np.empty((n, 3), dtype=object)
        X[:, 0] = texts
        X[:, 1] = parent_code_indices
        X[:, 2] = confidence_buckets

        return X

    def _generate_predictions_with_teacher_forcing(
        self,
        classifier: torchTextClassifiers,
        texts: list[str],
        ground_truth_codes: list[str],
        label_to_idx: dict[str, int],
        idx_to_label: dict[int, str],
        parent_code_indices: np.ndarray | None = None,
        confidence_buckets: np.ndarray | None = None,
    ) -> tuple[list[str], np.ndarray]:
        """Generate predictions with teacher forcing for training.

        During training, we use a mix of ground truth and model predictions
        to make the next level robust to cascading errors.

        Args:
            classifier: Trained classifier for this level.
            texts: Input texts.
            ground_truth_codes: True labels for this level.
            label_to_idx: Label to index mapping.
            idx_to_label: Index to label mapping.
            parent_code_indices: Parent code indices (for levels 2-5).
            confidence_buckets: Confidence buckets (for levels 2-5).

        Returns:
            Tuple of (predicted_codes, confidence_scores).
        """
        n = len(texts)

        # Prepare input
        X = self._prepare_input_with_features(texts, parent_code_indices, confidence_buckets)

        # Get model predictions
        result = classifier.predict(X, top_k=1)
        pred_indices = result["prediction"].numpy().flatten()
        pred_confidence = result["confidence"].numpy().flatten()

        # Apply teacher forcing: use ground truth for some samples
        mask = np.random.random(n) < self.config.teacher_forcing_ratio

        # Convert predictions to codes
        predicted_codes = []
        final_confidence = np.zeros(n)

        for i in range(n):
            if mask[i] and ground_truth_codes[i] in label_to_idx:
                # Use ground truth
                predicted_codes.append(ground_truth_codes[i])
                final_confidence[i] = 1.0  # Perfect confidence for ground truth
            else:
                # Use model prediction
                pred_label = idx_to_label[pred_indices[i]]
                predicted_codes.append(pred_label)
                final_confidence[i] = pred_confidence[i]

        return predicted_codes, final_confidence

    def train(
        self,
        df: pd.DataFrame,
        text_column: str = "text",
        code_column: str = "code",
        save_dir: str | None = None,
        mlflow_run_info: dict | None = None,
        resume_from: str | None = None,
        checkpoint_path: str | None = None,
    ) -> dict:
        """Train all level classifiers sequentially.

        Training proceeds L1 → L2 → L3 → L4 → L5, where each level
        receives predictions from the previous level as features.

        Args:
            df: DataFrame with text and COICOP code columns.
            text_column: Name of text column.
            code_column: Name of full COICOP code column.
            save_dir: Directory to save checkpoints during training.
            mlflow_run_info: MLflow run info for logging.
            resume_from: Path to a partial checkpoint to resume from.
            checkpoint_path: Path to save intermediate checkpoints after each level.

        Returns:
            Dictionary with training metrics per level.
        """
        from sklearn.model_selection import train_test_split

        from ..preprocessing.data_preparation import extract_levels

        # Ensure we have a copy with level columns
        df = df.copy()

        # Extract all level columns if not present
        if "level1" not in df.columns:
            levels = df[code_column].apply(extract_levels).apply(pd.Series)
            df = pd.concat([df, levels], axis=1)

        # Resume from a previous checkpoint if provided
        completed_levels: set[str] = set()
        if resume_from is not None:
            resume_path = Path(resume_from)
            if not resume_path.exists():
                raise FileNotFoundError(f"Resume path does not exist: {resume_path}")

            logger.info(f"Resuming training from {resume_path}")
            partial = HierarchicalCOICOPClassifier.load(resume_path)
            completed_levels = set(partial.level_classifiers.keys())
            logger.info(f"  Previously completed levels: {sorted(completed_levels)}")

            # Restore state from partial model
            self.tokenizer = partial.tokenizer
            self.level_classifiers = partial.level_classifiers
            self.level_label_names = partial.level_label_names
            self.level_label_to_idx = partial.level_label_to_idx
            self.level_idx_to_label = partial.level_idx_to_label
            self.parent_code_to_idx = partial.parent_code_to_idx
        else:
            # Get all texts for tokenizer training
            all_texts = df[text_column].tolist()
            self._init_tokenizer(all_texts)

        # Check if all levels already completed
        active_levels = COICOP_LEVELS[:self.config.max_level]
        if completed_levels and all(lv in completed_levels for lv in active_levels):
            logger.warning("All levels already completed — nothing to resume.")
            self._is_trained = True
            return {}

        # Get MLflow run_id for checkpoints
        mlflow_run_id = mlflow_run_info.get("run_id") if mlflow_run_info else None

        # Training metrics
        metrics = {}

        # Track parent predictions for next level (as dicts keyed by DataFrame index)
        parent_predictions: dict[int, str] | None = None
        parent_confidence: dict[int, float] | None = None

        for level_idx, level_name in enumerate(active_levels):
            is_completed = level_name in completed_levels

            logger.info(f"\n{'='*60}")
            if is_completed:
                logger.info(f"Skipping {level_name.upper()} (already trained)")
            else:
                logger.info(f"Training {level_name.upper()} classifier")
            logger.info(f"{'='*60}")

            # Filter to samples that have this level defined
            level_df = df[df[level_name].notna()].copy()

            if len(level_df) < self.config.min_samples_per_level:
                logger.warning(
                    f"Skipping {level_name}: insufficient samples "
                    f"({len(level_df)} < {self.config.min_samples_per_level})"
                )
                continue

            if is_completed:
                # Use loaded mappings — filter to known labels
                known_labels = set(self.level_label_to_idx[level_name].keys())
                level_df = level_df[level_df[level_name].isin(known_labels)].copy()
                texts = level_df[text_column].tolist()
                classifier = self.level_classifiers[level_name]

                # Prepare parent features for inference
                use_parent = level_idx > 0 and self.config.use_parent_features
                if use_parent:
                    parent_level = COICOP_LEVELS[level_idx - 1]
                    gt_parent_codes = level_df[parent_level].tolist()
                    if parent_predictions is not None:
                        parent_code_list = [
                            parent_predictions.get(idx, gt)
                            for gt, idx in zip(gt_parent_codes, level_df.index)
                        ]
                    else:
                        parent_code_list = gt_parent_codes
                    parent_code_indices = np.array([
                        self.parent_code_to_idx[level_name].get(code, 0)
                        for code in parent_code_list
                    ])
                    if parent_confidence is not None:
                        conf_values = np.array([
                            parent_confidence.get(idx, 0.9) for idx in level_df.index
                        ])
                        conf_buckets = self._discretize_confidence(conf_values)
                    else:
                        conf_buckets = self._discretize_confidence(
                            np.ones(len(texts)) * 0.9
                        )
                else:
                    parent_code_indices = None
                    conf_buckets = None

                # Generate predictions for next level
                logger.info(f"  Regenerating predictions for next level...")
                X_full = self._prepare_input_with_features(
                    texts, parent_code_indices, conf_buckets
                )
                result = self._batched_predict(classifier, X_full, top_k=1)
                pred_indices = result["prediction"].flatten()
                pred_conf = result["confidence"].flatten()

                parent_predictions = {}
                parent_confidence = {}
                for i, orig_idx in enumerate(level_df.index):
                    pred_label = self.level_idx_to_label[level_name][pred_indices[i]]
                    parent_predictions[orig_idx] = pred_label
                    parent_confidence[orig_idx] = pred_conf[i]

                continue

            # Get unique labels and build mappings
            label_names = self._get_level_codes(level_df, level_name)

            # Filter classes with enough samples
            label_counts = level_df[level_name].value_counts()
            valid_labels = label_counts[
                label_counts >= self.config.min_samples_per_class
            ].index.tolist()

            if len(valid_labels) < 2:
                logger.warning(f"Skipping {level_name}: fewer than 2 valid classes")
                continue

            # Filter to valid labels
            level_df = level_df[level_df[level_name].isin(valid_labels)].copy()
            label_names = sorted(valid_labels)

            self.level_label_names[level_name] = label_names
            self.level_label_to_idx[level_name] = {
                label: idx for idx, label in enumerate(label_names)
            }
            self.level_idx_to_label[level_name] = {
                idx: label for idx, label in enumerate(label_names)
            }

            num_classes = len(label_names)
            logger.info(f"  Classes: {num_classes}, Samples: {len(level_df)}")

            # Determine parent vocabulary for categorical features
            use_parent = level_idx > 0 and self.config.use_parent_features

            if use_parent:
                # Get parent level name and build parent code vocabulary
                parent_level = COICOP_LEVELS[level_idx - 1]

                # Get unique parent codes from current level's data
                parent_codes = sorted(
                    level_df[parent_level].dropna().unique().tolist()
                )
                self.parent_code_to_idx[level_name] = {
                    code: idx for idx, code in enumerate(parent_codes)
                }
                parent_vocab_size = len(parent_codes)
                logger.info(f"  Parent vocabulary size: {parent_vocab_size}")
            else:
                parent_vocab_size = None

            # Create classifier
            classifier = self._create_level_classifier(
                level_name, num_classes, parent_vocab_size
            )

            # Prepare training data
            texts = level_df[text_column].tolist()
            labels = level_df[level_name].tolist()
            y = np.array([self.level_label_to_idx[level_name][l] for l in labels])

            # Prepare parent features if needed
            if use_parent:
                parent_level = COICOP_LEVELS[level_idx - 1]

                # Use ground truth parent codes for training (with noise from prev level)
                # For first training pass, use ground truth parents
                gt_parent_codes = level_df[parent_level].tolist()

                if parent_predictions is not None:
                    # Mix with previous level's predictions (teacher forcing at parent level)
                    # Use stored predictions from last level
                    parent_code_list = []
                    for i, idx in enumerate(level_df.index):
                        if idx in parent_predictions and np.random.random() < (1 - self.config.teacher_forcing_ratio):
                            parent_code_list.append(parent_predictions[idx])
                        else:
                            parent_code_list.append(gt_parent_codes[i])
                else:
                    parent_code_list = gt_parent_codes

                # Convert to indices (with fallback for unknown codes)
                parent_code_indices = np.array([
                    self.parent_code_to_idx[level_name].get(code, 0)
                    for code in parent_code_list
                ])

                # Use uniform confidence for training (or from prev level if available)
                if parent_confidence is not None:
                    # Get confidence values for current level's samples
                    conf_values = np.array([
                        parent_confidence.get(idx, 0.9) for idx in level_df.index
                    ])
                    conf_buckets = self._discretize_confidence(conf_values)
                else:
                    # Use high confidence for ground truth
                    conf_buckets = self._discretize_confidence(
                        np.ones(len(texts)) * 0.9
                    )
            else:
                parent_code_indices = None
                conf_buckets = None

            # Prepare input
            X = self._prepare_input_with_features(texts, parent_code_indices, conf_buckets)

            # Train/val split
            indices = np.arange(len(texts))
            train_idx, val_idx = train_test_split(
                indices,
                test_size=0.2,
                random_state=42,
                stratify=y,
            )

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            logger.info(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")

            # Build per-level trainer_params with MLflow logger if available
            level_trainer_params = {"log_every_n_steps": 50}
            if mlflow_run_info:
                from ..tracking.mlflow_utils import make_trainer_params

                level_trainer_params.update(
                    make_trainer_params(**mlflow_run_info, prefix=level_name)
                )

            # Training config
            save_path = None
            if save_dir:
                save_path = str(Path(save_dir) / level_name)

            training_config = TrainingConfig(
                num_epochs=self.config.num_epochs,
                batch_size=self.config.batch_size,
                lr=self.config.lr,
                patience_early_stopping=self.config.patience,
                save_path=save_path or f"hierarchical_{level_name}",
                num_workers=self.config.num_workers,
                dataloader_params={"pin_memory": self.config.pin_memory},
                trainer_params=level_trainer_params,
            )

            # Train
            classifier.train(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                training_config=training_config,
                verbose=True,
            )

            self.level_classifiers[level_name] = classifier

            # Store metrics
            metrics[level_name] = {
                "num_classes": num_classes,
                "train_samples": len(train_idx),
                "val_samples": len(val_idx),
                "labels": label_names,
            }

            # Generate predictions for next level (on FULL dataset)
            logger.info(f"  Generating predictions for next level...")
            X_full = self._prepare_input_with_features(texts, parent_code_indices, conf_buckets)
            result = self._batched_predict(classifier, X_full, top_k=1)

            # Store predictions indexed by original DataFrame index
            pred_indices = result["prediction"].flatten()
            pred_confidence = result["confidence"].flatten()

            # Create dictionaries mapping original index to predictions
            # This handles non-contiguous DataFrame indices
            parent_predictions = {}
            parent_confidence = {}

            for i, orig_idx in enumerate(level_df.index):
                pred_label = self.level_idx_to_label[level_name][pred_indices[i]]
                parent_predictions[orig_idx] = pred_label
                parent_confidence[orig_idx] = pred_confidence[i]

            # Save intermediate checkpoint after each level
            if checkpoint_path:
                self._is_trained = True
                self.save(checkpoint_path, mlflow_run_id=mlflow_run_id)
                logger.info(f"  Checkpoint saved to {checkpoint_path}")

        self._is_trained = True

        logger.info(f"\n{'='*60}")
        logger.info(f"Training complete: {len(self.level_classifiers)} level classifiers")
        logger.info(f"{'='*60}")

        return metrics

    def fine_tune(
        self,
        df: pd.DataFrame,
        text_column: str = "text",
        code_column: str = "code",
        save_dir: str | None = None,
        mlflow_run_info: dict | None = None,
        levels: list[str] | None = None,
        lr: float | None = None,
        num_epochs: int | None = None,
        batch_size: int | None = None,
        patience: int | None = None,
        teacher_forcing_ratio: float | None = None,
        checkpoint_path: str | None = None,
    ) -> dict:
        """Fine-tune a pre-trained hierarchical classifier on new data.

        Reuses the existing tokenizer and label/parent mappings.
        Only updates the weights of the level classifiers.

        Args:
            df: DataFrame with text and COICOP code columns.
            text_column: Name of text column.
            code_column: Name of full COICOP code column.
            save_dir: Directory to save checkpoints during training.
            mlflow_run_info: MLflow run info for logging.
            levels: List of level names to fine-tune (e.g. ["level3", "level4"]).
                    None means all trained levels.
            lr: Learning rate override. Default: config.lr / 10.
            num_epochs: Number of epochs override. Default: 5.
            batch_size: Batch size override. Default: config.batch_size.
            patience: Early stopping patience override. Default: 3.
            teacher_forcing_ratio: Teacher forcing ratio override.
            checkpoint_path: Path to save intermediate checkpoints after each level.

        Returns:
            Dictionary with training metrics per level.
        """
        import pandas as pd
        from sklearn.model_selection import train_test_split

        from ..preprocessing.data_preparation import extract_levels

        if not self._is_trained:
            raise RuntimeError(
                "Classifier must be trained before fine-tuning. "
                "Use train() first or load a pre-trained model."
            )

        # Resolve which levels to fine-tune
        trained_levels = list(self.level_classifiers.keys())
        if levels is not None:
            for lvl in levels:
                if lvl not in trained_levels:
                    raise ValueError(
                        f"Level '{lvl}' not found in trained levels: {trained_levels}"
                    )
            ft_levels = levels
        else:
            ft_levels = trained_levels

        # Resolve hyperparameters with fine-tuning defaults
        ft_lr = lr if lr is not None else self.config.lr / 10
        ft_num_epochs = num_epochs if num_epochs is not None else 5
        ft_batch_size = batch_size if batch_size is not None else self.config.batch_size
        ft_patience = patience if patience is not None else 3
        ft_teacher_forcing_ratio = (
            teacher_forcing_ratio
            if teacher_forcing_ratio is not None
            else self.config.teacher_forcing_ratio
        )

        logger.info(f"Fine-tuning levels: {ft_levels}")
        logger.info(
            f"  lr={ft_lr}, epochs={ft_num_epochs}, batch_size={ft_batch_size}, "
            f"patience={ft_patience}, teacher_forcing={ft_teacher_forcing_ratio}"
        )

        # Prepare data: extract level columns
        df = df.copy()
        if "level1" not in df.columns:
            level_cols = df[code_column].apply(extract_levels).apply(pd.Series)
            df = pd.concat([df, level_cols], axis=1)

        # Training metrics
        metrics = {}

        # Track parent predictions for next level
        parent_predictions: dict[int, str] | None = None
        parent_confidence: dict[int, float] | None = None

        active_levels = COICOP_LEVELS[:self.config.max_level]

        for level_idx, level_name in enumerate(active_levels):
            if level_name not in trained_levels:
                continue

            # Filter to samples that have this level defined
            level_df = df[df[level_name].notna()].copy()

            # Filter to known labels only
            known_labels = set(self.level_label_to_idx[level_name].keys())
            original_count = len(level_df)
            level_df = level_df[level_df[level_name].isin(known_labels)]
            n_dropped = original_count - len(level_df)
            if n_dropped > 0:
                logger.warning(
                    f"  {level_name}: {n_dropped} echantillons ignores (labels inconnus)"
                )

            if len(level_df) < self.config.min_samples_per_level:
                logger.warning(
                    f"Skipping {level_name}: insufficient samples "
                    f"({len(level_df)} < {self.config.min_samples_per_level})"
                )
                continue

            should_train = level_name in ft_levels

            logger.info(f"\n{'='*60}")
            if should_train:
                logger.info(f"Fine-tuning {level_name.upper()} classifier")
            else:
                logger.info(
                    f"Generating predictions for {level_name.upper()} (not fine-tuned)"
                )
            logger.info(f"{'='*60}")

            classifier = self.level_classifiers[level_name]

            # Prepare training data
            texts = level_df[text_column].tolist()
            labels = level_df[level_name].tolist()
            y = np.array([self.level_label_to_idx[level_name][l] for l in labels])

            # Prepare parent features if needed
            use_parent = level_idx > 0 and self.config.use_parent_features
            if use_parent:
                parent_level = COICOP_LEVELS[level_idx - 1]
                gt_parent_codes = level_df[parent_level].tolist()

                if parent_predictions is not None:
                    parent_code_list = []
                    for i, idx in enumerate(level_df.index):
                        if (
                            idx in parent_predictions
                            and np.random.random() < (1 - ft_teacher_forcing_ratio)
                        ):
                            parent_code_list.append(parent_predictions[idx])
                        else:
                            parent_code_list.append(gt_parent_codes[i])
                else:
                    parent_code_list = gt_parent_codes

                parent_code_indices = np.array([
                    self.parent_code_to_idx[level_name].get(code, 0)
                    for code in parent_code_list
                ])

                if parent_confidence is not None:
                    conf_values = np.array([
                        parent_confidence.get(idx, 0.9) for idx in level_df.index
                    ])
                    conf_buckets = self._discretize_confidence(conf_values)
                else:
                    conf_buckets = self._discretize_confidence(
                        np.ones(len(texts)) * 0.9
                    )
            else:
                parent_code_indices = None
                conf_buckets = None

            X = self._prepare_input_with_features(texts, parent_code_indices, conf_buckets)

            if should_train:
                # Train/val split
                indices = np.arange(len(texts))
                train_idx, val_idx = train_test_split(
                    indices,
                    test_size=0.2,
                    random_state=42,
                    stratify=y,
                )

                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                logger.info(
                    f"  Classes: {len(known_labels)}, Samples: {len(level_df)}"
                )
                logger.info(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")

                # Build per-level trainer_params with MLflow logger if available
                level_trainer_params = {"log_every_n_steps": 50}
                if mlflow_run_info:
                    from ..tracking.mlflow_utils import make_trainer_params

                    level_trainer_params.update(
                        make_trainer_params(**mlflow_run_info, prefix=level_name)
                    )

                save_path = None
                if save_dir:
                    save_path = str(Path(save_dir) / level_name)

                training_config = TrainingConfig(
                    num_epochs=ft_num_epochs,
                    batch_size=ft_batch_size,
                    lr=ft_lr,
                    patience_early_stopping=ft_patience,
                    save_path=save_path or f"finetune_{level_name}",
                    num_workers=self.config.num_workers,
                    dataloader_params={"pin_memory": self.config.pin_memory},
                    trainer_params=level_trainer_params,
                )

                classifier.train(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    training_config=training_config,
                    verbose=True,
                )

                metrics[level_name] = {
                    "num_classes": len(known_labels),
                    "train_samples": len(train_idx),
                    "val_samples": len(val_idx),
                    "n_dropped": n_dropped,
                }

            # Generate predictions for next level (on FULL dataset)
            logger.info(f"  Generating predictions for next level...")
            X_full = self._prepare_input_with_features(
                texts, parent_code_indices, conf_buckets
            )
            result = self._batched_predict(classifier, X_full, top_k=1)

            pred_indices = result["prediction"].flatten()
            pred_confidence_arr = result["confidence"].flatten()

            parent_predictions = {}
            parent_confidence = {}
            for i, orig_idx in enumerate(level_df.index):
                pred_label = self.level_idx_to_label[level_name][pred_indices[i]]
                parent_predictions[orig_idx] = pred_label
                parent_confidence[orig_idx] = pred_confidence_arr[i]

            # Save intermediate checkpoint after each level
            if checkpoint_path and should_train:
                self.save(checkpoint_path)
                logger.info(f"  Checkpoint saved to {checkpoint_path}")

        logger.info(f"\n{'='*60}")
        logger.info(f"Fine-tuning complete: {len(metrics)} levels fine-tuned")
        logger.info(f"{'='*60}")

        return metrics

    def predict(
        self,
        texts: list[str],
        return_all_levels: bool = True,
        top_k: int = 1,
        confidence_threshold: float | None = None,
        beam_size: int = 1,
    ) -> dict:
        """Cascade predictions through all 5 levels.

        Args:
            texts: List of text strings to classify.
            return_all_levels: Whether to return predictions at each level.
            top_k: Number of top predictions per level.
            confidence_threshold: Minimum confidence per level; stop at the
                deepest level meeting this threshold. None disables thresholding.

        Returns:
            Dictionary with:
                - final_code: Most specific predicted code
                - final_level: Name of the deepest predicted level
                - final_confidence: Confidence for final prediction
                - combined_confidence: Product of per-level confidences up to final level
                - all_levels: (if return_all_levels) dict of level predictions
        """
        if not self._is_trained:
            raise RuntimeError("Classifier must be trained before prediction.")

        if beam_size > 1:
            return self._beam_predict(texts, return_all_levels, beam_size, confidence_threshold)

        n = len(texts)
        all_levels: dict[str, dict] = {}

        # Track parent info for cascade
        parent_predictions: list[str] | None = None
        parent_confidence: np.ndarray | None = None

        final_code = [""] * n
        final_confidence = np.zeros(n)
        final_level = [""] * n

        active_levels = COICOP_LEVELS[:self.config.max_level]

        for level_idx, level_name in enumerate(active_levels):
            if level_name not in self.level_classifiers:
                continue

            classifier = self.level_classifiers[level_name]

            # Check if this classifier was trained with parent features
            has_parent_features = level_name in self.parent_code_to_idx

            # Use parent features only if:
            # 1. Classifier was trained with parent features
            # 2. We have parent predictions from previous level
            use_parent = has_parent_features and parent_predictions is not None

            # Prepare input
            if use_parent:
                # Convert parent predictions to indices
                parent_code_to_idx = self.parent_code_to_idx.get(level_name, {})
                parent_code_indices = np.array([
                    parent_code_to_idx.get(code, 0) for code in parent_predictions
                ])
                conf_buckets = self._discretize_confidence(parent_confidence)
            elif has_parent_features:
                # Classifier expects parent features but we don't have them
                # Use default values (index 0, high confidence bucket)
                parent_code_indices = np.zeros(n, dtype=int)
                conf_buckets = self._discretize_confidence(np.ones(n) * 0.5)
            else:
                parent_code_indices = None
                conf_buckets = None

            X = self._prepare_input_with_features(texts, parent_code_indices, conf_buckets)

            # Predict
            result = classifier.predict(X, top_k=top_k)
            idx_to_label = self.level_idx_to_label[level_name]

            if top_k > 1:
                # result["prediction"] shape (n, top_k), result["confidence"] shape (n, top_k)
                pred_indices_2d = result["prediction"].numpy()
                pred_confidence_2d = result["confidence"].numpy()

                # Top-1 for cascade
                top1_indices = pred_indices_2d[:, 0].flatten()
                top1_confidence = pred_confidence_2d[:, 0].flatten()
                predictions = [idx_to_label[idx] for idx in top1_indices]

                # All top-K labels
                top_k_labels = [
                    [idx_to_label[pred_indices_2d[i, k]] for k in range(top_k)]
                    for i in range(n)
                ]
                top_k_confs = pred_confidence_2d.tolist()

                all_levels[level_name] = {
                    "predictions": top_k_labels,
                    "confidence": top_k_confs,
                }

                # Update final predictions (top-1)
                for i in range(n):
                    final_code[i] = predictions[i]
                    final_confidence[i] = top1_confidence[i]
                    final_level[i] = level_name

                # Prepare for next level (always top-1)
                parent_predictions = predictions
                parent_confidence = top1_confidence
            else:
                pred_indices = result["prediction"].numpy().flatten()
                pred_confidence = result["confidence"].numpy().flatten()

                predictions = [idx_to_label[idx] for idx in pred_indices]

                all_levels[level_name] = {
                    "predictions": predictions,
                    "confidence": pred_confidence.tolist(),
                }

                for i in range(n):
                    final_code[i] = predictions[i]
                    final_confidence[i] = pred_confidence[i]
                    final_level[i] = level_name

                parent_predictions = predictions
                parent_confidence = pred_confidence

        # Compute combined_confidence and apply threshold
        combined_confidence = [1.0] * n
        for i in range(n):
            product = 1.0
            selected_code = ""
            selected_level = ""
            selected_conf = 0.0
            threshold_applied = False
            for level_name in active_levels:
                if level_name not in all_levels:
                    continue
                level_conf = all_levels[level_name]["confidence"]
                if top_k > 1:
                    c = level_conf[i][0]
                    code = all_levels[level_name]["predictions"][i][0]
                else:
                    c = level_conf[i]
                    code = all_levels[level_name]["predictions"][i]
                if confidence_threshold is not None and c < confidence_threshold:
                    threshold_applied = True
                    break
                product *= c
                selected_code = code
                selected_level = level_name
                selected_conf = c
            combined_confidence[i] = product
            if confidence_threshold is not None and threshold_applied:
                final_code[i] = selected_code
                final_level[i] = selected_level
                final_confidence[i] = selected_conf

        result = {
            "final_code": final_code,
            "final_level": final_level,
            "final_confidence": final_confidence.tolist(),
            "combined_confidence": combined_confidence,
        }

        if return_all_levels:
            result["all_levels"] = all_levels

        return result

    def predict_single(self, text: str) -> dict:
        """Predict COICOP code for a single text with detailed output.

        Args:
            text: Single text string to classify.

        Returns:
            Dictionary with predictions at each level.
        """
        result = self.predict([text], return_all_levels=True)

        output = {
            "text": text,
            "final_code": result["final_code"][0],
            "final_confidence": result["final_confidence"][0],
            "levels": {},
        }

        for level_name, level_data in result["all_levels"].items():
            output["levels"][level_name] = {
                "code": level_data["predictions"][0],
                "confidence": level_data["confidence"][0],
            }

        return output

    def _beam_predict(
        self,
        texts: list[str],
        return_all_levels: bool,
        beam_size: int,
        confidence_threshold: float | None,
    ) -> dict:
        """Beam search cascade: maintain beam_size candidate paths at each level.

        At each level the beam_size current beams are each expanded with the
        top-beam_size predictions, then pruned back to beam_size by joint
        log-probability.  The top beam yields the final prediction.
        """
        import math

        n = len(texts)
        active_levels = COICOP_LEVELS[:self.config.max_level]

        # beams[i] = sorted list of (joint_log_prob, {level: code}, {level: conf})
        beams: list[list[tuple]] = [[(0.0, {}, {})] for _ in range(n)]

        for level_idx, level_name in enumerate(active_levels):
            if level_name not in self.level_classifiers:
                continue
            classifier = self.level_classifiers[level_name]
            idx_to_label = self.level_idx_to_label[level_name]
            has_parent = level_name in self.parent_code_to_idx and level_idx > 0
            prev_level = COICOP_LEVELS[level_idx - 1] if level_idx > 0 else None

            num_current_beams = len(beams[0])
            new_beams: list[list[tuple]] = [[] for _ in range(n)]

            for beam_idx in range(num_current_beams):
                if has_parent and prev_level:
                    parent_codes = [
                        beams[i][beam_idx][1].get(prev_level, "") if beam_idx < len(beams[i]) else ""
                        for i in range(n)
                    ]
                    parent_confs = [
                        beams[i][beam_idx][2].get(prev_level, 0.9) if beam_idx < len(beams[i]) else 0.9
                        for i in range(n)
                    ]
                    parent_code_to_idx = self.parent_code_to_idx[level_name]
                    parent_code_indices = np.array([parent_code_to_idx.get(c, 0) for c in parent_codes])
                    conf_buckets = self._discretize_confidence(np.array(parent_confs))
                else:
                    parent_code_indices = None
                    conf_buckets = None

                X = self._prepare_input_with_features(texts, parent_code_indices, conf_buckets)
                raw = self._batched_predict(classifier, X, top_k=beam_size)

                # Normalize to shape (n, beam_size)
                preds = raw["prediction"].reshape(n, -1)
                confs = raw["confidence"].reshape(n, -1)

                for i in range(n):
                    if beam_idx >= len(beams[i]):
                        continue
                    prev_lp, prev_codes, prev_confs = beams[i][beam_idx]
                    for k in range(preds.shape[1]):
                        code = idx_to_label[int(preds[i, k])]
                        conf = float(confs[i, k])
                        new_lp = prev_lp + math.log(max(conf, 1e-9))
                        new_beams[i].append((
                            new_lp,
                            {**prev_codes, level_name: code},
                            {**prev_confs, level_name: conf},
                        ))

            for i in range(n):
                new_beams[i].sort(key=lambda x: -x[0])
                beams[i] = new_beams[i][:beam_size]

        # Build return dict from top beam per text
        final_code: list[str] = []
        final_level_out: list[str] = []
        final_confidence: list[float] = []
        combined_conf: list[float] = []

        for i in range(n):
            best_lp, best_codes, best_confs = beams[i][0] if beams[i] else (0.0, {}, {})
            deepest = ""
            for lv in reversed(active_levels):
                if lv in best_codes:
                    deepest = lv
                    break
            final_code.append(best_codes.get(deepest, ""))
            final_level_out.append(deepest)
            final_confidence.append(best_confs.get(deepest, 0.0))
            combined_conf.append(math.exp(best_lp))

        result: dict = {
            "final_code": final_code,
            "final_level": final_level_out,
            "final_confidence": final_confidence,
            "combined_confidence": combined_conf,
        }

        if return_all_levels:
            all_levels: dict[str, dict] = {}
            for level_name in active_levels:
                if level_name not in self.level_classifiers:
                    continue
                if beam_size > 1:
                    preds_list = [
                        [beams[i][b][1].get(level_name, "") for b in range(len(beams[i]))]
                        for i in range(n)
                    ]
                    confs_list = [
                        [beams[i][b][2].get(level_name, 0.0) for b in range(len(beams[i]))]
                        for i in range(n)
                    ]
                else:
                    preds_list = [beams[i][0][1].get(level_name, "") if beams[i] else "" for i in range(n)]
                    confs_list = [beams[i][0][2].get(level_name, 0.0) if beams[i] else 0.0 for i in range(n)]
                all_levels[level_name] = {"predictions": preds_list, "confidence": confs_list}
            result["all_levels"] = all_levels

        return result

    def save(self, path: str | Path, mlflow_run_id: str | None = None) -> None:
        """Save the hierarchical classifier.

        Args:
            path: Directory path to save all components.
            mlflow_run_id: Optional MLflow run ID to store in metadata (for resume).
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save tokenizer
        tokenizer_path = path / "tokenizer.pkl"
        with open(tokenizer_path, "wb") as f:
            pickle.dump(self.tokenizer, f)

        # Save each level classifier
        classifiers_path = path / "classifiers"
        classifiers_path.mkdir(exist_ok=True)

        for level_name, classifier in self.level_classifiers.items():
            classifier.save(classifiers_path / level_name)

        # Save metadata
        metadata = {
            "config": {
                "ngram_min_n": self.config.ngram_min_n,
                "ngram_max_n": self.config.ngram_max_n,
                "ngram_num_tokens": self.config.ngram_num_tokens,
                "ngram_min_count": self.config.ngram_min_count,
                "ngram_len_word_ngrams": self.config.ngram_len_word_ngrams,
                "tokenizer_name": self.config.tokenizer_name,
                "embedding_dim": self.config.embedding_dim,
                "max_seq_length": self.config.max_seq_length,
                "batch_size": self.config.batch_size,
                "lr": self.config.lr,
                "num_epochs": self.config.num_epochs,
                "patience": self.config.patience,
                "min_samples_per_level": self.config.min_samples_per_level,
                "min_samples_per_class": self.config.min_samples_per_class,
                "use_parent_features": self.config.use_parent_features,
                "parent_embedding_dim": self.config.parent_embedding_dim,
                "confidence_buckets": self.config.confidence_buckets,
                "teacher_forcing_ratio": self.config.teacher_forcing_ratio,
                "predict_batch_size": self.config.predict_batch_size,
            },
            "level_label_names": self.level_label_names,
            "level_label_to_idx": self.level_label_to_idx,
            "level_idx_to_label": {
                level: {str(k): v for k, v in mapping.items()}
                for level, mapping in self.level_idx_to_label.items()
            },
            "parent_code_to_idx": self.parent_code_to_idx,
            "trained_levels": list(self.level_classifiers.keys()),
            "mlflow_run_id": mlflow_run_id,
        }

        with open(path / "hierarchical_metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

        logger.info(f"Hierarchical classifier saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> HierarchicalCOICOPClassifier:
        """Load a trained hierarchical classifier.

        Args:
            path: Directory path where model was saved.

        Returns:
            Loaded HierarchicalCOICOPClassifier instance.
        """
        path = Path(path)

        # Load metadata
        with open(path / "hierarchical_metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

        # Create config from saved values
        config = HierarchicalConfig(**metadata["config"])

        # Create instance
        instance = cls(config=config)

        # Restore mappings
        instance.level_label_names = metadata["level_label_names"]
        instance.level_label_to_idx = metadata["level_label_to_idx"]
        instance.level_idx_to_label = {
            level: {int(k): v for k, v in mapping.items()}
            for level, mapping in metadata["level_idx_to_label"].items()
        }
        instance.parent_code_to_idx = metadata["parent_code_to_idx"]

        # Load tokenizer
        with open(path / "tokenizer.pkl", "rb") as f:
            instance.tokenizer = pickle.load(f)

        # Load classifiers
        classifiers_path = path / "classifiers"
        for level_name in metadata["trained_levels"]:
            instance.level_classifiers[level_name] = torchTextClassifiers.load(
                classifiers_path / level_name
            )

        instance._is_trained = True

        logger.info(
            f"Hierarchical classifier loaded: {len(instance.level_classifiers)} levels"
        )

        return instance
