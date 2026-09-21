"""Basic flat COICOP classifier with n-gram tokenization.

Single-level classifier that predicts the full COICOP code directly,
without hierarchical decomposition or parent features.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from torchTextClassifiers import ModelConfig, TrainingConfig, torchTextClassifiers
from torchTextClassifiers.tokenizers import NGramTokenizer

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BasicConfig:
    """Configuration for basic flat COICOP classifier."""

    # NGram tokenizer settings
    ngram_min_n: int = 3
    ngram_max_n: int = 6
    ngram_num_tokens: int = 100_000
    ngram_min_count: int = 1
    ngram_len_word_ngrams: int = 1

    # Tokenizer override (None = use NGram, else HuggingFace pretrained name)
    tokenizer_name: str | None = None

    # Model settings
    embedding_dim: int = 128
    max_seq_length: int = 64

    # Training settings
    batch_size: int = 32
    lr: float = 0.1
    num_epochs: int = 20
    patience: int = 5
    min_samples_per_class: int = 2


class BasicCOICOPClassifier:
    """Flat single-level classifier that predicts full COICOP codes directly.

    Unlike the hierarchical classifier, this uses a single torchTextClassifiers
    instance with no levels, no parent features, and no teacher forcing.
    """

    def __init__(self, config: BasicConfig | None = None):
        self.config = config or BasicConfig()
        self.classifier: torchTextClassifiers | None = None
        self.tokenizer: NGramTokenizer | None = None  # or HuggingFaceTokenizer
        self.label_names: list[str] = []
        self.label_to_idx: dict[str, int] = {}
        self.idx_to_label: dict[int, str] = {}
        self._is_trained = False

    def _init_tokenizer(self, texts: list[str]) -> None:
        """Initialize tokenizer (HuggingFace pretrained or NGram)."""
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

    def train(
        self,
        df: pd.DataFrame,
        text_column: str = "product",
        code_column: str = "code8",
        save_dir: str | None = None,
        trainer_params: dict | None = None,
    ) -> dict:
        """Train the flat classifier.

        Args:
            df: DataFrame with text and code columns (already preprocessed).
            text_column: Name of text column.
            code_column: Name of COICOP code column.
            save_dir: Directory to save checkpoints during training.
            trainer_params: Optional trainer params (e.g. MLflow logger).

        Returns:
            Dictionary with training metrics.
        """
        from sklearn.model_selection import train_test_split

        df = df.copy()

        # Filter classes with too few samples
        label_counts = df[code_column].value_counts()
        valid_labels = label_counts[
            label_counts >= self.config.min_samples_per_class
        ].index.tolist()

        n_dropped = len(df) - df[code_column].isin(valid_labels).sum()
        if n_dropped > 0:
            logger.warning(
                f"Dropping {n_dropped} samples from classes with < {self.config.min_samples_per_class} samples"
            )
        df = df[df[code_column].isin(valid_labels)].copy()

        # Build label mappings
        self.label_names = sorted(valid_labels)
        self.label_to_idx = {label: idx for idx, label in enumerate(self.label_names)}
        self.idx_to_label = {idx: label for idx, label in enumerate(self.label_names)}

        num_classes = len(self.label_names)
        logger.info(f"Classes: {num_classes}, Samples: {len(df)}")

        # Init tokenizer
        texts = df[text_column].tolist()
        self._init_tokenizer(texts)

        # Prepare data
        X = np.array(texts)
        y = np.array([self.label_to_idx[label] for label in df[code_column]])

        # Stratified train/val split
        indices = np.arange(len(texts))
        train_idx, val_idx = train_test_split(
            indices, test_size=0.2, random_state=42, stratify=y,
        )
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        logger.info(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

        # Create model
        model_config = ModelConfig(
            embedding_dim=self.config.embedding_dim,
            num_classes=num_classes,
        )
        self.classifier = torchTextClassifiers(
            tokenizer=self.tokenizer,
            model_config=model_config,
        )

        # Training config
        # Ensure log_every_n_steps is not 1 (library default) to avoid
        # excessive MLflow HTTP calls when logging is enabled
        merged_trainer_params = {"log_every_n_steps": 50}
        if trainer_params:
            merged_trainer_params.update(trainer_params)

        training_config = TrainingConfig(
            num_epochs=self.config.num_epochs,
            batch_size=self.config.batch_size,
            lr=self.config.lr,
            patience_early_stopping=self.config.patience,
            save_path=save_dir or "basic_model",
            trainer_params=merged_trainer_params,
        )

        # Train
        self.classifier.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            training_config=training_config,
            verbose=True,
        )

        self._is_trained = True

        metrics = {
            "num_classes": num_classes,
            "train_samples": len(train_idx),
            "val_samples": len(val_idx),
            "total_samples": len(df),
            "dropped_samples": n_dropped,
        }

        logger.info(f"Training complete: {num_classes} classes")
        return metrics

    def fine_tune(
        self,
        df: pd.DataFrame,
        text_column: str = "product",
        code_column: str = "code8",
        save_dir: str | None = None,
        trainer_params: dict | None = None,
        lr: float | None = None,
        num_epochs: int | None = None,
        batch_size: int | None = None,
        patience: int | None = None,
    ) -> dict:
        """Fine-tune a trained basic classifier on new data.

        Only updates weights for labels already known to the model.

        Args:
            df: DataFrame with text and code columns (already preprocessed).
            text_column: Name of text column.
            code_column: Name of COICOP code column.
            save_dir: Directory to save checkpoints during training.
            trainer_params: Optional trainer params (e.g. MLflow logger).
            lr: Learning rate override (default: original lr / 10).
            num_epochs: Epochs override (default: 5).
            batch_size: Batch size override (default: same as original).
            patience: Early stopping patience override (default: 3).

        Returns:
            Dictionary with fine-tuning metrics.
        """
        from sklearn.model_selection import train_test_split

        if not self._is_trained:
            raise RuntimeError("Classifier must be trained before fine-tuning.")

        # Filter to known labels only
        known_labels = set(self.label_to_idx)
        mask = df[code_column].isin(known_labels)
        n_dropped = len(df) - mask.sum()
        if n_dropped > 0:
            logger.warning(f"Dropping {n_dropped} samples with unknown labels")
        df = df[mask].copy()

        if len(df) == 0:
            raise ValueError("No samples with known labels remain after filtering.")

        # Resolve fine-tuning defaults
        ft_lr = lr or self.config.lr / 10
        ft_epochs = num_epochs or 5
        ft_batch_size = batch_size or self.config.batch_size
        ft_patience = patience or 3

        logger.info(
            f"Fine-tuning: {len(df)} samples, lr={ft_lr}, "
            f"epochs={ft_epochs}, batch_size={ft_batch_size}"
        )

        # Prepare data
        texts = df[text_column].tolist()
        X = np.array(texts)
        y = np.array([self.label_to_idx[label] for label in df[code_column]])

        # Stratified train/val split
        indices = np.arange(len(texts))
        train_idx, val_idx = train_test_split(
            indices, test_size=0.2, random_state=42, stratify=y,
        )
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        logger.info(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

        # Training config with FT params
        ft_merged_trainer_params = {"log_every_n_steps": 50}
        if trainer_params:
            ft_merged_trainer_params.update(trainer_params)

        training_config = TrainingConfig(
            num_epochs=ft_epochs,
            batch_size=ft_batch_size,
            lr=ft_lr,
            patience_early_stopping=ft_patience,
            save_path=save_dir or "basic_model_ft",
            trainer_params=ft_merged_trainer_params,
        )

        # Fine-tune existing classifier weights
        self.classifier.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            training_config=training_config,
            verbose=True,
        )

        metrics = {
            "train_samples": len(train_idx),
            "val_samples": len(val_idx),
            "total_samples": len(df),
            "dropped_samples": n_dropped,
        }

        logger.info("Fine-tuning complete.")
        return metrics

    def predict(self, texts: list[str], top_k: int = 1) -> dict:
        """Predict COICOP codes for input texts.

        Args:
            texts: List of preprocessed text strings.
            top_k: Number of top predictions to return.

        Returns:
            Dict with "predictions" and "confidence" keys.
        """
        if not self._is_trained:
            raise RuntimeError("Classifier must be trained before prediction.")

        X = np.array(texts)
        result = self.classifier.predict(X, top_k=top_k)

        if top_k > 1:
            pred_indices = result["prediction"].numpy()  # (n, top_k)
            pred_confidence = result["confidence"].numpy()  # (n, top_k)

            predictions = [
                [self.idx_to_label[pred_indices[i, k]] for k in range(top_k)]
                for i in range(len(texts))
            ]
            confidence = pred_confidence.tolist()
        else:
            pred_indices = result["prediction"].numpy().flatten()
            pred_confidence = result["confidence"].numpy().flatten()

            predictions = [self.idx_to_label[idx] for idx in pred_indices]
            confidence = pred_confidence.tolist()

        return {"predictions": predictions, "confidence": confidence}

    def save(self, path: str | Path) -> None:
        """Save the classifier to disk.

        Args:
            path: Directory path to save all components.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save the torchTextClassifiers model
        self.classifier.save(path / "model")

        # Save tokenizer
        with open(path / "tokenizer.pkl", "wb") as f:
            pickle.dump(self.tokenizer, f)

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
                "min_samples_per_class": self.config.min_samples_per_class,
            },
            "label_names": self.label_names,
            "label_to_idx": self.label_to_idx,
            "idx_to_label": {str(k): v for k, v in self.idx_to_label.items()},
        }

        with open(path / "basic_metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

        logger.info(f"Basic classifier saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> BasicCOICOPClassifier:
        """Load a trained basic classifier.

        Args:
            path: Directory path where model was saved.

        Returns:
            Loaded BasicCOICOPClassifier instance.
        """
        path = Path(path)

        # Load metadata
        with open(path / "basic_metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

        config = BasicConfig(**metadata["config"])
        instance = cls(config=config)

        # Restore label mappings
        instance.label_names = metadata["label_names"]
        instance.label_to_idx = metadata["label_to_idx"]
        instance.idx_to_label = {
            int(k): v for k, v in metadata["idx_to_label"].items()
        }

        # Load tokenizer
        with open(path / "tokenizer.pkl", "rb") as f:
            instance.tokenizer = pickle.load(f)

        # Load classifier
        instance.classifier = torchTextClassifiers.load(path / "model")

        instance._is_trained = True

        logger.info(
            f"Basic classifier loaded: {len(instance.label_names)} classes"
        )
        return instance
