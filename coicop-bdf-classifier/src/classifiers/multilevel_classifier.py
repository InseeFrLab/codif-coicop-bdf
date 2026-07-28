"""Interpretable multi-level COICOP classifier (INSEE NAF-style).

Implements the approach described in
https://julber95.github.io/interpretable-text-classification/pages/naf_multilevel.html
applied to COICOP: a single shared token encoder feeding N *independent*
per-level classification heads, trained jointly on the summed cross-entropy
``L = L_level1 + ... + L_levelN``.

How this differs from :mod:`multihead_classifier`, which shares the same
skeleton:

- **Independent heads.** No parent->child masking at inference. Quoting the
  reference: *"The five heads never talk to each other... Nothing forces the
  section, division, group, class, and sub-class predictions for the same
  example to be mutually consistent."* Disagreement between levels is surfaced
  (``levels_consistent``) rather than suppressed.
- **FastText-style backbone by default.** ``n_attention_layers=0`` means the
  token encoder is an embedding table + RMSNorm, and each head mean-pools.
  Transformer blocks remain reachable by raising ``n_attention_layers``.
- **Explainability.** :meth:`MultilevelCOICOPClassifier.explain` returns
  per-word contributions for every level, with no extra dependency.

Two head types are supported:

- ``head_type="mean"`` (default): masked mean pooling + ``Linear(D, K)``.
  Attribution is *exact* — see :meth:`explain`.
- ``head_type="label-attention"``: per-class cross-attention (learned label
  queries over token embeddings) + ``Linear(D, 1)`` applied per class.
  Attribution uses the attention matrix the model already computes.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from torchTextClassifiers.contrib.multilevel import MultiLevelTextClassificationModel
from torchTextClassifiers.model.components import AttentionConfig, LabelAttentionConfig
from torchTextClassifiers.model.components.classification_head import ClassificationHead
from torchTextClassifiers.model.components.text_embedder import (
    SentenceEmbedder,
    SentenceEmbedderConfig,
    TokenEmbedder,
    TokenEmbedderConfig,
)
from torchTextClassifiers.tokenizers import NGramTokenizer

if TYPE_CHECKING:
    import pandas as pd

from ..preprocessing.data_preparation import COICOP_LEVELS

logger = logging.getLogger(__name__)

HEAD_TYPES = ("mean", "label-attention")
TOKENIZER_TYPES = ("ngram", "wordpiece")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class MultiLevelModelWithAttention(MultiLevelTextClassificationModel):
    """``MultiLevelTextClassificationModel`` without categorical features.

    Overrides ``forward`` for two reasons:

    1. The parent implementation always calls ``self.categorical_variable_net``,
       which raises ``TypeError`` when it is ``None``. This model uses no
       categorical features, so the call is skipped entirely (the parent's
       ``__init__`` accepts ``None`` without complaint).
    2. The parent discards the ``label_attention_matrix`` that
       ``SentenceEmbedder`` already computed, making attention weights
       unreachable. ``return_attention=True`` returns them.
    """

    def forward(  # type: ignore[override]
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_attention: bool = False,
        **kwargs,
    ):
        """Run a forward pass and return one logit tensor per level.

        Args:
            input_ids: Tokenised text, shape ``(batch, seq_len)``.
            attention_mask: Padding mask, shape ``(batch, seq_len)``.
            return_attention: Also return per-level label-attention matrices.

        Returns:
            ``list[Tensor (batch, num_classes_i)]``, or a
            ``(logits, attentions)`` tuple when ``return_attention`` is set.
            An attention entry is ``None`` for mean-pooling levels.
        """
        token_embeddings = self.token_embedder(input_ids, attention_mask)["token_embeddings"]

        logits: list[torch.Tensor] = []
        attentions: list[torch.Tensor | None] = []

        for sentence_embedder, classification_head in zip(
            self.sentence_embedders, self.classification_heads
        ):
            out = sentence_embedder(
                token_embeddings=token_embeddings,
                attention_mask=attention_mask,
                return_label_attention_matrix=return_attention,
            )
            # (B, D) for mean pooling, (B, K, D) for label attention.
            # squeeze(-1) collapses the (B, K, 1) label-attention head output
            # and is a no-op for the (B, K) mean-pooling output when K > 1.
            logits.append(classification_head(out["sentence_embedding"]).squeeze(-1))
            attentions.append(out["label_attention_matrix"])

        if return_attention:
            return logits, attentions
        return logits


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class MultilevelConfig:
    """Configuration for the interpretable multi-level COICOP classifier."""

    # Architecture
    head_type: str = "mean"  # "mean" | "label-attention"
    embedding_dim: int = 128
    max_seq_length: int = 64
    n_attention_layers: int = 0  # 0 = FastText-style, no transformer blocks
    n_attention_heads: int = 4
    n_kv_heads: int = 4
    n_label_attention_heads: int = 4
    max_level: int = 5
    # Tokenizer
    tokenizer_type: str = "ngram"  # "ngram" | "wordpiece"
    ngram_min_n: int = 3
    ngram_max_n: int = 6
    ngram_num_tokens: int = 100_000
    wordpiece_vocab_size: int = 5_000
    tokenizer_name: str | None = None  # HuggingFace pretrained name (overrides type)
    # Training
    batch_size: int = 32
    lr: float = 1e-3
    num_epochs: int = 20
    patience: int = 5
    loss_weights: list[float] | None = None
    min_samples_per_level: int = 50
    min_samples_per_class: int = 2
    # DataLoader
    num_workers: int = 0
    pin_memory: bool = True
    predict_batch_size: int = 512

    def __post_init__(self) -> None:
        if self.head_type not in HEAD_TYPES:
            raise ValueError(
                f"head_type must be one of {HEAD_TYPES}, got {self.head_type!r}"
            )
        if self.tokenizer_type not in TOKENIZER_TYPES:
            raise ValueError(
                f"tokenizer_type must be one of {TOKENIZER_TYPES}, "
                f"got {self.tokenizer_type!r}"
            )
        if (
            self.head_type == "label-attention"
            and self.embedding_dim % self.n_label_attention_heads != 0
        ):
            raise ValueError(
                f"embedding_dim ({self.embedding_dim}) must be divisible by "
                f"n_label_attention_heads ({self.n_label_attention_heads})."
            )
        if (
            self.n_attention_layers > 0
            and self.embedding_dim % self.n_attention_heads != 0
        ):
            raise ValueError(
                f"embedding_dim ({self.embedding_dim}) must be divisible by "
                f"n_attention_heads ({self.n_attention_heads})."
            )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class MultilevelDataset(Dataset):
    """Dataset yielding text plus a ``(n_levels,)`` integer label row.

    Missing labels (code absent at that depth, or a class pruned by
    ``min_samples_per_class``) are ``-100`` and ignored by the loss.
    """

    def __init__(self, texts: list[str], level_labels: dict[str, np.ndarray], tokenizer):
        self.texts = texts
        self.level_labels = level_labels
        self.tokenizer = tokenizer
        self.level_names = list(level_labels.keys())

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx):
        row = np.empty((len(self.level_names),), dtype=np.int64)
        for li, name in enumerate(self.level_names):
            row[li] = self.level_labels[name][idx]
        return self.texts[idx], row

    def collate_fn(self, batch):
        texts, label_rows = zip(*batch)
        tok_out = self.tokenizer.tokenize(list(texts))
        return {
            "input_ids": tok_out.input_ids,
            "attention_mask": tok_out.attention_mask,
            "labels": torch.tensor(np.stack(label_rows), dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------


class MultilevelLightningModule(pl.LightningModule):
    """Lightning wrapper implementing the summed per-level cross-entropy.

    The loss is computed locally rather than with
    ``contrib.MultiLevelCrossEntropyLoss``, which calls ``output.squeeze()``
    (corrupting shapes at batch size 1) and averages over *all* levels
    regardless of ``-100`` masking. Here each level contributes
    ``w_i * CE(logits_i, labels_i, ignore_index=-100)``, and a level with no
    valid label in the batch contributes exactly zero instead of NaN.
    """

    def __init__(
        self,
        model: MultiLevelModelWithAttention,
        level_names: list[str],
        lr: float = 1e-3,
        loss_weights: list[float] | None = None,
    ):
        super().__init__()
        self.model = model
        self.level_names = level_names
        self.lr = lr
        self.loss_weights = loss_weights or [1.0] * len(level_names)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        return self.model(input_ids, attention_mask)

    def _shared_step(self, batch, prefix: str) -> torch.Tensor:
        logits = self.model(batch["input_ids"], batch["attention_mask"])
        labels = batch["labels"]

        total = torch.zeros((), device=batch["input_ids"].device)

        for i, level_name in enumerate(self.level_names):
            level_labels = labels[:, i]
            mask = level_labels != -100
            if not bool(mask.any()):
                # No supervision for this level in this batch: contribute 0.
                # F.cross_entropy would return NaN here and poison the sum.
                continue

            level_loss = F.cross_entropy(
                logits[i], level_labels, ignore_index=-100, reduction="mean"
            )
            total = total + self.loss_weights[i] * level_loss
            self.log(f"{prefix}_loss_{level_name}", level_loss, on_epoch=True)

            if prefix == "val":
                preds = logits[i][mask].argmax(dim=-1)
                acc = (preds == level_labels[mask]).float().mean()
                self.log(f"val_accuracy_{level_name}", acc, on_epoch=True)

        self.log(f"{prefix}_loss", total, prog_bar=True, on_epoch=True)
        return total

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class MultilevelCOICOPClassifier:
    """Interpretable multi-level COICOP classifier with independent heads."""

    def __init__(self, config: MultilevelConfig | None = None):
        self.config = config or MultilevelConfig()
        self.tokenizer = None
        self.model: MultiLevelModelWithAttention | None = None
        self.level_names: list[str] = []
        self.level_label_names: dict[str, list[str]] = {}
        self.level_label_to_idx: dict[str, dict[str, int]] = {}
        self.level_idx_to_label: dict[str, dict[int, str]] = {}
        self._is_trained = False

    # -- Tokenizer ----------------------------------------------------------

    def _init_tokenizer(self, texts: list[str]) -> None:
        """Fit (or load) the tokenizer. ``texts`` must be the training split only."""
        if self.config.tokenizer_name is not None:
            from torchTextClassifiers.tokenizers import HuggingFaceTokenizer

            logger.info(f"Loading HuggingFace tokenizer: {self.config.tokenizer_name}...")
            self.tokenizer = HuggingFaceTokenizer.load_from_pretrained(
                self.config.tokenizer_name,
                output_dim=self.config.max_seq_length,
            )
        elif self.config.tokenizer_type == "wordpiece":
            from torchTextClassifiers.tokenizers.WordPiece import WordPieceTokenizer

            logger.info(
                f"Training WordPieceTokenizer (vocab_size={self.config.wordpiece_vocab_size})..."
            )
            self.tokenizer = WordPieceTokenizer(
                vocab_size=self.config.wordpiece_vocab_size,
                output_dim=self.config.max_seq_length,
            )
            self.tokenizer.train(texts)
        else:
            logger.info(
                f"Training NGramTokenizer (n={self.config.ngram_min_n}-{self.config.ngram_max_n}, "
                f"vocab_size={self.config.ngram_num_tokens})..."
            )
            self.tokenizer = NGramTokenizer(
                min_count=1,
                min_n=self.config.ngram_min_n,
                max_n=self.config.ngram_max_n,
                num_tokens=self.config.ngram_num_tokens,
                len_word_ngrams=1,
                training_text=texts,
                output_dim=self.config.max_seq_length,
                preprocess=True,
            )
        logger.info("Tokenizer ready.")

    # -- Model --------------------------------------------------------------

    def _build_model(self) -> MultiLevelModelWithAttention:
        """Construct the shared encoder and one independent head per level."""
        cfg = self.config
        d = cfg.embedding_dim

        attention_cfg = None
        if cfg.n_attention_layers > 0:
            attention_cfg = AttentionConfig(
                n_layers=cfg.n_attention_layers,
                n_head=cfg.n_attention_heads,
                n_kv_head=cfg.n_kv_heads,
                sequence_len=cfg.max_seq_length,
                positional_encoding=True,
            )
            attention_cfg.n_embd = d

        token_embedder = TokenEmbedder(
            TokenEmbedderConfig(
                vocab_size=self.tokenizer.vocab_size,
                embedding_dim=d,
                padding_idx=self.tokenizer.padding_idx,
                attention_config=attention_cfg,
            )
        )
        # MultiLevelTextClassificationModel never calls this, unlike the
        # single-level TextClassificationModel. Do it here so the FastText
        # backbone gets the library's intended init.
        token_embedder.init_weights()

        sentence_embedders: list[SentenceEmbedder] = []
        classification_heads: list[ClassificationHead] = []

        for level_name in self.level_names:
            num_classes = len(self.level_label_names[level_name])

            if cfg.head_type == "label-attention":
                sentence_embedders.append(
                    SentenceEmbedder(
                        SentenceEmbedderConfig(
                            aggregation_method=None,
                            label_attention_config=LabelAttentionConfig(
                                n_head=cfg.n_label_attention_heads,
                                num_classes=num_classes,
                                embedding_dim=d,
                            ),
                        )
                    )
                )
                # One score per class: (B, K, D) -> (B, K, 1) -> (B, K)
                classification_heads.append(ClassificationHead(input_dim=d, num_classes=1))
            else:
                sentence_embedders.append(
                    SentenceEmbedder(
                        SentenceEmbedderConfig(
                            aggregation_method="mean", label_attention_config=None
                        )
                    )
                )
                classification_heads.append(
                    ClassificationHead(input_dim=d, num_classes=num_classes)
                )

        return MultiLevelModelWithAttention(
            token_embedder=token_embedder,
            sentence_embedders=sentence_embedders,
            classification_heads=classification_heads,
            categorical_variable_net=None,
        )

    # -- Training -----------------------------------------------------------

    def train(
        self,
        df: pd.DataFrame,
        text_column: str = "text",
        code_column: str = "code",
        save_dir: str | None = None,
        mlflow_run_info: dict | None = None,
    ) -> dict:
        """Train the multi-level classifier.

        Args:
            df: DataFrame with text and COICOP code columns.
            text_column: Name of the text column.
            code_column: Name of the COICOP code column.
            save_dir: Directory for Lightning checkpoints.
            mlflow_run_info: Optional MLflow info dict for logging.

        Returns:
            Dict with training metrics.
        """
        import pandas as pd
        from sklearn.model_selection import train_test_split

        from ..preprocessing.data_preparation import extract_levels

        df = df.copy()
        if "level1" not in df.columns:
            level_cols = df[code_column].apply(extract_levels).apply(pd.Series)
            df = pd.concat([df, level_cols], axis=1)

        active_levels = COICOP_LEVELS[: self.config.max_level]

        self.level_names = []
        self.level_label_names = {}
        self.level_label_to_idx = {}
        self.level_idx_to_label = {}
        level_num_classes: dict[str, int] = {}

        for level_name in active_levels:
            level_df = df[df[level_name].notna()]

            if len(level_df) < self.config.min_samples_per_level:
                logger.warning(
                    f"Skipping {level_name}: insufficient samples "
                    f"({len(level_df)} < {self.config.min_samples_per_level})"
                )
                continue

            label_counts = level_df[level_name].value_counts()
            valid_labels = label_counts[
                label_counts >= self.config.min_samples_per_class
            ].index.tolist()

            if len(valid_labels) < 2:
                logger.warning(f"Skipping {level_name}: fewer than 2 valid classes")
                continue

            label_names = sorted(valid_labels)
            self.level_names.append(level_name)
            self.level_label_names[level_name] = label_names
            self.level_label_to_idx[level_name] = {
                label: idx for idx, label in enumerate(label_names)
            }
            self.level_idx_to_label[level_name] = dict(enumerate(label_names))
            level_num_classes[level_name] = len(label_names)
            logger.info(f"  {level_name}: {len(label_names)} classes, {len(level_df)} samples")

        if not self.level_names:
            raise ValueError("No valid levels found for training.")

        # Per-level label arrays (-100 = no supervision at this level)
        level_labels: dict[str, np.ndarray] = {}
        for level_name in self.level_names:
            mapped = df[level_name].map(self.level_label_to_idx[level_name])
            level_labels[level_name] = mapped.fillna(-100).astype(np.int64).values

        # Stratified split on the shallowest active level
        texts = df[text_column].tolist()
        primary_labels = level_labels[self.level_names[0]]
        valid_mask = primary_labels != -100
        valid_indices = np.where(valid_mask)[0]

        train_idx, val_idx = train_test_split(
            valid_indices,
            test_size=0.2,
            random_state=42,
            stratify=primary_labels[valid_indices],
        )
        # Rows with no label at the primary level can still supervise deeper
        # levels, so they go to train rather than being dropped.
        train_idx = np.concatenate([train_idx, np.where(~valid_mask)[0]])

        logger.info(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

        train_texts = [texts[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]

        # Fit the tokenizer on the training split only, so validation text
        # never leaks into the vocabulary.
        self._init_tokenizer(train_texts)

        train_level_labels = {k: v[train_idx] for k, v in level_labels.items()}
        val_level_labels = {k: v[val_idx] for k, v in level_labels.items()}

        train_ds = MultilevelDataset(train_texts, train_level_labels, self.tokenizer)
        val_ds = MultilevelDataset(val_texts, val_level_labels, self.tokenizer)

        dl_kwargs = dict(
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.num_workers > 0,
        )
        train_dl = DataLoader(
            train_ds, shuffle=True, collate_fn=train_ds.collate_fn, **dl_kwargs
        )
        val_dl = DataLoader(
            val_ds, shuffle=False, collate_fn=val_ds.collate_fn, **dl_kwargs
        )

        self.model = self._build_model()

        lightning_module = MultilevelLightningModule(
            model=self.model,
            level_names=self.level_names,
            lr=self.config.lr,
            loss_weights=self.config.loss_weights or [1.0] * len(self.level_names),
        )

        callbacks = [
            pl.callbacks.EarlyStopping(
                monitor="val_loss", patience=self.config.patience, mode="min"
            )
        ]
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            callbacks.append(
                pl.callbacks.ModelCheckpoint(
                    dirpath=save_dir,
                    filename="multilevel-{epoch:02d}-{val_loss:.4f}",
                    monitor="val_loss",
                    mode="min",
                    save_top_k=1,
                )
            )

        trainer_logger = None
        if mlflow_run_info:
            from ..tracking.mlflow_utils import NonFinalizingMLFlowLogger

            trainer_logger = NonFinalizingMLFlowLogger(
                experiment_name=mlflow_run_info["experiment_name"],
                run_id=mlflow_run_info["run_id"],
                tracking_uri=mlflow_run_info["tracking_uri"],
                prefix="multilevel",
            )

        trainer = pl.Trainer(
            max_epochs=self.config.num_epochs,
            callbacks=callbacks,
            logger=trainer_logger if trainer_logger else True,
            enable_progress_bar=True,
            accelerator="auto",
        )

        logger.info("Starting multi-level training...")
        trainer.fit(lightning_module, train_dl, val_dl)
        logger.info("Training complete.")

        ckpt_callback = next(
            (cb for cb in callbacks if isinstance(cb, pl.callbacks.ModelCheckpoint)), None
        )
        if ckpt_callback and ckpt_callback.best_model_path:
            best_ckpt = torch.load(
                ckpt_callback.best_model_path, map_location="cpu", weights_only=False
            )
            lightning_module.load_state_dict(best_ckpt["state_dict"])
            self.model = lightning_module.model
            logger.info(f"Loaded best checkpoint: {ckpt_callback.best_model_path}")

        self.model.eval()
        self._is_trained = True

        return {
            "train_samples": len(train_idx),
            "val_samples": len(val_idx),
            "levels": {
                level_name: {
                    "num_classes": level_num_classes[level_name],
                    "train_samples": int((train_level_labels[level_name] != -100).sum()),
                    "val_samples": int((val_level_labels[level_name] != -100).sum()),
                }
                for level_name in self.level_names
            },
        }

    # -- Prediction ---------------------------------------------------------

    def _level_probs(self, texts: list[str]) -> dict[str, np.ndarray]:
        """Run the model over ``texts`` and return per-level softmax probabilities."""
        self.model.eval()
        device = next(self.model.parameters()).device
        n = len(texts)
        all_probs: dict[str, np.ndarray] = {}

        for start in range(0, n, self.config.predict_batch_size):
            batch_texts = texts[start : start + self.config.predict_batch_size]
            tok = self.tokenizer.tokenize(batch_texts)

            with torch.no_grad():
                logits_list = self.model(
                    tok.input_ids.to(device), tok.attention_mask.to(device)
                )

            for li, level_name in enumerate(self.level_names):
                probs = F.softmax(logits_list[li], dim=-1).cpu().numpy()
                if level_name not in all_probs:
                    all_probs[level_name] = np.zeros((n, probs.shape[1]), dtype=np.float32)
                all_probs[level_name][start : start + len(batch_texts)] = probs

        return all_probs

    def predict(
        self,
        texts: list[str],
        return_all_levels: bool = True,
        top_k: int = 1,
        confidence_threshold: float | None = None,
    ) -> dict:
        """Predict COICOP codes, one independent decision per level.

        Unlike the hierarchical and multi-head classifiers, **no parent->child
        masking is applied**: each head's softmax stands on its own. Levels may
        therefore disagree, which ``levels_consistent`` reports.

        The returned dict matches the schema the other hierarchical predictors
        emit, so ``_HierarchicalBasePredictor`` and the top-k evaluation
        machinery work unchanged. Semantics specific to independent heads:

        - ``final_code`` / ``final_level`` / ``final_confidence``: the deepest
          active level's top-1. Each head predicts the full dotted prefix at
          its own depth, so this is a valid standalone code.
        - ``combined_confidence``: product of the per-level top-1 confidences.
          With independent heads this is an *agreement proxy*, not a cascade
          probability.
        - ``levels_consistent``: whether each level's top-1 is a string prefix
          of the next level's.
        """
        if not self._is_trained:
            raise RuntimeError("Classifier must be trained before prediction.")

        n = len(texts)
        all_probs = self._level_probs(texts)
        all_levels: dict[str, dict] = {}

        for level_name in self.level_names:
            probs = all_probs[level_name]
            idx_to_label = self.level_idx_to_label[level_name]

            if top_k > 1:
                k_actual = min(top_k, probs.shape[1])
                order = np.argsort(probs, axis=1)[:, ::-1][:, :k_actual]
                all_levels[level_name] = {
                    "predictions": [
                        [idx_to_label[order[i, k]] for k in range(k_actual)]
                        for i in range(n)
                    ],
                    "confidence": [
                        [float(probs[i, order[i, k]]) for k in range(k_actual)]
                        for i in range(n)
                    ],
                }
            else:
                argmax = probs.argmax(axis=1)
                all_levels[level_name] = {
                    "predictions": [idx_to_label[idx] for idx in argmax],
                    "confidence": [float(probs[i, argmax[i]]) for i in range(n)],
                }

        def top1(level_name: str, i: int) -> tuple[str, float]:
            data = all_levels[level_name]
            if top_k > 1:
                return data["predictions"][i][0], data["confidence"][i][0]
            return data["predictions"][i], data["confidence"][i]

        final_code = [""] * n
        final_level = [""] * n
        final_confidence = [0.0] * n
        combined_confidence = [0.0] * n
        levels_consistent = [True] * n

        for i in range(n):
            product = 1.0
            consistent = True
            previous_code: str | None = None

            for level_name in self.level_names:
                code, conf = top1(level_name, i)

                if confidence_threshold is not None and conf < confidence_threshold:
                    break

                if previous_code is not None and not code.startswith(previous_code):
                    consistent = False
                previous_code = code

                product *= conf
                final_code[i] = code
                final_level[i] = level_name
                final_confidence[i] = conf

            combined_confidence[i] = product if final_code[i] else 0.0
            levels_consistent[i] = consistent

        result = {
            "final_code": final_code,
            "final_level": final_level,
            "final_confidence": final_confidence,
            "combined_confidence": combined_confidence,
            "levels_consistent": levels_consistent,
        }
        if return_all_levels:
            result["all_levels"] = all_levels
        return result

    # -- Explainability -----------------------------------------------------

    def _word_spans(self, texts: list[str], tok) -> list[list[tuple[int, int, str]]]:
        """Group token positions into words using the tokenizer's word ids.

        Returns, per text, a list of ``(first_token, last_token, word)`` where
        the word is sliced out of the original text via the offset mapping.
        Tokens with a ``None`` word id (``[CLS]``/``[SEP]``/padding) are skipped.
        """
        spans: list[list[tuple[int, int, str]]] = []
        offsets = tok.offset_mapping

        for i, text in enumerate(texts):
            groups: dict[int, list[int]] = {}
            for pos, wid in enumerate(tok.word_ids[i]):
                if wid is None or (isinstance(wid, float) and np.isnan(wid)):
                    continue
                if int(tok.attention_mask[i, pos]) == 0:
                    continue
                groups.setdefault(int(wid), []).append(pos)

            row: list[tuple[int, int, str]] = []
            for wid in sorted(groups):
                positions = groups[wid]
                start = int(offsets[i, positions[0], 0])
                end = int(offsets[i, positions[-1], 1])
                word = text[start:end]
                if word.strip():
                    row.append((positions[0], positions[-1], word))
            spans.append(row)

        return spans

    def explain(
        self,
        texts: list[str],
        top_words: int = 10,
    ) -> list[dict[str, list[tuple[str, float]]]]:
        """Explain each level's prediction as per-word contributions.

        Two mechanisms, depending on ``head_type``:

        - ``"mean"``: the contribution of token *t* to class *c* is
          ``W_c . token_emb_t / n_tokens``. Since
          ``logit_c = W_c . mean(token_emb) + b_c``, these sum **exactly** to
          ``logit_c - b_c``. This is the completeness property that Integrated
          Gradients approximates with 20+ passes, available here in closed form
          because mean pooling makes the head additive over tokens.

          Completeness holds over *all tokens*. The returned word scores can
          fall short of ``logit_c - b_c`` by the contribution of tokens that
          map to no word — the n-gram tokenizer emits a trailing whole-sequence
          token, and WordPiece emits ``[CLS]``/``[SEP]``. Those are real inputs
          but correspond to no span of the user's text, so they are not
          displayed.
        - ``"label-attention"``: the model's own label-attention weights,
          averaged over heads and read off the predicted class's row. These say
          which tokens built the class representation — note the downstream
          linear layer may still weight that representation differently.

        Scores are summed token->word using the tokenizer's word ids.

        Args:
            texts: Texts to explain.
            top_words: Number of highest-scoring words to keep per level.

        Returns:
            One dict per text, mapping level name to a list of
            ``(word, score)`` sorted by descending absolute score.
        """
        if not self._is_trained:
            raise RuntimeError("Classifier must be trained before explaining.")

        self.model.eval()
        device = next(self.model.parameters()).device
        want_attention = self.config.head_type == "label-attention"

        tok = self.tokenizer.tokenize(
            texts, return_word_ids=True, return_offsets_mapping=True
        )
        input_ids = tok.input_ids.to(device)
        attention_mask = tok.attention_mask.to(device)

        with torch.no_grad():
            if want_attention:
                logits_list, attentions = self.model(
                    input_ids, attention_mask, return_attention=True
                )
            else:
                logits_list = self.model(input_ids, attention_mask)
                attentions = [None] * len(self.level_names)
                token_embeddings = self.model.token_embedder(input_ids, attention_mask)[
                    "token_embeddings"
                ]

        spans = self._word_spans(texts, tok)
        n_tokens = attention_mask.sum(dim=1).clamp(min=1)
        results: list[dict[str, list[tuple[str, float]]]] = []

        for i in range(len(texts)):
            per_level: dict[str, list[tuple[str, float]]] = {}

            for li, level_name in enumerate(self.level_names):
                predicted = int(logits_list[li][i].argmax())

                if want_attention:
                    # (n_head, K, seq_len) -> mean over heads -> predicted row
                    token_scores = attentions[li][i].mean(dim=0)[predicted]
                else:
                    # W_c . token_emb_t / n_tokens, exactly additive to the logit
                    weight = self.model.classification_heads[li].net.weight[predicted]
                    token_scores = (token_embeddings[i] @ weight) / n_tokens[i]

                token_scores = token_scores.detach().cpu().numpy()

                words = [
                    (word, float(token_scores[first : last + 1].sum()))
                    for first, last, word in spans[i]
                ]
                words.sort(key=lambda wc: abs(wc[1]), reverse=True)
                per_level[level_name] = words[:top_words]

            results.append(per_level)

        return results

    # -- Save / Load --------------------------------------------------------

    def save(self, path: str | Path, mlflow_run_id: str | None = None) -> None:
        """Save tokenizer, model weights and metadata to ``path``."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        with open(path / "tokenizer.pkl", "wb") as f:
            pickle.dump(self.tokenizer, f)

        torch.save(self.model.state_dict(), path / "model.ckpt")

        metadata = {
            # asdict() persists every field, so load() cannot silently reset
            # config values that were never written out.
            "config": asdict(self.config),
            "level_names": self.level_names,
            "level_label_names": self.level_label_names,
            "level_label_to_idx": self.level_label_to_idx,
            "level_idx_to_label": {
                level: {str(k): v for k, v in mapping.items()}
                for level, mapping in self.level_idx_to_label.items()
            },
            "mlflow_run_id": mlflow_run_id,
        }
        with open(path / "multilevel_metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

        logger.info(f"Multi-level classifier saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> MultilevelCOICOPClassifier:
        """Load a trained multi-level classifier from ``path``."""
        path = Path(path)

        with open(path / "multilevel_metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

        instance = cls(config=MultilevelConfig(**metadata["config"]))
        instance.level_names = metadata["level_names"]
        instance.level_label_names = metadata["level_label_names"]
        instance.level_label_to_idx = metadata["level_label_to_idx"]
        instance.level_idx_to_label = {
            level: {int(k): v for k, v in mapping.items()}
            for level, mapping in metadata["level_idx_to_label"].items()
        }

        with open(path / "tokenizer.pkl", "rb") as f:
            instance.tokenizer = pickle.load(f)

        instance.model = instance._build_model()
        instance.model.load_state_dict(
            torch.load(path / "model.ckpt", map_location="cpu", weights_only=True)
        )
        instance.model.eval()
        instance._is_trained = True

        logger.info(
            f"Multi-level classifier loaded: {len(instance.level_names)} levels "
            f"({', '.join(instance.level_names)}), head_type={instance.config.head_type}"
        )
        return instance
