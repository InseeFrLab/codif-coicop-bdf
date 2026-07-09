"""Multi-head hierarchical COICOP classifier with shared backbone.

This module implements a single shared-backbone model with N label-attention
classification heads (one per COICOP level). It uses the v2
``contrib.MultiLevelTextClassificationModel`` from torchTextClassifiers
(which wraps TokenEmbedder + list[SentenceEmbedder] + list[ClassificationHead]
+ CategoricalVariableNet under a unified forward).

Hierarchical consistency is enforced at inference via parent-child masking.

v1 → v2 migration notes (Phase 4):
- ``MultiHeadClassificationModel`` (custom nn.Module) →
  ``contrib.MultiLevelTextClassificationModel``
- ``MultiHeadLightningModule`` + manual ``F.cross_entropy`` →
  ``contrib.MultiLevelCrossEntropyLoss`` (expects ``labels: Tensor (batch, n_levels)``)
- Dataset ``collate_fn`` now returns ``{"labels": tensor}`` instead of
  ``{"labels": dict[str, tensor]}``. The integer label for level *i* is stored
  in the *i*-th column. Level names are indexed by ``self.level_names``.
"""

from __future__ import annotations

import logging
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# torchTextClassifiers v2 contrib
from torchTextClassifiers.contrib.multilevel import (
    MultiLevelCrossEntropyLoss,
    MultiLevelTextClassificationModel,
)

# torchTextClassifiers v2 components
from torchTextClassifiers.model.components import AttentionConfig, LabelAttentionConfig
from torchTextClassifiers.model.components.categorical_var_net import CategoricalVariableNet
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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class MultiHeadConfig:
    """Configuration for multi-head COICOP classifier."""

    # Tokenizer
    ngram_min_n: int = 3
    ngram_max_n: int = 6
    ngram_num_tokens: int = 100_000
    # Tokenizer override (None = use NGram, else HuggingFace pretrained name)
    tokenizer_name: str | None = None
    # Backbone
    embedding_dim: int = 128
    max_seq_length: int = 64
    n_attention_layers: int = 2
    n_attention_heads: int = 4
    n_kv_heads: int = 4
    # Per-level heads
    n_label_attention_heads: int = 4
    max_level: int = 4
    # Training
    batch_size: int = 32
    lr: float = 0.01
    num_epochs: int = 20
    patience: int = 5
    loss_weights: list[float] | None = None
    min_samples_per_level: int = 50
    min_samples_per_class: int = 2
    # DataLoader
    num_workers: int = 0
    pin_memory: bool = True
    predict_batch_size: int = 512


# ---------------------------------------------------------------------------
# Dataset — v2 expects labels as a (batch, n_levels) integer tensor
# ---------------------------------------------------------------------------


class MultiHeadDataset(Dataset):
    """Dataset for multi-head training with per-level labels.

    Labels are returned as a ``(batch, n_levels)`` integer tensor
    (``-100`` for missing samples at a level), not as a ``dict[str, tensor]``.
    Level *i* of the tensor corresponds to ``self.level_names[i]``.
    """

    def __init__(
        self,
        texts: list[str],
        level_labels: dict[str, np.ndarray],
        tokenizer,
    ):
        self.texts = texts
        self.level_labels = level_labels
        self.tokenizer = tokenizer
        self.level_names = list(level_labels.keys())
        self.level_indices = {name: i for i, name in enumerate(self.level_names)}

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        row = np.empty((len(self.level_names),), dtype=np.int64)
        for name, li in self.level_indices.items():
            row[li] = self.level_labels[name][idx]
        return self.texts[idx], row

    def collate_fn(self, batch):
        texts, label_rows = zip(*batch)
        tok_out = self.tokenizer.tokenize(list(texts))
        labels = torch.tensor(np.stack(label_rows), dtype=torch.long)
        # Dummy categorical_vars for the v2 forward (always passes 0 for the single dummy var)
        batch_size = len(texts)
        cat_vars = torch.zeros(batch_size, 1, dtype=torch.long)
        return {
            "input_ids": tok_out.input_ids,
            "attention_mask": tok_out.attention_mask,
            "labels": labels,
            "categorical_vars": cat_vars,
        }


# ---------------------------------------------------------------------------
# Lightning module — minimal wrapper around v2 components
# ---------------------------------------------------------------------------


class MultiHeadLightningModule(pl.LightningModule):
    """Lightning wrapper for ``contrib.MultiLevelTextClassificationModel``.

    The v2 model requires a ``categorical_vars`` tensor (even if dummy) because
    its internal ``CategoricalVariableNet`` does not guard on ``None``.
    """

    def __init__(
        self,
        model: MultiLevelTextClassificationModel,
        level_names: list[str],
        lr: float = 0.01,
        loss_weights: list[float] | None = None,
    ):
        super().__init__()
        self.model = model
        self.level_names = level_names
        self.lr = lr
        self.loss_fn = MultiLevelCrossEntropyLoss()
        self.loss_weights = loss_weights or [1.0] * len(level_names)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> list[torch.Tensor]:
        return self.model(input_ids, attention_mask)

    def _shared_step(self, batch, prefix: str):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        batch_size = input_ids.shape[0]
        # Dummy categorical vars: [0] * n_cats (1 dummy var = constant embedding)
        cat_vars = torch.zeros(batch_size, 1, dtype=torch.long, device=input_ids.device)

        logits = self.model(input_ids, attention_mask, categorical_vars=cat_vars)  # list[Tensor (B, K_i)]

        # Manual weighted loss: MultiLevelCrossEntropyLoss gives per-sample loss (B,)
        raw_loss = self.loss_fn(logits, batch["labels"]).mean()

        if self.loss_weights != [1.0] * len(self.level_names):
            weighted = sum(w * F.cross_entropy(logit, labels, ignore_index=-100) for w, logit, labels in zip(
                self.loss_weights, logits, [batch["labels"][:, i] for i in range(len(self.level_names))]
            )) / sum(self.loss_weights)
            batch_loss = weighted.mean() if weighted.dim() > 0 else weighted
        else:
            batch_loss = raw_loss

        self.log(f"{prefix}_loss", batch_loss, prog_bar=True, on_epoch=True)

        # Per-level losses & accuracy (valid only where label != -100)
        for i, level_name in enumerate(self.level_names):
            level_logits = logits[i]  # (B, K_i)
            level_labels = batch["labels"][:, i]  # (B,)
            level_loss = F.cross_entropy(level_logits, level_labels, ignore_index=-100, reduction="mean")
            self.log(f"{prefix}_loss_{level_name}", level_loss, prog_bar=False, on_epoch=True)

            if prefix == "val":
                mask = level_labels != -100
                if mask.any():
                    preds = level_logits[mask].argmax(dim=-1)
                    acc = (preds == level_labels[mask]).float().mean()
                    self.log(f"val_accuracy_{level_name}", acc, prog_bar=False, on_epoch=True)

        return batch_loss

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
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


# ---------------------------------------------------------------------------
# Top-level Classifier
# ---------------------------------------------------------------------------


class MultiHeadCOICOPClassifier:
    """Multi-head COICOP classifier with shared backbone.

    Manages the full pipeline: tokenization, training, prediction, save/load.
    """

    def __init__(self, config: MultiHeadConfig | None = None):
        self.config = config or MultiHeadConfig()
        self.tokenizer: BaseTokenizer | None = None
        # v2: model is now MultiLevelTextClassificationModel
        self.model: MultiLevelTextClassificationModel | None = None
        self.level_names: list[str] = []
        self.level_label_names: dict[str, list[str]] = {}
        self.level_label_to_idx: dict[str, dict[str, int]] = {}
        self.level_idx_to_label: dict[str, dict[int, str]] = {}
        self.valid_children: dict[str, dict[str, list[int]]] = {}
        self._is_trained = False

    # -- Tokenizer ----------------------------------------------------------

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

    # -- Label mapping helpers ----------------------------------------------

    def _build_valid_children(self) -> None:
        """Build parent->children index mapping for hierarchical masking."""
        for level_idx in range(1, len(self.level_names)):
            level_name = self.level_names[level_idx]
            parent_level = self.level_names[level_idx - 1]

            children_map: dict[str, list[int]] = defaultdict(list)
            for code, idx in self.level_label_to_idx[level_name].items():
                parent_code = ".".join(code.split(".")[:level_idx])
                if level_idx == 1:
                    parent_code = code.split(".")[0].zfill(2)
                children_map[parent_code].append(idx)

            self.valid_children[level_name] = dict(children_map)

    # -- Build v2 components ------------------------------------------------

    def _build_model(self) -> MultiLevelTextClassificationModel:
        """Construct the v2 model from config + level metadata.

        Returns
        -------
        MultiLevelTextClassificationModel
            Ready-to-train model with a shared token embedder, per-level
            SentenceEmbedder (with label attention), per-level classification
            head, and a dummy CategoricalVariableNet (we don't use categorical
            features).
        """
        vocab_size = self.tokenizer.vocab_size
        padding_idx = self.tokenizer.padding_idx
        D = self.config.embedding_dim
        max_len = self.config.max_seq_length
        n_attn_layers = self.config.n_attention_layers
        n_attn_heads = self.config.n_attention_heads
        n_kv_heads = self.config.n_kv_heads
        n_la_heads = self.config.n_label_attention_heads

        # Shared token embedder
        attention_cfg = AttentionConfig(
            n_layers=n_attn_layers,
            n_head=n_attn_heads,
            n_kv_head=n_kv_heads,
            sequence_len=max_len,
            positional_encoding=True,
        )
        attention_cfg.n_embd = D
        tok_embedder = TokenEmbedder(
            TokenEmbedderConfig(
                vocab_size=vocab_size,
                embedding_dim=D,
                padding_idx=padding_idx,
                attention_config=attention_cfg,
            )
        )

        # Per-level sentence embedders + classification heads
        sentence_embedders: list[SentenceEmbedder] = []
        classification_heads: list[ClassificationHead] = []

        for level_name in self.level_names:
            num_classes = len(self.level_label_names[level_name])

            la_cfg = LabelAttentionConfig(
                n_head=n_la_heads,
                num_classes=num_classes,
                embedding_dim=D,
            )
            se_cfg = SentenceEmbedderConfig(
                aggregation_method=None,  # label attention handles aggregation
                label_attention_config=la_cfg,
            )

            sent_embedder = SentenceEmbedder(se_cfg)
            # Classification head: Linear(D, 1) per class → output (B, K, 1) → squeeze → (B, K)
            cls_head = ClassificationHead(input_dim=D, num_classes=1)

            sentence_embedders.append(sent_embedder)
            classification_heads.append(cls_head)

        # Dummy CategoricalVariableNet (no categorical vars used yet).
        # We use SUM_TO_TEXT with a single 0-indexed embedding of dim D so that
        # x_cat has shape (B, D) and after expansion (B, num_cls, D) it adds
        # a constant bias to every label-head's sentence embedding.
        cat_net = CategoricalVariableNet(
            categorical_vocabulary_sizes=[1],
            categorical_embedding_dims=None,  # triggers SUM_TO_TEXT
            text_embedding_dim=D,
        )

        model = MultiLevelTextClassificationModel(
            token_embedder=tok_embedder,
            sentence_embedders=sentence_embedders,
            classification_heads=classification_heads,
            categorical_variable_net=cat_net,
        )
        return model

    # -- Training -----------------------------------------------------------

    def train(
        self,
        df: pd.DataFrame,
        text_column: str = "text",
        code_column: str = "code",
        save_dir: str | None = None,
        mlflow_run_info: dict | None = None,
    ) -> dict:
        """Train the multi-head classifier.

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

        # Extract level columns
        df = df.copy()
        if "level1" not in df.columns:
            level_cols = df[code_column].apply(extract_levels).apply(pd.Series)
            df = pd.concat([df, level_cols], axis=1)

        # Determine active levels
        active_levels = COICOP_LEVELS[: self.config.max_level]

        # Build label mappings per level
        self.level_names = []
        level_num_classes = {}

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
            self.level_idx_to_label[level_name] = {
                idx: label for idx, label in enumerate(label_names)
            }
            level_num_classes[level_name] = len(label_names)
            logger.info(f"  {level_name}: {len(label_names)} classes, {len(level_df)} samples")

        if not self.level_names:
            raise ValueError("No valid levels found for training.")

        self._build_valid_children()

        # Train tokenizer
        texts = df[text_column].tolist()
        self._init_tokenizer(texts)

        # Build per-level label arrays (use -100 for missing)
        level_labels: dict[str, np.ndarray] = {}
        for level_name in self.level_names:
            mapping = self.level_label_to_idx[level_name]
            mapped = df[level_name].map(mapping)
            level_labels[level_name] = mapped.fillna(-100).astype(np.int64).values

        # Stratified train/val split on level 1 labels
        primary_level = self.level_names[0]
        primary_labels = level_labels[primary_level]
        valid_mask = primary_labels != -100
        valid_indices = np.where(valid_mask)[0]

        train_idx, val_idx = train_test_split(
            valid_indices,
            test_size=0.2,
            random_state=42,
            stratify=primary_labels[valid_indices],
        )
        invalid_indices = np.where(~valid_mask)[0]
        train_idx = np.concatenate([train_idx, invalid_indices])

        logger.info(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

        # Build datasets
        train_texts = [texts[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        train_level_labels = {k: v[train_idx] for k, v in level_labels.items()}
        val_level_labels = {k: v[val_idx] for k, v in level_labels.items()}

        train_ds = MultiHeadDataset(train_texts, train_level_labels, self.tokenizer)
        val_ds = MultiHeadDataset(val_texts, val_level_labels, self.tokenizer)

        train_dl = DataLoader(
            train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=train_ds.collate_fn,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.num_workers > 0,
        )
        val_dl = DataLoader(
            val_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=val_ds.collate_fn,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.num_workers > 0,
        )

        # Build v2 model
        self.model = self._build_model()

        loss_weights = self.config.loss_weights or [1.0] * len(self.level_names)

        # Lightning module
        lightning_module = MultiHeadLightningModule(
            model=self.model,
            level_names=self.level_names,
            lr=self.config.lr,
            loss_weights=loss_weights,
        )

        # Callbacks
        callbacks = [
            pl.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.config.patience,
                mode="min",
            ),
        ]

        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            callbacks.append(
                pl.callbacks.ModelCheckpoint(
                    dirpath=save_dir,
                    filename="multihead-{epoch:02d}-{val_loss:.4f}",
                    monitor="val_loss",
                    mode="min",
                    save_top_k=1,
                )
            )

        # Logger (MLflow)
        trainer_logger = None
        if mlflow_run_info:
            from ..tracking.mlflow_utils import NonFinalizingMLFlowLogger

            trainer_logger = NonFinalizingMLFlowLogger(
                experiment_name=mlflow_run_info["experiment_name"],
                run_id=mlflow_run_info["run_id"],
                tracking_uri=mlflow_run_info["tracking_uri"],
                prefix="multihead",
            )

        # Trainer
        trainer = pl.Trainer(
            max_epochs=self.config.num_epochs,
            callbacks=callbacks,
            logger=trainer_logger if trainer_logger else True,
            enable_progress_bar=True,
            accelerator="auto",
        )

        logger.info("Starting multi-head training...")
        trainer.fit(lightning_module, train_dl, val_dl)
        logger.info("Training complete.")

        # Load best checkpoint
        ckpt_callback = None
        for cb in callbacks:
            if isinstance(cb, pl.callbacks.ModelCheckpoint):
                ckpt_callback = cb
                break

        if ckpt_callback and ckpt_callback.best_model_path:
            best_ckpt = torch.load(
                ckpt_callback.best_model_path, map_location="cpu", weights_only=False
            )
            lightning_module.load_state_dict(best_ckpt["state_dict"])
            self.model = lightning_module.model
            logger.info(f"Loaded best checkpoint: {ckpt_callback.best_model_path}")

        self._is_trained = True

        metrics = {
            "train_samples": len(train_idx),
            "val_samples": len(val_idx),
            "levels": {},
        }
        for level_name in self.level_names:
            train_valid = (train_level_labels[level_name] != -100).sum()
            val_valid = (val_level_labels[level_name] != -100).sum()
            metrics["levels"][level_name] = {
                "num_classes": level_num_classes[level_name],
                "train_samples": int(train_valid),
                "val_samples": int(val_valid),
            }

        return metrics

    # -- Prediction ---------------------------------------------------------

    def predict(
        self,
        texts: list[str],
        return_all_levels: bool = True,
        top_k: int = 1,
        confidence_threshold: float | None = None,
    ) -> dict:
        """Predict COICOP codes with hierarchical masking.

        Output format is identical to the v1 implementation so downstream
        code does not need to change.
        """
        if not self._is_trained:
            raise RuntimeError("Classifier must be trained before prediction.")

        self.model.eval()
        device = next(self.model.parameters()).device
        n = len(texts)

        # Batch prediction
        all_probs: dict[str, np.ndarray] = {}

        for start in range(0, n, self.config.predict_batch_size):
            batch_texts = texts[start : start + self.config.predict_batch_size]
            tok = self.tokenizer.tokenize(batch_texts)
            input_ids = tok.input_ids.to(device)
            attention_mask = tok.attention_mask.to(device)

            with torch.no_grad():
                cat_vars = torch.zeros(
                    len(batch_texts), 1, dtype=torch.long, device=device
                )
                logits_list = self.model(
                    input_ids, attention_mask, categorical_vars=cat_vars
                )  # list[Tensor (B, K_i)]

            for li, level_name in enumerate(self.level_names):
                probs = F.softmax(logits_list[li], dim=-1).cpu().numpy()
                if level_name not in all_probs:
                    num_classes = probs.shape[1]
                    all_probs[level_name] = np.zeros((n, num_classes), dtype=np.float32)
                all_probs[level_name][start : start + len(batch_texts)] = probs

        # -- Hierarchical masking -------------------------------------------
        for level_idx in range(1, len(self.level_names)):
            level_name = self.level_names[level_idx]
            parent_level = self.level_names[level_idx - 1]

            parent_argmax = np.argsort(all_probs[parent_level], axis=1)[:, -1]
            num_classes = all_probs[level_name].shape[1]

            for i in range(n):
                parent_code = self.level_idx_to_label[parent_level][parent_argmax[i]]
                valid_idx = self.valid_children.get(level_name, {}).get(parent_code, [])
                if valid_idx:
                    mask = np.zeros(num_classes, dtype=np.float32)
                    mask[valid_idx] = 1.0
                    all_probs[level_name][i] *= mask
                    total = all_probs[level_name][i].sum()
                    if total > 0:
                        all_probs[level_name][i] /= total
                    else:
                        all_probs[level_name][i][valid_idx] = 1.0 / len(valid_idx)

        # -- Extract predictions --------------------------------------------
        all_levels: dict[str, dict] = {}
        final_code = [""] * n
        final_confidence = np.zeros(n)
        final_level = [""] * n

        for level_name in self.level_names:
            probs = all_probs[level_name]

            if top_k > 1:
                top_k_actual = min(top_k, probs.shape[1])
                top_k_indices = np.argsort(probs, axis=1)[:, ::-1][:, :top_k_actual]
                top_k_labels = []
                top_k_confs = []
                for i in range(n):
                    labels_i = [
                        self.level_idx_to_label[level_name][top_k_indices[i, k]]
                        for k in range(top_k_actual)
                    ]
                    confs_i = [float(probs[i, top_k_indices[i, k]]) for k in range(top_k_actual)]
                    top_k_labels.append(labels_i)
                    top_k_confs.append(confs_i)

                all_levels[level_name] = {
                    "predictions": top_k_labels,
                    "confidence": top_k_confs,
                }

                for i in range(n):
                    final_code[i] = top_k_labels[i][0]
                    final_confidence[i] = top_k_confs[i][0]
                    final_level[i] = level_name
            else:
                argmax = np.argsort(probs, axis=1)[:, -1]
                predictions = [self.level_idx_to_label[level_name][idx] for idx in argmax]
                confidences = [float(probs[i, argmax[i]]) for i in range(n)]

                all_levels[level_name] = {
                    "predictions": predictions,
                    "confidence": confidences,
                }

                for i in range(n):
                    final_code[i] = predictions[i]
                    final_confidence[i] = confidences[i]
                    final_level[i] = level_name

        # -- Combined confidence & threshold --------------------------------
        combined_confidence = [1.0] * n
        for i in range(n):
            product = 1.0
            selected_code = ""
            selected_level = ""
            selected_conf = 0.0
            threshold_applied = False
            for level_name in self.level_names:
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
            "final_confidence": [float(c) for c in final_confidence],
            "combined_confidence": combined_confidence,
        }

        if return_all_levels:
            result["all_levels"] = all_levels

        return result

    # -- Save / Load --------------------------------------------------------

    def save(self, path: str | Path, mlflow_run_id: str | None = None) -> None:
        """Save the multi-head classifier.

        The model (``MultiLevelTextClassificationModel``) state dict is saved
        alongside the tokenizer and metadata.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save tokenizer
        with open(path / "tokenizer.pkl", "wb") as f:
            pickle.dump(self.tokenizer, f)

        # Save model state dict
        torch.save(self.model.state_dict(), path / "model.ckpt")

        metadata = {
            "config": {
                "ngram_min_n": self.config.ngram_min_n,
                "ngram_max_n": self.config.ngram_max_n,
                "ngram_num_tokens": self.config.ngram_num_tokens,
                "tokenizer_name": self.config.tokenizer_name,
                "embedding_dim": self.config.embedding_dim,
                "max_seq_length": self.config.max_seq_length,
                "n_attention_layers": self.config.n_attention_layers,
                "n_attention_heads": self.config.n_attention_heads,
                "n_kv_heads": self.config.n_kv_heads,
                "n_label_attention_heads": self.config.n_label_attention_heads,
                "max_level": self.config.max_level,
                "batch_size": self.config.batch_size,
                "lr": self.config.lr,
                "num_epochs": self.config.num_epochs,
                "patience": self.config.patience,
                "loss_weights": self.config.loss_weights,
                "min_samples_per_level": self.config.min_samples_per_level,
                "min_samples_per_class": self.config.min_samples_per_class,
                "predict_batch_size": self.config.predict_batch_size,
            },
            "level_names": self.level_names,
            "level_label_names": self.level_label_names,
            "level_label_to_idx": self.level_label_to_idx,
            "level_idx_to_label": {
                level: {str(k): v for k, v in mapping.items()}
                for level, mapping in self.level_idx_to_label.items()
            },
            "valid_children": self.valid_children,
            "mlflow_run_id": mlflow_run_id,
        }

        with open(path / "multihead_metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

        logger.info(f"Multi-head classifier saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> MultiHeadCOICOPClassifier:
        """Load a trained multi-head classifier.

        Rebuilds the v2 model architecture from metadata then loads the
        saved state dict.
        """
        path = Path(path)

        with open(path / "multihead_metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

        config = MultiHeadConfig(**metadata["config"])
        instance = cls(config=config)

        instance.level_names = metadata["level_names"]
        instance.level_label_names = metadata["level_label_names"]
        instance.level_label_to_idx = metadata["level_label_to_idx"]
        instance.level_idx_to_label = {
            level: {int(k): v for k, v in mapping.items()}
            for level, mapping in metadata["level_idx_to_label"].items()
        }
        instance.valid_children = metadata["valid_children"]

        with open(path / "tokenizer.pkl", "rb") as f:
            instance.tokenizer = pickle.load(f)

        # Rebuild the v2 model (same path as in _build_model)
        instance.model = instance._build_model()

        state_dict = torch.load(path / "model.ckpt", map_location="cpu", weights_only=True)
        instance.model.load_state_dict(state_dict)
        instance.model.eval()

        instance._is_trained = True

        logger.info(
            f"Multi-head classifier loaded: {len(instance.level_names)} levels "
            f"({', '.join(instance.level_names)})"
        )

        return instance
