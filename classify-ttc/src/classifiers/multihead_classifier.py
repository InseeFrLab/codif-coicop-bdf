"""Multi-head hierarchical COICOP classifier with shared backbone.

This module implements a single shared-backbone model with N label-attention
classification heads (one per COICOP level). Unlike the hierarchical classifier
that trains N independent models, this architecture shares parameters across
levels through a common transformer backbone.

Architecture:
    Input text -> [Shared NGramTokenizer] -> [Shared Backbone: Embedding + Transformer Blocks]
    -> Per-level LabelAttentionClassifier heads -> Per-level ClassificationHead -> logits

Hierarchical consistency is enforced at inference via parent-child masking.
"""

from __future__ import annotations

import logging
import math
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchTextClassifiers.model.components import (
    AttentionConfig,
    ClassificationHead,
    LabelAttentionConfig,
    TextEmbedderConfig,
)
from torchTextClassifiers.model.components.attention import Block, norm
from torchTextClassifiers.model.components.text_embedder import LabelAttentionClassifier
from torchTextClassifiers.tokenizers import NGramTokenizer

if TYPE_CHECKING:
    import pandas as pd

from ..preprocessing.data_preparation import COICOP_LEVELS

logger = logging.getLogger(__name__)


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
# Dataset
# ---------------------------------------------------------------------------


class MultiHeadDataset(Dataset):
    """Dataset for multi-head training with per-level labels."""

    def __init__(
        self,
        texts: list[str],
        level_labels: dict[str, np.ndarray],
        tokenizer: NGramTokenizer,
    ):
        """
        Args:
            texts: List of input text strings.
            level_labels: Dict mapping level name to array of label indices.
                          Use -100 for samples without a label at that level.
            tokenizer: Trained NGramTokenizer.
        """
        self.texts = texts
        self.level_labels = level_labels
        self.tokenizer = tokenizer
        self.level_names = list(level_labels.keys())

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        labels = {k: self.level_labels[k][idx] for k in self.level_names}
        return self.texts[idx], labels

    def collate_fn(self, batch):
        texts, label_dicts = zip(*batch)
        tok_out = self.tokenizer.tokenize(list(texts))
        labels = {
            k: torch.tensor([d[k] for d in label_dicts], dtype=torch.long)
            for k in self.level_names
        }
        return {
            "input_ids": tok_out.input_ids,
            "attention_mask": tok_out.attention_mask,
            "labels": labels,
        }


# ---------------------------------------------------------------------------
# PyTorch Model
# ---------------------------------------------------------------------------


class MultiHeadClassificationModel(nn.Module):
    """Shared backbone + per-level label-attention classification heads."""

    def __init__(
        self,
        vocab_size: int,
        padding_idx: int,
        embedding_dim: int,
        max_seq_length: int,
        n_attention_layers: int,
        n_attention_heads: int,
        n_kv_heads: int,
        n_label_attention_heads: int,
        level_num_classes: dict[str, int],
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.level_names = list(level_num_classes.keys())

        # Shared embedding
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

        # Shared transformer backbone
        head_dim = embedding_dim // n_attention_heads
        if embedding_dim % n_attention_heads != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by n_attention_heads ({n_attention_heads})."
            )
        if head_dim % 2 != 0:
            raise ValueError(
                f"embedding_dim / n_attention_heads must be even for rotary positional embeddings. "
                f"Got head_dim={head_dim} (embedding_dim={embedding_dim}, n_attention_heads={n_attention_heads})."
            )
        if embedding_dim % n_label_attention_heads != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by n_label_attention_heads ({n_label_attention_heads})."
            )

        attention_config = AttentionConfig(
            n_layers=n_attention_layers,
            n_head=n_attention_heads,
            n_kv_head=n_kv_heads,
            sequence_len=max_seq_length,
            positional_encoding=True,
        )
        attention_config.n_embd = embedding_dim

        self.blocks = nn.ModuleList([
            Block(attention_config, layer_idx)
            for layer_idx in range(n_attention_layers)
        ])

        # Precompute RoPE
        self.rotary_seq_len = max_seq_length * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        # Per-level label attention + classification heads
        self.label_attention = nn.ModuleDict()
        self.classification_heads = nn.ModuleDict()

        for level_name, num_classes in level_num_classes.items():
            la_config = LabelAttentionConfig(
                n_head=n_label_attention_heads,
                num_classes=num_classes,
            )
            te_config = TextEmbedderConfig(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                padding_idx=padding_idx,
                label_attention_config=la_config,
            )
            self.label_attention[level_name] = LabelAttentionClassifier(te_config)
            self.classification_heads[level_name] = ClassificationHead(
                input_dim=embedding_dim,
                num_classes=1,
            )

        self.init_weights()

    def init_weights(self):
        self.apply(self._init_weights)
        # Zero out c_proj weights in transformer blocks
        for block in self.blocks:
            nn.init.zeros_(block.mlp.c_proj.weight)
            nn.init.zeros_(block.attn.c_proj.weight)
        # Recompute RoPE
        head_dim = self.cos.shape[-1] * 2  # cos shape is [1, seq, 1, head_dim//2]
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos = cos
        self.sin = sin
        # bfloat16 embeddings on CUDA
        if self.embedding.weight.device.type == "cuda":
            self.embedding.to(dtype=torch.bfloat16)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=1.0)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = next(self.parameters()).device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
        return cos, sin

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass through shared backbone + all heads.

        Args:
            input_ids: (batch, seq_len) token indices.
            attention_mask: (batch, seq_len) 1=real token, 0=pad.

        Returns:
            Dict mapping level name to logits tensor (batch, num_classes).
        """
        if input_ids.dtype != torch.long:
            input_ids = input_ids.to(torch.long)

        seq_len = input_ids.shape[1]
        token_emb = self.embedding(input_ids)
        token_emb = norm(token_emb)

        cos_sin = (self.cos[:, :seq_len], self.sin[:, :seq_len])
        for block in self.blocks:
            token_emb = block(token_emb, cos_sin)
        token_emb = norm(token_emb)

        logits = {}
        for level_name in self.level_names:
            la_out = self.label_attention[level_name](
                token_emb, attention_mask
            )["sentence_embedding"]  # (B, num_classes, D)
            head_out = self.classification_heads[level_name](
                norm(la_out)
            ).squeeze(-1)  # (B, num_classes)
            logits[level_name] = head_out

        return logits


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------


class MultiHeadLightningModule(pl.LightningModule):
    """PyTorch Lightning wrapper for multi-head training."""

    def __init__(
        self,
        model: MultiHeadClassificationModel,
        level_names: list[str],
        lr: float = 0.01,
        loss_weights: list[float] | None = None,
    ):
        super().__init__()
        self.model = model
        self.level_names = level_names
        self.lr = lr

        if loss_weights is not None:
            self.loss_weights = loss_weights
        else:
            self.loss_weights = [1.0] * len(level_names)

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)

    def _compute_loss(self, batch, prefix: str):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        logits = self.model(input_ids, attention_mask)

        total_loss = torch.tensor(0.0, device=self.device)
        for i, level_name in enumerate(self.level_names):
            level_logits = logits[level_name]
            level_labels = labels[level_name]
            loss = F.cross_entropy(level_logits, level_labels, ignore_index=-100)
            weighted = self.loss_weights[i] * loss
            total_loss = total_loss + weighted
            self.log(f"{prefix}_loss_{level_name}", loss, prog_bar=False, on_epoch=True)

            if prefix == "val":
                # Accuracy for valid samples
                mask = level_labels != -100
                if mask.any():
                    preds = level_logits[mask].argmax(dim=-1)
                    acc = (preds == level_labels[mask]).float().mean()
                    self.log(
                        f"val_accuracy_{level_name}", acc, prog_bar=False, on_epoch=True
                    )

        self.log(f"{prefix}_loss", total_loss, prog_bar=True, on_epoch=True)
        return total_loss

    def training_step(self, batch, batch_idx):
        return self._compute_loss(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._compute_loss(batch, "val")

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
        self.tokenizer: NGramTokenizer | None = None
        self.model: MultiHeadClassificationModel | None = None
        self.level_names: list[str] = []
        self.level_label_names: dict[str, list[str]] = {}
        self.level_label_to_idx: dict[str, dict[str, int]] = {}
        self.level_idx_to_label: dict[str, dict[int, str]] = {}
        self.valid_children: dict[str, dict[str, list[int]]] = {}
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
                min_count=1,
                min_n=self.config.ngram_min_n,
                max_n=self.config.ngram_max_n,
                num_tokens=self.config.ngram_num_tokens,
                len_word_ngrams=1,
                training_text=texts,
                output_dim=self.config.max_seq_length,
            )
        logger.info("Tokenizer ready.")

    def _build_valid_children(self) -> None:
        """Build parent->children index mapping for hierarchical masking."""
        for level_idx in range(1, len(self.level_names)):
            level_name = self.level_names[level_idx]
            parent_level = self.level_names[level_idx - 1]

            children_map: dict[str, list[int]] = defaultdict(list)
            for code, idx in self.level_label_to_idx[level_name].items():
                parent_code = ".".join(code.split(".")[:level_idx])
                # Level 1 codes are 2-digit (e.g., "01"), no dot prefix
                if level_idx == 1:
                    parent_code = code.split(".")[0].zfill(2)
                children_map[parent_code].append(idx)

            self.valid_children[level_name] = dict(children_map)

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

            # Filter classes with enough samples
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

        # Build valid children hierarchy
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
        # Only split on samples that have the primary level
        valid_mask = primary_labels != -100
        valid_indices = np.where(valid_mask)[0]

        train_idx, val_idx = train_test_split(
            valid_indices,
            test_size=0.2,
            random_state=42,
            stratify=primary_labels[valid_indices],
        )

        # Add samples without primary level to training
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

        # Build model
        self.model = MultiHeadClassificationModel(
            vocab_size=self.tokenizer.vocab_size,
            padding_idx=self.tokenizer.padding_idx,
            embedding_dim=self.config.embedding_dim,
            max_seq_length=self.config.max_seq_length,
            n_attention_layers=self.config.n_attention_layers,
            n_attention_heads=self.config.n_attention_heads,
            n_kv_heads=self.config.n_kv_heads,
            n_label_attention_heads=self.config.n_label_attention_heads,
            level_num_classes=level_num_classes,
        )

        # Lightning module
        lightning_module = MultiHeadLightningModule(
            model=self.model,
            level_names=self.level_names,
            lr=self.config.lr,
            loss_weights=self.config.loss_weights,
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

        # Logger
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

        # Get best model weights if checkpoint callback exists
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

    def predict(
        self,
        texts: list[str],
        return_all_levels: bool = True,
        top_k: int = 1,
        confidence_threshold: float | None = None,
    ) -> dict:
        """Predict COICOP codes with hierarchical masking.

        Args:
            texts: List of text strings to classify.
            return_all_levels: Whether to return predictions at each level.
            top_k: Number of top predictions per level.
            confidence_threshold: Min confidence; stop at deepest level meeting it.

        Returns:
            Dict compatible with HierarchicalCOICOPClassifier.predict() output.
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
                logits = self.model(input_ids, attention_mask)

            for level_name in self.level_names:
                probs = F.softmax(logits[level_name], dim=-1).cpu().numpy()
                if level_name not in all_probs:
                    num_classes = probs.shape[1]
                    all_probs[level_name] = np.zeros((n, num_classes), dtype=np.float32)
                all_probs[level_name][start : start + len(batch_texts)] = probs

        # Apply hierarchical masking
        for level_idx in range(1, len(self.level_names)):
            level_name = self.level_names[level_idx]
            parent_level = self.level_names[level_idx - 1]

            # Use argsort (not argmax) for tie-breaking consistency with extraction
            parent_argmax = np.argsort(all_probs[parent_level], axis=1)[:, -1]
            num_classes = all_probs[level_name].shape[1]

            for i in range(n):
                parent_code = self.level_idx_to_label[parent_level][parent_argmax[i]]
                valid_idx = self.valid_children.get(level_name, {}).get(parent_code, [])
                if valid_idx:
                    mask = np.zeros(num_classes, dtype=np.float32)
                    mask[valid_idx] = 1.0
                    all_probs[level_name][i] *= mask
                    # Re-normalize
                    total = all_probs[level_name][i].sum()
                    if total > 0:
                        all_probs[level_name][i] /= total
                    else:
                        # Fallback: uniform over valid children
                        all_probs[level_name][i][valid_idx] = 1.0 / len(valid_idx)
                # If no valid children found, keep original probs

        # Extract predictions
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
                # Use argsort for consistency with masking tie-breaking
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

        # Combined confidence and threshold
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

    def save(self, path: str | Path, mlflow_run_id: str | None = None) -> None:
        """Save the multi-head classifier.

        Args:
            path: Directory to save all components.
            mlflow_run_id: Optional MLflow run ID for metadata.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save tokenizer
        with open(path / "tokenizer.pkl", "wb") as f:
            pickle.dump(self.tokenizer, f)

        # Save model state dict
        torch.save(self.model.state_dict(), path / "model.ckpt")

        # Save metadata
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

        Args:
            path: Directory where model was saved.

        Returns:
            Loaded MultiHeadCOICOPClassifier instance.
        """
        path = Path(path)

        # Load metadata
        with open(path / "multihead_metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

        # Create config
        config = MultiHeadConfig(**metadata["config"])
        instance = cls(config=config)

        # Restore mappings
        instance.level_names = metadata["level_names"]
        instance.level_label_names = metadata["level_label_names"]
        instance.level_label_to_idx = metadata["level_label_to_idx"]
        instance.level_idx_to_label = {
            level: {int(k): v for k, v in mapping.items()}
            for level, mapping in metadata["level_idx_to_label"].items()
        }
        instance.valid_children = metadata["valid_children"]

        # Load tokenizer
        with open(path / "tokenizer.pkl", "rb") as f:
            instance.tokenizer = pickle.load(f)

        # Rebuild model architecture
        level_num_classes = {
            level: len(labels) for level, labels in instance.level_label_names.items()
        }
        instance.model = MultiHeadClassificationModel(
            vocab_size=instance.tokenizer.vocab_size,
            padding_idx=instance.tokenizer.pad_token_id,
            embedding_dim=config.embedding_dim,
            max_seq_length=config.max_seq_length,
            n_attention_layers=config.n_attention_layers,
            n_attention_heads=config.n_attention_heads,
            n_kv_heads=config.n_kv_heads,
            n_label_attention_heads=config.n_label_attention_heads,
            level_num_classes=level_num_classes,
        )

        # Load state dict
        state_dict = torch.load(path / "model.ckpt", map_location="cpu", weights_only=True)
        instance.model.load_state_dict(state_dict)
        instance.model.eval()

        instance._is_trained = True

        logger.info(
            f"Multi-head classifier loaded: {len(instance.level_names)} levels "
            f"({', '.join(instance.level_names)})"
        )

        return instance
