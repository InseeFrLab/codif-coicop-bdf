"""Training script for COICOP classifiers."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import mlflow

from .classifiers.basic_classifier import BasicCOICOPClassifier, BasicConfig
from .preprocessing.data_preparation import load_annotations
from .classifiers.hierarchical_classifier import HierarchicalCOICOPClassifier, HierarchicalConfig
from .classifiers.multihead_classifier import MultiHeadCOICOPClassifier, MultiHeadConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

_SECRET_PARAMS = {"encryption_key"}


def _log_all_params(func_args: dict) -> None:
    """Log all function arguments to MLflow, excluding secrets and None values."""
    to_log = {}
    for k, v in func_args.items():
        if k in _SECRET_PARAMS:
            continue
        if v is None:
            continue
        if isinstance(v, (list, dict)):
            to_log[k] = str(v)
        else:
            to_log[k] = v
    mlflow.log_params(to_log)


def _evaluate_on_annotations(
    classifier,
    model_path: str | Path,
    classifier_type: str,
    eval_data_path: str,
    eval_text_column: str,
    eval_top_k: int,
    eval_filter_columns: list[str] | None = None,
    eval_code_column: str = "code",
    eval_beam_size: int = 1,
) -> dict[str, float]:
    """Predict on eval data and return top-k accuracy metrics dict.

    Args:
        classifier: Trained classifier instance (saved already).
        model_path: Path to the saved model directory.
        classifier_type: "hierarchical" or "cascade".
        eval_data_path: Path to evaluation parquet/csv file.
        eval_text_column: Name of text column in eval data.
        eval_top_k: Maximum K for top-k accuracy.

    Returns:
        Dict of {f"eval_level{N}_top-{K}": accuracy, ...}.
    """
    import pandas as pd

    from .predict import HierarchicalCOICOPPredictor
    from .evaluation.topk_accuracy import (
        compute_topk_accuracy,
        detect_levels,
        detect_max_k,
        ensure_true_labels,
    )

    logger.info(f"Running evaluation on {eval_data_path}...")

    # Load eval data
    eval_path = Path(eval_data_path)
    if eval_path.suffix == ".parquet":
        eval_df = pd.read_parquet(eval_path)
    else:
        eval_df = pd.read_csv(eval_path)

    logger.info(f"Loaded {len(eval_df)} evaluation samples")

    # Keep only the columns needed for prediction and evaluation
    keep_cols = [eval_text_column, eval_code_column]
    if eval_filter_columns:
        keep_cols += [c for c in eval_filter_columns if c in eval_df.columns]
    eval_df = eval_df[[c for c in keep_cols if c in eval_df.columns]]

    if classifier_type == "hierarchical":
        predictor = HierarchicalCOICOPPredictor(model_path)
        result_df = predictor.predict_dataframe(
            eval_df,
            text_column=eval_text_column,
            top_k=eval_top_k,
            beam_size=eval_beam_size,
        )
    elif classifier_type == "multihead":
        from .predict import MultiHeadCOICOPPredictor

        predictor = MultiHeadCOICOPPredictor(model_path)
        result_df = predictor.predict_dataframe(
            eval_df,
            text_column=eval_text_column,
            top_k=eval_top_k,
        )
    elif classifier_type == "basic":
        from .predict import BasicCOICOPPredictor

        predictor = BasicCOICOPPredictor(model_path)
        result_df = predictor.predict_dataframe(
            eval_df,
            text_column=eval_text_column,
            top_k=eval_top_k,
        )
    else:
        raise ValueError(f"Unknown classifier_type: {classifier_type}")

    # Ensure true label columns exist
    result_df = ensure_true_labels(result_df, eval_code_column)

    # Compute top-k accuracy per level
    levels = detect_levels(list(result_df.columns))
    ks = list(range(1, eval_top_k + 1))
    eval_metrics: dict[str, float] = {}

    for level in levels:
        max_k = detect_max_k(list(result_df.columns), level)
        row = compute_topk_accuracy(result_df, level, ks, max_k)
        for k in ks:
            key = f"top-{k}"
            if key in row and not (
                isinstance(row[key], float) and row[key] != row[key]
            ):
                eval_metrics[f"eval_level{level}_top-{k}"] = row[key]

    # Per-filter-column metrics
    if eval_filter_columns:
        for col in eval_filter_columns:
            if col not in result_df.columns:
                logger.warning(
                    f"Filter column '{col}' not found in eval data, skipping"
                )
                continue
            for val, label in [(True, "true"), (False, "false")]:
                subset = result_df[result_df[col].astype(bool) == val]
                if len(subset) == 0:
                    logger.warning(f"No rows with {col}={val}, skipping")
                    continue
                for level in levels:
                    max_k = detect_max_k(list(subset.columns), level)
                    row = compute_topk_accuracy(subset, level, ks, max_k)
                    for k in ks:
                        key = f"top-{k}"
                        if key in row and not (
                            isinstance(row[key], float) and row[key] != row[key]
                        ):
                            eval_metrics[f"eval_{col}_{label}_level{level}_top-{k}"] = (
                                row[key]
                            )
                eval_metrics[f"eval_{col}_{label}_N"] = len(subset)

    logger.info(f"Evaluation metrics: {eval_metrics}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        artifact_file = Path(tmp_dir) / "eval_predictions.parquet"
        result_df.to_parquet(artifact_file, index=False)
        mlflow.log_artifact(str(artifact_file), artifact_path="evaluation")
    logger.info("Evaluation predictions logged as MLflow artifact")

    return eval_metrics


def _finalize_mlflow_run(
    classifier,
    model_path: Path,
    classifier_type: str,
    pyfunc_module: str,
    eval_data_path: str | None,
    eval_text_column: str,
    eval_top_k: int,
    eval_filter_columns: list[str] | None,
    eval_code_column: str = "code",
    eval_beam_size: int = 1,
) -> None:
    """Run post-training evaluation, log artifacts/pyfunc, and end the MLflow run."""
    if eval_data_path:
        eval_prediction_type = (
            "greedy" if eval_beam_size <= 1 else f"beam_{eval_beam_size}"
        )
        mlflow.log_param("eval_prediction_type", eval_prediction_type)
        eval_metrics = _evaluate_on_annotations(
            classifier=classifier,
            model_path=model_path,
            classifier_type=classifier_type,
            eval_data_path=eval_data_path,
            eval_text_column=eval_text_column,
            eval_top_k=eval_top_k,
            eval_filter_columns=eval_filter_columns,
            eval_code_column=eval_code_column,
            eval_beam_size=eval_beam_size,
        )
        mlflow.log_metrics(eval_metrics)

    try:
        mlflow.log_artifacts(str(model_path), artifact_path="model")
    except Exception as exc:
        logger.warning("Failed to upload model artifacts to MLflow: %s", exc)

    try:
        mlflow.pyfunc.log_model(
            name="pyfunc_model",
            python_model=pyfunc_module,
            artifacts={
                "model_dir": str(model_path),
                "stopwords": "data/text/stopwords.json",
            },
        )
    except Exception as exc:
        logger.warning("Failed to log pyfunc model to MLflow: %s", exc)

    mlflow.end_run()


def train_hierarchical_classifier(
    annotations_path: str,
    output_dir: str,
    ngram_min_n: int = 3,
    ngram_max_n: int = 6,
    ngram_num_tokens: int = 100000,
    embedding_dim: int = 50,
    parent_embedding_dim: int = 32,
    max_seq_length: int = 64,
    batch_size: int = 32,
    lr: float = 0.1,
    num_epochs: int = 20,
    patience: int = 5,
    min_samples: int = 50,
    use_parent_features: bool = True,
    teacher_forcing_ratio: float = 0.9,
    mlflow_experiment: str | None = None,
    eval_data_path: str | None = None,
    eval_top_k: int = 5,
    eval_text_column: str = "text",
    eval_filter_columns: list[str] | None = None,
    eval_code_column: str = "code",
    eval_beam_size: int = 1,
    preprocess: bool = True,
    code_column: str = "code",
    resume_from: bool = False,
    encryption_key: str | None = None,
    max_level: int = 5,
    num_workers: int = 0,
    pin_memory: bool = True,
    tokenizer_name: str | None = None,
) -> HierarchicalCOICOPClassifier:
    """Train the hierarchical multi-level COICOP classifier.

    This classifier trains separate models for each COICOP level (1-5),
    with each level receiving predictions from the parent level as
    categorical features.

    Args:
        annotations_path: Path to annotations.parquet file
        output_dir: Directory to save trained models
        ngram_min_n: Minimum n-gram size for tokenizer
        ngram_max_n: Maximum n-gram size for tokenizer
        ngram_num_tokens: Vocabulary size for n-gram tokenizer
        embedding_dim: Text embedding dimension
        max_seq_length: Maximum sequence length
        batch_size: Training batch size
        lr: Learning rate
        num_epochs: Maximum epochs per classifier
        patience: Early stopping patience
        min_samples: Minimum samples per level
        use_parent_features: Whether to use parent predictions as features
        teacher_forcing_ratio: Ratio of ground truth to use during training
        mlflow_experiment: MLflow experiment name (optional)
        eval_data_path: Path to evaluation data for post-training metrics
        eval_top_k: Maximum K for top-k accuracy evaluation
        eval_text_column: Text column name in evaluation data
        resume_from: Whether to resume from a previous checkpoint in output_dir

    Returns:
        Trained HierarchicalCOICOPClassifier
    """
    _all_args = dict(locals())

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Compute model path early (used for checkpoint and resume)
    model_path = output_path / "hierarchical_model"
    resume_path = str(model_path) if resume_from else None

    # Load data
    logger.info(f"Loading annotations from {annotations_path}...")
    df = load_annotations(
        annotations_path,
        exclude_technical=True,
        encryption_key=encryption_key,
        preprocess=preprocess,
        code_column=code_column,
    )
    logger.info(f"Loaded {len(df)} samples (excluding 98.x and 99.x codes)")

    # Log data statistics
    unique_codes = df[code_column].nunique()
    unique_level1 = df["level1"].nunique()
    logger.info(f"Unique codes: {unique_codes}")
    logger.info(f"Level 1 categories: {unique_level1}")

    # Try to reuse MLflow run on resume
    saved_mlflow_run_id = None
    if resume_from and model_path.exists():
        import pickle

        meta_file = model_path / "hierarchical_metadata.pkl"
        if meta_file.exists():
            with open(meta_file, "rb") as f:
                saved_meta = pickle.load(f)
            saved_mlflow_run_id = saved_meta.get("mlflow_run_id")

    # Initialize MLflow if experiment name provided
    mlflow_run_info = None
    if mlflow_experiment:
        mlflow.set_experiment(mlflow_experiment)
        if saved_mlflow_run_id:
            logger.info(f"Resuming MLflow run {saved_mlflow_run_id}")
            mlflow.start_run(run_id=saved_mlflow_run_id)
        else:
            mlflow.start_run()
        _log_all_params(_all_args)
        mlflow.log_params(
            {
                "classifier_type": "hierarchical",
                "num_samples": len(df),
                "unique_codes": unique_codes,
                "unique_level1": unique_level1,
            }
        )

        run_id = mlflow.active_run().info.run_id
        mlflow_run_info = {
            "experiment_name": mlflow_experiment,
            "run_id": run_id,
            "tracking_uri": mlflow.get_tracking_uri(),
        }

    # Create config
    config = HierarchicalConfig(
        ngram_min_n=ngram_min_n,
        ngram_max_n=ngram_max_n,
        ngram_num_tokens=ngram_num_tokens,
        embedding_dim=embedding_dim,
        parent_embedding_dim=parent_embedding_dim,
        max_seq_length=max_seq_length,
        batch_size=batch_size,
        lr=lr,
        num_epochs=num_epochs,
        patience=patience,
        min_samples_per_level=min_samples,
        use_parent_features=use_parent_features,
        teacher_forcing_ratio=teacher_forcing_ratio,
        max_level=max_level,
        num_workers=num_workers,
        pin_memory=pin_memory,
        tokenizer_name=tokenizer_name,
    )

    # Create and train classifier
    classifier = HierarchicalCOICOPClassifier(config=config)

    logger.info("Starting hierarchical classifier training...")
    metrics = classifier.train(
        df=df,
        text_column="text",
        code_column=code_column,
        save_dir=str(output_path / "checkpoints"),
        mlflow_run_info=mlflow_run_info,
        resume_from=resume_path,
        checkpoint_path=str(model_path),
    )

    # Log metrics to MLflow
    if mlflow_experiment:
        for level_name, level_metrics in metrics.items():
            mlflow.log_metrics(
                {
                    f"{level_name}_num_classes": level_metrics["num_classes"],
                    f"{level_name}_train_samples": level_metrics["train_samples"],
                    f"{level_name}_val_samples": level_metrics["val_samples"],
                }
            )

    # Save the complete classifier (final save with MLflow run_id)
    mlflow_run_id = mlflow.active_run().info.run_id if mlflow_experiment else None
    classifier.save(model_path, mlflow_run_id=mlflow_run_id)
    logger.info(f"Model saved to {model_path}")

    if mlflow_experiment:
        _finalize_mlflow_run(
            classifier,
            model_path,
            "hierarchical",
            "src/mlflow_model_hierarchical.py",
            eval_data_path,
            eval_text_column,
            eval_top_k,
            eval_filter_columns,
            eval_code_column,
            eval_beam_size,
        )

    return classifier


def fine_tune_hierarchical_classifier(
    model_path: str,
    annotations_path: str,
    output_dir: str,
    levels: list[str] | None = None,
    lr: float | None = None,
    num_epochs: int | None = None,
    batch_size: int | None = None,
    patience: int | None = None,
    teacher_forcing_ratio: float | None = None,
    mlflow_experiment: str | None = None,
    eval_data_path: str | None = None,
    eval_top_k: int = 5,
    eval_text_column: str = "text",
    eval_filter_columns: list[str] | None = None,
    eval_code_column: str = "code",
    eval_beam_size: int = 1,
    preprocess: bool = True,
    code_column: str = "code",
    encryption_key: str | None = None,
    max_level: int | None = None,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
) -> HierarchicalCOICOPClassifier:
    """Fine-tune a pre-trained hierarchical classifier on new data.

    Args:
        model_path: Path to the pre-trained hierarchical model directory.
        annotations_path: Path to new training data (parquet or csv).
        output_dir: Directory to save the fine-tuned model.
        levels: List of level names to fine-tune (None = all).
        lr: Learning rate override.
        num_epochs: Number of epochs override.
        batch_size: Batch size override.
        patience: Early stopping patience override.
        teacher_forcing_ratio: Teacher forcing ratio override.
        mlflow_experiment: MLflow experiment name (optional).
        eval_data_path: Path to evaluation data for post-training metrics.
        eval_top_k: Maximum K for top-k accuracy evaluation.
        eval_text_column: Text column name in evaluation data.
        eval_filter_columns: Boolean columns to compute separate metrics for.

    Returns:
        Fine-tuned HierarchicalCOICOPClassifier.
    """
    _all_args = dict(locals())

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load pre-trained model
    logger.info(f"Loading pre-trained model from {model_path}...")
    classifier = HierarchicalCOICOPClassifier.load(model_path)

    # Override max_level if provided
    if max_level is not None:
        classifier.config.max_level = max_level

    # Override DataLoader settings if provided
    if num_workers is not None:
        classifier.config.num_workers = num_workers
    if pin_memory is not None:
        classifier.config.pin_memory = pin_memory

    # Load new data
    logger.info(f"Loading new training data from {annotations_path}...")
    df = load_annotations(
        annotations_path,
        exclude_technical=True,
        encryption_key=encryption_key,
        preprocess=preprocess,
        code_column=code_column,
    )
    logger.info(f"Loaded {len(df)} samples (excluding 98.x and 99.x codes)")

    # Log data statistics
    unique_codes = df[code_column].nunique()
    logger.info(f"Unique codes: {unique_codes}")

    # Initialize MLflow if experiment name provided
    mlflow_run_info = None
    if mlflow_experiment:
        mlflow.set_experiment(mlflow_experiment)
        mlflow.start_run()
        _log_all_params(_all_args)
        mlflow.log_params(
            {
                "classifier_type": "hierarchical",
                "task": "fine-tuning",
                "num_samples": len(df),
                "unique_codes": unique_codes,
            }
        )

        run_id = mlflow.active_run().info.run_id
        mlflow_run_info = {
            "experiment_name": mlflow_experiment,
            "run_id": run_id,
            "tracking_uri": mlflow.get_tracking_uri(),
        }

    # Fine-tune
    ft_model_path = output_path / "hierarchical_model"
    logger.info("Starting fine-tuning...")
    metrics = classifier.fine_tune(
        df=df,
        text_column="text",
        code_column=code_column,
        save_dir=str(output_path / "checkpoints"),
        mlflow_run_info=mlflow_run_info,
        levels=levels,
        lr=lr,
        num_epochs=num_epochs,
        batch_size=batch_size,
        patience=patience,
        teacher_forcing_ratio=teacher_forcing_ratio,
        checkpoint_path=str(ft_model_path),
    )

    # Log metrics to MLflow
    if mlflow_experiment:
        for level_name, level_metrics in metrics.items():
            mlflow.log_metrics(
                {
                    f"ft_{level_name}_num_classes": level_metrics["num_classes"],
                    f"ft_{level_name}_train_samples": level_metrics["train_samples"],
                    f"ft_{level_name}_val_samples": level_metrics["val_samples"],
                    f"ft_{level_name}_n_dropped": level_metrics["n_dropped"],
                }
            )

    # Save the fine-tuned model
    ft_model_path = output_path / "hierarchical_model"
    classifier.save(ft_model_path)
    logger.info(f"Fine-tuned model saved to {ft_model_path}")

    if mlflow_experiment:
        _finalize_mlflow_run(
            classifier,
            ft_model_path,
            "hierarchical",
            "src/mlflow_model_hierarchical.py",
            eval_data_path,
            eval_text_column,
            eval_top_k,
            eval_filter_columns,
            eval_code_column,
            eval_beam_size,
        )

    return classifier


def train_basic_classifier(
    data_path: str,
    output_dir: str,
    ngram_min_n: int = 3,
    ngram_max_n: int = 6,
    ngram_num_tokens: int = 100_000,
    embedding_dim: int = 128,
    max_seq_length: int = 64,
    batch_size: int = 32,
    lr: float = 0.1,
    num_epochs: int = 20,
    patience: int = 5,
    text_column: str = "product",
    code_column: str = "code8",
    mlflow_experiment: str | None = None,
    eval_data_path: str | None = None,
    eval_top_k: int = 5,
    eval_text_column: str = "product",
    eval_filter_columns: list[str] | None = None,
    eval_code_column: str = "code",
    preprocess: bool = False,
    encryption_key: str | None = None,
    tokenizer_name: str | None = None,
) -> BasicCOICOPClassifier:
    """Train the basic flat COICOP classifier.

    Reads a preprocessed parquet file directly (e.g. from build-training-data)
    and trains a single flat classifier on all COICOP codes.

    Args:
        data_path: Path to training parquet (product, code columns).
        output_dir: Directory to save trained model.
        ngram_min_n: Minimum n-gram size for tokenizer.
        ngram_max_n: Maximum n-gram size for tokenizer.
        ngram_num_tokens: Vocabulary size for n-gram tokenizer.
        embedding_dim: Text embedding dimension.
        max_seq_length: Maximum sequence length.
        batch_size: Training batch size.
        lr: Learning rate.
        num_epochs: Maximum epochs.
        patience: Early stopping patience.
        mlflow_experiment: MLflow experiment name (optional).
        eval_data_path: Path to evaluation data for post-training metrics.
        eval_top_k: Maximum K for top-k accuracy evaluation.
        eval_text_column: Text column name in evaluation data.

    Returns:
        Trained BasicCOICOPClassifier.
    """
    _all_args = dict(locals())

    from .preprocessing.data_preparation import read_parquet

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info(f"Loading training data from {data_path}...")
    df = read_parquet(data_path, encryption_key)
    if preprocess:
        import json
        from .preprocessing.data_preparation import preprocess_text

        with open("data/text/stopwords.json", "r", encoding="utf-8") as f:
            stopwords = json.load(f)
        df = preprocess_text(df, text_column, stopwords)
    logger.info(f"Loaded {len(df)} samples")

    unique_codes = df[code_column].nunique()
    logger.info(f"Unique codes: {unique_codes}")

    # Initialize MLflow if experiment name provided
    trainer_params = None
    if mlflow_experiment:
        mlflow.set_experiment(mlflow_experiment)
        mlflow.start_run()
        _log_all_params(_all_args)
        mlflow.log_params(
            {
                "classifier_type": "basic",
                "num_samples": len(df),
                "unique_codes": unique_codes,
            }
        )

        from .tracking.mlflow_utils import make_trainer_params

        run_id = mlflow.active_run().info.run_id
        trainer_params = make_trainer_params(
            experiment_name=mlflow_experiment,
            run_id=run_id,
            tracking_uri=mlflow.get_tracking_uri(),
        )

    # Create config and classifier
    config = BasicConfig(
        ngram_min_n=ngram_min_n,
        ngram_max_n=ngram_max_n,
        ngram_num_tokens=ngram_num_tokens,
        embedding_dim=embedding_dim,
        max_seq_length=max_seq_length,
        batch_size=batch_size,
        lr=lr,
        num_epochs=num_epochs,
        patience=patience,
        tokenizer_name=tokenizer_name,
    )
    classifier = BasicCOICOPClassifier(config=config)

    logger.info("Starting basic classifier training...")
    metrics = classifier.train(
        df=df,
        text_column=text_column,
        code_column=code_column,
        save_dir=str(output_path / "checkpoints"),
        trainer_params=trainer_params,
    )

    # Save model
    model_path = output_path / "basic_model"
    classifier.save(model_path)
    logger.info(f"Model saved to {model_path}")

    if mlflow_experiment:
        mlflow.log_metrics(
            {
                "num_classes": metrics["num_classes"],
                "train_samples": metrics["train_samples"],
                "val_samples": metrics["val_samples"],
                "dropped_samples": metrics["dropped_samples"],
            }
        )

        _finalize_mlflow_run(
            classifier,
            model_path,
            "basic",
            "src/mlflow_model_basic.py",
            eval_data_path,
            eval_text_column,
            eval_top_k,
            eval_filter_columns,
            eval_code_column,
        )

    return classifier


def train_multihead_classifier(
    annotations_path: str,
    output_dir: str,
    ngram_min_n: int = 3,
    ngram_max_n: int = 6,
    ngram_num_tokens: int = 100_000,
    embedding_dim: int = 128,
    max_seq_length: int = 64,
    n_attention_layers: int = 2,
    n_attention_heads: int = 4,
    n_kv_heads: int = 4,
    n_label_attention_heads: int = 4,
    batch_size: int = 32,
    lr: float = 0.01,
    num_epochs: int = 20,
    patience: int = 5,
    min_samples: int = 50,
    max_level: int = 4,
    loss_weights: list[float] | None = None,
    mlflow_experiment: str | None = None,
    eval_data_path: str | None = None,
    eval_top_k: int = 5,
    eval_text_column: str = "text",
    eval_filter_columns: list[str] | None = None,
    eval_code_column: str = "code",
    preprocess: bool = True,
    code_column: str = "code",
    encryption_key: str | None = None,
    num_workers: int = 0,
    pin_memory: bool = True,
    tokenizer_name: str | None = None,
) -> MultiHeadCOICOPClassifier:
    """Train the multi-head COICOP classifier with shared backbone.

    Args:
        annotations_path: Path to training data (parquet or csv).
        output_dir: Directory to save trained model.
        ngram_min_n: Minimum n-gram size for tokenizer.
        ngram_max_n: Maximum n-gram size for tokenizer.
        ngram_num_tokens: Vocabulary size for n-gram tokenizer.
        embedding_dim: Text embedding dimension.
        max_seq_length: Maximum sequence length.
        n_attention_layers: Number of transformer blocks in shared backbone.
        n_attention_heads: Heads per self-attention layer.
        n_kv_heads: KV heads (GQA if < n_attention_heads).
        n_label_attention_heads: Heads in each LabelAttentionClassifier.
        batch_size: Training batch size.
        lr: Learning rate.
        num_epochs: Maximum epochs.
        patience: Early stopping patience.
        min_samples: Minimum samples per level.
        max_level: Number of COICOP levels to train (1-5).
        loss_weights: Per-level loss weights (defaults to equal).
        mlflow_experiment: MLflow experiment name (optional).
        eval_data_path: Path to evaluation data for post-training metrics.
        eval_top_k: Maximum K for top-k accuracy evaluation.
        eval_text_column: Text column name in evaluation data.
        encryption_key: Parquet encryption key.
        num_workers: DataLoader workers.
        pin_memory: Pin memory for faster CPU->GPU transfer.

    Returns:
        Trained MultiHeadCOICOPClassifier.
    """
    _all_args = dict(locals())

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info(f"Loading annotations from {annotations_path}...")
    df = load_annotations(
        annotations_path,
        exclude_technical=True,
        encryption_key=encryption_key,
        preprocess=preprocess,
        code_column=code_column,
    )
    logger.info(f"Loaded {len(df)} samples (excluding 98.x and 99.x codes)")

    unique_codes = df[code_column].nunique()
    unique_level1 = df["level1"].nunique()
    logger.info(f"Unique codes: {unique_codes}")
    logger.info(f"Level 1 categories: {unique_level1}")

    # Initialize MLflow
    mlflow_run_info = None
    if mlflow_experiment:
        mlflow.set_experiment(mlflow_experiment)
        mlflow.start_run()
        _log_all_params(_all_args)
        mlflow.log_params(
            {
                "classifier_type": "multihead",
                "num_samples": len(df),
                "unique_codes": unique_codes,
                "unique_level1": unique_level1,
            }
        )

        run_id = mlflow.active_run().info.run_id
        mlflow_run_info = {
            "experiment_name": mlflow_experiment,
            "run_id": run_id,
            "tracking_uri": mlflow.get_tracking_uri(),
        }

    # Create config
    config = MultiHeadConfig(
        ngram_min_n=ngram_min_n,
        ngram_max_n=ngram_max_n,
        ngram_num_tokens=ngram_num_tokens,
        embedding_dim=embedding_dim,
        max_seq_length=max_seq_length,
        n_attention_layers=n_attention_layers,
        n_attention_heads=n_attention_heads,
        n_kv_heads=n_kv_heads,
        n_label_attention_heads=n_label_attention_heads,
        batch_size=batch_size,
        lr=lr,
        num_epochs=num_epochs,
        patience=patience,
        max_level=max_level,
        loss_weights=loss_weights,
        min_samples_per_level=min_samples,
        num_workers=num_workers,
        pin_memory=pin_memory,
        tokenizer_name=tokenizer_name,
    )

    # Create and train classifier
    classifier = MultiHeadCOICOPClassifier(config=config)

    logger.info("Starting multi-head classifier training...")
    model_path = output_path / "multihead_model"
    metrics = classifier.train(
        df=df,
        text_column="text",
        code_column=code_column,
        save_dir=str(output_path / "checkpoints"),
        mlflow_run_info=mlflow_run_info,
    )

    # Log metrics to MLflow
    if mlflow_experiment:
        for level_name, level_metrics in metrics["levels"].items():
            mlflow.log_metrics(
                {
                    f"{level_name}_num_classes": level_metrics["num_classes"],
                    f"{level_name}_train_samples": level_metrics["train_samples"],
                    f"{level_name}_val_samples": level_metrics["val_samples"],
                }
            )

    # Save the classifier
    mlflow_run_id = mlflow.active_run().info.run_id if mlflow_experiment else None
    classifier.save(model_path, mlflow_run_id=mlflow_run_id)
    logger.info(f"Model saved to {model_path}")

    if mlflow_experiment:
        _finalize_mlflow_run(
            classifier,
            model_path,
            "multihead",
            "src/mlflow_model_multihead.py",
            eval_data_path,
            eval_text_column,
            eval_top_k,
            eval_filter_columns,
            eval_code_column,
        )

    return classifier


def fine_tune_basic_classifier(
    model_path: str,
    data_path: str,
    output_dir: str,
    lr: float | None = None,
    num_epochs: int | None = None,
    batch_size: int | None = None,
    patience: int | None = None,
    code_column: str = "code8",
    text_column: str = "product",
    mlflow_experiment: str | None = None,
    eval_data_path: str | None = None,
    eval_top_k: int = 5,
    eval_text_column: str = "product",
    eval_filter_columns: list[str] | None = None,
    eval_code_column: str = "code",
    preprocess: bool = False,
    encryption_key: str | None = None,
) -> BasicCOICOPClassifier:
    """Fine-tune a pre-trained basic classifier on new data.

    Args:
        model_path: Path to the pre-trained basic model directory.
        data_path: Path to new training data (parquet).
        output_dir: Directory to save the fine-tuned model.
        lr: Learning rate override (default: original lr / 10).
        num_epochs: Number of epochs override (default: 5).
        batch_size: Batch size override.
        patience: Early stopping patience override (default: 3).
        code_column: Name of the COICOP code column.
        mlflow_experiment: MLflow experiment name (optional).
        eval_data_path: Path to evaluation data for post-training metrics.
        eval_top_k: Maximum K for top-k accuracy evaluation.
        eval_text_column: Text column name in evaluation data.
        eval_filter_columns: Boolean columns to compute separate metrics for.
        encryption_key: Parquet encryption key.

    Returns:
        Fine-tuned BasicCOICOPClassifier.
    """
    _all_args = dict(locals())

    from .preprocessing.data_preparation import read_parquet

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load pre-trained model
    logger.info(f"Loading pre-trained basic model from {model_path}...")
    classifier = BasicCOICOPClassifier.load(model_path)

    # Load new data
    logger.info(f"Loading training data from {data_path}...")
    df = read_parquet(data_path, encryption_key)
    if preprocess:
        import json
        from .preprocessing.data_preparation import preprocess_text

        with open("data/text/stopwords.json", "r", encoding="utf-8") as f:
            stopwords = json.load(f)
        df = preprocess_text(df, "product", stopwords)
    logger.info(f"Loaded {len(df)} samples")

    unique_codes = df[code_column].nunique()
    logger.info(f"Unique codes: {unique_codes}")

    # Initialize MLflow if experiment name provided
    trainer_params = None
    if mlflow_experiment:
        mlflow.set_experiment(mlflow_experiment)
        mlflow.start_run()
        _log_all_params(_all_args)
        mlflow.log_params(
            {
                "classifier_type": "basic",
                "task": "fine-tuning",
                "num_samples": len(df),
                "unique_codes": unique_codes,
            }
        )

        from .tracking.mlflow_utils import make_trainer_params

        run_id = mlflow.active_run().info.run_id
        trainer_params = make_trainer_params(
            experiment_name=mlflow_experiment,
            run_id=run_id,
            tracking_uri=mlflow.get_tracking_uri(),
        )

    # Fine-tune
    logger.info("Starting fine-tuning...")
    metrics = classifier.fine_tune(
        df=df,
        text_column=text_column,
        code_column=code_column,
        save_dir=str(output_path / "checkpoints"),
        trainer_params=trainer_params,
        lr=lr,
        num_epochs=num_epochs,
        batch_size=batch_size,
        patience=patience,
    )

    # Save the fine-tuned model
    ft_model_path = output_path / "basic_model"
    classifier.save(ft_model_path)
    logger.info(f"Fine-tuned model saved to {ft_model_path}")

    if mlflow_experiment:
        mlflow.log_metrics(
            {
                "ft_train_samples": metrics["train_samples"],
                "ft_val_samples": metrics["val_samples"],
                "ft_dropped_samples": metrics["dropped_samples"],
            }
        )

        _finalize_mlflow_run(
            classifier,
            ft_model_path,
            "basic",
            "src/mlflow_model_basic.py",
            eval_data_path,
            eval_text_column,
            eval_top_k,
            eval_filter_columns,
            eval_code_column,
        )

    return classifier
