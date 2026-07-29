"""Evaluation report module for COICOP classifier on annotated data."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from ..preprocessing.data_preparation import extract_levels, preprocess_text

logger = logging.getLogger(__name__)

# Column mappings per annotated file
_FILE_CONFIGS = {
    "ajouts_manuels_application": {
        "sep": ";",
        "text_col": "product",
        "code_col": "code1",
        "amount_col": "price",
        "store_type_col": "store_type",
        "has_hypermarche": False,
    },
    "depenses_manuelles_carnets": {
        "sep": ",",
        "text_col": "Nature de la dépense",
        "code_col": "coicop",
        "amount_col": "Montant",
        "store_type_col": "Enseigne ou type de magasin",
        "has_hypermarche": False,
    },
    "tickets_application": {
        "sep": ";",
        "text_col": "product",
        "code_col": "Code coicop",
        "amount_col": "Montant de la dépense",
        "store_type_col": "Nomen_mag",
        "has_hypermarche": True,
    },
}


def _detect_file_config(columns: list[str]) -> dict | None:
    """Detect which annotated file format based on columns present."""
    cols = set(columns)
    if "code1" in cols:
        return _FILE_CONFIGS["ajouts_manuels_application"]
    if "Nature de la dépense" in cols:
        return _FILE_CONFIGS["depenses_manuelles_carnets"]
    if "Code coicop" in cols:
        return _FILE_CONFIGS["tickets_application"]
    return None


def _parse_amount(s: str | float) -> float | None:
    """Convert amount string to float, handling comma decimals."""
    if pd.isna(s):
        return None
    s = str(s).strip()
    if not s:
        return None
    # Replace comma decimal separator with dot
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def load_annotated_file(path: str | Path) -> pd.DataFrame:
    """Load a single annotated CSV and normalize columns.

    Returns DataFrame with columns: product, code, amount, is_hypermarche, source_file.
    """
    path = Path(path)
    stem = path.stem

    # Try known config first, fall back to auto-detection
    if stem in _FILE_CONFIGS:
        config = _FILE_CONFIGS[stem]
    else:
        # Read first to detect
        probe = pd.read_csv(path, nrows=0, sep=";", encoding="utf-8")
        config = _detect_file_config(list(probe.columns))
        if config is None:
            probe = pd.read_csv(path, nrows=0, sep=",", encoding="utf-8")
            config = _detect_file_config(list(probe.columns))
        if config is None:
            raise ValueError(f"Cannot detect format for {path}")

    df = pd.read_csv(path, sep=config["sep"], encoding="utf-8", dtype=str)

    # Normalize columns
    result = pd.DataFrame()
    result["product"] = df[config["text_col"]]
    result["code"] = df[config["code_col"]]
    result["amount"] = df[config["amount_col"]].apply(_parse_amount)
    result["source_file"] = stem

    # Store type / hypermarche
    if config["has_hypermarche"]:
        result["is_hypermarche"] = df[config["store_type_col"]] == "Hypermarchés"
    else:
        result["is_hypermarche"] = None

    # Filter out rows with empty/null text or code
    result = result.dropna(subset=["product", "code"])
    result = result[result["product"].str.strip().astype(bool)]
    result = result[result["code"].str.strip().astype(bool)]

    logger.info(f"Loaded {len(result)} rows from {path.name}")
    return result.reset_index(drop=True)


def load_all_annotated(directory: str | Path) -> pd.DataFrame:
    """Load and concatenate all annotated CSV files from a directory."""
    directory = Path(directory)
    dfs = []
    for csv_file in sorted(directory.glob("*.csv")):
        try:
            dfs.append(load_annotated_file(csv_file))
        except Exception as e:
            logger.warning(f"Skipping {csv_file.name}: {e}")

    if not dfs:
        raise FileNotFoundError(f"No annotated CSV files found in {directory}")

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total annotated samples: {len(combined)}")
    return combined


class _UniformPredictor:
    """Wrapper providing uniform predict_dataframe interface."""

    def __init__(self, predictor, is_pyfunc: bool = False):
        self._predictor = predictor
        self._is_pyfunc = is_pyfunc

    @property
    def needs_preprocessing(self) -> bool:
        return not self._is_pyfunc

    def predict_dataframe(
        self, df: pd.DataFrame, text_column: str = "product", top_k: int = 1
    ) -> pd.DataFrame:
        if self._is_pyfunc:
            # MLflow pyfunc: expects DataFrame with "text" column
            input_df = pd.DataFrame({"text": df[text_column]})
            preds = self._predictor.predict(input_df)
            result_df = df.copy()
            # pyfunc returns a DataFrame or array with predictions
            if isinstance(preds, pd.DataFrame):
                for col in preds.columns:
                    result_df[col] = preds[col].values
            else:
                result_df["predicted_code"] = preds
            return result_df
        else:
            return self._predictor.predict_dataframe(
                df, text_column=text_column, top_k=top_k
            )


def load_predictor(model_path: str | Path) -> _UniformPredictor:
    """Load a predictor from a local model directory or MLflow URI.

    Returns a _UniformPredictor with uniform predict_dataframe interface.
    """
    model_path_str = str(model_path)

    # MLflow URI
    if model_path_str.startswith("runs:/") or model_path_str.startswith("models:/"):
        import mlflow

        pyfunc_model = mlflow.pyfunc.load_model(model_path_str)
        logger.info(f"Loaded MLflow pyfunc model from {model_path_str}")
        return _UniformPredictor(pyfunc_model, is_pyfunc=True)

    # Local model
    model_dir = Path(model_path)
    if (model_dir / "basic_metadata.pkl").exists():
        from ..predict import BasicCOICOPPredictor

        predictor = BasicCOICOPPredictor(model_dir)
        return _UniformPredictor(predictor, is_pyfunc=False)

    if (model_dir / "hierarchical_metadata.pkl").exists():
        from ..predict import HierarchicalCOICOPPredictor

        predictor = HierarchicalCOICOPPredictor(model_dir)
        return _UniformPredictor(predictor, is_pyfunc=False)

    raise FileNotFoundError(
        f"No recognized model found at {model_dir}. "
        "Expected basic_metadata.pkl or hierarchical_metadata.pkl."
    )


def _compute_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    """Compute accuracy and weighted F1 for a set of predictions."""
    n = len(y_true)
    if n == 0:
        return {"accuracy": float("nan"), "f1_weighted": float("nan"), "n": 0}
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    return {"accuracy": acc, "f1_weighted": f1, "n": n}


def _compute_topk_accuracy(
    true_codes: list[str],
    pred_top1: list[str],
    pred_alternatives: list[list[str]] | None,
    k: int,
) -> float:
    """Compute top-k accuracy.

    pred_alternatives[i] is a list of alternative codes for sample i (top-2, top-3, ...).
    """
    if not true_codes:
        return float("nan")

    hits = 0
    for i, true in enumerate(true_codes):
        candidates = [pred_top1[i]]
        if pred_alternatives and i < len(pred_alternatives):
            candidates.extend(pred_alternatives[i][: k - 1])
        if true in candidates:
            hits += 1
    return hits / len(true_codes)


def _extract_level_from_code(code: str, level: int) -> str | None:
    """Extract a specific COICOP level from a dotted code string."""
    parts = code.split(".")
    if level == 1:
        return parts[0].zfill(2) if len(parts) >= 1 else None
    if level <= len(parts):
        prefix = ".".join(parts[:level])
        # Level 1 should be zero-padded
        first = parts[0].zfill(2)
        return first + "." + ".".join(parts[1:level]) if level > 1 else first
    return None


def run_evaluation(
    model_path: str | Path,
    data_dir: str | Path,
    top_k: int = 5,
    text_column: str = "product",
    amount_threshold: float = 200.0,
) -> dict:
    """Run full evaluation on annotated data.

    Returns structured metrics dict.
    """
    # Load data
    df = load_all_annotated(data_dir)
    n_total = len(df)
    n_files = df["source_file"].nunique()

    # Load predictor
    predictor = load_predictor(model_path)

    # Preprocess text if needed (not for MLflow pyfunc models)
    if predictor.needs_preprocessing:
        stopwords_path = Path("data/text/stopwords.json")
        if stopwords_path.exists():
            with open(stopwords_path, "r", encoding="utf-8") as f:
                stopwords = json.load(f)
        else:
            stopwords = []
        df = preprocess_text(df, text_column, stopwords)

    # Run predictions
    logger.info(f"Running predictions on {len(df)} samples with top_k={top_k}...")
    result_df = predictor.predict_dataframe(df, text_column=text_column, top_k=top_k)

    # Extract true COICOP levels
    true_levels = result_df["code"].apply(extract_levels).apply(pd.Series)
    for col in true_levels.columns:
        result_df[f"true_{col}"] = true_levels[col]

    # Extract predicted levels from predicted_code
    if "predicted_code" in result_df.columns:
        pred_levels = (
            result_df["predicted_code"].apply(extract_levels).apply(pd.Series)
        )
        for col in pred_levels.columns:
            result_df[f"pred_{col}"] = pred_levels[col]

    # Build alternatives list for top-k
    alternatives = None
    if top_k > 1:
        alt_cols = [
            f"predicted_code_top{rank}" for rank in range(2, top_k + 1)
            if f"predicted_code_top{rank}" in result_df.columns
        ]
        if alt_cols:
            alternatives = []
            for _, row in result_df.iterrows():
                alts = [row[c] for c in alt_cols if pd.notna(row[c]) and str(row[c]).strip()]
                alternatives.append(alts)

    # --- Compute metrics ---
    metrics = {
        "model_path": str(model_path),
        "data_dir": str(data_dir),
        "n_files": n_files,
        "n_total": n_total,
        "n_after_preprocess": len(result_df),
        "date": datetime.now().strftime("%Y-%m-%d"),
        "top_k": top_k,
        "amount_threshold": amount_threshold,
    }

    # Overall metrics
    y_true = result_df["code"].tolist()
    y_pred = result_df["predicted_code"].tolist()
    overall = _compute_metrics(y_true, y_pred)
    overall["topk_accuracy"] = {}
    for k in range(1, top_k + 1):
        overall["topk_accuracy"][k] = _compute_topk_accuracy(
            y_true, y_pred, alternatives, k
        )
    metrics["overall"] = overall

    # By COICOP level
    level_metrics = {}
    for lvl in range(1, 6):
        true_col = f"true_level{lvl}"
        pred_col = f"pred_level{lvl}"
        if true_col in result_df.columns and pred_col in result_df.columns:
            valid = result_df[true_col].notna() & result_df[pred_col].notna()
            sub = result_df[valid]
            if len(sub) > 0:
                level_metrics[lvl] = _compute_metrics(
                    sub[true_col].tolist(), sub[pred_col].tolist()
                )
            else:
                level_metrics[lvl] = {"accuracy": float("nan"), "f1_weighted": float("nan"), "n": 0}
    metrics["by_level"] = level_metrics

    # By data source
    source_metrics = {}
    for source, group_df in result_df.groupby("source_file"):
        y_t = group_df["code"].tolist()
        y_p = group_df["predicted_code"].tolist()
        m = _compute_metrics(y_t, y_p)

        # Source-level alternatives
        src_alts = None
        if alternatives:
            src_indices = group_df.index.tolist()
            src_alts = [alternatives[i] for i in src_indices if i < len(alternatives)]

        m["topk_accuracy"] = {}
        for k in range(1, top_k + 1):
            m["topk_accuracy"][k] = _compute_topk_accuracy(y_t, y_p, src_alts, k)
        source_metrics[source] = m
    metrics["by_source"] = source_metrics

    # By spending amount
    amount_metrics = {}
    has_amount = result_df["amount"].notna()
    if has_amount.any():
        low = result_df[has_amount & (result_df["amount"] <= amount_threshold)]
        high = result_df[has_amount & (result_df["amount"] > amount_threshold)]

        for label, sub_df in [
            (f"<= {amount_threshold:.0f}€", low),
            (f"> {amount_threshold:.0f}€", high),
        ]:
            if len(sub_df) > 0:
                y_t = sub_df["code"].tolist()
                y_p = sub_df["predicted_code"].tolist()
                m = _compute_metrics(y_t, y_p)
                sub_alts = None
                if alternatives:
                    sub_indices = sub_df.index.tolist()
                    sub_alts = [alternatives[i] for i in sub_indices if i < len(alternatives)]
                m["topk_accuracy"] = {}
                for k in range(1, top_k + 1):
                    m["topk_accuracy"][k] = _compute_topk_accuracy(
                        y_t, y_p, sub_alts, k
                    )
                amount_metrics[label] = m
    metrics["by_amount"] = amount_metrics

    # By store type (only rows where is_hypermarche is not null)
    store_metrics = {}
    has_store = result_df["is_hypermarche"].notna()
    if has_store.any():
        store_df = result_df[has_store]
        hyper = store_df[store_df["is_hypermarche"] == True]  # noqa: E712
        autres = store_df[store_df["is_hypermarche"] == False]  # noqa: E712

        for label, sub_df in [("Hypermarchés", hyper), ("Autres", autres)]:
            if len(sub_df) > 0:
                y_t = sub_df["code"].tolist()
                y_p = sub_df["predicted_code"].tolist()
                m = _compute_metrics(y_t, y_p)
                sub_alts = None
                if alternatives:
                    sub_indices = sub_df.index.tolist()
                    sub_alts = [alternatives[i] for i in sub_indices if i < len(alternatives)]
                m["topk_accuracy"] = {}
                for k in range(1, top_k + 1):
                    m["topk_accuracy"][k] = _compute_topk_accuracy(
                        y_t, y_p, sub_alts, k
                    )
                store_metrics[label] = m
    metrics["by_store_type"] = store_metrics

    return metrics


def format_report(metrics: dict) -> str:
    """Format metrics dict as a readable text report."""
    lines = []
    top_k = metrics.get("top_k", 5)

    lines.append("=== COICOP Model Evaluation Report ===")
    lines.append(f"Model: {metrics['model_path']}")
    lines.append(
        f"Data:  {metrics['data_dir']} "
        f"({metrics['n_files']} files, {metrics['n_total']} total samples)"
    )
    lines.append(f"Date:  {metrics['date']}")
    lines.append("")

    # Overall
    overall = metrics["overall"]
    lines.append("--- Overall ---")
    lines.append(f"  Samples: {overall['n']}")
    if overall["topk_accuracy"]:
        for k in sorted(overall["topk_accuracy"]):
            val = overall["topk_accuracy"][k]
            label = f"Top-{k} accuracy"
            lines.append(f"  {label + ':':<20s} {val * 100:.2f}%")
    lines.append(f"  {'Weighted F1:':<20s} {overall['f1_weighted']:.4f}")
    lines.append("")

    # By COICOP level
    lines.append("--- By COICOP level ---")
    for lvl, m in sorted(metrics.get("by_level", {}).items()):
        acc_str = f"{m['accuracy'] * 100:.2f}%" if not pd.isna(m["accuracy"]) else "N/A"
        lines.append(f"  Level {lvl}: acc={acc_str} ({m['n']} samples)")
    lines.append("")

    # By data source
    lines.append("--- By data source ---")
    for source, m in sorted(metrics.get("by_source", {}).items()):
        lines.append(f"  {source} ({m['n']} samples):")
        topk = m.get("topk_accuracy", {})
        parts = []
        for k in sorted(topk):
            parts.append(f"Top-{k}: {topk[k] * 100:.2f}%")
        parts.append(f"F1: {m['f1_weighted']:.4f}")
        lines.append(f"    {('  '.join(parts))}")
    lines.append("")

    # By spending amount
    threshold = metrics.get("amount_threshold", 200)
    lines.append(f"--- By spending amount (threshold: {threshold:.0f}€) ---")
    for label, m in metrics.get("by_amount", {}).items():
        lines.append(f"  {label} ({m['n']} samples):")
        topk = m.get("topk_accuracy", {})
        parts = []
        for k in sorted(topk):
            parts.append(f"Top-{k}: {topk[k] * 100:.2f}%")
        parts.append(f"F1: {m['f1_weighted']:.4f}")
        lines.append(f"    {('  '.join(parts))}")
    if not metrics.get("by_amount"):
        lines.append("  (no amount data available)")
    lines.append("")

    # By store type
    lines.append("--- By store type ---")
    for label, m in metrics.get("by_store_type", {}).items():
        lines.append(f"  {label} ({m['n']} samples):")
        topk = m.get("topk_accuracy", {})
        parts = []
        for k in sorted(topk):
            parts.append(f"Top-{k}: {topk[k] * 100:.2f}%")
        parts.append(f"F1: {m['f1_weighted']:.4f}")
        lines.append(f"    {('  '.join(parts))}")
    if not metrics.get("by_store_type"):
        lines.append("  (no store type data available)")
    lines.append("")

    return "\n".join(lines)


def _flatten_metrics(metrics: dict, prefix: str = "eval") -> dict[str, float]:
    """Flatten structured metrics dict into MLflow-compatible flat keys."""
    flat = {}

    def _sanitize_key(key: str) -> str:
        """Make key MLflow-compatible (alphanumeric, underscores, dashes, dots)."""
        return re.sub(r"[^a-zA-Z0-9_\-.]", "_", key)

    # Overall
    overall = metrics.get("overall", {})
    flat[f"{prefix}.overall.accuracy"] = overall.get("accuracy", float("nan"))
    flat[f"{prefix}.overall.f1_weighted"] = overall.get("f1_weighted", float("nan"))
    for k, v in overall.get("topk_accuracy", {}).items():
        flat[f"{prefix}.overall.top_{k}"] = v

    # By level
    for lvl, m in metrics.get("by_level", {}).items():
        flat[f"{prefix}.level{lvl}.accuracy"] = m.get("accuracy", float("nan"))
        flat[f"{prefix}.level{lvl}.f1_weighted"] = m.get("f1_weighted", float("nan"))

    # By source
    for source, m in metrics.get("by_source", {}).items():
        key = _sanitize_key(source)
        flat[f"{prefix}.source.{key}.accuracy"] = m.get("accuracy", float("nan"))
        flat[f"{prefix}.source.{key}.f1_weighted"] = m.get("f1_weighted", float("nan"))
        for k, v in m.get("topk_accuracy", {}).items():
            flat[f"{prefix}.source.{key}.top_{k}"] = v

    # By amount
    for label, m in metrics.get("by_amount", {}).items():
        key = _sanitize_key(label)
        flat[f"{prefix}.amount.{key}.accuracy"] = m.get("accuracy", float("nan"))
        flat[f"{prefix}.amount.{key}.f1_weighted"] = m.get("f1_weighted", float("nan"))
        for k, v in m.get("topk_accuracy", {}).items():
            flat[f"{prefix}.amount.{key}.top_{k}"] = v

    # By store type
    for label, m in metrics.get("by_store_type", {}).items():
        key = _sanitize_key(label)
        flat[f"{prefix}.store.{key}.accuracy"] = m.get("accuracy", float("nan"))
        flat[f"{prefix}.store.{key}.f1_weighted"] = m.get("f1_weighted", float("nan"))
        for k, v in m.get("topk_accuracy", {}).items():
            flat[f"{prefix}.store.{key}.top_{k}"] = v

    return flat


def log_metrics_to_mlflow(
    metrics: dict,
    run_id: str | None = None,
    experiment_name: str | None = None,
) -> None:
    """Log evaluation metrics to MLflow.

    If run_id is provided, logs to that existing run.
    If experiment_name is provided without run_id, creates a new run.
    """
    import mlflow

    flat = _flatten_metrics(metrics)

    if run_id:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics(flat)
        logger.info(f"Logged {len(flat)} metrics to MLflow run {run_id}")
    elif experiment_name:
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run():
            mlflow.log_metrics(flat)
            mlflow.log_param("eval_data_dir", metrics.get("data_dir", ""))
            mlflow.log_param("eval_model_path", metrics.get("model_path", ""))
        logger.info(
            f"Logged {len(flat)} metrics to new MLflow run "
            f"in experiment '{experiment_name}'"
        )
    else:
        logger.warning("No MLflow run_id or experiment_name provided, skipping logging")
