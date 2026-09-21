"""Evaluate a prediction file (output from predict commands) by COICOP level.
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path

import duckdb
import pandas as pd

from .topk_accuracy import (
    compute_topk_accuracy,
    detect_levels,
    detect_max_k,
    ensure_predicted_levels,
    ensure_true_labels,
)

logger = logging.getLogger(__name__)


# ── File reading ──────────────────────────────────────


def _configure_s3(con: duckdb.DuckDBPyConnection) -> None:
    """Configure DuckDB S3 secret from environment variables."""
    con.execute(f"""
        CREATE SECRET secret_ls3 (
            TYPE S3,
            KEY_ID '{os.environ["AWS_ACCESS_KEY_ID"]}',
            SECRET '{os.environ["AWS_SECRET_ACCESS_KEY"]}',
            ENDPOINT '{os.environ["AWS_S3_ENDPOINT"]}',
            SESSION_TOKEN '{os.environ["AWS_SESSION_TOKEN"]}',
            REGION 'us-east-1',
            URL_STYLE 'path',
            SCOPE 's3://'
        );
    """)


def read_prediction_file(path: str | Path) -> pd.DataFrame:
    """Read a local or S3 parquet/CSV prediction file."""
    path_str = str(path)
    if path_str.startswith("s3://"):
        con = duckdb.connect()
        _configure_s3(con)
        return con.execute(f"SELECT * FROM '{path_str}'").df()
    path = Path(path_str)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, sep=";")


# ── Core evaluation ───────────────────────────────────


def evaluate_predictions(
    df: pd.DataFrame,
    code_column: str = "code",
    categorical_column: str | None = None,
    max_k: int = 5,
) -> dict:
    """Compute top-K accuracy by COICOP level on a prediction DataFrame.

    Args:
        df: DataFrame produced by a predict command (must contain predicted_level{N}
            columns or a predicted_code column from which they are derived).
        code_column: Name of the column holding the ground-truth COICOP code.
        categorical_column: Optional column to group results by (e.g. a source
            or store-type flag).
        max_k: Maximum K to report for top-K accuracy.

    Returns:
        dict with keys:
            - ``n_samples``: total row count.
            - ``levels``: mapping level -> top-k accuracy dict.
            - ``categorical_column``: name of the grouping column (if provided).
            - ``by_category``: mapping category_value -> levels dict (if provided).
    """
    df = ensure_true_labels(df, code_column=code_column)
    df = ensure_predicted_levels(df)

    levels = detect_levels(df.columns.tolist())
    if not levels:
        raise ValueError(
            "No predicted_level{N} columns found. "
            "Run a predict command with --top-k >= 1 first."
        )

    # Detect the maximum K available once (based on full-dataset columns).
    cols = df.columns.tolist()
    k_avail_per_level = {lvl: detect_max_k(cols, lvl) for lvl in levels}

    def _compute(sub: pd.DataFrame) -> dict[int, dict]:
        result = {}
        for level in levels:
            k_avail = k_avail_per_level[level]
            ks = list(range(1, min(max_k, k_avail) + 1))
            result[level] = compute_topk_accuracy(sub, level, ks, k_avail)
        return result

    results: dict = {
        "n_samples": len(df),
        "levels": _compute(df),
    }

    if categorical_column:
        if categorical_column not in df.columns:
            logger.warning(
                "Category column '%s' not found — skipping breakdown. "
                "Available columns: %s",
                categorical_column,
                list(df.columns),
            )
        else:
            results["categorical_column"] = categorical_column
            results["by_category"] = {
                str(val): _compute(grp)
                for val, grp in df.groupby(categorical_column, sort=True)
            }

    return results


# ── Formatting ────────────────────────────────────────


def _level_table_str(level_results: dict[int, dict], ks: list[int]) -> str:
    """Return an aligned table of top-K accuracy per COICOP level."""
    rows = []
    for level in sorted(level_results):
        row: dict = {"Level": f"level{level}"}
        for k in ks:
            key = f"top-{k}"
            row[key] = level_results[level].get(key, float("nan"))
        row["N"] = level_results[level].get("N", 0)
        rows.append(row)

    tbl = pd.DataFrame(rows).set_index("Level")

    def _fmt_pct(x):
        return f"{x:>7.2%}" if pd.notna(x) else "      -"

    def _fmt_n(x):
        return f"{int(x):>7,d}" if pd.notna(x) else "      -"

    formatters = {
        col: (_fmt_n if col == "N" else _fmt_pct) for col in tbl.columns
    }
    buf = io.StringIO()
    buf.write(tbl.to_string(formatters=formatters))
    return buf.getvalue()


def format_report(results: dict, prediction_path: str | None = None) -> str:
    """Format evaluation results as a human-readable report string."""
    lines: list[str] = []

    path_label = prediction_path or results.get("prediction_path", "")
    n = results["n_samples"]
    lines.append(f"Evaluation: {path_label}")
    lines.append(f"N: {n:,}")

    # Collect all K values present across any level.
    ks = sorted({
        int(key.split("-")[1])
        for lvl in results["levels"].values()
        for key in lvl
        if key.startswith("top-")
    })

    sep = "─" * 62
    lines.append("")
    lines.append(sep)
    lines.append("ACCURACY BY COICOP LEVEL")
    lines.append(sep)
    lines.append(_level_table_str(results["levels"], ks))

    if "by_category" in results:
        cat_col = results.get("categorical_column", "category")
        for cat_val, cat_levels in results["by_category"].items():
            n_cat = next(iter(cat_levels.values()), {}).get("N", 0)
            lines.append("")
            lines.append(sep)
            lines.append(f"{cat_col.upper()}: {cat_val}  (N={n_cat:,})")
            lines.append(sep)
            lines.append(_level_table_str(cat_levels, ks))

    return "\n".join(lines)


# ── High-level entry point ────────────────────────────


def run_evaluate_predictions(
    prediction_path: str | Path,
    code_column: str = "code",
    text_column: str = "product",
    categorical_column: str | None = None,
    max_k: int = 5,
) -> tuple[dict, str]:
    """Read a prediction file, evaluate it, and return (results, report).

    Args:
        prediction_path: Local path or S3 URI to a parquet/CSV prediction file.
        code_column: Column holding the ground-truth COICOP code.
        text_column: Column holding the product text (used for logging only).
        categorical_column: Optional column to group results by.
        max_k: Maximum K for top-K accuracy.

    Returns:
        Tuple of (results dict, formatted report string).
    """
    logger.info("Reading prediction file: %s", prediction_path)
    df = read_prediction_file(prediction_path)
    logger.info("Loaded %d rows", len(df))

    results = evaluate_predictions(
        df,
        code_column=code_column,
        categorical_column=categorical_column,
        max_k=max_k,
    )
    results["prediction_path"] = str(prediction_path)

    report = format_report(results, prediction_path=str(prediction_path))
    return results, report
