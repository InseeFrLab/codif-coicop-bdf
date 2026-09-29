"""Evaluate a prediction file (output from predict commands) by COICOP level.
"""

from __future__ import annotations

import html
import io
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

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


# ── HTML report ───────────────────────────────────────

_HTML_STYLE = """
body { font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 1100px;
       padding: 0 1rem; color: #1f2328; background: #fff; }
h1 { font-size: 1.5rem; } h2 { font-size: 1.15rem; margin-top: 2rem; }
table { border-collapse: collapse; margin: .5rem 0 1rem; font-size: .9rem; }
th, td { border: 1px solid #d0d7de; padding: .3rem .6rem; text-align: right; }
th { background: #f6f8fa; }
td:first-child, th:first-child { text-align: left; }
.meta td { text-align: left; }
.note { color: #59636e; font-size: .85rem; }
"""


def _pct(x) -> str:
    return f"{x:.2%}" if pd.notna(x) else "-"


def _level_table_html(level_results: dict[int, dict]) -> str:
    ks = sorted({
        int(key.split("-")[1])
        for lvl in level_results.values()
        for key in lvl
        if key.startswith("top-")
    })
    head = "".join(f"<th>top-{k}</th>" for k in ks)
    rows = []
    for level in sorted(level_results):
        res = level_results[level]
        cells = "".join(f"<td>{_pct(res.get(f'top-{k}'))}</td>" for k in ks)
        rows.append(
            f"<tr><td>level{level}</td>{cells}<td>{int(res.get('N', 0)):,}</td></tr>"
        )
    return (
        f"<table><tr><th>Niveau</th>{head}<th>N</th></tr>"
        + "".join(rows)
        + "</table>"
    )


def _finest_level(results: dict) -> int | None:
    """Deepest COICOP level that has at least one evaluable row."""
    levels = [lvl for lvl, res in results["levels"].items() if res.get("N", 0) > 0]
    return max(levels) if levels else None


def format_html_report(
    results: dict,
    df: pd.DataFrame,
    code_column: str = "code",
    meta: dict[str, str] | None = None,
    n_confusions: int = 30,
) -> str:
    """Render the evaluation as a standalone HTML page.

    Same top-K tables as the text report, plus, at the finest evaluable level,
    the top-1 accuracy per true code and the most frequent confusions.
    """
    df = ensure_predicted_levels(ensure_true_labels(df, code_column=code_column))
    esc = html.escape
    parts = [
        "<!doctype html><html lang='fr'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>Rapport d'évaluation TTC</title>",
        f"<style>{_HTML_STYLE}</style></head><body>",
        "<h1>Rapport d'évaluation TTC</h1>",
    ]

    meta_rows = {
        "Fichier évalué": results.get("prediction_path", ""),
        "N": f"{results['n_samples']:,}",
        "Généré le": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        **(meta or {}),
    }
    parts.append("<table class='meta'>")
    parts += [
        f"<tr><th>{esc(str(k))}</th><td>{esc(str(v))}</td></tr>"
        for k, v in meta_rows.items()
    ]
    parts.append("</table>")

    parts.append("<h2>Accuracy par niveau COICOP</h2>")
    parts.append(_level_table_html(results["levels"]))

    if "by_category" in results:
        cat_col = results.get("categorical_column", "category")
        parts.append(f"<h2>Par {esc(cat_col)}</h2>")
        for cat_val, cat_levels in results["by_category"].items():
            n_cat = next(iter(cat_levels.values()), {}).get("N", 0)
            parts.append(f"<h3>{esc(cat_val)} (N={n_cat:,})</h3>")
            parts.append(_level_table_html(cat_levels))

    level = _finest_level(results)
    if level is not None:
        true_col, pred_col = f"level{level}", f"predicted_level{level}"
        sub = df[df[true_col].notna() & df[pred_col].notna()]
        sub = sub.assign(_hit=sub[true_col].astype(str) == sub[pred_col].astype(str))

        per_code = (
            sub.groupby(true_col)["_hit"]
            .agg(N="size", accuracy="mean")
            .sort_values(["accuracy", "N"], ascending=[True, False])
        )
        parts.append(f"<h2>Accuracy top-1 par code (level{level})</h2>")
        parts.append(
            "<p class='note'>Triée de la plus faible à la plus forte. Les lignes "
            f"dont le code vrai ou prédit n'atteint pas le niveau {level} "
            "(codes techniques courts, par ex. 98.4) n'y figurent pas.</p>"
        )
        parts.append("<table><tr><th>Code</th><th>N</th><th>Top-1</th></tr>")
        parts += [
            f"<tr><td>{esc(str(code))}</td><td>{int(r.N):,}</td>"
            f"<td>{_pct(r.accuracy)}</td></tr>"
            for code, r in per_code.iterrows()
        ]
        parts.append("</table>")

        confusions = (
            sub[~sub["_hit"]]
            .groupby([true_col, pred_col])
            .size()
            .sort_values(ascending=False)
            .head(n_confusions)
        )
        parts.append(f"<h2>Confusions les plus fréquentes (level{level})</h2>")
        parts.append("<table><tr><th>Code vrai</th><th>Code prédit</th><th>N</th></tr>")
        parts += [
            f"<tr><td>{esc(str(t))}</td><td>{esc(str(p))}</td><td>{int(n):,}</td></tr>"
            for (t, p), n in confusions.items()
        ]
        parts.append("</table>")

    parts.append("</body></html>")
    return "\n".join(parts)


def write_text_output(text: str, path: str | Path, content_type: str = "text/plain") -> None:
    """Write a text file locally or to S3 (``s3://bucket/key``, via boto3)."""
    path_str = str(path)
    if path_str.startswith("s3://"):
        import boto3

        endpoint = os.environ.get("AWS_S3_ENDPOINT") or os.environ.get("AWS_ENDPOINT_URL")
        kwargs: dict = {}
        if endpoint:
            if not endpoint.startswith("http"):
                endpoint = f"https://{endpoint}"
            kwargs["endpoint_url"] = endpoint
        parsed = urlparse(path_str)
        boto3.client("s3", **kwargs).put_object(
            Bucket=parsed.netloc,
            Key=parsed.path.lstrip("/"),
            Body=text.encode("utf-8"),
            ContentType=f"{content_type}; charset=utf-8",
        )
    else:
        Path(path_str).parent.mkdir(parents=True, exist_ok=True)
        Path(path_str).write_text(text, encoding="utf-8")
    logger.info("Saved %s", path_str)


# ── High-level entry point ────────────────────────────


def run_evaluate_predictions(
    prediction_path: str | Path,
    code_column: str = "code",
    text_column: str = "product",
    categorical_column: str | None = None,
    max_k: int = 5,
    html_output: str | Path | None = None,
    report_meta: dict[str, str] | None = None,
) -> tuple[dict, str]:
    """Read a prediction file, evaluate it, and return (results, report).

    Args:
        prediction_path: Local path or S3 URI to a parquet/CSV prediction file.
        code_column: Column holding the ground-truth COICOP code.
        text_column: Column holding the product text (used for logging only).
        categorical_column: Optional column to group results by.
        max_k: Maximum K for top-K accuracy.
        html_output: If set, also write an HTML report there (local or S3).
        report_meta: Extra key/value pairs shown in the HTML report header.

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

    if html_output:
        page = format_html_report(results, df, code_column=code_column, meta=report_meta)
        write_text_output(page, html_output, content_type="text/html")

    return results, report
