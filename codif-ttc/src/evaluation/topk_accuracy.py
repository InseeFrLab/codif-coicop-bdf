"""Compute top-K accuracy (overall and per-code breakdown) from a predictions parquet."""

from __future__ import annotations

import argparse
import re
import sys

import pandas as pd

from ..preprocessing.data_preparation import extract_levels


def detect_levels(columns: list[str]) -> list[int]:
    """Return sorted list of level numbers that have a predicted_level{N} column."""
    levels = set()
    for col in columns:
        m = re.fullmatch(r"predicted_level(\d+)", col)
        if m:
            levels.add(int(m.group(1)))
    return sorted(levels)


def detect_max_k(columns: list[str], level: int) -> int:
    """Return the maximum K available for a given level (at least 1)."""
    max_k = 1
    for col in columns:
        m = re.fullmatch(rf"predicted_level{level}_top(\d+)", col)
        if m:
            max_k = max(max_k, int(m.group(1)))
    return max_k


def ensure_true_labels(df: pd.DataFrame, code_column: str = 'code') -> pd.DataFrame:
    """Add level1-level5 columns from `code` if they are missing."""
    if "level1" in df.columns:
        return df
    if code_column not in df.columns:
        print(f"ERROR: DataFrame has neither level1..level5 nor {code_column} column.", file=sys.stderr)
        sys.exit(1)
    levels = df[code_column].apply(extract_levels).apply(pd.Series)
    return pd.concat([df, levels], axis=1)


def ensure_predicted_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Derive predicted_level{N} columns from predicted_code if absent.

    Basic-classifier prediction files store the full code in `predicted_code`
    instead of per-level columns.  This converts them to the standard format
    expected by detect_levels / topk_hit.
    """
    if "predicted_level1" in df.columns or "predicted_code" not in df.columns:
        return df
    levels_data = [extract_levels(code) for code in df["predicted_code"]]
    for level_key in ["level1", "level2", "level3", "level4", "level5"]:
        df[f"predicted_{level_key}"] = [d.get(level_key, "") for d in levels_data]
    rank = 2
    while f"predicted_code_top{rank}" in df.columns:
        top_levels_data = [
            extract_levels(code) if code else {}
            for code in df[f"predicted_code_top{rank}"]
        ]
        for level_key in ["level1", "level2", "level3", "level4", "level5"]:
            df[f"predicted_{level_key}_top{rank}"] = [d.get(level_key, "") for d in top_levels_data]
        rank += 1
    return df


def topk_hit(df: pd.DataFrame, level: int, k: int) -> pd.Series:
    """Boolean Series: true if the true label appears in the top-k predictions.

    Top-1 prediction column is `predicted_level{N}`.
    Top-2..K columns are `predicted_level{N}_top{rank}`.
    """
    true_col = f"level{level}"
    pred_cols = [f"predicted_level{level}"]
    for rank in range(2, k + 1):
        col = f"predicted_level{level}_top{rank}"
        if col in df.columns:
            pred_cols.append(col)

    true_vals = df[true_col].astype(str)
    hit = pd.Series(False, index=df.index)
    for col in pred_cols:
        hit = hit | (df[col].astype(str) == true_vals)
    return hit


def compute_topk_accuracy(
    df: pd.DataFrame,
    level: int,
    ks: list[int],
    max_k_available: int,
) -> dict[str, float | int]:
    """Compute top-k accuracy for a single level, return dict with top-1..top-K and N."""
    true_col = f"level{level}"
    valid = df[true_col].notna() & df[f"predicted_level{level}"].notna()
    sub = df[valid]
    n = len(sub)
    row: dict[str, float | int] = {}
    for k in ks:
        if k > max_k_available:
            row[f"top-{k}"] = float("nan")
        else:
            row[f"top-{k}"] = topk_hit(sub, level, k).mean() if n > 0 else float("nan")
    row["N"] = n
    return row


def print_table(rows: list[dict], index_col: str) -> None:
    """Pretty-print a list of dicts as an aligned table."""
    tbl = pd.DataFrame(rows)
    tbl = tbl.set_index(index_col)
    # Format floats as percentages, ints as-is
    formatters = {}
    for col in tbl.columns:
        if col == "N":
            formatters[col] = lambda x: f"{int(x):>7d}" if pd.notna(x) else "      -"
        else:
            formatters[col] = lambda x: f"{x:>7.2%}" if pd.notna(x) else "      -"
    print(tbl.to_string(formatters=formatters))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute top-K accuracy from a predictions parquet file."
    )
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to predictions parquet file",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Maximum K to report (default: 5)",
    )
    parser.add_argument(
        "--group-by",
        type=str,
        default="level1",
        help="Level to group by for per-code breakdown (default: level1)",
    )
    parser.add_argument(
        "--filter",
        type=str,
        nargs="+",
        metavar="COL",
        help="Keep only rows where these boolean columns are True "
             "(e.g. --filter receips_from_app manual_from_app)",
    )

    parser.add_argument(
        "--code-column",
        type=str,
        default="code",
        help="The column containing the true code (default: code)",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.file)
    df = ensure_true_labels(df, args.code_column)
    df = ensure_predicted_levels(df)

    # ── Apply boolean filters ─────────────────────────────────────────
    if args.filter:
        for col in args.filter:
            if col not in df.columns:
                print(f"ERROR: filter column '{col}' not found in data.", file=sys.stderr)
                print(f"  Available boolean columns: {[c for c in df.columns if df[c].dtype == 'bool']}", file=sys.stderr)
                sys.exit(1)
            df = df[df[col].astype(bool)].copy()
        active_filters = ", ".join(args.filter)
        print(f"Filters applied: {active_filters}  ({len(df)} rows remaining)\n")

    if len(df) == 0:
        print("ERROR: no rows left after filtering.", file=sys.stderr)
        sys.exit(1)

    levels = detect_levels(list(df.columns))
    if not levels:
        print("ERROR: No predicted_level{N} columns found.", file=sys.stderr)
        sys.exit(1)

    ks = [k for k in range(1, args.top_k + 1)]

    # ── Overall accuracy ──────────────────────────────────────────────
    print("=" * 60)
    print("OVERALL TOP-K ACCURACY")
    print("=" * 60)
    overall_rows = []
    for level in levels:
        max_k = detect_max_k(list(df.columns), level)
        row = compute_topk_accuracy(df, level, ks, max_k)
        row["Level"] = f"level{level}"
        overall_rows.append(row)
    print_table(overall_rows, index_col="Level")

    # ── Per-code breakdown ────────────────────────────────────────────
    group_col = args.group_by
    if group_col not in df.columns:
        print(f"WARNING: group-by column '{group_col}' not found, skipping breakdown.", file=sys.stderr)
        return

    for level in levels:
        level_col = f"level{level}"
        if level_col == group_col:
            continue  # no point grouping a level by itself

        max_k = detect_max_k(list(df.columns), level)
        print("=" * 60)
        print(f"LEVEL{level} ACCURACY GROUPED BY {group_col.upper()}")
        print("=" * 60)

        group_rows = []
        for group_val, group_df in sorted(df.groupby(group_col)):
            if group_df[f"predicted_level{level}"].isna().all():
                continue
            row = compute_topk_accuracy(group_df, level, ks, max_k)
            row[group_col] = group_val
            group_rows.append(row)

        if group_rows:
            print_table(group_rows, index_col=group_col)
        else:
            print("  (no data)\n")


if __name__ == "__main__":
    main()
