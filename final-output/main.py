"""Produce final user-facing output from regex and LLM predictions."""

from __future__ import annotations

import argparse
import os
import sys
from textwrap import dedent

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import s3fs


PIPELINE_COLS = {
    "l_pr_product",
    "s_pr_product",
    "code",
    "coicop",
    "source",
    "n_obs",
    "shop_type_code",
    "_source_input_file",
    "method",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", required=True)
    p.add_argument("--run-date", required=True)
    p.add_argument("--bucket", default="projet-budget-famille")
    p.add_argument("--text-column", default="raw_product",
                   help="Original input column name for product text")
    p.add_argument("--shop-column", default="shop",
                   help="Original input column name for shop")
    p.add_argument("--budget-column", default="budget",
                   help="Original input column name for budget")
    p.add_argument("--annee-column", default="annee",
                   help="Original input column name for year")
    return p.parse_args()


def init_duckdb() -> duckdb.DuckDBPyConnection:
    endpoint = (
        os.environ.get("AWS_S3_ENDPOINT")
        or os.environ.get("AWS_ENDPOINT_URL", "").replace("https://", "").replace("http://", "")
        or "minio.lab.sspcloud.fr"
    )
    con = duckdb.connect()
    con.sql("INSTALL httpfs; LOAD httpfs;")
    con.sql(
        dedent(f"""
            CREATE OR REPLACE SECRET s3_secret (
                TYPE s3,
                KEY_ID '{os.environ.get("AWS_ACCESS_KEY_ID", "")}',
                SECRET '{os.environ.get("AWS_SECRET_ACCESS_KEY", "")}',
                SESSION_TOKEN '{os.environ.get("AWS_SESSION_TOKEN", "")}',
                ENDPOINT '{endpoint}',
                URL_STYLE 'path',
                USE_SSL true
            );
        """)
    )
    return con


def _fs() -> s3fs.S3FileSystem:
    endpoint = os.environ.get("AWS_S3_ENDPOINT") or os.environ.get("AWS_ENDPOINT_URL")
    if endpoint and not endpoint.startswith("http"):
        endpoint = f"https://{endpoint}"
    return s3fs.S3FileSystem(endpoint_url=endpoint)


def export_parquet(df: pd.DataFrame, s3_uri: str) -> None:
    path = s3_uri.removeprefix("s3://")
    tbl = pa.Table.from_pandas(df, preserve_index=False)
    with _fs().open(path, "wb") as f:
        pq.write_table(tbl, f)
    print(f"[final-output] exported {len(df)} rows → {s3_uri}", flush=True)


def export_csv(df: pd.DataFrame, s3_uri: str) -> None:
    path = s3_uri.removeprefix("s3://")
    with _fs().open(path, "w") as f:
        df.to_csv(f, index=False)
    print(f"[final-output] exported {len(df)} rows → {s3_uri}", flush=True)


def main() -> int:
    args = parse_args()
    run_root = f"s3://{args.bucket}/data/workflow_runs/{args.run_date}/{args.run_id}"
    con = init_duckdb()

    # Base: all user columns from raw_test.parquet (strip internal pipeline columns)
    # Read full first to capture _source_input_file before stripping
    raw_full = con.sql(f"SELECT * FROM read_parquet('{run_root}/preprocessing/raw_test.parquet')").df()
    input_file_path = (
        raw_full["_source_input_file"].iloc[0]
        if "_source_input_file" in raw_full.columns and len(raw_full) > 0
        else None
    )
    initial_cols = ["id"] + [c for c in raw_full.columns if c not in PIPELINE_COLS and c != "id"]
    raw = raw_full[initial_cols]
    print(f"[final-output] base: {len(raw)} rows, {len(initial_cols)} columns", flush=True)

    # Regex predictions: rows classified by regex
    regex = con.sql(f"""
        SELECT id, predict_code
        FROM read_parquet('{run_root}/codif-regex/REGEX_pred.parquet')
        WHERE predict_code IS NOT NULL
    """).df()
    print(f"[final-output] regex predictions: {len(regex)} rows", flush=True)

    # LLM decisions
    llm = con.sql(f"""
        SELECT id, llm_code, llm_explication, llm_confiance, llm_model
        FROM read_parquet('{run_root}/decide-coicop/predictions.parquet')
    """).df()
    print(f"[final-output] LLM decisions: {len(llm)} rows", flush=True)

    result = raw.merge(regex[["id", "predict_code"]], on="id", how="left")
    result = result.merge(llm, on="id", how="left")

    # LLM code takes precedence; fallback to regex; else NA
    result["predicted_code"] = result["llm_code"].where(
        result["llm_code"].notna(), result["predict_code"]
    )
    result["prediction_source"] = result.apply(
        lambda r: ("consensus" if r["llm_model"] == "consensus" else "llm")
        if pd.notna(r["llm_code"])
        else ("regex" if pd.notna(r["predict_code"]) else None),
        axis=1,
    )
    result["llm_comment"] = result["llm_explication"]

    result = result.drop(
        columns=["predict_code", "llm_code", "llm_explication", "llm_confiance", "llm_model"]
    )

    # Restore original column names (preprocessing renamed them to pipeline names)
    reverse_mapping = {}
    for pipeline_col, original_col in [
        ("raw_product", args.text_column),
        ("shop",        args.shop_column),
        ("budget",      args.budget_column),
        ("annee",       args.annee_column),
    ]:
        if original_col and original_col != pipeline_col and pipeline_col in result.columns:
            reverse_mapping[pipeline_col] = original_col
    if reverse_mapping:
        result = result.rename(columns=reverse_mapping)
        print(f"[final-output] restored column names: {reverse_mapping}", flush=True)

    # Export with original input filename and format
    output_basename = os.path.basename(input_file_path) if input_file_path else "predictions.parquet"
    output_uri = f"{run_root}/final-output/{output_basename}"
    print(f"[final-output] output file: {output_uri}", flush=True)

    if output_basename.lower().endswith(".csv"):
        export_csv(result, output_uri)
    else:
        export_parquet(result, output_uri)

    return 0


if __name__ == "__main__":
    sys.exit(main())
