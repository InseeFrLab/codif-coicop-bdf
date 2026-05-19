"""Render the Quarto accuracy report and upload the HTML to S3."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent
from urllib.parse import urlparse

import boto3
import duckdb


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", required=True)
    p.add_argument("--run-date", required=True)
    p.add_argument(
        "--bucket",
        default="projet-budget-famille",
        help="S3 bucket that holds workflow_runs (default: projet-budget-famille)",
    )
    return p.parse_args()


def s3_client():
    endpoint = os.environ.get("AWS_S3_ENDPOINT") or os.environ.get("AWS_ENDPOINT_URL")
    kwargs: dict = {}
    if endpoint:
        if not endpoint.startswith("http"):
            endpoint = f"https://{endpoint}"
        kwargs["endpoint_url"] = endpoint
    return boto3.client("s3", **kwargs)


def upload(local: Path, s3_uri: str) -> None:
    parsed = urlparse(s3_uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    s3_client().upload_file(
        str(local),
        bucket,
        key,
        ExtraArgs={"ContentType": "text/html"},
    )
    print(f"[report] uploaded {local} -> {s3_uri}", flush=True)


def is_prediction_mode(input_path: str) -> bool:
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
    row = con.sql(f"SELECT bool_and(code IS NULL) FROM read_parquet('{input_path}')").fetchone()
    return bool(row[0]) if row and row[0] is not None else False


def main() -> int:
    args = parse_args()
    run_root = f"s3://{args.bucket}/data/workflow_runs/{args.run_date}/{args.run_id}"
    input_path = f"{run_root}/decide-coicop/predictions.parquet"
    output_s3 = f"{run_root}/report/report.html"

    here = Path(__file__).resolve().parent
    out_html = here / "report.html"

    env = os.environ.copy()
    env["REPORT_INPUT_PATH"] = input_path
    env["REPORT_RUN_ID"] = args.run_id
    env["REPORT_RUN_DATE"] = args.run_date

    prediction = is_prediction_mode(input_path)
    qmd = "prediction_report.qmd" if prediction else "report.qmd"
    print(f"[report] mode={'prediction' if prediction else 'evaluation'}", flush=True)
    print(f"[report] rendering {here / qmd} (input={input_path})", flush=True)
    subprocess.run(
        ["quarto", "render", qmd, "--to", "html", "--output", "report.html"],
        cwd=here,
        env=env,
        check=True,
    )

    upload(out_html, output_s3)
    return 0


if __name__ == "__main__":
    sys.exit(main())
