"""Render the Quarto accuracy report and upload the HTML to S3."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

import boto3


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

    print(f"[report] rendering {here / 'report.qmd'} (input={input_path})", flush=True)
    subprocess.run(
        ["quarto", "render", "report.qmd", "--to", "html", "--output", "report.html"],
        cwd=here,
        env=env,
        check=True,
    )

    upload(out_html, output_s3)
    return 0


if __name__ == "__main__":
    sys.exit(main())
