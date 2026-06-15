"""Render the Quarto accuracy report, upload the HTML to S3, and log to MLflow."""

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

from coicop_metrics import (
    LEVELS,
    METHODS,
    accuracy,
    parse_step_timings,
    prediction_depth_distribution,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", required=True)
    p.add_argument("--run-date", required=True)
    p.add_argument(
        "--bucket",
        default="projet-budget-famille",
        help="S3 bucket that holds workflow_runs (default: projet-budget-famille)",
    )
    p.add_argument(
        "--input-file",
        default=None,
        help="Original input file path; used to locate the final-output file in prediction mode.",
    )
    # --- MLflow logging ---
    p.add_argument(
        "--experiment-name",
        default="codif-coicop-eval",
        help="MLflow experiment name to log this report's metrics under.",
    )
    p.add_argument(
        "--step-timings",
        default="",
        help='JSON object {step: [startedAt, finishedAt]} (RFC3339) of Argo step timings.',
    )
    # Pipeline parameters, logged as MLflow params for traceability.
    p.add_argument("--sample-size", default=None)
    p.add_argument("--model-name", default=None)
    p.add_argument("--decide-model", default=None)
    p.add_argument("--decide-concurrency", default=None)
    p.add_argument(
        "--ttc-model-uri",
        default=None,
        help="MLflow URI of the TTC model used by run-ttc, logged for traceability.",
    )
    p.add_argument("--skip-vector-db", default=None)
    p.add_argument("--skip-report", default=None)
    return p.parse_args()


def s3_endpoint() -> str:
    return (
        os.environ.get("AWS_S3_ENDPOINT")
        or os.environ.get("AWS_ENDPOINT_URL", "").replace("https://", "").replace("http://", "")
        or "minio.lab.sspcloud.fr"
    )


def connect_s3() -> duckdb.DuckDBPyConnection:
    """DuckDB connection configured to read parquet from the S3 bucket."""
    con = duckdb.connect()
    con.sql("INSTALL httpfs; LOAD httpfs;")
    con.sql(
        dedent(f"""
            CREATE OR REPLACE SECRET s3_secret (
                TYPE s3,
                KEY_ID '{os.environ.get("AWS_ACCESS_KEY_ID", "")}',
                SECRET '{os.environ.get("AWS_SECRET_ACCESS_KEY", "")}',
                SESSION_TOKEN '{os.environ.get("AWS_SESSION_TOKEN", "")}',
                ENDPOINT '{s3_endpoint()}',
                URL_STYLE 'path',
                USE_SSL true
            );
        """)
    )
    return con


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


def _git(args: list[str]):
    try:
        return subprocess.check_output(["git", *args]).decode("ascii").strip()
    except Exception:
        return None


def log_to_mlflow(args, run_root, output_s3, decide_path, prediction) -> None:
    """Best-effort MLflow logging. Never raises into the pipeline step."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        print("[report] MLFLOW_TRACKING_URI not set, skipping MLflow logging", flush=True)
        return

    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(run_name=f"{args.run_date}_{args.run_id}"):
        # ---- params: pipeline parameters + S3 output prefix ----
        params = {
            "run_id": args.run_id,
            "run_date": args.run_date,
            "bucket": args.bucket,
            "sample_size": args.sample_size,
            "model_name": args.model_name,
            "decide_model": args.decide_model,
            "decide_concurrency": args.decide_concurrency,
            "ttc_model_uri": args.ttc_model_uri,
            "skip_vector_db": args.skip_vector_db,
            "skip_report": args.skip_report,
            "output_prefix": run_root,
            "report_html": output_s3,
        }
        mlflow.log_params({k: v for k, v in params.items() if v is not None})
        mlflow.set_tag("mode", "prediction" if prediction else "evaluation")
        for tag, val in (("git.commit", _git(["rev-parse", "HEAD"])),
                         ("git.branch", _git(["rev-parse", "--abbrev-ref", "HEAD"]))):
            if val:
                mlflow.set_tag(tag, val)

        # ---- timing metrics (from Argo step durations) ----
        timings = parse_step_timings(args.step_timings)
        if timings:
            mlflow.log_metrics(timings)
        else:
            print("[report] no parsable step timings provided", flush=True)

        # ---- read the decide-coicop predictions (dot-separated codes) ----
        print(f"[report] loading decide-coicop predictions (mlflow): {decide_path}", flush=True)
        con = connect_s3()
        df = con.sql(f"SELECT * FROM read_parquet('{decide_path}')").df()
        mlflow.log_metric("n_obs_total", len(df))

        # Final-prediction distribution by COICOP level (logged in both modes).
        if "llm_code" in df.columns:
            dist = prediction_depth_distribution(df["llm_code"])
            for k in LEVELS:
                mlflow.log_metric(f"pred_depth_niv{k}_count", dist["depth"][k]["count"])
                mlflow.log_metric(f"pred_depth_niv{k}_pct", dist["depth"][k]["pct"])
            mlflow.log_dict(dist, "prediction_distribution.json")

        # ---- accuracy metrics (evaluation mode only) ----
        scorable = df
        if "code" in df.columns:
            scorable = df[df["code"].notna() & (df["code"].astype(str).str.len() > 0)].copy()
        if not prediction and len(scorable) > 0:
            mlflow.log_metric("n_scorable", len(scorable))
            for name, col in METHODS:
                if col not in scorable.columns:
                    continue
                for k in LEVELS:
                    n_ok, n_app, acc = accuracy(scorable["code"], scorable[col], k)
                    if name == "LLM":
                        # headline: final accuracy after decide-coicop
                        mlflow.log_metric(f"n_applicable_niv{k}", n_app)
                    if n_app:
                        mlflow.log_metric(f"accuracy_{name.lower()}_niv{k}", acc)
            if "llm_error" in scorable.columns:
                mlflow.log_metric("llm_error_count", int(scorable["llm_error"].notna().sum()))
            if "llm_code" in scorable.columns:
                mlflow.log_metric(
                    "llm_coverage", float(scorable["llm_code"].notna().mean())
                )
            if "llm_confiance" in scorable.columns:
                conf = scorable["llm_confiance"].dropna()
                if len(conf) > 0:
                    mlflow.log_metric("mean_llm_confiance", float(conf.astype(float).mean()))

        # ---- artifacts ----
        out_html = Path(__file__).resolve().parent / "report.html"
        if out_html.exists():
            mlflow.log_artifact(str(out_html))
        print(
            f"[report] logged metrics to MLflow experiment '{args.experiment_name}'",
            flush=True,
        )


def main() -> int:
    args = parse_args()
    run_root = f"s3://{args.bucket}/data/workflow_runs/{args.run_date}/{args.run_id}"
    decide_path = f"{run_root}/decide-coicop/predictions.parquet"
    final_output_path = f"{run_root}/final-output/predictions.parquet"
    output_s3 = f"{run_root}/report/report.html"

    here = Path(__file__).resolve().parent
    out_html = here / "report.html"

    env = os.environ.copy()
    env["REPORT_RUN_ID"] = args.run_id
    env["REPORT_RUN_DATE"] = args.run_date

    # prediction mode: ground-truth `code` is entirely NULL
    print(f"[report] loading decide-coicop predictions: {decide_path}", flush=True)
    con = connect_s3()
    row = con.sql(
        f"SELECT bool_and(code IS NULL) FROM read_parquet('{decide_path}')"
    ).fetchone()
    prediction = bool(row[0]) if row and row[0] is not None else False

    qmd = "prediction_report.qmd" if prediction else "report.qmd"
    if prediction:
        if args.input_file:
            final_output_path = f"{run_root}/final-output/{os.path.basename(args.input_file)}"
        env["REPORT_INPUT_PATH"] = final_output_path
        env["REPORT_DECIDE_PATH"] = decide_path
        print(f"[report] decide-coicop input (qmd): {decide_path}", flush=True)
    else:
        env["REPORT_INPUT_PATH"] = decide_path
    print(f"[report] mode={'prediction' if prediction else 'evaluation'}", flush=True)
    print(f"[report] rendering {here / qmd} (input={env['REPORT_INPUT_PATH']})", flush=True)
    subprocess.run(
        ["quarto", "render", qmd, "--to", "html", "--output", "report.html"],
        cwd=here,
        env=env,
        check=True,
    )

    upload(out_html, output_s3)

    try:
        log_to_mlflow(args, run_root, output_s3, decide_path, prediction)
    except Exception as exc:  # never fail the step on a tracking error
        print(f"[report] MLflow logging failed (non-fatal): {exc}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
