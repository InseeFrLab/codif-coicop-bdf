"""Render the Quarto accuracy report, upload the HTML to S3, and log to MLflow."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

import boto3
import duckdb
import pandas as pd

from codif_common.contracts import artifact, run_root as contracts_run_root
from codif_common.s3 import connect_secret, resolve_endpoint as s3_endpoint

from codif_common.metrics import (
    LEVELS,
    final_decision,
    parse_step_timings,
    prediction_depth_distribution,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", required=True)
    p.add_argument("--run-date", required=True)
    p.add_argument(
        "--bucket",
        default=None,
        help="Surcharge le bucket du registre (contracts.yaml). Équivalent à $COICOP_BUCKET.",
    )
    p.add_argument(
        "--input-file",
        default=None,
        help="Original input file path; used to locate the export-results file in prediction mode.",
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
    p.add_argument("--classify-rag-model", default=None)
    p.add_argument("--reconcile-llm-model", default=None)
    p.add_argument("--reconcile-llm-concurrency", default=None)
    p.add_argument(
        "--classify-ttc-model-uri",
        default=None,
        help="MLflow URI of the TTC model used by classify-ttc, logged for traceability.",
    )
    # Quelles vector DB ont servi. Loggué pour la même raison que les URI de
    # modèles : sans ça, deux runs bâtis sur des index différents mais aux
    # métriques différentes seraient indistinguables dans MLflow.
    p.add_argument(
        "--classify-rag-notices-collection",
        default=None,
        help="Collection Qdrant interrogée par classify-rag-notices, pour traçabilité.",
    )
    p.add_argument(
        "--classify-rag-annotations-collection",
        default=None,
        help="Collection Qdrant interrogée par classify-rag-annotations, pour traçabilité.",
    )
    p.add_argument("--skip-report", default=None)
    p.add_argument(
        "--reconciliation", choices=["llm", "sirus"], default="llm",
        help="Quelle étape de conciliation a tranché : `llm` (reconcile-llm, "
             "défaut) ou `sirus` (reconcile-sirus). Détermine le parquet lu.",
    )
    p.add_argument(
        "--reconcile-sirus-model-uri", default=None,
        help="Tracé dans MLflow pour savoir quel modèle a produit les codes.",
    )
    return p.parse_args()



def connect_s3() -> duckdb.DuckDBPyConnection:
    """Connexion DuckDB configurée pour lire les parquet du bucket S3."""
    return connect_secret()


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


def log_to_mlflow(args, run_root, output_s3, decide_path) -> None:
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
            "classify_rag_model": args.classify_rag_model,
            "reconcile_llm_model": args.reconcile_llm_model,
            "reconcile_llm_concurrency": args.reconcile_llm_concurrency,
            "classify_ttc_model_uri": args.classify_ttc_model_uri,
            "classify_rag_notices_collection": args.classify_rag_notices_collection,
            "classify_rag_annotations_collection": args.classify_rag_annotations_collection,
            "skip_report": args.skip_report,
            "output_prefix": run_root,
            "report_html": output_s3,
            # Quelle conciliation a tranché, et avec quel modèle : sans ça, deux
            # runs aux métriques différentes seraient indistinguables.
            "reconciliation": args.reconciliation,
            "reconcile_sirus_model_uri": args.reconcile_sirus_model_uri,
        }
        mlflow.log_params({k: v for k, v in params.items() if v is not None})
        mlflow.set_tag("mode", "production")
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

        # ---- read the reconcile-llm predictions (dot-separated codes) ----
        print(f"[report] loading reconcile-llm predictions (mlflow): {decide_path}", flush=True)
        con = connect_s3()
        df = con.sql(f"SELECT * FROM read_parquet('{decide_path}')").df()
        mlflow.log_metric("n_obs_total", len(df))

        # Final-prediction distribution by COICOP level (logged in both modes).
        # Porte sur la colonne de décision effective : codé en dur sur
        # `llm_code`, ce bloc restait muet en conciliation SIRUS.
        decision = final_decision(df, strict=False)[1]
        if decision and decision in df.columns:
            dist = prediction_depth_distribution(df[decision])
            for k in LEVELS:
                mlflow.log_metric(f"pred_depth_niv{k}_count", dist["depth"][k]["count"])
                mlflow.log_metric(f"pred_depth_niv{k}_pct", dist["depth"][k]["pct"])
            mlflow.log_dict(dist, "prediction_distribution.json")

        # Les métriques d'accuracy ont quitté ce rapport pour l'étape finale
        # `evaluate` : elles exigent une vérité terrain, que la production n'a
        # pas. Ne restent ici que les indicateurs calculables sans elle.

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
    if args.bucket:
        os.environ["COICOP_BUCKET"] = args.bucket
    RUN = {"run_date": args.run_date, "run_id": args.run_id}
    run_root = contracts_run_root(**RUN)
    # Les deux conciliations sont exclusives (paramètre Argo `reconciliation`) :
    # on lit celle qui a effectivement tourné. Le fichier porte, dans les deux
    # cas, la table fusionnée complète — donc le rapport peut scorer les 4
    # classifieurs de base à l'identique.
    conciliation_step = "reconcile-sirus" if args.reconciliation == "sirus" else "reconcile-llm"
    decide_path = artifact(conciliation_step, "predictions", **RUN)
    final_output_path = artifact("export-results", "predictions", **RUN)
    output_s3 = artifact("report", "html", **RUN)

    here = Path(__file__).resolve().parent
    out_html = here / "report.html"

    env = os.environ.copy()
    env["REPORT_RUN_ID"] = args.run_id
    env["REPORT_RUN_DATE"] = args.run_date

    # Un seul mode. L'évaluation, quand le fichier d'entrée porte des étiquettes,
    # est l'affaire de l'étape finale `evaluate` — ce rapport-ci ne suppose
    # aucune vérité terrain et se rend toujours.
    if args.input_file:
        final_output_path = artifact(
            "export-results", "deliverable", **RUN,
            filename=os.path.basename(args.input_file),
        )
    env["REPORT_INPUT_PATH"] = final_output_path
    env["REPORT_DECIDE_PATH"] = decide_path
    print(f"[report] livrable : {final_output_path}", flush=True)
    print(f"[report] conciliation : {decide_path}", flush=True)
    subprocess.run(
        ["quarto", "render", "prediction_report.qmd", "--to", "html", "--output", "report.html"],
        cwd=here,
        env=env,
        check=True,
    )

    upload(out_html, output_s3)

    try:
        log_to_mlflow(args, run_root, output_s3, decide_path)
    except Exception as exc:  # never fail the step on a tracking error
        print(f"[report] MLflow logging failed (non-fatal): {exc}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
