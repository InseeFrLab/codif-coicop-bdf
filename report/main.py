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

from coicop_metrics import (
    CANONICAL_LEVELS,
    LEVELS,
    METHODS,
    REGIME_LEVEL,
    TRUTH_COL_CANONICAL,
    accuracy,
    coverage_table,
    parse_step_timings,
    prediction_depth_distribution,
    regime_masks,
    truth_column,
    truth_depth_distribution,
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

        # ---- read the reconcile-llm predictions (dot-separated codes) ----
        print(f"[report] loading reconcile-llm predictions (mlflow): {decide_path}", flush=True)
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
        # Scored against the canonical truth (`code_lvl4`) so predictions and
        # ground truth live in the same pruned code space.
        truth_col = truth_column(df)
        scorable = df
        if truth_col in df.columns:
            scorable = df[
                df[truth_col].notna() & (df[truth_col].astype(str).str.len() > 0)
            ].copy()
        if not prediction and len(scorable) > 0:
            mlflow.log_metric("n_scorable", len(scorable))
            mlflow.log_param("truth_column", truth_col)
            truth = scorable[truth_col]
            depth = truth_depth_distribution(truth)
            for k in CANONICAL_LEVELS:
                mlflow.log_metric(
                    f"truth_shallower_than_niv{k}_count", depth["shallower_than"][k]["count"]
                )
            mlflow.log_dict(depth, "truth_depth_distribution.json")
            # Avec une vérité canonique, le niveau 5 est structurellement vide.
            strict_levels = (
                CANONICAL_LEVELS if truth_col == TRUTH_COL_CANONICAL else LEVELS
            )
            for name, col in METHODS:
                if col not in scorable.columns:
                    continue
                for k in strict_levels:
                    n_ok, n_app, acc = accuracy(truth, scorable[col], k)
                    if name == "LLM":
                        # headline: final accuracy after reconcile-llm
                        mlflow.log_metric(f"n_applicable_niv{k}", n_app)
                    if n_app:
                        mlflow.log_metric(f"accuracy_{name.lower()}_niv{k}", acc)
                # Inclusive convention: every row counts at every level (a truth
                # shallower than k must be predicted exactly).
                for k in CANONICAL_LEVELS:
                    _, n_all, acc_all = accuracy(
                        truth, scorable[col], k, inclusive=True
                    )
                    if n_all:
                        mlflow.log_metric(
                            f"accuracy_all_{name.lower()}_niv{k}", acc_all
                        )
            # Coverage vs accuracy-when-answering: the global accuracy above
            # counts a refusal to code (NULL, "", "N/A") as an error, which hides
            # whether a method is wrong or simply silent.
            cov = coverage_table(scorable, REGIME_LEVEL)
            for name in cov.index:
                slug = name.lower()
                mlflow.log_metric(f"coverage_{slug}", float(cov.loc[name, "couverture"]))
                mlflow.log_metric(
                    f"abstention_{slug}_count", int(cov.loc[name, "abstentions"])
                )
                acc_ans = cov.loc[name, f"accuracy niv{REGIME_LEVEL} sur réponses"]
                if pd.notna(acc_ans):
                    mlflow.log_metric(
                        f"accuracy_answered_{slug}_niv{REGIME_LEVEL}", float(acc_ans)
                    )

            # Per-regime accuracy at the survey target level: the pooled figures
            # above mix the consensus short-circuit (where `llm_code` is TTC
            # top-1, so LLM and TTC are identical by construction) with the rows
            # the judge really arbitrated. Only the latter measure the judge.
            masks = regime_masks(scorable)
            if masks is not None:
                for _label, suffix, mask in masks:
                    sub = scorable[mask]
                    mlflow.log_metric(f"n_{suffix}", len(sub))
                    if not len(sub):
                        continue
                    for name, col in METHODS:
                        if col not in sub.columns:
                            continue
                        _, n_sub, acc_sub = accuracy(
                            sub[truth_col], sub[col], REGIME_LEVEL, inclusive=True
                        )
                        if n_sub:
                            mlflow.log_metric(
                                f"accuracy_all_{name.lower()}_niv{REGIME_LEVEL}_{suffix}",
                                acc_sub,
                            )
                consensus_mask = masks[0][2]
                mlflow.log_metric("consensus_share", float(consensus_mask.mean()))
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

    # prediction mode: ground-truth `code` is entirely NULL
    print(f"[report] loading reconcile-llm predictions: {decide_path}", flush=True)
    con = connect_s3()
    row = con.sql(
        f"SELECT bool_and(code IS NULL) FROM read_parquet('{decide_path}')"
    ).fetchone()
    prediction = bool(row[0]) if row and row[0] is not None else False

    qmd = "prediction_report.qmd" if prediction else "report.qmd"
    if prediction:
        if args.input_file:
            final_output_path = artifact(
                "export-results", "deliverable", **RUN,
                filename=os.path.basename(args.input_file),
            )
        env["REPORT_INPUT_PATH"] = final_output_path
        env["REPORT_DECIDE_PATH"] = decide_path
        print(f"[report] reconcile-llm input (qmd): {decide_path}", flush=True)
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
