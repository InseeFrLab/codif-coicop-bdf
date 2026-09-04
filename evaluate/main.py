"""Étape finale FACULTATIVE : mesure la qualité d'un run.

Ne tourne que si le fichier d'entrée du run portait des étiquettes
(`--label-column` de `build-datasets`). Rend le rapport Quarto, le dépose sur S3
et logue les métriques dans MLflow.

Elle existe pour que le reste du pipeline n'ait plus qu'un seul mode. Avant, la
dualité production/évaluation était testée à treize endroits, et trois étapes de
classification calculaient leurs propres métriques — dont `rag-notices`, qui les
calculait même en production et loguait alors une accuracy ≈ 0 sans rien casser.

Trois artefacts, pas un. Le parquet de conciliation porte les quatre
classifieurs et la conciliation, mais pas les codes récupérés par les RAG
(retirés à la fusion) ni les lignes captées par la regex (elles n'entrent jamais
dans la chaîne) — donc ni recall de retrieval, ni accuracy de bout en bout.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

import boto3
import pandas as pd

from codif_common.contracts import artifact, run_root as contracts_run_root
from codif_common.metrics import (
    CANONICAL_LEVELS,
    LEVELS,
    METHODS,
    REGIME_LEVEL,
    TRUTH_COL_CANONICAL,
    accuracy,
    coverage_table,
    final_decision,
    regime_masks,
    truth_column,
    truth_depth_distribution,
)
from codif_common.s3 import connect_secret, resolve_endpoint

from internals import flatten_internal, load_classifier_records


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", required=True)
    p.add_argument("--run-date", required=True)
    p.add_argument("--bucket", default=None, help="Surcharge le bucket du registre.")
    p.add_argument(
        "--reconciliation", choices=["llm", "sirus"], default="llm",
        help="Quelle conciliation a tranché : détermine le parquet lu.",
    )
    p.add_argument(
        "--input-file", default=None,
        help="Fichier d'entrée du run ; sert à localiser le livrable export-results.",
    )
    p.add_argument(
        "--source-column", default=None,
        help=(
            "Colonne de provenance du produit. Renseignée, le rapport ajoute les "
            "accuracy ventilées par source. Vide, la section est omise. Ne "
            "restreint jamais le périmètre : c'est une dimension d'analyse."
        ),
    )
    p.add_argument("--experiment-name", default="codif-coicop-eval")
    p.add_argument("--step-timings", default="")
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
    s3_client().upload_file(
        str(local), parsed.netloc, parsed.path.lstrip("/"),
        ExtraArgs={"ContentType": "text/html"},
    )
    print(f"[evaluate] uploaded {local} -> {s3_uri}", flush=True)


def _git(args: list[str]):
    try:
        return subprocess.check_output(["git", *args]).decode("ascii").strip()
    except Exception:
        return None


def require_canonical_truth(df: pd.DataFrame) -> str:
    """Exige la vérité canonique, au lieu de se rabattre en silence.

    `truth_column()` retombe sur `code` quand `code_lvl4` manque, avec un simple
    avertissement. C'est un repli raisonnable pour un rapport rendu au fil de
    l'eau ; il ne l'est pas quand on a explicitement demandé une évaluation :
    mesurer des prédictions canoniques contre une vérité brute sous-estime
    l'accuracy sur près d'un quart des postes.

    `code_lvl4` naît en un seul endroit — `reconcile-llm`, et seulement s'il a
    reçu `--mapping-file`. C'est ce couplage, silencieux jusqu'ici, que cette
    fonction rend visible.
    """
    truth_col = truth_column(df)
    if truth_col != TRUTH_COL_CANONICAL:
        raise SystemExit(
            f"[evaluate] ÉCHEC : la vérité canonique « {TRUTH_COL_CANONICAL} » est "
            f"absente du parquet de conciliation.\n"
            f"  Colonnes présentes : {', '.join(sorted(df.columns))}\n"
            f"  Cause probable : `reconcile-llm` n'a pas reçu --mapping-file, donc "
            f"la vérité n'a pas été projetée dans l'espace de codes canonique.\n"
            f"  Mesurer contre « code » brut sous-estimerait l'accuracy sur près "
            f"d'un quart des postes : on préfère échouer."
        )
    return truth_col


def log_to_mlflow(args, df, scorable, truth_col, output_s3, internal=None) -> None:
    """Métriques d'évaluation. Best-effort : ne casse jamais l'étape."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        print("[evaluate] MLFLOW_TRACKING_URI absent, log MLflow ignoré", flush=True)
        return

    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(run_name=f"{args.run_date}_{args.run_id}"):
        mlflow.log_params({
            "run_id": args.run_id,
            "run_date": args.run_date,
            "reconciliation": args.reconciliation,
            "truth_column": truth_col,
            "report_html": output_s3,
        })
        mlflow.set_tag("mode", "evaluation")
        for tag, val in (("git.commit", _git(["rev-parse", "HEAD"])),
                         ("git.branch", _git(["rev-parse", "--abbrev-ref", "HEAD"]))):
            if val:
                mlflow.set_tag(tag, val)

        mlflow.log_metric("n_obs_total", len(df))
        mlflow.log_metric("n_scorable", len(scorable))
        truth = scorable[truth_col]

        depth = truth_depth_distribution(truth)
        for k in CANONICAL_LEVELS:
            mlflow.log_metric(
                f"truth_shallower_than_niv{k}_count", depth["shallower_than"][k]["count"]
            )
        mlflow.log_dict(depth, "truth_depth_distribution.json")

        # Avec une vérité canonique, le niveau 5 est structurellement vide.
        strict_levels = CANONICAL_LEVELS if truth_col == TRUTH_COL_CANONICAL else LEVELS
        for name, col in METHODS:
            if col not in scorable.columns:
                continue
            slug = name.lower()
            for k in strict_levels:
                n_ok, n_app, acc = accuracy(truth, scorable[col], k)
                if n_app:
                    mlflow.log_metric(f"accuracy_{slug}_niv{k}", acc)
            # Convention inclusive : toute observation compte à tout niveau.
            for k in CANONICAL_LEVELS:
                _, n_all, acc_all = accuracy(truth, scorable[col], k, inclusive=True)
                if n_all:
                    mlflow.log_metric(f"accuracy_all_{slug}_niv{k}", acc_all)

        # Couverture contre accuracy-sur-réponses : l'accuracy globale compte un
        # refus de coder comme une erreur, ce qui masque si une méthode se trompe
        # ou se tait.
        cov = coverage_table(scorable, REGIME_LEVEL)
        for name in cov.index:
            slug = name.lower()
            mlflow.log_metric(f"coverage_{slug}", float(cov.loc[name, "couverture"]))
            mlflow.log_metric(f"abstention_{slug}_count", int(cov.loc[name, "abstentions"]))
            acc_ans = cov.loc[name, f"accuracy niv{REGIME_LEVEL} sur réponses"]
            if pd.notna(acc_ans):
                mlflow.log_metric(f"accuracy_answered_{slug}_niv{REGIME_LEVEL}", float(acc_ans))

        # Par régime : les chiffres agrégés mélangent le court-circuit consensus
        # (où la conciliation vaut TTC top-1 par construction) et les lignes
        # réellement arbitrées. Seules les secondes mesurent le juge.
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
                            f"accuracy_all_{name.lower()}_niv{REGIME_LEVEL}_{suffix}", acc_sub
                        )
            mlflow.log_metric("consensus_share", float(masks[0][2].mean()))

        # Coût et latence de l'arbitrage : présents dans le parquet depuis
        # toujours, agrégés nulle part jusqu'ici.
        for col, metric in (
            ("llm_prompt_tokens", "llm_prompt_tokens_total"),
            ("llm_completion_tokens", "llm_completion_tokens_total"),
        ):
            if col in df.columns:
                mlflow.log_metric(metric, float(pd.to_numeric(df[col], errors="coerce").sum()))
        if "llm_latency_s" in df.columns:
            lat = pd.to_numeric(df["llm_latency_s"], errors="coerce").dropna()
            if len(lat):
                mlflow.log_metric("llm_latency_s_mean", float(lat.mean()))
                mlflow.log_metric("llm_latency_s_total", float(lat.sum()))
        if "llm_error" in df.columns:
            mlflow.log_metric("llm_error_count", int(df["llm_error"].notna().sum()))

        # Décomposition interne des classifieurs : recall du retriever, régimes
        # de réponse, fiabilité des confiances, distorsion de distribution. Ces
        # séries vivaient dans les étapes de classification, chacune dans sa
        # propre expérience et sur son propre périmètre. Ici, elles portent sur
        # les mêmes lignes et la même vérité que le reste du rapport.
        if internal:
            mlflow.log_metrics(internal)
            print(f"[evaluate] + {len(internal)} métriques de décomposition", flush=True)

        out_html = Path(__file__).resolve().parent / "evaluation_report.html"
        if out_html.exists():
            mlflow.log_artifact(str(out_html))
        print(f"[evaluate] métriques loguées dans « {args.experiment_name} »", flush=True)


def main() -> int:
    args = parse_args()
    if args.bucket:
        os.environ["COICOP_BUCKET"] = args.bucket
    RUN = {"run_date": args.run_date, "run_id": args.run_id}

    step = f"reconcile-{args.reconciliation}"
    decide_path = artifact(step, "predictions", **RUN)
    output_s3 = artifact("evaluate", "html", **RUN)

    print(f"[evaluate] lecture de la conciliation : {decide_path}", flush=True)
    con = connect_secret()
    df = con.sql(f"SELECT * FROM read_parquet('{decide_path}')").df()

    truth_col = require_canonical_truth(df)
    scorable = df[
        df[truth_col].notna() & (df[truth_col].astype(str).str.len() > 0)
    ].copy()
    if len(scorable) == 0:
        raise SystemExit(
            "[evaluate] ÉCHEC : aucune ligne étiquetée dans le parquet de "
            "conciliation. Le fichier d'entrée portait-il bien la colonne annoncée "
            "par `--label-column` ?"
        )
    print(f"[evaluate] {len(scorable)}/{len(df)} lignes évaluables", flush=True)

    # Les deux artefacts complémentaires : sans eux, ni recall de retrieval ni
    # accuracy de bout en bout. Le rapport les rend facultatifs pour rester
    # lisible quand un run partiel les laisse absents.
    env = {
        **os.environ,
        "EVAL_DECIDE_PATH": decide_path,
        # `predictions` porte `parsed` / `codable` / `confidence`, que la fusion
        # de reconcile-llm laisse tomber : sans lui, pas de régimes de réponse.
        "EVAL_RAGNOTICES_PATH": artifact("classify-rag-notices", "predictions", **RUN),
        "EVAL_RETRIEVED_PATH": artifact("classify-rag-notices", "retrieved_codes", **RUN),
        "EVAL_RAGANN_PATH": artifact("classify-rag-annotations", "predictions", **RUN),
        # Pour l'accuracy de bout en bout : le livrable porte les lignes captées
        # par la regex mais plus la vérité (retirée par PIPELINE_COLS), d'où la
        # jointure sur `observations` et la mise au format canonique.
        "EVAL_OBSERVATIONS_PATH": artifact("build-datasets", "observations", **RUN),
        "EVAL_MAPPING_PATH": artifact("prune-codes", "mapping_lvl4", **RUN),
        "EVAL_RUN_ID": args.run_id,
        "EVAL_RUN_DATE": args.run_date,
        "EVAL_SOURCE_COLUMN": args.source_column or "",
        "AWS_S3_ENDPOINT": resolve_endpoint(),
    }
    if args.input_file:
        env["EVAL_DELIVERABLE_PATH"] = artifact(
            "export-results", "deliverable", **RUN,
            filename=os.path.basename(args.input_file),
        )

    here = Path(__file__).resolve().parent
    out_html = here / "evaluation_report.html"
    print("[evaluate] rendu du rapport Quarto…", flush=True)
    subprocess.run(
        ["quarto", "render", "evaluation_report.qmd", "--to", "html",
         "--output", "evaluation_report.html"],
        cwd=here, env=env, check=True,
    )

    upload(out_html, output_s3)

    # Recalculé ici plutôt que récupéré du rapport : Quarto ne rend rien
    # d'exploitable en retour, et `internals` est justement partagé pour que les
    # deux passes donnent le même résultat.
    internal = flatten_internal(
        load_classifier_records(
            con, scorable, truth_col,
            ragnotices_path=env["EVAL_RAGNOTICES_PATH"],
            retrieved_path=env["EVAL_RETRIEVED_PATH"],
            ragann_path=env["EVAL_RAGANN_PATH"],
        )
    )
    log_to_mlflow(args, df, scorable, truth_col, output_s3, internal)
    print(f"[evaluate] terminé — {output_s3}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
