"""Produce final user-facing output from regex and LLM predictions."""

from __future__ import annotations

import argparse
import os
import sys

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import s3fs
from codif_common.contracts import artifact, run_root as contracts_run_root
from codif_common.s3 import connect_secret as init_duckdb, resolve_endpoint
from prune_codes.pruning import trunc_and_prune_lvl4


PIPELINE_COLS = {
    "l_pr_product",
    "s_pr_product",
    "code",
    "code_lvl4",
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
    p.add_argument(
        "--bucket",
        default=None,
        help="Surcharge le bucket du registre (contracts.yaml). Équivalent à $COICOP_BUCKET.",
    )
    p.add_argument("--text-column", default="raw_product",
                   help="Original input column name for product text")
    p.add_argument("--shop-column", default="shop",
                   help="Original input column name for shop")
    p.add_argument("--budget-column", default="budget",
                   help="Original input column name for budget")
    p.add_argument("--annee-column", default="annee",
                   help="Original input column name for year")
    p.add_argument(
        "--input-file", required=True,
        help="Fichier d'entrée du run. Sert à restaurer le nom du livrable ; "
             "les colonnes utilisateur, elles, viennent d'observations.parquet.",
    )
    p.add_argument(
        "--decision-source", choices=["llm", "sirus"], default="llm",
        help="Quelle étape de conciliation a produit le code final. `llm` "
             "(défaut) => reconcile-llm ; `sirus` => reconcile-sirus. Les deux "
             "sont exclusives (paramètre Argo `reconciliation`).",
    )
    p.add_argument(
        "--decision-file", default=None,
        help="Surcharge du parquet de conciliation à lire. Par défaut déduit de "
             "--decision-source et du run. Réservé aux lancements manuels : le "
             "workflow ne le passe pas, pour ne pas avoir deux façons de dire "
             "le même chemin.",
    )
    return p.parse_args()


# Schéma de décision de chaque conciliation, ramené aux noms internes utilisés
# par la logique de précédence ci-dessous. Nommer les colonnes honnêtement en
# amont (sirus_code et non llm_code) impose cette traduction, mais évite qu'un
# parquet mente sur son contenu.
DECISION_SCHEMAS = {
    "llm": {
        "step": "reconcile-llm",
        "code": "llm_code",
        "comment": "llm_explication",
        "confidence": "llm_confiance",
        "regime": "llm_model",
    },
    "sirus": {
        "step": "reconcile-sirus",
        "code": "sirus_code",
        "comment": None,    # SIRUS ne produit pas d'explication en texte
        "confidence": "sirus_proba",
        # Pas de régime : SIRUS livre un code et un score, sans verdict sur le
        # sort à leur réserver. Décider d'un seuil d'exploitation est une
        # question métier, instruite par la section « Calibration de SIRUS » du
        # rapport d'évaluation — pas figée dans le parquet de sortie.
        "regime": None,
    },
}


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
    print(f"[export-results] exported {len(df)} rows → {s3_uri}", flush=True)


def export_csv(df: pd.DataFrame, s3_uri: str) -> None:
    path = s3_uri.removeprefix("s3://")
    with _fs().open(path, "w") as f:
        df.to_csv(f, index=False)
    print(f"[export-results] exported {len(df)} rows → {s3_uri}", flush=True)


def main() -> int:
    args = parse_args()
    # Chemins depuis le registre : cette étape en désignait 4 en dur, dont
    # `mapping_lvl4.parquet`, invisible aussi bien dans les configs que dans le YAML.
    if args.bucket:
        os.environ["COICOP_BUCKET"] = args.bucket
    RUN = {"run_date": args.run_date, "run_id": args.run_id}
    run_root = contracts_run_root(**RUN)
    con = init_duckdb()

    # Base: all user columns from the preprocessing output (strip internal pipeline columns).
    base_path = artifact("build-datasets", "observations", **RUN)
    print(f"[export-results] loading base: {base_path}", flush=True)
    # Read full first to capture _source_input_file before stripping
    observations = con.sql(f"SELECT * FROM read_parquet('{base_path}')").df()
    input_file_path = (
        observations["_source_input_file"].iloc[0]
        if "_source_input_file" in observations.columns and len(observations) > 0
        else None
    )
    initial_cols = ["id"] + [c for c in observations.columns if c not in PIPELINE_COLS and c != "id"]
    raw = observations[initial_cols]
    print(f"[export-results] base: {len(raw)} rows, {len(initial_cols)} columns", flush=True)

    # Regex predictions: rows classified by regex
    print(f"[export-results] loading regex predictions: {artifact("classify-regex", "predictions", **RUN)}", flush=True)
    regex = con.sql(f"""
        SELECT id, predict_code
        FROM read_parquet('{artifact("classify-regex", "predictions", **RUN)}')
        WHERE predict_code IS NOT NULL
    """).df()
    print(f"[export-results] regex predictions: {len(regex)} rows", flush=True)

    # Décisions de la conciliation retenue (reconcile-llm ou reconcile-sirus).
    schema = DECISION_SCHEMAS[args.decision_source]
    decision_path = args.decision_file or artifact(schema["step"], "predictions", **RUN)
    cols = [c for c in (schema["code"], schema["comment"], schema["confidence"], schema["regime"]) if c]
    print(
        f"[export-results] loading {args.decision_source} decisions: {decision_path}",
        flush=True,
    )
    decisions = con.sql(f"""
        SELECT id, {", ".join(cols)}
        FROM read_parquet('{decision_path}')
    """).df()
    # Noms internes : la suite ne connaît plus la conciliation d'origine.
    decisions = decisions.rename(
        columns={
            schema["code"]: "_decision_code",
            schema["confidence"]: "_decision_confidence",
            **({schema["comment"]: "_decision_comment"} if schema["comment"] else {}),
            **({schema["regime"]: "_decision_regime"} if schema["regime"] else {}),
        }
    )
    # Une conciliation qui ne fournit ni explication ni régime voit les colonnes
    # internes créées vides : la suite est écrite une seule fois pour les deux.
    if not schema["comment"]:
        decisions["_decision_comment"] = pd.NA
    if not schema["regime"]:
        decisions["_decision_regime"] = pd.NA
    print(f"[export-results] {args.decision_source} decisions: {len(decisions)} rows", flush=True)

    result = raw.merge(regex[["id", "predict_code"]], on="id", how="left")
    result = result.merge(decisions, on="id", how="left")

    # Le code de la conciliation prime ; repli sur la regex ; sinon NA.
    result["predicted_code"] = result["_decision_code"].where(
        result["_decision_code"].notna(), result["predict_code"]
    )

    if args.decision_source == "llm":
        # reconcile-llm distingue deux régimes dans `llm_model` : le
        # court-circuit consensus et l'arbitrage effectif du juge.
        def _source(r):
            if pd.notna(r["_decision_code"]):
                return "consensus" if r["_decision_regime"] == "consensus" else "llm"
            return "regex" if pd.notna(r["predict_code"]) else None
    else:
        # SIRUS n'a pas de régime : il livre un code et un score, sans verdict.
        # Le score est exposé tel quel dans `prediction_confidence` ; c'est à
        # l'aval de décider, s'il le souhaite, à partir de quel niveau il
        # exploite le code sans relecture.
        def _source(r):
            if pd.notna(r["_decision_code"]):
                return "sirus"
            return "regex" if pd.notna(r["predict_code"]) else None

    result["prediction_source"] = result.apply(_source, axis=1)
    result["llm_comment"] = result["_decision_comment"]
    if args.decision_source == "sirus":
        result["prediction_confidence"] = result["_decision_confidence"]

    # Garde-fou final : quelle que soit la source de la décision (regex, consensus
    # ou LLM), le code final est tronqué au niveau 4 et élagué des hiérarchies
    # linéaires, en réutilisant la logique centralisée du module `prune`. Garantit
    # que la sortie ne contient que des codes prunés (idempotent sur un code déjà
    # pruné ; les codes NA restent NA).
    mapping_path = artifact("prune-codes", "mapping_lvl4", **RUN)
    print(f"[export-results] pruning final codes with: {mapping_path}", flush=True)
    mapping = con.sql(f"SELECT * FROM read_parquet('{mapping_path}')").df()
    result = trunc_and_prune_lvl4(result, mapping, code_name="predicted_code")
    result["predicted_code"] = result["predicted_code_tpruned"]
    result = result.drop(columns=["predicted_code_tpruned"])

    result = result.drop(
        columns=[
            "predict_code",
            "_decision_code",
            "_decision_comment",
            "_decision_confidence",
            "_decision_regime",
        ]
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
        print(f"[export-results] restored column names: {reverse_mapping}", flush=True)

    # Export with original input filename and format
    output_basename = os.path.basename(input_file_path) if input_file_path else "predictions.parquet"
    output_uri = artifact("export-results", "deliverable", **RUN, filename=output_basename)
    print(f"[export-results] output file: {output_uri}", flush=True)

    if output_basename.lower().endswith(".csv"):
        export_csv(result, output_uri)
    else:
        export_parquet(result, output_uri)

    return 0


if __name__ == "__main__":
    sys.exit(main())
