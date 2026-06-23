"""
Annotation RAG Pipeline
=======================
Codifies product descriptions with the annotation-based RAG.

For each product: embed it, retrieve the most similar already-coded products from
Qdrant, build a few-shot prompt from those examples, ask the LLM for a COICOP code,
parse the JSON answer, and export the predictions.

Two modes (mirrors the workflow's `skip-report` convention):
  - PRODUCTION (default): predict + export only. Input has no labels.
  - EVALUATION (`--skip-eval false`): additionally compute per-level accuracy and
    retrieval recall against the ground-truth `code` of a labeled input.

Usage:
    # production
    uv run scripts/1_run_rag.py --run-id <ID> --run-date <YYYY-MM-DD>
    # evaluation
    uv run scripts/1_run_rag.py --run-id <ID> --run-date <YYYY-MM-DD> --skip-eval false
"""
import argparse
import datetime
import logging
import os
import time

import mlflow
import pandas as pd
import yaml
from openai import OpenAI
from qdrant_client import QdrantClient
from tqdm import tqdm

# to remove
# os.chdir("codif-coicop-bdf/coicop-rag-annotations")
# import types
# args = types.SimpleNamespace(config="config/config.yaml",run_id="pipeline-5jqsd", run_date="2026-06-03",sample_size=100,model_name=None, experiment_name=None,)

from coicop_rag_annotations.eval import (
    detailed_evaluation,
    flatten_detailed,
    format_detailed_report,
)
from coicop_rag_annotations.generation_tools import generate_llm_responses
from coicop_rag_annotations.parsing import extract_json_from_response
from coicop_rag_annotations.prompt import load_prompt
from coicop_rag_annotations.utils import (
    build_location_text,
    create_duckdb_connection,
    embed_texts,
    expand_paths,
    is_present,
)


def setup_argument_parser():
    parser = argparse.ArgumentParser(description="Annotation RAG pipeline")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config YAML")
    parser.add_argument("--run-id", required=True, help="Workflow run identifier")
    parser.add_argument("--run-date", required=True, help="Workflow run date (YYYY-MM-DD)")
    parser.add_argument("--sample_size", type=int, help="Limit number of products")
    parser.add_argument(
        "--sample-annotations", type=int, default=None,
        help="Taille de la KB d'annotations (doit correspondre à ce qui a été passé à "
             "0_build_annotation_vector_db.py). Non vide → utilise la collection de test "
             "(suffix '_test' sur collection_name).",
    )
    parser.add_argument("--model_name", type=str, help="LLM model name (overrides config)")
    parser.add_argument("--experiment_name", type=str, help="MLflow experiment (overrides config)")
    parser.add_argument(
        "--skip-eval", default="true",
        help="If != 'true', evaluate predictions against ground-truth labels "
             "(evaluation mode). Default 'true' = production mode (predictions only).",
    )
    return parser


def build_examples_block(texts, codes) -> str:
    """Format retrieved (product, code) pairs as a numbered reference list."""
    return "\n".join(
        f'{i + 1}. "{text}" → code COICOP : {code}'
        for i, (text, code) in enumerate(zip(texts, codes))
    )


def main():
    args = setup_argument_parser().parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    if args.model_name:
        config["llm"]["model_name"] = args.model_name
    if args.experiment_name:
        config["mlflow"]["experiment_name"] = args.experiment_name
    config = expand_paths(config, run_id=args.run_id, run_date=args.run_date)
    if args.sample_annotations:
        config["qdrant"]["collection_name"] = config["qdrant"]["collection_name"] + "_test"

    product_col = config["annotations"]["product_col"]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Evaluation is opt-in (mirrors the workflow's `skip-report` convention):
    # default = production (predictions only); `--skip-eval false` = evaluation mode.
    do_eval = str(args.skip_eval).lower() != "true"
    input_path = config["data"]["s3_path_input"]

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=f"annotation-rag_{timestamp}"):
        logger.info("=" * 80)
        logger.info("STARTING ANNOTATION RAG PIPELINE (%s mode)",
                    "EVALUATION" if do_eval else "PRODUCTION")
        logger.info("=" * 80)

        mlflow.log_params({
            "collection_name": config["qdrant"]["collection_name"],
            "model_name": config["llm"]["model_name"],
            "embedding_model": config["embedding"]["model_name"],
            "retrieval_size": config["retrieval"]["size"],
            "mode": "evaluation" if do_eval else "production",
            "input": input_path,
            "vectordb_exclude_sources": ",".join(config["annotations"].get("exclude_sources") or []) or "none",
            "testset_include_sources": ",".join(config["eval"].get("include_sources") or []) or "all",
        })

        # -------------------------------------------------------------------
        # Clients + prompt
        # -------------------------------------------------------------------
        con = create_duckdb_connection()
        client_llmlab = OpenAI(
            base_url=os.environ["LLMLAB_URL"], api_key=os.environ["LLMLAB_API_KEY"]
        )
        client_qdrant = QdrantClient(
            url=os.environ["QDRANT_URL"],
            api_key=os.environ["QDRANT_API_KEY"],
            port=os.environ["QDRANT_API_PORT"],
        )
        prompt_template = load_prompt(config)

        # -------------------------------------------------------------------
        # Load input products to codify (labeled test set in eval mode,
        # unlabeled production data otherwise)
        # -------------------------------------------------------------------
        logger.info("STEP 1: loading input products from %s", input_path)
        observations = con.sql(
            f"SELECT * FROM read_parquet('{input_path}')"
        ).to_df()

        # Keep only configured sources in the test set (empty = all).
        # Production input carries source="prediction" (no ground truth) and must
        # NOT be filtered out → the source filter only applies in evaluation mode.
        include_sources = config["eval"].get("include_sources") or []
        if do_eval and include_sources and "source" in observations.columns:
            before = len(observations)
            observations = observations[observations["source"].isin(include_sources)]
            logger.info(
                f"  → test set filtered to sources {include_sources}: {before} → {len(observations)} rows"
            )
        elif not do_eval:
            logger.info("  → production mode: no source filter applied to the input")

        if args.sample_size:
            observations = observations.sample(n=min(args.sample_size, len(observations)),
                                     random_state=config["eval"]["seed"])
        observations_records = observations.to_dict(orient="records")
        mlflow.log_metric("num_products", len(observations_records))
        logger.info(f"  → {len(observations_records)} test products")

        # -------------------------------------------------------------------
        # Embed + retrieve
        # -------------------------------------------------------------------
        logger.info("STEP 2: embedding test products")
        # Same canonical representation as the index: product + purchase location.
        search_texts = [
            build_location_text(r[product_col], r.get("shop"), r.get("shop_type_name"))
            for r in observations_records
        ]
        search_embeddings = embed_texts(
            client_llmlab,
            config["embedding"]["model_name"],
            search_texts,
            batch_size=config["embedding"]["batch_size"],
        )

        logger.info("STEP 3: retrieving similar annotated examples from Qdrant")
        retrieved_texts, retrieved_codes = [], []
        for emb in tqdm(search_embeddings, desc="Vector search"):
            points = client_qdrant.query_points(
                collection_name=config["qdrant"]["collection_name"],
                query=emb,
                limit=config["retrieval"]["size"],
            ).model_dump()["points"]
            retrieved_texts.append([p["payload"]["text"] for p in points])
            retrieved_codes.append([p["payload"]["code"] for p in points])

        # -------------------------------------------------------------------
        # Build prompts + generate
        # -------------------------------------------------------------------
        logger.info("STEP 4: building prompts")
        messages = []
        for i, record in enumerate(observations_records):
            budget = is_present(record.get("budget"))
            price_bloc = (
                f"# Prix payé : {round(float(budget), 1)} euros"
                if budget is not None and float(budget) > 0 else ""
            )
            messages.append(
                prompt_template.compile(
                    # product carries the purchase location (same repr. as the index
                    # and the examples) → enseigne_bloc no longer needed.
                    product=search_texts[i],
                    examples=build_examples_block(retrieved_texts[i], retrieved_codes[i]),
                    enseigne_bloc="",
                    price_bloc=price_bloc,
                )
            )

        logger.info("STEP 5: generating LLM responses")
        t_start = time.perf_counter()
        llm_responses = generate_llm_responses(
            messages, client_llmlab, config, concurrency=config["llm"]["concurrency"]
        )
        inference_seconds = time.perf_counter() - t_start
        mlflow.log_metrics({
            "inference/total_time_seconds": inference_seconds,
            "inference/iterations_per_second": (
                len(messages) / inference_seconds if inference_seconds > 0 else 0.0
            ),
        })
        logger.info(
            "  → inference: %.1fs for %d products (%.2f it/s)",
            inference_seconds, len(messages),
            len(messages) / inference_seconds if inference_seconds > 0 else 0.0,
        )

        # -------------------------------------------------------------------
        # Parse + assemble records
        # -------------------------------------------------------------------
        logger.info("STEP 6: parsing responses")
        records = []
        n_parse_errors = 0
        for response, record, retr_codes in zip(llm_responses, observations_records, retrieved_codes):
            if response is None:
                parsed = {"parsed": False}
            else:
                parsed = extract_json_from_response(response.choices[0].message.content or "")
            if not parsed.get("parsed"):
                n_parse_errors += 1
            records.append({
                **record,
                "code_predict": parsed.get("code_predict"),
                "codable": parsed.get("codable"),
                "confidence": parsed.get("confidence"),
                "parsed": parsed.get("parsed", False),
                "list_retrieved_codes": list(retr_codes),
                "method": "rag-annotations",
            })
        mlflow.log_metric("parse_errors", n_parse_errors)
        logger.info(f"  → {n_parse_errors} parse errors / {len(records)}")

        # -------------------------------------------------------------------
        # Export predictions
        # -------------------------------------------------------------------
        logger.info("STEP 7: exporting predictions")
        df_pred = pd.DataFrame(records)
        df_pred_export = df_pred.copy()
        df_pred_export["list_retrieved_codes"] = df_pred_export["list_retrieved_codes"].apply(list)
        pred_path = config["data"]["s3_path_predictions"]
        con.sql(f"COPY df_pred_export TO '{pred_path}' (FORMAT PARQUET)")
        mlflow.log_param("predictions_path", pred_path)
        logger.info(f"  → predictions exported: {pred_path}")

        mlflow.log_dict(config, "config.yaml")

        # -------------------------------------------------------------------
        # Evaluate (opt-in: only in evaluation mode, and only if labels exist)
        # -------------------------------------------------------------------
        if not do_eval:
            logger.info("STEP 8: evaluation skipped (production mode)")
        elif "code" not in observations.columns:
            logger.warning(
                "STEP 8: evaluation requested but no ground-truth 'code' column in input "
                "— skipping evaluation."
            )
        else:
            logger.info("STEP 8: evaluating")
            target_level = config["eval"]["levels"]
            detail = detailed_evaluation(records, max_level=target_level, target_level=target_level)

            report = format_detailed_report(detail)
            print("\n" + report)
            with open("report.txt", "w", encoding="utf-8") as f:
                f.write(report)
            mlflow.log_artifact("report.txt")

            # Scalar metrics
            mlflow.log_metrics(flatten_detailed(detail))

            # Detailed tables (viewable in the MLflow UI)
            mlflow.log_table(pd.DataFrame(detail["by_level1"]),
                             "eval/accuracy_by_level1.json")
            for lvl, d in detail["distortion"].items():
                mlflow.log_table(pd.DataFrame(d["per_category"]),
                                 f"eval/distribution_level_{lvl}.json")
            mlflow.log_table(pd.DataFrame(detail["confidence"]["calibration_bins"]),
                             "eval/confidence_calibration.json")
            mlflow.log_table(pd.DataFrame(detail["confidence"]["threshold_sweep"]),
                             "eval/confidence_threshold_sweep.json")
            mlflow.log_table(pd.DataFrame(detail["codable"]["groups"]),
                             "eval/codable_reliability.json")

        logger.info("=" * 80)
        logger.info("ANNOTATION RAG PIPELINE COMPLETED")
        logger.info("=" * 80)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    main()
