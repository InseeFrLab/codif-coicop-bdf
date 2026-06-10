"""
Annotation RAG Pipeline
=======================
Runs the annotation-based RAG on the upstream evaluation test set and evaluates it.

For each test product: embed it, retrieve the most similar already-coded products
from Qdrant, build a few-shot prompt from those examples, ask the LLM for a COICOP
code, parse the JSON answer, then compute per-level accuracy and retrieval recall.

Usage:
    uv run scripts/1_run_rag.py --run-id <ID> --run-date <YYYY-MM-DD> [--sample_size N]
"""
import argparse
import datetime
import logging
import os

import mlflow
import pandas as pd
import yaml
from openai import OpenAI
from qdrant_client import QdrantClient
from tqdm import tqdm

from coicop_rag_annotations.eval import evaluate, flatten_metrics, format_report
from coicop_rag_annotations.generation_tools import generate_llm_responses
from coicop_rag_annotations.parsing import extract_json_from_response
from coicop_rag_annotations.prompt import load_prompt
from coicop_rag_annotations.utils import create_duckdb_connection, embed_texts, expand_paths


def setup_argument_parser():
    parser = argparse.ArgumentParser(description="Annotation RAG pipeline")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config YAML")
    parser.add_argument("--run-id", required=True, help="Workflow run identifier")
    parser.add_argument("--run-date", required=True, help="Workflow run date (YYYY-MM-DD)")
    parser.add_argument("--sample_size", type=int, help="Limit number of test products")
    parser.add_argument("--model_name", type=str, help="LLM model name (overrides config)")
    parser.add_argument("--experiment_name", type=str, help="MLflow experiment (overrides config)")
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

    product_col = config["annotations"]["product_col"]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=f"annotation-rag_{timestamp}"):
        logger.info("=" * 80)
        logger.info("STARTING ANNOTATION RAG PIPELINE")
        logger.info("=" * 80)

        mlflow.log_params({
            "collection_name": config["qdrant"]["collection_name"],
            "model_name": config["llm"]["model_name"],
            "embedding_model": config["embedding"]["model_name"],
            "retrieval_size": config["retrieval"]["size"],
            "test_set": config["eval"]["s3_path_test"],
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
        # Load evaluation test set (produced upstream by prune-annotations)
        # -------------------------------------------------------------------
        logger.info("STEP 1: loading evaluation test set (upstream)")
        test_df = con.sql(
            f"SELECT * FROM read_parquet('{config['eval']['s3_path_test']}')"
        ).to_df()
        if args.sample_size:
            test_df = test_df.sample(n=min(args.sample_size, len(test_df)),
                                     random_state=config["eval"]["seed"])
        test_records = test_df.to_dict(orient="records")
        mlflow.log_metric("num_products", len(test_records))
        logger.info(f"  → {len(test_records)} test products")

        # -------------------------------------------------------------------
        # Embed + retrieve
        # -------------------------------------------------------------------
        logger.info("STEP 2: embedding test products")
        search_embeddings = embed_texts(
            client_llmlab, config["embedding"]["model_name"],
            [str(r[product_col]) for r in test_records],
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
        for i, record in enumerate(test_records):
            shop = record.get("shop") or None
            shop_type = record.get("shop_type_name") or None
            if shop:
                shop_info = f"{shop} (type d'enseigne : {shop_type})" if shop_type else shop
                enseigne_bloc = f"# Enseigne d'achat : {shop_info}"
            else:
                enseigne_bloc = ""
            budget = record.get("budget")
            price_bloc = (
                f"# Prix payé : {round(budget, 1)} euros"
                if isinstance(budget, float) and budget else ""
            )
            messages.append(
                prompt_template.compile(
                    product=record[product_col],
                    examples=build_examples_block(retrieved_texts[i], retrieved_codes[i]),
                    enseigne_bloc=enseigne_bloc,
                    price_bloc=price_bloc,
                )
            )

        logger.info("STEP 5: generating LLM responses")
        llm_responses = generate_llm_responses(
            messages, client_llmlab, config, concurrency=config["llm"]["concurrency"]
        )

        # -------------------------------------------------------------------
        # Parse + assemble records
        # -------------------------------------------------------------------
        logger.info("STEP 6: parsing responses")
        records = []
        n_parse_errors = 0
        for response, record, retr_codes in zip(llm_responses, test_records, retrieved_codes):
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

        # -------------------------------------------------------------------
        # Evaluate
        # -------------------------------------------------------------------
        logger.info("STEP 8: evaluating")
        metrics = evaluate(records, levels=range(1, config["eval"]["levels"] + 1))
        report = format_report(metrics, n_total=len(records))
        print("\n" + report)

        with open("report.txt", "w", encoding="utf-8") as f:
            f.write(report)
        mlflow.log_artifact("report.txt")
        mlflow.log_metrics(flatten_metrics(metrics))
        mlflow.log_dict(config, "config.yaml")

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
