"""
Annotation RAG Pipeline
=======================
Codifies product descriptions with the annotation-based RAG.

For each product: embed it, retrieve the most similar already-coded products from
Qdrant, build a few-shot prompt from those examples, ask the LLM for a COICOP code,
parse the JSON answer, and export the predictions.

Un seul mode : prédire et exporter. L'évaluation se fait a posteriori, dans
l'étape finale `evaluate`, qui relit le parquet exporté ici.

Usage:
    uv run scripts/1_run_rag.py --run-id <ID> --run-date <YYYY-MM-DD>
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

from rag_annotations.eval import (
    detailed_evaluation,
    flatten_detailed,
    format_detailed_report,
)
from rag_annotations.generation_tools import generate_llm_responses
from rag_annotations.parsing import extract_json_from_response
from rag_annotations.prompt import load_prompt
from rag_annotations.utils import (
    build_location_text,
    create_duckdb_connection,
    embed_texts,
    expand_paths,
    is_present,
)
from codif_common.vector_index import validate_collection


def setup_argument_parser():
    parser = argparse.ArgumentParser(description="Annotation RAG pipeline")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config YAML")
    parser.add_argument("--run-id", required=True, help="Workflow run identifier")
    parser.add_argument("--run-date", required=True, help="Workflow run date (YYYY-MM-DD)")
    parser.add_argument("--sample_size", type=int, help="Limit number of products")
    parser.add_argument(
        "--collection-name",
        required=True,
        help=(
            "Nom complet de la collection Qdrant à interroger, produit par "
            "index-annotations-pipeline.yaml. Obligatoire : il n'y a plus de nom "
            "par défaut en config, précisément pour qu'un oubli échoue au lieu "
            "de retomber en silence sur l'index d'un autre run."
        ),
    )
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
    # Affecté après expand_paths : le nom vient d'Argo, pas de la config, et ne
    # doit pas traverser str.format() (une accolade y lèverait un KeyError).
    config["qdrant"]["collection_name"] = args.collection_name

    product_col = config["annotations"]["product_col"]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    input_path = config["data"]["s3_path_input"]

    # Valider l'index AVANT MLflow : échouer à l'intérieur d'un start_run
    # laisserait un run FAILED dans l'expérience.
    logger.info("Validation de la collection Qdrant...")
    _con_validate = create_duckdb_connection()
    _client_validate = QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ["QDRANT_API_KEY"],
        port=os.environ["QDRANT_API_PORT"],
    )
    index_manifest = validate_collection(
        con=_con_validate,
        client_qdrant=_client_validate,
        collection_name=config["qdrant"]["collection_name"],
        manifests_root=config["qdrant"]["manifests_root"],
        expected_dim=config["embedding"]["model_len"],
        expected_embedding_model=config["embedding"]["model_name"],
        param_name="classify-rag-annotations-collection",
        index_pipeline="argo/index-annotations-pipeline.yaml",
    )
    logger.info(
        "✓ Collection validée : %s (%s points)",
        config["qdrant"]["collection_name"],
        index_manifest["point_count_live"],
    )

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=f"annotation-rag_{timestamp}"):
        logger.info("=" * 80)
        logger.info("STARTING ANNOTATION RAG PIPELINE")
        logger.info("=" * 80)

        # Provenance de l'index : deux runs bâtis sur des index différents
        # seraient sinon indistinguables dans MLflow.
        mlflow.set_tag("index.git_sha", str(index_manifest.get("git_sha")))
        mlflow.set_tag("index.run_id", str(index_manifest.get("run_id")))

        mlflow.log_params({
            "collection_name": config["qdrant"]["collection_name"],
            "index_point_count": index_manifest["point_count_live"],
            "model_name": config["llm"]["model_name"],
            "embedding_model": config["embedding"]["model_name"],
            "retrieval_size": config["retrieval"]["size"],
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

        # Aucun filtre par source : on code tout ce qui arrive. Ce filtre
        # n'existait que pour l'évaluation, où certaines sources faussaient la
        # mesure ; la source est désormais une dimension d'analyse du rapport
        # d'évaluation, pas une restriction du périmètre codé.

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

        # L'évaluation a quitté cette étape pour l'étape finale `evaluate`, qui
        # la rejoue depuis le parquet exporté ci-dessus. `detailed_evaluation` et
        # ses fonctions restent dans `rag_annotations.eval`, désormais importées
        # de là-bas.

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
