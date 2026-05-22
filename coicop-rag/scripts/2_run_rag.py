"""
RAG COICOP Pipeline
===================
Pipeline for automatic COICOP coding using RAG (Retrieval-Augmented Generation)
"""
import os
import yaml
import datetime
import logging
import argparse
from tqdm import tqdm
import duckdb
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qdrant_client import QdrantClient
from openai import OpenAI
from langfuse import Langfuse
import mlflow
import subprocess
import random

from coicop_rag.data.parsing import extract_json_from_response
from coicop_rag.utils import create_duckdb_connection, expand_paths, merge_eval_and_retreived, truncate_code
from coicop_rag.eval.metrics import (
    compute_hierarchical_metrics,
    calculate_accuracy_at_level,
    flatten_metrics,
    write_metrics_report,
)
from coicop_rag.generation_tools import generate_llm_responses



# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    """Main pipeline execution"""
    
    logger.info("=" * 80)
    logger.info("STARTING RAG COICOP PIPELINE")
    logger.info("=" * 80)
    
    # ---------------------------------------------------------------------------
    # Parse arguments and load configuration
    # ---------------------------------------------------------------------------
    
    parser = setup_argument_parser()
    args = parser.parse_args()

    # config = load_config("config.yaml")
    config = load_config(args.config)
    config = merge_config_with_args(config, args)
    config = expand_paths(config, run_id=args.run_id, run_date=args.run_date)

    logger.info(f"✓ Configuration loaded: {config['llm']['model_name']}")

    # Timestamp for MLflow run names and plots; no longer used in S3 paths
    # (run_id already uniquely identifies the run folder).
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # ---------------------------------------------------------------------------
    # Setup MLflow tracking
    # ---------------------------------------------------------------------------
    
    logger.info("Setting up MLflow experiment tracking...")
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"run_{timestamp}"):
        logger.info(f"✓ MLflow run started: {mlflow.active_run().info.run_id}")
        mlflow.set_tag("git.commit", get_git_commit_hash())
        mlflow.set_tag("git.branch", get_git_branch())
        mlflow.set_tag("git.repo", "https://github.com/InseeFrLab/coicop-rag")

        
        # Log parameters
        mlflow.log_params({
            "collection_name": config['qdrant']['collection_name'],
            "model_name": config["llm"]["model_name"],
            "embedding_model": config["embedding"]["model_name"],
            "temperature": config["llm"]["temperature"],
            "max_tokens": config["llm"]["max_tokens"],
            "retrieval_size": config["retrieval"]["size"],
            "sample_size": config["annotations"]["sample_size"],
            "prompt_name": config["llm"]["prompt_name"],
            "prompt_version": config["llm"]["prompt_version"],
            "threshold_confidence": config["eval"]["threshold_confidence"],
        })
        
        # -----------------------------------------------------------------------
        # Initialize external service connections
        # -----------------------------------------------------------------------
        
        con, client_qdrant, client_llmlab = initialize_clients(config)
        
        # -----------------------------------------------------------------------
        # Load prompt template
        # -----------------------------------------------------------------------
        
        prompt_template = load_prompt_template(config)
        
        # -----------------------------------------------------------------------
        # Load and prepare annotations
        # -----------------------------------------------------------------------

        # Import RAG products (already split by 1_split_rules.py)
        mlflow.log_param("input_data_path", config['annotations']['s3_path_rag'])
        searched_products_rag, nature_annotation = load_and_prepare_annotations(con, config)

        mlflow.log_param("nature_annotation", nature_annotation)
        mlflow.log_metric("num_products", len(searched_products_rag))

        # -----------------------------------------------------------------------
        # Execute main pipeline steps
        # -----------------------------------------------------------------------

        # Step 1: Generate embeddings
        search_embeddings, embedding_dim = generate_embeddings(
            searched_products_rag,
            client_llmlab,
            config
        )
        mlflow.log_param("embedding_dimension", embedding_dim)
        
        # Step 2: Vector search
        qdrant_results_texts, qdrant_results_codes = perform_vector_search(
            search_embeddings,
            client_qdrant,
            config
        )
        
        # Step 3: Prepare prompts
        messages = prepare_prompts(
            searched_products_rag,
            qdrant_results_texts,
            qdrant_results_codes,
            prompt_template
        )

        log_prompts_sample(messages, n=6)
        
        # Step 4: Generate LLM responses
        llm_responses = generate_llm_responses(
            messages,
            client_llmlab,
            config,
            concurrency=config["llm"].get("concurrency", 8),
        )
        
        # Step 5: Parse responses
        llm_responses_parsed, n_parse_errors = parse_llm_responses(llm_responses)
        mlflow.log_metric("parse_errors", n_parse_errors)
        
        # Step 6: Create evaluation dataset
        df_eval, df_retrieved_codes = create_evaluation_dataframe(
            llm_responses_parsed=llm_responses_parsed,
            searched_products=searched_products_rag,
            qdrant_results_codes=qdrant_results_codes,
            con=con,
            path_mapping_lvl4=config["coicop"]["path_mapping_lvl4"],
        )

        # Step 6b: Plot confidence vs accuracy
        fig = plot_confidence_vs_accuracy(df_eval, level=4)
        mlflow.log_figure(fig, "confidence_vs_accuracy.png")
        plt.close(fig)

        # Step 7: Export RAG predictions
        eval_path, retrieved_path = export_predictions(
            con,
            df_eval,
            df_retrieved_codes,
            config,
        )
        
        mlflow.log_param("eval_output_path", eval_path)
        mlflow.log_param("retrieved_codes_output_path", retrieved_path)
        
        # Step 8: Compute and log metrics

        metrics, by_nature_metrics = compute_and_log_metrics(
            df_eval,
            df_retrieved_codes,
            config,
        )

        # Step 9 : get sample of tricky errors

        df_tricky_errors = get_tricky_errors(
            sample_size=40,
            df_eval=df_eval,
            df_retrieved_codes=df_retrieved_codes,
            config=config,
            level=4,
        )

        mlflow.log_table(
            df_tricky_errors, 
            artifact_file="tricky_errors.json"
        )
        
        # -----------------------------------------------------------------------
        # Generate and save metrics report
        # -----------------------------------------------------------------------
        
        logger.info("=" * 80)
        logger.info("GENERATING METRICS REPORT")
        logger.info("=" * 80)
        
        # write_metrics_report(metrics, "report.txt")
        write_metrics_report(
            metrics=metrics,
            output_path="report.txt",
            include_product_types=True,
            include_comparison=True,
            by_nature_metrics=by_nature_metrics,
            )
        
        mlflow.log_artifact("report.txt", artifact_path="reports")
        logger.info("✓ Metrics report saved")
        
        # Log config as artifact
        mlflow.log_dict(config, "config.yaml")
        
        # -----------------------------------------------------------------------
        # Pipeline completion
        # -----------------------------------------------------------------------
        
        logger.info("=" * 80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info(f"MLflow run ID: {mlflow.active_run().info.run_id}")
        logger.info("=" * 80)



# ============================================================================
# Logging Configuration
# ============================================================================

def setup_logging():
    """Configure logging with both console and file handlers"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                f'logs/pipeline_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
            )
        ]
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    return logging.getLogger(__name__)


logger = setup_logging()


# ============================================================================
# Configuration Management
# ============================================================================

def load_config(config_path='config.yaml'):
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    logger.info(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_argument_parser():
    """
    Setup command-line argument parser
    
    Arguments override values from config.yaml when provided
    """
    parser = argparse.ArgumentParser(
        description='RAG COICOP Pipeline - Automatic COICOP coding',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration file
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to config YAML file'
    )
    
    # Sample size override
    parser.add_argument(
        '--sample_size',
        type=int,
        help='Number of products to sample (overrides config)'
    )
    
    # Model parameters
    parser.add_argument(
        '--model_name',
        type=str,
        help='LLM model name (overrides config)'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        help='LLM temperature (overrides config)'
    )
    
    parser.add_argument(
        '--max_tokens',
        type=int,
        help='Maximum tokens for LLM generation (overrides config)'
    )
    
    # Retrieval parameters
    parser.add_argument(
        '--retrieval_size',
        type=int,
        help='Number of documents to retrieve (overrides config)'
    )
    
    # Data parameters
    parser.add_argument(
        '--collection_name',
        type=str,
        help='Qdrant collection name (overrides config)'
    )
    
    parser.add_argument(
        '--nature_annotation',
        type=str,
        help='Type of annotation to filter (overrides config)'
    )
    
    # Evaluation parameters
    parser.add_argument(
        '--threshold_confidence',
        type=float,
        help='Confidence threshold for evaluation (overrides config)'
    )
    
    # MLflow parameters
    parser.add_argument(
        '--experiment_name',
        type=str,
        help='MLflow experiment name (overrides config)'
    )

    # Workflow run identity
    parser.add_argument(
        '--run-id',
        required=True,
        help='Workflow run identifier'
    )
    parser.add_argument(
        '--run-date',
        required=True,
        help='Workflow run date (YYYY-MM-DD)'
    )

    return parser


def merge_config_with_args(config, args):
    """
    Merge command-line arguments with config file
    Command-line arguments take precedence over config file values
    
    Args:
        config: Configuration dictionary from YAML
        args: Parsed command-line arguments
        
    Returns:
        dict: Merged configuration
    """
    # Override config values with command-line arguments if provided
    if args.sample_size is not None:
        config['annotations']['sample_size'] = args.sample_size
        
    if args.model_name is not None:
        config['llm']['model_name'] = args.model_name
        
    if args.temperature is not None:
        config['llm']['temperature'] = args.temperature
        
    if args.max_tokens is not None:
        config['llm']['max_tokens'] = args.max_tokens
        
    if args.retrieval_size is not None:
        config['retrieval']['size'] = args.retrieval_size
        
    if args.collection_name is not None:
        config['qdrant']['collection_name'] = args.collection_name
        
    if args.nature_annotation is not None:
        config['annotations']['nature'] = args.nature_annotation
        
    if args.threshold_confidence is not None:
        config['eval']['threshold_confidence'] = args.threshold_confidence
        
    if args.experiment_name is not None:
        config['mlflow']['experiment_name'] = args.experiment_name
    
    return config


# ============================================================================
# Pipeline Steps
# ============================================================================

def initialize_clients(config):
    """
    Initialize connections to external services
    
    Args:
        config: Configuration dictionary
        
    Returns:
        tuple: (duckdb_connection, qdrant_client, llm_client)
    """
    logger.info("Initializing external service connections...")
    
    # DuckDB connection
    logger.info("  → Connecting to DuckDB...")
    con = create_duckdb_connection()
    
    # Qdrant connection
    logger.info("  → Connecting to Qdrant...")
    client_qdrant = QdrantClient(
        url=os.environ["QDRANT_URL"], 
        api_key=os.environ["QDRANT_API_KEY"],
        port=os.environ["QDRANT_API_PORT"]
    )
    logger.info(f"  → Qdrant collection: {config['qdrant']['collection_name']}")
    
    # LLM connection — llm.lab (génération et embedding sur le même serveur)
    logger.info("  → Connecting to llm.lab...")
    client_llmlab = OpenAI(
        base_url=os.environ["LLMLAB_URL"],
        api_key=os.environ["LLMLAB_API_KEY"],
    )

    available = [m.id for m in client_llmlab.models.list().data]

    expected_gen_model = config["llm"]["model_name"]
    if expected_gen_model not in available:
        raise ValueError(
            f"Modèle de génération '{expected_gen_model}' absent de llm.lab — disponibles : {available}"
        )
    logger.info("✔ Modèle de génération '%s' disponible sur llm.lab", expected_gen_model)

    expected_emb_model = config["embedding"]["model_name"]
    if expected_emb_model not in available:
        raise ValueError(
            f"Modèle d'embedding '{expected_emb_model}' absent de llm.lab — disponibles : {available}"
        )
    logger.info("✔ Modèle d'embedding '%s' disponible sur llm.lab", expected_emb_model)

    logger.info("✓ All clients initialized successfully")

    return con, client_qdrant, client_llmlab


def load_prompt_template(config):
    """
    Load prompt template from Langfuse
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Prompt template object
    """
    logger.info("Loading prompt template from Langfuse...")
    
    prompt_template = Langfuse().get_prompt(
        config["llm"]["prompt_name"], 
        version=int(config["llm"]["prompt_version"])
    )
    
    logger.info(
        f"✓ Prompt loaded: {config['llm']['prompt_name']} "
        f"v{config['llm']['prompt_version']}"
    )
    
    return prompt_template


def load_and_prepare_annotations(con, config):
    """
    Load RAG annotations from S3 (produced by 1_split_rules.py).

    Args:
        con: DuckDB connection
        config: Configuration dictionary

    Returns:
        tuple: (searched_products, nature_annotation)
    """
    logger.info("Loading RAG annotations...")

    annotations = con.sql(
        f"SELECT * FROM read_parquet('{config['annotations']['s3_path_rag']}')"
    ).to_df()

    nature_annotation = config["annotations"]["nature"]
    if nature_annotation:
        annotations = annotations.loc[annotations["source"] == nature_annotation]

    logger.info(
        f"✓ Annotations loaded: {len(annotations)} rows "
        f"(type: {nature_annotation or 'all'})"
    )

    searched_products = annotations.to_dict(orient="records")

    # Apply sampling if configured
    sample_size = int(config["annotations"]["sample_size"]) if config["annotations"]["sample_size"] else 0
    if sample_size:
        random.seed(42)
        searched_products = random.sample(searched_products, sample_size)
        logger.info(f"✓ Sampling applied: {sample_size} products")

    logger.info(f"✓ Total products to process: {len(searched_products)}")

    return searched_products, nature_annotation


def generate_embeddings(searched_products, client_emb, config):
    """
    Generate embeddings for all product descriptions
    
    Args:
        searched_products: List of product dictionaries
        client_emb: OpenAI client for embedding generation
        config: Configuration dictionary
        
    Returns:
        list: List of embedding vectors
    """
    logger.info("=" * 80)
    logger.info("STEP 1: GENERATING EMBEDDINGS")
    logger.info("=" * 80)
    
    search_embeddings = []
    
    for searched_product in tqdm(searched_products, desc="Generating embeddings"):
        response = client_emb.embeddings.create(
            model=config["embedding"]["model_name"],
            input=searched_product['l_pr_product']
        )
        search_embeddings.append(response.data[0].embedding)
    
    embedding_dim = len(search_embeddings[0])
    logger.info(
        f"✓ Embeddings generated: {len(search_embeddings)} vectors "
        f"(dimension: {embedding_dim})"
    )
    
    return search_embeddings, embedding_dim


def perform_vector_search(search_embeddings, client_qdrant, config):
    """
    Perform vector search in Qdrant to retrieve relevant documents
    
    Args:
        search_embeddings: List of embedding vectors
        client_qdrant: Qdrant client
        config: Configuration dictionary
        
    Returns:
        tuple: (texts, codes) - Retrieved document texts and COICOP codes
    """
    logger.info("=" * 80)
    logger.info("STEP 2: VECTOR SEARCH IN QDRANT")
    logger.info("=" * 80)
    
    qdrant_results_texts = []
    qdrant_results_codes = []
    
    for search_embedding in tqdm(search_embeddings, desc="Vector search"):
        points = client_qdrant.query_points(
            collection_name=config["qdrant"]["collection_name"],
            query=search_embedding,
            limit=config["retrieval"]["size"],
        )
        
        qdrant_results_texts.append(
            [point["payload"]["text"] for point in points.model_dump()["points"]]
        )
        qdrant_results_codes.append(
            [point["payload"]["code"] for point in points.model_dump()["points"]]
        )
    
    logger.info(
        f"✓ Vector searches completed: {len(qdrant_results_texts)} searches, "
        f"{len(qdrant_results_texts[0])} points per search"
    )
    
    return qdrant_results_texts, qdrant_results_codes


def prepare_prompts(searched_products, qdrant_results_texts, qdrant_results_codes, prompt_template):
    """
    Prepare prompts for LLM generation
    
    Args:
        searched_products: List of product dictionaries
        qdrant_results_texts: Retrieved document texts
        qdrant_results_codes: Retrieved COICOP codes
        prompt_template: Langfuse prompt template
        
    Returns:
        list: List of compiled prompt messages
    """
    logger.info("=" * 80)
    logger.info("STEP 3: PREPARING PROMPTS")
    logger.info("=" * 80)
    
    messages = []
    
    for i, searched_product in enumerate(searched_products):
        # Include store information if available
        shop = searched_product.get("shop") or None
        shop_type = searched_product.get("shop_type_name") or None
        if shop:
            shop_info = f"{shop} (type d'enseigne : {shop_type})" if shop_type else shop
            enseigne_bloc = (
                f"# Pour information, ce produit a été acheté dans cette enseigne : {shop_info}"
            )
        else:
            enseigne_bloc = None
        
        if searched_product["budget"] and isinstance(searched_product["budget"], float):
            price_bloc = (
                f"# Pour information, ce produit a coûté : {round(searched_product['budget'], 1)} euros."
            )
        else:
            price_bloc = None
        
        messages.append(
            prompt_template.compile(
                product=searched_product["l_pr_product"],
                enseigne_bloc=enseigne_bloc,
                price_bloc=price_bloc,
                proposed_codes="\n\n## ".join(qdrant_results_texts[i]),
                list_proposed_codes=qdrant_results_codes[i]
            )
        )
    
    logger.info(f"✓ Prompts prepared: {len(messages)}")
    
    return messages


def log_prompts_sample(messages, n, base_filename: str = "prompts/prompt"):
    n_max = len(messages)
    n = n_max if n > n_max else n
    index = random.sample(range(n_max), n)
    messages_to_log = [messages[m] for m in index]

    for idx, prompt in enumerate(messages_to_log):
        filename = f"{base_filename}_{idx}.md"
        # Concatène le contenu de tous les messages dans le prompt
        text = "\n\n".join(f"### {msg['role'].capitalize()}\n{msg['content']}" for msg in prompt)
        mlflow.log_text(text, filename)


# def generate_llm_responses(messages, client_gen, config):
#     """
#     Generate predictions using LLM
    
#     Args:
#         messages: List of prompt messages
#         client_gen: OpenAI client for generation
#         config: Configuration dictionary
        
#     Returns:
#         list: List of LLM response objects
#     """
#     logger.info("=" * 80)
#     logger.info("STEP 4: LLM GENERATION")
#     logger.info("=" * 80)
    
#     llm_responses = []
    
#     for message in tqdm(messages, desc="LLM generation"):
#         llm_responses.append(
#             client_gen.chat.completions.create(
#                 model=config["llm"]["model_name"],
#                 messages=message,
#                 temperature=config["llm"]["temperature"],
#                 max_tokens=config["llm"]["max_tokens"],
#                 response_format={"type": "json_object"}
#             )
#         )
    
#     logger.info(f"✓ LLM responses generated: {len(llm_responses)}")
    
#     return llm_responses


def parse_llm_responses(llm_responses):
    """
    Parse JSON responses from LLM
    
    Args:
        llm_responses: List of LLM response objects
        
    Returns:
        tuple: (parsed_responses, parse_errors_count)
    """
    logger.info("Parsing LLM responses...")
    
    llm_responses_parsed = []

    for idx, llm_response in enumerate(llm_responses):
        # Case 1: generation failed (worker returned None)
        if llm_response is None:
            logger.warning("Response %d is None (generation failed)", idx)
            llm_responses_parsed.append({'parsed': False})
            continue

        content = llm_response.choices[0].message.content or ""
        # "stop"   → model finished normally
        # "length" → truncated by max_tokens (JSON is likely incomplete)
        finish_reason = llm_response.choices[0].finish_reason

        try:
            parsed = extract_json_from_response(content)

            # Case 2: extract_json_from_response did not raise but failed to
            # parse the JSON (returns {'parsed': False})
            if not parsed.get('parsed', False):
                logger.warning(
                    "Response %d not parsed — finish_reason=%s — content: %r",
                    idx, finish_reason, content[:200],
                )
            llm_responses_parsed.append(parsed)

        except Exception as e:
            # Case 3: unexpected exception during parsing
            logger.warning(
                "Response %d parsing exception: %s — finish_reason=%s — content: %r",
                idx, e, finish_reason, content[:200],
            )
            llm_responses_parsed.append({'parsed': False})

    parse_errors = sum(dic == {'parsed': False} for dic in llm_responses_parsed)
    
    logger.info(
        f"✓ Responses parsed: {len(llm_responses_parsed)} "
        f"({parse_errors} errors)"
    )
    
    return llm_responses_parsed, parse_errors


def create_evaluation_dataframe(
        llm_responses_parsed,
        searched_products,
        qdrant_results_codes,
        con,
        path_mapping_lvl4: str,
    ):
    """
    Create evaluation dataframe combining predictions and ground truth.

    The new vector DB contains all COICOP codes (unpruned, levels 1-5), so
    LLM predictions can be at any level. This function:
      1. Truncates ``code_predict`` to level 4  → stored in ``code_predict``.
      2. Applies the pruning mapping table to the truncated code
         → stored in ``code_predict_tprune``.

    The ground-truth ``code`` column in the annotations is already pruned
    (produced by the upstream pruning step).

    Args:
        llm_responses_parsed: Parsed LLM responses.
        searched_products: Original product data with pruned annotations.
        qdrant_results_codes: Retrieved COICOP codes.
        con: Active DuckDB connection (used to load the mapping table).
        path_mapping_lvl4: S3 path to the level-4 pruning mapping parquet.
            Expected columns: ``code`` (level-4 code) and
            ``code_parent_equivalent`` (its pruned equivalent).

    Returns:
        tuple: (evaluation_df, retrieved_codes_df)
    """
    logger.info("=" * 80)
    logger.info("STEP 5: CREATING EVALUATION DATASET")
    logger.info("=" * 80)

    rows = []
    for pred, annotation in zip(llm_responses_parsed, searched_products):
        rows.append(pred | annotation)

    df_eval = pd.DataFrame(rows)
    df_eval["method"] = "rag-notices"

    # ── 1. Truncate predictions to level 4 ───────────────────────────────────
    df_eval["code_predict"] = df_eval["code_predict"].apply(
        lambda c: truncate_code(c, level=4)
    )

    # ── 2. Apply pruning mapping ──────────────────────────────────────────────
    mapping = con.sql(
        f"SELECT code, code_parent_equivalent FROM read_parquet('{path_mapping_lvl4}')"
    ).df()
    code_to_pruned = mapping.set_index("code")["code_parent_equivalent"].to_dict()

    df_eval["code_predict"] = df_eval["code_predict"].apply(
        lambda c: code_to_pruned.get(c, c)   # keep as-is if not in mapping
    )

    # ── 3. Build retrieved codes dataframe, truncated and pruned ─────────────
    def _truncate_and_prune(code: str) -> str:
        return code_to_pruned.get(truncate_code(code, level=4), truncate_code(code, level=4))

    df_retrieved_codes = pd.DataFrame(qdrant_results_codes)
    df_retrieved_codes.columns = df_retrieved_codes.columns.astype(str)
    code_cols = [c for c in df_retrieved_codes.columns if c != "id"]
    for col in code_cols:
        df_retrieved_codes[col] = df_retrieved_codes[col].apply(_truncate_and_prune)
    df_retrieved_codes["id"] = df_eval["id"]

    logger.info(f"✓ Evaluation dataset created: {len(df_eval)} rows")

    return df_eval, df_retrieved_codes
  


def plot_confidence_vs_accuracy(df_eval: pd.DataFrame, level: int = 4) -> plt.Figure:
    """
    Plot confidence (from LLM) against prediction accuracy at a given COICOP level.

    Produces two vertically stacked subplots:
      - Top: accuracy per confidence bin (bar chart)
      - Bottom: number of predictions per bin (histogram)

    Only rows with parsed=True and codable=True are included.

    Args:
        df_eval: Evaluation dataframe with columns 'confidence', 'code_predict',
                 'code', 'parsed', 'codable'.
        level: COICOP level at which to evaluate accuracy.

    Returns:
        matplotlib Figure
    """
    df = df_eval.copy()

    # Filter to parsed & codable rows
    mask = df.get("parsed", pd.Series(True, index=df.index)) == True
    if "codable" in df.columns:
        mask &= df["codable"] == True
    df = df[mask].copy()

    if df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data (parsed & codable)", ha="center", va="center")
        return fig

    # Correct prediction at the requested level
    df["correct"] = df.apply(
        lambda r: truncate_code(str(r["code_predict"]), level) == truncate_code(str(r["code"]), level),
        axis=1,
    )
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df = df.dropna(subset=["confidence"])

    # Weighted mean accuracy (computed on all rows, not as mean of bin means)
    overall_accuracy = df["correct"].mean()

    bins = np.arange(0.0, 1.05, 0.1)
    bin_labels = [f"{b:.1f}–{b+0.1:.1f}" for b in bins[:-1]]
    df["bin"] = pd.cut(df["confidence"], bins=bins, labels=bin_labels, include_lowest=True)

    grouped = df.groupby("bin", observed=False)["correct"]
    accuracy = grouped.mean()
    counts = grouped.count()
    proportions = counts / counts.sum()

    # Drop bins with no data for display
    has_data = counts > 0
    accuracy = accuracy[has_data]
    counts = counts[has_data]
    proportions = proportions[has_data]
    x_labels = list(accuracy.index.astype(str))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(
        f"Confidence vs Accuracy (level {level}) — parsed & codable (n={len(df)})",
        fontsize=13, fontweight="bold"
    )

    # Top: accuracy bars
    colors = ["#d9534f" if v < 0.5 else "#5cb85c" if v >= 0.7 else "#f0ad4e"
              for v in accuracy.values]
    bars = ax1.bar(x_labels, accuracy.values, color=colors, edgecolor="white", width=0.8)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Accuracy")
    ax1.axhline(overall_accuracy, color="steelblue", linestyle="--", linewidth=1.2,
                label=f"Overall accuracy = {overall_accuracy:.2%}")
    ax1.legend(fontsize=9)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    for bar, val in zip(bars, accuracy.values):
        if not np.isnan(val):
            ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                     f"{val:.0%}", ha="center", va="bottom", fontsize=8)

    # Bottom: proportion histogram
    ax2.bar(x_labels, proportions.values, color="steelblue", edgecolor="white", width=0.8, alpha=0.7)
    ax2.set_ylabel("Proportion")
    ax2.set_xlabel("Confidence bin")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    for bar, val in zip(ax2.patches, proportions.values):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.003,
                 f"{val:.0%}", ha="center", va="bottom", fontsize=8)

    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    return fig


def export_predictions(con, df_eval, df_retrieved_codes, config):
    """
    Export predictions to S3

    Args:
        con: DuckDB connection
        df_eval: Evaluation dataframe
        df_retrieved_codes: Retrieved codes dataframe
        config: Configuration dictionary (paths already expanded with run_id/run_date)

    Returns:
        tuple: (eval_path, retrieved_path)
    """
    logger.info("=" * 80)
    logger.info("STEP 6: EXPORTING PREDICTIONS")
    logger.info("=" * 80)

    eval_path = config['predictions']['s3_path']
    retrieved_path = config['predictions']['s3_path_retrieved_codes']
    
    # Export evaluation results
    con.sql(f"""
        COPY df_eval 
        TO '{eval_path}'
        (FORMAT PARQUET)
    """)
    logger.info(f"✓ Predictions exported: {eval_path}")
    
    # Export retrieved codes
    con.sql(f"""
        COPY df_retrieved_codes 
        TO '{retrieved_path}'
        (FORMAT PARQUET)
    """)
    logger.info(f"✓ Retrieved codes exported: {retrieved_path}")
    
    return eval_path, retrieved_path


def get_git_commit_hash():
    """Récupère le hash du commit Git actuel"""
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD']
        ).decode('ascii').strip()
    except:
        return None


def get_git_branch():
    """Récupère la branche Git actuelle"""
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD']
        ).decode('ascii').strip()
    except:
        return None


def compute_and_log_metrics(df_eval, df_retrieved_codes, config):
    """
    Compute evaluation metrics and log to MLflow.

    Also computes metrics broken down by annotation nature (``source`` column)
    if more than one nature is present in df_eval.

    Args:
        df_eval: Evaluation dataframe
        df_retrieved_codes: Retrieved codes dataframe
        config: Configuration dictionary

    Returns:
        tuple: (metrics, by_nature_metrics) where by_nature_metrics is a
        dict {nature: metrics_dict} or None if only one nature is present.
    """
    logger.info("=" * 80)
    logger.info("STEP 7: COMPUTING METRICS")
    logger.info("=" * 80)

    records = merge_eval_and_retreived(
        df_eval=df_eval,
        retrieved_codes=df_retrieved_codes,
        retrieval_size=config["retrieval"]["size"],
        code_name="code",
        col_retrieved_codes_name="list_retrieved_codes",
    )

    metrics = compute_hierarchical_metrics(
        records=records,
        threshold=config["eval"]["threshold_confidence"],
        predicted_col="code_predict",
        label_col="code",
        retrieved_col="list_retrieved_codes"
    )

    # Only log overall metrics to MLflow (by_product_type is in the report)
    metrics_mlflow = flatten_metrics(metrics, include_product_types=False)
    mlflow.log_metrics(metrics_mlflow)

    # Per-nature metrics (only if multiple natures present)
    by_nature_metrics = None
    if "source" in df_eval.columns:
        natures = df_eval["source"].dropna().unique().tolist()
        if len(natures) > 1:
            logger.info(f"  → Computing metrics per annotation nature: {natures}")
            by_nature_metrics = {}
            for nature in sorted(natures):
                nature_ids = set(df_eval.loc[df_eval["source"] == nature, "id"])
                nature_records = [r for r in records if r.get("id") in nature_ids]
                by_nature_metrics[nature] = compute_hierarchical_metrics(
                    records=nature_records,
                    threshold=config["eval"]["threshold_confidence"],
                    predicted_col="code_predict",
                    label_col="code",
                    retrieved_col="list_retrieved_codes",
                    by_product_type=False,
                )

    logger.info("✓ Metrics computed and logged")

    return metrics, by_nature_metrics


def get_tricky_errors(
    sample_size: int,
    df_eval,
    df_retrieved_codes,
    config,
    level,
):
    """
    Return a random sample of hard prediction errors for qualitative analysis.

    "Hard" errors are wrong predictions on products that are:
      - codable (the LLM flagged them as codable)
      - not in the uncodable/misc categories (codes starting with 98 or 99)

    Args:
        sample_size: Maximum number of errors to return.
        df_eval: Evaluation dataframe (predictions + annotations).
        df_retrieved_codes: Retrieved codes dataframe.
        config: Configuration dictionary (used for retrieval_size).
        level: COICOP level at which to evaluate correctness.

    Returns:
        DataFrame with columns: l_pr_product, shop, code, code_predict,
        confidence, codable, budget, in_retrieved.
    """
    # Merge predictions with retrieved codes into flat records
    records = merge_eval_and_retreived(
        df_eval=df_eval,
        retrieved_codes=df_retrieved_codes,
        retrieval_size=config["retrieval"]["size"],
        code_name="code",
        col_retrieved_codes_name="list_retrieved_codes",
    )

    # Identify which records are wrong at the requested level
    (
        overall_accuracy,
        result_list,
        retrieval_accuracy,
        generation_accuracy_when_retrieved,
        label_in_retrieved_list
    ) = calculate_accuracy_at_level(
        records=records,
        predicted_col="code_predict",
        label_col="code",
        level=level,
        retrieved_col="list_retrieved_codes"
    )

    # Keep only wrong predictions on codable, non-misc products
    errors_list = [x for x, m in zip(records, result_list) if not m]
    codable_errors = [x for x in errors_list if x["codable"]]
    real_errors = [x for x in codable_errors if x["code"] is not None and (x["code"][:2] not in ("98", "99"))]

    keys_to_keep = [
        "l_pr_product", "shop", "code",
        "code_predict", "confidence", "codable", "budget",
        "in_retrieved"
    ]
    sample_size = min(sample_size, len(real_errors))
    real_errors_sample = random.sample(real_errors, sample_size)

    res = [{ k: e.get(k) for k in keys_to_keep } for e in real_errors_sample]
    return pd.DataFrame(res)



# ============================================================================
# Entry Point
# ============================================================================


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error in pipeline: {e}", exc_info=True)
        raise