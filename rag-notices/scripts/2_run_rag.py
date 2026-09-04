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
from pathlib import Path
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

from rag_notices.data.parsing import extract_json_from_response
from rag_notices.utils import create_duckdb_connection, expand_paths, merge_eval_and_retreived, truncate_code
from codif_common.vector_index import validate_collection
from rag_notices.generation_tools import generate_llm_responses



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

    # Affecté après expand_paths : le nom vient d'Argo, pas de la config, et ne
    # doit pas traverser str.format() (cf. merge_config_with_args).
    config['qdrant']['collection_name'] = args.collection_name

    logger.info(f"✓ Configuration loaded: {config['llm']['model_name']}")

    # ---------------------------------------------------------------------------
    # Valider l'index AVANT tout travail coûteux — et avant MLflow, pour ne pas
    # laisser un run FAILED dans l'expérience si la collection est mauvaise.
    # ---------------------------------------------------------------------------

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
        collection_name=config['qdrant']['collection_name'],
        manifests_root=config['qdrant']['manifests_root'],
        expected_dim=config["embedding"]["model_len"],
        expected_embedding_model=config["embedding"]["model_name"],
        expected_strategy=config["qdrant"]["strategy"],
        param_name="classify-rag-notices-collection",
        index_pipeline="argo/index-notices-pipeline.yaml",
    )
    logger.info(
        f"✓ Collection validée : {config['qdrant']['collection_name']} "
        f"({index_manifest['point_count_live']} points, "
        f"bâtie le {index_manifest.get('run_date')} par {index_manifest.get('run_id')})"
    )

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
        # Provenance de l'index : sans ça, deux runs aux métriques différentes
        # mais bâtis sur des index différents sont indistinguables.
        mlflow.set_tag("index.git_sha", str(index_manifest.get("git_sha")))
        mlflow.set_tag("index.run_id", str(index_manifest.get("run_id")))

        mlflow.log_params({
            "collection_name": config['qdrant']['collection_name'],
            "index_point_count": index_manifest["point_count_live"],
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
        observations, nature_annotation = load_and_prepare_annotations(con, config)

        mlflow.log_param("nature_annotation", nature_annotation)
        mlflow.log_metric("num_products", len(observations))

        # -----------------------------------------------------------------------
        # Execute main pipeline steps
        # -----------------------------------------------------------------------

        # Step 1: Generate embeddings
        search_embeddings, embedding_dim = generate_embeddings(
            observations,
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
            observations,
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
            observations=observations,
            qdrant_results_codes=qdrant_results_codes,
            con=con,
            path_mapping_lvl4=config["coicop"]["path_mapping_lvl4"],
        )

        # Step 7: Export RAG predictions
        eval_path, retrieved_path = export_predictions(
            con,
            df_eval,
            df_retrieved_codes,
            config,
        )

        mlflow.log_param("eval_output_path", eval_path)
        mlflow.log_param("retrieved_codes_output_path", retrieved_path)

        # L'évaluation a quitté cette étape pour l'étape finale `evaluate`, qui la
        # rejoue depuis les deux parquets exportés ci-dessus.
        #
        # Ce module était le seul à évaluer SANS CONDITION — il n'a jamais connu la
        # dualité production/évaluation. En production la vérité est nulle, et la
        # garde correspondante de `eval/metrics.py` est commentée : chaque run
        # loguait donc dans MLflow une accuracy ≈ 0, sans que rien n'échoue.

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
    # Chaque run Argo part d'un clone neuf : le dossier n'existe pas encore, et
    # FileHandler ne le crée pas. Sans ça, l'étape échoue à l'import.
    Path("logs").mkdir(parents=True, exist_ok=True)
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
        required=True,
        help=(
            'Nom complet de la collection Qdrant à interroger, produit par '
            'index-notices-pipeline.yaml. Obligatoire : il n\'y a plus de nom '
            'par défaut en config, précisément pour qu\'un oubli échoue au lieu '
            'de retomber en silence sur l\'index d\'un autre run.'
        ),
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
        
    # `collection_name` n'est délibérément PAS traité ici : cette fonction
    # tourne avant `expand_paths`, qui applique str.format() à toute chaîne de
    # la config. Un nom contenant une accolade y lèverait un KeyError. Il est
    # affecté après l'expansion, dans main().


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
        tuple: (observations, nature_annotation)
    """
    logger.info(f"Loading RAG annotations from: {config['annotations']['s3_path_rag']}")

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

    observations = annotations.to_dict(orient="records")

    # Apply sampling if configured
    sample_size = int(config["annotations"]["sample_size"]) if config["annotations"]["sample_size"] else 0
    if sample_size:
        random.seed(42)
        observations = random.sample(observations, sample_size)
        logger.info(f"✓ Sampling applied: {sample_size} products")

    logger.info(f"✓ Total products to process: {len(observations)}")

    return observations, nature_annotation


def generate_embeddings(observations, client_emb, config):
    """
    Generate embeddings for all product descriptions
    
    Args:
        observations: List of product dictionaries
        client_emb: OpenAI client for embedding generation
        config: Configuration dictionary
        
    Returns:
        list: List of embedding vectors
    """
    logger.info("=" * 80)
    logger.info("STEP 1: GENERATING EMBEDDINGS")
    logger.info("=" * 80)
    
    search_embeddings = []
    
    for observation in tqdm(observations, desc="Generating embeddings"):
        response = client_emb.embeddings.create(
            model=config["embedding"]["model_name"],
            input=observation['l_pr_product']
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


def prepare_prompts(observations, qdrant_results_texts, qdrant_results_codes, prompt_template):
    """
    Prepare prompts for LLM generation
    
    Args:
        observations: List of product dictionaries
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
    
    for i, observation in enumerate(observations):
        # Include store information if available
        shop = observation.get("shop") or None
        shop_type = observation.get("shop_type_name") or None
        if shop:
            shop_info = f"{shop} (type d'enseigne : {shop_type})" if shop_type else shop
            enseigne_bloc = (
                f"# Pour information, ce produit a été acheté dans cette enseigne : {shop_info}"
            )
        else:
            enseigne_bloc = None
        
        if observation["budget"] and isinstance(observation["budget"], float):
            price_bloc = (
                f"# Pour information, ce produit a coûté : {round(observation['budget'], 1)} euros."
            )
        else:
            price_bloc = None
        
        messages.append(
            prompt_template.compile(
                product=observation["l_pr_product"],
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
        observations,
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
        observations: Original product data with pruned annotations.
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
    for pred, annotation in zip(llm_responses_parsed, observations):
        rows.append(pred | annotation)

    df_eval = pd.DataFrame(rows)
    df_eval["method"] = "rag-notices"

    # ── 1. Truncate predictions to level 4 ───────────────────────────────────
    df_eval["code_predict"] = df_eval["code_predict"].apply(
        lambda c: truncate_code(c, level=4)
    )

    # ── 2. Apply pruning mapping ──────────────────────────────────────────────
    logger.info(f"Loading pruning mapping from: {path_mapping_lvl4}")
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


# ============================================================================
# Entry Point
# ============================================================================


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error in pipeline: {e}", exc_info=True)
        raise