"""
Annotation Vector Database Creation
===================================
Reads the **already-pruned** annotation KB (`prune/annotations_train_pruned.parquet`,
produced by the `prune` module), embeds ALL of it and uploads it to a Qdrant collection.

There is no train/test split here: the upstream `preprocessing` already separates
train annotations from the test annotations. So every loaded annotation goes into
the vector DB; `1_run_rag.py` codifies the separate (also pruned) input set.

Optionally, the **suggester** examples (`prune/suggester_pruned.parquet`, already pruned)
are added to the index as extra retrieval candidates. Source filtering is applied here
(`exclude_sources` / `exclude_sources_prod` per `--skip-eval`); the KB can be capped with
`--sample-size`.

Usage:
    uv run scripts/0_build_annotation_vector_db.py --run-id <ID> --run-date <YYYY-MM-DD>
"""
import argparse
import logging
import os
import uuid

import pandas as pd
import yaml
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from coicop_rag_annotations.utils import (
    build_location_text,
    create_duckdb_connection,
    embed_texts,
    expand_paths,
)


def main():
    parser = argparse.ArgumentParser(description="Annotation vector database creation")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config YAML")
    parser.add_argument("--run-id", required=True, help="Workflow run identifier")
    parser.add_argument("--run-date", required=True, help="Workflow run date (YYYY-MM-DD)")
    parser.add_argument(
        "--skip-eval", default="true",
        help="Mirrors the run step: 'true' = production (default), else evaluation. "
             "Selects which sources are excluded from the index "
             "(annotations.exclude_sources_prod vs exclude_sources).",
    )
    parser.add_argument(
        "--sample-size", type=int, default=None,
        help="Limite le nombre d'annotations indexées dans la vector DB (KB). "
             "Vide = toutes. Échantillonnage sans remise, graine eval.seed.",
    )
    args = parser.parse_args()

    # Production indexe un périmètre de sources plus large que l'évaluation
    # (cf. exclude_sources_prod) ; do_eval suit la même convention que 1_run_rag.py.
    do_eval = str(args.skip_eval).lower() != "true"

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config = expand_paths(config, run_id=args.run_id, run_date=args.run_date)

    logger.info("=" * 80)
    logger.info("STARTING ANNOTATION VECTOR DB CREATION")
    logger.info("=" * 80)

    con = create_duckdb_connection()
    client_llmlab = OpenAI(
        base_url=os.environ["LLMLAB_URL"],
        api_key=os.environ["LLMLAB_API_KEY"],
    )
    client_qdrant = QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ["QDRANT_API_KEY"],
        port=os.environ["QDRANT_API_PORT"],
    )

    product_col = config["annotations"]["product_col"]

    # -----------------------------------------------------------------------
    # STEP 1: load annotations
    # -----------------------------------------------------------------------
    logger.info("STEP 1: loading annotations")
    annotations = con.sql(
        f"SELECT * FROM read_parquet('{config['annotations']['s3_path']}')"
    ).to_df()
    annotations = annotations[config["annotations"]["features"]]

    nature = config["annotations"]["nature"]
    if nature:
        annotations = annotations.loc[annotations["source"] == nature]
    logger.info(f"  → {len(annotations)} annotations loaded (nature={nature or 'all'})")

    # Exclude configured sources from the vector DB.
    # Évaluation : exclude_sources (KB = lot train restreint).
    # Production  : exclude_sources_prod (KB = toutes les annotations voulues).
    if do_eval:
        exclude_sources = config["annotations"].get("exclude_sources") or []
    else:
        exclude_sources = config["annotations"].get("exclude_sources_prod") or []
    logger.info(f"  → mode={'evaluation' if do_eval else 'production'}, exclude_sources={exclude_sources}")
    if exclude_sources:
        before = len(annotations)
        annotations = annotations[~annotations["source"].isin(exclude_sources)]
        logger.info(
            f"  → excluded sources {exclude_sources}: {before - len(annotations)} rows dropped "
            f"→ {len(annotations)} remaining"
        )

    # -----------------------------------------------------------------------
    # STEP 2: cleanup (les codes sont déjà tronqués/prunés en amont par `prune`)
    # -----------------------------------------------------------------------
    logger.info("STEP 2: drop des lignes sans libellé ou code exploitable")
    before = len(annotations)
    annotations = annotations.dropna(subset=[product_col, "code"])
    annotations = annotations[annotations[product_col].astype(str).str.strip() != ""]
    logger.info(f"  → {len(annotations)} usable annotations (dropped {before - len(annotations)})")

    # Échantillonnage optionnel de la KB (sans remise, graine reproductible).
    if args.sample_size and args.sample_size < len(annotations):
        seed = config.get("eval", {}).get("seed", 42)
        annotations = annotations.sample(n=args.sample_size, random_state=seed)
        logger.info(f"  → KB échantillonnée à {len(annotations)} annotations (seed={seed})")

    # -----------------------------------------------------------------------
    # STEP 3: add suggester examples to the index (déjà prunés par l'étape `prune`)
    # -----------------------------------------------------------------------
    # No train/test split here: preprocessing already split train (this input)
    # from the evaluation test set. ALL loaded annotations are indexed.
    kb_data = annotations
    suggester_excluded = "suggester" in exclude_sources
    if config.get("suggester", {}).get("enabled") and not suggester_excluded:
        logger.info("STEP 3: adding suggester examples to the index")
        suggester_df = con.sql(
            f"SELECT * FROM read_parquet('{config['suggester']['s3_path_pruned']}')"
        ).to_df()
        kb_data = pd.concat([annotations, suggester_df], ignore_index=True)
        logger.info(
            f"  → index = {len(annotations)} annotations + {len(suggester_df)} suggester "
            f"= {len(kb_data)} rows"
        )
    elif suggester_excluded:
        logger.info("STEP 3: suggester excluded via exclude_sources — index = annotations only")
    else:
        logger.info("STEP 3: suggester disabled — index = annotations only")

    # -----------------------------------------------------------------------
    # STEP 4: embed index product descriptions
    # -----------------------------------------------------------------------
    logger.info("STEP 4: generating embeddings for index products")
    kb_records = kb_data.to_dict(orient="records")
    # Canonical text = product + purchase location (shop / shop type) when known.
    texts = [
        build_location_text(r[product_col], r.get("shop"), r.get("shop_type_name"))
        for r in kb_records
    ]
    embeddings = embed_texts(
        client_llmlab,
        config["embedding"]["model_name"],
        texts,
        batch_size=config["embedding"]["batch_size"],
    )
    logger.info(f"  → {len(embeddings)} embeddings generated")

    # -----------------------------------------------------------------------
    # STEP 5: create Qdrant collection and upload
    # -----------------------------------------------------------------------
    logger.info("STEP 5: creating collection and uploading points")
    collection_name = config["qdrant"]["collection_name"]
    if client_qdrant.collection_exists(collection_name):
        client_qdrant.delete_collection(collection_name)
        logger.info(f"  → existing collection deleted: {collection_name}")
    client_qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=config["embedding"]["model_len"], distance=Distance.COSINE
        ),
    )

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "text": text,                       # enriched text (embedded + shown as example)
                "product": str(record[product_col]),  # raw product, for reference
                "code": record["code"],
                "coicop": record.get("coicop"),
                "source": record.get("source"),
            },
        )
        for record, text, embedding in zip(kb_records, texts, embeddings)
    ]

    batch_size = config["qdrant"]["upload_batch_size"]
    n_batches = (len(points) - 1) // batch_size + 1
    for i in range(0, len(points), batch_size):
        client_qdrant.upsert(collection_name=collection_name, points=points[i:i + batch_size])
        logger.info(f"  → batch {i // batch_size + 1}/{n_batches} uploaded")

    logger.info("=" * 80)
    logger.info("ANNOTATION VECTOR DB CREATION COMPLETED")
    logger.info(f"  collection: {collection_name} ({len(points)} points)")
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
