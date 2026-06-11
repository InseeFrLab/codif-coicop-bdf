"""
Annotation Vector Database Creation
===================================
Imports annotated **train** data (product descriptions + COICOP code), prunes the
codes to level 4, then embeds ALL of them and uploads them to a Qdrant collection.

There is no train/test split here: the upstream preprocessing already separates
train annotations (used by the TTC classifier — and this input) from the test
annotations used to evaluate the pipeline. So every loaded annotation goes into
the vector DB; `1_run_rag.py` evaluates on the separate upstream test set.

Optionally, the **suggester** examples are added to the index as extra retrieval
candidates. Their codes are level-5 and unpruned upstream, so they go through the
exact same `prune_annotation_lvl4` (truncate to level 4 + linear-hierarchy mapping)
as the annotations — guaranteeing consistent codes in the vector DB.

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

from coicop_rag_annotations.pruning import prune_annotation_lvl4
from coicop_rag_annotations.utils import (
    build_location_text,
    create_duckdb_connection,
    embed_texts,
    expand_paths,
)


def load_and_prune_suggester(con, config, product_col, mapping_table_lvl4, notices_raw):
    """
    Load suggester examples and prune their codes exactly like the annotations.

    Reads the configured source (CSV or parquet), aligns its columns to
    (`product_col`, 'code', 'coicop'), tags `source='suggester'`, then applies the
    same `prune_annotation_lvl4` (truncate to level 4 + linear-hierarchy mapping).

    Returns the cleaned/pruned DataFrame.
    """
    sug = config["suggester"]
    reader = "read_parquet" if sug["s3_path"].endswith(".parquet") else "read_csv_auto"

    raw_count = con.sql(f"SELECT COUNT(*) FROM {reader}('{sug['s3_path']}')").fetchone()[0]

    # The ENTIRE suggester is indexed (no sampling). dedup=true only collapses exact
    # (product, code, coicop) duplicates, mirroring the preprocessing loader.
    distinct = "DISTINCT" if sug.get("dedup", True) else ""
    df = con.sql(f"""
        SELECT {distinct}
            "{sug['product_col']}" AS "{product_col}",
            "{sug['code_col']}"    AS code,
            "{sug['coicop_col']}"  AS coicop
        FROM {reader}('{sug['s3_path']}')
    """).to_df()
    logger.info(f"  → {raw_count} suggester rows in source → {len(df)} after dedup={sug.get('dedup', True)}")
    df["source"] = "suggester"

    # Same pruning as the annotations → consistent level-4, mapped codes.
    df = prune_annotation_lvl4(df, mapping_table_lvl4, notices_raw)

    before = len(df)
    df = df.dropna(subset=[product_col, "code"])
    df = df[df[product_col].astype(str).str.strip() != ""]
    logger.info(f"  → {len(df)} usable suggester rows (dropped {before - len(df)} empty/invalid)")
    return df


def main():
    parser = argparse.ArgumentParser(description="Annotation vector database creation")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config YAML")
    parser.add_argument("--run-id", required=True, help="Workflow run identifier")
    parser.add_argument("--run-date", required=True, help="Workflow run date (YYYY-MM-DD)")
    args = parser.parse_args()

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

    # Exclude configured sources from the vector DB
    exclude_sources = config["annotations"].get("exclude_sources") or []
    if exclude_sources:
        before = len(annotations)
        annotations = annotations[~annotations["source"].isin(exclude_sources)]
        logger.info(
            f"  → excluded sources {exclude_sources}: {before - len(annotations)} rows dropped "
            f"→ {len(annotations)} remaining"
        )

    # -----------------------------------------------------------------------
    # STEP 2: prune codes to level 4
    # -----------------------------------------------------------------------
    logger.info("STEP 2: pruning annotation codes to level 4")
    mapping_table_lvl4 = con.sql(
        f"SELECT code, code_parent_equivalent FROM read_parquet('{config['coicop']['path_mapping_lvl4']}')"
    ).to_df()
    notices_raw = con.sql(
        f"SELECT * FROM read_csv_auto('{config['coicop']['path_raw']}')"
    ).to_df()
    annotations = prune_annotation_lvl4(annotations, mapping_table_lvl4, notices_raw)

    # Drop rows without a usable product description or code
    before = len(annotations)
    annotations = annotations.dropna(subset=[product_col, "code"])
    annotations = annotations[annotations[product_col].astype(str).str.strip() != ""]
    logger.info(f"  → {len(annotations)} usable annotations (dropped {before - len(annotations)})")

    # -----------------------------------------------------------------------
    # STEP 3: add suggester examples to the index
    # -----------------------------------------------------------------------
    # No train/test split here: preprocessing already split train (this input)
    # from the evaluation test set. ALL loaded annotations are indexed.
    index_df = annotations
    suggester_excluded = "suggester" in exclude_sources
    if config.get("suggester", {}).get("enabled") and not suggester_excluded:
        logger.info("STEP 3: adding suggester examples to the index")
        suggester_df = load_and_prune_suggester(
            con, config, product_col, mapping_table_lvl4, notices_raw
        )
        index_df = pd.concat([annotations, suggester_df], ignore_index=True)
        logger.info(
            f"  → index = {len(annotations)} annotations + {len(suggester_df)} suggester "
            f"= {len(index_df)} rows"
        )
    elif suggester_excluded:
        logger.info("STEP 3: suggester excluded via exclude_sources — index = annotations only")
    else:
        logger.info("STEP 3: suggester disabled — index = annotations only")

    # -----------------------------------------------------------------------
    # STEP 4: embed index product descriptions
    # -----------------------------------------------------------------------
    logger.info("STEP 4: generating embeddings for index products")
    train_records = index_df.to_dict(orient="records")
    # Canonical text = product + purchase location (shop / shop type) when known.
    texts = [
        build_location_text(r[product_col], r.get("shop"), r.get("shop_type_name"))
        for r in train_records
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
        for record, text, embedding in zip(train_records, texts, embeddings)
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
