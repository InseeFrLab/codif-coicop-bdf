"""
Vector Database Creation
========================
Embeds pruned COICOP notices and uploads them to a Qdrant vector database.
"""
import argparse
import logging
import os
import uuid

import yaml
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from coicop_rag.data.coicop_document import CoicopDocument
from coicop_rag.utils import create_duckdb_connection, get_parents


def main():
    parser = argparse.ArgumentParser(description="Vector database creation pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config YAML file"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    logger.info("=" * 80)
    logger.info("STARTING VECTOR DATABASE CREATION PIPELINE")
    logger.info("=" * 80)

    # -----------------------------------------------------------------------
    # Initialize clients
    # -----------------------------------------------------------------------

    con = create_duckdb_connection()

    client_emb = OpenAI(
        base_url=os.environ["VLLM_EMBEDDING_URL"],
        api_key=os.environ["VLLM_EMBEDDING_API_KEY"]
    )
    model_name = client_emb.models.list().data[0].id
    logger.info(f"Embedding model: {model_name}")

    client_qdrant = QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ["QDRANT_API_KEY"],
        port=os.environ["QDRANT_API_PORT"]
    )

    # -----------------------------------------------------------------------
    # Load COICOP notices
    # -----------------------------------------------------------------------

    logger.info("=" * 80)
    logger.info("STEP 1: LOADING COICOP NOTICES")
    logger.info("=" * 80)

    vectors_cfg = config["coicop"].get("vectors", {})
    truncate_level = vectors_cfg.get("truncate_level")
    prune = vectors_cfg.get("prune", False)
    logger.info(f"  truncate_level={truncate_level}, prune={prune}")

    notices_df = con.sql(
        f"SELECT * FROM read_csv_auto('{config['coicop']['path_raw']}')"
    ).to_df()

    columns_to_keep = [
        col for col in notices_df.columns
        if "column" not in col.lower() and not col.endswith("_en")
    ]
    notices_df = notices_df[columns_to_keep]
    logger.info(f"  → {len(notices_df)} notices loaded from path_raw")

    # Apply truncation: keep codes whose depth (number of dot-separated parts) <= truncate_level
    if truncate_level is not None:
        before = len(notices_df)
        notices_df = notices_df[
            notices_df["code"].apply(lambda c: len(str(c).split(".")) <= truncate_level)
        ]
        logger.info(f"  → {len(notices_df)} notices after truncation to level {truncate_level} (dropped {before - len(notices_df)})")

    # Apply pruning: keep only canonical codes (code == code_parent_equivalent in mapping)
    if prune:
        mapping_df = con.sql(
            f"SELECT code, code_parent_equivalent FROM read_parquet('{config['coicop']['path_mapping_lvl4']}')"
        ).to_df()
        canonical_codes = set(
            mapping_df.loc[mapping_df["code"] == mapping_df["code_parent_equivalent"], "code"]
        )
        before = len(notices_df)
        notices_df = notices_df[notices_df["code"].isin(canonical_codes)]
        logger.info(f"  → {len(notices_df)} notices after pruning (dropped {before - len(notices_df)} non-canonical codes)")

    notices = notices_df.to_dict(orient="records")
    logger.info(f"✓ {len(notices)} notices ready")

    # -----------------------------------------------------------------------
    # Enrich with parent lineage
    # -----------------------------------------------------------------------

    logger.info("=" * 80)
    logger.info("STEP 2: ENRICHING WITH PARENT LINEAGE")
    logger.info("=" * 80)

    for notice in notices:
        code = notice["code"]
        notice["parents"] = get_parents(code)
        notice["parents_labels"] = (
            notices_df.loc[
                notices_df["code"].isin(notice["parents"]),
                "label_fr"
            ].to_list()
        )
    logger.info(f"✓ Parent lineage added to {len(notices)} notices")

    # -----------------------------------------------------------------------
    # Build document chunks
    # -----------------------------------------------------------------------

    logger.info("=" * 80)
    logger.info("STEP 3: BUILDING DOCUMENT CHUNKS")
    logger.info("=" * 80)

    strategy = config["qdrant"]["strategy"]
    logger.info(f"Strategy: {strategy}")

    documents = []
    for notice in notices:
        doc = CoicopDocument(
            code=str(notice["code"]),
            label_fr=str(notice["label_fr"]),
            note_generale_fr=notice.get("note_generale_fr"),
            contenu_central_fr=notice.get("contenu_central_fr"),
            contenu_additionnel_fr=notice.get("contenu_additionnel_fr"),
            note_exclusion_fr=notice.get("note_exclusion_fr"),
            parents=notice.get("parents"),
            parents_labels=notice.get("parents_labels"),
        )
        chunk = doc.to_text_chunks(strategy=strategy)
        documents.append({
            "id": str(uuid.uuid4()),
            "text": chunk["text"],
            "metadata": {
                "code": doc.code,
                "label_fr": doc.label_fr,
                "strategy": chunk["type"],
            }
        })

    logger.info(f"✓ {len(documents)} document chunks created")

    # -----------------------------------------------------------------------
    # Create Qdrant collection
    # -----------------------------------------------------------------------

    logger.info("=" * 80)
    logger.info("STEP 4: CREATING QDRANT COLLECTION")
    logger.info("=" * 80)

    collection_name = config["qdrant"]["collection_name"]
    if client_qdrant.collection_exists(collection_name):
        client_qdrant.delete_collection(collection_name)
        logger.info(f"  → Existing collection deleted: {collection_name}")

    client_qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=config["embedding"]["model_len"],
            distance=Distance.COSINE
        )
    )
    logger.info(f"✓ Collection created: {collection_name}")

    # -----------------------------------------------------------------------
    # Generate embeddings
    # -----------------------------------------------------------------------

    logger.info("=" * 80)
    logger.info("STEP 5: GENERATING EMBEDDINGS")
    logger.info("=" * 80)

    embeddings = []
    for i, document in enumerate(documents):
        try:
            response = client_emb.embeddings.create(
                model=model_name,
                input=document["text"]
            )
            embeddings.append(response.data[0].embedding)
            if (i + 1) % 10 == 0:
                logger.info(f"  → {i + 1}/{len(documents)} embeddings generated")
        except Exception as e:
            logger.error(f"Failed to generate embedding for document {i}: {e}")
            continue

    logger.info(f"✓ {len(embeddings)} embeddings generated")

    # -----------------------------------------------------------------------
    # Upload to Qdrant
    # -----------------------------------------------------------------------

    logger.info("=" * 80)
    logger.info("STEP 6: UPLOADING TO QDRANT")
    logger.info("=" * 80)

    points = [
        PointStruct(
            id=document["id"],
            vector=embedding,
            payload={
                "text": document["text"],
                **document["metadata"]
            }
        )
        for document, embedding in zip(documents, embeddings)
    ]

    upload_batch_size = config["qdrant"]["upload_batch_size"]
    n_batches = (len(points) - 1) // upload_batch_size + 1
    logger.info(f"Uploading {len(points)} points in {n_batches} batches of {upload_batch_size}")

    for i in range(0, len(points), upload_batch_size):
        batch = points[i:i + upload_batch_size]
        batch_num = i // upload_batch_size + 1
        try:
            client_qdrant.upsert(
                collection_name=config["qdrant"]["collection_name"],
                points=batch
            )
            logger.info(f"  → Batch {batch_num}/{n_batches} uploaded")
        except Exception as e:
            logger.error(f"Failed to upload batch {batch_num}: {e}")
            continue

    logger.info("=" * 80)
    logger.info("VECTOR DATABASE CREATION PIPELINE COMPLETED SUCCESSFULLY!")
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
