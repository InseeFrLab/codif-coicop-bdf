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

from rag_notices.data.coicop_document import CoicopDocument
from rag_notices.utils import create_duckdb_connection, expand_paths, get_parents
from codif_common.vector_index import (
    build_collection_name,
    dumps,
    git_sha,
    log_paste_banner,
    write_manifest,
)


def main():
    parser = argparse.ArgumentParser(description="Vector database creation pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config YAML file"
    )
    parser.add_argument("--run-id", required=True, help="Workflow run identifier")
    parser.add_argument("--run-date", required=True, help="Workflow run date (YYYY-MM-DD)")
    parser.add_argument(
        "--collection-name",
        default=None,
        help=(
            "Nom complet de la collection à créer. Par défaut, composé depuis "
            "`qdrant.collection_base` et l'identité de CE run d'indexation. "
            "À ne renseigner que pour rejouer un nom précis."
        ),
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config = expand_paths(config, run_id=args.run_id, run_date=args.run_date)

    logger.info("=" * 80)
    logger.info("STARTING VECTOR DATABASE CREATION PIPELINE")
    logger.info("=" * 80)

    # -----------------------------------------------------------------------
    # Initialize clients
    # -----------------------------------------------------------------------

    con = create_duckdb_connection()

    client_llmlab = OpenAI(
        base_url=os.environ["LLMLAB_URL"],
        api_key=os.environ["LLMLAB_API_KEY"],
    )
    model_name = config["embedding"]["model_name"]
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

    # La nomenclature est déjà tronquée + prunée (codes canoniques) par l'étape
    # `prune` unifiée → on lit directement son artefact, sans re-traitement ici.
    logger.info(f"  Chargement nomenclature prunée : {config['coicop']['path_prunned_lvl4']}")
    notices_df = con.sql(
        f"SELECT * FROM read_parquet('{config['coicop']['path_prunned_lvl4']}')"
    ).to_df()

    columns_to_keep = [
        col for col in notices_df.columns
        if "column" not in col.lower() and not col.endswith("_en")
    ]
    notices_df = notices_df[columns_to_keep]
    logger.info(f"  → {len(notices_df)} notices prunées chargées depuis {config['coicop']['path_prunned_lvl4']}")

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

    # `collection_base`, pas `collection_name` : la clé est délibérément
    # renommée. Ce fichier lisait le nom à deux endroits (ici et dans l'upsert
    # plus bas) ; si la clé avait gardé son nom, oublier le second aurait créé
    # la collection unique tout en déversant les points dans l'ancienne
    # collection partagée, silencieusement. Renommée, l'oubli lève un KeyError.
    collection_name = args.collection_name or build_collection_name(
        base=config["qdrant"]["collection_base"],
        run_date=args.run_date,
        run_id=args.run_id,
    )
    logger.info(f"  Collection cible : {collection_name}")

    # Les noms étant uniques, ce cas ne se produit plus qu'en cas de `argo retry`
    # sur le même run_id — où recréer est le comportement voulu.
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

    # Un échec d'embedding est FATAL, et ce n'était pas le cas avant.
    # L'ancien `continue` laissait `embeddings` plus court que `documents`, et
    # le `zip` plus bas réappariait alors chaque vecteur au mauvais payload à
    # partir du premier échec — un index silencieusement faux. Tant que
    # l'indexation vivait dans le pipeline, une collection corrompue écrasait
    # la collection partagée et finissait par se remarquer ; désormais elle
    # reçoit un nom neuf qu'on recopie comme index béni.
    embeddings = []
    for i, document in enumerate(documents):
        try:
            response = client_llmlab.embeddings.create(
                model=model_name,
                input=document["text"]
            )
            embeddings.append(response.data[0].embedding)
            if (i + 1) % 10 == 0:
                logger.info(f"  → {i + 1}/{len(documents)} embeddings generated")
        except Exception as e:
            raise RuntimeError(
                f"Échec d'embedding sur le document {i} (code {document['metadata']['code']}) : {e}. "
                "Indexation interrompue : poursuivre produirait une collection "
                "aux payloads décalés."
            ) from e

    if len(embeddings) != len(documents):
        raise RuntimeError(
            f"{len(embeddings)} embeddings pour {len(documents)} documents — "
            "appariement impossible."
        )
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
        # Fatal, pour la même raison que les embeddings : un lot manquant
        # donnait une collection incomplète que rien ne signalait.
        client_qdrant.upsert(collection_name=collection_name, points=batch)
        logger.info(f"  → Batch {batch_num}/{n_batches} uploaded")

    # -----------------------------------------------------------------------
    # Manifest
    # -----------------------------------------------------------------------

    logger.info("=" * 80)
    logger.info("STEP 7: WRITING MANIFEST")
    logger.info("=" * 80)

    # Compté auprès de Qdrant après upload, jamais `len(points)` : c'est la
    # seule mesure qui atteste que les points sont bien arrivés.
    point_count = client_qdrant.count(collection_name=collection_name, exact=True).count

    source_csv = config["coicop"]["path_prunned_lvl4"]
    manifest = {
        "collection_name": collection_name,
        "kind": "notices",
        "mode": None,  # les notices dérivent d'un CSV statique : pas de mode
        "run_id": args.run_id,
        "run_date": args.run_date,
        "embedding_model": model_name,
        "embedding_dim": config["embedding"]["model_len"],
        "strategy": strategy,
        "sample_size": None,
        "point_count": point_count,
        "source_nomenclature": source_csv,
        "git_sha": git_sha(),
    }
    uri = write_manifest(con, config["qdrant"]["manifests_root"], manifest)
    logger.info(f"✓ Manifeste écrit : {uri}")
    logger.info(dumps(manifest))

    logger.info("=" * 80)
    logger.info("VECTOR DATABASE CREATION PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)

    log_paste_banner(logger, {"classify-rag-notices-collection": collection_name})


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    main()
