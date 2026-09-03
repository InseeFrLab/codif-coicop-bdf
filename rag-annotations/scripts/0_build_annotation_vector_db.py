"""
Annotation Vector Database Creation
===================================
Lit la KB **déjà prunée** (`prune-codes/annotations_train_pruned.parquet`),
l'embarque intégralement et la charge dans une collection Qdrant.

Ce qu'on indexe, ce sont les **produits déjà annotés** : `annotations_full` +
`suggester` au sens de build-datasets. Rien à voir avec les produits *à classer*
— d'où l'absence de toute notion de fichier d'entrée ici.

`--kb-scope` :
  full  (défaut) — tous les produits annotés ;
  train          — l'ancien split, conservé pour la transition et destiné à
                   disparaître. Il n'existait que faute de jeu de test
                   indépendant ; les nouveaux produits annotés en fournissent un.

Le **suggester** (`prune-codes/suggester_pruned.parquet`) est ajouté à l'index
comme candidats de retrieval supplémentaires. La KB peut être plafonnée par
`--sample-size` (essais rapides).

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

from rag_annotations.utils import (
    build_location_text,
    create_duckdb_connection,
    embed_texts,
    expand_paths,
)
from codif_common.vector_index import (
    build_collection_name,
    dumps,
    git_sha,
    log_paste_banner,
    write_manifest,
)


def main():
    parser = argparse.ArgumentParser(description="Annotation vector database creation")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config YAML")
    parser.add_argument("--run-id", required=True, help="Workflow run identifier")
    parser.add_argument("--run-date", required=True, help="Workflow run date (YYYY-MM-DD)")
    parser.add_argument(
        "--kb-scope", choices=["full", "train"], default="full",
        help="Quels produits annotés composent la KB. 'full' (défaut) : tous "
             "(annotations_full). 'train' : l'ancien split, conservé pour la "
             "transition et destiné à disparaître — il n'existait que faute de jeu "
             "de test indépendant. Sélectionne aussi le profil de sources exclues "
             "(annotations.exclude_sources_prod vs exclude_sources).",
    )
    parser.add_argument(
        "--sample-size", type=int, default=None,
        help="Limite le nombre d'annotations indexées dans la vector DB (KB). "
             "Vide = toutes. Échantillonnage sans remise, graine eval.seed.",
    )
    parser.add_argument(
        "--collection-name",
        default=None,
        help=(
            "Nom complet de la collection à créer. Par défaut, composé depuis "
            "`qdrant.collection_base`, le mode et l'identité de CE run "
            "d'indexation. À ne renseigner que pour rejouer un nom précis."
        ),
    )
    args = parser.parse_args()

    # `train` (l'ancien split) applique le profil de sources restreint,
    # `full` le profil large — les deux listes sont identiques aujourd'hui,
    # la distinction est conservée le temps de la transition.
    is_train_split = args.kb_scope == "train"

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
    logger.info(f"STEP 1: loading annotations from {config['annotations']['s3_path']}")
    annotations = con.sql(
        f"SELECT * FROM read_parquet('{config['annotations']['s3_path']}')"
    ).to_df()
    annotations = annotations[config["annotations"]["features"]]

    nature = config["annotations"]["nature"]
    if nature:
        annotations = annotations.loc[annotations["source"] == nature]
    logger.info(f"  → {len(annotations)} annotations loaded (nature={nature or 'all'})")

    # Exclude configured sources from the vector DB.
    # train : exclude_sources (profil restreint de l'ancien split).
    # full  : exclude_sources_prod (profil large, toutes les annotations voulues).
    if is_train_split:
        exclude_sources = config["annotations"].get("exclude_sources") or []
    else:
        exclude_sources = config["annotations"].get("exclude_sources_prod") or []
    logger.info(f"  → kb_scope={args.kb_scope}, exclude_sources={exclude_sources}")
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

    # -----------------------------------------------------------------------
    # STEP 3: add suggester examples to the index (déjà prunés par l'étape `prune`)
    # -----------------------------------------------------------------------
    # No train/test split here: preprocessing already split train (this input)
    # from the evaluation test set. ALL loaded annotations are indexed.
    kb_data = annotations
    suggester_excluded = "suggester" in exclude_sources
    if config.get("suggester", {}).get("enabled") and not suggester_excluded:
        logger.info(f"STEP 3: adding suggester examples from {config['suggester']['s3_path_pruned']}")
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

    # Échantillonnage optionnel de la KB (annotations + suggester confondus), sans
    # remise et avec graine reproductible. Appliqué APRÈS la fusion pour que le total
    # de points indexés == sample_size : on génère ainsi une petite vector DB rapide
    # à construire pour tester le code (et non sample_size + tout le suggester).
    if args.sample_size and args.sample_size < len(kb_data):
        seed = config.get("eval", {}).get("seed", 42)
        kb_data = kb_data.sample(n=args.sample_size, random_state=seed).reset_index(drop=True)
        logger.info(f"  → KB échantillonnée à {len(kb_data)} points (annotations + suggester, seed={seed})")

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
    # Le mode est dans le NOM, pas seulement dans le manifeste : une KB de
    # production contient *toutes* les annotations, donc l'interroger depuis un
    # run d'évaluation gonfle l'accuracy sans lever d'erreur. C'est le champ
    # qu'un humain confond le plus en recopiant, autant qu'il soit visible.
    # L'ancien suffixe `_test`, accordé par convention entre ce script et
    # 1_run_rag.py, disparaît : le nom unique le rend inutile, et
    # `sample_size` reste visible via le suffixe `__sampleN`.
    collection_name = args.collection_name or build_collection_name(
        base=config["qdrant"]["collection_base"],
        run_date=args.run_date,
        run_id=args.run_id,
        mode=args.kb_scope,
        sample_size=args.sample_size,
    )
    logger.info(f"  → collection cible : {collection_name}")
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

    # -----------------------------------------------------------------------
    # STEP 6: manifest
    # -----------------------------------------------------------------------
    logger.info("STEP 6: writing manifest")

    # Compté auprès de Qdrant, jamais `len(points)` : seule mesure qui atteste
    # que les points sont bien arrivés.
    point_count = client_qdrant.count(collection_name=collection_name, exact=True).count

    manifest = {
        "collection_name": collection_name,
        "kind": "annotations",
        "kb_scope": args.kb_scope,
        "run_id": args.run_id,
        "run_date": args.run_date,
        "embedding_model": config["embedding"]["model_name"],
        "embedding_dim": config["embedding"]["model_len"],
        "strategy": None,  # pas de découpage : une annotation = un point
        "sample_size": args.sample_size,
        "point_count": point_count,
        "source_annotations": config["annotations"]["s3_path"],
        "git_sha": git_sha(),
    }
    uri = write_manifest(con, config["qdrant"]["manifests_root"], manifest)
    logger.info(f"  ✓ manifeste écrit : {uri}")
    logger.info(dumps(manifest))

    logger.info("=" * 80)
    logger.info("ANNOTATION VECTOR DB CREATION COMPLETED")
    logger.info(f"  collection: {collection_name} ({point_count} points)")
    logger.info("=" * 80)

    log_paste_banner(logger, {"classify-rag-annotations-collection": collection_name})


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    main()
