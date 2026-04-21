# %%

import os
from datetime import date
import s3fs
import time
import uuid
import pandas as pd
from src.utils.data_management import (
    concat_path_from_key,
    load_stopwords,
)
from src.utils.load_config import load_config
from src.utils.logging import setup_logging
from src.utils.init_duckdb import init_duckdb
from src.data.load_data import load_data
from src.data.string_cleaning import preprocess_text, normalize_text
from src.data.check_data import duplicated_suggester, get_product_with_multiple_codes
from src.stats.calculate_features import aggregate_budget, statistics_annotations_data
from src.data.save_data import export_parquet_s3

from sklearn.model_selection import train_test_split


def main():
    # CONFIGURATION -------------------------------------

    logger = setup_logging()
    fs = s3fs.S3FileSystem(endpoint_url=os.environ["AWS_ENDPOINT_URL"])

    # -----------------------------------------------------------------------
    # Load config file and paths
    # -----------------------------------------------------------------------
    logger.info("Chargement du fichier de configuration depuis le fichier config.yaml")
    config = load_config()
    logger.info("Fin du chargement du fichier de configuration")

    # -----------------------------------------------------------------------
    # Init duckdb
    # -----------------------------------------------------------------------
    logger.info("Initialisation duckdb")
    con = init_duckdb(config)
    logger.info("Initialisation duckdb fait")

    # -----------------------------------------------------------------------
    # READING ALL FILES
    # -----------------------------------------------------------------------
    logger.info("[1/4] Chargement des fichiers annotés issus du S3 du projet")

    (
        annotations_copain,
        annotations_hors_copain,
        suggester,
        shops_mapping,
        uncodable_products,
        annotation_old
    ) = load_data(config, con)

    logger.info(
        f"Données annotées issues de l'application COPAIN importées, {len(annotations_copain)} lignes"
    )
    logger.info(
        f"Données annotées issues de l'annotation historique importées, {len(annotations_hors_copain)} lignes"
    )
    logger.info(f"Données du suggester importées, {len(suggester)} lignes")
    logger.info(
        f"Mapping entre nom d'enseignes et types d'enseignes, {len(shops_mapping)} lignes"
    )
    logger.info(
        f"Number of uncodable products to remove: {len(uncodable_products)}"
    )
    logger.info(
        f"Données annotées issues de la précédente enquête BdF (2017), {len(annotation_old)} lignes"
    )

    # Recode method column
    mapping_method = {
        "ticket_application": "receipts_from_app",
        "ajout_manuel_application": "manual_from_app",
        "depense_manuelle_carnet": "manual_from_book",
    }

    annotations_hors_copain["source"] = annotations_hors_copain["source"].replace(
        mapping_method
    )

    # Consolidate annotations data
    annotations = pd.concat(
        [annotations_copain, annotations_hors_copain, suggester], axis=0
    )

    logger.info("Fin du chargement et de la création du fichier d'annotation consolidé")

    # -----------------------------------------------------------------------
    # STANDARDIZING ANNOTATIONS DATA
    # -----------------------------------------------------------------------
    logger.info("[2/4] Début de la phase de standardisation des libellés")

    # Merge des types de magasins avant préprocessing
    annotations["shop"] = normalize_text(annotations["shop"])
    shops_mapping["shop"] = normalize_text(shops_mapping["shop"])

    # Merge shop names / shop types
    annotations = annotations.merge(
        shops_mapping[["shop", "shop_type_code", "shop_type_name"]],
        on="shop",
        how="left",
    )

    # Add old annotations
    annotations = pd.concat([annotations, annotation_old], axis=0, join='outer')

    # Préprocessing léger des libellés de produits
    annotations["l_pr_product"] = normalize_text(annotations["raw_product"])
    uncodable_products = normalize_text(uncodable_products)

    # Préprocessing fort des libellés de produits
    stopwords = load_stopwords()
    annotations["s_pr_product"] = annotations["l_pr_product"].copy()
    annotations = preprocess_text(annotations, "s_pr_product", stopwords)

    # Remove uncodable products
    logger.info(
        f"Number of annotations before removal of uncodable products: {len(annotations)}"
    )
    annotations = annotations.loc[~annotations["raw_product"].isin(uncodable_products)]
    logger.info(
        f"Number of annotations after removal of uncodable products: {len(annotations)}"
    )

    logger.info(
            f"Number of annotations with old data (bdF 2017): {len(annotations)}"
        )

    n_shop_types = annotations["shop_type_name"].isna().sum()
    logger.info(
        f"Proportion of shop types retrieved: {round(n_shop_types/len(annotations), 2)}%"
    )
    # Add unique id (UUID)
    annotations['id'] = [str(uuid.uuid4()) for _ in range(len(annotations))]

    # -----------------------------------------------------------------------
    # CONTROLING ANNOTATIONS
    # -----------------------------------------------------------------------
    
    logger.info("[3/4] Début de la phase de contrôle des données annotées")

    # Duplicated rows in suggester
    duplicated_suggester(suggester, logger)

    annotations_hors_copain = (
        annotations.loc[
            annotations["source"]
            .isin(["receipts_from_app", "manual_from_book", "manual_from_app"])
        ]
    )
    annotations_hors_copain_with_multiple_codes = get_product_with_multiple_codes(
        annotations_hors_copain, config, "raw_product", "code"
    )

    annotations_with_multiple_codes = get_product_with_multiple_codes(
        annotations, config, "raw_product", "code"
    )

    timestamp = date.today().isoformat()

    base_path = (
    f"{concat_path_from_key(config, 'paths', 'output')}"
    f"/annotations-consolidated-{timestamp}"
    )
    export_parquet_s3(annotations_hors_copain_with_multiple_codes, f"{base_path}/annotations_hors_copain_with_multiple_codes.parquet")
    export_parquet_s3(annotations_with_multiple_codes, f"{base_path}/annotations_with_multiple_codes.parquet")

    logger.info("Fin des contrôles sur les données annotées")

    # -----------------------------------------------------------------------
    # WRITE ANNOTATIONS PREPROCESSED
    # -----------------------------------------------------------------------

    logger.info("[4/4] Ecriture sur S3 du fichier")

    # Prepare dedup and sum budget (just 2024 test data)
    annotations_with_budget_test = aggregate_budget(
        annotations=annotations[annotations["source"].isin(["copain", "receipts_from_app", "manual_from_app", "manual_from_book"])],
        group_columns=["id", "raw_product", "l_pr_product", "s_pr_product", "source", "annee", "code", "coicop", "shop", "shop_type_name"],
        method_column="source",
        budget_column="budget",
    )
    logger.info(f"Nombre de lignes total du fichier BdF 2024: {len(annotations_with_budget_test)}")

    # Prepare dedup and sum budget (just 2017 BdF data)
    annotations_with_budget_old = aggregate_budget(
        annotations=annotations[~annotations["source"].isin(["copain", "receipts_from_app", "manual_from_app", "manual_from_book"])],
        group_columns=["id", "raw_product", "l_pr_product", "s_pr_product", "source", "annee", "code", "coicop", "shop", "shop_type_name"],
        method_column="source",
        budget_column="budget",
    )
    logger.info(f"Nombre de lignes total du fichier BdF 2017: {len(annotations_with_budget_old)}")

    # Splitr train / test set
    train_anno_wb, test_anno_wb = train_test_split(annotations_with_budget_test, test_size=0.5, random_state=42)
    train_anno_wb = pd.concat([train_anno_wb, annotations_with_budget_old], axis=0)

    logger.info(f"Nombre de lignes total du fichier train: {len(train_anno_wb)}")
    logger.info(f"Nombre de lignes total du fichier test: {len(test_anno_wb)}")
    
    # Export
    base_path = (
        f"{concat_path_from_key(config, 'paths', 'output')}"
        f"/annotations-consolidated-{timestamp}"
    )
    
    export_parquet_s3(train_anno_wb,  f"{base_path}/raw_train.parquet")
    export_parquet_s3(test_anno_wb,   f"{base_path}/raw_test.parquet")

    # COMPTER CEUX AVEC/SANS BUDGET (hors données suggester) ================
    statistics_annotations_data(con, pd.concat([train_anno_wb, test_anno_wb], axis=0))

    logger.info("Fin de la construction du dataset.")


if __name__ == "__main__":
    main()
