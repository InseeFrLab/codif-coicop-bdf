# %%

import argparse
import os
import s3fs
import time
import uuid
import pandas as pd
import pyarrow as pa
from src.utils.data_management import (
    concat_path_from_key,
    load_stopwords,
)
from src.utils.load_config import load_config
from src.utils.logging import setup_logging
from src.utils.init_duckdb import init_duckdb
from src.data.load_data import (
    load_data,
    load_input_file,
    load_shops_mapping,
    load_path_data,
)
from src.data.string_cleaning import preprocess_text, normalize_text
from src.data.check_data import duplicated_suggester, get_product_with_multiple_codes
from src.stats.calculate_features import aggregate_budget, statistics_annotations_data
from src.data.save_data import export_parquet_s3

from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True, help="Workflow run identifier")
    parser.add_argument(
        "--run-date", required=True, help="Workflow run date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--input-file",
        default=None,
        help="Path to input file for prediction (local or S3). Activates prediction mode.",
    )
    parser.add_argument(
        "--text-column",
        default="raw_product",
        help="Column in the input file containing the text to classify (default: raw_product).",
    )
    parser.add_argument(
        "--shop-column",
        default="shop",
        help="Column in the input file for the shop name (default: shop).",
    )
    parser.add_argument(
        "--budget-column",
        default="budget",
        help="Column in the input file for the budget amount (default: budget).",
    )
    parser.add_argument(
        "--annee-column",
        default="annee",
        help="Column in the input file for the year (default: annee).",
    )
    parser.add_argument(
        "--source-column",
        default="source",
        help="Column in the input file for the data source tag (default: source).",
    )
    return parser.parse_args()


def main():
    # CONFIGURATION -------------------------------------

    args = parse_args()
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
    # PREDICTION MODE
    # -----------------------------------------------------------------------
    if args.input_file:
        logger.info(
            f"Mode prédiction activé. Chargement du fichier : {args.input_file}"
        )
        (_, _, _, S3_APP_ANNOTATIONS, S3_SHOPS_MAPPING, S3_UNCODABLE_PRODUCTS) = load_path_data(config)

        df = load_input_file(args.input_file, con)
        logger.info(f"{len(df)} lignes chargées depuis le fichier d'entrée")

        if args.text_column not in df.columns:
            raise ValueError(
                f"Column '{args.text_column}' not found in input file. Found: {list(df.columns)}"
            )

        # Rename input columns to canonical pipeline names
        column_mapping = {
            args.text_column: "raw_product",
            args.shop_column: "shop",
            args.budget_column: "budget",
            args.annee_column: "annee",
            args.source_column: "source",
        }
        df = df.rename(columns={src: tgt for src, tgt in column_mapping.items()
                                 if src in df.columns and src != tgt})

        shops_mapping = load_shops_mapping(S3_SHOPS_MAPPING, con)

        uncodable_products = (
            con.sql(f"""
            SELECT DISTINCT produit AS raw_product
            FROM read_csv_auto('{S3_UNCODABLE_PRODUCTS}')
        """)
            .to_df()["raw_product"]
            .tolist()
        )

        if "shop" in df.columns:
            df["shop"] = normalize_text(df["shop"])
            shops_mapping["shop"] = normalize_text(shops_mapping["shop"])
            df = df.merge(
                shops_mapping[["shop", "shop_type_code", "shop_type_name"]],
                on="shop",
                how="left",
            )

        df["l_pr_product"] = normalize_text(df["raw_product"])
        uncodable_products = normalize_text(pd.Series(uncodable_products)).tolist()
        stopwords = load_stopwords()
        df["s_pr_product"] = df["l_pr_product"].copy()
        df = preprocess_text(df, "s_pr_product", stopwords)

        df = df.loc[~df["raw_product"].isin(uncodable_products)]
        df["id"] = [str(uuid.uuid4()) for _ in range(len(df))]
        logger.info(f"{len(df)} lignes après prétraitement")

        # Ensure schema matches training output so downstream steps don't crash
        prediction_defaults = {
            "annee": pd.NA,
            "code": pd.NA,
            "coicop": pd.NA,
            "shop": pd.NA,
            "shop_type_name": pd.NA,
            "budget": pd.NA,
            "n_obs": 1,
            "source": "prediction",
        }
        for col, default in prediction_defaults.items():
            if col not in df.columns:
                df[col] = default

        output_root = concat_path_from_key(config, "paths", "output_root").format(
            run_id=args.run_id, run_date=args.run_date
        )
        df["_source_input_file"] = args.input_file
        export_parquet_s3(df, f"{output_root}/raw_test.parquet")

        # raw_train.parquet vide : regex-codif ne doit pas traiter le suggester
        train_schema = pa.Schema.from_pandas(df, preserve_index=False)
        export_parquet_s3(df.iloc[0:0], f"{output_root}/raw_train.parquet", schema=train_schema)

        # Suggester dans un fichier dédié, lu directement par codif-lcs (bypass regex-codif)
        suggester = con.sql(f"""
            SELECT DISTINCT code, product AS raw_product, coicop
            FROM read_csv_auto('{S3_APP_ANNOTATIONS}')
        """).to_df()
        suggester["source"] = "suggester"
        suggester["annee"] = 2017
        suggester["l_pr_product"] = normalize_text(suggester["raw_product"])
        suggester["s_pr_product"] = suggester["l_pr_product"].copy()
        suggester = preprocess_text(suggester, "s_pr_product", stopwords)
        for col in ["shop", "shop_type_name", "budget", "n_obs"]:
            if col not in suggester.columns:
                suggester[col] = pd.NA
        suggester["id"] = [str(uuid.uuid4()) for _ in range(len(suggester))]
        export_parquet_s3(suggester, f"{output_root}/suggester.parquet")
        logger.info(
            f"Fichier de prédiction exporté : {output_root}/raw_test.parquet "
            f"({len(suggester)} lignes suggester dans suggester.parquet)"
        )
        return

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
        annotation_old,
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
    logger.info(f"Number of uncodable products to remove: {len(uncodable_products)}")
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
    annotations = pd.concat([annotations, annotation_old], axis=0, join="outer")

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

    logger.info(f"Number of annotations with old data (bdF 2017): {len(annotations)}")

    n_shop_types = annotations["shop_type_name"].isna().sum()
    logger.info(
        f"Proportion of shop types retrieved: {round(n_shop_types / len(annotations), 2)}%"
    )
    # Add unique id (UUID)
    annotations["id"] = [str(uuid.uuid4()) for _ in range(len(annotations))]

    # -----------------------------------------------------------------------
    # CONTROLING ANNOTATIONS
    # -----------------------------------------------------------------------

    logger.info("[3/4] Début de la phase de contrôle des données annotées")

    # Duplicated rows in suggester
    duplicated_suggester(suggester, logger)

    annotations_hors_copain = annotations.loc[
        annotations["source"].isin(
            ["receipts_from_app", "manual_from_book", "manual_from_app"]
        )
    ]
    annotations_hors_copain_with_multiple_codes = get_product_with_multiple_codes(
        annotations_hors_copain, config, "raw_product", "code"
    )

    annotations_with_multiple_codes = get_product_with_multiple_codes(
        annotations, config, "raw_product", "code"
    )

    output_root = concat_path_from_key(config, "paths", "output_root").format(
        run_id=args.run_id, run_date=args.run_date
    )
    qa_path = f"{output_root}/qa"
    export_parquet_s3(
        annotations_hors_copain_with_multiple_codes,
        f"{qa_path}/annotations_hors_copain_with_multiple_codes.parquet",
    )
    export_parquet_s3(
        annotations_with_multiple_codes,
        f"{qa_path}/annotations_with_multiple_codes.parquet",
    )

    logger.info("Fin des contrôles sur les données annotées")

    # -----------------------------------------------------------------------
    # WRITE ANNOTATIONS PREPROCESSED
    # -----------------------------------------------------------------------

    logger.info("[4/4] Ecriture sur S3 du fichier")

    # Prepare dedup and sum budget (just 2024 test data)
    annotations_with_budget_test = aggregate_budget(
        annotations=annotations[
            annotations["source"].isin(
                ["copain", "receipts_from_app", "manual_from_app", "manual_from_book"]
            )
        ],
        group_columns=[
            "id",
            "raw_product",
            "l_pr_product",
            "s_pr_product",
            "source",
            "annee",
            "code",
            "coicop",
            "shop",
            "shop_type_name",
        ],
        method_column="source",
        budget_column="budget",
    )
    logger.info(
        f"Nombre de lignes total du fichier BdF 2024: {len(annotations_with_budget_test)}"
    )

    # Prepare dedup and sum budget (just 2017 BdF data)
    annotations_with_budget_old = aggregate_budget(
        annotations=annotations[
            ~annotations["source"].isin(
                ["copain", "receipts_from_app", "manual_from_app", "manual_from_book"]
            )
        ],
        group_columns=[
            "id",
            "raw_product",
            "l_pr_product",
            "s_pr_product",
            "source",
            "annee",
            "code",
            "coicop",
            "shop",
            "shop_type_name",
        ],
        method_column="source",
        budget_column="budget",
    )
    logger.info(
        f"Nombre de lignes total du fichier BdF 2017: {len(annotations_with_budget_old)}"
    )

    # Splitr train / test set
    train_anno_wb, test_anno_wb = train_test_split(
        annotations_with_budget_test, test_size=0.5, random_state=42
    )
    train_anno_wb = pd.concat([train_anno_wb, annotations_with_budget_old], axis=0)

    logger.info(f"Nombre de lignes total du fichier train: {len(train_anno_wb)}")
    logger.info(f"Nombre de lignes total du fichier test: {len(test_anno_wb)}")

    # Export
    export_parquet_s3(train_anno_wb, f"{output_root}/raw_train.parquet")
    export_parquet_s3(test_anno_wb, f"{output_root}/raw_test.parquet")

    # COMPTER CEUX AVEC/SANS BUDGET (hors données suggester) ================
    statistics_annotations_data(con, pd.concat([train_anno_wb, test_anno_wb], axis=0))

    logger.info("Fin de la construction du dataset.")


if __name__ == "__main__":
    main()
