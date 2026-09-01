# %%

import argparse
import uuid

import pandas as pd
from sklearn.model_selection import train_test_split

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
)
from src.data.string_cleaning import preprocess_text, normalize_text
from src.data.check_data import duplicated_suggester, get_product_with_multiple_codes
from src.stats.calculate_features import aggregate_budget, statistics_annotations_data
from src.data.save_data import export_parquet_s3


# Colonnes de regroupement pour l'agrégation budget (dédoublonnage + somme).
BUDGET_GROUP_COLUMNS = [
    "id", "raw_product", "l_pr_product", "s_pr_product",
    "source", "annee", "code", "coicop", "shop", "shop_type_name",
]

# Sources considérées comme données BdF 2024 (le reste — dont 2017 et suggester —
# part dans le lot « historique » et n'est donc pas inclus dans le split test).
# NB : "copain" est conservé ici à titre de trace (source exclue du pipeline depuis
# 2026-06, cf. load_data.py) ; n'étant plus chargée, elle matche 0 ligne.
SOURCES_2024 = ["copain", "receipts_from_app", "manual_from_app", "manual_from_book"]

# Recodage des libellés de méthode hérités vers les noms canoniques.
SOURCE_METHOD_MAPPING = {
    "ticket_application": "receipts_from_app",
    "ajout_manuel_application": "manual_from_app",
    "depense_manuelle_carnet": "manual_from_book",
}


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


# ---------------------------------------------------------------------------
# Briques de standardisation des libellés, partagées par les deux pipelines
# (annotations et observations à prédire).
# ---------------------------------------------------------------------------
def merge_shop_types(df: pd.DataFrame, shops_mapping: pd.DataFrame) -> pd.DataFrame:
    """Normalise l'enseigne et lui rattache son type de magasin.

    `shops_mapping` est déjà normalisé et unique sur `shop` (cf. load_shops_mapping),
    ce qui garantit un merge `many_to_one` sans multiplication de lignes. Sans colonne
    `shop` (input de prédiction sans enseigne), le DataFrame est renvoyé tel quel.
    """
    if "shop" in df.columns:
        df["shop"] = normalize_text(df["shop"])
        df = df.merge(
            shops_mapping[["shop", "shop_type_code", "shop_type_name"]],
            on="shop",
            how="left",
            validate="many_to_one",
        )
    return df


def normalize_products(
    df: pd.DataFrame, uncodable_products, stopwords, logger
) -> pd.DataFrame:
    """Normalise les libellés produits, retire les non codables, attribue un UUID.

    Deux variantes de libellé sont produites :
    - `l_pr_product` : normalisation *légère* (`normalize_text`) — utilisée par les modèles ;
    - `s_pr_product` : normalisation *forte* (`preprocess_text`) — utilisée par les regex.
    """
    df["l_pr_product"] = normalize_text(df["raw_product"])
    df["s_pr_product"] = df["l_pr_product"].copy()
    df = preprocess_text(df, "s_pr_product", stopwords)

    # Les produits non codables sont comparés sur `raw_product` après normalisation
    # de la liste de référence (cohérent entre annotations et observations).
    uncodable = normalize_text(pd.Series(uncodable_products)).tolist()
    before = len(df)
    df = df.loc[~df["raw_product"].isin(uncodable)]
    logger.info(
        f"Produits non codables retirés : {before - len(df)} "
        f"({before} → {len(df)} lignes)"
    )

    df["id"] = [str(uuid.uuid4()) for _ in range(len(df))]
    return df


# ---------------------------------------------------------------------------
# Pipeline ANNOTATIONS — exécuté dans tous les cas.
# Construit le dataset annoté consolidé (2024 + 2017 + suggester), le split
# train/test, et le jeu complet.
# ---------------------------------------------------------------------------
def build_annotations(
    config,
    annotations_hors_copain,
    suggester,
    annotation_old,
    shops_mapping,
    uncodable_products,
    stopwords,
    output_root,
    logger,
):
    """Renvoie (annotations_full, raw_train, raw_test) et exporte les contrôles QA."""

    # -- Recodage de la colonne source (méthodes héritées) ------------------
    annotations_hors_copain["source"] = annotations_hors_copain["source"].replace(
        SOURCE_METHOD_MAPPING
    )

    # -- Consolidation des annotations 2024 + suggester ---------------------
    # copain retiré de la consolidation (source exclue du pipeline — cf. load_data.py).
    annotations = pd.concat(
        [annotations_hors_copain, suggester], axis=0
    )
    logger.info("Fin du chargement et de la création du fichier d'annotation consolidé")

    # -- Standardisation ----------------------------------------------------
    logger.info("[2/4] Début de la phase de standardisation des libellés")
    # Merge des types d'enseignes AVANT d'ajouter les annotations historiques
    # 2017 (qui portent déjà shop_type_code / shop_type_name).
    annotations = merge_shop_types(annotations, shops_mapping)
    annotations = pd.concat([annotations, annotation_old], axis=0, join="outer")
    annotations = normalize_products(annotations, uncodable_products, stopwords, logger)
    logger.info(f"Number of annotations with old data (bdF 2017): {len(annotations)}")

    n_shop_types = annotations["shop_type_name"].isna().sum()
    logger.info(
        f"Proportion of shop types retrieved: {round(n_shop_types / len(annotations), 2)}%"
    )

    # -- Contrôles qualité --------------------------------------------------
    logger.info("[3/4] Début de la phase de contrôle des données annotées")
    duplicated_suggester(suggester, logger)

    annotations_hors_copain_std = annotations.loc[
        annotations["source"].isin(
            ["receipts_from_app", "manual_from_book", "manual_from_app"]
        )
    ]
    annotations_hors_copain_with_multiple_codes = get_product_with_multiple_codes(
        annotations_hors_copain_std, config, "raw_product", "code"
    )
    annotations_with_multiple_codes = get_product_with_multiple_codes(
        annotations, config, "raw_product", "code"
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

    # -- Agrégation budget + split train/test -------------------------------
    logger.info("[4/4] Agrégation du budget et split train/test")

    # Dédoublonnage et somme du budget : données 2024 d'un côté, historique de l'autre.
    annotations_with_budget_test = aggregate_budget(
        annotations=annotations[annotations["source"].isin(SOURCES_2024)],
        group_columns=BUDGET_GROUP_COLUMNS,
        method_column="source",
        budget_column="budget",
    )
    logger.info(
        f"Nombre de lignes total du fichier BdF 2024: {len(annotations_with_budget_test)}"
    )
    annotations_with_budget_old = aggregate_budget(
        annotations=annotations[~annotations["source"].isin(SOURCES_2024)],
        group_columns=BUDGET_GROUP_COLUMNS,
        method_column="source",
        budget_column="budget",
    )
    logger.info(
        f"Nombre de lignes total du fichier BdF 2017: {len(annotations_with_budget_old)}"
    )

    # Split sur les seules données 2024 ; l'historique 2017 (+ suggester) rejoint le train.
    train_anno_wb, test_anno_wb = train_test_split(
        annotations_with_budget_test, test_size=0.5, random_state=42
    )
    train_anno_wb = pd.concat([train_anno_wb, annotations_with_budget_old], axis=0)

    # Jeu complet = train ∪ test (2024 agrégé + historique agrégé).
    annotations_full = pd.concat(
        [annotations_with_budget_test, annotations_with_budget_old], axis=0
    )

    logger.info(f"Nombre de lignes total du fichier complet: {len(annotations_full)}")
    logger.info(f"Nombre de lignes total du fichier train: {len(train_anno_wb)}")
    logger.info(f"Nombre de lignes total du fichier test: {len(test_anno_wb)}")

    return annotations_full, train_anno_wb, test_anno_wb


# ---------------------------------------------------------------------------
# Pipeline SUGGESTER — exécuté dans tous les cas.
# Fichier dédié (lu directement par classify-lcs), à partir de la liste produits.
# ---------------------------------------------------------------------------
def build_suggester(suggester, stopwords, logger):
    """Préprocesse le suggester (déjà tagué source='suggester', annee=2017)."""
    suggester = suggester.copy()
    suggester["l_pr_product"] = normalize_text(suggester["raw_product"])
    suggester["s_pr_product"] = suggester["l_pr_product"].copy()
    suggester = preprocess_text(suggester, "s_pr_product", stopwords)
    for col in ["shop", "shop_type_name", "budget", "n_obs"]:
        if col not in suggester.columns:
            suggester[col] = pd.NA
    suggester["id"] = [str(uuid.uuid4()) for _ in range(len(suggester))]
    logger.info(f"Suggester préprocessé : {len(suggester)} lignes")
    return suggester


# ---------------------------------------------------------------------------
# Pipeline OBSERVATIONS — exécuté uniquement si --input-file est fourni.
# Prépare les libellés à coder (sans vérité terrain).
# ---------------------------------------------------------------------------
def build_observations(args, con, shops_mapping, uncodable_products, stopwords, logger):
    """Préprocesse le fichier d'observations à coder (mode prédiction)."""
    logger.info(f"Mode prédiction activé. Chargement du fichier : {args.input_file}")
    df = load_input_file(args.input_file, con)
    logger.info(f"{len(df)} lignes chargées depuis le fichier d'entrée")

    if args.text_column not in df.columns:
        raise ValueError(
            f"Column '{args.text_column}' not found in input file. Found: {list(df.columns)}"
        )

    # Renommage des colonnes d'entrée vers les noms canoniques du pipeline.
    column_mapping = {
        args.text_column: "raw_product",
        args.shop_column: "shop",
        args.budget_column: "budget",
        args.annee_column: "annee",
        args.source_column: "source",
    }
    df = df.rename(
        columns={
            src: tgt for src, tgt in column_mapping.items()
            if src in df.columns and src != tgt
        }
    )

    df = merge_shop_types(df, shops_mapping)
    df = normalize_products(df, uncodable_products, stopwords, logger)
    logger.info(f"{len(df)} lignes après prétraitement")

    # Aligne le schéma sur celui des annotations pour que l'aval ne plante pas
    # (pas de vérité terrain en prédiction → code/coicop à NA, source='prediction').
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

    df["_source_input_file"] = args.input_file
    return df


def main():
    # -----------------------------------------------------------------------
    # Configuration
    # -----------------------------------------------------------------------
    args = parse_args()
    logger = setup_logging()

    logger.info("Chargement du fichier de configuration depuis le fichier config.yaml")
    config = load_config()
    logger.info("Fin du chargement du fichier de configuration")

    logger.info("Initialisation duckdb")
    con = init_duckdb(config)
    logger.info("Initialisation duckdb fait")

    stopwords = load_stopwords()
    output_root = concat_path_from_key(config, "paths", "output_root").format(
        run_id=args.run_id, run_date=args.run_date
    )

    # -----------------------------------------------------------------------
    # Chargement unique des sources annotées (annotations + ressources partagées)
    # -----------------------------------------------------------------------
    logger.info("[1/4] Chargement des fichiers annotés issus du S3 du projet")
    # copain retiré de l'unpacking (source exclue du pipeline — cf. load_data.py).
    (
        annotations_hors_copain,
        suggester,
        shops_mapping,
        uncodable_products,
        annotation_old,
    ) = load_data(config, con)

    # COPAIN exclu : plus de chargement ni de log d'import pour cette source.
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

    # -----------------------------------------------------------------------
    # Pipeline ANNOTATIONS (toujours) : fichier complet + train + test
    # -----------------------------------------------------------------------
    annotations_full, raw_train, raw_test = build_annotations(
        config,
        annotations_hors_copain,
        suggester,
        annotation_old,
        shops_mapping,
        uncodable_products,
        stopwords,
        output_root,
        logger,
    )

    logger.info("Écriture sur S3 des fichiers d'annotations (complet / train / test)")
    export_parquet_s3(annotations_full, f"{output_root}/annotations_full.parquet")
    export_parquet_s3(raw_train, f"{output_root}/raw_train.parquet")
    export_parquet_s3(raw_test, f"{output_root}/raw_test.parquet")

    statistics_annotations_data(con, pd.concat([raw_train, raw_test], axis=0))

    # -----------------------------------------------------------------------
    # Pipeline SUGGESTER (toujours)
    # -----------------------------------------------------------------------
    suggester_df = build_suggester(suggester, stopwords, logger)
    export_parquet_s3(suggester_df, f"{output_root}/suggester.parquet")

    # -----------------------------------------------------------------------
    # Pipeline OBSERVATIONS (si un fichier d'entrée est fourni)
    # -----------------------------------------------------------------------
    if args.input_file:
        observations = build_observations(
            args, con, shops_mapping, uncodable_products, stopwords, logger
        )
        export_parquet_s3(observations, f"{output_root}/observations.parquet")
        logger.info(
            f"Fichier d'observations à coder exporté : {output_root}/observations.parquet "
            f"({len(observations)} lignes)"
        )

    logger.info("Fin du preprocessing.")


if __name__ == "__main__":
    main()
