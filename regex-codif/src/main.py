import argparse
import os
import s3fs
import pandas as pd

from utils.load_config import load_config, expand_paths
from utils.logging import setup_logging
from utils.init_duckdb import init_duckdb
from utils.load_rules import load_regex_rules

from data.save_data import save_data_to_parquet
from data.apply_regex import apply_regex


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True, help="Workflow run identifier")
    parser.add_argument("--run-date", required=True, help="Workflow run date (YYYY-MM-DD)")
    parser.add_argument(
        "--input-file",
        default=None,
        help="Présence => mode production : le flux 'train' (KB) est l'ensemble des "
             "annotations (annotations_full) et le flux 'test' (à coder) les "
             "observations. Absence => mode évaluation (raw_train / raw_test).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Échantillonne le jeu à coder (test sans regex) à N lignes. Point UNIQUE "
             "de sampling des observations : tous les classifieurs aval (lcs, rag, "
             "rag-annotations, ttc) héritent du même jeu. Vide = tout.",
    )
    return parser.parse_args()


def main():
    # CONFIGURATION -------------------------------------

    args = parse_args()
    logger = setup_logging()

    # -----------------------------------------------------------------------
    # Load config file and paths
    # -----------------------------------------------------------------------
    logger.info("Chargement du fichier de configuration depuis le fichier config.yaml")
    config = expand_paths(load_config(), run_id=args.run_id, run_date=args.run_date)
    logger.info("Fin du chargement du fichier de configuration")

    # -----------------------------------------------------------------------
    # Init duckdb
    # -----------------------------------------------------------------------
    logger.info("Initialisation duckdb")
    con = init_duckdb(config)
    logger.info("Initialisation duckdb fait")

    # -----------------------------------------------------------------------
    # READ RULES
    # -----------------------------------------------------------------------
    rules = load_regex_rules("config/rules.yaml")

    # -----------------------------------------------------------------------
    # READ TRAIN AND TEST SET
    #
    # Production (--input-file) : 'train' = annotations_full (alimente la vector DB
    # d'annotations) et 'test' = observations à coder. Évaluation : split train/test
    # des annotations (raw_train / raw_test). Les sorties (raw_*_without_regex) gardent
    # les mêmes noms, donc tout l'aval est inchangé.
    # -----------------------------------------------------------------------
    is_prod = bool(args.input_file)
    kb_key = "train_set_prod" if is_prod else "train_set"
    observations_key = "test_set_prod" if is_prod else "test_set"
    logger.info(f"Mode {'production' if is_prod else 'évaluation'} : "
                f"kb_data={config['paths'][kb_key]}, observations={config['paths'][observations_key]}")

    logger.info("Chargement de la KB (kb_data).")
    query_definition = f"SELECT * FROM read_parquet('{config['paths'][kb_key]}')"
    kb_data = con.sql(query_definition).to_df()

    logger.info("Chargement des observations (jeu à coder).")
    query_definition = f"SELECT * FROM read_parquet('{config['paths'][observations_key]}')"
    observations = con.sql(query_definition).to_df()

    # -----------------------------------------------------------------------
    # CLASSIFICATION WITH REGEX
    # -----------------------------------------------------------------------

    # Apply rules on KB raws
    kb_data_regex_predicted, kb_data_without_regex = apply_regex(kb_data, rules, logger)

    # Apply rules on observations raws
    observations_regex_predicted, observations_without_regex = apply_regex(observations, rules, logger)

    # Échantillonnage CENTRALISÉ du jeu à coder (sans remise, graine 42) : c'est l'unique
    # point de sampling des observations. raw_test_without_regex est lu par lcs/ttc, et
    # par prune (→ annotations_test_pruned, lu par les RAG) → tous héritent du même jeu.
    if args.sample_size and args.sample_size < len(observations_without_regex):
        observations_without_regex = observations_without_regex.sample(
            n=args.sample_size, random_state=42
        )
        logger.info(f"Échantillon observations : {len(observations_without_regex)} libellés (seed=42)")

    # Final dataframe with regex predictions
    regex_predicted = pd.concat([kb_data_regex_predicted, observations_regex_predicted])

    # -----------------------------------------------------------------------
    # METRICS
    # -----------------------------------------------------------------------

    # contingence table
    logger.info("Nombre de libellés prédits selon leur code :")
    logger.info(regex_predicted["predict_code"].value_counts())

    if "code" in regex_predicted.columns:
        # Global accuracy
        accuracy = (regex_predicted["code"] == regex_predicted["predict_code"]).mean()
        logger.info(f"Accuracy : {accuracy:.2%}")
        # Accuracy with not null codes
        non_nuls = regex_predicted[regex_predicted["predict_code"].notna()]
        accuracy = (non_nuls["code"] == non_nuls["predict_code"]).mean()
        logger.info(f"Accuracy (with not null codes): {accuracy:.2%}")

        # Number of raw classified by regex
        logger.info(f"Prédictions par année:")
        logger.info(regex_predicted.groupby("annee")["code"].count())

        # Accuracy per code présents dans le fichier de règles
        rules_codes = [rule["code"] for rule in rules]
        regex_predicted["match"] = regex_predicted["code"] == regex_predicted["predict_code"]
        accuracy_par_code = regex_predicted[regex_predicted["code"].isin(rules_codes)].groupby("code")["match"].mean()
        logger.info("Accuracy par code :")
        logger.info(accuracy_par_code)

        non_nuls["match"] = non_nuls["code"] == non_nuls["predict_code"]
        accuracy_par_code_nn = non_nuls[non_nuls["code"].isin(rules_codes)].groupby("code")["match"].mean()
        logger.info("Accuracy par code (avec codes non nuls) :")
        logger.info(accuracy_par_code_nn)

        error_pred = non_nuls.loc[non_nuls["code"] != non_nuls["predict_code"], ["raw_product", "code", "predict_code"]]
        logger.info(error_pred)

        save_data_to_parquet(df=error_pred, path=config["paths"]["error_pred"])

    # -----------------------------------------------------------------------
    # EXPORT OUTPUT
    # -----------------------------------------------------------------------
    if config["S3"]["export"]:
        logger.info("Export du fichier final de prédiction avec des regex.")
        save_data_to_parquet(df=regex_predicted, path=config["paths"]["output"]["pred"])

        logger.info("Export des observations sans les libellés prédits par REGEX.")
        save_data_to_parquet(df=observations_without_regex, path=config["paths"]["output"]["test_set"])

        logger.info("Export de la KB (kb_data) sans les libellés prédits par REGEX.")
        save_data_to_parquet(df=kb_data_without_regex, path=config["paths"]["output"]["train_set"])


if __name__ == "__main__":
    main()
