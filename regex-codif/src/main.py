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
    # -----------------------------------------------------------------------
    logger.info("Chargement du train set.")
    query_definition = f"SELECT * FROM read_parquet('{config["paths"]["train_set"]}')"
    train_set = con.sql(query_definition).to_df()

    logger.info("Chargement du test set.")
    query_definition = f"SELECT * FROM read_parquet('{config["paths"]["test_set"]}')"
    test_set = con.sql(query_definition).to_df()

    # -----------------------------------------------------------------------
    # CLASSIFICATION WITH REGEX
    # -----------------------------------------------------------------------

    # Apply rules on train set raws
    train_set_regex_predicted, train_set_without_regex = apply_regex(train_set, rules, logger)

    # Apply rules on test set raws
    test_set_regex_predicted, test_set_without_regex = apply_regex(test_set, rules, logger)

    # Final dataframe with regex predictions
    regex_predicted = pd.concat([train_set_regex_predicted, test_set_regex_predicted])

    # -----------------------------------------------------------------------
    # METRICS
    # -----------------------------------------------------------------------

    # contingence table
    logger.info("Nombre de libellés prédits selon leur code :")
    logger.info(regex_predicted["predict_code"].value_counts())
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

        logger.info("Export du test set sans les libellés prédits par REGEX.")
        save_data_to_parquet(df=test_set_without_regex, path=config["paths"]["output"]["test_set"])

        logger.info("Export du train set sans les libellés prédits par REGEX.")
        save_data_to_parquet(df=train_set_without_regex, path=config["paths"]["output"]["train_set"])


if __name__ == "__main__":
    main()
