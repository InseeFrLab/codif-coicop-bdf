import argparse
import os
import s3fs

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
        help="Conservé pour compatibilité : la KB est toujours annotations_full et le "
             "jeu à coder toujours observations. Le pipeline n'a plus qu'un mode.",
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
    # READ KB AND OBSERVATIONS
    #
    # 'train' = annotations_full (ce qui alimente la vector DB d'annotations),
    # 'test' = les observations à coder. Les clés de config gardent leur suffixe
    # `_prod`, hérité de l'époque où le pipeline avait deux modes.
    # -----------------------------------------------------------------------
    kb_key, observations_key = "train_set_prod", "test_set_prod"
    logger.info(f"KB={config['paths'][kb_key]}, observations={config['paths'][observations_key]}")

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
    # La moitié gauche n'est plus exportée (cf. `regex_predicted` plus bas) ; la
    # droite l'est, la KB élaguée sert à l'indexation en aval.
    _kb_regex_hits, kb_data_without_regex = apply_regex(kb_data, rules, logger)

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

    # Prédictions regex exportées : les observations à coder, et elles seules.
    #
    # Ce fichier concaténait aussi `kb_data_regex_predicted`, les correspondances
    # trouvées sur la base de connaissance annotée — ~16 000 lignes historiques
    # qui n'ont rien à faire dans la sortie d'un run de codification. C'est le
    # mélange que signale le commentaire des métriques plus bas : elles se
    # calculaient dessus et « ne mesuraient rien d'interprétable ». Les métriques
    # sont parties dans `evaluate`, l'artefact mélangé était resté.
    #
    # La correction est sans effet de bord : les identifiants des deux
    # populations sont des UUID tirés séparément par `build-datasets`, donc les
    # lignes de KB ne se rattachaient jamais à un produit du livrable — elles ne
    # faisaient que gonfler le fichier, la ligne de log d'`export-results`, et le
    # dénominateur du tableau de bout en bout du rapport d'évaluation.
    #
    # `kb_data_without_regex`, lui, reste exporté : la KB en a besoin en aval.
    regex_predicted = observations_regex_predicted

    # -----------------------------------------------------------------------
    # METRICS
    # -----------------------------------------------------------------------

    # contingence table
    logger.info(
        f"Observations captées par la regex : {len(regex_predicted)} "
        f"sur {len(observations)}"
    )
    logger.info("Nombre de libellés prédits selon leur code :")
    logger.info(regex_predicted["predict_code"].value_counts())

    # Les métriques d'accuracy qui vivaient ici sont parties dans l'étape finale
    # `evaluate`. Elles se calculaient sur un mélange de la KB annotée et des
    # observations (le `concat` ci-dessus), ce qui ne mesurait rien d'interprétable.

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
