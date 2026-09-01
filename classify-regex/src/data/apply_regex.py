import pandas as pd

from utils.load_rules import apply_regex_rules


def apply_regex(df: pd.DataFrame, rules: dict, logger):

    logger.info("Calcul du code prédit à l'aide des regex.")
    df["predict_code"] = apply_regex_rules(df["s_pr_product"], rules)

    # split test_set to test_set without regex classification raws and test_set with regex predicted raws
    logger.info("Split du testset pour avoir un fichier d'input pour les modèles et un autre pour la prédiction finale.")
    df_regex_predicted = df[df["predict_code"].notna()]
    df_regex_predicted["method"] = "REGEX"
    logger.info(f"Nombre de libellés prédits par REGEX : {len(df_regex_predicted)}")

    df_without_regex = df[df["predict_code"].isna()].drop(columns="predict_code")
    logger.info(f"Nombre de libellés à prédire par les différents modèles : {len(df_without_regex)}")

    return df_regex_predicted, df_without_regex