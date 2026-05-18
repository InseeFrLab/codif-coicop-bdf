from typing import List, Union
import numpy as np
import pandas as pd
import unicodedata


def normalize_text(series: Union[pd.Series, List]) -> pd.Series:
    
    if isinstance(series, list):
        series = pd.Series(series)

    if not isinstance(series, pd.Series):
        raise TypeError("Input must be a pandas Series or a list of strings.")

    # Supprimer les multiples espaces
    series = series.str.replace(r"\s+", " ", regex=True)

    # Remplacer explicitement toutes les ligatures (minuscule et majuscule)
    ligatures = {"œ": "oe", "Œ": "Oe", "æ": "ae", "Æ": "Ae"}
    for lig, repl in ligatures.items():
        series = series.str.replace(lig, repl, regex=False)

    # Décomposer les accents
    series = series.str.normalize("NFKD")

    # Supprimer tous les caractères non-ASCII (enlève accents restants)
    series = series.str.encode("ascii", errors="ignore").str.decode("ascii")

    # Convertir en minuscules
    series = series.str.lower()

    return series


def preprocess_text(
    df: pd.DataFrame, text_feature: str, stopwords: Union[List[str], set[str]]
) -> pd.DataFrame:
    """
    Pipeline principal de prétraitement textuel.

    Args:
        df: DataFrame contenant le texte à traiter.
        text_feature: Nom de la colonne texte à prétraiter.
        stopwords: Liste ou ensemble de mots à exclure.

    Returns:
        DataFrame prétraité.
    """
    df[text_feature + "_orig"] = df[text_feature].copy()
    df[text_feature] = df[text_feature].str.lower()
    df = remove_noise(df, text_feature)
    df = tokenize_and_clean(df, text_feature)
    df = remove_empty_and_strip(
        df, text_feature
    )  # attention, la fonction retire les observations associées à des labels vides
    df = remove_stopwords(df, text_feature, stopwords)
    df[text_feature] = remove_accents(df, text_feature)
    return df


def remove_noise(df: pd.DataFrame, text_feature: str) -> pd.DataFrame:
    """
    Supprime le bruit textuel : mots inutiles, ponctuation, chiffres, etc.

    Args:
        df: DataFrame contenant la colonne texte.
        text_feature: Nom de la colonne texte.

    Returns:
        DataFrame nettoyé.
    """
    lib_to_remove = r"\brien\b|\rien du tout\b"
    words_to_remove = r"\brien\b|\rien du tout\b"

    # Traiter les valeurs manquantes
    df[text_feature] = df[text_feature].fillna(" ")
    # On supprime les libellés vide de sens
    df[text_feature] = df[text_feature].str.replace(lib_to_remove, "", regex=True)
    # On supprime toutes les ponctuations
    df[text_feature] = df[text_feature].str.replace(r"[^\w\s]+", " ", regex=True)
    # On supprime les stopwords custom
    df[text_feature] = df[text_feature].str.replace(words_to_remove, "", regex=True)
    # CHIFFRES : QUOI FAIRE ? TEMPORAIREMENT ON SUPPRIME
    df[text_feature] = df[text_feature].str.replace(r"[\d+]", " ", regex=True)
    # On supprime les mots d'une seule lettre
    df[text_feature] = df[text_feature].apply(
        lambda x: " ".join([w for w in x.split() if len(w) > 1])
    )
    # On supprime les multiple space
    df[text_feature] = df[text_feature].str.replace(r"\s\s+", " ", regex=True)
    return df


def tokenize_and_clean(df: pd.DataFrame, text_feature: str) -> pd.DataFrame:
    """
    Tokenise chaque texte, supprime les doublons tout en conservant l'ordre.

    Args:
        df: DataFrame contenant la colonne texte.
        text_feature: Nom de la colonne texte.

    Returns:
        DataFrame avec texte nettoyé.
    """
    libs_token = [lib.split() for lib in df[text_feature].to_list()]
    libs_token = [
        sorted(set(libs_token[i]), key=libs_token[i].index)
        for i in range(len(libs_token))
    ]
    df[text_feature] = [" ".join(libs_token[i]) for i in range(len(libs_token))]
    return df


def remove_empty_and_strip(df, text_feature):
    df[text_feature] = df[text_feature].str.strip()
    df[text_feature] = df[text_feature].replace(r"^\s*$", np.nan, regex=True)
    df = df.dropna(subset=[text_feature])
    return df


def remove_stopwords(df, text_feature, stopwords):
    """
    Remove stopwords from a string
    The list of stopwords is based on the file stopwords.json.

    Args:
        df: dataframe with feature to clean
        text_feature: feature in df to clean
        stopwords: list of stopwords to remove

    Return:
        df: dataframe with cleaned feature
    """
    libs_token = [lib.split() for lib in df[text_feature].to_list()]
    df[text_feature] = [
        " ".join([word for word in libs_token[i] if word not in stopwords])
        for i in range(len(libs_token))
    ]
    return df


def remove_accents(df: pd.DataFrame, text_feature: str) -> pd.Series:
    """
    Remove diacritical marks from a string by converting accented characters
    (e.g., é, à, ç, ü) to their ASCII base equivalents (e.g., e, a, c, u).

    This function performs Unicode normalization (NFKD) and drops combining
    accent characters, returning an ASCII-only representation while preserving
    letters, digits, spaces, and punctuation.

    Args:
        df: Dataframe with feature to normalize
        text_feature: feature name to normalize.

    Returns:
        A string with accents and diacritics removed.
    """

    return (
        df[text_feature]
        .astype(str)
        .apply(
            lambda x: (
                unicodedata.normalize(
                    "NFKD", x
                )  # Normalisation unicode (décompose les accents)
                .encode(
                    "ascii", "ignore"
                )  # Suppression des accents en gardant uniquement ASCII
                .decode("ascii")
            )
        )
    )

