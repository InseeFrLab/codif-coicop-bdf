"""Data preparation module for COICOP classification."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Union

import duckdb
import json
import numpy as np
import pandas as pd
import unidecode


if TYPE_CHECKING:
    from collections.abc import Sequence

COICOP_LEVELS = ["level1", "level2", "level3", "level4", "level5"]


def read_parquet(path: str | Path, encryption_key: str | None = None) -> pd.DataFrame:
    """Read parquet file, with optional DuckDB decryption."""
    if encryption_key:
        con = duckdb.connect()
        con.execute(f"PRAGMA add_parquet_key('encryption_key', '{encryption_key}');")
        return con.execute(
            f"SELECT * FROM read_parquet('{path}', encryption_config={{footer_key: 'encryption_key'}})"
        ).df()
    return pd.read_parquet(path)


def extract_levels(code: str) -> dict[str, str | None]:
    """Extract hierarchical levels from a COICOP code.

    Args:
        code: COICOP code string (e.g., "01.1.2.3.4")

    Returns:
        Dictionary with level1 through level5 keys
    """
    parts = code.split(".")
    return {
        "level1": parts[0].zfill(2),
        "level2": ".".join(parts[:2]) if len(parts) >= 2 else None,
        "level3": ".".join(parts[:3]) if len(parts) >= 3 else None,
        "level4": ".".join(parts[:4]) if len(parts) >= 4 else None,
        "level5": ".".join(parts[:5]) if len(parts) >= 5 else None,
    }


def load_coicop_hierarchy(path: str | Path) -> pd.DataFrame:
    """Load COICOP hierarchy definitions.

    Args:
        path: Path to the COICOP definitions CSV file

    Returns:
        DataFrame with columns: code, libelle, level1-level5
    """
    df = pd.read_csv(path, sep=";", encoding="utf-8")
    df.columns = ["libelle", "code"]

    # Extract hierarchical levels
    levels = df["code"].apply(extract_levels).apply(pd.Series)
    df = pd.concat([df, levels], axis=1)

    return df


def load_annotations(
    path: str | Path,
    exclude_technical: bool = True,
    encryption_key: str | None = None,
    preprocess: bool = True,
    code_column: str = "code",
) -> pd.DataFrame:
    """Load and preprocess annotation data.

    Args:
        path: Path to annotations.parquet file
        exclude_technical: Whether to exclude 98.x and 99.x technical codes
        encryption_key: Parquet encryption key for reading encrypted files
        preprocess: Whether to apply text preprocessing (unidecode, stopword removal, etc.)
        code_column: Name of the column containing COICOP codes

    Returns:
        DataFrame with product text and hierarchical labels
    """
    df = read_parquet(path, encryption_key)

    if preprocess:
        with open("data/text/stopwords.json", "r", encoding="utf-8") as json_file:
            stopwords = json.load(json_file)
        df = preprocess_text(df, 'product', stopwords)

    # Filter out technical codes if requested
    if exclude_technical:
        mask = ~df[code_column].str.startswith(("98", "99"))
        df = df[mask].copy()

    # Extract hierarchical levels from the code
    levels = df[code_column].apply(extract_levels).apply(pd.Series)
    df = pd.concat([df, levels], axis=1)

    # Clean product text
    df["text"] = df["product"].str.strip().str.lower()

    return df


def get_class_weights(labels: Sequence[str]) -> dict[str, float]:
    """Calculate class weights for imbalanced data.

    Args:
        labels: Sequence of label strings

    Returns:
        Dictionary mapping label to weight
    """
    label_counts = pd.Series(labels).value_counts()
    total = len(labels)
    n_classes = len(label_counts)

    weights = {}
    for label, count in label_counts.items():
        weights[label] = total / (n_classes * count)

    return weights



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
        DataFrame prétraitée.
    """
    df[text_feature + "_orig"] = df[text_feature].copy()
    df[text_feature] = df[text_feature].fillna("").map(unidecode.unidecode)
    df[text_feature] = df[text_feature].str.lower()
    df = remove_noise(df, text_feature)
    df = tokenize_and_clean(df, text_feature)
    df = remove_empty_and_strip(df, text_feature)
    df = remove_stopwords(df, text_feature, stopwords)
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
    libs_token = [lib.split() for lib in df[text_feature].to_list()]
    df[text_feature] = [
        " ".join([word for word in libs_token[i] if word not in stopwords])
        for i in range(len(libs_token))
    ]
    return df
