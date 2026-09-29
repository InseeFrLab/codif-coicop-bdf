"""Data preparation module for COICOP classification."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Union

import duckdb
import json
import os
import numpy as np
import pandas as pd


if TYPE_CHECKING:
    from collections.abc import Sequence

COICOP_LEVELS = ["level1", "level2", "level3", "level4", "level5"]


def configure_s3(con: duckdb.DuckDBPyConnection) -> None:
    """Configure DuckDB S3 secret from the AWS_* environment variables."""
    con.execute(f"""
        CREATE SECRET secret_ls3 (
            TYPE S3,
            KEY_ID '{os.environ["AWS_ACCESS_KEY_ID"]}',
            SECRET '{os.environ["AWS_SECRET_ACCESS_KEY"]}',
            ENDPOINT '{os.environ["AWS_S3_ENDPOINT"]}',
            SESSION_TOKEN '{os.environ["AWS_SESSION_TOKEN"]}',
            REGION 'us-east-1',
            URL_STYLE 'path',
            SCOPE 's3://'
        );
    """)


def read_parquet(path: str | Path, encryption_key: str | None = None) -> pd.DataFrame:
    """Read parquet from a local path or S3 URL (glob supported), optionally encrypted."""
    path = str(path)
    if path.startswith("s3://") or encryption_key:
        con = duckdb.connect()
        if path.startswith("s3://"):
            configure_s3(con)
        if encryption_key:
            con.execute(
                f"PRAGMA add_parquet_key('encryption_key', '{encryption_key}');"
            )
            return con.execute(
                f"SELECT * FROM read_parquet('{path}', encryption_config={{footer_key: 'encryption_key'}})"
            ).df()
        return con.execute(f"SELECT * FROM '{path}'").df()
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
        preprocess: Whether to apply text preprocessing (normalization, stopword removal, etc.)
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



def normalize_text(series: pd.Series) -> pd.Series:
    """Normalisation légère : copie de ``normalize_text`` de build-datasets.

    Copie de ``build-datasets/src/data/string_cleaning.py:normalize_text``, qui
    produit ``l_pr_product``, le texte que classify-ttc reçoit en production.
    L'appliquer aussi ici garantit que l'entraînement (texte brut) et la
    production (``l_pr_product``) donnent le même texte : ``unidecode``, utilisé
    auparavant, transcrivait ``€`` en ``eur`` et ``°`` en ``deg``, que
    ``normalize_text`` supprime. Toute modification doit être faite des deux côtés.
    """
    # Supprimer les multiples espaces
    series = series.str.replace(r"\s+", " ", regex=True)
    # Remplacer explicitement toutes les ligatures (NFKD ne les décompose pas)
    for lig, repl in {"œ": "oe", "Œ": "Oe", "æ": "ae", "Æ": "Ae"}.items():
        series = series.str.replace(lig, repl, regex=False)
    # Décomposer les accents, puis supprimer tout caractère non ASCII
    series = series.str.normalize("NFKD")
    series = series.str.encode("ascii", errors="ignore").str.decode("ascii")
    return series.str.lower()


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
    df[text_feature] = normalize_text(df[text_feature].fillna("").astype(str))
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
