"""Build a balanced training dataset from DDC extraction and synthetic data."""

from __future__ import annotations

import json
import logging
import os

import duckdb
import pandas as pd

from ..preprocessing.data_preparation import preprocess_text

logger = logging.getLogger(__name__)

STOPWORDS_PATH = "data/text/stopwords.json"


def _configure_s3(con: duckdb.DuckDBPyConnection) -> None:
    """Configure DuckDB S3 secret for the work bucket."""
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


def _read_parquet(path: str, encryption_key: str | None = None) -> pd.DataFrame:
    """Read parquet from local path or S3 URL (with glob support)."""
    if path.startswith("s3://") or encryption_key:
        con = duckdb.connect()
        if path.startswith("s3://"):
            _configure_s3(con)
        if encryption_key:
            con.execute(
                f"PRAGMA add_parquet_key('encryption_key', '{encryption_key}');"
            )
            return con.execute(
                f"SELECT * FROM read_parquet('{path}', encryption_config={{footer_key: 'encryption_key'}})"
            ).df()
        return con.execute(f"SELECT * FROM '{path}'").df()
    return pd.read_parquet(path)


def _write_parquet(
    df: pd.DataFrame, path: str, encryption_key: str | None = None
) -> None:
    """Write DataFrame to parquet on local path or S3 URL."""
    if path.startswith("s3://") or encryption_key:
        con = duckdb.connect()
        if path.startswith("s3://"):
            _configure_s3(con)
        if encryption_key:
            con.execute(
                f"PRAGMA add_parquet_key('encryption_key', '{encryption_key}');"
            )
        con.register("__output", df)
        copy = f"COPY __output TO '{path}' (FORMAT PARQUET"
        if encryption_key:
            copy += ", ENCRYPTION_CONFIG {footer_key: 'encryption_key'}"
        copy += ")"
        con.execute(copy)
        return
    df.to_parquet(path, index=False)


def _load_stopwords() -> list[str]:
    with open(STOPWORDS_PATH, encoding="utf-8") as f:
        return json.load(f)


def _extract_level4(code: pd.Series) -> pd.Series:
    return code.apply(lambda c: ".".join(str(c).split(".")[:4]))


def build_training_data(
    ddc_path: str,
    output_path: str,
    synthetic_path: str = "data/synthetic_data.csv",
    max_per_code: int = 1000,
    seed: int = 42,
    encryption_key: str | None = None,
    preprocess: bool = True,
    ddc_text_column: str = "description_ean",
    ddc_code_column: str = "coicop_code",
) -> None:
    """Build a balanced training dataset from DDC and synthetic data.

    Args:
        ddc_path: Path to DDC parquet (local, S3, or HTTP).
        output_path: Output parquet file path.
        synthetic_path: Path to synthetic data CSV (semicolon-separated).
        max_per_code: Max DDC rows per level-4 code before sampling.
        seed: Random seed for reproducible sampling.
    """
    stopwords = _load_stopwords()

    # --- DDC data ---
    logger.info("Reading DDC data from %s", ddc_path)
    ddc = _read_parquet(ddc_path, encryption_key)
    ddc = ddc.rename(columns={ddc_text_column: "product", ddc_code_column: "code"})
    ddc = ddc[["product", "code"]].copy()
    ddc["source"] = "ddc"

    logger.info("DDC rows before preprocessing: %d", len(ddc))
    if preprocess:
        ddc = preprocess_text(ddc, "product", stopwords)
    ddc = ddc.drop_duplicates(subset=["product", "code"])
    logger.info("DDC rows after preprocessing + dedup: %d", len(ddc))

    # --- Synthetic data ---
    logger.info("Reading synthetic data from %s", synthetic_path)
    synthetic = pd.read_csv(
        synthetic_path,
        sep=";",
        skiprows=1,
        header=None,
        usecols=[0, 1],
        names=["product", "code"],
    )
    synthetic = synthetic[["product", "code"]].copy()
    synthetic["source"] = "synthetic"

    logger.info("Synthetic rows before preprocessing: %d", len(synthetic))
    if preprocess:
        synthetic = preprocess_text(synthetic, "product", stopwords)
    synthetic = synthetic.drop_duplicates(subset=["product", "code"])
    logger.info("Synthetic rows after preprocessing + dedup: %d", len(synthetic))

    # --- 8-char code column (used downstream by training) ---
    ddc["code8"] = ddc["code"].str[:8]
    synthetic["code8"] = synthetic["code"].str[:8]

    # --- Balance at level 4, prioritizing DDC ---
    ddc["code_level4"] = _extract_level4(ddc["code"])
    synthetic["code_level4"] = _extract_level4(synthetic["code"])

    all_level4_codes = set(ddc["code_level4"]) | set(synthetic["code_level4"])

    parts: list[pd.DataFrame] = []

    for code_l4 in sorted(all_level4_codes):
        ddc_subset = ddc[ddc["code_level4"] == code_l4]
        synthetic_subset = synthetic[synthetic["code_level4"] == code_l4]
        n_ddc = len(ddc_subset)

        if n_ddc >= max_per_code:
            # DDC alone exceeds cap: sample DDC down, no synthetic needed
            parts.append(ddc_subset.sample(n=max_per_code, random_state=seed))
        elif n_ddc > 0:
            # DDC exists but below cap: keep all DDC, fill remainder with synthetic
            parts.append(ddc_subset)
            remaining = max_per_code - n_ddc
            if len(synthetic_subset) > 0:
                n_syn = min(len(synthetic_subset), remaining)
                parts.append(synthetic_subset.sample(n=n_syn, random_state=seed))
        else:
            # No DDC for this code: use synthetic, capped at max_per_code
            if len(synthetic_subset) > 0:
                n_to_add = min(len(synthetic_subset), max_per_code)
                parts.append(synthetic_subset.sample(n=n_to_add, random_state=seed))

    result = pd.concat(parts, ignore_index=True)
    result = result.drop(columns=["code_level4"])

    # --- Save ---
    _write_parquet(result, output_path, encryption_key)

    # --- Summary ---
    n_ddc = (result["source"] == "ddc").sum()
    n_synthetic = (result["source"] == "synthetic").sum()
    n_codes = result["code"].nunique()
    logger.info(
        "Training data saved to %s: %d rows (%d DDC, %d synthetic), %d unique codes",
        output_path,
        len(result),
        n_ddc,
        n_synthetic,
        n_codes,
    )
