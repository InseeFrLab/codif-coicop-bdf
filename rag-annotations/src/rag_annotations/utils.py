"""
Shared utilities for the annotation-based RAG.

Mostly mirrors `rag_notices.utils`, with an added `embed_texts` helper used by
both the vector-DB build script and the run script.
"""
import os
from typing import Any, List, Optional

import duckdb
import pandas as pd
from tqdm import tqdm


def expand_paths(obj: Any, **kwargs: str) -> Any:
    """Recursively apply str.format(**kwargs) to every string in a nested
    dict/list structure. Used to substitute {run_id} / {run_date} in config
    path templates."""
    if isinstance(obj, dict):
        return {k: expand_paths(v, **kwargs) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand_paths(v, **kwargs) for v in obj]
    if isinstance(obj, str):
        return obj.format(**kwargs)
    return obj


def create_duckdb_connection() -> duckdb.DuckDBPyConnection:
    """
    Create a DuckDB in-memory connection configured for S3/MinIO access.
    """
    con = duckdb.connect(database=":memory:")
    con.execute(f"""
        SET s3_access_key_id='{os.getenv("AWS_ACCESS_KEY_ID")}';
        SET s3_secret_access_key='{os.getenv("AWS_SECRET_ACCESS_KEY")}';
        SET s3_session_token='';
        SET s3_endpoint='minio.lab.sspcloud.fr';
        SET s3_region='us-east-1';
    """)
    return con


def truncate_code(code: str, level: int) -> Optional[str]:
    """
    Truncate a dotted COICOP code to a hierarchical level.

    e.g. truncate_code("01.1.2.3", 2) -> "01.1". Returns the code unchanged if
    it is already at/below `level`, and None for invalid input.
    """
    if code is None or not isinstance(code, str) or code == "":
        return None
    parts = code.split(".")
    if len(parts) <= level:
        return code
    return ".".join(parts[:level])


def get_parents(code: str) -> List[str]:
    """Return all parent codes of `code`, one per higher hierarchical level."""
    code_level = len(code.split("."))
    return [truncate_code(code, level) for level in range(1, code_level)]


def embed_texts(client, model: str, texts: List[str], batch_size: int = 64) -> List[List[float]]:
    """
    Embed a list of texts via an OpenAI-compatible embeddings endpoint.

    Sends the texts in batches (the embeddings API accepts a list input) to
    limit the number of round-trips.

    Args:
        client: OpenAI-compatible client (e.g. llm.lab).
        model: Embedding model name.
        texts: Texts to embed.
        batch_size: Number of texts per request.

    Returns:
        List of embedding vectors, in the same order as `texts`.
    """
    embeddings: List[List[float]] = []
    with tqdm(total=len(texts), desc="Embeddings", unit="doc") as pbar:
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            response = client.embeddings.create(model=model, input=batch)
            embeddings.extend(item.embedding for item in response.data)
            pbar.update(len(batch))
    return embeddings


def is_present(value: Any) -> Optional[Any]:
    """Return `value` if it is a usable (non-missing) scalar, else None.

    Handles None, NaN (float), pandas NA, and empty/whitespace strings.
    """
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, str) and not value.strip():
        return None
    return value


def build_location_text(product: Any, shop: Any = None, shop_type: Any = None) -> str:
    """
    Build the canonical descriptive text of a product, enriched with the
    purchase location when available. This is the single representation used
    everywhere: embedded into the vector DB, used for the query embedding, and
    shown as few-shot examples / "produit à coder" in the prompt — so the same
    product is represented identically on the index and query sides.

    Examples:
        "café"                                  (no location info)
        "café - magasin ou lieu d'achat : Super U"
        "café - magasin ou lieu d'achat : Super U (type : supermarché)"
    """
    product = str(product).strip()
    shop = is_present(shop)
    shop_type = is_present(shop_type)
    if shop is None:
        return product
    location = str(shop)
    if shop_type is not None:
        location += f" (type : {shop_type})"
    return f"{product} - magasin ou lieu d'achat : {location}"
