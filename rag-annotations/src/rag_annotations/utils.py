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

# Les fonctions génériques (chemins, codes, connexion S3) vivent désormais dans
# `codif_common`, où elles ne sont écrites qu'une fois. Elles sont ré-exportées
# ici pour que les imports existants continuent de fonctionner.
from codif_common.codes import get_parents, truncate_code
from codif_common.paths import expand_paths
from codif_common.s3 import connect_env as create_duckdb_connection


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
