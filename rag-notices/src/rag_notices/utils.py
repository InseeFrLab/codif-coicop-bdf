import os
from typing import Any, List, Dict, Optional
import pandas as pd
import duckdb

# Les fonctions génériques (chemins, codes, connexion S3) vivent désormais dans
# `codif_common`, où elles ne sont écrites qu'une fois. Elles sont ré-exportées
# ici pour que les imports existants continuent de fonctionner.
from codif_common.codes import get_parents, truncate_code
from codif_common.paths import expand_paths
from codif_common.s3 import connect_env as create_duckdb_connection


def merge_eval_and_retreived(
    df_eval: pd.DataFrame,
    retrieved_codes: pd.DataFrame,
    retrieval_size: int,
    code_name: str,
    col_retrieved_codes_name: str = "list_retrieved_codes"
) -> List[Dict]:
    """
    Merge evaluation predictions with retrieved codes and compute retrieval indicator.

    Combines the wide-format retrieved codes DataFrame (one column per retrieved code)
    into a single list column, joins it onto the evaluation DataFrame on 'id', then
    adds a boolean column indicating whether the ground truth code was retrieved.

    Args:
        df_eval: Evaluation DataFrame containing at least 'id' and the ground truth
            column named by code_name.
        retrieved_codes: DataFrame with 'id' and one numeric string column per
            retrieved code ("0", "1", ..., str(retrieval_size - 1)).
        retrieval_size: Number of retrieved codes per record (number of numeric columns).
        code_name: Name of the ground truth code column in df_eval.
        col_retrieved_codes_name: Name of the list column to create. Default:
            "list_retrieved_codes".

    Returns:
        List of dicts (records) with all evaluation fields, the retrieved codes list,
        and an 'in_retrieved' boolean flag.
    """
    cols = [str(i) for i in range(retrieval_size)]
    retrieved_codes[col_retrieved_codes_name] = retrieved_codes[cols].values.tolist()
    retrieved_codes = retrieved_codes.drop(cols, axis=1)

    df_eval = df_eval.merge(retrieved_codes, how="left", on="id")

    df_eval["in_retrieved"] = df_eval.apply(
        lambda row: row[code_name] in row[col_retrieved_codes_name],
        axis=1
    )

    return df_eval.to_dict('records')


