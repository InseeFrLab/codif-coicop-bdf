import os
from typing import Any, List, Dict, Optional
import pandas as pd
import duckdb


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
    Returns:
        Configured DuckDB connection.
    """
    con = duckdb.connect(database=":memory:")

    con.execute(f"""
        SET s3_endpoint='{os.getenv("AWS_S3_ENDPOINT")}';
        SET s3_access_key_id='{os.getenv("AWS_ACCESS_KEY_ID")}';
        SET s3_secret_access_key='{os.getenv("AWS_SECRET_ACCESS_KEY")}';
        SET s3_session_token='';
        SET s3_endpoint='minio.lab.sspcloud.fr';
        SET s3_region='us-east-1';
    """)
    return con


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


def truncate_code(code: str, level: int) -> Optional[str]:
    """
    Truncate code to specified hierarchical level
    
    Args:
        code: Full code (e.g., '08.1.2.3.4' or '08.1.6')
        level: Level to truncate to (1-5)
    
    Returns:
        Truncated code or original if already at or below target level,
        None if invalid
    """
    if code is None or not isinstance(code, str) or code == '':
        return None
    
    # Split by dot separator
    parts = code.split('.')
    
    # If code is already at or below target level, return as-is
    if len(parts) <= level:
        return code
    
    # Otherwise truncate to target level
    return '.'.join(parts[:level])

def get_parents(code: str) -> List[str]:
    """
    Get all parent codes for a given code by truncating at each hierarchical level.

    Args:
        code: The full hierarchical code (e.g., '08.1.2.3.4' or '08.1.6')

    Returns:
        List of parent codes, each representing a higher level in the hierarchy.
        Returns empty list if input is invalid or has no parents.
    """
    code_level = len(code.split('.'))
    parents = []
    for level in range(1, code_level):
        parents.append(
            truncate_code(code, level)
        )

    return parents