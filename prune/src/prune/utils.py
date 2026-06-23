import os
from typing import Any, Optional

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
    """Create a DuckDB in-memory connection configured for S3/MinIO access."""
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
    Truncate a dotted COICOP code to the given hierarchical level.

    e.g. truncate_code('08.1.2.3.4', 4) -> '08.1.2.3'. Returns the code unchanged
    if already at or below the target level, None if invalid.
    """
    if code is None or not isinstance(code, str) or code == '':
        return None
    parts = code.split('.')
    if len(parts) <= level:
        return code
    return '.'.join(parts[:level])
