"""Extract DDC data from S3 and apply COICOP code mapping."""

from __future__ import annotations

import logging
import os
import secrets
from datetime import datetime

import duckdb

logger = logging.getLogger(__name__)

BASE_S3_PATH = (
    "s3://projet-ddc/protected/iceberg-warehouse-prod/"
    "entrepot_ddc/complement_collecte/data"
)

DEFAULT_OUTPUT_PREFIX = (
    "s3://travail/projet-ml-classification-bdf/"
    "confidentiel/personnel_sensible/data/raw/sample"
)


def _build_source_patterns(
    annee: list[int], mois: list[int] | None
) -> list[str]:
    """Build S3 glob patterns for the requested year/month periods."""
    patterns = []
    if mois:
        for a in annee:
            for m in mois:
                patterns.append(f"{BASE_S3_PATH}/annee={a}/mois={m}/**/*.parquet")
    else:
        for a in annee:
            patterns.append(f"{BASE_S3_PATH}/annee={a}/**/*.parquet")
    return patterns


def _build_sample_sql(patterns: list[str], famille_circana_path: str) -> str:
    """Build the full SQL query for extraction."""
    statements = []

    # famille_circana view
    statements.append(
        f"CREATE VIEW famille_circana AS FROM '{famille_circana_path}';"
    )

    # One view per pattern, then UNION them
    view_names = []
    for i, pattern in enumerate(patterns):
        view_name = f"sample_{i}"
        view_names.append(view_name)
        statements.append(f"""CREATE VIEW {view_name} AS
    SELECT ddc.description_ean,
           ddc.variete,
           CASE WHEN ddc.variete[:2]='99'
                THEN famille_circana.coicop
                ELSE ddc.variete
            END AS coicop_code
    FROM
        (SELECT DISTINCT description_ean,
                variete,
                id_famille
        FROM
        '{pattern}') ddc
        LEFT JOIN famille_circana
        ON ddc.id_famille=famille_circana.id_famille
    WHERE len(coicop_code)>=10 AND coicop_code[:2] != '99';""")

    # Union all views
    if len(view_names) == 1:
        statements.append(
            f"CREATE VIEW sample AS FROM {view_names[0]};"
        )
    else:
        union_parts = "\n    UNION BY NAME\n    ".join(
            f"FROM {v}" for v in view_names
        )
        statements.append(f"CREATE VIEW sample AS\n    {union_parts};")

    return "\n\n".join(statements)


def extract_ddc(
    annee: list[int],
    mois: list[int] | None = None,
    output_s3_path: str | None = None,
    famille_circana_path: str = "data/famille_circana.csv",
    memory_limit: str = "6GB",
    dry_run: bool = False,
    encrypt: bool = False,
    encryption_key: str | None = None,
) -> None:
    """Extract DDC data from S3, apply COICOP mapping, and write to parquet.

    Parameters
    ----------
    annee : list[int]
        Year(s) to extract.
    mois : list[int] | None
        Month(s) to extract. If None, all months for the given years.
    output_s3_path : str | None
        Override the default S3 output path.
    famille_circana_path : str
        Path to the famille_circana CSV mapping file.
    memory_limit : str
        DuckDB memory limit.
    dry_run : bool
        If True, print the SQL without executing.
    encrypt : bool
        If True, encrypt the output parquet file (AES-GCM 256 bits).
    encryption_key : str | None
        Explicit encryption key (hex, 32 chars). Implies encrypt=True.
    """
    if encryption_key:
        encrypt = True
    patterns = _build_source_patterns(annee, mois)

    if output_s3_path is None:
        timestamp = datetime.now().strftime("%Y%m%d")
        output_s3_path = f"{DEFAULT_OUTPUT_PREFIX}/ddc_{timestamp}.parquet"

    sql = _build_sample_sql(patterns, famille_circana_path)

    if encrypt:
        copy_stmt = (
            f"COPY sample TO '{output_s3_path}' "
            "(ENCRYPTION_CONFIG {footer_key: 'encryption_key'});"
        )
    else:
        copy_stmt = f"COPY sample TO '{output_s3_path}';"

    if dry_run:
        print("-- S3 secrets configuration (omitted)")
        print(f"SET memory_limit = '{memory_limit}';\n")
        if encrypt:
            key_display = encryption_key if encryption_key else "<CLÉ_AUTO_GÉNÉRÉE>"
            print(f"PRAGMA add_parquet_key('encryption_key', '{key_display}');\n")
        print(sql)
        print()
        print(copy_stmt)
        return

    logger.info("Connecting to DuckDB...")
    con = duckdb.connect()

    # Configure S3 secrets
    con.execute(f"""
        CREATE SECRET secret_prod (
            TYPE S3,
            KEY_ID '{os.environ["MINIO_PROD_ACCESS_KEY_ID"]}',
            SECRET '{os.environ["MINIO_PROD_SECRET_ACCESS_KEY"]}',
            ENDPOINT '{os.environ["MINIO_PROD_S3_ENDPOINT"]}',
            SESSION_TOKEN '',
            REGION 'us-east-1',
            URL_STYLE 'path',
            SCOPE 's3://projet-ddc/'
        );
    """)
    con.execute(f"""
        CREATE SECRET secret_ls3 (
            TYPE S3,
            KEY_ID '{os.environ["AWS_ACCESS_KEY_ID"]}',
            SECRET '{os.environ["AWS_SECRET_ACCESS_KEY"]}',
            ENDPOINT '{os.environ["AWS_S3_ENDPOINT"]}',
            SESSION_TOKEN '{os.environ["AWS_SESSION_TOKEN"]}',
            REGION 'us-east-1',
            URL_STYLE 'path',
            SCOPE 's3://travail/projet-ml-classification-bdf'
        );
    """)

    con.execute(f"SET memory_limit = '{memory_limit}';")

    if encrypt:
        key = encryption_key if encryption_key else secrets.token_hex(16)
        con.execute(f"PRAGMA add_parquet_key('encryption_key', '{key}');")
        logger.info("Clé de chiffrement parquet : %s", key)
        logger.info(
            "Conservez cette clé : elle est nécessaire pour relire le fichier."
        )

    # Execute the sample construction SQL
    for stmt in sql.split(";"):
        stmt = stmt.strip()
        if stmt:
            con.execute(stmt)

    # Count rows
    count = con.execute("SELECT count(*) FROM sample").fetchone()[0]
    logger.info(f"Extracted {count} rows")

    # Write output
    logger.info(f"Writing to {output_s3_path}...")
    con.execute(copy_stmt)
    logger.info("Done.")

    con.close()
