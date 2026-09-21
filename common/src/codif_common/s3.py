"""Connexions DuckDB configurées pour S3/MinIO.

⚠️  **Il existe quatre dialectes dans le dépôt, et ils ne sont pas
interchangeables.** Ce module en expose deux — ceux dont toutes les copies
étaient rigoureusement identiques entre elles — et laisse les deux autres en
place, délibérément.

| Dialecte | Où | Ici ? |
|---|---|---|
| `SET s3_*`, endpoint en dur, jeton forcé vide | `prune-codes`, `rag-notices`, `rag-annotations` | ✅ `connect_env` |
| `CREATE OR REPLACE SECRET s3_secret`, endpoint en cascade, tolérant | `export-results`, `report` | ✅ `connect_secret` |
| `CREATE SECRET … REGION … SCOPE`, `os.environ[...]` strict | `reconcile-llm`, `classify-ttc` (×4) | ❌ laissé en place |
| R, `CREATE SECRET` sans `URL_STYLE` ni `SCOPE` | `classify-lcs/R/main.R` | ❌ hors Python |

Pourquoi ne pas tout unifier : le premier dialecte force
``s3_session_token=''`` et **ignore donc le jeton de session** qu'Argo injecte.
Le troisième pose un ``SCOPE``, dont `classify-ttc/src/data/extract_ddc.py`
dépend pour faire cohabiter deux secrets. Les fusionner changerait
l'authentification effective — et ça ne se vérifie pas hors du cluster. La
déduplication s'arrête donc là où le comportement commencerait à bouger.
"""

import os
from textwrap import dedent

import duckdb

# Endpoint MinIO du SSPCloud, utilisé quand rien n'est configuré.
DEFAULT_ENDPOINT = "minio.lab.sspcloud.fr"


def resolve_endpoint() -> str:
    """Endpoint S3, par ordre de priorité : ``AWS_S3_ENDPOINT``, puis
    ``AWS_ENDPOINT_URL`` débarrassé de son schéma, puis le défaut MinIO.

    Reproduit à l'identique la cascade de `export-results` et `report`.
    """
    return (
        os.environ.get("AWS_S3_ENDPOINT")
        or os.environ.get("AWS_ENDPOINT_URL", "").replace("https://", "").replace("http://", "")
        or DEFAULT_ENDPOINT
    )


def connect_env() -> duckdb.DuckDBPyConnection:
    """Connexion en mémoire configurée par ``SET s3_*`` (dialecte 1).

    Reproduit exactement ce que faisaient `prune_codes.utils`,
    `rag_notices.utils` et `rag_annotations.utils` — **y compris le jeton de
    session forcé à vide et l'endpoint en dur**. Ce n'est pas un oubli : le
    corriger changerait l'authentification de trois modules d'un coup, ce qui
    ne se teste pas en local. À reprendre séparément, avec un run de contrôle.
    """
    con = duckdb.connect(database=":memory:")
    con.execute(f"""
        SET s3_access_key_id='{os.getenv("AWS_ACCESS_KEY_ID")}';
        SET s3_secret_access_key='{os.getenv("AWS_SECRET_ACCESS_KEY")}';
        SET s3_session_token='';
        SET s3_endpoint='{DEFAULT_ENDPOINT}';
        SET s3_region='us-east-1';
    """)
    return con


def connect_secret() -> duckdb.DuckDBPyConnection:
    """Connexion configurée par ``CREATE OR REPLACE SECRET`` (dialecte 2).

    Reproduit exactement `export-results` et `report` : endpoint en cascade,
    jeton de session transmis, lectures d'environnement **tolérantes**
    (``.get(..., "")``) — des identifiants absents ne lèvent donc pas ici, mais
    produiront une erreur S3 à la première lecture.
    """
    con = duckdb.connect()
    con.sql("INSTALL httpfs; LOAD httpfs;")
    con.sql(
        dedent(f"""
            CREATE OR REPLACE SECRET s3_secret (
                TYPE s3,
                KEY_ID '{os.environ.get("AWS_ACCESS_KEY_ID", "")}',
                SECRET '{os.environ.get("AWS_SECRET_ACCESS_KEY", "")}',
                SESSION_TOKEN '{os.environ.get("AWS_SESSION_TOKEN", "")}',
                ENDPOINT '{resolve_endpoint()}',
                URL_STYLE 'path',
                USE_SSL true
            );
        """)
    )
    return con
