"""
Annotation Pruning
==================
Truncates annotation codes to level 4 and applies the COICOP linear hierarchy
mapping to normalize ground truth codes before the RAG pipeline.

Must be run after 0_prunning_coicop.py (requires the mapping table on S3).
"""
import argparse
import logging

import yaml

from coicop_rag.data.pruning import prune_annotation_lvl4
from coicop_rag.utils import create_duckdb_connection, expand_paths


def main():
    parser = argparse.ArgumentParser(description="Annotation pruning pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config YAML file"
    )
    parser.add_argument("--run-id", required=True, help="Workflow run identifier")
    parser.add_argument("--run-date", required=True, help="Workflow run date (YYYY-MM-DD)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config = expand_paths(config, run_id=args.run_id, run_date=args.run_date)

    logger.info("=" * 80)
    logger.info("STARTING ANNOTATION PRUNING PIPELINE")
    logger.info("=" * 80)

    con = create_duckdb_connection()

    # -----------------------------------------------------------------------
    # Load mapping table and raw notices (needed for label lookup)
    # -----------------------------------------------------------------------

    logger.info("Loading mapping table...")
    mapping_table_lvl4 = con.sql(
        f"SELECT * FROM read_parquet('{config['coicop']['path_mapping_lvl4']}')"
    ).to_df()
    logger.info(f"  → {len(mapping_table_lvl4)} entries in mapping table")

    logger.info("Loading raw COICOP notices...")
    notices_raw = con.sql(
        f"SELECT * FROM read_csv('{config['coicop']['path_raw']}')"
    ).to_df()
    logger.info(f"  → {len(notices_raw)} notices loaded")

    # -----------------------------------------------------------------------
    # Load raw annotations
    # -----------------------------------------------------------------------

    logger.info("=" * 80)
    logger.info("STEP 1: LOADING ANNOTATIONS")
    logger.info("=" * 80)

    annotations = con.sql(
        f"SELECT * FROM read_parquet('{config['annotations']['s3_path']}')"
    ).to_df()

    annotations = annotations[config["annotations"]["features"]]

    logger.info(f"✓ {len(annotations)} annotations loaded")

    # -----------------------------------------------------------------------
    # Truncate and prune codes
    # -----------------------------------------------------------------------

    logger.info("=" * 80)
    logger.info("STEP 2: TRUNCATING AND PRUNING CODES")
    logger.info("=" * 80)

    annotations = prune_annotation_lvl4(
        annotations,
        mapping_table_lvl4,
        notices_raw
    )

    logger.info(f"✓ {len(annotations)} annotations after pruning")

    # -----------------------------------------------------------------------
    # Export
    # -----------------------------------------------------------------------

    logger.info("=" * 80)
    logger.info("STEP 3: EXPORTING PRUNED ANNOTATIONS")
    logger.info("=" * 80)

    output_path = config["annotations"]["s3_path_rag"]
    con.sql(f"""
        COPY annotations
        TO '{output_path}'
        (FORMAT PARQUET)
    """)
    logger.info(f"✓ Pruned annotations exported: {output_path}")

    logger.info("=" * 80)
    logger.info("ANNOTATION PRUNING PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    main()
