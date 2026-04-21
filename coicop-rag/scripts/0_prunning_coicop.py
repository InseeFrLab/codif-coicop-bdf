"""
COICOP Pruning
==============
Prunes linear hierarchies from the raw COICOP notices and exports
the filtered notices and mapping table to S3.
"""
import argparse
import logging
import yaml
from coicop_rag.data.pruning import prune_linear_hierarchies
from coicop_rag.utils import create_duckdb_connection


def main():
    parser = argparse.ArgumentParser(description="COICOP pruning pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config YAML file"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    logger.info("=" * 80)
    logger.info("STARTING COICOP PRUNING PIPELINE")
    logger.info("=" * 80)

    con = create_duckdb_connection()

    logger.info("Loading raw COICOP notices...")
    notices_raw = con.sql(
        f"SELECT * FROM read_csv('{config['coicop']['path_raw']}')"
    ).to_df()
    logger.info(f"  → Coicop notices load from file: {config['coicop']['path_raw']}")
    logger.info(f"  → {len(notices_raw)} rows loaded")

    # Remove Level 5 (Poste)
    notices_raw = notices_raw.loc[notices_raw["type"] != "Poste"]
    logger.info(f"  → {len(notices_raw)} rows after removing 'Poste' level")

    logger.info("Pruning linear hierarchies...")
    notices_filtered, mapping_table = prune_linear_hierarchies(notices_raw)
    logger.info(f"  → {len(notices_filtered)} notices after pruning")
    logger.info(f"  → {len(mapping_table)} entries in mapping table")

    logger.info("Exporting results to S3...")
    con.sql(f"""
        COPY notices_filtered
        TO '{config["coicop"]["path_prunned_lvl4"]}'
        (FORMAT PARQUET)
    """)
    logger.info(f"  → Pruned notices: {config['coicop']['path_prunned_lvl4']}")

    con.sql(f"""
        COPY mapping_table
        TO '{config["coicop"]["path_mapping_lvl4"]}'
        (FORMAT PARQUET)
    """)
    logger.info(f"  → Mapping table: {config['coicop']['path_mapping_lvl4']}")

    logger.info("=" * 80)
    logger.info("COICOP PRUNING PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    main()
