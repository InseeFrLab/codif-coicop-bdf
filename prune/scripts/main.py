"""
Pruning unifié (troncature niveau 4 + élagage des hiérarchies linéaires)
========================================================================
Étape (module autonome `prune`) qui produit **tous** les artefacts prunés du
pipeline, que l'aval se contente ensuite de lire (plus aucun re-pruning à la
volée) :

  - nomenclature COICOP prunée + table de mapping (code → code_parent_equivalent) ;
  - annotations « train » (KB) prunées ;
  - annotations « test » (à coder) prunées ;
  - suggester pruné.

Usage :
    uv run scripts/main.py --config config/config.yaml --run-id <ID> --run-date <YYYY-MM-DD>
"""
import argparse
import logging

import yaml

from prune.pruning import prune_annotation_lvl4, prune_linear_hierarchies
from prune.utils import create_duckdb_connection, expand_paths


def _copy_to_parquet(con, df, name, path):
    """Écrit un DataFrame pandas vers un parquet S3 via DuckDB."""
    con.register(name, df)
    con.sql(f"COPY {name} TO '{path}' (FORMAT PARQUET)")


def prune_annotations_file(con, in_path, features, mapping_table_lvl4, notices_raw, logger):
    """Charge un fichier d'annotations, le restreint aux features, et applique
    `prune_annotation_lvl4` (troncature niveau 4 + mapping + resync libellé)."""
    df = con.sql(f"SELECT * FROM read_parquet('{in_path}')").to_df()
    df = df[features]
    logger.info(f"  → {len(df)} lignes chargées depuis {in_path}")
    return prune_annotation_lvl4(df, mapping_table_lvl4, notices_raw)


def load_and_prune_suggester(con, sug_cfg, product_col, mapping_table_lvl4, notices_raw, logger):
    """Charge le suggester (source autoportante), aligne ses colonnes sur
    (`product_col`, 'code', 'coicop'), tague `source='suggester'`, puis applique
    exactement le même `prune_annotation_lvl4` que les annotations."""
    reader = "read_parquet" if sug_cfg["s3_path"].endswith(".parquet") else "read_csv_auto"
    raw_count = con.sql(f"SELECT COUNT(*) FROM {reader}('{sug_cfg['s3_path']}')").fetchone()[0]

    distinct = "DISTINCT" if sug_cfg.get("dedup", True) else ""
    df = con.sql(f"""
        SELECT {distinct}
            "{sug_cfg['source_product_col']}" AS "{product_col}",
            "{sug_cfg['code_col']}"    AS code,
            "{sug_cfg['coicop_col']}"  AS coicop
        FROM {reader}('{sug_cfg['s3_path']}')
    """).to_df()
    logger.info(f"  → {raw_count} lignes suggester → {len(df)} après dedup={sug_cfg.get('dedup', True)}")
    df["source"] = "suggester"

    df = prune_annotation_lvl4(df, mapping_table_lvl4, notices_raw)

    before = len(df)
    df = df.dropna(subset=[product_col, "code"])
    df = df[df[product_col].astype(str).str.strip() != ""]
    logger.info(f"  → {len(df)} lignes suggester exploitables (retirées {before - len(df)})")
    return df


def main():
    parser = argparse.ArgumentParser(description="Pruning unifié")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config YAML")
    parser.add_argument("--run-id", required=True, help="Workflow run identifier")
    parser.add_argument("--run-date", required=True, help="Workflow run date (YYYY-MM-DD)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = expand_paths(cfg, run_id=args.run_id, run_date=args.run_date)
    out = cfg["outputs"]

    logger.info("=" * 80)
    logger.info("STARTING UNIFIED PRUNING PIPELINE")
    logger.info("=" * 80)

    con = create_duckdb_connection()

    # -----------------------------------------------------------------------
    # 1. Nomenclature COICOP → nomenclature prunée + table de mapping
    # -----------------------------------------------------------------------
    logger.info("STEP 1: pruning de la nomenclature COICOP")
    notices_raw = con.sql(
        f"SELECT * FROM read_csv('{cfg['nomenclature_raw']}')"
    ).to_df()
    logger.info(f"  → {len(notices_raw)} notices brutes chargées")

    # Le niveau 5 (Poste) est retiré avant l'élagage (incohérences au-delà du niveau 4).
    notices_for_prune = notices_raw.loc[notices_raw["type"] != "Poste"]
    logger.info(f"  → {len(notices_for_prune)} notices après retrait du niveau 'Poste'")

    nomenclature_pruned, mapping_table = prune_linear_hierarchies(notices_for_prune)
    logger.info(
        f"  → {len(nomenclature_pruned)} notices prunées, "
        f"{len(mapping_table)} entrées dans la table de mapping"
    )
    _copy_to_parquet(con, nomenclature_pruned, "nomenclature_pruned", out["nomenclature_pruned"])
    _copy_to_parquet(con, mapping_table, "mapping_table", out["mapping_lvl4"])
    logger.info(f"  ✓ {out['nomenclature_pruned']}")
    logger.info(f"  ✓ {out['mapping_lvl4']}")

    # -----------------------------------------------------------------------
    # 2. Annotations : flux 'train' (KB) et flux 'test' (à coder)
    #    (le libellé est resynchronisé depuis la nomenclature brute, tous niveaux)
    # -----------------------------------------------------------------------
    features = cfg["features"]

    logger.info("STEP 2a: pruning des annotations 'train' (KB)")
    train_pruned = prune_annotations_file(
        con, cfg["annotations_train"], features, mapping_table, notices_raw, logger
    )
    _copy_to_parquet(con, train_pruned, "annotations_train_pruned", out["annotations_train_pruned"])
    logger.info(f"  ✓ {out['annotations_train_pruned']} ({len(train_pruned)} lignes)")

    logger.info("STEP 2b: pruning des annotations 'test' (à coder)")
    test_pruned = prune_annotations_file(
        con, cfg["annotations_test"], features, mapping_table, notices_raw, logger
    )
    _copy_to_parquet(con, test_pruned, "annotations_test_pruned", out["annotations_test_pruned"])
    logger.info(f"  ✓ {out['annotations_test_pruned']} ({len(test_pruned)} lignes)")

    # -----------------------------------------------------------------------
    # 3. Suggester
    # -----------------------------------------------------------------------
    logger.info("STEP 3: pruning du suggester")
    suggester_pruned = load_and_prune_suggester(
        con, cfg["suggester"], cfg["product_col"], mapping_table, notices_raw, logger
    )
    _copy_to_parquet(con, suggester_pruned, "suggester_pruned", out["suggester_pruned"])
    logger.info(f"  ✓ {out['suggester_pruned']} ({len(suggester_pruned)} lignes)")

    logger.info("=" * 80)
    logger.info("UNIFIED PRUNING PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    main()
