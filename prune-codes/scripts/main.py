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

from prune_codes.pruning import prune_annotation_lvl4, prune_linear_hierarchies
from prune_codes.utils import create_duckdb_connection, expand_paths


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
    logger.info(f"  Chargement suggester : {sug_cfg['s3_path']}")
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
    parser.add_argument(
        "--only",
        choices=["all", "nomenclature", "kb"],
        default="all",
        help=(
            "Périmètre du pruning. 'all' (défaut) : tout, pour le pipeline de "
            "classification. 'nomenclature' : l'étape 1 seule (nomenclature + mapping), "
            "pour le pipeline d'indexation des notices. 'kb' : étape 1 + KB + suggester, "
            "sans le jeu à coder — pour le pipeline d'indexation des annotations. "
            "Toutes les variantes exécutent l'étape 1 : le mapping et la nomenclature "
            "brute qu'elle calcule en mémoire sont consommés par les étapes suivantes, "
            "qui ne peuvent donc pas s'en passer."
        ),
    )
    parser.add_argument(
        "--annotations-train",
        default=None,
        help=(
            "Remplace la source de la KB (défaut : la clé `annotations_train` de la "
            "config, soit la sortie de classify-regex). Le pipeline d'indexation des "
            "annotations y met `build-datasets/annotations_full.parquet` : la KB, ce "
            "sont les produits déjà annotés, et elle n'a pas à être filtrée par la regex."
        ),
    )
    parser.add_argument(
        "--suggester-path",
        default=None,
        help=(
            "Remplace la source du suggester (défaut : la clé `suggester.s3_path`). "
            "Le pipeline d'indexation y met `build-datasets/suggester.parquet`, "
            "c'est-à-dire le suggester préprocessé plutôt que le CSV brut."
        ),
    )
    parser.add_argument(
        "--suggester-product-col",
        default=None,
        help="Colonne produit de la source suggester (défaut : `suggester.source_product_col`).",
    )
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
    logger.info(f"  Chargement nomenclature : {cfg['nomenclature_raw']}")
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

    if args.only == "nomenclature":
        # Sortie anticipée : les étapes 2 et 3 liraient des sorties classify-regex
        # qui n'existent pas dans un run d'indexation des notices. `cfg["features"]`
        # et `cfg["suggester"]` ne sont donc pas lus non plus.
        logger.info("=" * 80)
        logger.info("PRUNING NOMENCLATURE TERMINÉ (--only nomenclature : étapes 2 et 3 ignorées)")
        logger.info("=" * 80)
        return

    # -----------------------------------------------------------------------
    # 2. Annotations : flux 'train' (KB) et flux 'test' (à coder)
    #    (le libellé est resynchronisé depuis la nomenclature brute, tous niveaux)
    # -----------------------------------------------------------------------
    features = cfg["features"]

    kb_source = args.annotations_train or cfg["annotations_train"]
    logger.info(f"STEP 2a: pruning de la KB (kb_data) depuis {kb_source}")
    kb_data_pruned = prune_annotations_file(
        con, kb_source, features, mapping_table, notices_raw, logger
    )
    _copy_to_parquet(con, kb_data_pruned, "annotations_train_pruned", out["annotations_train_pruned"])
    logger.info(f"  ✓ {out['annotations_train_pruned']} ({len(kb_data_pruned)} lignes)")

    if args.only == "kb":
        # Le jeu à coder n'a pas de sens ici : ce run construit une base
        # d'exemples annotés, pas une codification. `cfg["annotations_test"]`
        # pointe une sortie classify-regex que ce pipeline ne produit pas.
        logger.info("STEP 2b ignorée (--only kb : pas de jeu à coder)")
    else:
        logger.info("STEP 2b: pruning des observations (à coder)")
        observations_pruned = prune_annotations_file(
            con, cfg["annotations_test"], features, mapping_table, notices_raw, logger
        )
        _copy_to_parquet(con, observations_pruned, "annotations_test_pruned", out["annotations_test_pruned"])
        logger.info(f"  ✓ {out['annotations_test_pruned']} ({len(observations_pruned)} lignes)")

    # -----------------------------------------------------------------------
    # 3. Suggester
    # -----------------------------------------------------------------------
    logger.info("STEP 3: pruning du suggester")
    sug_cfg = dict(cfg["suggester"])
    if args.suggester_path:
        sug_cfg["s3_path"] = args.suggester_path
    if args.suggester_product_col:
        sug_cfg["source_product_col"] = args.suggester_product_col
    suggester_pruned = load_and_prune_suggester(
        con, sug_cfg, cfg["product_col"], mapping_table, notices_raw, logger
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
