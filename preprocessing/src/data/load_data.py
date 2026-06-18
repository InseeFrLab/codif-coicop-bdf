import logging

import duckdb
import pandas as pd

from src.data.string_cleaning import normalize_text
from src.utils.data_management import concat_path_from_key

logger = logging.getLogger(__name__)


def load_path_data(config):
    """
    Initialize filepath to annotations files
    """
    copain_anno = concat_path_from_key(config, "paths", "copain_annotations")
    additional_annotations = concat_path_from_key(
        config, "paths", "additional_annotations"
    )
    old_anno = concat_path_from_key(config, "paths", "old_annotations")
    app_anno = concat_path_from_key(config, "paths", "app_annotations")
    shops_mapping = concat_path_from_key(config, "paths", "shops_mapping")
    uncodable_products = concat_path_from_key(config, "paths", "uncodable_products")
    return copain_anno, additional_annotations, old_anno, app_anno, shops_mapping, uncodable_products


def load_copain_data(s3_copain_anno, con):
    """
    Load COPAIN data (manual annotations)

    CONSERVÉE COMME TRACE — la source `copain` est exclue de tout le pipeline
    depuis 2026-06. Cette fonction n'est plus appelée par `load_data()` ; pour
    réactiver copain, voir le bloc commenté dans `load_data()`.

    Args:
        s3_copain_anno : path for COPAIN annotations file
        con : connexion for duckdb database

    Return:
        anno_copain: COPAIN annotation dataframe
    """
    logger.info(f"Chargement annotations COPAIN : {s3_copain_anno}/**/*.parquet")
    query_definition = f"SELECT product as raw_product, code, coicop, day FROM read_parquet('{s3_copain_anno}/**/*.parquet', hive_partitioning = true)"

    anno_copain = con.sql(query_definition).to_df()
    anno_copain["source"] = "copain"
    anno_copain["annee"] = anno_copain["day"].dt.year
    return anno_copain


def load_additionnal_anno(s3_add_anno, con):
    """
    Load historical annotations files from S3 files
    """
    logger.info(f"Chargement annotations historiques : {s3_add_anno}")
    additionnal_annotations = con.sql(
        f"""
        FROM read_csv_auto(
            '{s3_add_anno}',
            delim=';',
            encoding='UTF-8'
        )
        SELECT
            -- Calcul du budget :
            -- 1) Nettoie la colonne "Montant de la dépense"
            --    - Supprime les occurrences de '//' (REPLACE)
            --    - Remplace les virgules par des points pour normaliser le séparateur décimal
            -- 2) Extrait uniquement la partie numérique valide en début de chaîne (regexp_extract)
            -- 3) Convertit les chaînes vides en NULL (NULLIF)
            -- 4) Cast en FLOAT
            -- 5) Agrège via SUM pour obtenir le total par groupe
            SUM(
                CAST(
                    NULLIF(
                        regexp_extract(
                            REPLACE(
                                REPLACE("prix", '//', ''),
                                ',', '.'
                            ),
                            '^[0-9]+(\\.[0-9]+)?',
                            0
                        ),
                        ''
                    ) AS FLOAT
                )
            ) AS budget,
            "produit" AS raw_product,
            "ecoicopv2" AS code,
            "libelle_ecoicopv2" AS coicop,
            "magasin" AS shop,
            "fichier_source" AS source
        GROUP BY
            "produit",
            "ecoicopv2",
            "libelle_ecoicopv2",
            "magasin",
            "fichier_source"
        """
    ).to_df()
    additionnal_annotations["annee"] = 2024

    return additionnal_annotations


def load_old_annotations(s3_old_anno, con) -> pd.DataFrame:
    logger.info(f"Chargement annotations BdF 2017 : {s3_old_anno}")
    old_annotations = con.sql(
        f"""
        FROM read_csv_auto(
            '{s3_old_anno}',
            delim=';',
            encoding='cp1252',
            types={{'code_mag_bdf2026': 'VARCHAR'}}
        )
        SELECT
            SUM(
                    CAST(
                        NULLIF(
                            regexp_extract(
                                REPLACE(
                                    REPLACE("prix", '//', ''),
                                    ',', '.'
                                ),
                                '^[0-9]+(\\.[0-9]+)?',
                                0
                            ),
                            ''
                        ) AS FLOAT
                    )
                ) AS budget,
                "produit" AS raw_product,
                "code_ecoicopv2" AS code,
                "libelle_ecoicopv2" AS coicop,
                "code_mag_bdf2026" AS shop_type_code,
                "lib_mag_bdf2026" AS shop_type_name
        GROUP BY
            "produit",
            "code_ecoicopv2",
            "libelle_ecoicopv2",
            "code_mag_bdf2026",
            "lib_mag_bdf2026"
    """).to_df()

    old_annotations["shop_type_code"] = pd.to_numeric(old_annotations['shop_type_code'], errors='coerce').astype('Int64')
    old_annotations["source"] = "bdf_2017"
    old_annotations["annee"] = 2017
    return old_annotations


def load_shops_mapping(s3_shop_types, con) -> pd.DataFrame:
    """
    Load shop types since annotated receipts file

    """
    # on récupère la nomenclature des enseignes qui ont été déjà codées dans l'input des tickets de caisse
    logger.info(f"Chargement mapping enseignes : {s3_shop_types}")
    shops_mapping = con.sql(
        f"FROM read_csv_auto('{s3_shop_types}') "
        'SELECT DISTINCT shop, code_mag as shop_type_code, "Nomen_mag" as shop_type_name'
    ).to_df()

    # shops_mapping = shops_mapping[~shops_mapping["shop"].isin(["//", None])]

    # Normalise la clé de jointure comme les sites de merge, PUIS déduplique dessus,
    # afin que la table de droite du merge soit garantie unique sur `shop`.
    # Sans cela, une enseigne associée à plusieurs types (code_mag/Nomen_mag) — ou
    # plusieurs orthographes qui se confondent après normalisation — multiplie les
    # lignes du fichier d'entrée lors du merge.
    shops_mapping["shop"] = normalize_text(shops_mapping["shop"])
    before = len(shops_mapping)
    shops_mapping = shops_mapping.drop_duplicates(subset="shop", keep="first")
    dropped = before - len(shops_mapping)
    if dropped:
        logger.info(
            f"shops_mapping : {dropped} lignes d'enseignes ambiguës "
            "(même shop, types multiples) supprimées — on garde la 1ère occurrence"
        )
    return shops_mapping


def load_input_file(path: str, con) -> pd.DataFrame:
    """
    Load an arbitrary CSV or parquet file (local or S3) for prediction mode.
    Returns a DataFrame guaranteed to have a 'raw_product' column.
    """
    ext = path.split("?")[0].lower()
    if ext.endswith(".parquet"):
        df = con.sql(f"SELECT * FROM read_parquet('{path}')").to_df()
    else:
        df = con.sql(f"SELECT * FROM read_csv_auto('{path}', delim=';')").to_df()

    return df


def load_data(config, con):
    """
    Load all annotations files
    """
    # -----------------------------------------------------------------------
    # READING PATH ANNOTATIONS
    # -----------------------------------------------------------------------

    (
        S3_COPAIN_ANNOTATIONS,  # inutilisé — copain exclu du pipeline (cf. bloc ci-dessous)
        S3_ADDITIONAL_ANNOTATIONS,
        S3_OLD_ANNOTATIONS,
        S3_APP_ANNOTATIONS,
        S3_SHOPS_MAPPING,
        S3_UNCODABLE_PRODUCTS
    ) = load_path_data(config)

    # -----------------------------------------------------------------------
    # READING COPAIN ANNOTATIONS
    # -----------------------------------------------------------------------
    # COPAIN EXCLU DU PIPELINE (2026-06) : la source `copain` n'est plus chargée
    # ni intégrée au dataset annoté consolidé. On évite ainsi de lire le gros glob
    # S3 `output-annotation/**/*.parquet` pour rien.
    # Pour réactiver copain : décommenter l'appel ci-dessous, le réintégrer dans le
    # tuple de retour de load_data(), dans l'unpacking de main(), dans la signature
    # de build_annotations() et dans SOURCES_2024 (main.py).
    #
    # annotations_copain = load_copain_data(S3_COPAIN_ANNOTATIONS, con)

    # -----------------------------------------------------------------------
    # READING HISTORICAL ANNOTATIONS
    # -----------------------------------------------------------------------

    annotations_hors_copain = load_additionnal_anno(S3_ADDITIONAL_ANNOTATIONS, con)

    # -----------------------------------------------------------------------
    # READING OLD ANNOTATIONS (BdF 2017)
    # -----------------------------------------------------------------------

    annotations_old = load_old_annotations(S3_OLD_ANNOTATIONS, con)


    # -----------------------------------------------------------------------
    # ADD SHOP TYPE IN ANNOTATIONS DATAFRAMES
    # -----------------------------------------------------------------------

    shops_mapping = load_shops_mapping(S3_SHOPS_MAPPING, con)

    # -----------------------------------------------------------------------
    # READING SUGGESTER LIST
    # -----------------------------------------------------------------------

    logger.info(f"Chargement suggester (liste produits) : {S3_APP_ANNOTATIONS}")
    suggester = con.sql(f"""
        SELECT
            DISTINCT code, product as raw_product, coicop
        FROM read_csv_auto('{S3_APP_ANNOTATIONS}')
        """).to_df()

    suggester["source"] = "suggester"
    suggester["annee"] = 2017

    # Reading uncodable products
    logger.info(f"Chargement produits non codables : {S3_UNCODABLE_PRODUCTS}")
    uncodable_products = con.sql(f"""
        SELECT
            DISTINCT produit AS raw_product
        FROM read_csv_auto('{S3_UNCODABLE_PRODUCTS}')
        """).to_df()["raw_product"].tolist()

    # copain retiré du tuple de retour (exclu du pipeline — cf. bloc COPAIN ci-dessus).
    return annotations_hors_copain, suggester, shops_mapping, uncodable_products, annotations_old
