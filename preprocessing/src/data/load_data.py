import duckdb
import pandas as pd
from src.utils.data_management import concat_path_from_key


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

    Args:
        s3_copain_anno : path for COPAIN annotations file
        con : connexion for duckdb database

    Return:
        anno_copain: COPAIN annotation dataframe
    """
    query_definition = f"SELECT product as raw_product, code, coicop, day FROM read_parquet('{s3_copain_anno}/**/*.parquet', hive_partitioning = true)"

    anno_copain = con.sql(query_definition).to_df()
    anno_copain["source"] = "copain"
    anno_copain["annee"] = anno_copain["day"].dt.year
    return anno_copain


def load_additionnal_anno(s3_add_anno, con):
    """
    Load historical annotations files from S3 files
    """
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
    shops_mapping = con.sql(
        f"FROM read_csv_auto('{s3_shop_types}') "
        'SELECT DISTINCT shop, code_mag as shop_type_code, "Nomen_mag" as shop_type_name'
    ).to_df()

    # shops_mapping = shops_mapping[~shops_mapping["shop"].isin(["//", None])]
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
        S3_COPAIN_ANNOTATIONS,
        S3_ADDITIONAL_ANNOTATIONS,
        S3_OLD_ANNOTATIONS,
        S3_APP_ANNOTATIONS,
        S3_SHOPS_MAPPING,
        S3_UNCODABLE_PRODUCTS
    ) = load_path_data(config)

    # -----------------------------------------------------------------------
    # READING COPAIN ANNOTATIONS
    # -----------------------------------------------------------------------

    # TODO : récupérer le nom des magasins

    annotations_copain = load_copain_data(S3_COPAIN_ANNOTATIONS, con)

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

    suggester = con.sql(f"""
        SELECT
            DISTINCT code, product as raw_product, coicop
        FROM read_csv_auto('{S3_APP_ANNOTATIONS}')
        """).to_df()

    suggester["source"] = "suggester"
    suggester["annee"] = 2017

    # Reading uncodable products
    uncodable_products = con.sql(f"""
        SELECT
            DISTINCT produit AS raw_product
        FROM read_csv_auto('{S3_UNCODABLE_PRODUCTS}')
        """).to_df()["raw_product"].tolist()

    return annotations_copain, annotations_hors_copain, suggester, shops_mapping, uncodable_products, annotations_old
