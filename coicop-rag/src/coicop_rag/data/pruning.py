"""
coicop_linear_pruning.py

Utilities to identify and prune strictly linear hierarchies
in a COICOP-like nomenclature stored as a pandas DataFrame.

A strictly linear hierarchy is defined as a chain where each code
has at most one child, across all generations. In such cases,
child codes are considered equivalent to their most aggregated parent.
"""

import logging
import pandas as pd
from coicop_rag.utils import truncate_code


logger = logging.getLogger(__name__)


def compute_number_of_children(notices_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the number of children for each code and attach it to the DataFrame.

    Parameters
    ----------
    notices_raw : pd.DataFrame
        Input DataFrame containing at least 'code' and 'parent' columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with an added 'nb_children' column.
    """
    children_count = (
        notices_raw
        .groupby("parent")["code"]
        .nunique()
        .rename("nb_children")
    )

    notices_raw = notices_raw.merge(
        children_count,
        left_on="code",
        right_index=True,
        how="left"
    )

    notices_raw["nb_children"] = (
        notices_raw["nb_children"]
        .fillna(0)
        .astype(int)
    )

    return notices_raw


def compute_equivalence_groups(notices_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Identify strictly linear hierarchies and assign an equivalence group id
    to each code belonging to such a hierarchy.

    Parameters
    ----------
    notices_raw : pd.DataFrame
        DataFrame with 'code', 'parent' and 'nb_children' columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with an added 'groupe_equivalent' column (nullable Int64).
    """
    # Codes that have at most one child
    linear_codes = set(
        notices_raw.loc[notices_raw["nb_children"] <= 1, "code"]
    )

    # Parent -> child mapping restricted to linear relations
    linear_relations = notices_raw[
        notices_raw["parent"].isin(linear_codes)
        & notices_raw["code"].isin(linear_codes)
    ]

    next_code = dict(
        zip(linear_relations["parent"], linear_relations["code"])
    )

    # Roots of linear chains
    roots = notices_raw.loc[
        notices_raw["code"].isin(linear_codes)
        & (
            notices_raw["parent"].isna()
            | ~notices_raw["parent"].isin(linear_codes)
        ),
        "code"
    ]

    equivalence_map: dict[str, int] = {}
    group_id = 0

    for root in roots:
        current = root
        lineage = [current]

        while current in next_code:
            current = next_code[current]
            lineage.append(current)

        for code in lineage:
            equivalence_map[code] = group_id

        group_id += 1

    notices_raw["groupe_equivalent"] = (
        notices_raw["code"]
        .map(equivalence_map)
        .astype("Int64")
    )

    return notices_raw


def prune_equivalent_children(
    notices_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove child codes that are strictly equivalent to their parent,
    keeping only the most aggregated code in each equivalence group.

    Also returns a mapping table allowing conversion from any code
    belonging to a linear group to the retained (parent) code.

    Parameters
    ----------
    notices_raw : pd.DataFrame
        DataFrame with a 'groupe_equivalent' column.

    Returns
    -------
    notices_filtered : pd.DataFrame
        Pruned COICOP nomenclature.

    mapping_table : pd.DataFrame
        Table with columns ['code', 'code_parent_equivalent'] allowing
        replacement of removed child codes by their retained parent code.
    """
    initial_count = len(notices_raw)

    notices_raw = notices_raw.copy()
    notices_raw["level"] = notices_raw["code"].str.count(r"\.") + 1

    equivalent_codes = notices_raw[notices_raw["groupe_equivalent"].notna()]
    non_equivalent_codes = notices_raw[notices_raw["groupe_equivalent"].isna()]

    # Retained (most aggregated) code per equivalence group
    retained = (
        equivalent_codes
        .sort_values("level")
        .groupby("groupe_equivalent", as_index=False)
        .first()[["groupe_equivalent", "code"]]
        .rename(columns={"code": "code_parent_equivalent"})
    )

    # Build full mapping: every code in a linear group -> retained parent code
    mapping_table = (
        equivalent_codes[["code", "groupe_equivalent"]]
        .merge(retained, on="groupe_equivalent", how="left")
        .drop(columns="groupe_equivalent")
        .sort_values("code")
        .reset_index(drop=True)
    )

    # Keep only the retained parent codes + all non-equivalent codes
    notices_filtered = pd.concat(
        [
            non_equivalent_codes,
            notices_raw[
                notices_raw["code"].isin(retained["code_parent_equivalent"])
            ],
        ],
        ignore_index=True,
    )

    removed_count = initial_count - len(notices_filtered)

    logger.info(
        "COICOP linear pruning completed: %d codes removed (%d -> %d).",
        removed_count,
        initial_count,
        len(notices_filtered),
    )

    return notices_filtered, mapping_table


def prune_linear_hierarchies(
    notices_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    High-level convenience function that:
    1. Computes number of children per code
    2. Identifies equivalence groups in strictly linear hierarchies
    3. Removes redundant equivalent child codes
    4. Returns a mapping table from removed children to retained parent codes

    Parameters
    ----------
    notices_raw : pd.DataFrame

    Returns
    -------
    notices_filtered : pd.DataFrame
        Pruned COICOP nomenclature.

    mapping_table : pd.DataFrame
        Table allowing conversion from any code belonging to a linear group
        to the retained parent code.
    """
    notices_raw = compute_number_of_children(notices_raw)
    notices_raw = compute_equivalence_groups(notices_raw)

    notices_filtered, mapping_table = prune_equivalent_children(notices_raw)

    return notices_filtered, mapping_table

def prune_annotation_lvl4(
    annotations: pd.DataFrame,
    mapping_table_lvl4: pd.DataFrame,
    notices_raw: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prune children coicop codes using a linear hierarchy mapping.

    The mapping table only works for coicop codes at level 4 max (nomenclature inconsistencies at level 5 +).
    ==> Annotation codes must be truncated at level 4 max first.

    This function replaces child codes belonging to strictly linear 
    hierarchies by their retained parent code, and synchronizes
    the corresponding COICOP label.

    Parameters
    ----------
    annotations : pd.DataFrame
        DataFrame containing at least:
        - 'code' : COICOP code (may be level 5 or 6)
        - 'coicop' : COICOP label associated with the code (before )

    mapping_table_lvl4 : pd.DataFrame
        Mapping table relevant for level4 max, wih columns:
        - 'code'
        - 'code_parent_equivalent'

    notices_raw : pd.DataFrame
        Raw COICOP nomenclature, containing at least:
        - 'code'
        - 'label_fr'

    Returns
    -------
    pd.DataFrame
        A copy of `annotations` where:
        - 'code' has been truncated to level 4 max, and replaced by the parent equivalent code when relevant
        - 'coicop' has been replaced by the parent label
    """
    annotations = annotations.copy()
    

    # ------------------------------------------------------------------
    # 0. Add truncated code column (max = level 4)
    # ------------------------------------------------------------------
    annotations["code_truncate4"] = (
        annotations["code"]
        .apply(lambda x: truncate_code(x, level=4))
    )

    # ------------------------------------------------------------------
    # 1. Build fast lookup structures
    # ------------------------------------------------------------------
    code_mapping = (
        mapping_table_lvl4
        .set_index("code")["code_parent_equivalent"]
    )

    label_mapping = (
        notices_raw  # all levels, no matter
        .set_index("code")["label_fr"]
    )

    # ------------------------------------------------------------------
    # 2. Replace codes using the equivalence mapping
    # ------------------------------------------------------------------

    annotations["code_mapped"] = (
        annotations["code_truncate4"]
        .map(code_mapping)
        .fillna(annotations["code_truncate4"])
    )

    # ------------------------------------------------------------------
    # 3. Replace labels using the mapped codes
    # ------------------------------------------------------------------

    annotations["coicop_mapped"] = (
        annotations["code_mapped"]
        .map(label_mapping)
        .fillna(annotations["coicop"]) # Special BdF codes like "98", "99" ==> inconsistent with notices
    )

    # ------------------------------------------------------------------
    # 4. Finalize output (overwrite original columns)
    # ------------------------------------------------------------------

    annotations["code"] = annotations["code_mapped"]
    annotations["coicop"] = annotations["coicop_mapped"]

    annotations = annotations.drop(
        columns=["code_mapped", "code_truncate4", "coicop_mapped"]
    )

    return annotations



def trunc_and_prune_lvl4(
    df: pd.DataFrame,
    mapping_table_lvl4: pd.DataFrame,
    notices_raw: pd.DataFrame,
    code_name: str,
    coicop_name=None,
) -> pd.DataFrame:
    """
    Prune children coicop codes using a linear hierarchy mapping.

    The mapping table only works for coicop codes at level 4 max (nomenclature inconsistencies at level 5 +).
    ==> df codes must be truncated at level 4 max first.

    This function replaces child codes belonging to strictly linear 
    hierarchies by their retained parent code, and synchronizes
    the corresponding COICOP label.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least:
        - code_name : COICOP code (may be level 5 or 6)
        - coicop_name : COICOP label associated with the code (before )

    mapping_table_lvl4 : pd.DataFrame
        Mapping table relevant for level4 max, wih columns:
        - 'code'
        - 'code_parent_equivalent'

    notices_raw : pd.DataFrame
        Raw COICOP nomenclature, containing at least:
        - 'code'
        - 'label_fr'

    Returns
    -------
    pd.DataFrame
        A copy of `df` where:
        - code_name has been truncated to level 4 max, and replaced by the parent equivalent code when relevant
        - coicop_name has been replaced by the parent label
    """
    df = df.copy()
    

    # ------------------------------------------------------------------
    # 0. Add truncated code column (max = level 4)
    # ------------------------------------------------------------------
    df["code_truncate4"] = (
        df[code_name]
        .apply(lambda x: truncate_code(x, level=4))
    )

    # ------------------------------------------------------------------
    # 1. Build fast lookup structures
    # ------------------------------------------------------------------
    code_mapping = (
        mapping_table_lvl4
        .set_index("code")["code_parent_equivalent"]
    )

    label_mapping = (
        notices_raw  # all levels, no matter
        .set_index("code")["label_fr"]
    )

    # ------------------------------------------------------------------
    # 2. Replace codes using the equivalence mapping
    # ------------------------------------------------------------------

    df["code_mapped"] = (
        df["code_truncate4"]
        .map(code_mapping)
        .fillna(df["code_truncate4"])
    )

    # ------------------------------------------------------------------
    # 3. Replace labels using the mapped codes
    # ------------------------------------------------------------------
    if coicop_name:
        df["coicop_mapped"] = (
            df["code_mapped"]
            .map(label_mapping)
            .fillna(df[coicop_name]) # Special BdF codes like "98", "99" ==> inconsistent with notices
        )

    # ------------------------------------------------------------------
    # 4. Finalize output (overwrite original columns)
    # ------------------------------------------------------------------

    df[f"{code_name}_tpruned"] = df["code_mapped"]
    
    if coicop_name:
        df[f"{coicop_name}_tpruned"] = df["coicop_mapped"]

    df = df.drop(
        columns=["code_mapped", "code_truncate4", "coicop_mapped"],
        errors='ignore'  # if coicop_mapped absent
    )

    return df




def _trunc_and_prune_lvl4(
    code: str,
    mapping_table_lvl4: pd.DataFrame,
) -> str:
    """
    Prune children coicop codes using a linear hierarchy mapping.

    The mapping table only works for coicop codes at level 4 max (nomenclature inconsistencies at level 5 +).
    ==> df codes must be truncated at level 4 max first.

    This function replaces child codes belonging to strictly linear 
    hierarchies by their retained parent code

    mapping_table_lvl4 : pd.DataFrame
        Mapping table relevant for level4 max, wih columns:
        - 'code'
        - 'code_parent_equivalent'


    """

    code_truncate4 = truncate_code(code, level=4)

    code_mapping = (
        mapping_table_lvl4
        .set_index("code")["code_parent_equivalent"]
    )

    code_tpruned = code_mapping.get(code_truncate4)

    if pd.isna(code_tpruned):
        return code_truncate4

    return code_tpruned