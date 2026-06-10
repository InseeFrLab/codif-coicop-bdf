"""
Annotation pruning (subset of coicop_rag.data.pruning).

Truncates annotation codes to level 4 and applies the COICOP linear-hierarchy
mapping table (produced upstream by `0_prunning_coicop.py` in coicop-rag) to
normalize ground-truth codes. We reuse the existing mapping table on S3 rather
than recomputing the linear hierarchies.
"""
import pandas as pd

from coicop_rag_annotations.utils import truncate_code


def prune_annotation_lvl4(
    annotations: pd.DataFrame,
    mapping_table_lvl4: pd.DataFrame,
    notices_raw: pd.DataFrame,
) -> pd.DataFrame:
    """
    Truncate `annotations['code']` to level 4 and replace codes belonging to
    strictly linear hierarchies by their retained parent code, synchronizing the
    `coicop` label.

    Args:
        annotations: Must contain 'code' and 'coicop' columns.
        mapping_table_lvl4: Columns 'code' and 'code_parent_equivalent'.
        notices_raw: Raw COICOP nomenclature with 'code' and 'label_fr'.

    Returns:
        Copy of `annotations` with 'code'/'coicop' overwritten by their pruned values.
    """
    annotations = annotations.copy()

    annotations["code_truncate4"] = annotations["code"].apply(
        lambda x: truncate_code(x, level=4)
    )

    code_mapping = mapping_table_lvl4.set_index("code")["code_parent_equivalent"]
    label_mapping = notices_raw.set_index("code")["label_fr"]

    annotations["code_mapped"] = (
        annotations["code_truncate4"].map(code_mapping).fillna(annotations["code_truncate4"])
    )
    annotations["coicop_mapped"] = (
        annotations["code_mapped"].map(label_mapping).fillna(annotations["coicop"])
    )

    annotations["code"] = annotations["code_mapped"]
    annotations["coicop"] = annotations["coicop_mapped"]

    return annotations.drop(columns=["code_mapped", "code_truncate4", "coicop_mapped"])
