import pandas as pd


def aggregate_budget(
    annotations, group_columns, method_column="source", budget_column="budget"
):
    """
    Calculate total budget for each key [product, code, coicop, shop, shop_type_name]
    """
    if group_columns is None:
        group_columns = ["id", "raw_product", "l_pr_product", "s_pr_product", "annee", "source", "code", "coicop", "shop", "shop_type_name"]

    annotations["budget"] = annotations["budget"].fillna(0)
    annotations = annotations.groupby(group_columns, as_index=False, dropna=False).agg(
        budget=(budget_column, "sum"),
        n_obs=(budget_column, "size")
    )
    annotations["budget"] = annotations["budget"].replace(0, pd.NA)
    return annotations


def statistics_annotations_data(con, annotations):
    """
    Create some statistics indicators on annotations data
    """
    con.register("annotations", annotations)
    res = con.sql("""
        SELECT 
            COUNT(*) AS total_rows,
            COUNT(*) FILTER (WHERE budget IS NULL OR budget = 0) AS missing_count,
            ROUND(100.0 * COUNT(*) FILTER (WHERE budget IS NULL OR budget = 0)  / COUNT(*), 2) AS missing_percent
        FROM annotations
        WHERE source <> 'suggester'
    """).to_df()
    return res
