def duplicated_suggester(suggester, logger):
    """
    Check whether suggester coicop codes are duplicated or not

    Args:
        suggester: dataframe with suggester data
        logger: logging function
    """

    doublons = suggester.duplicated(subset=["code", "raw_product"], keep=False)
    nb_doublons = len(suggester[doublons])
    no_duplicates = nb_doublons == 0

    if no_duplicates:
        logger.info("Pas de doublons libellé * coicop sur les données du suggester")
    else:
        logger.info(
            f"Présence de {nb_doublons} doublons libellé * coicop sur les données du suggester"
        )


def get_product_with_multiple_codes(
    annotations, config, product_name="raw_product", code_name="code"
):
    """
    Check whether each product has one or multiple coicop codes and list them

    Args:
        annotation: consolidated annotations dataframe
        config: YAML configuration files
    """

    products_with_multiple_codes = annotations.groupby(product_name)[
        code_name
    ].nunique()
    products_with_multiple_codes = products_with_multiple_codes.loc[
        products_with_multiple_codes > 1
    ].index

    # Filtrer les lignes correspondantes
    annotations_with_multiple_codes = annotations.loc[
        annotations[product_name].isin(products_with_multiple_codes)
    ]
    annotations_with_multiple_codes = annotations_with_multiple_codes.sort_values(
        by=product_name
    )
    return annotations_with_multiple_codes
