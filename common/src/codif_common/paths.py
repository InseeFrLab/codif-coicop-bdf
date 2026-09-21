"""Substitution des variables de run dans les chemins de configuration.

Cette fonction existait en **quatre exemplaires identiques** (`prune_codes`,
`rag_notices`, `rag_annotations`, `classify-regex`) avant d'être rapatriée ici.
"""

from typing import Any


def expand_paths(obj: Any, **kwargs: str) -> Any:
    """Applique récursivement ``str.format(**kwargs)`` à toute chaîne d'une
    structure imbriquée (dict / list).

    Sert à substituer ``{run_id}`` et ``{run_date}`` dans les gabarits de
    chemins des `config.yaml`.

    Attention : la substitution porte sur **toute** chaîne de la config, pas
    seulement les chemins. Une valeur contenant une accolade non destinée au
    formatage lèvera un ``KeyError`` — c'est la raison pour laquelle les noms de
    collections Qdrant, qui viennent d'Argo, sont affectés *après* l'expansion
    et non à travers elle.
    """
    if isinstance(obj, dict):
        return {k: expand_paths(v, **kwargs) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand_paths(v, **kwargs) for v in obj]
    if isinstance(obj, str):
        return obj.format(**kwargs)
    return obj
