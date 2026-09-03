"""Utilitaires de `prune-codes`.

Les fonctions génériques (chemins, codes, connexion S3) vivent dans
`codif_common` et sont ré-exportées ici : les imports existants continuent de
fonctionner.

`ABSTENTION_SENTINELS` et `is_answer` restent ICI, et c'est délibéré — voir le
commentaire ci-dessous. Les déplacer dans `codif_common` obligerait
`prune-codes` à en dépendre pour du vocabulaire qui lui appartient, sans
bénéfice.
"""

from typing import Any

from codif_common.codes import get_parents, truncate_code
from codif_common.paths import expand_paths
from codif_common.s3 import connect_env as create_duckdb_connection


# Un classifieur peut refuser de coder. Le refus arrive dans les données sous
# plusieurs formes selon la brique : NULL, chaîne vide, ou sentinelle textuelle
# (`llm_code` vaut "N/A" quand l'arbitrage LLM échoue). Toutes comptent comme
# abstention, pas comme code.
#
# Défini ici plutôt que dans `report/` : c'est du vocabulaire de code COICOP,
# et `report/` n'est pas installable comme dépendance (il tire jupyter,
# matplotlib, seaborn). `report.coicop_metrics` et le module `reconcile-sirus/` importent
# tous les deux depuis ici, pour qu'un ajout de sentinelle profite aux deux.
ABSTENTION_SENTINELS = frozenset(
    {"", "-", "?", "n/a", "na", "nan", "nat", "none", "null"}
)


def is_answer(value: Any) -> bool:
    """True si ``value`` porte un code, False si c'est une abstention.

    Un refus de coder (NULL, chaîne vide, ``"N/A"`` …) n'est pas un code faux :
    c'est l'absence de réponse. Les deux comptent comme erreur dans les mesures
    d'accuracy, mais les distinguer permet de séparer « coder faux » de « ne pas
    coder » — deux défauts qui n'appellent pas la même correction.
    """
    if value is None:
        return False
    if isinstance(value, float) and value != value:  # NaN, sans dépendre de numpy
        return False
    return str(value).strip().lower() not in ABSTENTION_SENTINELS
