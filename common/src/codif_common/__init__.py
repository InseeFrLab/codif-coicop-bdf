"""Socle partagé du pipeline de codification COICOP.

Regroupe ce qui existait en plusieurs exemplaires recopiés entre modules. Les
symboles les plus utilisés sont ré-exportés ici pour que l'import courant tienne
sur une ligne ; les autres s'importent depuis leur sous-module
(``codif_common.s3``, ``codif_common.vector_index``).

Ce paquet ne dépend d'aucun autre membre du workspace, et ne doit jamais en
dépendre : c'est ce qui garantit qu'aucun cycle ne peut apparaître.
"""

from codif_common.codes import get_parents, truncate_code
from codif_common.paths import expand_paths

__all__ = ["expand_paths", "get_parents", "truncate_code"]
