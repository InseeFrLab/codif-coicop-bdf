# `common/` — socle partagé

Regroupe le code qui existait en plusieurs exemplaires recopiés entre modules.
Paquet importable : **`codif_common`**.

```python
from codif_common import expand_paths, truncate_code, get_parents
from codif_common.s3 import connect_env, connect_secret
from codif_common.vector_index import build_collection_name, validate_collection
```

## Contenu

| Module | Rôle | Remplaçait |
|---|---|---|
| `paths.py` | `expand_paths` — substitue `{run_id}`/`{run_date}` dans les configs | 4 copies |
| `codes.py` | `truncate_code`, `get_parents` | 4 et 2 copies |
| `s3.py` | connexions DuckDB configurées pour S3 | 5 copies, sur 2 des 4 dialectes |
| `vector_index.py` | nommage, manifeste et validation des collections Qdrant | 2 copies |

## Règle d'or

**Ce module ne dépend d'aucun autre membre du workspace.** C'est ce qui rend
tout cycle impossible : n'importe quel module peut en dépendre sans risque.

Corollaire : `is_answer` et `ABSTENTION_SENTINELS` restent dans
`prune_codes.utils`, où ils vivaient déjà et d'où `report` et `reconcile-sirus`
les importent. Les déplacer ici obligerait `prune-codes` à dépendre de
`codif_common`, ce qui est possible, mais sans bénéfice qui le justifie
aujourd'hui.

## Ce qui n'y est délibérément pas

**Deux des quatre dialectes de connexion S3.** Voir l'en-tête de `s3.py` : le
dialecte `SET s3_*` force le jeton de session à vide, celui à `SCOPE` fait
cohabiter deux secrets dans `classify-ttc`. Les unifier changerait
l'authentification effective, ce qui ne se vérifie pas hors du cluster.

**Le nettoyage de texte**, dupliqué entre `build-datasets` et `classify-ttc`.
Les deux traitent les accents dans un ordre différent : l'un compare les mots
vides à des mots accentués, l'autre à des mots désaccentués. Les fusionner
changerait les prédictions.

**`setup_logging`**, dont les trois copies divergent sur trois points de
comportement — dont le `mkdir("logs")` sans lequel `rag-notices` ne démarre pas.

Ces trois exclusions ne sont pas des oublis : ce sont les endroits où
dédupliquer reviendrait à modifier le comportement sans filet.
