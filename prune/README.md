# prune — pruning unifié COICOP

Étape unique du pipeline qui applique le même algorithme de pruning
(troncature niveau 4 + élagage des hiérarchies linéaires → code canonique
`code_parent_equivalent`) et produit **tous** les artefacts prunés, que l'aval
se contente de lire.

Remplace les anciennes étapes `prune-coicop` / `prune-annotations` et le pruning
qui était fait à la volée dans les builds de vector DB (coicop-rag et
coicop-rag-annotations).

## Sorties (par run, sous `…/{run_date}/{run_id}/prune/`)

| Fichier | Contenu | Consommé par |
|---|---|---|
| `nomenclature_pruned.parquet` | nomenclature prunée (niveaux 1 à 4) | `create-vector-db` (coicop-rag) |
| `mapping_lvl4.parquet` | table `code → code_parent_equivalent` | scoring `run-rag` |
| `annotations_train_pruned.parquet` | KB d'annotations prunées | `create-vector-db-annotations` |
| `annotations_test_pruned.parquet` | jeu à coder pruné | `run-rag` et `run-rag-annotations` |
| `suggester_pruned.parquet` | suggester pruné | `create-vector-db-annotations` |

## Usage

```bash
uv sync --locked
uv run scripts/main.py --config config/config.yaml --run-id <ID> --run-date <YYYY-MM-DD>
```

Le dépôt est un workspace `uv` : le lock est à la racine, et cette commande n'installe
que les dépendances de ce module (voir « Environnement Python » dans le README racine).

Entrées : la nomenclature COICOP brute + les sorties de `codif-regex`
(`raw_train_without_regex` = KB, `raw_test_without_regex` = à coder) + le
suggester (CSV). Voir `config/config.yaml`.
