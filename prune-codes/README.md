# prune-codes — pruning unifié COICOP

Étape unique qui applique le même algorithme de pruning (troncature niveau 4 +
élagage des hiérarchies linéaires → code canonique `code_parent_equivalent`) et
produit **tous** les artefacts prunés, que l'aval se contente de lire.

Remplace les anciennes étapes `prune-coicop` / `prune-annotations` et le pruning
qui était fait à la volée dans les builds de vector DB (rag-notices et
rag-annotations).

Elle apparaît dans les **trois** pipelines Argo, avec un périmètre différent à
chaque fois (voir `--only` ci-dessous) :

| Pipeline | Invocation | Ce qui est produit |
|---|---|---|
| `argo/codif-pipeline.yaml` | (défaut, `--only all`) | tout |
| `argo/index-notices-pipeline.yaml` | `--only nomenclature` | nomenclature + mapping |
| `argo/index-annotations-pipeline.yaml` | `--only kb` + surcharges de sources | nomenclature, mapping, KB, suggester |

## Sorties (par run, sous `…/{run_date}/{run_id}/prune-codes/`)

| Fichier | Contenu | Produit par `--only` | Consommé par |
|---|---|---|---|
| `nomenclature_pruned.parquet` | nomenclature prunée (niveaux 1 à 4) | `all`, `nomenclature`, `kb` | `index-notices` (rag-notices) |
| `mapping_lvl4.parquet` | table `code → code_parent_equivalent` | `all`, `nomenclature`, `kb` | scoring `classify-rag-notices`, normalisation des codes dans `reconcile-llm` |
| `annotations_train_pruned.parquet` | KB d'annotations prunées | `all`, `kb` | `index-annotations` |
| `annotations_test_pruned.parquet` | jeu à coder pruné | `all` | `classify-rag-notices` et `classify-rag-annotations` |
| `suggester_pruned.parquet` | suggester pruné | `all`, `kb` | `index-annotations` |

## Usage

```bash
uv sync --locked
uv run scripts/main.py --config config/config.yaml --run-id <ID> --run-date <YYYY-MM-DD>
```

Le dépôt est un workspace `uv` : le lock est à la racine, et cette commande n'installe
que les dépendances de ce module (voir « Environnement Python » dans le README racine).

Entrées par défaut : la nomenclature COICOP brute + les sorties de `classify-regex`
(`raw_train_without_regex` = la KB, `raw_test_without_regex` = le jeu à coder — les
suffixes `train`/`test` sont hérités du split supprimé) + le suggester (CSV).
Voir `config/config.yaml`.

## `--only {all,nomenclature,kb}`

Périmètre du pruning. Défaut `all` — comportement inchangé, celui du pipeline de
classification.

- **`nomenclature`** — l'étape 1 seule (nomenclature prunée + mapping), puis sortie
  anticipée. C'est ce dont a besoin `argo/index-notices-pipeline.yaml` : les étapes 2
  et 3 liraient des sorties `classify-regex` qui n'existent pas dans un run
  d'indexation autonome. `features` et `suggester` ne sont pas lus dans la config.
- **`kb`** — étape 1 + KB + suggester, **sans le jeu à coder**. C'est ce dont a besoin
  `argo/index-annotations-pipeline.yaml` : ce run construit une base d'exemples
  annotés, pas une codification.

Il n'existe **volontairement pas** d'option `annotations` ou `suggester` seules : les
étapes 2 et 3 consomment le mapping et la nomenclature brute calculés en mémoire par
l'étape 1. Toutes les variantes exécutent donc l'étape 1 — elles ne peuvent pas s'en
passer.

### Surcharge des sources

Trois options permettent au pipeline d'indexation des annotations de brancher le
pruning sur `build-datasets` plutôt que sur `classify-regex` / le CSV brut :

| Option | Défaut (config) | Ce qu'y met `index-annotations-pipeline.yaml` |
|---|---|---|
| `--annotations-train` | `annotations_train` (sortie `classify-regex`) | `build-datasets/annotations_full.parquet` |
| `--suggester-path` | `suggester.s3_path` (CSV brut) | `build-datasets/suggester.parquet` (préprocessé, porte déjà `l_pr_product`) |
| `--suggester-product-col` | `suggester.source_product_col` | `l_pr_product` |

La KB, ce sont les **produits déjà annotés** : elle n'a pas à être filtrée par la regex,
d'où le contournement de `classify-regex`.

Exemple, tel que lancé par le pipeline d'indexation des annotations :

```bash
BD="s3://projet-budget-famille/data/workflow_runs/<date>/<run_id>/build-datasets"
uv run scripts/main.py --config config/config.yaml \
  --run-id <ID> --run-date <YYYY-MM-DD> \
  --only kb \
  --annotations-train "$BD/annotations_full.parquet" \
  --suggester-path "$BD/suggester.parquet" \
  --suggester-product-col l_pr_product
```
