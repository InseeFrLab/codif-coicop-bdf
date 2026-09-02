# coicop-rag-annotations

RAG de codification COICOP **basé sur des exemples annotés** (descriptions de
produits déjà codifiées), par opposition à [`coicop-rag`](../rag-notices/) qui
récupère des **notices** de la nomenclature.

Idée : pour coder un produit, on récupère dans une base vectorielle les produits
*déjà annotés* les plus proches sémantiquement, et on fournit ces couples
`(description → code)` comme exemples few-shot à un LLM.

## Architecture

Les deux scripts vivent dans **deux pipelines Argo distincts** : l'indexation est
sortie du pipeline de classification.

```
argo/index-annotations-pipeline.yaml
  build-datasets ──→ prune-codes (--only kb) ──→ index-annotations
                                                  (0_build_annotation_vector_db.py)

argo/codif-pipeline.yaml
  … ──→ classify-rag-annotations  (1_run_rag.py)
```

```
0_build_annotation_vector_db.py        1_run_rag.py
   lit la KB déjà prunée                   charge le test set déjà pruné
   (prune-codes/annotations_train_pruned)        (prune-codes/annotations_test_pruned)
   (+ suggester déjà pruné) → index  ──►    embed + retrieve (Qdrant)
   embed → Qdrant                          prompt few-shot → LLM → parse → éval
```

Pourquoi séparés : embedder toute la KB n'a pas à être repayé à chaque codification, et
tant que la collection portait un nom fixe partagé, une réindexation détruisait la base
que lisait un run concurrent. Même organisation que l'entraînement de `classify-ttc` et
de `reconcile-sirus` : l'artefact coûteux est produit à part, son identifiant est
recopié dans `argo/params.yaml` (clé `classify-rag-annotations-collection`). Le log de
fin d'`index-annotations` affiche la ligne exacte à recopier.

**Ce qu'indexe le pipeline d'indexation** : les *produits déjà annotés*, c'est-à-dire
`annotations_full` + `suggester` au sens de `build-datasets`. Rien d'autre. Il n'y a
donc **aucun paramètre `input_file`** : ce paramètre désigne les produits *à classer*,
ce qui n'a rien à voir avec la constitution d'une base d'exemples. Et `classify-regex`
n'est pas dans la chaîne : la KB n'a pas à être filtrée des produits que la regex sait
coder.

**`kb-scope`** (paramètre Argo, `--kb-scope` côté script) :

| Valeur | KB | Statut |
|---|---|---|
| `full` (défaut) | `build-datasets/annotations_full.parquet` — tous les produits annotés | régime cible |
| `train` | `build-datasets/raw_train.parquet` — l'ancien split | **transitoire, destiné à disparaître** |

`train` n'existait que faute de jeu de test indépendant ; les nouveaux produits annotés
en fournissent un naturellement. `kb-scope` sélectionne aussi le profil de sources
exclues (`annotations.exclude_sources` pour `train`, `exclude_sources_prod` pour `full`
— listes identiques aujourd'hui, la distinction ne survit que le temps de la
transition).

En mode évaluation, `1_run_rag.py` **avertit** (sans bloquer) si la collection a été
bâtie en `kb-scope=full` : une telle KB contient tous les produits annotés, donc
potentiellement les réponses attendues. L'avertissement plutôt que le refus, parce que
ce cas n'est plus la norme.

Le pruning (troncature niveau 4 + mapping) est centralisé dans le module
[`prune-codes/`](../prune-codes/) ; ce module **lit** les artefacts prunés et ne prune plus lui-même.

- `scripts/0_build_annotation_vector_db.py` — lit la KB d'annotations **déjà prunée**
  (`prune-codes/annotations_train_pruned.parquet`), ajoute optionnellement le **suggester**
  **déjà pruné** (`prune-codes/suggester_pruned.parquet`), embed **toutes** les descriptions
  et les charge dans Qdrant. Filtrage des sources via `exclude_sources` /
  `exclude_sources_prod` selon `--kb-scope` ; échantillonnage optionnel de la KB via
  `--sample-size` (appliqué **après** la fusion avec le suggester, pour que le total de
  points indexés vaille exactement `sample-size`). Écrit enfin un **manifeste** JSON plat
  sous `s3://projet-budget-famille/data/vector_db_manifests/{collection_name}.json`
  (`qdrant.manifests_root`) : `kb_scope`, modèle d'embedding et dimension, `sample_size`,
  nombre de points **compté auprès de Qdrant**, run source et SHA git.
- `scripts/1_run_rag.py` — codifie l'input (`prune-codes/annotations_test_pruned.parquet`) et
  exporte les prédictions. En mode **évaluation** (`--skip-eval false`), calcule en plus
  accuracy + recall de retrieval par niveau COICOP (MLflow + `report.txt`) et filtre le
  test set par `include_sources`. En **production** (défaut), prédictions seulement, pas
  de filtre source. Exige `--collection-name` et **valide la collection avant MLflow**
  (existence, dimension, modèle d'embedding, collection non vide), en relisant le
  manifeste écrit par le constructeur.
- `prompts/annotation_rag.md` — template de prompt (sections `<<<SYSTEM>>>` /
  `<<<USER>>>`). Source de vérité locale ; peut être migré vers Langfuse.
- `src/rag_annotations/` — utilitaires (embeddings, génération LLM
  concurrente, parsing, évaluation, chargement du prompt).

### Représentation texte enrichie par le lieu d'achat

Le code COICOP d'un produit dépend du **lieu d'achat** (« café » en supermarché ≠ « café »
au comptoir d'un bar). On indexe donc une représentation canonique unique
(`build_location_text`) qui combine produit + magasin + type de magasin quand on en
dispose — par ex. `café - magasin ou lieu d'achat : Super U (type : supermarché)`.

Cette **même** représentation sert partout : texte embeddé dans Qdrant, texte de la
requête (test), et exemples few-shot du prompt → l'espace d'embedding requête/index est
cohérent et le LLM voit le lieu d'achat dans les exemples. Quand l'info manque (ex. le
suggester), on retombe sur le seul libellé produit.

## Configuration

Tout est dans `config/config.yaml`. Paramètres clés :

| Clé | Rôle |
|---|---|
| `annotations.s3_path` | KB **prunée** à indexer (`prune-codes/annotations_train_pruned.parquet`) |
| `annotations.product_col` | colonne texte à embedder (`l_pr_product`) |
| `annotations.exclude_sources` / `exclude_sources_prod` | sources exclues de l'index, respectivement pour `--kb-scope train` et `full` |
| `suggester.enabled` | ajouter les exemples du suggester à l'index |
| `suggester.s3_path_pruned` | suggester déjà pruné (`prune-codes/suggester_pruned.parquet`) |
| `qdrant.collection_base` | **racine** du nom de collection (`coicop_annotations`) — voir ci-dessous |
| `qdrant.manifests_root` | où sont écrits/relus les manifestes de collection |
| `retrieval.size` | nombre d'exemples annotés récupérés par produit |
| `llm.use_langfuse` | `false` → prompt local ; `true` → Langfuse |
| `data.s3_path_input` | produits à codifier (test labellisé en éval, données prod sinon) |
| `data.s3_path_predictions` | sortie : prédictions du RAG |
| `eval.levels` | profondeur d'évaluation (niveaux 1..levels) |

### Nommage des collections Qdrant

Il n'y a plus de collection au nom fixe (`coicop_annotations_without_copain_2017` a
disparu, tout comme le suffixe `_test` que le constructeur et le consommateur
s'accordaient par convention). Chaque build compose un nom **unique** :

```
{qdrant.collection_base}__{kb_scope}__{run_date}__{run_id}[__sample{N}]
# ex. coicop_annotations__full__2026-09-02__index-annotations-b3x9q
```

Le `kb_scope` est dans le **nom**, pas seulement dans le manifeste : c'est le champ
qu'un humain confond le plus en recopiant, et interroger une KB `full` depuis un run
d'évaluation gonfle l'accuracy sans lever d'erreur. Le suffixe `__sampleN` est visible
pour la même raison : sans lui, un index jouet de 100 points et un index complet
portent des noms de même forme.

La clé de config est `qdrant.collection_base`, et surtout **pas** `collection_name` :
ce fichier est lu par le constructeur *et* par le consommateur, et une clé
`collection_name` inciterait le consommateur à retomber en silence sur une collection
périmée. Le nom complet lui est donc passé en argument, obligatoirement.

Le nom seul ne peut pas porter tout ce qui détermine la validité d'un index (modèle
d'embedding, `kb_scope`, run source) : deux collections de même dimension bâties
différemment sont indistinguables et interroger la mauvaise ne lève aucune erreur — le
RAG rend juste de moins bons résultats. D'où le manifeste, écrit à côté de la collection
et relu au démarrage du consommateur.

## Exécution (Onyxia, secrets via Vault)

Variables d'environnement attendues (mêmes que `coicop-rag`, injectées par Vault) :
`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `LLMLAB_URL`, `LLMLAB_API_KEY`,
`QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_API_PORT`, `MLFLOW_TRACKING_URI`,
`MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD` (+ `LANGFUSE_*` si
`use_langfuse: true`).

```bash
cd rag-annotations
uv sync
mkdir -p logs

# 1. Construire la base vectorielle — RUN D'INDEXATION, identité propre.
#    index = KB (annotations_full par défaut) + suggester
uv run scripts/0_build_annotation_vector_db.py \
  --run-id <INDEX_ID> --run-date <YYYY-MM-DD> --kb-scope full
#    → journalise : classify-rag-annotations-collection: coicop_annotations__full__<date>__<INDEX_ID>
#    options : --sample-size N (KB plafonnée), --collection-name (forcer un nom exact)

# 2a. PRODUCTION (défaut) : codifie l'input, exporte les prédictions, pas d'évaluation
#     --collection-name est OBLIGATOIRE
uv run scripts/1_run_rag.py --run-id <ID> --run-date <YYYY-MM-DD> \
  --collection-name coicop_annotations__full__<date>__<INDEX_ID>

# 2b. ÉVALUATION (opt-in) : idem + métriques vs labels (input labellisé requis)
uv run scripts/1_run_rag.py --run-id <ID> --run-date <YYYY-MM-DD> \
  --collection-name coicop_annotations__train__<date>__<INDEX_ID> \
  --skip-eval false
```

`--collection-name` est **optionnel côté indexation** (vide = nom composé) et
**obligatoire côté classification** : il n'y a plus de nom par défaut en config,
précisément pour qu'un oubli échoue en quelques secondes au lieu de retomber en silence
sur l'index d'un autre run. Les deux runs ont des identités **distinctes** : le
`--run-id` / `--run-date` de l'indexation sert à nommer la collection, celui de la
classification à localiser les entrées et sorties du run de codification.

### Modes production / évaluation

`1_run_rag.py` produit **toujours** des prédictions (`data.s3_path_predictions`).
L'**évaluation** (accuracy + recall par niveau) est **opt-in** via `--skip-eval false`
— cohérent avec la convention `skip-report` de `argo/codif-pipeline.yaml` (paramètre
booléen string, testé par `!= "true"`). En production, l'input n'a pas de labels et
l'évaluation est ignorée (un garde-fou la saute aussi s'il manque la colonne `code`).

## Prérequis upstream

Tous produits par l'étape `prune-codes` (module [`prune-codes/`](../prune-codes/)),
sous le `run_date`/`run_id` du run concerné.

Pour **`0_build_annotation_vector_db.py`** (run d'indexation, `prune-codes --only kb`) :

- KB d'annotations **prunée** (`prune-codes/annotations_train_pruned.parquet`) → indexée.
  Dans `argo/index-annotations-pipeline.yaml`, sa source est surchargée via
  `prune-codes --annotations-train` et pointe sur `build-datasets/annotations_full.parquet`
  (ou `raw_train.parquet` si `kb-scope=train`), **pas** sur une sortie `classify-regex` ;
- suggester **pruné** (`prune-codes/suggester_pruned.parquet`) si `suggester.enabled` —
  source surchargée sur `build-datasets/suggester.parquet` (le suggester préprocessé, qui
  porte déjà `l_pr_product`) plutôt que sur le CSV brut.

Pour **`1_run_rag.py`** (run de classification) :

- input à codifier **pruné** (`prune-codes/annotations_test_pruned.parquet`) : split test
  labellisé en mode évaluation, observations à codifier en production ;
- une collection Qdrant **déjà construite** avec son manifeste, dont le nom est passé en
  argument.

## Rapport d'évaluation (mode `--skip-eval false`)

Loggé dans MLflow (scalaires `log_metrics` + tables `log_table` + `report.txt`) :

- **Accuracy / recall par niveau** (1..`eval.levels`) sur `all_parsed` et `parsed_and_codable`.
- **Accuracy par catégorie COICOP niveau 1** (`eval/accuracy_by_level1.json`) : support et
  précision par grande catégorie, au niveau 1 et au niveau cible.
- **Accuracy par source des données de test** (accuracy globale + niveau 1 par `source`) —
  **uniquement dans `report.txt`** (pas en scalaire ni en table MLflow).
- **Distorsion de distribution** niveaux 1 et 2 (`eval/distribution_level_{1,2}.json`) :
  parts vraies vs prédites par catégorie + `diff`, et indicateurs agrégés **TV distance**
  et **KL** (détecte les sur/sous-prédictions systématiques d'une catégorie).
- **Fiabilité de la confiance LLM** : **AUROC** confiance↔justesse, courbe de **calibration**
  (accuracy par bin), et **balayage de seuils** (couverture vs accuracy si on filtre sur
  `confidence ≥ t`) → indique si la confiance permet d'écarter les mauvaises prédictions.
- **Fiabilité du flag `codable`** : accuracy et couverture pour `codable` True/False/manquant,
  et **lift** d'accuracy si on ne garde que `codable=True`.

## Limites de cette première version

- Le texte du suggester est utilisé tel quel (pas la normalisation `l_pr_product`
  du prétraitement) ; cohérence des **codes** garantie, pas du prétraitement texte.
- Embeddings recalculés à chaque run (pas de cache).
- Prompt local par défaut ; le pousser dans Langfuse pour le versionner.
