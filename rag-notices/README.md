# RAG classifier for product

Using coicop classification.

## Starting

```bash
uv sync --locked
```

Le dépôt est un workspace `uv` : le lock est à la racine, et cette commande n'installe
que les dépendances de ce module (voir « Environnement Python » dans le README racine).

## Workflow

Les deux scripts de ce module vivent désormais dans **deux pipelines Argo distincts** :

```
argo/index-notices-pipeline.yaml : prune-codes (--only nomenclature) ──→ index-notices
argo/codif-pipeline.yaml         : … ──→ classify-rag-notices
```

L'indexation est **sortie du pipeline de classification**. Deux raisons :

- embedder toute la nomenclature n'a aucune raison d'être repayé à chaque codification ;
- tant que la collection portait un nom fixe partagé par tous les runs, une
  réindexation détruisait la base que lisait un run concurrent.

Même organisation que l'entraînement de `classify-ttc` et de `reconcile-sirus` :
l'artefact coûteux est produit à part, et son identifiant est recopié dans le fichier
de paramètres du pipeline de classification (`argo/params.yaml`, clé
`classify-rag-notices-collection`). Le log de fin d'`index-notices` affiche la ligne
exacte à recopier.

Le pipeline d'indexation est **autonome au sens strict** : la nomenclature dérive d'un
CSV statique, aucune donnée d'un run de classification n'y entre. D'où
`prune-codes --only nomenclature`, qui s'arrête après l'étape 1 — les étapes suivantes
liraient des sorties `classify-regex` qui n'existent pas ici.

> Le pruning (nomenclature, mapping, annotations, suggester) est centralisé dans le
> module **`prune-codes/`** ; les anciens scripts `0_prunning_coicop.py` et
> `1_prune_annotations.py` ont été retirés d'ici. `index-notices` lit directement
> la nomenclature prunée (`prune-codes/nomenclature_pruned.parquet`).

### 0_create_vector_db.py

Encode les notices COICOP **prunées** dans une base vectorielle Qdrant.

1. **Génération des embeddings** : les notices sont encodées via le modèle d'embedding hébergé sur llm.lab (`LLMLAB_URL`, `LLMLAB_API_KEY`). Un échec d'embedding est **fatal** : poursuivre décalerait l'appariement vecteur ↔ payload et produirait un index silencieusement faux.
2. **Stockage vectoriel** : les embeddings sont indexés dans Qdrant (`QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_API_PORT`)
3. **Manifeste** : un JSON plat est écrit sous `s3://projet-budget-famille/data/vector_db_manifests/{collection_name}.json` (`qdrant.manifests_root`), portant le modèle d'embedding, sa dimension, la stratégie de découpage, la nomenclature source, le nombre de points **compté auprès de Qdrant** et le SHA git du code qui a bâti l'index.

### 2_run_rag.py

Classifie les annotations via le pipeline RAG.

1. **Validation de la collection** : avant tout travail coûteux — et **avant MLflow**, pour ne pas laisser un run FAILED dans l'expérience — le script relit le manifeste de la collection passée en argument et vérifie qu'elle existe, que sa dimension, son modèle d'embedding et sa stratégie correspondent à ce que ce run attend, et qu'elle n'est pas vide.
2. **Gestion des prompts** : les templates sont stockés dans Langfuse (`LANGFUSE_BASE_URL`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`)
3. **Retrieval et génération** :
   - Les contextes pertinents sont récupérés depuis Qdrant
   - La génération finale utilise le modèle hébergé sur llm.lab (`LLMLAB_URL`, `LLMLAB_API_KEY`)
4. **Logging MLflow** : les métriques sont enregistrées dans MLflow (`MLFLOW_TRACKING_URI`), avec la provenance de l'index (`index.run_id`, `index.git_sha`) — sans quoi deux runs aux métriques différentes mais bâtis sur des index différents seraient indistinguables.

## Nommage des collections Qdrant

Il n'y a plus de collection au nom fixe (`coicop_lineage` a disparu). Chaque build
compose un nom **unique** :

```
{qdrant.collection_base}__{run_date}__{run_id}        # ex. coicop_notices__2026-09-02__index-notices-a7k2p
```

La clé de config est `qdrant.collection_base`, et surtout **pas** `collection_name` :
ce même fichier de config est lu par le constructeur *et* par le consommateur, et une
clé `collection_name` inciterait le consommateur à retomber en silence sur une
collection périmée. Le nom complet lui est donc passé en argument, obligatoirement.

Le nom seul ne peut pas porter tout ce qui détermine la validité d'un index : deux
collections de même dimension bâties avec des réglages différents sont indistinguables
et interroger la mauvaise ne lève aucune erreur — le RAG rend juste de moins bons
résultats. D'où le manifeste, écrit à côté de la collection et relu au démarrage du
consommateur.

## Exécution locale / interactive (hors Argo)

Pour déboguer `scripts/2_run_rag.py` en interactif (exécution ligne par ligne), le script lit
ses credentials via `os.environ[...]`. Ces variables doivent donc être présentes dans
l'environnement **avant** de démarrer la session.

### 1. Injecter un secret dans Vault

Créer un secret (via l'UI Onyxia *Mes secrets*)
contenant les clés suivantes :

```
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
DDC_ENCRYPTION_KEY
LANGFUSE_BASE_URL
LANGFUSE_PUBLIC_KEY
LANGFUSE_SECRET_KEY
LLMLAB_API_KEY
LLMLAB_URL
MLFLOW_TRACKING_PASSWORD
MLFLOW_TRACKING_URI
MLFLOW_TRACKING_USERNAME
QDRANT_API_KEY
QDRANT_API_PORT
QDRANT_URL
```

Au lancement du service Onyxia, référencer le chemin de ce secret dans la section *Vault* pour
qu'Onyxia injecte chaque clé comme variable d'environnement.

### 2. Lancer

```bash
cd rag-notices
uv sync --locked
mkdir -p logs prompts   # le logger écrit dans logs/ (sinon échec au démarrage)

uv run python           # session interactive : os.environ est déjà peuplé par Vault
```

En une fois :

```bash
# Construire l'index (run d'indexation, identité propre)
uv run scripts/0_create_vector_db.py --run-id <INDEX_ID> --run-date <YYYY-MM-DD>
#   → journalise : classify-rag-notices-collection: coicop_notices__<date>__<INDEX_ID>

# Codifier (run de classification, autre identité) — le nom de collection est OBLIGATOIRE
uv run scripts/2_run_rag.py --run-id <ID> --run-date <YYYY-MM-DD> \
  --collection_name coicop_notices__<date>__<INDEX_ID> \
  --sample_size 100
```

⚠️ Les deux scripts n'orthographient pas le drapeau pareil : `0_create_vector_db.py`
attend `--collection-name` (tirets), `2_run_rag.py` attend `--collection_name`
(underscore, cohérent avec ses autres options historiques).

`--collection-name` est **optionnel côté indexation** (vide = nom composé depuis
`qdrant.collection_base` et l'identité du run) et **obligatoire côté classification** :
il n'y a plus de nom par défaut en config, précisément pour qu'un oubli échoue en
quelques secondes au lieu de retomber en silence sur l'index d'un autre run.

### Pré-requis

- **`0_create_vector_db.py`** : pour son couple `run-id` / `run-date`, la nomenclature
  prunée (`prune-codes/nomenclature_pruned.parquet`) doit exister — c'est ce que produit
  `prune-codes --only nomenclature` en amont dans `argo/index-notices-pipeline.yaml`.
- **`2_run_rag.py`** : pour son couple `run-id` / `run-date`, les sorties de l'étape
  `prune-codes` (`prune-codes/annotations_test_pruned.parquet`,
  `prune-codes/nomenclature_pruned.parquet`, `prune-codes/mapping_lvl4.parquet`) ; une
  collection Qdrant **déjà construite** avec son manifeste, dont le nom est passé en
  argument ; et le prompt Langfuse `prompt-multi-level` (chemins/versions dans
  `config/config.yaml`).
