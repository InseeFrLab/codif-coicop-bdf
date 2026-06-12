# RAG classifier for product

Using coicop classification.

## Starting

```python
uv sync
```

## Workflow

Ce module fournit deux étapes du pipeline (`argo/codif-pipeline.yaml`), en aval de
l'étape de pruning unifiée (module [`prune/`](../prune/)) :

```
prune ──→ create-vector-db ──→ run-rag
```

> Le pruning (nomenclature, mapping, annotations, suggester) est désormais centralisé
> dans le module **`prune/`** ; les anciens scripts `0_prunning_coicop.py` et
> `1_prune_annotations.py` ont été retirés d'ici. `create-vector-db` lit directement
> la nomenclature prunée (`prune/nomenclature_pruned.parquet`).

### 0_create_vector_db.py

Encode les notices COICOP **prunées** dans une base vectorielle Qdrant.

1. **Génération des embeddings** : les notices sont encodées via le modèle d'embedding hébergé sur llm.lab (`LLMLAB_URL`, `LLMLAB_API_KEY`)
2. **Stockage vectoriel** : les embeddings sont indexés dans Qdrant (`QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_API_PORT`)

### 2_run_rag.py

Classifie les annotations via le pipeline RAG.

3. **Gestion des prompts** : les templates sont stockés dans Langfuse (`LANGFUSE_BASE_URL`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`)
4. **Retrieval et génération** :
   - Les contextes pertinents sont récupérés depuis Qdrant
   - La génération finale utilise le modèle hébergé sur llm.lab (`LLMLAB_URL`, `LLMLAB_API_KEY`)
5. **Logging MLflow** : les métriques sont enregistrées dans MLflow (`MLFLOW_TRACKING_URI`)

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
cd coicop-rag
uv sync
mkdir -p logs prompts   # le logger écrit dans logs/ (sinon échec au démarrage)

uv run python           # session interactive : os.environ est déjà peuplé par Vault
```

En une fois : `uv run scripts/2_run_rag.py --run-id <ID> --run-date <YYYY-MM-DD> --sample_size 100`.

### Pré-requis

Pour le couple `run-id` / `run-date` choisi, doivent déjà exister sur S3/services : les sorties
de l'étape `prune` (`prune/annotations_test_pruned.parquet`, `prune/nomenclature_pruned.parquet`,
`prune/mapping_lvl4.parquet`), la collection Qdrant `coicop_lineage` et le prompt Langfuse
`prompt-multi-level` (chemins/versions dans `config/config.yaml`).
