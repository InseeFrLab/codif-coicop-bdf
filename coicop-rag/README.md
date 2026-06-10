# RAG classifier for product

Using coicop classification.

## Starting

```python
uv sync
```

## Workflow

Le pipeline est orchestré via Argo Workflows (`argo/pipeline.yaml`) selon le DAG suivant :

```
                   ┌──→ prune-coicop ──→ create-vector-db ──┐
preprocessing ─────┤                                         ├──→ run-rag
                   └──→ prune-annotations ───────────────────┘
```

### preprocessing (construction-dataset)

Construit le dataset d'annotations à partir des sources brutes (COPAIN, historique, suggester).

- Clone et exécute le repo `construction-dataset`
- Exporte le dataset consolidé sur S3

### 0_prunning_coicop.py

Élague les hiérarchies linéaires de la nomenclature COICOP brute et exporte les notices filtrées ainsi qu'une table de correspondance vers S3.

- Supprime le niveau 5 (Poste) de la nomenclature
- Produit les notices prunées et la table de mapping niveau 4

### 0_create_vector_db.py

Encode les notices COICOP dans une base vectorielle Qdrant (plusieurs stratégies d'indexation).

1. **Génération des embeddings** : les notices sont encodées via le modèle d'embedding hébergé sur llm.lab (`LLMLAB_URL`, `LLMLAB_API_KEY`)
2. **Stockage vectoriel** : les embeddings sont indexés dans Qdrant (`QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_API_PORT`)

### 1_prune_annotations.py

Tronque les codes d'annotation au niveau 4 et applique la table de correspondance COICOP pour normaliser les codes de vérité terrain.

- Dépend de `0_prunning_coicop.py` (requiert la table de mapping sur S3)

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

Pour le couple `run-id` / `run-date` choisi, doivent déjà exister sur S3/services : la sortie
de `prune-annotations`, la table de mapping de prunning, la collection Qdrant `coicop_lineage`
et le prompt Langfuse `prompt-multi-level` (chemins/versions dans `config/config.yaml`).
