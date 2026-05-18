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
