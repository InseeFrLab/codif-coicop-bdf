# codif-coicop-bdf

Pipeline d'orchestration pour la codification automatique des produits de l'enquête Budget de Famille (BDF) selon la nomenclature COICOP.

## Pipeline

Le pipeline est orchestré via Argo Workflows (`argo/pipeline.yaml`) selon le DAG suivant :

```
                   ┌──→ prune-coicop ──→ create-vector-db ──┐  (skippable)
preprocessing ─────┤                                         ├──→ run-rag
                   └──→ prune-annotations ──→ codif-regex ───┘
                                                └──→ codif-lcs
```

### preprocessing

Construit le dataset d'annotations à partir des sources brutes (COPAIN, historique, suggester).

- Clone et exécute le repo [`construction-dataset`](https://git.lab.sspcloud.fr/ssplab/experimentation-bdf/construction-dataset)
- Exporte le dataset consolidé sur S3

### prune-coicop *(skippable)*

Élague les hiérarchies linéaires de la nomenclature COICOP brute.

- Supprime le niveau 5 (Poste) de la nomenclature
- Produit les notices prunées et la table de mapping niveau 4 sur S3

### create-vector-db *(skippable)*

Encode les notices COICOP dans une base vectorielle Qdrant.

- Génère les embeddings via le modèle VLLM (`VLLM_EMBEDDING_URL`)
- Indexe les vecteurs dans Qdrant (`QDRANT_URL`)

### prune-annotations

Tronque les codes d'annotation au niveau 4 et normalise les codes de vérité terrain via la table de mapping COICOP.

### codif-regex

Codification des libellés produits par approche regex.

- Clone et exécute le repo [`regex_codif`](https://git.lab.sspcloud.fr/ssplab/experimentation-bdf/regex_codif)

### codif-lcs

Codification des libellés produits par approche LCS (Longest Common Subsequence) en R.

- Clone et exécute le repo [`stats-annotations`](https://git.lab.sspcloud.fr/ssplab/experimentation-bdf/stats-annotations)

### run-rag

Classifie les annotations via un pipeline RAG (Retrieval-Augmented Generation).

- Récupère les contextes pertinents depuis Qdrant
- Génère les codes COICOP via le modèle VLLM (`VLLM_GENERATION_URL`)
- Enregistre les métriques dans MLflow (`MLFLOW_TRACKING_URI`)
- Trace les expériences dans Langfuse (`LANGFUSE_BASE_URL`)

## Lancer le workflow

### Prérequis

Le secret Kubernetes `secret-codif-coicop-bdf` doit exister dans le namespace et contenir les clés suivantes :

```
AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN,
AWS_S3_ENDPOINT, AWS_ENDPOINT_URL,
VLLM_EMBEDDING_URL, VLLM_EMBEDDING_API_KEY,
VLLM_GENERATION_URL, VLLM_GENERATION_API_KEY,
QDRANT_URL, QDRANT_API_KEY, QDRANT_API_PORT,
LANGFUSE_BASE_URL, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY,
MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD,
OLLAMA_URL, OLLAMA_API_KEY,
DDC_ENCRYPTION_KEY
```

### Avec la CLI Argo

```bash
# Pipeline complet
argo submit argo/pipeline.yaml

# Limiter à N annotations (utile pour tester)
argo submit argo/pipeline.yaml -p sample_size=100

# Utiliser un modèle spécifique pour run-rag
argo submit argo/pipeline.yaml -p model-name=openai/gpt-oss-120b

# Sauter la construction de la vector DB (si déjà construite)
argo submit argo/pipeline.yaml -p skip-vector-db=true

# Combinaison de paramètres
argo submit argo/pipeline.yaml -p skip-vector-db=true -p sample_size=100
```

### Avec kubectl (sans CLI Argo)

```bash
python3 -c "
import sys, yaml
with open('argo/pipeline.yaml') as f:
    w = yaml.safe_load(f)
# Modifier les paramètres souhaités, ex:
for p in w['spec']['arguments']['parameters']:
    if p['name'] == 'skip-vector-db':
        p['value'] = 'true'
    if p['name'] == 'sample_size':
        p['value'] = '100'
print(yaml.dump(w, default_flow_style=False))
" | kubectl create -f -
```

### Paramètres disponibles

| Paramètre | Défaut | Description |
|---|---|---|
| `sample_size` | *(vide)* | Nombre d'annotations à traiter (toutes si vide) |
| `model-name` | *(vide)* | Modèle LLM à utiliser pour `run-rag` (défaut du config si vide) |
| `skip-vector-db` | `false` | Si `true`, saute `prune-coicop` et `create-vector-db` |
