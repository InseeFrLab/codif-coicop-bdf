# codif-coicop-bdf

Pipeline d'orchestration pour la codification automatique des produits de l'enquête Budget de Famille (BDF) selon la nomenclature COICOP.

## Pipeline

Le pipeline est orchestré via Argo Workflows (`argo/pipeline.yaml`) selon le DAG suivant :

```
                   ┌──→ prune-coicop ──→ create-vector-db ─────────────┐  (skippable)
   preprocessing ──┤                                                   ├──→ run-rag ─┐
                   └──→ codif-regex ──→ prune-annotations ─────────────┘             │
                                 ├──→ codif-lcs ─────────────────────────────────────┼──→ decide-coicop
                                 └──→ run-ttc  ──────────────────────────────────────┘
```

### preprocessing

Construit le dataset d'annotations à partir des sources brutes (COPAIN, historique, suggester).

- Code dans [`preprocessing/`](./preprocessing/) (ex-repo `construction-dataset`)
- Exporte le dataset consolidé sur S3

### prune-coicop *(skippable)*

Élague les hiérarchies linéaires de la nomenclature COICOP brute.

- Code dans [`coicop-rag/`](./coicop-rag/) (`scripts/0_prunning_coicop.py`)
- Supprime le niveau 5 (Poste) de la nomenclature
- Produit les notices prunées et la table de mapping niveau 4 sur S3

### create-vector-db *(skippable)*

Encode les notices COICOP dans une base vectorielle Qdrant.

- Code dans [`coicop-rag/`](./coicop-rag/) (`scripts/0_create_vector_db.py`)
- Génère les embeddings via le modèle VLLM (`VLLM_EMBEDDING_URL`)
- Indexe les vecteurs dans Qdrant (`QDRANT_URL`)

### prune-annotations

Tronque les codes d'annotation au niveau 4 et normalise les codes de vérité terrain via la table de mapping COICOP.

- Code dans [`coicop-rag/`](./coicop-rag/) (`scripts/1_prune_annotations.py`)

### codif-regex

Codification des libellés produits par approche regex.

- Code dans [`regex-codif/`](./regex-codif/) (ex-repo `regex_codif`)

### codif-lcs

Codification des libellés produits par approche LCS (Longest Common Subsequence) en R.

- Code dans [`stats-annotations/`](./stats-annotations/)

### run-rag

Classifie les annotations via un pipeline RAG (Retrieval-Augmented Generation).

- Code dans [`coicop-rag/`](./coicop-rag/) (`scripts/2_run_rag.py`)
- Récupère les contextes pertinents depuis Qdrant
- Génère les codes COICOP via le modèle VLLM (`VLLM_GENERATION_URL`)
- Enregistre les métriques dans MLflow (`MLFLOW_TRACKING_URI`)
- Trace les expériences dans Langfuse (`LANGFUSE_BASE_URL`)

### run-ttc

Prédictions TTC via un classifieur pré-entraîné.

- Code dans [`coicop-bdf-classifier/`](./coicop-bdf-classifier/) (ex-repo `coicop_bdf_classifier`, bientôt archivé en amont)
- L'étape argo utilise actuellement l'image pré-construite `ghcr.io/micedre/coicop_bdf_classifier:latest`
- Étape destinée à être supprimée à terme

### decide-coicop

Arbitrage final des prédictions par un LLM-as-judge : fusionne les sorties de `codif-lcs`, `run-rag` et `run-ttc` et sélectionne le meilleur code COICOP par observation.

- Code dans [`coicop-bdf-classifier/`](./coicop-bdf-classifier/) (sous-commande `uv run main.py decide-coicop`)
- Entrées :
  - `s3://.../codif-lcs/raw_test_LCS.parquet`
  - `s3://.../run-rag/predictions.parquet`
  - `s3://.../run-ttc/predictions.parquet`
- Sortie : `s3://.../decide-coicop/predictions.parquet`
- Utilise un endpoint OpenAI-compatible (`OPENAI_API_KEY`, optionnellement `OPENAI_BASE_URL` pour un backend non-OpenAI)
- Court-circuit consensus : si les trois sources convergent (et que la confiance TTC ≥ 0.90), aucune requête LLM n'est émise
- Filtrage de nomenclature : seules les sections COICOP pertinentes sont envoyées au prompt (réduction ×4–10 du nombre de tokens)
- Reprise automatique : relancer l'étape avec le même `run_id`/`run_date` reprend les observations non traitées depuis le fichier de sortie existant

## Structure du dépôt

Ce dépôt rassemble le code de toutes les étapes du pipeline, auparavant dispersé dans plusieurs repos.

| Dossier | Origine | Rôle |
|---|---|---|
| [`argo/`](./argo/) | — | Workflows Argo (`pipeline.yaml`, `ttc-pipeline.yaml`, `rbac.yaml`) |
| [`preprocessing/`](./preprocessing/) | `construction-dataset` | Étape `preprocessing` |
| [`regex-codif/`](./regex-codif/) | `regex_codif` | Étape `codif-regex` |
| [`coicop-rag/`](./coicop-rag/) | `coicop-rag` | Étapes `prune-coicop`, `prune-annotations`, `create-vector-db`, `run-rag` |
| [`stats-annotations/`](./stats-annotations/) | `stats-annotations` | Étape `codif-lcs` (R) |
| [`coicop-bdf-classifier/`](./coicop-bdf-classifier/) | `coicop_bdf_classifier` | Classifieur TTC (source vendorisée, image pré-construite encore utilisée par `run-ttc`) |

Chaque sous-dossier Python conserve son propre `pyproject.toml` / `uv.lock` et peut être développé et exécuté indépendamment.

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
DDC_ENCRYPTION_KEY,
OPENAI_API_KEY, OPENAI_BASE_URL   # requis pour decide-coicop (OPENAI_BASE_URL optionnel)
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
| `decide-model` | `gemma4-26b-moe` | Modèle LLM utilisé par `decide-coicop` (vide → défaut `gpt-4o` de la commande) |
| `decide-concurrency` | `5` | Nombre d'appels LLM parallèles de `decide-coicop` |
