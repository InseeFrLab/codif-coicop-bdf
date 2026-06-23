# codif-coicop-bdf

Pipeline d'orchestration pour la codification automatique des produits de l'enquête Budget de Famille (BDF) selon la nomenclature COICOP.

## Pipeline

Le pipeline est orchestré via Argo Workflows (`argo/codif-pipeline.yaml`) selon le DAG suivant :

```
                                     ┌──→ create-vector-db ──────────────┐  (skippable)
   preprocessing ──┐   ┌──→ prune ───┤                                   ├──→ run-rag(-annotations) ─┐
    (input_file)   └──→┤             └──→ create-vector-db-annotations ───┘                           │
                       └──→ codif-regex ─┬──→ codif-lcs ──────────────────────────────────────────────┼──→ decide-coicop ──→ final-output ──→ report (opt-in)
                         (→ prune)       └──→ run-ttc ──────────────────────────────────────────────┘
```

Le **mode** est dérivé d'`input_file` : non vide ⇒ **production** (on code ces observations), vide ⇒ **évaluation** (on code le split test des annotations). L'étape **`prune`** centralise tout le pruning et produit les artefacts lus par l'aval.

### preprocessing

Construit le dataset d'annotations à partir des sources brutes (COPAIN, historique, suggester).

- Code dans [`preprocessing/`](./preprocessing/) (ex-repo `construction-dataset`)
- Exporte le dataset consolidé sur S3

### prune

Étape **unique** de pruning (troncature niveau 4 + élagage des hiérarchies linéaires → code canonique). Produit tous les artefacts prunés sous `…/{run}/prune/`, lus par l'aval.

- Code dans [`prune/`](./prune/) (`scripts/main.py`)
- Sorties : `nomenclature_pruned`, `mapping_lvl4`, `annotations_train_pruned` (KB), `annotations_test_pruned` (à coder), `suggester_pruned`

### create-vector-db *(skippable)*

Encode les notices COICOP **prunées** dans une base vectorielle Qdrant.

- Code dans [`coicop-rag/`](./coicop-rag/) (`scripts/0_create_vector_db.py`)
- Lit `prune/nomenclature_pruned.parquet` ; embeddings via VLLM, index Qdrant

### create-vector-db-annotations *(skippable)*

Encode la **KB d'annotations** prunée (+ suggester) dans une vector DB Qdrant, pour la RAG sur exemples annotés.

- Code dans [`coicop-rag-annotations/`](./coicop-rag-annotations/) (`scripts/0_build_annotation_vector_db.py`)
- Lit `prune/annotations_train_pruned.parquet` et `prune/suggester_pruned.parquet`

### codif-regex

Codification des libellés produits par approche regex.

- Code dans [`regex-codif/`](./regex-codif/) (ex-repo `regex_codif`)

### codif-lcs

Codification des libellés produits par approche LCS (Longest Common Subsequence) en R.

- Code dans [`stats-annotations/`](./stats-annotations/)

### run-rag

Code le jeu à coder via un RAG sur les **notices** de la nomenclature COICOP.

- Code dans [`coicop-rag/`](./coicop-rag/) (`scripts/2_run_rag.py`)
- Récupère les notices proches depuis Qdrant, génère via VLLM (`VLLM_GENERATION_URL`)
- Métriques MLflow (`MLFLOW_TRACKING_URI`), traces Langfuse (`LANGFUSE_BASE_URL`)

### run-rag-annotations

Code le jeu à coder via un RAG sur des **exemples déjà annotés** (few-shot).

- Code dans [`coicop-rag-annotations/`](./coicop-rag-annotations/) (`scripts/1_run_rag.py`)
- Récupère les annotations proches depuis Qdrant ; éval (accuracy/recall) opt-in via le mode

### run-ttc

Prédictions TTC via un classifieur pré-entraîné.

- Code dans [`coicop-bdf-classifier/`](./coicop-bdf-classifier/) (ex-repo `coicop_bdf_classifier`, bientôt archivé en amont)
- L'étape argo utilise actuellement l'image pré-construite `ghcr.io/micedre/coicop_bdf_classifier:latest`
- Étape destinée à être supprimée à terme

### decide-coicop

Arbitrage final des prédictions par un LLM-as-judge : fusionne les sorties de `codif-lcs`, `run-rag`, `run-rag-annotations` et `run-ttc` et sélectionne le meilleur code COICOP par observation.

- Code dans [`coicop-bdf-classifier/`](./coicop-bdf-classifier/) (sous-commande `uv run main.py decide-coicop`)
- Entrées :
  - `s3://.../codif-lcs/raw_test_LCS.parquet`
  - `s3://.../run-rag/predictions.parquet`
  - `s3://.../rag-annotation/predictions.parquet`
  - `s3://.../run-ttc/predictions.parquet`
- Sortie : `s3://.../decide-coicop/predictions.parquet`
- Utilise un endpoint OpenAI-compatible (`LLMLAB_API_KEY`, optionnellement `LLMLAB_URL` pour un backend non-OpenAI)
- Court-circuit consensus : si les trois sources convergent (et que la confiance TTC ≥ 0.90), aucune requête LLM n'est émise
- Filtrage de nomenclature : seules les sections COICOP pertinentes sont envoyées au prompt (réduction ×4–10 du nombre de tokens)
- Reprise automatique : relancer l'étape avec le même `run_id`/`run_date` reprend les observations non traitées depuis le fichier de sortie existant

### report *(opt-in)*

Rapport d'exactitude Quarto (HTML auto-contenu) sur la sortie de `decide-coicop`.

- Code dans [`report/`](./report/) — Quarto + Python (pandas, duckdb, matplotlib, seaborn)
- Déclenché par `skip-report=false` ; désactivé par défaut
- Entrée : `s3://.../decide-coicop/predictions.parquet`
- Sortie : `s3://.../report/report.html`
- Contenu :
  - Accuracy globale par niveau COICOP (1 à 5) pour **LCS, RAG, TTC, LLM**
  - Accuracy par `shop`, `shop_type_name` et **quartile de `budget`**
  - Matrice de confusion (top 20 paires `code` vs `llm_code` au niveau 4)
  - Calibration : accuracy par bucket de `llm_confiance`
  - Consensus vs désaccord des sources amont : apport de l'arbitrage LLM
- Méthodologie *accuracy par niveau* : tronquer `code` et la prédiction aux `k` premiers segments ; les observations dont la vérité a moins de `k` niveaux sont exclues du dénominateur à ce niveau

## Structure du dépôt

Ce dépôt rassemble le code de toutes les étapes du pipeline, auparavant dispersé dans plusieurs repos.

| Dossier | Origine | Rôle |
|---|---|---|
| [`argo/`](./argo/) | — | Workflows Argo (`codif-pipeline.yaml`, `ttc-pipeline.yaml`, `rbac.yaml`) |
| [`preprocessing/`](./preprocessing/) | `construction-dataset` | Étape `preprocessing` |
| [`regex-codif/`](./regex-codif/) | `regex_codif` | Étape `codif-regex` |
| [`prune/`](./prune/) | — | Étape `prune` (pruning unifié) |
| [`coicop-rag/`](./coicop-rag/) | `coicop-rag` | Étapes `create-vector-db`, `run-rag` |
| [`coicop-rag-annotations/`](./coicop-rag-annotations/) | — | Étapes `create-vector-db-annotations`, `run-rag-annotations` |
| [`stats-annotations/`](./stats-annotations/) | `stats-annotations` | Étape `codif-lcs` (R) |
| [`coicop-bdf-classifier/`](./coicop-bdf-classifier/) | `coicop_bdf_classifier` | Étapes `run-ttc`, `decide-coicop` |
| [`final-output/`](./final-output/) | — | Étape `final-output` (livrable utilisateur) |
| [`report/`](./report/) | — | Rapport Quarto d'exactitude (étape `report`, opt-in) |

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
LLMLAB_API_KEY, LLMLAB_URL   # requis pour decide-coicop (LLMLAB_URL optionnel)
```

### Avec la CLI Argo

```bash
# Pipeline complet (mode prod si input_file est fourni dans le YAML, sinon éval)
argo submit argo/codif-pipeline.yaml

# Lancer en production sur un fichier d'observations à coder
argo submit argo/codif-pipeline.yaml \
  -p input_file=s3://.../workflow_inputs/mon_fichier.csv \
  -p text_column=NAT_DEP -p shop_column=MAG_DEP -p budget_column=MONT_DEP

# Limiter le volume pour tester (sampling centralisé à codif-regex, hérité par tous
# les classifieurs ; en prod, -p sample-observations=100)
argo submit argo/codif-pipeline.yaml -p sample-annotations=100

# Modèle spécifique pour run-rag
argo submit argo/codif-pipeline.yaml -p model-name=openai/gpt-oss-120b

# Sauter la (re)construction des vector DB (déjà construites)
argo submit argo/codif-pipeline.yaml -p skip-vector-db=true

# Activer le rapport d'exactitude (désactivé par défaut)
argo submit argo/codif-pipeline.yaml -p skip-report=false
```

Voir aussi la fiche [`argo/argo_helper.md`](./argo/argo_helper.md) et le fichier de paramètres [`argo/params.yaml`](./argo/params.yaml).

### Paramètres disponibles

| Paramètre | Défaut | Description |
|---|---|---|
| `input_file` | *(csv BDF)* | Non vide ⇒ **production** (code ces observations) ; vide ⇒ **évaluation** (code le split test des annotations). Pilote le mode partout. |
| `sample-annotations` | *(vide)* | Plafonne la KB d'annotations indexée (vector DB). En éval, sert aussi de cap au jeu à coder. |
| `sample-observations` | *(vide)* | Plafonne le jeu à coder (production). Échantillonné **une fois** à `codif-regex`, hérité par tous les classifieurs. |
| `model-name` | `gemma4-26b-moe` | Modèle LLM pour `run-rag` / `run-rag-annotations` |
| `skip-vector-db` | `true` | Si `true`, saute `create-vector-db` et `create-vector-db-annotations` |
| `decide-model` | `gemma4-26b-moe` | Modèle LLM utilisé par `decide-coicop` |
| `decide-concurrency` | `5` | Nombre d'appels LLM parallèles de `decide-coicop` |
| `skip-report` | `false` | Si `false`, génère le rapport Quarto après `final-output` |
