# codif-coicop-bdf

Pipeline d'orchestration pour la codification automatique des produits de l'enquête Budget de Famille (BDF) selon la nomenclature COICOP.

## Pipeline

Le pipeline est orchestré via Argo Workflows (`argo/codif-pipeline.yaml`) selon le DAG suivant :

```
build-datasets  (input_file : production / évaluation)
  └─→ classify-regex ─┬─→ classify-lcs ────────────────────────────────────────────────────┐
                      ├─→ classify-ttc ────────────────────────────────────────────────────┤
                      └─→ prune-codes ─┬─→ index-notices ─────→ classify-rag-notices ──────┤
                                       └─→ index-annotations ─→ classify-rag-annotations ──┘
                                                                                           │
                                                       ┌───────────────────────────────────┘
                                                       │  les 4 classifieurs convergent
                                                       └─→ reconcile-llm  OU  reconcile-sirus   (exclusifs : paramètre `reconciliation`)
                                                               └─→ export-results ─→ report  (opt-in)
```

Le **mode** est dérivé d'`input_file` : non vide ⇒ **production** (on code ces observations), vide ⇒ **évaluation** (on code le split test des annotations). L'étape **`prune-codes`** centralise tout le pruning et produit les artefacts lus par l'aval.

### build-datasets

Construit le dataset d'annotations à partir des sources brutes (COPAIN, historique, suggester).

- Code dans [`build-datasets/`](./build-datasets/) (ex-repo `construction-dataset`)
- Exporte le dataset consolidé sur S3

### prune-codes

Étape **unique** de pruning (troncature niveau 4 + élagage des hiérarchies linéaires → code canonique). Produit tous les artefacts prunés sous `…/{run}/prune-codes/`, lus par l'aval.

- Code dans [`prune-codes/`](./prune-codes/) (`scripts/main.py`)
- Sorties : `nomenclature_pruned`, `mapping_lvl4`, `annotations_train_pruned` (KB), `annotations_test_pruned` (à coder), `suggester_pruned`

### index-notices *(skippable)*

Encode les notices COICOP **prunées** dans une base vectorielle Qdrant.

- Code dans [`rag-notices/`](./rag-notices/) (`scripts/0_create_vector_db.py`)
- Lit `prune-codes/nomenclature_pruned.parquet` ; embeddings via VLLM, index Qdrant

### index-annotations *(skippable)*

Encode la **KB d'annotations** prunée (+ suggester) dans une vector DB Qdrant, pour la RAG sur exemples annotés.

- Code dans [`rag-annotations/`](./rag-annotations/) (`scripts/0_build_annotation_vector_db.py`)
- Lit `prune-codes/annotations_train_pruned.parquet` et `prune-codes/suggester_pruned.parquet`

### classify-regex

Codification des libellés produits par approche regex.

- Code dans [`classify-regex/`](./classify-regex/) (ex-repo `regex_codif`)

### classify-lcs

Codification des libellés produits par approche LCS (Longest Common Subsequence) en R.

- Code dans [`classify-lcs/`](./classify-lcs/)

### classify-rag-notices

Code le jeu à coder via un RAG sur les **notices** de la nomenclature COICOP.

- Code dans [`rag-notices/`](./rag-notices/) (`scripts/2_run_rag.py`)
- Récupère les notices proches depuis Qdrant, génère via VLLM (`VLLM_GENERATION_URL`)
- Métriques MLflow (`MLFLOW_TRACKING_URI`), traces Langfuse (`LANGFUSE_BASE_URL`)

### classify-rag-annotations

Code le jeu à coder via un RAG sur des **exemples déjà annotés** (few-shot).

- Code dans [`rag-annotations/`](./rag-annotations/) (`scripts/1_run_rag.py`)
- Récupère les annotations proches depuis Qdrant ; éval (accuracy/recall) opt-in via le mode

### classify-ttc

Prédictions TTC via un classifieur pré-entraîné.

- Code dans [`classify-ttc/`](./classify-ttc/) (ex-repo `coicop_bdf_classifier`, bientôt archivé en amont)
- L'étape argo utilise actuellement l'image pré-construite `ghcr.io/micedre/coicop_bdf_classifier:latest`
- Étape destinée à être supprimée à terme

### reconcile-llm

Arbitrage final des prédictions par un LLM-as-judge : fusionne les sorties de `classify-lcs`, `classify-rag-notices`, `classify-rag-annotations` et `classify-ttc` et sélectionne le meilleur code COICOP par observation.

- Code dans [`reconcile-llm/`](./reconcile-llm/) (`uv run main.py reconcile-llm`)
- Entrées :
  - `s3://.../classify-lcs/raw_test_LCS.parquet`
  - `s3://.../classify-rag-notices/predictions.parquet`
  - `s3://.../rag-annotation/predictions.parquet`
  - `s3://.../classify-ttc/predictions.parquet`
- Sortie : `s3://.../reconcile-llm/predictions.parquet`
- Utilise un endpoint OpenAI-compatible (`LLMLAB_API_KEY`, optionnellement `LLMLAB_URL` pour un backend non-OpenAI)
- Court-circuit consensus : si les trois sources convergent (et que la confiance TTC ≥ 0.90), aucune requête LLM n'est émise
- Filtrage de nomenclature : seules les sections COICOP pertinentes sont envoyées au prompt (réduction ×4–10 du nombre de tokens)
- Reprise automatique : relancer l'étape avec le même `run_id`/`run_date` reprend les observations non traitées depuis le fichier de sortie existant

### report *(opt-in)*

Rapport d'exactitude Quarto (HTML auto-contenu) sur la sortie de `reconcile-llm`.

- Code dans [`report/`](./report/) — Quarto + Python (pandas, duckdb, matplotlib, seaborn)
- Déclenché par `skip-report=false` ; désactivé par défaut
- Entrée : `s3://.../reconcile-llm/predictions.parquet`
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
| [`build-datasets/`](./build-datasets/) | `construction-dataset` | Étape `build-datasets` |
| [`classify-regex/`](./classify-regex/) | `regex_codif` | Étape `classify-regex` |
| [`prune-codes/`](./prune-codes/) | — | Étape `prune-codes` (pruning unifié) |
| [`rag-notices/`](./rag-notices/) | `coicop-rag` | Étapes `index-notices`, `classify-rag-notices` |
| [`rag-annotations/`](./rag-annotations/) | — | Étapes `index-annotations`, `classify-rag-annotations` |
| [`classify-lcs/`](./classify-lcs/) | `stats-annotations` | Étape `classify-lcs` (R) |
| [`classify-ttc/`](./classify-ttc/) | `coicop_bdf_classifier` | Étape `classify-ttc` |
| [`reconcile-llm/`](./reconcile-llm/) | — | Étape `reconcile-llm` (arbitrage LLM) |
| [`export-results/`](./export-results/) | — | Étape `export-results` (livrable utilisateur) |
| [`report/`](./report/) | — | Rapport Quarto d'exactitude (étape `report`, opt-in) |

Chaque sous-dossier Python conserve son propre `pyproject.toml` (ses dépendances lui
appartiennent), mais tous sont membres d'un même **workspace `uv`**.

## Environnement Python

Un `pyproject.toml` à la racine déclare les 10 modules Python comme membres d'un workspace, ce
qui donne **un seul `uv.lock`** pour tout le dépôt : une seule version de `pandas`, `duckdb`,
`pyarrow`… partagée par toutes les étapes. C'est nécessaire parce que les étapes se passent des
Parquet : avec un lock par module, `classify-regex` écrivait en pandas 3 ce que `classify-ttc` relisait
en pandas 2.

```bash
# Travailler sur un module : n'installe que SES dépendances, aux versions du lock racine
cd prune-codes/ && uv sync --locked
uv run scripts/main.py --help

# Ajouter une dépendance à un module, depuis n'importe où dans le dépôt
uv add --package prune polars

# Faire monter un paquet pour TOUT le dépôt (le lock est commun)
uv lock --upgrade-package duckdb
```

L'environnement (`.venv`) vit à la racine du workspace et est reconstruit par chaque `uv sync` :
enchaîner deux modules est normal et rapide. `--locked` fait échouer la commande si le lock ne
correspond plus aux `pyproject.toml` — c'est ce que fait le pipeline, plutôt que de re-résoudre
les dépendances en silence au démarrage d'une étape.

`classify-lcs/` (R) n'est pas concerné : ses dépendances sont installées par `R/main.R`.

Python ≥ 3.13 partout (`.python-version` à la racine).

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
LLMLAB_API_KEY, LLMLAB_URL   # requis pour reconcile-llm (LLMLAB_URL optionnel)
```

### Avec la CLI Argo

```bash
# Pipeline complet (mode prod si input_file est fourni dans le YAML, sinon éval)
argo submit argo/codif-pipeline.yaml

# Lancer en production sur un fichier d'observations à coder
argo submit argo/codif-pipeline.yaml \
  -p input_file=s3://.../workflow_inputs/mon_fichier.csv \
  -p text_column=NAT_DEP -p shop_column=MAG_DEP -p budget_column=MONT_DEP

# Limiter le volume pour tester (sampling centralisé à classify-regex, hérité par tous
# les classifieurs ; en prod, -p sample-observations=100)
argo submit argo/codif-pipeline.yaml -p sample-annotations=100

# Modèle spécifique pour classify-rag-notices
argo submit argo/codif-pipeline.yaml -p classify-rag-model=openai/gpt-oss-120b

# Sauter la (re)construction des vector DB (déjà construites)
argo submit argo/codif-pipeline.yaml -p skip-index=true

# Activer le rapport d'exactitude (désactivé par défaut)
argo submit argo/codif-pipeline.yaml -p skip-report=false
```

Voir aussi la fiche [`argo/argo_helper.md`](./argo/argo_helper.md) et le fichier de paramètres [`argo/params.yaml`](./argo/params.yaml).

### Paramètres disponibles

| Paramètre | Défaut | Description |
|---|---|---|
| `input_file` | *(csv BDF)* | Non vide ⇒ **production** (code ces observations) ; vide ⇒ **évaluation** (code le split test des annotations). Pilote le mode partout. |
| `sample-annotations` | *(vide)* | Plafonne la KB d'annotations indexée (vector DB). En éval, sert aussi de cap au jeu à coder. |
| `sample-observations` | *(vide)* | Plafonne le jeu à coder (production). Échantillonné **une fois** à `classify-regex`, hérité par tous les classifieurs. |
| `classify-rag-model` | `gemma4-26b-moe` | Modèle LLM pour `classify-rag-notices` / `classify-rag-annotations` |
| `skip-index` | `true` | Si `true`, saute `index-notices` et `index-annotations` |
| `reconcile-llm-model` | `gemma4-26b-moe` | Modèle LLM utilisé par `reconcile-llm` |
| `reconcile-llm-concurrency` | `5` | Nombre d'appels LLM parallèles de `reconcile-llm` |
| `skip-report` | `false` | Si `false`, génère le rapport Quarto après `export-results` |
