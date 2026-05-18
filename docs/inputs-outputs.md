# Inputs et outputs du pipeline

Ce document liste, pour chaque étape du workflow Argo (`argo/pipeline.yaml`), les sources de données lues et les résultats écrits. Il est reconstitué à partir des fichiers de configuration (`*/config*.yaml`) et des scripts.

## Convention de stockage

Depuis l'harmonisation des I/O, **chaque exécution du workflow écrit dans un dossier dédié** :

```
s3://projet-budget-famille/data/workflow_runs/{run_date}/{run_id}/<step>/
```

- `run_id` = `{{workflow.name}}` (ex : `pipeline-abc12`) — identifiant unique de l'exécution Argo.
- `run_date` = `{{workflow.creationTimestamp.Y}}-{{...m}}-{{...d}}` — date de soumission du workflow (`YYYY-MM-DD`).

Les deux valeurs sont calculées au lancement du workflow et propagées à chaque étape via les paramètres Argo `run_id` / `run_date`, eux-mêmes passés aux scripts via les flags `--run-id` / `--run-date`. Les chemins dans les fichiers de config contiennent des placeholders `{run_id}` / `{run_date}` qui sont substitués à l'exécution par la helper `expand_paths()` (présente dans `regex-codif/src/utils/load_config.py` et `coicop-rag/src/coicop_rag/utils.py`).

## Environnement de stockage

### S3 / MinIO

- **Bucket** : `projet-budget-famille`
- **Endpoint** : `minio.lab.sspcloud.fr` (via env `AWS_S3_ENDPOINT` / `AWS_ENDPOINT_URL`)
- **Credentials** : injectés par le secret Kubernetes `secret-codif-coicop-bdf` (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`)

### Qdrant (base vectorielle, **partagée entre runs**)

- **URL** : env `QDRANT_URL` + `QDRANT_API_PORT`
- **Collection** : `coicop_lineage` (définie dans `coicop-rag/config/config.yaml`, clé `qdrant.collection_name`) — une seule collection, recréée par `create-vector-db`.

### MLflow / Langfuse / vLLM / Ollama

Env injectés depuis le secret : `MLFLOW_TRACKING_URI` / `MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD`, `LANGFUSE_*`, `VLLM_EMBEDDING_URL` / `VLLM_EMBEDDING_API_KEY`, `VLLM_GENERATION_URL` / `VLLM_GENERATION_API_KEY`, `OLLAMA_URL` / `OLLAMA_API_KEY`. Expérience MLflow par défaut : `test` (clé `mlflow.experiment_name`).

---

## Arborescence type d'un run

```
s3://projet-budget-famille/data/workflow_runs/2026-04-21/pipeline-abc12/
├── preprocessing/
│   ├── raw_train.parquet
│   ├── raw_test.parquet
│   └── qa/
│       ├── annotations_with_multiple_codes.parquet
│       └── annotations_hors_copain_with_multiple_codes.parquet
├── codif-regex/
│   ├── REGEX_pred.parquet
│   ├── raw_train_without_regex.parquet
│   ├── raw_test_without_regex.parquet
│   └── error_regex_pred.parquet
├── codif-lcs/
│   ├── raw_test_LCS.parquet
│   ├── analyse_codif_LCS.parquet
│   └── eval/
│       └── <tab_acc_*>.parquet          (tables générées par export_liste_df)
├── prune-annotations/
│   └── raw_test_without_regex_pruned.parquet
├── run-rag/
│   ├── predictions.parquet
│   └── retrieved_codes.parquet
└── run-ttc/
    └── predictions.parquet
```

## Étape par étape

### 1. `preprocessing`

Code : [`preprocessing/`](../preprocessing/) · config : [`preprocessing/config.yaml`](../preprocessing/config.yaml)

**Lit** (sources statiques, chemins inchangés — tout sur `s3://projet-budget-famille/`) :

| Variable config | Chemin S3 |
|---|---|
| `paths.copain_annotations` | `data/output-annotation/` |
| `paths.additional_annotations` | `data/codification-manuelle-anterieure/annotations_test_2024.csv` |
| `paths.old_annotations` | `data/codification-manuelle-anterieure/annotations_BDF_2017.csv` |
| `paths.app_annotations` | `data/input-annotation/liste_produits_fr_copain.csv` |
| `paths.shops_mapping` | `data/input-annotation/liste_magasins.csv` |
| `paths.uncodable_products` | `data/codification-manuelle-anterieure/produit_non_annotable.csv` |

**Écrit** — sous `{output_root} = s3://projet-budget-famille/data/workflow_runs/{run_date}/{run_id}/preprocessing` :

- `<output_root>/raw_train.parquet`
- `<output_root>/raw_test.parquet`
- `<output_root>/qa/annotations_with_multiple_codes.parquet`
- `<output_root>/qa/annotations_hors_copain_with_multiple_codes.parquet`

---

### 2. `codif-regex`

Code : [`regex-codif/`](../regex-codif/) · config : [`regex-codif/config/config.yaml`](../regex-codif/config/config.yaml) · règles : [`regex-codif/config/rules.yaml`](../regex-codif/config/rules.yaml)

**Lit** (sortie de `preprocessing` du même run) :

| Variable config | Chemin S3 (après expansion) |
|---|---|
| `paths.train_set` | `…/workflow_runs/{run_date}/{run_id}/preprocessing/raw_train.parquet` |
| `paths.test_set`  | `…/workflow_runs/{run_date}/{run_id}/preprocessing/raw_test.parquet` |

**Écrit** (`{run}` = `…/workflow_runs/{run_date}/{run_id}/codif-regex`) :

| Variable config | Chemin S3 |
|---|---|
| `paths.output.pred` | `<run>/REGEX_pred.parquet` |
| `paths.output.test_set` | `<run>/raw_test_without_regex.parquet` |
| `paths.output.train_set` | `<run>/raw_train_without_regex.parquet` |
| `paths.error_pred` | `<run>/error_regex_pred.parquet` |

---

### 3. `codif-lcs`

Code : [`stats-annotations/`](../stats-annotations/) — pas de fichier de config YAML : les chemins sont construits dans [`R/main.R`](../stats-annotations/R/main.R) à partir des arguments `--run-id=<id>` / `--run-date=<YYYY-MM-DD>` passés à `Rscript`.

**Lit** (sorties de `codif-regex` du même run) :

| Constante R | Chemin S3 |
|---|---|
| `path` | `…/workflow_runs/{run_date}/{run_id}/codif-regex/raw_test_without_regex.parquet` |
| `sug_path` | `…/workflow_runs/{run_date}/{run_id}/codif-regex/raw_train_without_regex.parquet` (filtré `source = 'suggester'`) |

**Écrit** (paths construits à partir de `lcs_output_dir = data/workflow_runs/{run_date}/{run_id}/codif-lcs`) :

| Chemin S3 | Contenu |
|---|---|
| `<lcs_output_dir>/raw_test_LCS.parquet` | Comparaison test set vs règles LCS |
| `<lcs_output_dir>/analyse_codif_LCS.parquet` | Table d'analyse qualité |
| `<lcs_output_dir>/eval/<name>.parquet` | Tables d'évaluation (via `export_liste_df` dans `fonctions.R`) |

---

### 4. `prune-coicop` *(skippable, paths partagés)*

Code : [`coicop-rag/scripts/0_prunning_coicop.py`](../coicop-rag/scripts/0_prunning_coicop.py) · config : [`coicop-rag/config/config.yaml`](../coicop-rag/config/config.yaml)

Cette étape ne dépend pas du run en cours : elle transforme la nomenclature COICOP brute, qui change rarement. Ses sorties sont donc **partagées entre tous les runs** (paths stables).

**Lit** :

| Variable config | Chemin S3 |
|---|---|
| `coicop.path_raw` | `data/coicop-2018_envoi_rmes_20251022.csv` |

**Écrit** (paths stables) :

| Variable config | Chemin S3 |
|---|---|
| `coicop.path_prunned_lvl4` | `data/workflow_outputs/prunning/coicop-2018_envoi_rmes_20251022_prunned_lvl4.parquet` |
| `coicop.path_mapping_lvl4` | `data/workflow_outputs/prunning/coicop-2018_envoi_rmes_20251022_mapping_prunning_lvl4.parquet` |

---

### 5. `prune-annotations`

Code : [`coicop-rag/scripts/1_prune_annotations.py`](../coicop-rag/scripts/1_prune_annotations.py) · config : [`coicop-rag/config/config.yaml`](../coicop-rag/config/config.yaml)

**Lit** :

| Variable config | Chemin S3 |
|---|---|
| `annotations.s3_path` | `…/workflow_runs/{run_date}/{run_id}/codif-regex/raw_test_without_regex.parquet` (sortie de `codif-regex`) |
| `coicop.path_mapping_lvl4` | `data/workflow_outputs/prunning/…_mapping_prunning_lvl4.parquet` (sortie partagée de `prune-coicop`) |

**Écrit** :

| Variable config | Chemin S3 |
|---|---|
| `annotations.s3_path_rag` | `…/workflow_runs/{run_date}/{run_id}/prune-annotations/raw_test_without_regex_pruned.parquet` |

---

### 6. `create-vector-db` *(skippable, sorties partagées)*

Code : [`coicop-rag/scripts/0_create_vector_db.py`](../coicop-rag/scripts/0_create_vector_db.py) · config : [`coicop-rag/config/config.yaml`](../coicop-rag/config/config.yaml)

Comme `prune-coicop`, ses entrées et sorties sont stables (la collection Qdrant est partagée).

**Lit** : `coicop.path_mapping_lvl4` (partagé).

**Appelle** : vLLM embedding (`VLLM_EMBEDDING_URL`, modèle `Qwen/Qwen3-Embedding-8B`, dim 4096).

**Écrit** : collection Qdrant `coicop_lineage` (recréée à chaque exécution).

---

### 7. `run-rag`

Code : [`coicop-rag/scripts/2_run_rag.py`](../coicop-rag/scripts/2_run_rag.py) · config : [`coicop-rag/config/config.yaml`](../coicop-rag/config/config.yaml)

**Lit** :

| Variable config | Chemin S3 |
|---|---|
| `annotations.s3_path_rag` | `…/workflow_runs/{run_date}/{run_id}/prune-annotations/raw_test_without_regex_pruned.parquet` (sortie de `prune-annotations`) |

**Appelle** : Qdrant `coicop_lineage` (top-k = `retrieval.size` = 10), vLLM génération, Langfuse, MLflow (expérience `test`).

**Écrit** :

| Variable config | Chemin S3 |
|---|---|
| `predictions.s3_path` | `…/workflow_runs/{run_date}/{run_id}/run-rag/predictions.parquet` |
| `predictions.s3_path_retrieved_codes` | `…/workflow_runs/{run_date}/{run_id}/run-rag/retrieved_codes.parquet` |

Plus de suffixe `{timestamp}` dans les noms : le `run_id` rend déjà les fichiers uniques entre runs.

Paramètres CLI surchargeables : `--sample_size`, `--model_name`, `--temperature`, `--max_tokens`, `--retrieval_size`, `--collection_name`, `--threshold_confidence`, `--experiment_name`.

---

### 8. `run-ttc`

Code : [`coicop-bdf-classifier/`](../coicop-bdf-classifier/) (source vendorisée). L'étape Argo utilise encore l'image pré-construite `ghcr.io/micedre/coicop_bdf_classifier:latest` ; les chemins sont définis **dans `argo/pipeline.yaml`** via des placeholders Argo.

**Lit** :

| Argument CLI | Chemin |
|---|---|
| `--model` | `mlflow-artifacts:/10/cacf2603514b4887bbfb77e2654c9bc1/artifacts/model` |
| `--file` | `s3://projet-budget-famille/data/workflow_runs/{{inputs.parameters.run_date}}/{{inputs.parameters.run_id}}/codif-regex/raw_test_without_regex.parquet` |

**Écrit** :

| Argument CLI | Chemin |
|---|---|
| `--output` | `s3://projet-budget-famille/data/workflow_runs/{{inputs.parameters.run_date}}/{{inputs.parameters.run_id}}/run-ttc/predictions.parquet` |

Prédit top-10 COICOP (`--top-k 10`) sur la colonne `l_pr_product`. `run-ttc` classifie les items que `codif-regex` n'a pas pu coder (même rôle que `run-rag` et `codif-lcs`).

---

## Graphe des données après harmonisation

```
sources statiques (data/…)                   prune-coicop ──► data/workflow_outputs/prunning/ ──┐
       │                                                                                        │
       ▼                                                                                        │
preprocessing ──► <run>/preprocessing/                                                          │
       │                                                                                        │
       └──► codif-regex ──► <run>/codif-regex/ ──┬──► prune-annotations ──► <run>/prune-annotations/ ──► run-rag ──► <run>/run-rag/
                                                 │
                                                 ├──► codif-lcs ──► <run>/codif-lcs/
                                                 │
                                                 └──► run-ttc ──► <run>/run-ttc/
                                                                                                │
                                                                     create-vector-db ◄─────────┘
                                                                         │
                                                                         ▼
                                                                 Qdrant "coicop_lineage" ──► utilisé par run-rag
```

`<run>` = `s3://projet-budget-famille/data/workflow_runs/{run_date}/{run_id}/`.
