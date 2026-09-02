# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Monorepo for the automatic COICOP codification pipeline of the INSEE Budget de Famille (BDF) survey. Orchestrated via Argo Workflows (`argo/codif-pipeline.yaml`). Each subdirectory is a Python (or R) module with its own `pyproject.toml`. The Python modules are members of a single **uv workspace**: one `uv.lock` at the repo root, one resolved version per package across the whole pipeline.

## Pipeline DAG

**Three separate Argo workflows.** The expensive vector-DB indexing is built **outside** the
classification pipeline and passed in by name — the same arrangement as `classify-ttc` and
`reconcile-sirus` training.

```
① argo/index-notices-pipeline.yaml
   prune-codes (--only nomenclature) ─→ index-notices        → collection Qdrant, nom unique

② argo/index-annotations-pipeline.yaml
   build-datasets ─→ prune-codes (--only kb) ─→ index-annotations   → collection Qdrant, nom unique

        └── les deux noms sont recopiés dans argo/params.yaml ──┐
                                                                ▼
③ argo/codif-pipeline.yaml   (input_file : production / évaluation)
build-datasets
  └─→ classify-regex ─┬─→ classify-lcs ────────────────────────────────────┐
                      ├─→ classify-ttc ────────────────────────────────────┤
                      └─→ prune-codes ─┬─→ classify-rag-notices ───────────┤
                                       └─→ classify-rag-annotations ───────┘
                                                                           │
                                       ┌───────────────────────────────────┘
                                       │  les 4 classifieurs convergent
                                       └─→ reconcile-llm  OU  reconcile-sirus   (exclusifs : paramètre `reconciliation`)
                                               └─→ export-results ─→ report  (skip-report)
```

Le pruning (troncature niv.4 + élagage des hiérarchies linéaires) est centralisé dans le module `prune-codes/` : une étape unique, après `classify-regex`, qui produit tous les artefacts prunés (nomenclature, mapping, annotations train/test, suggester) sous `…/{run}/prune-codes/`. L'aval ne fait que lire. Le drapeau `--only {all,nomenclature,kb}` restreint son périmètre pour les pipelines d'indexation.

| Module | Argo step | Workflow | Language |
|---|---|---|---|
| `build-datasets/` | `build-datasets` | ② et ③ | Python |
| `prune-codes/` | `prune-codes` | ①, ② et ③ | Python |
| `rag-notices/` | `index-notices` | ① | Python |
| `rag-notices/` | `classify-rag-notices` | ③ | Python |
| `rag-annotations/` | `index-annotations` | ② | Python |
| `rag-annotations/` | `classify-rag-annotations` | ③ | Python |
| `classify-regex/` | `classify-regex` | ③ | Python |
| `classify-lcs/` | `classify-lcs` | ③ | R |
| `classify-ttc/` | `classify-ttc` (entraînement hors pipeline) | ③ | Python |
| `reconcile-llm/` | `reconcile-llm` | ③ | Python |
| `reconcile-sirus/` | `reconcile-sirus` (entraînement hors pipeline) | ③ | Python + R |
| `report/` | `report` | ③ | Python + Quarto |
| `export-results/` | `export-results` | ③ | Python |

## Running the Pipeline

**Build the vector DBs first** (once; they are reused across classification runs):

```bash
argo submit argo/index-notices-pipeline.yaml --watch
argo submit argo/index-annotations-pipeline.yaml --watch
```

Each prints, at the end, the exact line to paste into `argo/params.yaml`:

```
classify-rag-notices-collection: coicop_notices__2026-09-02__index-notices-a7k2p
classify-rag-annotations-collection: coicop_annotations__full__2026-09-02__index-annotations-b3x9q
```

Then the classification pipeline:

```bash
# Full pipeline (collection names come from params.yaml)
argo submit argo/codif-pipeline.yaml --parameter-file argo/params.yaml

# Test run on a sample: sampling is centralized at classify-regex and inherited
# by every classifier, so all four codify exactly the same rows
argo submit argo/codif-pipeline.yaml --parameter-file argo/params.yaml -p sample-observations=100

# Disable the accuracy report (on by default in the YAML)
argo submit argo/codif-pipeline.yaml --parameter-file argo/params.yaml -p skip-report=true
```

Key pipeline parameters:
- `input_file` — non-empty = **production** (codifies these observations), empty = **evaluation** (codifies the annotation test split). Drives the prod/eval mode everywhere.
- `classify-rag-notices-collection` / `classify-rag-annotations-collection` — **required**, no default. Qdrant collections produced by workflows ① and ②. Argo has no required-parameter mechanism, so the guard is written twice: a `[ -z ] && exit 1` in the container script *and* `required=True` in argparse. An unset name must fail in seconds, not silently fall back to some other run's index.
- `sample-observations` — cap the to-codify set, in **both** modes; sampled once at `classify-regex`. To cap the indexed KB instead, that is `kb-sample-size` of workflow ②.
- `classify-rag-model` (LLM for classify-rag-notices), `reconcile-llm-model` (default `gemma4-26b-moe`), `reconcile-llm-concurrency` (default `5`), `skip-report`.
- `reconciliation` — `llm` (default, `reconcile-llm`) or `sirus` (`reconcile-sirus`). **Mutually exclusive**: the other step is skipped via `when:`, and `export-results`/`report` depend on both (legacy `dependencies:` tolerates a Skipped node).
- `reconcile-sirus-model-uri` — MLflow artifact URI, required when `reconciliation: sirus`. Training happens **outside the pipeline** (`cd reconcile-sirus/ && ./train.sh <date>/<run_id>`), so the model can never come from the run it scores — train-on-test is impossible by construction (same pattern as `classify-ttc-model-uri`).
- `reconcile-sirus` applies **no threshold**: it emits `sirus_code` + `sirus_proba` per product and nothing else. Deciding what score is good enough to skip review is a business call, informed by the "Calibration de SIRUS" section of the evaluation report.

## Developing a Module

The repo is a **uv workspace**: one `pyproject.toml` per module for its own dependencies, but
a single `uv.lock` at the root — so every step runs the same pandas/duckdb/pyarrow. Syncing from
a module directory installs only that module's dependencies, into the shared `.venv` at the
workspace root:

```bash
cd <module>/
uv sync --locked   # only this module's deps, versions from the root lock
uv run python main.py ...

uv add --package <module> <pkg>      # add a dependency, from anywhere in the repo
uv lock --upgrade-package <pkg>      # bump a package for the whole repo
```

`--locked` fails if the lock no longer matches the `pyproject.toml` files instead of silently
re-resolving; the Argo steps use `uv sync --locked --no-dev`.

**pandas stays on 2.x** and it is not a preference: `mlflow` declares `pandas<3`, and five
modules depend on mlflow. The reason is documented in the root `pyproject.toml`; the day mlflow
supports pandas 3, `uv lock --upgrade-package pandas` moves the whole repo at once.

No package index is configured in the repo — uv resolves against PyPI directly. Python ≥ 3.13
required everywhere (`.python-version` at the root).

Inter-module data exchange goes through S3 (parquet files). The path convention is `s3://<bucket>/<run_id>/<run_date>/<step_name>/`.

## Module Architecture Notes

**`prune-codes/`** — Étape unique de pruning (troncature niveau 4 + élagage des hiérarchies linéaires). Produit tous les artefacts prunés sous `…/{run}/prune-codes/` (nomenclature, mapping, annotations train/test, suggester), lus par les modules RAG. `scripts/main.py`.

**`rag-notices/`** — Two scripts, now in **two different workflows** : `0_create_vector_db.py` (workflow ①) encode la nomenclature **prunée** dans Qdrant ; `2_run_rag.py` (workflow ③) fait le RAG sur notices contre une collection existante, dont le nom lui est passé par `--collection_name` (obligatoire). Embeddings et génération via VLLM (OpenAI-compatible), métriques MLflow, prompts tracés dans Langfuse.

**`rag-annotations/`** — RAG sur exemples annotés, également scindé : `0_build_annotation_vector_db.py` (workflow ②) indexe la KB — `annotations_full` + suggester au sens de `build-datasets`, **sans filtrage regex** — avec `--kb-scope {full,train}` ; `1_run_rag.py` (workflow ③) codifie l'input pruné, avec `--collection-name` obligatoire et `--skip-eval` pour le calcul des métriques.

### Vector DBs (workflows ① et ②)

Les collections **ne portent plus de nom fixe partagé**. Chaque indexation en crée une nouvelle :

```
{base}[__{kb_scope}]__{run_date}__{run_id}[__sampleN]
coicop_notices__2026-09-02__index-notices-a7k2p
coicop_annotations__full__2026-09-02__index-annotations-b3x9q
```

Auparavant, `coicop_lineage` et `coicop_annotations_without_copain_2017` étaient partagées par tous les runs et **détruites puis recréées** à chaque indexation : une réindexation cassait la base que lisait un run concurrent, et le défaut `skip-index=true` faisait coder contre une vector DB de provenance inconnue.

Deux conséquences pour qui touche à ce code :

- La clé de config s'appelle `qdrant.collection_base`, **pas** `collection_name`. Le renommage est délibéré : le même `config.yaml` est lu par le constructeur et par le consommateur. Une clé `collection_name` y ferait retomber le consommateur en silence sur une collection périmée ; et côté constructeur, `0_create_vector_db.py` lisait le nom à deux endroits — en oublier un aurait déversé les points dans l'ancienne collection tout en créant la nouvelle, vide.
- Chaque collection est accompagnée d'un **manifeste** JSON sous `s3://…/data/vector_db_manifests/{collection_name}.json` (modèle d'embedding, dimension, stratégie, `kb_scope`, nombre de points, sha git). Les étapes `classify-rag-*` le relisent **avant `mlflow.set_experiment`** pour valider la collection : échouer à l'intérieur d'un `start_run` laisserait un run FAILED qui pollue l'expérience. Le nom seul ne peut pas porter ces informations — deux collections de même dimension bâties avec des stratégies différentes sont indistinguables, et interroger la mauvaise ne lève aucune erreur.

`kb-scope=train` est **transitoire** : le split train/test n'existait que faute de jeu de test indépendant. Les nouveaux produits annotés en fournissent un, donc toute la base historique peut servir de KB (`full`, le défaut). L'option disparaîtra.

**`reconcile-llm/`** — Module autonome : LLM-as-judge fusionnant les sorties de `classify-lcs`, `classify-rag-notices`, `classify-rag-annotations`, `classify-ttc`. Normalise d'abord les codes LCS/TTC (troncature niv.4 + élagage, via `prune-codes`, dépendance path). Consensus short-circuit : si les quatre s'accordent et que la confiance TTC ≥ 0.90, aucun appel LLM. Reprise supportée : relancer avec le même `run_id`/`run_date` reprend depuis la sortie existante. Entrée `main.py reconcile-llm`.

**`reconcile-sirus/`** — Conciliation alternative au juge LLM, par règles interprétables (SIRUS). Une seule étape Argo, `reconcile-sirus` (**Python pur** : le modèle est une liste de règles en JSON et le scoring une moyenne, donc pas de R ni de compilation en production). L'**entraînement est hors pipeline**, comme celui de `classify-ttc/` : `cd reconcile-sirus/ && ./train.sh <date>/<run_id>` enchaîne la construction de la table candidat-level (Python, réutilisant `reconcile_llm.load_all_observations`), l'ajustement (R) et les mesures + log MLflow (Python). L'équivalence Python ↔ `sirus.predict` est prouvée par un test golden bit-exact et re-vérifiée à chaque entraînement. Voir `reconcile-sirus/README.md` — en particulier sur l'exploitation du score, dont la plage atteignable est une propriété du modèle et non du problème.

**`classify-lcs/`** — R scripts only; entry point is `R/main.R`.

**`classify-ttc/`** — Classifieur neuronal COICOP (torchtextclassifiers : hierarchical/multihead/basic, train/predict/serve ; étape `classify-ttc` via `predict-basic`). A son propre `CLAUDE.md`.

## Argo gotchas (hard-won — do not "clean up")

**`git -c http.version=HTTP/1.1 clone` on all 17 clone sites.** The image's git (2.54.0, linked
against libcurl3-gnutls) cannot parse GitHub's HTTP/2 ref advertisement: it fails on `expected
flush after ref listing`, then asks for a Username, which looks exactly like an authentication
problem and is not one — GitHub answers `200` with the correct content-type (verified under
`GIT_CURL_VERBOSE`). Forcing HTTP/1.1 fixes the transport while keeping git protocol v2. The
`curl` binary in the same image is OpenSSL-based and works, which makes the diagnosis
misleading.

**Never write `git clone … && cd …` under `set -e`.** POSIX exempts every command of an `&&`
list except the last, so a failed clone does **not** abort the script: execution continues in
the wrong directory and the real error is masked by a confusing `No pyproject.toml found`. The
17 sites use two separate statements for this reason.

**Argo has no required-parameter mechanism**, and `argo submit --parameter-file` silently
accepts unknown keys — that is how `rereconciliation:` sat dead in `params.yaml` from its own
commit. Any parameter that must not be empty needs a guard in the container script *and* in
argparse.

## Required Kubernetes Secret

`secret-codif-coicop-bdf` must contain AWS credentials, VLLM endpoints (embedding + generation), Qdrant, Langfuse, MLflow, Ollama, `DDC_ENCRYPTION_KEY`, and `LLMLAB_API_KEY` (+ optional `LLMLAB_URL`). See `README.md` for the full key list.
