# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Monorepo for the automatic COICOP codification pipeline of the INSEE Budget de Famille (BDF) survey. Orchestrated via Argo Workflows (`argo/codif-pipeline.yaml`). Each subdirectory is a Python (or R) module with its own `pyproject.toml`. The Python modules are members of a single **uv workspace**: one `uv.lock` at the repo root, one resolved version per package across the whole pipeline.

## Pipeline DAG

```
                                      ┌──→ create-vector-db ──────────────┐  (skippable)
   preprocessing ──┐   ┌──→ prune ────┤                                   ├──→ run-rag(-annotations) ─┐
                   └──→┤              └──→ create-vector-db-annotations ───┘                           │
                       └──→ codif-regex ─┬──→ codif-lcs ──────────────────────────────────────────────┼──→ CONCILIATION ──→ final-output ──→ report
                         (→ prune)       └──→ run-ttc  ───────────────────────────────────────────────┘        │
                                                                                                       ┌───────┴────────┐
                                                                              (paramètre conciliation) │                │
                                                                                  decide-coicop (llm)  │   sirus-predict (sirus)
                                                                                   — les deux sont exclusifs —
```

Le pruning (troncature niv.4 + élagage des hiérarchies linéaires) est centralisé dans le module `prune/` : une étape unique, après `codif-regex`, qui produit tous les artefacts prunés (nomenclature, mapping, annotations train/test, suggester) sous `…/{run}/prune/`. L'aval ne fait que lire.

| Module | Argo step | Language |
|---|---|---|
| `preprocessing/` | `preprocessing` | Python |
| `prune/` | `prune` | Python |
| `coicop-rag/` | `create-vector-db`, `run-rag` | Python |
| `coicop-rag-annotations/` | `create-vector-db-annotations`, `run-rag-annotations` | Python |
| `regex-codif/` | `codif-regex` | Python |
| `stats-annotations/` | `codif-lcs` | R |
| `codif-ttc/` | `run-ttc` | Python |
| `decide-coicop/` | `decide-coicop` | Python |
| `sirus/` | `sirus-predict` (entraînement hors pipeline) | Python + R |
| `report/` | `report` | Python + Quarto |
| `final-output/` | `final-output` | Python |

## Running the Pipeline

```bash
# Full pipeline
argo submit argo/codif-pipeline.yaml

# Test run on a sample (sampling is centralized at codif-regex and inherited by
# every classifier; in eval, sample-annotations also caps the to-codify split)
argo submit argo/codif-pipeline.yaml -p sample-annotations=100

# Skip vector DB rebuild (already built)
argo submit argo/codif-pipeline.yaml -p skip-vector-db=true

# Enable accuracy report (off by default)
argo submit argo/codif-pipeline.yaml -p skip-report=false
```

Key pipeline parameters:
- `input_file` — non-empty = **production** (codifies these observations), empty = **evaluation** (codifies the annotation test split). Drives the prod/eval mode everywhere.
- `sample-annotations` — cap the annotation KB indexed in the vector DB.
- `sample-observations` — cap the to-codify set (production only); sampled once at `codif-regex` so all classifiers share the same rows. In eval, `sample-annotations` is used instead.
- `model-name` (LLM for run-rag), `decide-model` (default `gemma4-26b-moe`), `decide-concurrency` (default `5`), `skip-vector-db`, `skip-report`.
- `conciliation` — `llm` (default, `decide-coicop`) or `sirus` (`sirus-predict`). **Mutually exclusive**: the other step is skipped via `when:`, and `final-output`/`report` depend on both (legacy `dependencies:` tolerates a Skipped node).
- `sirus-model-uri` — MLflow artifact URI, required when `conciliation: sirus`. Training happens **outside the pipeline** (`cd sirus/ && ./train.sh <date>/<run_id>`), so the model can never come from the run it scores — train-on-test is impossible by construction (same pattern as `ttc-model-uri`).
- `sirus-predict` applies **no threshold**: it emits `sirus_code` + `sirus_proba` per product and nothing else. Deciding what score is good enough to skip review is a business call, informed by the "Calibration de SIRUS" section of the evaluation report.

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

**`prune/`** — Étape unique de pruning (troncature niveau 4 + élagage des hiérarchies linéaires). Produit tous les artefacts prunés sous `…/{run}/prune/` (nomenclature, mapping, annotations train/test, suggester), lus par les modules RAG. `scripts/main.py`.

**`coicop-rag/`** — Two scripts (`0_create_vector_db.py`, `2_run_rag.py`) : encode la nomenclature **prunée** dans Qdrant puis fait le RAG sur notices. Vector DB uses Qdrant + VLLM embeddings. LLM generation via VLLM (OpenAI-compatible). Metrics logged to MLflow, prompts traced in Langfuse.

**`coicop-rag-annotations/`** — RAG sur exemples annotés : `0_build_annotation_vector_db.py` indexe la KB prunée (+ suggester), `1_run_rag.py` codifie l'input pruné (éval/prod via `--skip-eval`).

**`decide-coicop/`** — Module autonome : LLM-as-judge fusionnant les sorties de `codif-lcs`, `run-rag`, `run-rag-annotations`, `run-ttc`. Normalise d'abord les codes LCS/TTC (troncature niv.4 + élagage, via `prune`, dépendance path). Consensus short-circuit : si les quatre s'accordent et que la confiance TTC ≥ 0.90, aucun appel LLM. Reprise supportée : relancer avec le même `run_id`/`run_date` reprend depuis la sortie existante. Entrée `main.py decide-coicop`.

**`sirus/`** — Conciliation alternative au juge LLM, par règles interprétables (SIRUS). Une seule étape Argo, `sirus-predict` (**Python pur** : le modèle est une liste de règles en JSON et le scoring une moyenne, donc pas de R ni de compilation en production). L'**entraînement est hors pipeline**, comme celui de `codif-ttc/` : `cd sirus/ && ./train.sh <date>/<run_id>` enchaîne la construction de la table candidat-level (Python, réutilisant `decide_coicop.load_all_observations`), l'ajustement (R) et les mesures + log MLflow (Python). L'équivalence Python ↔ `sirus.predict` est prouvée par un test golden bit-exact et re-vérifiée à chaque entraînement. Voir `sirus/README.md` — en particulier sur l'exploitation du score, dont la plage atteignable est une propriété du modèle et non du problème.

**`stats-annotations/`** — R scripts only; entry point is `R/main.R`.

**`codif-ttc/`** — Classifieur neuronal COICOP (torchtextclassifiers : hierarchical/multihead/basic, train/predict/serve ; étape `run-ttc` via `predict-basic`). A son propre `CLAUDE.md`.

## Required Kubernetes Secret

`secret-codif-coicop-bdf` must contain AWS credentials, VLLM endpoints (embedding + generation), Qdrant, Langfuse, MLflow, Ollama, `DDC_ENCRYPTION_KEY`, and `LLMLAB_API_KEY` (+ optional `LLMLAB_URL`). See `README.md` for the full key list.
