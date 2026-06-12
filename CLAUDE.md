# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Monorepo for the automatic COICOP codification pipeline of the INSEE Budget de Famille (BDF) survey. Orchestrated via Argo Workflows (`argo/codif-pipeline.yaml`). Each subdirectory is an independent Python (or R) module with its own `pyproject.toml` / `uv.lock`.

## Pipeline DAG

```
                                      ┌──→ create-vector-db ──────────────┐  (skippable)
   preprocessing ──┐   ┌──→ prune ────┤                                   ├──→ run-rag(-annotations) ─┐
                   └──→┤              └──→ create-vector-db-annotations ───┘                           │
                       └──→ codif-regex ─┬──→ codif-lcs ──────────────────────────────────────────────┼──→ decide-coicop ──→ final-output ──→ report
                         (→ prune)       └──→ run-ttc  ───────────────────────────────────────────────┘
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
| `coicop-bdf-classifier/` | `run-ttc`, `decide-coicop` | Python |
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

## Developing a Module

Each Python module is self-contained:

```bash
cd <module>/
uv sync          # install deps
uv run python main.py ...
```

PyPI packages are fetched through the INSEE Nexus proxy (configured per-module in `pyproject.toml [tool.uv]`). Python ≥ 3.13 required everywhere.

Inter-module data exchange goes through S3 (parquet files). The path convention is `s3://<bucket>/<run_id>/<run_date>/<step_name>/`.

## Module Architecture Notes

**`prune/`** — Étape unique de pruning (troncature niveau 4 + élagage des hiérarchies linéaires). Produit tous les artefacts prunés sous `…/{run}/prune/` (nomenclature, mapping, annotations train/test, suggester), lus par les modules RAG. `scripts/main.py`.

**`coicop-rag/`** — Two scripts (`0_create_vector_db.py`, `2_run_rag.py`) : encode la nomenclature **prunée** dans Qdrant puis fait le RAG sur notices. Vector DB uses Qdrant + VLLM embeddings. LLM generation via VLLM (OpenAI-compatible). Metrics logged to MLflow, prompts traced in Langfuse.

**`coicop-rag-annotations/`** — RAG sur exemples annotés : `0_build_annotation_vector_db.py` indexe la KB prunée (+ suggester), `1_run_rag.py` codifie l'input pruné (éval/prod via `--skip-eval`).

**`decide-coicop`** (subcommand in `coicop-bdf-classifier/`) — LLM-as-judge merging outputs from `codif-lcs`, `run-rag`, `run-rag-annotations`, `run-ttc`. Consensus short-circuit: if all three agree and TTC confidence ≥ 0.90, no LLM call is made. Supports resume: re-running with the same `run_id`/`run_date` resumes from existing output.

**`stats-annotations/`** — R scripts only; entry point is `R/main.R`.

**`coicop-bdf-classifier/`** — Has its own `CLAUDE.md` covering classifier-specific commands, architecture, and conventions.

## Required Kubernetes Secret

`secret-codif-coicop-bdf` must contain AWS credentials, VLLM endpoints (embedding + generation), Qdrant, Langfuse, MLflow, Ollama, `DDC_ENCRYPTION_KEY`, and `LLMLAB_API_KEY` (+ optional `LLMLAB_URL`). See `README.md` for the full key list.
