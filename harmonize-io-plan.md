# Harmonize pipeline I/O under a per-run folder

## Context

Today, every pipeline step has its own hardcoded S3 layout (see `docs/inputs-outputs.md`). Consequences:
- Dated paths in `regex-codif/config/config.yaml` (`2026-03-26`) and `argo/pipeline.yaml` run-ttc step (`2026-04-15`) drift out of sync with the latest `preprocessing` output and must be edited by hand.
- `stats-annotations` has paths embedded in R code — no config at all.
- Two concurrent runs can overwrite each other's outputs (same paths, no isolation).
- Reproducing a past run requires digging through S3 and inferring which files belonged together.

Goal: every run writes into a **single isolated folder** keyed by the Argo workflow name and creation date. All steps read their upstream dependencies from that same folder. Static reference data (source inputs, pruned COICOP nomenclature, Qdrant collection) stays at fixed paths.

## Decisions (from Q&A)

| Topic | Choice |
|---|---|
| Folder pattern | `s3://projet-budget-famille/data/workflow_runs/<YYYY-MM-DD>/<workflow.name>/<step>/...` |
| Scope | Per-run for pipeline outputs; **shared** for `prune-coicop` outputs (`data/workflow_outputs/prunning/`) and the Qdrant collection (`coicop_lineage`) |
| Run-id plumbing | Argo passes `run_id` + `run_date` as step params → scripts accept `--run-id` / `--run-date` CLI flags → configs hold path templates with `{run_id}` / `{run_date}` placeholders |
| Run-id source | `{{workflow.name}}` |
| Date source | `{{workflow.creationTimestamp.Y}}-{{workflow.creationTimestamp.m}}-{{workflow.creationTimestamp.d}}` |
| Qdrant collection | Single shared `coicop_lineage`, recreated by `create-vector-db` (unchanged) |

## Target S3 layout

```
s3://projet-budget-famille/
├── data/                                                 # sources + shared reference data
│   ├── output-annotation/                                # (source, unchanged)
│   ├── input-annotation/                                 # (source, unchanged)
│   ├── codification-manuelle-anterieure/                 # (source, unchanged)
│   ├── coicop-2018_envoi_rmes_20251022.csv               # (source, unchanged)
│   │
│   ├── workflow_outputs/prunning/                        # prune-coicop outputs (stable, shared)
│   │   ├── coicop-...-prunned_lvl4.parquet
│   │   └── coicop-...-mapping_prunning_lvl4.parquet
│   │
│   └── workflow_runs/                                    # per-run outputs (new)
│       └── 2026-04-21/
│           └── pipeline-abc12/
│               ├── preprocessing/
│               │   ├── raw_train.parquet
│               │   ├── raw_test.parquet
│               │   └── qa/
│               │       ├── annotations_with_multiple_codes.parquet
│               │       └── annotations_hors_copain_with_multiple_codes.parquet
│               ├── codif-regex/
│               │   ├── REGEX_pred.parquet
│               │   ├── raw_train_without_regex.parquet
│               │   ├── raw_test_without_regex.parquet
│               │   └── error_regex_pred.parquet
│               ├── codif-lcs/
│               │   ├── raw_test_LCS.parquet
│               │   └── analyse_codif_LCS.parquet
│               ├── prune-annotations/
│               │   └── raw_test_without_regex_pruned.parquet
│               ├── run-rag/
│               │   ├── predictions.parquet
│               │   └── retrieved_codes.parquet
│               └── run-ttc/
│                   └── predictions.parquet
```

Existing filenames preserved to minimize code churn. Qdrant collection name unchanged.

## Data flow after harmonization

```
sources (data/*)              prune-coicop ──► data/workflow_outputs/prunning/ ──┐
   │                                                                             │
   ▼                                                                              │
preprocessing ──► <run>/preprocessing/                                           │
                         │                                                        │
                         ▼                                                        ▼
           ┌─────► codif-regex ──► <run>/codif-regex/ ──┐                 create-vector-db ──► Qdrant "coicop_lineage"
           │                                            │                         │
           │                                            ▼                         │
           │                                  codif-lcs ──► <run>/codif-lcs/     │
           │                                                                      │
           │                                  prune-annotations ◄──────────────────
           │                                         │
           │                                         ▼
           │                                <run>/prune-annotations/
           │                                         │
           │                                         ▼
           └─────► run-rag ◄── Qdrant ────────────── ┘ ──► <run>/run-rag/
                         │
                         └ mlflow + langfuse

<run>/preprocessing/raw_test.parquet ──► run-ttc ──► <run>/run-ttc/predictions.parquet
```

## Workflow-level changes (`argo/pipeline.yaml`)

Add two top-level parameters that resolve from Argo built-ins:

```yaml
spec:
  arguments:
    parameters:
      - name: sample_size
        value: ""
      - name: model-name
        value: ""
      - name: skip-vector-db
        value: "false"
      - name: run_id                                              # NEW
        value: "{{workflow.name}}"
      - name: run_date                                            # NEW
        value: "{{workflow.creationTimestamp.Y}}-{{workflow.creationTimestamp.m}}-{{workflow.creationTimestamp.d}}"
```

Each step template that writes or reads pipeline data takes `run_id` and `run_date` as `inputs.parameters`, and passes them as `--run-id {{inputs.parameters.run_id}} --run-date {{inputs.parameters.run_date}}` in the container command. The DAG tasks add:

```yaml
arguments:
  parameters:
    - name: run_id
      value: "{{workflow.parameters.run_id}}"
    - name: run_date
      value: "{{workflow.parameters.run_date}}"
```

`prune-coicop` and `create-vector-db` do **not** need run-id (their I/O is stable), but receiving the params does no harm and keeps the template uniform. Plan: leave them unchanged (don't add args) since they don't write per-run outputs.

## Per-step changes

### 1. `preprocessing/`

- **config.yaml**: add
  ```yaml
  paths:
    output_root: "/data/workflow_runs/{run_date}/{run_id}/preprocessing"
  ```
  (drop the legacy `paths.output: "/data/output-annotation-consolidated"`).
- **main.py**: parse `--run-id` / `--run-date` via argparse; replace the `timestamp = date.today().isoformat()` + `annotations-consolidated-<ts>/` scheme with expansion of `{run_id}` / `{run_date}` in `output_root`. QA files go to `<output_root>/qa/`.
- Outputs: `<root>/raw_train.parquet`, `<root>/raw_test.parquet`, `<root>/qa/annotations_with_multiple_codes.parquet`, `<root>/qa/annotations_hors_copain_with_multiple_codes.parquet`.

### 2. `regex-codif/`

- **config/config.yaml**: replace all hardcoded paths with templates:
  ```yaml
  paths:
    train_set: "s3://projet-budget-famille/data/workflow_runs/{run_date}/{run_id}/preprocessing/raw_train.parquet"
    test_set:  "s3://projet-budget-famille/data/workflow_runs/{run_date}/{run_id}/preprocessing/raw_test.parquet"
    error_pred: "s3://projet-budget-famille/data/workflow_runs/{run_date}/{run_id}/codif-regex/error_regex_pred.parquet"
    output:
      pred:     "s3://projet-budget-famille/data/workflow_runs/{run_date}/{run_id}/codif-regex/REGEX_pred.parquet"
      test_set: "s3://projet-budget-famille/data/workflow_runs/{run_date}/{run_id}/codif-regex/raw_test_without_regex.parquet"
      train_set: "s3://projet-budget-famille/data/workflow_runs/{run_date}/{run_id}/codif-regex/raw_train_without_regex.parquet"
  ```
- **src/main.py**: add argparse for `--run-id` / `--run-date`; after `load_config()`, run a helper `expand_paths(config, run_id=..., run_date=...)` that does `.format(run_id=run_id, run_date=run_date)` on every string in the nested structure (add to `src/utils/load_config.py`).

### 3. `stats-annotations/` (codif-lcs)

- **R/main.R**: take CLI args via `commandArgs(trailingOnly = TRUE)` (expect `--run-id=...` and `--run-date=...` or positional); build input paths as `glue("s3://{BUCKET}/data/workflow_runs/{run_date}/{run_id}/codif-regex/raw_test_without_regex.parquet")` (and similarly for `raw_train_without_regex.parquet`).
- **R/calcul_lcs_libel.R**: same substitution for the two `aws.s3::s3write_using` outputs (`raw_test_LCS.parquet`, `analyse_codif_LCS.parquet`). `main.R` must pass `run_id` and `run_date` as globals (or source the file after assigning them) so they're visible in the sourced scripts.

### 4. `coicop-rag/` (prune-annotations + run-rag)

- **config/config.yaml**: template the per-run paths:
  ```yaml
  annotations:
    s3_path:     "s3://projet-budget-famille/data/workflow_runs/{run_date}/{run_id}/codif-regex/raw_test_without_regex.parquet"
    s3_path_rag: "s3://projet-budget-famille/data/workflow_runs/{run_date}/{run_id}/prune-annotations/raw_test_without_regex_pruned.parquet"
    ...
  predictions:
    s3_path:                  "s3://projet-budget-famille/data/workflow_runs/{run_date}/{run_id}/run-rag/predictions.parquet"
    s3_path_retrieved_codes:  "s3://projet-budget-famille/data/workflow_runs/{run_date}/{run_id}/run-rag/retrieved_codes.parquet"
  ```
  Keep `coicop.path_raw` / `path_prunned_lvl4` / `path_mapping_lvl4` as-is (stable paths).
- **src/coicop_rag/utils.py** (or new module): add `expand_paths(config, run_id, run_date)` helper applying `.format(...)` recursively. Invoked by `1_prune_annotations.py` and `2_run_rag.py` after loading config.
- **scripts/1_prune_annotations.py**: add argparse for `--run-id`/`--run-date` (keep existing `--config`); call `expand_paths`.
- **scripts/2_run_rag.py**: same; also **remove the `{timestamp}` substitution** in `predictions.s3_path` since `run_id` already uniquely identifies the run (no more files per launch attempt — overwrite behavior is acceptable since a workflow run is idempotent).
- **scripts/0_prunning_coicop.py** and **0_create_vector_db.py**: no change (stable paths).

### 5. `run-ttc` (argo-only)

The classifier CLI already accepts `--file` and `--output` flags. Rewrite those values in `argo/pipeline.yaml` to reference the run-scoped paths:

```yaml
- name: run-ttc
  inputs:
    parameters:
      - name: run_id
      - name: run_date
  container:
    args:
      - predict-basic
      - "--model"
      - "mlflow-artifacts:/10/cacf2603514b4887bbfb77e2654c9bc1/artifacts/model"
      - "--file"
      - "s3://projet-budget-famille/data/workflow_runs/{{inputs.parameters.run_date}}/{{inputs.parameters.run_id}}/preprocessing/raw_test.parquet"
      - "--output"
      - "s3://projet-budget-famille/data/workflow_runs/{{inputs.parameters.run_date}}/{{inputs.parameters.run_id}}/run-ttc/predictions.parquet"
      ...
```

Also switch `run-ttc`'s dependency from `codif-regex` to `preprocessing` (its real input is `raw_test.parquet` from preprocessing, not anything regex). Current DAG edge is misleading.

## Commit-sized todo

The pipeline is **not runnable between commits 2 and 6** — upstream outputs live at new paths while some downstream steps still look at old paths. This is acceptable on a feature branch; the final merge makes everything work together. Each commit is self-consistent for review and can be reverted independently.

1. **Thread `run_id` / `run_date` through `argo/pipeline.yaml`** — add top-level parameters resolved from Argo built-ins, add them as `inputs.parameters` on every per-run step template, pass them via DAG `arguments.parameters`. Scripts don't yet receive them on the command line — this commit is infrastructure-only and has no behavior change.
2. **Migrate `preprocessing/`** — argparse in `main.py`, `output_root` template in `config.yaml`, argo step passes `--run-id` / `--run-date` flags. Verify: `uv sync --frozen && uv run python main.py --help`.
3. **Migrate `regex-codif/`** — add `expand_paths()` helper in `src/utils/load_config.py`, argparse in `src/main.py`, templated `config/config.yaml`, argo step passes flags. Verify: `uv sync --frozen && uv run python src/main.py --help`.
4. **Migrate `stats-annotations/`** — `commandArgs()` parsing in `R/main.R`, path building via `glue()`, pass `run_id`/`run_date` into sourced `calcul_lcs_libel.R`, argo step passes `--run-id=...` / `--run-date=...` args to `Rscript`. Verify: syntactic grep (no R available locally).
5. **Migrate `coicop-rag/` (prune-annotations + run-rag)** — add `expand_paths()` helper in `src/coicop_rag/utils.py` or new module, argparse in both scripts, templated `config/config.yaml`, drop `{timestamp}` logic from `run-rag`, argo steps pass flags. Verify: `uv sync --frozen && uv run scripts/1_prune_annotations.py --help && uv run scripts/2_run_rag.py --help`.
6. **Migrate `run-ttc`** — argo-only: rewrite `--file` and `--output` args to per-run paths, swap dependency edge to `preprocessing`. Verify: `kubectl create --dry-run=client -f argo/pipeline.yaml`.
7. **Update `docs/inputs-outputs.md`** to reflect the new harmonized layout; drop the "Limites connues" section entries that are now resolved.

## Critical files

- `/home/onyxia/work/codif-coicop-bdf/argo/pipeline.yaml`
- `/home/onyxia/work/codif-coicop-bdf/preprocessing/{main.py,config.yaml}`
- `/home/onyxia/work/codif-coicop-bdf/regex-codif/{src/main.py,src/utils/load_config.py,config/config.yaml}`
- `/home/onyxia/work/codif-coicop-bdf/stats-annotations/R/{main.R,calcul_lcs_libel.R}`
- `/home/onyxia/work/codif-coicop-bdf/coicop-rag/{scripts/1_prune_annotations.py,scripts/2_run_rag.py,config/config.yaml,src/coicop_rag/utils.py}`
- `/home/onyxia/work/codif-coicop-bdf/docs/inputs-outputs.md`

## Verification

Per-commit static checks (done in the plan above per step).

End-to-end (manual, requires cluster + S3):
- Push branch and submit: `argo submit argo/pipeline.yaml -p sample_size=10`. Expected folder appears at `s3://projet-budget-famille/data/workflow_runs/<YYYY-MM-DD>/<workflow-name>/` with one subfolder per step.
- Two consecutive runs produce two disjoint folders — no overwrites.
- `skip-vector-db=true` run still completes successfully (prune-coicop + create-vector-db skipped; shared paths + Qdrant collection reused).

## Out of scope

- `argo/ttc-pipeline.yaml` (standalone run-ttc workflow) — kept as-is; would need a `source_run_id` parameter to read from a specific preprocessing run. Revisit when/if ttc-pipeline stays around.
- Per-run Qdrant collections (rejected — shared collection kept).
- Auto-cleanup / S3 lifecycle rules for old run folders — separate concern.
- Migration of historical runs into the new layout — not needed.
