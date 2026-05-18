# Consolidate pipeline sub-repos into codif-coicop-bdf

## Context

`codif-coicop-bdf` currently contains only an argo workflow (`argo/pipeline.yaml`) that orchestrates the BDF → COICOP codification pipeline by cloning and running five separate Git repositories at runtime:

| Argo step | Source repo | Entry point |
|---|---|---|
| `preprocessing` | `construction-dataset` | `uv run main.py` |
| `codif-regex` | `regex_codif` | `uv run src/main.py` |
| `codif-lcs` | `stats-annotations` | `Rscript R/main.R` |
| `prune-coicop`, `prune-annotations`, `create-vector-db`, `run-rag` | `coicop-rag` | `uv run scripts/<N>_*.py` |
| `run-ttc` | `coicop_bdf_classifier` (pre-built image `ghcr.io/micedre/...`) | `python main.py predict-basic` |

Managing five repos (some on GitLab SSP Cloud, some on GitHub) is painful: PRs, CI, versions, and branches must be coordinated across them. The goal is to **consolidate all source code into this repo as a monorepo**, so that the argo workflow only clones `codif-coicop-bdf` and cd's into the relevant subdirectory.

Decisions (from clarifying questions):
- **Layout**: subdirs per source repo, each keeps its own `pyproject.toml` / `config/` / tooling. Minimal refactoring.
- **Classifier**: vendor `coicop_bdf_classifier` too (its GitHub home at `micedre/coicop_bdf_classifier` will be archived). `run-ttc` argo step stays pointing to the pre-built image for now — it is slated for deletion later.
- **Argo strategy**: each step now clones `https://github.com/InseeFrLab/codif-coicop-bdf.git` and cd's into its subdirectory.
- **Drop**: `coicop-rag/old/` (deprecated rule-based classifier), Quarto reports (`*.qmd`, `_quarto.yml`, `rapport_style.css`) in `regex_codif` + `stats-annotations` (pure reporting, not in pipeline).
- **Keep**: each source repo's `README.md` as a subdir README; top-level README.md becomes the pipeline overview (already close to that today).

## Target structure

```
codif-coicop-bdf/
├── argo/
│   ├── pipeline.yaml          # UPDATED: clone URLs + cd into subdirs
│   ├── rbac.yaml              # unchanged
│   └── ttc-pipeline.yaml      # unchanged
├── preprocessing/             # from construction-dataset
│   ├── main.py
│   ├── pyproject.toml
│   ├── uv.lock
│   ├── .python-version
│   ├── config.yaml
│   ├── data/stopwords.json
│   ├── src/{data,stats,utils}/
│   └── README.md
├── regex-codif/               # from regex_codif
│   ├── pyproject.toml
│   ├── uv.lock
│   ├── .python-version
│   ├── config/{config.yaml,rules.yaml}
│   ├── src/{__init__.py,main.py,data/,utils/}
│   └── README.md
├── coicop-rag/                # from coicop-rag
│   ├── pyproject.toml
│   ├── uv.lock
│   ├── .python-version
│   ├── config/config.yaml
│   ├── scripts/{0_prunning_coicop.py,0_create_vector_db.py,1_prune_annotations.py,2_run_rag.py,eval.py}
│   ├── src/coicop_rag/
│   ├── tests/
│   └── README.md
│   # DROP: old/, logs/
├── stats-annotations/         # from stats-annotations
│   ├── R/{main.R,fonctions.R,calcul_distances.R,calcul_lcs_libel.R,...}
│   ├── C/distance_gcd_batch_cpp.cpp
│   ├── stats-annotations.Rproj
│   └── README.md
│   # DROP: index.qmd, resultats.qmd, _quarto.yml, rapport_style.css, .gitlab-ci.yml
├── coicop-bdf-classifier/     # from coicop_bdf_classifier
│   ├── main.py
│   ├── pyproject.toml
│   ├── uv.lock
│   ├── .python-version
│   ├── Dockerfile
│   ├── src/
│   ├── tests/
│   ├── data/
│   ├── docs/
│   ├── CLAUDE.md
│   └── README.md
│   # DROP: argo/ (its own argo folder), .github/ (handled at monorepo level later)
├── LICENSE                    # unchanged
└── README.md                  # UPDATED: add repo layout section, keep pipeline overview
```

## Argo workflow changes (`argo/pipeline.yaml`)

All templates currently have the pattern:

```yaml
args:
  - |
    set -e
    git clone --branch main https://<host>/<repo>.git repo && cd repo
    uv sync                # (some steps) or nothing
    uv run <entrypoint>
```

They must become:

```yaml
args:
  - |
    set -e
    git clone --branch main https://github.com/InseeFrLab/codif-coicop-bdf.git repo && cd repo/<subdir>
    uv sync                # preserve per-step
    uv run <entrypoint>
```

Exact per-step diffs:

| Template | New `cd` target | Entry point (unchanged) |
|---|---|---|
| `preprocessing` | `repo/preprocessing` | `uv sync && uv run main.py` |
| `codif-regex` | `repo/regex-codif` | `uv sync && uv run src/main.py` |
| `codif-lcs` | `repo/stats-annotations` | `Rscript R/main.R` |
| `prune-coicop` | `repo/coicop-rag` | `uv run scripts/0_prunning_coicop.py --config {{inputs.parameters.config}}` |
| `prune-annotations` | `repo/coicop-rag` | `uv run scripts/1_prune_annotations.py --config {{inputs.parameters.config}}` |
| `create-vector-db` | `repo/coicop-rag` | `uv run scripts/0_create_vector_db.py --config {{inputs.parameters.config}}` |
| `run-rag` | `repo/coicop-rag` | `mkdir -p logs prompts` then `uv run scripts/2_run_rag.py $ARGS` (arg-building unchanged) |
| `run-ttc` | n/a (no clone, uses pre-built image) | unchanged — slated for deletion later |

Container images, env vars, resources, and DAG dependencies (`dag.tasks`) remain unchanged.

## Commit-sized todo

Each step is a single commit, lands independently, and the pipeline keeps working after each one (because `argo/pipeline.yaml` is updated only in step 7, after all source is in place).

1. **Vendor `preprocessing/`** from `construction-dataset` (copy working tree minus `.git`, `data/`+stopwords kept, include `uv.lock`, `README.md`). Verify `cd preprocessing && uv sync --frozen` succeeds.
2. **Vendor `regex-codif/`** from `regex_codif` (drop `index.qmd`, `resultats.qmd`, `_quarto.yml`, `rapport_style.css`, `.gitlab-ci.yml`). Verify `uv sync --frozen`.
3. **Vendor `coicop-rag/`** from `coicop-rag` (drop `old/`, `logs/`). Verify `uv sync --frozen` and that `scripts/*.py` are present.
4. **Vendor `stats-annotations/`** from `stats-annotations` (drop `index.qmd`, `resultats.qmd`, `_quarto.yml`, `rapport_style.css`, `.gitlab-ci.yml`). Verify `Rscript -e "source('R/main.R')"` at least parses (don't run end-to-end — needs S3 + data).
5. **Vendor `coicop-bdf-classifier/`** from `coicop_bdf_classifier` (drop nested `argo/`, nested `.github/`). Verify `uv sync --frozen`.
6. **Update top-level `README.md`** with a "Repo layout" section describing each subdir, and a short note that run-ttc still uses the pre-built image pending removal.
7. **Update `argo/pipeline.yaml`**: rewrite each step's `git clone` URL to `https://github.com/InseeFrLab/codif-coicop-bdf.git` and add `cd repo/<subdir>`. `run-ttc` unchanged. Validate YAML with `yq '.' argo/pipeline.yaml > /dev/null` and Argo lint (`argo lint argo/pipeline.yaml` if CLI available, else `kubectl create --dry-run=client -f argo/pipeline.yaml`).

## Critical files

- `/home/onyxia/work/codif-coicop-bdf/argo/pipeline.yaml` — the workflow being rewritten
- `/home/onyxia/work/codif-coicop-bdf/README.md` — needs an updated layout section
- `/home/onyxia/work/construction-dataset/` → `preprocessing/`
- `/home/onyxia/work/regex_codif/` → `regex-codif/`
- `/home/onyxia/work/coicop-rag/` → `coicop-rag/` (minus `old/`, `logs/`)
- `/home/onyxia/work/stats-annotations/` → `stats-annotations/` (minus `.qmd`, `_quarto.yml`)
- `/home/onyxia/work/coicop_bdf_classifier/` → `coicop-bdf-classifier/` (minus its own `argo/`, `.github/`)

## Verification

Static checks per commit (fast, local):
- For each Python subdir after vendoring: `cd <subdir> && uv sync --frozen && uv run python -c "print('ok')"` to confirm the uv environment resolves.
- For `stats-annotations/`: `Rscript -e 'parse("R/main.R")'` to confirm the script parses.
- After step 7: `yq '.' argo/pipeline.yaml > /dev/null` and if available `argo lint argo/pipeline.yaml` or `kubectl create -f argo/pipeline.yaml --dry-run=client`.

End-to-end (manual, requires cluster + S3 credentials):
- Push the branch to `InseeFrLab/codif-coicop-bdf` so the argo clone URL resolves.
- Submit the pipeline with a small sample: `argo submit argo/pipeline.yaml -p sample_size=10 -p skip-vector-db=true`. Expected: `preprocessing` → `prune-annotations` → `codif-regex` → `codif-lcs` + `run-rag` (skipped deps as parameterized). Verify each pod succeeds and S3 outputs match the pre-consolidation outputs.
- Full run (no skip, default sample size) once smoke test passes.

## Deferred / out of scope

- Removing `run-ttc` (user flagged for later).
- Unifying `pyproject.toml` / Python packages into one workspace.
- Monorepo-level CI (`.github/workflows`) replacing the per-repo CI that lived in the source repos.
- Building a custom Docker image for the classifier to replace `ghcr.io/micedre/coicop_bdf_classifier:latest`.
