# Running the `codif-pipeline`

Quick guide to launch the `codif-pipeline.yaml` Argo workflow from a terminal
(SSPCloud VS Code service), in the same namespace as the Argo Workflows service.

## Prerequisites (check once)

```bash
argo version                                    # CLI installed?
kubectl get secret secret-codif-coicop-bdf      # credentials secret present?
kubectl auth can-i create workflows.argoproj.io # allowed to launch?  -> "yes"
```

- **Credentials are handled automatically**: the YAML reads them from the
  `secret-codif-coicop-bdf` secret via `secretKeyRef`. Nothing to export.
- From the terminal you use your pod's *service account*; no login needed.

## Installing the Argo CLI (if `argo version` fails)

The CLI is a single static binary downloaded from the GitHub releases. Match the
version to the Argo Workflows server (`v3.6.5` here):

```bash
# Linux x86_64 — install into ~/.local/bin (no root required)
ARGO_VERSION=v3.6.5
curl -sLO "https://github.com/argoproj/argo-workflows/releases/download/${ARGO_VERSION}/argo-linux-amd64.gz"
gunzip argo-linux-amd64.gz
chmod +x argo-linux-amd64
mkdir -p ~/.local/bin
mv argo-linux-amd64 ~/.local/bin/argo

# Make sure ~/.local/bin is on your PATH (usually already the case on SSPCloud)
export PATH="$HOME/.local/bin:$PATH"

argo version    # verify
```

> Tip: check the server version with `argo version` once it works, or in the
> Argo Workflows UI, and keep client and server aligned.

## Prerequisite: the two vector DBs

Since indexing left this pipeline, `codif-pipeline.yaml` **does not build the Qdrant
collections** — it queries existing ones, named by parameter. Build them once with their own
workflows (see [`index_annotations_helper.md`](./index_annotations_helper.md)):

```bash
argo submit index-notices-pipeline.yaml --watch
argo submit index-annotations-pipeline.yaml --watch
```

Each prints, at the end, the exact line to paste into `params.yaml`:

```
classify-rag-notices-collection: coicop_notices__2026-09-02__index-notices-a7k2p
classify-rag-annotations-collection: coicop_annotations__full__2026-09-02__index-annotations-b3x9q
```

They are reusable: rebuild them only when the nomenclature or the annotation base changes.

## Every run is preceded by a smoke run

`codif-pipeline.yaml` runs the **whole DAG twice**: first on a small sample
(`smoke-observations`, default 100), then for real. If the smoke pass fails, the real
run never starts.

```
smoke (≈8 min, 100 lignes)  ──→  full (≈2 h)
        │                          ▲
        └── échec ─────────────────┘  jamais atteint
```

It costs ~8 min measured in production on the real CSV — about 7% of a full run — which
is what makes gating every run bearable.

### There is nothing to launch

The smoke pass is **not a separate command**. Your usual submit already does it:

```bash
argo submit codif-pipeline.yaml --parameter-file params.yaml --watch
```

`argo get` then shows two passes under the workflow:

```
STEP        TEMPLATE   DURATION
 ├─ smoke   pipeline   8m      ← 100 rows
 └─ full    pipeline   2h      ← starts only if smoke succeeded
```

Their S3 outputs live under `…/{run_id}-smoke/` and `…/{run_id}/`, so they never mix, and
the smoke logs to its own MLflow experiment (`codif-coicop-smoke`) so its 100-row metrics
are never mistaken for real ones.

### Two switches

**Smoke only** — check a branch in ~8 min without committing to the 2-hour run. The `full`
pass shows as `Skipped`:

```bash
argo submit codif-pipeline.yaml --parameter-file params.yaml \
  -p smoke-only=true -p git-branch=<your-branch> --watch
```

**Skip the smoke** — a false-positive must not immobilise production. The `smoke` pass shows
as `Skipped` and `full` runs anyway:

```bash
argo submit codif-pipeline.yaml --parameter-file params.yaml -p skip-smoke=true --watch
```

Both rely on the same mechanism: the legacy `dependencies:` form expands to
`Succeeded || Skipped || Daemoned`, so a **skipped** task satisfies the gate while a
**failed** one does not.

### Reading a failure

If the smoke fails, `full` never starts and the workflow stops. Find which step broke:

```bash
argo get  @latest              # which step is ✖
argo logs @latest -f           # its logs
```

The smoke's artifacts stay under `…/{run_id}-smoke/` for inspection — a failed smoke leaves
everything it managed to produce.

**What it does not catch**: volume-related failures (OOM, rate limits, timeouts that only
appear at 16 000 rows), and a deliverable that is produced but wrong — the success criterion
is the state of the steps, not the content they produce.

## Launch

```bash
cd argo

# 1. Validate the YAML (good reflex before any submit)
argo lint codif-pipeline.yaml

# 2. Submit WITH your parameters (params.yaml file)
export ARGO_NAMESPACE=projet-budget-famille
argo submit codif-pipeline.yaml --parameter-file params.yaml --watch
```

- `params.yaml` (next to this guide) holds the overridable parameters; edit it
  for your run. Any value left out falls back to the YAML default.
- `--watch` streams the DAG tree live.

### Quick variant without a file

For a one-off override, without touching `params.yaml`:

```bash
argo submit codif-pipeline.yaml -p sample-observations=500 -p skip-report=true --watch
```

> ⚠️ `argo submit` **silently accepts an unknown parameter name** — it just registers a new one
> that nothing reads. A typo in `-p` or in `params.yaml` produces no error and no effect. This
> is how `rereconciliation:` sat dead in `params.yaml` from its own commit.

> ⚠️ Never pass `run_id` / `run_date`: they are computed automatically and keep
> the S3 paths `workflow_runs/{run_date}/{run_id}/` consistent across steps.

## Monitor

```bash
argo list                  # all runs
argo get   @latest         # DAG status of the latest run
argo logs  @latest -f      # follow logs live
```

(`@latest` = last submitted workflow; otherwise use the name `codif-xxxxx`.)

## Troubleshooting

```bash
argo stop     @latest   # graceful stop
argo delete   @latest   # delete
argo resubmit @latest   # rerun as-is
```

## Main parameters

| Parameter | Default | Role |
|---|---|---|
| `git-branch` | `main` | branch every step clones. **Your local changes are invisible until pushed** — the pods clone from GitHub, not from your disk |
| `input_file` | `s3://…/BDF_data_…_a_codif.csv` | input CSV file on S3 |
| `text_column` | `NAT_DEP` | label column to classify |
| `shop_column` / `budget_column` | `MAG_DEP` / `MONT_DEP` | shop / amount |
| `annee_column` / `source_column` | `""` | optional |
| `skip-smoke` | `"false"` | `"true"` runs the real pass without the smoke gate |
| `smoke-only` | `"false"` | `"true"` runs the smoke pass and stops there |
| `smoke-observations` | `"100"` | size of the smoke sample |
| `smoke-experiment` | `codif-coicop-smoke` | MLflow experiment of the smoke pass |
| `sample-observations` | `""` | cap the to-codify set, in **both** modes (empty = all); sampled **once** at `classify-regex` so all classifiers code the same rows. To cap the indexed KB instead, that is `kb-sample-size` of `index-annotations-pipeline.yaml` |
| `classify-rag-notices-collection` | `""` | **required** — Qdrant collection from `index-notices-pipeline.yaml` |
| `classify-rag-annotations-collection` | `""` | **required** — Qdrant collection from `index-annotations-pipeline.yaml` |
| `classify-rag-model` / `reconcile-llm-model` | `gemma4-26b-moe` | LLM models |
| `reconcile-llm-concurrency` | `5` | arbitration parallelism |
| `classify-ttc-model-uri` | `mlflow-artifacts:/10/…/model` | TTC model (MLflow) |
| `skip-report` | `false` | `"true"` skips the Quarto report (it runs by default) |
| `report-experiment` | `codif-coicop-eval` | report's MLflow experiment |
| `reconciliation` | `llm` | which step decides the final code: `llm` (`reconcile-llm`) or `sirus` (`reconcile-sirus`). **Mutually exclusive** — the other is skipped |
| `reconcile-sirus-model-uri` | `""` | SIRUS model (MLflow), required when `reconciliation: sirus`. Produced out of pipeline by `reconcile-sirus/train.sh` |

The two `*-collection` parameters have no default **on purpose**. Argo cannot express a required
parameter, so the guard is written twice — a `[ -z ] && exit 1` in the container script and
`required=True` in argparse. An unset name fails in seconds with a message naming the workflow
to run, instead of silently querying some other run's index.

> The parameter is spelled `reconciliation`. Older docs and `params.yaml` said `conciliation`
> or `rereconciliation`; since an unknown key is accepted in silence, those commands ran with
> the `llm` default even when `sirus` was requested.
