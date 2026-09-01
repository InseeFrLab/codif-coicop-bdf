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
argo submit codif-pipeline.yaml -p sample-annotations=500 -p skip-report=false --watch
```

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
| `input_file` | `s3://…/BDF_data_…_a_codif.csv` | input CSV file on S3 |
| `text_column` | `NAT_DEP` | label column to classify |
| `shop_column` / `budget_column` | `MAG_DEP` / `MONT_DEP` | shop / amount |
| `annee_column` / `source_column` | `""` | optional |
| `sample-annotations` | `""` | cap the annotation KB indexed in the vector DB (empty = all) |
| `sample-observations` | `""` | cap the to-codify set, prod only (empty = all); sampled **once** at `classify-regex` so all classifiers code the same rows; in eval, inference uses `sample-annotations` |
| `classify-rag-model` / `reconcile-llm-model` | `gemma4-26b-moe` | LLM models |
| `reconcile-llm-concurrency` | `5` | arbitration parallelism |
| `classify-ttc-model-uri` | `mlflow-artifacts:/10/…/model` | TTC model (MLflow) |
| `skip-index` | `true` | skip Qdrant rebuild |
| `skip-report` | `false` | generate the Quarto report |
| `report-experiment` | `codif-coicop-eval` | report's MLflow experiment |
| `conciliation` | `llm` | which step decides the final code: `llm` (`reconcile-llm`) or `sirus` (`reconcile-sirus`). **Mutually exclusive** — the other is skipped |
| `reconcile-sirus-model-uri` | `""` | SIRUS model (MLflow), required when `conciliation: sirus`. Produced out of pipeline by `reconcile-sirus/train.sh` |
