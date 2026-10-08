# Running the `codif-pipeline`

From a terminal of your SSPCloud VS Code service, in the same namespace as the Argo Workflows service.
You don't need to log in or export anything: the pod's service account is used, and the credentials come from the `secret-codif-coicop-bdf` secret.

## 1. Install the Argo CLI

The CLI version must match the server (`v3.6.5`).

```bash
ARGO_VERSION=v3.6.5
curl -sLO "https://github.com/argoproj/argo-workflows/releases/download/${ARGO_VERSION}/argo-linux-amd64.gz"
gunzip argo-linux-amd64.gz && chmod +x argo-linux-amd64
mkdir -p ~/.local/bin && mv argo-linux-amd64 ~/.local/bin/argo
export PATH="$HOME/.local/bin:$PATH"
```

## 2. Check the setup

```bash
argo version                                    # CLI installed
kubectl get secret secret-codif-coicop-bdf      # credentials secret present
kubectl auth can-i create workflows.argoproj.io # -> "yes"
```

## 3. Launch the pipeline

```bash
cd argo
export ARGO_NAMESPACE=projet-budget-famille
argo submit codif-pipeline.yaml --parameter-file params.yaml --watch
```

- Edit `params.yaml` for your run. Any parameter left out keeps its default from the YAML.
- The pods clone `git-branch` (default `main`) from GitHub, so **local changes are invisible until they are pushed**.
- Every run first does a **smoke pass** on 100 rows (~8 min). If it fails, the full run never starts.
  - Check a branch without launching the full run: `-p smoke-only=true -p git-branch=<branch>`
  - Skip the smoke pass: `-p skip-smoke=true`
- Never pass `run_id` or `run_date`: they are computed automatically.

> ⚠️ `argo submit` accepts an **unknown parameter name without any error**. A typo in `-p` or in `params.yaml` has no effect.

## 4. The vector DBs (only when the nomenclature or the KB changes)

The pipeline queries two existing Qdrant collections, named in `params.yaml`. To rebuild them (details in [`index_annotations_helper.md`](./index_annotations_helper.md)):

```bash
argo submit index-notices-pipeline.yaml --watch
argo submit index-annotations-pipeline.yaml --watch
```

Each workflow prints, at the end, the line to paste into `params.yaml`:

```
classify-rag-notices-collection: coicop_notices__2026-09-02__index-notices-a7k2p
classify-rag-annotations-collection: coicop_annotations__2026-09-02__index-annotations-b3x9q
```

## 5. Debugging

```bash
argo lint codif-pipeline.yaml   # validate the YAML before submitting
argo list                       # all runs
argo get  @latest               # DAG status: which step is ✖
argo logs @latest -f            # follow logs
argo stop @latest               # graceful stop
argo resubmit @latest           # rerun as-is
```

`@latest` is the last submitted workflow; otherwise use its name (`codif-xxxxx`).
Outputs are under `s3://projet-budget-famille/data/workflow_runs/{run_date}/{run_id}/`, and those of the smoke pass under `…/{run_id}-smoke/`.
The full list of parameters is in the root `CLAUDE.md` and in `codif-pipeline.yaml`.
