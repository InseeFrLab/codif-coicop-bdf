#!/usr/bin/env bash
# Entraîne le modèle SIRUS sur un run d'évaluation passé, HORS pipeline Argo.
# (L'entraînement produit un modèle pour un run futur ; cf. sirus/README.md.)
#
# Usage : ./train.sh 2026-06-29/codif-vvkv9
# Réglages optionnels : NUM_RULE=20 MAX_DEPTH=2 SEED=42 EXPERIMENT=... ./train.sh ...
#
# Affiche en fin d'exécution l'URI MLflow à recopier dans argo/params.yaml.
set -euo pipefail

RUN="${1:?usage: ./train.sh <date>/<run_id>}"
RUN_DATE="${RUN%%/*}"; RUN_ID="${RUN##*/}"
ROOT="${SIRUS_RUNS_ROOT:-s3://projet-budget-famille/data/workflow_runs}/$RUN"
ART="artifacts/$RUN_ID"

# Seul contrôle préalable : sans MLflow, l'échec n'arriverait qu'à la 3ᵉ étape,
# après plusieurs minutes d'ajustement R.
if [ -z "${MLFLOW_TRACKING_URI:-}" ]; then
  echo "MLFLOW_TRACKING_URI absente : le modèle et ses métriques n'auraient" >&2
  echo "nulle part à aller. Activer l'option MLflow du service SSP Cloud." >&2
  exit 1
fi

mkdir -p "$ART"
uv sync --locked
Rscript R/install_deps.R

# `rag-annotation` au SINGULIER, contrairement au nom de l'étape Argo.
uv run main.py build-table \
  --lcs-file "$ROOT/codif-lcs/raw_test_LCS.parquet" \
  --rag-file "$ROOT/run-rag/predictions.parquet" \
  --rag-annotations-file "$ROOT/rag-annotation/predictions.parquet" \
  --ttc-file "$ROOT/run-ttc/predictions.parquet" \
  --mapping-file "$ROOT/prune/mapping_lvl4.parquet" \
  --tocodify-file "$ROOT/codif-regex/raw_test_without_regex.parquet" \
  --out "$ART/features.parquet"

Rscript R/fit_sirus.R --features="$ART/features.parquet" --out-dir="$ART" \
  --num-rule="${NUM_RULE:-20}" --max-depth="${MAX_DEPTH:-2}" --seed="${SEED:-42}"

# Les artefacts restent dans $ART : si le log MLflow échoue, relancer cette
# seule commande suffit, sans réajuster le modèle.
uv run main.py finalize --artifacts-dir "$ART" \
  --experiment "${EXPERIMENT:-codif-coicop-sirus}" \
  --run-id "$RUN_ID" --run-date "$RUN_DATE"
