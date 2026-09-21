"""Accès au registre des artefacts (`contracts.yaml`).

Une étape ne construit plus le chemin d'une autre : elle le demande.

    from codif_common.contracts import artifact

    path = artifact("prune-codes", "mapping_lvl4", run_date=..., run_id=...)
    # → s3://projet-budget-famille/data/workflow_runs/2026-09-03/codif-abc/prune-codes/mapping_lvl4.parquet

Le pendant R est dans `classify-lcs/R/main.R`, qui lit le même fichier.
"""

import functools
import os
from pathlib import Path
from typing import Any, Dict, List

import yaml

# Remonte depuis src/codif_common/ jusqu'à la racine du dépôt.
_DEFAULT_REGISTRY = Path(__file__).resolve().parents[3] / "contracts.yaml"


@functools.lru_cache(maxsize=None)
def load_registry(path: str | Path | None = None) -> Dict[str, Any]:
    """Charge et met en cache le registre.

    Le chemin peut être forcé par ``$COICOP_CONTRACTS`` — utile pour un test ou
    pour rejouer un run ancien avec un registre d'époque.
    """
    path = path or os.environ.get("COICOP_CONTRACTS") or _DEFAULT_REGISTRY
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def bucket() -> str:
    """Nom du bucket, surchargeable par ``$COICOP_BUCKET``."""
    return os.environ.get("COICOP_BUCKET") or load_registry()["bucket"]


def run_root(run_date: str, run_id: str) -> str:
    """Racine S3 du run, URI complète."""
    template = load_registry()["run_root"]
    return f"s3://{bucket()}/{template.format(run_date=run_date, run_id=run_id)}"


def artifact(step: str, key: str, *, run_date: str, run_id: str, **fmt: str) -> str:
    """URI complète d'un artefact déclaré.

    Lève un ``KeyError`` explicite si l'étape ou la clé n'existe pas — c'est
    voulu : une faute de frappe doit échouer ici, pas produire un chemin S3
    plausible que personne n'a jamais écrit.

    ``**fmt`` sert aux rares gabarits à variable supplémentaire, comme
    ``export-results.deliverable`` qui porte le nom du fichier d'entrée.
    """
    steps = load_registry()["steps"]
    if step not in steps:
        raise KeyError(
            f"Étape « {step} » absente de contracts.yaml. Connues : {sorted(steps)}"
        )
    outputs = steps[step].get("outputs") or {}
    if key not in outputs:
        raise KeyError(
            f"Sortie « {key} » non déclarée pour l'étape « {step} ». "
            f"Déclarées : {sorted(outputs) or '(aucune)'}"
        )
    relative = outputs[key].format(run_date=run_date, run_id=run_id, **fmt)
    return f"{run_root(run_date, run_id)}/{relative}"


def external(key: str) -> str:
    """URI complète d'une entrée externe (ni produite ni versionnée par un run)."""
    ext = load_registry()["external"]
    if key not in ext:
        raise KeyError(
            f"Entrée externe « {key} » non déclarée. Connues : {sorted(ext)}"
        )
    return f"s3://{bucket()}/{ext[key]}"


def consumers(step: str, key: str) -> List[str]:
    """Étapes qui déclarent consommer ``step.key``.

    Répond à « qui casse si je change ce fichier ? » — question qui, avant le
    registre, ne se traitait qu'en fouillant sept mécanismes différents.
    """
    ref = f"{step}.{key}"
    return sorted(
        name for name, spec in load_registry()["steps"].items()
        if ref in (spec.get("inputs") or [])
    )


def dangling_inputs() -> List[str]:
    """Entrées déclarées qui ne désignent aucune sortie déclarée.

    C'est le contrôle qui aurait rendu visible le décalage entre
    `build-datasets` (qui écrit le suggester sous `build-datasets/`) et
    `classify-lcs` (qui le cherchait à la racine du run) — décalage qu'un
    `tryCatch` avalait sans un mot.
    """
    steps = load_registry()["steps"]
    known = {
        f"{name}.{key}"
        for name, spec in steps.items()
        for key in (spec.get("outputs") or {})
    }
    return sorted(
        f"{name} → {ref}"
        for name, spec in steps.items()
        for ref in (spec.get("inputs") or [])
        if ref not in known
    )
