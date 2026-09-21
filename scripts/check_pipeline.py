#!/usr/bin/env python3
"""Contrôles de cohérence des workflows Argo, sans cluster ni exécution.

Répond à une seule question : « quelqu'un a-t-il renommé quelque chose en
oubliant un endroit ? » — la famille de pannes qui a coûté trois lancements
ratés en une semaine.

Ce que ça NE fait pas : vérifier que le pipeline code correctement. C'est un
contrôle de cohérence du dépôt avec lui-même, rien de plus.

    uv run python scripts/check_pipeline.py .

Sortie non nulle si un défaut bloquant est trouvé. Les avertissements
(paramètre déclaré que plus personne n'utilise) n'échouent pas.
"""

import re
import sys
from pathlib import Path

import yaml

WF_PARAM = re.compile(r"\{\{workflow\.parameters\.([A-Za-z0-9_-]+)\}\}")
IN_PARAM = re.compile(r"\{\{inputs\.parameters\.([A-Za-z0-9_-]+)\}\}")

# Quel fichier de paramètres alimente quel workflow. `argo submit
# --parameter-file` accepte une clé inconnue SANS RIEN DIRE : elle est
# simplement enregistrée et jamais lue. C'est ainsi que `rereconciliation:` est
# resté mort dans params.yaml depuis son commit, faisant partir les runs en
# conciliation LLM alors qu'on demandait SIRUS.
PARAM_FILES = {"argo/params.yaml": "argo/codif-pipeline.yaml"}


def check_workflow(path: Path) -> tuple[list[str], list[str]]:
    """Renvoie (erreurs, avertissements) pour un manifeste de workflow."""
    errors: list[str] = []
    warnings: list[str] = []
    raw = path.read_text()
    doc = yaml.safe_load(raw)

    declared = {p["name"] for p in doc["spec"].get("arguments", {}).get("parameters", [])}
    referenced = set(WF_PARAM.findall(raw))

    for missing in sorted(referenced - declared):
        errors.append(f"{{{{workflow.parameters.{missing}}}}} utilisé mais non déclaré")
    for unused in sorted(declared - referenced):
        warnings.append(f"paramètre « {unused} » déclaré mais jamais utilisé")

    templates = {t["name"] for t in doc["spec"]["templates"]}
    for t in doc["spec"]["templates"]:
        tasks = (t.get("dag") or {}).get("tasks", [])
        names = {x["name"] for x in tasks}
        for task in tasks:
            if task["template"] not in templates:
                errors.append(
                    f"tâche « {task['name']} » → template inexistant « {task['template']} »"
                )
            for dep in task.get("dependencies", []):
                if dep not in names:
                    errors.append(
                        f"tâche « {task['name']} » dépend de « {dep} », qui n'existe pas"
                    )

    # Une valeur qu'un template attend de son appelant doit lui être déclarée.
    for t in doc["spec"]["templates"]:
        declared_in = {p["name"] for p in (t.get("inputs") or {}).get("parameters", [])}
        for missing in sorted(set(IN_PARAM.findall(yaml.safe_dump(t))) - declared_in):
            errors.append(
                f"template « {t['name']} » : inputs.parameters.{missing} non déclaré"
            )

    # Les 11 étapes lancent `uv sync --locked` : sans le drapeau, une étape peut
    # réinstaller des versions différentes de celles verrouillées.
    for i, line in enumerate(raw.splitlines(), 1):
        if re.search(r"uv (sync|run)\b", line) and "--locked" not in line:
            errors.append(f"ligne {i} : `uv` sans --locked → {line.strip()[:70]}")

    return errors, warnings


def check_param_file(param_path: Path, wf_path: Path) -> list[str]:
    declared = {
        p["name"]
        for p in yaml.safe_load(wf_path.read_text())["spec"]
        .get("arguments", {})
        .get("parameters", [])
    }
    params = yaml.safe_load(param_path.read_text()) or {}
    return [
        f"« {k} » inconnu de {wf_path.name} — argo l'accepterait en silence"
        for k in params
        if k not in declared
    ]


def main(root: Path) -> int:
    failed = False

    for wf in sorted((root / "argo").glob("*pipeline.yaml")):
        errors, warnings = check_workflow(wf)
        print(f"=== {wf.name} ===")
        for w in warnings:
            print(f"  ⚠ {w}")
        for e in errors:
            print(f"  ✗ {e}")
        if not errors:
            print("  ✓ cohérent")
        failed |= bool(errors)

    for pf, wf in PARAM_FILES.items():
        pf_path, wf_path = root / pf, root / wf
        if not pf_path.exists():
            continue
        errors = check_param_file(pf_path, wf_path)
        print(f"=== {pf_path.name} vs {wf_path.name} ===")
        for e in errors:
            print(f"  ✗ {e}")
        if not errors:
            print("  ✓ cohérent")
        failed |= bool(errors)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1] if len(sys.argv) > 1 else ".")))
