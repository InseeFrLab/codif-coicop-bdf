#!/usr/bin/env python
"""Conciliation COICOP par règles interprétables (SIRUS).

Trois sous-commandes, réparties sur les deux étapes Argo :

    entraînement   build-table  (Python)  →  R/fit_sirus.R  (R)  →  finalize (Python)
                   hors pipeline, enchaîné par `train.sh`
    sirus-predict  predict      (Python)  —  étape Argo, si `conciliation: sirus`

`build-table` et `predict` partagent `src/candidates.py` : c'est ce qui garantit
que les sentinelles et l'ordre des features vus à l'entraînement sont ceux vus à
la prédiction.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from decide_coicop import _read_parquet, _write_parquet, load_all_observations  # noqa: E402

from src.candidates import (  # noqa: E402
    FEATURES,
    build_candidate_table,
    features_sha256,
    log_diagnostics,
    reconcile_population,
)
from src.scorer import (  # noqa: E402
    load_rules,
    pick_best,
    resolve_model_path,
    score,
    scorable_mask,
)
from src.train import (  # noqa: E402
    build_calibration,
    check_drift,
    feature_distribution,
    log_to_mlflow,
    split_by_product,
    verify_scorer_against_r,
)

logging.basicConfig(
    level=logging.INFO, format="[sirus] %(levelname)s %(message)s", stream=sys.stdout
)
logger = logging.getLogger("sirus")


def _merged_table(args: argparse.Namespace) -> pd.DataFrame:
    """Fusion des 4 classifieurs + troncature niveau 4, via `decide-coicop`.

    Aucune de ces deux logiques n'est réimplémentée ici : c'est la même fonction
    que celle utilisée par le juge LLM, donc les deux conciliations raisonnent
    exactement sur les mêmes candidats.
    """
    if not args.mapping_file:
        logger.warning(
            "--mapping-file absent : les codes candidats ne seront ni tronqués au "
            "niveau 4 ni élagués, donc pas comparables entre eux. À n'utiliser que "
            "pour un test."
        )
    return load_all_observations(
        lcs_path=args.lcs_file,
        rag_path=args.rag_file,
        ttc_path=args.ttc_file,
        rag_annotations_path=args.rag_annotations_file,
        mapping_path=args.mapping_file,
    )


def _share_without_candidate(artifacts_dir: Path) -> float | None:
    """Part de produits sans aucun candidat, relue des diagnostics de build-table.

    Sert de référence au contrôle de dérive de `sirus-predict` : mesurée sur un
    run où les 4 classifieurs fonctionnaient, elle donne le point de comparaison
    qui révèle un changement de format en amont.
    """
    chemin = artifacts_dir / "features.diagnostics.json"
    if not chemin.exists():
        return None
    diag = json.loads(chemin.read_text(encoding="utf-8"))
    n = diag.get("n_products_in") or 0
    if not n:
        return None
    return len(diag.get("ids_without_candidate", [])) / n


def _tocodify_ids(path: str | None) -> set | None:
    """Identifiants réellement à coder, pour détecter les produits perdus en amont."""
    if not path:
        return None
    df = _read_parquet(path)
    if "id" not in df.columns:
        logger.warning("--tocodify-file sans colonne `id` : contrôle de population ignoré")
        return None
    return set(df["id"])


# --------------------------------------------------------------------------- #
# build-table : prépare les features d'entraînement (1/3, hors pipeline)
# --------------------------------------------------------------------------- #
def cmd_build_table(args: argparse.Namespace) -> int:
    merged = _merged_table(args)
    table, diag = build_candidate_table(merged)
    diag = reconcile_population(table, _tocodify_ids(args.tocodify_file), diag)
    log_diagnostics(diag)

    if "correcte" not in table.columns:
        logger.error(
            "aucune vérité terrain dans ce run (`code_lvl4` vide) : l'entraînement "
            "est impossible. Il faut pointer sur un run d'ÉVALUATION (soumis avec "
            "input_file vide) ; un run de production ne porte pas de vérité."
        )
        return 1

    table = split_by_product(
        table, frac=args.split_frac, seed=args.split_seed
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Parquet, lu tel quel par R via duckdb : il porte les types et les doubles
    # exacts, donc R voit exactement les mêmes valeurs de features que Python.
    # Un CSV intermédiaire imposerait un formatage à 17 chiffres et un
    # `colClasses` explicite, deux mécanismes silencieux dont dépendrait
    # l'exactitude des seuils appris.
    table.to_parquet(out, index=False)

    diag["features_sha256"] = features_sha256()
    Path(args.out).with_suffix(".diagnostics.json").write_text(
        json.dumps(diag, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    logger.info(
        "table de features écrite : %s (%d lignes, %d train / %d test)",
        args.out,
        len(table),
        int((table["split"] == "train").sum()),
        int((table["split"] == "test").sum()),
    )
    return 0


# --------------------------------------------------------------------------- #
# finalize : mesure, calibration, MLflow (étape 3/3)
# --------------------------------------------------------------------------- #
def cmd_finalize(args: argparse.Namespace) -> int:
    art = Path(args.artifacts_dir)
    features = pd.read_parquet(art / "features.parquet")
    test = features[features["split"] == "test"].reset_index(drop=True)

    # Auto-contrôle : le scorer Python doit reproduire sirus.predict au bit près
    # sur CE modèle, avec CETTE version de R. Échoue l'étape sinon — mieux vaut
    # ne pas livrer un modèle qu'on ne sait pas scorer.
    rules_eval = load_rules(art / "rules_eval.json")
    if not verify_scorer_against_r(rules_eval, test, art / "proba_eval_R.csv"):
        return 1

    proba_test = score(rules_eval, test[FEATURES])
    calib = build_calibration(test, proba_test)

    # Le modèle LIVRÉ est le réajustement sur 100 %, dont la distribution de
    # proba n'est pas identique à celle du modèle 80 % qui a servi à mesurer.
    # Les deux plages sont logguées : c'est le diagnostic qui dit sur quelle
    # étendue de scores les métriques ci-dessus ont été établies, et sur quelle
    # étendue le modèle livré se prononcera réellement.
    rules_final = load_rules(art / "rules.json")
    proba_full = score(rules_final, features[FEATURES])
    shipped = {"proba_min": float(proba_full.min()), "proba_max": float(proba_full.max())}
    logger.info(
        "plage de proba — modèle calibré (80 %%) : [%.4f, %.4f] | modèle livré (100 %%) : [%.4f, %.4f]",
        calib["proba_min"],
        calib["proba_max"],
        shipped["proba_min"],
        shipped["proba_max"],
    )
    # rules.json est enrichi de la calibration : c'est lui qui voyage dans
    # MLflow, donc c'est lui qui porte les métriques et le balayage indicatif.
    # Aucun seuil n'y est inscrit : le pipeline n'en applique pas.
    payload = json.loads((art / "rules.json").read_text(encoding="utf-8"))
    payload["calibration"] = calib
    payload["shipped_model_proba_range"] = shipped
    payload["features_sha256"] = features_sha256()
    payload["training"] = {
        "run_id": args.run_id,
        "run_date": args.run_date,
        "n_products": int(features["id"].nunique()),
        "n_candidates": int(len(features)),
        "split_frac": args.split_frac,
        "split_seed": args.split_seed,
        "date": datetime.now(timezone.utc).date().isoformat(),
        # Référence du contrôle de dérive de `sirus-predict` : la part de
        # produits sans aucun candidat, mesurée quand les 4 classifieurs
        # fonctionnaient. Un bond sur un run futur signale un changement de
        # format en amont.
        "share_without_candidate": _share_without_candidate(art),
        # Distributions d'entraînement : `sirus-predict` les compare à celles du
        # run pour détecter une dérive amont (cf. le cas RAG-ANN). Résumées par
        # `feature_distribution`, la même fonction que celle utilisée à la
        # relecture — les confiances y sont mises en tranches, sans quoi la
        # comparaison de valeurs continues alerterait sur un run sain.
        "feature_distributions": feature_distribution(features),
    }
    (art / "rules.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    uri = log_to_mlflow(art, payload, experiment=args.experiment, run_name=f"{args.run_date}_{args.run_id}")
    logger.info("=" * 72)
    logger.info("Modèle SIRUS loggué. Pour l'utiliser, mettre dans argo/params.yaml :")
    logger.info("    sirus-model-uri: %s", uri)
    logger.info("=" * 72)
    logger.info(
        "Pour choisir un seuil d'exploitation du score : voir la section "
        "« Calibration de SIRUS » du rapport d'évaluation, qui donne l'accuracy "
        "par tranche de probabilité et l'arbitrage volume/fiabilité par seuil."
    )
    return 0


# --------------------------------------------------------------------------- #
# predict : étape sirus-predict, Python pur
# --------------------------------------------------------------------------- #
def cmd_predict(args: argparse.Namespace) -> int:
    model_dir = resolve_model_path(args.model_uri)
    rules = load_rules(model_dir)
    meta = rules.meta

    if meta.get("features_sha256") and meta["features_sha256"] != features_sha256():
        logger.warning(
            "la feature engineering a changé depuis l'entraînement de ce modèle "
            "(hash de src/candidates.py différent). Les seuils des règles ont été "
            "appris contre l'ancienne version : vérifier qu'aucune sentinelle ni "
            "l'ordre des features n'a bougé, ou réentraîner."
        )

    merged = _merged_table(args)

    # Run sans aucune observation : cas LÉGITIME, pas une erreur. Il se produit
    # quand l'amont n'a rien laissé à coder — par exemple si la regex a tout
    # capté, ou sur un échantillonnage à zéro. On écrit une sortie vide et on
    # sort en succès : faire échouer l'étape bloquerait un pipeline qui n'a
    # simplement rien à faire.
    if merged.empty:
        logger.info(
            "aucune observation à coder dans ce run : sortie vide écrite, rien à "
            "faire. Ce n'est pas une erreur (l'amont n'a rien laissé à coder)."
        )
        _write_parquet(merged, args.output_file)
        return 0

    # Pas de vérité terrain utilisée en production, même si la colonne existe :
    # truth_col=None rend l'erreur impossible plutôt qu'improbable.
    table, diag = build_candidate_table(merged, truth_col=None)
    diag = reconcile_population(table, _tocodify_ids(args.tocodify_file), diag)
    log_diagnostics(diag)
    check_drift(meta, table, diag)

    mask = scorable_mask(rules, table)
    scorable = table[mask].reset_index(drop=True)
    if scorable.empty:
        # Il y avait des observations, mais pas un seul candidat scorable : soit
        # les 4 classifieurs se sont tous abstenus, soit leurs codes ont changé
        # de forme (dé-zéro-paddés, tronqués autrement) et sont tous rejetés,
        # soit toutes les divisions COICOP sont inconnues du modèle. Aucun de ces
        # cas n'est normal — d'où l'échec, plutôt qu'une sortie entièrement vide
        # qui laisserait croire que le run a été traité.
        logger.error(
            "%d observation(s) en entrée mais aucun candidat scorable. Regarder "
            "les compteurs de rejet ci-dessus : s'ils sont élevés, la forme des "
            "codes émis en amont a probablement changé. Étape en échec.",
            len(merged),
        )
        return 1

    proba = score(rules, scorable[FEATURES])
    decided = pick_best(scorable, proba)

    # Produits sans candidat exploitable : absents de `decided`, ils
    # disparaîtraient de la sortie sans ce rattrapage explicite.
    all_ids = set(merged["id"])
    manquants = sorted(all_ids - set(decided["id"]))
    if manquants:
        decided = pd.concat(
            [
                decided,
                pd.DataFrame(
                    {
                        "id": manquants,
                        "sirus_code": pd.NA,
                        "sirus_proba": np.nan,
                        # Aucun candidat proposé par les 4 classifieurs : il n'y
                        # a pas d'argmax à prendre. `sirus_code` à NA et
                        # `sirus_n_candidats` à 0 le disent sans colonne dédiée.
                        "sirus_n_candidats": 0,
                    }
                ),
            ],
            ignore_index=True,
        )

    decided["sirus_model_uri"] = str(args.model_uri)
    decided["sirus_timestamp"] = datetime.now(timezone.utc).isoformat()

    # La table fusionnée complète est conservée en sortie : le report en a besoin
    # pour scorer les 4 classifieurs de base, comme il le fait aujourd'hui depuis
    # la sortie de decide-coicop.
    out = merged.merge(decided, on="id", how="left")
    if args.extra_columns_file:
        extra = _read_parquet(args.extra_columns_file)
        nouvelles = [c for c in extra.columns if c not in out.columns and c != "id"]
        if nouvelles:
            out = out.merge(extra[["id", *nouvelles]], on="id", how="left")
            logger.info("colonnes supplémentaires jointes : %s", nouvelles)

    assert len(out) == len(merged), "la jointure a changé le nombre de lignes"
    _write_parquet(out, args.output_file)

    n_sans_code = int(decided["sirus_code"].isna().sum())
    logger.info(
        "%d produit(s) codés, %d sans candidat exploitable",
        len(decided) - n_sans_code,
        n_sans_code,
    )
    logger.info(
        "score attribué — min/médiane/max : %.4f / %.4f / %.4f. La sortie étant "
        "une moyenne de sorties de règles, elle n'atteint jamais 0 ni 1 : c'est "
        "pourquoi un seuil d'exploitation, s'il en faut un en aval, ne se lit pas "
        "comme une probabilité usuelle (voir la section « Calibration de SIRUS » "
        "du rapport d'évaluation).",
        float(proba.min()),
        float(np.median(proba)),
        float(proba.max()),
    )
    logger.info("écrit : %s (%d lignes)", args.output_file, len(out))
    return 0


# --------------------------------------------------------------------------- #
def _add_source_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--lcs-file", required=True)
    p.add_argument("--rag-file", required=True)
    p.add_argument("--ttc-file", required=True)
    p.add_argument("--rag-annotations-file", default=None)
    p.add_argument("--mapping-file", default=None)
    p.add_argument(
        "--tocodify-file",
        default=None,
        help="Parquet de l'ensemble réellement à coder (codif-regex). Sert à "
        "détecter les produits perdus par la table de base LCS.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sirus", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    bt = sub.add_parser("build-table", help="Construit la table candidat-level d'entraînement")
    _add_source_args(bt)
    bt.add_argument("--out", required=True)
    bt.add_argument("--split-frac", type=float, default=0.8)
    bt.add_argument("--split-seed", type=int, default=42)
    bt.set_defaults(func=cmd_build_table)

    fi = sub.add_parser("finalize", help="Mesure, calibration et log MLflow du modèle ajusté")
    fi.add_argument("--artifacts-dir", required=True)
    fi.add_argument("--experiment", default="codif-coicop-sirus")
    fi.add_argument("--run-id", default="")
    fi.add_argument("--run-date", default="")
    fi.add_argument("--split-frac", type=float, default=0.8)
    fi.add_argument("--split-seed", type=int, default=42)
    fi.set_defaults(func=cmd_finalize)

    pr = sub.add_parser("predict", help="Applique un modèle SIRUS à un run")
    _add_source_args(pr)
    pr.add_argument("--model-uri", required=True)
    pr.add_argument("--output-file", required=True)
    pr.add_argument("--extra-columns-file", default=None)
    pr.set_defaults(func=cmd_predict)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
