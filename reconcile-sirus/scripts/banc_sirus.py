#!/usr/bin/env python
"""Banc de test des réglages d'entraînement SIRUS, par validation croisée.

Trois questions, toutes jugées sur la même cible : l'accuracy de SIRUS sur les
produits à budget ≤ 50 € où il a un vrai choix à faire (≥ 2 codes candidats).

1. entraîner sur tous les produits, ou seulement sur ceux à budget ≤ 50 € ;
2. ``num_rule`` à 10, 15 ou 20 (``max_depth`` et graine inchangés) ;
3. garder ou retirer du train les produits à candidat unique.

Plan factoriel complet (prix × num_rule × population), validation croisée par
produit : les folds sont tirés une fois, les filtres ne touchent que le train,
et le fold de test est identique pour toutes les configurations — d'où des
comparaisons appariées (McNemar) sur l'ensemble des prédictions hors
échantillon.

Aucun modèle n'est livré : seuls des modèles d'évaluation sont ajustés
(``fit_sirus.R --eval-only=true``) et l'expérience MLflow est distincte de
celle des modèles de production.

Usage (depuis ``reconcile-sirus/``) :

    uv run python scripts/banc_sirus.py 2026-10-07/codif-2jcqh
    uv run python scripts/banc_sirus.py 2026-10-07/codif-2jcqh --smoke --no-mlflow
    uv run python scripts/banc_sirus.py 2026-10-07/codif-2jcqh --log-only
"""

from __future__ import annotations

import argparse
import itertools
import logging
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

MODULE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE))

from reconcile_llm import _read_parquet, load_all_observations  # noqa: E402

from src.candidates import FEATURES, build_candidate_table, features_sha256, reconcile_population  # noqa: E402
from src.scorer import load_rules, route_and_score  # noqa: E402
from src.train import keep_multi_candidates, verify_scorer_against_r  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[banc] %(levelname)s %(message)s", stream=sys.stdout)
logger = logging.getLogger("banc")

SEUIL_PRIX = 50.0
MAX_DEPTH = 2
SEED = 42


# --------------------------------------------------------------------------- #
# Données
# --------------------------------------------------------------------------- #
def charger_table(root: str) -> pd.DataFrame:
    """Table candidat (mêmes entrées et fonctions que `build-table`) + budget."""
    merged = load_all_observations(
        lcs_path=f"{root}/classify-lcs/raw_test_LCS.parquet",
        rag_path=f"{root}/classify-rag-notices/predictions.parquet",
        ttc_path=f"{root}/classify-ttc/predictions.parquet",
        rag_annotations_path=f"{root}/classify-rag-annotations/predictions.parquet",
        mapping_path=f"{root}/prune-codes/mapping_lvl4.parquet",
    )
    table, diag = build_candidate_table(merged)
    tocodify = _read_parquet(f"{root}/classify-regex/raw_test_without_regex.parquet")
    reconcile_population(table, set(tocodify["id"]), diag)
    if "correcte" not in table.columns:
        raise SystemExit("aucune vérité terrain dans ce run : banc impossible")

    # Budget = MONT_DEP de la ligne, celui qu'utilise `evaluate`.
    obs = _read_parquet(f"{root}/build-datasets/observations.parquet")[["id", "budget"]]
    table = table.merge(obs, on="id", how="left", validate="many_to_one")
    table["n_candidats"] = table.groupby("id")["id"].transform("size")
    return table


def attribuer_folds(table: pd.DataFrame, n_folds: int, seed: int) -> pd.DataFrame:
    """Un fold par produit (jamais par ligne : les candidats d'un produit restent ensemble)."""
    ids = np.sort(table["id"].unique())
    rng = np.random.default_rng(seed)
    folds = pd.Series(rng.permutation(len(ids)) % n_folds, index=ids)
    return table.assign(fold=table["id"].map(folds).astype(int))


# --------------------------------------------------------------------------- #
# Configurations
# --------------------------------------------------------------------------- #
def configurations(num_rules: list[int]) -> list[dict]:
    return [
        {"prix": prix, "num_rule": nr, "population": pop, "nom": f"prix-{prix}_rules-{nr}_pop-{pop}"}
        for prix, nr, pop in itertools.product(("tous", "le50"), num_rules, ("tous", "multi"))
    ]


def table_fold(table: pd.DataFrame, cfg: dict, fold: int) -> pd.DataFrame:
    """Train filtré selon la configuration, test = le fold entier."""
    t = table.assign(split=np.where(table["fold"] == fold, "test", "train"))
    train, test = t[t["split"] == "train"], t[t["split"] == "test"]
    if cfg["prix"] == "le50":
        # NaN exclus : un budget inconnu n'est pas « ≤ 50 € ».
        train = train[train["budget"] <= SEUIL_PRIX]
    if cfg["population"] == "multi":
        train = keep_multi_candidates(train)
    return pd.concat([train, test], ignore_index=True)


def ajuster(table: pd.DataFrame, cfg: dict, fold: int, out_root: Path) -> Path:
    """Écrit la table du fold et lance `fit_sirus.R --eval-only` (sauté si déjà fait)."""
    out = out_root / cfg["nom"] / f"fold{fold}"
    if (out / "rules_eval.json").exists() and (out / "proba_eval_R.csv").exists():
        return out
    out.mkdir(parents=True, exist_ok=True)
    feat = table_fold(table, cfg, fold)
    feat[["id", "code_candidat", *FEATURES, "correcte", "split"]].to_parquet(out / "features.parquet", index=False)
    cmd = [
        "Rscript", "R/fit_sirus.R",
        f"--features={out / 'features.parquet'}", f"--out-dir={out}",
        f"--num-rule={cfg['num_rule']}", f"--max-depth={MAX_DEPTH}", f"--seed={SEED}",
        f"--features-sha256={features_sha256()}", "--eval-only=true",
    ]
    debut = time.monotonic()
    res = subprocess.run(cmd, cwd=MODULE, capture_output=True, text=True)
    (out / "fit.log").write_text(res.stdout + "\n" + res.stderr, encoding="utf-8")
    if res.returncode != 0:
        # Pas de rules_eval.json partiel : la reprise relancerait sinon un fold cassé comme fait.
        (out / "rules_eval.json").unlink(missing_ok=True)
        raise RuntimeError(f"fit_sirus.R en échec pour {cfg['nom']}/fold{fold} (voir {out / 'fit.log'})")
    logger.info("%s/fold%d ajusté en %.0f s", cfg["nom"], fold, time.monotonic() - debut)
    return out


# --------------------------------------------------------------------------- #
# Évaluation
# --------------------------------------------------------------------------- #
def predire_fold(table: pd.DataFrame, cfg: dict, fold: int, out: Path) -> tuple[pd.DataFrame, int]:
    """Décision produit du fold de test, comme en production (routage + argmax)."""
    test = table[table["fold"] == fold].reset_index(drop=True)
    rules = load_rules(out / "rules_eval.json")
    if not verify_scorer_against_r(rules, test, out / "proba_eval_R.csv"):
        raise RuntimeError(f"auto-contrôle R ↔ Python en échec : {cfg['nom']}/fold{fold}")

    decided, _ = route_and_score(rules, test)
    produits = test.groupby("id").agg(
        budget=("budget", "first"), n_candidats=("n_candidats", "first"), borne_haute=("correcte", "max")
    )
    choix = decided.merge(
        test[["id", "code_candidat", "correcte"]],
        left_on=["id", "sirus_code"], right_on=["id", "code_candidat"], how="left",
    ).set_index("id")[["sirus_code", "sirus_proba", "sirus_route", "correcte"]]
    # Un produit sans candidat scorable (division inconnue) n'a pas de code : compté faux.
    pred = produits.join(choix, how="left")
    pred["correcte"] = pred["correcte"].fillna(0).astype(int)
    pred = pred.reset_index().assign(config=cfg["nom"], fold=fold, **{k: cfg[k] for k in ("prix", "num_rule", "population")})
    return pred, len(rules.rules)


def masques(pred: pd.DataFrame) -> dict[str, pd.Series]:
    multi = pred["n_candidats"] >= 2
    le50 = pred["budget"] <= SEUIL_PRIX
    gt50 = pred["budget"] > SEUIL_PRIX
    return {
        "multi_le50": multi & le50,
        "multi_all": multi,
        "multi_gt50": multi & gt50,
        "product_le50": le50,
        "product_all": pd.Series(True, index=pred.index),
    }


def metriques(pred: pd.DataFrame) -> dict[str, float]:
    m = {f"acc_{k}": float(pred.loc[v, "correcte"].mean()) for k, v in masques(pred).items()}
    cible = masques(pred)["multi_le50"]
    m["upper_bound_multi_le50"] = float(pred.loc[cible, "borne_haute"].mean())
    m["n_multi_le50"] = int(cible.sum())
    p = pred["sirus_proba"].dropna()
    m["proba_min"], m["proba_max"] = float(p.min()), float(p.max())
    return m


def mcnemar_exact(b: int, c: int) -> float:
    """p-valeur bilatérale exacte du test de McNemar (binomiale sur les discordants)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    queue = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * queue)


def comparaisons(oof: pd.DataFrame, cfgs: list[dict]) -> pd.DataFrame:
    """Paires de configurations ne différant que par un facteur, sur la cible."""
    cible = oof[(oof["n_candidats"] >= 2) & (oof["budget"] <= SEUIL_PRIX)]
    large = cible.pivot(index="id", columns="config", values="correcte")
    par_nom = {c["nom"]: c for c in cfgs}
    lignes = []
    for a, b in itertools.combinations(par_nom, 2):
        ca, cb = par_nom[a], par_nom[b]
        diff = [k for k in ("prix", "num_rule", "population") if ca[k] != cb[k]]
        if len(diff) != 1:
            continue
        question = {"prix": "Q1 prix", "num_rule": "Q2 num_rule", "population": "Q3 population"}[diff[0]]
        # Sens de lecture : B = variante testée (le50, multi, plus de règles).
        if diff[0] == "num_rule" and ca["num_rule"] > cb["num_rule"]:
            a, b = b, a
        if diff[0] != "num_rule" and ca[diff[0]] != "tous":
            a, b = b, a
        xa, xb = large[a], large[b]
        b_seul = int(((xb == 1) & (xa == 0)).sum())
        a_seul = int(((xa == 1) & (xb == 0)).sum())
        lignes.append({
            "question": question, "config_A": a, "config_B": b,
            "acc_A": float(xa.mean()), "acc_B": float(xb.mean()), "ecart_B_moins_A_pts": 100 * float(xb.mean() - xa.mean()),
            "B_juste_A_faux": b_seul, "A_juste_B_faux": a_seul, "p_mcnemar": mcnemar_exact(b_seul, a_seul),
            "n_produits": int(len(large)),
        })
    if not lignes:
        return pd.DataFrame()
    return pd.DataFrame(lignes).sort_values(["question", "config_A", "config_B"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# MLflow
# --------------------------------------------------------------------------- #
def log_mlflow(args, cfgs, par_fold, synthese, out_root: Path) -> None:
    import mlflow

    mlflow.set_experiment(args.experiment)
    run_date, run_id = args.run.split("/")
    with mlflow.start_run(run_name=f"banc_{run_date}_{run_id}"):
        mlflow.log_params({
            "run": args.run, "folds": args.folds, "split_seed": SEED, "seuil_prix": SEUIL_PRIX,
            "max_depth": MAX_DEPTH, "seed": SEED, "num_rules": ",".join(map(str, args.num_rules)),
            "features_sha256": features_sha256(), "critere": "acc_multi_le50",
        })
        for nom in ("synthese.csv", "comparaisons.csv", "oof_predictions.parquet"):
            mlflow.log_artifact(str(out_root / nom))
        for cfg in cfgs:
            with mlflow.start_run(run_name=cfg["nom"], nested=True):
                mlflow.log_params({
                    "prix": cfg["prix"], "population": cfg["population"],
                    "num_rule": cfg["num_rule"], "max_depth": MAX_DEPTH, "folds": args.folds,
                })
                ligne = synthese.loc[cfg["nom"]].drop(["prix", "population", "num_rule"])
                mlflow.log_metrics({k: float(v) for k, v in ligne.items() if not pd.isna(v)})
                for _, r in par_fold[par_fold["config"] == cfg["nom"]].iterrows():
                    mlflow.log_metrics(
                        {k: float(r[k]) for k in r.index if k.startswith("acc_") or k == "num_rule_selected"},
                        step=int(r["fold"]),
                    )
                for k in range(args.folds):
                    d = out_root / cfg["nom"] / f"fold{k}"
                    for f in ("rules_printed.txt", "rules_eval.json"):
                        if (d / f).exists():
                            mlflow.log_artifact(str(d / f), artifact_path=f"fold{k}")
    logger.info("résultats logués dans l'expérience MLflow « %s »", args.experiment)


# --------------------------------------------------------------------------- #
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", help="<date>/<run_id>, ex. 2026-10-07/codif-2jcqh")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--jobs", type=int, default=6)
    p.add_argument("--num-rules", type=lambda s: [int(x) for x in s.split(",")], default=[10, 15, 20])
    p.add_argument("--experiment", default="codif-coicop-sirus-banc")
    p.add_argument("--runs-root", default=os.environ.get("SIRUS_RUNS_ROOT", "s3://projet-budget-famille/data/workflow_runs"))
    p.add_argument("--no-mlflow", action="store_true")
    p.add_argument("--log-only", action="store_true", help="Relance l'évaluation et le log MLflow sans ajuster (folds déjà faits)")
    p.add_argument("--smoke", action="store_true", help="Une seule configuration (le50, multi, plus petit num_rule), fold 0")
    args = p.parse_args()

    if not args.no_mlflow and not os.environ.get("MLFLOW_TRACKING_URI"):
        raise SystemExit("MLFLOW_TRACKING_URI absente (ou passer --no-mlflow)")

    run_id = args.run.split("/")[-1]
    out_root = MODULE / "artifacts" / f"banc-{run_id}"
    out_root.mkdir(parents=True, exist_ok=True)

    cache = out_root / "table.parquet"
    if cache.exists():
        table = pd.read_parquet(cache)
    else:
        table = attribuer_folds(charger_table(f"{args.runs_root}/{args.run}"), args.folds, SEED)
        table.to_parquet(cache, index=False)
    prod = table.drop_duplicates("id")
    logger.info(
        "%d produits (%d candidats) ; cible ≤ %.0f € à ≥ 2 candidats : %d produits ; budget manquant : %d",
        len(prod), len(table), SEUIL_PRIX,
        int(((prod["n_candidats"] >= 2) & (prod["budget"] <= SEUIL_PRIX)).sum()), int(prod["budget"].isna().sum()),
    )

    cfgs = configurations(args.num_rules)
    folds = list(range(args.folds))
    if args.smoke:
        cfgs = [c for c in cfgs if c["prix"] == "le50" and c["population"] == "multi"][:1]
        folds = [0]
    taches = [(c, k) for c in cfgs for k in folds]

    if not args.log_only:
        logger.info("%d ajustements (%d en parallèle)", len(taches), args.jobs)
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            futurs = {ex.submit(ajuster, table, c, k, out_root): (c["nom"], k) for c, k in taches}
            for f in as_completed(futurs):
                f.result()

    preds, lignes_fold = [], []
    for c, k in taches:
        pred, nr = predire_fold(table, c, k, out_root / c["nom"] / f"fold{k}")
        preds.append(pred)
        lignes_fold.append({"config": c["nom"], "fold": k, "num_rule_selected": nr, **metriques(pred)})
    oof = pd.concat(preds, ignore_index=True)
    par_fold = pd.DataFrame(lignes_fold)

    synthese = pd.DataFrame([
        {"config": c["nom"], **{k: c[k] for k in ("prix", "num_rule", "population")}, **metriques(oof[oof["config"] == c["nom"]])}
        for c in cfgs
    ]).set_index("config")
    stats = par_fold.groupby("config").agg(
        acc_multi_le50_std_folds=("acc_multi_le50", "std"), num_rule_selected_mean=("num_rule_selected", "mean")
    )
    synthese = synthese.join(stats).sort_values("acc_multi_le50", ascending=False)
    comp = comparaisons(oof, cfgs)

    synthese.to_csv(out_root / "synthese.csv")
    comp.to_csv(out_root / "comparaisons.csv", index=False)
    par_fold.to_csv(out_root / "par_fold.csv", index=False)
    oof.to_parquet(out_root / "oof_predictions.parquet", index=False)

    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.4f}".format):
        print("\n=== Synthèse (tri sur acc_multi_le50, prédictions hors échantillon) ===")
        print(synthese[["acc_multi_le50", "acc_multi_le50_std_folds", "acc_multi_all", "acc_product_le50",
                        "upper_bound_multi_le50", "num_rule_selected_mean", "proba_min", "proba_max"]])
        if len(comp):
            print("\n=== Comparaisons appariées sur la cible (McNemar exact) ===")
            print(comp.drop(columns=["n_produits"]))

    if not args.no_mlflow:
        log_mlflow(args, cfgs, par_fold, synthese, out_root)
    logger.info("sorties locales : %s", out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
