"""Mesure, calibration et log MLflow de l'entraînement (`reconcile-sirus/train.sh`).

Le découpage 80/20 vit ici (et non dans R) pour que l'aléa soit à un seul
endroit : R reçoit un parquet portant déjà une colonne ``split``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .candidates import FEATURES
from .scorer import Rules, score

logger = logging.getLogger("sirus.train")


def split_by_product(table: pd.DataFrame, *, frac: float, seed: int) -> pd.DataFrame:
    """Ajoute une colonne ``split`` valant "train"/"test".

    Le découpage est **par produit** (`id`), jamais par ligne : les candidats
    concurrents d'un même produit doivent rester du même côté, sinon le modèle
    voit à l'entraînement des candidats du produit qu'il évaluera en test.
    """
    ids = np.sort(table["id"].unique())
    rng = np.random.default_rng(seed)
    train_ids = set(rng.choice(ids, size=round(frac * len(ids)), replace=False))
    out = table.copy()
    out["split"] = np.where(out["id"].isin(train_ids), "train", "test")
    return out


def verify_scorer_against_r(
    rules: Rules, test: pd.DataFrame, proba_r_path: str | Path
) -> bool:
    """Le scorer Python reproduit-il ``sirus.predict`` sur ce modèle précis ?

    Le test golden prouve l'équivalence au jour où on l'a figé ; ceci la prouve
    à chaque entraînement, sur ce modèle et cette version de R. Si les deux
    divergent, mieux vaut ne pas livrer le modèle du tout : on ne saurait pas le
    scorer en production.
    """
    # Tout en chaînes : `proba_R` doit garder ses 17 chiffres (un parse flottant
    # de pandas les préserverait, mais autant ne pas en dépendre), et un code
    # comme "01.1" serait sinon lu comme le flottant 1.1 — la jointure
    # échouerait sur un type incompatible.
    ref = pd.read_csv(proba_r_path, dtype=str)
    got = score(rules, test[FEATURES])

    cle = ["id", "code_candidat"]
    fusion = (
        test[cle]
        .astype({c: "string" for c in cle})
        .assign(proba_py=got)
        .merge(
            ref.astype({c: "string" for c in cle}),
            on=cle,
            how="inner",
            validate="one_to_one",
        )
    )
    if len(fusion) != len(test):
        logger.error(
            "auto-contrôle impossible : %d lignes appariées sur %d entre la table "
            "de test et proba_eval_R.csv.",
            len(fusion),
            len(test),
        )
        return False

    attendu = fusion["proba_R"].map(float).to_numpy()
    obtenu = fusion["proba_py"].to_numpy()
    if np.array_equal(obtenu, attendu):
        logger.info(
            "auto-contrôle OK : le scorer Python reproduit sirus.predict au bit "
            "près sur les %d lignes de test.",
            len(fusion),
        )
        return True

    ecart = np.abs(obtenu - attendu)
    pire = int(np.argmax(ecart))

    # Distinguer le bruit d'arrondi de la vraie divergence. `_r_mean` reproduit
    # l'algorithme de mean() de R et donne normalement l'égalité bit-à-bit ;
    # mais sur une plateforme où np.longdouble n'a pas de précision étendue
    # (ARM, Windows), il reste ~1 ULP d'écart. Quelques ULP sont bénins ; un
    # écart plus grand signale une perte de précision à l'export ou un scorer
    # qui ne reproduit plus la sémantique du paquet.
    tolerance_ulp = 8 * np.spacing(np.maximum(np.abs(attendu), 1e-300))
    if np.all(ecart <= tolerance_ulp):
        logger.warning(
            "auto-contrôle : écart max %.3e sur %d/%d lignes, dans le bruit "
            "d'arrondi (quelques ULP). Attendu si np.longdouble n'offre pas de "
            "précision étendue sur cette plateforme ; sans conséquence sur les "
            "décisions. Aucune ligne ne change de côté du seuil.",
            float(ecart.max()),
            int((ecart > 0).sum()),
            len(fusion),
        )
        return True

    logger.error(
        "AUTO-CONTRÔLE EN ÉCHEC : le scorer Python diverge de sirus.predict. "
        "%d/%d lignes concernées, écart max %.3e (ligne %s : Python %.17g vs R %s). "
        "Ne pas livrer ce modèle — soit l'export des règles perd de la précision, "
        "soit le scorer ne reproduit plus la sémantique du paquet.",
        int((ecart > 0).sum()),
        len(fusion),
        float(ecart.max()),
        fusion.iloc[pire][["id", "code_candidat"]].to_dict(),
        obtenu[pire],
        fusion.iloc[pire]["proba_R"],
    )
    return False


# Features suivies par le contrôle de dérive, et comment les résumer.
#
# Les confiances sont CONTINUES : compter leurs valeurs brutes donnerait des
# milliers de modalités uniques, et une référence tronquée aux plus fréquentes ne
# couvrirait qu'une poignée de lignes — la comparaison alerterait sur un run
# parfaitement sain. On les résume donc en tranches qui portent le sens utile :
# le classifieur a-t-il voté, et a-t-il hésité ?
#
# `conf_ragann` est suivie en particulier parce que l'expérimentation a montré
# qu'elle est quasi binaire (sentinelle ou 1.0) et qu'elle porte la moitié des
# règles : un ajustement de prompt en amont la déplace et repointe l'essentiel
# du modèle.
_TRANCHES_CONFIANCE = [-np.inf, -0.5, 0.5, 0.9, 0.999, np.inf]
_ETIQUETTES_CONFIANCE = ["absent", "faible", "moyenne", "haute", "certaine"]
DRIFT_FEATURES = ("conf_ragann", "conf_ttc", "nb_votants", "code_candidat_n1")


def feature_distribution(table: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Résumé distributionnel des features suivies, comparable d'un run à l'autre.

    Utilisé **des deux côtés** — écrit par `finalize`, relu par `check_drift` —
    pour que la comparaison porte sur des grandeurs construites à l'identique.
    Un résumé calculé différemment de part et d'autre produirait des écarts qui
    ne diraient rien de la dérive.
    """
    out: dict[str, dict[str, int]] = {}
    for col in DRIFT_FEATURES:
        if col not in table.columns:
            continue
        if col.startswith("conf_"):
            serie = pd.cut(
                pd.to_numeric(table[col], errors="coerce"),
                bins=_TRANCHES_CONFIANCE,
                labels=_ETIQUETTES_CONFIANCE,
            )
        else:
            serie = table[col].astype("string")
        out[col] = {str(k): int(v) for k, v in serie.value_counts().items()}
    return out


def check_drift(meta: dict, table: pd.DataFrame, diag: dict) -> None:
    """Compare le run à ce que le modèle a vu à l'entraînement.

    Sert à détecter une **dérive amont** : un classifieur dont le prompt, le
    modèle ou le format de sortie a changé continue de produire des codes, mais
    plus le même signal. Le modèle, lui, applique des seuils appris contre
    l'ancien signal — il reste confiant en se trompant, et rien ne le signale
    autrement.

    Deux comparaisons, toutes deux disponibles sans vérité terrain :

    - la part de produits sans aucun candidat exploitable. Un bond ici veut dire
      que les codes émis en amont ne sont plus reconnus (changement de format le
      plus souvent), et c'est le symptôme le plus visible ;
    - la distribution des features discriminantes, notamment ``conf_ragann``,
      dont l'expérimentation a montré qu'elle est quasi binaire et qu'elle porte
      la moitié des règles : un simple ajustement de prompt en amont la déplace
      et repointe l'essentiel du modèle.

    Émet des logs, ne fait jamais échouer l'étape : l'ampleur acceptable d'une
    dérive est un jugement métier, pas un seuil qu'on puisse coder ici.
    """
    entrainement = (meta.get("training") or {}).get("feature_distributions") or {}
    if not entrainement:
        logger.info(
            "modèle sans distributions d'entraînement enregistrées : contrôle de "
            "dérive impossible (modèle produit par une version antérieure de "
            "`finalize`)."
        )
        return

    n_produits = diag.get("n_products_in", 0)
    if n_produits:
        part_sans = len(diag.get("ids_without_candidate", [])) / n_produits
        # Référence d'entraînement : la part de produits sans candidat y était
        # mesurée sur des données où les 4 classifieurs fonctionnaient.
        part_ref = (meta.get("training") or {}).get("share_without_candidate")
        if part_ref is not None and part_sans - part_ref > 0.10:
            logger.error(
                "part de produits sans aucun candidat : %.1f %% sur ce run contre "
                "%.1f %% à l'entraînement. Un tel écart signale que les codes émis "
                "en amont ne sont plus reconnus — vérifier leur format avant "
                "d'exploiter cette sortie.",
                100 * part_sans,
                100 * part_ref,
            )

    # Le résumé du run est calculé par la MÊME fonction que celle qui a produit
    # la référence : sans ça, l'écart mesuré refléterait la façon de compter et
    # non la dérive.
    observe = feature_distribution(table)
    for col, ref in entrainement.items():
        if col not in observe:
            continue
        total_ref = sum(ref.values())
        obs = pd.Series(observe[col])
        total_obs = int(obs.sum())
        if not total_ref or not total_obs:
            continue
        # Distance en variation totale sur les modalités vues à l'entraînement :
        # une seule mesure, lisible, et robuste à l'apparition de modalités
        # nouvelles (qui gonflent la distance sans la faire déborder).
        modalites = set(ref) | set(obs.index)
        distance = 0.5 * sum(
            abs(obs.get(m, 0) / total_obs - ref.get(m, 0) / total_ref) for m in modalites
        )
        if distance > 0.20:
            logger.warning(
                "`%s` : distribution éloignée de celle de l'entraînement "
                "(distance %.2f). Les seuils des règles ont été appris contre "
                "l'ancienne distribution : suspecter une évolution du classifieur "
                "amont plutôt qu'un problème de SIRUS.",
                col,
                distance,
            )
        else:
            logger.info("`%s` : distribution conforme (distance %.2f).", col, distance)


def build_calibration(test: pd.DataFrame, proba: np.ndarray) -> dict:
    """Accuracy et calibration du modèle, mesurées sur le test 20 %.

    Tout ce qui est renvoyé ici est de l'**information d'analyse**, pas un
    réglage : le pipeline n'applique aucun seuil, `reconcile-sirus` livre un code
    et un score. Le balayage volume/fiabilité est loggué dans MLflow pour
    pouvoir comparer deux entraînements ; l'analyse qui sert à *décider* d'un
    seuil d'exploitation vit dans le rapport d'évaluation, qui la recalcule sur
    les données du run.

    Les tableaux bruts ``proba``/``correct`` sont conservés précisément pour que
    n'importe quel seuil puisse être réévalué hors ligne, sans réentraîner.
    """
    df = test[["id", "correcte"]].copy()
    df["proba"] = proba

    # Décision au niveau produit : le candidat de plus haute proba.
    best = df.loc[df.groupby("id", sort=False)["proba"].idxmax()].reset_index(drop=True)

    # Borne haute : le bon code est-il proposé par au moins un classifieur ?
    # Au-delà, aucun modèle de conciliation ne peut faire mieux — il faudrait un
    # meilleur candidat en amont.
    borne_haute = float(df.groupby("id")["correcte"].max().mean())

    metrics = {
        "accuracy_candidate": float(((proba > 0.5).astype(int) == test["correcte"]).mean()),
        "accuracy_product": float(best["correcte"].mean()),
        "upper_bound": borne_haute,
        "n_test_products": int(len(best)),
    }

    # Balayage volume/fiabilité : à titre INDICATIF, pour que MLflow porte de
    # quoi comparer deux entraînements. Aucun seuil n'en est déduit, et le
    # pipeline n'en applique aucun.
    sweep = []
    for seuil in np.round(np.arange(0.05, 1.0, 0.01), 2):
        confiant = best["proba"] >= seuil
        n = int(confiant.sum())
        sweep.append(
            {
                "seuil": float(seuil),
                "volume": float(confiant.mean()),
                "reliability": float(best.loc[confiant, "correcte"].mean()) if n else float("nan"),
                "n": n,
            }
        )

    bins = (
        best.assign(tranche=pd.cut(best["proba"], bins=np.arange(0, 1.1, 0.1)))
        .groupby("tranche", observed=True)
        .agg(n=("correcte", "size"), pct_correct=("correcte", "mean"))
        .reset_index()
    )

    logger.info(
        "accuracy candidat %.1f %% | produit %.1f %% | borne haute %.1f %%",
        100 * metrics["accuracy_candidate"],
        100 * metrics["accuracy_product"],
        100 * metrics["upper_bound"],
    )
    logger.info(
        "plage de proba atteignable : [%.4f, %.4f] — la sortie étant une moyenne "
        "de sorties de règles, elle n'atteint jamais 0 ni 1",
        float(best["proba"].min()),
        float(best["proba"].max()),
    )

    return {
        "metrics": metrics,
        "proba_min": float(best["proba"].min()),
        "proba_max": float(best["proba"].max()),
        "threshold_sweep": sweep,
        "bins": [
            {"lo": float(r.tranche.left), "hi": float(r.tranche.right),
             "n": int(r.n), "pct_correct": float(r.pct_correct)}
            for r in bins.itertuples()
        ],
        # Tableaux bruts : permettent de réévaluer n'importe quel seuil hors
        # ligne, sans réentraîner.
        "proba": [float(x) for x in best["proba"]],
        "correct": [int(x) for x in best["correcte"]],
    }


def log_to_mlflow(
    artifacts_dir: Path, payload: dict, *, experiment: str, run_name: str
) -> str:
    """Logue métriques et artefacts, et renvoie l'URI à recopier dans params.yaml."""
    import mlflow

    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name or None) as run:
        calib = payload["calibration"]
        mlflow.log_metrics({k: v for k, v in calib["metrics"].items()})
        mlflow.log_metrics(
            {
                "proba_min": calib["proba_min"],
                "proba_max": calib["proba_max"],
                "num_rule_selected": payload["hyperparams"]["num_rule_selected"],
                "num_trees": payload["hyperparams"]["num_trees"],
            }
        )
        mlflow.log_params(payload["hyperparams"])

        # Balayage volume/fiabilité : information permettant de comparer deux
        # entraînements. Le pipeline n'applique aucun seuil ; l'analyse qui sert
        # à en décider un vit dans le rapport d'évaluation.
        try:
            mlflow.log_table(
                pd.DataFrame(calib["threshold_sweep"]), artifact_file="threshold_sweep.json"
            )
        except Exception as exc:  # noqa: BLE001 - le log ne doit pas faire échouer l'étape
            logger.warning("log_table du balayage de seuils indisponible : %s", exc)

        # `artifact_path="model"` : donne une URI de la même forme que
        # classify-ttc-model-uri (mlflow-artifacts:/<exp>/<run>/artifacts/model).
        for nom in ("rules.json", "rules_printed.txt", "rules_eval.json"):
            chemin = artifacts_dir / nom
            if chemin.exists():
                mlflow.log_artifact(str(chemin), artifact_path="model")

        uri = f"mlflow-artifacts:/{run.info.experiment_id}/{run.info.run_id}/artifacts/model"
        (artifacts_dir / "model_uri.txt").write_text(uri, encoding="utf-8")
        return uri
