"""Réimplémentation Python de ``sirus.predict`` pour ``type = "classif"``.

Pourquoi réimplémenter plutôt qu'appeler R : le paquet `sirus` a été archivé du
CRAN (2026-01-15), il faut le compiler depuis une source patchée, et il
**abandonne sur tout NA** (``sirus:::data.check``) — un niveau de facteur inédit
tuerait l'étape en production. Le scoring, lui, est une moyenne. On garde donc R
pour l'entraînement (rare, hors pipeline) et Python pour la production.

L'équivalence est prouvée par ``tests/test_scorer_golden.py`` (fichier de
référence produit par R) et re-vérifiée à chaque entraînement par l'auto-contrôle
de ``reconcile-sirus/train.sh`` sur son test 20 %.

Correspondance avec le paquet (sirus 0.3.3) :

    R/sirus.R:399-400     type == 'classif'  ->  pred <- apply(data.rule, 1, mean)
    R/sirus.R:405         aucune règle       ->  pred <- rep(sirus.m$mean, n)
    R/sirus_utility.R:373 numérique          ->  X < seuil  /  X >= seuil
                          facteur            ->  X %in% niveaux
                          plusieurs conditions -> ET logique
    R/sirus_utility.R:407 sortie             ->  outputs[1] si la règle tient,
                                                 outputs[2] sinon

ATTENTION — l'agrégation est une moyenne **non pondérée**. Le champ
``rule.weights`` existe dans l'objet R en classification (rempli à 1/K) et
``sirus.print`` l'affiche, mais ``sirus.predict`` ne le lit jamais : le remplacer
par des valeurs arbitraires ne change pas les prédictions. La régression ridge à
coefficients positifs est l'agrégation de la variante *régression* (Bénard et al.
2021, AISTATS). Pour la classification, l'article (EJS 15:427-505, éq. 3.3) dit
explicitement « we simply average », et précise avoir testé une agrégation
linéaire régularisée puis l'avoir écartée. Ne pas « corriger » ce fichier en y
ajoutant des poids.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("sirus.scorer")

_MLFLOW_PREFIXES = ("runs:/", "models:/", "mlflow-artifacts:/")

# Version du contrat rules.json que ce scorer sait lire. À incrémenter côté R
# (R/export_rules.R) ET ici dès qu'un champ change de sens — jamais d'un seul
# côté, sinon on perd la seule protection contre une relecture erronée.
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Condition:
    var: str
    op: str  # "<" | ">=" | "in"
    value: float | None = None
    levels: tuple[str, ...] = ()


@dataclass(frozen=True)
class Rule:
    conditions: tuple[Condition, ...]
    out_true: float
    out_false: float


@dataclass(frozen=True)
class Rules:
    features: tuple[str, ...]
    factor_levels: dict[str, tuple[str, ...]]
    mean: float
    rules: tuple[Rule, ...]
    meta: dict

    @property
    def factor_features(self) -> frozenset[str]:
        return frozenset(self.factor_levels)


def resolve_model_path(model_uri: str | Path) -> Path:
    """Résout une URI MLflow en chemin local ; laisse passer un chemin tel quel.

    Même mécanique que ``classify-ttc/src/predict.py::_resolve_mlflow_path``, pour
    que ``reconcile-sirus-model-uri`` se comporte exactement comme ``classify-ttc-model-uri``.
    """
    text = str(model_uri).strip()
    if not text:
        raise ValueError(
            "URI de modèle SIRUS vide. Renseigner `reconcile-sirus-model-uri` dans "
            "argo/params.yaml : c'est l'URI qu'affiche `reconcile-sirus/train.sh` en fin "
            "d'entraînement (celui-ci se lance à la main, hors pipeline), et "
            "qu'on retrouve dans MLflow. Un chemin local vers un dossier "
            "contenant rules.json est aussi accepté."
        )
    if text.startswith(_MLFLOW_PREFIXES):
        import mlflow

        logger.info("téléchargement des artefacts MLflow depuis %s", text)
        local = mlflow.artifacts.download_artifacts(artifact_uri=text)
        logger.info("téléchargés dans %s", local)
        return Path(local)
    return Path(text)


def load_rules(path: str | Path) -> Rules:
    """Charge ``rules.json``. Chaque flottant y est une chaîne, convertie ici.

    Les nombres sont sérialisés en chaînes (``sprintf("%.17g")`` côté R, et les
    seuils verbatim car ils sont déjà des chaînes dans l'objet modèle). C'est la
    seule forme qui round-trip : ``jsonlite::toJSON`` arrondit à 4 chiffres par
    défaut et ``digits = NA`` n'en donne que 15, ce qui ne suffit pas.
    """
    path = Path(path)
    if path.is_dir():
        path = path / "rules.json"
    payload = json.loads(path.read_text(encoding="utf-8"))

    # Le contrat doit échouer bruyamment s'il évolue. Ce scorer est la seule
    # chose qui sait interpréter rules.json, et il doit rester correct des
    # années après que le paquet R qui a produit le fichier soit devenu
    # difficile à réinstaller (`sirus` est archivé du CRAN). Une version de
    # schéma inconnue signifie qu'on lirait un fichier selon des conventions qui
    # ne sont plus les siennes — mieux vaut refuser que produire des
    # probabilités plausibles et fausses.
    version = payload.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"rules.json en schema_version={version!r}, ce scorer implémente la "
            f"version {SCHEMA_VERSION}. Réentraîner le modèle, ou porter le "
            "scorer sur l'ancien schéma en connaissance de cause."
        )

    if payload.get("type") != "classif":
        raise ValueError(
            f"type de modèle non géré : {payload.get('type')!r}. Ce scorer "
            "n'implémente que l'agrégation par moyenne de la classification ; "
            "la régression utilise une ridge et exigerait poids et intercept."
        )

    rules = []
    for raw in payload["rules"]:
        conditions = []
        for cond in raw["conditions"]:
            if cond["op"] == "in":
                conditions.append(
                    Condition(var=cond["var"], op="in", levels=tuple(cond["levels"]))
                )
            elif cond["op"] in ("<", ">="):
                conditions.append(
                    Condition(var=cond["var"], op=cond["op"], value=float(cond["value"]))
                )
            else:
                raise ValueError(f"opérateur de condition inconnu : {cond['op']!r}")
        out_true, out_false = (float(v) for v in raw["outputs"])
        rules.append(Rule(conditions=tuple(conditions), out_true=out_true, out_false=out_false))

    return Rules(
        features=tuple(payload["features"]),
        factor_levels={k: tuple(v) for k, v in payload.get("factor_levels", {}).items()},
        mean=float(payload["mean"]),
        rules=tuple(rules),
        meta=payload,
    )


def scorable_mask(rules: Rules, table: pd.DataFrame) -> pd.Series:
    """Lignes que le modèle peut scorer.

    Un niveau de facteur inédit (nouvelle division COICOP) doit être **écarté**,
    pas scoré. En R il ferait échouer ``data.check`` ; côté Python, ``isin()``
    renverrait tranquillement ``False`` et produirait une probabilité *plausible*
    que R n'aurait jamais produite. Écarter est le seul comportement qui
    reproduit la sémantique de R tout en restant exploitable en production : le
    candidat est perdu, le produit part en reprise s'il n'en a pas d'autre.
    """
    mask = pd.Series(True, index=table.index)
    for var, levels in rules.factor_levels.items():
        if var not in table.columns:
            continue
        known = table[var].astype("string").isin(levels)
        n_unknown = int((~known).sum())
        if n_unknown:
            unseen = sorted(set(table.loc[~known, var].astype("string").dropna()))
            logger.error(
                "%d candidat(s) écarté(s) : `%s` porte %d modalité(s) absente(s) du "
                "modèle (%s). Le modèle ne peut pas les scorer — volume perdu, pas "
                "erreur de codage. Un réentraînement les intégrerait.",
                n_unknown,
                var,
                len(unseen),
                unseen[:10],
            )
        mask &= known
    return mask


def score(rules: Rules, table: pd.DataFrame) -> np.ndarray:
    """Probabilité qu'un candidat soit le bon code, une valeur par ligne."""
    missing = [f for f in rules.features if f not in table.columns]
    if missing:
        raise ValueError(f"features absentes de la table à scorer : {missing}")
    if tuple(f for f in table.columns if f in rules.features) != rules.features:
        logger.warning(
            "l'ordre des features de la table diffère de celui du modèle. Sans "
            "effet sur le score (indexation par nom), mais signale que FEATURES a "
            "peut-être bougé d'un seul côté."
        )

    if not rules.rules:
        # Aucune règle sélectionnée : R renvoie le taux de base (sirus.R:405).
        logger.warning("modèle sans aucune règle : repli sur la moyenne (%.6f)", rules.mean)
        return np.full(len(table), rules.mean, dtype="float64")

    numeric_vars = {c.var for r in rules.rules for c in r.conditions if c.op != "in"}
    for var in numeric_vars:
        col = pd.to_numeric(table[var], errors="coerce")
        if col.isna().any():
            raise ValueError(
                f"valeurs manquantes dans `{var}`. NaN vaut False pour `<` ET `>=`, "
                "soit une branche que R n'emprunte jamais (il abandonne sur NA). "
                "build_candidate_table rend ce cas impossible : une occurrence ici "
                "signale une table construite autrement."
            )

    outputs = np.empty((len(table), len(rules.rules)), dtype="float64")
    for j, rule in enumerate(rules.rules):
        held = np.ones(len(table), dtype=bool)
        for cond in rule.conditions:
            if cond.op == "in":
                held &= table[cond.var].astype("string").isin(cond.levels).to_numpy()
            elif cond.op == "<":
                held &= table[cond.var].to_numpy(dtype="float64") < cond.value
            else:
                held &= table[cond.var].to_numpy(dtype="float64") >= cond.value
        outputs[:, j] = np.where(held, rule.out_true, rule.out_false)

    # Moyenne non pondérée — cf. l'avertissement en tête de module.
    return _r_mean(outputs)


def _r_mean(a: np.ndarray) -> np.ndarray:
    """Moyenne par ligne, avec l'algorithme exact de ``mean()`` de R.

    ``numpy.mean`` et R ne donnent pas le même dernier bit : R (summary.c,
    ``rmean``) accumule en ``long double`` puis fait une passe de correction sur
    les résidus, là où numpy somme en float64 par paires. Sur 20 règles l'écart
    est d'environ 1 ULP (~1.7e-16) — négligeable numériquement, mais il suffit à
    faire échouer une comparaison bit-à-bit, et on perdrait alors la capacité de
    distinguer « bruit d'arrondi » de « le portage a réellement divergé ».

    Reproduire l'algorithme rend l'équivalence exacte et donc vérifiable :

        s = somme(x) / n            (en long double)
        t = somme(x - s)            (en long double)
        moyenne = s + t / n

    Vérifié bit-exact contre ``sirus.predict`` sur l'intégralité d'un jeu de
    test. Sur une plateforme où ``np.longdouble`` n'offre pas plus de précision
    que ``double`` (ARM, Windows), on retomberait à ~1 ULP d'écart :
    ``verify_scorer_against_r`` le traite alors comme un avertissement et non
    comme une erreur.
    """
    x = a.astype(np.longdouble)
    n = x.shape[1]
    s = np.add.reduce(x, axis=1) / n
    t = np.add.reduce(x - s[:, None], axis=1)
    return (s + t / n).astype(np.float64)


def pick_best(table: pd.DataFrame, proba: np.ndarray) -> pd.DataFrame:
    """Un seul candidat retenu par produit : celui de plus haute probabilité.

    Renvoie ``id``, ``sirus_code``, ``sirus_proba``, ``sirus_n_candidats``.

    **Aucun verdict.** Cette étape livre un code et un score, elle ne dit pas
    s'il faut les exploiter sans relecture : décider d'un seuil est une question
    métier, qui s'instruit sur la table accuracy-vs-seuil du rapport
    d'évaluation. Un verdict calculé ici serait de toute façon entièrement
    déduit de ``sirus_proba``, donc sans information propre, tout en figeant
    dans le parquet un réglage que l'aval ne pourrait plus rediscuter sans
    réentraîner.

    Clé sur ``id`` (UUID stable), jamais sur un index positionnel : c'est ce qui
    permet à `export-results` et au report de joindre le résultat.
    """
    scored = table[["id", "code_candidat"]].copy()
    scored["sirus_proba"] = proba
    # Ex æquo : on garde le premier, comme `slice_max(..., with_ties = FALSE)`.
    best_idx = scored.groupby("id", sort=False)["sirus_proba"].idxmax()
    best = scored.loc[best_idx].copy()
    n_cand = scored.groupby("id", sort=False).size().rename("sirus_n_candidats")

    out = best.rename(columns={"code_candidat": "sirus_code"}).merge(
        n_cand, left_on="id", right_index=True, how="left"
    )
    return out.reset_index(drop=True)
