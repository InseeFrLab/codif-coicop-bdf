"""Construction de la table candidat-level qui alimente SIRUS.

**Ce module est le contrat entre l'entraînement et la prédiction.** Les deux
chemins (l'entraînement manuel et l'étape `reconcile-sirus`) l'appellent, et un
écart entre eux serait invisible : les prédictions resteraient plausibles mais
fausses, parce que les seuils des règles ont été appris contre *ces* valeurs de
sentinelles exactes.
D'où le module unique, et le hash de ce fichier embarqué dans `rules.json`
(cf. ``features_sha256`` dans ``scorer.py``).

Le cadrage : on ne demande pas au modèle « quel classifieur croire ? » (4 classes)
mais « ce candidat précis est-il le bon code ? » (binaire). Une ligne = un couple
(produit, code candidat distinct proposé par au moins un des 4 classifieurs).
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from prune_codes.utils import is_answer

logger = logging.getLogger("sirus.candidates")

# Ordre load-bearing : `sirus.predict` vérifie `all(colnames == data.names)`.
# Le scorer Python indexe par nom, donc l'ordre ne l'affecte pas, mais un écart
# d'ordre signale que FEATURES a été modifié d'un seul côté.
FEATURES = [
    "vote_lcs",
    "vote_rag",
    "vote_ragann",
    "vote_ttc",
    "nb_votants",
    "conf_rag",
    "conf_ragann",
    "conf_ttc",
    "dist_lcs",
    "code_candidat_n1",
]

# Sentinelles utilisées quand un classifieur n'a PAS voté pour le candidat de la
# ligne. Choisies hors de la plage réelle des valeurs (confiances dans [0, 1],
# `lcs_distance` dans [0, ~0.8]) pour ne jamais entrer en collision avec une
# vraie valeur faible.
#
# NE JAMAIS LES CHANGER sans réentraîner : un seuil appris comme
# `conf_ttc < 0.79` sépare « TTC a voté avec une confiance faible » de « TTC n'a
# pas voté » précisément parce que l'absence vaut -1.
CONF_MISSING = -1.0
DIST_MISSING = 1.5

# (nom, colonne de code, colonne de score, colonne de feature, sentinelle)
VOTERS = [
    ("lcs", "lcs_code", "lcs_distance", "dist_lcs", DIST_MISSING),
    ("rag", "rag_code", "rag_confidence", "conf_rag", CONF_MISSING),
    ("ragann", "ragann_code", "ragann_confidence", "conf_ragann", CONF_MISSING),
    ("ttc", "ttc_code_1", "ttc_conf_1", "conf_ttc", CONF_MISSING),
]

# Un code COICOP exploitable : division sur 2 chiffres, puis des segments
# pointés. Filtre ce qui n'est ni une abstention connue ni un code — un libellé
# renvoyé par erreur par un LLM, par exemple, ne doit pas devenir un candidat.
CODE_RE = re.compile(r"^\d{2}(\.\d+)*$")

# Colonne de vérité terrain : la forme canonique produite par `reconcile-llm`
# (tronquée niveau 4 + hiérarchies linéaires élaguées). JAMAIS `code`, qui est
# l'annotation brute niveau 5 : comparer des candidats prunés à une vérité non
# prunée donnerait une accuracy proche de zéro.
TRUTH_COL = "code_lvl4"


def features_sha256() -> str:
    """Hash de ce fichier, embarqué dans ``rules.json`` à l'entraînement.

    `reconcile-sirus` le compare au sien : un écart signifie que la feature
    engineering a bougé depuis l'entraînement du modèle chargé.
    """
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _is_proposal(series: pd.Series) -> pd.Series:
    """Masque des valeurs qui sont un vrai code candidat.

    Rejette les NULL/NaN, les abstentions textuelles (`prune_codes.utils.is_answer`,
    partagé avec le report) et tout ce qui n'a pas la forme d'un code COICOP.
    """
    answered = series.map(is_answer)
    well_formed = series.astype("string").str.strip().str.match(CODE_RE).fillna(False)
    return (answered & well_formed).astype(bool)


def build_candidate_table(
    merged: pd.DataFrame,
    *,
    truth_col: str | None = TRUTH_COL,
) -> tuple[pd.DataFrame, dict]:
    """Passe la table large de `load_all_observations` en table candidat-level.

    Parameters
    ----------
    merged
        Sortie de ``reconcile_llm.load_all_observations(..., mapping_path=...)``.
        Le `mapping_path` n'est pas optionnel en pratique : sans lui les codes
        candidats ne sont pas tronqués au niveau 4 et ne sont donc comparables ni
        entre eux ni à la vérité.
    truth_col
        Colonne de vérité terrain. ``None`` ou colonne absente/entièrement nulle
        (mode production) ⇒ la colonne ``correcte`` est **absente** du résultat,
        pas remplie de NaN : une colonne float toute-NaN passerait jusqu'à R et
        déclencherait son ``data.check``.

    Returns
    -------
    (table, diagnostics)
        ``table`` a une ligne par (``id``, ``code_candidat``) avec les 10
        features, et ``correcte`` en mode évaluation. ``diagnostics`` porte de
        quoi rendre visible ce qui a été écarté — voir ``log_diagnostics``.
    """
    if "id" not in merged.columns:
        raise ValueError("colonne `id` absente : la table candidat-level est clé sur `id`")
    if not merged["id"].is_unique:
        n_dup = int(merged["id"].duplicated().sum())
        raise ValueError(
            f"{n_dup} `id` dupliqué(s) en entrée. `load_all_observations` fait des "
            "jointures `how=\"left\"` sur `id` : un doublon dans un parquet amont "
            "démultiplie silencieusement les lignes. À investiguer en amont."
        )

    df = merged.copy()
    diagnostics: dict = {"n_products_in": len(df)}

    # Un classifieur peut manquer entièrement (run antérieur à son intégration,
    # ou `--rag-annotations-file` non fourni). On crée la colonne vide plutôt que
    # d'échouer : le modèle a besoin des 10 features, et l'absence se traduira en
    # `vote_* = 0` + sentinelle, ce que SIRUS sait interpréter.
    for name, code_col, score_col, _, _ in VOTERS:
        for col in (code_col, score_col):
            if col not in df.columns:
                logger.warning(
                    "colonne `%s` absente : le classifieur %s est traité comme "
                    "n'ayant rien proposé sur tout le run",
                    col,
                    name,
                )
                df[col] = pd.NA

    # --- Candidats : une ligne par (id, code distinct réellement proposé) ------
    proposal_masks = {code_col: _is_proposal(df[code_col]) for _, code_col, _, _, _ in VOTERS}
    diagnostics["n_rejected_not_a_proposal"] = {
        code_col: int((~mask & df[code_col].notna()).sum())
        for code_col, mask in proposal_masks.items()
    }

    code_cols = [code_col for _, code_col, _, _, _ in VOTERS]
    long = df[["id", *code_cols]].melt(
        id_vars="id", value_vars=code_cols, var_name="_source", value_name="code_candidat"
    )
    # Rejouer le masque sur la table longue : `melt` a perdu l'alignement.
    long = long[_is_proposal(long["code_candidat"])]
    long["code_candidat"] = long["code_candidat"].astype("string").str.strip()
    candidates = long.drop_duplicates(subset=["id", "code_candidat"])[["id", "code_candidat"]]

    # --- Features -------------------------------------------------------------
    join_cols = ["id", *code_cols, *[s for _, _, s, _, _ in VOTERS]]
    if truth_col and truth_col in df.columns:
        join_cols.append(truth_col)
    table = candidates.merge(df[join_cols], on="id", how="left")

    vote_cols = []
    for name, code_col, score_col, feature_col, sentinel in VOTERS:
        vote_col = f"vote_{name}"
        voted = (
            _is_proposal(table[code_col])
            & (table[code_col].astype("string").str.strip() == table["code_candidat"])
        )
        table[vote_col] = voted.astype("int64")
        vote_cols.append(vote_col)
        # Le `fillna(sentinel)` est DANS la branche « a voté » : c'est le
        # `coalesce(score, sentinelle)` du R. Le mettre après le `where` ne
        # serait pas équivalent — un classifieur qui a voté sans score doit
        # recevoir la sentinelle, pas un NaN.
        score = pd.to_numeric(table[score_col], errors="coerce").fillna(sentinel)
        table[feature_col] = np.where(voted.to_numpy(), score.to_numpy(), sentinel).astype(
            "float64"
        )

    table["nb_votants"] = table[vote_cols].sum(axis=1).astype("int64")
    if not (table["nb_votants"] >= 1).all():
        n_bad = int((table["nb_votants"] < 1).sum())
        raise AssertionError(
            f"{n_bad} candidat(s) sans aucun votant. Un candidat n'existe que parce "
            "qu'un classifieur l'a proposé : ce cas signifie que la détection des "
            "propositions et le calcul des votes se contredisent."
        )

    # Division COICOP du candidat. Équivalent à une troncature niveau 1 : toutes
    # les divisions sont sur 2 chiffres zéro-paddés ("01".."13", "98", "99").
    table["code_candidat_n1"] = table["code_candidat"].str.split(".").str[0].astype("string")

    # --- Cible (évaluation seulement) -----------------------------------------
    has_truth = bool(
        truth_col
        and truth_col in table.columns
        and table[truth_col].notna().any()
    )
    if has_truth:
        table["correcte"] = (
            table["code_candidat"] == table[truth_col].astype("string").str.strip()
        ).astype("int64")
    diagnostics["mode"] = "evaluation" if has_truth else "production"

    out_cols = ["id", "code_candidat", *FEATURES] + (["correcte"] if has_truth else [])
    # Tri stable : le `melt` + `drop_duplicates` ci-dessus laisse un ordre qui
    # dépend de l'ordre des colonnes de vote. Un ordre déterministe est requis
    # par le test golden et rend les parquets de deux runs comparables.
    table = (
        table[out_cols]
        .sort_values(["id", "code_candidat"], kind="stable")
        .reset_index(drop=True)
    )

    ids_with = set(table["id"])
    diagnostics["ids_without_candidate"] = sorted(set(df["id"]) - ids_with)
    diagnostics["n_products_with_candidates"] = len(ids_with)
    diagnostics["n_candidates"] = len(table)
    diagnostics["candidates_per_product"] = (
        table.groupby("id").size().value_counts().sort_index().to_dict()
    )
    return table, diagnostics


def reconcile_population(
    table: pd.DataFrame, tocodify_ids: set | None, diagnostics: dict
) -> dict:
    """Compare la population scorée à l'ensemble réellement à coder.

    `load_all_observations` prend la sortie de `classify-lcs` comme table de base et
    joint le reste en `how="left"`. Un produit absent de la sortie LCS disparaît
    donc de la table candidat-level *et* du décompte « sans candidat » — il
    s'évapore. Ce contrôle le rend visible.
    """
    if tocodify_ids is None:
        return diagnostics
    missing = tocodify_ids - set(table["id"]) - set(diagnostics["ids_without_candidate"])
    diagnostics["n_products_missing_from_lcs_base"] = len(missing)
    diagnostics["ids_missing_from_lcs_base"] = sorted(missing)[:50]
    if missing:
        logger.error(
            "%d produit(s) à coder sont absents de la table de base LCS : ils ne "
            "seront ni scorés ni comptés comme « sans candidat ». Exemples : %s",
            len(missing),
            sorted(missing)[:5],
        )
    return diagnostics


def log_diagnostics(diagnostics: dict) -> None:
    """Rend visible tout ce qui a été écarté — rien ne doit disparaître en silence."""
    d = diagnostics
    logger.info(
        "mode=%s | %d produits en entrée → %d avec candidats (%d candidats, %.2f/produit)",
        d.get("mode"),
        d.get("n_products_in", 0),
        d.get("n_products_with_candidates", 0),
        d.get("n_candidates", 0),
        d.get("n_candidates", 0) / max(d.get("n_products_with_candidates", 1), 1),
    )
    logger.info("répartition candidats/produit : %s", d.get("candidates_per_product"))
    rejected = {k: v for k, v in (d.get("n_rejected_not_a_proposal") or {}).items() if v}
    if rejected:
        logger.warning(
            "valeurs non exploitables comme code candidat (abstention ou forme "
            "invalide), par colonne : %s",
            rejected,
        )
    n_no_cand = len(d.get("ids_without_candidate", []))
    if n_no_cand:
        logger.warning(
            "%d produit(s) sans aucun candidat : structurellement non codables "
            "automatiquement, ils partiront en reprise manuelle.",
            n_no_cand,
        )
