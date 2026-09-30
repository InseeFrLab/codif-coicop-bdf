"""Traçabilité du code livré : quels classifieurs le portaient, et par quel chemin.

Le livrable dit quel code a été retenu (`predicted_code`) et d'où il vient
(`prediction_source`). Ces colonnes-ci disent en plus **sur quoi il repose** :
combien des 4 classifieurs l'avaient proposé, avec quelle confiance chacun s'est
prononcé, et si la conciliation a réellement choisi ou seulement entériné un
candidat unique.

Tout se calcule à partir du parquet de conciliation : `reconcile-llm` comme
`reconcile-sirus` y recopient la table fusionnée des classifieurs, codes déjà
tronqués au niveau 4 et élagués par `load_all_observations`. Ils sont donc dans
le même espace que `predicted_code`, lui-même passé par le garde-fou
`trunc_and_prune_lvl4` — à condition d'appeler ces fonctions APRÈS ce garde-fou.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import pandas as pd
from prune_codes.utils import is_answer

# (nom, colonne de code, colonne de score dans le parquet de conciliation,
#  colonne livrée). TTC ne vote que par son top-1, comme dans les `VOTERS` de
# reconcile-sirus/src/candidates.py : ses top-2/3 ne sont pas des propositions.
CLASSIFIERS = [
    ("lcs", "lcs_code", "lcs_distance", "lcs_distance"),
    ("rag", "rag_code", "rag_confidence", "rag_confidence"),
    ("ragann", "ragann_code", "ragann_confidence", "ragann_confidence"),
    ("ttc", "ttc_code_1", "ttc_conf_1", "ttc_confidence"),
]

# Même définition d'un code exploitable que `reconcile-sirus/src/candidates.py`
# (CODE_RE) : un libellé renvoyé par erreur n'est pas une proposition. Recopiée
# plutôt qu'importée — export-results ne dépend pas de reconcile-sirus, et
# modifier candidates.py changerait le hash embarqué dans les modèles SIRUS.
CODE_RE = re.compile(r"^\d{2}(\.\d+)*$")

# Valeurs de `reconciliation_type`.
CANDIDAT_UNIQUE = "candidat_unique"  # SIRUS : un seul code proposé, retenu sans modèle
CHOIX_SIRUS = "choix_sirus"  # SIRUS : argmax du score parmi au moins deux candidats
CONSENSUS = "consensus"  # juge LLM : court-circuit, le juge n'a pas été appelé
CHOIX_LLM = "choix_llm"  # juge LLM : arbitrage effectif
REGEX = "regex"
AUCUN_CODE = "aucun_code"

# Colonnes propres à chaque conciliation, nécessaires au type de conciliation.
REGIME_COLUMNS = {
    "sirus": ["sirus_route", "sirus_n_candidats"],
    "llm": ["llm_model"],
}

# Colonnes livrées, dans l'ordre où elles terminent le fichier.
DELIVERED_COLUMNS = [
    "reconciliation_type",
    "n_classifiers_agreeing",
    *[f"proposed_by_{name}" for name, _, _, _ in CLASSIFIERS],
    *[out for _, _, _, out in CLASSIFIERS],
]


def decision_columns(available: Iterable[str], source: str) -> list[str]:
    """Colonnes à lire dans le parquet de conciliation, parmi celles présentes.

    Un run antérieur à un classifieur (RAG-annotations) ou au routage SIRUS
    (`sirus_route`) n'a pas toutes les colonnes : on lit ce qui existe, et le
    reste ressort NA au lieu de faire échouer l'étape.
    """
    available = set(available)
    wanted = [c for _, code, score, _ in CLASSIFIERS for c in (code, score)]
    wanted += REGIME_COLUMNS[source]
    return [c for c in wanted if c in available]


def _proposal(series: pd.Series) -> pd.Series:
    """Code proposé, nettoyé ; NA pour une abstention ou une valeur non-code."""
    text = series.astype("string").str.strip()
    ok = series.map(is_answer).astype(bool) & text.str.match(CODE_RE).fillna(False).astype(bool)
    return text.where(ok, pd.NA)


def add_classifier_traceability(result: pd.DataFrame, in_reconciliation: pd.Series) -> pd.DataFrame:
    """Ajoute votes, décompte et scores bruts des 4 classifieurs.

    ``in_reconciliation`` marque les lignes dont le code vient de la conciliation.
    Ailleurs (code regex, aucun code), les classifieurs n'ont pas décidé du code
    livré : votes et décompte y valent NA, pas False/0.
    """
    out = result.copy()
    code = out["predicted_code"].astype("string").str.strip()
    votes = []
    for name, code_col, score_col, out_col in CLASSIFIERS:
        proposed = (
            _proposal(out[code_col]) if code_col in out.columns
            else pd.Series(pd.NA, index=out.index, dtype="string")
        )
        vote = (proposed == code).fillna(False).astype("boolean")
        vote[~in_reconciliation] = pd.NA
        out[f"proposed_by_{name}"] = vote
        votes.append(vote)
        # Score brut, même quand le classifieur a proposé un autre code : c'est
        # la confiance d'un classifieur en désaccord qu'on veut pouvoir lire.
        out[out_col] = (
            pd.to_numeric(out[score_col], errors="coerce").astype("Float64")
            if score_col in out.columns
            else pd.Series(pd.NA, index=out.index, dtype="Float64")
        )
    n = pd.concat(votes, axis=1).astype("Int64").sum(axis=1, min_count=1).astype("Int64")
    n[~in_reconciliation] = pd.NA
    out["n_classifiers_agreeing"] = n
    return out


def reconciliation_type(result: pd.DataFrame, source: str) -> pd.Series:
    """Par quel chemin le code livré a été obtenu.

    ``result`` porte ``_decision_code`` (code de la conciliation), ``predict_code``
    (regex) et les colonnes de régime de la conciliation.
    """
    has_decision = result["_decision_code"].notna()
    kind = pd.Series(pd.NA, index=result.index, dtype="string")

    if source == "sirus":
        if "sirus_route" in result.columns:
            route = result["sirus_route"].astype("string")
            kind[has_decision & (route == "candidat_unique")] = CANDIDAT_UNIQUE
            kind[has_decision & (route == "modele")] = CHOIX_SIRUS
        # Run antérieur au routage (ou route manquante) : le nombre de candidats
        # dit la même chose. Un candidat unique n'offrait rien à choisir.
        if "sirus_n_candidats" in result.columns:
            n = pd.to_numeric(result["sirus_n_candidats"], errors="coerce")
            todo = has_decision & kind.isna()
            kind[todo & (n == 1)] = CANDIDAT_UNIQUE
            kind[todo & (n >= 2)] = CHOIX_SIRUS
    else:
        regime = result["llm_model"] if "llm_model" in result.columns else pd.Series(pd.NA, index=result.index)
        kind[has_decision & (regime == "consensus")] = CONSENSUS
        kind[has_decision & kind.isna()] = CHOIX_LLM

    todo = kind.isna()
    kind[todo & result["predict_code"].notna()] = REGEX
    kind[kind.isna()] = AUCUN_CODE
    return kind
