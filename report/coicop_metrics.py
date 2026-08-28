"""Shared COICOP metric helpers.

Used by both ``report.qmd`` (so the rendered report and the metrics logged to
MLflow never diverge) and ``main.py`` (which logs to MLflow).

COICOP codes are stored as dot-separated strings (e.g. ``"01.1.2.3.4"``).
Level-k accuracy compares the first ``k`` segments of the prediction with the
ground truth.

Two conventions coexist, and both are reported (see ``accuracy_table``):

``inclusive=False`` (strict, historical)
    Rows whose ground truth is shallower than ``k`` are **excluded** from the
    denominator: that depth cannot be evaluated for them. A prediction shorter
    than ``k`` counts as an error. Answers "among the observations codeable at
    depth k, what share did we get right?".

``inclusive=True``
    **Every** row with a ground truth counts, whatever its depth. The truth and
    the prediction are compared on ``parts[:k]``, so a truth shallower than
    ``k`` demands a prediction equal to it. This is only meaningful on
    *canonical* codes (``code_lvl4``): once linear hierarchies are pruned, a
    truth of ``01.3`` means ``01.3`` **is** the level-4 answer, not a truncation
    of it. Answers "over the whole population, what share did we place
    correctly in the level-k grid?".
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd
from prune.utils import ABSTENTION_SENTINELS as prune_abstention_sentinels
from prune.utils import is_answer

# Prediction columns produced by decide-coicop, in display order.
# ``llm_code`` is the final prediction (after the LLM arbitration step).
# ``ragann_code`` (RAG on annotated examples) is only present when the
# annotation-RAG brick ran — methods absent from the data are skipped.
METHODS = [
    ("LCS", "lcs_code"),
    ("RAG", "rag_code"),
    ("RAG-annot", "ragann_code"),
    ("TTC", "ttc_code_1"),
    ("LLM", "llm_code"),
    ("SIRUS", "sirus_code"),
]

# Les deux conciliations possibles (paramètre Argo `conciliation`), de la plus
# ancienne à la plus récente. Elles sont EXCLUSIVES : un run porte l'une ou
# l'autre, jamais les deux.
CONCILIATIONS = [
    ("LLM", "llm_code"),
    ("SIRUS", "sirus_code"),
]
LEVELS = [1, 2, 3, 4, 5]

# Canonical codes never exceed 4 segments (level 5 is truncated away by the
# `prune` step), so the inclusive convention saturates at level 4: comparing
# parts[:4] already compares the codes in full.
CANONICAL_LEVELS = [1, 2, 3, 4]

# Ground-truth columns produced by decide-coicop: the raw annotation, and its
# canonical form (truncated to level 4 + linear hierarchies pruned).
TRUTH_COL_RAW = "code"
TRUTH_COL_CANONICAL = "code_lvl4"

# decide-coicop records the arbitration regime in `llm_model`: the consensus
# short-circuit (`try_consensus_decision` — every available source agrees with
# TTC top-1 and its confidence is >= 0.90) retains that code without calling the
# judge, so `llm_code == ttc_code_1` by construction on those rows. Pooling the
# two regimes makes the LLM and TTC columns partly tautological, hence the
# per-regime split below.
REGIME_COL = "llm_model"
CONSENSUS_LABEL = "consensus"
# (label affiché, suffixe de métrique MLflow)
REGIMES = [("Consensus", "consensus"), ("Arbitré", "arbitrated")]
# Niveau auquel le découpage par régime est rapporté (niveau cible de l'enquête),
# partagé par report.qmd et main.py pour que les deux ne divergent pas.
REGIME_LEVEL = 4

# Le vocabulaire de l'abstention (« ce classifieur a-t-il répondu ? ») vit dans
# `prune.utils` : c'est du vocabulaire de code COICOP, partagé avec le module
# `sirus/`, qui ne peut pas dépendre de `report/` (jupyter, matplotlib, seaborn).
# Ré-exporté ici pour que report.qmd, prediction_report.qmd et main.py continuent
# de l'importer depuis `coicop_metrics` sans changement.
ABSTENTION_SENTINELS = prune_abstention_sentinels

# Colonnes booléennes par lesquelles une brique déclare explicitement que
# l'observation n'est pas codable, indépendamment du code qu'elle émet.
CODABLE_FLAGS = {"RAG": "codable", "RAG-annot": "ragann_codable"}


def available_methods(data: pd.DataFrame) -> list[tuple[str, str]]:
    """METHODS whose prediction column is present in ``data`` (backward compatible
    with runs produced before a given method existed)."""
    return [(name, col) for name, col in METHODS if col in data.columns]


def final_decision(data: pd.DataFrame, *, strict: bool = True):
    """Nom d'affichage et colonne de la conciliation qui a tranché dans ce run.

    Les deux conciliations étant exclusives, un run ne porte qu'une des deux
    colonnes. Résoudre ici plutôt que d'écrire ``llm_code`` en dur partout
    permet au rapport de rendre dans les deux modes — et les runs antérieurs à
    SIRUS continuent de tomber sur ``llm_code``.

    ``strict=True`` (défaut) lève si aucune colonne n'est présente : c'est le bon
    comportement pour le rapport d'évaluation, dont tout l'objet est de scorer un
    code final. ``strict=False`` renvoie ``(None, None)``, pour le rapport de
    production qui sait déjà se dégrader section par section.
    """
    for name, col in CONCILIATIONS:
        if col in data.columns:
            return name, col
    if not strict:
        return None, None
    raise KeyError(
        "aucune colonne de conciliation dans ces données "
        f"({[c for _, c in CONCILIATIONS]}) : le parquet ne vient ni de "
        "decide-coicop ni de sirus-predict."
    )


def truth_column(data: pd.DataFrame) -> str:
    """Ground-truth column to score against.

    Predictions live in the canonical (pruned) code space, so the truth must
    too — otherwise a correct canonical prediction such as ``01.3`` is scored
    against a raw annotation ``01.3.0.0.1`` and counted wrong. Falls back to the
    raw ``code`` for runs produced before ``code_lvl4`` existed.
    """
    return (
        TRUTH_COL_CANONICAL
        if TRUTH_COL_CANONICAL in data.columns
        else TRUTH_COL_RAW
    )


def code_parts(s) -> list[str]:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return []
    s = str(s).strip()
    if not s:
        return []
    return s.split(".")


def level_result(truth: str, pred: str, k: int, *, inclusive: bool = False):
    """Return True/False/None for one observation at level ``k``.

    None means the observation is not counted at all (see the module docstring
    for the two conventions): with ``inclusive=False`` when the truth is
    shallower than ``k``, with ``inclusive=True`` only when there is no truth.
    """
    tp = code_parts(truth)
    pp = code_parts(pred)
    if inclusive:
        if not tp:
            return None
        if not pp:
            return False
        return tp[:k] == pp[:k]
    if len(tp) < k:
        return None
    if len(pp) < k:
        return False
    return tp[:k] == pp[:k]


def accuracy_series(
    truth: pd.Series, pred: pd.Series, k: int, *, inclusive: bool = False
) -> pd.Series:
    """Series of True/False/NA indexed like truth."""
    out = [level_result(t, p, k, inclusive=inclusive) for t, p in zip(truth, pred)]
    return pd.Series(out, index=truth.index, dtype="object")


def accuracy(
    truth: pd.Series, pred: pd.Series, k: int, *, inclusive: bool = False
) -> tuple[int, int, float]:
    s = accuracy_series(truth, pred, k, inclusive=inclusive)
    applicable = s.notna()
    n_app = int(applicable.sum())
    n_ok = int((s == True).sum())  # noqa: E712
    return n_ok, n_app, (n_ok / n_app if n_app else float("nan"))


def accuracy_table(
    data: pd.DataFrame, *, inclusive: bool = False, levels: list[int] | None = None
) -> pd.DataFrame:
    """Accuracy per method and per level, scored against ``truth_column(data)``.

    With ``inclusive=True`` the denominator is the same at every level (all rows
    carrying a ground truth), so it is reported once in the table caption rather
    than per column.
    """
    truth = data[truth_column(data)]
    levels = levels or (CANONICAL_LEVELS if inclusive else LEVELS)
    rows = []
    for name, col in available_methods(data):
        row = {"méthode": name}
        for k in levels:
            _, n_app, acc = accuracy(truth, data[col], k, inclusive=inclusive)
            row[f"niv{k}" if inclusive else f"niv{k} (n={n_app})"] = acc
        rows.append(row)
    return pd.DataFrame(rows).set_index("méthode")


# `is_answer` est importé de `prune.utils` (voir ABSTENTION_SENTINELS ci-dessus)
# et ré-exporté ici : report.qmd et prediction_report.qmd l'importent depuis ce
# module. Voir ``coverage_table`` pour séparer « coder faux » de « ne pas coder ».


def answer_mask(pred: pd.Series) -> pd.Series:
    """Boolean mask of the rows where ``pred`` carries a code."""
    return pred.map(is_answer)


def coverage_table(
    data: pd.DataFrame, k: int = REGIME_LEVEL, *, inclusive: bool = True
) -> pd.DataFrame:
    """Split each method's accuracy into coverage × accuracy-when-answering.

    The global accuracy conflates two very different failures: coding the wrong
    thing, and refusing to code. Under the inclusive convention an abstention is
    counted as an error, so the global figure factorises exactly:

        accuracy globale = couverture × accuracy sur les réponses

    A method can therefore look mediocre because it is wrong, or because it is
    silent — two problems with opposite remedies.
    """
    truth = data[truth_column(data)]
    n_total = len(data)
    rows = []
    for name, col in available_methods(data):
        answered = answer_mask(data[col])
        n_ans = int(answered.sum())
        _, _, acc_all = accuracy(truth, data[col], k, inclusive=inclusive)
        _, _, acc_ans = accuracy(
            truth[answered], data[col][answered], k, inclusive=inclusive
        )
        rows.append(
            {
                "méthode": name,
                "couverture": (n_ans / n_total) if n_total else float("nan"),
                "abstentions": n_total - n_ans,
                f"accuracy niv{k} sur réponses": acc_ans,
                f"accuracy niv{k} globale": acc_all,
            }
        )
    return pd.DataFrame(rows).set_index("méthode")


def declared_refusal_table(data: pd.DataFrame, k: int = REGIME_LEVEL) -> pd.DataFrame | None:
    """Cross a brick's ``codable`` flag with whether it emitted a code anyway.

    The two signals can disagree: a brick may flag an observation as not codable
    and still return a code (or the reverse). Returns None when no flag column is
    present in ``data``.
    """
    truth = data[truth_column(data)]
    rows = []
    for name, flag in CODABLE_FLAGS.items():
        col = dict(METHODS).get(name)
        if flag not in data.columns or col not in data.columns:
            continue
        declared = data[flag].fillna(False).astype(bool)
        answered = answer_mask(data[col])
        for decl_label, decl_mask in (("codable", declared), ("non codable", ~declared)):
            for ans_label, ans_mask in (("code émis", answered), ("abstention", ~answered)):
                mask = decl_mask & ans_mask
                # Sur les lignes d'abstention l'accuracy vaut 0 par construction
                # (pas de code = erreur) : on la laisse vide plutôt que d'afficher
                # un chiffre qui n'apporte rien.
                acc = float("nan")
                if ans_label == "code émis" and mask.any():
                    _, _, acc = accuracy(truth[mask], data[col][mask], k, inclusive=True)
                rows.append(
                    {
                        "méthode": name,
                        "drapeau": f"{flag} = {decl_label}",
                        "sortie": ans_label,
                        "n": int(mask.sum()),
                        f"accuracy niv{k}": acc,
                    }
                )
    if not rows:
        return None
    return pd.DataFrame(rows)


def regime_masks(data: pd.DataFrame) -> list[tuple[str, str, pd.Series]] | None:
    """Split ``data`` into consensus vs judge-arbitrated rows.

    Returns ``[(label, metric_suffix, mask), …]``, or None when ``REGIME_COL`` is
    absent (runs produced before decide-coicop tagged the regime).
    """
    if REGIME_COL not in data.columns:
        return None
    consensus = data[REGIME_COL] == CONSENSUS_LABEL
    masks = {"consensus": consensus, "arbitrated": ~consensus}
    return [(label, suffix, masks[suffix]) for label, suffix in REGIMES]


def regime_accuracy_table(
    data: pd.DataFrame, k: int = 4, *, inclusive: bool = True
) -> pd.DataFrame | None:
    """Accuracy at level ``k`` per method, split by arbitration regime.

    The pooled figures mix two regimes that measure different things. On the
    consensus rows the judge was never called and ``llm_code`` *is* TTC top-1, so
    the LLM and TTC columns are identical there by construction — those rows say
    nothing about the judge while pulling both figures up (they are the easy
    cases, selected on TTC confidence). Only the arbitrated subset compares the
    judge with the sources it actually arbitrated.

    Returns None when the regime column is absent.
    """
    masks = regime_masks(data)
    if masks is None:
        return None
    truth = data[truth_column(data)]
    rows = []
    for name, col in available_methods(data):
        row = {"méthode": name}
        _, n_all, acc_all = accuracy(truth, data[col], k, inclusive=inclusive)
        row[f"Ensemble (n={n_all})"] = acc_all
        for label, _suffix, mask in masks:
            sub = data[mask]
            _, n_sub, acc_sub = accuracy(
                sub[truth_column(data)], sub[col], k, inclusive=inclusive
            )
            row[f"{label} (n={n_sub})"] = acc_sub
        rows.append(row)
    return pd.DataFrame(rows).set_index("méthode")


def truth_depth_distribution(truth: pd.Series) -> dict:
    """How deep the ground truth actually goes.

    Needed to read the inclusive accuracy: at level ``k``, the rows whose truth
    is shallower than ``k`` are the ones the strict convention drops and the
    inclusive one requires to be predicted *exactly*.
    """
    depths = [len(code_parts(t)) for t in truth]
    depths = [d for d in depths if d]
    total = len(depths)
    per_depth = {
        d: {
            "count": sum(1 for x in depths if x == d),
            "pct": (sum(1 for x in depths if x == d) / total * 100)
            if total
            else float("nan"),
        }
        for d in sorted(set(depths))
    }
    shallower = {
        k: {
            "count": sum(1 for x in depths if x < k),
            "pct": (sum(1 for x in depths if x < k) / total * 100)
            if total
            else float("nan"),
        }
        for k in CANONICAL_LEVELS
    }
    return {"total": total, "per_depth": per_depth, "shallower_than": shallower}


def prediction_depth_distribution(pred: pd.Series) -> dict:
    """Distribution of predictions by COICOP depth and by level-1 division.

    For each level k in 1..5, count (and percentage of) predictions that reach
    at least depth k. Also returns the share of predictions per level-1
    division. ``total`` is the number of non-empty predictions (the denominator
    for percentages).
    """
    parts = [code_parts(p) for p in pred]
    total = sum(1 for p in parts if p)
    depth = {}
    for k in LEVELS:
        count = sum(1 for p in parts if len(p) >= k)
        pct = (count / total * 100) if total else float("nan")
        depth[k] = {"count": count, "pct": pct}

    div_counts: dict[str, int] = {}
    for p in parts:
        if p:
            div_counts[p[0]] = div_counts.get(p[0], 0) + 1
    divisions = {
        div: {"count": c, "pct": (c / total * 100 if total else float("nan"))}
        for div, c in sorted(div_counts.items())
    }
    return {"total": total, "depth": depth, "divisions": divisions}


def _parse_ts(value: str):
    """Parse an RFC3339 timestamp; return None if absent/unresolved."""
    if not value:
        return None
    value = value.strip()
    # An unsubstituted Argo template (e.g. "{{tasks...}}") is not a timestamp.
    if not value or value.startswith("{{"):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    # Une tâche *skippée* (conciliation non retenue, cf. le paramètre
    # `conciliation`) voit ses `startedAt`/`finishedAt` résolus par Argo en zéro
    # temps, "0001-01-01T00:00:00Z". Cela parse sans erreur et produirait une
    # durée de l'ordre de -6e10 secondes dans MLflow.
    if parsed.year < 2000:
        return None
    return parsed


def parse_step_timings(raw: str) -> dict[str, float]:
    """Parse Argo step timings JSON into durations in seconds.

    ``raw`` is a JSON object ``{step: [startedAt, finishedAt]}`` using RFC3339
    timestamps. Returns ``duration_<step>_seconds`` for every step whose two
    timestamps parse, plus ``codification_total_seconds`` spanning from the
    preprocessing start to the decide-coicop finish. Missing/unresolved values
    are skipped rather than raising.
    """
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}

    parsed: dict[str, tuple] = {}
    out: dict[str, float] = {}
    for step, pair in data.items():
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        start, end = _parse_ts(pair[0]), _parse_ts(pair[1])
        parsed[step] = (start, end)
        if start and end:
            duree = (end - start).total_seconds()
            # Une durée négative ou nulle n'est pas une mesure : c'est le signe
            # d'un horodatage non résolu qui aurait franchi les filtres.
            if duree > 0:
                out["duration_" + step.replace("-", "_") + "_seconds"] = duree

    # Durée totale de codification : début de preprocessing → fin de la
    # conciliation. Les deux conciliations étant exclusives (paramètre
    # `conciliation`), on prend celle qui a effectivement tourné : l'autre est
    # skippée et ses horodatages ont été écartés ci-dessus.
    start = parsed.get("preprocessing", (None, None))[0]
    end = (
        parsed.get("decide-coicop", (None, None))[1]
        or parsed.get("sirus-predict", (None, None))[1]
    )
    if start and end and (end - start).total_seconds() > 0:
        out["codification_total_seconds"] = (end - start).total_seconds()
    return out
