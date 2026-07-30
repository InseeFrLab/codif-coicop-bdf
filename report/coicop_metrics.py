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


def available_methods(data: pd.DataFrame) -> list[tuple[str, str]]:
    """METHODS whose prediction column is present in ``data`` (backward compatible
    with runs produced before a given method existed)."""
    return [(name, col) for name, col in METHODS if col in data.columns]


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
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


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
            key = "duration_" + step.replace("-", "_") + "_seconds"
            out[key] = (end - start).total_seconds()

    # Total codification span: first step start -> decide-coicop finish.
    start = parsed.get("preprocessing", (None, None))[0]
    end = parsed.get("decide-coicop", (None, None))[1]
    if start and end:
        out["codification_total_seconds"] = (end - start).total_seconds()
    return out
