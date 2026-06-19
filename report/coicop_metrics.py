"""Shared COICOP metric helpers.

Used by both ``report.qmd`` (so the rendered report and the metrics logged to
MLflow never diverge) and ``main.py`` (which logs to MLflow).

COICOP codes are stored as dot-separated strings (e.g. ``"01.1.2.3.4"``).
Level-k accuracy compares the first ``k`` segments of the prediction with the
ground truth.
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


def available_methods(data: pd.DataFrame) -> list[tuple[str, str]]:
    """METHODS whose prediction column is present in ``data`` (backward compatible
    with runs produced before a given method existed)."""
    return [(name, col) for name, col in METHODS if col in data.columns]


def code_parts(s) -> list[str]:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return []
    s = str(s).strip()
    if not s:
        return []
    return s.split(".")


def level_result(truth: str, pred: str, k: int):
    """Return True/False/None.

    None means the observation is not applicable at level k
    (truth is shallower than k, so we can't evaluate that depth).
    """
    tp = code_parts(truth)
    if len(tp) < k:
        return None
    pp = code_parts(pred)
    if len(pp) < k:
        return False
    return tp[:k] == pp[:k]


def accuracy_series(truth: pd.Series, pred: pd.Series, k: int) -> pd.Series:
    """Series of True/False/NA indexed like truth."""
    out = [level_result(t, p, k) for t, p in zip(truth, pred)]
    return pd.Series(out, index=truth.index, dtype="object")


def accuracy(truth: pd.Series, pred: pd.Series, k: int) -> tuple[int, int, float]:
    s = accuracy_series(truth, pred, k)
    applicable = s.notna()
    n_app = int(applicable.sum())
    n_ok = int((s == True).sum())  # noqa: E712
    return n_ok, n_app, (n_ok / n_app if n_app else float("nan"))


def accuracy_table(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, col in available_methods(data):
        row = {"méthode": name}
        for k in LEVELS:
            _, n_app, acc = accuracy(data["code"], data[col], k)
            row[f"niv{k} (n={n_app})"] = acc
        rows.append(row)
    return pd.DataFrame(rows).set_index("méthode")


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
