"""
Evaluation for the annotation-based RAG.

Beyond per-level accuracy / retrieval recall, this module provides:
  - accuracy broken down by COICOP level-1 category,
  - distribution-distortion indicators (true vs predicted) at levels 1 and 2,
  - confidence-reliability indicators (calibration, AUROC, threshold sweep) to
    tell whether the LLM `confidence` can filter out wrong predictions,
  - `codable`-reliability indicators (does the `codable` flag separate good from
    bad predictions?).
"""
import math
from collections import Counter
from typing import Dict, List, Optional

import numpy as np

from coicop_rag_annotations.utils import is_present, truncate_code


def accuracy_by_level(
    records: List[Dict],
    levels: range,
    predicted_col: str = "code_predict",
    label_col: str = "code",
    retrieved_col: str = "list_retrieved_codes",
) -> Dict[int, Dict[str, float]]:
    """
    Per-level accuracy and retrieval recall.

    For each level, codes are truncated and compared. Observations whose ground
    truth is shallower than the level are excluded from that level's denominator.
    """
    out: Dict[int, Dict[str, float]] = {}
    for lvl in levels:
        correct = total = ret_hit = 0
        for r in records:
            label = truncate_code(r.get(label_col), lvl)
            if label is None:
                continue
            total += 1
            if truncate_code(r.get(predicted_col), lvl) == label:
                correct += 1
            retrieved = r.get(retrieved_col) or []
            if any(truncate_code(c, lvl) == label for c in retrieved):
                ret_hit += 1
        out[lvl] = {
            "accuracy": correct / total if total else 0.0,
            "retrieval_recall": ret_hit / total if total else 0.0,
            "n": total,
        }
    return out


def evaluate(
    records: List[Dict],
    levels: range = range(1, 5),
) -> Dict[str, Dict]:
    """
    Compute metrics on all-parsed and parsed-and-codable subsets.

    Returns a nested dict: {subset: {level: {accuracy, retrieval_recall, n}}}.
    """
    parsed = [r for r in records if r.get("parsed") is True]
    codable = [r for r in parsed if r.get("codable") is True]
    return {
        "all_parsed": accuracy_by_level(parsed, levels),
        "parsed_and_codable": accuracy_by_level(codable, levels),
    }


def format_report(metrics: Dict[str, Dict], n_total: int) -> str:
    """Render `evaluate()` output as a plain-text report."""
    lines = ["=" * 70, "ANNOTATION-RAG EVALUATION", "=" * 70, f"Total observations: {n_total}"]
    for subset, by_level in metrics.items():
        lines += ["", "-" * 70, f"Subset: {subset.upper()}", "-" * 70,
                  f"{'Level':<8}{'N':<10}{'Accuracy':<14}{'Retrieval recall':<18}",
                  f"{'-'*7} {'-'*9} {'-'*13} {'-'*17}"]
        for lvl, m in by_level.items():
            lines.append(
                f"{lvl:<8}{m['n']:<10}{m['accuracy']:<14.4f}{m['retrieval_recall']:<18.4f}"
            )
    lines += ["", "=" * 70]
    return "\n".join(lines)


def flatten_metrics(metrics: Dict[str, Dict]) -> Dict[str, float]:
    """Flatten to `subset/level_N/<metric>` keys for mlflow.log_metrics."""
    flat: Dict[str, float] = {}
    for subset, by_level in metrics.items():
        for lvl, m in by_level.items():
            flat[f"{subset}/level_{lvl}/accuracy"] = m["accuracy"]
            flat[f"{subset}/level_{lvl}/retrieval_recall"] = m["retrieval_recall"]
            flat[f"{subset}/level_{lvl}/n"] = m["n"]
    return flat


# ===========================================================================
# Detailed indicators
# ===========================================================================

def _correct_at(record: Dict, level: int, predicted_col: str, label_col: str) -> Optional[bool]:
    """True/False if the prediction matches the truncated label at `level`;
    None if the record has no ground truth at that level."""
    label = truncate_code(record.get(label_col), level)
    if label is None:
        return None
    return truncate_code(record.get(predicted_col), level) == label


def accuracy_by_category(
    records: List[Dict],
    group_level: int = 1,
    target_level: int = 4,
    predicted_col: str = "code_predict",
    label_col: str = "code",
) -> List[Dict]:
    """
    Accuracy broken down by the true COICOP category at `group_level` (e.g. 1).

    For each category: support (n + share) and accuracy both at `group_level`
    (did we land in the right top category?) and at `target_level` (fine code).
    """
    groups: Dict[str, list] = {}
    for r in records:
        gtrue = truncate_code(r.get(label_col), group_level)
        if gtrue is None:
            continue
        cg = truncate_code(r.get(predicted_col), group_level) == gtrue
        ct = _correct_at(r, target_level, predicted_col, label_col)
        groups.setdefault(gtrue, []).append((cg, ct))

    total = sum(len(v) for v in groups.values())
    rows = []
    for cat, lst in groups.items():
        n = len(lst)
        tvals = [ct for _, ct in lst if ct is not None]
        rows.append({
            "level1": cat,
            "n": n,
            "share": n / total if total else 0.0,
            f"accuracy_level{group_level}": sum(1 for cg, _ in lst if cg) / n,
            f"accuracy_level{target_level}": (sum(tvals) / len(tvals)) if tvals else None,
        })
    rows.sort(key=lambda d: d["n"], reverse=True)
    return rows


def accuracy_by_source(
    records: List[Dict],
    source_col: str = "source",
    group_level: int = 1,
    target_level: int = 4,
    predicted_col: str = "code_predict",
    label_col: str = "code",
) -> List[Dict]:
    """
    Accuracy broken down by the test-data source (e.g. copain, manual_from_app…).

    For each source: support (n + share), accuracy at `group_level` (level-1 top
    category) and at `target_level` (global / fine code).
    """
    groups: Dict[str, list] = {}
    for r in records:
        gtrue = truncate_code(r.get(label_col), group_level)
        if gtrue is None:
            continue
        src = is_present(r.get(source_col))
        src = str(src) if src is not None else "<unknown>"
        cg = truncate_code(r.get(predicted_col), group_level) == gtrue
        ct = _correct_at(r, target_level, predicted_col, label_col)
        groups.setdefault(src, []).append((cg, ct))

    total = sum(len(v) for v in groups.values())
    rows = []
    for src, lst in groups.items():
        n = len(lst)
        tvals = [ct for _, ct in lst if ct is not None]
        rows.append({
            "source": src,
            "n": n,
            "share": n / total if total else 0.0,
            f"accuracy_level{group_level}": sum(1 for cg, _ in lst if cg) / n,
            f"accuracy_level{target_level}": (sum(tvals) / len(tvals)) if tvals else None,
        })
    rows.sort(key=lambda d: d["n"], reverse=True)
    return rows


def distribution_distortion(
    records: List[Dict],
    level: int,
    predicted_col: str = "code_predict",
    label_col: str = "code",
) -> Dict:
    """
    Compare the distribution of TRUE vs PREDICTED codes at `level`.

    Returns per-category shares + aggregate distortion indicators:
      - tv_distance: total variation distance ½·Σ|pred_share − true_share| (0…1),
      - kl_divergence: KL(true ‖ pred) with smoothing.
    A large positive `diff` means the model over-predicts that category.
    """
    true_counts: Counter = Counter()
    pred_counts: Counter = Counter()
    n = 0
    for r in records:
        true = truncate_code(r.get(label_col), level)
        if true is None:
            continue
        n += 1
        true_counts[true] += 1
        pred = truncate_code(r.get(predicted_col), level)
        pred_counts[pred if pred is not None else "<none>"] += 1

    if n == 0:
        return {"level": level, "n": 0, "tv_distance": 0.0, "kl_divergence": 0.0,
                "n_categories": 0, "per_category": []}

    cats = sorted(set(true_counts) | set(pred_counts))
    eps = 1e-9
    tv = kl = 0.0
    per_cat = []
    for c in cats:
        t, p = true_counts.get(c, 0), pred_counts.get(c, 0)
        ts, ps = t / n, p / n
        tv += abs(ps - ts)
        if ts > 0:
            kl += ts * math.log((ts + eps) / (ps + eps))
        per_cat.append({"category": c, "true_n": t, "pred_n": p,
                        "true_share": ts, "pred_share": ps, "diff": ps - ts})
    per_cat.sort(key=lambda d: abs(d["diff"]), reverse=True)
    return {"level": level, "n": n, "tv_distance": 0.5 * tv, "kl_divergence": kl,
            "n_categories": len(cats), "per_category": per_cat}


def _auroc(scores: np.ndarray, labels: np.ndarray) -> Optional[float]:
    """AUROC = P(score | correct > score | wrong), via Mann-Whitney with tie
    handling. None if a class is missing."""
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks within ties
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (ranks[order[i]] + ranks[order[j]]) / 2
        i = j + 1
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def confidence_reliability(
    records: List[Dict],
    target_level: int,
    predicted_col: str = "code_predict",
    label_col: str = "code",
    confidence_col: str = "confidence",
    n_bins: int = 10,
    thresholds=(0.5, 0.6, 0.7, 0.8, 0.9),
) -> Dict:
    """
    Can the LLM `confidence` filter out wrong predictions (correctness at
    `target_level`)? Returns calibration bins, AUROC, mean confidence by
    correctness, and a coverage/accuracy threshold sweep.
    """
    scores, correct = [], []
    for r in records:
        if r.get("parsed") is not True:
            continue
        is_correct = _correct_at(r, target_level, predicted_col, label_col)
        if is_correct is None:
            continue
        conf = r.get(confidence_col)
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            continue
        if math.isnan(conf):
            continue
        scores.append(conf)
        correct.append(is_correct)

    result = {"target_level": target_level, "n": len(scores), "auroc": None,
              "mean_conf_correct": None, "mean_conf_incorrect": None,
              "calibration_bins": [], "threshold_sweep": []}
    if not scores:
        return result

    s = np.asarray(scores, float)
    c = np.asarray(correct, bool)
    result["auroc"] = _auroc(s, c)
    if c.any():
        result["mean_conf_correct"] = float(s[c].mean())
    if (~c).any():
        result["mean_conf_incorrect"] = float(s[~c].mean())

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (s >= lo) & (s <= hi) if i == n_bins - 1 else (s >= lo) & (s < hi)
        cnt = int(mask.sum())
        result["calibration_bins"].append({
            "bin": f"[{lo:.1f},{hi:.1f}{']' if i == n_bins - 1 else ')'}",
            "n": cnt,
            "mean_confidence": float(s[mask].mean()) if cnt else None,
            "accuracy": float(c[mask].mean()) if cnt else None,
        })

    n = len(s)
    for t in thresholds:
        keep = s >= t
        kept = int(keep.sum())
        result["threshold_sweep"].append({
            "threshold": t,
            "coverage": kept / n,
            "n_kept": kept,
            "accuracy_kept": float(c[keep].mean()) if kept else None,
            "accuracy_dropped": float(c[~keep].mean()) if kept < n else None,
        })
    return result


def codable_reliability(
    records: List[Dict],
    target_level: int,
    predicted_col: str = "code_predict",
    label_col: str = "code",
) -> Dict:
    """
    Does the `codable` flag separate good from bad predictions (at
    `target_level`)? Returns accuracy/coverage for codable True/False/missing
    and the accuracy lift of keeping only codable=True vs the overall accuracy.
    """
    buckets: Dict[str, list] = {"true": [], "false": [], "missing": []}
    for r in records:
        is_correct = _correct_at(r, target_level, predicted_col, label_col)
        if is_correct is None:
            continue
        codable = r.get("codable")
        key = "true" if codable is True else "false" if codable is False else "missing"
        buckets[key].append(is_correct)

    total = sum(len(v) for v in buckets.values())

    def stat(lst):
        n = len(lst)
        return {"n": n, "accuracy": (sum(lst) / n if n else None),
                "coverage": (n / total if total else 0.0)}

    overall = (sum(sum(v) for v in buckets.values()) / total) if total else None
    true_acc = stat(buckets["true"])["accuracy"]
    return {
        "target_level": target_level,
        "n": total,
        "overall_accuracy": overall,
        "lift": (true_acc - overall) if (true_acc is not None and overall is not None) else None,
        "true": stat(buckets["true"]),
        "false": stat(buckets["false"]),
        "missing": stat(buckets["missing"]),
        "groups": [{"codable": k, **stat(buckets[k])} for k in ("true", "false", "missing")],
    }


def detailed_evaluation(records: List[Dict], max_level: int = 4, target_level: int = 4) -> Dict:
    """Run every indicator and return them in one structured dict."""
    # Distribution distortion is computed only on predictions the LLM flagged as
    # codable (uncodable rows have null/unreliable predictions).
    codable_records = [r for r in records if r.get("codable") is True]
    return {
        "n_total": len(records),
        "target_level": target_level,
        "overall": evaluate(records, range(1, max_level + 1)),
        "by_level1": accuracy_by_category(records, group_level=1, target_level=target_level),
        "by_source": accuracy_by_source(records, group_level=1, target_level=target_level),
        "distortion": {lvl: distribution_distortion(codable_records, lvl) for lvl in (1, 2)},
        "confidence": confidence_reliability(records, target_level),
        "codable": codable_reliability(records, target_level),
    }


def flatten_detailed(detail: Dict) -> Dict[str, float]:
    """Scalar metrics from `detailed_evaluation` for mlflow.log_metrics (None dropped)."""
    flat: Dict[str, float] = dict(flatten_metrics(detail["overall"]))
    for lvl, d in detail["distortion"].items():
        flat[f"distortion/level_{lvl}/tv_distance"] = d["tv_distance"]
        flat[f"distortion/level_{lvl}/kl_divergence"] = d["kl_divergence"]
    conf = detail["confidence"]
    flat["confidence/auroc"] = conf["auroc"]
    flat["confidence/mean_conf_correct"] = conf["mean_conf_correct"]
    flat["confidence/mean_conf_incorrect"] = conf["mean_conf_incorrect"]
    for row in conf["threshold_sweep"]:
        pct = int(round(row["threshold"] * 100))
        flat[f"confidence/accuracy_at_{pct}"] = row["accuracy_kept"]
        flat[f"confidence/coverage_at_{pct}"] = row["coverage"]
    cod = detail["codable"]
    flat["codable/overall_accuracy"] = cod["overall_accuracy"]
    flat["codable/accuracy_true"] = cod["true"]["accuracy"]
    flat["codable/accuracy_false"] = cod["false"]["accuracy"]
    flat["codable/coverage_true"] = cod["true"]["coverage"]
    flat["codable/lift"] = cod["lift"]
    return {k: v for k, v in flat.items() if v is not None}


def _fmt(x, spec=".4f"):
    return format(x, spec) if isinstance(x, (int, float)) else "N/A"


def format_detailed_report(detail: Dict) -> str:
    """Render `detailed_evaluation` as a human-readable text report."""
    L = ["=" * 78, "ANNOTATION-RAG — DETAILED EVALUATION", "=" * 78,
         f"Total observations: {detail['n_total']}   (target level = {detail['target_level']})"]

    # Overall per-level
    L += ["", "── Overall accuracy / retrieval recall by level " + "─" * 30]
    L.append(format_report(detail["overall"], detail["n_total"]))

    # Accuracy by level-1 category
    tgt = detail["target_level"]
    L += ["", "── Accuracy by COICOP level-1 category " + "─" * 39,
          f"{'lvl1':<6}{'n':<8}{'share':<9}{'acc@1':<9}{f'acc@{tgt}':<9}"]
    for row in detail["by_level1"]:
        L.append(f"{row['level1']:<6}{row['n']:<8}{_fmt(row['share'],'.3f'):<9}"
                 f"{_fmt(row['accuracy_level1'],'.3f'):<9}{_fmt(row.get(f'accuracy_level{tgt}'),'.3f'):<9}")

    # Accuracy by test-data source (global = acc@target, plus level-1 acc)
    if detail.get("by_source"):
        L += ["", "── Accuracy by test-data source " + "─" * 46,
              f"{'source':<22}{'n':<8}{'share':<9}{'acc@1':<9}{f'acc@{tgt} (global)':<18}"]
        for row in detail["by_source"]:
            L.append(f"{str(row['source']):<22}{row['n']:<8}{_fmt(row['share'],'.3f'):<9}"
                     f"{_fmt(row['accuracy_level1'],'.3f'):<9}{_fmt(row.get(f'accuracy_level{tgt}'),'.3f'):<18}")

    # Distribution distortion (codable predictions only)
    for lvl, d in detail["distortion"].items():
        L += ["", f"── Distribution distortion @ level {lvl} — codable only (n={d['n']}) "
              f"(TV={_fmt(d['tv_distance'],'.4f')}, KL={_fmt(d['kl_divergence'],'.4f')}) " + "─" * 8,
              f"{'category':<10}{'true%':<9}{'pred%':<9}{'diff':<9}"]
        for row in d["per_category"][:12]:
            L.append(f"{str(row['category']):<10}{row['true_share']*100:<9.2f}"
                     f"{row['pred_share']*100:<9.2f}{row['diff']*100:+.2f}")
        if len(d["per_category"]) > 12:
            L.append(f"... ({len(d['per_category']) - 12} more categories)")

    # Confidence reliability
    conf = detail["confidence"]
    L += ["", "── Confidence reliability " + "─" * 52,
          f"AUROC (confidence vs correctness): {_fmt(conf['auroc'])}   "
          f"(n={conf['n']}; mean conf correct={_fmt(conf['mean_conf_correct'],'.3f')}, "
          f"incorrect={_fmt(conf['mean_conf_incorrect'],'.3f')})",
          "Calibration:", f"  {'bin':<12}{'n':<8}{'mean_conf':<12}{'accuracy':<10}"]
    for b in conf["calibration_bins"]:
        L.append(f"  {b['bin']:<12}{b['n']:<8}{_fmt(b['mean_confidence'],'.3f'):<12}{_fmt(b['accuracy'],'.3f'):<10}")
    L += ["Threshold sweep (keep confidence ≥ t):",
          f"  {'t':<6}{'coverage':<11}{'acc_kept':<11}{'acc_dropped':<12}"]
    for row in conf["threshold_sweep"]:
        L.append(f"  {row['threshold']:<6}{_fmt(row['coverage'],'.3f'):<11}"
                 f"{_fmt(row['accuracy_kept'],'.3f'):<11}{_fmt(row['accuracy_dropped'],'.3f'):<12}")

    # Codable reliability
    cod = detail["codable"]
    L += ["", "── `codable` reliability " + "─" * 53,
          f"Overall accuracy: {_fmt(cod['overall_accuracy'],'.3f')}   "
          f"lift(codable=True): {_fmt(cod['lift'],'+.3f')}",
          f"  {'codable':<10}{'n':<8}{'coverage':<11}{'accuracy':<10}"]
    for g in cod["groups"]:
        L.append(f"  {g['codable']:<10}{g['n']:<8}{_fmt(g['coverage'],'.3f'):<11}{_fmt(g['accuracy'],'.3f'):<10}")

    L += ["", "=" * 78]
    return "\n".join(L)
