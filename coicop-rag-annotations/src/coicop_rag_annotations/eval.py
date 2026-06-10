"""
Compact evaluation for the annotation-based RAG.

Computes, per COICOP level, the prediction accuracy and the retrieval recall
(was the ground-truth code among the retrieved annotated examples?). Reported
on two record sets: all parsed predictions, and the parsed-and-codable subset.
"""
from typing import Dict, List

from coicop_rag_annotations.utils import truncate_code


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
