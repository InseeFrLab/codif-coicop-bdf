import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from typing import Optional


SOURCE_FLAGS = ["manual_from_app", "manual_from_books", "receips_from_app", "suggester"]


def _filter_by_sources(
    df: pd.DataFrame,
    sources: Optional[list[str]] = None,
    source_logic: str = "any",
) -> pd.DataFrame:
    """
    Filter dataframe by source boolean flags.

    Parameters
    ----------
    df : pd.DataFrame
    sources : list[str] or None
        Which boolean columns to filter on. None = no filtering (all data).
        Valid values: 'manual_from_app', 'manual_from_books', 'receips_from_app', 'suggester'
    source_logic : str
        'any'  → keep rows where ANY of the listed sources is True
        'all'  → keep rows where ALL of the listed sources are True
        'only' → keep rows where ONLY the listed sources are True (others are False)
        'none' → keep rows where NONE of the listed sources are True

    Returns
    -------
    Filtered DataFrame
    """
    if sources is None:
        return df

    invalid = set(sources) - set(SOURCE_FLAGS)
    if invalid:
        raise ValueError(f"Unknown source flags: {invalid}. Valid: {SOURCE_FLAGS}")

    if source_logic == "any":
        mask = df[sources].any(axis=1)
    elif source_logic == "all":
        mask = df[sources].all(axis=1)
    elif source_logic == "only":
        others = [s for s in SOURCE_FLAGS if s not in sources]
        mask = df[sources].all(axis=1) & ~df[others].any(axis=1)
    elif source_logic == "none":
        mask = ~df[sources].any(axis=1)
    else:
        raise ValueError(f"source_logic must be 'any', 'all', 'only', or 'none', got '{source_logic}'")

    return df[mask]


def evaluate_by_confidence(
    df: pd.DataFrame,
    levels: list[int] = [1, 2, 3, 4],
    confidence_thresholds: list[float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    average: str = "weighted",
    sources: Optional[list[str]] = None,
    source_logic: str = "any",
) -> pd.DataFrame:
    """
    Calculate accuracy, F1, precision, recall for each level at each confidence threshold,
    optionally filtered by source flags.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: level{i}, predicted_level{i}, confidence_level{i}
    levels : list[int]
        Which levels to evaluate (default [1,2,3,4]).
    confidence_thresholds : list[float]
        Minimum confidence values to filter on.
    average : str
        Averaging method for F1/precision/recall ('weighted', 'macro', 'micro').
    sources : list[str] or None
        Boolean flag columns to filter on. None = all data.
    source_logic : str
        'any' | 'all' | 'only' | 'none' — how to combine source flags.

    Returns
    -------
    pd.DataFrame with columns:
        level, threshold, accuracy, f1, precision, recall, coverage, n_samples, source_filter
    """
    filtered_df = _filter_by_sources(df, sources, source_logic)
    total_n = len(filtered_df)
    source_label = f"{source_logic}({', '.join(sources)})" if sources else "all"

    results = []

    for level in levels:
        true_col = f"level{level}"
        pred_col = f"predicted_level{level}"
        conf_col = f"confidence_level{level}"

        if not all(c in filtered_df.columns for c in [true_col, pred_col, conf_col]):
            print(f"⚠️  Skipping level {level}: missing columns")
            continue

        for thresh in confidence_thresholds:
            mask = filtered_df[conf_col] >= thresh
            subset = filtered_df[mask].dropna(subset=[true_col, pred_col])

            n = len(subset)
            coverage = n / total_n if total_n > 0 else 0

            if n == 0:
                results.append({
                    "level": level,
                    "threshold": thresh,
                    "accuracy": np.nan,
                    "f1": np.nan,
                    "precision": np.nan,
                    "recall": np.nan,
                    "coverage": 0.0,
                    "n_samples": 0,
                    "source_filter": source_label,
                })
                continue

            y_true = subset[true_col].astype(str)
            y_pred = subset[pred_col].astype(str)

            results.append({
                "level": level,
                "threshold": thresh,
                "accuracy": accuracy_score(y_true, y_pred),
                "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
                "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
                "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
                "coverage": coverage,
                "n_samples": n,
                "source_filter": source_label,
            })

    return pd.DataFrame(results)


def evaluate_all_sources(
    df: pd.DataFrame,
    levels: list[int] = [1, 2, 3, 4],
    confidence_thresholds: list[float] = [0.0, 0.3, 0.5, 0.7, 0.9],
    average: str = "weighted",
) -> pd.DataFrame:
    """
    Run evaluate_by_confidence for each source flag individually + all data combined.
    Useful for comparing performance across data sources at a glance.

    Returns a single concatenated DataFrame with a 'source_filter' column.
    """
    all_results = []

    # All data
    all_results.append(evaluate_by_confidence(
        df, levels, confidence_thresholds, average, sources=None
    ))

    # Each source individually
    for src in SOURCE_FLAGS:
        if src in df.columns and df[src].any():
            all_results.append(evaluate_by_confidence(
                df, levels, confidence_thresholds, average,
                sources=[src], source_logic="any"
            ))

    # None of the sources (if relevant)
    if all(s in df.columns for s in SOURCE_FLAGS):
        all_results.append(evaluate_by_confidence(
            df, levels, confidence_thresholds, average,
            sources=SOURCE_FLAGS, source_logic="none"
        ))

    return pd.concat(all_results, ignore_index=True)


def classification_report_by_level(
    df: pd.DataFrame,
    level: int,
    min_confidence: float = 0.0,
    sources: Optional[list[str]] = None,
    source_logic: str = "any",
) -> str:
    """
    Return sklearn classification_report for a given level filtered by confidence and source.
    """
    filtered_df = _filter_by_sources(df, sources, source_logic)

    true_col = f"level{level}"
    pred_col = f"predicted_level{level}"
    conf_col = f"confidence_level{level}"

    mask = filtered_df[conf_col] >= min_confidence
    subset = filtered_df[mask].dropna(subset=[true_col, pred_col])

    y_true = subset[true_col].astype(str)
    y_pred = subset[pred_col].astype(str)

    source_label = f"{source_logic}({', '.join(sources)})" if sources else "all"
    header = f"Level {level} | confidence >= {min_confidence} | source: {source_label} | n={len(subset)}\n"

    return header + classification_report(y_true, y_pred, zero_division=0)


def plot_accuracy_vs_coverage(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None,
):
    """
    Plot accuracy and coverage vs confidence threshold for each level.
    If multiple source_filters exist, they are overlaid with different colors.
    A single shared legend is placed outside the plots at the bottom.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    levels = sorted(results_df["level"].unique())
    source_filters = results_df["source_filter"].unique()

    fig, axes = plt.subplots(1, len(levels), figsize=(6 * len(levels), 5), sharey=False)
    if len(levels) == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, len(source_filters)))

    for ax, level in zip(axes, levels):
        for color, src in zip(colors, source_filters):
            data = results_df[(results_df["level"] == level) & (results_df["source_filter"] == src)]
            if data.empty:
                continue

            ax.plot(data["threshold"], data["accuracy"], "o-", color=color)
            #ax.plot(data["threshold"], data["f1"], "s--", color=color, alpha=0.6)

        ax.set_title(f"Level {level}")
        ax.set_xlabel("Confidence Threshold")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)

    # Build shared legend handles
    legend_handles = []
    for color, src in zip(colors, source_filters):
        legend_handles.append(Line2D([0], [0], marker="o", linestyle="-", color=color, label=f"{src} (acc)"))
        #legend_handles.append(Line2D([0], [0], marker="s", linestyle="--", color=color, alpha=0.6, label=f"{src} (f1)"))

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(len(legend_handles), 4),
        fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
        frameon=True,
    )

    plt.suptitle("Accuracy / F1 vs Confidence Threshold by Source", fontsize=14)
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")

    plt.show()


# ──────────────────────────────────────────────
# Usage examples
# ──────────────────────────────────────────────
if __name__ == "__main__":
    # df = pd.read_csv("your_data.csv")

    # 1) Evaluate all data
    # results = evaluate_by_confidence(df)

    # 2) Evaluate only manual_from_app entries
    # results = evaluate_by_confidence(df, sources=["manual_from_app"])

    # 3) Evaluate only suggester entries with confidence >= 0.5
    # results = evaluate_by_confidence(df, sources=["suggester"], confidence_thresholds=[0.5, 0.7, 0.9])

    # 4) Evaluate entries from NEITHER manual nor suggester
    # results = evaluate_by_confidence(df, sources=SOURCE_FLAGS, source_logic="none")

    # 5) Compare all sources side by side
    # results = evaluate_all_sources(df)
    # print(results.to_string(index=False))
    # plot_accuracy_vs_coverage(results)

    # 6) Detailed report for suggester data, level 3, confidence >= 0.5
    # print(classification_report_by_level(df, level=3, min_confidence=0.5, sources=["suggester"]))

    print("Import this module and use the functions with your DataFrame.")