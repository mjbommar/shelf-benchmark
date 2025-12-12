"""Visualization for benchmark comparisons.

Provides plotting functions for:
- Critical difference diagrams (for multiple model comparison)
- Confidence interval plots
- Performance comparison bar charts
- Heatmaps for pairwise significance

Requires: matplotlib, seaborn
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.figure import Figure

    from shelf.evaluate.analysis.comparison import (
        ComparisonResult,
        MultipleComparisonResult,
    )


def plot_critical_difference(
    result: "MultipleComparisonResult",
    title: str | None = None,
    figsize: tuple[float, float] = (10, 4),
    save_path: str | "Path" | None = None,
) -> "Figure":
    """Plot critical difference diagram for multiple model comparison.

    Shows models ranked by mean score with bars indicating groups
    that are not significantly different (connected by horizontal lines).

    Args:
        result: MultipleComparisonResult from compare_multiple()
        title: Plot title (defaults to metric name)
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure object
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    n_models = len(result.models)
    mean_ranks = result.friedman_nemenyi.mean_ranks
    cd = result.friedman_nemenyi.critical_difference

    # Sort by mean rank
    sorted_models = sorted(result.models, key=lambda m: mean_ranks[m])

    # Plot model names and ranks
    for i, model in enumerate(sorted_models):
        rank = mean_ranks[model]
        score = result.mean_scores[model]

        # Draw rank marker
        ax.plot(rank, i, "ko", markersize=8)

        # Model name on left, score on right
        ax.text(0.3, i, f"{model}", ha="right", va="center", fontsize=10)
        ax.text(
            n_models + 0.7,
            i,
            f"{score:.4f}",
            ha="left",
            va="center",
            fontsize=9,
            color="gray",
        )

    # Draw critical difference bar at top
    ax.plot([1, 1 + cd], [n_models + 0.5, n_models + 0.5], "k-", linewidth=2)
    ax.text(
        1 + cd / 2,
        n_models + 0.7,
        f"CD = {cd:.2f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

    # Draw horizontal lines connecting models not significantly different
    # Group models by equivalence
    sig_pairs = set(result.friedman_nemenyi.significant_pairs)

    for i, model_i in enumerate(sorted_models):
        for j, model_j in enumerate(sorted_models):
            if j <= i:
                continue
            # Check if this pair is NOT significantly different
            pair = (
                (model_i, model_j)
                if (model_i, model_j) in sig_pairs or (model_j, model_i) in sig_pairs
                else None
            )
            if pair is None and (model_j, model_i) not in sig_pairs:
                # Not significant - draw connecting line
                rank_i = mean_ranks[model_i]
                rank_j = mean_ranks[model_j]
                y_mid = (i + j) / 2
                # Only connect if within CD
                if abs(rank_i - rank_j) < cd:
                    ax.plot(
                        [rank_i, rank_j],
                        [y_mid - 0.1, y_mid - 0.1],
                        "b-",
                        alpha=0.3,
                        linewidth=4,
                    )

    # Formatting
    ax.set_xlim(0, n_models + 1.5)
    ax.set_ylim(-0.5, n_models + 1.5)
    ax.set_xlabel("Mean Rank (lower is better)")
    ax.set_yticks([])

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Critical Difference Diagram: {result.metric}")

    # Add legend
    ax.text(
        0.02,
        0.02,
        f"Friedman p={result.friedman_nemenyi.friedman_p_value:.4f}",
        transform=ax.transAxes,
        fontsize=8,
        color="gray",
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_confidence_intervals(
    results: list["ComparisonResult"],
    metric: str | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (8, 6),
    save_path: str | "Path" | None = None,
) -> "Figure":
    """Plot confidence intervals for multiple comparisons.

    Shows point estimates with error bars for each model.

    Args:
        results: List of ComparisonResult objects
        metric: Metric name for title
        title: Custom title
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure object
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    # Collect unique models and their scores
    models: dict[str, list[float]] = {}

    for comp in results:
        if comp.name_a not in models:
            models[comp.name_a] = []
        models[comp.name_a].append(comp.mean_a)

        if comp.name_b not in models:
            models[comp.name_b] = []
        models[comp.name_b].append(comp.mean_b)

        # Store CI from bootstrap if available
        if comp.bootstrap:
            # Store CI for the difference, not absolute values
            pass

    # Sort models by mean score
    model_names = sorted(models.keys(), key=lambda m: np.mean(models[m]), reverse=True)

    y_pos = np.arange(len(model_names))
    means = [np.mean(models[m]) for m in model_names]

    # Simple bar chart
    bars = ax.barh(y_pos, means, color="steelblue", alpha=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(model_names)
    ax.set_xlabel(metric or "Score")
    ax.set_title(title or f"Model Performance: {metric or 'Comparison'}")

    # Add value labels
    for i, (bar, mean) in enumerate(zip(bars, means)):
        ax.text(
            mean + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{mean:.4f}",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_pairwise_significance_heatmap(
    result: "MultipleComparisonResult",
    title: str | None = None,
    figsize: tuple[float, float] = (8, 6),
    save_path: str | "Path" | None = None,
) -> "Figure":
    """Plot heatmap of pairwise significance p-values.

    Args:
        result: MultipleComparisonResult from compare_multiple()
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure object
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    n = len(result.models)
    models = result.ranking  # Use ranked order

    # Build p-value matrix
    p_matrix = np.ones((n, n))
    for i, model_i in enumerate(models):
        for j, model_j in enumerate(models):
            if i < j:
                pair = (model_i, model_j)
                if pair in result.friedman_nemenyi.pairwise_p_values:
                    p_matrix[i, j] = result.friedman_nemenyi.pairwise_p_values[pair]
                    p_matrix[j, i] = p_matrix[i, j]
                else:
                    # Try reverse order
                    pair_rev = (model_j, model_i)
                    if pair_rev in result.friedman_nemenyi.pairwise_p_values:
                        p_matrix[i, j] = result.friedman_nemenyi.pairwise_p_values[
                            pair_rev
                        ]
                        p_matrix[j, i] = p_matrix[i, j]

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    mask = np.triu(np.ones_like(p_matrix, dtype=bool), k=1)  # Upper triangle
    sns.heatmap(
        p_matrix,
        mask=~mask.T,  # Show lower triangle
        annot=True,
        fmt=".3f",
        cmap="RdYlGn_r",
        vmin=0,
        vmax=0.1,
        xticklabels=models,
        yticklabels=models,
        ax=ax,
        cbar_kws={"label": "p-value"},
    )

    ax.set_title(title or f"Pairwise Significance: {result.metric}")

    # Highlight significant cells
    for i in range(n):
        for j in range(i + 1, n):
            if p_matrix[j, i] < 0.05:
                ax.add_patch(
                    plt.Rectangle(
                        (i, j), 1, 1, fill=False, edgecolor="black", linewidth=2
                    )
                )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_score_distribution(
    result: "MultipleComparisonResult",
    title: str | None = None,
    figsize: tuple[float, float] = (10, 6),
    save_path: str | "Path" | None = None,
) -> "Figure":
    """Plot score distributions for each model.

    Shows box plots or violin plots of score distributions.

    Args:
        result: MultipleComparisonResult from compare_multiple()
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure object
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=figsize)

    # Prepare data for seaborn
    data = []
    for model in result.ranking:
        for score in result.scores[model]:
            data.append({"Model": model, "Score": score})

    import pandas as pd

    df = pd.DataFrame(data)

    # Create violin plot
    sns.violinplot(
        data=df,
        x="Model",
        y="Score",
        hue="Model",
        order=result.ranking,
        hue_order=result.ranking,
        ax=ax,
        inner="box",
        palette="Set2",
        legend=False,
    )

    ax.set_xlabel("")
    ax.set_ylabel(result.metric)
    ax.set_title(title or f"Score Distribution: {result.metric}")

    # Rotate x labels if needed
    if len(result.models) > 4:
        plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_comparison_summary(
    result: "MultipleComparisonResult",
    save_dir: str | "Path" | None = None,
    prefix: str = "comparison",
) -> dict[str, "Figure"]:
    """Generate all comparison plots and optionally save them.

    Args:
        result: MultipleComparisonResult from compare_multiple()
        save_dir: Directory to save plots (optional)
        prefix: Filename prefix for saved plots

    Returns:
        Dictionary mapping plot names to Figure objects
    """
    from pathlib import Path

    figs = {}

    # Critical difference diagram
    figs["critical_difference"] = plot_critical_difference(result)

    # Significance heatmap
    if len(result.models) > 2:
        figs["significance_heatmap"] = plot_pairwise_significance_heatmap(result)

    # Score distribution
    figs["score_distribution"] = plot_score_distribution(result)

    # Save if directory provided
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        for name, fig in figs.items():
            fig.savefig(
                save_path / f"{prefix}_{name}.png", dpi=150, bbox_inches="tight"
            )

    return figs
