"""Task correlation and independence analysis for SHELF benchmark.

This module provides tools to analyze task correlations and independence,
which are critical for:
1. Defending against "tasks are redundant" reviewer concerns
2. Computing effective sample size (accounting for correlated tasks)
3. Identifying task families for reporting
4. Validating that aggregate scores measure diverse capabilities

Key insight: Low task correlations (mean r < 0.3) indicate tasks measure
independent capabilities. High correlations (r > 0.7) suggest redundancy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class TaskCorrelationResult:
    """Result of task correlation analysis.

    Attributes:
        correlation_matrix: Task x Task correlation matrix
        task_names: List of task names (ordering matches matrix)
        mean_correlation: Mean of all pairwise correlations
        median_correlation: Median of all pairwise correlations
        min_correlation: Minimum correlation (most independent pair)
        max_correlation: Maximum correlation (most redundant pair)
        negative_count: Number of negative correlations
        high_correlation_pairs: Pairs with |r| > threshold
        effective_n: Effective sample size accounting for correlations
        task_families: Groups of highly correlated tasks (|r| > 0.7)
    """

    correlation_matrix: np.ndarray
    task_names: list[str]
    mean_correlation: float
    median_correlation: float
    min_correlation: float
    max_correlation: float
    negative_count: int
    high_correlation_pairs: list[tuple[str, str, float]]
    effective_n: float
    task_families: list[list[str]]
    model_scores: dict[str, dict[str, float]] = field(default_factory=dict)

    def summary(self) -> str:
        """Generate human-readable summary."""
        n_tasks = len(self.task_names)
        n_pairs = n_tasks * (n_tasks - 1) // 2
        pct_negative = self.negative_count / n_pairs * 100 if n_pairs > 0 else 0

        lines = [
            "Task Correlation Analysis",
            "=" * 50,
            f"Tasks: {n_tasks}",
            f"Pairwise correlations: {n_pairs}",
            "",
            "Correlation Statistics:",
            f"  Mean:   {self.mean_correlation:.3f}",
            f"  Median: {self.median_correlation:.3f}",
            f"  Range:  [{self.min_correlation:.3f}, {self.max_correlation:.3f}]",
            f"  Negative: {self.negative_count} ({pct_negative:.1f}%)",
            "",
            f"Effective sample size: {self.effective_n:.1f} (of {n_tasks} nominal)",
        ]

        if self.high_correlation_pairs:
            lines.extend(
                [
                    "",
                    f"High correlation pairs (|r| > 0.7): {len(self.high_correlation_pairs)}",
                ]
            )
            for t1, t2, r in self.high_correlation_pairs[:5]:
                lines.append(f"  {t1} ↔ {t2}: r={r:.3f}")
            if len(self.high_correlation_pairs) > 5:
                lines.append(f"  ... and {len(self.high_correlation_pairs) - 5} more")

        if self.task_families:
            lines.extend(
                [
                    "",
                    f"Task families (highly correlated groups): {len(self.task_families)}",
                ]
            )
            for family in self.task_families[:3]:
                lines.append(f"  • {', '.join(family)}")

        return "\n".join(lines)


def compute_task_correlation_matrix(
    results: list[dict[str, Any]],
    metric: str = "primary_score",
) -> tuple[np.ndarray, list[str], dict[str, dict[str, float]]]:
    """Compute correlation matrix between tasks across models.

    Args:
        results: List of evaluation result dictionaries
        metric: Metric to use for correlation (default: primary_score)

    Returns:
        Tuple of (correlation_matrix, task_names, model_scores)
        - correlation_matrix: n_tasks x n_tasks Pearson correlation matrix
        - task_names: List of task names (ordering matches matrix rows/cols)
        - model_scores: Dict[model_key, Dict[task_name, score]]
    """
    # Extract scores: model -> task -> score
    model_scores: dict[str, dict[str, float]] = {}

    for result in results:
        model_key = result.get("model_key") or result.get("model")
        task_name = result.get("task")

        if not model_key or not task_name:
            continue

        # Get score
        score = result.get(metric)
        if score is None:
            score = result.get("metrics", {}).get(metric)
        if score is None:
            score = result.get("primary_score")

        if score is not None:
            if model_key not in model_scores:
                model_scores[model_key] = {}
            model_scores[model_key][task_name] = float(score)

    # Get all tasks that have data from multiple models
    all_tasks = set()
    for task_dict in model_scores.values():
        all_tasks.update(task_dict.keys())

    # Filter to tasks with data from at least 3 models
    valid_tasks = []
    for task in sorted(all_tasks):
        count = sum(1 for m in model_scores if task in model_scores[m])
        if count >= 3:
            valid_tasks.append(task)

    if len(valid_tasks) < 2:
        raise ValueError(
            f"Need at least 2 tasks with 3+ models, found {len(valid_tasks)}"
        )

    # Get models that have all valid tasks
    valid_models = []
    for model_key, task_dict in model_scores.items():
        if all(task in task_dict for task in valid_tasks):
            valid_models.append(model_key)

    if len(valid_models) < 3:
        raise ValueError(
            f"Need at least 3 models with all tasks, found {len(valid_models)}"
        )

    # Build score matrix: rows = models, cols = tasks
    n_models = len(valid_models)
    n_tasks = len(valid_tasks)
    score_matrix = np.zeros((n_models, n_tasks))

    for i, model_key in enumerate(valid_models):
        for j, task_name in enumerate(valid_tasks):
            score_matrix[i, j] = model_scores[model_key][task_name]

    # Compute correlation matrix (tasks x tasks)
    # np.corrcoef expects observations in columns, so transpose
    corr_matrix = np.corrcoef(score_matrix.T)

    return corr_matrix, valid_tasks, model_scores


def compute_effective_sample_size(correlation_matrix: np.ndarray) -> float:
    """Estimate effective sample size accounting for task correlations.

    Uses eigenvalue decomposition to estimate the number of independent
    "dimensions" in the task space. This is important for power analysis
    and multiple comparison correction.

    Args:
        correlation_matrix: n x n correlation matrix

    Returns:
        Effective sample size (between 1 and n)

    Method:
        Based on "effective number of tests" from Galwey (2009):
        N_eff = sum(eigenvalues)^2 / sum(eigenvalues^2)

        This gives 1 for perfectly correlated tasks and n for independent tasks.
    """
    # Get eigenvalues
    eigenvalues = np.linalg.eigvalsh(correlation_matrix)

    # Filter to positive eigenvalues (numerical stability)
    eigenvalues = eigenvalues[eigenvalues > 0]

    if len(eigenvalues) == 0:
        return 1.0

    # Galwey's formula for effective number of tests
    sum_eig = np.sum(eigenvalues)
    sum_eig_sq = np.sum(eigenvalues**2)

    if sum_eig_sq == 0:
        return float(len(eigenvalues))

    n_eff = (sum_eig**2) / sum_eig_sq

    return float(n_eff)


def identify_task_families(
    correlation_matrix: np.ndarray,
    task_names: list[str],
    threshold: float = 0.7,
) -> list[list[str]]:
    """Identify groups of highly correlated tasks.

    Uses connected components to find task "families" that are highly
    correlated with each other (|r| > threshold).

    Args:
        correlation_matrix: Task correlation matrix
        task_names: Task names (ordering matches matrix)
        threshold: Correlation threshold for grouping (default: 0.7)

    Returns:
        List of task families (groups of correlated task names)
    """
    n = len(task_names)

    # Build adjacency based on high correlation
    parent = list(range(n))

    def find(x: int) -> int:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Union tasks with |r| > threshold
    for i in range(n):
        for j in range(i + 1, n):
            if abs(correlation_matrix[i, j]) > threshold:
                union(i, j)

    # Group by root
    groups: dict[int, list[str]] = {}
    for i, task in enumerate(task_names):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(task)

    # Return only groups with 2+ members
    return [sorted(g) for g in groups.values() if len(g) >= 2]


def analyze_task_correlations(
    results: list[dict[str, Any]],
    metric: str = "primary_score",
    high_correlation_threshold: float = 0.7,
) -> TaskCorrelationResult:
    """Comprehensive task correlation analysis.

    This is the main entry point for task correlation analysis.

    Args:
        results: List of evaluation result dictionaries
        metric: Metric to analyze (default: primary_score)
        high_correlation_threshold: Threshold for "high" correlation

    Returns:
        TaskCorrelationResult with full analysis

    Example:
        >>> results = load_results_from_dir("results/v0.3.0/baselines")
        >>> analysis = analyze_task_correlations(results)
        >>> print(analysis.summary())
        >>> print(f"Tasks are {'independent' if analysis.mean_correlation < 0.3 else 'correlated'}")
    """
    # Compute correlation matrix
    corr_matrix, task_names, model_scores = compute_task_correlation_matrix(
        results, metric
    )

    n_tasks = len(task_names)

    # Extract pairwise correlations (upper triangle)
    pairwise_corrs = []
    high_corr_pairs = []

    for i in range(n_tasks):
        for j in range(i + 1, n_tasks):
            r = corr_matrix[i, j]
            pairwise_corrs.append(r)

            if abs(r) > high_correlation_threshold:
                high_corr_pairs.append((task_names[i], task_names[j], r))

    pairwise_corrs = np.array(pairwise_corrs)

    # Sort high correlation pairs by |r|
    high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    # Compute statistics
    mean_corr = float(np.mean(pairwise_corrs))
    median_corr = float(np.median(pairwise_corrs))
    min_corr = float(np.min(pairwise_corrs))
    max_corr = float(np.max(pairwise_corrs))
    negative_count = int(np.sum(pairwise_corrs < 0))

    # Effective sample size
    effective_n = compute_effective_sample_size(corr_matrix)

    # Task families
    task_families = identify_task_families(
        corr_matrix, task_names, high_correlation_threshold
    )

    return TaskCorrelationResult(
        correlation_matrix=corr_matrix,
        task_names=task_names,
        mean_correlation=mean_corr,
        median_correlation=median_corr,
        min_correlation=min_corr,
        max_correlation=max_corr,
        negative_count=negative_count,
        high_correlation_pairs=high_corr_pairs,
        effective_n=effective_n,
        task_families=task_families,
        model_scores=model_scores,
    )


def compute_rank_consistency(
    model_scores: dict[str, dict[str, float]],
) -> dict[str, dict[str, Any]]:
    """Compute rank consistency statistics for each model.

    Measures how consistently a model ranks across different tasks.
    High variance indicates a "specialist" model, low variance indicates
    a "generalist" model.

    Args:
        model_scores: Dict[model_key, Dict[task_name, score]]

    Returns:
        Dict[model_key, stats] where stats includes:
        - mean_rank: Average rank across tasks
        - std_rank: Standard deviation of ranks
        - rank_range: (min_rank, max_rank)
        - best_tasks: Tasks where model ranks in top 3
        - worst_tasks: Tasks where model ranks in bottom 3
    """
    # Get all tasks
    all_tasks = set()
    for task_dict in model_scores.values():
        all_tasks.update(task_dict.keys())
    all_tasks = sorted(all_tasks)

    # Get all models
    all_models = sorted(model_scores.keys())
    n_models = len(all_models)

    # Compute rank per task
    model_ranks: dict[str, list[int]] = {m: [] for m in all_models}

    for task in all_tasks:
        # Get scores for this task
        task_scores = []
        for model in all_models:
            score = model_scores[model].get(task)
            if score is not None:
                task_scores.append((model, score))

        if len(task_scores) < 2:
            continue

        # Rank by score (higher is better = rank 1)
        task_scores.sort(key=lambda x: x[1], reverse=True)

        for rank, (model, _) in enumerate(task_scores, 1):
            model_ranks[model].append(rank)

    # Compute statistics per model
    results: dict[str, dict[str, Any]] = {}

    for model in all_models:
        ranks = model_ranks[model]
        if not ranks:
            continue

        results[model] = {
            "mean_rank": float(np.mean(ranks)),
            "std_rank": float(np.std(ranks)) if len(ranks) > 1 else 0.0,
            "min_rank": int(np.min(ranks)),
            "max_rank": int(np.max(ranks)),
            "rank_range": int(np.max(ranks)) - int(np.min(ranks)),
            "n_tasks": len(ranks),
            "n_top3": sum(1 for r in ranks if r <= 3),
            "n_bottom3": sum(1 for r in ranks if r > n_models - 3),
        }

    return results


def compute_task_champion_diversity(
    model_scores: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Analyze diversity of task champions.

    High champion diversity indicates tasks measure different capabilities.
    If one model wins all tasks, the benchmark may be too narrow.

    Args:
        model_scores: Dict[model_key, Dict[task_name, score]]

    Returns:
        Dict with:
        - champions: Dict[task_name, winning_model]
        - n_unique_champions: Number of distinct winning models
        - champion_distribution: Dict[model, n_wins]
        - diversity_ratio: n_unique_champions / n_tasks
    """
    all_tasks = set()
    for task_dict in model_scores.values():
        all_tasks.update(task_dict.keys())
    all_tasks = sorted(all_tasks)

    champions: dict[str, str] = {}
    champion_counts: dict[str, int] = {}

    for task in all_tasks:
        best_model = None
        best_score = -float("inf")

        for model, task_dict in model_scores.items():
            score = task_dict.get(task)
            if score is not None and score > best_score:
                best_score = score
                best_model = model

        if best_model:
            champions[task] = best_model
            champion_counts[best_model] = champion_counts.get(best_model, 0) + 1

    n_tasks = len(champions)
    n_unique = len(champion_counts)

    return {
        "champions": champions,
        "n_unique_champions": n_unique,
        "n_tasks": n_tasks,
        "champion_distribution": champion_counts,
        "diversity_ratio": n_unique / n_tasks if n_tasks > 0 else 0,
    }
