"""High-level comparison functions for benchmark analysis.

This module provides the main API for comparing models and data generations:
- compare_two(): Compare two models/conditions
- compare_multiple(): Compare many models with multiple comparison correction
- compare_by_group(): Compare grouped by model, commit, task, etc.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike

from shelf.evaluate.analysis.bootstrap import (
    BootstrapDifferenceResult,
    bootstrap_paired_difference,
)
from shelf.evaluate.analysis.significance import (
    FriedmanNemenyiResult,
    SignificanceResult,
    cohens_d,
    friedman_nemenyi_test,
    paired_permutation_test,
    paired_t_test,
    wilcoxon_test,
)


@dataclass
class ComparisonResult:
    """Result of comparing two models/conditions.

    Attributes:
        name_a: Name of first model/condition
        name_b: Name of second model/condition
        metric: Name of metric being compared
        mean_a: Mean score for A
        mean_b: Mean score for B
        difference: Mean difference (A - B)
        relative_diff_pct: Relative difference as percentage
        significance: Statistical significance test result
        bootstrap: Bootstrap analysis result (optional)
        n_samples: Number of samples compared
        is_paired: Whether comparison is paired
    """

    name_a: str
    name_b: str
    metric: str
    mean_a: float
    mean_b: float
    difference: float
    relative_diff_pct: float
    significance: SignificanceResult
    bootstrap: BootstrapDifferenceResult | None = None
    n_samples: int = 0
    is_paired: bool = True

    @property
    def winner(self) -> str | None:
        """Return name of significantly better model, or None if no sig diff."""
        if not self.significance.significant:
            return None
        return self.name_a if self.difference > 0 else self.name_b

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Comparing {self.name_a} vs {self.name_b} on {self.metric}",
            f"  {self.name_a}: {self.mean_a:.4f}",
            f"  {self.name_b}: {self.mean_b:.4f}",
            f"  Difference: {self.difference:+.4f} ({self.relative_diff_pct:+.1f}%)",
            f"  {self.significance}",
        ]
        if self.bootstrap:
            lines.append(f"  Bootstrap: {self.bootstrap}")
        if self.winner:
            lines.append(f"  Winner: {self.winner}")
        else:
            lines.append("  No significant difference")
        return "\n".join(lines)


@dataclass
class MultipleComparisonResult:
    """Result of comparing multiple models.

    Attributes:
        models: List of model names
        metric: Name of metric
        scores: Dict mapping model name to array of scores
        mean_scores: Dict mapping model name to mean score
        ranking: Models ranked by mean score (best first)
        friedman_nemenyi: Friedman + Nemenyi test result
        pairwise: Dict of pairwise comparison results
        equivalence_groups: Groups of statistically equivalent models
    """

    models: list[str]
    metric: str
    scores: dict[str, np.ndarray]
    mean_scores: dict[str, float]
    ranking: list[str]
    friedman_nemenyi: FriedmanNemenyiResult
    pairwise: dict[tuple[str, str], ComparisonResult] = field(default_factory=dict)
    equivalence_groups: list[list[str]] = field(default_factory=list)

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Multiple Model Comparison: {self.metric}",
            f"Models: {len(self.models)}, Tasks/Datasets: {self.friedman_nemenyi.n_blocks}",
            "",
            "Ranking (by mean score):",
        ]
        for i, model in enumerate(self.ranking, 1):
            rank = self.friedman_nemenyi.mean_ranks[model]
            lines.append(
                f"  {i}. {model}: {self.mean_scores[model]:.4f} (mean rank: {rank:.2f})"
            )

        lines.extend(
            [
                "",
                str(self.friedman_nemenyi),
            ]
        )

        if self.equivalence_groups:
            lines.extend(
                ["", "Equivalence groups (models not significantly different):"]
            )
            for group in self.equivalence_groups:
                lines.append(f"  {', '.join(group)}")

        return "\n".join(lines)


def load_results_from_dir(
    results_dir: str | Path,
    pattern: str = "*.json",
) -> list[dict[str, Any]]:
    """Load evaluation results from a directory.

    Args:
        results_dir: Directory containing result JSON files
        pattern: Glob pattern for result files

    Returns:
        List of result dictionaries
    """
    results_dir = Path(results_dir)
    results = []

    for result_file in sorted(results_dir.glob(pattern)):
        with open(result_file) as f:
            result = json.load(f)
            result["_source_file"] = str(result_file)
            results.append(result)

    return results


def extract_scores(
    results: list[dict[str, Any]],
    metric: str,
    group_by: str = "model",
) -> dict[str, list[float]]:
    """Extract metric scores grouped by a field.

    Args:
        results: List of result dictionaries
        metric: Metric name to extract (e.g., "ndcg@10", "macro_f1")
        group_by: Field to group by (e.g., "model", "commit_id")

    Returns:
        Dict mapping group name to list of scores
    """
    scores: dict[str, list[float]] = {}

    for result in results:
        # Get group key
        group_key = result.get(group_by) or result.get("context", {}).get(
            group_by, "unknown"
        )

        # Get metric value
        metrics = result.get("metrics", {})
        if metric in metrics:
            score = metrics[metric]
        elif "." in metric:
            # Handle nested metrics like "retrieval.ndcg@10"
            parts = metric.split(".")
            score = metrics
            for part in parts:
                score = score.get(part, {})
            if isinstance(score, dict):
                score = None
        else:
            score = None

        if score is not None:
            if group_key not in scores:
                scores[group_key] = []
            scores[group_key].append(float(score))

    return scores


def compare_two(
    scores_a: ArrayLike,
    scores_b: ArrayLike,
    name_a: str = "Model A",
    name_b: str = "Model B",
    metric: str = "score",
    test: Literal["wilcoxon", "t-test", "permutation", "bootstrap"] = "bootstrap",
    alpha: float = 0.05,
    paired: bool = True,
) -> ComparisonResult:
    """Compare two models/conditions.

    Args:
        scores_a: Scores for model/condition A
        scores_b: Scores for model/condition B
        name_a: Name for A
        name_b: Name for B
        metric: Name of the metric
        test: Statistical test to use
        alpha: Significance level
        paired: Whether samples are paired

    Returns:
        ComparisonResult with full comparison details
    """
    scores_a = np.asarray(scores_a)
    scores_b = np.asarray(scores_b)

    mean_a = float(np.mean(scores_a))
    mean_b = float(np.mean(scores_b))
    difference = mean_a - mean_b
    relative_diff = (difference / mean_b * 100) if mean_b != 0 else 0

    # Run significance test
    if test == "wilcoxon":
        sig_result = wilcoxon_test(scores_a, scores_b, alpha=alpha)
    elif test == "t-test":
        sig_result = paired_t_test(scores_a, scores_b, alpha=alpha)
    elif test == "permutation":
        sig_result = paired_permutation_test(scores_a, scores_b, alpha=alpha)
    elif test == "bootstrap":
        # Use bootstrap for significance
        boot_result = bootstrap_paired_difference(
            scores_a, scores_b, ci_level=1 - alpha
        )
        effect_d, effect_interp = cohens_d(scores_a, scores_b, paired=paired)
        sig_result = SignificanceResult(
            test_name="Bootstrap",
            statistic=boot_result.mean_diff,
            p_value=boot_result.p_value,
            significant=boot_result.significant,
            alpha=alpha,
            effect_size=effect_d,
            effect_size_interpretation=effect_interp,
            confidence_interval=(boot_result.ci_lower, boot_result.ci_upper),
        )
    else:
        raise ValueError(f"Unknown test: {test}")

    # Bootstrap CI for difference
    bootstrap = bootstrap_paired_difference(scores_a, scores_b, ci_level=1 - alpha)

    return ComparisonResult(
        name_a=name_a,
        name_b=name_b,
        metric=metric,
        mean_a=mean_a,
        mean_b=mean_b,
        difference=difference,
        relative_diff_pct=relative_diff,
        significance=sig_result,
        bootstrap=bootstrap,
        n_samples=len(scores_a),
        is_paired=paired,
    )


def compare_multiple(
    scores: dict[str, ArrayLike],
    metric: str = "score",
    alpha: float = 0.05,
    include_pairwise: bool = True,
) -> MultipleComparisonResult:
    """Compare multiple models using Friedman test + Nemenyi post-hoc.

    This is the standard approach for comparing multiple ML models
    across multiple datasets/tasks, as recommended in ML benchmark
    literature.

    Args:
        scores: Dict mapping model name to array of scores
                (one score per dataset/task)
        metric: Name of the metric
        alpha: Significance level
        include_pairwise: Whether to compute pairwise comparisons

    Returns:
        MultipleComparisonResult with full analysis
    """
    # Convert to numpy arrays
    scores_np = {k: np.asarray(v) for k, v in scores.items()}
    models = list(scores_np.keys())

    # Compute mean scores and ranking
    mean_scores = {k: float(np.mean(v)) for k, v in scores_np.items()}
    ranking = sorted(models, key=lambda m: mean_scores[m], reverse=True)

    # Friedman + Nemenyi test
    fn_result = friedman_nemenyi_test(scores_np, alpha=alpha)

    # Pairwise comparisons (optional)
    pairwise = {}
    if include_pairwise:
        for i, model_a in enumerate(models):
            for model_b in models[i + 1 :]:
                comp = compare_two(
                    scores_np[model_a],
                    scores_np[model_b],
                    name_a=model_a,
                    name_b=model_b,
                    metric=metric,
                    alpha=alpha,
                )
                pairwise[(model_a, model_b)] = comp

    # Find equivalence groups (models not significantly different)
    equivalence_groups = _find_equivalence_groups(models, fn_result.significant_pairs)

    return MultipleComparisonResult(
        models=models,
        metric=metric,
        scores=scores_np,
        mean_scores=mean_scores,
        ranking=ranking,
        friedman_nemenyi=fn_result,
        pairwise=pairwise,
        equivalence_groups=equivalence_groups,
    )


def _find_equivalence_groups(
    models: list[str],
    significant_pairs: list[tuple[str, str]],
) -> list[list[str]]:
    """Find groups of models that are not significantly different.

    Uses connected components to find equivalence classes.
    """
    # Build graph of "not significant" relationships
    not_sig = set()
    all_pairs = set()
    for i, a in enumerate(models):
        for b in models[i + 1 :]:
            all_pairs.add((a, b))

    sig_set = set(significant_pairs)
    for pair in all_pairs:
        if pair not in sig_set and (pair[1], pair[0]) not in sig_set:
            not_sig.add(pair)

    # Find connected components via union-find
    parent = {m: m for m in models}

    def find(x: str) -> str:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x: str, y: str) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for a, b in not_sig:
        union(a, b)

    # Group by root
    groups: dict[str, list[str]] = {}
    for m in models:
        root = find(m)
        if root not in groups:
            groups[root] = []
        groups[root].append(m)

    # Return groups with more than one member (actual equivalence)
    return [sorted(g) for g in groups.values() if len(g) > 1]


def compare_by_group(
    results: list[dict[str, Any]],
    metric: str,
    group_by: str = "model",
    alpha: float = 0.05,
) -> MultipleComparisonResult:
    """Compare results grouped by a field (model, commit_id, etc.).

    High-level convenience function that:
    1. Extracts scores grouped by the specified field
    2. Runs multiple comparison analysis

    Args:
        results: List of evaluation result dictionaries
        metric: Metric to compare
        group_by: Field to group by
        alpha: Significance level

    Returns:
        MultipleComparisonResult
    """
    scores = extract_scores(results, metric, group_by)

    if len(scores) < 2:
        raise ValueError(f"Need at least 2 groups to compare, found {len(scores)}")

    # Ensure all groups have same number of scores (required for Friedman)
    min_len = min(len(v) for v in scores.values())
    scores_aligned = {k: v[:min_len] for k, v in scores.items()}

    return compare_multiple(scores_aligned, metric=metric, alpha=alpha)


def compare_commits(
    results: list[dict[str, Any]],
    metric: str,
    commit_a: str,
    commit_b: str,
    alpha: float = 0.05,
) -> ComparisonResult:
    """Compare results between two commit IDs / data generations.

    Useful for checking if prompt/code changes affected performance.

    Args:
        results: List of evaluation results
        metric: Metric to compare
        commit_a: First commit ID
        commit_b: Second commit ID
        alpha: Significance level

    Returns:
        ComparisonResult comparing the two commits
    """
    # Filter results by commit
    results_a = [
        r
        for r in results
        if r.get("commit_id") == commit_a
        or r.get("context", {}).get("commit_id") == commit_a
        or r.get("context", {}).get("data_commit") == commit_a
    ]
    results_b = [
        r
        for r in results
        if r.get("commit_id") == commit_b
        or r.get("context", {}).get("commit_id") == commit_b
        or r.get("context", {}).get("data_commit") == commit_b
    ]

    if not results_a:
        raise ValueError(f"No results found for commit {commit_a}")
    if not results_b:
        raise ValueError(f"No results found for commit {commit_b}")

    # Extract scores
    scores_a = [r.get("metrics", {}).get(metric) for r in results_a]
    scores_b = [r.get("metrics", {}).get(metric) for r in results_b]

    scores_a = [s for s in scores_a if s is not None]
    scores_b = [s for s in scores_b if s is not None]

    return compare_two(
        scores_a,
        scores_b,
        name_a=f"commit:{commit_a}",
        name_b=f"commit:{commit_b}",
        metric=metric,
        alpha=alpha,
        paired=False,  # Different data, so unpaired
    )


@dataclass
class FacetAnalysisResult:
    """Result of analyzing performance across facet values.

    Attributes:
        facet: Name of the facet field (e.g., "form", "lcc")
        metric: Name of the metric analyzed
        model: Model name (if applicable)
        facet_scores: Dict mapping facet value to list of scores
        facet_means: Dict mapping facet value to mean score
        ranking: Facet values ranked by mean score (best first)
        variance_analysis: ANOVA or Kruskal-Wallis result
        best_facet: Facet value with highest score
        worst_facet: Facet value with lowest score
        score_range: (min_mean, max_mean) across facets
    """

    facet: str
    metric: str
    model: str | None
    facet_scores: dict[str, list[float]]
    facet_means: dict[str, float]
    ranking: list[str]
    variance_analysis: SignificanceResult | None
    best_facet: str
    worst_facet: str
    score_range: tuple[float, float]

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Facet Analysis: {self.facet}",
            f"Metric: {self.metric}",
            f"Model: {self.model or 'N/A'}",
            f"Facet values: {len(self.facet_scores)}",
            "",
            f"Best: {self.best_facet} ({self.facet_means[self.best_facet]:.4f})",
            f"Worst: {self.worst_facet} ({self.facet_means[self.worst_facet]:.4f})",
            f"Range: {self.score_range[1] - self.score_range[0]:.4f}",
        ]

        if self.variance_analysis:
            lines.extend(
                [
                    "",
                    f"Variance test: {self.variance_analysis}",
                ]
            )

        lines.extend(["", "Ranking:"])
        for i, facet_val in enumerate(self.ranking[:10], 1):
            mean = self.facet_means[facet_val]
            count = len(self.facet_scores[facet_val])
            lines.append(f"  {i}. {facet_val}: {mean:.4f} (n={count})")
        if len(self.ranking) > 10:
            lines.append(f"  ... and {len(self.ranking) - 10} more")

        return "\n".join(lines)


def compare_by_facet(
    results: list[dict[str, Any]],
    metric: str,
    facet: str,
    model: str | None = None,
    min_samples: int = 5,
    alpha: float = 0.05,
) -> FacetAnalysisResult:
    """Analyze how performance varies by a facet field.

    Useful for understanding which document types, forms, or regions
    a model performs better or worse on.

    Args:
        results: List of evaluation results with stratified_metrics
        metric: Metric to analyze
        facet: Facet field to analyze (e.g., "form", "lcc", "register")
        model: Filter to specific model (optional)
        min_samples: Minimum samples per facet value to include
        alpha: Significance level for variance test

    Returns:
        FacetAnalysisResult with full facet breakdown

    Example:
        >>> # Analyze performance by document form
        >>> analysis = compare_by_facet(results, "macro_f1", "form")
        >>> print(analysis.summary())
        >>> print(f"Model struggles with: {analysis.worst_facet}")
    """
    from scipy import stats

    # Extract facet scores from stratified_metrics
    facet_scores: dict[str, list[float]] = {}

    for result in results:
        # Filter by model if specified
        if model:
            result_model = result.get("context", {}).get("model_name")
            if result_model != model:
                continue

        # Get stratified metrics
        stratified = result.get("stratified_metrics", {})
        if not stratified:
            continue

        # Look for facet keys (format: "field=value")
        for key, metrics in stratified.items():
            if not key.startswith(f"{facet}="):
                continue

            facet_value = key.split("=", 1)[1]
            score = metrics.get(metric)

            if score is not None:
                if facet_value not in facet_scores:
                    facet_scores[facet_value] = []
                facet_scores[facet_value].append(float(score))

    # If no stratified_metrics, try per_query_metrics or per_class_metrics
    if not facet_scores:
        # Try extracting from per-query or per-class breakdowns
        for result in results:
            if model:
                result_model = result.get("context", {}).get("model_name")
                if result_model != model:
                    continue

            per_class = result.get("per_class_metrics", {})
            if per_class and facet == "class":
                for class_name, class_metrics in per_class.items():
                    score = class_metrics.get(metric)
                    if score is not None:
                        if class_name not in facet_scores:
                            facet_scores[class_name] = []
                        facet_scores[class_name].append(float(score))

    if not facet_scores:
        raise ValueError(
            f"No scores found for facet '{facet}' and metric '{metric}'. "
            "Ensure results have stratified_metrics populated."
        )

    # Filter by min_samples
    facet_scores = {k: v for k, v in facet_scores.items() if len(v) >= min_samples}

    if len(facet_scores) < 2:
        raise ValueError(
            f"Need at least 2 facet values with >= {min_samples} samples. "
            f"Found: {len(facet_scores)}"
        )

    # Compute means and ranking
    facet_means = {k: float(np.mean(v)) for k, v in facet_scores.items()}
    ranking = sorted(facet_means.keys(), key=lambda k: facet_means[k], reverse=True)

    best_facet = ranking[0]
    worst_facet = ranking[-1]
    score_range = (facet_means[worst_facet], facet_means[best_facet])

    # Variance analysis: Kruskal-Wallis (non-parametric ANOVA)
    variance_result = None
    if len(facet_scores) >= 3:
        groups = list(facet_scores.values())
        try:
            h_stat, p_value = stats.kruskal(*groups)
            variance_result = SignificanceResult(
                test_name="Kruskal-Wallis",
                statistic=float(h_stat),
                p_value=float(p_value),
                significant=p_value < alpha,
                alpha=alpha,
                details={
                    "n_groups": len(groups),
                    "interpretation": "Significant variation across facets"
                    if p_value < alpha
                    else "No significant variation across facets",
                },
            )
        except Exception as e:
            # May fail with too few samples
            variance_result = SignificanceResult(
                test_name="Kruskal-Wallis",
                statistic=0.0,
                p_value=1.0,
                significant=False,
                alpha=alpha,
                details={"error": str(e)},
            )

    return FacetAnalysisResult(
        facet=facet,
        metric=metric,
        model=model,
        facet_scores=facet_scores,
        facet_means=facet_means,
        ranking=ranking,
        variance_analysis=variance_result,
        best_facet=best_facet,
        worst_facet=worst_facet,
        score_range=score_range,
    )


def compare_generation_models(
    results: list[dict[str, Any]],
    metric: str,
    alpha: float = 0.05,
) -> MultipleComparisonResult | ComparisonResult:
    """Compare performance across different data generation models.

    Useful for checking if the generating model (gpt-5.1 vs gpt-5.2)
    affects benchmark performance.

    Args:
        results: List of evaluation results
        metric: Metric to compare
        alpha: Significance level

    Returns:
        MultipleComparisonResult if >2 models, ComparisonResult if 2 models
    """
    # Extract scores grouped by generation model
    # First, try data_provenance
    model_scores: dict[str, list[float]] = {}

    for result in results:
        # Get generation model from provenance
        provenance = result.get("data_provenance", {})
        gen_model = provenance.get("primary_model")

        # Fallback to context
        if not gen_model:
            gen_model = (
                result.get("context", {}).get("extra", {}).get("generation_model")
            )

        if not gen_model:
            gen_model = "unknown"

        # Get metric value
        score = result.get("metrics", {}).get(metric)
        if score is None:
            score = result.get("primary_score")

        if score is not None:
            if gen_model not in model_scores:
                model_scores[gen_model] = []
            model_scores[gen_model].append(float(score))

    if len(model_scores) < 2:
        raise ValueError(
            f"Need at least 2 generation models to compare. Found: {list(model_scores.keys())}"
        )

    # Use appropriate comparison
    models = list(model_scores.keys())
    if len(models) == 2:
        return compare_two(
            model_scores[models[0]],
            model_scores[models[1]],
            name_a=f"gen:{models[0]}",
            name_b=f"gen:{models[1]}",
            metric=metric,
            alpha=alpha,
            paired=False,
        )
    else:
        # Convert to ArrayLike compatible type
        scores_arrays: dict[str, ArrayLike] = {
            k: np.array(v) for k, v in model_scores.items()
        }
        return compare_multiple(scores_arrays, metric=metric, alpha=alpha)
