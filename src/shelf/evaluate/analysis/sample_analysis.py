"""Sample-level analysis for practitioner-focused insights.

This module provides tools for analyzing per-sample evaluation results,
enabling practitioners to understand model reliability, robustness, and
failure modes.

Key features:
- Sample-level variance and reliability metrics
- Error stratification by document attributes (form, register, length, etc.)
- Bootstrap CIs from per-sample data (much tighter than task-level)
- Failure mode analysis

References:
- Reliability analysis best practices in ML
- Error analysis for model debugging
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from shelf.evaluate.results import PerSampleResults


@dataclass
class SampleVarianceResult:
    """Result of sample-level variance analysis.

    Attributes:
        accuracy: Overall accuracy
        accuracy_std: Standard deviation of accuracy across bootstrap samples
        accuracy_ci: 95% confidence interval for accuracy
        error_rate: Overall error rate (1 - accuracy)
        error_rate_std: Standard deviation of error rate
        n_samples: Number of samples analyzed
        n_correct: Number correct
        n_incorrect: Number incorrect
    """

    accuracy: float
    accuracy_std: float
    accuracy_ci: tuple[float, float]
    error_rate: float
    error_rate_std: float
    n_samples: int
    n_correct: int
    n_incorrect: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "accuracy": self.accuracy,
            "accuracy_std": self.accuracy_std,
            "accuracy_ci": list(self.accuracy_ci),
            "error_rate": self.error_rate,
            "error_rate_std": self.error_rate_std,
            "n_samples": self.n_samples,
            "n_correct": self.n_correct,
            "n_incorrect": self.n_incorrect,
        }


@dataclass
class StratifiedErrorResult:
    """Result of error stratification analysis.

    Attributes:
        stratify_field: Field used for stratification
        strata: Dict mapping stratum values to error statistics
        worst_strata: List of (stratum, error_rate) sorted by error rate desc
        best_strata: List of (stratum, error_rate) sorted by error rate asc
        error_rate_range: (min_error_rate, max_error_rate) across strata
    """

    stratify_field: str
    strata: dict[str, dict[str, float]]
    worst_strata: list[tuple[str, float]]
    best_strata: list[tuple[str, float]]
    error_rate_range: tuple[float, float]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "stratify_field": self.stratify_field,
            "strata": self.strata,
            "worst_strata": self.worst_strata,
            "best_strata": self.best_strata,
            "error_rate_range": list(self.error_rate_range),
        }


@dataclass
class ReliabilityResult:
    """Result of reliability/robustness analysis.

    Attributes:
        overall_accuracy: Accuracy across all samples
        worst_case_accuracy: Accuracy on worst-performing stratum
        worst_stratum: Name of worst-performing stratum
        reliability_gap: Gap between overall and worst-case accuracy
        variance_across_strata: Variance of accuracy across strata
        is_reliable: Whether model is considered reliable (gap < threshold)
    """

    overall_accuracy: float
    worst_case_accuracy: float
    worst_stratum: str
    reliability_gap: float
    variance_across_strata: float
    is_reliable: bool
    reliability_threshold: float = 0.1  # Default: 10% gap threshold

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_accuracy": self.overall_accuracy,
            "worst_case_accuracy": self.worst_case_accuracy,
            "worst_stratum": self.worst_stratum,
            "reliability_gap": self.reliability_gap,
            "variance_across_strata": self.variance_across_strata,
            "is_reliable": self.is_reliable,
            "reliability_threshold": self.reliability_threshold,
        }


@dataclass
class SampleAnalysisResult:
    """Complete sample-level analysis result.

    Combines variance, stratified error, and reliability analyses.
    """

    model_key: str
    task: str
    variance: SampleVarianceResult
    stratified_errors: dict[str, StratifiedErrorResult]
    reliability: ReliabilityResult
    sample_count: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_key": self.model_key,
            "task": self.task,
            "variance": self.variance.to_dict(),
            "stratified_errors": {
                k: v.to_dict() for k, v in self.stratified_errors.items()
            },
            "reliability": self.reliability.to_dict(),
            "sample_count": self.sample_count,
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Sample Analysis: {self.model_key} on {self.task}",
            "=" * 50,
            f"Samples: {self.sample_count}",
            "",
            "Variance Analysis:",
            f"  Accuracy: {self.variance.accuracy:.3f} (±{self.variance.accuracy_std:.3f})",
            f"  95% CI: [{self.variance.accuracy_ci[0]:.3f}, {self.variance.accuracy_ci[1]:.3f}]",
            f"  Error rate: {self.variance.error_rate:.3f}",
            "",
            "Reliability:",
            f"  Overall: {self.reliability.overall_accuracy:.3f}",
            f"  Worst-case: {self.reliability.worst_case_accuracy:.3f} ({self.reliability.worst_stratum})",
            f"  Gap: {self.reliability.reliability_gap:.3f}",
            f"  Reliable: {'Yes' if self.reliability.is_reliable else 'No'}",
        ]

        # Add worst strata from each stratification
        for strat_field, result in self.stratified_errors.items():
            lines.extend(
                [
                    "",
                    f"Errors by {strat_field}:",
                ]
            )
            for stratum, error_rate in result.worst_strata[:3]:
                lines.append(f"  {stratum}: {error_rate:.1%} error rate")

        return "\n".join(lines)


def compute_sample_variance(
    results: PerSampleResults,
    n_bootstrap: int = 1000,
    random_seed: int = 42,
) -> SampleVarianceResult:
    """Compute sample-level variance with bootstrap confidence intervals.

    Args:
        results: Per-sample results
        n_bootstrap: Number of bootstrap iterations
        random_seed: Random seed for reproducibility

    Returns:
        SampleVarianceResult with variance statistics
    """
    correct_mask = results.get_correct_mask()
    n_samples = len(correct_mask)
    n_correct = sum(correct_mask)
    n_incorrect = n_samples - n_correct

    accuracy = n_correct / n_samples if n_samples > 0 else 0.0
    error_rate = 1.0 - accuracy

    # Bootstrap confidence interval
    rng = np.random.default_rng(random_seed)
    correct_array = np.array(correct_mask, dtype=float)

    bootstrap_accs = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, n_samples, size=n_samples)
        bootstrap_accs.append(np.mean(correct_array[indices]))

    bootstrap_accs = np.array(bootstrap_accs)
    accuracy_std = float(np.std(bootstrap_accs))
    ci_lower = float(np.percentile(bootstrap_accs, 2.5))
    ci_upper = float(np.percentile(bootstrap_accs, 97.5))

    return SampleVarianceResult(
        accuracy=accuracy,
        accuracy_std=accuracy_std,
        accuracy_ci=(ci_lower, ci_upper),
        error_rate=error_rate,
        error_rate_std=accuracy_std,  # Same for error rate
        n_samples=n_samples,
        n_correct=n_correct,
        n_incorrect=n_incorrect,
    )


def compute_stratified_errors(
    results: PerSampleResults,
    stratify_field: str,
    min_samples: int = 10,
) -> StratifiedErrorResult:
    """Compute error rates stratified by a metadata field.

    Args:
        results: Per-sample results
        stratify_field: Metadata field to stratify by
        min_samples: Minimum samples per stratum to include

    Returns:
        StratifiedErrorResult with per-stratum error statistics
    """
    strata_stats = results.compute_stratified_accuracy(stratify_field)

    # Filter to strata with enough samples
    filtered_strata = {
        k: v for k, v in strata_stats.items() if v["count"] >= min_samples
    }

    if not filtered_strata:
        # Return empty result if no strata have enough samples
        return StratifiedErrorResult(
            stratify_field=stratify_field,
            strata={},
            worst_strata=[],
            best_strata=[],
            error_rate_range=(0.0, 0.0),
        )

    # Sort by error rate
    sorted_by_error = sorted(
        filtered_strata.items(), key=lambda x: x[1]["error_rate"], reverse=True
    )
    worst_strata = [(k, v["error_rate"]) for k, v in sorted_by_error[:5]]
    best_strata = [(k, v["error_rate"]) for k, v in sorted_by_error[-5:][::-1]]

    error_rates = [v["error_rate"] for v in filtered_strata.values()]
    error_rate_range = (min(error_rates), max(error_rates))

    return StratifiedErrorResult(
        stratify_field=stratify_field,
        strata=filtered_strata,
        worst_strata=worst_strata,
        best_strata=best_strata,
        error_rate_range=error_rate_range,
    )


def compute_reliability(
    results: PerSampleResults,
    stratify_fields: list[str] | None = None,
    threshold: float = 0.1,
    min_samples: int = 10,
) -> ReliabilityResult:
    """Compute reliability metrics based on worst-case performance.

    A model is considered reliable if the gap between overall accuracy
    and worst-case stratum accuracy is below the threshold.

    Args:
        results: Per-sample results
        stratify_fields: Fields to check for worst-case (default: all available)
        threshold: Maximum acceptable accuracy gap
        min_samples: Minimum samples per stratum

    Returns:
        ReliabilityResult with reliability assessment
    """
    # Compute overall accuracy
    correct_mask = results.get_correct_mask()
    overall_accuracy = sum(correct_mask) / len(correct_mask) if correct_mask else 0.0

    # Default stratification fields
    if stratify_fields is None:
        # Check what metadata fields are available
        if results.samples:
            available_fields = set(results.samples[0].metadata.keys())
            stratify_fields = list(available_fields)
        else:
            stratify_fields = []

    # Find worst-case stratum across all stratification fields
    worst_accuracy = overall_accuracy
    worst_stratum = "overall"
    all_accuracies: list[float] = [overall_accuracy]

    for strat_field in stratify_fields:
        strata_stats = results.compute_stratified_accuracy(strat_field)

        for stratum, stats in strata_stats.items():
            if stats["count"] < min_samples:
                continue

            acc = stats["accuracy"]
            all_accuracies.append(acc)

            if acc < worst_accuracy:
                worst_accuracy = acc
                worst_stratum = f"{strat_field}={stratum}"

    # Compute variance across strata
    variance = float(np.var(all_accuracies)) if len(all_accuracies) > 1 else 0.0

    # Compute reliability gap
    reliability_gap = overall_accuracy - worst_accuracy
    is_reliable = reliability_gap <= threshold

    return ReliabilityResult(
        overall_accuracy=overall_accuracy,
        worst_case_accuracy=worst_accuracy,
        worst_stratum=worst_stratum,
        reliability_gap=reliability_gap,
        variance_across_strata=variance,
        is_reliable=is_reliable,
        reliability_threshold=threshold,
    )


def analyze_samples(
    results: PerSampleResults,
    stratify_fields: list[str] | None = None,
    reliability_threshold: float = 0.1,
    min_samples_per_stratum: int = 10,
    n_bootstrap: int = 1000,
    random_seed: int = 42,
) -> SampleAnalysisResult:
    """Comprehensive sample-level analysis.

    This is the main entry point for sample analysis.

    Args:
        results: Per-sample results
        stratify_fields: Fields to stratify by (default: auto-detect)
        reliability_threshold: Threshold for reliability assessment
        min_samples_per_stratum: Minimum samples per stratum
        n_bootstrap: Bootstrap iterations for variance
        random_seed: Random seed

    Returns:
        SampleAnalysisResult with complete analysis
    """
    # Auto-detect stratification fields if not provided
    if stratify_fields is None and results.samples:
        available_fields = set(results.samples[0].metadata.keys())
        # Filter to meaningful fields
        default_fields = [
            "form",
            "form_category",
            "register",
            "audience",
            "lcc",
            "length_bucket",
        ]
        stratify_fields = [f for f in default_fields if f in available_fields]

    # Compute variance
    variance = compute_sample_variance(
        results, n_bootstrap=n_bootstrap, random_seed=random_seed
    )

    # Compute stratified errors for each field
    stratified_errors: dict[str, StratifiedErrorResult] = {}
    for strat_field in stratify_fields or []:
        stratified_errors[strat_field] = compute_stratified_errors(
            results, strat_field, min_samples=min_samples_per_stratum
        )

    # Compute reliability
    reliability = compute_reliability(
        results,
        stratify_fields=stratify_fields,
        threshold=reliability_threshold,
        min_samples=min_samples_per_stratum,
    )

    return SampleAnalysisResult(
        model_key=results.model_key,
        task=results.task,
        variance=variance,
        stratified_errors=stratified_errors,
        reliability=reliability,
        sample_count=results.sample_count,
    )


def load_and_analyze_samples(
    per_sample_path: str | Path,
    stratify_fields: list[str] | None = None,
    **kwargs: Any,
) -> SampleAnalysisResult:
    """Load per-sample results and run analysis.

    Convenience function that combines loading and analysis.

    Args:
        per_sample_path: Path to per-sample results file (.jsonl.gz)
        stratify_fields: Fields to stratify by
        **kwargs: Additional arguments to analyze_samples

    Returns:
        SampleAnalysisResult with complete analysis
    """
    results = PerSampleResults.load(per_sample_path)
    return analyze_samples(results, stratify_fields=stratify_fields, **kwargs)


def compare_sample_reliability(
    results_list: list[PerSampleResults],
    stratify_fields: list[str] | None = None,
    threshold: float = 0.1,
) -> list[dict[str, Any]]:
    """Compare reliability across multiple models.

    Args:
        results_list: List of per-sample results from different models
        stratify_fields: Fields to check for worst-case
        threshold: Reliability threshold

    Returns:
        List of model reliability summaries, sorted by reliability gap
    """
    summaries = []

    for results in results_list:
        reliability = compute_reliability(
            results, stratify_fields=stratify_fields, threshold=threshold
        )

        summaries.append(
            {
                "model_key": results.model_key,
                "task": results.task,
                "overall_accuracy": reliability.overall_accuracy,
                "worst_case_accuracy": reliability.worst_case_accuracy,
                "worst_stratum": reliability.worst_stratum,
                "reliability_gap": reliability.reliability_gap,
                "is_reliable": reliability.is_reliable,
            }
        )

    # Sort by reliability gap (most reliable first)
    summaries.sort(key=lambda x: x["reliability_gap"])

    return summaries
