"""Bootstrap methods for confidence intervals and hypothesis testing.

Bootstrap provides non-parametric, assumption-free confidence intervals
and significance tests. Particularly useful when:
- Sample size is small
- Distribution is non-normal
- Standard error formulas don't exist for the statistic

References:
- Efron & Tibshirani (1993). An Introduction to the Bootstrap.
- Berg-Kirkpatrick et al. (2012). An Empirical Investigation of
  Statistical Significance in NLP. EMNLP.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike


@dataclass
class BootstrapResult:
    """Result of bootstrap analysis.

    Attributes:
        estimate: Point estimate of the statistic
        ci_lower: Lower bound of confidence interval
        ci_upper: Upper bound of confidence interval
        ci_level: Confidence level (e.g., 0.95 for 95% CI)
        se: Bootstrap standard error
        n_bootstrap: Number of bootstrap iterations
        bootstrap_distribution: Full bootstrap distribution (optional)
    """

    estimate: float
    ci_lower: float
    ci_upper: float
    ci_level: float
    se: float
    n_bootstrap: int
    bootstrap_distribution: np.ndarray | None = None

    def __str__(self) -> str:
        ci_pct = int(self.ci_level * 100)
        return f"{self.estimate:.4f} ({ci_pct}% CI: [{self.ci_lower:.4f}, {self.ci_upper:.4f}])"

    @property
    def ci_width(self) -> float:
        """Width of confidence interval."""
        return self.ci_upper - self.ci_lower

    def contains(self, value: float) -> bool:
        """Check if value falls within confidence interval."""
        return self.ci_lower <= value <= self.ci_upper


def bootstrap_ci(
    data: ArrayLike,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    method: str = "percentile",
    random_state: int | None = 42,
    return_distribution: bool = False,
) -> BootstrapResult:
    """Compute bootstrap confidence interval for a statistic.

    Args:
        data: Input data array
        statistic: Function that computes the statistic of interest
        n_bootstrap: Number of bootstrap samples
        ci_level: Confidence level (0.95 = 95% CI)
        method: CI method ("percentile", "basic", or "bca")
        random_state: Random seed for reproducibility
        return_distribution: Whether to return full bootstrap distribution

    Returns:
        BootstrapResult with estimate and confidence interval

    Example:
        >>> scores = [0.82, 0.85, 0.79, 0.88, 0.84]
        >>> result = bootstrap_ci(scores, statistic=np.mean)
        >>> print(result)
        0.8360 (95% CI: [0.8040, 0.8680])
    """
    data = np.asarray(data)
    rng = np.random.default_rng(random_state)
    n = len(data)

    # Original estimate
    theta_hat = statistic(data)

    # Generate bootstrap distribution
    bootstrap_stats = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        # Sample with replacement
        indices = rng.integers(0, n, size=n)
        bootstrap_sample = data[indices]
        bootstrap_stats[i] = statistic(bootstrap_sample)

    # Compute confidence interval
    alpha = 1 - ci_level

    if method == "percentile":
        # Simple percentile method
        ci_lower = float(np.percentile(bootstrap_stats, 100 * alpha / 2))
        ci_upper = float(np.percentile(bootstrap_stats, 100 * (1 - alpha / 2)))

    elif method == "basic":
        # Basic bootstrap (pivot method)
        q_lower = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
        q_upper = np.percentile(bootstrap_stats, 100 * alpha / 2)
        ci_lower = float(2 * theta_hat - q_lower)
        ci_upper = float(2 * theta_hat - q_upper)

    elif method == "bca":
        # Bias-corrected and accelerated (BCa) bootstrap
        # More accurate but computationally intensive
        from scipy import stats as scipy_stats

        # Bias correction factor
        z0 = scipy_stats.norm.ppf(np.mean(bootstrap_stats < theta_hat))

        # Acceleration factor (jackknife estimate)
        jackknife_stats = np.zeros(n)
        for i in range(n):
            jackknife_sample = np.delete(data, i)
            jackknife_stats[i] = statistic(jackknife_sample)
        jack_mean = np.mean(jackknife_stats)
        a = np.sum((jack_mean - jackknife_stats) ** 3) / (
            6 * np.sum((jack_mean - jackknife_stats) ** 2) ** 1.5 + 1e-10
        )

        # Adjusted percentiles
        z_alpha_lower = scipy_stats.norm.ppf(alpha / 2)
        z_alpha_upper = scipy_stats.norm.ppf(1 - alpha / 2)

        alpha_lower = scipy_stats.norm.cdf(
            z0 + (z0 + z_alpha_lower) / (1 - a * (z0 + z_alpha_lower))
        )
        alpha_upper = scipy_stats.norm.cdf(
            z0 + (z0 + z_alpha_upper) / (1 - a * (z0 + z_alpha_upper))
        )

        ci_lower = float(np.percentile(bootstrap_stats, 100 * alpha_lower))
        ci_upper = float(np.percentile(bootstrap_stats, 100 * alpha_upper))

    else:
        raise ValueError(
            f"Unknown method: {method}. Use 'percentile', 'basic', or 'bca'"
        )

    return BootstrapResult(
        estimate=float(theta_hat),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_level=ci_level,
        se=float(np.std(bootstrap_stats, ddof=1)),
        n_bootstrap=n_bootstrap,
        bootstrap_distribution=bootstrap_stats if return_distribution else None,
    )


@dataclass
class BootstrapDifferenceResult:
    """Result of bootstrap test for difference between two samples.

    Attributes:
        mean_diff: Mean difference (A - B)
        ci_lower: Lower bound of CI for difference
        ci_upper: Upper bound of CI for difference
        ci_level: Confidence level
        p_value: p-value for null hypothesis that difference = 0
        significant: Whether difference is significant (CI excludes 0)
        se: Standard error of the difference
        n_bootstrap: Number of bootstrap iterations
    """

    mean_diff: float
    ci_lower: float
    ci_upper: float
    ci_level: float
    p_value: float
    significant: bool
    se: float
    n_bootstrap: int

    def __str__(self) -> str:
        ci_pct = int(self.ci_level * 100)
        sig = "*" if self.significant else ""
        return (
            f"Δ = {self.mean_diff:+.4f} ({ci_pct}% CI: [{self.ci_lower:+.4f}, {self.ci_upper:+.4f}])"
            f" p={self.p_value:.4f}{sig}"
        )


def bootstrap_paired_difference(
    x: ArrayLike,
    y: ArrayLike,
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    random_state: int | None = 42,
) -> BootstrapDifferenceResult:
    """Bootstrap test for paired difference.

    Tests whether the mean difference between paired samples is significantly
    different from zero. More robust than t-test for non-normal data.

    Args:
        x: First sample (e.g., model A scores)
        y: Second sample (e.g., model B scores)
        n_bootstrap: Number of bootstrap samples
        ci_level: Confidence level for CI
        random_state: Random seed

    Returns:
        BootstrapDifferenceResult with difference statistics

    Example:
        >>> model_a = [0.82, 0.85, 0.79, 0.88, 0.84]
        >>> model_b = [0.80, 0.83, 0.78, 0.85, 0.82]
        >>> result = bootstrap_paired_difference(model_a, model_b)
        >>> print(result)
        Δ = +0.0200 (95% CI: [+0.0080, +0.0340]) p=0.0023*
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if len(x) != len(y):
        raise ValueError(f"Arrays must have same length: {len(x)} vs {len(y)}")

    rng = np.random.default_rng(random_state)
    n = len(x)

    # Observed difference
    diff = x - y
    observed_mean_diff = np.mean(diff)

    # Bootstrap the difference
    bootstrap_diffs = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        bootstrap_diffs[i] = np.mean(diff[indices])

    # Confidence interval
    alpha = 1 - ci_level
    ci_lower = float(np.percentile(bootstrap_diffs, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_diffs, 100 * (1 - alpha / 2)))

    # p-value: proportion of bootstrap samples on opposite side of zero
    # Two-sided test
    if observed_mean_diff >= 0:
        p_value = 2 * np.mean(bootstrap_diffs <= 0)
    else:
        p_value = 2 * np.mean(bootstrap_diffs >= 0)
    p_value = min(p_value, 1.0)  # Cap at 1

    # Significance: CI doesn't include zero
    significant = (ci_lower > 0) or (ci_upper < 0)

    return BootstrapDifferenceResult(
        mean_diff=float(observed_mean_diff),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_level=ci_level,
        p_value=float(p_value),
        significant=significant,
        se=float(np.std(bootstrap_diffs, ddof=1)),
        n_bootstrap=n_bootstrap,
    )


def bootstrap_ratio(
    x: ArrayLike,
    y: ArrayLike,
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    random_state: int | None = 42,
) -> BootstrapResult:
    """Bootstrap confidence interval for ratio of means.

    Useful for relative improvement calculations.

    Args:
        x: Numerator sample
        y: Denominator sample
        n_bootstrap: Number of bootstrap samples
        ci_level: Confidence level
        random_state: Random seed

    Returns:
        BootstrapResult for mean(x) / mean(y)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    rng = np.random.default_rng(random_state)

    n_x, n_y = len(x), len(y)

    # Observed ratio
    ratio_hat = np.mean(x) / np.mean(y)

    # Bootstrap
    bootstrap_ratios = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        x_boot = x[rng.integers(0, n_x, size=n_x)]
        y_boot = y[rng.integers(0, n_y, size=n_y)]
        bootstrap_ratios[i] = np.mean(x_boot) / np.mean(y_boot)

    alpha = 1 - ci_level
    ci_lower = float(np.percentile(bootstrap_ratios, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_ratios, 100 * (1 - alpha / 2)))

    return BootstrapResult(
        estimate=float(ratio_hat),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_level=ci_level,
        se=float(np.std(bootstrap_ratios, ddof=1)),
        n_bootstrap=n_bootstrap,
        bootstrap_distribution=None,
    )
