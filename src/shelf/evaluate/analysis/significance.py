"""Statistical significance tests for benchmark comparison.

This module provides various statistical tests for comparing model performance:
- Paired tests (when evaluating on same samples)
- Unpaired tests (when samples differ)
- Multiple comparison tests (when comparing many models)

The choice of test depends on:
1. Number of models being compared (2 vs many)
2. Whether evaluation is on same samples (paired vs unpaired)
3. Distribution assumptions (parametric vs non-parametric)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats


@dataclass
class SignificanceResult:
    """Result of a statistical significance test.

    Attributes:
        test_name: Name of the statistical test used
        statistic: Test statistic value
        p_value: p-value (probability of observing result under null hypothesis)
        significant: Whether difference is significant at given alpha
        alpha: Significance level used
        effect_size: Effect size measure (e.g., Cohen's d)
        effect_size_interpretation: Interpretation (small/medium/large)
        confidence_interval: CI for the difference (if applicable)
        details: Additional test-specific details
    """

    test_name: str
    statistic: float
    p_value: float
    significant: bool
    alpha: float
    effect_size: float | None = None
    effect_size_interpretation: str | None = None
    confidence_interval: tuple[float, float] | None = None
    details: dict | None = None

    def __str__(self) -> str:
        sig_marker = (
            "***"
            if self.p_value < 0.001
            else ("**" if self.p_value < 0.01 else ("*" if self.p_value < 0.05 else ""))
        )
        result = f"{self.test_name}: p={self.p_value:.4f}{sig_marker}"
        if self.effect_size is not None:
            result += f", d={self.effect_size:.3f} ({self.effect_size_interpretation})"
        return result


def cohens_d(x: ArrayLike, y: ArrayLike, paired: bool = True) -> tuple[float, str]:
    """Compute Cohen's d effect size.

    Args:
        x: First sample
        y: Second sample
        paired: Whether samples are paired (uses different formula)

    Returns:
        Tuple of (effect_size, interpretation)

    Interpretation thresholds (Cohen, 1988):
        - |d| < 0.2: negligible
        - 0.2 <= |d| < 0.5: small
        - 0.5 <= |d| < 0.8: medium
        - |d| >= 0.8: large
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if paired:
        # For paired data, use the SD of differences
        diff = x - y
        d = np.mean(diff) / np.std(diff, ddof=1)
    else:
        # Pooled standard deviation
        nx, ny = len(x), len(y)
        pooled_std = np.sqrt(
            ((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1))
            / (nx + ny - 2)
        )
        d = (np.mean(x) - np.mean(y)) / pooled_std

    # Interpretation
    abs_d = abs(d)
    if abs_d < 0.2:
        interpretation = "negligible"
    elif abs_d < 0.5:
        interpretation = "small"
    elif abs_d < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"

    return d, interpretation


def wilcoxon_test(
    x: ArrayLike,
    y: ArrayLike,
    alpha: float = 0.05,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
) -> SignificanceResult:
    """Wilcoxon signed-rank test for paired samples.

    Non-parametric alternative to paired t-test. Tests whether the
    distribution of differences is symmetric around zero.

    Args:
        x: First sample (e.g., scores from model A)
        y: Second sample (e.g., scores from model B)
        alpha: Significance level
        alternative: Direction of test

    Returns:
        SignificanceResult with test outcome
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Remove ties (where x == y)
    diff = x - y
    nonzero_mask = diff != 0
    if nonzero_mask.sum() < 10:
        # Too few non-tied pairs for reliable test
        return SignificanceResult(
            test_name="Wilcoxon signed-rank",
            statistic=np.nan,
            p_value=1.0,
            significant=False,
            alpha=alpha,
            details={"warning": "Too few non-tied pairs for reliable test"},
        )

    result = stats.wilcoxon(x, y, alternative=alternative)
    effect_d, effect_interp = cohens_d(x, y, paired=True)

    return SignificanceResult(
        test_name="Wilcoxon signed-rank",
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        significant=result.pvalue < alpha,
        alpha=alpha,
        effect_size=effect_d,
        effect_size_interpretation=effect_interp,
        details={
            "n_pairs": len(x),
            "n_nonzero_diffs": int(nonzero_mask.sum()),
            "mean_diff": float(np.mean(diff)),
            "median_diff": float(np.median(diff)),
        },
    )


def paired_t_test(
    x: ArrayLike,
    y: ArrayLike,
    alpha: float = 0.05,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
) -> SignificanceResult:
    """Paired t-test for comparing two related samples.

    Parametric test assuming differences are normally distributed.
    Use wilcoxon_test for non-parametric alternative.

    Args:
        x: First sample
        y: Second sample
        alpha: Significance level
        alternative: Direction of test

    Returns:
        SignificanceResult with test outcome
    """
    x = np.asarray(x)
    y = np.asarray(y)

    result = stats.ttest_rel(x, y, alternative=alternative)
    effect_d, effect_interp = cohens_d(x, y, paired=True)

    # Confidence interval for mean difference
    diff = x - y
    se = stats.sem(diff)
    df = len(diff) - 1
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    ci = (float(np.mean(diff) - t_crit * se), float(np.mean(diff) + t_crit * se))

    return SignificanceResult(
        test_name="Paired t-test",
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        significant=result.pvalue < alpha,
        alpha=alpha,
        effect_size=effect_d,
        effect_size_interpretation=effect_interp,
        confidence_interval=ci,
        details={
            "n_pairs": len(x),
            "mean_diff": float(np.mean(diff)),
            "std_diff": float(np.std(diff, ddof=1)),
            "df": df,
        },
    )


def paired_permutation_test(
    x: ArrayLike,
    y: ArrayLike,
    alpha: float = 0.05,
    n_permutations: int = 10000,
    random_state: int | None = 42,
) -> SignificanceResult:
    """Permutation test for paired samples.

    Non-parametric, assumption-free test. Under the null hypothesis,
    the sign of each difference is equally likely to be positive or negative.

    Args:
        x: First sample
        y: Second sample
        alpha: Significance level
        n_permutations: Number of permutations
        random_state: Random seed for reproducibility

    Returns:
        SignificanceResult with test outcome
    """
    x = np.asarray(x)
    y = np.asarray(y)
    rng = np.random.default_rng(random_state)

    diff = x - y
    observed_stat = np.mean(diff)

    # Generate null distribution by randomly flipping signs
    n = len(diff)
    null_stats = []
    for _ in range(n_permutations):
        signs = rng.choice([-1, 1], size=n)
        null_stats.append(np.mean(diff * signs))

    null_stats = np.array(null_stats)

    # Two-sided p-value
    p_value = np.mean(np.abs(null_stats) >= np.abs(observed_stat))

    effect_d, effect_interp = cohens_d(x, y, paired=True)

    return SignificanceResult(
        test_name="Paired permutation test",
        statistic=observed_stat,
        p_value=float(p_value),
        significant=bool(p_value < alpha),
        alpha=alpha,
        effect_size=effect_d,
        effect_size_interpretation=effect_interp,
        details={
            "n_pairs": n,
            "n_permutations": n_permutations,
            "mean_diff": float(observed_stat),
        },
    )


def mcnemar_test(
    correct_a: ArrayLike,
    correct_b: ArrayLike,
    alpha: float = 0.05,
) -> SignificanceResult:
    """McNemar's test for paired binary outcomes.

    Tests whether two classifiers have the same error rate on the same data.
    Specifically designed for classification accuracy comparison.

    Args:
        correct_a: Binary array (1 = correct, 0 = incorrect) for model A
        correct_b: Binary array for model B
        alpha: Significance level

    Returns:
        SignificanceResult with test outcome
    """
    correct_a = np.asarray(correct_a, dtype=bool)
    correct_b = np.asarray(correct_b, dtype=bool)

    # Build contingency table
    # b = A correct, B incorrect
    # c = A incorrect, B correct
    b = np.sum(correct_a & ~correct_b)
    c = np.sum(~correct_a & correct_b)

    # McNemar's test with continuity correction
    if b + c == 0:
        # No discordant pairs
        return SignificanceResult(
            test_name="McNemar's test",
            statistic=0.0,
            p_value=1.0,
            significant=False,
            alpha=alpha,
            details={"warning": "No discordant pairs", "b": int(b), "c": int(c)},
        )

    # Chi-squared statistic with continuity correction
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - stats.chi2.cdf(chi2, df=1)

    # Effect size: odds ratio
    odds_ratio = b / c if c > 0 else float("inf")

    return SignificanceResult(
        test_name="McNemar's test",
        statistic=float(chi2),
        p_value=float(p_value),
        significant=p_value < alpha,
        alpha=alpha,
        effect_size=float(np.log(odds_ratio)) if odds_ratio != float("inf") else None,
        effect_size_interpretation="log odds ratio",
        details={
            "b_a_correct_b_incorrect": int(b),
            "c_a_incorrect_b_correct": int(c),
            "odds_ratio": float(odds_ratio) if odds_ratio != float("inf") else None,
            "accuracy_a": float(np.mean(correct_a)),
            "accuracy_b": float(np.mean(correct_b)),
        },
    )


@dataclass
class FriedmanNemenyiResult:
    """Result of Friedman test with Nemenyi post-hoc.

    Attributes:
        friedman_statistic: Friedman test statistic
        friedman_p_value: p-value for Friedman test
        significant_overall: Whether any difference exists
        n_groups: Number of groups (models) compared
        n_blocks: Number of blocks (datasets/tasks)
        mean_ranks: Mean rank for each group
        critical_difference: CD for Nemenyi test
        pairwise_p_values: Matrix of pairwise p-values
        significant_pairs: List of significantly different pairs
        alpha: Significance level used
    """

    friedman_statistic: float
    friedman_p_value: float
    significant_overall: bool
    n_groups: int
    n_blocks: int
    mean_ranks: dict[str, float]
    critical_difference: float
    pairwise_p_values: dict[tuple[str, str], float]
    significant_pairs: list[tuple[str, str]]
    alpha: float

    def __str__(self) -> str:
        lines = [
            f"Friedman test: χ²={self.friedman_statistic:.2f}, p={self.friedman_p_value:.4f}",
            f"Critical difference (α={self.alpha}): {self.critical_difference:.3f}",
            f"Significant pairs: {len(self.significant_pairs)}",
        ]
        if self.significant_pairs:
            for a, b in self.significant_pairs[:5]:
                lines.append(f"  {a} vs {b}: p={self.pairwise_p_values[(a, b)]:.4f}")
            if len(self.significant_pairs) > 5:
                lines.append(f"  ... and {len(self.significant_pairs) - 5} more")
        return "\n".join(lines)


def friedman_nemenyi_test(
    data: dict[str, ArrayLike],
    alpha: float = 0.05,
) -> FriedmanNemenyiResult:
    """Friedman test with Nemenyi post-hoc for multiple model comparison.

    Standard approach for comparing multiple ML models across multiple datasets.
    The Friedman test checks if any model is significantly different, then
    Nemenyi post-hoc identifies which pairs differ.

    Args:
        data: Dictionary mapping model name to array of scores
              Each array should have same length (one score per dataset/task)
        alpha: Significance level

    Returns:
        FriedmanNemenyiResult with full comparison details

    Example:
        >>> data = {
        ...     "BERT": [0.85, 0.82, 0.88, 0.79],
        ...     "RoBERTa": [0.87, 0.84, 0.86, 0.81],
        ...     "MiniLM": [0.82, 0.80, 0.84, 0.77],
        ... }
        >>> result = friedman_nemenyi_test(data)
        >>> print(result)
    """
    try:
        import scikit_posthocs as sp
    except ImportError:
        raise ImportError(
            "scikit-posthocs is required for Nemenyi test. "
            "Install with: pip install scikit-posthocs"
        )

    # Convert to matrix format (rows = blocks/datasets, cols = groups/models)
    model_names = list(data.keys())
    matrix = np.column_stack([np.asarray(data[name]) for name in model_names])

    n_blocks, n_groups = matrix.shape

    # Friedman test
    friedman_stat, friedman_p = stats.friedmanchisquare(*matrix.T)

    # Compute ranks (lower score = better rank, so we negate for ranking)
    # Actually, higher score is usually better, so rank directly
    ranks = np.zeros_like(matrix)
    for i in range(n_blocks):
        ranks[i] = stats.rankdata(-matrix[i])  # Negate so higher score = rank 1

    mean_ranks = {name: float(ranks[:, j].mean()) for j, name in enumerate(model_names)}

    # Nemenyi post-hoc test
    # scikit-posthocs expects data in specific format
    nemenyi_p = sp.posthoc_nemenyi_friedman(matrix)

    # Critical difference for Nemenyi test
    # CD = q_α * sqrt(k(k+1)/(6n))
    # where k = n_groups, n = n_blocks
    # q_α from studentized range distribution (approximation for Nemenyi)
    q_alpha = stats.studentized_range.ppf(1 - alpha, n_groups, np.inf)
    cd = q_alpha * np.sqrt(n_groups * (n_groups + 1) / (6 * n_blocks))

    # Build pairwise p-values dict and find significant pairs
    pairwise_p = {}
    significant_pairs = []
    for i, name_i in enumerate(model_names):
        for j, name_j in enumerate(model_names):
            if i < j:
                p_val = nemenyi_p.iloc[i, j]
                pairwise_p[(name_i, name_j)] = float(p_val)
                if p_val < alpha:
                    significant_pairs.append((name_i, name_j))

    return FriedmanNemenyiResult(
        friedman_statistic=float(friedman_stat),
        friedman_p_value=float(friedman_p),
        significant_overall=friedman_p < alpha,
        n_groups=n_groups,
        n_blocks=n_blocks,
        mean_ranks=mean_ranks,
        critical_difference=float(cd),
        pairwise_p_values=pairwise_p,
        significant_pairs=significant_pairs,
        alpha=alpha,
    )


def multiple_comparison_correction(
    p_values: ArrayLike,
    method: Literal["bonferroni", "holm", "fdr_bh", "fdr_by"] = "holm",
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply multiple comparison correction to p-values.

    Args:
        p_values: Array of p-values from multiple tests
        method: Correction method:
            - "bonferroni": Conservative, controls family-wise error rate
            - "holm": Less conservative step-down procedure
            - "fdr_bh": Benjamini-Hochberg, controls false discovery rate
            - "fdr_by": Benjamini-Yekutieli, more conservative FDR
        alpha: Significance level

    Returns:
        Tuple of (reject_array, corrected_p_values)
    """
    from statsmodels.stats.multitest import multipletests

    reject, corrected_p, _, _ = multipletests(p_values, alpha=alpha, method=method)
    return reject, corrected_p
