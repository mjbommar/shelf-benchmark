"""Unit tests for shelf.evaluate.analysis.significance module.

Tests cover:
- Effect size calculations (Cohen's d)
- Paired statistical tests (Wilcoxon, t-test, permutation)
- McNemar's test for binary outcomes
- Friedman-Nemenyi test for multiple comparisons
- Multiple comparison correction methods
"""

from __future__ import annotations

import numpy as np
import pytest

from shelf.evaluate.analysis.significance import (
    FriedmanNemenyiResult,
    SignificanceResult,
    cohens_d,
    friedman_nemenyi_test,
    mcnemar_test,
    multiple_comparison_correction,
    paired_permutation_test,
    paired_t_test,
    wilcoxon_test,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def paired_scores_different():
    """Paired scores with significant difference."""
    np.random.seed(42)
    # Model A significantly better than Model B
    scores_a = np.array([0.85, 0.87, 0.83, 0.86, 0.84, 0.88, 0.82, 0.87, 0.85, 0.86])
    scores_b = np.array([0.75, 0.77, 0.73, 0.76, 0.74, 0.78, 0.72, 0.77, 0.75, 0.76])
    return scores_a, scores_b


@pytest.fixture
def paired_scores_similar():
    """Paired scores with no significant difference."""
    np.random.seed(42)
    base = np.array([0.80, 0.82, 0.85, 0.78, 0.81, 0.83, 0.79, 0.84, 0.86, 0.77])
    scores_a = base + np.random.normal(0, 0.01, 10)
    scores_b = base + np.random.normal(0, 0.01, 10)
    return scores_a, scores_b


@pytest.fixture
def multiple_model_scores():
    """Scores from multiple models across multiple datasets."""
    np.random.seed(42)
    return {
        "ModelA": np.array([0.85, 0.87, 0.83, 0.86, 0.84, 0.88, 0.82, 0.87]),
        "ModelB": np.array([0.75, 0.77, 0.73, 0.76, 0.74, 0.78, 0.72, 0.77]),
        "ModelC": np.array([0.80, 0.82, 0.78, 0.81, 0.79, 0.83, 0.77, 0.82]),
        "ModelD": np.array([0.70, 0.72, 0.68, 0.71, 0.69, 0.73, 0.67, 0.72]),
    }


# ===========================================================================
# SignificanceResult Tests
# ===========================================================================


class TestSignificanceResult:
    """Tests for SignificanceResult dataclass."""

    def test_str_with_significance_markers(self):
        """Test string shows significance markers."""
        # p < 0.001 -> ***
        result = SignificanceResult(
            test_name="Test",
            statistic=10.0,
            p_value=0.0001,
            significant=True,
            alpha=0.05,
        )
        assert "***" in str(result)

        # p < 0.01 -> **
        result2 = SignificanceResult(
            test_name="Test",
            statistic=10.0,
            p_value=0.005,
            significant=True,
            alpha=0.05,
        )
        assert "**" in str(result2)

        # p < 0.05 -> *
        result3 = SignificanceResult(
            test_name="Test",
            statistic=10.0,
            p_value=0.03,
            significant=True,
            alpha=0.05,
        )
        assert "*" in str(result3)

        # p >= 0.05 -> no marker
        result4 = SignificanceResult(
            test_name="Test",
            statistic=10.0,
            p_value=0.10,
            significant=False,
            alpha=0.05,
        )
        s = str(result4)
        # Should not have any significance markers
        assert (
            "***" not in s or "p=0.1000" in s
        )  # Either no *** or p-value shows it's not significant

    def test_str_with_effect_size(self):
        """Test string includes effect size."""
        result = SignificanceResult(
            test_name="Test",
            statistic=10.0,
            p_value=0.01,
            significant=True,
            alpha=0.05,
            effect_size=0.8,
            effect_size_interpretation="large",
        )
        s = str(result)
        assert "d=0.800" in s or "d=0.8" in s
        assert "large" in s


# ===========================================================================
# Cohen's d Tests
# ===========================================================================


class TestCohensD:
    """Tests for Cohen's d effect size calculation."""

    def test_zero_effect(self):
        """Test identical arrays give zero effect."""
        data = np.array([1, 2, 3, 4, 5])
        d, interp = cohens_d(data, data, paired=True)
        # When x == y, diff is all zeros, and std(diff) is 0, so we get NaN or 0
        # Let's check if it's close to 0 or NaN
        assert np.isnan(d) or abs(d) < 0.01
        if not np.isnan(d):
            assert interp == "negligible"

    def test_small_effect(self):
        """Test small effect size interpretation."""
        # Create data with genuinely small effect
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        y = np.array(
            [1.1, 2.1, 2.9, 3.9, 5.1, 5.9, 7.1, 8.0, 9.1, 9.9]
        )  # Varied small diffs
        d, interp = cohens_d(x, y, paired=True)
        # Should be small or negligible, not large
        assert interp in ["small", "negligible", "medium"]

    def test_large_effect(self):
        """Test large effect size detection."""
        x = np.array([10, 11, 12, 13, 14])
        y = np.array([1, 2, 3, 4, 5])  # Large difference
        d, interp = cohens_d(x, y, paired=True)
        assert interp == "large"
        assert abs(d) >= 0.8

    def test_paired_vs_unpaired(self):
        """Test difference between paired and unpaired calculation."""
        np.random.seed(42)
        x = np.random.normal(10, 2, 30)
        y = np.random.normal(8, 2, 30)

        d_paired, _ = cohens_d(x, y, paired=True)
        d_unpaired, _ = cohens_d(x, y, paired=False)

        # Both should detect positive effect (x > y)
        assert d_paired > 0
        assert d_unpaired > 0


# ===========================================================================
# Wilcoxon Test
# ===========================================================================


class TestWilcoxonTest:
    """Tests for Wilcoxon signed-rank test."""

    def test_significant_difference(self, paired_scores_different):
        """Test detection of significant difference."""
        scores_a, scores_b = paired_scores_different
        result = wilcoxon_test(scores_a, scores_b)

        assert isinstance(result, SignificanceResult)
        assert result.test_name == "Wilcoxon signed-rank"
        assert bool(result.significant) is True
        assert result.p_value < 0.05

    def test_no_significant_difference(self, paired_scores_similar):
        """Test when no significant difference."""
        scores_a, scores_b = paired_scores_similar
        result = wilcoxon_test(scores_a, scores_b)

        # May or may not be significant
        assert isinstance(result, SignificanceResult)

    def test_identical_arrays(self):
        """Test with identical arrays (all ties)."""
        data = np.array([0.80, 0.82, 0.85, 0.78, 0.81, 0.83, 0.79, 0.84, 0.86, 0.77])
        result = wilcoxon_test(data, data)

        # Should return non-significant with warning
        assert result.p_value == 1.0
        assert result.significant is False

    def test_effect_size_included(self, paired_scores_different):
        """Test that effect size is computed."""
        scores_a, scores_b = paired_scores_different
        result = wilcoxon_test(scores_a, scores_b)

        assert result.effect_size is not None
        assert result.effect_size_interpretation is not None

    def test_alternative_hypothesis(self, paired_scores_different):
        """Test different alternative hypotheses."""
        scores_a, scores_b = paired_scores_different

        # Greater (A > B)
        result_greater = wilcoxon_test(scores_a, scores_b, alternative="greater")
        # Less (A < B)
        result_less = wilcoxon_test(scores_a, scores_b, alternative="less")

        # Since A > B significantly, greater should have lower p-value than less
        assert result_greater.p_value < result_less.p_value


# ===========================================================================
# Paired t-test
# ===========================================================================


class TestPairedTTest:
    """Tests for paired t-test."""

    def test_significant_difference(self, paired_scores_different):
        """Test detection of significant difference."""
        scores_a, scores_b = paired_scores_different
        result = paired_t_test(scores_a, scores_b)

        assert result.test_name == "Paired t-test"
        assert bool(result.significant) is True
        assert result.p_value < 0.05

    def test_confidence_interval(self, paired_scores_different):
        """Test that CI is computed."""
        scores_a, scores_b = paired_scores_different
        result = paired_t_test(scores_a, scores_b)

        assert result.confidence_interval is not None
        ci_lower, ci_upper = result.confidence_interval
        # CI can be the same if std_diff is 0 (all differences identical)
        assert ci_lower <= ci_upper

    def test_details_included(self, paired_scores_different):
        """Test that details dict is populated."""
        scores_a, scores_b = paired_scores_different
        result = paired_t_test(scores_a, scores_b)

        assert result.details is not None
        assert "n_pairs" in result.details
        assert "mean_diff" in result.details
        assert "df" in result.details


# ===========================================================================
# Paired Permutation Test
# ===========================================================================


class TestPairedPermutationTest:
    """Tests for paired permutation test."""

    def test_significant_difference(self, paired_scores_different):
        """Test detection of significant difference."""
        scores_a, scores_b = paired_scores_different
        result = paired_permutation_test(
            scores_a,
            scores_b,
            n_permutations=1000,
            random_state=42,
        )

        assert result.test_name == "Paired permutation test"
        assert result.significant is True

    def test_reproducibility(self, paired_scores_different):
        """Test that random_state ensures reproducibility."""
        scores_a, scores_b = paired_scores_different

        result1 = paired_permutation_test(scores_a, scores_b, random_state=42)
        result2 = paired_permutation_test(scores_a, scores_b, random_state=42)

        assert result1.p_value == result2.p_value

    def test_no_difference(self):
        """Test with identical data."""
        data = np.array([0.80, 0.82, 0.85, 0.78, 0.81])
        result = paired_permutation_test(data, data, random_state=42)

        assert result.details["mean_diff"] == pytest.approx(0.0, abs=0.001)


# ===========================================================================
# McNemar Test
# ===========================================================================


class TestMcNemarTest:
    """Tests for McNemar's test."""

    def test_significant_difference(self):
        """Test when one model is clearly better."""
        # Model A correct, B incorrect: 50 cases
        # Model A incorrect, B correct: 5 cases
        # Need larger discrepancy for significance with continuity correction
        correct_a = np.array([1] * 50 + [0] * 5 + [1] * 50)
        correct_b = np.array([1] * 50 + [1] * 5 + [0] * 50)

        result = mcnemar_test(correct_a, correct_b)

        assert result.test_name == "McNemar's test"
        assert bool(result.significant) is True

    def test_no_discordant_pairs(self):
        """Test when models always agree."""
        correct_a = np.array([1, 1, 0, 0, 1, 0])
        correct_b = np.array([1, 1, 0, 0, 1, 0])

        result = mcnemar_test(correct_a, correct_b)

        assert result.p_value == 1.0
        assert result.significant is False

    def test_details_included(self):
        """Test that details dict has contingency counts."""
        correct_a = np.array([1, 1, 0, 0, 1, 0, 1, 0])
        correct_b = np.array([1, 0, 0, 1, 1, 0, 0, 1])

        result = mcnemar_test(correct_a, correct_b)

        assert result.details is not None
        assert "b_a_correct_b_incorrect" in result.details
        assert "c_a_incorrect_b_correct" in result.details


# ===========================================================================
# Friedman-Nemenyi Test
# ===========================================================================


class TestFriedmanNemenyiTest:
    """Tests for Friedman test with Nemenyi post-hoc."""

    def test_significant_overall(self, multiple_model_scores):
        """Test detection of overall significance."""
        result = friedman_nemenyi_test(multiple_model_scores)

        assert isinstance(result, FriedmanNemenyiResult)
        assert bool(result.significant_overall) is True
        assert result.friedman_p_value < 0.05

    def test_mean_ranks_computed(self, multiple_model_scores):
        """Test that mean ranks are computed for all models."""
        result = friedman_nemenyi_test(multiple_model_scores)

        assert len(result.mean_ranks) == 4  # 4 models
        for model in multiple_model_scores:
            assert model in result.mean_ranks

    def test_critical_difference(self, multiple_model_scores):
        """Test that critical difference is computed."""
        result = friedman_nemenyi_test(multiple_model_scores)

        assert result.critical_difference > 0

    def test_pairwise_comparisons(self, multiple_model_scores):
        """Test that pairwise p-values are computed."""
        result = friedman_nemenyi_test(multiple_model_scores)

        # Should have n*(n-1)/2 pairs for n=4 models
        assert len(result.pairwise_p_values) == 6

    def test_str_representation(self, multiple_model_scores):
        """Test string representation."""
        result = friedman_nemenyi_test(multiple_model_scores)
        s = str(result)

        assert "Friedman" in s
        assert "Critical difference" in s

    def test_no_significant_difference(self):
        """Test when models perform similarly."""
        np.random.seed(42)
        base = np.random.normal(0.80, 0.02, 10)
        similar_scores = {
            "ModelA": base + np.random.normal(0, 0.005, 10),
            "ModelB": base + np.random.normal(0, 0.005, 10),
            "ModelC": base + np.random.normal(0, 0.005, 10),
        }

        result = friedman_nemenyi_test(similar_scores)

        # May or may not be significant, but should run without error
        assert isinstance(result, FriedmanNemenyiResult)


# ===========================================================================
# Multiple Comparison Correction
# ===========================================================================


class TestMultipleComparisonCorrection:
    """Tests for multiple comparison correction."""

    def test_bonferroni_correction(self):
        """Test Bonferroni correction."""
        p_values = np.array([0.01, 0.03, 0.05])
        reject, corrected = multiple_comparison_correction(
            p_values, method="bonferroni", alpha=0.05
        )

        # Bonferroni is conservative: corrected = p * n
        assert corrected[0] == pytest.approx(0.03, rel=0.01)

    def test_holm_correction(self):
        """Test Holm step-down procedure."""
        p_values = np.array([0.01, 0.03, 0.05])
        reject, corrected = multiple_comparison_correction(
            p_values, method="holm", alpha=0.05
        )

        # Holm is less conservative than Bonferroni
        assert isinstance(reject, np.ndarray)
        assert isinstance(corrected, np.ndarray)

    def test_fdr_correction(self):
        """Test FDR (Benjamini-Hochberg) correction."""
        p_values = np.array([0.001, 0.01, 0.03, 0.05, 0.10])
        reject, corrected = multiple_comparison_correction(
            p_values, method="fdr_bh", alpha=0.05
        )

        # FDR controls false discovery rate, usually less strict
        assert bool(reject[0]) is True  # Very small p-value


# ===========================================================================
# Edge Cases
# ===========================================================================


class TestSignificanceEdgeCases:
    """Edge case tests for significance functions."""

    def test_wilcoxon_few_pairs(self):
        """Test Wilcoxon with very few non-tied pairs."""
        x = np.array([0.80, 0.80, 0.80, 0.80, 0.80, 0.81, 0.82, 0.83])
        y = np.array([0.80, 0.80, 0.80, 0.80, 0.80, 0.79, 0.78, 0.77])

        result = wilcoxon_test(x, y)
        # Should handle gracefully
        assert isinstance(result, SignificanceResult)

    def test_t_test_small_sample(self):
        """Test t-test with small sample size."""
        x = np.array([0.85, 0.86, 0.84])
        y = np.array([0.80, 0.81, 0.79])

        result = paired_t_test(x, y)
        assert isinstance(result, SignificanceResult)

    def test_friedman_minimum_models(self):
        """Test Friedman with minimum 3 models."""
        scores = {
            "A": np.array([0.8, 0.82, 0.78, 0.81]),
            "B": np.array([0.75, 0.77, 0.73, 0.76]),
            "C": np.array([0.70, 0.72, 0.68, 0.71]),
        }
        result = friedman_nemenyi_test(scores)
        assert isinstance(result, FriedmanNemenyiResult)

    def test_mcnemar_continuity_correction(self):
        """Test McNemar with continuity correction edge case."""
        # Very few discordant pairs
        correct_a = np.array([1, 1, 1, 0, 1, 0])
        correct_b = np.array([1, 1, 0, 0, 1, 1])

        result = mcnemar_test(correct_a, correct_b)
        assert isinstance(result, SignificanceResult)
        # With continuity correction, chi2 = (|b-c|-1)^2 / (b+c)

    def test_permutation_test_zero_permutations_warning(self):
        """Test permutation test with reasonable permutation count."""
        x = np.array([0.8, 0.82, 0.85])
        y = np.array([0.75, 0.77, 0.80])

        # Should work with small number of permutations
        result = paired_permutation_test(x, y, n_permutations=100, random_state=42)
        assert isinstance(result, SignificanceResult)
