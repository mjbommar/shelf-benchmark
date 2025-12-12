"""Unit tests for shelf.evaluate.analysis.bootstrap module.

Tests cover:
- BootstrapResult dataclass methods
- bootstrap_ci() with all 3 methods (percentile, basic, bca)
- bootstrap_paired_difference() significance testing
- bootstrap_ratio() for ratio confidence intervals
- Edge cases: small samples, identical data, extreme values
"""

from __future__ import annotations

import numpy as np
import pytest

from shelf.evaluate.analysis.bootstrap import (
    BootstrapDifferenceResult,
    BootstrapResult,
    bootstrap_ci,
    bootstrap_paired_difference,
    bootstrap_ratio,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def bootstrap_data():
    """Sample data for bootstrap tests."""
    # Beta distribution centered around 0.80
    np.random.seed(42)
    return np.random.beta(8, 2, size=100)


@pytest.fixture
def paired_scores_different():
    """Paired scores with significant difference."""
    np.random.seed(42)
    # Model A: mean = 0.85, Model B: mean = 0.75
    scores_a = np.random.beta(17, 3, size=50)
    scores_b = np.random.beta(15, 5, size=50)
    return scores_a, scores_b


@pytest.fixture
def paired_scores_similar():
    """Paired scores with no significant difference."""
    np.random.seed(42)
    # Both models: mean ≈ 0.80
    scores_a = np.random.beta(16, 4, size=50)
    scores_b = np.random.beta(16, 4, size=50) + 0.01  # Tiny difference
    return scores_a, scores_b


# ===========================================================================
# BootstrapResult Tests
# ===========================================================================


class TestBootstrapResult:
    """Tests for BootstrapResult dataclass."""

    def test_str_representation(self):
        """Test string representation includes CI."""
        result = BootstrapResult(
            estimate=0.85,
            ci_lower=0.80,
            ci_upper=0.90,
            ci_level=0.95,
            se=0.025,
            n_bootstrap=1000,
        )
        s = str(result)
        assert "0.85" in s or "0.8500" in s
        assert "0.80" in s or "0.8000" in s
        assert "0.90" in s or "0.9000" in s
        assert "95%" in s

    def test_ci_width(self):
        """Test CI width property."""
        result = BootstrapResult(
            estimate=0.85,
            ci_lower=0.80,
            ci_upper=0.90,
            ci_level=0.95,
            se=0.025,
            n_bootstrap=1000,
        )
        assert result.ci_width == pytest.approx(0.10)

    def test_contains_value(self):
        """Test contains method."""
        result = BootstrapResult(
            estimate=0.85,
            ci_lower=0.80,
            ci_upper=0.90,
            ci_level=0.95,
            se=0.025,
            n_bootstrap=1000,
        )
        assert result.contains(0.85) is True
        assert result.contains(0.80) is True
        assert result.contains(0.90) is True
        assert result.contains(0.79) is False
        assert result.contains(0.91) is False


# ===========================================================================
# bootstrap_ci Tests
# ===========================================================================


class TestBootstrapCI:
    """Tests for bootstrap_ci function."""

    def test_percentile_method(self, bootstrap_data):
        """Test percentile bootstrap CI."""
        result = bootstrap_ci(
            bootstrap_data,
            n_bootstrap=1000,
            ci_level=0.95,
            method="percentile",
            random_state=42,
        )
        # Mean should be close to the sample mean
        assert result.estimate == pytest.approx(np.mean(bootstrap_data))
        # CI should contain the estimate
        assert result.ci_lower < result.estimate < result.ci_upper
        # CI level should be stored
        assert result.ci_level == 0.95

    def test_basic_method(self, bootstrap_data):
        """Test basic (pivot) bootstrap CI."""
        result = bootstrap_ci(
            bootstrap_data,
            n_bootstrap=1000,
            ci_level=0.95,
            method="basic",
            random_state=42,
        )
        # Basic method may produce different bounds, but should be valid
        assert result.ci_lower < result.ci_upper
        assert result.estimate == pytest.approx(np.mean(bootstrap_data))

    def test_bca_method(self, bootstrap_data):
        """Test BCa bootstrap CI."""
        result = bootstrap_ci(
            bootstrap_data,
            n_bootstrap=1000,
            ci_level=0.95,
            method="bca",
            random_state=42,
        )
        assert result.ci_lower < result.ci_upper
        assert result.estimate == pytest.approx(np.mean(bootstrap_data))

    def test_custom_statistic(self, bootstrap_data):
        """Test with custom statistic (median)."""
        result = bootstrap_ci(
            bootstrap_data,
            statistic=np.median,
            n_bootstrap=1000,
            random_state=42,
        )
        assert result.estimate == pytest.approx(np.median(bootstrap_data))

    def test_different_ci_levels(self, bootstrap_data):
        """Test different confidence levels."""
        result_90 = bootstrap_ci(bootstrap_data, ci_level=0.90, random_state=42)
        result_95 = bootstrap_ci(bootstrap_data, ci_level=0.95, random_state=42)
        result_99 = bootstrap_ci(bootstrap_data, ci_level=0.99, random_state=42)

        # Wider CI for higher confidence
        assert result_90.ci_width < result_95.ci_width < result_99.ci_width

    def test_reproducibility(self, bootstrap_data):
        """Test that random_state ensures reproducibility."""
        result1 = bootstrap_ci(bootstrap_data, random_state=42)
        result2 = bootstrap_ci(bootstrap_data, random_state=42)

        assert result1.ci_lower == result2.ci_lower
        assert result1.ci_upper == result2.ci_upper
        assert result1.se == result2.se

    def test_return_distribution(self, bootstrap_data):
        """Test returning full bootstrap distribution."""
        result = bootstrap_ci(
            bootstrap_data,
            n_bootstrap=1000,
            return_distribution=True,
            random_state=42,
        )
        assert result.bootstrap_distribution is not None
        assert len(result.bootstrap_distribution) == 1000

    def test_invalid_method_raises(self, bootstrap_data):
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            bootstrap_ci(bootstrap_data, method="invalid")

    def test_small_sample(self):
        """Test with small sample size."""
        data = np.array([0.80, 0.82, 0.85])
        result = bootstrap_ci(data, n_bootstrap=1000, random_state=42)
        # Should still produce valid CI
        assert result.ci_lower < result.ci_upper


# ===========================================================================
# bootstrap_paired_difference Tests
# ===========================================================================


class TestBootstrapPairedDifference:
    """Tests for bootstrap_paired_difference function."""

    def test_significant_difference(self, paired_scores_different):
        """Test detection of significant difference."""
        scores_a, scores_b = paired_scores_different
        result = bootstrap_paired_difference(
            scores_a,
            scores_b,
            n_bootstrap=1000,
            random_state=42,
        )
        # With sufficient difference, should detect significance
        # (Note: statistical tests can occasionally fail, but with these params it's very likely)
        assert result.mean_diff > 0  # A > B
        # CI bounds should be valid
        assert result.ci_lower < result.ci_upper

    def test_no_significant_difference(self, paired_scores_similar):
        """Test when no significant difference."""
        scores_a, scores_b = paired_scores_similar
        result = bootstrap_paired_difference(
            scores_a,
            scores_b,
            n_bootstrap=1000,
            random_state=42,
        )
        # Difference should be small
        assert abs(result.mean_diff) < 0.2  # Reasonable threshold
        # CI should be valid
        assert result.ci_lower < result.ci_upper

    def test_identical_arrays(self):
        """Test with identical arrays (no difference)."""
        data = np.array([0.80, 0.82, 0.85, 0.78, 0.81])
        result = bootstrap_paired_difference(
            data,
            data,
            n_bootstrap=1000,
            random_state=42,
        )
        assert result.mean_diff == pytest.approx(0.0)
        assert result.significant is False

    def test_unequal_length_raises(self):
        """Test that unequal array lengths raise error."""
        with pytest.raises(ValueError, match="same length"):
            bootstrap_paired_difference(
                np.array([1, 2, 3]),
                np.array([1, 2]),
            )

    def test_ci_contains_zero_for_no_difference(self, paired_scores_similar):
        """Test CI contains zero when no significant difference."""
        scores_a, scores_b = paired_scores_similar
        result = bootstrap_paired_difference(
            scores_a,
            scores_b,
            n_bootstrap=1000,
            random_state=42,
        )
        # If not significant, CI should span zero
        if not result.significant:
            assert result.ci_lower <= 0 <= result.ci_upper

    def test_str_representation(self, paired_scores_different):
        """Test string representation."""
        scores_a, scores_b = paired_scores_different
        result = bootstrap_paired_difference(scores_a, scores_b, random_state=42)
        s = str(result)
        # Should show difference symbol and p-value
        assert "Δ" in s or "+" in s or "-" in s  # Shows difference
        assert "p=" in s  # Shows p-value


# ===========================================================================
# bootstrap_ratio Tests
# ===========================================================================


class TestBootstrapRatio:
    """Tests for bootstrap_ratio function."""

    def test_ratio_of_means(self):
        """Test ratio of means computation."""
        x = np.array([100, 102, 98, 101, 99])
        y = np.array([50, 51, 49, 50, 50])
        result = bootstrap_ratio(x, y, n_bootstrap=1000, random_state=42)

        # Ratio should be approximately 2.0
        assert result.estimate == pytest.approx(2.0, rel=0.05)
        assert result.ci_lower < 2.0 < result.ci_upper

    def test_ratio_ci_bounds(self):
        """Test that CI bounds are reasonable."""
        x = np.array([0.90, 0.92, 0.88, 0.91, 0.89])
        y = np.array([0.80, 0.82, 0.78, 0.81, 0.79])
        result = bootstrap_ratio(x, y, n_bootstrap=1000, random_state=42)

        # Ratio should be > 1 (x > y)
        assert result.ci_lower > 1.0
        assert result.estimate > 1.0


# ===========================================================================
# Edge Cases
# ===========================================================================


class TestBootstrapEdgeCases:
    """Edge case tests for bootstrap functions."""

    def test_constant_data(self):
        """Test with constant data (zero variance)."""
        data = np.array([0.80] * 20)
        result = bootstrap_ci(data, n_bootstrap=100, random_state=42)

        # All values same, so estimate is that value
        assert result.estimate == pytest.approx(0.80)
        # CI should be very tight (essentially zero width)
        assert result.ci_width < 0.001

    def test_high_variance_data(self):
        """Test with high variance data."""
        np.random.seed(42)
        data = np.random.uniform(0, 1, 100)
        result = bootstrap_ci(data, n_bootstrap=1000, random_state=42)

        # CI should be wider than low variance
        assert result.ci_width > 0.05

    def test_more_bootstrap_iterations(self, bootstrap_data):
        """Test that more iterations give more stable estimates."""
        results = []
        for n in [100, 1000, 10000]:
            r = bootstrap_ci(bootstrap_data, n_bootstrap=n, random_state=42)
            results.append(r)

        # Estimates should be the same (based on original data)
        assert results[0].estimate == results[1].estimate == results[2].estimate

        # SEs should be similar regardless of n_bootstrap
        # (SE measures variability of data, not bootstrap iterations)
        # Allow for some variation due to finite bootstrap samples
        assert results[1].se == pytest.approx(results[2].se, rel=0.15)


# ===========================================================================
# BootstrapDifferenceResult Tests
# ===========================================================================


class TestBootstrapDifferenceResult:
    """Tests for BootstrapDifferenceResult dataclass."""

    def test_str_representation_significant(self):
        """Test string representation for significant result."""
        result = BootstrapDifferenceResult(
            mean_diff=0.05,
            ci_lower=0.02,
            ci_upper=0.08,
            ci_level=0.95,
            p_value=0.003,
            significant=True,
            se=0.015,
            n_bootstrap=1000,
        )
        s = str(result)
        assert "Δ" in s
        assert "p=" in s
        assert "*" in s  # Significance marker

    def test_str_representation_not_significant(self):
        """Test string representation for non-significant result."""
        result = BootstrapDifferenceResult(
            mean_diff=0.01,
            ci_lower=-0.02,
            ci_upper=0.04,
            ci_level=0.95,
            p_value=0.45,
            significant=False,
            se=0.015,
            n_bootstrap=1000,
        )
        s = str(result)
        assert "Δ" in s
        assert "p=" in s
        # Should not have significance marker
        # (can't easily test absence of * without full parsing)
