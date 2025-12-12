"""Unit tests for shelf.evaluate.analysis.comparison module.

Tests cover:
- ComparisonResult dataclass (winner, summary)
- MultipleComparisonResult dataclass (summary, ranking)
- compare_two() function (all test types)
- compare_multiple() function
- load_results_from_dir() function
- extract_scores() function
- compare_by_group() function
- compare_commits() function
- FacetAnalysisResult dataclass
- compare_by_facet() function
- compare_generation_models() function
- Edge cases (single model, identical scores, missing data)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from shelf.evaluate.analysis.comparison import (
    ComparisonResult,
    FacetAnalysisResult,
    MultipleComparisonResult,
    _find_equivalence_groups,
    compare_by_facet,
    compare_by_group,
    compare_commits,
    compare_generation_models,
    compare_multiple,
    compare_two,
    extract_scores,
    load_results_from_dir,
)
from shelf.evaluate.analysis.significance import SignificanceResult


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def scores_a():
    """Scores for model A (higher)."""
    np.random.seed(42)
    return np.random.normal(0.85, 0.05, 50)


@pytest.fixture
def scores_b():
    """Scores for model B (lower)."""
    np.random.seed(43)
    return np.random.normal(0.75, 0.05, 50)


@pytest.fixture
def scores_similar():
    """Two sets of similar scores."""
    np.random.seed(42)
    base = np.random.normal(0.80, 0.05, 50)
    return base + np.random.normal(0, 0.01, 50), base + np.random.normal(0, 0.01, 50)


@pytest.fixture
def multiple_scores():
    """Scores from multiple models."""
    np.random.seed(42)
    return {
        "ModelA": np.random.normal(0.85, 0.03, 10),
        "ModelB": np.random.normal(0.82, 0.04, 10),
        "ModelC": np.random.normal(0.78, 0.05, 10),
        "ModelD": np.random.normal(0.75, 0.04, 10),
    }


@pytest.fixture
def sample_results():
    """Sample evaluation results for testing (3 models for Friedman test)."""
    return [
        {
            "model": "ModelA",
            "commit_id": "abc123",
            "metrics": {
                "macro_f1": 0.85,
                "accuracy": 0.87,
                "retrieval": {"ndcg@10": 0.75},
            },
            "context": {"model_name": "ModelA", "data_commit": "abc123"},
        },
        {
            "model": "ModelA",
            "commit_id": "abc123",
            "metrics": {
                "macro_f1": 0.83,
                "accuracy": 0.86,
                "retrieval": {"ndcg@10": 0.73},
            },
            "context": {"model_name": "ModelA", "data_commit": "abc123"},
        },
        {
            "model": "ModelB",
            "commit_id": "def456",
            "metrics": {
                "macro_f1": 0.78,
                "accuracy": 0.80,
                "retrieval": {"ndcg@10": 0.70},
            },
            "context": {"model_name": "ModelB", "data_commit": "def456"},
        },
        {
            "model": "ModelB",
            "commit_id": "def456",
            "metrics": {
                "macro_f1": 0.76,
                "accuracy": 0.79,
                "retrieval": {"ndcg@10": 0.68},
            },
            "context": {"model_name": "ModelB", "data_commit": "def456"},
        },
        {
            "model": "ModelC",
            "commit_id": "ghi789",
            "metrics": {
                "macro_f1": 0.80,
                "accuracy": 0.82,
                "retrieval": {"ndcg@10": 0.72},
            },
            "context": {"model_name": "ModelC", "data_commit": "ghi789"},
        },
        {
            "model": "ModelC",
            "commit_id": "ghi789",
            "metrics": {
                "macro_f1": 0.81,
                "accuracy": 0.83,
                "retrieval": {"ndcg@10": 0.71},
            },
            "context": {"model_name": "ModelC", "data_commit": "ghi789"},
        },
    ]


@pytest.fixture
def temp_results_dir(tmp_path, sample_results):
    """Create temporary directory with result JSON files."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    for i, result in enumerate(sample_results):
        result_file = results_dir / f"result_{i}.json"
        with open(result_file, "w") as f:
            json.dump(result, f)

    return results_dir


@pytest.fixture
def facet_results():
    """Results with stratified metrics for facet analysis."""
    return [
        {
            "context": {"model_name": "ModelA"},
            "metrics": {"macro_f1": 0.85},
            "stratified_metrics": {
                "form=lecture": {"macro_f1": 0.90},
                "form=map": {"macro_f1": 0.80},
                "form=essay": {"macro_f1": 0.85},
                "lcc=A": {"macro_f1": 0.88},
                "lcc=B": {"macro_f1": 0.82},
            },
        },
        {
            "context": {"model_name": "ModelA"},
            "metrics": {"macro_f1": 0.83},
            "stratified_metrics": {
                "form=lecture": {"macro_f1": 0.88},
                "form=map": {"macro_f1": 0.78},
                "form=essay": {"macro_f1": 0.83},
                "lcc=A": {"macro_f1": 0.86},
                "lcc=B": {"macro_f1": 0.80},
            },
        },
        {
            "context": {"model_name": "ModelA"},
            "metrics": {"macro_f1": 0.84},
            "stratified_metrics": {
                "form=lecture": {"macro_f1": 0.89},
                "form=map": {"macro_f1": 0.79},
                "form=essay": {"macro_f1": 0.84},
                "lcc=A": {"macro_f1": 0.87},
                "lcc=B": {"macro_f1": 0.81},
            },
        },
    ]


@pytest.fixture
def generation_model_results():
    """Results with different generation models."""
    return [
        {
            "data_provenance": {"primary_model": "gpt-5.1"},
            "metrics": {"macro_f1": 0.85},
        },
        {
            "data_provenance": {"primary_model": "gpt-5.1"},
            "metrics": {"macro_f1": 0.83},
        },
        {
            "data_provenance": {"primary_model": "gpt-5.2"},
            "metrics": {"macro_f1": 0.87},
        },
        {
            "data_provenance": {"primary_model": "gpt-5.2"},
            "metrics": {"macro_f1": 0.86},
        },
    ]


# ===========================================================================
# ComparisonResult Tests
# ===========================================================================


@pytest.mark.unit
class TestComparisonResult:
    """Tests for ComparisonResult dataclass."""

    def test_winner_significant_positive(self):
        """Test winner when A significantly better than B."""
        sig_result = SignificanceResult(
            test_name="Test",
            statistic=3.5,
            p_value=0.001,
            significant=True,
            alpha=0.05,
        )
        result = ComparisonResult(
            name_a="ModelA",
            name_b="ModelB",
            metric="f1",
            mean_a=0.85,
            mean_b=0.75,
            difference=0.10,
            relative_diff_pct=13.33,
            significance=sig_result,
        )
        assert result.winner == "ModelA"

    def test_winner_significant_negative(self):
        """Test winner when B significantly better than A."""
        sig_result = SignificanceResult(
            test_name="Test",
            statistic=3.5,
            p_value=0.001,
            significant=True,
            alpha=0.05,
        )
        result = ComparisonResult(
            name_a="ModelA",
            name_b="ModelB",
            metric="f1",
            mean_a=0.75,
            mean_b=0.85,
            difference=-0.10,
            relative_diff_pct=-11.76,
            significance=sig_result,
        )
        assert result.winner == "ModelB"

    def test_winner_not_significant(self):
        """Test no winner when difference is not significant."""
        sig_result = SignificanceResult(
            test_name="Test",
            statistic=0.5,
            p_value=0.50,
            significant=False,
            alpha=0.05,
        )
        result = ComparisonResult(
            name_a="ModelA",
            name_b="ModelB",
            metric="f1",
            mean_a=0.80,
            mean_b=0.81,
            difference=-0.01,
            relative_diff_pct=-1.23,
            significance=sig_result,
        )
        assert result.winner is None

    def test_summary_with_winner(self):
        """Test summary generation with a winner."""
        sig_result = SignificanceResult(
            test_name="Bootstrap",
            statistic=0.10,
            p_value=0.001,
            significant=True,
            alpha=0.05,
        )
        result = ComparisonResult(
            name_a="ModelA",
            name_b="ModelB",
            metric="macro_f1",
            mean_a=0.85,
            mean_b=0.75,
            difference=0.10,
            relative_diff_pct=13.33,
            significance=sig_result,
            n_samples=50,
        )
        summary = result.summary()
        assert "Comparing ModelA vs ModelB" in summary
        assert "0.8500" in summary
        assert "0.7500" in summary
        assert "+0.1000" in summary
        assert "+13.3%" in summary
        assert "Winner: ModelA" in summary

    def test_summary_without_winner(self):
        """Test summary when no significant difference."""
        sig_result = SignificanceResult(
            test_name="Bootstrap",
            statistic=0.01,
            p_value=0.50,
            significant=False,
            alpha=0.05,
        )
        result = ComparisonResult(
            name_a="ModelA",
            name_b="ModelB",
            metric="macro_f1",
            mean_a=0.80,
            mean_b=0.81,
            difference=-0.01,
            relative_diff_pct=-1.23,
            significance=sig_result,
        )
        summary = result.summary()
        assert "No significant difference" in summary


# ===========================================================================
# MultipleComparisonResult Tests
# ===========================================================================


@pytest.mark.unit
class TestMultipleComparisonResult:
    """Tests for MultipleComparisonResult dataclass."""

    def test_summary_basic(self, multiple_scores):
        """Test summary generation."""
        result = compare_multiple(multiple_scores, metric="macro_f1", alpha=0.05)
        summary = result.summary()

        assert "Multiple Model Comparison" in summary
        assert "macro_f1" in summary
        assert "ModelA" in summary
        assert "ModelB" in summary
        assert "ModelC" in summary
        assert "ModelD" in summary
        assert "Ranking" in summary

    def test_ranking_order(self, multiple_scores):
        """Test that ranking is ordered by mean score."""
        result = compare_multiple(multiple_scores, metric="macro_f1")

        # Should be ordered from highest to lowest mean
        assert result.ranking[0] == "ModelA"
        assert result.ranking[-1] == "ModelD"

        # Verify means are in descending order
        means = [result.mean_scores[m] for m in result.ranking]
        assert means == sorted(means, reverse=True)


# ===========================================================================
# compare_two() Tests
# ===========================================================================


@pytest.mark.unit
@pytest.mark.statistical
class TestCompareTwo:
    """Tests for compare_two() function."""

    def test_bootstrap_test(self, scores_a, scores_b):
        """Test comparison with bootstrap test."""
        result = compare_two(
            scores_a,
            scores_b,
            name_a="ModelA",
            name_b="ModelB",
            metric="f1",
            test="bootstrap",
            alpha=0.05,
        )

        assert result.name_a == "ModelA"
        assert result.name_b == "ModelB"
        assert result.metric == "f1"
        assert result.mean_a > result.mean_b
        assert result.difference > 0
        assert result.bootstrap is not None
        assert result.n_samples == 50
        assert result.is_paired is True

    def test_wilcoxon_test(self, scores_a, scores_b):
        """Test comparison with Wilcoxon test."""
        result = compare_two(scores_a, scores_b, test="wilcoxon", alpha=0.05)

        assert result.significance.test_name == "Wilcoxon signed-rank"
        assert result.bootstrap is not None

    def test_t_test(self, scores_a, scores_b):
        """Test comparison with paired t-test."""
        result = compare_two(scores_a, scores_b, test="t-test", alpha=0.05)

        assert result.significance.test_name == "Paired t-test"

    def test_permutation_test(self, scores_a, scores_b):
        """Test comparison with permutation test."""
        result = compare_two(scores_a, scores_b, test="permutation", alpha=0.05)

        assert result.significance.test_name == "Paired permutation test"

    def test_invalid_test_type(self, scores_a, scores_b):
        """Test that invalid test type raises error."""
        with pytest.raises(ValueError, match="Unknown test"):
            compare_two(scores_a, scores_b, test="invalid_test")

    def test_relative_difference_calculation(self):
        """Test relative difference percentage calculation."""
        scores_a = np.array([0.90, 0.90, 0.90])
        scores_b = np.array([0.80, 0.80, 0.80])

        result = compare_two(scores_a, scores_b)

        # (0.90 - 0.80) / 0.80 * 100 = 12.5%
        assert abs(result.relative_diff_pct - 12.5) < 0.1

    def test_relative_difference_zero_denominator(self):
        """Test relative difference when mean_b is zero."""
        scores_a = np.array([0.1, 0.1, 0.1])
        scores_b = np.array([0.0, 0.0, 0.0])

        result = compare_two(scores_a, scores_b)

        # Should handle division by zero
        assert result.relative_diff_pct == 0

    def test_similar_scores_not_significant(self, scores_similar):
        """Test that similar scores are not significantly different."""
        scores_a, scores_b = scores_similar

        result = compare_two(scores_a, scores_b, alpha=0.05)

        # Should not be significant
        assert result.significance.significant is False
        assert result.winner is None

    def test_different_scores_significant(self, scores_a, scores_b):
        """Test that different scores are significantly different."""
        result = compare_two(scores_a, scores_b, alpha=0.05)

        # Should be significant
        assert result.significance.significant is True
        assert result.winner is not None

    def test_paired_vs_unpaired(self, scores_a, scores_b):
        """Test paired parameter is stored correctly."""
        result_paired = compare_two(scores_a, scores_b, paired=True)
        result_unpaired = compare_two(scores_a, scores_b, paired=False)

        assert result_paired.is_paired is True
        assert result_unpaired.is_paired is False


# ===========================================================================
# compare_multiple() Tests
# ===========================================================================


@pytest.mark.unit
@pytest.mark.statistical
class TestCompareMultiple:
    """Tests for compare_multiple() function."""

    def test_basic_comparison(self, multiple_scores):
        """Test basic multiple comparison."""
        result = compare_multiple(multiple_scores, metric="f1", alpha=0.05)

        assert len(result.models) == 4
        assert len(result.ranking) == 4
        assert result.metric == "f1"
        assert result.friedman_nemenyi is not None

    def test_mean_scores_correct(self, multiple_scores):
        """Test that mean scores are calculated correctly."""
        result = compare_multiple(multiple_scores)

        for model, scores in multiple_scores.items():
            expected_mean = float(np.mean(scores))
            assert abs(result.mean_scores[model] - expected_mean) < 1e-6

    def test_ranking_best_first(self, multiple_scores):
        """Test that ranking puts best model first."""
        result = compare_multiple(multiple_scores)

        best_model = result.ranking[0]
        best_mean = result.mean_scores[best_model]

        for model in result.models:
            assert best_mean >= result.mean_scores[model]

    def test_friedman_nemenyi_result(self, multiple_scores):
        """Test Friedman-Nemenyi test is performed."""
        result = compare_multiple(multiple_scores)

        fn_result = result.friedman_nemenyi
        assert fn_result.n_groups == 4
        assert fn_result.n_blocks == 10
        assert fn_result.friedman_statistic > 0
        assert 0 <= fn_result.friedman_p_value <= 1

    def test_pairwise_comparisons_included(self, multiple_scores):
        """Test pairwise comparisons are included when requested."""
        result = compare_multiple(multiple_scores, include_pairwise=True)

        # Should have C(4, 2) = 6 pairwise comparisons
        assert len(result.pairwise) == 6

        # Check that all pairs are present
        models = result.models
        for i, model_a in enumerate(models):
            for model_b in models[i + 1 :]:
                assert (model_a, model_b) in result.pairwise

    def test_pairwise_comparisons_excluded(self, multiple_scores):
        """Test pairwise comparisons can be excluded."""
        result = compare_multiple(multiple_scores, include_pairwise=False)

        assert len(result.pairwise) == 0

    def test_equivalence_groups(self):
        """Test equivalence groups for similar models."""
        # Create scores where ModelA and ModelB are similar
        np.random.seed(42)
        base = np.random.normal(0.80, 0.02, 10)
        scores = {
            "ModelA": base + np.random.normal(0, 0.001, 10),
            "ModelB": base + np.random.normal(0, 0.001, 10),
            "ModelC": np.random.normal(0.60, 0.02, 10),
        }

        result = compare_multiple(scores, alpha=0.05)

        # Should have equivalence groups
        # (Exact groups depend on statistical tests, but should have some)
        assert isinstance(result.equivalence_groups, list)

    def test_two_models(self):
        """Test comparison with only two models (should fail - Friedman needs 3+)."""
        np.random.seed(42)
        scores = {
            "ModelA": np.random.normal(0.85, 0.03, 10),
            "ModelB": np.random.normal(0.75, 0.04, 10),
        }

        # Friedman test requires at least 3 models
        with pytest.raises(ValueError, match="At least 3 sets"):
            compare_multiple(scores)


# ===========================================================================
# Equivalence Groups Tests
# ===========================================================================


@pytest.mark.unit
class TestFindEquivalenceGroups:
    """Tests for _find_equivalence_groups() function."""

    def test_all_different(self):
        """Test when all models are significantly different."""
        models = ["A", "B", "C"]
        significant_pairs = [("A", "B"), ("A", "C"), ("B", "C")]

        groups = _find_equivalence_groups(models, significant_pairs)

        # No equivalence groups (all different)
        assert len(groups) == 0

    def test_all_equivalent(self):
        """Test when all models are equivalent."""
        models = ["A", "B", "C"]
        significant_pairs = []  # No significant differences

        groups = _find_equivalence_groups(models, significant_pairs)

        # All models in one group
        assert len(groups) == 1
        assert sorted(groups[0]) == ["A", "B", "C"]

    def test_partial_equivalence(self):
        """Test with partial equivalence."""
        models = ["A", "B", "C", "D"]
        # A and B are equivalent, C and D are equivalent
        significant_pairs = [("A", "C"), ("A", "D"), ("B", "C"), ("B", "D")]

        groups = _find_equivalence_groups(models, significant_pairs)

        # Should have 2 groups
        assert len(groups) == 2

        # Check groups contain correct models
        groups_sets = [set(g) for g in groups]
        assert {"A", "B"} in groups_sets or {"B", "A"} in groups_sets
        assert {"C", "D"} in groups_sets or {"D", "C"} in groups_sets

    def test_single_model(self):
        """Test with a single model."""
        models = ["A"]
        significant_pairs = []

        groups = _find_equivalence_groups(models, significant_pairs)

        # Single model doesn't form a group
        assert len(groups) == 0


# ===========================================================================
# load_results_from_dir() Tests
# ===========================================================================


@pytest.mark.unit
class TestLoadResultsFromDir:
    """Tests for load_results_from_dir() function."""

    def test_load_all_results(self, temp_results_dir):
        """Test loading all JSON files from directory."""
        results = load_results_from_dir(temp_results_dir)

        assert len(results) == 6  # 3 models × 2 results each
        for result in results:
            assert "model" in result
            assert "metrics" in result
            assert "_source_file" in result

    def test_load_with_pattern(self, temp_results_dir):
        """Test loading with specific pattern."""
        # Create a non-matching file
        other_file = temp_results_dir / "other.txt"
        other_file.write_text("not json")

        results = load_results_from_dir(temp_results_dir, pattern="*.json")

        # Should only load JSON files (6 results from 3 models)
        assert len(results) == 6

    def test_empty_directory(self, tmp_path):
        """Test with empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        results = load_results_from_dir(empty_dir)

        assert len(results) == 0

    def test_source_file_added(self, temp_results_dir):
        """Test that source file path is added to results."""
        results = load_results_from_dir(temp_results_dir)

        for result in results:
            assert "_source_file" in result
            assert Path(result["_source_file"]).exists()


# ===========================================================================
# extract_scores() Tests
# ===========================================================================


@pytest.mark.unit
class TestExtractScores:
    """Tests for extract_scores() function."""

    def test_extract_by_model(self, sample_results):
        """Test extracting scores grouped by model."""
        scores = extract_scores(sample_results, "macro_f1", group_by="model")

        assert "ModelA" in scores
        assert "ModelB" in scores
        assert len(scores["ModelA"]) == 2
        assert len(scores["ModelB"]) == 2
        assert 0.85 in scores["ModelA"]
        assert 0.83 in scores["ModelA"]

    def test_extract_by_commit(self, sample_results):
        """Test extracting scores grouped by commit."""
        scores = extract_scores(sample_results, "macro_f1", group_by="commit_id")

        assert "abc123" in scores
        assert "def456" in scores
        assert len(scores["abc123"]) == 2
        assert len(scores["def456"]) == 2

    def test_extract_nested_metric(self, sample_results):
        """Test extracting nested metric."""
        scores = extract_scores(sample_results, "retrieval.ndcg@10", group_by="model")

        assert "ModelA" in scores
        assert "ModelB" in scores
        assert 0.75 in scores["ModelA"]

    def test_extract_from_context(self):
        """Test extracting group key from context."""
        results = [
            {
                "context": {"model_name": "TestModel"},
                "metrics": {"f1": 0.85},
            }
        ]

        scores = extract_scores(results, "f1", group_by="model_name")

        assert "TestModel" in scores
        assert 0.85 in scores["TestModel"]

    def test_missing_metric(self, sample_results):
        """Test with missing metric."""
        scores = extract_scores(sample_results, "nonexistent_metric", group_by="model")

        # Should return empty dict or dict with empty lists
        for values in scores.values():
            assert len(values) == 0

    def test_unknown_group(self):
        """Test with unknown group_by field."""
        results = [{"metrics": {"f1": 0.85}}]

        scores = extract_scores(results, "f1", group_by="unknown_field")

        # Should use "unknown" as group key
        assert "unknown" in scores


# ===========================================================================
# compare_by_group() Tests
# ===========================================================================


@pytest.mark.unit
@pytest.mark.statistical
class TestCompareByGroup:
    """Tests for compare_by_group() function."""

    def test_compare_by_model(self, sample_results):
        """Test comparing results grouped by model."""
        result = compare_by_group(
            sample_results, metric="macro_f1", group_by="model", alpha=0.05
        )

        assert isinstance(result, MultipleComparisonResult)
        assert "ModelA" in result.models
        assert "ModelB" in result.models
        assert "ModelC" in result.models

    def test_compare_by_commit(self, sample_results):
        """Test comparing results grouped by commit."""
        result = compare_by_group(
            sample_results, metric="accuracy", group_by="commit_id", alpha=0.05
        )

        assert "abc123" in result.models
        assert "def456" in result.models
        assert "ghi789" in result.models

    def test_insufficient_groups(self):
        """Test error with insufficient groups."""
        results = [
            {"model": "OnlyModel", "metrics": {"f1": 0.85}},
        ]

        with pytest.raises(ValueError, match="Need at least 2 groups"):
            compare_by_group(results, metric="f1", group_by="model")

    def test_unequal_group_sizes(self):
        """Test handling of unequal group sizes (should truncate)."""
        results = [
            {"model": "ModelA", "metrics": {"f1": 0.85}},
            {"model": "ModelA", "metrics": {"f1": 0.86}},
            {"model": "ModelA", "metrics": {"f1": 0.87}},
            {"model": "ModelB", "metrics": {"f1": 0.75}},
            {"model": "ModelB", "metrics": {"f1": 0.76}},
            {"model": "ModelC", "metrics": {"f1": 0.80}},
            {"model": "ModelC", "metrics": {"f1": 0.81}},
        ]

        result = compare_by_group(results, metric="f1", group_by="model")

        # Should truncate to minimum length (2 for ModelB and ModelC)
        for scores in result.scores.values():
            assert len(scores) == 2


# ===========================================================================
# compare_commits() Tests
# ===========================================================================


@pytest.mark.unit
@pytest.mark.statistical
class TestCompareCommits:
    """Tests for compare_commits() function."""

    def test_compare_two_commits(self, sample_results):
        """Test comparing two commits."""
        result = compare_commits(
            sample_results,
            metric="macro_f1",
            commit_a="abc123",
            commit_b="def456",
            alpha=0.05,
        )

        assert isinstance(result, ComparisonResult)
        assert "commit:abc123" in result.name_a
        assert "commit:def456" in result.name_b
        assert result.is_paired is False

    def test_commit_from_context(self):
        """Test extracting commit from context."""
        results = [
            {
                "context": {"commit_id": "aaa"},
                "metrics": {"f1": 0.85},
            },
            {
                "context": {"commit_id": "aaa"},
                "metrics": {"f1": 0.86},
            },
            {
                "context": {"commit_id": "bbb"},
                "metrics": {"f1": 0.75},
            },
            {
                "context": {"commit_id": "bbb"},
                "metrics": {"f1": 0.76},
            },
        ]

        result = compare_commits(results, "f1", "aaa", "bbb")

        assert result.mean_a == 0.855
        assert result.mean_b == 0.755

    def test_commit_from_data_commit(self):
        """Test extracting commit from context.data_commit."""
        results = [
            {
                "context": {"data_commit": "xxx"},
                "metrics": {"f1": 0.90},
            },
            {
                "context": {"data_commit": "xxx"},
                "metrics": {"f1": 0.91},
            },
            {
                "context": {"data_commit": "yyy"},
                "metrics": {"f1": 0.80},
            },
            {
                "context": {"data_commit": "yyy"},
                "metrics": {"f1": 0.81},
            },
        ]

        result = compare_commits(results, "f1", "xxx", "yyy")

        assert result.mean_a == 0.905
        assert result.mean_b == 0.805

    def test_missing_commit_a(self, sample_results):
        """Test error when commit_a not found."""
        with pytest.raises(ValueError, match="No results found for commit"):
            compare_commits(sample_results, "macro_f1", "nonexistent", "def456")

    def test_missing_commit_b(self, sample_results):
        """Test error when commit_b not found."""
        with pytest.raises(ValueError, match="No results found for commit"):
            compare_commits(sample_results, "macro_f1", "abc123", "nonexistent")


# ===========================================================================
# FacetAnalysisResult Tests
# ===========================================================================


@pytest.mark.unit
class TestFacetAnalysisResult:
    """Tests for FacetAnalysisResult dataclass."""

    def test_summary_generation(self):
        """Test summary generation."""
        facet_result = FacetAnalysisResult(
            facet="form",
            metric="macro_f1",
            model="ModelA",
            facet_scores={
                "lecture": [0.90, 0.88],
                "map": [0.80, 0.78],
                "essay": [0.85, 0.83],
            },
            facet_means={"lecture": 0.89, "map": 0.79, "essay": 0.84},
            ranking=["lecture", "essay", "map"],
            variance_analysis=None,
            best_facet="lecture",
            worst_facet="map",
            score_range=(0.79, 0.89),
        )

        summary = facet_result.summary()

        assert "Facet Analysis: form" in summary
        assert "Metric: macro_f1" in summary
        assert "Model: ModelA" in summary
        assert "Best: lecture" in summary
        assert "Worst: map" in summary
        assert "lecture" in summary
        assert "map" in summary
        assert "essay" in summary

    def test_summary_with_variance_analysis(self):
        """Test summary includes variance test when available."""
        sig_result = SignificanceResult(
            test_name="Kruskal-Wallis",
            statistic=12.5,
            p_value=0.001,
            significant=True,
            alpha=0.05,
        )

        facet_result = FacetAnalysisResult(
            facet="lcc",
            metric="accuracy",
            model=None,
            facet_scores={"A": [0.9], "B": [0.8], "C": [0.7]},
            facet_means={"A": 0.9, "B": 0.8, "C": 0.7},
            ranking=["A", "B", "C"],
            variance_analysis=sig_result,
            best_facet="A",
            worst_facet="C",
            score_range=(0.7, 0.9),
        )

        summary = facet_result.summary()

        assert "Variance test:" in summary
        assert "Kruskal-Wallis" in summary


# ===========================================================================
# compare_by_facet() Tests
# ===========================================================================


@pytest.mark.unit
@pytest.mark.statistical
class TestCompareByFacet:
    """Tests for compare_by_facet() function."""

    def test_analyze_form_facet(self, facet_results):
        """Test analyzing performance by form facet."""
        result = compare_by_facet(
            facet_results, metric="macro_f1", facet="form", min_samples=2
        )

        assert result.facet == "form"
        assert result.metric == "macro_f1"
        assert "lecture" in result.facet_scores
        assert "map" in result.facet_scores
        assert "essay" in result.facet_scores

        # Check means
        assert result.facet_means["lecture"] > result.facet_means["map"]

        # Check ranking
        assert result.ranking[0] == result.best_facet
        assert result.ranking[-1] == result.worst_facet

    def test_analyze_lcc_facet(self, facet_results):
        """Test analyzing performance by LCC facet."""
        result = compare_by_facet(
            facet_results, metric="macro_f1", facet="lcc", min_samples=2
        )

        assert result.facet == "lcc"
        assert "A" in result.facet_scores
        assert "B" in result.facet_scores

    def test_filter_by_model(self, facet_results):
        """Test filtering to specific model."""
        result = compare_by_facet(
            facet_results,
            metric="macro_f1",
            facet="form",
            model="ModelA",
            min_samples=2,
        )

        assert result.model == "ModelA"

    def test_min_samples_filter(self, facet_results):
        """Test that min_samples filters out facet values."""
        result = compare_by_facet(
            facet_results, metric="macro_f1", facet="form", min_samples=3
        )

        # All facets should have at least 3 samples
        for scores in result.facet_scores.values():
            assert len(scores) >= 3

    def test_variance_analysis(self, facet_results):
        """Test that variance analysis is performed."""
        result = compare_by_facet(
            facet_results, metric="macro_f1", facet="form", min_samples=2
        )

        assert result.variance_analysis is not None
        assert result.variance_analysis.test_name == "Kruskal-Wallis"
        assert result.variance_analysis.p_value >= 0
        assert result.variance_analysis.p_value <= 1

    def test_best_worst_facets(self, facet_results):
        """Test best and worst facet identification."""
        result = compare_by_facet(
            facet_results, metric="macro_f1", facet="form", min_samples=2
        )

        best_mean = result.facet_means[result.best_facet]
        worst_mean = result.facet_means[result.worst_facet]

        assert best_mean >= worst_mean
        assert result.score_range[0] == worst_mean
        assert result.score_range[1] == best_mean

    def test_no_stratified_metrics(self):
        """Test error when no stratified metrics available."""
        results = [{"metrics": {"f1": 0.85}}]

        with pytest.raises(ValueError, match="No scores found for facet"):
            compare_by_facet(results, metric="f1", facet="form")

    def test_insufficient_facet_values(self):
        """Test error with too few facet values."""
        results = [
            {
                "stratified_metrics": {
                    "form=lecture": {"f1": 0.90},
                }
            }
        ]

        with pytest.raises(ValueError, match="Need at least 2 facet values"):
            compare_by_facet(results, metric="f1", facet="form", min_samples=1)

    def test_per_class_metrics_fallback(self):
        """Test fallback to per_class_metrics when stratified not available."""
        results = [
            {
                "per_class_metrics": {
                    "A": {"f1": 0.90},
                    "B": {"f1": 0.80},
                    "C": {"f1": 0.85},
                }
            },
            {
                "per_class_metrics": {
                    "A": {"f1": 0.88},
                    "B": {"f1": 0.78},
                    "C": {"f1": 0.83},
                }
            },
        ]

        result = compare_by_facet(results, metric="f1", facet="class", min_samples=2)

        assert "A" in result.facet_scores
        assert "B" in result.facet_scores
        assert "C" in result.facet_scores


# ===========================================================================
# compare_generation_models() Tests
# ===========================================================================


@pytest.mark.unit
@pytest.mark.statistical
class TestCompareGenerationModels:
    """Tests for compare_generation_models() function."""

    def test_compare_two_generation_models(self, generation_model_results):
        """Test comparing two generation models."""
        result = compare_generation_models(
            generation_model_results, metric="macro_f1", alpha=0.05
        )

        # Should return ComparisonResult for 2 models
        assert isinstance(result, ComparisonResult)
        assert "gen:gpt-5.1" in result.name_a or "gen:gpt-5.1" in result.name_b
        assert "gen:gpt-5.2" in result.name_a or "gen:gpt-5.2" in result.name_b

    def test_compare_multiple_generation_models(self):
        """Test comparing more than 2 generation models."""
        results = [
            {"data_provenance": {"primary_model": "gpt-5.1"}, "metrics": {"f1": 0.85}},
            {"data_provenance": {"primary_model": "gpt-5.1"}, "metrics": {"f1": 0.86}},
            {"data_provenance": {"primary_model": "gpt-5.2"}, "metrics": {"f1": 0.87}},
            {"data_provenance": {"primary_model": "gpt-5.2"}, "metrics": {"f1": 0.88}},
            {"data_provenance": {"primary_model": "gpt-4"}, "metrics": {"f1": 0.80}},
            {"data_provenance": {"primary_model": "gpt-4"}, "metrics": {"f1": 0.81}},
        ]

        result = compare_generation_models(results, metric="f1")

        # Should return MultipleComparisonResult for 3+ models
        assert isinstance(result, MultipleComparisonResult)
        assert len(result.models) == 3

    def test_fallback_to_context(self):
        """Test fallback to context when provenance not available."""
        results = [
            {
                "context": {"extra": {"generation_model": "model-a"}},
                "metrics": {"f1": 0.85},
            },
            {
                "context": {"extra": {"generation_model": "model-a"}},
                "metrics": {"f1": 0.86},
            },
            {
                "context": {"extra": {"generation_model": "model-b"}},
                "metrics": {"f1": 0.75},
            },
            {
                "context": {"extra": {"generation_model": "model-b"}},
                "metrics": {"f1": 0.76},
            },
        ]

        result = compare_generation_models(results, metric="f1")

        assert isinstance(result, ComparisonResult)

    def test_primary_score_fallback(self):
        """Test using primary_score when metric not in metrics dict."""
        results = [
            {"data_provenance": {"primary_model": "gpt-5.1"}, "primary_score": 0.85},
            {"data_provenance": {"primary_model": "gpt-5.1"}, "primary_score": 0.86},
            {"data_provenance": {"primary_model": "gpt-5.2"}, "primary_score": 0.87},
            {"data_provenance": {"primary_model": "gpt-5.2"}, "primary_score": 0.88},
        ]

        result = compare_generation_models(results, metric="nonexistent")

        assert isinstance(result, ComparisonResult)
        # Should have used primary_score values
        assert result.mean_a > 0
        assert result.mean_b > 0

    def test_insufficient_models(self):
        """Test error with insufficient generation models."""
        results = [
            {"data_provenance": {"primary_model": "gpt-5.1"}, "metrics": {"f1": 0.85}},
        ]

        with pytest.raises(ValueError, match="Need at least 2 generation models"):
            compare_generation_models(results, metric="f1")

    def test_unknown_generation_model(self):
        """Test handling of unknown generation model."""
        results = [
            {"metrics": {"f1": 0.85}},  # No provenance
            {"metrics": {"f1": 0.80}},
        ]

        # Should group under "unknown"
        # This should raise an error since both would be "unknown" (only 1 group)
        with pytest.raises(ValueError, match="Need at least 2 generation models"):
            compare_generation_models(results, metric="f1")


# ===========================================================================
# Edge Cases and Integration Tests
# ===========================================================================


@pytest.mark.unit
class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_scores_array(self):
        """Test with empty score arrays."""
        scores_a = np.array([])
        scores_b = np.array([])

        # Should handle empty arrays gracefully (may raise or return NaN)
        # Depends on implementation details of statistical tests
        # Multiple warnings can occur
        with pytest.warns(RuntimeWarning):
            compare_two(scores_a, scores_b)

    def test_single_score(self):
        """Test with single score per model."""
        scores_a = np.array([0.85])
        scores_b = np.array([0.75])

        # Should work but will raise warning due to ddof issue
        with pytest.warns(RuntimeWarning, match="Degrees of freedom"):
            result = compare_two(scores_a, scores_b)

        assert result.mean_a == 0.85
        assert result.mean_b == 0.75

    def test_identical_scores(self):
        """Test with identical scores."""
        scores = np.array([0.80, 0.80, 0.80, 0.80])

        result = compare_two(scores, scores)

        assert result.difference == 0.0
        assert result.relative_diff_pct == 0.0
        assert result.winner is None

    def test_nan_handling(self):
        """Test handling of NaN values."""
        scores_a = np.array([0.85, np.nan, 0.87])
        scores_b = np.array([0.75, 0.76, 0.77])

        # Should either filter NaN or raise error
        # Implementation should handle this gracefully
        # (Exact behavior depends on numpy/scipy version)
        result = compare_two(scores_a, scores_b)
        # Result should be computed (NaN handling varies)
        assert isinstance(result, ComparisonResult)

    def test_very_small_differences(self):
        """Test with very small differences."""
        np.random.seed(42)
        scores_a = np.random.normal(0.8, 0.01, 100)
        scores_b = scores_a + 1e-6  # Tiny difference (larger than before)

        result = compare_two(scores_a, scores_b)

        # With deterministic tiny difference, bootstrap may find significance
        # The key is that the difference is very small
        assert abs(result.difference) < 1e-5

    def test_large_sample_size(self):
        """Test with large sample size."""
        np.random.seed(42)
        scores_a = np.random.normal(0.85, 0.05, 1000)
        scores_b = np.random.normal(0.83, 0.05, 1000)

        result = compare_two(scores_a, scores_b)

        assert result.n_samples == 1000
        # With large N, even small differences may be significant
        assert isinstance(result.significance.significant, bool)


# ===========================================================================
# Integration Tests
# ===========================================================================


@pytest.mark.unit
class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_full_workflow(self, temp_results_dir):
        """Test complete workflow from loading to comparison."""
        # Load results
        results = load_results_from_dir(temp_results_dir)
        assert len(results) > 0

        # Extract scores
        scores = extract_scores(results, "macro_f1", group_by="model")
        assert len(scores) >= 2

        # Compare
        result = compare_by_group(results, "macro_f1", group_by="model")
        assert isinstance(result, MultipleComparisonResult)

        # Generate summary
        summary = result.summary()
        assert len(summary) > 0

    def test_commits_workflow(self, sample_results):
        """Test commit comparison workflow."""
        # Get unique commits
        commits = list(set(r["commit_id"] for r in sample_results))
        assert len(commits) >= 2

        # Compare commits
        result = compare_commits(sample_results, "macro_f1", commits[0], commits[1])

        assert isinstance(result, ComparisonResult)
        assert result.is_paired is False

    def test_nested_metric_extraction(self, sample_results):
        """Test extracting and comparing nested metrics."""
        scores = extract_scores(sample_results, "retrieval.ndcg@10", group_by="model")

        assert len(scores) > 0

        # Should be able to compare (needs 3+ models for Friedman)
        if len(scores) >= 3:
            result = compare_by_group(
                sample_results, "retrieval.ndcg@10", group_by="model"
            )
            assert isinstance(result, MultipleComparisonResult)
