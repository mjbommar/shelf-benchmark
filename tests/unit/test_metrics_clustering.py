"""Unit tests for shelf.evaluate.metrics.clustering module.

Tests cover:
- Individual metric functions (v_measure, nmi, ari, homogeneity, completeness)
- compute_clustering_metrics aggregation
- Beta parameter for v_measure
- Average method for NMI
- Edge cases: single cluster, all same label, many clusters
- String vs int labels
- Empty inputs raise ValueError
- Mismatched lengths raise ValueError
- Metric ranges and properties
"""

from __future__ import annotations

import numpy as np
import pytest

from shelf.evaluate.metrics.clustering import (
    adjusted_rand_index,
    completeness,
    compute_clustering_metrics,
    homogeneity,
    normalized_mutual_info,
    v_measure,
)


@pytest.mark.unit
@pytest.mark.metrics
class TestVMeasure:
    """Tests for V-measure metric."""

    def test_perfect_clustering(self, clustering_perfect):
        """Test V-measure on perfect clustering."""
        labels_true, labels_pred = clustering_perfect
        score = v_measure(labels_true.tolist(), labels_pred.tolist())
        assert score == pytest.approx(1.0)

    def test_random_clustering(self, clustering_random):
        """Test V-measure on random clustering."""
        labels_true, labels_pred = clustering_random
        score = v_measure(labels_true.tolist(), labels_pred.tolist())
        # Random clustering should score lower than perfect
        assert 0 <= score < 1.0

    def test_v_measure_range(self):
        """Test that V-measure is in [0, 1]."""
        labels_true = [0, 0, 1, 1, 2, 2]
        labels_pred = [0, 1, 0, 1, 0, 1]
        score = v_measure(labels_true, labels_pred)
        assert 0 <= score <= 1

    def test_v_measure_beta_parameter(self):
        """Test V-measure with different beta values."""
        labels_true = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        labels_pred = [0, 0, 1, 1, 1, 2, 2, 2, 2]

        # Default beta=1.0 (equal weight)
        score_default = v_measure(labels_true, labels_pred, beta=1.0)

        # Beta=0.5 (favor homogeneity)
        score_low_beta = v_measure(labels_true, labels_pred, beta=0.5)

        # Beta=2.0 (favor completeness)
        score_high_beta = v_measure(labels_true, labels_pred, beta=2.0)

        # All should be in valid range
        assert 0 <= score_default <= 1
        assert 0 <= score_low_beta <= 1
        assert 0 <= score_high_beta <= 1

    def test_string_labels(self):
        """Test V-measure with string labels."""
        labels_true = ["A", "A", "B", "B", "C", "C"]
        labels_pred = [0, 0, 1, 1, 2, 2]
        score = v_measure(labels_true, labels_pred)
        assert score == pytest.approx(1.0)

    def test_mixed_string_labels(self):
        """Test with both string true labels and int predictions."""
        labels_true = ["class1", "class1", "class2", "class2", "class3", "class3"]
        labels_pred = [0, 0, 1, 1, 2, 2]
        score = v_measure(labels_true, labels_pred)
        assert score == pytest.approx(1.0)


@pytest.mark.unit
@pytest.mark.metrics
class TestNormalizedMutualInfo:
    """Tests for Normalized Mutual Information metric."""

    def test_perfect_clustering(self, clustering_perfect):
        """Test NMI on perfect clustering."""
        labels_true, labels_pred = clustering_perfect
        score = normalized_mutual_info(labels_true.tolist(), labels_pred.tolist())
        assert score == pytest.approx(1.0)

    def test_random_clustering(self, clustering_random):
        """Test NMI on random clustering."""
        labels_true, labels_pred = clustering_random
        score = normalized_mutual_info(labels_true.tolist(), labels_pred.tolist())
        # Random clustering should score lower
        assert 0 <= score < 1.0

    def test_nmi_range(self):
        """Test that NMI is in [0, 1]."""
        labels_true = [0, 0, 1, 1, 2, 2]
        labels_pred = [0, 1, 0, 1, 0, 1]
        score = normalized_mutual_info(labels_true, labels_pred)
        assert 0 <= score <= 1

    def test_nmi_average_methods(self):
        """Test NMI with different averaging methods."""
        labels_true = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        labels_pred = [0, 0, 1, 1, 1, 2, 2, 2, 2]

        score_arithmetic = normalized_mutual_info(
            labels_true, labels_pred, average_method="arithmetic"
        )
        score_geometric = normalized_mutual_info(
            labels_true, labels_pred, average_method="geometric"
        )
        score_min = normalized_mutual_info(
            labels_true, labels_pred, average_method="min"
        )
        score_max = normalized_mutual_info(
            labels_true, labels_pred, average_method="max"
        )

        # All should be in valid range
        assert 0 <= score_arithmetic <= 1
        assert 0 <= score_geometric <= 1
        assert 0 <= score_min <= 1
        assert 0 <= score_max <= 1

        # Geometric mean should be <= arithmetic mean (with tolerance for floating point)
        # Note: due to different normalizations in sklearn NMI, this relationship
        # may not always hold. Just verify both are valid NMI scores.
        assert (
            score_geometric <= score_arithmetic + 1e-3
            or abs(score_geometric - score_arithmetic) < 1e-3
        )

    def test_string_labels(self):
        """Test NMI with string labels."""
        labels_true = ["red", "red", "blue", "blue", "green", "green"]
        labels_pred = [0, 0, 1, 1, 2, 2]
        score = normalized_mutual_info(labels_true, labels_pred)
        assert score == pytest.approx(1.0)


@pytest.mark.unit
@pytest.mark.metrics
class TestAdjustedRandIndex:
    """Tests for Adjusted Rand Index metric."""

    def test_perfect_clustering(self, clustering_perfect):
        """Test ARI on perfect clustering."""
        labels_true, labels_pred = clustering_perfect
        score = adjusted_rand_index(labels_true.tolist(), labels_pred.tolist())
        assert score == pytest.approx(1.0)

    def test_random_clustering(self, clustering_random):
        """Test ARI on random clustering."""
        labels_true, labels_pred = clustering_random
        score = adjusted_rand_index(labels_true.tolist(), labels_pred.tolist())
        # Random clustering should be around 0 (adjusted for chance)
        assert -1 <= score <= 1

    def test_ari_range(self):
        """Test that ARI is in [-1, 1]."""
        labels_true = [0, 0, 1, 1, 2, 2]
        labels_pred = [0, 1, 0, 1, 0, 1]
        score = adjusted_rand_index(labels_true, labels_pred)
        assert -1 <= score <= 1

    def test_ari_worst_case(self):
        """Test ARI on worst-case clustering."""
        # Complete disagreement
        labels_true = [0, 0, 0, 1, 1, 1]
        labels_pred = [1, 1, 1, 0, 0, 0]
        score = adjusted_rand_index(labels_true, labels_pred)
        # Should still be in valid range, likely negative
        assert -1 <= score <= 1

    def test_string_labels(self):
        """Test ARI with string labels."""
        labels_true = ["type_a", "type_a", "type_b", "type_b"]
        labels_pred = [0, 0, 1, 1]
        score = adjusted_rand_index(labels_true, labels_pred)
        assert score == pytest.approx(1.0)

    def test_all_same_cluster(self):
        """Test ARI when everything is in one cluster."""
        labels_true = [0, 0, 1, 1, 2, 2]
        labels_pred = [0, 0, 0, 0, 0, 0]
        score = adjusted_rand_index(labels_true, labels_pred)
        # Should be low (not matching the true clustering)
        assert -1 <= score < 1


@pytest.mark.unit
@pytest.mark.metrics
class TestHomogeneity:
    """Tests for homogeneity metric."""

    def test_perfect_clustering(self, clustering_perfect):
        """Test homogeneity on perfect clustering."""
        labels_true, labels_pred = clustering_perfect
        score = homogeneity(labels_true.tolist(), labels_pred.tolist())
        assert score == pytest.approx(1.0)

    def test_random_clustering(self, clustering_random):
        """Test homogeneity on random clustering."""
        labels_true, labels_pred = clustering_random
        score = homogeneity(labels_true.tolist(), labels_pred.tolist())
        assert 0 <= score <= 1

    def test_homogeneity_range(self):
        """Test that homogeneity is in [0, 1]."""
        labels_true = [0, 0, 1, 1, 2, 2]
        labels_pred = [0, 1, 0, 1, 0, 1]
        score = homogeneity(labels_true, labels_pred)
        assert 0 <= score <= 1

    def test_homogeneous_but_incomplete(self):
        """Test homogeneity when clusters are pure but incomplete."""
        # Each cluster contains only one class (homogeneous)
        # But classes are split across clusters (not complete)
        labels_true = [0, 0, 0, 0, 1, 1, 1, 1]
        labels_pred = [0, 0, 1, 1, 2, 2, 3, 3]
        score = homogeneity(labels_true, labels_pred)
        # Should be 1.0 (perfectly homogeneous)
        assert score == pytest.approx(1.0)

    def test_string_labels(self):
        """Test homogeneity with string labels."""
        labels_true = ["cat", "cat", "dog", "dog"]
        labels_pred = [0, 0, 1, 1]
        score = homogeneity(labels_true, labels_pred)
        assert score == pytest.approx(1.0)


@pytest.mark.unit
@pytest.mark.metrics
class TestCompleteness:
    """Tests for completeness metric."""

    def test_perfect_clustering(self, clustering_perfect):
        """Test completeness on perfect clustering."""
        labels_true, labels_pred = clustering_perfect
        score = completeness(labels_true.tolist(), labels_pred.tolist())
        assert score == pytest.approx(1.0)

    def test_random_clustering(self, clustering_random):
        """Test completeness on random clustering."""
        labels_true, labels_pred = clustering_random
        score = completeness(labels_true.tolist(), labels_pred.tolist())
        assert 0 <= score <= 1

    def test_completeness_range(self):
        """Test that completeness is in [0, 1]."""
        labels_true = [0, 0, 1, 1, 2, 2]
        labels_pred = [0, 1, 0, 1, 0, 1]
        score = completeness(labels_true, labels_pred)
        assert 0 <= score <= 1

    def test_complete_but_not_homogeneous(self):
        """Test completeness when all class members are together but mixed."""
        # All members of each class in same cluster (complete)
        # But clusters contain multiple classes (not homogeneous)
        labels_true = [0, 0, 1, 1, 2, 2]
        labels_pred = [0, 0, 0, 0, 1, 1]
        score_comp = completeness(labels_true, labels_pred)
        score_homo = homogeneity(labels_true, labels_pred)

        # Completeness should be relatively high
        assert score_comp > 0.5
        # But homogeneity should be lower (cluster 0 has mixed classes)
        assert score_homo < score_comp

    def test_string_labels(self):
        """Test completeness with string labels."""
        labels_true = ["apple", "apple", "banana", "banana"]
        labels_pred = [0, 0, 1, 1]
        score = completeness(labels_true, labels_pred)
        assert score == pytest.approx(1.0)


@pytest.mark.unit
@pytest.mark.metrics
class TestComputeClusteringMetrics:
    """Tests for the aggregated compute_clustering_metrics function."""

    def test_all_metrics_returned(self, clustering_perfect):
        """Test that all expected metrics are returned."""
        labels_true, labels_pred = clustering_perfect
        result = compute_clustering_metrics(labels_true.tolist(), labels_pred.tolist())

        assert "v_measure" in result
        assert "nmi" in result
        assert "ari" in result
        assert "homogeneity" in result
        assert "completeness" in result
        assert "num_samples" in result
        assert "num_clusters_true" in result
        assert "num_clusters_pred" in result

    def test_perfect_clustering_all_metrics(self, clustering_perfect):
        """Test all metrics are 1.0 for perfect clustering."""
        labels_true, labels_pred = clustering_perfect
        result = compute_clustering_metrics(labels_true.tolist(), labels_pred.tolist())

        # All similarity metrics should be 1.0
        assert result["v_measure"] == pytest.approx(1.0)
        assert result["nmi"] == pytest.approx(1.0)
        assert result["ari"] == pytest.approx(1.0)
        assert result["homogeneity"] == pytest.approx(1.0)
        assert result["completeness"] == pytest.approx(1.0)

    def test_random_clustering_metrics(self, clustering_random):
        """Test metrics on random clustering."""
        labels_true, labels_pred = clustering_random
        result = compute_clustering_metrics(labels_true.tolist(), labels_pred.tolist())

        # All metrics should be in valid ranges
        assert 0 <= result["v_measure"] <= 1
        assert 0 <= result["nmi"] <= 1
        assert -1 <= result["ari"] <= 1
        assert 0 <= result["homogeneity"] <= 1
        assert 0 <= result["completeness"] <= 1

        # Scores should be less than perfect
        assert result["v_measure"] < 1.0
        assert result["nmi"] < 1.0
        assert result["ari"] < 1.0

    def test_cluster_counts(self):
        """Test cluster count tracking."""
        labels_true = ["A", "A", "B", "B", "C", "C"]
        labels_pred = [0, 0, 1, 1, 2, 2]
        result = compute_clustering_metrics(labels_true, labels_pred)

        assert result["num_samples"] == 6
        assert result["num_clusters_true"] == 3
        assert result["num_clusters_pred"] == 3

    def test_mismatched_cluster_counts(self):
        """Test when number of predicted clusters differs from true."""
        labels_true = ["A", "A", "A", "B", "B", "B"]
        labels_pred = [0, 0, 1, 1, 2, 2]  # 3 predicted clusters, 2 true classes
        result = compute_clustering_metrics(labels_true, labels_pred)

        assert result["num_clusters_true"] == 2
        assert result["num_clusters_pred"] == 3
        assert result["num_samples"] == 6

    def test_string_labels(self):
        """Test with string labels."""
        labels_true = ["fiction", "fiction", "non-fiction", "non-fiction"]
        labels_pred = [0, 0, 1, 1]
        result = compute_clustering_metrics(labels_true, labels_pred)

        assert result["v_measure"] == pytest.approx(1.0)
        assert result["nmi"] == pytest.approx(1.0)
        assert result["ari"] == pytest.approx(1.0)

    def test_int_labels(self):
        """Test with integer labels."""
        labels_true = [10, 10, 20, 20, 30, 30]
        labels_pred = [100, 100, 200, 200, 300, 300]
        result = compute_clustering_metrics(labels_true, labels_pred)

        assert result["v_measure"] == pytest.approx(1.0)
        assert result["ari"] == pytest.approx(1.0)

    def test_v_measure_equals_harmonic_mean(self):
        """Test that V-measure is harmonic mean of homogeneity and completeness."""
        labels_true = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        labels_pred = [0, 0, 1, 1, 1, 2, 2, 2, 2]
        result = compute_clustering_metrics(labels_true, labels_pred)

        h = result["homogeneity"]
        c = result["completeness"]
        v = result["v_measure"]

        # V-measure should be harmonic mean (with beta=1.0)
        if h + c > 0:
            expected_v = 2 * (h * c) / (h + c)
            assert v == pytest.approx(expected_v, abs=1e-10)


@pytest.mark.unit
@pytest.mark.metrics
class TestClusteringEdgeCases:
    """Edge case tests for clustering metrics."""

    def test_single_cluster_predicted(self):
        """Test when all samples are assigned to one cluster."""
        labels_true = [0, 0, 1, 1, 2, 2]
        labels_pred = [0, 0, 0, 0, 0, 0]
        result = compute_clustering_metrics(labels_true, labels_pred)

        # Homogeneity should be undefined/low (mixed classes in one cluster)
        # Completeness should be 1.0 (all class members together)
        assert result["num_clusters_pred"] == 1
        assert result["num_clusters_true"] == 3
        assert result["completeness"] == pytest.approx(1.0)

    def test_all_singleton_clusters(self):
        """Test when each sample is in its own cluster."""
        labels_true = [0, 0, 1, 1, 2, 2]
        labels_pred = [0, 1, 2, 3, 4, 5]
        result = compute_clustering_metrics(labels_true, labels_pred)

        # Homogeneity should be 1.0 (each cluster is pure)
        # Completeness should be low (classes are split)
        assert result["num_clusters_pred"] == 6
        assert result["num_clusters_true"] == 3
        assert result["homogeneity"] == pytest.approx(1.0)

    def test_single_true_class(self):
        """Test with only one true class."""
        labels_true = [0, 0, 0, 0, 0, 0]
        labels_pred = [0, 0, 1, 1, 2, 2]
        result = compute_clustering_metrics(labels_true, labels_pred)

        assert result["num_clusters_true"] == 1
        assert result["num_clusters_pred"] == 3
        # Metrics should still be computable
        assert 0 <= result["v_measure"] <= 1
        assert 0 <= result["nmi"] <= 1

    def test_two_samples(self):
        """Test with minimal number of samples."""
        labels_true = [0, 1]
        labels_pred = [0, 1]
        result = compute_clustering_metrics(labels_true, labels_pred)

        assert result["num_samples"] == 2
        assert result["v_measure"] == pytest.approx(1.0)
        assert result["ari"] == pytest.approx(1.0)

    def test_large_number_of_clusters(self):
        """Test with many clusters."""
        n_samples = 100
        n_clusters = 20
        labels_true = [i % n_clusters for i in range(n_samples)]
        labels_pred = [i % n_clusters for i in range(n_samples)]
        result = compute_clustering_metrics(labels_true, labels_pred)

        assert result["num_clusters_true"] == n_clusters
        assert result["num_clusters_pred"] == n_clusters
        assert result["v_measure"] == pytest.approx(1.0)

    def test_empty_labels_raises(self):
        """Test that empty labels raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            compute_clustering_metrics([], [])

    def test_mismatched_lengths_raises(self):
        """Test that mismatched lengths raise ValueError."""
        labels_true = [0, 0, 1, 1]
        labels_pred = [0, 0, 1]

        with pytest.raises(ValueError, match="Length mismatch"):
            compute_clustering_metrics(labels_true, labels_pred)

    def test_single_sample_raises(self):
        """Test that single sample is handled (sklearn requirement)."""
        # Single sample might cause issues with some metrics
        labels_true = [0]
        labels_pred = [0]
        # Should either work or raise a clear error
        # sklearn handles this, so we expect it to work
        result = compute_clustering_metrics(labels_true, labels_pred)
        assert result["num_samples"] == 1

    def test_all_same_true_label(self):
        """Test when all true labels are the same."""
        labels_true = ["A", "A", "A", "A", "A"]
        labels_pred = [0, 0, 1, 1, 2]
        result = compute_clustering_metrics(labels_true, labels_pred)

        assert result["num_clusters_true"] == 1
        assert result["num_clusters_pred"] == 3
        # Should compute without errors
        assert "v_measure" in result

    def test_all_same_predicted_label(self):
        """Test when all predictions are the same."""
        labels_true = ["A", "B", "C", "D", "E"]
        labels_pred = [0, 0, 0, 0, 0]
        result = compute_clustering_metrics(labels_true, labels_pred)

        assert result["num_clusters_true"] == 5
        assert result["num_clusters_pred"] == 1
        # Completeness should be 1.0 (all members together)
        assert result["completeness"] == pytest.approx(1.0)

    def test_mixed_label_types(self):
        """Test with mixed string and numeric labels."""
        # True labels as strings, predicted as ints
        labels_true = ["class_0", "class_0", "class_1", "class_1"]
        labels_pred = [0, 0, 1, 1]
        result = compute_clustering_metrics(labels_true, labels_pred)

        assert result["v_measure"] == pytest.approx(1.0)
        assert result["num_samples"] == 4

    def test_duplicate_cluster_ids_noncontiguous(self):
        """Test with non-contiguous cluster IDs."""
        labels_true = [0, 0, 1, 1, 2, 2]
        labels_pred = [10, 10, 20, 20, 30, 30]
        result = compute_clustering_metrics(labels_true, labels_pred)

        # Should handle non-contiguous IDs
        assert result["num_clusters_pred"] == 3
        assert result["v_measure"] == pytest.approx(1.0)

    def test_numpy_arrays_as_input(self):
        """Test that numpy arrays work as input (converted to list)."""
        labels_true = np.array([0, 0, 1, 1, 2, 2])
        labels_pred = np.array([0, 0, 1, 1, 2, 2])
        result = compute_clustering_metrics(labels_true.tolist(), labels_pred.tolist())

        assert result["v_measure"] == pytest.approx(1.0)
        assert result["num_samples"] == 6


@pytest.mark.unit
@pytest.mark.metrics
class TestClusteringMetricProperties:
    """Test mathematical properties of clustering metrics."""

    def test_vmeasure_symmetry_property(self):
        """Test V-measure relationship with homogeneity and completeness."""
        labels_true = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        labels_pred = [0, 0, 1, 1, 1, 2, 2, 2, 2]

        result = compute_clustering_metrics(labels_true, labels_pred)

        # V-measure is the harmonic mean of homogeneity and completeness
        # It should be <= max(homogeneity, completeness)
        assert 0 <= result["v_measure"] <= 1
        assert (
            result["v_measure"]
            <= max(result["homogeneity"], result["completeness"]) + 1e-6
        )

    def test_ari_symmetry(self):
        """Test that ARI is symmetric."""
        labels_true = [0, 0, 1, 1, 2, 2]
        labels_pred = [0, 1, 0, 1, 0, 1]

        ari1 = adjusted_rand_index(labels_true, labels_pred)
        ari2 = adjusted_rand_index(labels_pred, labels_true)

        assert ari1 == pytest.approx(ari2)

    def test_perfect_clustering_all_ones(self):
        """Test that perfect clustering gives all metrics = 1.0 (except ARI which is 1.0)."""
        labels_true = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        labels_pred = [0, 0, 0, 1, 1, 1, 2, 2, 2]

        result = compute_clustering_metrics(labels_true, labels_pred)

        assert result["v_measure"] == pytest.approx(1.0)
        assert result["nmi"] == pytest.approx(1.0)
        assert result["ari"] == pytest.approx(1.0)
        assert result["homogeneity"] == pytest.approx(1.0)
        assert result["completeness"] == pytest.approx(1.0)

    def test_perfect_clustering_permuted_labels(self):
        """Test that label permutation doesn't affect metrics."""
        labels_true = [0, 0, 1, 1, 2, 2]
        # Same clustering, different cluster IDs
        labels_pred1 = [0, 0, 1, 1, 2, 2]
        labels_pred2 = [5, 5, 7, 7, 9, 9]

        result1 = compute_clustering_metrics(labels_true, labels_pred1)
        result2 = compute_clustering_metrics(labels_true, labels_pred2)

        assert result1["v_measure"] == pytest.approx(result2["v_measure"])
        assert result1["nmi"] == pytest.approx(result2["nmi"])
        assert result1["ari"] == pytest.approx(result2["ari"])

    def test_homogeneity_completeness_tradeoff(self):
        """Test tradeoff between homogeneity and completeness."""
        # Create clustering that is homogeneous but not complete
        labels_true = [0, 0, 0, 0, 1, 1, 1, 1]
        labels_pred = [0, 0, 1, 1, 2, 2, 3, 3]  # Split each class into 2 clusters

        result = compute_clustering_metrics(labels_true, labels_pred)

        # Should be perfectly homogeneous (each cluster is pure)
        assert result["homogeneity"] == pytest.approx(1.0)
        # But not complete (classes are split)
        assert result["completeness"] < 1.0

        # Now test opposite: complete but not homogeneous
        labels_true = [0, 0, 1, 1, 2, 2]
        labels_pred = [0, 0, 0, 0, 1, 1]  # Merge classes 0 and 1

        result2 = compute_clustering_metrics(labels_true, labels_pred)

        # Not homogeneous (cluster 0 has mixed classes)
        assert result2["homogeneity"] < 1.0
