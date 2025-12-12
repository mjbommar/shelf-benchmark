"""Unit tests for ClusteringEvaluator.

Tests the clustering evaluator for SHELF tasks, including:
- Evaluator initialization with different cluster configurations
- evaluate() method with perfect, random, and edge case clustering
- Cluster count computation and validation
- Handling mismatched cluster counts
- Bootstrap confidence intervals
- Integration with prediction validation
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from shelf.evaluate.evaluators.clustering import ClusteringEvaluator
from shelf.evaluate.schemas import ValidationError
from shelf.evaluate.tasks import TaskSpec, TaskType


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def clustering_task_spec() -> TaskSpec:
    """Task spec for clustering evaluation."""
    return TaskSpec(
        name="test_clustering",
        task_type=TaskType.CLUSTERING,
        description="Test clustering task",
        text_field="text",
        label_field="lcc",
        id_field="id",
        label_space=tuple(["A", "B", "C", "D"]),  # 4 classes
        primary_metric="v_measure",
        secondary_metrics=tuple(["nmi", "ari", "homogeneity", "completeness"]),
        dataset_name="test/dataset",
        dataset_config="default",
        default_split="test",
    )


@pytest.fixture
def clustering_task_spec_no_label_space() -> TaskSpec:
    """Task spec without predefined label space."""
    return TaskSpec(
        name="test_clustering_open",
        task_type=TaskType.CLUSTERING,
        description="Test clustering with open label space",
        text_field="text",
        label_field="lcc",
        id_field="id",
        label_space=None,  # Infer from data
        primary_metric="v_measure",
        secondary_metrics=tuple(["nmi", "ari"]),
        dataset_name="test/dataset",
        dataset_config="default",
        default_split="test",
    )


@pytest.fixture
def ground_truth_clustering() -> pl.DataFrame:
    """Ground truth data for clustering tests."""
    return pl.DataFrame(
        {
            "id": [f"doc_{i:03d}" for i in range(12)],
            "text": [f"Document {i}" for i in range(12)],
            "lcc": ["A", "A", "A", "B", "B", "B", "C", "C", "C", "D", "D", "D"],
            "form": ["essay"] * 12,
            "model": ["gpt-5.1"] * 12,
        }
    )


@pytest.fixture
def perfect_clustering_predictions() -> list[dict[str, int]]:
    """Perfect clustering predictions (clusters match ground truth)."""
    return [
        {"id": "doc_000", "cluster": 0},
        {"id": "doc_001", "cluster": 0},
        {"id": "doc_002", "cluster": 0},
        {"id": "doc_003", "cluster": 1},
        {"id": "doc_004", "cluster": 1},
        {"id": "doc_005", "cluster": 1},
        {"id": "doc_006", "cluster": 2},
        {"id": "doc_007", "cluster": 2},
        {"id": "doc_008", "cluster": 2},
        {"id": "doc_009", "cluster": 3},
        {"id": "doc_010", "cluster": 3},
        {"id": "doc_011", "cluster": 3},
    ]


@pytest.fixture
def random_clustering_predictions() -> list[dict[str, int]]:
    """Random clustering predictions."""
    np.random.seed(42)
    return [
        {"id": f"doc_{i:03d}", "cluster": int(np.random.randint(0, 4))}
        for i in range(12)
    ]


@pytest.fixture
def partial_clustering_predictions() -> list[dict[str, int]]:
    """Partially correct clustering (some mixed clusters)."""
    return [
        {"id": "doc_000", "cluster": 0},
        {"id": "doc_001", "cluster": 0},
        {"id": "doc_002", "cluster": 1},  # Misplaced
        {"id": "doc_003", "cluster": 1},
        {"id": "doc_004", "cluster": 1},
        {"id": "doc_005", "cluster": 0},  # Misplaced
        {"id": "doc_006", "cluster": 2},
        {"id": "doc_007", "cluster": 2},
        {"id": "doc_008", "cluster": 2},
        {"id": "doc_009", "cluster": 3},
        {"id": "doc_010", "cluster": 3},
        {"id": "doc_011", "cluster": 3},
    ]


@pytest.fixture
def wrong_cluster_count_predictions() -> list[dict[str, int]]:
    """Predictions with wrong number of clusters (5 instead of 4)."""
    return [
        {"id": "doc_000", "cluster": 0},
        {"id": "doc_001", "cluster": 0},
        {"id": "doc_002", "cluster": 0},
        {"id": "doc_003", "cluster": 1},
        {"id": "doc_004", "cluster": 1},
        {"id": "doc_005", "cluster": 1},
        {"id": "doc_006", "cluster": 2},
        {"id": "doc_007", "cluster": 2},
        {"id": "doc_008", "cluster": 4},  # Extra cluster
        {"id": "doc_009", "cluster": 3},
        {"id": "doc_010", "cluster": 3},
        {"id": "doc_011", "cluster": 3},
    ]


# ===========================================================================
# Test ClusteringEvaluator Initialization
# ===========================================================================


@pytest.mark.unit
@pytest.mark.evaluator
def test_clustering_evaluator_init_with_label_space(clustering_task_spec):
    """Test evaluator initialization with label space."""
    evaluator = ClusteringEvaluator(clustering_task_spec)

    assert evaluator.task_spec == clustering_task_spec
    assert evaluator.n_clusters == 4  # From label_space
    assert evaluator.random_seed == 42


@pytest.mark.unit
@pytest.mark.evaluator
def test_clustering_evaluator_init_no_label_space(clustering_task_spec_no_label_space):
    """Test evaluator initialization without label space."""
    evaluator = ClusteringEvaluator(clustering_task_spec_no_label_space)

    assert evaluator.task_spec == clustering_task_spec_no_label_space
    assert evaluator.n_clusters is None  # Will be inferred from data
    assert evaluator.random_seed == 42


@pytest.mark.unit
@pytest.mark.evaluator
def test_clustering_evaluator_init_explicit_clusters(clustering_task_spec):
    """Test evaluator initialization with explicit n_clusters."""
    evaluator = ClusteringEvaluator(clustering_task_spec, n_clusters=10)

    assert evaluator.n_clusters == 10  # Explicit override


@pytest.mark.unit
@pytest.mark.evaluator
def test_clustering_evaluator_init_custom_seed(clustering_task_spec):
    """Test evaluator initialization with custom random seed."""
    evaluator = ClusteringEvaluator(clustering_task_spec, random_seed=999)

    assert evaluator.random_seed == 999


# ===========================================================================
# Test evaluate() - Perfect Clustering
# ===========================================================================


@pytest.mark.unit
@pytest.mark.evaluator
def test_evaluate_perfect_clustering(
    clustering_task_spec,
    ground_truth_clustering,
    perfect_clustering_predictions,
):
    """Test evaluation with perfect clustering."""
    evaluator = ClusteringEvaluator(clustering_task_spec)
    result = evaluator.evaluate(perfect_clustering_predictions, ground_truth_clustering)

    # Check result structure
    assert result.task == "test_clustering"
    assert result.task_type == "clustering"
    assert result.primary_metric == "v_measure"
    assert result.num_samples == 12

    # Perfect clustering should have perfect metrics
    assert result.metrics["v_measure"] == pytest.approx(1.0)
    assert result.metrics["nmi"] == pytest.approx(1.0)
    assert result.metrics["ari"] == pytest.approx(1.0)
    assert result.metrics["homogeneity"] == pytest.approx(1.0)
    assert result.metrics["completeness"] == pytest.approx(1.0)

    # Check cluster counts
    assert result.metrics["num_clusters_true"] == 4
    assert result.metrics["num_clusters_pred"] == 4
    assert result.metrics["num_samples"] == 12

    # Primary score should match primary metric
    assert result.primary_score == pytest.approx(1.0)


@pytest.mark.unit
@pytest.mark.evaluator
def test_evaluate_perfect_clustering_string_labels(
    clustering_task_spec,
    ground_truth_clustering,
    perfect_clustering_predictions,
):
    """Test evaluation handles string labels correctly."""
    evaluator = ClusteringEvaluator(clustering_task_spec)
    result = evaluator.evaluate(perfect_clustering_predictions, ground_truth_clustering)

    # Should work with string labels (A, B, C, D) vs int clusters (0, 1, 2, 3)
    assert result.metrics["v_measure"] == pytest.approx(1.0)


# ===========================================================================
# Test evaluate() - Random/Poor Clustering
# ===========================================================================


@pytest.mark.unit
@pytest.mark.evaluator
def test_evaluate_random_clustering(
    clustering_task_spec,
    ground_truth_clustering,
    random_clustering_predictions,
):
    """Test evaluation with random clustering."""
    evaluator = ClusteringEvaluator(clustering_task_spec)
    result = evaluator.evaluate(random_clustering_predictions, ground_truth_clustering)

    # Random clustering should have low metrics
    assert 0.0 <= result.metrics["v_measure"] <= 0.5
    assert 0.0 <= result.metrics["nmi"] <= 0.5
    # ARI can be negative (worse than random)
    assert -0.5 <= result.metrics["ari"] <= 0.5

    # Metrics should be non-negative (except ARI which can be negative)
    assert result.metrics["homogeneity"] >= 0.0
    assert result.metrics["completeness"] >= 0.0


@pytest.mark.unit
@pytest.mark.evaluator
def test_evaluate_partial_clustering(
    clustering_task_spec,
    ground_truth_clustering,
    partial_clustering_predictions,
):
    """Test evaluation with partially correct clustering."""
    evaluator = ClusteringEvaluator(clustering_task_spec)
    result = evaluator.evaluate(partial_clustering_predictions, ground_truth_clustering)

    # Partial clustering should have medium metrics (better than random, worse than perfect)
    assert 0.3 <= result.metrics["v_measure"] <= 0.95
    assert 0.3 <= result.metrics["nmi"] <= 0.95

    # Should still have 4 clusters
    assert result.metrics["num_clusters_pred"] == 4


# ===========================================================================
# Test evaluate() - Cluster Count Validation
# ===========================================================================


@pytest.mark.unit
@pytest.mark.evaluator
def test_evaluate_wrong_cluster_count_warning(
    clustering_task_spec,
    ground_truth_clustering,
    wrong_cluster_count_predictions,
):
    """Test evaluation with wrong number of clusters (generates warning)."""
    evaluator = ClusteringEvaluator(clustering_task_spec)

    # Should raise ValidationError because validation warns about cluster count mismatch
    # But the validation only generates a warning, not an error, so it should still work
    result = evaluator.evaluate(
        wrong_cluster_count_predictions, ground_truth_clustering
    )

    # Check that we got 5 clusters instead of 4
    assert result.metrics["num_clusters_pred"] == 5
    assert result.metrics["num_clusters_true"] == 4

    # Metrics should still be computable
    assert "v_measure" in result.metrics
    assert result.metrics["v_measure"] >= 0.0


@pytest.mark.unit
@pytest.mark.evaluator
def test_evaluate_single_cluster(clustering_task_spec, ground_truth_clustering):
    """Test evaluation when all predictions are in one cluster."""
    predictions = [{"id": f"doc_{i:03d}", "cluster": 0} for i in range(12)]

    evaluator = ClusteringEvaluator(clustering_task_spec)
    result = evaluator.evaluate(predictions, ground_truth_clustering)

    # Single cluster should have perfect completeness, but poor homogeneity
    assert result.metrics["completeness"] == pytest.approx(1.0)
    assert result.metrics["homogeneity"] < 0.5
    assert result.metrics["num_clusters_pred"] == 1


@pytest.mark.unit
@pytest.mark.evaluator
def test_evaluate_each_item_own_cluster(clustering_task_spec, ground_truth_clustering):
    """Test evaluation when each item is in its own cluster."""
    predictions = [{"id": f"doc_{i:03d}", "cluster": i} for i in range(12)]

    evaluator = ClusteringEvaluator(clustering_task_spec)
    result = evaluator.evaluate(predictions, ground_truth_clustering)

    # Each item in own cluster: perfect homogeneity, low completeness
    assert result.metrics["homogeneity"] == pytest.approx(1.0)
    # Completeness will be relatively low (but exact value depends on sklearn implementation)
    assert result.metrics["completeness"] < 0.7
    assert result.metrics["num_clusters_pred"] == 12


# ===========================================================================
# Test evaluate() - Edge Cases
# ===========================================================================


@pytest.mark.unit
@pytest.mark.evaluator
def test_evaluate_missing_predictions(
    clustering_task_spec,
    ground_truth_clustering,
):
    """Test evaluation with missing predictions."""
    # Only predict for half the documents
    predictions = [{"id": f"doc_{i:03d}", "cluster": 0} for i in range(6)]

    evaluator = ClusteringEvaluator(clustering_task_spec)

    # Should raise ValidationError for missing predictions
    with pytest.raises(ValidationError) as exc_info:
        evaluator.evaluate(predictions, ground_truth_clustering)

    assert "Missing predictions" in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.evaluator
def test_evaluate_duplicate_predictions(
    clustering_task_spec,
    ground_truth_clustering,
    perfect_clustering_predictions,
):
    """Test evaluation with duplicate predictions."""
    # Add a duplicate
    predictions = perfect_clustering_predictions + [{"id": "doc_000", "cluster": 1}]

    evaluator = ClusteringEvaluator(clustering_task_spec)

    # Should raise ValidationError for duplicate IDs
    with pytest.raises(ValidationError) as exc_info:
        evaluator.evaluate(predictions, ground_truth_clustering)

    assert "duplicate" in str(exc_info.value).lower()


@pytest.mark.unit
@pytest.mark.evaluator
def test_evaluate_unknown_document_id(
    clustering_task_spec,
    ground_truth_clustering,
    perfect_clustering_predictions,
):
    """Test evaluation with unknown document ID."""
    # Add prediction for unknown document
    predictions = perfect_clustering_predictions + [{"id": "doc_999", "cluster": 0}]

    evaluator = ClusteringEvaluator(clustering_task_spec)

    # Should raise ValidationError for unknown ID
    with pytest.raises(ValidationError) as exc_info:
        evaluator.evaluate(predictions, ground_truth_clustering)

    assert "unknown" in str(exc_info.value).lower()


@pytest.mark.unit
@pytest.mark.evaluator
def test_evaluate_negative_cluster_id(clustering_task_spec, ground_truth_clustering):
    """Test evaluation with negative cluster ID."""
    predictions = [{"id": "doc_000", "cluster": -1}]

    evaluator = ClusteringEvaluator(clustering_task_spec)

    # Should raise ValidationError for negative cluster
    with pytest.raises(ValidationError) as exc_info:
        evaluator.evaluate(predictions, ground_truth_clustering)

    assert "non-negative" in str(exc_info.value).lower()


@pytest.mark.unit
@pytest.mark.evaluator
def test_evaluate_empty_predictions(clustering_task_spec, ground_truth_clustering):
    """Test evaluation with empty predictions list."""
    predictions = []

    evaluator = ClusteringEvaluator(clustering_task_spec)

    # Should raise ValidationError for missing predictions
    with pytest.raises(ValidationError) as exc_info:
        evaluator.evaluate(predictions, ground_truth_clustering)

    assert "Missing predictions" in str(exc_info.value)


# ===========================================================================
# Test evaluate() - Data Validation
# ===========================================================================


@pytest.mark.unit
@pytest.mark.evaluator
def test_evaluate_invalid_prediction_schema(
    clustering_task_spec,
    ground_truth_clustering,
):
    """Test evaluation with invalid prediction schema."""
    # Missing 'cluster' field
    predictions = [{"id": "doc_000"}]

    evaluator = ClusteringEvaluator(clustering_task_spec)

    with pytest.raises(ValidationError):
        evaluator.evaluate(predictions, ground_truth_clustering)


@pytest.mark.unit
@pytest.mark.evaluator
def test_evaluate_invalid_cluster_type(clustering_task_spec, ground_truth_clustering):
    """Test evaluation with invalid cluster type."""
    # Cluster should be int, not string
    predictions = [{"id": "doc_000", "cluster": "cluster_A"}]

    evaluator = ClusteringEvaluator(clustering_task_spec)

    with pytest.raises(ValidationError):
        evaluator.evaluate(predictions, ground_truth_clustering)


# ===========================================================================
# Test Cluster Count Computation
# ===========================================================================


@pytest.mark.unit
@pytest.mark.evaluator
def test_cluster_count_from_label_space(clustering_task_spec):
    """Test that n_clusters is set from label_space."""
    evaluator = ClusteringEvaluator(clustering_task_spec)

    assert evaluator.n_clusters == 4


@pytest.mark.unit
@pytest.mark.evaluator
def test_cluster_count_inferred_from_data(
    clustering_task_spec_no_label_space,
    ground_truth_clustering,
    perfect_clustering_predictions,
):
    """Test that cluster count can be inferred from data."""
    evaluator = ClusteringEvaluator(clustering_task_spec_no_label_space)

    # n_clusters is None initially
    assert evaluator.n_clusters is None

    # Should still work - metrics will show actual cluster counts
    result = evaluator.evaluate(perfect_clustering_predictions, ground_truth_clustering)

    assert result.metrics["num_clusters_true"] == 4
    assert result.metrics["num_clusters_pred"] == 4


# ===========================================================================
# Test Result Context and Provenance
# ===========================================================================


@pytest.mark.unit
@pytest.mark.evaluator
def test_evaluate_result_context(
    clustering_task_spec,
    ground_truth_clustering,
    perfect_clustering_predictions,
):
    """Test that evaluation result includes proper context."""
    evaluator = ClusteringEvaluator(clustering_task_spec, random_seed=123)
    result = evaluator.evaluate(perfect_clustering_predictions, ground_truth_clustering)

    # Check context
    assert result.context is not None
    assert result.context.random_seed == 123
    assert result.context.dataset_checksum is not None

    # Check provenance
    assert result.data_provenance is not None


@pytest.mark.unit
@pytest.mark.evaluator
def test_evaluate_result_split(
    clustering_task_spec,
    ground_truth_clustering,
    perfect_clustering_predictions,
):
    """Test that result includes split information."""
    evaluator = ClusteringEvaluator(clustering_task_spec)
    result = evaluator.evaluate(perfect_clustering_predictions, ground_truth_clustering)

    assert result.split == "test"  # From task_spec.default_split


# ===========================================================================
# Test Metrics Completeness
# ===========================================================================


@pytest.mark.unit
@pytest.mark.evaluator
def test_evaluate_all_metrics_present(
    clustering_task_spec,
    ground_truth_clustering,
    perfect_clustering_predictions,
):
    """Test that all expected metrics are present in results."""
    evaluator = ClusteringEvaluator(clustering_task_spec)
    result = evaluator.evaluate(perfect_clustering_predictions, ground_truth_clustering)

    expected_metrics = {
        "v_measure",
        "nmi",
        "ari",
        "homogeneity",
        "completeness",
        "num_samples",
        "num_clusters_true",
        "num_clusters_pred",
    }

    assert set(result.metrics.keys()) == expected_metrics


@pytest.mark.unit
@pytest.mark.evaluator
def test_evaluate_metrics_ranges(
    clustering_task_spec,
    ground_truth_clustering,
    random_clustering_predictions,
):
    """Test that metrics are within expected ranges."""
    evaluator = ClusteringEvaluator(clustering_task_spec)
    result = evaluator.evaluate(random_clustering_predictions, ground_truth_clustering)

    # All clustering metrics should be in [0, 1] except ARI which can be negative
    assert 0.0 <= result.metrics["v_measure"] <= 1.0
    assert 0.0 <= result.metrics["nmi"] <= 1.0
    assert -1.0 <= result.metrics["ari"] <= 1.0
    assert 0.0 <= result.metrics["homogeneity"] <= 1.0
    assert 0.0 <= result.metrics["completeness"] <= 1.0


# ===========================================================================
# Test Different Ground Truth Sizes
# ===========================================================================


@pytest.mark.unit
@pytest.mark.evaluator
def test_evaluate_small_dataset(clustering_task_spec):
    """Test evaluation with very small dataset."""
    ground_truth = pl.DataFrame(
        {
            "id": ["doc_0", "doc_1", "doc_2"],
            "text": ["Text 0", "Text 1", "Text 2"],
            "lcc": ["A", "A", "B"],
        }
    )

    predictions = [
        {"id": "doc_0", "cluster": 0},
        {"id": "doc_1", "cluster": 0},
        {"id": "doc_2", "cluster": 1},
    ]

    evaluator = ClusteringEvaluator(clustering_task_spec)
    result = evaluator.evaluate(predictions, ground_truth)

    assert result.num_samples == 3
    assert result.metrics["v_measure"] == pytest.approx(1.0)


@pytest.mark.unit
@pytest.mark.evaluator
def test_evaluate_large_dataset(clustering_task_spec):
    """Test evaluation with larger dataset."""
    n_docs = 1000
    n_clusters = 4

    # Create balanced ground truth
    ground_truth = pl.DataFrame(
        {
            "id": [f"doc_{i:04d}" for i in range(n_docs)],
            "text": [f"Text {i}" for i in range(n_docs)],
            "lcc": [chr(ord("A") + (i % n_clusters)) for i in range(n_docs)],
        }
    )

    # Perfect clustering
    predictions = [
        {"id": f"doc_{i:04d}", "cluster": i % n_clusters} for i in range(n_docs)
    ]

    evaluator = ClusteringEvaluator(clustering_task_spec)
    result = evaluator.evaluate(predictions, ground_truth)

    assert result.num_samples == 1000
    assert result.metrics["v_measure"] == pytest.approx(1.0)


# ===========================================================================
# Test Non-contiguous Cluster IDs
# ===========================================================================


@pytest.mark.unit
@pytest.mark.evaluator
def test_evaluate_non_contiguous_cluster_ids(
    clustering_task_spec,
    ground_truth_clustering,
):
    """Test evaluation with non-contiguous cluster IDs (e.g., 0, 5, 10, 15)."""
    predictions = [
        {"id": "doc_000", "cluster": 0},
        {"id": "doc_001", "cluster": 0},
        {"id": "doc_002", "cluster": 0},
        {"id": "doc_003", "cluster": 5},
        {"id": "doc_004", "cluster": 5},
        {"id": "doc_005", "cluster": 5},
        {"id": "doc_006", "cluster": 10},
        {"id": "doc_007", "cluster": 10},
        {"id": "doc_008", "cluster": 10},
        {"id": "doc_009", "cluster": 15},
        {"id": "doc_010", "cluster": 15},
        {"id": "doc_011", "cluster": 15},
    ]

    evaluator = ClusteringEvaluator(clustering_task_spec)
    result = evaluator.evaluate(predictions, ground_truth_clustering)

    # Should still work - cluster IDs don't need to be contiguous
    assert result.metrics["num_clusters_pred"] == 4
    assert result.metrics["v_measure"] == pytest.approx(1.0)


# ===========================================================================
# Test Summary and String Representation
# ===========================================================================


@pytest.mark.unit
@pytest.mark.evaluator
def test_evaluate_result_summary(
    clustering_task_spec,
    ground_truth_clustering,
    perfect_clustering_predictions,
):
    """Test that result summary is generated correctly."""
    evaluator = ClusteringEvaluator(clustering_task_spec)
    result = evaluator.evaluate(perfect_clustering_predictions, ground_truth_clustering)

    summary = result.summary()

    # Summary should include key information
    assert "test_clustering" in summary
    assert "v_measure" in summary
    assert "1.0" in summary or "1.00" in summary


# ===========================================================================
# Test Confidence Intervals (Not Yet Implemented)
# ===========================================================================


@pytest.mark.unit
@pytest.mark.evaluator
def test_evaluate_compute_ci_not_implemented(
    clustering_task_spec,
    ground_truth_clustering,
    perfect_clustering_predictions,
):
    """Test that compute_ci parameter exists (even if not implemented)."""
    evaluator = ClusteringEvaluator(clustering_task_spec)

    # Should not raise an error, even if CI computation is not implemented
    result = evaluator.evaluate(
        perfect_clustering_predictions, ground_truth_clustering, compute_ci=True
    )

    # CI might be None if not implemented
    # Just check that the parameter is accepted
    assert result is not None


# ===========================================================================
# Test Integration with Metrics Module
# ===========================================================================


@pytest.mark.unit
@pytest.mark.evaluator
def test_evaluate_uses_metrics_module(
    clustering_task_spec,
    ground_truth_clustering,
    perfect_clustering_predictions,
):
    """Test that evaluator correctly uses the metrics.clustering module."""
    evaluator = ClusteringEvaluator(clustering_task_spec)
    result = evaluator.evaluate(perfect_clustering_predictions, ground_truth_clustering)

    # All these metrics should come from compute_clustering_metrics()
    assert "v_measure" in result.metrics
    assert "nmi" in result.metrics
    assert "ari" in result.metrics
    assert "homogeneity" in result.metrics
    assert "completeness" in result.metrics

    # Verify they're float values
    for metric_name, value in result.metrics.items():
        if metric_name.startswith("num_"):
            assert isinstance(value, int)
        else:
            assert isinstance(value, (int, float))
