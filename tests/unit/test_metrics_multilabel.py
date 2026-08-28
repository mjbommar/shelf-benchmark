"""Unit tests for shelf.evaluate.metrics.multilabel module.

Tests cover:
- binarize_labels with a fixed vocabulary
- F1 averaging variants (micro, macro, samples, weighted)
- subset_accuracy and hamming_loss
- Threshold-free ranking metrics (LRAP, mAP, coverage error)
- Per-label breakdown
- compute_multilabel_metrics aggregation
- Edge cases: empty label sets, single-label rows, all-labels rows,
  a label that never appears, zero division
"""

from __future__ import annotations

import numpy as np
import pytest
from shelf.evaluate.metrics.multilabel import (
    binarize_labels,
    compute_multilabel_metrics,
    hamming_loss,
    label_cardinality,
    label_coverage_error,
    label_ranking_average_precision,
    mean_average_precision,
    multilabel_f1,
    per_label_metrics,
    subset_accuracy,
)

LABELS = ["a", "b", "c"]


@pytest.fixture
def perfect_multilabel() -> tuple[np.ndarray, np.ndarray]:
    """Ground truth and identical predictions."""
    y_true = binarize_labels([["a"], ["a", "b"], ["b", "c"]], LABELS)
    return y_true, y_true.copy()


@pytest.fixture
def partial_multilabel() -> tuple[np.ndarray, np.ndarray]:
    """Ground truth with partially overlapping predictions."""
    y_true = binarize_labels([["a"], ["a", "b"], ["b", "c"]], LABELS)
    y_pred = binarize_labels([["a"], ["a"], ["b"]], LABELS)
    return y_true, y_pred


class TestBinarizeLabels:
    """Tests for binarize_labels."""

    @pytest.mark.unit
    def test_basic_binarization(self):
        """Test label sets map onto the fixed column order."""
        m = binarize_labels([["a"], ["a", "b"], []], LABELS)
        assert m.tolist() == [[1, 0, 0], [1, 1, 0], [0, 0, 0]]

    @pytest.mark.unit
    def test_column_order_follows_vocabulary(self):
        """Test column order is the caller's, not sorted or observed order."""
        m = binarize_labels([["a", "c"]], ["c", "b", "a"])
        assert m.tolist() == [[1, 0, 1]]

    @pytest.mark.unit
    def test_empty_label_set(self):
        """Test a document with no labels becomes an all-zero row."""
        m = binarize_labels([[]], LABELS)
        assert m.tolist() == [[0, 0, 0]]
        assert m.sum() == 0

    @pytest.mark.unit
    def test_all_labels_row(self):
        """Test a document carrying every label becomes an all-one row."""
        m = binarize_labels([["a", "b", "c"]], LABELS)
        assert m.tolist() == [[1, 1, 1]]

    @pytest.mark.unit
    def test_unknown_label_is_ignored(self):
        """Test labels outside the vocabulary are dropped, not errors."""
        m = binarize_labels([["a", "zzz"]], LABELS)
        assert m.tolist() == [[1, 0, 0]]

    @pytest.mark.unit
    def test_duplicate_label_is_idempotent(self):
        """Test a repeated label still sets the cell exactly once."""
        m = binarize_labels([["a", "a"]], LABELS)
        assert m.tolist() == [[1, 0, 0]]

    @pytest.mark.unit
    def test_shape(self):
        """Test output shape is (n_samples, n_labels)."""
        m = binarize_labels([["a"], ["b"]], LABELS)
        assert m.shape == (2, 3)

    @pytest.mark.unit
    def test_no_samples(self):
        """Test an empty corpus yields a (0, n_labels) matrix."""
        m = binarize_labels([], LABELS)
        assert m.shape == (0, 3)


class TestF1Variants:
    """Tests for the F1 averaging variants."""

    @pytest.mark.unit
    def test_perfect_predictions(self, perfect_multilabel):
        """Test every averaging returns 1.0 for perfect predictions."""
        y_true, y_pred = perfect_multilabel
        for average in ("micro", "macro", "samples", "weighted"):
            assert multilabel_f1(y_true, y_pred, average) == pytest.approx(1.0)

    @pytest.mark.unit
    def test_partial_predictions_between_zero_and_one(self, partial_multilabel):
        """Test partial predictions score strictly between 0 and 1."""
        y_true, y_pred = partial_multilabel
        for average in ("micro", "macro", "samples", "weighted"):
            score = multilabel_f1(y_true, y_pred, average)
            assert 0.0 < score < 1.0

    @pytest.mark.unit
    def test_micro_f1_known_value(self, partial_multilabel):
        """Test micro F1 against a hand-computed value."""
        y_true, y_pred = partial_multilabel
        # TP = a(row0), a(row1), b(row2) = 3; predicted = 3; true = 5
        # precision = 3/3 = 1.0, recall = 3/5 = 0.6 -> F1 = 2*.6/1.6 = 0.75
        assert multilabel_f1(y_true, y_pred, "micro") == pytest.approx(0.75)

    @pytest.mark.unit
    def test_all_zero_predictions_score_zero(self):
        """Test predicting nothing yields F1 of 0 without a zero-division error."""
        y_true = binarize_labels([["a"], ["b"]], LABELS)
        y_pred = binarize_labels([[], []], LABELS)
        for average in ("micro", "macro", "samples", "weighted"):
            assert multilabel_f1(y_true, y_pred, average) == pytest.approx(0.0)

    @pytest.mark.unit
    def test_label_never_appearing_scores_zero_in_macro(self):
        """Test a never-seen label contributes 0.0 to macro F1."""
        # "c" appears in neither truth nor predictions.
        y_true = binarize_labels([["a"], ["b"]], LABELS)
        y_pred = binarize_labels([["a"], ["b"]], LABELS)
        # micro is perfect, macro is dragged down by the unused third label.
        assert multilabel_f1(y_true, y_pred, "micro") == pytest.approx(1.0)
        assert multilabel_f1(y_true, y_pred, "macro") == pytest.approx(2 / 3)

    @pytest.mark.unit
    def test_macro_below_micro_when_rare_label_missed(self):
        """Test macro punishes rare-label failure more than micro."""
        y_true = binarize_labels([["a"], ["a"], ["a"], ["c"]], LABELS)
        y_pred = binarize_labels([["a"], ["a"], ["a"], ["a"]], LABELS)
        assert multilabel_f1(y_true, y_pred, "macro") < multilabel_f1(
            y_true, y_pred, "micro"
        )


class TestSubsetAccuracyAndHamming:
    """Tests for set-level correctness metrics."""

    @pytest.mark.unit
    def test_perfect_subset_accuracy(self, perfect_multilabel):
        """Test exact match ratio is 1.0 when all rows match."""
        y_true, y_pred = perfect_multilabel
        assert subset_accuracy(y_true, y_pred) == pytest.approx(1.0)
        assert hamming_loss(y_true, y_pred) == pytest.approx(0.0)

    @pytest.mark.unit
    def test_partial_subset_accuracy(self, partial_multilabel):
        """Test only fully-correct rows count toward subset accuracy."""
        y_true, y_pred = partial_multilabel
        # Only row 0 matches exactly.
        assert subset_accuracy(y_true, y_pred) == pytest.approx(1 / 3)

    @pytest.mark.unit
    def test_hamming_loss_counts_cells(self, partial_multilabel):
        """Test Hamming loss is the per-cell error rate."""
        y_true, y_pred = partial_multilabel
        # 2 wrong cells (b in row 1, c in row 2) out of 3 rows * 3 labels.
        assert hamming_loss(y_true, y_pred) == pytest.approx(2 / 9)

    @pytest.mark.unit
    def test_empty_true_and_empty_pred_is_exact_match(self):
        """Test predicting nothing for a document with no labels is correct."""
        y_true = binarize_labels([[]], LABELS)
        y_pred = binarize_labels([[]], LABELS)
        assert subset_accuracy(y_true, y_pred) == pytest.approx(1.0)
        assert hamming_loss(y_true, y_pred) == pytest.approx(0.0)

    @pytest.mark.unit
    def test_all_labels_row_predicted_exactly(self):
        """Test a row with every label can be matched exactly."""
        y_true = binarize_labels([["a", "b", "c"]], LABELS)
        assert subset_accuracy(y_true, y_true.copy()) == pytest.approx(1.0)


class TestRankingMetrics:
    """Tests for threshold-free ranking metrics."""

    @pytest.mark.unit
    def test_lrap_perfect_ranking(self):
        """Test LRAP is 1.0 when true labels rank above false ones."""
        y_true = binarize_labels([["a"], ["b", "c"]], LABELS)
        y_score = np.array([[0.9, 0.1, 0.05], [0.1, 0.9, 0.8]])
        assert label_ranking_average_precision(y_true, y_score) == pytest.approx(1.0)

    @pytest.mark.unit
    def test_lrap_imperfect_ranking_below_one(self):
        """Test LRAP drops when a false label outranks a true one."""
        y_true = binarize_labels([["c"]], LABELS)
        y_score = np.array([[0.9, 0.5, 0.1]])
        assert label_ranking_average_precision(y_true, y_score) < 1.0

    @pytest.mark.unit
    def test_lrap_drops_rows_with_no_true_labels(self):
        """Test rows with an empty label set do not corrupt LRAP."""
        y_true = binarize_labels([["a"], []], LABELS)
        y_score = np.array([[0.9, 0.1, 0.05], [0.3, 0.3, 0.3]])
        assert label_ranking_average_precision(y_true, y_score) == pytest.approx(1.0)

    @pytest.mark.unit
    def test_lrap_all_rows_empty_returns_zero(self):
        """Test LRAP is 0.0 rather than nan when nothing is labelled."""
        y_true = binarize_labels([[], []], LABELS)
        y_score = np.full((2, 3), 0.5)
        assert label_ranking_average_precision(y_true, y_score) == pytest.approx(0.0)

    @pytest.mark.unit
    def test_map_micro_perfect(self):
        """Test micro mAP is 1.0 for a perfect ranking."""
        y_true = binarize_labels([["a"], ["b"]], LABELS)
        y_score = np.array([[0.9, 0.1, 0.0], [0.1, 0.9, 0.0]])
        assert mean_average_precision(y_true, y_score, "micro") == pytest.approx(1.0)

    @pytest.mark.unit
    def test_map_macro_ignores_labels_with_no_positives(self):
        """Test a label with no positive example does not produce nan."""
        # "c" has no positives anywhere.
        y_true = binarize_labels([["a"], ["b"]], LABELS)
        y_score = np.array([[0.9, 0.1, 0.4], [0.1, 0.9, 0.4]])
        value = mean_average_precision(y_true, y_score, "macro")
        assert not np.isnan(value)
        assert value == pytest.approx(1.0)

    @pytest.mark.unit
    def test_map_no_positives_at_all_returns_zero(self):
        """Test mAP is 0.0 when the ground truth has no positive cell."""
        y_true = binarize_labels([[], []], LABELS)
        y_score = np.full((2, 3), 0.5)
        assert mean_average_precision(y_true, y_score, "micro") == pytest.approx(0.0)
        assert mean_average_precision(y_true, y_score, "macro") == pytest.approx(0.0)

    @pytest.mark.unit
    def test_coverage_error_best_case(self):
        """Test coverage error equals label cardinality for a perfect ranking."""
        y_true = binarize_labels([["a"], ["a", "b"]], LABELS)
        y_score = np.array([[0.9, 0.1, 0.05], [0.9, 0.8, 0.05]])
        # Row 0 needs depth 1, row 1 needs depth 2 -> mean 1.5
        assert label_coverage_error(y_true, y_score) == pytest.approx(1.5)

    @pytest.mark.unit
    def test_coverage_error_all_empty_returns_zero(self):
        """Test coverage error is 0.0 rather than nan when nothing is labelled."""
        y_true = binarize_labels([[], []], LABELS)
        y_score = np.full((2, 3), 0.5)
        assert label_coverage_error(y_true, y_score) == pytest.approx(0.0)


class TestPerLabelMetrics:
    """Tests for the per-label breakdown."""

    @pytest.mark.unit
    def test_all_labels_present(self, partial_multilabel):
        """Test every vocabulary label appears in the breakdown."""
        y_true, y_pred = partial_multilabel
        result = per_label_metrics(y_true, y_pred, LABELS)
        assert set(result.keys()) == set(LABELS)

    @pytest.mark.unit
    def test_support_and_predicted_counts(self, partial_multilabel):
        """Test support and num_predicted reflect the indicator matrices."""
        y_true, y_pred = partial_multilabel
        result = per_label_metrics(y_true, y_pred, LABELS)
        assert result["a"]["support"] == 2
        assert result["a"]["num_predicted"] == 2
        assert result["c"]["support"] == 1
        assert result["c"]["num_predicted"] == 0

    @pytest.mark.unit
    def test_never_predicted_label_scores_zero(self, partial_multilabel):
        """Test a label never predicted gets 0.0 rather than a nan."""
        y_true, y_pred = partial_multilabel
        result = per_label_metrics(y_true, y_pred, LABELS)
        assert result["c"]["precision"] == pytest.approx(0.0)
        assert result["c"]["recall"] == pytest.approx(0.0)
        assert result["c"]["f1"] == pytest.approx(0.0)

    @pytest.mark.unit
    def test_label_never_appearing_anywhere(self):
        """Test a label absent from truth and predictions is reported as zero."""
        y_true = binarize_labels([["a"], ["b"]], LABELS)
        result = per_label_metrics(y_true, y_true.copy(), LABELS)
        assert result["c"]["support"] == 0
        assert result["c"]["num_predicted"] == 0
        assert result["c"]["f1"] == pytest.approx(0.0)

    @pytest.mark.unit
    def test_mismatched_label_count_raises(self, partial_multilabel):
        """Test a vocabulary that does not match the matrix width raises."""
        y_true, y_pred = partial_multilabel
        with pytest.raises(ValueError, match="must match matrix columns"):
            per_label_metrics(y_true, y_pred, ["a", "b"])


class TestLabelCardinality:
    """Tests for label_cardinality."""

    @pytest.mark.unit
    def test_mean_labels_per_sample(self):
        """Test cardinality is the mean number of positives per row."""
        y = binarize_labels([["a"], ["a", "b"], ["a", "b", "c"]], LABELS)
        assert label_cardinality(y) == pytest.approx(2.0)

    @pytest.mark.unit
    def test_empty_rows_count_as_zero(self):
        """Test empty rows pull cardinality down."""
        y = binarize_labels([["a"], []], LABELS)
        assert label_cardinality(y) == pytest.approx(0.5)

    @pytest.mark.unit
    def test_no_samples_returns_zero(self):
        """Test cardinality of an empty corpus is 0.0, not nan."""
        assert label_cardinality(binarize_labels([], LABELS)) == pytest.approx(0.0)


class TestComputeMultilabelMetrics:
    """Tests for the compute_multilabel_metrics aggregator."""

    @pytest.mark.unit
    def test_returns_all_f1_variants(self, partial_multilabel):
        """Test all four F1 averagings are reported together."""
        y_true, y_pred = partial_multilabel
        result = compute_multilabel_metrics(y_true, y_pred, LABELS)
        for key in ("micro_f1", "macro_f1", "samples_f1", "weighted_f1"):
            assert key in result
            assert 0.0 <= result[key] <= 1.0

    @pytest.mark.unit
    def test_returns_set_level_metrics(self, partial_multilabel):
        """Test subset accuracy and Hamming loss are reported."""
        y_true, y_pred = partial_multilabel
        result = compute_multilabel_metrics(y_true, y_pred, LABELS)
        assert result["subset_accuracy"] == pytest.approx(1 / 3)
        assert result["hamming_loss"] == pytest.approx(2 / 9)
        assert result["num_exact_match"] == 1

    @pytest.mark.unit
    def test_counts_and_cardinality(self, partial_multilabel):
        """Test sample counts and label cardinalities are reported."""
        y_true, y_pred = partial_multilabel
        result = compute_multilabel_metrics(y_true, y_pred, LABELS)
        assert result["num_samples"] == 3
        assert result["num_labels"] == 3
        assert result["label_cardinality_true"] == pytest.approx(5 / 3)
        assert result["label_cardinality_pred"] == pytest.approx(1.0)
        assert result["empty_prediction_rate"] == pytest.approx(0.0)

    @pytest.mark.unit
    def test_empty_prediction_rate(self):
        """Test the rate of documents given no labels at all."""
        y_true = binarize_labels([["a"], ["b"]], LABELS)
        y_pred = binarize_labels([["a"], []], LABELS)
        result = compute_multilabel_metrics(y_true, y_pred, LABELS)
        assert result["empty_prediction_rate"] == pytest.approx(0.5)

    @pytest.mark.unit
    def test_ranking_metrics_omitted_without_scores(self, partial_multilabel):
        """Test ranking metrics are absent when no score matrix is given."""
        y_true, y_pred = partial_multilabel
        result = compute_multilabel_metrics(y_true, y_pred, LABELS)
        for key in ("lrap", "map_micro", "map_macro", "coverage_error"):
            assert key not in result

    @pytest.mark.unit
    def test_ranking_metrics_included_with_scores(self, partial_multilabel):
        """Test ranking metrics appear when a score matrix is given."""
        y_true, y_pred = partial_multilabel
        y_score = np.array([[0.9, 0.2, 0.1], [0.8, 0.4, 0.1], [0.2, 0.7, 0.3]])
        result = compute_multilabel_metrics(y_true, y_pred, LABELS, y_score=y_score)
        for key in ("lrap", "map_micro", "map_macro", "coverage_error"):
            assert key in result
        assert 0.0 <= result["lrap"] <= 1.0

    @pytest.mark.unit
    def test_per_label_included_by_default(self, partial_multilabel):
        """Test the per-label breakdown and label order are returned."""
        y_true, y_pred = partial_multilabel
        result = compute_multilabel_metrics(y_true, y_pred, LABELS)
        assert set(result["per_label"].keys()) == set(LABELS)
        assert result["labels"] == LABELS

    @pytest.mark.unit
    def test_per_label_can_be_disabled(self, partial_multilabel):
        """Test the per-label breakdown can be skipped."""
        y_true, y_pred = partial_multilabel
        result = compute_multilabel_metrics(
            y_true, y_pred, LABELS, compute_per_label=False
        )
        assert "per_label" not in result
        assert "labels" not in result

    @pytest.mark.unit
    def test_perfect_predictions(self, perfect_multilabel):
        """Test perfect predictions score 1.0 across the board."""
        y_true, y_pred = perfect_multilabel
        result = compute_multilabel_metrics(y_true, y_pred, LABELS)
        assert result["micro_f1"] == pytest.approx(1.0)
        assert result["macro_f1"] == pytest.approx(1.0)
        assert result["subset_accuracy"] == pytest.approx(1.0)
        assert result["hamming_loss"] == pytest.approx(0.0)

    @pytest.mark.unit
    def test_empty_input_raises(self):
        """Test an empty corpus raises rather than returning nan metrics."""
        empty = binarize_labels([], LABELS)
        with pytest.raises(ValueError, match="cannot be empty"):
            compute_multilabel_metrics(empty, empty, LABELS)

    @pytest.mark.unit
    def test_shape_mismatch_raises(self):
        """Test misaligned prediction and truth matrices raise."""
        y_true = binarize_labels([["a"], ["b"]], LABELS)
        y_pred = binarize_labels([["a"]], LABELS)
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_multilabel_metrics(y_true, y_pred, LABELS)

    @pytest.mark.unit
    def test_label_count_mismatch_raises(self, partial_multilabel):
        """Test a vocabulary that does not match the matrix width raises."""
        y_true, y_pred = partial_multilabel
        with pytest.raises(ValueError, match="must match matrix columns"):
            compute_multilabel_metrics(y_true, y_pred, ["a", "b"])

    @pytest.mark.unit
    def test_score_shape_mismatch_raises(self, partial_multilabel):
        """Test a score matrix of the wrong shape raises."""
        y_true, y_pred = partial_multilabel
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_multilabel_metrics(y_true, y_pred, LABELS, y_score=np.zeros((2, 3)))

    @pytest.mark.unit
    def test_all_empty_predictions_does_not_raise(self):
        """Test the aggregate survives a model that predicts nothing."""
        y_true = binarize_labels([["a"], ["b", "c"]], LABELS)
        y_pred = binarize_labels([[], []], LABELS)
        result = compute_multilabel_metrics(y_true, y_pred, LABELS)
        assert result["micro_f1"] == pytest.approx(0.0)
        assert result["macro_f1"] == pytest.approx(0.0)
        assert result["subset_accuracy"] == pytest.approx(0.0)
        assert result["empty_prediction_rate"] == pytest.approx(1.0)

    @pytest.mark.unit
    def test_all_labels_predicted_for_every_row(self):
        """Test predicting every label gives perfect recall, poor precision."""
        y_true = binarize_labels([["a"], ["b"]], LABELS)
        y_pred = binarize_labels([["a", "b", "c"], ["a", "b", "c"]], LABELS)
        result = compute_multilabel_metrics(y_true, y_pred, LABELS)
        assert result["subset_accuracy"] == pytest.approx(0.0)
        assert result["label_cardinality_pred"] == pytest.approx(3.0)
        assert result["per_label"]["a"]["recall"] == pytest.approx(1.0)
        assert result["per_label"]["a"]["precision"] == pytest.approx(0.5)
