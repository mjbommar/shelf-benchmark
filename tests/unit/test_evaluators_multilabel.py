"""Unit tests for MultiLabelClassificationEvaluator.

Tests cover:
1. validate_multilabel_predictions()
2. MultiLabelClassificationEvaluator initialization
3. evaluate() with perfect / partial / degenerate predictions
4. Per-label metrics and per-sample capture
5. Bootstrap confidence intervals (compute_ci=True)
6. evaluate_embedder_with_classifier() with a mock embedder
7. Multi-label edge cases: empty label sets, single-label rows,
   all-labels rows, a label that never appears in train
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest
from shelf.evaluate.evaluators.multilabel import (
    MultiLabelClassificationEvaluator,
    validate_multilabel_predictions,
)
from shelf.evaluate.results import EvaluationResult
from shelf.evaluate.schemas import ValidationError
from shelf.evaluate.tasks import TaskSpec, TaskType

TOPIC_SPACE = ("Art", "Ethics", "Physics", "Rare")


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def multilabel_task_spec():
    """Create a sample multi-label task spec."""
    return TaskSpec(
        name="test_topic_classification",
        task_type=TaskType.MULTILABEL,
        description="Test multi-label topic classification task",
        text_field="text",
        label_field="topics",
        id_field="id",
        label_space=TOPIC_SPACE,
        primary_metric="macro_f1",
        secondary_metrics=("micro_f1", "samples_f1", "subset_accuracy"),
        dataset_name="test/dataset",
        dataset_config="default",
        default_split="test",
    )


@pytest.fixture
def open_vocab_task_spec():
    """Task spec with no declared label space."""
    return TaskSpec(
        name="test_open_topics",
        task_type=TaskType.MULTILABEL,
        description="Open-vocabulary multi-label task",
        text_field="text",
        label_field="topics",
        id_field="id",
        label_space=None,
        primary_metric="macro_f1",
        secondary_metrics=("micro_f1",),
        dataset_name="test/dataset",
        dataset_config="default",
        default_split="test",
    )


@pytest.fixture
def ground_truth_df():
    """Ground truth covering single-label, multi-label and all-label rows."""
    return pl.DataFrame(
        {
            "id": ["doc_001", "doc_002", "doc_003", "doc_004"],
            "text": [
                "A short essay on painting.",
                "A long treatise on moral philosophy and painting. " * 60,
                "Quantum field theory notes.",
                "A survey touching everything at once.",
            ],
            # doc_001: single label, doc_004: all labels
            "topics": [
                ["Art"],
                ["Art", "Ethics"],
                ["Physics"],
                ["Art", "Ethics", "Physics", "Rare"],
            ],
            "register": ["academic", "academic", "technical", "casual"],
            "audience": ["students", "scholars", "scholars", "general"],
            "git_commit": ["abc123"] * 4,
            "model": ["gpt-5.1"] * 4,
        }
    )


@pytest.fixture
def perfect_predictions():
    """Predictions matching the ground truth exactly."""
    return [
        {"id": "doc_001", "predictions": ["Art"]},
        {"id": "doc_002", "predictions": ["Art", "Ethics"]},
        {"id": "doc_003", "predictions": ["Physics"]},
        {"id": "doc_004", "predictions": ["Art", "Ethics", "Physics", "Rare"]},
    ]


@pytest.fixture
def partial_predictions():
    """Predictions with misses, extras and one empty set."""
    return [
        {"id": "doc_001", "predictions": ["Art"]},  # exact
        {"id": "doc_002", "predictions": ["Art"]},  # missed Ethics
        {"id": "doc_003", "predictions": ["Physics", "Art"]},  # extra Art
        {"id": "doc_004", "predictions": []},  # predicted nothing
    ]


# ===========================================================================
# Validation Tests
# ===========================================================================


@pytest.mark.unit
class TestValidateMultilabelPredictions:
    """Tests for validate_multilabel_predictions()."""

    def test_valid_predictions(self, perfect_predictions):
        """Test well-formed predictions validate cleanly."""
        result = validate_multilabel_predictions(
            perfect_predictions,
            ground_truth_ids={"doc_001", "doc_002", "doc_003", "doc_004"},
            label_space=set(TOPIC_SPACE),
        )
        assert result.valid
        assert result.errors == []
        assert result.num_predictions == 4

    def test_empty_prediction_set_is_a_warning_not_an_error(self):
        """Test an empty label set is accepted with a warning."""
        result = validate_multilabel_predictions(
            [{"id": "doc_001", "predictions": []}],
            ground_truth_ids={"doc_001"},
            label_space=set(TOPIC_SPACE),
        )
        assert result.valid
        assert len(result.warnings) == 1
        assert "empty prediction set" in result.warnings[0]

    def test_unknown_label_is_error(self):
        """Test a label outside the label space is rejected."""
        result = validate_multilabel_predictions(
            [{"id": "doc_001", "predictions": ["Chemistry"]}],
            ground_truth_ids={"doc_001"},
            label_space=set(TOPIC_SPACE),
        )
        assert not result.valid
        assert any("invalid label" in e for e in result.errors)

    def test_unknown_label_allowed_without_label_space(self):
        """Test open vocabulary accepts any label."""
        result = validate_multilabel_predictions(
            [{"id": "doc_001", "predictions": ["Chemistry"]}],
            ground_truth_ids={"doc_001"},
            label_space=None,
        )
        assert result.valid

    def test_duplicate_id_is_error(self):
        """Test the same document predicted twice is rejected."""
        result = validate_multilabel_predictions(
            [
                {"id": "doc_001", "predictions": ["Art"]},
                {"id": "doc_001", "predictions": ["Ethics"]},
            ],
            ground_truth_ids={"doc_001"},
        )
        assert not result.valid
        assert any("duplicate id" in e for e in result.errors)

    def test_duplicate_label_within_row_is_error(self):
        """Test a repeated label inside one prediction set is rejected."""
        result = validate_multilabel_predictions(
            [{"id": "doc_001", "predictions": ["Art", "Art"]}],
            ground_truth_ids={"doc_001"},
        )
        assert not result.valid
        assert any("duplicate labels" in e for e in result.errors)

    def test_unknown_id_is_error(self):
        """Test a prediction for an unknown document is rejected."""
        result = validate_multilabel_predictions(
            [{"id": "nope", "predictions": ["Art"]}],
            ground_truth_ids={"doc_001"},
        )
        assert not result.valid
        assert any("unknown id" in e for e in result.errors)

    def test_missing_predictions_is_error(self):
        """Test documents with no prediction at all are reported."""
        result = validate_multilabel_predictions(
            [{"id": "doc_001", "predictions": ["Art"]}],
            ground_truth_ids={"doc_001", "doc_002"},
        )
        assert not result.valid
        assert any("Missing predictions" in e for e in result.errors)

    def test_malformed_row_is_error(self):
        """Test a structurally invalid row is reported, not raised."""
        result = validate_multilabel_predictions(
            [{"id": 42, "predictions": ["Art"]}],
            ground_truth_ids={"doc_001"},
        )
        assert not result.valid

    def test_empty_set_with_non_string_id_is_error(self):
        """Test an empty prediction set still requires a string id."""
        result = validate_multilabel_predictions(
            [{"id": 42, "predictions": []}],
            ground_truth_ids={"doc_001"},
        )
        assert not result.valid
        assert any("must be a string" in e for e in result.errors)


# ===========================================================================
# Initialization Tests
# ===========================================================================


@pytest.mark.unit
@pytest.mark.evaluator
class TestMultiLabelEvaluatorInit:
    """Test MultiLabelClassificationEvaluator initialization."""

    def test_init_basic(self, multilabel_task_spec):
        """Test basic initialization with task spec."""
        evaluator = MultiLabelClassificationEvaluator(multilabel_task_spec)
        assert evaluator.task_spec == multilabel_task_spec
        assert evaluator.random_seed == 42
        assert evaluator.filter_by == {}
        assert evaluator.stratify_by == []

    def test_init_with_random_seed(self, multilabel_task_spec):
        """Test initialization with custom random seed."""
        evaluator = MultiLabelClassificationEvaluator(
            multilabel_task_spec, random_seed=123
        )
        assert evaluator.random_seed == 123

    def test_init_with_stratify_by_string(self, multilabel_task_spec):
        """Test a single stratify field is normalized to a list."""
        evaluator = MultiLabelClassificationEvaluator(
            multilabel_task_spec, stratify_by="register"
        )
        assert evaluator.stratify_by == ["register"]


# ===========================================================================
# evaluate() Tests
# ===========================================================================


@pytest.mark.unit
@pytest.mark.evaluator
class TestMultiLabelEvaluatorEvaluate:
    """Test MultiLabelClassificationEvaluator.evaluate()."""

    def test_perfect_predictions(
        self, multilabel_task_spec, ground_truth_df, perfect_predictions
    ):
        """Test perfect predictions score 1.0 on every metric."""
        evaluator = MultiLabelClassificationEvaluator(multilabel_task_spec)
        result = evaluator.evaluate(perfect_predictions, ground_truth_df)

        assert isinstance(result, EvaluationResult)
        assert result.primary_metric == "macro_f1"
        assert result.primary_score == pytest.approx(1.0)
        assert result.metrics["micro_f1"] == pytest.approx(1.0)
        assert result.metrics["samples_f1"] == pytest.approx(1.0)
        assert result.metrics["subset_accuracy"] == pytest.approx(1.0)
        assert result.metrics["hamming_loss"] == pytest.approx(0.0)
        assert result.num_correct == 4
        assert result.misclassified_ids == []

    def test_task_type_recorded(
        self, multilabel_task_spec, ground_truth_df, perfect_predictions
    ):
        """Test the result records the multilabel task type."""
        evaluator = MultiLabelClassificationEvaluator(multilabel_task_spec)
        result = evaluator.evaluate(perfect_predictions, ground_truth_df)
        assert result.task_type == "multilabel"

    def test_partial_predictions(
        self, multilabel_task_spec, ground_truth_df, partial_predictions
    ):
        """Test partial predictions score between 0 and 1."""
        evaluator = MultiLabelClassificationEvaluator(multilabel_task_spec)
        result = evaluator.evaluate(partial_predictions, ground_truth_df)

        assert 0.0 < result.metrics["micro_f1"] < 1.0
        # Only doc_001 is an exact set match.
        assert result.metrics["subset_accuracy"] == pytest.approx(0.25)
        assert result.num_correct == 1
        assert result.misclassified_ids is not None
        assert set(result.misclassified_ids) == {"doc_002", "doc_003", "doc_004"}

    def test_empty_prediction_set_counted(
        self, multilabel_task_spec, ground_truth_df, partial_predictions
    ):
        """Test a document given no labels shows in empty_prediction_rate."""
        evaluator = MultiLabelClassificationEvaluator(multilabel_task_spec)
        result = evaluator.evaluate(partial_predictions, ground_truth_df)
        assert result.metrics["empty_prediction_rate"] == pytest.approx(0.25)

    def test_label_cardinalities_reported(
        self, multilabel_task_spec, ground_truth_df, perfect_predictions
    ):
        """Test true and predicted label cardinality are both reported."""
        evaluator = MultiLabelClassificationEvaluator(multilabel_task_spec)
        result = evaluator.evaluate(perfect_predictions, ground_truth_df)
        # 1 + 2 + 1 + 4 = 8 labels over 4 documents
        assert result.metrics["label_cardinality_true"] == pytest.approx(2.0)
        assert result.metrics["label_cardinality_pred"] == pytest.approx(2.0)

    def test_per_label_metrics_cover_full_label_space(
        self, multilabel_task_spec, ground_truth_df, partial_predictions
    ):
        """Test every declared label appears in the per-label breakdown."""
        evaluator = MultiLabelClassificationEvaluator(multilabel_task_spec)
        result = evaluator.evaluate(partial_predictions, ground_truth_df)
        assert result.per_class_metrics is not None
        assert set(result.per_class_metrics.keys()) == set(TOPIC_SPACE)

    def test_label_never_predicted_scores_zero(
        self, multilabel_task_spec, ground_truth_df, partial_predictions
    ):
        """Test a label present in truth but never predicted scores 0.0."""
        evaluator = MultiLabelClassificationEvaluator(multilabel_task_spec)
        result = evaluator.evaluate(partial_predictions, ground_truth_df)
        assert result.per_class_metrics is not None
        rare = result.per_class_metrics["Rare"]
        assert rare["support"] == 1
        assert rare["num_predicted"] == 0
        assert rare["f1"] == pytest.approx(0.0)

    def test_all_empty_predictions(self, multilabel_task_spec, ground_truth_df):
        """Test a model that predicts nothing scores 0 without raising."""
        predictions = [
            {"id": doc_id, "predictions": []}
            for doc_id in ground_truth_df["id"].to_list()
        ]
        evaluator = MultiLabelClassificationEvaluator(multilabel_task_spec)
        result = evaluator.evaluate(predictions, ground_truth_df)

        assert result.metrics["macro_f1"] == pytest.approx(0.0)
        assert result.metrics["micro_f1"] == pytest.approx(0.0)
        assert result.metrics["subset_accuracy"] == pytest.approx(0.0)
        assert result.metrics["empty_prediction_rate"] == pytest.approx(1.0)

    def test_all_labels_predicted_everywhere(
        self, multilabel_task_spec, ground_truth_df
    ):
        """Test predicting the whole vocabulary gives perfect recall."""
        predictions = [
            {"id": doc_id, "predictions": list(TOPIC_SPACE)}
            for doc_id in ground_truth_df["id"].to_list()
        ]
        evaluator = MultiLabelClassificationEvaluator(multilabel_task_spec)
        result = evaluator.evaluate(predictions, ground_truth_df)

        assert result.metrics["label_cardinality_pred"] == pytest.approx(4.0)
        # Only doc_004 (which carries every label) is an exact match.
        assert result.metrics["subset_accuracy"] == pytest.approx(0.25)
        assert result.per_class_metrics is not None
        for label in TOPIC_SPACE:
            assert result.per_class_metrics[label]["recall"] == pytest.approx(1.0)

    def test_open_vocabulary_infers_labels_from_data(
        self, open_vocab_task_spec, ground_truth_df, perfect_predictions
    ):
        """Test the observed union is used when no label space is declared."""
        evaluator = MultiLabelClassificationEvaluator(open_vocab_task_spec)
        result = evaluator.evaluate(perfect_predictions, ground_truth_df)
        assert result.per_class_metrics is not None
        assert set(result.per_class_metrics.keys()) == set(TOPIC_SPACE)
        assert result.metrics["num_labels"] == 4

    def test_invalid_predictions_raise(
        self, multilabel_task_spec, ground_truth_df, perfect_predictions
    ):
        """Test validation errors surface as ValidationError."""
        bad = list(perfect_predictions)
        bad[0] = {"id": "doc_001", "predictions": ["Chemistry"]}
        evaluator = MultiLabelClassificationEvaluator(multilabel_task_spec)
        with pytest.raises(ValidationError):
            evaluator.evaluate(bad, ground_truth_df)

    def test_missing_predictions_raise(
        self, multilabel_task_spec, ground_truth_df, perfect_predictions
    ):
        """Test omitting a document raises rather than silently scoring."""
        evaluator = MultiLabelClassificationEvaluator(multilabel_task_spec)
        with pytest.raises(ValidationError):
            evaluator.evaluate(perfect_predictions[:2], ground_truth_df)

    def test_save_samples_captures_sorted_label_sets(
        self, multilabel_task_spec, ground_truth_df, partial_predictions
    ):
        """Test per-sample capture stores sorted label sets and metadata."""
        evaluator = MultiLabelClassificationEvaluator(multilabel_task_spec)
        result = evaluator.evaluate(
            partial_predictions,
            ground_truth_df,
            save_samples=True,
            model_key="test-model",
        )

        samples = result.per_sample_results
        assert samples is not None
        assert samples.model_key == "test-model"
        assert samples.task_type == "multilabel"
        assert len(samples.samples) == 4

        by_id = {s.id: s for s in samples.samples}
        assert by_id["doc_002"].y_true == ["Art", "Ethics"]
        assert by_id["doc_002"].y_pred == ["Art"]
        assert by_id["doc_002"].correct is False
        assert by_id["doc_001"].correct is True
        # "register" and "audience" are known stratification fields.
        assert by_id["doc_001"].metadata["register"] == "academic"
        assert by_id["doc_001"].metadata["length_bucket"] == "short"
        assert by_id["doc_002"].metadata["length_bucket"] == "long"

    def test_compute_ci_returns_intervals(
        self, multilabel_task_spec, ground_truth_df, partial_predictions
    ):
        """Test bootstrap CIs are produced for the F1 family."""
        evaluator = MultiLabelClassificationEvaluator(multilabel_task_spec)
        result = evaluator.evaluate(
            partial_predictions, ground_truth_df, compute_ci=True
        )

        cis = result.confidence_intervals
        assert cis is not None
        for key in ("micro_f1", "macro_f1", "samples_f1", "subset_accuracy"):
            lower, upper = cis[key]
            assert 0.0 <= lower <= upper <= 1.0

    def test_compute_ci_is_deterministic(
        self, multilabel_task_spec, ground_truth_df, partial_predictions
    ):
        """Test the same seed gives identical confidence intervals."""
        results = [
            MultiLabelClassificationEvaluator(
                multilabel_task_spec, random_seed=7
            ).evaluate(partial_predictions, ground_truth_df, compute_ci=True)
            for _ in range(2)
        ]
        assert results[0].confidence_intervals == results[1].confidence_intervals

    def test_ci_perfect_predictions_are_degenerate(
        self, multilabel_task_spec, ground_truth_df, perfect_predictions
    ):
        """Test perfect predictions give (1.0, 1.0) on every resample.

        This pins the closed-form contingency math the vectorized bootstrap
        uses against the metric functions it replaces.
        """
        evaluator = MultiLabelClassificationEvaluator(multilabel_task_spec)
        result = evaluator.evaluate(
            perfect_predictions, ground_truth_df, compute_ci=True
        )
        cis = result.confidence_intervals
        assert cis is not None
        for key in ("micro_f1", "samples_f1", "subset_accuracy"):
            assert cis[key] == (pytest.approx(1.0), pytest.approx(1.0))
        # macro_f1 is NOT degenerate here: "Rare" occurs in a single document,
        # so a resample that misses it scores that label 0.0 - which is the
        # same behaviour the non-bootstrapped macro_f1 has.
        assert cis["macro_f1"][1] == pytest.approx(1.0)
        assert cis["macro_f1"][0] < 1.0

    def test_ci_all_empty_predictions_are_degenerate(
        self, multilabel_task_spec, ground_truth_df
    ):
        """Test a model predicting nothing gives (0.0, 0.0) on every resample."""
        predictions = [
            {"id": doc_id, "predictions": []}
            for doc_id in ground_truth_df["id"].to_list()
        ]
        evaluator = MultiLabelClassificationEvaluator(multilabel_task_spec)
        result = evaluator.evaluate(predictions, ground_truth_df, compute_ci=True)
        cis = result.confidence_intervals
        assert cis is not None
        for key in ("micro_f1", "macro_f1", "samples_f1", "subset_accuracy"):
            assert cis[key] == (pytest.approx(0.0), pytest.approx(0.0))

    def test_ci_brackets_point_estimate(self, multilabel_task_spec):
        """Test the bootstrap interval contains the observed score."""
        rng = np.random.default_rng(0)
        n = 200
        ids = [f"doc_{i:04d}" for i in range(n)]
        truth = [
            sorted(rng.choice(TOPIC_SPACE, size=rng.integers(1, 4), replace=False))
            for _ in range(n)
        ]
        # Predictions agree with the truth about 70% of the time.
        preds = [
            {
                "id": doc_id,
                "predictions": labels
                if rng.random() < 0.7
                else [str(rng.choice(TOPIC_SPACE))],
            }
            for doc_id, labels in zip(ids, truth)
        ]
        gt = pl.DataFrame(
            {"id": ids, "text": ["text"] * n, "topics": [list(t) for t in truth]}
        )

        evaluator = MultiLabelClassificationEvaluator(multilabel_task_spec)
        result = evaluator.evaluate(preds, gt, compute_ci=True)
        cis = result.confidence_intervals
        assert cis is not None
        for key in ("micro_f1", "macro_f1", "samples_f1", "subset_accuracy"):
            lower, upper = cis[key]
            assert lower <= result.metrics[key] <= upper

    def test_ci_block_size_does_not_change_replicate_count(
        self, multilabel_task_spec, ground_truth_df, partial_predictions
    ):
        """Test a block size larger than n_bootstrap still produces an interval."""
        evaluator = MultiLabelClassificationEvaluator(multilabel_task_spec)
        y_true = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [1, 1, 1, 1]])
        y_pred = np.array([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 1, 0], [0, 0, 0, 0]])
        cis = evaluator._bootstrap_confidence_intervals(
            y_true, y_pred, n_bootstrap=50, block_size=500
        )
        for key in ("micro_f1", "macro_f1", "samples_f1", "subset_accuracy"):
            lower, upper = cis[key]
            assert 0.0 <= lower <= upper <= 1.0

    def test_no_ci_by_default(
        self, multilabel_task_spec, ground_truth_df, perfect_predictions
    ):
        """Test confidence intervals are opt-in."""
        evaluator = MultiLabelClassificationEvaluator(multilabel_task_spec)
        result = evaluator.evaluate(perfect_predictions, ground_truth_df)
        assert result.confidence_intervals is None

    def test_null_ground_truth_labels_treated_as_empty(self, multilabel_task_spec):
        """Test a null topics cell is scored as an empty label set."""
        gt = pl.DataFrame(
            {
                "id": ["doc_001", "doc_002"],
                "text": ["one", "two"],
                "topics": [["Art"], None],
            }
        )
        predictions = [
            {"id": "doc_001", "predictions": ["Art"]},
            {"id": "doc_002", "predictions": []},
        ]
        evaluator = MultiLabelClassificationEvaluator(multilabel_task_spec)
        result = evaluator.evaluate(predictions, gt)
        assert result.metrics["subset_accuracy"] == pytest.approx(1.0)


# ===========================================================================
# evaluate_embedder_with_classifier() Tests
# ===========================================================================


def _mock_embedder(dim: int = 8):
    """Build a mock embedder returning deterministic random embeddings."""
    embedder = MagicMock()
    embedder.model_name = "MockEmbedder"
    embedder.embedding_dim = dim
    rng = np.random.default_rng(0)

    def encode(texts, batch_size=32, show_progress=True):
        return rng.standard_normal((len(texts), dim))

    embedder.encode = encode
    return embedder


@pytest.fixture
def train_df():
    """Training frame where 'Rare' never appears as a positive example."""
    return pl.DataFrame(
        {
            "id": [f"train_{i:03d}" for i in range(8)],
            "text": [f"training document {i}" for i in range(8)],
            "topics": [
                ["Art"],
                ["Art", "Ethics"],
                ["Ethics"],
                ["Physics"],
                ["Art"],
                ["Physics", "Ethics"],
                ["Art"],
                ["Physics"],
            ],
            "register": ["academic"] * 8,
        }
    )


@pytest.mark.unit
@pytest.mark.evaluator
class TestMultiLabelEvaluatorEmbedder:
    """Test evaluate_embedder_with_classifier()."""

    def test_basic_embedder_evaluation(
        self, multilabel_task_spec, train_df, ground_truth_df
    ):
        """Test the frozen-embedding + OvR-logreg protocol runs end to end."""
        evaluator = MultiLabelClassificationEvaluator(multilabel_task_spec)
        with patch.object(
            evaluator, "_load_ground_truth", side_effect=[train_df, ground_truth_df]
        ):
            result = evaluator.evaluate_embedder_with_classifier(_mock_embedder())

        assert isinstance(result, EvaluationResult)
        assert result.context is not None
        assert result.context.extra["embedding_dim"] == 8
        assert (
            result.context.extra["classifier"]
            == "OneVsRestClassifier(LogisticRegression)"
        )
        assert result.context.extra["train_size"] == 8
        assert result.metrics["num_samples"] == 4
        assert result.metrics["num_labels"] == 4

    def test_label_absent_from_train_is_always_negative(
        self, multilabel_task_spec, train_df, ground_truth_df
    ):
        """Test a label with no positive train example scores F1 of 0.0.

        It must still occupy a column so it counts against macro-F1 rather
        than being silently dropped from the denominator.
        """
        evaluator = MultiLabelClassificationEvaluator(multilabel_task_spec)
        with patch.object(
            evaluator, "_load_ground_truth", side_effect=[train_df, ground_truth_df]
        ):
            result = evaluator.evaluate_embedder_with_classifier(_mock_embedder())

        assert result.per_class_metrics is not None
        rare = result.per_class_metrics["Rare"]
        assert rare["num_predicted"] == 0
        assert rare["f1"] == pytest.approx(0.0)
        assert result.metrics["num_labels"] == 4

    def test_ranking_metrics_present(
        self, multilabel_task_spec, train_df, ground_truth_df
    ):
        """Test threshold-free ranking metrics are computed from scores."""
        evaluator = MultiLabelClassificationEvaluator(multilabel_task_spec)
        with patch.object(
            evaluator, "_load_ground_truth", side_effect=[train_df, ground_truth_df]
        ):
            result = evaluator.evaluate_embedder_with_classifier(_mock_embedder())

        for key in ("lrap", "map_micro", "map_macro", "coverage_error"):
            assert key in result.metrics

    def test_save_samples(self, multilabel_task_spec, train_df, ground_truth_df):
        """Test per-sample capture works in the embedder path."""
        evaluator = MultiLabelClassificationEvaluator(multilabel_task_spec)
        with patch.object(
            evaluator, "_load_ground_truth", side_effect=[train_df, ground_truth_df]
        ):
            result = evaluator.evaluate_embedder_with_classifier(
                _mock_embedder(), save_samples=True
            )

        assert result.per_sample_results is not None
        assert len(result.per_sample_results.samples) == 4
        for sample in result.per_sample_results.samples:
            assert isinstance(sample.y_true, list)
            assert isinstance(sample.y_pred, list)

    def test_compute_ci(self, multilabel_task_spec, train_df, ground_truth_df):
        """Test CIs can be requested from the embedder path."""
        evaluator = MultiLabelClassificationEvaluator(multilabel_task_spec)
        with patch.object(
            evaluator, "_load_ground_truth", side_effect=[train_df, ground_truth_df]
        ):
            result = evaluator.evaluate_embedder_with_classifier(
                _mock_embedder(), compute_ci=True
            )
        assert result.confidence_intervals is not None

    def test_random_forest_head(self, multilabel_task_spec, train_df, ground_truth_df):
        """Test the random forest head is supported."""
        evaluator = MultiLabelClassificationEvaluator(multilabel_task_spec)
        with patch.object(
            evaluator, "_load_ground_truth", side_effect=[train_df, ground_truth_df]
        ):
            result = evaluator.evaluate_embedder_with_classifier(
                _mock_embedder(),
                classifier="random_forest",
                classifier_params={"n_estimators": 5},
            )
        assert result.context is not None
        assert (
            result.context.extra["classifier"]
            == "OneVsRestClassifier(RandomForestClassifier)"
        )

    def test_unknown_classifier_raises(
        self, multilabel_task_spec, train_df, ground_truth_df
    ):
        """Test an unsupported head name raises."""
        evaluator = MultiLabelClassificationEvaluator(multilabel_task_spec)
        with (
            patch.object(
                evaluator, "_load_ground_truth", side_effect=[train_df, ground_truth_df]
            ),
            pytest.raises(ValueError, match="Unsupported classifier type"),
        ):
            evaluator.evaluate_embedder_with_classifier(
                _mock_embedder(), classifier="svm"
            )

    def test_no_positive_training_example_raises(
        self, multilabel_task_spec, ground_truth_df
    ):
        """Test a training split with no positive label at all raises."""
        empty_train = pl.DataFrame(
            {
                "id": ["train_000", "train_001"],
                "text": ["a", "b"],
                "topics": [[], []],
            }
        )
        evaluator = MultiLabelClassificationEvaluator(multilabel_task_spec)
        with (
            patch.object(
                evaluator,
                "_load_ground_truth",
                side_effect=[empty_train, ground_truth_df],
            ),
            pytest.raises(ValueError, match="No label has a positive"),
        ):
            evaluator.evaluate_embedder_with_classifier(_mock_embedder())

    def test_wrong_train_embedding_count_raises(
        self, multilabel_task_spec, train_df, ground_truth_df
    ):
        """Test a mismatched train embedding count raises."""
        embedder = MagicMock()
        embedder.model_name = "MockEmbedder"
        embedder.embedding_dim = 8
        embedder.encode = lambda texts, batch_size=32, show_progress=True: np.zeros(
            (2, 8)
        )

        evaluator = MultiLabelClassificationEvaluator(multilabel_task_spec)
        with (
            patch.object(
                evaluator, "_load_ground_truth", side_effect=[train_df, ground_truth_df]
            ),
            pytest.raises(ValueError, match="train embeddings for 8 texts"),
        ):
            evaluator.evaluate_embedder_with_classifier(embedder)

    def test_wrong_test_embedding_count_raises(
        self, multilabel_task_spec, train_df, ground_truth_df
    ):
        """Test a mismatched test embedding count raises."""
        embedder = MagicMock()
        embedder.model_name = "MockEmbedder"
        embedder.embedding_dim = 8
        calls = [0]

        def encode(texts, batch_size=32, show_progress=True):
            calls[0] += 1
            return np.zeros((len(texts) if calls[0] == 1 else 2, 8))

        embedder.encode = encode

        evaluator = MultiLabelClassificationEvaluator(multilabel_task_spec)
        with (
            patch.object(
                evaluator, "_load_ground_truth", side_effect=[train_df, ground_truth_df]
            ),
            pytest.raises(ValueError, match="test embeddings for 4 texts"),
        ):
            evaluator.evaluate_embedder_with_classifier(embedder)
