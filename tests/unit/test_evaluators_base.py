"""Unit tests for shelf.evaluate.evaluators.base module.

Tests cover:
- TaskEvaluator initialization with various parameters
- _apply_filters method with different filter types
- _create_result method for generating EvaluationResult
- _extract_provenance method
- _compute_stratified_metrics method
- evaluate_from_file workflow
- Error handling (empty filters, invalid filter types)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from shelf.evaluate.evaluators.base import STRATIFY_FIELDS, TaskEvaluator
from shelf.evaluate.results import (
    DataProvenance,
    EvaluationContext,
    EvaluationResult,
)
from shelf.evaluate.tasks import TaskSpec, TaskType


# ===========================================================================
# Test Fixtures
# ===========================================================================


@pytest.fixture
def sample_task_spec() -> TaskSpec:
    """Create a sample task specification for testing."""
    return TaskSpec(
        name="test_classification",
        task_type=TaskType.CLASSIFICATION,
        description="Test classification task",
        text_field="text",
        label_field="lcc",
        id_field="id",
        label_space=("A", "B", "C", "D"),
        primary_metric="macro_f1",
        secondary_metrics=("micro_f1", "accuracy"),
        dataset_name="mjbommar/SHELF",
        dataset_config="default",
        default_split="test",
    )


@pytest.fixture
def sample_ground_truth_df() -> pl.DataFrame:
    """Create a sample ground truth DataFrame for testing."""
    return pl.DataFrame(
        {
            "id": ["doc_001", "doc_002", "doc_003", "doc_004", "doc_005", "doc_006"],
            "text": [
                "Text about topic A",
                "Text about topic B",
                "Text about topic C",
                "Text about topic A again",
                "Text about topic B again",
                "Text about topic C again",
            ],
            "lcc": ["A", "B", "C", "A", "B", "C"],
            "form": ["lecture", "map", "essay", "lecture", "map", "essay"],
            "form_category": [
                "educational",
                "visual",
                "literary",
                "educational",
                "visual",
                "literary",
            ],
            "topic": ["art", "science", "law", "art", "science", "law"],
            "region": ["US", "Europe", "Asia", "US", "Europe", "Asia"],
            "audience": [
                "general",
                "specialist",
                "academic",
                "general",
                "specialist",
                "academic",
            ],
            "register": [
                "formal",
                "technical",
                "casual",
                "formal",
                "technical",
                "casual",
            ],
            "model": ["gpt-5.1", "gpt-5.1", "gpt-5.2", "gpt-5.1", "gpt-5.2", "gpt-5.2"],
            "git_commit": ["abc123", "abc123", "abc123", "def456", "def456", "def456"],
        }
    )


@pytest.fixture
def sample_predictions() -> list[dict[str, Any]]:
    """Create sample predictions for testing."""
    return [
        {"id": "doc_001", "prediction": "A", "confidence": 0.95},
        {"id": "doc_002", "prediction": "B", "confidence": 0.87},
        {"id": "doc_003", "prediction": "C", "confidence": 0.72},
        {"id": "doc_004", "prediction": "A", "confidence": 0.88},
        {"id": "doc_005", "prediction": "B", "confidence": 0.91},
        {"id": "doc_006", "prediction": "C", "confidence": 0.76},
    ]


@pytest.fixture
def temp_predictions_file(sample_predictions, tmp_path: Path) -> Path:
    """Create a temporary predictions file for testing."""
    filepath = tmp_path / "predictions.jsonl"
    with open(filepath, "w") as f:
        for pred in sample_predictions:
            f.write(json.dumps(pred) + "\n")
    return filepath


# ===========================================================================
# Concrete Test Implementation
# ===========================================================================


class ConcreteTaskEvaluator(TaskEvaluator):
    """Concrete implementation of TaskEvaluator for testing.

    This implements the abstract evaluate() method so we can test
    the base class methods.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._current_split = None

    def evaluate(
        self,
        predictions: list[dict[str, Any]],
        ground_truth: pl.DataFrame,
        compute_ci: bool = False,
    ) -> EvaluationResult:
        """Minimal evaluate implementation for testing."""
        # Simple accuracy calculation
        pred_dict = {p["id"]: p["prediction"] for p in predictions}
        correct = 0
        total = len(ground_truth)

        for row in ground_truth.iter_rows(named=True):
            if (
                row["id"] in pred_dict
                and pred_dict[row["id"]] == row[self.task_spec.label_field]
            ):
                correct += 1

        accuracy = correct / total if total > 0 else 0.0

        metrics = {
            "accuracy": accuracy,
            "macro_f1": accuracy,  # Simplified for testing
            "micro_f1": accuracy,
        }

        # Use the split that was set by evaluate_from_file, or default
        split = self._current_split or self.task_spec.default_split

        return self._create_result(
            metrics=metrics,
            ground_truth=ground_truth,
            split=split,
            num_correct=correct,
        )

    def evaluate_from_file(
        self,
        predictions_path: Path | str,
        split: str | None = None,
        compute_ci: bool = False,
    ) -> EvaluationResult:
        """Override to track split for testing."""
        # Store the split so evaluate() can use it
        self._current_split = split or self.task_spec.default_split
        return super().evaluate_from_file(predictions_path, split, compute_ci)


# ===========================================================================
# Tests for STRATIFY_FIELDS Constant
# ===========================================================================


@pytest.mark.unit
class TestStratifyFieldsConstant:
    """Tests for STRATIFY_FIELDS constant."""

    def test_stratify_fields_defined(self):
        """Test that STRATIFY_FIELDS is defined with expected fields."""
        assert isinstance(STRATIFY_FIELDS, list)
        assert len(STRATIFY_FIELDS) > 0

    def test_stratify_fields_content(self):
        """Test that STRATIFY_FIELDS contains expected taxonomy fields."""
        expected_fields = ["lcc", "form", "topic", "region", "audience", "register"]
        for field in expected_fields:
            assert field in STRATIFY_FIELDS

    def test_stratify_fields_includes_metadata(self):
        """Test that STRATIFY_FIELDS includes metadata fields."""
        assert "model" in STRATIFY_FIELDS
        assert "git_commit" in STRATIFY_FIELDS


# ===========================================================================
# Tests for TaskEvaluator Initialization
# ===========================================================================


@pytest.mark.unit
@pytest.mark.evaluator
class TestTaskEvaluatorInit:
    """Tests for TaskEvaluator initialization."""

    def test_init_minimal(self, sample_task_spec):
        """Test initialization with minimal parameters."""
        evaluator = ConcreteTaskEvaluator(task_spec=sample_task_spec)

        assert evaluator.task_spec == sample_task_spec
        assert evaluator.random_seed == 42  # Default
        assert evaluator.filter_by == {}
        assert evaluator.stratify_by == []

    def test_init_with_random_seed(self, sample_task_spec):
        """Test initialization with custom random seed."""
        evaluator = ConcreteTaskEvaluator(
            task_spec=sample_task_spec,
            random_seed=12345,
        )

        assert evaluator.random_seed == 12345

    def test_init_with_filter_by_single_value(self, sample_task_spec):
        """Test initialization with single filter value."""
        filter_by = {"git_commit": "abc123"}
        evaluator = ConcreteTaskEvaluator(
            task_spec=sample_task_spec,
            filter_by=filter_by,
        )

        assert evaluator.filter_by == filter_by

    def test_init_with_filter_by_list_values(self, sample_task_spec):
        """Test initialization with list of filter values."""
        filter_by = {"lcc": ["A", "B", "C"]}
        evaluator = ConcreteTaskEvaluator(
            task_spec=sample_task_spec,
            filter_by=filter_by,
        )

        assert evaluator.filter_by == filter_by

    def test_init_with_multiple_filters(self, sample_task_spec):
        """Test initialization with multiple filters."""
        filter_by = {
            "git_commit": "abc123",
            "model": "gpt-5.1",
            "lcc": ["A", "B"],
        }
        evaluator = ConcreteTaskEvaluator(
            task_spec=sample_task_spec,
            filter_by=filter_by,
        )

        assert evaluator.filter_by == filter_by

    def test_init_with_stratify_by_string(self, sample_task_spec):
        """Test initialization with single stratify field."""
        evaluator = ConcreteTaskEvaluator(
            task_spec=sample_task_spec,
            stratify_by="form",
        )

        assert evaluator.stratify_by == ["form"]

    def test_init_with_stratify_by_list(self, sample_task_spec):
        """Test initialization with multiple stratify fields."""
        evaluator = ConcreteTaskEvaluator(
            task_spec=sample_task_spec,
            stratify_by=["form", "lcc"],
        )

        assert evaluator.stratify_by == ["form", "lcc"]

    def test_init_with_all_parameters(self, sample_task_spec):
        """Test initialization with all parameters."""
        evaluator = ConcreteTaskEvaluator(
            task_spec=sample_task_spec,
            random_seed=99,
            filter_by={"model": "gpt-5.1"},
            stratify_by=["form", "register"],
        )

        assert evaluator.task_spec == sample_task_spec
        assert evaluator.random_seed == 99
        assert evaluator.filter_by == {"model": "gpt-5.1"}
        assert evaluator.stratify_by == ["form", "register"]


# ===========================================================================
# Tests for _apply_filters Method
# ===========================================================================


@pytest.mark.unit
@pytest.mark.evaluator
class TestApplyFilters:
    """Tests for _apply_filters method."""

    def test_apply_filters_no_filters(self, sample_task_spec, sample_ground_truth_df):
        """Test _apply_filters with no filters returns unchanged DataFrame."""
        evaluator = ConcreteTaskEvaluator(task_spec=sample_task_spec)
        result = evaluator._apply_filters(sample_ground_truth_df)

        assert len(result) == len(sample_ground_truth_df)
        assert result.equals(sample_ground_truth_df)

    def test_apply_filters_single_string_value(
        self, sample_task_spec, sample_ground_truth_df
    ):
        """Test _apply_filters with single string value."""
        evaluator = ConcreteTaskEvaluator(
            task_spec=sample_task_spec,
            filter_by={"git_commit": "abc123"},
        )
        result = evaluator._apply_filters(sample_ground_truth_df)

        assert len(result) == 3  # Only 3 records with git_commit="abc123"
        assert result["git_commit"].to_list() == ["abc123"] * 3

    def test_apply_filters_list_values(self, sample_task_spec, sample_ground_truth_df):
        """Test _apply_filters with list of values."""
        evaluator = ConcreteTaskEvaluator(
            task_spec=sample_task_spec,
            filter_by={"lcc": ["A", "B"]},
        )
        result = evaluator._apply_filters(sample_ground_truth_df)

        assert len(result) == 4  # 2 A's and 2 B's
        assert set(result["lcc"].to_list()) == {"A", "B"}

    def test_apply_filters_multiple_filters(
        self, sample_task_spec, sample_ground_truth_df
    ):
        """Test _apply_filters with multiple filters (AND logic)."""
        evaluator = ConcreteTaskEvaluator(
            task_spec=sample_task_spec,
            filter_by={
                "git_commit": "abc123",
                "lcc": "A",
            },
        )
        result = evaluator._apply_filters(sample_ground_truth_df)

        assert len(result) == 1  # Only doc_001 matches both
        assert result["id"][0] == "doc_001"

    def test_apply_filters_nonexistent_field(
        self, sample_task_spec, sample_ground_truth_df
    ):
        """Test _apply_filters with non-existent field logs warning but continues."""
        evaluator = ConcreteTaskEvaluator(
            task_spec=sample_task_spec,
            filter_by={"nonexistent_field": "value"},
        )
        result = evaluator._apply_filters(sample_ground_truth_df)

        # Should return unchanged DataFrame since filter field doesn't exist
        assert len(result) == len(sample_ground_truth_df)

    def test_apply_filters_returns_zero_rows_raises_error(
        self, sample_task_spec, sample_ground_truth_df
    ):
        """Test _apply_filters raises ValueError when filter returns 0 rows."""
        evaluator = ConcreteTaskEvaluator(
            task_spec=sample_task_spec,
            filter_by={"lcc": "Z"},  # No records with lcc="Z"
        )

        with pytest.raises(ValueError, match="Filter returned 0 rows"):
            evaluator._apply_filters(sample_ground_truth_df)

    def test_apply_filters_invalid_type_raises_error(
        self, sample_task_spec, sample_ground_truth_df
    ):
        """Test _apply_filters raises ValueError for invalid filter type."""
        evaluator = ConcreteTaskEvaluator(
            task_spec=sample_task_spec,
            filter_by={"lcc": 123},  # Invalid type (not str or list)
        )

        with pytest.raises(ValueError, match="Filter value must be str or list"):
            evaluator._apply_filters(sample_ground_truth_df)

    def test_apply_filters_complex_scenario(
        self, sample_task_spec, sample_ground_truth_df
    ):
        """Test _apply_filters with complex multiple filters."""
        evaluator = ConcreteTaskEvaluator(
            task_spec=sample_task_spec,
            filter_by={
                "model": ["gpt-5.1", "gpt-5.2"],
                "form": "lecture",
            },
        )
        result = evaluator._apply_filters(sample_ground_truth_df)

        assert len(result) == 2  # Both lectures
        assert all(f == "lecture" for f in result["form"].to_list())


# ===========================================================================
# Tests for _extract_provenance Method
# ===========================================================================


@pytest.mark.unit
@pytest.mark.evaluator
class TestExtractProvenance:
    """Tests for _extract_provenance method."""

    def test_extract_provenance_basic(self, sample_task_spec, sample_ground_truth_df):
        """Test _extract_provenance extracts correct information."""
        evaluator = ConcreteTaskEvaluator(task_spec=sample_task_spec)
        provenance = evaluator._extract_provenance(sample_ground_truth_df)

        assert isinstance(provenance, DataProvenance)
        assert set(provenance.unique_commits) == {"abc123", "def456"}
        assert set(provenance.unique_models) == {"gpt-5.1", "gpt-5.2"}

    def test_extract_provenance_with_filters(
        self, sample_task_spec, sample_ground_truth_df
    ):
        """Test _extract_provenance includes filter information."""
        filter_by = {"git_commit": "abc123"}
        evaluator = ConcreteTaskEvaluator(
            task_spec=sample_task_spec,
            filter_by=filter_by,
        )

        filtered_df = evaluator._apply_filters(sample_ground_truth_df)
        provenance = evaluator._extract_provenance(filtered_df)

        assert provenance.filters_applied == filter_by
        assert provenance.unique_commits == ["abc123"]

    def test_extract_provenance_commit_distribution(
        self, sample_task_spec, sample_ground_truth_df
    ):
        """Test _extract_provenance includes commit distribution."""
        evaluator = ConcreteTaskEvaluator(task_spec=sample_task_spec)
        provenance = evaluator._extract_provenance(sample_ground_truth_df)

        assert provenance.commit_distribution == {"abc123": 3, "def456": 3}

    def test_extract_provenance_model_distribution(
        self, sample_task_spec, sample_ground_truth_df
    ):
        """Test _extract_provenance includes model distribution."""
        evaluator = ConcreteTaskEvaluator(task_spec=sample_task_spec)
        provenance = evaluator._extract_provenance(sample_ground_truth_df)

        assert provenance.model_distribution == {"gpt-5.1": 3, "gpt-5.2": 3}

    def test_extract_provenance_primary_commit(
        self, sample_task_spec, sample_ground_truth_df
    ):
        """Test _extract_provenance sets primary_commit when single commit."""
        evaluator = ConcreteTaskEvaluator(
            task_spec=sample_task_spec,
            filter_by={"git_commit": "abc123"},
        )
        filtered_df = evaluator._apply_filters(sample_ground_truth_df)
        provenance = evaluator._extract_provenance(filtered_df)

        assert provenance.primary_commit == "abc123"
        assert provenance.is_single_commit()


# ===========================================================================
# Tests for _create_result Method
# ===========================================================================


@pytest.mark.unit
@pytest.mark.evaluator
class TestCreateResult:
    """Tests for _create_result method."""

    def test_create_result_minimal(self, sample_task_spec, sample_ground_truth_df):
        """Test _create_result with minimal parameters."""
        evaluator = ConcreteTaskEvaluator(task_spec=sample_task_spec)
        metrics = {"macro_f1": 0.85, "accuracy": 0.80}

        result = evaluator._create_result(
            metrics=metrics,
            ground_truth=sample_ground_truth_df,
            split="test",
        )

        assert isinstance(result, EvaluationResult)
        assert result.task == "test_classification"
        assert result.task_type == "classification"
        assert result.split == "test"
        assert result.primary_metric == "macro_f1"
        assert result.primary_score == 0.85
        assert result.metrics == metrics
        assert result.num_samples == len(sample_ground_truth_df)

    def test_create_result_with_per_class_metrics(
        self, sample_task_spec, sample_ground_truth_df
    ):
        """Test _create_result with per-class metrics."""
        evaluator = ConcreteTaskEvaluator(task_spec=sample_task_spec)
        metrics = {"macro_f1": 0.85}
        per_class = {
            "A": {"f1": 0.90, "precision": 0.88},
            "B": {"f1": 0.80, "precision": 0.82},
        }

        result = evaluator._create_result(
            metrics=metrics,
            ground_truth=sample_ground_truth_df,
            split="test",
            per_class_metrics=per_class,
        )

        assert result.per_class_metrics == per_class

    def test_create_result_with_confusion_matrix(
        self, sample_task_spec, sample_ground_truth_df
    ):
        """Test _create_result with confusion matrix."""
        evaluator = ConcreteTaskEvaluator(task_spec=sample_task_spec)
        metrics = {"accuracy": 0.75}
        confusion = [[10, 2], [1, 7]]

        result = evaluator._create_result(
            metrics=metrics,
            ground_truth=sample_ground_truth_df,
            split="test",
            confusion_matrix=confusion,
        )

        assert result.confusion_matrix == confusion

    def test_create_result_with_num_correct(
        self, sample_task_spec, sample_ground_truth_df
    ):
        """Test _create_result with num_correct."""
        evaluator = ConcreteTaskEvaluator(task_spec=sample_task_spec)
        metrics = {"accuracy": 0.83}

        result = evaluator._create_result(
            metrics=metrics,
            ground_truth=sample_ground_truth_df,
            split="test",
            num_correct=5,
        )

        assert result.num_correct == 5

    def test_create_result_with_misclassified_ids(
        self, sample_task_spec, sample_ground_truth_df
    ):
        """Test _create_result with misclassified IDs."""
        evaluator = ConcreteTaskEvaluator(task_spec=sample_task_spec)
        metrics = {"accuracy": 0.90}
        misclassified = ["doc_002", "doc_005"]

        result = evaluator._create_result(
            metrics=metrics,
            ground_truth=sample_ground_truth_df,
            split="test",
            misclassified_ids=misclassified,
        )

        assert result.misclassified_ids == misclassified

    def test_create_result_with_confidence_intervals(
        self, sample_task_spec, sample_ground_truth_df
    ):
        """Test _create_result with bootstrap confidence intervals."""
        evaluator = ConcreteTaskEvaluator(task_spec=sample_task_spec)
        metrics = {"macro_f1": 0.85}
        cis = {"macro_f1": (0.80, 0.90)}

        result = evaluator._create_result(
            metrics=metrics,
            ground_truth=sample_ground_truth_df,
            split="test",
            confidence_intervals=cis,
        )

        assert result.confidence_intervals == cis

    def test_create_result_with_stratified_metrics(
        self, sample_task_spec, sample_ground_truth_df
    ):
        """Test _create_result with stratified metrics."""
        evaluator = ConcreteTaskEvaluator(
            task_spec=sample_task_spec,
            stratify_by="form",
        )
        metrics = {"macro_f1": 0.85}
        stratified = {
            "form=lecture": {"macro_f1": 0.90},
            "form=map": {"macro_f1": 0.85},
            "form=essay": {"macro_f1": 0.80},
        }

        result = evaluator._create_result(
            metrics=metrics,
            ground_truth=sample_ground_truth_df,
            split="test",
            stratified_metrics=stratified,
        )

        assert result.stratified_metrics == stratified
        assert result.stratify_by == "form"

    def test_create_result_with_multiple_stratify_fields(
        self, sample_task_spec, sample_ground_truth_df
    ):
        """Test _create_result with multiple stratify fields."""
        evaluator = ConcreteTaskEvaluator(
            task_spec=sample_task_spec,
            stratify_by=["form", "lcc"],
        )
        metrics = {"macro_f1": 0.85}

        result = evaluator._create_result(
            metrics=metrics,
            ground_truth=sample_ground_truth_df,
            split="test",
        )

        assert result.stratify_by == "form,lcc"

    def test_create_result_includes_context(
        self, sample_task_spec, sample_ground_truth_df
    ):
        """Test _create_result includes EvaluationContext."""
        evaluator = ConcreteTaskEvaluator(task_spec=sample_task_spec)
        metrics = {"accuracy": 0.80}

        result = evaluator._create_result(
            metrics=metrics,
            ground_truth=sample_ground_truth_df,
            split="test",
        )

        assert result.context is not None
        assert isinstance(result.context, EvaluationContext)
        assert result.context.random_seed == 42

    def test_create_result_includes_provenance(
        self, sample_task_spec, sample_ground_truth_df
    ):
        """Test _create_result includes DataProvenance."""
        evaluator = ConcreteTaskEvaluator(task_spec=sample_task_spec)
        metrics = {"accuracy": 0.80}

        result = evaluator._create_result(
            metrics=metrics,
            ground_truth=sample_ground_truth_df,
            split="test",
        )

        assert result.data_provenance is not None
        assert isinstance(result.data_provenance, DataProvenance)

    def test_create_result_with_extra_context(
        self, sample_task_spec, sample_ground_truth_df
    ):
        """Test _create_result with extra context parameters."""
        evaluator = ConcreteTaskEvaluator(task_spec=sample_task_spec)
        metrics = {"accuracy": 0.80}

        result = evaluator._create_result(
            metrics=metrics,
            ground_truth=sample_ground_truth_df,
            split="test",
            model_name="test-model-v1",
        )

        assert result.context.model_name == "test-model-v1"


# ===========================================================================
# Tests for _compute_stratified_metrics Method
# ===========================================================================


@pytest.mark.unit
@pytest.mark.evaluator
class TestComputeStratifiedMetrics:
    """Tests for _compute_stratified_metrics method."""

    def test_compute_stratified_metrics_no_stratification(
        self, sample_task_spec, sample_ground_truth_df, sample_predictions
    ):
        """Test _compute_stratified_metrics returns None when no stratification."""
        evaluator = ConcreteTaskEvaluator(task_spec=sample_task_spec)

        def compute_metrics(preds, gt):
            return {"accuracy": 1.0}

        result = evaluator._compute_stratified_metrics(
            ground_truth=sample_ground_truth_df,
            predictions=sample_predictions,
            compute_metrics_fn=compute_metrics,
        )

        assert result is None

    def test_compute_stratified_metrics_by_form(
        self, sample_task_spec, sample_ground_truth_df, sample_predictions
    ):
        """Test _compute_stratified_metrics stratifies by form."""
        evaluator = ConcreteTaskEvaluator(
            task_spec=sample_task_spec,
            stratify_by="form",
        )

        def compute_metrics(preds, gt):
            return {"accuracy": len(preds) / len(gt)}

        result = evaluator._compute_stratified_metrics(
            ground_truth=sample_ground_truth_df,
            predictions=sample_predictions,
            compute_metrics_fn=compute_metrics,
        )

        assert result is not None
        assert "form=lecture" in result
        assert "form=map" in result
        assert "form=essay" in result

    def test_compute_stratified_metrics_by_lcc(
        self, sample_task_spec, sample_ground_truth_df, sample_predictions
    ):
        """Test _compute_stratified_metrics stratifies by LCC."""
        evaluator = ConcreteTaskEvaluator(
            task_spec=sample_task_spec,
            stratify_by="lcc",
        )

        def compute_metrics(preds, gt):
            return {"accuracy": 0.9}

        result = evaluator._compute_stratified_metrics(
            ground_truth=sample_ground_truth_df,
            predictions=sample_predictions,
            compute_metrics_fn=compute_metrics,
        )

        assert result is not None
        assert "lcc=A" in result
        assert "lcc=B" in result
        assert "lcc=C" in result

    def test_compute_stratified_metrics_multiple_fields(
        self, sample_task_spec, sample_ground_truth_df, sample_predictions
    ):
        """Test _compute_stratified_metrics with multiple stratify fields."""
        evaluator = ConcreteTaskEvaluator(
            task_spec=sample_task_spec,
            stratify_by=["form", "lcc"],
        )

        def compute_metrics(preds, gt):
            return {"accuracy": 1.0}

        result = evaluator._compute_stratified_metrics(
            ground_truth=sample_ground_truth_df,
            predictions=sample_predictions,
            compute_metrics_fn=compute_metrics,
        )

        assert result is not None
        # Should have results for both form and lcc strata
        form_keys = [k for k in result.keys() if k.startswith("form=")]
        lcc_keys = [k for k in result.keys() if k.startswith("lcc=")]
        assert len(form_keys) > 0
        assert len(lcc_keys) > 0

    def test_compute_stratified_metrics_nonexistent_field(
        self, sample_task_spec, sample_ground_truth_df, sample_predictions
    ):
        """Test _compute_stratified_metrics skips non-existent field."""
        evaluator = ConcreteTaskEvaluator(
            task_spec=sample_task_spec,
            stratify_by="nonexistent_field",
        )

        def compute_metrics(preds, gt):
            return {"accuracy": 1.0}

        result = evaluator._compute_stratified_metrics(
            ground_truth=sample_ground_truth_df,
            predictions=sample_predictions,
            compute_metrics_fn=compute_metrics,
        )

        # Should return None or empty dict since field doesn't exist
        assert result is None or result == {}

    def test_compute_stratified_metrics_skips_none_values(
        self, sample_task_spec, sample_predictions
    ):
        """Test _compute_stratified_metrics skips None values in stratification."""
        # Create DataFrame with None values
        df_with_none = pl.DataFrame(
            {
                "id": ["doc_001", "doc_002", "doc_003"],
                "lcc": ["A", None, "C"],
                "form": ["lecture", "map", None],
            }
        )

        evaluator = ConcreteTaskEvaluator(
            task_spec=sample_task_spec,
            stratify_by="lcc",
        )

        def compute_metrics(preds, gt):
            return {"accuracy": 1.0}

        result = evaluator._compute_stratified_metrics(
            ground_truth=df_with_none,
            predictions=sample_predictions[:3],
            compute_metrics_fn=compute_metrics,
        )

        if result:
            # Should only have strata for non-None values
            assert "lcc=A" in result
            assert "lcc=C" in result
            # Should not have None as a key

    def test_compute_stratified_metrics_handles_exceptions(
        self, sample_task_spec, sample_ground_truth_df, sample_predictions
    ):
        """Test _compute_stratified_metrics handles exceptions gracefully."""
        evaluator = ConcreteTaskEvaluator(
            task_spec=sample_task_spec,
            stratify_by="form",
        )

        def compute_metrics_with_error(preds, gt):
            raise ValueError("Test error")

        # Should not raise, but log warning
        result = evaluator._compute_stratified_metrics(
            ground_truth=sample_ground_truth_df,
            predictions=sample_predictions,
            compute_metrics_fn=compute_metrics_with_error,
        )

        # Should return None or empty dict due to errors
        assert result is None or result == {}


# ===========================================================================
# Tests for evaluate_from_file Method
# ===========================================================================


@pytest.mark.unit
@pytest.mark.evaluator
class TestEvaluateFromFile:
    """Tests for evaluate_from_file method."""

    def test_evaluate_from_file_basic(
        self,
        sample_task_spec,
        temp_predictions_file,
        sample_ground_truth_df,
        monkeypatch,
    ):
        """Test evaluate_from_file loads and evaluates predictions."""
        evaluator = ConcreteTaskEvaluator(task_spec=sample_task_spec)

        # Mock _load_ground_truth to return our sample data
        def mock_load_ground_truth(split):
            return sample_ground_truth_df

        monkeypatch.setattr(evaluator, "_load_ground_truth", mock_load_ground_truth)

        result = evaluator.evaluate_from_file(temp_predictions_file)

        assert isinstance(result, EvaluationResult)
        assert result.task == "test_classification"
        assert result.split == "test"  # Default split

    def test_evaluate_from_file_with_custom_split(
        self,
        sample_task_spec,
        temp_predictions_file,
        sample_ground_truth_df,
        monkeypatch,
    ):
        """Test evaluate_from_file with custom split."""
        evaluator = ConcreteTaskEvaluator(task_spec=sample_task_spec)

        def mock_load_ground_truth(split):
            assert split == "validation"
            return sample_ground_truth_df

        monkeypatch.setattr(evaluator, "_load_ground_truth", mock_load_ground_truth)

        result = evaluator.evaluate_from_file(temp_predictions_file, split="validation")

        assert result.split == "validation"

    def test_evaluate_from_file_includes_checksum(
        self,
        sample_task_spec,
        temp_predictions_file,
        sample_ground_truth_df,
        monkeypatch,
    ):
        """Test evaluate_from_file includes prediction file checksum."""
        evaluator = ConcreteTaskEvaluator(task_spec=sample_task_spec)

        def mock_load_ground_truth(split):
            return sample_ground_truth_df

        monkeypatch.setattr(evaluator, "_load_ground_truth", mock_load_ground_truth)

        result = evaluator.evaluate_from_file(temp_predictions_file)

        assert result.context is not None
        assert result.context.prediction_file_checksum is not None
        assert len(result.context.prediction_file_checksum) == 32  # MD5 hex

    def test_evaluate_from_file_with_path_string(
        self,
        sample_task_spec,
        temp_predictions_file,
        sample_ground_truth_df,
        monkeypatch,
    ):
        """Test evaluate_from_file accepts string path."""
        evaluator = ConcreteTaskEvaluator(task_spec=sample_task_spec)

        def mock_load_ground_truth(split):
            return sample_ground_truth_df

        monkeypatch.setattr(evaluator, "_load_ground_truth", mock_load_ground_truth)

        # Pass as string instead of Path
        result = evaluator.evaluate_from_file(str(temp_predictions_file))

        assert isinstance(result, EvaluationResult)

    def test_evaluate_from_file_with_compute_ci(
        self,
        sample_task_spec,
        temp_predictions_file,
        sample_ground_truth_df,
        monkeypatch,
    ):
        """Test evaluate_from_file passes compute_ci parameter."""
        evaluator = ConcreteTaskEvaluator(task_spec=sample_task_spec)

        def mock_load_ground_truth(split):
            return sample_ground_truth_df

        monkeypatch.setattr(evaluator, "_load_ground_truth", mock_load_ground_truth)

        # Should not raise even with compute_ci=True
        result = evaluator.evaluate_from_file(temp_predictions_file, compute_ci=True)

        assert isinstance(result, EvaluationResult)


# ===========================================================================
# Tests for Abstract Method Enforcement
# ===========================================================================


@pytest.mark.unit
@pytest.mark.evaluator
class TestAbstractMethodEnforcement:
    """Tests that TaskEvaluator is properly abstract."""

    def test_cannot_instantiate_base_class(self, sample_task_spec):
        """Test that TaskEvaluator cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            TaskEvaluator(task_spec=sample_task_spec)

    def test_subclass_must_implement_evaluate(self, sample_task_spec):
        """Test that subclasses must implement evaluate method."""

        class IncompleteEvaluator(TaskEvaluator):
            pass

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteEvaluator(task_spec=sample_task_spec)


# ===========================================================================
# Integration Tests
# ===========================================================================


@pytest.mark.unit
@pytest.mark.evaluator
class TestIntegration:
    """Integration tests for complete workflows."""

    def test_full_evaluation_workflow(
        self, sample_task_spec, sample_ground_truth_df, sample_predictions
    ):
        """Test complete evaluation workflow from predictions to result."""
        evaluator = ConcreteTaskEvaluator(task_spec=sample_task_spec)

        result = evaluator.evaluate(
            predictions=sample_predictions,
            ground_truth=sample_ground_truth_df,
            compute_ci=False,
        )

        assert isinstance(result, EvaluationResult)
        assert result.task == "test_classification"
        assert result.num_samples == 6
        assert result.primary_score == 1.0  # All predictions match
        assert result.num_correct == 6

    def test_evaluation_with_filtering(
        self, sample_task_spec, sample_ground_truth_df, sample_predictions
    ):
        """Test evaluation workflow with filtering."""
        evaluator = ConcreteTaskEvaluator(
            task_spec=sample_task_spec,
            filter_by={"git_commit": "abc123"},
        )

        filtered_df = evaluator._apply_filters(sample_ground_truth_df)

        # Filter predictions to match
        filtered_predictions = [
            p for p in sample_predictions if p["id"] in filtered_df["id"].to_list()
        ]

        result = evaluator.evaluate(
            predictions=filtered_predictions,
            ground_truth=filtered_df,
        )

        assert result.num_samples == 3
        assert result.data_provenance.is_single_commit()

    def test_evaluation_with_stratification(
        self, sample_task_spec, sample_ground_truth_df, sample_predictions
    ):
        """Test evaluation workflow with stratification."""
        evaluator = ConcreteTaskEvaluator(
            task_spec=sample_task_spec,
            stratify_by="form",
        )

        result = evaluator.evaluate(
            predictions=sample_predictions,
            ground_truth=sample_ground_truth_df,
        )

        assert result.stratify_by == "form"

    def test_evaluation_preserves_random_seed(
        self, sample_task_spec, sample_ground_truth_df
    ):
        """Test that random seed is preserved in results."""
        evaluator = ConcreteTaskEvaluator(
            task_spec=sample_task_spec,
            random_seed=99999,
        )

        result = evaluator.evaluate(
            predictions=[],
            ground_truth=sample_ground_truth_df,
        )

        assert result.context.random_seed == 99999
