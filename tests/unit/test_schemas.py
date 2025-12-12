"""Unit tests for shelf.evaluate.schemas module.

Tests cover:
- ClassificationPrediction validation (id, prediction, confidence)
- MultiLabelPrediction validation (id, predictions list, confidences list)
- RetrievalPrediction validation (query_id, ranked_doc_ids, scores)
- ClusteringPrediction validation (id, cluster)
- PairPrediction validation (pair_id/id alias, score, prediction)
- ValidationResult dataclass
- ValidationError exception
- load_predictions_jsonl function
- validate_classification_predictions function
- validate_retrieval_predictions function
- validate_clustering_predictions function
- validate_pair_predictions function
- Edge cases: empty inputs, boundary values, missing fields, extra fields
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError as PydanticValidationError

from shelf.evaluate.schemas import (
    ClassificationPrediction,
    ClusteringPrediction,
    MultiLabelPrediction,
    PairPrediction,
    RetrievalPrediction,
    ValidationError,
    ValidationResult,
    load_predictions_jsonl,
    validate_classification_predictions,
    validate_clustering_predictions,
    validate_pair_predictions,
    validate_retrieval_predictions,
)


# ===========================================================================
# ClassificationPrediction Tests
# ===========================================================================


@pytest.mark.unit
@pytest.mark.schema
class TestClassificationPrediction:
    """Tests for ClassificationPrediction schema."""

    def test_valid_prediction_minimal(self):
        """Test valid prediction with minimal fields."""
        pred = ClassificationPrediction(id="doc_001", prediction="A")
        assert pred.id == "doc_001"
        assert pred.prediction == "A"
        assert pred.confidence is None

    def test_valid_prediction_with_confidence(self):
        """Test valid prediction with confidence score."""
        pred = ClassificationPrediction(id="doc_001", prediction="A", confidence=0.95)
        assert pred.id == "doc_001"
        assert pred.prediction == "A"
        assert pred.confidence == pytest.approx(0.95)

    def test_confidence_zero(self):
        """Test confidence can be 0.0."""
        pred = ClassificationPrediction(id="doc_001", prediction="A", confidence=0.0)
        assert pred.confidence == pytest.approx(0.0)

    def test_confidence_one(self):
        """Test confidence can be 1.0."""
        pred = ClassificationPrediction(id="doc_001", prediction="A", confidence=1.0)
        assert pred.confidence == pytest.approx(1.0)

    def test_confidence_below_zero_fails(self):
        """Test confidence below 0 raises error."""
        with pytest.raises(
            PydanticValidationError, match="confidence must be between 0 and 1"
        ):
            ClassificationPrediction(id="doc_001", prediction="A", confidence=-0.1)

    def test_confidence_above_one_fails(self):
        """Test confidence above 1 raises error."""
        with pytest.raises(
            PydanticValidationError, match="confidence must be between 0 and 1"
        ):
            ClassificationPrediction(id="doc_001", prediction="A", confidence=1.1)

    def test_missing_id_fails(self):
        """Test missing id raises error."""
        with pytest.raises(PydanticValidationError):
            ClassificationPrediction(prediction="A")

    def test_missing_prediction_fails(self):
        """Test missing prediction raises error."""
        with pytest.raises(PydanticValidationError):
            ClassificationPrediction(id="doc_001")

    def test_extra_fields_ignored(self):
        """Test that extra fields are ignored (Pydantic v2 default behavior)."""
        # In Pydantic v2, strict=True is about type coercion, not extra fields
        # Extra fields are silently ignored by default
        pred = ClassificationPrediction(
            id="doc_001", prediction="A", extra_field="value"
        )
        assert pred.id == "doc_001"
        assert pred.prediction == "A"
        assert not hasattr(pred, "extra_field")

    def test_strict_mode_rejects_wrong_types(self):
        """Test strict mode rejects wrong types."""
        with pytest.raises(PydanticValidationError):
            ClassificationPrediction(id=123, prediction="A")


# ===========================================================================
# MultiLabelPrediction Tests
# ===========================================================================


@pytest.mark.unit
@pytest.mark.schema
class TestMultiLabelPrediction:
    """Tests for MultiLabelPrediction schema."""

    def test_valid_prediction_minimal(self):
        """Test valid prediction with minimal fields."""
        pred = MultiLabelPrediction(id="doc_001", predictions=["topic1", "topic2"])
        assert pred.id == "doc_001"
        assert pred.predictions == ["topic1", "topic2"]
        assert pred.confidences is None

    def test_valid_prediction_with_confidences(self):
        """Test valid prediction with confidence scores."""
        pred = MultiLabelPrediction(
            id="doc_001",
            predictions=["topic1", "topic2"],
            confidences=[0.9, 0.8],
        )
        assert pred.id == "doc_001"
        assert pred.predictions == ["topic1", "topic2"]
        assert pred.confidences == [pytest.approx(0.9), pytest.approx(0.8)]

    def test_single_prediction(self):
        """Test single prediction in list."""
        pred = MultiLabelPrediction(id="doc_001", predictions=["topic1"])
        assert len(pred.predictions) == 1
        assert pred.predictions[0] == "topic1"

    def test_empty_predictions_fails(self):
        """Test empty predictions list raises error."""
        with pytest.raises(
            PydanticValidationError, match="predictions cannot be empty"
        ):
            MultiLabelPrediction(id="doc_001", predictions=[])

    def test_confidence_out_of_range_fails(self):
        """Test confidence out of range raises error."""
        with pytest.raises(
            PydanticValidationError, match="all confidences must be between 0 and 1"
        ):
            MultiLabelPrediction(
                id="doc_001",
                predictions=["topic1", "topic2"],
                confidences=[0.9, 1.5],
            )

    def test_confidence_negative_fails(self):
        """Test negative confidence raises error."""
        with pytest.raises(
            PydanticValidationError, match="all confidences must be between 0 and 1"
        ):
            MultiLabelPrediction(
                id="doc_001",
                predictions=["topic1"],
                confidences=[-0.1],
            )

    def test_mismatched_lengths_allowed(self):
        """Test mismatched predictions/confidences lengths is allowed by schema."""
        # Schema doesn't enforce length matching, that's left to validation
        pred = MultiLabelPrediction(
            id="doc_001",
            predictions=["topic1", "topic2"],
            confidences=[0.9],
        )
        assert len(pred.predictions) == 2
        assert len(pred.confidences) == 1


# ===========================================================================
# RetrievalPrediction Tests
# ===========================================================================


@pytest.mark.unit
@pytest.mark.schema
class TestRetrievalPrediction:
    """Tests for RetrievalPrediction schema."""

    def test_valid_prediction_minimal(self):
        """Test valid prediction with minimal fields."""
        pred = RetrievalPrediction(
            query_id="q1", ranked_doc_ids=["doc_1", "doc_2", "doc_3"]
        )
        assert pred.query_id == "q1"
        assert pred.ranked_doc_ids == ["doc_1", "doc_2", "doc_3"]
        assert pred.scores is None

    def test_valid_prediction_with_scores(self):
        """Test valid prediction with scores."""
        pred = RetrievalPrediction(
            query_id="q1",
            ranked_doc_ids=["doc_1", "doc_2"],
            scores=[0.9, 0.7],
        )
        assert pred.query_id == "q1"
        assert pred.scores == [pytest.approx(0.9), pytest.approx(0.7)]

    def test_single_doc(self):
        """Test single document in ranking."""
        pred = RetrievalPrediction(query_id="q1", ranked_doc_ids=["doc_1"])
        assert len(pred.ranked_doc_ids) == 1

    def test_empty_ranked_docs_fails(self):
        """Test empty ranked_doc_ids raises error."""
        with pytest.raises(
            PydanticValidationError, match="ranked_doc_ids cannot be empty"
        ):
            RetrievalPrediction(query_id="q1", ranked_doc_ids=[])

    def test_duplicate_docs_fails(self):
        """Test duplicate doc IDs raise error."""
        with pytest.raises(
            PydanticValidationError, match="ranked_doc_ids contains duplicates"
        ):
            RetrievalPrediction(
                query_id="q1", ranked_doc_ids=["doc_1", "doc_2", "doc_1"]
            )

    def test_scores_can_be_negative(self):
        """Test scores can be negative (e.g., negative similarity)."""
        pred = RetrievalPrediction(
            query_id="q1",
            ranked_doc_ids=["doc_1", "doc_2"],
            scores=[0.5, -0.3],
        )
        assert pred.scores == [pytest.approx(0.5), pytest.approx(-0.3)]

    def test_scores_can_be_greater_than_one(self):
        """Test scores can be > 1 (not bounded like confidence)."""
        pred = RetrievalPrediction(
            query_id="q1",
            ranked_doc_ids=["doc_1"],
            scores=[10.5],
        )
        assert pred.scores == [pytest.approx(10.5)]


# ===========================================================================
# ClusteringPrediction Tests
# ===========================================================================


@pytest.mark.unit
@pytest.mark.schema
class TestClusteringPrediction:
    """Tests for ClusteringPrediction schema."""

    def test_valid_prediction(self):
        """Test valid clustering prediction."""
        pred = ClusteringPrediction(id="doc_001", cluster=0)
        assert pred.id == "doc_001"
        assert pred.cluster == 0

    def test_cluster_zero(self):
        """Test cluster can be 0."""
        pred = ClusteringPrediction(id="doc_001", cluster=0)
        assert pred.cluster == 0

    def test_cluster_large_value(self):
        """Test cluster can be large value."""
        pred = ClusteringPrediction(id="doc_001", cluster=999)
        assert pred.cluster == 999

    def test_cluster_negative_fails(self):
        """Test negative cluster raises error."""
        with pytest.raises(
            PydanticValidationError, match="cluster must be non-negative"
        ):
            ClusteringPrediction(id="doc_001", cluster=-1)

    def test_missing_id_fails(self):
        """Test missing id raises error."""
        with pytest.raises(PydanticValidationError):
            ClusteringPrediction(cluster=0)

    def test_missing_cluster_fails(self):
        """Test missing cluster raises error."""
        with pytest.raises(PydanticValidationError):
            ClusteringPrediction(id="doc_001")


# ===========================================================================
# PairPrediction Tests
# ===========================================================================


@pytest.mark.unit
@pytest.mark.schema
class TestPairPrediction:
    """Tests for PairPrediction schema."""

    def test_valid_prediction_with_score(self):
        """Test valid prediction with score only."""
        pred = PairPrediction(pair_id="pair_001", score=0.95)
        assert pred.pair_id == "pair_001"
        assert pred.score == pytest.approx(0.95)
        assert pred.prediction is None

    def test_valid_prediction_with_prediction(self):
        """Test valid prediction with binary prediction only."""
        pred = PairPrediction(pair_id="pair_001", prediction=1)
        assert pred.pair_id == "pair_001"
        assert pred.prediction == 1
        assert pred.score is None

    def test_valid_prediction_with_both(self):
        """Test valid prediction with both score and prediction."""
        pred = PairPrediction(pair_id="pair_001", score=0.95, prediction=1)
        assert pred.pair_id == "pair_001"
        assert pred.score == pytest.approx(0.95)
        assert pred.prediction == 1

    def test_id_alias_works(self):
        """Test 'id' alias for pair_id."""
        pred = PairPrediction(id="pair_001", score=0.5)
        assert pred.pair_id == "pair_001"

    def test_neither_score_nor_prediction_fails(self):
        """Test missing both score and prediction raises error."""
        with pytest.raises(
            PydanticValidationError,
            match="either score or prediction must be provided",
        ):
            PairPrediction(pair_id="pair_001")

    def test_prediction_zero(self):
        """Test prediction can be 0."""
        pred = PairPrediction(pair_id="pair_001", prediction=0)
        assert pred.prediction == 0

    def test_prediction_one(self):
        """Test prediction can be 1."""
        pred = PairPrediction(pair_id="pair_001", prediction=1)
        assert pred.prediction == 1

    def test_prediction_invalid_value_fails(self):
        """Test invalid prediction value raises error."""
        with pytest.raises(PydanticValidationError, match="prediction must be 0 or 1"):
            PairPrediction(pair_id="pair_001", prediction=2)

    def test_score_can_be_negative(self):
        """Test score can be negative."""
        pred = PairPrediction(pair_id="pair_001", score=-0.5)
        assert pred.score == pytest.approx(-0.5)

    def test_score_can_be_zero(self):
        """Test score can be 0."""
        pred = PairPrediction(pair_id="pair_001", score=0.0)
        assert pred.score == pytest.approx(0.0)

    def test_score_infinity_fails(self):
        """Test infinite score raises error."""
        with pytest.raises(
            PydanticValidationError, match="score must be a finite number"
        ):
            PairPrediction(pair_id="pair_001", score=float("inf"))

    def test_score_negative_infinity_fails(self):
        """Test negative infinite score raises error."""
        with pytest.raises(
            PydanticValidationError, match="score must be a finite number"
        ):
            PairPrediction(pair_id="pair_001", score=float("-inf"))

    def test_score_nan_fails(self):
        """Test NaN score raises error."""
        with pytest.raises(PydanticValidationError):
            PairPrediction(pair_id="pair_001", score=float("nan"))


# ===========================================================================
# ValidationResult Tests
# ===========================================================================


@pytest.mark.unit
@pytest.mark.schema
class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_valid_result(self):
        """Test valid ValidationResult creation."""
        result = ValidationResult(
            valid=True,
            errors=[],
            warnings=[],
            num_predictions=10,
        )
        assert result.valid is True
        assert result.errors == []
        assert result.warnings == []
        assert result.num_predictions == 10

    def test_invalid_result_with_errors(self):
        """Test invalid ValidationResult with errors."""
        result = ValidationResult(
            valid=False,
            errors=["Error 1", "Error 2"],
            warnings=["Warning 1"],
            num_predictions=5,
        )
        assert result.valid is False
        assert len(result.errors) == 2
        assert len(result.warnings) == 1

    def test_empty_result(self):
        """Test result with zero predictions."""
        result = ValidationResult(
            valid=True,
            errors=[],
            warnings=[],
            num_predictions=0,
        )
        assert result.num_predictions == 0


# ===========================================================================
# ValidationError Tests
# ===========================================================================


@pytest.mark.unit
@pytest.mark.schema
class TestValidationError:
    """Tests for ValidationError exception."""

    def test_single_error(self):
        """Test ValidationError with single error."""
        error = ValidationError(["Error message"])
        assert len(error.errors) == 1
        assert "Error message" in str(error)
        assert "1 error(s)" in str(error)

    def test_multiple_errors(self):
        """Test ValidationError with multiple errors."""
        errors = ["Error 1", "Error 2", "Error 3"]
        error = ValidationError(errors)
        assert len(error.errors) == 3
        assert "3 error(s)" in str(error)

    def test_many_errors_truncated(self):
        """Test ValidationError truncates long error lists in message."""
        errors = [f"Error {i}" for i in range(20)]
        error = ValidationError(errors)
        message = str(error)
        assert "20 error(s)" in message
        # Should show first 10 and indicate more
        assert "10 more errors" in message

    def test_error_list_accessible(self):
        """Test errors list is accessible."""
        errors = ["Error 1", "Error 2"]
        error = ValidationError(errors)
        assert error.errors == errors


# ===========================================================================
# load_predictions_jsonl Tests
# ===========================================================================


@pytest.mark.unit
@pytest.mark.schema
class TestLoadPredictionsJsonl:
    """Tests for load_predictions_jsonl function."""

    def test_load_valid_jsonl(
        self, make_temp_jsonl_file, valid_classification_predictions
    ):
        """Test loading valid JSONL file."""
        path = make_temp_jsonl_file(valid_classification_predictions)
        predictions = load_predictions_jsonl(path)
        assert len(predictions) == 3
        assert predictions[0]["id"] == "doc_001"
        assert predictions[0]["prediction"] == "A"

    def test_load_empty_file(self, make_temp_jsonl_file):
        """Test loading empty JSONL file."""
        path = make_temp_jsonl_file([])
        predictions = load_predictions_jsonl(path)
        assert predictions == []

    def test_load_with_blank_lines(self, tmp_path):
        """Test loading JSONL with blank lines (should skip)."""
        path = tmp_path / "predictions.jsonl"
        with open(path, "w") as f:
            f.write('{"id": "doc_001", "prediction": "A"}\n')
            f.write("\n")
            f.write('{"id": "doc_002", "prediction": "B"}\n')
            f.write("   \n")
            f.write('{"id": "doc_003", "prediction": "C"}\n')

        predictions = load_predictions_jsonl(path)
        assert len(predictions) == 3

    def test_load_invalid_json_fails(self, tmp_path):
        """Test loading invalid JSON raises error."""
        path = tmp_path / "invalid.jsonl"
        with open(path, "w") as f:
            f.write('{"id": "doc_001", "prediction": "A"}\n')
            f.write('{"id": "doc_002", "prediction":}\n')  # Invalid JSON

        with pytest.raises(json.JSONDecodeError, match="Invalid JSON on line 2"):
            load_predictions_jsonl(path)

    def test_load_nonexistent_file_fails(self):
        """Test loading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_predictions_jsonl("/nonexistent/path/file.jsonl")

    def test_load_path_as_string(
        self, make_temp_jsonl_file, valid_classification_predictions
    ):
        """Test loading with path as string."""
        path = make_temp_jsonl_file(valid_classification_predictions)
        predictions = load_predictions_jsonl(str(path))
        assert len(predictions) == 3

    def test_load_path_as_path_object(
        self, make_temp_jsonl_file, valid_classification_predictions
    ):
        """Test loading with Path object."""
        path = make_temp_jsonl_file(valid_classification_predictions)
        predictions = load_predictions_jsonl(Path(path))
        assert len(predictions) == 3


# ===========================================================================
# validate_classification_predictions Tests
# ===========================================================================


@pytest.mark.unit
@pytest.mark.schema
class TestValidateClassificationPredictions:
    """Tests for validate_classification_predictions function."""

    def test_valid_predictions(
        self,
        valid_classification_predictions,
        ground_truth_ids,
        classification_label_space,
    ):
        """Test validation of valid predictions."""
        result = validate_classification_predictions(
            valid_classification_predictions,
            ground_truth_ids,
            classification_label_space,
        )
        assert result.valid is True
        assert result.errors == []
        assert result.num_predictions == 3

    def test_valid_without_label_space(
        self, valid_classification_predictions, ground_truth_ids
    ):
        """Test validation without label space (any label allowed)."""
        result = validate_classification_predictions(
            valid_classification_predictions,
            ground_truth_ids,
            label_space=None,
        )
        assert result.valid is True

    def test_missing_predictions(self, ground_truth_ids, classification_label_space):
        """Test missing predictions detected."""
        predictions = [
            {"id": "doc_001", "prediction": "A"},
            # Missing doc_002 and doc_003
        ]
        result = validate_classification_predictions(
            predictions, ground_truth_ids, classification_label_space
        )
        assert result.valid is False
        assert any("Missing predictions for 2 documents" in e for e in result.errors)

    def test_duplicate_ids(self, ground_truth_ids, classification_label_space):
        """Test duplicate prediction IDs detected."""
        predictions = [
            {"id": "doc_001", "prediction": "A"},
            {"id": "doc_001", "prediction": "B"},  # Duplicate
            {"id": "doc_002", "prediction": "C"},
        ]
        result = validate_classification_predictions(
            predictions, ground_truth_ids, classification_label_space
        )
        assert result.valid is False
        assert any("duplicate id 'doc_001'" in e for e in result.errors)

    def test_unknown_id(self, ground_truth_ids, classification_label_space):
        """Test unknown document ID detected."""
        predictions = [
            {"id": "doc_001", "prediction": "A"},
            {"id": "doc_002", "prediction": "B"},
            {"id": "doc_999", "prediction": "C"},  # Unknown
        ]
        result = validate_classification_predictions(
            predictions, ground_truth_ids, classification_label_space
        )
        assert result.valid is False
        assert any("unknown id 'doc_999'" in e for e in result.errors)

    def test_invalid_label(self, ground_truth_ids, classification_label_space):
        """Test invalid label detected."""
        predictions = [
            {"id": "doc_001", "prediction": "A"},
            {"id": "doc_002", "prediction": "Z"},  # Not in label space
            {"id": "doc_003", "prediction": "C"},
        ]
        result = validate_classification_predictions(
            predictions, ground_truth_ids, classification_label_space
        )
        assert result.valid is False
        assert any("invalid label 'Z'" in e for e in result.errors)

    def test_invalid_schema(self, ground_truth_ids, classification_label_space):
        """Test invalid prediction schema detected."""
        predictions = [
            {"id": "doc_001", "prediction": "A"},
            {"id": "doc_002"},  # Missing prediction
            {"id": "doc_003", "prediction": "C"},
        ]
        result = validate_classification_predictions(
            predictions, ground_truth_ids, classification_label_space
        )
        assert result.valid is False
        assert any("Prediction 1:" in e for e in result.errors)

    def test_empty_predictions(self, ground_truth_ids, classification_label_space):
        """Test empty predictions list."""
        result = validate_classification_predictions(
            [], ground_truth_ids, classification_label_space
        )
        assert result.valid is False
        assert result.num_predictions == 0


# ===========================================================================
# validate_retrieval_predictions Tests
# ===========================================================================


@pytest.mark.unit
@pytest.mark.schema
class TestValidateRetrievalPredictions:
    """Tests for validate_retrieval_predictions function."""

    def test_valid_predictions(self, valid_retrieval_predictions):
        """Test validation of valid retrieval predictions."""
        query_ids = {"q1", "q2", "q3"}
        corpus_ids = {f"doc_{i}" for i in range(1, 9)}

        result = validate_retrieval_predictions(
            valid_retrieval_predictions, query_ids, corpus_ids
        )
        assert result.valid is True
        assert result.errors == []
        assert result.num_predictions == 3

    def test_missing_query(self):
        """Test missing query predictions detected."""
        predictions = [
            {"query_id": "q1", "ranked_doc_ids": ["doc_1", "doc_2"]},
            # Missing q2
        ]
        query_ids = {"q1", "q2"}
        corpus_ids = {"doc_1", "doc_2"}

        result = validate_retrieval_predictions(predictions, query_ids, corpus_ids)
        assert result.valid is False
        assert any("Missing predictions for 1 queries" in e for e in result.errors)

    def test_duplicate_query_ids(self):
        """Test duplicate query IDs detected."""
        predictions = [
            {"query_id": "q1", "ranked_doc_ids": ["doc_1"]},
            {"query_id": "q1", "ranked_doc_ids": ["doc_2"]},  # Duplicate
        ]
        query_ids = {"q1"}
        corpus_ids = {"doc_1", "doc_2"}

        result = validate_retrieval_predictions(predictions, query_ids, corpus_ids)
        assert result.valid is False
        assert any("duplicate query_id 'q1'" in e for e in result.errors)

    def test_unknown_query_id(self):
        """Test unknown query ID detected."""
        predictions = [
            {"query_id": "q999", "ranked_doc_ids": ["doc_1"]},
        ]
        query_ids = {"q1"}
        corpus_ids = {"doc_1"}

        result = validate_retrieval_predictions(predictions, query_ids, corpus_ids)
        assert result.valid is False
        assert any("unknown query_id 'q999'" in e for e in result.errors)

    def test_unknown_doc_ids(self):
        """Test unknown document IDs in ranking detected."""
        predictions = [
            {"query_id": "q1", "ranked_doc_ids": ["doc_1", "doc_999"]},
        ]
        query_ids = {"q1"}
        corpus_ids = {"doc_1", "doc_2"}

        result = validate_retrieval_predictions(predictions, query_ids, corpus_ids)
        assert result.valid is False
        assert any("unknown doc IDs in ranking" in e for e in result.errors)

    def test_invalid_schema(self):
        """Test invalid prediction schema detected."""
        predictions = [
            {"query_id": "q1", "ranked_doc_ids": []},  # Empty list
        ]
        query_ids = {"q1"}
        corpus_ids = {"doc_1"}

        result = validate_retrieval_predictions(predictions, query_ids, corpus_ids)
        assert result.valid is False
        assert any("Prediction 0:" in e for e in result.errors)


# ===========================================================================
# validate_clustering_predictions Tests
# ===========================================================================


@pytest.mark.unit
@pytest.mark.schema
class TestValidateClusteringPredictions:
    """Tests for validate_clustering_predictions function."""

    def test_valid_predictions(self, valid_clustering_predictions):
        """Test validation of valid clustering predictions."""
        ground_truth_ids = {"doc_001", "doc_002", "doc_003", "doc_004"}

        result = validate_clustering_predictions(
            valid_clustering_predictions, ground_truth_ids
        )
        assert result.valid is True
        assert result.errors == []
        assert result.num_predictions == 4

    def test_valid_with_expected_clusters(self, valid_clustering_predictions):
        """Test validation with expected cluster count."""
        ground_truth_ids = {"doc_001", "doc_002", "doc_003", "doc_004"}

        result = validate_clustering_predictions(
            valid_clustering_predictions, ground_truth_ids, expected_clusters=3
        )
        assert result.valid is True
        # Should have 3 clusters (0, 1, 2)
        assert result.warnings == []

    def test_wrong_cluster_count_warning(self, valid_clustering_predictions):
        """Test wrong cluster count generates warning."""
        ground_truth_ids = {"doc_001", "doc_002", "doc_003", "doc_004"}

        result = validate_clustering_predictions(
            valid_clustering_predictions, ground_truth_ids, expected_clusters=5
        )
        assert result.valid is True  # Just a warning, not error
        assert any("Expected 5 clusters, got 3" in w for w in result.warnings)

    def test_missing_predictions(self):
        """Test missing predictions detected."""
        predictions = [
            {"id": "doc_001", "cluster": 0},
            # Missing doc_002
        ]
        ground_truth_ids = {"doc_001", "doc_002"}

        result = validate_clustering_predictions(predictions, ground_truth_ids)
        assert result.valid is False
        assert any("Missing predictions for 1 documents" in e for e in result.errors)

    def test_duplicate_ids(self):
        """Test duplicate prediction IDs detected."""
        predictions = [
            {"id": "doc_001", "cluster": 0},
            {"id": "doc_001", "cluster": 1},  # Duplicate
        ]
        ground_truth_ids = {"doc_001"}

        result = validate_clustering_predictions(predictions, ground_truth_ids)
        assert result.valid is False
        assert any("duplicate id 'doc_001'" in e for e in result.errors)

    def test_unknown_id(self):
        """Test unknown document ID detected."""
        predictions = [
            {"id": "doc_999", "cluster": 0},
        ]
        ground_truth_ids = {"doc_001"}

        result = validate_clustering_predictions(predictions, ground_truth_ids)
        assert result.valid is False
        assert any("unknown id 'doc_999'" in e for e in result.errors)

    def test_invalid_schema(self):
        """Test invalid prediction schema detected."""
        predictions = [
            {"id": "doc_001", "cluster": -1},  # Negative cluster
        ]
        ground_truth_ids = {"doc_001"}

        result = validate_clustering_predictions(predictions, ground_truth_ids)
        assert result.valid is False
        assert any("Prediction 0:" in e for e in result.errors)


# ===========================================================================
# validate_pair_predictions Tests
# ===========================================================================


@pytest.mark.unit
@pytest.mark.schema
class TestValidatePairPredictions:
    """Tests for validate_pair_predictions function."""

    def test_valid_predictions(self, valid_pair_predictions):
        """Test validation of valid pair predictions."""
        pair_ids = {"pair_001", "pair_002", "pair_003", "pair_004"}

        result = validate_pair_predictions(valid_pair_predictions, pair_ids)
        assert result.valid is True
        assert result.errors == []
        assert result.num_predictions == 4

    def test_missing_pairs(self):
        """Test missing pair predictions detected."""
        predictions = [
            {"pair_id": "pair_001", "score": 0.9},
            # Missing pair_002
        ]
        pair_ids = {"pair_001", "pair_002"}

        result = validate_pair_predictions(predictions, pair_ids)
        assert result.valid is False
        assert any("Missing predictions for 1 pairs" in e for e in result.errors)

    def test_duplicate_pair_ids(self):
        """Test duplicate pair IDs detected."""
        predictions = [
            {"pair_id": "pair_001", "score": 0.9},
            {"pair_id": "pair_001", "score": 0.8},  # Duplicate
        ]
        pair_ids = {"pair_001"}

        result = validate_pair_predictions(predictions, pair_ids)
        assert result.valid is False
        assert any("duplicate pair_id 'pair_001'" in e for e in result.errors)

    def test_unknown_pair_id(self):
        """Test unknown pair ID detected."""
        predictions = [
            {"pair_id": "pair_999", "score": 0.9},
        ]
        pair_ids = {"pair_001"}

        result = validate_pair_predictions(predictions, pair_ids)
        assert result.valid is False
        assert any("unknown pair_id 'pair_999'" in e for e in result.errors)

    def test_invalid_schema(self):
        """Test invalid prediction schema detected."""
        predictions = [
            {"pair_id": "pair_001"},  # Missing both score and prediction
        ]
        pair_ids = {"pair_001"}

        result = validate_pair_predictions(predictions, pair_ids)
        assert result.valid is False
        assert any("Prediction 0:" in e for e in result.errors)

    def test_id_alias_works(self):
        """Test 'id' alias works in validation."""
        predictions = [
            {"id": "pair_001", "score": 0.9},  # Using 'id' instead of 'pair_id'
        ]
        pair_ids = {"pair_001"}

        result = validate_pair_predictions(predictions, pair_ids)
        assert result.valid is True


# ===========================================================================
# Integration and Edge Case Tests
# ===========================================================================


@pytest.mark.unit
@pytest.mark.schema
class TestSchemaIntegration:
    """Integration and edge case tests."""

    def test_classification_end_to_end(self, make_temp_jsonl_file):
        """Test full classification validation pipeline."""
        predictions = [
            {"id": "doc_001", "prediction": "A", "confidence": 0.95},
            {"id": "doc_002", "prediction": "B", "confidence": 0.87},
            {"id": "doc_003", "prediction": "C", "confidence": 0.72},
        ]
        path = make_temp_jsonl_file(predictions)

        # Load predictions
        loaded = load_predictions_jsonl(path)
        assert len(loaded) == 3

        # Validate
        ground_truth = {"doc_001", "doc_002", "doc_003"}
        label_space = {"A", "B", "C"}
        result = validate_classification_predictions(loaded, ground_truth, label_space)

        assert result.valid is True

    def test_retrieval_end_to_end(self, make_temp_jsonl_file):
        """Test full retrieval validation pipeline."""
        predictions = [
            {
                "query_id": "q1",
                "ranked_doc_ids": ["doc_1", "doc_2"],
                "scores": [0.9, 0.7],
            },
            {"query_id": "q2", "ranked_doc_ids": ["doc_3"], "scores": [0.85]},
        ]
        path = make_temp_jsonl_file(predictions)

        # Load predictions
        loaded = load_predictions_jsonl(path)

        # Validate
        query_ids = {"q1", "q2"}
        corpus_ids = {"doc_1", "doc_2", "doc_3"}
        result = validate_retrieval_predictions(loaded, query_ids, corpus_ids)

        assert result.valid is True

    def test_validation_with_multiple_errors(self):
        """Test validation collects multiple errors."""
        predictions = [
            {"id": "doc_001", "prediction": "A"},
            {"id": "doc_999", "prediction": "Z"},  # Unknown ID and invalid label
            {"id": "doc_001", "prediction": "B"},  # Duplicate ID
        ]
        ground_truth = {"doc_001", "doc_002"}
        label_space = {"A", "B", "C"}

        result = validate_classification_predictions(
            predictions, ground_truth, label_space
        )

        assert result.valid is False
        # Should have multiple errors: unknown ID, invalid label, duplicate ID, missing doc_002
        assert len(result.errors) >= 3

    def test_empty_ground_truth(self):
        """Test validation with empty ground truth."""
        predictions = [{"id": "doc_001", "prediction": "A"}]
        ground_truth = set()
        label_space = {"A"}

        result = validate_classification_predictions(
            predictions, ground_truth, label_space
        )

        assert result.valid is False
        assert any("unknown id" in e for e in result.errors)
