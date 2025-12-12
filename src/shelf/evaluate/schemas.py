"""Pydantic schemas for prediction file validation.

These schemas define the expected format for prediction files
and provide clear error messages when validation fails.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class ClassificationPrediction(BaseModel):
    """Single-label classification prediction.

    Attributes:
        id: Document identifier
        prediction: Predicted class label
        confidence: Optional confidence score (0-1)
    """

    model_config = ConfigDict(strict=True)

    id: str
    prediction: str
    confidence: float | None = None

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float | None) -> float | None:
        if v is not None and (v < 0 or v > 1):
            raise ValueError("confidence must be between 0 and 1")
        return v


class MultiLabelPrediction(BaseModel):
    """Multi-label classification prediction.

    Attributes:
        id: Document identifier
        predictions: List of predicted labels
        confidences: Optional confidence scores per label
    """

    model_config = ConfigDict(strict=True)

    id: str
    predictions: list[str]
    confidences: list[float] | None = None

    @field_validator("predictions")
    @classmethod
    def validate_predictions(cls, v: list[str]) -> list[str]:
        if len(v) == 0:
            raise ValueError("predictions cannot be empty")
        return v

    @field_validator("confidences")
    @classmethod
    def validate_confidences(cls, v: list[float] | None) -> list[float] | None:
        if v is not None:
            for conf in v:
                if conf < 0 or conf > 1:
                    raise ValueError("all confidences must be between 0 and 1")
        return v


class RetrievalPrediction(BaseModel):
    """Retrieval results for a single query.

    Attributes:
        query_id: Query identifier
        ranked_doc_ids: Ordered list of document IDs (most relevant first)
        scores: Optional relevance scores
    """

    model_config = ConfigDict(strict=True)

    query_id: str
    ranked_doc_ids: list[str]
    scores: list[float] | None = None

    @field_validator("ranked_doc_ids")
    @classmethod
    def validate_ranked_doc_ids(cls, v: list[str]) -> list[str]:
        if len(v) == 0:
            raise ValueError("ranked_doc_ids cannot be empty")
        # Check for duplicates
        if len(v) != len(set(v)):
            raise ValueError("ranked_doc_ids contains duplicates")
        return v


class ClusteringPrediction(BaseModel):
    """Clustering assignment.

    Attributes:
        id: Document identifier
        cluster: Assigned cluster index
    """

    model_config = ConfigDict(strict=True)

    id: str
    cluster: int

    @field_validator("cluster")
    @classmethod
    def validate_cluster(cls, v: int) -> int:
        if v < 0:
            raise ValueError("cluster must be non-negative")
        return v


class PairPrediction(BaseModel):
    """Pair classification prediction.

    Supports either a similarity score or a binary prediction.
    At least one of score/prediction must be provided.

    Attributes:
        pair_id: Pair identifier
        score: Similarity/confidence score (e.g., cosine similarity)
        prediction: Binary prediction (0 = different, 1 = same)
    """

    model_config = ConfigDict(strict=True)

    pair_id: str = Field(validation_alias=AliasChoices("pair_id", "id"))
    score: float | None = None
    prediction: int | None = None

    @field_validator("score")
    @classmethod
    def validate_score(cls, v: float | None) -> float | None:
        if v is None:
            return v
        if not (float("-inf") < v < float("inf")):
            raise ValueError("score must be a finite number")
        return v

    @field_validator("prediction")
    @classmethod
    def validate_prediction(cls, v: int | None) -> int | None:
        if v is None:
            return v
        if v not in (0, 1):
            raise ValueError("prediction must be 0 or 1")
        return v

    @model_validator(mode="after")
    def validate_required_fields(self):
        if self.score is None and self.prediction is None:
            raise ValueError("either score or prediction must be provided")
        return self


@dataclass
class ValidationResult:
    """Result of prediction validation.

    Attributes:
        valid: Whether validation passed
        errors: List of error messages
        warnings: List of warning messages
        num_predictions: Number of predictions validated
    """

    valid: bool
    errors: list[str]
    warnings: list[str]
    num_predictions: int


class ValidationError(Exception):
    """Raised when prediction validation fails."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        message = f"Validation failed with {len(errors)} error(s):\n" + "\n".join(
            f"  - {e}" for e in errors[:10]
        )
        if len(errors) > 10:
            message += f"\n  ... and {len(errors) - 10} more errors"
        super().__init__(message)


def load_predictions_jsonl(path: Path | str) -> list[dict[str, Any]]:
    """Load predictions from a JSONL file.

    Args:
        path: Path to JSONL file

    Returns:
        List of prediction dictionaries

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file contains invalid JSON
    """
    path = Path(path)
    predictions = []
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                predictions.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(
                    f"Invalid JSON on line {line_num}: {e.msg}",
                    e.doc,
                    e.pos,
                ) from e
    return predictions


def validate_classification_predictions(
    predictions: list[dict[str, Any]],
    ground_truth_ids: set[str],
    label_space: set[str] | None = None,
) -> ValidationResult:
    """Validate classification predictions.

    Args:
        predictions: List of prediction dictionaries
        ground_truth_ids: Set of valid document IDs
        label_space: Valid labels (None = any label allowed)

    Returns:
        ValidationResult with errors and warnings
    """
    errors: list[str] = []
    warnings: list[str] = []
    seen_ids: set[str] = set()

    for i, pred in enumerate(predictions):
        # Validate schema
        try:
            validated = ClassificationPrediction.model_validate(pred)
        except Exception as e:
            errors.append(f"Prediction {i}: {e}")
            continue

        # Check for duplicate IDs
        if validated.id in seen_ids:
            errors.append(f"Prediction {i}: duplicate id '{validated.id}'")
        seen_ids.add(validated.id)

        # Check ID exists in ground truth
        if validated.id not in ground_truth_ids:
            errors.append(f"Prediction {i}: unknown id '{validated.id}'")

        # Check label is valid
        if label_space is not None and validated.prediction not in label_space:
            errors.append(
                f"Prediction {i}: invalid label '{validated.prediction}' "
                f"(valid: {sorted(label_space)[:5]}...)"
            )

    # Check for missing predictions
    missing_ids = ground_truth_ids - seen_ids
    if missing_ids:
        errors.append(
            f"Missing predictions for {len(missing_ids)} documents "
            f"(e.g., {list(missing_ids)[:3]})"
        )

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        num_predictions=len(predictions),
    )


def validate_retrieval_predictions(
    predictions: list[dict[str, Any]],
    query_ids: set[str],
    corpus_ids: set[str],
) -> ValidationResult:
    """Validate retrieval predictions.

    Args:
        predictions: List of prediction dictionaries
        query_ids: Set of valid query IDs
        corpus_ids: Set of valid corpus document IDs

    Returns:
        ValidationResult with errors and warnings
    """
    errors: list[str] = []
    warnings: list[str] = []
    seen_query_ids: set[str] = set()

    for i, pred in enumerate(predictions):
        # Validate schema
        try:
            validated = RetrievalPrediction.model_validate(pred)
        except Exception as e:
            errors.append(f"Prediction {i}: {e}")
            continue

        # Check for duplicate query IDs
        if validated.query_id in seen_query_ids:
            errors.append(f"Prediction {i}: duplicate query_id '{validated.query_id}'")
        seen_query_ids.add(validated.query_id)

        # Check query ID exists
        if validated.query_id not in query_ids:
            errors.append(f"Prediction {i}: unknown query_id '{validated.query_id}'")

        # Check all ranked doc IDs exist in corpus
        invalid_docs = set(validated.ranked_doc_ids) - corpus_ids
        if invalid_docs:
            errors.append(
                f"Prediction {i}: unknown doc IDs in ranking: {list(invalid_docs)[:3]}"
            )

    # Check for missing queries
    missing_queries = query_ids - seen_query_ids
    if missing_queries:
        errors.append(
            f"Missing predictions for {len(missing_queries)} queries "
            f"(e.g., {list(missing_queries)[:3]})"
        )

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        num_predictions=len(predictions),
    )


def validate_clustering_predictions(
    predictions: list[dict[str, Any]],
    ground_truth_ids: set[str],
    expected_clusters: int | None = None,
) -> ValidationResult:
    """Validate clustering predictions.

    Args:
        predictions: List of prediction dictionaries
        ground_truth_ids: Set of valid document IDs
        expected_clusters: Expected number of clusters (optional)

    Returns:
        ValidationResult with errors and warnings
    """
    errors: list[str] = []
    warnings: list[str] = []
    seen_ids: set[str] = set()
    cluster_ids: set[int] = set()

    for i, pred in enumerate(predictions):
        # Validate schema
        try:
            validated = ClusteringPrediction.model_validate(pred)
        except Exception as e:
            errors.append(f"Prediction {i}: {e}")
            continue

        # Check for duplicate IDs
        if validated.id in seen_ids:
            errors.append(f"Prediction {i}: duplicate id '{validated.id}'")
        seen_ids.add(validated.id)

        # Check ID exists
        if validated.id not in ground_truth_ids:
            errors.append(f"Prediction {i}: unknown id '{validated.id}'")

        cluster_ids.add(validated.cluster)

    # Check for missing predictions
    missing_ids = ground_truth_ids - seen_ids
    if missing_ids:
        errors.append(
            f"Missing predictions for {len(missing_ids)} documents "
            f"(e.g., {list(missing_ids)[:3]})"
        )

    # Check cluster count
    if expected_clusters is not None and len(cluster_ids) != expected_clusters:
        warnings.append(
            f"Expected {expected_clusters} clusters, got {len(cluster_ids)}"
        )

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        num_predictions=len(predictions),
    )


def validate_pair_predictions(
    predictions: list[dict[str, Any]],
    pair_ids: set[str],
) -> ValidationResult:
    """Validate pair classification predictions.

    Args:
        predictions: List of prediction dictionaries
        pair_ids: Set of valid pair IDs

    Returns:
        ValidationResult with errors and warnings
    """
    errors: list[str] = []
    warnings: list[str] = []
    seen_pair_ids: set[str] = set()

    for i, pred in enumerate(predictions):
        # Validate schema
        try:
            validated = PairPrediction.model_validate(pred)
        except Exception as e:
            errors.append(f"Prediction {i}: {e}")
            continue

        # Check for duplicate pair IDs
        if validated.pair_id in seen_pair_ids:
            errors.append(f"Prediction {i}: duplicate pair_id '{validated.pair_id}'")
        seen_pair_ids.add(validated.pair_id)

        # Check pair ID exists
        if validated.pair_id not in pair_ids:
            errors.append(f"Prediction {i}: unknown pair_id '{validated.pair_id}'")

    # Check for missing predictions
    missing_pairs = pair_ids - seen_pair_ids
    if missing_pairs:
        errors.append(
            f"Missing predictions for {len(missing_pairs)} pairs "
            f"(e.g., {list(missing_pairs)[:3]})"
        )

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        num_predictions=len(predictions),
    )
