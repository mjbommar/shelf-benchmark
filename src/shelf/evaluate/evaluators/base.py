"""Base evaluator class for SHELF tasks.

Evaluators are stateless - they take predictions and ground truth,
and return results. They don't hold data.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl
from datasets import load_dataset

from shelf.evaluate.results import (
    EvaluationContext,
    EvaluationResult,
    compute_data_checksum,
    compute_file_checksum,
)
from shelf.evaluate.schemas import load_predictions_jsonl
from shelf.evaluate.tasks import TaskSpec

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class TaskEvaluator(ABC):
    """Base class for all task evaluators.

    Evaluators are stateless - they take predictions and ground truth,
    and return results. They don't hold data.

    Subclasses must implement:
    - evaluate(): Core evaluation logic

    Subclasses may optionally override:
    - _load_ground_truth(): Custom data loading
    - _validate_predictions(): Custom validation
    """

    def __init__(self, task_spec: TaskSpec, random_seed: int = 42):
        """Initialize evaluator.

        Args:
            task_spec: Task specification
            random_seed: Random seed for reproducibility
        """
        self.task_spec = task_spec
        self.random_seed = random_seed

    @abstractmethod
    def evaluate(
        self,
        predictions: list[dict[str, Any]],
        ground_truth: pl.DataFrame,
        compute_ci: bool = False,
    ) -> EvaluationResult:
        """Evaluate predictions against ground truth.

        Args:
            predictions: List of prediction dicts (validated against schema)
            ground_truth: DataFrame with ground truth labels
            compute_ci: Whether to compute bootstrap confidence intervals

        Returns:
            EvaluationResult with all metrics and context
        """
        ...

    def evaluate_from_file(
        self,
        predictions_path: Path | str,
        split: str | None = None,
        compute_ci: bool = False,
    ) -> EvaluationResult:
        """Load predictions from file and evaluate.

        Args:
            predictions_path: Path to predictions JSONL file
            split: Dataset split to evaluate on (default: task default)
            compute_ci: Whether to compute confidence intervals

        Returns:
            EvaluationResult with all metrics
        """
        split = split or self.task_spec.default_split
        predictions_path = Path(predictions_path)
        prediction_file_checksum = compute_file_checksum(predictions_path)
        predictions = load_predictions_jsonl(predictions_path)
        ground_truth = self._load_ground_truth(split)
        result = self.evaluate(predictions, ground_truth, compute_ci)

        if result.context is not None:
            result.context.prediction_file_checksum = prediction_file_checksum

        return result

    def _load_ground_truth(self, split: str) -> pl.DataFrame:
        """Load ground truth from local parquet or HuggingFace dataset.

        Tries to load from local data/hf_dataset/ first, then falls back
        to HuggingFace Hub.

        Args:
            split: Dataset split to load

        Returns:
            Polars DataFrame with ground truth data
        """
        # Try local parquet files first
        local_path = Path("data/hf_dataset") / f"{split}.parquet"
        if local_path.exists():
            df = pl.read_parquet(local_path)
            # The 'text' field should be body-only (no title) to avoid label leakage.
            # If 'text' is missing (old dataset format), fall back to body-only.
            if "text" not in df.columns:
                if "body" in df.columns:
                    logger.warning(
                        "Dataset missing 'text' field. Using 'body' only. "
                        "Consider regenerating the dataset with prepare_hf_dataset.py"
                    )
                    df = df.with_columns(pl.col("body").alias("text"))
                elif "title" in df.columns:
                    logger.warning(
                        "Dataset missing 'text' and 'body' fields. Using 'title' as fallback."
                    )
                    df = df.with_columns(pl.col("title").alias("text"))
            return df

        # Fall back to HuggingFace Hub
        from datasets import Dataset

        dataset = load_dataset(
            self.task_spec.dataset_name,
            self.task_spec.dataset_config,
            split=split,
        )

        # Ensure we have a Dataset (not DatasetDict)
        if not isinstance(dataset, Dataset):
            raise TypeError(
                f"Expected Dataset, got {type(dataset).__name__}. "
                f"Make sure to specify a split."
            )

        # Convert to Polars DataFrame via Arrow
        result = pl.from_arrow(dataset.data.table)

        # Ensure we return a DataFrame, not a Series
        if isinstance(result, pl.Series):
            raise TypeError(
                "Expected DataFrame from Arrow table conversion, got Series"
            )

        return result

    def _create_result(
        self,
        metrics: dict[str, float],
        ground_truth: pl.DataFrame,
        split: str,
        per_class_metrics: dict[str, dict[str, float]] | None = None,
        confusion_matrix: list[list[int]] | None = None,
        per_query_metrics: dict[str, dict[str, float]] | None = None,
        num_correct: int | None = None,
        misclassified_ids: list[str] | None = None,
        confidence_intervals: dict[str, tuple[float, float]] | None = None,
        **extra_context: Any,
    ) -> EvaluationResult:
        """Create evaluation result with full context.

        Args:
            metrics: Dictionary of computed metrics
            ground_truth: Ground truth DataFrame (for checksum)
            split: Dataset split evaluated
            per_class_metrics: Per-class breakdown
            confusion_matrix: Confusion matrix
            per_query_metrics: Per-query breakdown
            num_correct: Number of correct predictions
            misclassified_ids: List of misclassified IDs
            confidence_intervals: Bootstrap CIs
            **extra_context: Additional context for EvaluationContext

        Returns:
            Complete EvaluationResult
        """
        # Compute dataset checksum
        data_records = ground_truth.to_dicts()
        dataset_checksum = compute_data_checksum(data_records)

        # Capture context
        context = EvaluationContext.capture(
            random_seed=self.random_seed,
            dataset_checksum=dataset_checksum,
            model_name=extra_context.pop("model_name", None),
            **extra_context,
        )

        # Get primary metric value
        primary_metric = self.task_spec.primary_metric
        primary_score = metrics.get(primary_metric, 0.0)

        return EvaluationResult(
            task=self.task_spec.name,
            task_type=self.task_spec.task_type.value,
            split=split,
            primary_metric=primary_metric,
            primary_score=primary_score,
            metrics=metrics,
            per_class_metrics=per_class_metrics,
            confusion_matrix=confusion_matrix,
            per_query_metrics=per_query_metrics,
            num_samples=len(ground_truth),
            num_correct=num_correct,
            misclassified_ids=misclassified_ids,
            confidence_intervals=confidence_intervals,
            context=context,
        )
