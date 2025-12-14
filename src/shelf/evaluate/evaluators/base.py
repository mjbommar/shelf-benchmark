"""Base evaluator class for SHELF tasks.

Evaluators are stateless - they take predictions and ground truth,
and return results. They don't hold data.

Supports:
- Filtering by data generation commit or model
- Stratified metrics by facets (form, LCC, register, etc.)
- Full provenance tracking for reproducibility
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl
from datasets import load_dataset

from shelf.evaluate.results import (
    DataProvenance,
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


# Standard facet fields available for stratification
STRATIFY_FIELDS = [
    "lcc",  # Library of Congress Classification
    "form",  # Document form (lecture, map, etc.)
    "form_category",  # Form category (broader)
    "topic",  # Topic
    "region",  # Geographic region
    "audience",  # Target audience
    "register",  # Writing register
    "model",  # Generating model (gpt-5.1, etc.)
    "git_commit",  # Data generation commit
]


class TaskEvaluator(ABC):
    """Base class for all task evaluators.

    Evaluators are stateless - they take predictions and ground truth,
    and return results. They don't hold data.

    Subclasses must implement:
    - evaluate(): Core evaluation logic

    Subclasses may optionally override:
    - _load_ground_truth(): Custom data loading
    - _validate_predictions(): Custom validation

    Features:
    - filter_by: Filter data by commit ID, generating model, or any field
    - stratify_by: Compute metrics stratified by facets (form, LCC, etc.)
    - Automatic provenance tracking for reproducibility
    """

    def __init__(
        self,
        task_spec: TaskSpec,
        random_seed: int = 42,
        filter_by: dict[str, str | list[str]] | None = None,
        stratify_by: str | list[str] | None = None,
    ):
        """Initialize evaluator.

        Args:
            task_spec: Task specification
            random_seed: Random seed for reproducibility
            filter_by: Filter data by field values. Examples:
                - {"git_commit": "abc123"} - single commit
                - {"model": "gpt-5.1"} - single model
                - {"lcc": ["A", "B", "C"]} - multiple values
            stratify_by: Field(s) to compute stratified metrics by.
                Examples: "form", "lcc", ["form", "register"]
        """
        self.task_spec = task_spec
        self.random_seed = random_seed
        self.filter_by = filter_by or {}
        self.stratify_by = (
            [stratify_by] if isinstance(stratify_by, str) else (stratify_by or [])
        )

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
        to HuggingFace Hub. Applies any configured filters.

        Args:
            split: Dataset split to load

        Returns:
            Polars DataFrame with ground truth data (filtered if filter_by is set)
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
        else:
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
            df = pl.from_arrow(dataset.data.table)

            # Ensure we return a DataFrame, not a Series
            if isinstance(df, pl.Series):
                raise TypeError(
                    "Expected DataFrame from Arrow table conversion, got Series"
                )

        # Apply filters if configured
        df = self._apply_filters(df)

        # Filter out records with empty text (data quality check)
        # This catches any bad records that made it into the dataset
        df = self._filter_empty_text(df)

        return df

    def _filter_empty_text(self, df: pl.DataFrame) -> pl.DataFrame:
        """Filter out records with empty or whitespace-only text.

        This is a data quality safeguard that removes records with empty
        text fields, which are typically failed generation artifacts.

        Args:
            df: Input DataFrame

        Returns:
            Filtered DataFrame with empty text records removed
        """
        text_field = self.task_spec.text_field

        if text_field not in df.columns:
            return df

        original_len = len(df)

        # Filter out null, empty string, and whitespace-only text
        df = df.filter(
            pl.col(text_field).is_not_null()
            & (pl.col(text_field).str.strip_chars() != "")
        )

        filtered_count = original_len - len(df)
        if filtered_count > 0:
            logger.warning(
                f"Filtered out {filtered_count} records with empty text field "
                f"(data quality issue). {len(df)} records remaining."
            )

        return df

    def _apply_filters(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply configured filters to DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            Filtered DataFrame
        """
        if not self.filter_by:
            return df

        original_len = len(df)

        for field, values in self.filter_by.items():
            if field not in df.columns:
                logger.warning(
                    f"Filter field '{field}' not found in data. Available: {df.columns}"
                )
                continue

            if isinstance(values, str):
                df = df.filter(pl.col(field) == values)
            elif isinstance(values, list):
                df = df.filter(pl.col(field).is_in(values))
            else:
                raise ValueError(
                    f"Filter value must be str or list, got {type(values)}"
                )

        if len(df) == 0:
            raise ValueError(
                f"Filter returned 0 rows. Original: {original_len}, "
                f"Filters: {self.filter_by}"
            )

        logger.info(
            f"Filtered data: {original_len} -> {len(df)} rows "
            f"({len(df) / original_len * 100:.1f}%)"
        )

        return df

    def _extract_provenance(self, df: pl.DataFrame) -> DataProvenance:
        """Extract provenance information from data.

        Args:
            df: DataFrame with ground truth data

        Returns:
            DataProvenance instance
        """
        return DataProvenance.from_data(
            data=df.to_dicts(),
            filters_applied=self.filter_by if self.filter_by else None,
        )

    def _create_result(
        self,
        metrics: dict[str, float],
        ground_truth: pl.DataFrame,
        split: str,
        per_class_metrics: dict[str, dict[str, float]] | None = None,
        confusion_matrix: list[list[int]] | None = None,
        confusion_matrix_labels: list[str] | None = None,
        top_confusions: list[dict[str, Any]] | None = None,
        stratified_confusion_matrices: (
            dict[str, dict[str, list[list[int]]]] | None
        ) = None,
        per_query_metrics: dict[str, dict[str, float]] | None = None,
        num_correct: int | None = None,
        misclassified_ids: list[str] | None = None,
        confidence_intervals: dict[str, tuple[float, float]] | None = None,
        stratified_metrics: dict[str, dict[str, float]] | None = None,
        **extra_context: Any,
    ) -> EvaluationResult:
        """Create evaluation result with full context.

        Args:
            metrics: Dictionary of computed metrics
            ground_truth: Ground truth DataFrame (for checksum)
            split: Dataset split evaluated
            per_class_metrics: Per-class breakdown
            confusion_matrix: Confusion matrix
            confusion_matrix_labels: Labels for confusion matrix indices
            top_confusions: Pre-computed top confused class pairs
            stratified_confusion_matrices: Confusion matrices by stratum
                e.g., {"audience": {"Physicians": [[...]], ...}}
            per_query_metrics: Per-query breakdown
            num_correct: Number of correct predictions
            misclassified_ids: List of misclassified IDs
            confidence_intervals: Bootstrap CIs
            stratified_metrics: Metrics broken down by stratify_by field
            **extra_context: Additional context for EvaluationContext

        Returns:
            Complete EvaluationResult
        """
        # Compute dataset checksum
        data_records = ground_truth.to_dicts()
        dataset_checksum = compute_data_checksum(data_records)

        # Extract provenance
        provenance = self._extract_provenance(ground_truth)

        # Capture context with provenance info
        context = EvaluationContext.capture(
            random_seed=self.random_seed,
            dataset_checksum=dataset_checksum,
            model_name=extra_context.pop("model_name", None),
            data_commit=provenance.primary_commit,
            **extra_context,
        )

        # Get primary metric value
        primary_metric = self.task_spec.primary_metric
        primary_score = metrics.get(primary_metric, 0.0)

        # Determine stratify_by field for result
        stratify_by_field = self.stratify_by[0] if len(self.stratify_by) == 1 else None
        if len(self.stratify_by) > 1:
            stratify_by_field = ",".join(self.stratify_by)

        return EvaluationResult(
            task=self.task_spec.name,
            task_type=self.task_spec.task_type.value,
            split=split,
            primary_metric=primary_metric,
            primary_score=primary_score,
            metrics=metrics,
            per_class_metrics=per_class_metrics,
            confusion_matrix=confusion_matrix,
            confusion_matrix_labels=confusion_matrix_labels,
            top_confusions=top_confusions,
            stratified_confusion_matrices=stratified_confusion_matrices,
            per_query_metrics=per_query_metrics,
            num_samples=len(ground_truth),
            num_correct=num_correct,
            misclassified_ids=misclassified_ids,
            confidence_intervals=confidence_intervals,
            stratified_metrics=stratified_metrics,
            stratify_by=stratify_by_field,
            data_provenance=provenance,
            context=context,
        )

    def _compute_stratified_metrics(
        self,
        ground_truth: pl.DataFrame,
        predictions: list[dict[str, Any]],
        compute_metrics_fn: Any,  # Callable that computes metrics
    ) -> dict[str, dict[str, float]] | None:
        """Compute metrics stratified by configured fields.

        Args:
            ground_truth: DataFrame with ground truth data
            predictions: List of predictions
            compute_metrics_fn: Function to compute metrics for a subset

        Returns:
            Dict mapping stratum values to metrics, or None if no stratification
        """
        if not self.stratify_by:
            return None

        stratified: dict[str, dict[str, float]] = {}
        id_field = self.task_spec.id_field

        for field in self.stratify_by:
            if field not in ground_truth.columns:
                logger.warning(f"Stratify field '{field}' not found in data. Skipping.")
                continue

            # Group by field value
            unique_values = ground_truth[field].unique().to_list()
            for value in unique_values:
                if value is None:
                    continue

                stratum_key = f"{field}={value}"
                subset = ground_truth.filter(pl.col(field) == value)

                # Get predictions for this subset
                subset_ids = set(subset[id_field].to_list())
                subset_preds = [p for p in predictions if p["id"] in subset_ids]

                if not subset_preds:
                    continue

                # Compute metrics for this stratum
                try:
                    stratum_metrics = compute_metrics_fn(subset_preds, subset)
                    stratified[stratum_key] = stratum_metrics
                except Exception as e:
                    logger.warning(f"Failed to compute metrics for {stratum_key}: {e}")

        return stratified if stratified else None
