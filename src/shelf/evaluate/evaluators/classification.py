"""Classification evaluator for SHELF tasks.

Evaluates models on single-label classification tasks like LCC classification,
LCGFT form classification, audience classification, and register classification.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import polars as pl

from shelf.evaluate.evaluators.base import TaskEvaluator
from shelf.evaluate.metrics.classification import (
    compute_classification_metrics,
    compute_stratified_confusion_matrices,
    extract_top_confusions,
)
from shelf.evaluate.results import (
    EvaluationResult,
    PerSampleResult,
    PerSampleResults,
)
from shelf.evaluate.schemas import (
    ValidationError,
    validate_classification_predictions,
)
from shelf.evaluate.tasks import TaskSpec

# Metadata fields to capture for stratification analysis
STRATIFICATION_FIELDS = [
    "form",
    "form_category",
    "register",
    "audience",
    "lcc",
    "topic",
    "region",
]

if TYPE_CHECKING:
    from shelf.evaluate.adapters.protocols import TextClassifier, TextEmbedder

logger = logging.getLogger(__name__)


class ClassificationEvaluator(TaskEvaluator):
    """Evaluator for single-label classification tasks.

    Supports two modes:
    1. From predictions file: Pre-computed class labels
    2. From classifier model: Direct prediction via TextClassifier protocol

    For classification:
    - Ground truth labels come from the specified label_field
    - Predictions are single class labels per document
    - Primary metric is typically macro_f1 (configurable via task spec)

    Example:
        from shelf.evaluate.evaluators import ClassificationEvaluator
        from shelf.evaluate.registry import get_task

        task_spec = get_task("lcc_classification")
        evaluator = ClassificationEvaluator(task_spec)

        # From predictions file
        result = evaluator.evaluate_from_file("predictions.jsonl")

        # Or from classifier
        result = evaluator.evaluate_classifier(classifier, split="test")
        print(result.summary())
    """

    def __init__(
        self,
        task_spec: TaskSpec,
        random_seed: int = 42,
    ):
        """Initialize classification evaluator.

        Args:
            task_spec: Task specification
            random_seed: Random seed for reproducibility
        """
        super().__init__(task_spec, random_seed)

    def evaluate(
        self,
        predictions: list[dict[str, Any]],
        ground_truth: pl.DataFrame,
        compute_ci: bool = False,
        save_samples: bool = False,
        model_key: str | None = None,
    ) -> EvaluationResult:
        """Evaluate classification predictions.

        Args:
            predictions: List of {"id": str, "prediction": str}
            ground_truth: DataFrame with ground truth labels
            compute_ci: Whether to compute confidence intervals (not yet implemented)
            save_samples: Whether to capture per-sample results for detailed analysis
            model_key: Model identifier for per-sample results

        Returns:
            EvaluationResult with classification metrics
        """
        id_field = self.task_spec.id_field
        label_field = self.task_spec.label_field

        # Validate predictions against ground truth
        validation = validate_classification_predictions(
            predictions=predictions,
            ground_truth_ids=set(ground_truth[id_field].to_list()),
            label_space=set(self.task_spec.label_space)
            if self.task_spec.label_space
            else None,
        )
        if not validation.valid:
            raise ValidationError(validation.errors)

        # Build prediction dict: id -> predicted_label
        pred_dict: dict[str, str] = {}
        for pred in predictions:
            doc_id = pred["id"]
            predicted_label = pred["prediction"]
            pred_dict[doc_id] = predicted_label

        # Get ground truth and build per-sample results
        y_true: list[str] = []
        y_pred: list[str] = []
        misclassified_ids: list[str] = []
        per_sample_list: list[PerSampleResult] = []

        # Get available columns for metadata extraction
        available_cols = set(ground_truth.columns)
        metadata_fields = [f for f in STRATIFICATION_FIELDS if f in available_cols]

        # Collect strata values for stratified confusion matrices
        # Also add length_bucket as a stratification field
        strata_fields = metadata_fields + ["length_bucket"]
        strata: dict[str, list[str | None]] = {f: [] for f in strata_fields}

        for row in ground_truth.iter_rows(named=True):
            doc_id = row[id_field]
            true_label = row[label_field]

            if doc_id not in pred_dict:
                logger.warning(f"Missing prediction for document: {doc_id}")
                continue

            predicted_label = pred_dict[doc_id]
            is_correct = true_label == predicted_label

            y_true.append(true_label)
            y_pred.append(predicted_label)

            if not is_correct:
                misclassified_ids.append(doc_id)

            # Collect strata values for stratified confusion matrices
            for field in metadata_fields:
                value = row.get(field)
                strata[field].append(str(value) if value is not None else None)

            # Compute length bucket and add to strata
            text_field = self.task_spec.text_field
            length_bucket: str | None = None
            if text_field in row and row[text_field]:
                text_len = len(row[text_field])
                if text_len < 500:
                    length_bucket = "short"
                elif text_len < 2000:
                    length_bucket = "medium"
                else:
                    length_bucket = "long"
            strata["length_bucket"].append(length_bucket)

            # Capture per-sample result with metadata for stratification
            if save_samples:
                metadata: dict[str, Any] = {}
                for field in metadata_fields:
                    if field in row and row[field] is not None:
                        metadata[field] = row[field]

                # Add text length bucket for length analysis
                if length_bucket:
                    metadata["length_bucket"] = length_bucket
                    metadata["text_length"] = len(row[text_field])

                per_sample_list.append(
                    PerSampleResult(
                        id=doc_id,
                        y_true=true_label,
                        y_pred=predicted_label,
                        correct=is_correct,
                        metadata=metadata,
                    )
                )

        if not y_true:
            raise ValueError("No valid predictions found matching ground truth IDs")

        # Get label space from task spec or infer from data
        labels = None
        if self.task_spec.label_space:
            labels = list(self.task_spec.label_space)

        # Compute metrics
        metrics_result = compute_classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            labels=labels,
            compute_per_class=True,
            compute_confusion_matrix=True,
        )

        # Extract components
        per_class = metrics_result.pop("per_class", None)
        conf_matrix = metrics_result.pop("confusion_matrix", None)
        labels_order = metrics_result.pop("labels", None)

        # Build metrics dict (just the scalar metrics)
        metrics = {
            k: v for k, v in metrics_result.items() if isinstance(v, (int, float))
        }

        # Compute stratified confusion matrices
        stratified_conf_matrices = compute_stratified_confusion_matrices(
            y_true=y_true,
            y_pred=y_pred,
            strata=strata,
            labels=labels_order,
            min_samples=10,
        )

        # Extract top confusions from confusion matrix
        top_confusions = None
        if conf_matrix and labels_order:
            top_confusions = extract_top_confusions(
                confusion_matrix=conf_matrix,
                labels=labels_order,
                n=10,
                min_count=1,
            )

        result = self._create_result(
            metrics=metrics,
            ground_truth=ground_truth,
            split=self.task_spec.default_split,
            per_class_metrics=per_class,
            confusion_matrix=conf_matrix,
            confusion_matrix_labels=labels_order,
            top_confusions=top_confusions,
            stratified_confusion_matrices=stratified_conf_matrices,
            num_correct=metrics.get("num_correct"),
            misclassified_ids=misclassified_ids[:100],  # Limit to first 100
        )

        # Attach per-sample results if captured
        if save_samples and per_sample_list:
            result.per_sample_results = PerSampleResults(
                task=self.task_spec.name,
                task_type="classification",
                model_key=model_key or "unknown",
                split=self.task_spec.default_split,
                samples=per_sample_list,
            )

        return result

    def evaluate_classifier(
        self,
        classifier: "TextClassifier",
        split: str | None = None,
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> EvaluationResult:
        """Evaluate a classifier directly on classification task.

        Args:
            classifier: TextClassifier instance
            split: Dataset split (default: task default, usually "test")
            batch_size: Batch size for prediction
            show_progress: Whether to show progress bars

        Returns:
            EvaluationResult with classification metrics
        """
        split = split or self.task_spec.default_split

        logger.info(f"Loading data from split: {split}")

        # Load ground truth
        ground_truth = self._load_ground_truth(split)

        logger.info(f"Documents to classify: {len(ground_truth)}")

        # Get text field
        text_field = self.task_spec.text_field
        id_field = self.task_spec.id_field
        label_field = self.task_spec.label_field

        # Extract texts and IDs
        texts = ground_truth[text_field].to_list()
        doc_ids = ground_truth[id_field].to_list()
        y_true = ground_truth[label_field].to_list()

        # Predict
        logger.info("Running classifier predictions...")
        y_pred = classifier.predict(texts, batch_size=batch_size)

        if len(y_pred) != len(doc_ids):
            raise ValueError(
                f"Classifier returned {len(y_pred)} predictions for {len(doc_ids)} documents "
                f"({type(classifier).__name__})"
            )

        # Build misclassified IDs
        misclassified_ids = [
            doc_id
            for doc_id, true_label, pred_label in zip(doc_ids, y_true, y_pred)
            if true_label != pred_label
        ]

        # Get label space from task spec or infer from data
        labels = None
        if self.task_spec.label_space:
            labels = list(self.task_spec.label_space)

        # Compute metrics
        metrics_result = compute_classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            labels=labels,
            compute_per_class=True,
            compute_confusion_matrix=True,
        )

        # Extract components
        per_class = metrics_result.pop("per_class", None)
        conf_matrix = metrics_result.pop("confusion_matrix", None)
        labels_order = metrics_result.pop("labels", None)

        # Build metrics dict (just the scalar metrics)
        metrics = {
            k: v for k, v in metrics_result.items() if isinstance(v, (int, float))
        }

        # Extract top confusions from confusion matrix
        top_confusions = None
        if conf_matrix and labels_order:
            top_confusions = extract_top_confusions(
                confusion_matrix=conf_matrix,
                labels=labels_order,
                n=10,
                min_count=1,
            )

        return self._create_result(
            metrics=metrics,
            ground_truth=ground_truth,
            split=split,
            per_class_metrics=per_class,
            confusion_matrix=conf_matrix,
            confusion_matrix_labels=labels_order,
            top_confusions=top_confusions,
            num_correct=metrics.get("num_correct"),
            misclassified_ids=misclassified_ids[:100],
            model_name=getattr(classifier, "model_name", None),
        )

    def evaluate_embedder_with_classifier(
        self,
        embedder: "TextEmbedder",
        split: str | None = None,
        train_split: str = "train",
        batch_size: int = 32,
        show_progress: bool = True,
        save_samples: bool = False,
    ) -> EvaluationResult:
        """Evaluate embedder by training a simple classifier on embeddings.

        This method:
        1. Encodes train split with embedder
        2. Trains a LogisticRegression classifier
        3. Encodes test split and predicts
        4. Evaluates predictions

        This is useful for evaluating embedding quality on classification tasks.

        Args:
            embedder: TextEmbedder instance
            split: Evaluation split (default: task default, usually "test")
            train_split: Training split for classifier
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bars
            save_samples: Whether to capture per-sample results for detailed analysis

        Returns:
            EvaluationResult with classification metrics
        """
        from sklearn.linear_model import LogisticRegression

        split = split or self.task_spec.default_split

        logger.info(f"Loading train data from split: {train_split}")
        train_df = self._load_ground_truth(train_split)

        logger.info(f"Loading test data from split: {split}")
        test_df = self._load_ground_truth(split)

        # Get field names
        text_field = self.task_spec.text_field
        id_field = self.task_spec.id_field
        label_field = self.task_spec.label_field

        # Extract data
        train_texts = train_df[text_field].to_list()
        train_labels = train_df[label_field].to_list()

        test_texts = test_df[text_field].to_list()
        test_ids = test_df[id_field].to_list()
        y_true = test_df[label_field].to_list()

        logger.info(f"Train: {len(train_texts)}, Test: {len(test_texts)}")

        # Encode train
        logger.info("Encoding training data...")
        train_embeddings = embedder.encode(
            train_texts,
            batch_size=batch_size,
            show_progress=show_progress,
        )

        # Encode test
        logger.info("Encoding test data...")
        test_embeddings = embedder.encode(
            test_texts,
            batch_size=batch_size,
            show_progress=show_progress,
        )

        if len(train_embeddings) != len(train_texts):
            raise ValueError(
                f"Embedder returned {len(train_embeddings)} train embeddings for {len(train_texts)} texts "
                f"({type(embedder).__name__})"
            )

        if len(test_embeddings) != len(test_texts):
            raise ValueError(
                f"Embedder returned {len(test_embeddings)} test embeddings for {len(test_texts)} texts "
                f"({type(embedder).__name__})"
            )

        # Train classifier
        logger.info("Training LogisticRegression classifier...")
        clf = LogisticRegression(
            max_iter=1000,
            random_state=self.random_seed,
        )
        clf.fit(train_embeddings, train_labels)

        # Predict
        logger.info("Predicting on test data...")
        y_pred = clf.predict(test_embeddings).tolist()

        if len(y_pred) != len(test_ids):
            raise ValueError(
                f"Classifier produced {len(y_pred)} predictions for {len(test_ids)} test samples"
            )

        # Build misclassified IDs and per-sample results
        misclassified_ids: list[str] = []
        per_sample_list: list[PerSampleResult] = []

        # Get available columns for metadata extraction
        available_cols = set(test_df.columns)
        metadata_fields = [f for f in STRATIFICATION_FIELDS if f in available_cols]

        # Collect strata values for stratified confusion matrices
        strata_fields = metadata_fields + ["length_bucket"]
        strata: dict[str, list[str | None]] = {f: [] for f in strata_fields}

        for i, (doc_id, true_label, pred_label) in enumerate(
            zip(test_ids, y_true, y_pred)
        ):
            is_correct = true_label == pred_label

            if not is_correct:
                misclassified_ids.append(doc_id)

            # Get row from test_df for metadata
            row = test_df.row(i, named=True)

            # Collect strata values
            for field in metadata_fields:
                value = row.get(field)
                strata[field].append(str(value) if value is not None else None)

            # Compute length bucket
            length_bucket: str | None = None
            if text_field in row and row[text_field]:
                text_len = len(row[text_field])
                if text_len < 500:
                    length_bucket = "short"
                elif text_len < 2000:
                    length_bucket = "medium"
                else:
                    length_bucket = "long"
            strata["length_bucket"].append(length_bucket)

            if save_samples:
                metadata: dict[str, Any] = {}

                for field in metadata_fields:
                    if field in row and row[field] is not None:
                        metadata[field] = row[field]

                # Add text length bucket
                if length_bucket:
                    metadata["length_bucket"] = length_bucket
                    metadata["text_length"] = len(row[text_field])

                per_sample_list.append(
                    PerSampleResult(
                        id=doc_id,
                        y_true=true_label,
                        y_pred=pred_label,
                        correct=is_correct,
                        metadata=metadata,
                    )
                )

        # Get label space
        labels = None
        if self.task_spec.label_space:
            labels = list(self.task_spec.label_space)

        # Compute metrics
        metrics_result = compute_classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            labels=labels,
            compute_per_class=True,
            compute_confusion_matrix=True,
        )

        # Extract components
        per_class = metrics_result.pop("per_class", None)
        conf_matrix = metrics_result.pop("confusion_matrix", None)
        labels_order = metrics_result.pop("labels", None)

        # Build metrics dict
        metrics = {
            k: v for k, v in metrics_result.items() if isinstance(v, (int, float))
        }

        # Compute stratified confusion matrices
        stratified_conf_matrices = compute_stratified_confusion_matrices(
            y_true=y_true,
            y_pred=y_pred,
            strata=strata,
            labels=labels_order,
            min_samples=10,
        )

        # Extract top confusions from confusion matrix
        top_confusions = None
        if conf_matrix and labels_order:
            top_confusions = extract_top_confusions(
                confusion_matrix=conf_matrix,
                labels=labels_order,
                n=10,
                min_count=1,
            )

        result = self._create_result(
            metrics=metrics,
            ground_truth=test_df,
            split=split,
            per_class_metrics=per_class,
            confusion_matrix=conf_matrix,
            confusion_matrix_labels=labels_order,
            top_confusions=top_confusions,
            stratified_confusion_matrices=stratified_conf_matrices,
            num_correct=metrics.get("num_correct"),
            misclassified_ids=misclassified_ids[:100],
            model_name=embedder.model_name,
            embedding_dim=embedder.embedding_dim,
            classifier="LogisticRegression",
            train_size=len(train_texts),
        )

        # Attach per-sample results if captured
        if save_samples and per_sample_list:
            result.per_sample_results = PerSampleResults(
                task=self.task_spec.name,
                task_type="classification",
                model_key=embedder.model_name or "unknown",
                split=split,
                samples=per_sample_list,
            )

        return result
