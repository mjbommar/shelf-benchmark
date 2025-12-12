"""Classification evaluator for SHELF tasks.

Evaluates models on single-label classification tasks like LCC classification,
LCGFT form classification, audience classification, and register classification.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import polars as pl

from shelf.evaluate.evaluators.base import TaskEvaluator
from shelf.evaluate.metrics.classification import compute_classification_metrics
from shelf.evaluate.results import EvaluationResult
from shelf.evaluate.schemas import (
    ValidationError,
    validate_classification_predictions,
)
from shelf.evaluate.tasks import TaskSpec

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
    ) -> EvaluationResult:
        """Evaluate classification predictions.

        Args:
            predictions: List of {"id": str, "prediction": str}
            ground_truth: DataFrame with ground truth labels
            compute_ci: Whether to compute confidence intervals (not yet implemented)

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

        # Get ground truth
        y_true: list[str] = []
        y_pred: list[str] = []
        misclassified_ids: list[str] = []

        for row in ground_truth.iter_rows(named=True):
            doc_id = row[id_field]
            true_label = row[label_field]

            if doc_id not in pred_dict:
                logger.warning(f"Missing prediction for document: {doc_id}")
                continue

            predicted_label = pred_dict[doc_id]
            y_true.append(true_label)
            y_pred.append(predicted_label)

            if true_label != predicted_label:
                misclassified_ids.append(doc_id)

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

        return self._create_result(
            metrics=metrics,
            ground_truth=ground_truth,
            split=self.task_spec.default_split,
            per_class_metrics=per_class,
            confusion_matrix=conf_matrix,
            num_correct=metrics.get("num_correct"),
            misclassified_ids=misclassified_ids[:100],  # Limit to first 100
            labels=labels_order,
        )

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

        return self._create_result(
            metrics=metrics,
            ground_truth=ground_truth,
            split=split,
            per_class_metrics=per_class,
            confusion_matrix=conf_matrix,
            num_correct=metrics.get("num_correct"),
            misclassified_ids=misclassified_ids[:100],
            model_name=getattr(classifier, "model_name", None),
            labels=labels_order,
        )

    def evaluate_embedder_with_classifier(
        self,
        embedder: "TextEmbedder",
        split: str | None = None,
        train_split: str = "train",
        batch_size: int = 32,
        show_progress: bool = True,
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

        # Build misclassified IDs
        misclassified_ids = [
            doc_id
            for doc_id, true_label, pred_label in zip(test_ids, y_true, y_pred)
            if true_label != pred_label
        ]

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

        return self._create_result(
            metrics=metrics,
            ground_truth=test_df,
            split=split,
            per_class_metrics=per_class,
            confusion_matrix=conf_matrix,
            num_correct=metrics.get("num_correct"),
            misclassified_ids=misclassified_ids[:100],
            model_name=embedder.model_name,
            embedding_dim=embedder.embedding_dim,
            classifier="LogisticRegression",
            train_size=len(train_texts),
            labels=labels_order,
        )
