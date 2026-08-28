"""Multi-label classification evaluator for SHELF tasks.

Evaluates models on multi-label classification tasks like topic
classification, where each document carries 1-4 labels from a fixed
vocabulary.

Mirrors :class:`~shelf.evaluate.evaluators.classification.ClassificationEvaluator`
(same train/test separation, same frozen-embedding + logistic-regression
protocol, same per-sample capture) so that single-label and multi-label
results are directly comparable. The only structural difference is that
predictions are *sets* of labels rather than single labels, which changes the
metric set (see :mod:`shelf.evaluate.metrics.multilabel`).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from shelf.evaluate.evaluators.base import TaskEvaluator
from shelf.evaluate.metrics.multilabel import (
    binarize_labels,
    compute_multilabel_metrics,
)
from shelf.evaluate.results import (
    EvaluationResult,
    PerSampleResult,
    PerSampleResults,
)
from shelf.evaluate.schemas import (
    MultiLabelPrediction,
    ValidationError,
    ValidationResult,
)
from shelf.evaluate.tasks import TaskSpec

# Metadata fields to capture for stratification analysis.
# Kept identical to the single-label evaluator so downstream analysis code
# sees the same metadata keys for both task families.
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
    from shelf.evaluate.adapters.protocols import TextEmbedder

logger = logging.getLogger(__name__)


def validate_multilabel_predictions(
    predictions: list[dict[str, Any]],
    ground_truth_ids: set[str],
    label_space: set[str] | None = None,
) -> ValidationResult:
    """Validate multi-label predictions against schema and ground truth.

    Checks the same invariants as
    :func:`shelf.evaluate.schemas.validate_classification_predictions`:
    schema conformance, no duplicate IDs, all IDs known, labels inside the
    label space, and no missing predictions.

    Note: an *empty* prediction list is a legitimate model output (the model
    declined to assign any label) but ``MultiLabelPrediction`` rejects it, so
    empty lists are validated structurally here and reported as a warning
    rather than an error.

    Args:
        predictions: List of {"id": str, "predictions": list[str]} dicts
        ground_truth_ids: Set of valid document IDs
        label_space: Valid labels (None = any label allowed)

    Returns:
        ValidationResult with errors and warnings
    """
    errors: list[str] = []
    warnings: list[str] = []
    seen_ids: set[str] = set()

    for i, pred in enumerate(predictions):
        raw_labels = pred.get("predictions")

        # An empty prediction set is meaningful but is rejected by the pydantic
        # schema, so handle it structurally before delegating.
        if isinstance(raw_labels, list) and len(raw_labels) == 0:
            doc_id = pred.get("id")
            if not isinstance(doc_id, str):
                errors.append(f"Prediction {i}: 'id' must be a string")
                continue
            warnings.append(f"Prediction {i}: empty prediction set for '{doc_id}'")
            labels: list[str] = []
        else:
            try:
                validated = MultiLabelPrediction.model_validate(pred)
            except Exception as e:
                errors.append(f"Prediction {i}: {e}")
                continue
            doc_id = validated.id
            labels = validated.predictions

        if doc_id in seen_ids:
            errors.append(f"Prediction {i}: duplicate id '{doc_id}'")
        seen_ids.add(doc_id)

        if doc_id not in ground_truth_ids:
            errors.append(f"Prediction {i}: unknown id '{doc_id}'")

        if len(labels) != len(set(labels)):
            errors.append(f"Prediction {i}: duplicate labels for '{doc_id}'")

        if label_space is not None:
            for label in labels:
                if label not in label_space:
                    errors.append(
                        f"Prediction {i}: invalid label '{label}' "
                        f"(valid: {sorted(label_space)[:5]}...)"
                    )

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


class MultiLabelClassificationEvaluator(TaskEvaluator):
    """Evaluator for multi-label classification tasks.

    Supports two modes:
    1. From predictions file: Pre-computed label sets per document
    2. From an embedder: Frozen embeddings + One-vs-Rest logistic regression

    For multi-label classification:
    - Ground truth is a list-valued column (e.g., ``topics``)
    - Predictions are *sets* of labels per document
    - Primary metric is typically macro_f1 (configurable via task spec)

    Example:
        from shelf.evaluate.evaluators import MultiLabelClassificationEvaluator
        from shelf.evaluate.registry import get_task

        task_spec = get_task("topic_classification")
        evaluator = MultiLabelClassificationEvaluator(task_spec)

        # From predictions file
        result = evaluator.evaluate_from_file("predictions.jsonl")

        # Or from an embedder
        result = evaluator.evaluate_embedder_with_classifier(embedder)
        print(result.summary())
    """

    def __init__(
        self,
        task_spec: TaskSpec,
        random_seed: int = 42,
        filter_by: dict[str, str | list[str]] | None = None,
        stratify_by: str | list[str] | None = None,
    ):
        """Initialize multi-label classification evaluator.

        Args:
            task_spec: Task specification
            random_seed: Random seed for reproducibility
            filter_by: Filter data by field values
            stratify_by: Field(s) to compute stratified metrics by
        """
        super().__init__(task_spec, random_seed, filter_by, stratify_by)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _label_vocabulary(self, *label_sets: list[list[str]]) -> list[str]:
        """Resolve the ordered label vocabulary.

        Prefers the task spec's declared label space so that the metric
        denominators (and per-label breakdown) are stable regardless of which
        labels happen to appear in a particular split or subset. Falls back to
        the sorted union observed in the data.
        """
        if self.task_spec.label_space:
            return list(self.task_spec.label_space)

        observed: set[str] = set()
        for sets in label_sets:
            for labels in sets:
                observed.update(labels)
        return sorted(observed)

    @staticmethod
    def _normalize_label_list(value: Any) -> list[str]:
        """Coerce a ground-truth cell into a list of label strings.

        Polars returns list columns as Python lists (or numpy arrays when the
        frame came from Arrow); nulls become None.
        """
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return [str(v) for v in value if v is not None]

    def _length_bucket(self, row: dict[str, Any]) -> str | None:
        """Compute the text length bucket for a ground truth row."""
        text_field = self.task_spec.text_field
        text = row.get(text_field)
        if not text:
            return None
        text_len = len(text)
        if text_len < 500:
            return "short"
        if text_len < 2000:
            return "medium"
        return "long"

    def _build_result(
        self,
        *,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_score: np.ndarray | None,
        labels: list[str],
        ground_truth: pl.DataFrame,
        split: str,
        misclassified_ids: list[str],
        per_sample_list: list[PerSampleResult],
        save_samples: bool,
        model_key: str,
        compute_ci: bool = False,
        **extra_context: Any,
    ) -> EvaluationResult:
        """Compute metrics and assemble the EvaluationResult."""
        metrics_result = compute_multilabel_metrics(
            y_true=y_true,
            y_pred=y_pred,
            labels=labels,
            y_score=y_score,
            compute_per_label=True,
        )

        per_label = metrics_result.pop("per_label", None)
        metrics_result.pop("labels", None)
        num_exact_match = metrics_result.get("num_exact_match")

        metrics = {
            k: v for k, v in metrics_result.items() if isinstance(v, (int, float))
        }

        confidence_intervals = None
        if compute_ci:
            confidence_intervals = self._bootstrap_confidence_intervals(
                y_true=y_true,
                y_pred=y_pred,
            )

        result = self._create_result(
            metrics=metrics,
            ground_truth=ground_truth,
            split=split,
            per_class_metrics=per_label,
            num_correct=num_exact_match,
            misclassified_ids=misclassified_ids[:100],
            confidence_intervals=confidence_intervals,
            **extra_context,
        )

        if save_samples and per_sample_list:
            result.per_sample_results = PerSampleResults(
                task=self.task_spec.name,
                task_type=self.task_spec.task_type.value,
                model_key=model_key,
                split=split,
                samples=per_sample_list,
            )

        return result

    def _bootstrap_confidence_intervals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_bootstrap: int = 1000,
        ci_level: float = 0.95,
        block_size: int = 100,
    ) -> dict[str, tuple[float, float]]:
        """Bootstrap CIs for the F1 family by resampling documents.

        Resampling is done via per-document multiplicity weights rather than
        by materializing a resampled indicator matrix. Every reported metric
        is a closed-form function of the per-document (or per-label) TP/FP/FN
        counts, so a whole block of bootstrap replicates reduces to a single
        matrix product. The naive loop is prohibitively slow at SHELF scale
        (8,507 documents x 112 labels x 1,000 replicates); this is exact, not
        an approximation of it.

        Args:
            y_true: Binary indicator matrix of ground truth labels
            y_pred: Binary indicator matrix of predicted labels
            n_bootstrap: Number of bootstrap resamples
            ci_level: Confidence level
            block_size: Replicates computed per matrix product (memory knob)

        Returns:
            Dict mapping metric name to (lower, upper)
        """
        rng = np.random.default_rng(self.random_seed)
        n = y_true.shape[0]

        yt = np.asarray(y_true, dtype=bool)
        yp = np.asarray(y_pred, dtype=bool)

        # Per-document, per-label contingency contributions.
        tp = (yt & yp).astype(np.float64)
        fp = (~yt & yp).astype(np.float64)
        fn = (yt & ~yp).astype(np.float64)

        # Per-document scalars for the row-averaged metrics.
        row_tp = tp.sum(axis=1)
        row_denom = 2 * row_tp + fp.sum(axis=1) + fn.sum(axis=1)
        row_f1 = np.divide(2 * row_tp, row_denom, out=np.zeros(n), where=row_denom > 0)
        row_exact = (yt == yp).all(axis=1).astype(np.float64)

        def _f1(t: np.ndarray, p: np.ndarray, f: np.ndarray) -> np.ndarray:
            denom = 2 * t + p + f
            return np.divide(2 * t, denom, out=np.zeros_like(denom), where=denom > 0)

        collected: dict[str, list[np.ndarray]] = {
            "micro_f1": [],
            "macro_f1": [],
            "samples_f1": [],
            "subset_accuracy": [],
        }

        remaining = n_bootstrap
        while remaining > 0:
            block = min(block_size, remaining)
            remaining -= block

            # Multiplicity weights are exactly equivalent to sampling n
            # document indices with replacement.
            weights = rng.multinomial(n, np.full(n, 1.0 / n), size=block).astype(
                np.float64
            )

            tp_l = weights @ tp
            fp_l = weights @ fp
            fn_l = weights @ fn

            collected["macro_f1"].append(_f1(tp_l, fp_l, fn_l).mean(axis=1))
            collected["micro_f1"].append(
                _f1(tp_l.sum(axis=1), fp_l.sum(axis=1), fn_l.sum(axis=1))
            )
            collected["samples_f1"].append(weights @ row_f1 / n)
            collected["subset_accuracy"].append(weights @ row_exact / n)

        alpha = 1 - ci_level
        return {
            name: (
                float(np.percentile(np.concatenate(parts), 100 * alpha / 2)),
                float(np.percentile(np.concatenate(parts), 100 * (1 - alpha / 2))),
            )
            for name, parts in collected.items()
        }

    # ------------------------------------------------------------------
    # Evaluation entry points
    # ------------------------------------------------------------------

    def evaluate(
        self,
        predictions: list[dict[str, Any]],
        ground_truth: pl.DataFrame,
        compute_ci: bool = False,
        save_samples: bool = False,
        model_key: str | None = None,
    ) -> EvaluationResult:
        """Evaluate multi-label predictions.

        Args:
            predictions: List of {"id": str, "predictions": list[str]}
            ground_truth: DataFrame with ground truth label lists
            compute_ci: Whether to compute bootstrap confidence intervals
            save_samples: Whether to capture per-sample results
            model_key: Model identifier for per-sample results

        Returns:
            EvaluationResult with multi-label metrics
        """
        id_field = self.task_spec.id_field
        label_field = self.task_spec.label_field

        validation = validate_multilabel_predictions(
            predictions=predictions,
            ground_truth_ids=set(ground_truth[id_field].to_list()),
            label_space=set(self.task_spec.label_space)
            if self.task_spec.label_space
            else None,
        )
        if not validation.valid:
            raise ValidationError(validation.errors)
        if validation.warnings:
            # Empty prediction sets are common and legitimate; summarise rather
            # than emitting one line per document.
            logger.warning(
                f"{len(validation.warnings)} prediction(s) raised warnings "
                f"(first: {validation.warnings[0]})"
            )

        pred_dict: dict[str, list[str]] = {
            pred["id"]: list(pred.get("predictions") or []) for pred in predictions
        }

        true_sets: list[list[str]] = []
        pred_sets: list[list[str]] = []
        misclassified_ids: list[str] = []
        per_sample_list: list[PerSampleResult] = []

        available_cols = set(ground_truth.columns)
        metadata_fields = [f for f in STRATIFICATION_FIELDS if f in available_cols]

        for row in ground_truth.iter_rows(named=True):
            doc_id = row[id_field]
            if doc_id not in pred_dict:
                logger.warning(f"Missing prediction for document: {doc_id}")
                continue

            true_labels = self._normalize_label_list(row[label_field])
            predicted_labels = pred_dict[doc_id]
            is_exact = set(true_labels) == set(predicted_labels)

            true_sets.append(true_labels)
            pred_sets.append(predicted_labels)

            if not is_exact:
                misclassified_ids.append(doc_id)

            if save_samples:
                metadata: dict[str, Any] = {
                    field: row[field]
                    for field in metadata_fields
                    if row.get(field) is not None
                }
                length_bucket = self._length_bucket(row)
                if length_bucket:
                    metadata["length_bucket"] = length_bucket
                    metadata["text_length"] = len(row[self.task_spec.text_field])

                per_sample_list.append(
                    PerSampleResult(
                        id=doc_id,
                        y_true=sorted(true_labels),
                        y_pred=sorted(predicted_labels),
                        correct=is_exact,
                        metadata=metadata,
                    )
                )

        if not true_sets:
            raise ValueError("No valid predictions found matching ground truth IDs")

        labels = self._label_vocabulary(true_sets, pred_sets)

        return self._build_result(
            y_true=binarize_labels(true_sets, labels),
            y_pred=binarize_labels(pred_sets, labels),
            y_score=None,
            labels=labels,
            ground_truth=ground_truth,
            split=self.task_spec.default_split,
            misclassified_ids=misclassified_ids,
            per_sample_list=per_sample_list,
            save_samples=save_samples,
            model_key=model_key or "unknown",
            compute_ci=compute_ci,
        )

    def evaluate_embedder_with_classifier(
        self,
        embedder: TextEmbedder,
        split: str | None = None,
        train_split: str = "train",
        batch_size: int = 32,
        show_progress: bool = True,
        save_samples: bool = False,
        compute_ci: bool = False,
        classifier: str | None = None,
        classifier_params: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        """Evaluate an embedder by training a multi-label head on embeddings.

        This method:
        1. Encodes train split with embedder
        2. Trains a One-vs-Rest LogisticRegression over the frozen embeddings
        3. Encodes test split and predicts label sets
        4. Evaluates predictions

        The protocol deliberately matches
        ``ClassificationEvaluator.evaluate_embedder_with_classifier`` (same
        embedder, same linear head, same seed) so multi-label and single-label
        scores measure the same underlying representation quality.

        Args:
            embedder: TextEmbedder instance
            split: Evaluation split (default: task default, usually "test")
            train_split: Training split for the classifier
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bars
            save_samples: Whether to capture per-sample results
            compute_ci: Whether to compute bootstrap confidence intervals
            classifier: Which head to train (logistic_regression|random_forest)
            classifier_params: Optional override parameters for the classifier

        Returns:
            EvaluationResult with multi-label metrics
        """
        split = split or self.task_spec.default_split
        classifier_name = (classifier or "logistic_regression").lower()
        classifier_params = classifier_params or {}

        logger.info(f"Loading train data from split: {train_split}")
        train_df = self._load_ground_truth(train_split)

        logger.info(f"Loading test data from split: {split}")
        test_df = self._load_ground_truth(split)

        text_field = self.task_spec.text_field
        id_field = self.task_spec.id_field
        label_field = self.task_spec.label_field

        train_texts = train_df[text_field].to_list()
        train_label_sets = [
            self._normalize_label_list(v) for v in train_df[label_field].to_list()
        ]

        test_texts = test_df[text_field].to_list()
        test_ids = test_df[id_field].to_list()
        test_label_sets = [
            self._normalize_label_list(v) for v in test_df[label_field].to_list()
        ]

        logger.info(f"Train: {len(train_texts)}, Test: {len(test_texts)}")

        logger.info("Encoding training data...")
        train_embeddings = embedder.encode(
            train_texts,
            batch_size=batch_size,
            show_progress=show_progress,
        )

        logger.info("Encoding test data...")
        test_embeddings = embedder.encode(
            test_texts,
            batch_size=batch_size,
            show_progress=show_progress,
        )

        if len(train_embeddings) != len(train_texts):
            raise ValueError(
                f"Embedder returned {len(train_embeddings)} train embeddings for "
                f"{len(train_texts)} texts ({type(embedder).__name__})"
            )

        if len(test_embeddings) != len(test_texts):
            raise ValueError(
                f"Embedder returned {len(test_embeddings)} test embeddings for "
                f"{len(test_texts)} texts ({type(embedder).__name__})"
            )

        labels = self._label_vocabulary(train_label_sets, test_label_sets)
        y_train = binarize_labels(train_label_sets, labels)
        y_true = binarize_labels(test_label_sets, labels)

        clf, clf_label, trained_columns = self._train_classifier(
            classifier_name,
            train_embeddings,
            y_train,
            classifier_params,
        )

        logger.info("Predicting on test data...")
        y_pred = np.zeros_like(y_true)
        y_score = np.zeros(y_true.shape, dtype=float)
        if trained_columns:
            raw_pred = np.asarray(clf.predict(test_embeddings))
            raw_score = self._decision_scores(
                clf, test_embeddings, len(trained_columns)
            )
            y_pred[:, trained_columns] = raw_pred
            y_score[:, trained_columns] = raw_score

        if y_pred.shape[0] != len(test_ids):
            raise ValueError(
                f"Classifier produced {y_pred.shape[0]} predictions for "
                f"{len(test_ids)} test samples"
            )

        misclassified_ids: list[str] = []
        per_sample_list: list[PerSampleResult] = []

        available_cols = set(test_df.columns)
        metadata_fields = [f for f in STRATIFICATION_FIELDS if f in available_cols]

        for i, doc_id in enumerate(test_ids):
            predicted_labels = [labels[j] for j in np.flatnonzero(y_pred[i])]
            true_labels = test_label_sets[i]
            is_exact = set(true_labels) == set(predicted_labels)

            if not is_exact:
                misclassified_ids.append(doc_id)

            if save_samples:
                row = test_df.row(i, named=True)
                metadata: dict[str, Any] = {
                    field: row[field]
                    for field in metadata_fields
                    if row.get(field) is not None
                }
                length_bucket = self._length_bucket(row)
                if length_bucket:
                    metadata["length_bucket"] = length_bucket
                    metadata["text_length"] = len(row[text_field])

                per_sample_list.append(
                    PerSampleResult(
                        id=doc_id,
                        y_true=sorted(true_labels),
                        y_pred=sorted(predicted_labels),
                        correct=is_exact,
                        metadata=metadata,
                    )
                )

        return self._build_result(
            y_true=y_true,
            y_pred=y_pred,
            y_score=y_score,
            labels=labels,
            ground_truth=test_df,
            split=split,
            misclassified_ids=misclassified_ids,
            per_sample_list=per_sample_list,
            save_samples=save_samples,
            model_key=embedder.model_name or "unknown",
            compute_ci=compute_ci,
            model_name=embedder.model_name,
            embedding_dim=embedder.embedding_dim,
            classifier=clf_label,
            train_size=len(train_texts),
        )

    # ------------------------------------------------------------------
    # Classifier head
    # ------------------------------------------------------------------

    @staticmethod
    def _decision_scores(clf: Any, x: Any, num_columns: int) -> np.ndarray:
        """Extract continuous per-label scores for ranking metrics.

        Prefers ``predict_proba``; falls back to ``decision_function`` and
        finally to the hard predictions (which degrades the ranking metrics
        but never breaks the evaluation).
        """
        if hasattr(clf, "predict_proba"):
            scores = np.asarray(clf.predict_proba(x), dtype=float)
        elif hasattr(clf, "decision_function"):
            scores = np.asarray(clf.decision_function(x), dtype=float)
        else:  # pragma: no cover - all sklearn heads used here support one
            scores = np.asarray(clf.predict(x), dtype=float)

        # A single-column multi-label problem can come back 1-D.
        if scores.ndim == 1:
            scores = scores.reshape(-1, num_columns)
        return scores

    def _train_classifier(
        self,
        classifier_name: str,
        train_embeddings: Any,
        y_train: np.ndarray,
        classifier_params: dict[str, Any],
    ) -> tuple[Any, str, list[int]]:
        """Instantiate and fit the requested multi-label classifier.

        Labels with zero positive examples in the training split are dropped
        before fitting (a binary sub-problem with a single class cannot be fit
        and would otherwise raise). They are re-inserted as all-negative
        columns by the caller, which is the correct behaviour: a model trained
        without ever seeing a label cannot predict it, and its F1 of 0.0
        should count against macro-F1.

        Args:
            classifier_name: Head to train (logistic_regression|random_forest)
            train_embeddings: Frozen embeddings for the train split
            y_train: Binary indicator matrix for the train split
            classifier_params: Parameter overrides for the base estimator

        Returns:
            (fitted classifier, human-readable label, trained column indices)
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.multiclass import OneVsRestClassifier

        trained_columns = [
            int(i) for i in np.flatnonzero(np.asarray(y_train).sum(axis=0) > 0)
        ]
        dropped = y_train.shape[1] - len(trained_columns)
        if dropped:
            logger.warning(
                f"{dropped} label(s) have no positive training examples and "
                "will always be predicted negative."
            )
        if not trained_columns:
            raise ValueError("No label has a positive training example")

        y_fit = np.asarray(y_train)[:, trained_columns]

        name = classifier_name.lower()
        if name in ("logreg", "logistic", "logistic_regression"):
            params: dict[str, Any] = {
                "max_iter": 1000,
                "random_state": self.random_seed,
                "class_weight": "balanced",
            }
            params.update(classifier_params)
            logger.info("Training OneVsRest LogisticRegression classifier...")
            base = LogisticRegression(**params)
            clf_label = "OneVsRestClassifier(LogisticRegression)"
        elif name in ("rf", "random_forest", "random_forest_classifier"):
            params = {
                "n_estimators": 300,
                "max_depth": None,
                "n_jobs": -1,
                "random_state": self.random_seed,
                "class_weight": "balanced",
            }
            params.update(classifier_params)
            logger.info("Training OneVsRest RandomForestClassifier...")
            base = RandomForestClassifier(**params)
            clf_label = "OneVsRestClassifier(RandomForestClassifier)"
        else:
            raise ValueError(f"Unsupported classifier type: {classifier_name}")

        clf = OneVsRestClassifier(base, n_jobs=-1)
        clf.fit(train_embeddings, y_fit)

        return clf, clf_label, trained_columns
