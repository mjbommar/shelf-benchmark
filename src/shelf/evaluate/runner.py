"""Main evaluation runner for SHELF benchmark.

This module provides the high-level evaluate() function that
serves as the primary entry point for evaluation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from shelf.evaluate.evaluators.classification import ClassificationEvaluator
from shelf.evaluate.evaluators.clustering import ClusteringEvaluator
from shelf.evaluate.evaluators.pair import PairClassificationEvaluator
from shelf.evaluate.evaluators.retrieval import RetrievalEvaluator
from shelf.evaluate.registry import get_task
from shelf.evaluate.results import EvaluationResult, compute_file_checksum
from shelf.evaluate.tasks import TaskType

if TYPE_CHECKING:
    from shelf.evaluate.adapters.protocols import TextEmbedder
    from shelf.evaluate.tasks import TaskSpec

logger = logging.getLogger(__name__)


def evaluate(
    task: str,
    predictions: Path | str | list[dict[str, Any]] | None = None,
    model: "TextEmbedder | None" = None,
    split: str | None = None,
    output_path: Path | str | None = None,
    max_queries: int | None = None,
    batch_size: int = 32,
    show_progress: bool = True,
    **kwargs: Any,
) -> EvaluationResult:
    """Main evaluation entry point.

    This is the primary function for evaluating models on SHELF tasks.
    It supports two modes:

    1. From predictions file:
        result = evaluate("lcc_classification", predictions="preds.jsonl")

    2. From model directly:
        result = evaluate("lcc_retrieval", model=my_embedder)

    Args:
        task: Task name from registry (e.g., "lcc_retrieval", "lcc_classification")
        predictions: Path to predictions file or list of prediction dicts
        model: Model implementing appropriate protocol (alternative to predictions)
        split: Dataset split to evaluate on (default: task default)
        output_path: Path to save results JSON (optional)
        max_queries: Maximum queries to evaluate (for testing)
        batch_size: Batch size for model inference
        show_progress: Whether to show progress bars
        **kwargs: Additional arguments passed to evaluator

    Returns:
        EvaluationResult with all metrics and context

    Raises:
        ValueError: If neither predictions nor model is provided
        ValueError: If task is not found

    Examples:
        # Evaluate predictions file
        result = evaluate("lcc_classification", predictions="preds.jsonl")

        # Evaluate embedder on retrieval
        from shelf.evaluate.adapters import SentenceTransformerEmbedder
        embedder = SentenceTransformerEmbedder.from_pretrained("all-MiniLM-L6-v2")
        result = evaluate("lcc_retrieval", model=embedder)

        # Save results
        result = evaluate("lcc_retrieval", model=embedder, output_path="results.json")
    """
    # Get task spec
    task_spec = get_task(task)
    split = split or task_spec.default_split
    prediction_file_checksum: str | None = None

    logger.info(f"Evaluating task: {task} (type: {task_spec.task_type.value})")

    # Create appropriate evaluator
    evaluator = _create_evaluator(task_spec, **kwargs)

    # Run evaluation
    if predictions is not None:
        # Load predictions if path provided
        if isinstance(predictions, (str, Path)):
            from shelf.evaluate.schemas import load_predictions_jsonl

            predictions_path = Path(predictions)
            prediction_file_checksum = compute_file_checksum(predictions_path)
            predictions = load_predictions_jsonl(predictions_path)

        # Evaluate from predictions
        ground_truth = evaluator._load_ground_truth(split)
        result = evaluator.evaluate(predictions, ground_truth)

    elif model is not None:
        # Evaluate model directly
        if task_spec.task_type == TaskType.RETRIEVAL:
            if not isinstance(evaluator, RetrievalEvaluator):
                raise ValueError(f"Expected RetrievalEvaluator for {task}")

            # Check if model is a retriever (has retrieve method) vs embedder
            if hasattr(model, "retrieve") and callable(model.retrieve):
                # Use retriever interface (e.g., BM25Retriever)
                result = evaluator.evaluate_retriever(
                    retriever=model,
                    split=split,
                    max_queries=max_queries,
                    show_progress=show_progress,
                )
            else:
                # Use embedder interface (e.g., SentenceTransformerEmbedder, TfidfEmbedder)
                result = evaluator.evaluate_embedder(
                    embedder=model,
                    split=split,
                    max_queries=max_queries,
                    batch_size=batch_size,
                    show_progress=show_progress,
                )
        elif task_spec.task_type == TaskType.CLASSIFICATION:
            if not isinstance(evaluator, ClassificationEvaluator):
                raise ValueError(f"Expected ClassificationEvaluator for {task}")

            # Check if model is a classifier (has predict method) vs embedder
            if hasattr(model, "predict") and callable(model.predict):
                # Use classifier interface
                result = evaluator.evaluate_classifier(
                    classifier=model,
                    split=split,
                    batch_size=batch_size,
                    show_progress=show_progress,
                )
            else:
                # Use embedder interface (train LogisticRegression on embeddings)
                result = evaluator.evaluate_embedder_with_classifier(
                    embedder=model,
                    split=split,
                    batch_size=batch_size,
                    show_progress=show_progress,
                )
        elif task_spec.task_type == TaskType.CLUSTERING:
            if not isinstance(evaluator, ClusteringEvaluator):
                raise ValueError(f"Expected ClusteringEvaluator for {task}")

            # Clustering only supports embedder interface
            result = evaluator.evaluate_embedder(
                embedder=model,
                split=split,
                batch_size=batch_size,
                show_progress=show_progress,
            )
        elif task_spec.task_type == TaskType.PAIR_CLASSIFICATION:
            if not isinstance(evaluator, PairClassificationEvaluator):
                raise ValueError(f"Expected PairClassificationEvaluator for {task}")

            # Pair classification only supports embedder interface
            result = evaluator.evaluate_embedder(
                embedder=model,
                split=split,
                batch_size=batch_size,
                show_progress=show_progress,
            )
        else:
            raise NotImplementedError(
                f"Direct model evaluation not yet implemented for {task_spec.task_type}"
            )
    else:
        raise ValueError("Must provide either predictions or model")

    if prediction_file_checksum and result.context is not None:
        result.context.prediction_file_checksum = prediction_file_checksum

    # Save results if requested
    if output_path is not None:
        result.to_json(output_path)
        logger.info(f"Results saved to: {output_path}")

    return result


def evaluate_all(
    model: "TextEmbedder",
    tasks: list[str] | None = None,
    task_types: list[TaskType] | None = None,
    split: str | None = None,
    output_dir: Path | str | None = None,
    **kwargs: Any,
) -> dict[str, EvaluationResult]:
    """Evaluate a model on multiple tasks.

    Args:
        model: Model implementing TextEmbedder protocol
        tasks: List of task names (default: all compatible tasks)
        task_types: List of task types to include (default: [RETRIEVAL, CLUSTERING])
        split: Dataset split to evaluate on
        output_dir: Directory to save results (one file per task)
        **kwargs: Additional arguments passed to evaluate()

    Returns:
        Dictionary mapping task_name to EvaluationResult

    Examples:
        from shelf.evaluate import evaluate_all
        from shelf.evaluate.adapters import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder.from_pretrained("all-MiniLM-L6-v2")
        results = evaluate_all(embedder)

        for task, result in results.items():
            print(f"{task}: {result.primary_metric}={result.primary_score:.4f}")
    """
    from shelf.evaluate.registry import list_tasks

    # Default to retrieval tasks (most relevant for embedders)
    if tasks is None:
        if task_types is None:
            task_types = [TaskType.RETRIEVAL]
        tasks = []
        for tt in task_types:
            tasks.extend(list_tasks(tt))

    results: dict[str, EvaluationResult] = {}

    for task_name in tasks:
        logger.info(f"Evaluating: {task_name}")
        try:
            result = evaluate(
                task=task_name,
                model=model,
                split=split,
                **kwargs,
            )
            results[task_name] = result

            # Save individual result if output_dir specified
            if output_dir is not None:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                result.to_json(output_dir / f"{task_name}.json")

        except NotImplementedError as e:
            logger.warning(f"Skipping {task_name}: {e}")
        except Exception as e:
            logger.error(f"Error evaluating {task_name}: {e}")
            raise

    return results


def _create_evaluator(task_spec: "TaskSpec", **kwargs: Any):
    """Create appropriate evaluator for task type.

    Args:
        task_spec: Task specification
        **kwargs: Additional arguments for evaluator

    Returns:
        TaskEvaluator instance
    """
    if task_spec.task_type == TaskType.RETRIEVAL:
        return RetrievalEvaluator(task_spec, **kwargs)
    elif task_spec.task_type == TaskType.CLASSIFICATION:
        return ClassificationEvaluator(task_spec, **kwargs)
    elif task_spec.task_type == TaskType.CLUSTERING:
        return ClusteringEvaluator(task_spec, **kwargs)
    elif task_spec.task_type == TaskType.MULTILABEL:
        # TODO: Implement MultiLabelEvaluator
        raise NotImplementedError("MultiLabelEvaluator not yet implemented")
    elif task_spec.task_type == TaskType.PAIR_CLASSIFICATION:
        return PairClassificationEvaluator(task_spec, **kwargs)
    else:
        raise ValueError(f"Unknown task type: {task_spec.task_type}")
