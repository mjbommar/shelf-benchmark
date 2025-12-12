"""SHELF Evaluation Harness.

A clean, modular framework for evaluating models on SHELF benchmark tasks.

Primary interface:
    from shelf.evaluate import evaluate
    result = evaluate("lcc_retrieval", model=embedder)

Two paths for users:
    1. Quick: Model → Adapter → evaluate() → Results
    2. Flexible: Model → predictions.jsonl → evaluate() → Results

Example:
    from shelf.evaluate import evaluate
    from shelf.evaluate.adapters import SentenceTransformerEmbedder

    embedder = SentenceTransformerEmbedder.from_pretrained("all-MiniLM-L6-v2")
    result = evaluate("lcc_retrieval", model=embedder)
    print(f"NDCG@10: {result.metrics['ndcg@10']:.4f}")
"""

from shelf.evaluate.tasks import TaskSpec, TaskType
from shelf.evaluate.results import EvaluationContext, EvaluationResult
from shelf.evaluate.registry import TASK_REGISTRY, get_task, list_tasks
from shelf.evaluate.runner import evaluate, evaluate_all

__all__ = [
    # Core types
    "TaskSpec",
    "TaskType",
    "EvaluationContext",
    "EvaluationResult",
    # Registry
    "TASK_REGISTRY",
    "get_task",
    "list_tasks",
    # Runner
    "evaluate",
    "evaluate_all",
]
