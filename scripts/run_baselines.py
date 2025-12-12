#!/usr/bin/env python
"""Run baseline evaluations on SHELF benchmark.

This script runs all baseline models across all task types to establish
reproducible baseline results for the SHELF benchmark.

Baseline Models:
- TF (Term Frequency): Bag-of-words with SVD
- BM25: Okapi BM25 sparse retrieval
- TF-IDF: TF-IDF with SVD
- BERT: bert-base-uncased via sentence-transformers
- RoBERTa: roberta-base via sentence-transformers
- MiniLM: all-MiniLM-L6-v2 via sentence-transformers

Task Types:
- Retrieval: lcc_retrieval, form_retrieval, category_retrieval
- Classification: lcc_classification, lcgft_category_classification, register_classification
- Clustering: lcc_clustering, lcgft_clustering
- Pair Classification: same_lcc_pairs, same_form_pairs

Usage:
    # Run all baselines (full evaluation)
    python scripts/run_baselines.py

    # Run specific models
    python scripts/run_baselines.py --models tf bm25 tfidf

    # Run specific task types
    python scripts/run_baselines.py --task-types retrieval classification

    # Quick test with limited queries
    python scripts/run_baselines.py --max-queries 50

    # Save results
    python scripts/run_baselines.py --output results/baselines/

    # Skip slow neural models
    python scripts/run_baselines.py --skip-neural
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Model configurations
MODEL_CONFIGS = {
    "tf": {
        "type": "tf",
        "name": "TF",
        "description": "Term Frequency with SVD",
        "params": {"embedding_dim": 256},
    },
    "bm25": {
        "type": "bm25",
        "name": "BM25",
        "description": "Okapi BM25 sparse retrieval",
        "params": {"k1": 1.5, "b": 0.75},
    },
    "tfidf": {
        "type": "tfidf",
        "name": "TF-IDF",
        "description": "TF-IDF with SVD",
        "params": {"embedding_dim": 256},
    },
    "bert": {
        "type": "sentence_transformer",
        "name": "BERT",
        "description": "bert-base-uncased via sentence-transformers",
        "model_name": "bert-base-uncased",
    },
    "roberta": {
        "type": "sentence_transformer",
        "name": "RoBERTa",
        "description": "roberta-base via sentence-transformers",
        "model_name": "roberta-base",
    },
    "minilm": {
        "type": "sentence_transformer",
        "name": "MiniLM",
        "description": "all-MiniLM-L6-v2 via sentence-transformers",
        "model_name": "all-MiniLM-L6-v2",
    },
}

# Task configurations by type
TASK_CONFIGS = {
    "retrieval": [
        "lcc_retrieval",
        "form_retrieval",
        "category_retrieval",
    ],
    "classification": [
        "lcc_classification",
        "lcgft_category_classification",
        "register_classification",
    ],
    "clustering": [
        "lcc_clustering",
        "lcgft_clustering",
    ],
    "pair_classification": [
        "same_lcc_pairs",
        "same_form_pairs",
    ],
}


def get_version_info() -> dict[str, str]:
    """Get version information for reproducibility."""
    import sklearn
    import numpy as np
    import scipy

    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }

    try:
        import sentence_transformers

        versions["sentence_transformers"] = sentence_transformers.__version__
    except ImportError:
        pass

    try:
        import torch

        versions["torch"] = torch.__version__
    except ImportError:
        pass

    return versions


def create_model(model_key: str):  # noqa: C901
    """Create a model instance from configuration.

    Args:
        model_key: Key from MODEL_CONFIGS

    Returns:
        Model instance (embedder or retriever)
    """
    config = MODEL_CONFIGS[model_key]
    model_type = str(config["type"])

    if model_type == "tf":
        from shelf.evaluate.adapters import TfEmbedder

        params = config.get("params")
        if params is not None and isinstance(params, dict):
            embedding_dim = int(params.get("embedding_dim", 256))
            return TfEmbedder(embedding_dim=embedding_dim)
        return TfEmbedder()

    elif model_type == "tfidf":
        from shelf.evaluate.adapters import TfidfEmbedder

        params = config.get("params")
        if params is not None and isinstance(params, dict):
            embedding_dim = int(params.get("embedding_dim", 256))
            return TfidfEmbedder(embedding_dim=embedding_dim)
        return TfidfEmbedder()

    elif model_type == "bm25":
        from shelf.evaluate.adapters import BM25Retriever

        params = config.get("params")
        if params is not None and isinstance(params, dict):
            k1 = float(params.get("k1", 1.5))
            b = float(params.get("b", 0.75))
            return BM25Retriever(k1=k1, b=b)
        return BM25Retriever()

    elif model_type == "sentence_transformer":
        from shelf.evaluate.adapters import SentenceTransformerEmbedder

        model_name = config.get("model_name")
        if not isinstance(model_name, str):
            raise ValueError(f"model_name must be a string, got {type(model_name)}")
        return SentenceTransformerEmbedder.from_pretrained(model_name)

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def evaluate_retrieval(
    model,
    model_key: str,
    tasks: list[str],
    max_queries: int | None = None,
    output_dir: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Run retrieval evaluation for a model.

    Args:
        model: Model instance (embedder or retriever)
        model_key: Model configuration key
        tasks: List of retrieval task names
        max_queries: Maximum queries to evaluate
        output_dir: Directory to save results

    Returns:
        Dict mapping task_name to result dict
    """
    from shelf.evaluate import evaluate
    from shelf.evaluate.registry import get_task

    results = {}
    model_config = MODEL_CONFIGS[model_key]
    model_name = model_config["name"]

    for task_name in tasks:
        task_spec = get_task(task_name)
        logger.info(f"Evaluating {model_name} on {task_name}...")

        try:
            result = evaluate(
                task=task_name,
                model=model,
                max_queries=max_queries,
                show_progress=True,
            )

            task_result = {
                "model": model_name,
                "model_key": model_key,
                "task": task_name,
                "task_type": task_spec.task_type.value,
                "primary_metric": result.primary_metric,
                "primary_score": result.primary_score,
                "metrics": result.metrics,
            }

            results[task_name] = task_result

            # Print summary
            logger.info(
                f"  {task_name}: {result.primary_metric}={result.primary_score:.4f}"
            )

            # Save individual result
            if output_dir:
                output_path = output_dir / f"{model_key}_{task_name}.json"
                result.to_json(output_path)
                logger.info(f"  Saved: {output_path}")

            # Reset model for next task (sparse models need refitting)
            if hasattr(model, "reset"):
                model.reset()

        except Exception as e:
            logger.error(f"  Error evaluating {task_name}: {e}")
            results[task_name] = {"error": str(e)}

    return results


def evaluate_classification(
    model,
    model_key: str,
    tasks: list[str],
    output_dir: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Run classification evaluation for a model.

    Args:
        model: Model instance (embedder)
        model_key: Model configuration key
        tasks: List of classification task names
        output_dir: Directory to save results

    Returns:
        Dict mapping task_name to result dict
    """
    from shelf.evaluate import evaluate
    from shelf.evaluate.registry import get_task

    results = {}
    model_config = MODEL_CONFIGS[model_key]
    model_name = model_config["name"]

    for task_name in tasks:
        task_spec = get_task(task_name)
        logger.info(f"Evaluating {model_name} on {task_name}...")

        try:
            result = evaluate(
                task=task_name,
                model=model,
                show_progress=True,
            )

            task_result = {
                "model": model_name,
                "model_key": model_key,
                "task": task_name,
                "task_type": task_spec.task_type.value,
                "primary_metric": result.primary_metric,
                "primary_score": result.primary_score,
                "metrics": result.metrics,
            }

            results[task_name] = task_result

            # Print summary
            logger.info(
                f"  {task_name}: {result.primary_metric}={result.primary_score:.4f}"
            )

            # Save individual result
            if output_dir:
                output_path = output_dir / f"{model_key}_{task_name}.json"
                result.to_json(output_path)
                logger.info(f"  Saved: {output_path}")

            # Reset model for next task
            if hasattr(model, "reset"):
                model.reset()

        except Exception as e:
            logger.error(f"  Error evaluating {task_name}: {e}")
            results[task_name] = {"error": str(e)}

    return results


def evaluate_clustering(
    model,
    model_key: str,
    tasks: list[str],
    output_dir: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Run clustering evaluation for a model.

    Args:
        model: Model instance (embedder)
        model_key: Model configuration key
        tasks: List of clustering task names
        output_dir: Directory to save results

    Returns:
        Dict mapping task_name to result dict
    """
    from shelf.evaluate import evaluate
    from shelf.evaluate.registry import get_task

    results = {}
    model_config = MODEL_CONFIGS[model_key]
    model_name = model_config["name"]

    for task_name in tasks:
        task_spec = get_task(task_name)
        logger.info(f"Evaluating {model_name} on {task_name}...")

        try:
            result = evaluate(
                task=task_name,
                model=model,
                show_progress=True,
            )

            task_result = {
                "model": model_name,
                "model_key": model_key,
                "task": task_name,
                "task_type": task_spec.task_type.value,
                "primary_metric": result.primary_metric,
                "primary_score": result.primary_score,
                "metrics": result.metrics,
            }

            results[task_name] = task_result

            # Print summary
            logger.info(
                f"  {task_name}: {result.primary_metric}={result.primary_score:.4f}"
            )

            # Save individual result
            if output_dir:
                output_path = output_dir / f"{model_key}_{task_name}.json"
                result.to_json(output_path)
                logger.info(f"  Saved: {output_path}")

            # Reset model for next task
            if hasattr(model, "reset"):
                model.reset()

        except Exception as e:
            logger.error(f"  Error evaluating {task_name}: {e}")
            results[task_name] = {"error": str(e)}

    return results


def evaluate_pair_classification(
    model,
    model_key: str,
    tasks: list[str],
    output_dir: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Run pair classification evaluation for a model.

    Args:
        model: Model instance (embedder or retriever)
        model_key: Model configuration key
        tasks: List of pair classification task names
        output_dir: Directory to save results

    Returns:
        Dict mapping task_name to result dict
    """
    from shelf.evaluate.evaluators.pair import PairClassificationEvaluator
    from shelf.evaluate.registry import get_task

    results = {}
    model_config = MODEL_CONFIGS[model_key]
    model_name = model_config["name"]
    model_type = model_config["type"]

    for task_name in tasks:
        task_spec = get_task(task_name)
        logger.info(f"Evaluating {model_name} on {task_name}...")

        try:
            evaluator = PairClassificationEvaluator(task_spec)

            # Choose evaluation method based on model type
            if model_type == "bm25":
                result = evaluator.evaluate_bm25(model, show_progress=True)
            elif model_type in ("tf", "tfidf"):
                result = evaluator.evaluate_tfidf(model, show_progress=True)
            else:
                result = evaluator.evaluate_embedder(model, show_progress=True)

            task_result = {
                "model": model_name,
                "model_key": model_key,
                "task": task_name,
                "task_type": task_spec.task_type.value,
                "primary_metric": result.primary_metric,
                "primary_score": result.primary_score,
                "metrics": result.metrics,
            }

            results[task_name] = task_result

            # Print summary
            logger.info(
                f"  {task_name}: {result.primary_metric}={result.primary_score:.4f}"
            )

            # Save individual result
            if output_dir:
                output_path = output_dir / f"{model_key}_{task_name}.json"
                result.to_json(output_path)
                logger.info(f"  Saved: {output_path}")

            # Reset model for next task
            if hasattr(model, "reset"):
                model.reset()

        except Exception as e:
            logger.error(f"  Error evaluating {task_name}: {e}")
            import traceback

            traceback.print_exc()
            results[task_name] = {"error": str(e)}

    return results


def run_baselines(
    models: list[str],
    task_types: list[str],
    max_queries: int | None = None,
    output_dir: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Run baseline evaluations.

    Args:
        models: List of model keys to evaluate
        task_types: List of task types to evaluate
        max_queries: Maximum queries for retrieval tasks
        output_dir: Directory to save results

    Returns:
        Dict of all results
    """
    all_results: dict[str, dict[str, Any]] = {}

    for model_key in models:
        if model_key not in MODEL_CONFIGS:
            logger.warning(f"Unknown model: {model_key}, skipping")
            continue

        model_config = MODEL_CONFIGS[model_key]
        model_name = model_config["name"]
        model_type = model_config["type"]

        logger.info("=" * 60)
        logger.info(f"Model: {model_name} ({model_config['description']})")
        logger.info("=" * 60)

        # Create model once per model (reused across task types)
        model = create_model(model_key)

        for task_type in task_types:
            if task_type not in TASK_CONFIGS:
                logger.warning(f"Unknown task type: {task_type}, skipping")
                continue

            tasks = TASK_CONFIGS[task_type]
            logger.info(f"\n--- {task_type.upper()} TASKS ---")

            # Check model compatibility with task type
            if model_type == "bm25":
                # BM25 only works for retrieval and pair classification
                if task_type not in ("retrieval", "pair_classification"):
                    logger.info(
                        f"  Skipping {task_type} (BM25 only supports retrieval/pair)"
                    )
                    continue

            if task_type == "retrieval":
                results = evaluate_retrieval(
                    model, model_key, tasks, max_queries, output_dir
                )
            elif task_type == "classification":
                results = evaluate_classification(model, model_key, tasks, output_dir)
            elif task_type == "clustering":
                results = evaluate_clustering(model, model_key, tasks, output_dir)
            elif task_type == "pair_classification":
                results = evaluate_pair_classification(
                    model, model_key, tasks, output_dir
                )
            else:
                continue

            # Store results with model prefix
            for task_name, task_result in results.items():
                result_key = f"{model_key}_{task_name}"
                all_results[result_key] = task_result

    return all_results


def print_summary_table(results: dict[str, dict[str, Any]]) -> None:
    """Print summary table of results.

    Args:
        results: Dict of all results
    """
    print("\n" + "=" * 80)
    print("BASELINE SUMMARY")
    print("=" * 80)

    # Group by task type
    by_task_type: dict[str, list[tuple[str, dict]]] = {}
    for key, result in results.items():
        if "error" in result:
            continue
        task_type = result.get("task_type", "unknown")
        if task_type not in by_task_type:
            by_task_type[task_type] = []
        by_task_type[task_type].append((key, result))

    for task_type, task_results in sorted(by_task_type.items()):
        print(f"\n--- {task_type.upper()} ---")
        print(f"{'Model':<15} {'Task':<35} {'Metric':<15} {'Score':<10}")
        print("-" * 75)

        for key, result in sorted(
            task_results, key=lambda x: (x[1]["task"], x[1]["model"])
        ):
            model = result["model"]
            task = result["task"]
            metric = result["primary_metric"]
            score = result["primary_score"]
            print(f"{model:<15} {task:<35} {metric:<15} {score:<10.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Run SHELF baseline evaluations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_CONFIGS.keys()),
        choices=list(MODEL_CONFIGS.keys()),
        help="Models to evaluate (default: all)",
    )
    parser.add_argument(
        "--task-types",
        nargs="+",
        default=list(TASK_CONFIGS.keys()),
        choices=list(TASK_CONFIGS.keys()),
        help="Task types to evaluate (default: all)",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Maximum queries for retrieval tasks (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for results (default: print only)",
    )
    parser.add_argument(
        "--skip-neural",
        action="store_true",
        help="Skip neural models (bert, roberta, minilm)",
    )
    parser.add_argument(
        "--sparse-only",
        action="store_true",
        help="Run only sparse models (tf, bm25, tfidf)",
    )
    args = parser.parse_args()

    # Filter models
    models = args.models
    if args.skip_neural or args.sparse_only:
        neural_models = {"bert", "roberta", "minilm"}
        models = [m for m in models if m not in neural_models]

    if not models:
        logger.error("No models to evaluate")
        sys.exit(1)

    # Setup output directory
    output_dir = None
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Print configuration
    logger.info("SHELF Baseline Evaluation")
    logger.info(f"Models: {models}")
    logger.info(f"Task types: {args.task_types}")
    logger.info(f"Max queries: {args.max_queries or 'all'}")
    logger.info(f"Output directory: {output_dir or 'none (print only)'}")

    # Get version info
    versions = get_version_info()
    logger.info(f"Python: {versions['python']}")
    logger.info(f"sklearn: {versions['sklearn']}")

    # Run baselines
    results = run_baselines(
        models=models,
        task_types=args.task_types,
        max_queries=args.max_queries,
        output_dir=output_dir,
    )

    # Print summary
    print_summary_table(results)

    # Save summary
    if output_dir:
        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "versions": versions,
            "config": {
                "models": models,
                "task_types": args.task_types,
                "max_queries": args.max_queries,
            },
            "results": results,
        }

        summary_path = output_dir / "baseline_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
