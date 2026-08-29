#!/usr/bin/env python
"""Run all SHELF baselines with full reproducibility.

This script reads the configuration from config.yaml and runs all model-task
combinations with proper versioning, checksums, and provenance tracking.

Usage:
    # Run all baselines (full evaluation)
    python scripts/baselines/run_all.py

    # Run specific models
    python scripts/baselines/run_all.py --models minilm bge_base

    # Run specific task types
    python scripts/baselines/run_all.py --task-types retrieval classification

    # Quick test with limited samples
    python scripts/baselines/run_all.py --max-queries 100 --max-samples 500

    # Skip models already evaluated
    python scripts/baselines/run_all.py --skip-existing

    # Dry run (show what would be run)
    python scripts/baselines/run_all.py --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from shelf.evaluate.efficiency import (
    compute_aggregate_efficiency,
    compute_efficiency_metrics,
    get_size_category,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def collect_all_evaluation_texts(
    tasks_config: dict[str, list[str]],
    dataset_repo: str = "mjbommar/SHELF",
) -> list[str]:
    """Collect all unique texts needed for evaluation tasks.

    Collects texts in both formats used by evaluators:
    1. Body-only (main tasks use 'text' field)
    2. Title + body (pair tasks concatenate title and body)

    Args:
        tasks_config: Task configuration dict with task types and names
        dataset_repo: HuggingFace dataset repository ID

    Returns:
        List of unique text strings
    """
    from datasets import load_dataset

    texts = set()

    # Main dataset texts (body only - 'text' field).
    #
    # Honour SHELF_DATA_DIR the way the evaluators do. Without this the cache
    # is built from the hub repo no matter which corpus is being scored, so a
    # 3,016-document Gutenberg run embedded 50,968 texts -- slow enough to
    # dominate the run, and heavy enough to contribute to an OOM.
    from shelf.evaluate.evaluators.base import data_root

    local = data_root()
    local_splits = [
        s for s in ("train", "validation", "test") if (local / f"{s}.parquet").exists()
    ]
    if local_splits:
        import polars as pl

        logger.info(f"Loading main dataset texts from {local}...")
        for split in local_splits:
            df = pl.read_parquet(local / f"{split}.parquet")
            col = "text" if "text" in df.columns else "body"
            split_texts = [t for t in df[col].to_list() if t]
            texts.update(split_texts)
            logger.info(f"  {split}: {len(split_texts)} texts (body only)")
    else:
        logger.info("Loading main dataset texts...")
        ds = load_dataset(dataset_repo)
        for split in ["train", "validation", "test"]:
            split_texts = ds[split]["text"]
            texts.update(split_texts)
            logger.info(f"  {split}: {len(split_texts)} texts (body only)")

    # Pair dataset texts (title + body format)
    pair_tasks = tasks_config.get("pair_classification", [])
    if local_splits and pair_tasks:
        logger.info("  (skipping hub pair texts: scoring a local corpus)")
        pair_tasks = []
    for task_name in pair_tasks:
        logger.info(f"Loading pair texts from {task_name}...")
        try:
            pair_ds = load_dataset(dataset_repo, task_name, split="test")

            # Pair datasets have doc_a_title, doc_a_body, doc_b_title, doc_b_body
            for row in pair_ds:
                text_a = f"{row['doc_a_title']}\n\n{row['doc_a_body']}"
                text_b = f"{row['doc_b_title']}\n\n{row['doc_b_body']}"
                texts.add(text_a)
                texts.add(text_b)

        except Exception as e:
            logger.warning(f"Could not load {task_name}: {e}")

    logger.info(f"Total unique texts collected: {len(texts)}")
    return list(texts)


# Path to this script's directory
SCRIPT_DIR = Path(__file__).parent
CONFIG_PATH = SCRIPT_DIR / "config.yaml"


def load_config(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_git_info() -> dict[str, str | bool]:
    """Get current git commit and dirty status."""
    info: dict[str, str | bool] = {
        "commit": "unknown",
        "branch": "unknown",
        "dirty": False,
    }
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            info["commit"] = result.stdout.strip()

        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            info["branch"] = result.stdout.strip()

        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            info["dirty"] = bool(result.stdout.strip())

    except Exception:
        pass

    return info


def get_version_info() -> dict[str, str]:
    """Get version information for reproducibility."""
    import scipy
    import sklearn

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
        versions["sentence_transformers"] = "not installed"

    try:
        import torch

        versions["torch"] = torch.__version__
        versions["cuda_available"] = str(torch.cuda.is_available())
        if torch.cuda.is_available():
            versions["cuda_version"] = torch.version.cuda or "unknown"
    except ImportError:
        versions["torch"] = "not installed"

    try:
        from importlib.metadata import version

        versions["shelf"] = version("shelf")
    except Exception:
        versions["shelf"] = "unknown"

    return versions


def compute_dataset_checksum(dataset_version: str) -> str:
    """Compute checksum of the dataset metadata."""
    metadata_path = Path("data/hf_dataset/metadata.json")
    if metadata_path.exists():
        with open(metadata_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    return f"v{dataset_version}"


def create_model(model_config: dict[str, Any]):
    """Create a model instance from configuration.

    Args:
        model_config: Model configuration dict

    Returns:
        Model instance (embedder or retriever)
    """
    model_type = str(model_config["type"])

    if model_type == "tf":
        from shelf.evaluate.adapters import TfEmbedder

        params = model_config.get("params", {})
        embedding_dim = int(params.get("embedding_dim", 256))
        return TfEmbedder(embedding_dim=embedding_dim)

    elif model_type == "tfidf":
        from shelf.evaluate.adapters import TfidfEmbedder

        params = model_config.get("params", {})
        embedding_dim = int(params.get("embedding_dim", 256))
        return TfidfEmbedder(embedding_dim=embedding_dim)

    elif model_type == "bm25":
        from shelf.evaluate.adapters import BM25Retriever

        params = model_config.get("params", {})
        k1 = float(params.get("k1", 1.5))
        b = float(params.get("b", 0.75))
        return BM25Retriever(k1=k1, b=b)

    elif model_type == "sentence_transformer":
        from shelf.evaluate.adapters import SentenceTransformerEmbedder

        model_name = model_config.get("model_name")
        if not model_name:
            raise ValueError("sentence_transformer requires model_name")
        return SentenceTransformerEmbedder.from_pretrained(model_name)

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def evaluate_task(
    model,
    model_key: str,
    model_config: dict[str, Any],
    task_name: str,
    task_type: str,
    max_queries: int | None = None,
    batch_size: int = 32,
    show_progress: bool = True,
) -> dict[str, Any] | None:
    """Evaluate a single model on a single task.

    Returns:
        Result dict or None if evaluation fails
    """
    from shelf.evaluate import evaluate
    from shelf.evaluate.evaluators.pair import PairClassificationEvaluator
    from shelf.evaluate.registry import get_task

    task_spec = get_task(task_name)
    model_type = model_config["type"]
    model_name = model_config["name"]

    logger.info(f"  Evaluating {model_name} on {task_name}...")

    try:
        # Handle pair classification separately for TF/TF-IDF/BM25
        if task_type == "pair_classification" and model_type in ("tf", "tfidf", "bm25"):
            evaluator = PairClassificationEvaluator(task_spec)
            if model_type == "bm25":
                result = evaluator.evaluate_bm25(model, show_progress=show_progress)
            else:
                result = evaluator.evaluate_tfidf(model, show_progress=show_progress)
        else:
            # Standard evaluation via runner
            result = evaluate(
                task=task_name,
                model=model,
                max_queries=max_queries,
                batch_size=batch_size,
                show_progress=show_progress,
            )

        return {
            "model": model_name,
            "model_key": model_key,
            "model_type": model_type,
            "task": task_name,
            "task_type": task_type,
            "primary_metric": result.primary_metric,
            "primary_score": result.primary_score,
            "metrics": result.metrics,
            # The evaluator computes and serialises per-class metrics; this dict
            # used to drop them, which was misdiagnosed as an evaluator gap.
            "per_class_metrics": getattr(result, "per_class_metrics", None),
            "num_samples": result.num_samples,
            "context": result.context.to_dict() if result.context else None,
        }

    except Exception as e:
        logger.error(f"    Error: {e}")
        import traceback

        traceback.print_exc()
        return {
            "model": model_name,
            "model_key": model_key,
            "task": task_name,
            "task_type": task_type,
            "error": str(e),
        }


def compute_shelf_score(
    results: dict[str, dict[str, Any]],
    weights: dict[str, float],
    metrics: dict[str, str],
    exclude_tasks: set[str] | None = None,
) -> dict[str, float]:
    """Compute aggregate SHELF Score for each model.

    Args:
        results: All evaluation results
        weights: Task type weights
        metrics: Primary metric for each task type

    Returns:
        Dict mapping model_key to SHELF Score
    """
    # Group results by model and task type
    model_scores: dict[str, dict[str, list[float]]] = {}

    for key, result in results.items():
        if "error" in result:
            continue

        if exclude_tasks and result.get("task") in exclude_tasks:
            # Reported on its own, deliberately not folded into the aggregate.
            continue

        model_key = result["model_key"]
        task_type = result["task_type"]
        score = result["primary_score"]

        if model_key not in model_scores:
            model_scores[model_key] = {}
        if task_type not in model_scores[model_key]:
            model_scores[model_key][task_type] = []

        model_scores[model_key][task_type].append(score)

    # Compute weighted average for each model
    shelf_scores: dict[str, float] = {}
    for model_key, type_scores in model_scores.items():
        weighted_sum = 0.0
        total_weight = 0.0

        for task_type, scores in type_scores.items():
            if task_type in weights and scores:
                avg_score = sum(scores) / len(scores)
                weight = weights[task_type]
                weighted_sum += avg_score * weight
                total_weight += weight

        if total_weight > 0:
            shelf_scores[model_key] = weighted_sum / total_weight
        else:
            # No weighted task type ran for this model, so there is no aggregate
            # to report. Emitting 0.0 would read as "scored zero" rather than
            # "not scored", which is how a model that skipped the hardest
            # category ends up looking comparable to one that did not.
            logger.info(
                "No weighted task types ran for %s; omitting it from SHELF scores "
                "rather than reporting 0.0",
                model_key,
            )

    return shelf_scores


def get_efficiency_dict(model_config: dict[str, Any]) -> dict[str, Any]:
    """Get efficiency metrics dict for a model from its config.

    For dense models, computes full efficiency metrics.
    For sparse models, returns placeholder dict with nulls.

    Args:
        model_config: Model configuration from YAML

    Returns:
        Dict with efficiency metrics for inclusion in result JSON
    """
    num_params = model_config.get("num_params")

    if num_params is None:
        # Sparse model - no parameter-based efficiency metrics
        return {
            "num_params": None,
            "embedding_dim": model_config.get("params", {}).get("embedding_dim", 256),
            "size_category": "sparse",
            "flops_per_token": None,
            "relative_compute": None,
            "shelf_eff": None,
            "shelf_compute": None,
            "pareto_optimal": None,
            "size_rank": None,
        }

    # Dense model - compute efficiency metrics (without score-dependent metrics)
    embedding_dim = model_config.get("embedding_dim", 0)
    size_category = model_config.get("size_category") or get_size_category(num_params)

    metrics = compute_efficiency_metrics(
        num_params=num_params,
        embedding_dim=embedding_dim,
        size_category=size_category,
        shelf_score=None,  # Computed later after all tasks complete
    )

    return metrics.to_dict()


def print_summary_table(
    results: dict[str, dict[str, Any]],
    shelf_scores: dict[str, float],
    models_config: dict[str, dict[str, Any]],
) -> None:
    """Print summary table of results."""
    print("\n" + "=" * 90)
    print("SHELF BASELINE RESULTS SUMMARY")
    print("=" * 90)

    # Group by task type
    by_task_type: dict[str, list[tuple[str, dict]]] = {}
    for key, result in results.items():
        if "error" in result:
            continue
        task_type = result.get("task_type", "unknown")
        if task_type not in by_task_type:
            by_task_type[task_type] = []
        by_task_type[task_type].append((key, result))

    # Preferred display order, then anything else that actually ran. Appending
    # the remainder rather than hard-coding the list means a newly added task
    # type shows up instead of silently vanishing from every summary.
    preferred = [
        "retrieval",
        "classification",
        "multilabel",
        "clustering",
        "pair_classification",
    ]
    ordered = preferred + sorted(set(by_task_type) - set(preferred))

    for task_type in ordered:
        if task_type not in by_task_type:
            continue

        task_results = by_task_type[task_type]
        print(f"\n--- {task_type.upper().replace('_', ' ')} ---")
        print(f"{'Model':<20} {'Task':<30} {'Metric':<12} {'Score':<10}")
        print("-" * 75)

        for key, result in sorted(
            task_results, key=lambda x: (x[1]["task"], -x[1]["primary_score"])
        ):
            model = result["model"]
            task = result["task"]
            metric = result["primary_metric"]
            score = result["primary_score"]
            print(f"{model:<20} {task:<30} {metric:<12} {score:<10.4f}")

    # Print SHELF Scores
    print("\n" + "=" * 90)
    print("SHELF SCORES (Aggregate)")
    print("=" * 90)
    print(f"{'Rank':<6} {'Model':<25} {'SHELF Score':<12}")
    print("-" * 45)

    for rank, (model_key, score) in enumerate(
        sorted(shelf_scores.items(), key=lambda x: -x[1]), 1
    ):
        model_name = models_config.get(model_key, {}).get("name", model_key)
        print(f"{rank:<6} {model_name:<25} {score:<12.4f}")


def save_manifest(
    output_dir: Path,
    config: dict[str, Any],
    models_run: list[str],
    tasks_run: list[str],
    version_info: dict[str, str],
    git_info: dict[str, str | bool],
    dataset_checksum: str,
    start_time: datetime,
    end_time: datetime,
) -> None:
    """Save run manifest for reproducibility."""
    manifest = {
        "run_id": f"{start_time.strftime('%Y%m%d_%H%M%S')}",
        "config_version": config.get("version", "unknown"),
        "dataset_version": config.get("dataset_version", "unknown"),
        "dataset_checksum": dataset_checksum,
        "models_evaluated": models_run,
        "tasks_evaluated": tasks_run,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": (end_time - start_time).total_seconds(),
        "versions": version_info,
        "git": git_info,
        "reproducibility": config.get("reproducibility", {}),
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    logger.info(f"Manifest saved to: {manifest_path}")


def harvest_existing_results(
    output_dir: Path,
    models_config: dict[str, Any],
    tasks_config: dict[str, Any],
    produced: dict[str, Any],
) -> tuple[dict[str, Any], int]:
    """Fold every per-task result on disk into this run's results.

    summary.json describes the whole output directory, not the last
    invocation. Without this, evaluating a single model rewrites the summary
    with that single model and silently discards the rest of the table --
    which is how a 22-model table was reduced to one fine-tune entry, and how
    a headline number came to cite a run that no longer existed.

    Results produced by the current run always win over what is on disk.
    """
    task_names = [t for task_list in tasks_config.values() for t in task_list]
    merged = dict(produced)
    recovered = 0
    for model_key in models_config:
        for task_name in task_names:
            key = f"{model_key}_{task_name}"
            if key in merged:
                continue
            path = output_dir / f"{key}.json"
            if not path.exists():
                continue
            try:
                with open(path) as f:
                    merged[key] = json.load(f)
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning(f"  could not read {path.name}: {exc}")
                continue
            recovered += 1
    return merged, recovered


def main():
    parser = argparse.ArgumentParser(
        description="Run SHELF baseline evaluations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(CONFIG_PATH),
        help="Path to config file (default: scripts/baselines/config.yaml)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Models to evaluate (default: all from config)",
    )
    parser.add_argument(
        "--task-types",
        nargs="+",
        default=None,
        choices=["retrieval", "classification", "clustering", "pair_classification"],
        help="Task types to evaluate (default: all)",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Specific tasks to evaluate (overrides --task-types)",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Maximum queries for retrieval tasks (for testing)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding (default: 32)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: results/v{version}/baselines)",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Rebuild summary.json from results already on disk; run nothing.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip model-task combinations that already have results",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be run without executing",
    )
    parser.add_argument(
        "--sparse-only",
        action="store_true",
        help="Run only sparse models (tf, tfidf, bm25)",
    )
    parser.add_argument(
        "--dense-only",
        action="store_true",
        help="Run only dense models (sentence transformers)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce logging output",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable embedding cache (re-embed for each task)",
    )
    args = parser.parse_args()

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    # Load configuration
    config = load_config(Path(args.config))
    dataset_version = config.get("dataset_version", "0.3.0")

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = (
            Path(config["output"]["base_dir"]) / f"v{dataset_version}" / "baselines"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Get reproducibility info
    version_info = get_version_info()
    git_info = get_git_info()
    dataset_checksum = compute_dataset_checksum(dataset_version)

    # Determine which models to run
    models_config = config.get("models", {})
    if args.models:
        models_to_run = [m for m in args.models if m in models_config]
    else:
        models_to_run = list(models_config.keys())

    if args.aggregate_only:
        logger.info("Aggregate-only: rebuilding summary from existing results")
        models_to_run = []

    # Filter by sparse/dense
    if args.sparse_only:
        models_to_run = [
            m
            for m in models_to_run
            if models_config[m]["type"] in ("tf", "tfidf", "bm25")
        ]
    elif args.dense_only:
        models_to_run = [
            m
            for m in models_to_run
            if models_config[m]["type"] == "sentence_transformer"
        ]

    # Determine which tasks to run
    tasks_config = config.get("tasks", {})
    if args.tasks:
        tasks_to_run = [(t, _get_task_type(t, tasks_config)) for t in args.tasks]
    elif args.task_types:
        tasks_to_run = []
        for tt in args.task_types:
            for task in tasks_config.get(tt, []):
                tasks_to_run.append((task, tt))
    else:
        tasks_to_run = []
        for tt, task_list in tasks_config.items():
            for task in task_list:
                tasks_to_run.append((task, tt))

    # Print run configuration
    logger.info("=" * 70)
    logger.info("SHELF Baseline Evaluation")
    logger.info("=" * 70)
    logger.info(f"Config: {args.config}")
    logger.info(f"Dataset version: {dataset_version}")
    logger.info(f"Dataset checksum: {dataset_checksum}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Models: {len(models_to_run)} ({', '.join(models_to_run[:5])}...)")
    logger.info(f"Tasks: {len(tasks_to_run)}")
    logger.info(f"Python: {version_info['python']}")
    logger.info(f"sklearn: {version_info['sklearn']}")
    logger.info(f"Git: {git_info['commit']} (dirty={git_info['dirty']})")

    if args.dry_run:
        print("\n--- DRY RUN ---")
        print("\nModels to evaluate:")
        for m in models_to_run:
            mc = models_config[m]
            print(f"  {m}: {mc['name']} ({mc['description']})")

        print("\nTasks to evaluate:")
        for task, task_type in tasks_to_run:
            print(f"  {task} ({task_type})")

        print("\nCombinations:")
        count = 0
        for model_key in models_to_run:
            mc = models_config[model_key]
            supports = mc.get("supports", [])
            for task, task_type in tasks_to_run:
                if task_type in supports:
                    count += 1
                    print(f"  {mc['name']} x {task}")
        print(f"\nTotal: {count} evaluations")
        return

    # Pre-collect all texts for caching (dense models only)
    all_texts: list[str] | None = None
    use_cache = not args.no_cache
    has_dense_models = any(
        models_config[m]["type"] == "sentence_transformer" for m in models_to_run
    )

    if use_cache and has_dense_models:
        logger.info("=" * 60)
        logger.info("Collecting all texts for embedding cache...")
        logger.info("=" * 60)
        all_texts = collect_all_evaluation_texts(
            tasks_config=tasks_config,
            dataset_repo=config.get("dataset_repo", "mjbommar/SHELF"),
        )

    # Run evaluations
    start_time = datetime.now(UTC)
    all_results: dict[str, dict[str, Any]] = {}
    tasks_completed: list[str] = []

    for model_key in models_to_run:
        model_config = models_config[model_key]
        model_name = model_config["name"]
        model_type = model_config["type"]
        supports = model_config.get("supports", [])

        logger.info("=" * 60)
        logger.info(f"Model: {model_name} ({model_config['description']})")
        logger.info("=" * 60)

        # First pass: check which tasks need to run vs already exist
        tasks_needing_run = []
        for task_name, task_type in tasks_to_run:
            # Check if model supports this task type
            if task_type not in supports:
                logger.info(f"  Skipping {task_name} (not supported)")
                continue

            # Check if already exists
            result_path = output_dir / f"{model_key}_{task_name}.json"
            if args.skip_existing and result_path.exists():
                logger.info(f"  Skipping {task_name} (exists)")
                # Load existing result
                with open(result_path) as f:
                    existing = json.load(f)
                all_results[f"{model_key}_{task_name}"] = existing
                continue

            tasks_needing_run.append((task_name, task_type))

        # If no tasks need to run, skip model loading entirely
        if not tasks_needing_run:
            logger.info(f"  All tasks complete for {model_name}, skipping model load")
            continue

        logger.info(f"  {len(tasks_needing_run)} tasks to run")

        # Create model once per model
        try:
            model = create_model(model_config)
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            continue

        # For dense models with caching, embed all texts once and create CachedEmbedder
        eval_model = model  # Default: use model directly
        if use_cache and model_type == "sentence_transformer" and all_texts is not None:
            from shelf.evaluate.adapters.cached import CachedEmbedder

            logger.info(f"Embedding {len(all_texts)} texts for cache...")
            embeddings = model.encode(
                all_texts,
                batch_size=args.batch_size,
                show_progress=not args.quiet,
            )
            cache = {text: emb for text, emb in zip(all_texts, embeddings)}
            logger.info(
                f"Cache built: {len(cache)} entries, "
                f"{embeddings.nbytes / 1024 / 1024:.1f} MB"
            )

            eval_model = CachedEmbedder(
                cache=cache,
                model_name=model_name,
                embedding_dim=model.embedding_dim,
                # Instruction-retrieval tasks prefix their queries, so those
                # texts are never in a cache built from raw corpus text.
                # Without a fallback the cache raises and every dense model
                # fails those tasks while sparse models pass.
                fallback=model,
            )

        for task_name, task_type in tasks_needing_run:
            result_path = output_dir / f"{model_key}_{task_name}.json"

            # Run evaluation
            result = evaluate_task(
                model=eval_model,
                model_key=model_key,
                model_config=model_config,
                task_name=task_name,
                task_type=task_type,
                max_queries=args.max_queries,
                batch_size=args.batch_size,
                show_progress=not args.quiet,
            )

            if result:
                # Add efficiency metrics (without SHELF score for now)
                result["efficiency"] = get_efficiency_dict(model_config)

                result_key = f"{model_key}_{task_name}"
                all_results[result_key] = result

                # Save individual result
                with open(result_path, "w") as f:
                    json.dump(result, f, indent=2, default=str)
                logger.info(f"    Saved: {result_path}")

                if "error" not in result:
                    logger.info(
                        f"    {result['primary_metric']}={result['primary_score']:.4f}"
                    )
                    tasks_completed.append(task_name)

            # Reset model for next task (sparse models need refitting)
            if hasattr(model, "reset"):
                model.reset()

        # Log cache stats if using cached embedder
        if (
            use_cache
            and model_type == "sentence_transformer"
            and hasattr(eval_model, "get_stats")
        ):
            stats = eval_model.get_stats()
            logger.info(f"Cache stats for {model_name}: {stats['hits']} hits")

    end_time = datetime.now(UTC)

    # Compute SHELF scores
    shelf_score_config = config.get("shelf_score", {})
    weights = shelf_score_config.get("weights", {})
    metrics = shelf_score_config.get("metrics", {})
    excluded = set(shelf_score_config.get("exclude_tasks") or [])
    if excluded:
        logger.info(
            "Excluding %d task(s) from the SHELF aggregate: %s",
            len(excluded),
            ", ".join(sorted(excluded)),
        )
    all_results, recovered = harvest_existing_results(
        output_dir, models_config, tasks_config, all_results
    )
    if recovered:
        logger.info(
            f"Recovered {recovered} prior result(s) from {output_dir} "
            "so the summary covers the full directory"
        )

    shelf_scores = compute_shelf_score(all_results, weights, metrics, excluded)

    # Compute complete efficiency metrics (with SHELF scores, Pareto, size ranks)
    model_shelf_data = {
        model_key: {"shelf_score": score} for model_key, score in shelf_scores.items()
    }
    efficiency_by_model = compute_aggregate_efficiency(model_shelf_data, models_config)

    # Update all results with complete efficiency metrics and SHELF scores
    logger.info("Updating results with complete efficiency metrics...")
    for result_key, result in all_results.items():
        if "error" in result:
            continue

        model_key = result["model_key"]

        # Add SHELF score
        if model_key in shelf_scores:
            result["shelf_score"] = round(shelf_scores[model_key], 6)

        # Update efficiency with SHELF-dependent metrics
        if model_key in efficiency_by_model:
            result["efficiency"] = efficiency_by_model[model_key]

        # Re-save the updated result
        result_path = output_dir / f"{result_key}.json"
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2, default=str)

    # Print summary
    print_summary_table(all_results, shelf_scores, models_config)

    # Save summary
    # On --aggregate-only, keep the dataset_version the results were actually
    # produced under. Taking it from the current config restamps historical
    # results with today's version, which silently rewrites provenance.
    stamped_version = dataset_version
    if args.aggregate_only:
        try:
            prior = json.loads((output_dir / "summary.json").read_text())
            stamped_version = prior.get("dataset_version", dataset_version)
        except (OSError, json.JSONDecodeError):
            pass

    summary = {
        "timestamp": end_time.isoformat(),
        "dataset_version": stamped_version,
        "dataset_checksum": dataset_checksum,
        "versions": version_info,
        "git": git_info,
        "shelf_scores": shelf_scores,
        "efficiency": efficiency_by_model,
        "results": all_results,
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"\nSummary saved to: {summary_path}")

    # Save manifest
    save_manifest(
        output_dir=output_dir,
        config=config,
        models_run=models_to_run,
        tasks_run=list(set(tasks_completed)),
        version_info=version_info,
        git_info=git_info,
        dataset_checksum=dataset_checksum,
        start_time=start_time,
        end_time=end_time,
    )

    logger.info(f"\nTotal time: {(end_time - start_time).total_seconds():.1f}s")
    logger.info(f"Results: {len(all_results)} evaluations")
    logger.info(f"Errors: {sum(1 for r in all_results.values() if 'error' in r)}")


def _get_task_type(task_name: str, tasks_config: dict[str, list[str]]) -> str:
    """Get task type from task name."""
    for task_type, tasks in tasks_config.items():
        if task_name in tasks:
            return task_type
    return "unknown"


if __name__ == "__main__":
    main()
