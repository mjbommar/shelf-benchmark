#!/usr/bin/env python
"""Add efficiency metrics to existing baseline results.

This script retroactively adds efficiency metrics (SHELF_eff, Pareto optimality,
size ranks, etc.) to existing evaluation result JSON files.

Usage:
    # Update all results in a directory
    python scripts/baselines/add_efficiency_metrics.py results/v0.3.0/baselines

    # Dry run (show what would be updated)
    python scripts/baselines/add_efficiency_metrics.py results/v0.3.0/baselines --dry-run

    # Update and recompute SHELF scores from tasks
    python scripts/baselines/add_efficiency_metrics.py results/v0.3.0/baselines --recompute-shelf
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import yaml

# Add src to path for imports
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from shelf.evaluate.efficiency import (
    compute_efficiency_metrics,
    compute_shelf_eff,
    find_pareto_optimal,
    compute_size_ranks,
    get_size_category,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Path to config
SCRIPT_DIR = Path(__file__).parent
CONFIG_PATH = SCRIPT_DIR / "config.yaml"


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def compute_shelf_score(
    model_results: dict[str, dict],
    weights: dict[str, float],
    primary_metrics: dict[str, str],
) -> float:
    """Compute aggregate SHELF score from task results.

    Args:
        model_results: Dict mapping task_name to result dict
        weights: Task type weights from config
        primary_metrics: Primary metric names by task type

    Returns:
        Weighted average SHELF score
    """
    # Map task names to task types
    task_type_map = {
        "lcc_retrieval": "retrieval",
        "form_retrieval": "retrieval",
        "category_retrieval": "retrieval",
        "lcc_classification": "classification",
        "lcgft_category_classification": "classification",
        "register_classification": "classification",
        "lcc_clustering": "clustering",
        "lcgft_clustering": "clustering",
        "register_clustering": "clustering",
        "geographic_clustering": "clustering",
        "same_lcc_pairs": "pair_classification",
        "same_form_pairs": "pair_classification",
        "same_register_pairs": "pair_classification",
        "same_audience_pairs": "pair_classification",
        "same_topic_pairs": "pair_classification",
        "topic_overlap_pairs": "pair_classification",
    }

    # Collect scores by task type
    type_scores: dict[str, list[float]] = defaultdict(list)

    for task_name, result in model_results.items():
        task_type = task_type_map.get(task_name)
        if task_type is None:
            continue

        primary_metric = primary_metrics.get(task_type)
        if primary_metric is None:
            continue

        # Get score from metrics or primary_score
        score = None
        if "metrics" in result and primary_metric in result["metrics"]:
            score = result["metrics"][primary_metric]
        elif primary_metric == result.get("primary_metric"):
            score = result.get("primary_score")

        if score is not None:
            type_scores[task_type].append(score)

    # Compute weighted average
    total_weight = 0.0
    weighted_sum = 0.0

    for task_type, scores in type_scores.items():
        if not scores:
            continue
        weight = weights.get(task_type, 0.0)
        avg_score = sum(scores) / len(scores)
        weighted_sum += weight * avg_score
        total_weight += weight

    if total_weight == 0:
        return 0.0

    return weighted_sum / total_weight


def load_results_from_directory(results_dir: Path) -> dict[str, dict[str, dict]]:
    """Load all result JSON files from a directory.

    Returns:
        Dict mapping model_key to dict of task_name to result dict
    """
    results_by_model: dict[str, dict[str, dict]] = defaultdict(dict)

    for json_file in results_dir.glob("*.json"):
        if json_file.name == "manifest.json":
            continue
        if json_file.name == "summary.json":
            continue

        try:
            with open(json_file) as f:
                result = json.load(f)

            model_key = result.get("model_key")
            task_name = result.get("task")

            if model_key and task_name:
                results_by_model[model_key][task_name] = result
                results_by_model[model_key][task_name]["_file_path"] = str(json_file)

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Error loading {json_file}: {e}")

    return dict(results_by_model)


def update_result_file(
    file_path: Path,
    efficiency_metrics: dict,
    shelf_score: float | None = None,
    dry_run: bool = False,
) -> bool:
    """Update a single result JSON file with efficiency metrics.

    Args:
        file_path: Path to result JSON file
        efficiency_metrics: Efficiency metrics dict to add
        shelf_score: Optional SHELF score to add at top level
        dry_run: If True, only print what would be updated

    Returns:
        True if file was updated (or would be updated in dry run)
    """
    with open(file_path) as f:
        result = json.load(f)

    # Check if already has efficiency metrics
    if "efficiency" in result and not dry_run:
        logger.debug(f"Skipping {file_path.name} (already has efficiency metrics)")
        return False

    # Add efficiency metrics
    result["efficiency"] = efficiency_metrics

    # Add SHELF score if provided
    if shelf_score is not None:
        result["shelf_score"] = round(shelf_score, 6)

    if dry_run:
        logger.info(f"Would update {file_path.name}: efficiency={efficiency_metrics}")
        return True

    # Write back
    with open(file_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Updated {file_path.name}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Add efficiency metrics to existing baseline results"
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Directory containing result JSON files",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without making changes",
    )
    parser.add_argument(
        "--recompute-shelf",
        action="store_true",
        help="Recompute SHELF scores from individual task results",
    )

    args = parser.parse_args()

    if not args.results_dir.exists():
        logger.error(f"Results directory not found: {args.results_dir}")
        return 1

    # Load config
    logger.info(f"Loading config from {args.config}")
    config = load_config(args.config)
    model_configs = config.get("models", {})
    shelf_config = config.get("shelf_score", {})
    weights = shelf_config.get("weights", {})
    primary_metrics = shelf_config.get("metrics", {})

    # Load all results
    logger.info(f"Loading results from {args.results_dir}")
    results_by_model = load_results_from_directory(args.results_dir)
    logger.info(f"Found results for {len(results_by_model)} models")

    # Compute SHELF scores for each model
    shelf_scores: dict[str, float] = {}
    for model_key, model_results in results_by_model.items():
        shelf_score = compute_shelf_score(model_results, weights, primary_metrics)
        shelf_scores[model_key] = shelf_score
        logger.info(f"  {model_key}: SHELF score = {shelf_score:.4f}")

    # Build data for Pareto and size rank computation
    model_tuples = {}  # For Pareto
    model_tuples_with_category = {}  # For size ranks

    for model_key in results_by_model:
        model_config = model_configs.get(model_key, {})
        num_params = model_config.get("num_params")

        if num_params is None:
            logger.debug(f"Skipping {model_key} (no param count - sparse model)")
            continue

        shelf_score = shelf_scores.get(model_key, 0.0)
        size_category = model_config.get("size_category") or get_size_category(
            num_params
        )

        model_tuples[model_key] = (shelf_score, num_params)
        model_tuples_with_category[model_key] = (shelf_score, num_params, size_category)

    # Compute Pareto set and size ranks
    pareto_set = find_pareto_optimal(model_tuples) if model_tuples else set()
    size_ranks = (
        compute_size_ranks(model_tuples_with_category)
        if model_tuples_with_category
        else {}
    )

    logger.info(f"Pareto-optimal models: {sorted(pareto_set)}")

    # Update each result file
    updated_count = 0
    skipped_count = 0

    for model_key, model_results in results_by_model.items():
        model_config = model_configs.get(model_key, {})
        num_params = model_config.get("num_params")

        # Compute efficiency metrics
        if num_params is not None:
            embedding_dim = model_config.get("embedding_dim", 0)
            size_category = model_config.get("size_category") or get_size_category(
                num_params
            )
            shelf_score = shelf_scores.get(model_key)

            metrics = compute_efficiency_metrics(
                num_params=num_params,
                embedding_dim=embedding_dim,
                size_category=size_category,
                shelf_score=shelf_score,
            )
            metrics.pareto_optimal = model_key in pareto_set
            metrics.size_rank = size_ranks.get(model_key)

            efficiency_dict = metrics.to_dict()
        else:
            # Sparse model - minimal efficiency info
            efficiency_dict = {
                "num_params": None,
                "embedding_dim": model_config.get("params", {}).get(
                    "embedding_dim", 256
                ),
                "size_category": "sparse",
                "flops_per_token": None,
                "relative_compute": None,
                "shelf_eff": None,
                "shelf_compute": None,
                "pareto_optimal": None,
                "size_rank": None,
            }

        # Update each task result file for this model
        for task_name, result in model_results.items():
            file_path = result.get("_file_path")
            if file_path is None:
                continue

            file_path = Path(file_path)
            shelf_score_to_add = (
                shelf_scores.get(model_key) if args.recompute_shelf else None
            )

            if update_result_file(
                file_path, efficiency_dict, shelf_score_to_add, args.dry_run
            ):
                updated_count += 1
            else:
                skipped_count += 1

    # Print summary
    action = "Would update" if args.dry_run else "Updated"
    logger.info(f"\n{action} {updated_count} files, skipped {skipped_count} files")

    # Print efficiency rankings
    if model_tuples:
        logger.info("\nEfficiency Rankings (SHELF_eff):")
        eff_rankings = []
        for model_key, (shelf_score, num_params) in model_tuples.items():
            shelf_eff = compute_shelf_eff(shelf_score, num_params)
            eff_rankings.append(
                (model_key, shelf_score, shelf_eff, model_key in pareto_set)
            )

        eff_rankings.sort(key=lambda x: x[2], reverse=True)

        logger.info(f"{'Model':<15} {'SHELF':>8} {'SHELF_eff':>10} {'Pareto':>8}")
        logger.info("-" * 45)
        for model_key, shelf_score, shelf_eff, is_pareto in eff_rankings:
            pareto_str = "Yes" if is_pareto else ""
            logger.info(
                f"{model_key:<15} {shelf_score:>8.4f} {shelf_eff:>10.2f} {pareto_str:>8}"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
