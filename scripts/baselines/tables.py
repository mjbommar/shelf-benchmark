#!/usr/bin/env python
"""Generate publication-ready tables from SHELF baseline results.

This script generates LaTeX tables and markdown tables from the baseline
evaluation results.

Usage:
    # Generate all tables
    python scripts/baselines/tables.py

    # Generate tables for specific result directory
    python scripts/baselines/tables.py --results-dir results/v0.3.0/baselines

    # Output format
    python scripts/baselines/tables.py --format latex
    python scripts/baselines/tables.py --format markdown
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
CONFIG_PATH = SCRIPT_DIR / "config.yaml"


def load_config(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_results(results_dir: Path) -> dict[str, dict[str, Any]]:
    """Load all results from a directory."""
    results = {}
    summary_path = results_dir / "summary.json"

    if summary_path.exists():
        with open(summary_path) as f:
            data = json.load(f)
            return data.get("results", {})

    # Load individual files
    for result_file in results_dir.glob("*.json"):
        if result_file.name in ("summary.json", "manifest.json"):
            continue
        with open(result_file) as f:
            result = json.load(f)
            key = result_file.stem
            results[key] = result

    return results


def format_score(score: float, bold: bool = False, precision: int = 3) -> str:
    """Format a score for display."""
    formatted = f"{score:.{precision}f}"
    if bold:
        return f"\\textbf{{{formatted}}}"
    return formatted


def get_best_scores(
    results: dict[str, dict[str, Any]], task_names: list[str], model_keys: list[str]
) -> dict[str, str]:
    """Find best model for each task."""
    best: dict[str, str] = {}
    for task in task_names:
        best_score = -1.0
        best_model = ""
        for model_key in model_keys:
            key = f"{model_key}_{task}"
            if key in results and "error" not in results[key]:
                score = results[key]["primary_score"]
                if score > best_score:
                    best_score = score
                    best_model = model_key
        if best_model:
            best[task] = best_model
    return best


def generate_retrieval_table(
    results: dict[str, dict[str, Any]],
    models_config: dict[str, dict[str, Any]],
    model_keys: list[str],
    output_format: str = "latex",
) -> str:
    """Generate retrieval results table."""
    tasks = ["lcc_retrieval", "form_retrieval", "category_retrieval"]
    task_labels = ["LCC", "Form", "Category"]
    metric = "ndcg@10"

    best = get_best_scores(results, tasks, model_keys)

    lines = []
    if output_format == "latex":
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"\centering")
        lines.append(r"\caption{Retrieval Results (NDCG@10)}")
        lines.append(r"\label{tab:retrieval}")
        lines.append(r"\begin{tabular}{l" + "c" * len(tasks) + "c}")
        lines.append(r"\toprule")
        header = "Model & " + " & ".join(task_labels) + r" & Avg \\"
        lines.append(header)
        lines.append(r"\midrule")

        for model_key in model_keys:
            model_name = models_config.get(model_key, {}).get("name", model_key)
            scores = []
            for task in tasks:
                key = f"{model_key}_{task}"
                if key in results and "error" not in results[key]:
                    score = results[key]["primary_score"]
                    is_best = best.get(task) == model_key
                    scores.append(format_score(score, bold=is_best))
                else:
                    scores.append("--")

            # Compute average
            valid_scores = [
                results[f"{model_key}_{t}"]["primary_score"]
                for t in tasks
                if f"{model_key}_{t}" in results and "error" not in results[f"{model_key}_{t}"]
            ]
            avg = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
            avg_str = format_score(avg) if valid_scores else "--"

            row = f"{model_name} & " + " & ".join(scores) + f" & {avg_str} \\\\"
            lines.append(row)

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")

    else:  # markdown
        header = "| Model | " + " | ".join(task_labels) + " | Avg |"
        sep = "|" + "|".join(["---"] * (len(tasks) + 2)) + "|"
        lines.append(header)
        lines.append(sep)

        for model_key in model_keys:
            model_name = models_config.get(model_key, {}).get("name", model_key)
            scores = []
            for task in tasks:
                key = f"{model_key}_{task}"
                if key in results and "error" not in results[key]:
                    score = results[key]["primary_score"]
                    is_best = best.get(task) == model_key
                    score_str = f"{score:.3f}"
                    if is_best:
                        score_str = f"**{score_str}**"
                    scores.append(score_str)
                else:
                    scores.append("--")

            valid_scores = [
                results[f"{model_key}_{t}"]["primary_score"]
                for t in tasks
                if f"{model_key}_{t}" in results and "error" not in results[f"{model_key}_{t}"]
            ]
            avg = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
            avg_str = f"{avg:.3f}" if valid_scores else "--"

            row = f"| {model_name} | " + " | ".join(scores) + f" | {avg_str} |"
            lines.append(row)

    return "\n".join(lines)


def generate_classification_table(
    results: dict[str, dict[str, Any]],
    models_config: dict[str, dict[str, Any]],
    model_keys: list[str],
    output_format: str = "latex",
) -> str:
    """Generate classification results table."""
    tasks = ["lcc_classification", "lcgft_category_classification", "register_classification"]
    task_labels = ["LCC", "LCGFT", "Register"]
    metric = "macro_f1"

    best = get_best_scores(results, tasks, model_keys)

    lines = []
    if output_format == "latex":
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"\centering")
        lines.append(r"\caption{Classification Results (Macro F1)}")
        lines.append(r"\label{tab:classification}")
        lines.append(r"\begin{tabular}{l" + "c" * len(tasks) + "c}")
        lines.append(r"\toprule")
        header = "Model & " + " & ".join(task_labels) + r" & Avg \\"
        lines.append(header)
        lines.append(r"\midrule")

        for model_key in model_keys:
            model_name = models_config.get(model_key, {}).get("name", model_key)
            scores = []
            for task in tasks:
                key = f"{model_key}_{task}"
                if key in results and "error" not in results[key]:
                    score = results[key]["primary_score"]
                    is_best = best.get(task) == model_key
                    scores.append(format_score(score, bold=is_best))
                else:
                    scores.append("--")

            valid_scores = [
                results[f"{model_key}_{t}"]["primary_score"]
                for t in tasks
                if f"{model_key}_{t}" in results and "error" not in results[f"{model_key}_{t}"]
            ]
            avg = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
            avg_str = format_score(avg) if valid_scores else "--"

            row = f"{model_name} & " + " & ".join(scores) + f" & {avg_str} \\\\"
            lines.append(row)

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")

    else:  # markdown
        header = "| Model | " + " | ".join(task_labels) + " | Avg |"
        sep = "|" + "|".join(["---"] * (len(tasks) + 2)) + "|"
        lines.append(header)
        lines.append(sep)

        for model_key in model_keys:
            model_name = models_config.get(model_key, {}).get("name", model_key)
            scores = []
            for task in tasks:
                key = f"{model_key}_{task}"
                if key in results and "error" not in results[key]:
                    score = results[key]["primary_score"]
                    is_best = best.get(task) == model_key
                    score_str = f"{score:.3f}"
                    if is_best:
                        score_str = f"**{score_str}**"
                    scores.append(score_str)
                else:
                    scores.append("--")

            valid_scores = [
                results[f"{model_key}_{t}"]["primary_score"]
                for t in tasks
                if f"{model_key}_{t}" in results and "error" not in results[f"{model_key}_{t}"]
            ]
            avg = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
            avg_str = f"{avg:.3f}" if valid_scores else "--"

            row = f"| {model_name} | " + " | ".join(scores) + f" | {avg_str} |"
            lines.append(row)

    return "\n".join(lines)


def generate_clustering_table(
    results: dict[str, dict[str, Any]],
    models_config: dict[str, dict[str, Any]],
    model_keys: list[str],
    output_format: str = "latex",
) -> str:
    """Generate clustering results table."""
    tasks = ["lcc_clustering", "lcgft_clustering", "register_clustering", "geographic_clustering"]
    task_labels = ["LCC", "LCGFT", "Register", "Geographic"]
    metric = "v_measure"

    best = get_best_scores(results, tasks, model_keys)

    lines = []
    if output_format == "latex":
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"\centering")
        lines.append(r"\caption{Clustering Results (V-Measure)}")
        lines.append(r"\label{tab:clustering}")
        lines.append(r"\begin{tabular}{l" + "c" * len(tasks) + "c}")
        lines.append(r"\toprule")
        header = "Model & " + " & ".join(task_labels) + r" & Avg \\"
        lines.append(header)
        lines.append(r"\midrule")

        for model_key in model_keys:
            model_name = models_config.get(model_key, {}).get("name", model_key)
            scores = []
            for task in tasks:
                key = f"{model_key}_{task}"
                if key in results and "error" not in results[key]:
                    score = results[key]["primary_score"]
                    is_best = best.get(task) == model_key
                    scores.append(format_score(score, bold=is_best))
                else:
                    scores.append("--")

            valid_scores = [
                results[f"{model_key}_{t}"]["primary_score"]
                for t in tasks
                if f"{model_key}_{t}" in results and "error" not in results[f"{model_key}_{t}"]
            ]
            avg = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
            avg_str = format_score(avg) if valid_scores else "--"

            row = f"{model_name} & " + " & ".join(scores) + f" & {avg_str} \\\\"
            lines.append(row)

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")

    else:  # markdown
        header = "| Model | " + " | ".join(task_labels) + " | Avg |"
        sep = "|" + "|".join(["---"] * (len(tasks) + 2)) + "|"
        lines.append(header)
        lines.append(sep)

        for model_key in model_keys:
            model_name = models_config.get(model_key, {}).get("name", model_key)
            scores = []
            for task in tasks:
                key = f"{model_key}_{task}"
                if key in results and "error" not in results[key]:
                    score = results[key]["primary_score"]
                    is_best = best.get(task) == model_key
                    score_str = f"{score:.3f}"
                    if is_best:
                        score_str = f"**{score_str}**"
                    scores.append(score_str)
                else:
                    scores.append("--")

            valid_scores = [
                results[f"{model_key}_{t}"]["primary_score"]
                for t in tasks
                if f"{model_key}_{t}" in results and "error" not in results[f"{model_key}_{t}"]
            ]
            avg = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
            avg_str = f"{avg:.3f}" if valid_scores else "--"

            row = f"| {model_name} | " + " | ".join(scores) + f" | {avg_str} |"
            lines.append(row)

    return "\n".join(lines)


def generate_pair_table(
    results: dict[str, dict[str, Any]],
    models_config: dict[str, dict[str, Any]],
    model_keys: list[str],
    output_format: str = "latex",
) -> str:
    """Generate pair classification results table."""
    tasks = [
        "same_lcc_pairs",
        "same_form_pairs",
        "same_register_pairs",
        "same_audience_pairs",
        "same_topic_pairs",
        "topic_overlap_pairs",
    ]
    task_labels = ["LCC", "Form", "Reg.", "Aud.", "Topic", "Overlap"]

    best = get_best_scores(results, tasks, model_keys)

    lines = []
    if output_format == "latex":
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"\centering")
        lines.append(r"\caption{Pair Classification Results (F1)}")
        lines.append(r"\label{tab:pairs}")
        lines.append(r"\begin{tabular}{l" + "c" * len(tasks) + "c}")
        lines.append(r"\toprule")
        header = "Model & " + " & ".join(task_labels) + r" & Avg \\"
        lines.append(header)
        lines.append(r"\midrule")

        for model_key in model_keys:
            model_name = models_config.get(model_key, {}).get("name", model_key)
            scores = []
            for task in tasks:
                key = f"{model_key}_{task}"
                if key in results and "error" not in results[key]:
                    score = results[key]["primary_score"]
                    is_best = best.get(task) == model_key
                    scores.append(format_score(score, bold=is_best))
                else:
                    scores.append("--")

            valid_scores = [
                results[f"{model_key}_{t}"]["primary_score"]
                for t in tasks
                if f"{model_key}_{t}" in results and "error" not in results[f"{model_key}_{t}"]
            ]
            avg = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
            avg_str = format_score(avg) if valid_scores else "--"

            row = f"{model_name} & " + " & ".join(scores) + f" & {avg_str} \\\\"
            lines.append(row)

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")

    else:  # markdown
        header = "| Model | " + " | ".join(task_labels) + " | Avg |"
        sep = "|" + "|".join(["---"] * (len(tasks) + 2)) + "|"
        lines.append(header)
        lines.append(sep)

        for model_key in model_keys:
            model_name = models_config.get(model_key, {}).get("name", model_key)
            scores = []
            for task in tasks:
                key = f"{model_key}_{task}"
                if key in results and "error" not in results[key]:
                    score = results[key]["primary_score"]
                    is_best = best.get(task) == model_key
                    score_str = f"{score:.3f}"
                    if is_best:
                        score_str = f"**{score_str}**"
                    scores.append(score_str)
                else:
                    scores.append("--")

            valid_scores = [
                results[f"{model_key}_{t}"]["primary_score"]
                for t in tasks
                if f"{model_key}_{t}" in results and "error" not in results[f"{model_key}_{t}"]
            ]
            avg = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
            avg_str = f"{avg:.3f}" if valid_scores else "--"

            row = f"| {model_name} | " + " | ".join(scores) + f" | {avg_str} |"
            lines.append(row)

    return "\n".join(lines)


def generate_leaderboard_table(
    results: dict[str, dict[str, Any]],
    models_config: dict[str, dict[str, Any]],
    model_keys: list[str],
    weights: dict[str, float],
    output_format: str = "latex",
) -> str:
    """Generate SHELF Score leaderboard table."""
    # Compute scores per task type
    model_type_avgs: dict[str, dict[str, float]] = {}

    task_types = {
        "retrieval": ["lcc_retrieval", "form_retrieval", "category_retrieval"],
        "classification": ["lcc_classification", "lcgft_category_classification", "register_classification"],
        "clustering": ["lcc_clustering", "lcgft_clustering", "register_clustering", "geographic_clustering"],
        "pair_classification": [
            "same_lcc_pairs",
            "same_form_pairs",
            "same_register_pairs",
            "same_audience_pairs",
            "same_topic_pairs",
            "topic_overlap_pairs",
        ],
    }

    for model_key in model_keys:
        model_type_avgs[model_key] = {}
        for task_type, tasks in task_types.items():
            valid_scores = [
                results[f"{model_key}_{t}"]["primary_score"]
                for t in tasks
                if f"{model_key}_{t}" in results and "error" not in results[f"{model_key}_{t}"]
            ]
            if valid_scores:
                model_type_avgs[model_key][task_type] = sum(valid_scores) / len(valid_scores)

    # Compute SHELF scores
    shelf_scores: dict[str, float] = {}
    for model_key, type_avgs in model_type_avgs.items():
        weighted_sum = 0.0
        total_weight = 0.0
        for task_type, avg in type_avgs.items():
            if task_type in weights:
                weighted_sum += avg * weights[task_type]
                total_weight += weights[task_type]
        if total_weight > 0:
            shelf_scores[model_key] = weighted_sum / total_weight

    # Sort by SHELF score
    sorted_models = sorted(shelf_scores.keys(), key=lambda m: -shelf_scores.get(m, 0))

    lines = []
    type_labels = ["Ret.", "Clf.", "Clust.", "Pair"]
    type_keys = ["retrieval", "classification", "clustering", "pair_classification"]

    if output_format == "latex":
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"\centering")
        lines.append(r"\caption{SHELF Leaderboard}")
        lines.append(r"\label{tab:leaderboard}")
        lines.append(r"\begin{tabular}{rl" + "c" * len(type_labels) + "c}")
        lines.append(r"\toprule")
        header = r"Rank & Model & " + " & ".join(type_labels) + r" & SHELF \\"
        lines.append(header)
        lines.append(r"\midrule")

        for rank, model_key in enumerate(sorted_models, 1):
            model_name = models_config.get(model_key, {}).get("name", model_key)
            type_scores = []
            for tk in type_keys:
                if tk in model_type_avgs[model_key]:
                    type_scores.append(format_score(model_type_avgs[model_key][tk]))
                else:
                    type_scores.append("--")

            shelf_score = shelf_scores.get(model_key, 0.0)
            shelf_str = format_score(shelf_score, bold=(rank == 1))

            row = f"{rank} & {model_name} & " + " & ".join(type_scores) + f" & {shelf_str} \\\\"
            lines.append(row)

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")

    else:  # markdown
        header = "| Rank | Model | " + " | ".join(type_labels) + " | SHELF |"
        sep = "|" + "|".join(["---"] * (len(type_labels) + 3)) + "|"
        lines.append(header)
        lines.append(sep)

        for rank, model_key in enumerate(sorted_models, 1):
            model_name = models_config.get(model_key, {}).get("name", model_key)
            type_scores = []
            for tk in type_keys:
                if tk in model_type_avgs[model_key]:
                    score_str = f"{model_type_avgs[model_key][tk]:.3f}"
                    type_scores.append(score_str)
                else:
                    type_scores.append("--")

            shelf_score = shelf_scores.get(model_key, 0.0)
            shelf_str = f"{shelf_score:.3f}"
            if rank == 1:
                shelf_str = f"**{shelf_str}**"

            row = f"| {rank} | {model_name} | " + " | ".join(type_scores) + f" | {shelf_str} |"
            lines.append(row)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate tables from SHELF baseline results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Results directory (default: auto-detect latest)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(CONFIG_PATH),
        help="Config file path",
    )
    parser.add_argument(
        "--format",
        choices=["latex", "markdown", "both"],
        default="both",
        help="Output format (default: both)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save tables (default: {results_dir}/tables)",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(Path(args.config))
    models_config = config.get("models", {})
    weights = config.get("shelf_score", {}).get("weights", {})

    # Find results directory
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        # Auto-detect latest version
        base_dir = Path(config["output"]["base_dir"])
        versions = sorted(base_dir.glob("v*"), reverse=True)
        if not versions:
            logger.error("No results found")
            return
        results_dir = versions[0] / "baselines"

    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return

    logger.info(f"Loading results from: {results_dir}")
    results = load_results(results_dir)
    logger.info(f"Loaded {len(results)} results")

    # Get model keys that have results
    model_keys = []
    for key in sorted(set(k.rsplit("_", 1)[0] for k in results.keys() if "_" in k)):
        if key in models_config:
            model_keys.append(key)

    logger.info(f"Models with results: {model_keys}")

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = results_dir / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate tables
    formats = ["latex", "markdown"] if args.format == "both" else [args.format]

    for fmt in formats:
        ext = "tex" if fmt == "latex" else "md"

        # Retrieval table
        table = generate_retrieval_table(results, models_config, model_keys, fmt)
        output_path = output_dir / f"retrieval.{ext}"
        with open(output_path, "w") as f:
            f.write(table)
        logger.info(f"Saved: {output_path}")

        # Classification table
        table = generate_classification_table(results, models_config, model_keys, fmt)
        output_path = output_dir / f"classification.{ext}"
        with open(output_path, "w") as f:
            f.write(table)
        logger.info(f"Saved: {output_path}")

        # Clustering table
        table = generate_clustering_table(results, models_config, model_keys, fmt)
        output_path = output_dir / f"clustering.{ext}"
        with open(output_path, "w") as f:
            f.write(table)
        logger.info(f"Saved: {output_path}")

        # Pair classification table
        table = generate_pair_table(results, models_config, model_keys, fmt)
        output_path = output_dir / f"pair_classification.{ext}"
        with open(output_path, "w") as f:
            f.write(table)
        logger.info(f"Saved: {output_path}")

        # Leaderboard table
        table = generate_leaderboard_table(results, models_config, model_keys, weights, fmt)
        output_path = output_dir / f"leaderboard.{ext}"
        with open(output_path, "w") as f:
            f.write(table)
        logger.info(f"Saved: {output_path}")

    logger.info(f"\nAll tables saved to: {output_dir}")


if __name__ == "__main__":
    main()
