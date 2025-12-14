"""Evaluation CLI commands.

Commands for running evaluations and viewing results.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional

import typer
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="eval",
    help="Run evaluations and view results",
)
console = Console()

# Default paths
DEFAULT_CONFIG_PATH = (
    Path(__file__).parent.parent.parent.parent / "scripts" / "baselines" / "config.yaml"
)
DEFAULT_RESULTS_DIR = Path(__file__).parent.parent.parent.parent / "results"


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_results_dir(config_path: Path) -> Path:
    """Get results directory based on config."""
    cfg = load_config(config_path)
    dataset_version = cfg.get("dataset_version", "0.3.0")
    return DEFAULT_RESULTS_DIR / f"v{dataset_version}" / "baselines"


def format_params(num_params: int | None) -> str:
    """Format parameter count as human-readable string."""
    if num_params is None:
        return "-"
    if num_params >= 1_000_000_000:
        return f"{num_params / 1_000_000_000:.1f}B"
    if num_params >= 1_000_000:
        return f"{num_params / 1_000_000:.1f}M"
    return str(num_params)


@app.command("run")
def cmd_run(
    config: Annotated[
        Path, typer.Option("--config", "-c", help="Path to config file")
    ] = DEFAULT_CONFIG_PATH,
    models: Annotated[
        Optional[list[str]], typer.Option("--models", "-m", help="Models to evaluate")
    ] = None,
    tasks: Annotated[
        Optional[list[str]], typer.Option("--tasks", "-t", help="Tasks to evaluate")
    ] = None,
    task_types: Annotated[
        Optional[list[str]], typer.Option("--task-types", help="Task types to evaluate")
    ] = None,
    skip_existing: Annotated[
        bool, typer.Option("--skip-existing", "-s", help="Skip existing results")
    ] = False,
    dense_only: Annotated[
        bool, typer.Option("--dense-only", help="Run only dense models")
    ] = False,
    sparse_only: Annotated[
        bool, typer.Option("--sparse-only", help="Run only sparse models")
    ] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show what would be run")
    ] = False,
    batch_size: Annotated[
        int, typer.Option("--batch-size", "-b", help="Batch size for embedding")
    ] = 32,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Reduce output")] = False,
    jsonl_log: Annotated[
        Optional[Path], typer.Option("--jsonl-log", help="Path to JSONL log file")
    ] = None,
    no_cache: Annotated[
        bool, typer.Option("--no-cache", help="Disable embedding cache")
    ] = False,
    save_samples: Annotated[
        bool,
        typer.Option(
            "--save-samples", help="Save per-sample results for detailed analysis"
        ),
    ] = False,
):
    """Run baseline evaluations.

    Examples:
        shelf eval run                           # Run all
        shelf eval run --models minilm bge_base  # Specific models
        shelf eval run --skip-existing           # Skip completed
        shelf eval run --dense-only              # Only dense models
        shelf eval run --jsonl-log eval.jsonl    # Log to JSONL file
        shelf eval run --save-samples            # Save per-sample results
    """
    from shelf.evaluate.output import (
        JSONLHandler,
        NullHandler,
        RichHandler,
        create_composite,
    )
    from shelf.evaluate.runner import EvaluationOrchestrator, RunConfig

    # Load and configure
    if not config.exists():
        console.print(f"[red]Error:[/red] Config file not found: {config}")
        raise typer.Exit(1)

    try:
        run_config = RunConfig.from_yaml(config)
    except Exception as e:
        console.print(f"[red]Error loading config:[/red] {e}")
        raise typer.Exit(1)

    # Apply filters
    if models:
        try:
            run_config = run_config.with_models(models)
        except ValueError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)

    if tasks:
        try:
            run_config = run_config.with_tasks(tasks)
        except ValueError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)

    if task_types:
        try:
            run_config = run_config.with_task_types(task_types)
        except ValueError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)

    # Filter by dense/sparse
    if dense_only or sparse_only:
        filtered_models = []
        for model_key in run_config.models:
            model_cfg = run_config.get_model_config(model_key)
            model_type = model_cfg.get("type", "")
            is_sparse = model_type in ("tf", "tfidf", "bm25")

            if dense_only and not is_sparse:
                filtered_models.append(model_key)
            elif sparse_only and is_sparse:
                filtered_models.append(model_key)

        run_config = run_config.with_models(filtered_models)

    # Apply other options
    run_config = run_config.with_skip_existing(skip_existing)

    # Apply batch size, cache, and sample settings via dataclass.replace
    from dataclasses import replace

    run_config = replace(
        run_config,
        batch_size=batch_size,
        use_cache=not no_cache,
        show_progress=not quiet,
        save_samples=save_samples,
    )

    # Setup output handlers
    handlers: list = []

    if not quiet:
        handlers.append(RichHandler(console=console))

    if jsonl_log:
        handlers.append(JSONLHandler(jsonl_log))

    if handlers:
        output = create_composite(*handlers)
    else:
        output = NullHandler()

    # Create orchestrator
    orchestrator = EvaluationOrchestrator(
        config=run_config,
        output=output,
        dry_run=dry_run,
    )

    # Run evaluation
    try:
        result = orchestrator.run()

        if dry_run:
            console.print("[bold]Dry run complete[/bold] - no evaluations executed")
        elif result.failed_tasks > 0:
            raise typer.Exit(1)

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error during evaluation:[/red] {e}")
        raise typer.Exit(1)


@app.command("status")
def cmd_status(
    config: Annotated[
        Path, typer.Option("--config", "-c", help="Path to config file")
    ] = DEFAULT_CONFIG_PATH,
):
    """Show evaluation progress and status."""
    if not config.exists():
        console.print(f"[red]Error:[/red] Config file not found: {config}")
        raise typer.Exit(1)

    cfg = load_config(config)
    models_cfg = cfg.get("models", {})
    tasks_cfg = cfg.get("tasks", {})
    results_dir = get_results_dir(config)

    # Count total tasks per task type
    all_tasks = []
    for task_type, task_list in tasks_cfg.items():
        for task_name in task_list:
            all_tasks.append((task_name, task_type))

    # Check status of each model
    table = Table(
        title="Evaluation Status",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Model", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Progress", justify="center", style="green")
    table.add_column("Last Updated", style="dim")

    total_done = 0
    total_possible = 0

    for model_key, model_cfg in models_cfg.items():
        supports = model_cfg.get("supports", [])
        model_name = model_cfg.get("name", model_key)

        # Count applicable tasks
        applicable_tasks = [t for t, tt in all_tasks if tt in supports]
        total_tasks = len(applicable_tasks)
        total_possible += total_tasks

        # Check which results exist
        done_tasks = 0
        latest_timestamp = None

        for task_name in applicable_tasks:
            result_path = results_dir / f"{model_key}_{task_name}.json"
            if result_path.exists():
                done_tasks += 1
                total_done += 1
                # Get timestamp
                try:
                    with open(result_path) as f:
                        result = json.load(f)
                    ts_str = result.get("context", {}).get("timestamp")
                    if ts_str:
                        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                        if latest_timestamp is None or ts > latest_timestamp:
                            latest_timestamp = ts
                except Exception:
                    pass

        # Determine status
        if done_tasks == total_tasks:
            status = "[green]✓ Done[/green]"
        elif done_tasks > 0:
            status = "[yellow]⏳ Running[/yellow]"
        else:
            status = "[dim]○ Pending[/dim]"

        # Format timestamp
        ts_display = (
            latest_timestamp.strftime("%Y-%m-%d %H:%M") if latest_timestamp else "-"
        )

        table.add_row(
            model_name,
            status,
            f"{done_tasks}/{total_tasks}",
            ts_display,
        )

    console.print(table)
    console.print()

    # Summary
    pct = (total_done / total_possible * 100) if total_possible > 0 else 0
    console.print(
        f"Progress: [cyan]{total_done}[/cyan]/[cyan]{total_possible}[/cyan] tasks ([green]{pct:.0f}%[/green])"
    )


def _load_all_results(results_dir: Path) -> dict[str, dict]:
    """Load all result JSON files from a directory.

    Returns:
        Dict mapping "model_task" to result dict
    """
    all_results = {}
    for result_path in results_dir.glob("*.json"):
        if result_path.name in ("summary.json", "manifest.json"):
            continue
        try:
            with open(result_path) as f:
                result = json.load(f)
            # Key is model_task (e.g., "minilm_lcc_classification")
            key = result_path.stem
            all_results[key] = result
        except Exception:
            pass
    return all_results


def _compute_confidence_interval(
    scores: list[float], confidence: float = 0.95
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for mean."""
    import numpy as np

    if len(scores) < 2:
        mean = float(np.mean(scores))
        return (mean, mean)

    # Simple percentile bootstrap
    n_bootstrap = 1000
    rng = np.random.default_rng(42)
    bootstrap_means = []

    for _ in range(n_bootstrap):
        sample = rng.choice(scores, size=len(scores), replace=True)
        bootstrap_means.append(np.mean(sample))

    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

    return (float(lower), float(upper))


@app.command("results")
def cmd_results(
    config: Annotated[
        Path, typer.Option("--config", "-c", help="Path to config file")
    ] = DEFAULT_CONFIG_PATH,
    sort_by: Annotated[
        str, typer.Option("--sort", help="Sort by: shelf, shelf_eff, params")
    ] = "shelf",
    limit: Annotated[
        int, typer.Option("--limit", "-n", help="Number of models to show")
    ] = 20,
    by_task_type: Annotated[
        bool, typer.Option("--by-task-type", help="Show rankings by task type")
    ] = False,
    by_task: Annotated[
        bool, typer.Option("--by-task", help="Show per-task breakdown")
    ] = False,
    stats: Annotated[
        bool, typer.Option("--stats", help="Show statistical summary")
    ] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show all tables")
    ] = False,
):
    """Show benchmark results summary.

    Examples:
        shelf eval results                    # Basic rankings
        shelf eval results --by-task-type     # Rankings per task type
        shelf eval results --by-task          # Per-task breakdown
        shelf eval results --stats            # Statistical summary
        shelf eval results -v                 # All tables
    """
    import numpy as np

    if not config.exists():
        console.print(f"[red]Error:[/red] Config file not found: {config}")
        raise typer.Exit(1)

    cfg = load_config(config)
    models_cfg = cfg.get("models", {})
    results_dir = get_results_dir(config)

    if not results_dir.exists():
        console.print(f"[yellow]No results found in {results_dir}[/yellow]")
        raise typer.Exit(0)

    # Load all results
    all_results = _load_all_results(results_dir)
    if not all_results:
        console.print("[yellow]No results found[/yellow]")
        raise typer.Exit(0)

    # If verbose, enable all tables
    if verbose:
        by_task_type = True
        by_task = True
        stats = True

    # Organize results by model and task type
    model_data: dict[str, dict] = {}  # model_key -> aggregated data
    task_type_scores: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    task_scores: dict[str, dict[str, float]] = defaultdict(
        dict
    )  # task -> model -> score

    for result_key, result in all_results.items():
        model_key = result.get("model_key")
        model_name = result.get("model")
        task_name = result.get("task")
        task_type = result.get("task_type")
        primary_score = result.get("primary_score")

        if not all([model_key, task_name, task_type, primary_score is not None]):
            continue

        # Type narrowing for type checker
        assert model_key is not None
        assert task_name is not None
        assert task_type is not None
        assert primary_score is not None

        # Store per-task scores
        task_scores[task_name][model_key] = primary_score

        # Store per-task-type scores
        task_type_scores[task_type][model_key].append(primary_score)

        # Initialize model data if needed
        if model_key not in model_data:
            efficiency = result.get("efficiency", {})
            model_cfg = models_cfg.get(model_key, {})
            model_type = model_cfg.get("type", result.get("model_type", ""))

            model_data[model_key] = {
                "name": model_name or model_key,
                "shelf_score": result.get("shelf_score"),
                "shelf_eff": efficiency.get("shelf_eff"),
                "pareto_optimal": efficiency.get("pareto_optimal"),
                "size_category": efficiency.get(
                    "size_category",
                    "sparse" if model_type in ["tf", "tfidf", "bm25"] else "base",
                ),
                "num_params": efficiency.get("num_params"),
                "num_params_torch": efficiency.get("num_params_torch"),
                "hidden_size": efficiency.get("hidden_size"),
                "context_window": efficiency.get("context_window"),
                "throughput_bytes_sec": efficiency.get("throughput_bytes_sec"),
                "all_scores": [],
            }

        model_data[model_key]["all_scores"].append(primary_score)

    # Compute SHELF scores if missing
    shelf_config = cfg.get("shelf_score", {})
    weights = shelf_config.get("weights", {})

    for model_key, data in model_data.items():
        if data["shelf_score"] is None:
            weighted_sum = 0.0
            total_weight = 0.0
            for task_type, model_scores_map in task_type_scores.items():
                if model_key in model_scores_map and task_type in weights:
                    scores = model_scores_map[model_key]
                    avg_score = sum(scores) / len(scores)
                    weight = weights[task_type]
                    weighted_sum += avg_score * weight
                    total_weight += weight
            data["shelf_score"] = (
                weighted_sum / total_weight if total_weight > 0 else 0.0
            )

    # Sort models
    if sort_by == "shelf_eff":
        sorted_models = sorted(
            model_data.items(), key=lambda x: x[1].get("shelf_eff") or 0, reverse=True
        )
    elif sort_by == "params":
        sorted_models = sorted(
            model_data.items(),
            key=lambda x: x[1].get("num_params") or float("inf"),
        )
    else:  # shelf
        sorted_models = sorted(
            model_data.items(), key=lambda x: x[1].get("shelf_score") or 0, reverse=True
        )

    # Display header
    dataset_version = cfg.get("dataset_version", "0.3.0")
    console.print()
    console.print(
        Panel(
            f"[bold]SHELF Benchmark Results[/bold]\nDataset: v{dataset_version}",
            border_style="cyan",
        )
    )
    console.print()

    # === Main SHELF Score Rankings ===
    table = Table(
        title="SHELF Score Rankings",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Rank", justify="right", style="dim")
    table.add_column("Model", style="cyan")
    table.add_column("SHELF", justify="right", style="green")
    table.add_column("SHELF_eff", justify="right", style="yellow")
    table.add_column("Pareto", justify="center")
    table.add_column("Size", style="magenta")
    table.add_column("Hidden", justify="right", style="dim")
    table.add_column("Context", justify="right", style="dim")
    table.add_column("KB/s", justify="right", style="dim")

    for rank, (model_key, data) in enumerate(sorted_models[:limit], 1):
        shelf_score = data.get("shelf_score", 0)
        shelf_eff = data.get("shelf_eff")
        pareto = data.get("pareto_optimal")
        size_cat = data.get("size_category", "-")
        hidden_size = data.get("hidden_size")
        context_window = data.get("context_window")
        throughput = data.get("throughput_bytes_sec")

        pareto_str = "[green]✓[/green]" if pareto else ("-" if pareto is None else "")
        hidden_str = str(hidden_size) if hidden_size else "-"
        context_str = str(context_window) if context_window else "-"
        # Convert bytes/sec to KB/sec
        throughput_str = f"{throughput / 1024:.0f}" if throughput else "-"

        table.add_row(
            str(rank),
            data.get("name", model_key),
            f"{shelf_score:.4f}",
            f"{shelf_eff:.2f}" if shelf_eff else "-",
            pareto_str,
            size_cat,
            hidden_str,
            context_str,
            throughput_str,
        )

    console.print(table)

    # === Best by Size Category ===
    console.print()
    cat_table = Table(
        title="Best by Size Category",
        show_header=True,
        header_style="bold cyan",
    )
    cat_table.add_column("Category", style="magenta")
    cat_table.add_column("Model", style="cyan")
    cat_table.add_column("SHELF", justify="right", style="green")
    cat_table.add_column("SHELF_eff", justify="right", style="yellow")

    best_by_category: dict[str, tuple[str, dict]] = {}
    for model_key, data in model_data.items():
        cat = data.get("size_category", "unknown")
        if cat not in best_by_category or (data.get("shelf_score") or 0) > (
            best_by_category[cat][1].get("shelf_score") or 0
        ):
            best_by_category[cat] = (model_key, data)

    for cat in ["small", "base", "large", "sparse"]:
        if cat in best_by_category:
            model_key, data = best_by_category[cat]
            shelf_eff = data.get("shelf_eff")
            cat_table.add_row(
                cat,
                data.get("name", model_key),
                f"{data.get('shelf_score', 0):.4f}",
                f"{shelf_eff:.2f}" if shelf_eff else "-",
            )

    console.print(cat_table)

    # === Rankings by Task Type ===
    if by_task_type:
        console.print()
        for task_type in [
            "classification",
            "retrieval",
            "clustering",
            "pair_classification",
        ]:
            if task_type not in task_type_scores:
                continue

            type_table = Table(
                title=f"Rankings: {task_type.replace('_', ' ').title()}",
                show_header=True,
                header_style="bold cyan",
            )
            type_table.add_column("Rank", justify="right", style="dim")
            type_table.add_column("Model", style="cyan")
            type_table.add_column("Avg Score", justify="right", style="green")
            type_table.add_column("Tasks", justify="right", style="dim")

            # Sort models by average score for this task type
            type_rankings = []
            for model_key, scores in task_type_scores[task_type].items():
                avg_score = sum(scores) / len(scores)
                model_name = model_data.get(model_key, {}).get("name", model_key)
                type_rankings.append((model_key, model_name, avg_score, len(scores)))

            type_rankings.sort(key=lambda x: x[2], reverse=True)

            for rank, (model_key, model_name, avg_score, num_tasks) in enumerate(
                type_rankings[:limit], 1
            ):
                type_table.add_row(
                    str(rank),
                    model_name,
                    f"{avg_score:.4f}",
                    str(num_tasks),
                )

            console.print(type_table)
            console.print()

    # === Per-Task Breakdown ===
    if by_task:
        console.print()
        # Group tasks by type
        tasks_by_type: dict[str, list[str]] = defaultdict(list)
        for result in all_results.values():
            task_name = result.get("task")
            task_type = result.get("task_type")
            if task_name and task_type and task_name not in tasks_by_type[task_type]:
                tasks_by_type[task_type].append(task_name)

        for task_type in [
            "classification",
            "retrieval",
            "clustering",
            "pair_classification",
        ]:
            if task_type not in tasks_by_type:
                continue

            task_table = Table(
                title=f"Per-Task Scores: {task_type.replace('_', ' ').title()}",
                show_header=True,
                header_style="bold cyan",
            )
            task_table.add_column("Task", style="cyan")
            task_table.add_column("Best Model", style="green")
            task_table.add_column("Best Score", justify="right", style="green")
            task_table.add_column("Avg Score", justify="right")
            task_table.add_column("Range", justify="right", style="dim")

            for task_name in sorted(tasks_by_type[task_type]):
                if task_name not in task_scores:
                    continue

                scores_map = task_scores[task_name]
                if not scores_map:
                    continue

                scores_list = list(scores_map.values())
                best_model = max(scores_map.keys(), key=lambda k: scores_map[k])
                best_score = scores_map[best_model]
                avg_score = sum(scores_list) / len(scores_list)
                min_score = min(scores_list)
                max_score = max(scores_list)

                best_model_name = model_data.get(best_model, {}).get("name", best_model)
                # Truncate long task names
                task_display = (
                    task_name[:25] + "..." if len(task_name) > 28 else task_name
                )

                task_table.add_row(
                    task_display,
                    best_model_name,
                    f"{best_score:.4f}",
                    f"{avg_score:.4f}",
                    f"{min_score:.3f}-{max_score:.3f}",
                )

            console.print(task_table)
            console.print()

    # === Statistical Summary ===
    if stats:
        console.print()
        stats_table = Table(
            title="Statistical Summary (95% CI)",
            show_header=True,
            header_style="bold cyan",
        )
        stats_table.add_column("Model", style="cyan")
        stats_table.add_column("Mean", justify="right", style="green")
        stats_table.add_column("Std", justify="right")
        stats_table.add_column("Min", justify="right", style="dim")
        stats_table.add_column("Max", justify="right", style="dim")
        stats_table.add_column("95% CI", justify="right", style="yellow")
        stats_table.add_column("N", justify="right", style="dim")

        # Sort by mean score
        stats_data = []
        for model_key, data in model_data.items():
            scores = data.get("all_scores", [])
            if not scores:
                continue

            mean_score = float(np.mean(scores))
            std_score = float(np.std(scores)) if len(scores) > 1 else 0.0
            min_score = float(np.min(scores))
            max_score = float(np.max(scores))
            ci_lower, ci_upper = _compute_confidence_interval(scores)

            stats_data.append(
                {
                    "model_key": model_key,
                    "name": data.get("name", model_key),
                    "mean": mean_score,
                    "std": std_score,
                    "min": min_score,
                    "max": max_score,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "n": len(scores),
                }
            )

        stats_data.sort(key=lambda x: x["mean"], reverse=True)

        for item in stats_data[:limit]:
            stats_table.add_row(
                item["name"],
                f"{item['mean']:.4f}",
                f"{item['std']:.4f}",
                f"{item['min']:.4f}",
                f"{item['max']:.4f}",
                f"[{item['ci_lower']:.3f}, {item['ci_upper']:.3f}]",
                str(item["n"]),
            )

        console.print(stats_table)

        # === Head-to-Head: Top Models vs Baseline ===
        if len(stats_data) >= 2:
            console.print()
            # Compare top models to TF-IDF baseline (or first sparse model)
            baseline_key = None
            for item in stats_data:
                mk = item["model_key"]
                if mk in ["tfidf", "tf", "bm25"]:
                    baseline_key = mk
                    break

            if baseline_key:
                baseline_data = next(
                    d for d in stats_data if d["model_key"] == baseline_key
                )

                compare_table = Table(
                    title=f"Comparison vs Baseline ({baseline_data['name']})",
                    show_header=True,
                    header_style="bold cyan",
                )
                compare_table.add_column("Model", style="cyan")
                compare_table.add_column("Mean", justify="right", style="green")
                compare_table.add_column("Δ vs Baseline", justify="right")
                compare_table.add_column("% Improvement", justify="right")

                for item in stats_data[:limit]:
                    if item["model_key"] == baseline_key:
                        continue

                    delta = item["mean"] - baseline_data["mean"]
                    pct_improvement = (
                        (delta / baseline_data["mean"] * 100)
                        if baseline_data["mean"] > 0
                        else 0
                    )

                    delta_style = "green" if delta > 0 else "red"
                    delta_str = f"[{delta_style}]{delta:+.4f}[/{delta_style}]"
                    pct_str = f"[{delta_style}]{pct_improvement:+.1f}%[/{delta_style}]"

                    compare_table.add_row(
                        item["name"],
                        f"{item['mean']:.4f}",
                        delta_str,
                        pct_str,
                    )

                console.print(compare_table)


@app.command("efficiency")
def cmd_efficiency(
    config: Annotated[
        Path, typer.Option("--config", "-c", help="Path to config file")
    ] = DEFAULT_CONFIG_PATH,
    limit: Annotated[
        int, typer.Option("--limit", "-n", help="Number of models to show")
    ] = 20,
):
    """Show efficiency rankings (best performance per parameter)."""
    # Delegate to results with efficiency sorting
    cmd_results(config=config, sort_by="shelf_eff", limit=limit)


@app.command("analyze")
def cmd_analyze(
    config: Annotated[
        Path, typer.Option("--config", "-c", help="Path to config file")
    ] = DEFAULT_CONFIG_PATH,
    correlation: Annotated[
        bool, typer.Option("--correlation", help="Show task correlation analysis")
    ] = False,
    pairwise: Annotated[
        bool, typer.Option("--pairwise", help="Show pairwise significance matrix")
    ] = False,
    compare: Annotated[
        Optional[list[str]],
        typer.Option(
            "--compare",
            help="Compare two models (e.g., --compare bge_large --compare tfidf)",
        ),
    ] = None,
    equivalence: Annotated[
        bool, typer.Option("--equivalence", help="Show equivalence groups")
    ] = False,
    champions: Annotated[
        bool, typer.Option("--champions", help="Show task champion analysis")
    ] = False,
    alpha: Annotated[float, typer.Option("--alpha", help="Significance level")] = 0.05,
    correction: Annotated[
        str,
        typer.Option(
            "--correction",
            help="Multiple comparison correction: bonferroni, holm, fdr_bh",
        ),
    ] = "holm",
    export_json: Annotated[
        Optional[Path], typer.Option("--export-json", help="Export analysis to JSON")
    ] = None,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show all analysis tables")
    ] = False,
):
    """Statistical analysis for peer review.

    Provides comprehensive statistical analysis that addresses common
    peer reviewer concerns about benchmark validity:

    - Task independence (correlation analysis)
    - Statistical significance with proper corrections
    - Effect sizes (Cohen's d) for meaningful interpretation
    - Equivalence groups (which models are NOT significantly different)
    - Task champion diversity (do different models win different tasks?)

    Examples:
        shelf eval analyze                      # Summary statistics
        shelf eval analyze --correlation        # Task correlation matrix
        shelf eval analyze --pairwise           # Pairwise significance
        shelf eval analyze --compare bge_large --compare tfidf
        shelf eval analyze --equivalence        # Models with no sig diff
        shelf eval analyze -v                   # All analyses
    """
    import numpy as np

    if not config.exists():
        console.print(f"[red]Error:[/red] Config file not found: {config}")
        raise typer.Exit(1)

    cfg = load_config(config)
    results_dir = get_results_dir(config)

    if not results_dir.exists():
        console.print(f"[yellow]No results found in {results_dir}[/yellow]")
        raise typer.Exit(0)

    # Load all results
    all_results = _load_all_results(results_dir)
    if not all_results:
        console.print("[yellow]No results found[/yellow]")
        raise typer.Exit(0)

    # Convert to list format for analysis functions
    results_list = list(all_results.values())

    # If verbose, enable all analyses
    if verbose:
        correlation = True
        pairwise = True
        equivalence = True
        champions = True

    # Default: show summary if no specific analysis requested
    show_summary = not any([correlation, pairwise, compare, equivalence, champions])

    # Display header
    dataset_version = cfg.get("dataset_version", "0.3.0")
    console.print()
    console.print(
        Panel(
            f"[bold]SHELF Statistical Analysis[/bold]\n"
            f"Dataset: v{dataset_version} | α={alpha} | Correction: {correction}",
            border_style="cyan",
        )
    )

    export_data: dict = {
        "version": dataset_version,
        "alpha": alpha,
        "correction": correction,
    }

    # === Summary Statistics ===
    if show_summary:
        _display_analysis_summary(results_list, alpha, correction, console, export_data)

    # === Task Correlation Analysis ===
    if correlation:
        _display_correlation_analysis(results_list, console, export_data)

    # === Pairwise Significance Matrix ===
    if pairwise:
        _display_pairwise_significance(
            results_list, alpha, correction, console, export_data
        )

    # === Two-Model Comparison ===
    if compare and len(compare) >= 2:
        _display_model_comparison(
            compare[0], compare[1], results_list, alpha, console, export_data
        )

    # === Equivalence Groups ===
    if equivalence:
        _display_equivalence_groups(results_list, alpha, console, export_data)

    # === Task Champion Analysis ===
    if champions:
        _display_champion_analysis(results_list, console, export_data)

    # Export to JSON if requested
    if export_json:
        import json

        with open(export_json, "w") as f:
            # Convert numpy arrays to lists for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.integer):
                    return int(obj)
                return obj

            json.dump(export_data, f, indent=2, default=convert_numpy)
        console.print(f"\n[green]Analysis exported to {export_json}[/green]")


def _display_analysis_summary(
    results_list: list[dict],
    alpha: float,
    correction: str,
    console: Console,
    export_data: dict,
) -> None:
    """Display summary statistics for peer review."""
    import numpy as np

    from shelf.evaluate.analysis import (
        analyze_task_correlations,
    )
    from shelf.evaluate.analysis.significance import multiple_comparison_correction

    console.print()
    console.print("[bold cyan]═══ Statistical Summary for Peer Review ═══[/bold cyan]")

    # Get model scores per task
    model_scores: dict[str, dict[str, float]] = {}
    for result in results_list:
        model_key = result.get("model_key") or result.get("model")
        task = result.get("task")
        score = result.get("primary_score")
        if model_key and task and score is not None:
            if model_key not in model_scores:
                model_scores[model_key] = {}
            model_scores[model_key][task] = score

    # Aggregate scores per model
    model_means = {}
    for model, task_scores in model_scores.items():
        model_means[model] = float(np.mean(list(task_scores.values())))

    sorted_models = sorted(model_means.items(), key=lambda x: x[1], reverse=True)
    n_models = len(sorted_models)

    # Task correlation analysis
    try:
        corr_result = analyze_task_correlations(results_list)

        console.print()
        console.print(
            Panel(
                f"[bold]Task Independence[/bold]\n"
                f"Mean correlation: r = {corr_result.mean_correlation:.3f}\n"
                f"Median correlation: r = {corr_result.median_correlation:.3f}\n"
                f"Negative correlations: {corr_result.negative_count} "
                f"({corr_result.negative_count / (len(corr_result.task_names) * (len(corr_result.task_names) - 1) // 2) * 100:.0f}%)\n"
                f"Effective sample size: {corr_result.effective_n:.1f} of {len(corr_result.task_names)} tasks\n"
                f"Task families (r > 0.7): {len(corr_result.task_families)}",
                title="Task Correlation Summary",
                border_style="green"
                if corr_result.mean_correlation < 0.3
                else "yellow",
            )
        )

        export_data["task_correlation"] = {
            "mean": corr_result.mean_correlation,
            "median": corr_result.median_correlation,
            "n_negative": corr_result.negative_count,
            "effective_n": corr_result.effective_n,
            "n_task_families": len(corr_result.task_families),
        }
    except Exception as e:
        console.print(f"[yellow]Could not compute task correlation: {e}[/yellow]")

    # Pairwise comparisons summary
    console.print()

    # Find top 5 comparisons by effect size
    key_comparisons = []
    sorted_model_names = [m[0] for m in sorted_models]

    if n_models >= 2:
        # Compare adjacent models in ranking
        for i in range(min(5, n_models - 1)):
            model_a = sorted_model_names[i]
            model_b = sorted_model_names[i + 1]

            tasks_both = set(model_scores[model_a].keys()) & set(
                model_scores[model_b].keys()
            )
            if len(tasks_both) >= 3:
                scores_a = [model_scores[model_a][t] for t in tasks_both]
                scores_b = [model_scores[model_b][t] for t in tasks_both]

                from shelf.evaluate.analysis.significance import cohens_d, wilcoxon_test

                d, d_interp = cohens_d(scores_a, scores_b)
                wilcox = wilcoxon_test(scores_a, scores_b, alpha=alpha)

                key_comparisons.append(
                    {
                        "model_a": model_a,
                        "model_b": model_b,
                        "mean_a": float(np.mean(scores_a)),
                        "mean_b": float(np.mean(scores_b)),
                        "diff": float(np.mean(scores_a)) - float(np.mean(scores_b)),
                        "cohens_d": d,
                        "d_interpretation": d_interp,
                        "p_value": wilcox.p_value,
                        "significant": wilcox.significant,
                    }
                )

    if key_comparisons:
        # Apply multiple comparison correction
        p_values = [c["p_value"] for c in key_comparisons]
        reject, corrected_p = multiple_comparison_correction(
            p_values, method=correction, alpha=alpha
        )

        for i, comp in enumerate(key_comparisons):
            comp["p_corrected"] = float(corrected_p[i])
            comp["significant_corrected"] = bool(reject[i])

        sig_table = Table(
            title=f"Key Model Comparisons (α={alpha}, {correction} correction)",
            show_header=True,
            header_style="bold cyan",
        )
        sig_table.add_column("Comparison", style="cyan")
        sig_table.add_column("Δ Score", justify="right")
        sig_table.add_column("Cohen's d", justify="right")
        sig_table.add_column("p-value", justify="right")
        sig_table.add_column("p-corrected", justify="right")
        sig_table.add_column("Sig?", justify="center")

        for comp in key_comparisons:
            d_style = (
                "green"
                if abs(comp["cohens_d"]) >= 0.8
                else ("yellow" if abs(comp["cohens_d"]) >= 0.5 else "")
            )
            p_style = "green" if comp["significant_corrected"] else ""
            sig_str = "[green]✓[/green]" if comp["significant_corrected"] else ""

            sig_table.add_row(
                f"{comp['model_a']} vs {comp['model_b']}",
                f"{comp['diff']:+.4f}",
                f"[{d_style}]{comp['cohens_d']:.2f} ({comp['d_interpretation']})[/{d_style}]"
                if d_style
                else f"{comp['cohens_d']:.2f} ({comp['d_interpretation']})",
                f"{comp['p_value']:.4f}",
                f"[{p_style}]{comp['p_corrected']:.4f}[/{p_style}]"
                if p_style
                else f"{comp['p_corrected']:.4f}",
                sig_str,
            )

        console.print(sig_table)
        export_data["key_comparisons"] = key_comparisons

    # Summary statistics
    console.print()
    summary_panel = Panel(
        f"[bold]Key Takeaways[/bold]\n\n"
        f"• Models evaluated: {n_models}\n"
        f"• Significant differences (after {correction} correction): "
        f"{sum(1 for c in key_comparisons if c.get('significant_corrected', False))}/{len(key_comparisons)}\n"
        f"• Large effects (|d| ≥ 0.8): {sum(1 for c in key_comparisons if abs(c['cohens_d']) >= 0.8)}\n"
        f"• Recommendation: Report effect sizes alongside p-values",
        title="Reviewer Summary",
        border_style="cyan",
    )
    console.print(summary_panel)


def _display_correlation_analysis(
    results_list: list[dict],
    console: Console,
    export_data: dict,
) -> None:
    """Display task correlation matrix and analysis."""

    from shelf.evaluate.analysis import analyze_task_correlations

    console.print()
    console.print("[bold cyan]═══ Task Correlation Analysis ═══[/bold cyan]")

    try:
        corr_result = analyze_task_correlations(results_list)

        # Display correlation matrix as table
        n_tasks = len(corr_result.task_names)

        if n_tasks <= 12:
            # Show full matrix for small number of tasks
            corr_table = Table(
                title="Task Correlation Matrix",
                show_header=True,
                header_style="bold",
            )
            corr_table.add_column("Task", style="cyan")
            for task in corr_result.task_names:
                short_name = task[:8] + ".." if len(task) > 10 else task
                corr_table.add_column(short_name, justify="right", width=7)

            for i, task_i in enumerate(corr_result.task_names):
                row_values = [task_i[:15] + ".." if len(task_i) > 17 else task_i]
                for j in range(n_tasks):
                    r = corr_result.correlation_matrix[i, j]
                    if i == j:
                        row_values.append("[dim]1.00[/dim]")
                    elif abs(r) > 0.7:
                        row_values.append(f"[red]{r:.2f}[/red]")
                    elif abs(r) > 0.5:
                        row_values.append(f"[yellow]{r:.2f}[/yellow]")
                    elif r < -0.5:
                        row_values.append(f"[blue]{r:.2f}[/blue]")
                    else:
                        row_values.append(f"{r:.2f}")
                corr_table.add_row(*row_values)

            console.print(corr_table)
        else:
            console.print(
                f"[dim]({n_tasks} tasks - matrix too large to display, showing statistics)[/dim]"
            )

        # Statistics table
        console.print()
        stats_table = Table(
            title="Correlation Statistics", show_header=True, header_style="bold cyan"
        )
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", justify="right")
        stats_table.add_column("Interpretation")

        stats_table.add_row(
            "Mean r",
            f"{corr_result.mean_correlation:.3f}",
            "[green]Low[/green]"
            if corr_result.mean_correlation < 0.3
            else "[yellow]Moderate[/yellow]"
            if corr_result.mean_correlation < 0.5
            else "[red]High[/red]",
        )
        stats_table.add_row("Median r", f"{corr_result.median_correlation:.3f}", "")
        stats_table.add_row(
            "Range",
            f"[{corr_result.min_correlation:.3f}, {corr_result.max_correlation:.3f}]",
            "",
        )
        stats_table.add_row(
            "Negative pairs",
            f"{corr_result.negative_count}",
            "Shows tasks measure opposing capabilities",
        )
        stats_table.add_row(
            "High |r| > 0.7",
            f"{len(corr_result.high_correlation_pairs)}",
            "Potentially redundant task pairs",
        )
        stats_table.add_row(
            "Effective N",
            f"{corr_result.effective_n:.1f}",
            f"of {n_tasks} nominal tasks",
        )

        console.print(stats_table)

        # High correlation pairs
        if corr_result.high_correlation_pairs:
            console.print()
            high_table = Table(
                title="Highly Correlated Task Pairs (|r| > 0.7)",
                show_header=True,
                header_style="bold yellow",
            )
            high_table.add_column("Task 1", style="cyan")
            high_table.add_column("Task 2", style="cyan")
            high_table.add_column("r", justify="right")
            high_table.add_column("Note")

            for t1, t2, r in corr_result.high_correlation_pairs[:10]:
                note = "Strong positive" if r > 0 else "Strong negative (opposing)"
                high_table.add_row(t1, t2, f"{r:.3f}", note)

            console.print(high_table)

        # Task families
        if corr_result.task_families:
            console.print()
            console.print("[bold]Task Families (groups with r > 0.7):[/bold]")
            for i, family in enumerate(corr_result.task_families, 1):
                console.print(f"  {i}. {', '.join(family)}")

        export_data["correlation"] = {
            "matrix": corr_result.correlation_matrix.tolist(),
            "task_names": corr_result.task_names,
            "mean": corr_result.mean_correlation,
            "median": corr_result.median_correlation,
            "effective_n": corr_result.effective_n,
            "task_families": corr_result.task_families,
        }

    except Exception as e:
        console.print(f"[red]Error computing correlation: {e}[/red]")


def _display_pairwise_significance(
    results_list: list[dict],
    alpha: float,
    correction: str,
    console: Console,
    export_data: dict,
) -> None:
    """Display pairwise significance matrix with effect sizes."""
    import numpy as np

    from shelf.evaluate.analysis.significance import (
        cohens_d,
        multiple_comparison_correction,
        wilcoxon_test,
    )

    console.print()
    console.print("[bold cyan]═══ Pairwise Significance Analysis ═══[/bold cyan]")

    # Get model scores
    model_scores: dict[str, dict[str, float]] = {}
    for result in results_list:
        model_key = result.get("model_key") or result.get("model")
        task = result.get("task")
        score = result.get("primary_score")
        if model_key and task and score is not None:
            if model_key not in model_scores:
                model_scores[model_key] = {}
            model_scores[model_key][task] = score

    # Sort models by mean score
    model_means = {m: float(np.mean(list(s.values()))) for m, s in model_scores.items()}
    sorted_models = sorted(
        model_means.keys(), key=lambda x: model_means[x], reverse=True
    )

    # Limit to top N models for readability
    top_n = min(12, len(sorted_models))
    top_models = sorted_models[:top_n]

    # Compute all pairwise comparisons
    comparisons = []
    for i, model_a in enumerate(top_models):
        for model_b in top_models[i + 1 :]:
            tasks_both = set(model_scores[model_a].keys()) & set(
                model_scores[model_b].keys()
            )
            if len(tasks_both) < 3:
                continue

            scores_a = [model_scores[model_a][t] for t in tasks_both]
            scores_b = [model_scores[model_b][t] for t in tasks_both]

            d, d_interp = cohens_d(scores_a, scores_b)
            wilcox = wilcoxon_test(scores_a, scores_b, alpha=alpha)

            comparisons.append(
                {
                    "model_a": model_a,
                    "model_b": model_b,
                    "cohens_d": d,
                    "d_interpretation": d_interp,
                    "p_value": wilcox.p_value,
                    "mean_diff": float(np.mean(scores_a)) - float(np.mean(scores_b)),
                }
            )

    if not comparisons:
        console.print("[yellow]Insufficient data for pairwise comparisons[/yellow]")
        return

    # Apply multiple comparison correction
    p_values = [c["p_value"] for c in comparisons]
    reject, corrected_p = multiple_comparison_correction(
        p_values, method=correction, alpha=alpha
    )

    for i, comp in enumerate(comparisons):
        comp["p_corrected"] = float(corrected_p[i])
        comp["significant_corrected"] = bool(reject[i])

    # Display as table
    pw_table = Table(
        title=f"Pairwise Comparisons (top {top_n} models, {correction} correction)",
        show_header=True,
        header_style="bold cyan",
    )
    pw_table.add_column("Model A", style="cyan")
    pw_table.add_column("Model B", style="cyan")
    pw_table.add_column("Δ Score", justify="right")
    pw_table.add_column("Cohen's d", justify="right")
    pw_table.add_column("p (raw)", justify="right")
    pw_table.add_column("p (corr)", justify="right")
    pw_table.add_column("Sig?", justify="center")

    # Sort by effect size magnitude
    comparisons.sort(key=lambda x: abs(x["cohens_d"]), reverse=True)

    for comp in comparisons[:20]:  # Show top 20 by effect size
        d = comp["cohens_d"]
        d_style = "green" if abs(d) >= 0.8 else ("yellow" if abs(d) >= 0.5 else "dim")
        sig_str = "[green]✓[/green]" if comp["significant_corrected"] else ""

        pw_table.add_row(
            comp["model_a"],
            comp["model_b"],
            f"{comp['mean_diff']:+.4f}",
            f"[{d_style}]{d:.2f}[/{d_style}]",
            f"{comp['p_value']:.4f}",
            f"{comp['p_corrected']:.4f}",
            sig_str,
        )

    console.print(pw_table)

    # Summary
    n_significant = sum(1 for c in comparisons if c["significant_corrected"])
    n_large_effect = sum(1 for c in comparisons if abs(c["cohens_d"]) >= 0.8)

    console.print()
    console.print(
        f"[bold]Summary:[/bold] {n_significant}/{len(comparisons)} pairs significant after {correction} correction"
    )
    console.print(
        f"         {n_large_effect}/{len(comparisons)} pairs have large effect size (|d| ≥ 0.8)"
    )

    export_data["pairwise"] = comparisons


def _display_model_comparison(
    model_a: str,
    model_b: str,
    results_list: list[dict],
    alpha: float,
    console: Console,
    export_data: dict,
) -> None:
    """Display detailed comparison of two specific models."""
    import numpy as np

    from shelf.evaluate.analysis.significance import (
        cohens_d,
        paired_t_test,
        wilcoxon_test,
    )

    console.print()
    console.print(
        f"[bold cyan]═══ Detailed Comparison: {model_a} vs {model_b} ═══[/bold cyan]"
    )

    # Get scores for both models
    scores_a: dict[str, float] = {}
    scores_b: dict[str, float] = {}

    for result in results_list:
        model_key = result.get("model_key") or result.get("model")
        task = result.get("task")
        score = result.get("primary_score")

        if model_key == model_a and task and score is not None:
            scores_a[task] = score
        elif model_key == model_b and task and score is not None:
            scores_b[task] = score

    if not scores_a:
        console.print(f"[red]Model '{model_a}' not found in results[/red]")
        return
    if not scores_b:
        console.print(f"[red]Model '{model_b}' not found in results[/red]")
        return

    # Common tasks
    common_tasks = sorted(set(scores_a.keys()) & set(scores_b.keys()))
    if len(common_tasks) < 3:
        console.print(
            f"[red]Need at least 3 common tasks, found {len(common_tasks)}[/red]"
        )
        return

    arr_a = [scores_a[t] for t in common_tasks]
    arr_b = [scores_b[t] for t in common_tasks]

    # Overall statistics
    mean_a = float(np.mean(arr_a))
    mean_b = float(np.mean(arr_b))
    diff = mean_a - mean_b

    d, d_interp = cohens_d(arr_a, arr_b)
    t_test = paired_t_test(arr_a, arr_b, alpha=alpha)
    wilcox = wilcoxon_test(arr_a, arr_b, alpha=alpha)

    # Summary panel
    winner = model_a if diff > 0 else model_b
    winner_str = f"[green]{winner}[/green]" if abs(diff) > 0.01 else "[dim]Tie[/dim]"

    console.print(
        Panel(
            f"[bold]{model_a}[/bold]: mean = {mean_a:.4f}\n"
            f"[bold]{model_b}[/bold]: mean = {mean_b:.4f}\n"
            f"\n"
            f"Difference: {diff:+.4f} ({diff / mean_b * 100:+.1f}%)\n"
            f"Cohen's d: {d:.3f} ({d_interp})\n"
            f"\n"
            f"Paired t-test: p = {t_test.p_value:.4f}\n"
            f"Wilcoxon: p = {wilcox.p_value:.4f}\n"
            f"95% CI: [{t_test.confidence_interval[0]:.4f}, {t_test.confidence_interval[1]:.4f}]\n"
            f"\n"
            f"Winner: {winner_str}",
            title="Overall Comparison",
            border_style="cyan",
        )
    )

    # Per-task breakdown
    console.print()
    task_table = Table(
        title="Per-Task Breakdown",
        show_header=True,
        header_style="bold cyan",
    )
    task_table.add_column("Task", style="cyan")
    task_table.add_column(model_a, justify="right")
    task_table.add_column(model_b, justify="right")
    task_table.add_column("Δ", justify="right")
    task_table.add_column("Winner", justify="center")

    wins_a = 0
    wins_b = 0

    for task in common_tasks:
        sa = scores_a[task]
        sb = scores_b[task]
        task_diff = sa - sb

        if task_diff > 0.001:
            wins_a += 1
            winner_cell = f"[green]{model_a}[/green]"
        elif task_diff < -0.001:
            wins_b += 1
            winner_cell = f"[yellow]{model_b}[/yellow]"
        else:
            winner_cell = "[dim]Tie[/dim]"

        diff_style = "green" if task_diff > 0 else ("red" if task_diff < 0 else "dim")

        task_table.add_row(
            task[:30] + ".." if len(task) > 32 else task,
            f"{sa:.4f}",
            f"{sb:.4f}",
            f"[{diff_style}]{task_diff:+.4f}[/{diff_style}]",
            winner_cell,
        )

    console.print(task_table)

    console.print()
    console.print(
        f"[bold]Win counts:[/bold] {model_a}: {wins_a}, {model_b}: {wins_b}, Ties: {len(common_tasks) - wins_a - wins_b}"
    )

    export_data["comparison"] = {
        "model_a": model_a,
        "model_b": model_b,
        "mean_a": mean_a,
        "mean_b": mean_b,
        "difference": diff,
        "cohens_d": d,
        "p_t_test": t_test.p_value,
        "p_wilcoxon": wilcox.p_value,
        "wins_a": wins_a,
        "wins_b": wins_b,
    }


def _display_equivalence_groups(
    results_list: list[dict],
    alpha: float,
    console: Console,
    export_data: dict,
) -> None:
    """Display groups of statistically equivalent models."""

    from shelf.evaluate.analysis import compare_multiple

    console.print()
    console.print("[bold cyan]═══ Equivalence Groups ═══[/bold cyan]")
    console.print("[dim](Models NOT significantly different from each other)[/dim]")

    # Get model scores
    model_scores: dict[str, list[float]] = {}
    for result in results_list:
        model_key = result.get("model_key") or result.get("model")
        score = result.get("primary_score")
        if model_key and score is not None:
            if model_key not in model_scores:
                model_scores[model_key] = []
            model_scores[model_key].append(score)

    if len(model_scores) < 3:
        console.print(
            "[yellow]Need at least 3 models for equivalence analysis[/yellow]"
        )
        return

    # Ensure all models have same number of scores
    min_len = min(len(v) for v in model_scores.values())
    aligned_scores = {k: v[:min_len] for k, v in model_scores.items()}

    try:
        result = compare_multiple(aligned_scores, alpha=alpha)

        # Display ranking with equivalence group annotations
        console.print()
        rank_table = Table(
            title="Rankings with Equivalence Groups",
            show_header=True,
            header_style="bold cyan",
        )
        rank_table.add_column("Rank", justify="right", style="dim")
        rank_table.add_column("Model", style="cyan")
        rank_table.add_column("Mean Score", justify="right")
        rank_table.add_column("Mean Rank", justify="right")
        rank_table.add_column("Equiv. Group")

        # Assign group colors
        group_colors = ["green", "yellow", "blue", "magenta", "red", "cyan"]
        model_to_group: dict[str, int] = {}
        for i, group in enumerate(result.equivalence_groups):
            for model in group:
                model_to_group[model] = i

        for rank, model in enumerate(result.ranking, 1):
            mean_score = result.mean_scores[model]
            mean_rank = result.friedman_nemenyi.mean_ranks[model]

            group_idx = model_to_group.get(model)
            if group_idx is not None:
                color = group_colors[group_idx % len(group_colors)]
                group_str = f"[{color}]Group {group_idx + 1}[/{color}]"
            else:
                group_str = "[dim]Unique[/dim]"

            rank_table.add_row(
                str(rank),
                model,
                f"{mean_score:.4f}",
                f"{mean_rank:.2f}",
                group_str,
            )

        console.print(rank_table)

        # Display equivalence groups explicitly
        if result.equivalence_groups:
            console.print()
            console.print(
                "[bold]Equivalence Groups (models with no significant difference):[/bold]"
            )
            for i, group in enumerate(result.equivalence_groups, 1):
                color = group_colors[(i - 1) % len(group_colors)]
                members = ", ".join(group)
                console.print(f"  [{color}]Group {i}:[/{color}] {members}")
        else:
            console.print()
            console.print(
                "[green]All models are significantly different from each other[/green]"
            )

        # Friedman test summary
        console.print()
        console.print(
            f"[bold]Friedman test:[/bold] χ² = {result.friedman_nemenyi.friedman_statistic:.2f}, "
            f"p = {result.friedman_nemenyi.friedman_p_value:.4f}"
        )
        console.print(
            f"[bold]Critical difference:[/bold] {result.friedman_nemenyi.critical_difference:.3f}"
        )
        console.print(
            f"[bold]Significant pairs:[/bold] {len(result.friedman_nemenyi.significant_pairs)} of {len(result.models) * (len(result.models) - 1) // 2}"
        )

        export_data["equivalence"] = {
            "groups": result.equivalence_groups,
            "ranking": result.ranking,
            "friedman_p": result.friedman_nemenyi.friedman_p_value,
            "critical_difference": result.friedman_nemenyi.critical_difference,
        }

    except Exception as e:
        console.print(f"[red]Error computing equivalence: {e}[/red]")


def _display_champion_analysis(
    results_list: list[dict],
    console: Console,
    export_data: dict,
) -> None:
    """Display task champion diversity analysis."""
    from shelf.evaluate.analysis import (
        compute_rank_consistency,
        compute_task_champion_diversity,
    )

    console.print()
    console.print("[bold cyan]═══ Task Champion Analysis ═══[/bold cyan]")
    console.print("[dim](Do different models win different tasks?)[/dim]")

    # Get model scores
    model_scores: dict[str, dict[str, float]] = {}
    for result in results_list:
        model_key = result.get("model_key") or result.get("model")
        task = result.get("task")
        score = result.get("primary_score")
        if model_key and task and score is not None:
            if model_key not in model_scores:
                model_scores[model_key] = {}
            model_scores[model_key][task] = score

    # Champion diversity
    champ_result = compute_task_champion_diversity(model_scores)

    console.print()
    console.print(
        Panel(
            f"[bold]Champion Diversity[/bold]\n"
            f"Total tasks: {champ_result['n_tasks']}\n"
            f"Unique champions: {champ_result['n_unique_champions']}\n"
            f"Diversity ratio: {champ_result['diversity_ratio']:.2f}\n"
            f"\n"
            f"[green]High diversity indicates tasks measure different capabilities[/green]",
            title="Summary",
            border_style="green" if champ_result["diversity_ratio"] > 0.3 else "yellow",
        )
    )

    # Champion distribution
    console.print()
    dist_table = Table(
        title="Champion Distribution (wins per model)",
        show_header=True,
        header_style="bold cyan",
    )
    dist_table.add_column("Model", style="cyan")
    dist_table.add_column("Wins", justify="right", style="green")
    dist_table.add_column("% of Tasks", justify="right")

    sorted_champs = sorted(
        champ_result["champion_distribution"].items(), key=lambda x: x[1], reverse=True
    )

    for model, wins in sorted_champs:
        pct = wins / champ_result["n_tasks"] * 100
        dist_table.add_row(model, str(wins), f"{pct:.1f}%")

    console.print(dist_table)

    # Per-task champions
    console.print()
    task_table = Table(
        title="Task Champions",
        show_header=True,
        header_style="bold cyan",
    )
    task_table.add_column("Task", style="cyan")
    task_table.add_column("Champion", style="green")

    for task, model in sorted(champ_result["champions"].items()):
        task_table.add_row(task[:40] + ".." if len(task) > 42 else task, model)

    console.print(task_table)

    # Rank consistency
    console.print()
    rank_result = compute_rank_consistency(model_scores)

    rank_table = Table(
        title="Rank Consistency (how stable is each model's ranking?)",
        show_header=True,
        header_style="bold cyan",
    )
    rank_table.add_column("Model", style="cyan")
    rank_table.add_column("Mean Rank", justify="right")
    rank_table.add_column("Std", justify="right")
    rank_table.add_column("Range", justify="right")
    rank_table.add_column("Top-3", justify="right", style="green")

    sorted_rank = sorted(rank_result.items(), key=lambda x: x[1]["mean_rank"])

    for model, stats in sorted_rank[:15]:
        rank_table.add_row(
            model,
            f"{stats['mean_rank']:.1f}",
            f"{stats['std_rank']:.1f}",
            f"{stats['min_rank']}-{stats['max_rank']}",
            str(stats["n_top3"]),
        )

    console.print(rank_table)

    export_data["champions"] = {
        "diversity_ratio": champ_result["diversity_ratio"],
        "n_unique_champions": champ_result["n_unique_champions"],
        "distribution": champ_result["champion_distribution"],
        "per_task": champ_result["champions"],
    }
