"""Model management CLI commands.

Commands for listing, adding, and removing models from the benchmark config.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Optional

import typer
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

app = typer.Typer(
    name="models",
    help="Manage embedding models for SHELF benchmark",
)
console = Console()

# Default config path
DEFAULT_CONFIG_PATH = (
    Path(__file__).parent.parent.parent.parent / "scripts" / "baselines" / "config.yaml"
)


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def save_config(config: dict, config_path: Path) -> None:
    """Save configuration to YAML file."""
    with open(config_path, "w") as f:
        yaml.dump(
            config, f, default_flow_style=False, sort_keys=False, allow_unicode=True
        )


def format_params(num_params: int | None) -> str:
    """Format parameter count as human-readable string."""
    if num_params is None:
        return "-"
    if num_params >= 1_000_000_000:
        return f"{num_params / 1_000_000_000:.1f}B"
    if num_params >= 1_000_000:
        return f"{num_params / 1_000_000:.1f}M"
    if num_params >= 1_000:
        return f"{num_params / 1_000:.1f}K"
    return str(num_params)


def get_model_info_from_hf(model_id: str) -> dict | None:
    """Fetch model info from HuggingFace Hub.

    Returns dict with num_params, embedding_dim, hidden_size, and context_window,
    or None if fetch fails.
    """
    try:
        from huggingface_hub import HfApi
        from transformers import AutoConfig

        # Try to get config
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

        # Get hidden size
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(config, "d_model", None)

        # For sentence transformers, embedding_dim might differ from hidden_size
        # (e.g., if there's a pooling layer that changes dimensions)
        # Default to hidden_size for now
        embedding_dim = hidden_size

        # Get context window (max position embeddings)
        context_window = getattr(config, "max_position_embeddings", None)

        # Try to count parameters
        num_params = None
        try:
            # Check if model card has parameter count
            api = HfApi()
            model_info = api.model_info(model_id)
            if model_info.safetensors:
                # Get total params from safetensors metadata
                num_params = model_info.safetensors.get("total", None)

            if num_params is None and hasattr(model_info, "num_parameters"):
                num_params = model_info.num_parameters
        except Exception:
            pass

        # If we still don't have params, estimate from config
        if num_params is None and hidden_size:
            # Rough estimate: hidden_size^2 * num_layers * 4 (for BERT-like)
            num_layers = getattr(config, "num_hidden_layers", 12)
            vocab_size = getattr(config, "vocab_size", 30522)
            # Very rough estimate
            num_params = (
                hidden_size * hidden_size * num_layers * 4 + vocab_size * hidden_size
            )

        return {
            "num_params": num_params,
            "embedding_dim": embedding_dim,
            "hidden_size": hidden_size,
            "context_window": context_window,
        }
    except Exception as e:
        console.print(f"[yellow]Warning:[/yellow] Could not fetch model info: {e}")
        return None


def get_size_category(num_params: int) -> str:
    """Determine size category from parameter count.

    Returns explicit size ranges for clarity:
    - <10M: Very small models (e.g., distilled, micro)
    - 10M-50M: Small models (e.g., MiniLM, small variants)
    - 50M-100M: Base-small models
    - 100M-300M: Base models (e.g., BERT-base, 110M models)
    - 300M-1B: Large models (e.g., BERT-large, 335M models)
    - >1B: Very large models
    """
    if num_params < 10_000_000:
        return "<10M"
    elif num_params < 50_000_000:
        return "10M-50M"
    elif num_params < 100_000_000:
        return "50M-100M"
    elif num_params < 300_000_000:
        return "100M-300M"
    elif num_params < 1_000_000_000:
        return "300M-1B"
    else:
        return ">1B"


def generate_model_key(model_id: str) -> str:
    """Generate a model key from HuggingFace model ID."""
    # Take last part of path
    name = model_id.split("/")[-1]
    # Convert to snake_case
    name = re.sub(r"[-.]", "_", name)
    name = re.sub(r"[^a-zA-Z0-9_]", "", name)
    name = name.lower()
    # Shorten common suffixes
    name = re.sub(r"_v\d+(_\d+)?$", "", name)
    name = re.sub(r"_en$", "", name)
    return name


@app.command("list")
def cmd_list(
    config: Annotated[
        Path, typer.Option("--config", "-c", help="Path to config file")
    ] = DEFAULT_CONFIG_PATH,
    category: Annotated[
        Optional[str], typer.Option("--category", help="Filter by size category")
    ] = None,
):
    """List all configured models."""
    if not config.exists():
        console.print(f"[red]Error:[/red] Config file not found: {config}")
        raise typer.Exit(1)

    cfg = load_config(config)
    models = cfg.get("models", {})

    # Create table
    table = Table(
        title="Configured Models",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Key", style="cyan")
    table.add_column("Name", style="white")
    table.add_column("HuggingFace Model", style="dim")
    table.add_column("Params", justify="right", style="green")
    table.add_column("Dim", justify="right", style="yellow")
    table.add_column("Hidden", justify="right", style="dim")
    table.add_column("Context", justify="right", style="dim")
    table.add_column("Category", style="magenta")

    dense_count = 0
    sparse_count = 0

    for key, model in models.items():
        model_type = model.get("type", "")

        # Filter by category if specified
        if category:
            model_category = model.get(
                "size_category",
                "sparse" if model_type in ["tf", "tfidf", "bm25"] else "base",
            )
            if model_category != category:
                continue

        if model_type == "sentence_transformer":
            dense_count += 1
            num_params = model.get("num_params")
            embedding_dim = model.get("embedding_dim")
            hidden_size = model.get("hidden_size")
            context_window = model.get("context_window")
            size_category = model.get("size_category", "base")
            hf_model = model.get("model_name", "-")

            table.add_row(
                key,
                model.get("name", key),
                hf_model[:35] + "..." if len(hf_model) > 38 else hf_model,
                format_params(num_params),
                str(embedding_dim) if embedding_dim else "-",
                str(hidden_size) if hidden_size else "-",
                str(context_window) if context_window else "-",
                size_category,
            )
        else:
            sparse_count += 1
            params = model.get("params", {})
            embedding_dim = params.get("embedding_dim")

            table.add_row(
                key,
                model.get("name", key),
                f"[dim]{model_type}[/dim]",
                "-",
                str(embedding_dim) if embedding_dim else "-",
                "-",
                "-",
                "sparse",
            )

    console.print(table)
    console.print()
    console.print(
        f"[cyan]{dense_count}[/cyan] dense models, [cyan]{sparse_count}[/cyan] sparse models"
    )


@app.command("info")
def cmd_info(
    model_key: Annotated[str, typer.Argument(help="Model key to show info for")],
    config: Annotated[
        Path, typer.Option("--config", "-c", help="Path to config file")
    ] = DEFAULT_CONFIG_PATH,
):
    """Show detailed information about a model."""
    if not config.exists():
        console.print(f"[red]Error:[/red] Config file not found: {config}")
        raise typer.Exit(1)

    cfg = load_config(config)
    models = cfg.get("models", {})

    if model_key not in models:
        console.print(f"[red]Error:[/red] Model '{model_key}' not found")
        console.print("Use [cyan]shelf models list[/cyan] to see available models")
        raise typer.Exit(1)

    model = models[model_key]
    model_type = model.get("type", "")

    # Create info panel
    info_lines = []
    info_lines.append(f"[bold]Key:[/bold] {model_key}")
    info_lines.append(f"[bold]Name:[/bold] {model.get('name', model_key)}")
    info_lines.append(f"[bold]Type:[/bold] {model_type}")
    info_lines.append(f"[bold]Description:[/bold] {model.get('description', '-')}")

    if model_type == "sentence_transformer":
        info_lines.append(
            f"[bold]HuggingFace Model:[/bold] {model.get('model_name', '-')}"
        )
        info_lines.append(
            f"[bold]Parameters:[/bold] {format_params(model.get('num_params'))}"
        )
        info_lines.append(
            f"[bold]Embedding Dim:[/bold] {model.get('embedding_dim', '-')}"
        )
        info_lines.append(f"[bold]Hidden Size:[/bold] {model.get('hidden_size', '-')}")
        info_lines.append(
            f"[bold]Context Window:[/bold] {model.get('context_window', '-')}"
        )
        info_lines.append(
            f"[bold]Size Category:[/bold] {model.get('size_category', '-')}"
        )
    else:
        params = model.get("params", {})
        for k, v in params.items():
            info_lines.append(f"[bold]{k}:[/bold] {v}")

    info_lines.append(f"[bold]Supports:[/bold] {', '.join(model.get('supports', []))}")

    panel = Panel(
        "\n".join(info_lines),
        title=f"[bold cyan]{model.get('name', model_key)}[/bold cyan]",
        border_style="cyan",
    )
    console.print(panel)


@app.command("add")
def cmd_add(
    model_id: Annotated[
        str, typer.Argument(help="HuggingFace model ID (e.g., BAAI/bge-large-en-v1.5)")
    ],
    config: Annotated[
        Path, typer.Option("--config", "-c", help="Path to config file")
    ] = DEFAULT_CONFIG_PATH,
    key: Annotated[
        Optional[str],
        typer.Option("--key", "-k", help="Model key (auto-generated if not provided)"),
    ] = None,
    name: Annotated[
        Optional[str], typer.Option("--name", "-n", help="Display name")
    ] = None,
    no_fetch: Annotated[
        bool, typer.Option("--no-fetch", help="Don't fetch info from HuggingFace")
    ] = False,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip all prompts (non-interactive mode)"),
    ] = False,
):
    """Add a new sentence-transformer model from HuggingFace."""
    if not config.exists():
        console.print(f"[red]Error:[/red] Config file not found: {config}")
        raise typer.Exit(1)

    cfg = load_config(config)
    models = cfg.get("models", {})

    # Generate or validate key
    model_key = key or generate_model_key(model_id)

    if model_key in models:
        console.print(f"[red]Error:[/red] Model key '{model_key}' already exists")
        console.print("Use [cyan]--key[/cyan] to specify a different key")
        raise typer.Exit(1)

    # Fetch model info from HuggingFace
    num_params = None
    embedding_dim = None
    hidden_size = None
    context_window = None

    if not no_fetch:
        with console.status("Fetching model info from HuggingFace..."):
            info = get_model_info_from_hf(model_id)
            if info:
                num_params = info.get("num_params")
                embedding_dim = info.get("embedding_dim")
                hidden_size = info.get("hidden_size")
                context_window = info.get("context_window")

    # Show what we found
    console.print()
    console.print(f"[bold green]✓[/bold green] Model: [cyan]{model_id}[/cyan]")
    if num_params:
        console.print(f"  Parameters: {num_params:,} ({format_params(num_params)})")
    if embedding_dim:
        console.print(f"  Embedding dim: {embedding_dim}")
    if hidden_size:
        console.print(f"  Hidden size: {hidden_size}")
    if context_window:
        console.print(f"  Context window: {context_window}")

    # Prompt for missing values (skip if --yes)
    if num_params is None and not yes:
        num_params_str = Prompt.ask("  Parameters (e.g., 110000000)", default="")
        if num_params_str:
            num_params = int(
                num_params_str.replace(",", "")
                .replace("M", "000000")
                .replace("B", "000000000")
            )

    if embedding_dim is None and not yes:
        embedding_dim_str = Prompt.ask("  Embedding dim (e.g., 768)", default="768")
        embedding_dim = int(embedding_dim_str)
    elif embedding_dim is None:
        embedding_dim = 768  # Default for non-interactive

    # Determine size category
    size_category = get_size_category(num_params) if num_params else "base"
    console.print(f"  Size category: {size_category}")

    # Prompt for key and name (skip if --yes)
    if not yes:
        model_key = Prompt.ask("  Model key", default=model_key)
        display_name = name or Prompt.ask(
            "  Display name", default=model_id.split("/")[-1]
        )
    else:
        display_name = name or model_id.split("/")[-1]

    # Confirm (skip if --yes)
    if not yes:
        console.print()
        if not Confirm.ask(f"Add model '{model_key}'?"):
            console.print("[yellow]Cancelled[/yellow]")
            raise typer.Exit(0)

    # Create model config
    model_config = {
        "type": "sentence_transformer",
        "name": display_name,
        "description": f"{model_id} ({format_params(num_params)} params, {embedding_dim} dims)",
        "model_name": model_id,
        "num_params": num_params,
        "embedding_dim": embedding_dim,
        "hidden_size": hidden_size,
        "context_window": context_window,
        "size_category": size_category,
        "supports": [
            "retrieval",
            "classification",
            "clustering",
            "pair_classification",
        ],
    }

    # Add to config
    models[model_key] = model_config
    cfg["models"] = models

    # Save
    save_config(cfg, config)

    console.print()
    console.print(
        f"[bold green]✓[/bold green] Added [cyan]{model_key}[/cyan] to {config.name}"
    )


@app.command("remove")
def cmd_remove(
    model_key: Annotated[str, typer.Argument(help="Model key to remove")],
    config: Annotated[
        Path, typer.Option("--config", "-c", help="Path to config file")
    ] = DEFAULT_CONFIG_PATH,
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Skip confirmation")
    ] = False,
):
    """Remove a model from the config."""
    if not config.exists():
        console.print(f"[red]Error:[/red] Config file not found: {config}")
        raise typer.Exit(1)

    cfg = load_config(config)
    models = cfg.get("models", {})

    if model_key not in models:
        console.print(f"[red]Error:[/red] Model '{model_key}' not found")
        raise typer.Exit(1)

    model = models[model_key]
    model_name = model.get("name", model_key)

    # Confirm
    if not force:
        if not Confirm.ask(f"Remove model '[cyan]{model_key}[/cyan]' ({model_name})?"):
            console.print("[yellow]Cancelled[/yellow]")
            raise typer.Exit(0)

    # Remove
    del models[model_key]
    cfg["models"] = models

    # Also remove from model_groups if present
    model_groups = cfg.get("model_groups", {})
    for group_name, group in model_groups.items():
        group_models = group.get("models", [])
        if model_key in group_models:
            group_models.remove(model_key)
            group["models"] = group_models
    cfg["model_groups"] = model_groups

    # Save
    save_config(cfg, config)

    console.print(
        f"[bold green]✓[/bold green] Removed [cyan]{model_key}[/cyan] from {config.name}"
    )
