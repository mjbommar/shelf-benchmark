"""
CLI driver for SHELF taxonomy tools.

Uses Typer for CLI and Rich for console output.
"""

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Annotated, Optional

import orjson
import typer
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)
from rich.table import Table

from .taxonomies import TaxonomyType, Taxonomy
from .taxonomies.loaders import load_taxonomy_from_frequencies
from .benchmark import (
    LCBenchmarkGenerator,
    LCGeneratorConfig,
    get_taxonomy_stats,
    LCC_NAMES,
    LCGFT_CHILDREN,
)

app = typer.Typer(
    name="shelf",
    help="SHELF: Synthetic Harness for Evaluating LLM Fitness",
    add_completion=False,
)
console = Console()

# Subcommand for benchmark generation
gen_app = typer.Typer(help="Generate synthetic benchmark documents")
app.add_typer(gen_app, name="gen")

# Import and add model management and evaluation subcommands
from shelf.cli_cmds.models import app as models_app
from shelf.cli_cmds.eval import app as eval_app

app.add_typer(models_app, name="models")
app.add_typer(eval_app, name="eval")


def get_default_paths() -> tuple[Path, Path, Path]:
    """Get default paths relative to project root."""
    project_root = Path(__file__).parent.parent.parent
    frequencies_dir = project_root / "data" / "frequencies"
    loc_labels_dir = project_root / "data" / "loc_parsed"
    output_dir = project_root / "data" / "taxonomies"
    return frequencies_dir, loc_labels_dir, output_dir


def save_taxonomy_fast(taxonomy: Taxonomy, output_path: Path, labels_only_path: Path):
    """Save taxonomy using orjson for speed."""
    # Full taxonomy
    data = taxonomy.model_dump(mode="json")
    output_path.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))

    # Labels-only version
    labels_only = [
        {"id": label.id, "label": label.label, "frequency": label.frequency}
        for label in taxonomy.labels
    ]
    labels_only_path.write_bytes(orjson.dumps(labels_only, option=orjson.OPT_INDENT_2))


@app.command("list")
def cmd_list():
    """List available taxonomy types."""
    table = Table(title="Available Taxonomy Types")
    table.add_column("Type", style="cyan")
    table.add_column("Name", style="green")

    type_names = {
        TaxonomyType.LCC_MAIN: "LCC Main Classes (21)",
        TaxonomyType.LCC_SUBCLASS: "LCC Subclasses (~350)",
        TaxonomyType.LCSH_TOPICAL: "LCSH Topical Subjects (~51k)",
        TaxonomyType.LCSH_GEOGRAPHIC: "LCSH Geographic (~15k)",
        TaxonomyType.LCSH_FULL: "LCSH Full Headings (~486k)",
        TaxonomyType.LCGFT: "Genre/Form Terms (~550)",
        TaxonomyType.LCDGT: "Demographic Terms",
        TaxonomyType.SUDOC_AGENCY: "SuDoc Agencies (~480)",
        TaxonomyType.CORP_NAMES: "Corporate Names (~13k)",
    }

    for t in TaxonomyType:
        table.add_row(t.value, type_names.get(t, t.name))

    console.print(table)


@app.command("info")
def cmd_info(
    taxonomy: Annotated[
        str, typer.Argument(help="Taxonomy type (e.g., lcgft, lcsh_topical)")
    ],
):
    """Show information about a taxonomy."""
    frequencies_dir, loc_labels_dir, _ = get_default_paths()

    try:
        taxonomy_type = TaxonomyType(taxonomy)
    except ValueError:
        console.print(f"[red]Error:[/red] Unknown taxonomy type '{taxonomy}'")
        console.print("Use [cyan]shelf list[/cyan] to see available types.")
        raise typer.Exit(1)

    with console.status(f"Loading {taxonomy_type.value}..."):
        try:
            tax = load_taxonomy_from_frequencies(
                taxonomy_type,
                frequencies_dir,
                loc_labels_dir if loc_labels_dir.exists() else None,
            )
        except FileNotFoundError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)

    # Summary table
    table = Table(title=f"{tax.name}")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Type", tax.type.value)
    table.add_row("Total Labels", f"{tax.size:,}")
    table.add_row("Total Corpus Uses", f"{tax.total_corpus_uses:,}")

    console.print(table)
    console.print()

    # Coverage analysis
    coverage_table = Table(title="Coverage Analysis")
    coverage_table.add_column("Top N", justify="right", style="cyan")
    coverage_table.add_column("Coverage", justify="right", style="green")

    counts = sorted([label.frequency for label in tax.labels], reverse=True)
    total = sum(counts)

    for n in [10, 25, 50, 100, 250, 500, 1000, 2000]:
        if n > len(counts):
            break
        top_n_sum = sum(counts[:n])
        coverage = top_n_sum / total * 100 if total > 0 else 0
        coverage_table.add_row(f"{n:,}", f"{coverage:.1f}%")

    console.print(coverage_table)
    console.print()

    # Top labels
    top_table = Table(title="Top 20 Labels")
    top_table.add_column("Freq", justify="right", style="cyan")
    top_table.add_column("Label", style="white")

    for label in tax.labels[:20]:
        top_table.add_row(f"{label.frequency:,}", label.label[:60])

    console.print(top_table)


@app.command("extract")
def cmd_extract(
    taxonomy: Annotated[str, typer.Argument(help="Taxonomy type")],
    top_n: Annotated[
        Optional[int], typer.Option("--top-n", "-n", help="Extract top N labels")
    ] = None,
    min_freq: Annotated[
        Optional[int], typer.Option("--min-freq", help="Minimum frequency")
    ] = None,
    output_dir: Annotated[
        Optional[Path], typer.Option("--output-dir", "-o", help="Output directory")
    ] = None,
):
    """Extract top N labels from a taxonomy."""
    frequencies_dir, loc_labels_dir, default_output = get_default_paths()
    output_dir = output_dir or default_output
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        taxonomy_type = TaxonomyType(taxonomy)
    except ValueError:
        console.print(f"[red]Error:[/red] Unknown taxonomy type '{taxonomy}'")
        raise typer.Exit(1)

    with console.status(f"Loading {taxonomy_type.value}..."):
        tax = load_taxonomy_from_frequencies(
            taxonomy_type,
            frequencies_dir,
            loc_labels_dir if loc_labels_dir.exists() else None,
        )

    console.print(f"  Loaded [cyan]{tax.size:,}[/cyan] labels")

    if top_n:
        tax = tax.top_n(top_n)
        console.print(f"  Extracted top [cyan]{top_n}[/cyan] labels")
        console.print(f"  Coverage: [green]{tax.corpus_coverage:.1%}[/green]")

    if min_freq:
        tax = tax.filter_by_min_frequency(min_freq)
        console.print(
            f"  Filtered to [cyan]{tax.size}[/cyan] labels with freq >= {min_freq}"
        )

    # Output filenames
    suffix = f"_top{top_n}" if top_n else (f"_minfreq{min_freq}" if min_freq else "")
    output_path = output_dir / f"{taxonomy_type.value}{suffix}.json"
    labels_path = output_dir / f"{taxonomy_type.value}{suffix}_labels.json"

    with console.status("Saving..."):
        save_taxonomy_fast(tax, output_path, labels_path)

    console.print(f"  Saved: [green]{output_path}[/green]")
    console.print(f"  Labels: [green]{labels_path}[/green]")


@app.command("extract-all")
def cmd_extract_all(
    output_dir: Annotated[
        Optional[Path], typer.Option("--output-dir", "-o", help="Output directory")
    ] = None,
    workers: Annotated[
        int, typer.Option("--workers", "-w", help="Number of parallel workers")
    ] = 4,
):
    """Extract standard sets from all taxonomies (parallel)."""
    frequencies_dir, loc_labels_dir, default_output = get_default_paths()
    output_dir = output_dir or default_output
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define standard extractions: (taxonomy_type, top_n, description)
    extractions = [
        (TaxonomyType.LCC_MAIN, None, "All 21 main classes"),
        (TaxonomyType.LCC_SUBCLASS, 50, "Top 50 subclasses"),
        (TaxonomyType.LCC_SUBCLASS, 100, "Top 100 subclasses"),
        (TaxonomyType.LCSH_TOPICAL, 100, "Top 100 topical subjects"),
        (TaxonomyType.LCSH_TOPICAL, 500, "Top 500 topical subjects"),
        (TaxonomyType.LCSH_TOPICAL, 1000, "Top 1000 topical subjects"),
        (TaxonomyType.LCSH_TOPICAL, 2000, "Top 2000 topical subjects"),
        (TaxonomyType.LCSH_GEOGRAPHIC, 100, "Top 100 geographic"),
        (TaxonomyType.LCSH_GEOGRAPHIC, 500, "Top 500 geographic"),
        (TaxonomyType.LCGFT, 50, "Top 50 genre/form"),
        (TaxonomyType.LCGFT, 100, "Top 100 genre/form"),
        (TaxonomyType.LCGFT, None, "All genre/form"),
        (TaxonomyType.SUDOC_AGENCY, 50, "Top 50 SuDoc agencies"),
        (TaxonomyType.CORP_NAMES, 100, "Top 100 corporate names"),
    ]

    # Cache loaded taxonomies to avoid re-reading
    taxonomy_cache: dict[TaxonomyType, Taxonomy] = {}

    def load_if_needed(taxonomy_type: TaxonomyType) -> Taxonomy:
        if taxonomy_type not in taxonomy_cache:
            taxonomy_cache[taxonomy_type] = load_taxonomy_from_frequencies(
                taxonomy_type,
                frequencies_dir,
                loc_labels_dir if loc_labels_dir.exists() else None,
            )
        return taxonomy_cache[taxonomy_type]

    def process_extraction(
        taxonomy_type: TaxonomyType, top_n: int | None, desc: str
    ) -> dict:
        try:
            tax = load_if_needed(taxonomy_type)

            if top_n:
                tax = tax.top_n(top_n)

            suffix = f"_top{top_n}" if top_n else ""
            output_path = output_dir / f"{taxonomy_type.value}{suffix}.json"
            labels_path = output_dir / f"{taxonomy_type.value}{suffix}_labels.json"

            save_taxonomy_fast(tax, output_path, labels_path)

            return {
                "taxonomy": taxonomy_type.value,
                "description": desc,
                "labels": tax.size,
                "coverage": f"{tax.corpus_coverage:.1%}",
                "file": output_path.name,
                "status": "success",
            }
        except Exception as e:
            return {
                "taxonomy": taxonomy_type.value,
                "description": desc,
                "status": "error",
                "error": str(e),
            }

    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Extracting taxonomies...", total=len(extractions))

        # Process in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_extraction, tax_type, top_n, desc): (
                    tax_type,
                    top_n,
                    desc,
                )
                for tax_type, top_n, desc in extractions
            }

            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                progress.advance(task)

                if result.get("status") == "success":
                    console.print(
                        f"  [green]✓[/green] {result['taxonomy']}: "
                        f"{result['labels']} labels, {result['coverage']} coverage"
                    )
                else:
                    console.print(
                        f"  [red]✗[/red] {result['taxonomy']}: {result.get('error', 'Unknown error')}"
                    )

    # Save summary
    summary_path = output_dir / "extraction_summary.json"
    summary_path.write_bytes(orjson.dumps(results, option=orjson.OPT_INDENT_2))

    # Results table
    console.print()
    table = Table(title="Extraction Summary")
    table.add_column("Taxonomy", style="cyan")
    table.add_column("Labels", justify="right", style="green")
    table.add_column("Coverage", justify="right")
    table.add_column("File")

    for r in sorted(results, key=lambda x: x.get("file", "")):
        if r.get("status") == "success":
            table.add_row(
                r["taxonomy"],
                str(r["labels"]),
                r["coverage"],
                r["file"],
            )

    console.print(table)
    console.print(f"\n[green]Summary saved to:[/green] {summary_path}")


# =============================================================================
# Generation Commands (shelf gen ...)
# =============================================================================


@gen_app.command("stats")
def cmd_gen_stats():
    """Show LC taxonomy statistics for generation."""
    stats = get_taxonomy_stats()

    table = Table(title="LC Taxonomy Statistics")
    table.add_column("Dimension", style="cyan")
    table.add_column("Count", justify="right", style="green")

    table.add_row("LCC Main Classes", str(stats["lcc_classes"]))
    table.add_row("LCGFT Categories", str(stats["lcgft_categories"]))
    table.add_row("LCGFT Total Forms", str(stats["lcgft_total_terms"]))
    table.add_row("LCSH Topics (in taxonomy)", str(stats["lcsh_total_topics"]))
    table.add_row("LCDGT Audience Groups", str(stats["lcdgt_groups"]))

    console.print(table)
    console.print()

    # Show LCGFT categories
    cat_table = Table(title="LCGFT Categories (document types)")
    cat_table.add_column("Category", style="cyan")
    cat_table.add_column("Forms", justify="right", style="green")

    for cat, children in sorted(LCGFT_CHILDREN.items(), key=lambda x: -len(x[1])):
        if children:  # Only show non-empty categories
            cat_table.add_row(cat, str(len(children)))

    console.print(cat_table)


@gen_app.command("sample")
def cmd_gen_sample(
    n: Annotated[int, typer.Option("--count", "-n", help="Number of documents")] = 10,
    seed: Annotated[int, typer.Option("--seed", "-s", help="Random seed")] = 42,
):
    """Generate and display sample documents."""
    config = LCGeneratorConfig(seed=seed, total_documents=n)
    gen = LCBenchmarkGenerator(config)

    docs = gen.generate_batch(n)

    for doc in docs:
        console.print(f"\n[bold cyan]{doc.id}[/bold cyan]")
        console.print(f"  LCC: [green]{doc.lcc_class}[/green] ({doc.lcc_name})")
        console.print(
            f"  LCGFT: [yellow]{doc.lcgft_category}[/yellow] > {doc.lcgft_form}"
        )
        console.print(f"  Topics: {', '.join(doc.topics)}")
        if doc.lcdgt_audience:
            console.print(f"  Audience: {doc.lcdgt_audience}")
        if doc.geographic:
            console.print(f"  Geographic: {', '.join(doc.geographic)}")


@gen_app.command("create")
def cmd_gen_create(
    n: Annotated[int, typer.Option("--count", "-n", help="Number of documents")] = 1000,
    seed: Annotated[int, typer.Option("--seed", "-s", help="Random seed")] = 42,
    stratified: Annotated[
        bool, typer.Option("--stratified", help="Use stratified sampling")
    ] = False,
    output: Annotated[
        Optional[Path], typer.Option("--output", "-o", help="Output JSONL file")
    ] = None,
):
    """Generate benchmark documents and save to file."""
    _, _, default_output = get_default_paths()
    output = output or default_output / "benchmark_documents.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)

    config = LCGeneratorConfig(seed=seed, total_documents=n)
    gen = LCBenchmarkGenerator(config)

    with console.status(f"Generating {n} documents..."):
        if stratified:
            docs = gen.generate_stratified()
            console.print(f"Generated [cyan]{len(docs)}[/cyan] stratified documents")
        else:
            docs = gen.generate_batch(n)
            console.print(f"Generated [cyan]{len(docs)}[/cyan] random documents")

    # Get distribution
    dist = gen.get_distribution(docs)

    # Save as JSONL
    with console.status("Saving..."):
        with open(output, "wb") as f:
            for doc in docs:
                f.write(orjson.dumps(doc.model_dump(mode="json")))
                f.write(b"\n")

    console.print(f"Saved to: [green]{output}[/green]")
    console.print()

    # Summary table
    table = Table(title="Generation Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")

    table.add_row("Total Documents", f"{dist['total_documents']:,}")
    table.add_row("Unique LCC Classes", f"{dist['unique_lcc_classes']}")
    table.add_row("Unique LCGFT Categories", f"{dist['unique_lcgft_categories']}")
    table.add_row("Unique LCGFT Forms", f"{dist['unique_lcgft_forms']}")
    table.add_row("Unique Topics", f"{dist['unique_topics']}")

    console.print(table)

    # LCC distribution
    console.print()
    lcc_table = Table(title="LCC Distribution")
    lcc_table.add_column("Class", style="cyan")
    lcc_table.add_column("Name")
    lcc_table.add_column("Count", justify="right", style="green")

    for lcc, count in sorted(dist["lcc_distribution"].items()):
        from .benchmark import LCCClass

        lcc_table.add_row(lcc, LCC_NAMES.get(LCCClass(lcc), ""), str(count))

    console.print(lcc_table)


@gen_app.command("distribution")
def cmd_gen_distribution(
    input_file: Annotated[Path, typer.Argument(help="Input JSONL file")],
):
    """Analyze distribution of a generated benchmark file."""
    if not input_file.exists():
        console.print(f"[red]Error:[/red] File not found: {input_file}")
        raise typer.Exit(1)

    # Load documents
    from .benchmark import LCDocument

    docs = []
    with open(input_file, "rb") as f:
        for line in f:
            data = orjson.loads(line)
            docs.append(LCDocument(**data))

    console.print(f"Loaded [cyan]{len(docs):,}[/cyan] documents from {input_file}")

    # Analyze distribution
    config = LCGeneratorConfig()
    gen = LCBenchmarkGenerator(config)
    dist = gen.get_distribution(docs)

    # Summary
    table = Table(title="Distribution Summary")
    table.add_column("Dimension", style="cyan")
    table.add_column("Unique Values", justify="right", style="green")

    table.add_row("LCC Classes", f"{dist['unique_lcc_classes']}")
    table.add_row("LCGFT Categories", f"{dist['unique_lcgft_categories']}")
    table.add_row("LCGFT Forms", f"{dist['unique_lcgft_forms']}")
    table.add_row("Topics", f"{dist['unique_topics']}")

    console.print(table)

    # Top categories
    console.print()
    cat_table = Table(title="Top 15 LCGFT Categories")
    cat_table.add_column("Category", style="cyan")
    cat_table.add_column("Count", justify="right", style="green")

    for cat, count in list(dist["lcgft_category_distribution"].items())[:15]:
        cat_table.add_row(cat, str(count))

    console.print(cat_table)

    # Top forms
    console.print()
    form_table = Table(title="Top 15 LCGFT Forms")
    form_table.add_column("Form", style="cyan")
    form_table.add_column("Count", justify="right", style="green")

    for form, count in list(dist["lcgft_form_distribution"].items())[:15]:
        form_table.add_row(form, str(count))

    console.print(form_table)


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    sys.exit(main())
