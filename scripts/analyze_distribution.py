#!/usr/bin/env python
"""Analyze distribution of generated benchmark documents.

Computes per-dimension distributions and 2D co-occurrence matrices
for all taxonomy dimensions.

Usage:
    python scripts/analyze_distribution.py
    python scripts/analyze_distribution.py --artifacts-dir data/artifacts --output-dir data/analysis
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

console = Console()


def load_artifacts(artifacts_dir: Path) -> list[dict]:
    """Load all artifact JSON files."""
    docs = []
    for f in sorted(artifacts_dir.glob("*.json")):
        with open(f) as fp:
            docs.append(json.load(fp))
    return docs


def extract_dimensions(docs: list[dict]) -> dict[str, list]:
    """Extract dimension values from documents."""
    return {
        "lcc_code": [d["lcc_code"] for d in docs],
        "lcc_name": [d["lcc_name"] for d in docs],
        "lcgft_category": [d["lcgft_category"] for d in docs],
        "lcgft_form": [d["lcgft_form"] for d in docs],
        "target_length": [d["target_length"] for d in docs],
        "register": [d["register"] for d in docs],
        "audience": [d.get("audience") or "(none)" for d in docs],
        # Flatten multi-value fields
        "topics": [t for d in docs for t in d.get("topics", [])],
        "geographic": [g for d in docs for g in d.get("geographic", [])] or ["(none)"],
        # Generation metadata
        "model": [d.get("model", "(unknown)") for d in docs],
    }


def compute_distributions(dimensions: dict[str, list]) -> dict[str, Counter]:
    """Compute frequency distribution for each dimension."""
    return {name: Counter(values) for name, values in dimensions.items()}


def compute_cooccurrence(
    docs: list[dict], dim1: str, dim2: str
) -> tuple[np.ndarray, list[str], list[str]]:
    """Compute co-occurrence matrix between two dimensions."""
    # Extract values
    if dim1 == "topics":
        vals1 = [tuple(sorted(d.get("topics", []))) for d in docs]
        # Flatten to individual topics
        all_vals1 = sorted(set(t for d in docs for t in d.get("topics", [])))
    elif dim1 == "geographic":
        vals1 = [tuple(sorted(d.get("geographic", []))) for d in docs]
        all_vals1 = sorted(set(g for d in docs for g in d.get("geographic", []))) or [
            "(none)"
        ]
    elif dim1 == "audience":
        vals1 = [d.get(dim1) or "(none)" for d in docs]
        all_vals1 = sorted(set(vals1))
    else:
        vals1 = [d[dim1] for d in docs]
        all_vals1 = sorted(set(vals1))

    if dim2 == "topics":
        vals2 = [tuple(sorted(d.get("topics", []))) for d in docs]
        all_vals2 = sorted(set(t for d in docs for t in d.get("topics", [])))
    elif dim2 == "geographic":
        vals2 = [tuple(sorted(d.get("geographic", []))) for d in docs]
        all_vals2 = sorted(set(g for d in docs for g in d.get("geographic", []))) or [
            "(none)"
        ]
    elif dim2 == "audience":
        vals2 = [d.get(dim2) or "(none)" for d in docs]
        all_vals2 = sorted(set(vals2))
    else:
        vals2 = [d[dim2] for d in docs]
        all_vals2 = sorted(set(vals2))

    # Build matrix
    idx1 = {v: i for i, v in enumerate(all_vals1)}
    idx2 = {v: i for i, v in enumerate(all_vals2)}
    matrix = np.zeros((len(all_vals1), len(all_vals2)), dtype=int)

    for d in docs:
        # Get values for this doc
        if dim1 == "topics":
            v1_list = d.get("topics", [])
        elif dim1 == "geographic":
            v1_list = d.get("geographic", []) or ["(none)"]
        elif dim1 == "audience":
            v1_list = [d.get(dim1) or "(none)"]
        else:
            v1_list = [d[dim1]]

        if dim2 == "topics":
            v2_list = d.get("topics", [])
        elif dim2 == "geographic":
            v2_list = d.get("geographic", []) or ["(none)"]
        elif dim2 == "audience":
            v2_list = [d.get(dim2) or "(none)"]
        else:
            v2_list = [d[dim2]]

        for v1 in v1_list:
            for v2 in v2_list:
                if v1 in idx1 and v2 in idx2:
                    matrix[idx1[v1], idx2[v2]] += 1

    return matrix, all_vals1, all_vals2


def print_distribution(name: str, dist: Counter, top_n: int = 20) -> None:
    """Print a distribution table."""
    table = Table(title=f"{name} Distribution ({len(dist)} unique)")
    table.add_column("Value", style="cyan")
    table.add_column("Count", justify="right", style="green")
    table.add_column("Percent", justify="right", style="yellow")

    total = sum(dist.values())
    for value, count in dist.most_common(top_n):
        pct = 100 * count / total
        table.add_row(str(value)[:50], str(count), f"{pct:.1f}%")

    if len(dist) > top_n:
        table.add_row(f"... and {len(dist) - top_n} more", "", "")

    console.print(table)
    console.print()


def print_matrix(
    matrix: np.ndarray,
    rows: list[str],
    cols: list[str],
    title: str,
    max_rows: int = 15,
    max_cols: int = 10,
) -> None:
    """Print a co-occurrence matrix as a table."""
    # Truncate if needed
    if len(rows) > max_rows:
        # Keep top rows by sum
        row_sums = matrix.sum(axis=1)
        top_idx = np.argsort(row_sums)[-max_rows:][::-1]
        matrix = matrix[top_idx, :]
        rows = [rows[i] for i in top_idx]

    if len(cols) > max_cols:
        # Keep top cols by sum
        col_sums = matrix.sum(axis=0)
        top_idx = np.argsort(col_sums)[-max_cols:][::-1]
        matrix = matrix[:, top_idx]
        cols = [cols[i] for i in top_idx]

    table = Table(title=title)
    table.add_column("", style="cyan")
    for col in cols:
        table.add_column(str(col)[:12], justify="right")

    for i, row in enumerate(rows):
        values = [
            str(matrix[i, j]) if matrix[i, j] > 0 else "·" for j in range(len(cols))
        ]
        table.add_row(str(row)[:25], *values)

    console.print(table)
    console.print()


def save_distribution_json(
    distributions: dict[str, Counter],
    output_path: Path,
) -> None:
    """Save distributions to JSON."""
    data = {name: dict(dist.most_common()) for name, dist in distributions.items()}
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def save_matrix_csv(
    matrix: np.ndarray,
    rows: list[str],
    cols: list[str],
    output_path: Path,
) -> None:
    """Save co-occurrence matrix to CSV."""
    with open(output_path, "w") as f:
        f.write("," + ",".join(f'"{c}"' for c in cols) + "\n")
        for i, row in enumerate(rows):
            values = ",".join(str(matrix[i, j]) for j in range(len(cols)))
            f.write(f'"{row}",{values}\n')


def main():
    parser = argparse.ArgumentParser(
        description="Analyze distribution of generated benchmark documents"
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("data/artifacts"),
        help="Directory containing artifact JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save analysis outputs (optional)",
    )
    args = parser.parse_args()

    # Load data
    console.print(f"[bold]Loading artifacts from {args.artifacts_dir}[/bold]")
    docs = load_artifacts(args.artifacts_dir)
    console.print(f"Loaded {len(docs):,} documents\n")

    if not docs:
        console.print("[red]No documents found![/red]")
        return

    # Extract dimensions
    dimensions = extract_dimensions(docs)

    # Compute distributions
    distributions = compute_distributions(dimensions)

    # Print 1D distributions
    console.print("[bold underline]Per-Dimension Distributions[/bold underline]\n")

    # Generation metadata first
    print_distribution("model", distributions["model"])

    for name in [
        "lcc_code",
        "lcgft_category",
        "lcgft_form",
        "target_length",
        "register",
        "audience",
        "topics",
        "geographic",
    ]:
        print_distribution(name, distributions[name])

    # Word count stats
    word_counts = [d["word_count"] for d in docs]
    console.print("[bold]Word Count Statistics[/bold]")
    console.print(f"  Min: {min(word_counts):,}")
    console.print(f"  Max: {max(word_counts):,}")
    console.print(f"  Mean: {np.mean(word_counts):,.0f}")
    console.print(f"  Median: {np.median(word_counts):,.0f}")
    console.print(f"  Std: {np.std(word_counts):,.0f}")
    console.print()

    # 2D co-occurrence matrices
    console.print("[bold underline]Co-occurrence Matrices[/bold underline]\n")

    matrix_pairs = [
        ("lcc_code", "lcgft_category"),
        ("lcc_code", "target_length"),
        ("lcc_code", "register"),
        ("lcgft_category", "target_length"),
        ("lcgft_category", "register"),
        ("target_length", "register"),
    ]

    for dim1, dim2 in matrix_pairs:
        matrix, rows, cols = compute_cooccurrence(docs, dim1, dim2)
        print_matrix(matrix, rows, cols, f"{dim1} × {dim2}")

    # Coverage summary
    console.print("[bold underline]Coverage Summary[/bold underline]\n")
    coverage = Table(title="Dimension Coverage")
    coverage.add_column("Dimension", style="cyan")
    coverage.add_column("Unique in Sample", justify="right", style="green")
    coverage.add_column("Total Possible", justify="right", style="yellow")
    coverage.add_column("Coverage", justify="right", style="magenta")

    possible = {
        "lcc_code": 21,
        "lcgft_category": 14,
        "lcgft_form": 143,
        "target_length": 8,
        "register": 8,
        "audience": 24,
        "topics": 113,
        "geographic": 44,
    }

    for name, total in possible.items():
        unique = len(distributions[name])
        pct = 100 * unique / total
        coverage.add_row(name, str(unique), str(total), f"{pct:.0f}%")

    console.print(coverage)

    # Save outputs if requested
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)

        # Save distributions
        dist_path = args.output_dir / "distributions.json"
        save_distribution_json(distributions, dist_path)
        console.print(f"\nSaved distributions to {dist_path}")

        # Save matrices
        for dim1, dim2 in matrix_pairs:
            matrix, rows, cols = compute_cooccurrence(docs, dim1, dim2)
            matrix_path = args.output_dir / f"matrix_{dim1}_x_{dim2}.csv"
            save_matrix_csv(matrix, rows, cols, matrix_path)
        console.print(f"Saved {len(matrix_pairs)} matrices to {args.output_dir}")


if __name__ == "__main__":
    main()
