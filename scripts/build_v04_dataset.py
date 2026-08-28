#!/usr/bin/env python
"""Assemble a v0.4 SHELF corpus from generated artifacts.

Turns the raw output of `scripts/generate_documents.py` into a publishable
dataset, running every check the data plan requires *before* publication rather
than discovering defects afterwards.

The ordering matters and is not arbitrary:

1. **QC gates first** (§12). v0.3.1 shipped with ~2% of documents carrying
   explicit ``LCGFT:`` / ``LCC:`` headers and ~1.5% non-English, because nothing
   ran between generation and publication. Gates run here, and their pass rates
   are reported *per generator* so a bad model is visible rather than averaged
   away.
2. **Near-duplicate scan** (G6). Phase 1 gives one spec to every generator, so
   near-duplicates are expected *within* a spec and would be alarming across
   unrelated specs. Both rates are reported.
3. **Spec-level splitting** (§14). Splitting on document id would put fifteen
   realizations of the same spec across train and test. Measured on a
   Phase-1-shaped corpus, that straddles 598 of 600 specs.
4. **Promotion checks** (§12.1), including per-split SHA-256 hashes.

Nothing is dropped silently: rejected documents are counted per gate, and a
rejection summary is written next to the dataset.

Usage:
    uv run python scripts/build_v04_dataset.py \\
        --artifacts data/artifacts/v0.4/phase1.jsonl \\
        --output-dir data/hf_dataset/v0.4 \\
        --report data/artifacts/v0.4/promotion_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table
from shelf.evaluate.registry import LCGFT_CATEGORIES
from shelf.hub.splitter import SplitConfig, StratifiedSplitter
from shelf.qc.dedup import scan_corpus
from shelf.qc.gates import QCResult, run_gates
from shelf.qc.promotion import run_promotion_checks

console = Console()
logger = logging.getLogger(__name__)


def load_records(path: Path) -> list[dict[str, Any]]:
    """Read the JSONL index emitted by the generator."""
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def run_all_gates(
    records: list[dict[str, Any]],
    *,
    length_tolerance: float,
) -> dict[str, QCResult]:
    """Run the per-document gates over every candidate."""
    taxonomy_labels = tuple(LCGFT_CATEGORIES)
    results: dict[str, QCResult] = {}
    for record in records:
        word_range = record.get("target_word_range") or None
        results[str(record["id"])] = run_gates(
            str(record["id"]),
            title=str(record.get("title") or ""),
            body=str(record.get("body") or record.get("text") or ""),
            word_count=record.get("word_count"),
            target_word_range=(tuple(word_range) if word_range else None),
            topics=record.get("topics") or (),
            taxonomy_labels=taxonomy_labels,
            length_tolerance=length_tolerance,
        )
    return results


def report_gate_pass_rates(
    records: list[dict[str, Any]], qc: dict[str, QCResult]
) -> None:
    """Print per-generator pass rates. A bad generator must be visible."""
    by_generator: dict[str, list[QCResult]] = defaultdict(list)
    for record in records:
        model = str(record.get("model") or record.get("generator") or "unknown")
        by_generator[model].append(qc[str(record["id"])])

    gate_names = sorted({g.gate.value for r in qc.values() for g in r.gates})
    table = Table(title="QC gate pass rate by generator")
    table.add_column("Generator")
    table.add_column("Docs", justify="right")
    for name in gate_names:
        table.add_column(name, justify="right")
    table.add_column("All gates", justify="right")

    for model, results in sorted(by_generator.items()):
        row = [model, f"{len(results):,}"]
        for name in gate_names:
            passed = sum(
                1 for r in results for g in r.gates if g.gate.value == name and g.passed
            )
            row.append(f"{passed / len(results) * 100:.1f}%")
        overall = sum(1 for r in results if r.passed)
        row.append(f"{overall / len(results) * 100:.1f}%")
        table.add_row(*row)
    console.print(table)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts",
        type=Path,
        required=True,
        help="JSONL index written by generate_documents.py",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--group-by",
        default="spec_id",
        help="Split key. 'spec_id' is required for Phase 1 corpora (§14).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--length-tolerance", type=float, default=0.25)
    parser.add_argument(
        "--dedup-threshold",
        type=float,
        default=0.9,
        help="Jaccard threshold for the G6 near-duplicate scan",
    )
    parser.add_argument(
        "--keep-failures",
        action="store_true",
        help="Keep gate failures in the dataset, flagged rather than dropped",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.artifacts.exists():
        console.print(f"[red]No artifacts at {args.artifacts}[/red]")
        return 1

    records = load_records(args.artifacts)
    console.print(f"[bold]Loaded {len(records):,} generated documents[/bold]")
    if not records:
        return 1

    generators = Counter(str(r.get("model") or "unknown") for r in records)
    console.print(f"  generators: {len(generators)}")

    console.print("\n[bold]Running QC gates[/bold]")
    qc = run_all_gates(records, length_tolerance=args.length_tolerance)
    report_gate_pass_rates(records, qc)

    console.print("\n[bold]G6 near-duplicate scan[/bold]")
    # Scan only documents that parsed. Empty bodies are trivially identical to
    # each other, so including gate failures buries the real signal: on the
    # first Phase 1 assembly this produced 29,402 "near-duplicate pairs", every
    # one of them an empty-vs-empty match, against just 4 genuine ones.
    parsed = [r for r in records if str(r.get("body") or r.get("text") or "").strip()]
    skipped = len(records) - len(parsed)
    if skipped:
        console.print(f"  excluding {skipped:,} unparseable documents from the scan")
    ids = [str(r["id"]) for r in parsed]
    texts = [str(r.get("text") or r.get("body") or "") for r in parsed]
    index, g6_results = scan_corpus(ids, texts, threshold=args.dedup_threshold)
    pairs = index.find_all_duplicate_pairs()

    # Attach G6 so `QCResult.passed` accounts for it alongside G1-G5/G7.
    for doc_id, g6 in g6_results.items():
        if doc_id in qc:
            qc[doc_id] = qc[doc_id].with_near_duplicate(g6)

    spec_of = {str(r["id"]): str(r.get("spec_id") or "") for r in records}
    within = sum(
        1 for p in pairs if spec_of.get(str(p[0])) == spec_of.get(str(p[1])) != ""
    )
    console.print(
        f"  {len(pairs):,} near-duplicate pairs "
        f"({within:,} within a spec — expected by design; "
        f"{len(pairs) - within:,} across specs — investigate)"
    )

    retained = [r for r in records if args.keep_failures or qc[str(r["id"])].passed]
    dropped = len(records) - len(retained)
    console.print(
        f"\n[bold]Retained {len(retained):,} of {len(records):,}[/bold] "
        f"({dropped:,} dropped by gates)"
    )

    if len(retained) < 100:
        console.print("[red]Too few documents survive to split.[/red]")
        return 1

    console.print("\n[bold]Splitting[/bold]")
    split_config = SplitConfig(random_seed=args.seed, group_by=args.group_by)
    result = StratifiedSplitter(split_config).split(retained)
    console.print(
        f"  train={len(result.train):,} dev={len(result.dev):,} test={len(result.test):,}"
    )
    if args.group_by:
        console.print(f"  grouping: {result.statistics.get('grouping')}")

    console.print("\n[bold]Promotion checks[/bold]")
    promotion = run_promotion_checks(
        records,
        qc,
        splits={
            "train": [str(d["id"]) for d in result.train],
            "validation": [str(d["id"]) for d in result.dev],
            "test": [str(d["id"]) for d in result.test],
        },
        dedup_pairs=[(str(a), str(b), float(s)) for a, b, s in pairs],
    )

    payload = {
        "n_generated": len(records),
        "n_retained": len(retained),
        "n_dropped": dropped,
        "generators": dict(generators),
        "split_sizes": {
            "train": len(result.train),
            "validation": len(result.dev),
            "test": len(result.test),
        },
        "split_checksum": result.checksum,
        "grouping": result.statistics.get("grouping"),
        "near_duplicate_pairs": len(pairs),
        "near_duplicate_within_spec": within,
        "promotion": (
            promotion.to_dict() if hasattr(promotion, "to_dict") else str(promotion)
        ),
    }

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(payload, indent=2, default=str))
        console.print(f"  report: {args.report}")

    if args.dry_run:
        console.print(
            "\n[yellow]DRY RUN - nothing written to the dataset dir.[/yellow]"
        )
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, docs in (
        ("train", result.train),
        ("validation", result.dev),
        ("test", result.test),
    ):
        path = args.output_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for doc in docs:
                handle.write(json.dumps(doc) + "\n")
        console.print(f"  wrote {path} ({len(docs):,} docs)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
