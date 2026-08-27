#!/usr/bin/env python
"""Stage the v0.4 release for the HuggingFace Hub.

Follows the v0.3.1 publishing shape (`shelf.hub.push_folder_to_hub`): one repo
folder holding a parquet directory per config plus a README whose YAML front
matter declares every config's `data_files`.

**`default` is left untouched.** v0.4 ships as additional configs rather than a
merged corpus, for a measured reason: v0.4 alone caps its largest generator at
9.2% with 14 of 15 generators above 5%, while pooling it into v0.3.1 leaves gpt-5.2 at
49.9% and only 2 generators above 5%. Merging would destroy the generator
balance v0.4 exists to provide, and would silently invalidate every published
v0.3.1 baseline.

Nothing is uploaded without `--upload`. The default is a dry run that stages the
folder and prints what would be pushed.

Usage:
    uv run python scripts/publish_v04.py                 # stage + report
    uv run python scripts/publish_v04.py --upload        # actually push
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()

# Slices to publish, in card order. Each maps a local assembled dataset to the
# config name readers will use.
SLICES: tuple[tuple[str, str, str], ...] = (
    (
        "v0_4_core",
        "data/hf_dataset/v0.4",
        "Generator-balanced core: 15 generators, largest share 9.2%. Split on spec_id.",
    ),
    (
        "v0_4_supplement",
        "data/hf_dataset/v0.4_subclass",
        "Supplementary documents over the 21-class task, single generator.",
    ),
    (
        "v0_4_minimal_pairs",
        "data/hf_dataset/v0.4_minimal_pairs",
        "Minimal pairs varying exactly one facet (form or subject). Split on pair_id.",
    ),
    (
        "v0_4_holdout",
        "data/hf_dataset/v0.4_holdout",
        "Held-out probe from claude-fable-5, a generator absent from the core.",
    ),
)

TRANSFER = (
    "transfer_gutenberg",
    "data/transfer/gutenberg/records.jsonl",
    "Natural, human-catalogued Project Gutenberg passages. "
    "KNOWN CONTAMINATED - never pool with synthetic slices.",
)

CAVEATS = """
## Known limitations in the v0.4 slices

These are measured, not suspected, and are published alongside the data rather
than left for a reader to discover.

**No subclass tier.** An LCC subclass tier was planned and is not shipped: the
specification blocks assigned 80 subclasses but generation used the parent-class
description, so those documents carry 16 parent classes and no subclass label.
They ship as `v0_4_supplement`.

**Empty-document rate.** `v0_4_supplement` and `v0_4_minimal_pairs` were generated
before a fix for a reasoning-budget defect landed: on short length targets,
reasoning tokens consumed the entire output cap, yielding a title truncated
mid-word and no body. 13-15% of raw generations were affected. Those documents
are removed by the QC gates, so the published slices are clean, but they are
~14% smaller than their nominal spec count.

**Mild generator confound on register and length.** In `v0_4_core`, generator is
independent of the labels that matter -- LCC class (Cramer's V 0.018) and LCGFT
category (0.028) -- but is measurably correlated with `register` (0.062),
`target_length` (0.066) and `prompt_variant_id` (0.085). The cause is
non-uniform QC removal: the empty-body defect hit short documents hardest, and
two generators failed entirely for part of the run. Effect sizes are small but
the p-values are unambiguous. Any analysis conditioned on register or length
carries this confound.

**Do not pool `transfer_gutenberg` with synthetic slices.** It is in the
pretraining data of essentially every model that would be evaluated on it. SHELF
is the clean-synthetic condition; Gutenberg is the contaminated-natural one; the
gap between them is the measurement. Pooling them measures neither.

**Prompt variants differ from v0.3.1.** v0.4 documents were generated with four
new system-prompt variants and form-conditional output formatting; `default` was
generated with a single prompt. `prompt_variant_id` records which. A controlled
A/B measured spurious markdown on non-markdown forms dropping from 26.7% to
~1.3% (Fisher exact p < 0.00001).
"""


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def stage(out_dir: Path, upload: bool) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    table = Table(title="v0.4 release staging")
    table.add_column("Config")
    table.add_column("train", justify="right")
    table.add_column("validation", justify="right")
    table.add_column("test", justify="right")
    table.add_column("total", justify="right")

    staged: list[tuple[str, int]] = []
    for config, source, _desc in SLICES:
        src = Path(source)
        if not src.exists():
            console.print(f"[yellow]skipping {config}: {src} not built[/yellow]")
            continue
        counts = {}
        target = out_dir / config
        target.mkdir(parents=True, exist_ok=True)
        for split in ("train", "validation", "test"):
            rows = (
                load_jsonl(src / f"{split}.jsonl")
                if (src / f"{split}.jsonl").exists()
                else []
            )
            counts[split] = len(rows)
            if rows:
                import polars as pl

                frame = pl.DataFrame(rows, infer_schema_length=None)
                if "model" in frame.columns:
                    # claude-opus-5 was reachable through both OpenRouter and the
                    # native API, so it lands under two strings for one model.
                    # provider_served already records the route.
                    frame = frame.with_columns(
                        pl.col("model").str.replace(r"^anthropic/", "")
                    )
                frame.write_parquet(target / f"{split}-00000-of-00001.parquet")
        total = sum(counts.values())
        staged.append((config, total))
        table.add_row(
            config,
            f"{counts['train']:,}",
            f"{counts['validation']:,}",
            f"{counts['test']:,}",
            f"{total:,}",
        )

    # Transfer slice is a single split: it is a probe, not a trainable corpus.
    tpath = Path(TRANSFER[1])
    if tpath.exists():
        import polars as pl

        rows = load_jsonl(tpath)
        target = out_dir / TRANSFER[0]
        target.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(rows, infer_schema_length=None).write_parquet(
            target / "test-00000-of-00001.parquet"
        )
        staged.append((TRANSFER[0], len(rows)))
        table.add_row(TRANSFER[0], "-", "-", f"{len(rows):,}", f"{len(rows):,}")

    console.print(table)

    core = Path("data/hf_dataset/v0.4/train.jsonl")
    if core.exists():
        gens = Counter()
        for split in ("train", "validation", "test"):
            for row in load_jsonl(Path("data/hf_dataset/v0.4") / f"{split}.jsonl"):
                model = str(row.get("model", "?"))
                gens[model.removeprefix("anthropic/")] += 1
        total = sum(gens.values())
        console.print(
            f"\n  v0_4_core generators: {len(gens)}, "
            f"largest share {gens.most_common(1)[0][1] / total * 100:.1f}%"
        )

    (out_dir / "CAVEATS.md").write_text(CAVEATS.strip() + "\n")
    console.print(f"  caveats written to {out_dir / 'CAVEATS.md'}")
    console.print(
        f"\n[bold]Staged {sum(n for _, n in staged):,} documents "
        f"across {len(staged)} new configs -> {out_dir}[/bold]"
    )
    console.print(
        "  [yellow]`default` and the six pair configs are untouched.[/yellow]"
    )

    if not upload:
        console.print(
            "\n[yellow]DRY RUN - nothing uploaded. Re-run with --upload to push.[/yellow]"
        )
        return 0

    from shelf.hub.dataset import push_folder_to_hub

    url = push_folder_to_hub(
        out_dir,
        repo_id=ARGS.repo_id,
        commit_message="Add v0.4 slices (core, subclass, minimal pairs, holdout, transfer)",
    )
    console.print(f"[green]{url}[/green]")
    return 0


def main() -> int:
    global ARGS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data/hf_release/v0.4"))
    parser.add_argument("--repo-id", default="mjbommar/SHELF")
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Actually push. Without this the run only stages locally.",
    )
    ARGS = parser.parse_args()
    return stage(ARGS.out_dir, ARGS.upload)


if __name__ == "__main__":
    raise SystemExit(main())
