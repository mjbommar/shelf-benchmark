#!/usr/bin/env python
"""Assemble the human-validation kit (Phase 5).

Everything else in this plan is a machine measuring a machine. Without a human
ceiling, "model X reaches 0.89 macro-F1" has no interpretable upper bound: it is
not known whether 0.89 is near-perfect, mediocre, or above what a trained
cataloguer would achieve on the same documents.

This produces the materials a person needs and the scoring that consumes their
output. It cannot produce the labels -- that is the point.

Outputs
    sample.jsonl        stratified documents, gold labels withheld
    gold.jsonl          the withheld labels, for scoring only
    instructions.md     coding guidance
    coding_sheet.csv    one row per document, blank label columns
    README.md           how to run it and what comes back

Usage:
    uv run python scripts/build_annotation_kit.py --n 400 --coders 2
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

from rich.console import Console

console = Console()

INSTRUCTIONS = """# SHELF human validation — coding instructions

You are assigning library classification labels to {n} documents. Work
independently: do not discuss any document with the other coder until both of
you have finished. Disagreement is data, not a problem to resolve in advance.

## What you are labelling

For each document, record two labels.

**1. LCC class** — one letter from the Library of Congress Classification:

{lcc_table}

**2. LCGFT category** — one of the fourteen Library of Congress Genre/Form
categories:

{cat_table}

## How to decide

- Judge the document **as it is**, not what it might have been. Classify what is
  on the page.
- LCC asks *what is this about*. LCGFT asks *what kind of thing is this*. They
  are independent: a joke about military science is `U` and `Recreational
  works`. Unusual combinations are deliberate and common in this corpus.
- Titles are omitted on purpose. Classify from the body alone.
- If two labels seem equally defensible, pick one and mark `uncertain` = `y`.
  Do not leave a cell blank. A forced choice plus an uncertainty flag is far
  more informative than a gap.
- Spend roughly a minute per document. First reading is usually right; this
  measures achievable agreement, not maximum effort.

## Documents that are not classifiable

Some documents may be too short, incoherent, or not English. Mark
`unusable` = `y` and still record your best-guess labels. The rate at which this
happens is itself a corpus-quality measurement.

## Returning your work

Fill in `coding_sheet.csv` and return it unchanged apart from your entries.
Do not reorder or delete rows.
"""

README = """# SHELF human validation kit

{n} documents, stratified across LCC class and LCGFT category, sampled with
seed {seed} from `{source}`.

## What this is for

Two things that nothing else in the benchmark can supply:

1. **A human ceiling.** Model scores are uninterpretable without one.
2. **Inter-annotator agreement.** If two trained coders agree only 70% of the
   time on LCC class, then a model scoring 0.89 against the generated labels is
   measuring something other than what a cataloguer would call correct, and the
   benchmark's ceiling is the label noise rather than the task.

## Who should do this

Someone with cataloguing or library-science familiarity. Two coders working
independently. This is deliberately not crowdsourceable: the point is expert
agreement, and a non-expert's disagreement rate would measure the wrong thing.

## Procedure

1. Give each coder `sample.jsonl`, `instructions.md` and their own copy of
   `coding_sheet.csv`.
2. Coders work independently and do not discuss documents.
3. Return both completed sheets.
4. Score:

       uv run python scripts/score_annotations.py \\
           --coder-a coder_a.csv --coder-b coder_b.csv --gold gold.jsonl

That reports Cohen's kappa between coders, each coder against the generated
labels, and the resulting human ceiling per task.

## What is deliberately withheld

`sample.jsonl` carries no `lcc_code`, `lcgft_category`, `lcgft_form`, `topics`
or `title`. Gold labels live in `gold.jsonl`, which coders must not see.
"""


def load_documents(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def stratified_sample(docs: list[dict], n: int, seed: int) -> list[dict]:
    """Sample evenly across LCC class, then across category within class."""
    rng = random.Random(seed)
    by_class: dict[str, list[dict]] = {}
    for doc in docs:
        code = str(doc.get("lcc_code") or "")
        text = str(doc.get("text") or doc.get("body") or "")
        # A 15-word document cannot be classified by anyone; including them
        # would measure the corpus's short tail rather than coder agreement.
        if code and len(text.split()) >= 60:
            by_class.setdefault(code, []).append(doc)
    if not by_class:
        return []

    per_class = max(1, n // len(by_class))
    picked: list[dict] = []
    for code in sorted(by_class):
        pool = by_class[code]
        rng.shuffle(pool)
        picked.extend(pool[:per_class])
    rng.shuffle(picked)
    return picked[:n]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("data/artifacts/v0.4/phase1.jsonl")
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/annotation/v0.4"))
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--coders", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from shelf.evaluate.registry import LCC_CODES, LCGFT_CATEGORIES

    if not args.source.exists():
        console.print(f"[red]No source corpus at {args.source}[/red]")
        return 1

    docs = load_documents(args.source)
    sample = stratified_sample(docs, args.n, args.seed)
    if not sample:
        console.print(
            "[red]No eligible documents (need >=60 words and an LCC code).[/red]"
        )
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    withheld = {
        "lcc_code",
        "lcc_name",
        "lcc_uri",
        "lcgft_category",
        "lcgft_form",
        "topics",
        "title",
        "lcc_subclass",
        "lcc_subclass_name",
    }

    with (args.out_dir / "sample.jsonl").open("w", encoding="utf-8") as handle:
        for i, doc in enumerate(sample):
            blind = {k: v for k, v in doc.items() if k not in withheld}
            blind["annotation_id"] = f"doc_{i:04d}"
            handle.write(json.dumps(blind) + "\n")

    with (args.out_dir / "gold.jsonl").open("w", encoding="utf-8") as handle:
        for i, doc in enumerate(sample):
            handle.write(
                json.dumps(
                    {
                        "annotation_id": f"doc_{i:04d}",
                        "id": doc.get("id"),
                        "lcc_code": doc.get("lcc_code"),
                        "lcgft_category": doc.get("lcgft_category"),
                        "lcgft_form": doc.get("lcgft_form"),
                        "model": doc.get("model"),
                    }
                )
                + "\n"
            )

    from shelf.sampler.generator import LCC_SEMANTIC_DESCRIPTIONS

    lcc_table = "\n".join(
        f"- `{c}` — {LCC_SEMANTIC_DESCRIPTIONS.get(c, c)}" for c in LCC_CODES
    )
    cat_table = "\n".join(f"- {c}" for c in LCGFT_CATEGORIES)
    (args.out_dir / "instructions.md").write_text(
        INSTRUCTIONS.format(n=len(sample), lcc_table=lcc_table, cat_table=cat_table)
    )
    (args.out_dir / "README.md").write_text(
        README.format(n=len(sample), seed=args.seed, source=args.source)
    )

    for coder in range(args.coders):
        name = f"coding_sheet_coder_{chr(ord('a') + coder)}.csv"
        with (args.out_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "annotation_id",
                    "lcc_code",
                    "lcgft_category",
                    "uncertain",
                    "unusable",
                    "notes",
                ]
            )
            for i in range(len(sample)):
                writer.writerow([f"doc_{i:04d}", "", "", "", "", ""])

    dist = Counter(d.get("lcc_code") for d in sample)
    console.print(f"[bold]{len(sample)} documents -> {args.out_dir}[/bold]")
    console.print(f"  LCC classes covered: {len(dist)} of {len(LCC_CODES)}")
    console.print(f"  per-class range: {min(dist.values())}-{max(dist.values())}")
    console.print(
        f"  generators represented: {len(Counter(d.get('model') for d in sample))}"
    )
    console.print(f"  coding sheets: {args.coders}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
