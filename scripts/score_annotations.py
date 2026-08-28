#!/usr/bin/env python
"""Score returned human annotations (Phase 5).

Reports three quantities, and the order matters:

1. **Inter-coder agreement** (Cohen's kappa). This is the headline. It bounds
   what any model can meaningfully achieve: if two trained cataloguers agree
   only 70% of the time on LCC class, a model scoring 0.89 against the generated
   labels is not 89% correct in any human sense -- it is agreeing with a label
   whose own reliability is 0.70.
2. **Each coder against the generated label.** Whether the corpus's labels are
   the ones a human would assign. A low number here is a finding about SHELF,
   not about the coder.
3. **The human ceiling** -- coders' mean agreement with the generated label --
   which is the figure model scores should be reported against.

Kappa is used rather than raw agreement because chance agreement is high with
21 classes and non-uniform usage, and raw percentages would flatter everyone.

Usage:
    uv run python scripts/score_annotations.py \\
        --coder-a coder_a.csv --coder-b coder_b.csv --gold gold.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table
from sklearn.metrics import cohen_kappa_score

console = Console()


def read_sheet(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row.get("annotation_id") or "").strip()
            if key:
                rows[key] = {k: (v or "").strip() for k, v in row.items()}
    return rows


def read_gold(path: Path) -> dict[str, dict]:
    gold: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                record = json.loads(line)
                gold[record["annotation_id"]] = record
    return gold


def paired(a: dict, b: dict, ids: list[str], field: str) -> tuple[list, list]:
    left, right = [], []
    for key in ids:
        x, y = a.get(key, {}).get(field, ""), b.get(key, {}).get(field, "")
        if x and y:
            left.append(x)
            right.append(y)
    return left, right


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coder-a", type=Path, required=True)
    parser.add_argument("--coder-b", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    args = parser.parse_args()

    a, b, gold = (
        read_sheet(args.coder_a),
        read_sheet(args.coder_b),
        read_gold(args.gold),
    )
    ids = sorted(set(a) & set(b) & set(gold))
    if not ids:
        console.print(
            "[red]No overlapping annotation_ids across the three files.[/red]"
        )
        return 1

    gold_rows = {
        k: {
            "lcc_code": gold[k].get("lcc_code") or "",
            "lcgft_category": gold[k].get("lcgft_category") or "",
        }
        for k in ids
    }

    table = Table(title=f"Human validation ({len(ids)} documents)")
    table.add_column("Task")
    table.add_column("Coder A vs B", justify="right")
    table.add_column("A vs generated", justify="right")
    table.add_column("B vs generated", justify="right")
    table.add_column("Human ceiling", justify="right")

    for field, label in (
        ("lcc_code", "LCC class"),
        ("lcgft_category", "LCGFT category"),
    ):
        ab = paired(a, b, ids, field)
        ag = paired(a, gold_rows, ids, field)
        bg = paired(b, gold_rows, ids, field)
        if not ab[0]:
            table.add_row(label, "no data", "-", "-", "-")
            continue
        k_ab = cohen_kappa_score(*ab)
        acc_ag = (
            sum(x == y for x, y in zip(*ag, strict=True)) / len(ag[0]) if ag[0] else 0.0
        )
        acc_bg = (
            sum(x == y for x, y in zip(*bg, strict=True)) / len(bg[0]) if bg[0] else 0.0
        )
        table.add_row(
            label,
            f"κ={k_ab:.3f}",
            f"{acc_ag:.3f}",
            f"{acc_bg:.3f}",
            f"{(acc_ag + acc_bg) / 2:.3f}",
        )
    console.print(table)

    for name, sheet in (("A", a), ("B", b)):
        unc = sum(
            1 for k in ids if sheet.get(k, {}).get("uncertain", "").lower() == "y"
        )
        unu = sum(1 for k in ids if sheet.get(k, {}).get("unusable", "").lower() == "y")
        console.print(
            f"  coder {name}: {unc} uncertain ({unc / len(ids) * 100:.1f}%), "
            f"{unu} unusable ({unu / len(ids) * 100:.1f}%)"
        )

    console.print(
        "\n[bold]Report model scores against the human ceiling, not against 1.0.[/bold]\n"
        "A κ below ~0.6 between trained coders means the labels themselves are the\n"
        "limiting factor and the task should be reported as label-noise-bound."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
