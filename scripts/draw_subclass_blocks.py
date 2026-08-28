#!/usr/bin/env python
"""Draw spec blocks whose documents carry an LCC subclass (Phase 2).

`DocumentSampler` draws topics *for the class it drew*, so the subclass has to
be chosen first and the document constrained to its parent. Swapping the class
in afterwards would leave a document with, say, literature topics under a
mathematics label.

Uses the caption-hierarchy descriptions deliberately: a controlled A/B showed
the prose rewrite raised both the description copy rate (29.3% -> 35.5%) and the
TF-IDF baseline (0.639 -> 0.724), so it is not shipped.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from shelf.sampler.dimensions import LCCSubclassSampler
from shelf.sampler.document import DocumentSampler
from shelf.sampler.generator import LengthSampler, PromptVariantSampler, RegisterSampler
from shelf.sampler.specs import DocumentSpec, SpecBlock, save_spec_block

console = Console()


def draw(n_blocks: int, per_block: int, base_seed: int) -> list[SpecBlock]:
    blocks: list[SpecBlock] = []
    for index in range(n_blocks):
        seed = base_seed + index
        block_id = f"subclass-block-{index:02d}-seed{seed}"
        subclasses = LCCSubclassSampler(seed=seed)
        docs = DocumentSampler(seed=seed)
        lengths = LengthSampler(seed=seed)
        registers = RegisterSampler(seed=seed)
        variants = PromptVariantSampler(seed=seed)

        specs = []
        for _ in range(per_block):
            lcc = subclasses.sample()
            document = docs.with_lcc_classes([lcc.code]).sample()
            document.lcc = lcc
            specs.append(
                DocumentSpec.from_document(
                    document,
                    target_length=lengths.sample(),
                    register=registers.sample(),
                    prompt_variant=variants.sample(),
                    block_id=block_id,
                )
            )
        blocks.append(
            SpecBlock(
                block_id=block_id,
                seed=seed,
                specs=tuple(specs),
                sampler_config={"specs_per_block": per_block, "tier": "lcc_subclass"},
            )
        )
    return blocks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-blocks", type=int, default=2)
    parser.add_argument("--specs-per-block", type=int, default=750)
    parser.add_argument("--base-seed", type=int, default=1000)
    parser.add_argument(
        "--out-dir", type=Path, default=Path("data/artifacts/spec_blocks_subclass")
    )
    args = parser.parse_args()

    blocks = draw(args.n_blocks, args.specs_per_block, args.base_seed)
    for block in blocks:
        path = save_spec_block(block, args.out_dir / f"{block.block_id}.jsonl")
        subclasses = {s.lcc_subclass for s in block.specs}
        console.print(
            f"  {path}  {len(block)} specs, {len(subclasses)} distinct subclasses, "
            f"dups={len(block.duplicate_spec_ids())}"
        )
    total = {s.lcc_subclass for b in blocks for s in b.specs}
    console.print(
        f"[bold]{sum(len(b) for b in blocks)} specs, {len(total)} subclasses[/bold]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
