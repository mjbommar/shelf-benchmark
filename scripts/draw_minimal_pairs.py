#!/usr/bin/env python
"""Draw minimal-pair spec blocks (Phase 3).

A minimal pair holds topics, audience, register and length constant and varies
**exactly one** facet. That is what makes it a control: any difference a model
detects between the two documents is attributable to the varied facet rather
than to incidental vocabulary.

This attacks the shortcut measured throughout the plan. `same_lcc_pairs` as
shipped is 100% S2 positives against 81% S6 negatives -- "same subject vs.
totally unrelated" -- so topical similarity alone nearly solves it. A pair that
shares its topics and differs only in form cannot be solved that way.

Two axes are drawn:

* ``form``    -- same subject, same topics, different LCGFT form
* ``subject`` -- same form, same topics, different LCC class

Both members of a pair share a ``pair_id``; ``pair_role`` is ``a`` or ``b``.
Because the two members are deliberately near-identical in specification, they
must never straddle a split -- group on ``pair_id`` exactly as Phase 1 groups on
``spec_id``.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import replace
from pathlib import Path

from rich.console import Console
from shelf.sampler.document import DocumentSampler
from shelf.sampler.generator import LengthSampler, PromptVariantSampler, RegisterSampler
from shelf.sampler.specs import DocumentSpec, SpecBlock, save_spec_block

console = Console()


def draw_pairs(n_pairs: int, seed: int, block_id: str) -> list[DocumentSpec]:
    rng = random.Random(seed)
    docs = DocumentSampler(seed=seed)
    lengths = LengthSampler(seed=seed)
    registers = RegisterSampler(seed=seed)
    variants = PromptVariantSampler(seed=seed)

    specs: list[DocumentSpec] = []
    for index in range(n_pairs):
        axis = "form" if index % 2 == 0 else "subject"
        base = docs.sample()
        length, register, variant = (
            lengths.sample(),
            registers.sample(),
            variants.sample(),
        )
        pair_id = f"{block_id}-pair-{index:05d}"

        a = DocumentSpec.from_document(
            base,
            target_length=length,
            register=register,
            prompt_variant=variant,
            block_id=block_id,
        )

        # Redraw only the varied facet; everything else is carried across.
        for _ in range(60):
            other = docs.sample()
            if axis == "form" and other.lcgft.form != base.lcgft.form:
                b = replace(
                    a,
                    lcgft_form=other.lcgft.form,
                    lcgft_category=other.lcgft.category,
                )
                break
            if axis == "subject" and other.lcc.code != base.lcc.code:
                b = replace(a, lcc_code=other.lcc.code, lcc_name=other.lcc.name)
                break
        else:
            continue

        specs.append(replace(a, pair_id=pair_id, pair_role="a", pair_axis=axis))
        specs.append(replace(b, pair_id=pair_id, pair_role="b", pair_axis=axis))
        rng.random()
    return specs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-pairs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=3000)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/artifacts/spec_blocks_pairs/minimal-pairs-00.jsonl"),
    )
    args = parser.parse_args()

    block_id = "minimal-pairs-00"
    specs = draw_pairs(args.n_pairs, args.seed, block_id)
    block = SpecBlock(
        block_id=block_id,
        seed=args.seed,
        specs=tuple(specs),
        sampler_config={"tier": "minimal_pairs", "n_pairs": len(specs) // 2},
    )
    path = save_spec_block(block, args.out)
    axes = {}
    for spec in specs:
        axes[spec.pair_axis] = axes.get(spec.pair_axis, 0) + 1
    console.print(f"  {path}")
    console.print(
        f"  {len(specs)} specs = {len(specs) // 2} pairs; by axis {axes}; "
        f"duplicate spec_ids={len(block.duplicate_spec_ids())}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
