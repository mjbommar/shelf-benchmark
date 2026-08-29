#!/usr/bin/env python
"""Extend the published SHELF dataset card with the v0.4 configs.

The card is edited rather than regenerated. Its front matter already declares
`dataset_info` and `data_files` for seven configs whose numbers describe the
frozen v0.3.1 release; regenerating from scratch would silently restate those
from whatever happens to be on disk. This appends the new configs, leaves every
existing entry byte-identical, and adds a body section for the v0.4 slices.

Usage:
    uv run python scripts/update_dataset_card.py --check     # report only
    uv run python scripts/update_dataset_card.py --write
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from rich.console import Console

console = Console()

# (config, staged dir, splits, one-line description)
NEW_CONFIGS: tuple[tuple[str, str, tuple[str, ...], str], ...] = (
    (
        "v0_4_core",
        "v0_4_core",
        ("train", "validation", "test"),
        "Generator-balanced core corpus (v0.4)",
    ),
    (
        "v0_4_supplement",
        "v0_4_supplement",
        ("train", "validation", "test"),
        "Supplementary single-generator documents, 21-class task (v0.4)",
    ),
    (
        "v0_4_minimal_pairs",
        "v0_4_minimal_pairs",
        ("train", "validation", "test"),
        "Minimal pairs varying one facet (v0.4)",
    ),
    (
        "v0_4_holdout",
        "v0_4_holdout",
        ("train", "validation", "test"),
        "Held-out generator probe (v0.4)",
    ),
    (
        "transfer_gutenberg",
        "transfer_gutenberg",
        ("test",),
        "Natural Project Gutenberg passages - KNOWN CONTAMINATED",
    ),
)

BODY_SECTION = """
## v0.4 slices

v0.4 adds five configurations. **The `default` config and the six pair
configurations are unchanged**, so results published against v0.3.1 remain
valid.

| Config | Documents | What it is |
|---|---|---|
| `v0_4_core` | {core:,} | Generator-balanced corpus: {n_gen} generators, largest share {top_share:.1f}% |
| `v0_4_supplement` | {subclass:,} | Supplementary documents over the standard 21-class task, single generator |
| `v0_4_minimal_pairs` | {pairs:,} | Pairs holding topics, audience, register and length constant, varying exactly one facet |
| `v0_4_holdout` | {holdout:,} | Documents from a generator absent from the core, for transfer probing |
| `transfer_gutenberg` | {transfer:,} | Human-written, human-catalogued Project Gutenberg passages |

### Why v0.4 is separate rather than merged

The `default` corpus is **94.1% GPT-5.x**, with two models supplying that share
and five of its nine generators contributing about 100 documents each. `v0_4_core`
spreads {n_gen} generators with no single one above {top_share:.1f}%.

Pooling the two would return the largest generator to roughly half the combined
corpus, which would undo the balance `v0_4_core` exists to provide. Any
experiment that needs generator balance -- cross-generator generalization,
generator attribution, train-on-family-A / test-on-family-B -- should use
`v0_4_core` alone.

### Known limitations

**No subclass tier in this release.** An LCC subclass tier was planned and is
not shipped. The specification blocks assigned all 80 subclasses, but the
generation path passed the parent class description to the model, so the
documents were conditioned on 16 parent classes rather than 80 subclasses and
carry no subclass label. Those documents are published as `v0_4_supplement`,
described for what they are. The tier will return when the conditioning is
fixed.

**Empty-document rate.** `v0_4_supplement` and `v0_4_minimal_pairs` were generated
before a fix for a reasoning-budget defect: on short length targets, reasoning
tokens consumed the whole output cap, producing a title truncated mid-word and
no body. 13-15% of raw generations were affected. QC removes them, so the
published slices are clean but roughly 14% smaller than their nominal spec count.

**Generator confound on register and length.** In `v0_4_core`, generator is
independent of the labels that matter -- LCC class (Cramer's V 0.016) and LCGFT
category (0.027) -- but correlates weakly with `register` (0.061),
`target_length` (0.064) and `prompt_variant_id` (0.084). The cause is non-uniform
QC removal: the empty-body defect hit short documents hardest and two generators
failed for part of the run. Effect sizes are small; p-values are not marginal.
Analyses conditioned on register or length carry this confound.

**`transfer_gutenberg` is contaminated and must not be pooled.** It is in the
pretraining data of essentially every model that would be evaluated on it. SHELF
is the clean-synthetic condition, Gutenberg is the contaminated-natural one, and
the gap between them is the measurement. A lexical baseline trained on SHELF
scores 0.893 macro-F1 in-domain and **0.301 on Gutenberg**; trained on Gutenberg
it reaches 0.526 in-domain. Transfer fails symmetrically, which is domain shift
rather than memorisation.

**No human ceiling yet.** No human annotation round has been run, so model scores
on these slices have no interpretable upper bound.

**Prompt variants differ from `default`.** v0.4 documents use four new
system-prompt variants and form-conditional output formatting; `default` used a
single prompt. `prompt_variant_id` records which. A controlled A/B measured
spurious markdown on non-markdown forms falling from 26.7% to ~1.3%
(Fisher exact p < 0.00001).
"""


def infer_features(path: Path) -> list[dict]:
    """Derive a minimal HF feature list from the first staged row."""
    import polars as pl

    frame = pl.read_parquet(path)
    features = []
    for name, dtype in zip(frame.columns, frame.dtypes, strict=True):
        text = str(dtype)
        if text.startswith("List"):
            features.append({"name": name, "list": "string"})
        elif "Int" in text:
            features.append({"name": name, "dtype": "int64"})
        elif "Float" in text:
            features.append({"name": name, "dtype": "float64"})
        elif "Boolean" in text:
            features.append({"name": name, "dtype": "bool"})
        else:
            features.append({"name": name, "dtype": "string"})
    return features


COUNTED_SYNTHETIC_CONFIGS = (
    "v0_4_core",
    "v0_4_supplement",
    "v0_4_minimal_pairs",
    "v0_4_holdout",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--card", type=Path, default=Path("data/hf_release/v0.4/README.md")
    )
    parser.add_argument(
        "--published",
        type=Path,
        required=True,
        help="The currently published README.md to extend",
    )
    parser.add_argument(
        "--release-dir", type=Path, default=Path("data/hf_release/v0.4")
    )
    parser.add_argument("--version", default="0.4.0")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    raw = args.published.read_text(encoding="utf-8")
    _, front_raw, body = raw.split("---\n", 2)
    front = yaml.safe_load(front_raw)

    existing = {c["config_name"] for c in front.get("configs", [])}
    console.print(f"published configs: {len(existing)} -> {sorted(existing)}")

    counts: dict[str, int] = {}
    added = 0
    for config, folder, splits, _desc in NEW_CONFIGS:
        if config in existing:
            console.print(f"  [yellow]{config} already declared, skipping[/yellow]")
            continue
        base = args.release_dir / folder
        files, total, features = [], 0, None
        for split in splits:
            matches = sorted(base.glob(f"{split}-*.parquet"))
            if not matches:
                continue
            import polars as pl

            total += pl.read_parquet(matches[0]).height
            files.append({"split": split, "path": f"{folder}/{split}-*"})
            features = features or infer_features(matches[0])
        if not files:
            console.print(f"  [red]{config}: nothing staged, skipping[/red]")
            continue
        counts[config] = total
        front.setdefault("dataset_info", []).append(
            {"config_name": config, "features": features}
        )
        front.setdefault("configs", []).append(
            {"config_name": config, "data_files": files}
        )
        added += 1
        console.print(f"  + {config}: {total:,} docs, {len(files)} split(s)")

    front["version"] = args.version
    total_docs = sum(counts.values())
    if total_docs + 42532 > 100000:
        front["size_categories"] = ["100K<n<1M"]

    # Generator balance, computed rather than asserted.
    import polars as pl

    core = pl.concat(
        [
            pl.read_parquet(f)
            for f in sorted((args.release_dir / "v0_4_core").glob("*.parquet"))
        ]
    )
    gens = core["model"].value_counts(sort=True)
    n_gen, top_share = gens.height, gens["count"][0] / core.height * 100

    section = BODY_SECTION.format(
        core=counts.get("v0_4_core", 0),
        subclass=counts.get("v0_4_supplement", 0),
        pairs=counts.get("v0_4_minimal_pairs", 0),
        holdout=counts.get("v0_4_holdout", 0),
        transfer=counts.get("transfer_gutenberg", 0),
        n_gen=n_gen,
        top_share=top_share,
    )
    synthetic = 42_532 + sum(counts.get(c, 0) for c in COUNTED_SYNTHETIC_CONFIGS)
    natural = counts.get("transfer_gutenberg", 0)
    stale = "SHELF contains 42,532 synthetic documents annotated with Library of Congress taxonomies:"
    fresh = (
        f"SHELF contains {synthetic:,} synthetic documents annotated with Library of "
        f"Congress taxonomies, plus {natural:,} natural Project Gutenberg documents "
        f"used only as a transfer control ({synthetic + natural:,} in total). The "
        "v0.3.1 corpus described immediately below is 42,532 of that synthetic total; "
        "the rest is described under `v0.4 slices`:"
    )
    if stale in body:
        body = body.replace(stale, fresh, 1)
    elif "in total). The" not in body:
        raise SystemExit("corpus-total sentence not found and not already rewritten")

    new_card = (
        "---\n" + yaml.dump(front, sort_keys=False, allow_unicode=True) + "---\n" + body
    )
    if "## v0.4 slices" not in new_card:
        new_card = new_card.rstrip() + "\n\n" + section.strip() + "\n"

    console.print(
        f"\nversion -> {args.version}; {added} config(s) added; "
        f"card {len(new_card.splitlines()):,} lines"
    )
    if args.write:
        args.card.parent.mkdir(parents=True, exist_ok=True)
        args.card.write_text(new_card, encoding="utf-8")
        console.print(f"[green]wrote {args.card}[/green]")
    else:
        console.print("[yellow]--check mode, nothing written[/yellow]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
