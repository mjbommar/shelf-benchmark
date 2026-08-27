"""Publish the pooled `all` config to the SHELF dataset repo.

Additive. Existing configs are untouched, including the generator-balanced
`v0_4_core` and the v0.3.1 `default`. The pooled config exists for work
where sample count dominates -- fitting probes, rank agreement -- and it
carries the imbalance that pooling creates, stated on the card rather than
hidden.

Nothing uploads without --upload.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

SPLITS = ("train", "validation", "test")

CARD_SECTION = """

## The `all` config

A single pooled corpus of every synthetic SHELF document: the v0.3.1
`default` corpus plus every v0.4 slice.

| | documents |
|---|---|
| `default` (v0.3.1) | 42,532 |
| `v0_4_core` | 18,345 |
| `v0_4_supplement` | 1,043 |
| `v0_4_minimal_pairs` | 687 |
| `v0_4_holdout` | 292 |
| **`all`** | **62,899** |

Splits: train {train:,} / validation {validation:,} / test {test:,}. Each
document keeps the split it was assigned in its source config.

**This config is not generator balanced, and that is the trade.** Pooling
returns the largest generator to {top_share:.1f}% of the corpus, against
9.2% in `v0_4_core`. Use `all` when sample count matters more than
balance, and `v0_4_core` when it does not. Reporting a generator-sensitive
result on `all` without saying so would be misleading.

Every row carries `source_config` and `source_version`, so any component
slice can be recovered exactly:

```python
from datasets import load_dataset
ds = load_dataset("mjbommar/SHELF", "all")
core = ds["train"].filter(lambda r: r["source_config"] == "v0_4_core")
```

Schema is the union of both generations ({n_cols} columns), so no column is
dropped; columns absent from a source are null. `text` is always
populated. Provider routing prefixes are normalised, so one model is one
id. Titles carrying a leading markdown heading or `Title:` label were
cleaned (169 rows). Deduplicated on normalised body text: zero duplicates
were found across the two corpora, as expected from disjoint spec blocks.

**The Gutenberg transfer control is deliberately excluded.** It is natural
text used to measure whether SHELF scores transfer, and pooling it into
the corpus would destroy that measurement. It remains a separate config.
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="data/hf_dataset/all")
    ap.add_argument("--release-dir", default="data/hf_release/pooled")
    ap.add_argument("--repo-id", default="mjbommar/SHELF")
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()

    src, out = Path(args.source), Path(args.release_dir)
    if not src.exists():
        logger.error(f"{src} not built. Run scripts/build_pooled_dataset.py first.")
        return 1

    target = out / "all"
    target.mkdir(parents=True, exist_ok=True)

    import polars as pl

    counts: dict[str, int] = {}
    models: Counter[str] = Counter()
    n_cols = 0
    for split in SPLITS:
        pq = src / f"{split}.parquet"
        if not pq.exists():
            continue
        df = pl.read_parquet(pq)
        counts[split] = df.height
        n_cols = max(n_cols, df.width)
        if "model" in df.columns:
            models.update(df["model"].to_list())
        df.write_parquet(target / f"{split}-00000-of-00001.parquet")
        logger.info(f"  {split:<11} {df.height:>7,} staged")

    total = sum(counts.values())
    top_share = (models.most_common(1)[0][1] / total * 100) if models else 0.0
    logger.info(f"\n  TOTAL {total:,} documents, {len(models)} generators")
    logger.info(f"  largest generator share {top_share:.1f}%")

    section = CARD_SECTION.format(
        train=counts.get("train", 0),
        validation=counts.get("validation", 0),
        test=counts.get("test", 0),
        top_share=top_share,
        n_cols=n_cols,
    )
    (out / "ALL_CONFIG.md").write_text(section.strip() + "\n")
    logger.info(f"  card section -> {out / 'ALL_CONFIG.md'}")

    entry = {
        "config_name": "all",
        "data_files": [
            {"split": s, "path": f"all/{s}-00000-of-00001.parquet"}
            for s in SPLITS
            if s in counts
        ],
    }
    (out / "config_entry.json").write_text(json.dumps(entry, indent=2))

    if not args.upload:
        logger.info("\nDRY RUN - nothing uploaded. Re-run with --upload.")
        return 0

    from huggingface_hub import HfApi

    api = HfApi()
    api.upload_folder(
        folder_path=str(target),
        path_in_repo="all",
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message=f"Add pooled 'all' config ({total:,} synthetic documents)",
    )
    logger.info(f"\nUploaded to https://huggingface.co/datasets/{args.repo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
