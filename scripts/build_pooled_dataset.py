"""Combine every synthetic SHELF corpus into one maximum-sample dataset.

The published configs are deliberately separate: `v0_4_core` is generator
balanced and `default` (v0.3.1) is not, so pooling them returns gpt-5.2 to
roughly half the corpus. That balance is worth keeping, so this script is
*additive* -- it writes a new `all` config and leaves every existing slice
untouched. Use `all` when sample count matters more than generator balance
(fitting probes, rank-agreement work); use `v0_4_core` when balance matters.

The Gutenberg slice is deliberately NOT pooled. It is the natural-text
transfer control, and folding it into the training corpus would destroy the
measurement it exists to support.

Sources
    v0.3.1  `default` from the hub          42,532
    v0.4    core / supplement / pairs / holdout  20,367
    ------------------------------------------------
    total                                    62,899

Schema is the union of both, so no column is dropped. Columns absent from a
source are null. `text` is always populated: v0.3.1 carries it, and for v0.4
it is filled from `body` (in v0.3.1 the two are identical anyway).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Local v0.4 slices, mapped to the config name they ship under.
V04_SOURCES: tuple[tuple[str, str], ...] = (
    ("v0_4_core", "data/hf_dataset/v0.4"),
    ("v0_4_supplement", "data/hf_dataset/v0.4_subclass"),
    ("v0_4_minimal_pairs", "data/hf_dataset/v0.4_minimal_pairs"),
    ("v0_4_holdout", "data/hf_dataset/v0.4_holdout"),
)

SPLITS = ("train", "validation", "test")

# Generators sometimes emit the markdown heading or the field label into the
# title. It is 0.8% of v0.4 and ~0% of v0.3.1, but title sits next to the
# label, so it gets normalised rather than left to a downstream probe.
_TITLE_NOISE = re.compile(r"^\s*#+\s*|^\s*(?:title|TITLE)\s*:\s*", re.I)


def clean_title(title: str | None) -> tuple[str, bool]:
    """Strip a leading markdown heading or 'Title:' label. Returns (title, changed)."""
    if not title:
        return "", False
    cleaned = title
    for _ in range(3):  # '# Title: X' needs two passes
        stripped = _TITLE_NOISE.sub("", cleaned, count=1)
        if stripped == cleaned:
            break
        cleaned = stripped
    cleaned = cleaned.strip()
    return cleaned, cleaned != (title or "").strip()


def normalise_model(model: str | None) -> str:
    """Fold provider-routing prefixes into one id per model.

    claude-opus-5 was reachable through both OpenRouter and the native API,
    so one model appears under two strings and splits its own share.
    provider_served still records which route served each row.
    """
    return (model or "").removeprefix("anthropic/")


def body_key(row: dict[str, Any]) -> str:
    """Content key for duplicate detection: normalised body text."""
    body = (row.get("body") or row.get("text") or "").strip().lower()
    return hashlib.sha256(" ".join(body.split()).encode()).hexdigest()


def load_v04() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for config, path in V04_SOURCES:
        base = Path(path)
        if not base.exists():
            logger.warning(f"  {config}: {base} missing, skipping")
            continue
        for split in SPLITS:
            f = base / f"{split}.jsonl"
            if not f.exists():
                continue
            for line in f.open():
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                row["source_config"] = config
                row["source_version"] = "0.4.0"
                row["split"] = split
                rows.append(row)
        logger.info(f"  {config}: loaded")
    return rows


def load_v031(cache_dir: str | None) -> list[dict[str, Any]]:
    from datasets import load_dataset

    kwargs: dict[str, Any] = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    ds = load_dataset("mjbommar/SHELF", "default", **kwargs)
    rows: list[dict[str, Any]] = []
    for split in SPLITS:
        if split not in ds:
            continue
        for row in ds[split]:
            row = dict(row)
            row["source_config"] = "default"
            row["source_version"] = "0.3.1"
            row["split"] = split
            rows.append(row)
    logger.info(f"  default (v0.3.1): loaded {len(rows):,}")
    return rows


def build(out_dir: Path, cache_dir: str | None) -> int:
    logger.info("Loading sources")
    rows = load_v031(cache_dir) + load_v04()
    logger.info(f"\nLoaded {len(rows):,} rows before cleaning")

    # Union schema, so nothing is dropped.
    columns: set[str] = set()
    for row in rows:
        columns.update(row)
    columns |= {"text", "source_config", "source_version", "split"}
    logger.info(f"Union schema: {len(columns)} columns")

    seen: dict[str, str] = {}
    kept: list[dict[str, Any]] = []
    titles_fixed = 0
    dupes = 0
    empty = 0

    for row in rows:
        row["model"] = normalise_model(row.get("model"))
        title, changed = clean_title(row.get("title"))
        titles_fixed += changed
        row["title"] = title

        body = (row.get("body") or "").strip()
        if not body:
            empty += 1
            continue
        # text is redundant with body in v0.3.1; make it always present.
        if not (row.get("text") or "").strip():
            row["text"] = row["body"]

        key = body_key(row)
        if key in seen:
            dupes += 1
            continue
        seen[key] = row.get("id", "")

        kept.append({col: row.get(col) for col in sorted(columns)})

    logger.info(f"  titles normalised : {titles_fixed:,}")
    logger.info(f"  empty bodies dropped: {empty:,}")
    logger.info(f"  duplicate bodies dropped: {dupes:,}")
    logger.info(f"  kept: {len(kept):,}")

    out_dir.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    handles = {s: (out_dir / f"{s}.jsonl").open("w") for s in SPLITS}
    try:
        for row in kept:
            split = row.get("split") or "train"
            if split not in handles:
                split = "train"
            handles[split].write(json.dumps(row, default=str) + "\n")
            counts[split] += 1
    finally:
        for h in handles.values():
            h.close()

    logger.info("\nPooled dataset")
    for split in SPLITS:
        logger.info(f"  {split:<11} {counts[split]:>7,}")
    logger.info(f"  {'TOTAL':<11} {sum(counts.values()):>7,}")

    by_source = Counter(r["source_config"] for r in kept)
    logger.info("\nBy source config")
    for src, n in by_source.most_common():
        logger.info(f"  {src:<20} {n:>7,}")

    by_model = Counter(r.get("model") or "?" for r in kept)
    top, top_n = by_model.most_common(1)[0]
    logger.info(
        f"\nGenerators: {len(by_model)}; largest share {top} "
        f"{top_n / len(kept) * 100:.1f}%  <- pooling costs balance, as expected"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", default="data/hf_dataset/all")
    ap.add_argument("--cache-dir", default="data/hf_cache")
    args = ap.parse_args()
    return build(Path(args.output_dir), args.cache_dir)


if __name__ == "__main__":
    sys.exit(main())
