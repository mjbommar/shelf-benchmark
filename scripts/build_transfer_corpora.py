"""Stage the natural-text corpora used as transfer and rank-agreement controls.

Both are written in the schema the evaluators expect, so either can be
scored by pointing ``SHELF_DATA_DIR`` at its directory:

    SHELF_DATA_DIR=data/hf_dataset/transfer_gutenberg  python scripts/baselines/run_all.py ...

Corpora
    gutenberg   3,016 Project Gutenberg passages, human written and
                human catalogued. Full running text, so it is the closest
                natural analogue of a SHELF document and the primary
                transfer control.

    lcshbench   English records from LCSHBench (kltng/lcshbench, CC0),
                real catalogue records from Harvard, Columbia, and
                Princeton carrying real LCC classes.

**The two are not interchangeable.** Gutenberg is running prose; LCSHBench
is bibliographic metadata -- title, abstract, and table of contents, median
about 520 characters. A model that reads full documents well may read
catalogue stubs badly for reasons that have nothing to do with the
taxonomy, so report them separately and never pool them.

LCSHBench is read from raw JSONL rather than through ``datasets``: its
``heading_types`` field is a struct keyed by heading text, so schema
inference fails on the larger configs.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

LCSH_BASE = "https://huggingface.co/datasets/kltng/lcshbench/resolve/main"
SPLITS = ("train", "validation", "test")
# The 21 LCC classes SHELF uses; LCSHBench also emits an empty class.
VALID_LCC = set("ABCDEFGHJKLMNPQRSTUVZ")


def stratified_split(
    rows: list[dict[str, Any]],
    label_key: str,
    seed: int,
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> dict[str, list[dict[str, Any]]]:
    """Split within each label so every class appears in every split."""
    import random

    rng = random.Random(seed)
    by_label: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_label.setdefault(row[label_key], []).append(row)

    out: dict[str, list[dict[str, Any]]] = {s: [] for s in SPLITS}
    for label in sorted(by_label):
        group = by_label[label][:]
        rng.shuffle(group)
        n = len(group)
        n_train = int(n * fractions[0])
        n_val = int(n * fractions[1])
        out["train"].extend(group[:n_train])
        out["validation"].extend(group[n_train : n_train + n_val])
        out["test"].extend(group[n_train + n_val :])
    return out


def write(out_dir: Path, splits: dict[str, list[dict[str, Any]]], name: str) -> None:
    import polars as pl

    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"\n{name}")
    for split in SPLITS:
        rows = splits.get(split, [])
        if not rows:
            continue
        with (out_dir / f"{split}.jsonl").open("w") as f:
            for row in rows:
                f.write(json.dumps(row, default=str) + "\n")
        pl.DataFrame(rows, infer_schema_length=None).write_parquet(
            out_dir / f"{split}.parquet"
        )
        logger.info(f"  {split:<11} {len(rows):>6,}")
    total = sum(len(v) for v in splits.values())
    classes = {r["lcc_code"] for rows in splits.values() for r in rows}
    logger.info(f"  {'TOTAL':<11} {total:>6,}   ({len(classes)} LCC classes)")
    logger.info(f"  -> {out_dir}")


def build_gutenberg(src: Path, out_dir: Path, seed: int) -> None:
    rows = [json.loads(line) for line in src.open() if line.strip()]
    for row in rows:
        row["corpus"] = "gutenberg"
        row["is_synthetic"] = False
        if not (row.get("text") or "").strip():
            row["text"] = row.get("body") or ""
    rows = [r for r in rows if (r.get("text") or "").strip()]
    rows = [r for r in rows if r.get("lcc_code") in VALID_LCC]
    write(
        out_dir,
        stratified_split(rows, "lcc_code", seed),
        "Gutenberg (natural, full text)",
    )


def fetch(url: str, dest: Path) -> Path:
    if dest.exists():
        logger.info(f"  cached {dest.name}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"  downloading {url}")
    urllib.request.urlretrieve(url, dest)
    return dest


def build_lcshbench(out_dir: Path, cache: Path, seed: int, language: str) -> None:
    dev = fetch(f"{LCSH_BASE}/dev/dev.jsonl", cache / "lcsh_dev.jsonl")
    test = fetch(f"{LCSH_BASE}/test/test.jsonl", cache / "lcsh_test.jsonl")

    def convert(path: Path) -> list[dict[str, Any]]:
        out = []
        for line in path.open():
            if not line.strip():
                continue
            r = json.loads(line)
            if language and r.get("language") != language:
                continue
            code = (r.get("lc_class") or "").strip()
            if code not in VALID_LCC:
                continue
            # Catalogue records carry no body; the readable surface is the
            # title plus whatever descriptive fields the agency supplied.
            parts = [
                r.get("title") or "",
                r.get("abstract") or "",
                r.get("toc") or "",
                r.get("notes") or "",
            ]
            text = "\n\n".join(p.strip() for p in parts if p and p.strip())
            if not text.strip():
                continue
            out.append(
                {
                    "id": r.get("id"),
                    "title": (r.get("title") or "").strip(),
                    "body": text,
                    "text": text,
                    "lcc_code": code,
                    "language": r.get("language"),
                    "word_count": len(text.split()),
                    "corpus": "lcshbench",
                    "is_synthetic": False,
                    "n_agencies": r.get("n_agencies"),
                    "catalogs": r.get("catalogs"),
                }
            )
        return out

    dev_rows, test_rows = convert(dev), convert(test)
    # dev supplies train and validation; the published test split stays test.
    dev_split = stratified_split(dev_rows, "lcc_code", seed, (0.75, 0.25, 0.0))
    splits = {
        "train": dev_split["train"],
        "validation": dev_split["validation"],
        "test": test_rows,
    }
    label = f"LCSHBench ({language or 'all languages'}, catalogue metadata)"
    write(out_dir, splits, label)
    lens = sorted(len(r["text"]) for rows in splits.values() for r in rows)
    if lens:
        logger.info(
            f"  median text length {lens[len(lens) // 2]:,} chars "
            "-- metadata, not running prose; do not pool with Gutenberg"
        )
    logger.info(
        f"  class balance: {Counter(r['lcc_code'] for r in splits['test']).most_common(3)}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gutenberg-src", default="data/transfer/gutenberg/records.jsonl")
    ap.add_argument("--out-root", default="data/hf_dataset")
    ap.add_argument("--cache", default="data/transfer/cache")
    ap.add_argument("--language", default="English", help="'' for all languages")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.out_root)
    src = Path(args.gutenberg_src)
    if src.exists():
        build_gutenberg(src, root / "transfer_gutenberg", args.seed)
    else:
        logger.warning(f"skipping Gutenberg: {src} not found")

    build_lcshbench(
        root / "transfer_lcshbench", Path(args.cache), args.seed, args.language
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
