"""Measure how much label signal sits on the surface of a corpus.

Counts how often a label's exact name appears in its own document. Reported
alone the number means nothing: a document about birds contains the word
"birds", which is why extractive summarisation works. It becomes readable
only beside a natural corpus measured the same way.

Exhaustive by default -- no subsampling -- so the number is exact and a
rerun reproduces it. Length truncation is offered because a longer document
has more chances to contain any given string.

    python scripts/surface_signal.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

SPLITS = ("train", "validation", "test")


def load(path: Path, cols: list[str]):
    import polars as pl

    frames = []
    for s in SPLITS:
        f = path / f"{s}.parquet"
        if f.exists():
            have = [c for c in cols if c in pl.read_parquet_schema(f)]
            frames.append(pl.read_parquet(f, columns=have))
    if not frames:
        raise FileNotFoundError(path)
    return pl.concat(frames, how="diagonal_relaxed").to_dicts()


def rate(rows, field: str, trunc: int | None = None, subset: str | None = None):
    """Exact share of label values appearing verbatim in their own document."""
    hit = tot = 0
    for r in rows:
        if subset is not None and r.get("source_config") != subset:
            continue
        value = r.get(field)
        if value is None:
            continue
        text = r.get("text") or ""
        if trunc:
            text = " ".join(text.split()[:trunc])
        text = text.lower()
        for v in value if isinstance(value, list) else [value]:
            if not v:
                continue
            tot += 1
            if str(v).lower() in text:
                hit += 1
    return (hit / tot * 100 if tot else 0.0), hit, tot


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--synthetic", default="data/hf_dataset/all")
    ap.add_argument("--natural", default="data/hf_dataset/transfer_gutenberg")
    ap.add_argument("--trunc", type=int, default=200)
    ap.add_argument("--output", default="results/transfer/surface_signal.json")
    args = ap.parse_args()

    cols = ["text", "lcc_name", "topics", "lcgft_form", "source_config"]
    nat = load(Path(args.natural), cols)
    syn = load(Path(args.synthetic), cols)
    report: dict = {"trunc_words": args.trunc, "exhaustive": True}

    logger.info(f"natural  n={len(nat):,}   synthetic n={len(syn):,}\n")
    logger.info(
        f"{'corpus':<22}{'full text':>12}{'first ' + str(args.trunc) + 'w':>14}"
    )
    logger.info("-" * 48)
    for label, rows in (("Gutenberg (natural)", nat), ("SHELF (all)", syn)):
        full, h1, t1 = rate(rows, "lcc_name")
        cut, h2, t2 = rate(rows, "lcc_name", args.trunc)
        logger.info(f"{label:<22}{full:>11.1f}%{cut:>13.1f}%")
        report[label] = {"full": full, "truncated": cut, "n": t1}

    logger.info(f"\nBy generation, first {args.trunc} words (exhaustive):")
    logger.info(f"{'generation':<14}{'lcc_name':>10}{'topics':>9}{'form':>8}")
    logger.info("-" * 41)
    for cfg, label in (("default", "first"), ("v0_4_core", "second")):
        vals = {}
        for f in ("lcc_name", "topics", "lcgft_form"):
            vals[f], _, _ = rate(syn, f, args.trunc, subset=cfg)
        logger.info(
            f"{label:<14}{vals['lcc_name']:>9.1f}%{vals['topics']:>8.1f}%"
            f"{vals['lcgft_form']:>7.1f}%"
        )
        report[f"generation_{label}"] = vals

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    logger.info(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
