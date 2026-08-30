"""Is the sham arm a fair control, and how hard does masking bite?

Two questions a reader should not have to take on trust.

Is the control matched? The sham removes the same number of tokens as the
masked arm, chosen at random instead of by label. If the counts differ, a
score gap could be document damage rather than leakage. Checked per document,
not on the mean: means can agree while individual documents do not.

How much of each corpus does masking touch? Only documents that contain their
own label terms are affected. That share differs sharply between corpora, so
the masking arm is a stronger test on some than on others, and the ablation
should be read with that in mind.

Note on arithmetic: word counts subtract to negative values on some rows, and
doing this in unsigned integers wraps them to 4294967295. Cast first.

Usage:
    uv run python scripts/masking_corpora_audit.py
    uv run python scripts/masking_corpora_audit.py --output results/masking_corpora_audit.json
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import statistics
from pathlib import Path

import polars as pl

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

BASES = ("all", "transfer_gutenberg", "transfer_lcshbench")


def load(name: str) -> pl.DataFrame | None:
    files = sorted(glob.glob(f"data/hf_dataset/{name}/*.parquet"))
    if not files:
        return None
    return pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    rows = []
    for base in BASES:
        u, m, s = load(base), load(f"{base}_masked"), load(f"{base}_sham")
        if any(x is None for x in (u, m, s)):
            logger.warning(f"{base}: an arm is missing; skipping")
            continue
        col = "text" if "text" in u.columns else "body"
        wu = [len(t.split()) for t in u[col].to_list()]
        wm = [len(t.split()) for t in m[col].to_list()]
        ws = [len(t.split()) for t in s[col].to_list()]
        if not (len(wu) == len(wm) == len(ws)):
            logger.warning(f"{base}: arms differ in row count; skipping")
            continue

        # Signed, deliberately: see the note above.
        dm = [a - b for a, b in zip(wu, wm, strict=True)]
        ds = [a - b for a, b in zip(wu, ws, strict=True)]
        matched = sum(1 for a, b in zip(dm, ds, strict=True) if a == b)
        touched = sum(1 for a in dm if a > 0)
        n = len(dm)

        rows.append(
            {
                "corpus": base,
                "n_documents": n,
                "sham_matched_documents": matched,
                "sham_matched_pct": 100.0 * matched / n,
                "documents_masking_touches": touched,
                "documents_masking_touches_pct": 100.0 * touched / n,
                "mean_tokens_removed": statistics.mean(dm),
                "mean_words_unmasked": statistics.mean(wu),
            }
        )

    logger.info(
        f"{'corpus':<20}{'n':>8}{'sham matched':>14}{'masking touches':>17}{'tokens cut':>12}"
    )
    logger.info("-" * 71)
    for r in rows:
        logger.info(
            f"{r['corpus']:<20}{r['n_documents']:>8,}"
            f"{r['sham_matched_pct']:>13.1f}%"
            f"{r['documents_masking_touches_pct']:>16.1f}%"
            f"{r['mean_tokens_removed']:>12.2f}"
        )

    logger.info(
        "\nsham matched must be 100%: each document loses the same count in "
        "both arms, so a masked-sham gap is about which tokens went, not how "
        "many. masking touches is the share of documents containing their own "
        "label terms; where it is low the masked arm is a weaker test."
    )

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"rows": rows}, indent=2))
        logger.info(f"  wrote {out}")

    return 0 if all(r["sham_matched_pct"] == 100.0 for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
