"""How much of each corpus does an encoder actually read?

Every dense model in the sweep has a maximum sequence length, and every
document longer than it is cut. That cut is not evenly distributed across the
three corpora, so a cross-corpus comparison of dense models is partly a
comparison of how much text each model was allowed to see.

The lexical baselines have no such limit. The transfer matrix is built from
TF-IDF, so nothing here touches it.

Usage:
    uv run python scripts/truncation_audit.py
    uv run python scripts/truncation_audit.py --output results/truncation.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import statistics
from pathlib import Path

logger = logging.getLogger(__name__)

# Representative of the sweep: 12 of 20 dense models cap at 512.
TOKENIZER = "BAAI/bge-base-en-v1.5"
CAPS = (256, 384, 512, 1024)

CORPORA = {
    "shelf": "data/hf_dataset/all",
    "gutenberg": "data/hf_dataset/transfer_gutenberg",
    "lcshbench": "data/hf_dataset/transfer_lcshbench",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", default="test")
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    import polars as pl
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    rng = random.Random(args.seed)

    rows = []
    header = (
        f"{'corpus':<12}{'n':>6}{'median':>8}{'mean':>8}{'p90':>8}"
        + "".join(f"{'>' + str(c):>8}" for c in CAPS)
        + f"{'kept@512':>10}"
    )
    logger.info(header)
    logger.info("-" * len(header))

    for name, d in CORPORA.items():
        path = Path(d) / f"{args.split}.parquet"
        if not path.exists():
            logger.warning(f"{name:<12} missing {path}")
            continue
        df = pl.read_parquet(path)
        col = "text" if "text" in df.columns else "body"
        texts = [t for t in df[col].to_list() if t]
        sample = rng.sample(texts, min(args.sample, len(texts)))
        lens = [
            len(tok.encode(t, truncation=False, add_special_tokens=True))
            for t in sample
        ]
        n = len(lens)

        over = {c: 100.0 * sum(1 for x in lens if x > c) / n for c in CAPS}
        # Share of all tokens an encoder capped at 512 actually reads.
        kept512 = 100.0 * sum(min(x, 512) for x in lens) / sum(lens)

        logger.info(
            f"{name:<12}{n:>6}{statistics.median(lens):>8.0f}{statistics.mean(lens):>8.0f}"
            f"{sorted(lens)[int(0.9 * n)]:>8.0f}"
            + "".join(f"{over[c]:>7.1f}%" for c in CAPS)
            + f"{kept512:>9.1f}%"
        )
        rows.append(
            {
                "corpus": name,
                "n_sampled": n,
                "median_tokens": statistics.median(lens),
                "mean_tokens": statistics.mean(lens),
                "p90_tokens": sorted(lens)[int(0.9 * n)],
                "pct_over": over,
                "pct_tokens_kept_at_512": kept512,
            }
        )

    logger.info(
        "\nkept@512 is the share of tokens an encoder capped at 512 reads. "
        "It differs by corpus, so dense cross-corpus comparisons are not "
        "matched on how much text the model saw. Lexical baselines read "
        "the whole document and are unaffected."
    )

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "tokenizer": TOKENIZER,
                    "split": args.split,
                    "sample": args.sample,
                    "seed": args.seed,
                    "caps": list(CAPS),
                    "rows": rows,
                },
                indent=2,
            )
        )
        logger.info(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
