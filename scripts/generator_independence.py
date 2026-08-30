"""Is the writing model independent of the label?

The corpus draws generator, genre, and subject separately, so a reader should
be able to check that the writing model carries no information about the
label. This measures Cramer's V between the generator and each label, with
the chi-square p-value beside it.

The figure belongs to a slice. On the generator-balanced ``v0_4_core`` the
association is negligible; across the wider v0.4 generation, which adds
single-generator supplementary documents, it rises. Report the slice with
the number.

Normalises the ``anthropic/`` prefix first. The raw files carry
``claude-opus-5`` and ``anthropic/claude-opus-5`` as two strings for one
model, so an un-normalised count finds one generator too many and understates
every share.

Usage:
    uv run python scripts/generator_independence.py
    uv run python scripts/generator_independence.py --output results/generator_independence.json
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import chi2_contingency

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

LABELS = ("lcc_code", "lcgft_category")

# v0_4_core is the balanced slice; the wider generation is every v0.4 slice.
CORE_DIR = "v0.4"
WIDE_DIRS = ("v0.4", "v0.4_subclass", "v0.4_holdout", "v0.4_minimal_pairs")


def load(name: str) -> pl.DataFrame:
    files = sorted(glob.glob(f"data/hf_dataset/{name}/*.jsonl")) or sorted(
        glob.glob(f"data/hf_dataset/{name}/*.parquet")
    )
    if not files:
        raise FileNotFoundError(f"no data under data/hf_dataset/{name}")
    read = pl.read_ndjson if files[0].endswith("jsonl") else pl.read_parquet
    return pl.concat([read(f) for f in files], how="diagonal_relaxed")


def normalise(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(pl.col("model").str.replace(r"^anthropic/", "").alias("m"))


def cramers_v(df: pl.DataFrame, label: str) -> tuple[float, float, int]:
    tab = df.pivot(values="m", index="m", on=label, aggregate_function="len")
    m = tab.drop("m").fill_null(0).to_numpy().astype(float)
    chi2, p, _, _ = chi2_contingency(m)
    n = m.sum()
    return float(np.sqrt(chi2 / (n * (min(m.shape) - 1)))), float(p), int(n)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    slices = {
        "v0_4_core": normalise(load(CORE_DIR)),
        "v0_4_generation": normalise(
            pl.concat([load(d) for d in WIDE_DIRS], how="diagonal_relaxed")
        ),
    }

    rows = []
    logger.info(
        f"{'slice':<18}{'label':<18}{'n':>8}{'generators':>12}{'V':>9}{'p':>12}"
    )
    logger.info("-" * 77)
    for sname, df in slices.items():
        for label in LABELS:
            if label not in df.columns:
                continue
            v, p, n = cramers_v(df, label)
            g = df["m"].n_unique()
            logger.info(f"{sname:<18}{label:<18}{n:>8,}{g:>12}{v:>9.4f}{p:>12.3g}")
            rows.append(
                {
                    "slice": sname,
                    "label": label,
                    "n": n,
                    "generators": g,
                    "cramers_v": v,
                    "chi2_p": p,
                }
            )

    logger.info(
        "\nCramer's V is 0 when the generator says nothing about the label. "
        "The core slice is the one to cite for generator independence; the "
        "wider generation adds single-generator documents and is higher."
    )

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "measure": "Cramer's V between generation model and label",
                    "anthropic_prefix_normalised": True,
                    "core_slice": CORE_DIR,
                    "wide_slices": list(WIDE_DIRS),
                    "rows": rows,
                },
                indent=2,
            )
        )
        logger.info(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
