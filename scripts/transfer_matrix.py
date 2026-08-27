"""Measure how far LCC classification transfers between corpora.

The paper's central result. A lexical classifier is fit on one corpus and
scored on another, in every direction, so the question "does a SHELF score
predict performance on real catalogue text" gets a number rather than an
argument.

TF-IDF with logistic regression is the right probe here precisely because
it is weak: it has no pretraining, so contamination in the natural corpora
cannot explain any of the result. Whatever gap appears is distributional.

    python scripts/transfer_matrix.py \\
        --corpus shelf=data/hf_dataset/all \\
        --corpus gutenberg=data/hf_dataset/transfer_gutenberg \\
        --corpus lcshbench=data/hf_dataset/transfer_lcshbench

Every cell is macro-F1 over the 21 LCC classes. Read the diagonal as the
in-domain ceiling for that corpus and the off-diagonal as transfer.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def load_corpus(path: Path, max_rows: int | None) -> tuple[list[str], list[str]]:
    import polars as pl

    frames = []
    for split in ("train", "validation", "test"):
        f = path / f"{split}.parquet"
        if f.exists():
            frames.append(pl.read_parquet(f, columns=["text", "lcc_code"]))
    if not frames:
        raise FileNotFoundError(f"no parquet splits under {path}")
    df = pl.concat(frames)
    df = df.filter(
        pl.col("text").is_not_null()
        & (pl.col("text").str.strip_chars() != "")
        & pl.col("lcc_code").is_not_null()
    )
    if max_rows and df.height > max_rows:
        df = df.sample(n=max_rows, seed=42)
    return df["text"].to_list(), df["lcc_code"].to_list()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", action="append", required=True, metavar="NAME=DIR")
    ap.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Subsample each corpus, for a like-for-like comparison.",
    )
    ap.add_argument(
        "--balance",
        action="store_true",
        help="Subsample every corpus to the size of the smallest.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split

    corpora: dict[str, tuple[list[str], list[str]]] = {}
    for entry in args.corpus:
        name, _, path = entry.partition("=")
        texts, labels = load_corpus(Path(path), args.max_rows)
        corpora[name] = (texts, labels)
        logger.info(f"{name:<12} {len(texts):>7,} docs, {len(set(labels)):>2} classes")

    if args.balance:
        import random

        n = min(len(t) for t, _ in corpora.values())
        rng = random.Random(args.seed)
        for name, (texts, labels) in corpora.items():
            idx = rng.sample(range(len(texts)), n)
            corpora[name] = ([texts[i] for i in idx], [labels[i] for i in idx])
        logger.info(f"\nBalanced every corpus to {n:,} documents")

    # Hold out a test split per corpus so the diagonal is honest.
    splits: dict[str, Any] = {}
    for name, (texts, labels) in corpora.items():
        tr_x, te_x, tr_y, te_y = train_test_split(
            texts, labels, test_size=0.3, random_state=args.seed, stratify=labels
        )
        splits[name] = (tr_x, tr_y, te_x, te_y)

    names = list(corpora)
    results: dict[str, dict[str, float]] = {}

    for train_name in names:
        tr_x, tr_y, _, _ = splits[train_name]
        vec = TfidfVectorizer(max_features=50_000, sublinear_tf=True)
        xtr = vec.fit_transform(tr_x)
        clf = LogisticRegression(max_iter=2000)
        clf.fit(xtr, tr_y)
        results[train_name] = {}
        for test_name in names:
            _, _, te_x, te_y = splits[test_name]
            pred = clf.predict(vec.transform(te_x))
            # zero_division set explicitly: sklearn <1.4 defaults differ.
            results[train_name][test_name] = float(
                f1_score(te_y, pred, average="macro", zero_division=0.0)
            )

    width = max(len(n) for n in names) + 2
    logger.info("\nLCC macro-F1, TF-IDF + logistic regression")
    logger.info("-" * (width + 12 * len(names)))
    logger.info("train \\ test".ljust(width) + "".join(n.rjust(12) for n in names))
    logger.info("-" * (width + 12 * len(names)))
    for train_name in names:
        row = "".join(f"{results[train_name][t]:>12.4f}" for t in names)
        logger.info(train_name.ljust(width) + row)
    logger.info("-" * (width + 12 * len(names)))
    logger.info(
        "Diagonal is the in-domain ceiling; off-diagonal is transfer.\n"
        "The probe has no pretraining, so contamination cannot explain a gap."
    )

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "metric": "macro_f1",
                    "probe": "tfidf+logreg",
                    "seed": args.seed,
                    "balanced": args.balance,
                    "n_per_corpus": {k: len(v[0]) for k, v in corpora.items()},
                    "matrix": results,
                },
                indent=2,
            )
        )
        logger.info(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
