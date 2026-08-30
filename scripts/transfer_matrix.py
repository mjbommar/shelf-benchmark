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


def load_corpus(
    path: Path, max_rows: int | None, truncate_words: int | None = None
) -> tuple[list[str], list[str], list[str] | None]:
    """Texts, labels, and a grouping key if the corpus has one.

    The grouping key is ``spec_id``. Documents written from one specification
    are near-siblings, so splitting them at the document level leaks: 598 of
    600 specifications straddled a split when this was done by document. Where
    the column exists the split must group on it.
    """
    import polars as pl

    frames = []
    for split in ("train", "validation", "test"):
        f = path / f"{split}.parquet"
        if not f.exists():
            continue
        cols = pl.read_parquet_schema(f)
        want = ["text", "lcc_code"] + (["spec_id"] if "spec_id" in cols else [])
        frames.append(pl.read_parquet(f, columns=want))
    if not frames:
        raise FileNotFoundError(f"no parquet splits under {path}")
    df = pl.concat(frames, how="diagonal_relaxed")
    df = df.filter(
        pl.col("text").is_not_null()
        & (pl.col("text").str.strip_chars() != "")
        & pl.col("lcc_code").is_not_null()
    )
    if max_rows and df.height > max_rows:
        df = df.sample(n=max_rows, seed=42)
    texts = df["text"].to_list()
    if truncate_words:
        texts = [" ".join(t.split()[:truncate_words]) for t in texts]
    groups = None
    if "spec_id" in df.columns:
        g = df["spec_id"].to_list()
        # A row with no spec_id is its own group, so it never leaks.
        groups = [x if x is not None else f"__row{i}" for i, x in enumerate(g)]
    return texts, df["lcc_code"].to_list(), groups


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
    ap.add_argument(
        "--truncate-words",
        type=int,
        default=None,
        help=(
            "Cut every document to this many words before vectorising. "
            "TF-IDF gains with length and the corpora differ sharply in it "
            "(median 486, 609, and 134 tokens), so a length-matched arm "
            "separates 'easier corpus' from 'longer documents'."
        ),
    )
    ap.add_argument(
        "--n-boot",
        type=int,
        default=1000,
        help="Bootstrap resamples of the TEST DOCUMENTS for each cell.",
    )
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split

    corpora: dict[str, tuple[list[str], list[str], list[str] | None]] = {}
    for entry in args.corpus:
        name, _, path = entry.partition("=")
        texts, labels, groups = load_corpus(
            Path(path), args.max_rows, args.truncate_words
        )
        corpora[name] = (texts, labels, groups)
        ng = f", {len(set(groups)):,} spec groups" if groups else ", no spec_id"
        logger.info(
            f"{name:<12} {len(texts):>7,} docs, {len(set(labels)):>2} classes{ng}"
        )

    if args.balance:
        import random

        n = min(len(t) for t, _, _ in corpora.values())
        rng = random.Random(args.seed)
        for name, (texts, labels, groups) in corpora.items():
            idx = rng.sample(range(len(texts)), n)
            corpora[name] = (
                [texts[i] for i in idx],
                [labels[i] for i in idx],
                [groups[i] for i in idx] if groups else None,
            )
        logger.info(f"\nBalanced every corpus to {n:,} documents")

    # Hold out a test split per corpus so the diagonal is honest.
    splits: dict[str, Any] = {}
    split_kind: dict[str, str] = {}
    for name, (texts, labels, groups) in corpora.items():
        if groups is not None:
            # Group on spec_id: sibling documents from one brief must not sit
            # on both sides. Section 3 of the paper argues this and an earlier
            # version of this script then split by document anyway.
            from sklearn.model_selection import GroupShuffleSplit

            gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=args.seed)
            tr_i, te_i = next(gss.split(texts, labels, groups=groups))
            tr_x = [texts[i] for i in tr_i]
            tr_y = [labels[i] for i in tr_i]
            te_x = [texts[i] for i in te_i]
            te_y = [labels[i] for i in te_i]
            straddling = len({groups[i] for i in tr_i} & {groups[i] for i in te_i})
            split_kind[name] = f"grouped on spec_id ({straddling} straddling)"
        else:
            tr_x, te_x, tr_y, te_y = train_test_split(
                texts, labels, test_size=0.3, random_state=args.seed, stratify=labels
            )
            split_kind[name] = "stratified by label (no spec_id in this corpus)"
        splits[name] = (tr_x, tr_y, te_x, te_y)
        logger.info(f"  {name:<12} split: {split_kind[name]}")

    import random

    names = list(corpora)
    results: dict[str, dict[str, float]] = {}
    cis: dict[str, dict[str, list[float]]] = {n: {} for n in names}

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
            point = float(f1_score(te_y, pred, average="macro", zero_division=0.0))
            results[train_name][test_name] = point

            # Document-resampled interval, as the pre-registration requires.
            # An ordering of point estimates 0.02 apart is not a finding
            # without one.
            rng = random.Random(args.seed)
            n = len(te_y)
            boots = []
            for _ in range(args.n_boot):
                idx = [rng.randrange(n) for _ in range(n)]
                by = [te_y[i] for i in idx]
                bp = [pred[i] for i in idx]
                boots.append(
                    float(f1_score(by, bp, average="macro", zero_division=0.0))
                )
            boots.sort()
            cis[train_name][test_name] = [
                boots[int(0.025 * len(boots))],
                boots[int(0.975 * len(boots))],
            ]

    width = max(len(n) for n in names) + 2
    logger.info("\nLCC macro-F1, TF-IDF + logistic regression")
    logger.info("-" * (width + 12 * len(names)))
    logger.info("train \\ test".ljust(width) + "".join(n.rjust(12) for n in names))
    logger.info("-" * (width + 12 * len(names)))
    for train_name in names:
        row = "".join(f"{results[train_name][t]:>12.4f}" for t in names)
        logger.info(train_name.ljust(width) + row)
        ci_row = "".join(
            f"{'[' + format(cis[train_name][t][0], '.2f') + ',' + format(cis[train_name][t][1], '.2f') + ']':>12}"
            for t in names
        )
        logger.info(" " * width + ci_row)
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
                    "split": split_kind,
                    "truncate_words": args.truncate_words,
                    "n_boot": args.n_boot,
                    "bootstrap": "test documents resampled with replacement",
                    "matrix": results,
                    "ci": cis,
                },
                indent=2,
            )
        )
        logger.info(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
