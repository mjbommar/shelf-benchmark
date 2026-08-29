"""Build same-subject pair sets for the natural corpora.

Phase B of docs/PREREGISTRATION.md. The mining policy is fixed there and is
applied identically to every corpus, so that a rank comparison across corpora
is not a comparison of two different pair-construction schemes.

Policy, from the pre-registration §5:

- Equal positives and negatives, so chance is AUC 0.5.
- Quotas balanced across the 21 LCC classes.
- **Random negatives, not hard negatives.** Hard-negative mining needs an
  embedding model, which would couple the task to one of the systems under
  test.
- No duplicate pairs and no self-pairs.
- A document may appear in several pairs; the cap is recorded in the output,
  because it is what makes the pairs non-independent and forces the
  document-clustered bootstrap used when scoring.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def build(rows, n_per_side: int, seed: int, max_uses: int):
    rng = random.Random(seed)
    by_class: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        code = r.get("lcc_code")
        if code and (r.get("text") or "").strip():
            by_class[code].append(i)
    classes = sorted(c for c in by_class if len(by_class[c]) >= 2)
    if len(classes) < 2:
        raise SystemExit("need at least two usable classes")

    uses: dict[int, int] = defaultdict(int)
    seen: set[tuple[int, int]] = set()
    pos: list[tuple[int, int]] = []
    neg: list[tuple[int, int]] = []

    def take(a: int, b: int) -> bool:
        if a == b:
            return False
        key = (min(a, b), max(a, b))
        if key in seen:
            return False
        if uses[a] >= max_uses or uses[b] >= max_uses:
            return False
        seen.add(key)
        uses[a] += 1
        uses[b] += 1
        return True

    # Positives: quota per class, so no class dominates.
    per_class = max(1, n_per_side // len(classes))
    for c in classes:
        pool = by_class[c]
        tries = 0
        made = 0
        while made < per_class and tries < per_class * 60:
            tries += 1
            a, b = rng.choice(pool), rng.choice(pool)
            if take(a, b):
                pos.append((a, b))
                made += 1
    guard = 0
    while len(pos) < n_per_side and guard < n_per_side * 200:
        guard += 1
        c = rng.choice(classes)
        pool = by_class[c]
        a, b = rng.choice(pool), rng.choice(pool)
        if take(a, b):
            pos.append((a, b))

    # Negatives: random cross-class, quota by the class of the first document.
    for c in classes:
        others = [x for x in classes if x != c]
        made = 0
        tries = 0
        while made < per_class and tries < per_class * 60:
            tries += 1
            a = rng.choice(by_class[c])
            b = rng.choice(by_class[rng.choice(others)])
            if take(a, b):
                neg.append((a, b))
                made += 1
    guard = 0
    while len(neg) < n_per_side and guard < n_per_side * 200:
        guard += 1
        c = rng.choice(classes)
        others = [x for x in classes if x != c]
        a = rng.choice(by_class[c])
        b = rng.choice(by_class[rng.choice(others)])
        if take(a, b):
            neg.append((a, b))

    n = min(len(pos), len(neg), n_per_side)  # keep the sides equal
    return pos[:n], neg[:n], uses


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True, help="<data_root>/pairs/<name>")
    ap.add_argument("--n-per-side", type=int, default=2000)
    ap.add_argument("--max-uses", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import polars as pl

    src = Path(args.source)
    frames = [
        pl.read_parquet(src / f"{s}.parquet")
        for s in ("train", "validation", "test")
        if (src / f"{s}.parquet").exists()
    ]
    rows = pl.concat(frames, how="diagonal_relaxed").to_dicts()

    # A corpus of D documents used at most U times supports D*U/2 pairs. Asking
    # for more than that spins forever, which is what a first run did on
    # Gutenberg's 627-document test split.
    capacity = (len(rows) * args.max_uses) // 2
    requested = args.n_per_side * 2
    if requested > capacity:
        args.n_per_side = max(100, capacity // 2)
        logger.warning(
            f"  {src.name}: {len(rows):,} documents at {args.max_uses} uses "
            f"support {capacity:,} pairs; reducing to "
            f"{args.n_per_side * 2:,} ({args.n_per_side:,} per side)"
        )
    logger.info(f"{src.name}: {len(rows):,} documents")

    pos, neg, uses = build(rows, args.n_per_side, args.seed, args.max_uses)
    logger.info(f"  positives {len(pos):,}  negatives {len(neg):,}")

    out_rows = []
    for k, (pairs, label) in enumerate(((pos, 1), (neg, 0))):
        for j, (a, b) in enumerate(pairs):
            ra, rb = rows[a], rows[b]
            out_rows.append(
                {
                    "id": f"pair_{k}{j:06d}",
                    "doc_a_id": ra.get("id"),
                    "doc_a_title": ra.get("title") or "",
                    "doc_a_body": ra.get("text") or "",
                    "doc_b_id": rb.get("id"),
                    "doc_b_title": rb.get("title") or "",
                    "doc_b_body": rb.get("text") or "",
                    "label": label,
                    "label_field": "lcc_code",
                }
            )
    random.Random(args.seed).shuffle(out_rows)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(out_rows, infer_schema_length=None).write_parquet(out / "test.parquet")
    n_docs = len({u for u in uses if uses[u] > 0})
    stats = {
        "n_pairs": len(out_rows),
        "n_positive": len(pos),
        "n_negative": len(neg),
        "n_distinct_documents": n_docs,
        "max_uses_per_document": args.max_uses,
        "observed_max_uses": max(uses.values()) if uses else 0,
        "pairs_per_document": round(len(out_rows) / max(1, n_docs), 2),
        "negatives": "random cross-class (not hard negatives)",
        "seed": args.seed,
    }
    (out / "_pair_stats.json").write_text(json.dumps(stats, indent=2))
    logger.info(
        f"  {len(out_rows):,} pairs over {n_docs:,} distinct documents "
        f"({stats['pairs_per_document']} pairs/doc) -> {out}"
    )
    logger.info(
        "  pairs are NOT independent: intervals must resample documents, not pairs."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
