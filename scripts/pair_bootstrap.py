"""AUC intervals for pair tasks, resampling documents rather than pairs.

Pre-registration section 5. A pair set of 4,000 built from ~3,000 documents
is not 4,000 independent observations: the same document appears in several
pairs, so its errors are shared. Resampling pairs would understate the
interval.

This resamples DOCUMENTS. Every pair whose two endpoints both survive the
resample travels with them, which is the standard cluster bootstrap for
dyadic data. The effective sample size is therefore the document count, not
the pair count, and the intervals widen accordingly -- which is the honest
result.

Scores are computed here rather than read from saved predictions: the pair
evaluator does not write per-sample output by default. Models are built
through the same factory the sweep uses, so lexical baselines (tf, tfidf,
bm25) and pooled transformer models are covered, not only
sentence-transformers.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def auc(pairs: list[tuple[float, int]]) -> float:
    """Rank-based AUC. pairs = [(score, label)]."""
    pos = [s for s, y in pairs if y == 1]
    neg = [s for s, y in pairs if y == 0]
    if not pos or not neg:
        return float("nan")
    order = sorted(range(len(pairs)), key=lambda i: pairs[i][0])
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and pairs[order[j + 1]][0] == pairs[order[i]][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    rsum = sum(ranks[i] for i, (_, y) in enumerate(pairs) if y == 1)
    n1, n0 = len(pos), len(neg)
    return (rsum - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pairs",
        required=True,
        help="pair parquet, e.g. data/.../pairs/same_lcc/test.parquet",
    )
    ap.add_argument("--model", required=True, help="model key from config.yaml")
    ap.add_argument("--output", default=None, help="where to store the result")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    p = Path(args.pairs)
    if not p.exists():
        logger.error(f"{p} not found")
        return 2

    import sys as _sys

    import numpy as np
    import polars as pl
    import yaml

    _sys.path.insert(0, str(Path(__file__).resolve().parent / "baselines"))
    import run_all as R

    recs = pl.read_parquet(p).to_dicts()

    # Embed each distinct document once, then score every pair by cosine.
    texts: dict[str, str] = {}
    for r in recs:
        for side in ("a", "b"):
            did = r[f"doc_{side}_id"]
            if did not in texts:
                title = r.get(f"doc_{side}_title") or ""
                body = r.get(f"doc_{side}_body") or ""
                texts[did] = f"{title}\n\n{body}".strip()
    ids = sorted(texts)
    mcfg = yaml.safe_load(Path("scripts/baselines/config.yaml").read_text())["models"]
    if args.model not in mcfg:
        logger.error(f"unknown model key '{args.model}'")
        return 2
    model = R.create_model(mcfg[args.model])
    emb = np.asarray(model.encode([texts[i] for i in ids]))
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.where(norms == 0, 1.0, norms)
    idx = {d: i for i, d in enumerate(ids)}

    rows = [
        {
            "doc_a_id": r["doc_a_id"],
            "doc_b_id": r["doc_b_id"],
            "score": float(np.dot(emb[idx[r["doc_a_id"]]], emb[idx[r["doc_b_id"]]])),
            "label": int(r["label"]),
        }
        for r in recs
    ]

    by_doc: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        for d in (r["doc_a_id"], r["doc_b_id"]):
            by_doc.setdefault(d, []).append(i)
    docs = sorted(by_doc)
    point = auc([(float(r["score"]), int(r["label"])) for r in rows])

    rng = random.Random(args.seed)
    stats = []
    for _ in range(args.n_boot):
        # Multiplicity-weighted cluster bootstrap. Drawing documents into a
        # SET discards multiplicity and gives a ~63% subsample, not a
        # bootstrap. Keep each document's draw count k_d; a pair (a, b) then
        # enters the replicate k_a * k_b times, the correct weighting for
        # dyadic data.
        counts: dict[str, int] = {}
        for _ in range(len(docs)):
            d = docs[rng.randrange(len(docs))]
            counts[d] = counts.get(d, 0) + 1
        sel: list[tuple[float, int]] = []
        for r in rows:
            ka = counts.get(r["doc_a_id"], 0)
            kb = counts.get(r["doc_b_id"], 0)
            if ka and kb:
                sel.extend([(float(r["score"]), int(r["label"]))] * (ka * kb))
        if len({y for _, y in sel}) == 2:
            v = auc(sel)
            if v == v:
                stats.append(v)
    stats.sort()
    lo = stats[int(0.025 * len(stats))]
    hi = stats[int(0.975 * len(stats))]

    logger.info(
        f"pairs {len(rows):,} over {len(docs):,} documents "
        f"({len(rows) / len(docs):.2f} pairs/doc)"
    )
    logger.info(f"AUC {point:.4f}   95% document-clustered CI [{lo:.4f}, {hi:.4f}]")
    logger.info("Interval resamples DOCUMENTS; pairs are not independent.")
    rec = {
        "model": args.model,
        "pairs_file": str(p),
        "auc": point,
        "ci": [lo, hi],
        "n_pairs": len(rows),
        "n_documents": len(docs),
        "n_boot": len(stats),
        "bootstrap": "multiplicity-weighted document cluster",
    }
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec, indent=2))
        logger.info(f"  wrote {out}")
    print(json.dumps(rec))
    return 0


if __name__ == "__main__":
    sys.exit(main())
