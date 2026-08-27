"""Score the topic-leakage arms on leakage *and* on fidelity.

An arm that cuts verbatim echo by drifting off-topic has not improved the
corpus, so both are reported side by side and neither is read alone.

  verbatim rate   how often a topic string appears in its own document
  fidelity        cosine between the document and its topic names, embedded
                  with a sentence encoder; drops if the document stops being
                  about the topic
  lexical probe   macro-F1 of TF-IDF predicting LCC from the document; a
                  drop here is the *goal*, provided fidelity holds

Paired bootstrap over specifications, since every arm saw the same specs.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def verbatim(row: dict) -> tuple[int, int]:
    text = (row.get("text") or "").lower()
    if not text.strip():
        return 0, 0
    hit = sum(1 for t in row["topics"] if t and str(t).lower() in text)
    return hit, len(row["topics"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="results/experiments/topic_leakage_ab.jsonl")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with Path(args.input).open() as fh:
        all_rows = [json.loads(line) for line in fh]
    rows = [r for r in all_rows if (r.get("text") or "").strip()]
    arms = sorted({r["arm"] for r in rows})
    logger.info(f"{len(rows):,} non-empty documents across {len(arms)} arms")

    empty: defaultdict[str, int] = defaultdict(int)
    for r in all_rows:
        if not (r.get("text") or "").strip():
            empty[r["arm"]] += 1

    # --- embed once, for fidelity -------------------------------------
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    docs = [r["text"] for r in rows]
    topic_strings = [", ".join(r["topics"]) for r in rows]
    d_emb = model.encode(docs, normalize_embeddings=True, show_progress_bar=False)
    t_emb = model.encode(
        topic_strings, normalize_embeddings=True, show_progress_bar=False
    )
    for r, d, t in zip(rows, d_emb, t_emb, strict=True):
        r["_fidelity"] = float((d * t).sum())

    # --- per-arm summary ----------------------------------------------
    logger.info(
        f"\n{'arm':<15}{'n':>5}{'empty':>7}{'verbatim':>11}{'fidelity':>11}{'words':>8}"
    )
    logger.info("-" * 57)
    summary = {}
    for arm in arms:
        sub = [r for r in rows if r["arm"] == arm]
        hit = sum(verbatim(r)[0] for r in sub)
        tot = sum(verbatim(r)[1] for r in sub)
        fid = statistics.mean(r["_fidelity"] for r in sub)
        wc = statistics.median(len(r["text"].split()) for r in sub)
        summary[arm] = {"verbatim": hit / tot, "fidelity": fid, "n": len(sub)}
        logger.info(
            f"{arm:<15}{len(sub):>5}{empty[arm]:>7}{hit / tot * 100:>10.1f}%"
            f"{fid:>11.4f}{wc:>8.0f}"
        )

    # --- paired bootstrap against control ------------------------------
    by_spec: dict[int, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        by_spec[r["spec_idx"]][f"{r['arm']}|{r['model']}"] = r
    specs = sorted(by_spec)
    rng = random.Random(args.seed)

    logger.info("\nPaired change vs control (2,000 bootstrap resamples of specs)")
    logger.info(
        f"{'arm':<15}{'d verbatim':>14}{'95% CI':>20}{'d fidelity':>13}{'95% CI':>20}"
    )
    logger.info("-" * 82)
    models = sorted({r["model"] for r in rows})
    for arm in arms:
        if arm == "control":
            continue
        dv, df = [], []
        for _ in range(args.n_boot):
            picked = [specs[rng.randrange(len(specs))] for _ in range(len(specs))]
            ch = ct = th = tt = 0
            fa, fc = [], []
            for s in picked:
                for m in models:
                    a = by_spec[s].get(f"{arm}|{m}")
                    c = by_spec[s].get(f"control|{m}")
                    if not a or not c:
                        continue
                    h1, t1 = verbatim(a)
                    h2, t2 = verbatim(c)
                    th += h1
                    tt += t1
                    ch += h2
                    ct += t2
                    fa.append(a["_fidelity"])
                    fc.append(c["_fidelity"])
            if tt and ct and fa:
                dv.append(th / tt - ch / ct)
                df.append(statistics.mean(fa) - statistics.mean(fc))
        dv.sort()
        df.sort()
        lo_v, hi_v = dv[int(0.025 * len(dv))], dv[int(0.975 * len(dv))]
        lo_f, hi_f = df[int(0.025 * len(df))], df[int(0.975 * len(df))]
        logger.info(
            f"{arm:<15}{statistics.mean(dv) * 100:>13.1f}%"
            f"{f'[{lo_v * 100:+.1f}, {hi_v * 100:+.1f}]':>20}"
            f"{statistics.mean(df):>13.4f}"
            f"{f'[{lo_f:+.4f}, {hi_f:+.4f}]':>20}"
        )

    logger.info(
        "\nA verbatim drop is only a win if the fidelity interval contains 0 or\n"
        "is positive. A negative fidelity interval means the arm bought its\n"
        "leakage reduction by writing about something else."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
