"""Is the clustering ranking a property of the models, or of the seed?

Pre-registration section 6. K-means is run with ``n_init=10`` at a single
fixed ``random_state``, so every model shares one initialisation draw. If the
model ordering moves when that draw changes, then cross-corpus clustering
agreement is partly measuring initialisation noise.

Runs each model at five fixed seeds and reports:

- median ARI per model, the value used downstream;
- rank stability, the Spearman correlation between the model ordering at each
  pair of seeds;
- the decision fixed in advance: **if median rank stability is below 0.90,
  clustering is dropped from the rank-agreement claim.**

Embeddings are computed once per model and reused across seeds, so the cost
is one sweep rather than five.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import statistics
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

SEEDS = (0, 1, 2, 3, 4)
STABILITY_FLOOR = 0.90


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True, help="data/hf_dataset/<name>")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--label-field", default="lcc_code")
    ap.add_argument("--split", default="test")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    import polars as pl
    import yaml
    from scipy.stats import spearmanr
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    cfg = yaml.safe_load(Path("scripts/baselines/config.yaml").read_text())
    mcfg = cfg["models"]

    df = pl.read_parquet(Path(args.corpus) / f"{args.split}.parquet")
    texts = df["text"].to_list()
    labels = df[args.label_field].to_list()
    keep = [i for i, (t, y) in enumerate(zip(texts, labels)) if t and y]
    texts = [texts[i] for i in keep]
    labels = [labels[i] for i in keep]
    k = len(set(labels))
    logger.info(f"{Path(args.corpus).name}: {len(texts):,} docs, k={k}\n")

    per_model: dict[str, dict[int, float]] = {}
    for key in args.models:
        spec = mcfg.get(key)
        if not spec or spec.get("type") != "sentence_transformer":
            logger.info(f"  {key}: not a sentence-transformer, skipped")
            continue
        name = spec.get("model_name") or key
        try:
            st = SentenceTransformer(name)
            emb = st.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        except Exception as exc:
            logger.warning(f"  {key}: {exc}")
            continue
        scores = {}
        for seed in SEEDS:
            km = KMeans(n_clusters=k, n_init=10, random_state=seed)
            scores[seed] = float(adjusted_rand_score(labels, km.fit_predict(emb)))
        per_model[key] = scores
        med = statistics.median(scores.values())
        spread = max(scores.values()) - min(scores.values())
        logger.info(
            f"  {key:<24} median ARI {med:.4f}  spread across seeds {spread:.4f}"
        )
        del emb

    if len(per_model) < 4:
        logger.error("too few models to assess stability")
        return 1

    models = sorted(per_model)
    stabilities = []
    for a, b in itertools.combinations(SEEDS, 2):
        xa = [per_model[m][a] for m in models]
        xb = [per_model[m][b] for m in models]
        stabilities.append(float(spearmanr(xa, xb).statistic))
    med_stab = statistics.median(stabilities)

    logger.info(
        f"\n  rank stability across seed pairs: median {med_stab:.3f}, "
        f"min {min(stabilities):.3f}, max {max(stabilities):.3f}"
    )
    verdict = (
        "clustering is stable enough to carry the rank-agreement claim"
        if med_stab >= STABILITY_FLOOR
        else "UNSTABLE: drop clustering from the rank-agreement claim"
    )
    logger.info(f"  pre-registered floor {STABILITY_FLOOR}: {verdict}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "corpus": args.corpus,
                "seeds": list(SEEDS),
                "per_model": per_model,
                "median_ari": {
                    m: statistics.median(v.values()) for m, v in per_model.items()
                },
                "rank_stability_pairs": stabilities,
                "median_rank_stability": med_stab,
                "floor": STABILITY_FLOOR,
                "verdict": verdict,
            },
            indent=2,
        )
    )
    logger.info(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
