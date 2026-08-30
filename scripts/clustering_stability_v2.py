"""Clustering seed stability, measured THROUGH the evaluator.

Replaces clustering_stability.py, which a reviewer correctly faulted on two
counts: it covered only the 13 sentence-transformer models, excluding both
lexical baselines and all three large models, so the gate was decided on the
subset least likely to fail it; and it re-implemented the encode-and-cluster
pipeline rather than calling the evaluator, so for three of thirteen models
its ARI fell outside the seed range entirely -- the difference was
preprocessing, not seed.

This drives the real ClusteringEvaluator at five seeds, so the stability
measured is the stability of the clustering the paper reports.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "baselines"))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

SEEDS = (0, 1, 2, 3, 4)
FLOOR = 0.90


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="lcc_clustering")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    import run_all as R
    import yaml
    from scipy.stats import spearmanr
    from shelf.evaluate.evaluators.clustering import ClusteringEvaluator
    from shelf.evaluate.registry import get_task

    mcfg = yaml.safe_load(Path("scripts/baselines/config.yaml").read_text())["models"]
    spec = get_task(args.task)
    per_model: dict[str, dict[int, float]] = {}

    # Embed once per model, then let the five seeds share the cache. Without
    # this, evaluate_embedder re-encodes the whole corpus on every seed: on the
    # pooled corpus that is five passes over 63k documents per model, hours of
    # work to produce embeddings that are now identical by construction. The
    # seed is meant to vary the clustering, not the representation.
    texts = None
    if any(mcfg[k].get("type") == "sentence_transformer" for k in args.models):
        texts = R.collect_all_evaluation_texts(
            tasks_config={"clustering": [args.task]},
        )
        logger.info(f"cache corpus: {len(texts)} texts")

    for key in args.models:
        try:
            model = R.create_model(mcfg[key])
        except Exception as exc:
            logger.warning(f"  {key}: cannot build ({exc})")
            continue

        eval_model = model
        if texts is not None and mcfg[key].get("type") == "sentence_transformer":
            from shelf.evaluate.adapters.cached import CachedEmbedder

            # Batch size has to follow the model's context budget. A model
            # pinned to 8192 tokens allocates 4 GiB in one block at the
            # default batch and OOMs a 16 GiB card, which is how the first
            # attempt at this gate died. Scale it down as context grows.
            seq = int((mcfg[key].get("params") or {}).get("max_seq_length") or 512)
            bs = 32 if seq <= 512 else 16 if seq <= 2048 else 4
            logger.info(f"  {key}: seq={seq} batch={bs}")
            embs = model.encode(texts, batch_size=bs, show_progress=False)
            eval_model = CachedEmbedder(
                cache=dict(zip(texts, embs, strict=True)),
                model_name=key,
                embedding_dim=model.embedding_dim,
                fallback=model,
            )

        scores: dict[int, float] = {}
        for seed in SEEDS:
            try:
                ev = ClusteringEvaluator(spec, random_seed=seed)
                res = ev.evaluate_embedder(eval_model)
                scores[seed] = float((res.metrics or {}).get("ari", float("nan")))
            except Exception as exc:
                logger.warning(f"  {key} seed {seed}: {type(exc).__name__}: {exc}")
        good = {k: v for k, v in scores.items() if v == v}
        if len(good) < len(SEEDS):
            logger.warning(f"  {key}: only {len(good)}/{len(SEEDS)} seeds usable")
            continue
        per_model[key] = good
        logger.info(
            f"  {key:<24} median ARI {statistics.median(good.values()):.4f}  "
            f"spread {max(good.values()) - min(good.values()):.4f}"
        )

    if len(per_model) < 6:
        # Write the per-model values anyway. Stability needs three models,
        # but the ARI measurements are valid on their own and a caller
        # measuring a small batch for a merge should not lose an hour of GPU
        # work because this script cannot compute one summary statistic.
        logger.error(
            "too few models to assess stability; writing per-model values only"
        )
        Path(args.output).write_text(
            json.dumps(
                {
                    "task": args.task,
                    "seeds": list(SEEDS),
                    "n_models": len(per_model),
                    "per_model": {
                        m: {str(k): v for k, v in d.items()}
                        for m, d in per_model.items()
                    },
                    "median_ari": {
                        m: statistics.median(d.values()) for m, d in per_model.items()
                    },
                    "rank_stability": None,
                    "note": "fewer than three models: stability not computed",
                    "measured_through": "ClusteringEvaluator (not a reimplementation)",
                },
                indent=2,
            )
        )
        return 1

    models = sorted(per_model)
    stab = []
    for a, b in itertools.combinations(SEEDS, 2):
        xa = [per_model[m][a] for m in models]
        xb = [per_model[m][b] for m in models]
        stab.append(float(spearmanr(xa, xb).statistic))
    med = statistics.median(stab)
    verdict = (
        "stable enough to carry the rank-agreement claim"
        if med >= FLOOR
        else "UNSTABLE: drop clustering from the rank-agreement claim"
    )
    logger.info(
        f"\n  {len(models)} models, rank stability median {med:.3f} "
        f"(min {min(stab):.3f}, max {max(stab):.3f})"
    )
    logger.info(f"  pre-registered floor {FLOOR}: {verdict}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "task": args.task,
                "seeds": list(SEEDS),
                "n_models": len(models),
                "per_model": per_model,
                "median_ari": {
                    m: statistics.median(v.values()) for m, v in per_model.items()
                },
                "rank_stability": stab,
                "median_rank_stability": med,
                "floor": FLOOR,
                "verdict": verdict,
                "measured_through": "ClusteringEvaluator (not a reimplementation)",
            },
            indent=2,
        )
    )
    logger.info(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
