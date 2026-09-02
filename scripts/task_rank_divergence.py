"""Do the coarse and fine subject tasks order models the same way?

Compares model rankings on 21-class LCC subject classification against
133-class genre-form classification, within one corpus and over the same
documents. If the two tasks ordered models differently, the fine task would
be necessary for model selection. It does not.

The interval resamples MODELS, not documents: the question is whether the
ordering would hold for a different set of models.

Excludes weight duplicates by default. ogbert-110m-base and
ogbert-110m-sentence resolve to one safetensors blob with one SHA-256, and a
pair that is one model always agrees with itself, which raises the
correlation.

Usage:
    uv run python scripts/task_rank_divergence.py \\
        --results results/pooled/baselines \\
        --output results/transfer/task_rank_divergence.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DEFAULT_EXCLUDE = "ogbert_2m_sentence,ogbert_110m_sentence"


def load(results: Path, task: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for f in sorted(results.glob(f"*_{task}.json")):
        try:
            d = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if "error" in d or "primary_score" not in d:
            continue
        out[d.get("model_key") or f.name[: -len(f"_{task}.json")]] = float(
            d["primary_score"]
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results/pooled/baselines")
    ap.add_argument("--task-a", default="lcc_classification")
    ap.add_argument("--task-b", default="form_classification")
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude-models", default=DEFAULT_EXCLUDE)
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    from scipy.stats import spearmanr

    results = Path(args.results)
    excluded = {m.strip() for m in args.exclude_models.split(",") if m.strip()}

    a, b = load(results, args.task_a), load(results, args.task_b)
    shared_all = sorted(set(a) & set(b))
    shared = [m for m in shared_all if m not in excluded]
    dropped = [m for m in shared_all if m in excluded]
    if dropped:
        logger.info(f"excluded {len(dropped)}: {', '.join(dropped)}")

    xs = [a[m] for m in shared]
    ys = [b[m] for m in shared]
    rho = float(spearmanr(xs, ys).statistic)

    rng = random.Random(args.seed)
    boots = []
    n = len(shared)
    for _ in range(args.n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        bx = [xs[i] for i in idx]
        by = [ys[i] for i in idx]
        if len(set(bx)) < 2 or len(set(by)) < 2:
            continue
        r = spearmanr(bx, by).statistic
        if r == r:
            boots.append(float(r))
    boots.sort()
    lo = boots[int(0.025 * len(boots))]
    hi = boots[int(0.975 * len(boots))]

    logger.info(
        f"{args.task_a} vs {args.task_b} over {n} distinct models: "
        f"rho={rho:.4f}  95% CI [{lo:.2f}, {hi:.2f}]"
    )
    logger.info("Interval resamples MODELS, not documents.")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "comparison": f"{args.task_a} vs {args.task_b} (within {results})",
                    "n_models": n,
                    "excluded_models": sorted(dropped),
                    "spearman": rho,
                    "ci": [lo, hi],
                    "n_boot": args.n_boot,
                    "seed": args.seed,
                    "resamples": "models, not documents",
                },
                indent=2,
            )
        )
        logger.info(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
