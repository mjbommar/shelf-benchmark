"""Measure whether SHELF ranks models the way natural corpora rank them.

SHELF scores do not transfer to natural text in absolute terms -- we
measured that ourselves (0.8932 in-domain against 0.3010 on Gutenberg,
symmetrically). Rank agreement is the weaker claim that survives: a
benchmark can be useless for predicting an absolute score and still be
useful for choosing between models.

This script takes the per-model scores each corpus produced for one task
and reports Spearman and Kendall rank correlation between corpora, with a
bootstrap interval over models.

    python scripts/rank_agreement.py \\
        --corpus shelf=results/pooled/baselines \\
        --corpus gutenberg=results/transfer_gutenberg/baselines \\
        --corpus lcshbench=results/transfer_lcshbench/baselines \\
        --task lcc_classification

Read the result against two precedents, neither bibliographic: Majurski
and Matuszek report Spearman 0.91 between synthetic and human-curated
benchmarks (TMLR 2026), and YourBench reports rho = 1 for rankings from
document-grounded synthetic evaluation.

A caution the number cannot express: Gutenberg is running prose and
LCSHBench is catalogue metadata. Low agreement with LCSHBench may say more
about text length than about the taxonomy, so report the pair separately
rather than averaging them.
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


def load_scores(results_dir: Path, task: str, metric: str) -> dict[str, float]:
    """Per-model score for one task, read from the per-task result files."""
    scores: dict[str, float] = {}
    for path in sorted(results_dir.glob(f"*_{task}.json")):
        model = path.name[: -len(f"_{task}.json")]
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(f"  skipping {path.name}: {exc}")
            continue
        if "error" in data:
            continue
        value = data.get(metric)
        if value is None:
            value = data.get("primary_score")
        if value is None:
            metrics = data.get("metrics") or {}
            value = metrics.get(metric) or metrics.get("macro_f1")
        if value is None:
            continue
        scores[model] = float(value)
    return scores


def spearman(a: list[float], b: list[float]) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(a, b).statistic)


def kendall(a: list[float], b: list[float]) -> float:
    from scipy.stats import kendalltau

    return float(kendalltau(a, b).statistic)


def bootstrap_ci(
    models: list[str],
    x: dict[str, float],
    y: dict[str, float],
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    """Percentile interval, resampling *models* rather than documents.

    The uncertainty that matters here is whether the ordering would hold
    for a different set of models, not for a different set of documents.
    """
    import random

    rng = random.Random(seed)
    stats: list[float] = []
    n = len(models)
    for _ in range(n_boot):
        picked = [models[rng.randrange(n)] for _ in range(n)]
        xs = [x[m] for m in picked]
        ys = [y[m] for m in picked]
        if len(set(xs)) < 2 or len(set(ys)) < 2:
            continue
        stats.append(spearman(xs, ys))
    if not stats:
        return (float("nan"), float("nan"))
    stats.sort()
    lo = stats[int(0.025 * len(stats))]
    hi = stats[min(int(0.975 * len(stats)), len(stats) - 1)]
    return (lo, hi)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--corpus",
        action="append",
        required=True,
        metavar="NAME=DIR",
        help="Named results directory; repeat for each corpus.",
    )
    ap.add_argument("--task", default="lcc_classification")
    ap.add_argument("--metric", default="macro_f1")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    corpora: dict[str, dict[str, float]] = {}
    for entry in args.corpus:
        if "=" not in entry:
            logger.error(f"--corpus needs NAME=DIR, got {entry!r}")
            return 2
        name, _, path = entry.partition("=")
        scores = load_scores(Path(path), args.task, args.metric)
        if not scores:
            logger.warning(f"{name}: no results for task {args.task} in {path}")
        corpora[name] = scores
        logger.info(f"{name:<12} {len(scores):>3} models scored")

    names = [n for n in corpora if corpora[n]]
    if len(names) < 2:
        logger.error("\nNeed at least two corpora with results. Run the sweeps first.")
        return 1

    report: dict[str, Any] = {
        "task": args.task,
        "metric": args.metric,
        "n_bootstrap": args.n_boot,
        "per_corpus_models": {n: len(corpora[n]) for n in names},
        "pairs": [],
    }

    logger.info(f"\nRank agreement on {args.task} ({args.metric})")
    logger.info("-" * 72)
    logger.info(f"{'pair':<26}{'n':>4}{'spearman':>11}{'95% CI':>18}{'kendall':>10}")
    logger.info("-" * 72)

    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared = sorted(set(corpora[a]) & set(corpora[b]))
            if len(shared) < 3:
                logger.info(f"{a} vs {b:<14}{len(shared):>4}   too few shared models")
                continue
            xs = [corpora[a][m] for m in shared]
            ys = [corpora[b][m] for m in shared]
            rho = spearman(xs, ys)
            tau = kendall(xs, ys)
            lo, hi = bootstrap_ci(
                shared, corpora[a], corpora[b], args.n_boot, args.seed
            )
            logger.info(
                f"{a + ' vs ' + b:<26}{len(shared):>4}{rho:>11.3f}"
                f"{f'[{lo:.2f}, {hi:.2f}]':>18}{tau:>10.3f}"
            )
            report["pairs"].append(
                {
                    "a": a,
                    "b": b,
                    "n_models": len(shared),
                    "spearman": rho,
                    "spearman_ci": [lo, hi],
                    "kendall": tau,
                    "models": shared,
                }
            )

    logger.info("-" * 72)
    logger.info(
        "Interval resamples MODELS, not documents: the question is whether the\n"
        "ordering would hold for a different set of models."
    )
    logger.info(
        "Gutenberg is running prose; LCSHBench is catalogue metadata (median\n"
        "596 chars). Report those pairs separately -- do not average them."
    )

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        logger.info(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
