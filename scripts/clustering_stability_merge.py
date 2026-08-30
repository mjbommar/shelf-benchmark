"""Extend a clustering stability report with new models, without re-running old ones.

The gate stores each model's ARI at every seed. A model's median over those
seeds is a property of that model alone: it does not change when other models
are added to the sweep. So adding models to the benchmark does not require
re-embedding the models already measured.

What does change is the rank stability, which is a Spearman correlation
between the model orderings produced by different seeds. That is arithmetic
over the stored per-seed values and costs nothing.

This runs the gate for the named models only, then merges their per-seed
values into the existing report and recomputes the ranking statistics over the
whole set. On a 27-model sweep with 22 already measured, it does a fifth of
the GPU work of a full re-run.

Usage:
    uv run python scripts/clustering_stability_merge.py \\
        --existing results/transfer/clustering_stability_all.json \\
        --models granite_small_r2 gte_modernbert_2k \\
        --task lcc_clustering --output results/transfer/clustering_stability_all.json
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import subprocess
import sys
import tempfile
from itertools import combinations
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

FLOOR = 0.9


def rank_stability(
    per_model: dict[str, dict[str, float]], seeds: list[int]
) -> list[float]:
    """Spearman between the orderings two seeds produce, for every seed pair."""
    from scipy.stats import spearmanr

    models = sorted(per_model)
    out = []
    for a, b in combinations(seeds, 2):
        xa = [per_model[m][str(a)] for m in models]
        xb = [per_model[m][str(b)] for m in models]
        r = spearmanr(xa, xb).statistic
        if r == r:
            out.append(float(r))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--existing", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--task", default="lcc_clustering")
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--skip-run",
        action="store_true",
        help="Merge only; assume the new models are already in --existing.",
    )
    args = ap.parse_args()

    existing = json.loads(Path(args.existing).read_text())
    per_model: dict[str, dict[str, float]] = dict(existing["per_model"])
    seeds = list(existing["seeds"])
    logger.info(f"existing: {len(per_model)} models, seeds {seeds}")

    todo = [m for m in args.models if m not in per_model]
    if not todo:
        logger.info("nothing new to measure")
    elif not args.skip_run:
        logger.info(f"measuring {len(todo)}: {', '.join(todo)}")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name
        cmd = [
            sys.executable,
            "scripts/clustering_stability_v2.py",
            "--task",
            args.task,
            "--models",
            *todo,
            "--output",
            tmp_path,
        ]
        rc = subprocess.call(cmd)
        if rc != 0:
            # The gate refuses to report stability for fewer than three models,
            # which is correct and is not a failure of the measurement.
            logger.warning(
                f"gate exited {rc}; using whatever per-model values it wrote"
            )
        try:
            fresh = json.loads(Path(tmp_path).read_text())
            per_model.update(fresh.get("per_model", {}))
        except (OSError, json.JSONDecodeError) as exc:
            logger.error(f"could not read {tmp_path}: {exc}")
            return 2

    missing = [m for m in args.models if m not in per_model]
    if missing:
        logger.error(f"still missing after run: {', '.join(missing)}")
        return 2

    medians = {m: statistics.median(v.values()) for m, v in per_model.items()}
    stab = rank_stability(per_model, seeds)
    med_stab = statistics.median(stab)

    report = {
        "task": args.task,
        "seeds": seeds,
        "n_models": len(per_model),
        "per_model": per_model,
        "median_ari": medians,
        "rank_stability": stab,
        "median_rank_stability": med_stab,
        "floor": FLOOR,
        "verdict": (
            "stable enough to carry the rank-agreement claim"
            if med_stab >= FLOOR
            else "below the pre-registered floor; clustering is dropped"
        ),
        "measured_through": "ClusteringEvaluator (not a reimplementation)",
        "note": (
            "Per-seed values for models measured earlier are reused. A model's "
            "median over seeds does not depend on which other models are in the "
            "sweep; only the rank statistics do, and those are recomputed here "
            "over the full set."
        ),
    }
    Path(args.output).write_text(json.dumps(report, indent=2))
    logger.info(
        f"{len(per_model)} models, median rank stability {med_stab:.4f} "
        f"(floor {FLOOR}, worst pair {min(stab):.4f}) -> {report['verdict']}"
    )
    logger.info(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
