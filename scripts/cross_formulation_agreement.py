"""Rank agreement between corpora, per task formulation.

Implements the decision rule frozen in docs/PREREGISTRATION.md section 3.
The rule is applied as written; it is not adjusted after seeing results.

    supports the broad claim   rho >= 0.6 and interval lower bound > 0,
                               for >= 3 of 4 formulations against BOTH
                               natural corpora
    supports a narrow claim    holds for classification, fails for two or
                               more others
    contradicts                lower bound crosses zero for classification

Primary metrics are fixed: macro-F1 for classification, binary nDCG@10 for
retrieval, ARI for clustering, AUC for pairs. Intervals resample MODELS,
because the question is whether an ordering survives a different model set.
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

# task -> (metric key inside "metrics", fallback to primary_score)
PRIMARY = {
    "lcc_classification": "macro_f1",
    "lcc_retrieval": "ndcg@10",
    "lcc_clustering": "ari",
    "same_lcc_pairs": "auc_roc",
}
THRESHOLD = 0.6


def load(results_dir: Path, task: str) -> dict[str, float]:
    key = PRIMARY[task]
    out: dict[str, float] = {}
    for f in sorted(results_dir.glob(f"*_{task}.json")):
        try:
            r = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if "error" in r:
            continue
        m = r.get("model_key") or r.get("model")
        v = (r.get("metrics") or {}).get(key)
        if v is None:
            # Do NOT fall back to primary_score. For clustering that is
            # v_measure, which the pre-registration rejects for not being
            # chance-corrected; for pairs it is threshold-tuned F1. Silently
            # substituting a forbidden metric is how three earlier bugs in
            # this project produced plausible wrong numbers.
            raise SystemExit(
                f"{f.name}: metric '{key}' missing. Refusing to fall back to "
                "primary_score, which may be a metric the pre-registration "
                "rejects. Re-run the task or fix the metric key."
            )
        if m and v is not None:
            out[m] = float(v)
    return out


def spearman(a, b):
    from scipy.stats import spearmanr

    return float(spearmanr(a, b).statistic)


def boot(models, x, y, n, seed):
    rng = random.Random(seed)
    stats = []
    for _ in range(n):
        pick = [models[rng.randrange(len(models))] for _ in range(len(models))]
        xs = [x[m] for m in pick]
        ys = [y[m] for m in pick]
        if len(set(xs)) > 1 and len(set(ys)) > 1:
            stats.append(spearman(xs, ys))
    if not stats:
        return float("nan"), float("nan")
    stats.sort()
    return stats[int(0.025 * len(stats))], stats[int(0.975 * len(stats))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shelf", default="results/pooled/baselines")
    ap.add_argument("--natural", action="append", required=True, metavar="NAME=DIR")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="results/transfer/cross_formulation.json")
    args = ap.parse_args()

    nat = {}
    for e in args.natural:
        n, _, d = e.partition("=")
        nat[n] = Path(d)

    report = {"threshold": THRESHOLD, "pairs": []}
    holds = dict.fromkeys(nat, 0)
    tested = dict.fromkeys(nat, 0)

    logger.info(f"{'task':<20}{'corpus':<14}{'n':>4}{'rho':>8}{'95% CI':>18}  verdict")
    logger.info("-" * 74)
    for task in PRIMARY:
        s = load(Path(args.shelf), task)
        if not s:
            logger.info(f"{task:<20}{'—':<14}{'—':>4}   no SHELF results yet")
            continue
        for name, d in nat.items():
            o = load(d, task)
            shared = sorted(set(s) & set(o))
            if len(shared) < 4:
                logger.info(
                    f"{task:<20}{name:<14}{len(shared):>4}   too few shared models"
                )
                continue
            xs = [s[m] for m in shared]
            ys = [o[m] for m in shared]
            rho = spearman(xs, ys)
            lo, hi = boot(shared, s, o, args.n_boot, args.seed)
            ok = rho >= THRESHOLD and lo > 0
            tested[name] += 1
            holds[name] += 1 if ok else 0
            logger.info(
                f"{task:<20}{name:<14}{len(shared):>4}{rho:>8.3f}"
                f"{f'[{lo:.2f}, {hi:.2f}]':>18}  {'holds' if ok else 'FAILS'}"
            )
            report["pairs"].append(
                {
                    "task": task,
                    "corpus": name,
                    "n_models": len(shared),
                    "spearman": rho,
                    "ci": [lo, hi],
                    "holds": ok,
                    "models": shared,
                }
            )

    logger.info("-" * 74)
    for n in nat:
        logger.info(
            f"  {n}: {holds[n]} hold of {tested[n]} tested "
            f"(of {len(PRIMARY)} formulations)"
        )
        if tested[n] < len(PRIMARY):
            logger.info(
                f"     {len(PRIMARY) - tested[n]} formulation(s) NOT TESTED; "
                "the broad claim requires all four"
            )
    # Prereg section 3 requires >= 3 of the FOUR formulations against BOTH
    # corpora. Counting "3 of 3 tested" as sufficient let an untested
    # formulation count toward the broad claim.
    n_formulations = len(PRIMARY)
    broad = all(holds[n] >= 3 and tested[n] == n_formulations for n in nat)
    cls = [p for p in report["pairs"] if p["task"] == "lcc_classification"]
    contradicted = any(p["ci"][0] <= 0 for p in cls) if cls else False

    if contradicted:
        verdict = "CONTRADICTED: classification interval crosses zero"
    elif broad:
        verdict = "BROAD CLAIM SUPPORTED"
    else:
        verdict = "NARROW CLAIM ONLY: title must scope to classification"
    logger.info(f"\n  VERDICT (pre-registered rule): {verdict}")
    report["verdict"] = verdict

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    logger.info(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
