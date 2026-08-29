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
    """Scores for one task, searching the sibling _pairs directory too.

    Pair sweeps write to `results/<corpus>_pairs/baselines` while every other
    task writes to `results/<corpus>/baselines`. Reading only the latter made
    pair results invisible: the driver reported "too few shared models" and
    the pre-registered rule then counted the formulation as untested. Search
    both, so a formulation cannot be silently dropped because of where its
    output landed.
    """
    key = PRIMARY[task]
    out: dict[str, float] = {}
    search = [results_dir]
    sibling = (
        results_dir.parent.parent
        / f"{results_dir.parent.name}_pairs"
        / results_dir.name
    )
    if sibling.is_dir():
        search.append(sibling)
    files = [f for d in search for f in sorted(d.glob(f"*_{task}.json"))]
    for f in files:
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
    ap.add_argument(
        "--shelf-pairs",
        default=None,
        help=(
            "Directory holding SHELF pair results built on the SAME mining "
            "policy as the natural corpora. Without this the SHELF arm reads "
            "the published hub pair config, which uses a different scheme "
            "(no dedup, no use cap, no class quota) -- comparing two mining "
            "schemes rather than two corpora."
        ),
    )
    ap.add_argument("--natural", action="append", required=True, metavar="NAME=DIR")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--clustering-medians",
        action="append",
        default=[],
        metavar="NAME=FILE",
        help=(
            "Optional clustering_stability_v2 output per corpus. When given, "
            "lcc_clustering uses the per-model MEDIAN ARI across seeds instead "
            "of the single seed-42 result, as the pre-registration specifies."
        ),
    )
    ap.add_argument("--output", default="results/transfer/cross_formulation.json")
    args = ap.parse_args()

    nat = {}
    for e in args.natural:
        n, _, d = e.partition("=")
        nat[n] = Path(d)

    medians: dict[str, dict[str, float]] = {}
    stability: dict[str, tuple] = {}
    dropped: set[str] = set()
    for e in args.clustering_medians:
        n, _, f = e.partition("=")
        _st = json.loads(Path(f).read_text())
        medians[n] = _st["median_ari"]
        _stab = _st.get("median_rank_stability")
        _floor = _st.get("floor", 0.90)
        stability[n] = (_stab, _floor)
        if _stab is not None and _stab < _floor:
            # Prereg section 6: below the floor clustering is DROPPED from the
            # claim. Reading median_ari while ignoring the gate let a file
            # whose own verdict said "drop clustering" still count toward a
            # broad verdict.
            dropped.add(n)
            logger.warning(
                f"  {n}: clustering rank stability {_stab:.3f} below the "
                f"pre-registered floor {_floor}; DROPPING clustering from the "
                "rank-agreement claim for this corpus."
            )
        else:
            logger.info(
                f"  {n}: clustering uses the five-seed median "
                f"({len(medians[n])} models)"
            )

    report = {
        "threshold": THRESHOLD,
        "pairs": [],
        # A corpus dropped by the stability gate did NOT use the seed median
        # for its reported row -- it has no reported row. Listing it here
        # asserted a false provenance about how the pre-registered clustering
        # metric was computed, in the only durable record of the verdict.
        "clustering_uses_seed_median": sorted(set(medians) - dropped),
        "clustering_dropped_by_gate": sorted(dropped),
        "clustering_stability": {
            n: {"median_rank_stability": s, "floor": f}
            for n, (s, f) in stability.items()
        },
    }
    holds = dict.fromkeys(nat, 0)
    tested = dict.fromkeys(nat, 0)

    logger.info(f"{'task':<20}{'corpus':<14}{'n':>4}{'rho':>8}{'95% CI':>18}  verdict")
    logger.info("-" * 74)
    for task in PRIMARY:
        if task == "same_lcc_pairs" and args.shelf_pairs:
            s = load(Path(args.shelf_pairs), task)
        else:
            s = load(Path(args.shelf), task)
        if task == "same_lcc_pairs" and not args.shelf_pairs:
            logger.warning(
                "  same_lcc_pairs: SHELF arm is using the hub pair config, "
                "a different mining policy from the natural corpora. Pass "
                "--shelf-pairs for a like-for-like comparison."
            )
        if task == "lcc_clustering" and not medians:
            logger.warning(
                "  lcc_clustering: using the SINGLE seed-42 ARI. The "
                "pre-registration fixes per-model MEDIAN ARI over five seeds "
                "and a 0.9 rank-stability gate that can drop clustering from "
                "the claim entirely. Pass --clustering-medians, or treat any "
                "clustering row below as provisional."
            )
        if task == "lcc_clustering" and "shelf" in dropped:
            # The gate applies to the SHELF arm too. clustering_stability_v2
            # writes the SHELF file by default, so this is the axis most
            # likely to fail, and skipping it let a file whose own verdict
            # read "UNSTABLE: drop clustering" still produce a broad verdict.
            logger.info(
                f"{task:<20}{'ALL':<14}   DROPPED (SHELF failed the stability "
                "gate; the whole clustering row is withdrawn)"
            )
            continue
        if task == "lcc_clustering" and "shelf" in medians:
            s = dict(medians["shelf"])
        if not s:
            logger.info(f"{task:<20}{'—':<14}{'—':>4}   no SHELF results yet")
            continue
        for name, d in nat.items():
            if task == "lcc_clustering" and name in dropped:
                logger.info(f"{task:<20}{name:<14}   DROPPED (stability gate)")
                continue
            o = load(d, task)
            if task == "lcc_clustering" and name in medians:
                o = dict(medians[name])
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
            # SHELF failing the gate withdraws clustering for every corpus,
            # so a natural corpus can be short a formulation without being in
            # `dropped` itself. Count that as dropped, not untested.
            n_dropped = 1 if (n in dropped or "shelf" in dropped) else 0
            logger.info(
                f"     {len(PRIMARY) - tested[n] - n_dropped} formulation(s) "
                f"NOT TESTED"
                + (f", {n_dropped} DROPPED by the stability gate" if n_dropped else "")
                + "; the broad claim requires all four"
            )
    # Prereg section 3 requires >= 3 of the FOUR formulations against BOTH
    # corpora. Counting "3 of 3 tested" as sufficient let an untested
    # formulation count toward the broad claim.
    n_formulations = len(PRIMARY)
    broad = all(holds[n] >= 3 and tested[n] == n_formulations for n in nat)
    cls = [p for p in report["pairs"] if p["task"] == "lcc_classification"]
    contradicted = any(p["ci"][0] <= 0 for p in cls) if cls else False

    # Provisional if ANY corpus's clustering row used seed-42 ARI, not only
    # when no medians at all were supplied.
    # A corpus is provisional if its clustering row used seed-42 ARI. That
    # includes the SHELF arm, which is not in report["pairs"] because those
    # rows name only the natural corpus -- omitting it let a run with medians
    # for both natural corpora print a clean verdict while SHELF ran on
    # seed 42.
    has_clustering = any(p["task"] == "lcc_clustering" for p in report["pairs"])
    prov = sorted(
        {
            p["corpus"]
            for p in report["pairs"]
            if p["task"] == "lcc_clustering" and p["corpus"] not in medians
        }
        | ({"shelf"} if has_clustering and "shelf" not in medians else set())
    )
    clustering_provisional = bool(prov)
    if contradicted:
        verdict = "CONTRADICTED: classification interval crosses zero"
    elif broad:
        verdict = "BROAD CLAIM SUPPORTED"
    else:
        verdict = "NARROW CLAIM ONLY: title must scope to classification"
    if clustering_provisional:
        verdict += (
            "  [PROVISIONAL: clustering used seed-42 ARI for "
            + ", ".join(prov)
            + "; not the pre-registered five-seed median, stability gate "
            "not applied there]"
        )
    logger.info(f"\n  VERDICT (pre-registered rule): {verdict}")
    report["verdict"] = verdict

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    logger.info(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
