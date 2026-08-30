"""Does label leakage inflate scores, or decide which model wins?

Pre-registration section 4. Three conditions per corpus -- unmasked, masked,
and a sham that removes an equal number of random tokens -- so that a score
change can be attributed to removing label signal rather than to damaging the
document.

Reports, per corpus and task:

1. score change, masked and sham against unmasked;
2. rank correlation, masked against unmasked;
3. rank correlation, masked SHELF against masked natural corpora.

The interpretation was fixed before running:

- masking lowers scores but preserves ranking, and sham does not reproduce
  the drop -> leakage inflates scores uniformly; the ranking claim survives;
- masking changes the ranking beyond the sham's effect -> leakage was
  distorting model selection, and masking is a corpus fix rather than a
  robustness check;
- masked-against-natural agreement exceeds unmasked-against-natural ->
  leakage was actively moving the corpus away from natural behaviour.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import statistics
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PRIMARY = {"lcc_classification": "macro_f1", "lcc_retrieval": "ndcg@10"}


def load(d: Path, task: str) -> dict[str, float]:
    key = PRIMARY[task]
    out = {}
    for f in sorted(d.glob(f"*_{task}.json")):
        try:
            r = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if "error" in r:
            continue
        m = r.get("model_key") or r.get("model")
        v = (r.get("metrics") or {}).get(key)
        if v is None:
            raise SystemExit(
                f"{f.name}: metric '{key}' missing; refusing to "
                "fall back to primary_score"
            )
        if m and v is not None:
            out[m] = float(v)
    return out


def spearman(a, b):
    from scipy.stats import spearmanr

    return float(spearmanr(a, b).statistic)


def _boot_rho_diff(models, u, m, s, n=2000, seed=42):
    """Interval on rho(u,m) - rho(u,s), resampling models once per replicate."""
    rng = random.Random(seed)
    st = []
    for _ in range(n):
        p = [models[rng.randrange(len(models))] for _ in range(len(models))]
        xu = [u[k] for k in p]
        if len(set(xu)) < 2:
            continue
        xm, xs = [m[k] for k in p], [s[k] for k in p]
        if len(set(xm)) < 2 or len(set(xs)) < 2:
            continue
        st.append(spearman(xu, xm) - spearman(xu, xs))
    if not st:
        return float("nan"), float("nan")
    st.sort()
    return st[int(0.025 * len(st))], st[int(0.975 * len(st))]


def _boot_mean_diff(dm: list[float], ds: list[float], n=2000, seed=42):
    """Interval on (mean masked delta - mean sham delta), paired by model."""
    rng = random.Random(seed)
    diffs = [a - b for a, b in zip(dm, ds, strict=True)]
    st = []
    for _ in range(n):
        samp = [diffs[rng.randrange(len(diffs))] for _ in range(len(diffs))]
        st.append(statistics.mean(samp))
    st.sort()
    return st[int(0.025 * len(st))], st[int(0.975 * len(st))]


def boot_rho(models, x, y, n=2000, seed=42):
    rng = random.Random(seed)
    st = []
    for _ in range(n):
        p = [models[rng.randrange(len(models))] for _ in range(len(models))]
        xs, ys = [x[m] for m in p], [y[m] for m in p]
        if len(set(xs)) > 1 and len(set(ys)) > 1:
            st.append(spearman(xs, ys))
    if not st:
        return float("nan"), float("nan")
    st.sort()
    return st[int(0.025 * len(st))], st[int(0.975 * len(st))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--corpus",
        action="append",
        required=True,
        metavar="NAME=BASEDIR",
        help=(
            "BASEDIR without the _masked/_sham suffix, or an explicit "
            "UNMASKED:MASKED:SHAM triple when the unmasked results do not "
            "share the arms' directory prefix"
        ),
    )
    ap.add_argument("--output", default="results/transfer/masking_ablation.json")
    ap.add_argument(
        "--exclude-models",
        default="ogbert_110m_sentence",
        help=(
            "Model keys to drop before correlating. Defaults to the weight "
            "duplicate: ogbert-110m-base and ogbert-110m-sentence are one "
            "safetensors blob with one SHA-256, and a pair that is one model "
            "always agrees with itself."
        ),
    )
    args = ap.parse_args()

    # A corpus is either a prefix (base, base_masked, base_sham) or an explicit
    # triple. The triple exists because the pooled corpus keeps its unmasked
    # results in results/pooled while its arms are results/all_masked and
    # results/all_sham, so no single prefix reaches all three.
    corpora = {}
    for e in args.corpus:
        n, _, d = e.partition("=")
        parts = d.split(":")
        if len(parts) == 3:
            corpora[n] = tuple(parts)
        elif len(parts) == 1:
            corpora[n] = (d, f"{d}_masked", f"{d}_sham")
        else:
            ap.error(f"--corpus {n}: expected BASEDIR or UNMASKED:MASKED:SHAM")

    excluded = {m.strip() for m in args.exclude_models.split(",") if m.strip()}
    if excluded:
        logger.info(f"excluding {len(excluded)}: {', '.join(sorted(excluded))}")

    report: dict = {
        "conditions": ["unmasked", "masked", "sham"],
        "excluded_models": sorted(excluded),
        "rows": [],
    }

    for task in PRIMARY:
        logger.info(f"\n=== {task} ===")
        logger.info(
            f"{'corpus':<12}{'n':>4}{'unmask':>9}{'mask':>9}{'sham':>9}"
            f"{'d_mask':>9}{'d_sham':>9}{'rho m/u':>10}{'95% CI':>16}"
        )
        logger.info("-" * 87)
        masked_scores = {}
        for name, (d_u, d_m, d_s) in corpora.items():
            u = load(Path(f"{d_u}/baselines"), task)
            m = load(Path(f"{d_m}/baselines"), task)
            s = load(Path(f"{d_s}/baselines"), task)
            shared = sorted((set(u) & set(m) & set(s)) - excluded)
            if len(shared) < 4:
                logger.info(f"{name:<12}{len(shared):>4}   incomplete")
                continue
            mu_u = statistics.mean(u[k] for k in shared)
            mu_m = statistics.mean(m[k] for k in shared)
            mu_s = statistics.mean(s[k] for k in shared)
            rho = spearman([u[k] for k in shared], [m[k] for k in shared])
            lo, hi = boot_rho(shared, u, m)
            # The pre-registered rule turns on the sham's effect ON RANKING,
            # not only on scores. Without this the rule is undecidable.
            rho_s = spearman([u[k] for k in shared], [s[k] for k in shared])
            slo, shi = boot_rho(shared, u, s)
            # The rule's ranking arm is rho(u,m) < rho(u,s). Two overlapping
            # CIs are not a test of that; bootstrap the DIFFERENCE.
            rlo, rhi = _boot_rho_diff(shared, u, m, s)
            # Per-model paired deltas, so the headline is not a mean of means
            # over heterogeneous models.
            dm = [m[k] - u[k] for k in shared]
            ds = [s[k] - u[k] for k in shared]
            dlo, dhi = _boot_mean_diff(dm, ds)
            score_sep = dhi < 0  # masking beats sham on scores
            rank_changed = rhi < 0  # masking degrades ranking beyond sham
            verdict = (
                "leakage distorted model selection"
                if rank_changed
                else "leakage inflated scores uniformly; ranking preserved"
                if score_sep
                else "not separable from sham"
            )
            logger.info(
                f"{name:<12}{len(shared):>4}{mu_u:>9.4f}{mu_m:>9.4f}{mu_s:>9.4f}"
                f"{statistics.mean(dm):>9.4f}{statistics.mean(ds):>9.4f}"
                f"{rho:>10.3f}{f'[{lo:.2f}, {hi:.2f}]':>16}"
            )
            logger.info(
                f"{'':>16}sham rank rho {rho_s:.3f} [{slo:.2f}, {shi:.2f}]   "
                f"paired (mask-sham) delta {statistics.mean(dm) - statistics.mean(ds):+.4f} "
                f"[{dlo:+.4f}, {dhi:+.4f}]"
                f"{'  <- masking effect exceeds sham' if score_sep else '  <- NOT separable from sham'}"
            )
            logger.info(
                f"{'':>16}rank arm: rho(u,m)-rho(u,s) = {rho - rho_s:+.3f} "
                f"[{rlo:+.3f}, {rhi:+.3f}]   VERDICT: {verdict}"
            )
            masked_scores[name] = m
            report["rows"].append(
                {
                    "task": task,
                    "corpus": name,
                    "n_models": len(shared),
                    "mean_unmasked": mu_u,
                    "mean_masked": mu_m,
                    "mean_sham": mu_s,
                    "delta_masked": mu_m - mu_u,
                    "delta_sham": mu_s - mu_u,
                    "rho_masked_vs_unmasked": rho,
                    "ci": [lo, hi],
                    # These were computed, printed, and then dropped. The
                    # paper's "spans zero" claim had no artifact behind it.
                    "rho_sham_vs_unmasked": rho_s,
                    "rho_diff_masked_minus_sham": rho - rho_s,
                    "rho_diff_ci": [rlo, rhi],
                    "paired_mean_diff": statistics.mean(dm) - statistics.mean(ds),
                    "paired_mean_diff_ci": [dlo, dhi],
                    "paired_mean_diff_resamples": "models",
                    "paired_mean_diff_note": (
                        "The pre-registration fixes documents as the resampling "
                        "unit for classification and retrieval. This interval "
                        "resamples models instead, because the quantity is a "
                        "difference of per-model means and the per-document "
                        "predictions are not retained by the sweep. Recorded as "
                        "a deviation rather than presented as the frozen test."
                    ),
                }
            )

        # masked SHELF against masked natural corpora
        if "shelf" in masked_scores:
            for name, sc in masked_scores.items():
                if name == "shelf":
                    continue
                # Same exclusion as the per-corpus rows above: without it the
                # cross-corpus lines silently ran on 22 models while the table
                # above them ran on 21.
                shared = sorted((set(masked_scores["shelf"]) & set(sc)) - excluded)
                if len(shared) < 4:
                    continue
                rho = spearman(
                    [masked_scores["shelf"][k] for k in shared], [sc[k] for k in shared]
                )
                lo, hi = boot_rho(shared, masked_scores["shelf"], sc)
                logger.info(
                    f"  masked shelf vs masked {name}: rho {rho:.3f} "
                    f"[{lo:.2f}, {hi:.2f}]  n={len(shared)}"
                )
                report["rows"].append(
                    {
                        "task": task,
                        "comparison": f"masked_shelf_vs_masked_{name}",
                        "spearman": rho,
                        "ci": [lo, hi],
                        "n_models": len(shared),
                    }
                )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    logger.info(f"\nwrote {out}")
    logger.info(
        "\nRead d_mask against d_sham. A drop the sham reproduces is document "
        "damage, not leakage."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
