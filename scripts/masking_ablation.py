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
        v = (r.get("metrics") or {}).get(key, r.get("primary_score"))
        if m and v is not None:
            out[m] = float(v)
    return out


def spearman(a, b):
    from scipy.stats import spearmanr

    return float(spearmanr(a, b).statistic)


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
        help="BASEDIR without the _masked/_sham suffix",
    )
    ap.add_argument("--output", default="results/transfer/masking_ablation.json")
    args = ap.parse_args()

    corpora = {}
    for e in args.corpus:
        n, _, d = e.partition("=")
        corpora[n] = d

    report: dict = {"conditions": ["unmasked", "masked", "sham"], "rows": []}

    for task in PRIMARY:
        logger.info(f"\n=== {task} ===")
        logger.info(
            f"{'corpus':<12}{'n':>4}{'unmask':>9}{'mask':>9}{'sham':>9}"
            f"{'d_mask':>9}{'d_sham':>9}{'rho m/u':>10}{'95% CI':>16}"
        )
        logger.info("-" * 87)
        masked_scores = {}
        for name, base in corpora.items():
            u = load(Path(f"{base}/baselines"), task)
            m = load(Path(f"{base}_masked/baselines"), task)
            s = load(Path(f"{base}_sham/baselines"), task)
            shared = sorted(set(u) & set(m) & set(s))
            if len(shared) < 4:
                logger.info(f"{name:<12}{len(shared):>4}   incomplete")
                continue
            mu_u = statistics.mean(u[k] for k in shared)
            mu_m = statistics.mean(m[k] for k in shared)
            mu_s = statistics.mean(s[k] for k in shared)
            rho = spearman([u[k] for k in shared], [m[k] for k in shared])
            lo, hi = boot_rho(shared, u, m)
            logger.info(
                f"{name:<12}{len(shared):>4}{mu_u:>9.4f}{mu_m:>9.4f}{mu_s:>9.4f}"
                f"{mu_m - mu_u:>9.4f}{mu_s - mu_u:>9.4f}{rho:>10.3f}"
                f"{f'[{lo:.2f}, {hi:.2f}]':>16}"
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
                }
            )

        # masked SHELF against masked natural corpora
        if "shelf" in masked_scores:
            for name, sc in masked_scores.items():
                if name == "shelf":
                    continue
                shared = sorted(set(masked_scores["shelf"]) & set(sc))
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
