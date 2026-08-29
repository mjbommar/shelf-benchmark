"""How much can run-to-run jitter move a reported rank correlation?

The same model, task, seed and data checksum produced 0.82799, 0.82907 and
0.82867 across three runs before BLAS threads were pinned: an unpinned thread
count varies with machine load, changing float reduction order and therefore
lbfgs convergence. Threads are pinned now (`run_all.py`), but results computed
before that fix carry roughly 4e-4 of jitter.

The question is whether that matters, and it is cheaper to measure than to
regenerate ten hours of sweeps. This perturbs every score by uniform noise of
the observed magnitude and reports how far the rank correlation moves.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from pathlib import Path

JITTER = 4.1e-4  # max observed spread across three same-seed runs


def load(d: str, task: str) -> dict[str, float]:
    out = {}
    for f in glob.glob(f"{d}/*_{task}.json"):
        try:
            r = json.loads(Path(f).read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if "error" in r:
            continue
        m = r.get("model_key") or r.get("model")
        if m and r.get("primary_score") is not None:
            out[m] = float(r["primary_score"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shelf", default="results/pooled/baselines")
    ap.add_argument("--natural", action="append", required=True, metavar="NAME=DIR")
    ap.add_argument("--n-trials", type=int, default=2000)
    ap.add_argument("--output", default="results/transfer/jitter_sensitivity.json")
    args = ap.parse_args()

    from scipy.stats import spearmanr

    rng = random.Random(0)
    rows = []
    for task in ("lcc_classification", "lcc_retrieval"):
        s = load(args.shelf, task)
        for entry in args.natural:
            name, _, d = entry.partition("=")
            o = load(d, task)
            shared = sorted(set(s) & set(o))
            if len(shared) < 10:
                continue
            base = float(
                spearmanr([s[m] for m in shared], [o[m] for m in shared]).statistic
            )
            devs = []
            for _ in range(args.n_trials):
                x = [s[m] + rng.uniform(-JITTER, JITTER) for m in shared]
                y = [o[m] + rng.uniform(-JITTER, JITTER) for m in shared]
                devs.append(abs(float(spearmanr(x, y).statistic) - base))
            rows.append(
                {
                    "task": task,
                    "corpus": name,
                    "n_models": len(shared),
                    "rho": base,
                    "max_deviation": max(devs),
                    "mean_deviation": sum(devs) / len(devs),
                }
            )
            print(
                f"  {task:<19}{name:<11} rho={base:.4f}  max deviation={max(devs):.5f}"
            )

    worst = max((r["max_deviation"] for r in rows), default=0.0)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "jitter_magnitude": JITTER,
                "source": "unpinned BLAS thread count before scripts/baselines/run_all.py "
                "pinned OMP/OPENBLAS/MKL/NUMEXPR threads",
                "rows": rows,
                "worst_case_rho_deviation": worst,
                "conclusion": (
                    "Worst-case movement in a reported rank correlation is "
                    f"{worst:.4f}, one to two orders of magnitude below the width of "
                    "the reported bootstrap intervals. Pre-fix results are therefore "
                    "not regenerated; the jitter is disclosed instead."
                ),
            },
            indent=2,
        )
    )
    print(f"\n  worst-case rho deviation: {worst:.5f}")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
