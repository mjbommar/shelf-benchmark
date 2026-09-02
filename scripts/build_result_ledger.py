"""Describe the whole result set, not the last run that touched it.

Each results directory carries a `manifest.json` written by the sweep that
last ran there. After an incremental run that manifest names one model, which
is accurate about that invocation and misleading about the directory: a reader
finds a manifest listing a single model beside 600 result files.

This walks the result files themselves and writes a ledger of what is actually
present: every model, every task, which cells exist, which are error stubs,
which models are excluded from correlations and why, and the code and corpus
revisions. It is derived from the files, so it cannot drift from them.

Usage:
    uv run python scripts/build_result_ledger.py --output results/LEDGER.json
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Why a model may be absent from a rank correlation. Stated here so the ledger
# records the reason and not only the fact.
EXCLUSIONS = {
    "ogbert_2m_sentence": "retired from the reported model panel",
    "ogbert_110m_sentence": (
        "shares one safetensors blob and SHA-256 with ogbert-110m-base; "
        "counting both adds a pair that always agrees with itself"
    ),
    "gte_modernbert_8k": (
        "the same weights as gte-modernbert-2k at a larger context budget; "
        "belongs to the truncation experiment, not the model panel"
    ),
}

RESTRICTED_ALSO_DROPS = {
    "ogbert_110m_base": "36M-to-139M OGBert line, well below deployable quality",
    "ogbert_v1_mlm": "masked-LM checkpoint without sentence training",
    "roberta": "mean pooling without sentence training, scores 0.6715",
}


def git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results")
    ap.add_argument("--output", default="results/LEDGER.json")
    args = ap.parse_args()

    root = Path(args.results)
    dirs = sorted(p for p in root.glob("*/baselines") if p.is_dir())

    arms: dict[str, dict] = {}
    for d in dirs:
        models: set[str] = set()
        tasks: set[str] = set()
        cells = stubs = 0
        checksums: set[str] = set()
        for f in sorted(d.glob("*.json")):
            if f.name in ("summary.json", "manifest.json"):
                continue
            try:
                j = json.loads(f.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            key = j.get("model_key")
            task = j.get("task")
            if not key or not task:
                continue
            models.add(key)
            tasks.add(task)
            if "error" in j or "primary_score" not in j:
                stubs += 1
            else:
                cells += 1
            c = (j.get("context") or {}).get("dataset_checksum")
            if c:
                checksums.add(str(c))
        if not models:
            continue
        arms[d.parent.name] = {
            "path": str(d),
            "n_models": len(models),
            "n_tasks": len(tasks),
            "scored_cells": cells,
            "error_stubs": stubs,
            "models": sorted(models),
            "tasks": sorted(tasks),
            "dataset_checksums": sorted(checksums),
        }

    # Tasks that fail for every model are a config defect, not a coverage gap.
    task_fail: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for a in arms.values():
        for f in Path(a["path"]).glob("*.json"):
            if f.name in ("summary.json", "manifest.json"):
                continue
            try:
                j = json.loads(f.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            t = j.get("task")
            if not t:
                continue
            bad = "error" in j or "primary_score" not in j
            task_fail[t][1] += 1
            task_fail[t][0] += 1 if bad else 0
    always_fail = sorted(t for t, (b, n) in task_fail.items() if n and b == n)

    ledger = {
        "generated_from": "the result files themselves, not from a run log",
        "code_revision": git("rev-parse", "HEAD"),
        "code_dirty": bool(git("status", "--porcelain")),
        "n_arms": len(arms),
        "total_scored_cells": sum(a["scored_cells"] for a in arms.values()),
        "total_error_stubs": sum(a["error_stubs"] for a in arms.values()),
        "arms": arms,
        "excluded_from_rank_correlations": EXCLUSIONS,
        "additionally_excluded_in_restricted_analysis": RESTRICTED_ALSO_DROPS,
        "tasks_failing_for_every_model": always_fail,
        "note": (
            "A task listed in tasks_failing_for_every_model is a configuration "
            "defect rather than missing coverage; lcc_subclass_classification "
            "names a column the pooled corpus does not carry."
        ),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ledger, indent=2))

    logger.info(f"{'arm':<34}{'models':>8}{'tasks':>7}{'cells':>8}{'stubs':>7}")
    logger.info("-" * 64)
    for name, a in sorted(arms.items()):
        logger.info(
            f"{name:<34}{a['n_models']:>8}{a['n_tasks']:>7}"
            f"{a['scored_cells']:>8}{a['error_stubs']:>7}"
        )
    logger.info("-" * 64)
    logger.info(
        f"{'total':<34}{'':>8}{'':>7}"
        f"{ledger['total_scored_cells']:>8}{ledger['total_error_stubs']:>7}"
    )
    if always_fail:
        logger.info(f"\ntasks failing for every model: {', '.join(always_fail)}")
    logger.info(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
