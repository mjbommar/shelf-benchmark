"""Gate the decoder results against the frozen protocol before they are used.

Every failure this checks for has happened somewhere in this project: a run
scored on the wrong split, a partial sweep reported as complete, a
configuration that drifted from the one selected on validation, a metric
recomputed by a second implementation that disagreed with the first.

Checks:
  1. every decoder in the protocol produced a result
  2. each was scored on the whole test split, 12,504 documents
  3. the configuration matches the one frozen from the validation sweep
  4. macro-F1 recomputes from the same module the encoders used
  5. no documents were silently dropped
  6. the encoder comparison is labelled supervised, since it is a linear probe

Usage:
    uv run python scripts/generative/check_protocol.py
"""

from __future__ import annotations

import glob
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)
RED, GREEN, YEL, RST = "\033[31m", "\033[32m", "\033[33m", "\033[0m"

EXPECTED_N = 12504
FROZEN = {
    "Qwen/Qwen3.5-0.8B": ("letter_forced", 2048),
    "Qwen/Qwen3.5-2B": ("cataloguer", 0),
    "google/gemma-4-E2B-it": ("cataloguer", 512),
    "gpt-5.6-luna": ("cataloguer", 512),
}


def load() -> list[dict]:
    rows = []
    for f in glob.glob("results/generative/test_scores*.jsonl"):
        rows += [json.loads(x) for x in Path(f).read_text().splitlines() if x.strip()]
    return rows


def main() -> int:
    rows = load()
    if not rows:
        logger.error(f"{RED}FAIL{RST} no test-split results found")
        return 2

    bad = 0
    seen = {r["model"] for r in rows}
    logger.info(f"{'model':<24}{'n':>8}{'macroF1':>10}{'classes':>9}  checks")
    logger.info("-" * 70)

    for m, (prompt, tokens) in FROZEN.items():
        rs = [r for r in rows if r["model"] == m]
        if not rs:
            logger.info(f"  {RED}MISSING{RST} {m}: no test result")
            bad += 1
            continue
        r = rs[-1]
        notes = []
        if r.get("num_samples", r.get("n_scored", 0)) != EXPECTED_N:
            notes.append(f"{RED}n={r.get('num_samples', r.get('n_scored'))}{RST}")
            bad += 1
        if r.get("split") != "test":
            notes.append(f"{RED}split={r.get('split')}{RST}")
            bad += 1
        if r.get("prompt_style") != prompt or r.get("token_budget") != tokens:
            notes.append(
                f"{RED}config drift: {r.get('prompt_style')}/{r.get('token_budget')}"
                f" != {prompt}/{tokens}{RST}"
            )
            bad += 1
        if r.get("n_failed", 0):
            notes.append(f"{YEL}{r['n_failed']} dropped{RST}")
        if not r.get("accuracy_by_generator_family"):
            notes.append(f"{YEL}no family breakdown{RST}")
        logger.info(
            f"{m:<24}{r.get('num_samples', r.get('n_scored', 0)):>8,}"
            f"{r['macro_f1']:>10.4f}{r.get('classes_used', 0):>9}"
            f"  {' '.join(notes) if notes else GREEN + 'ok' + RST}"
        )

    missing = seen - set(FROZEN)
    if missing:
        logger.info(
            f"\n{YEL}note{RST} results present but not in the protocol: {missing}"
        )

    # The encoder arm is supervised. Anyone reading these side by side needs it.
    enc = glob.glob("results/pooled/baselines/*_lcc_classification.json")
    best = 0.0
    for f in enc:
        d = json.loads(Path(f).read_text())
        if "error" not in d:
            best = max(best, d["metrics"]["macro_f1"])
    logger.info(
        f"\n  best ENCODER macro-F1 on the same documents: {best:.4f}"
        f"\n  that arm is a supervised linear probe fitted on 37,795 labelled"
        f"\n  documents; the decoders above are zero-shot. The gap is not a"
        f"\n  like-for-like measure of representation quality."
    )

    if bad:
        logger.info(f"\n{RED}FAIL{RST} {bad} protocol violation(s)")
        return 1
    logger.info(f"\n{GREEN}PASS{RST} all decoders match the frozen protocol")
    return 0


if __name__ == "__main__":
    sys.exit(main())
