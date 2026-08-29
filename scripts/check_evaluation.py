"""Gate results before they reach a paper, a card, or a claim.

Every check here exists because its absence produced a wrong number in this
project. See docs/EVALUATION_CHECKLIST.md for the incident behind each one.

    uv run python scripts/check_evaluation.py --results results/pooled/baselines

Exit status is non-zero if any check fails, so it can gate a build.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

GREEN, RED, YELLOW, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[0m"


def load(results_dir: Path) -> tuple[dict, dict, list]:
    """Return (model -> tasks with real results, model -> tasks that errored)."""
    ok: dict[str, set[str]] = defaultdict(set)
    err: dict[str, set[str]] = defaultdict(set)
    unreadable: list[str] = []
    for f in sorted(results_dir.glob("*.json")):
        if f.name.startswith(("summary", "manifest", "baseline_summary")):
            continue
        try:
            r = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            unreadable.append(f.name)
            continue
        model = r.get("model_key") or r.get("model")
        task = r.get("task")
        if not model or not task:
            unreadable.append(f.name)
            continue
        (err if "error" in r else ok)[model].add(task)
    return dict(ok), dict(err), unreadable


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True)
    ap.add_argument(
        "--exclude-models",
        default="finetune",
        help=(
            "Comma-separated substrings; matching models are dropped from the "
            "completeness denominator. Default excludes fine-tuned models, "
            "which are not zero-shot baselines and would mislead if mixed into "
            "a baseline table. Pass an empty string to count everything."
        ),
    )
    ap.add_argument("--config", default="scripts/baselines/config.yaml")
    ap.add_argument(
        "--min-large-params",
        type=int,
        default=300_000_000,
        help="A model at or above this size counts as 'large' for B2.",
    )
    args = ap.parse_args()

    import yaml

    results = Path(args.results)
    if not results.is_dir():
        print(f"{RED}FAIL{RESET} results directory not found: {results}")
        return 2

    # Accept either the run directory or its baselines/ subdirectory. Pointing
    # this at the parent used to glob nothing and then report every model as
    # "not started", which reads like a finding rather than a bad path.
    if not any(results.glob("*.json")) and (results / "baselines").is_dir():
        results = results / "baselines"
        print(f"    (descended into {results})")

    # A gate that scores an empty directory is worse than no gate: "0 of 23
    # complete" looks like a real answer. Refuse instead.
    n_files = len([f for f in results.glob("*.json") if f.name != "summary.json"])
    if n_files == 0:
        print(
            f"{RED}FAIL{RESET} no result files in {results}. "
            "Nothing was checked -- this is a bad path or an empty run, "
            "not a finding about model coverage."
        )
        return 2

    cfg = yaml.safe_load(Path(args.config).read_text())
    models_cfg = cfg.get("models", {})
    drop = [x for x in args.exclude_models.split(",") if x]
    if drop:
        excluded = [m for m in models_cfg if any(d in m for d in drop)]
        if excluded:
            print(
                "    excluded from completeness (not zero-shot baselines): "
                + ", ".join(sorted(excluded))
            )
        models_cfg = {m: v for m, v in models_cfg.items() if m not in excluded}
    tasks_cfg = cfg.get("tasks", {})
    all_tasks = {t for tl in tasks_cfg.values() for t in tl}

    ok, err, unreadable = load(results)
    failures: list[str] = []
    warnings: list[str] = []

    print(f"Auditing {results}\n")

    # --- B3: count results, never files -------------------------------
    n_files = len(
        [
            f
            for f in results.glob("*.json")
            if not f.name.startswith(("summary", "manifest", "baseline_summary"))
        ]
    )
    n_ok = sum(len(v) for v in ok.values())
    n_err = sum(len(v) for v in err.values())
    print(
        f"B3  files={n_files}  real results={n_ok}  error stubs={n_err}"
        f"  unreadable={len(unreadable)}"
    )
    if n_err:
        detail = {m: sorted(t) for m, t in err.items()}
        print(f"    {YELLOW}error stubs must never be counted as results:{RESET}")
        for m, ts in list(detail.items())[:5]:
            print(f"      {m}: {', '.join(ts)}")
    if unreadable:
        failures.append(f"B3: {len(unreadable)} unreadable result files")

    # --- B1: completeness ---------------------------------------------
    # --- B1: completeness ---------------------------------------------
    # A task that errors for EVERY model is structurally impossible on this
    # corpus rather than a gap in the sweep: lcc_subclass_classification needs
    # a column the pooled corpus does not carry, because that tier was
    # withdrawn. Excluding it keeps the gate reachable. It is printed so the
    # exclusion is visible rather than silent.
    attempted = set(ok) | set(err)
    impossible = {
        t
        for t in all_tasks
        if any(t in err.get(m, set()) for m in attempted)
        and not any(t in ok.get(m, set()) for m in attempted)
    }
    if impossible:
        print(
            "    structurally impossible on this corpus, excluded from B1: "
            + ", ".join(sorted(impossible))
        )

    expected = {
        m: {
            t
            for t in all_tasks
            if _supports(models_cfg[m], t, tasks_cfg) and t not in impossible
        }
        for m in models_cfg
    }
    complete = [m for m in expected if expected[m] and ok.get(m, set()) >= expected[m]]
    partial = [
        m
        for m in expected
        if expected[m] and 0 < len(ok.get(m, set())) < len(expected[m])
    ]
    missing = [m for m in expected if expected[m] and not ok.get(m)]
    print(
        f"\nB1  models complete={len(complete)}  partial={len(partial)}  "
        f"not started={len(missing)}  (of {len(expected)} configured)"
    )
    if partial or missing:
        warnings.append(
            f"B1: sweep is PARTIAL ({len(complete)}/{len(expected)} models complete). "
            "Any number from it must be labelled partial in the same sentence."
        )
        if missing:
            print(
                f"    not started: {', '.join(sorted(missing)[:8])}"
                + (" ..." if len(missing) > 8 else "")
            )

    # --- B2: is the partial sample biased? -----------------------------
    def params(m):
        return models_cfg.get(m, {}).get("num_params") or 0

    done_large = [m for m in ok if params(m) >= args.min_large_params]
    todo_large = [
        m for m in expected if params(m) >= args.min_large_params and m not in ok
    ]
    print(
        f"\nB2  large models (>= {args.min_large_params / 1e6:.0f}M): "
        f"{len(done_large)} done, {len(todo_large)} missing"
    )
    if todo_large and (partial or missing):
        failures.append(
            f"B2: BIASED SAMPLE -- {len(todo_large)} large models missing while "
            f"{len(done_large)} present. A partial sweep finishes small models "
            "first, so this is a small-model table, not a random subset. "
            f"Missing: {', '.join(sorted(todo_large))}"
        )

    # --- A1/A2: one corpus per claim -----------------------------------
    # Provenance lives in context.dataset_checksum, with the seed and library
    # versions beside it. Several checksums per directory is EXPECTED: a pair
    # task reads different data from a classification task. What must not
    # happen is one task carrying two checksums inside one corpus, which means
    # results were mixed across builds of the data.
    stamped = 0
    per_task_checksums: dict[str, set[str]] = defaultdict(set)
    for f in results.glob("*.json"):
        if f.name.startswith(("summary", "manifest", "baseline_summary")):
            continue
        try:
            r = json.loads(f.read_text())
        except Exception:
            continue
        if "error" in r:
            continue  # error stubs legitimately carry no provenance
        ctx = r.get("context") or {}
        cks = ctx.get("dataset_checksum")
        if cks and ctx.get("random_seed") is not None:
            stamped += 1
        if cks and r.get("task"):
            per_task_checksums[r["task"]].add(cks)
    print(f"\nA2  results with checksum and seed: {stamped}/{n_ok}")
    if stamped < n_ok:
        warnings.append(f"A2: {n_ok - stamped} results lack a checksum or seed.")

    mixed = {t: c for t, c in per_task_checksums.items() if len(c) > 1}
    print(f"A1  tasks whose results mix data builds: {len(mixed)}")
    if mixed:
        failures.append(
            "A1: MIXED DATA BUILDS within one corpus for "
            f"{len(mixed)} task(s): {', '.join(sorted(mixed)[:5])}. "
            "All results for a task must come from one build of the data."
        )

    # --- A4: does the evidence cover the scope of the claim? -----------
    # Not fully automatable, but the common failure is mechanical: a claim
    # spanning several task formulations resting on results for one. Report
    # which formulations have results so the gap is visible.
    families = {
        "classification": [t for t in all_tasks if "classification" in t],
        "retrieval": [t for t in all_tasks if "retrieval" in t],
        "clustering": [t for t in all_tasks if "clustering" in t],
        "pairs": [t for t in all_tasks if t.endswith("_pairs")],
    }
    covered = {
        fam: sum(1 for t in ts if any(t in v for v in ok.values()))
        for fam, ts in families.items()
    }
    missing = [f for f, c in covered.items() if c == 0]
    print(
        "\nA4  task families with results: "
        + ", ".join(f"{f}={c}" for f, c in covered.items())
    )
    if missing:
        warnings.append(
            f"A4: no results for {', '.join(missing)}. A claim spanning task "
            "formulations must not rest on the ones that happen to be run."
        )

    # --- G1: summary must not be narrower than the directory -----------
    for name in ("summary.json", "baseline_summary.json"):
        sp = results / name
        if not sp.exists():
            continue
        try:
            summ = json.loads(sp.read_text())
        except Exception:
            failures.append(f"G1: {name} is unreadable")
            continue
        in_summary = len(summ.get("results", {}))
        print(f"\nG1  {name}: {in_summary} entries vs {n_ok} real results on disk")
        if in_summary < n_ok:
            failures.append(
                f"G1: {name} holds {in_summary} entries but {n_ok} results exist. "
                "A partial run overwrote the aggregate. Rebuild with "
                "--aggregate-only."
            )

    # --- report --------------------------------------------------------
    print()
    if failures:
        for f in failures:
            print(f"{RED}FAIL{RESET} {f}")
    for w in warnings:
        print(f"{YELLOW}WARN{RESET} {w}")
    if not failures and not warnings:
        print(f"{GREEN}PASS{RESET} results are complete, unbiased, and stamped.")
    print(
        "\nStatistics and metric checks (D, E) are not automatable. Read "
        "docs/EVALUATION_CHECKLIST.md before writing them up."
    )
    return 1 if failures else 0


def _supports(model_cfg: dict, task: str, tasks_cfg: dict) -> bool:
    supports = model_cfg.get("supports", [])
    for task_type, task_list in tasks_cfg.items():
        if task in task_list:
            return task_type in supports
    return False


if __name__ == "__main__":
    sys.exit(main())
