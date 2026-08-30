"""Check every table number in the paper against the result files.

Each table is hand-transcribed, so a number can drift from its artifact
without anything failing. This reads the LaTeX, finds each table by its
label, and compares it to the results on disk.

Two traps are encoded here because both caught the author mid-audit:

  * clustering ``primary_score`` is v_measure, not the adjusted Rand index
    the paper reports. Use ``metrics.ari``.
  * pair ``primary_score`` is f1, which is degenerate on these tasks --
    three of six sit at exactly 0.667, the all-positive classifier. The
    paper reports ``metrics.auc_roc``.

Getting either wrong makes every row look broken, which reads like a
finding about the paper rather than a bug in the checker.

Usage:
    uv run python scripts/verify_paper_numbers.py
    uv run python scripts/verify_paper_numbers.py --paper ../shelf-paper
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import re
import statistics
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

RED, GREEN, RESET = "\033[31m", "\033[32m", "\033[0m"

# Display names in the paper that differ from the name in the result file.
ALIAS = {
    "DistilBERT": "distilbert-base-uncased",
    "OGBert-110M Sent.": "OGBert-110M Sentence",
    "ogbert-2m-sent.": "ogbert-2m-sentence",
}

# label -> (section file, columns, metric or None for primary_score)
PER_MODEL = {
    "tab:baselines-cls": (
        "04b_baselines.tex",
        [
            "lcc_classification",
            "lcgft_category_classification",
            "form_classification",
            "register_classification",
        ],
        None,
    ),
    "tab:headroom": (
        "06b_headroom.tex",
        ["lcc_classification", "form_classification"],
        None,
    ),
}

# label -> (section file, [(row label, task, metric)], summary stat per column)
SUMMARY = {
    "tab:baselines-clu": (
        "04b_baselines.tex",
        [
            "lcc_clustering",
            "lcgft_clustering",
            "register_clustering",
            "geographic_clustering",
        ],
        "ari",
        ["best", "median", "worst"],
    ),
    "tab:retrieval": (
        "04c_retrieval.tex",
        ["lcc_retrieval", "form_retrieval", "category_retrieval"],
        None,
        ["best", "median", "bm25"],
    ),
    "tab:pairs": (
        "04c_retrieval.tex",
        [
            "same_lcc_pairs",
            "same_topic_pairs",
            "topic_overlap_pairs",
            "same_form_pairs",
            "same_register_pairs",
            "same_audience_pairs",
        ],
        "auc_roc",
        ["best", "median"],
    ),
    "tab:instruction": (
        "04c_retrieval.tex",
        [
            "instruction_same_subject_diff_form",
            "instruction_same_topic_diff_subject",
            "instruction_same_form_diff_subject",
            "instruction_same_audience_diff_register",
        ],
        None,
        ["best", "median"],
    ),
}


# Tables whose numbers come from a derived artifact rather than per-model
# result files. Each entry maps a row label in the table to the artifact row
# and the fields to compare, in table-column order.
ARTIFACT_TABLES = {
    "tab:masking": {
        "file": "06_surface.tex",
        "artifact": "results/transfer/masking_ablation.json",
        "task": "lcc_classification",
        "rows": {
            "shelf": "SHELF",
            "gutenberg": "Gutenberg",
            "lcshbench": "LCSHBench",
        },
        "fields": [
            "mean_unmasked",
            "mean_masked",
            "delta_masked",
            "mean_sham",
            "delta_sham",
        ],
    },
}


def check_artifact_tables(sections: Path, tol: float) -> tuple[int, int]:
    """Compare tables built from a derived JSON artifact."""
    checked = bad = 0
    for label, spec in ARTIFACT_TABLES.items():
        path = Path(spec["artifact"])
        if not path.exists():
            logger.warning(f"  {label}: artifact {path} missing")
            continue
        data = json.loads(path.read_text())
        rows = {
            (r["corpus"], r["task"]): r
            for r in data.get("rows", [])
            if "corpus" in r and "task" in r
        }
        tex = (sections / spec["file"]).read_text()
        blk = table_block(tex, label)
        if blk is None:
            logger.warning(f"  {label}: not found in {spec['file']}")
            continue
        for corpus, display in spec["rows"].items():
            line = next(
                (ln for ln in blk.splitlines() if ln.strip().startswith(display)), None
            )
            if line is None:
                logger.info(f"  {RED}MISMATCH{RESET} {label}: no row for {display}")
                bad += 1
                continue
            vals = [float(x) for x in re.findall(r"-?\d\.\d{4}", line)]
            art = rows.get((corpus, spec["task"]))
            if art is None or len(vals) != len(spec["fields"]):
                logger.info(f"  {RED}MISMATCH{RESET} {label} {display}: shape")
                bad += 1
                continue
            for field, paper in zip(spec["fields"], vals, strict=True):
                checked += 1
                if abs(round(art[field], 4) - paper) >= tol:
                    bad += 1
                    logger.info(
                        f"  {RED}MISMATCH{RESET} {label} {display} {field}: "
                        f"paper={paper} artifact={round(art[field], 4)}"
                    )
    return checked, bad


def table_block(tex: str, label: str) -> str | None:
    for m in re.finditer(r"\\begin\{table\}.*?\\end\{table\}", tex, re.S):
        if label in m.group(0):
            return m.group(0)
    return None


def scores(results: Path, task: str, metric: str | None) -> dict[str, float]:
    out: dict[str, float] = {}
    for f in glob.glob(str(results / f"*_{task}.json")):
        d = json.loads(Path(f).read_text())
        if "error" in d:
            continue
        v = (d.get("metrics") or {}).get(metric) if metric else d.get("primary_score")
        if v is not None:
            out[d["model_key"]] = float(v)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paper", default="../shelf-paper")
    ap.add_argument("--results", default="results/pooled/baselines")
    ap.add_argument("--tol", type=float, default=5e-5)
    args = ap.parse_args()

    sections = Path(args.paper) / "latex" / "sections"
    results = Path(args.results)
    if not sections.is_dir():
        logger.error(f"{RED}FAIL{RESET} no section directory at {sections}")
        return 2

    name2key = {}
    for f in glob.glob(str(results / "*_lcc_classification.json")):
        d = json.loads(Path(f).read_text())
        name2key[d["model"]] = d["model_key"]
    if not name2key:
        logger.error(
            f"{RED}FAIL{RESET} no results under {results}; nothing was checked"
        )
        return 2

    checked = bad = 0

    for label, (fname, cols, metric) in PER_MODEL.items():
        tex = (sections / fname).read_text()
        blk = table_block(tex, label)
        if blk is None:
            logger.warning(f"  {label}: not found in {fname}")
            continue
        for line in blk.splitlines():
            m = re.match(r"\s*([A-Za-z0-9ént\.\+\- ]+?)\s*&(.+)\\\\\s*$", line)
            if not m:
                continue
            disp = m.group(1).strip()
            key = name2key.get(ALIAS.get(disp, disp))
            vals = [float(x) for x in re.findall(r"-?\d\.\d{3,4}", m.group(2))]
            if key is None or len(vals) != len(cols):
                continue
            for col, paper in zip(cols, vals, strict=True):
                art = scores(results, col, metric).get(key)
                checked += 1
                if art is None or abs(round(art, 4) - paper) >= args.tol:
                    bad += 1
                    logger.info(
                        f"  {RED}MISMATCH{RESET} {label} {disp} {col}: "
                        f"paper={paper} artifact={art if art is None else round(art, 4)}"
                    )

    for label, (fname, tasks, metric, stats) in SUMMARY.items():
        tex = (sections / fname).read_text()
        blk = table_block(tex, label)
        if blk is None:
            logger.warning(f"  {label}: not found in {fname}")
            continue
        rows = {}
        for line in blk.splitlines():
            m = re.match(r"\s*([A-Za-z0-9 ,\.\-]+?)\s*&(.+)\\\\\s*$", line)
            if not m:
                continue
            rows[m.group(1).strip().lower()] = [
                float(x.replace("$", ""))
                for x in re.findall(r"-?\$?\d\.\d{3,4}", m.group(2))
            ]
        # Tables are laid out either task-per-row or stat-per-row.
        by_task = {t: scores(results, t, metric) for t in tasks}
        stat_fn = {
            "best": max,
            "median": statistics.median,
            "worst": min,
        }
        for rowname, vals in rows.items():
            if rowname in stat_fn and len(vals) == len(tasks):
                for t, paper in zip(tasks, vals, strict=True):
                    v = by_task[t]
                    if not v:
                        continue
                    art = stat_fn[rowname](v.values())
                    checked += 1
                    if abs(round(art, 4) - paper) >= args.tol:
                        bad += 1
                        logger.info(
                            f"  {RED}MISMATCH{RESET} {label} {rowname} {t}: "
                            f"paper={paper} artifact={round(art, 4)}"
                        )
            elif rowname == "bm25" and len(vals) == len(tasks):
                for t, paper in zip(tasks, vals, strict=True):
                    art = by_task[t].get("bm25")
                    checked += 1
                    if art is None or abs(round(art, 4) - paper) >= args.tol:
                        bad += 1
                        logger.info(
                            f"  {RED}MISMATCH{RESET} {label} bm25 {t}: "
                            f"paper={paper} artifact={art if art is None else round(art, 4)}"
                        )

    c2, b2 = check_artifact_tables(sections, args.tol)
    checked += c2
    bad += b2

    if checked == 0:
        logger.error(
            f"{RED}FAIL{RESET} nothing was checked; the parser matched no rows"
        )
        return 2
    if bad:
        logger.info(
            f"\n{RED}FAIL{RESET} {bad} of {checked} numbers do not match their artifacts"
        )
        return 1
    logger.info(
        f"\n{GREEN}PASS{RESET} all {checked} table numbers match the result files"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
