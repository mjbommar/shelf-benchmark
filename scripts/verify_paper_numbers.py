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
# Display name in the paper -> name recorded in the result files, and the
# reverse, because the completeness check needs to go both ways.
ALIAS = {
    "DistilBERT": "distilbert-base-uncased",
    "distilbert-base-uncased": "DistilBERT",
    "OGBert 110M": "OGBert-110M Base",
    "OGBert-110M Base": "OGBert 110M",
    "OGBert 36M": "ogbert-v1-mlm",
    "ogbert-v1-mlm": "OGBert 36M",
}

# Historical result files remain on disk for provenance, but these
# configurations are no longer part of the reported panel.
RETIRED_MODELS = {"ogbert_2m_sentence", "ogbert_110m_sentence"}

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
        ["best", "median"],
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

TASK_ROW_LABELS = {
    "tab:baselines-clu": {
        "subject": "lcc_clustering",
        "genre form": "lcgft_clustering",
        "register": "register_clustering",
        "geography": "geographic_clustering",
    },
    "tab:retrieval": {
        "subject": "lcc_retrieval",
        "genre form": "form_retrieval",
        "category": "category_retrieval",
    },
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


def check_decoder_table(sections: Path, tol: float) -> tuple[int, int]:
    """Check the zero-shot decoder table against its result files.

    These numbers come from a separate arm with its own runner, so the
    per-model machinery above cannot see them. Without this they would be the
    only table in the paper not tied to an artifact.
    """
    import glob as _glob

    rows: dict[str, dict] = {}
    for f in _glob.glob("results/generative/test_scores*.jsonl"):
        for line in Path(f).read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                rows[d["model"]] = d
    if not rows:
        logger.warning("  tab:decoders: no decoder results on disk")
        return 0, 0

    display = {
        "Qwen3.5-0.8B": "Qwen/Qwen3.5-0.8B",
        "Gemma-4-E2B": "google/gemma-4-E2B-it",
        "Qwen3.5-2B": "Qwen/Qwen3.5-2B",
        "GPT-5.6-luna": "gpt-5.6-luna",
    }
    tex = (sections / "06c_decoders.tex").read_text()
    blk = table_block(tex, "tab:decoders")
    if blk is None:
        logger.warning("  tab:decoders: not found")
        return 0, 0

    checked = bad = 0
    for disp, key in display.items():
        line = next((x for x in blk.splitlines() if x.strip().startswith(disp)), None)
        art = rows.get(key)
        if line is None or art is None:
            logger.info(f"  {RED}MISMATCH{RESET} tab:decoders {disp}: missing")
            bad += 1
            continue
        vals = [float(x) for x in re.findall(r"0\.\d{4}", line)]
        for field, paper in zip(("macro_f1", "accuracy"), vals, strict=False):
            checked += 1
            if abs(round(art[field], 4) - paper) >= tol:
                bad += 1
                logger.info(
                    f"  {RED}MISMATCH{RESET} tab:decoders {disp} {field}: "
                    f"paper={paper} artifact={round(art[field], 4)}"
                )
        # every decoder must have been scored on the whole test split
        n = art.get("num_samples", art.get("n_scored"))
        checked += 1
        if n != 12504:
            bad += 1
            logger.info(f"  {RED}MISMATCH{RESET} tab:decoders {disp}: n={n} not 12504")
    return checked, bad


def check_timing_table(sections: Path) -> tuple[int, int]:
    """Check the inference-rate table against the timing artifact."""
    art = Path("results/generative/timing.json")
    if not art.exists():
        logger.warning("  tab:timing: results/generative/timing.json missing")
        return 0, 0
    rows = {r["model"]: r for r in json.loads(art.read_text())["rows"]}
    api_path = Path("results/generative/test_scores_api.jsonl")
    if api_path.exists():
        for line in api_path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("model") == "gpt-5.6-luna" and rec.get("seconds"):
                rows["GPT-5.6-luna (API)"] = {
                    "docs_per_s": rec["n_scored"] / rec["seconds"]
                }
    tex = (sections / "06c_decoders.tex").read_text()
    blk = table_block(tex, "tab:timing")
    if blk is None:
        logger.warning("  tab:timing: not found")
        return 0, 0

    display = {
        "TF-IDF + logistic": "TF-IDF + logistic",
        "MiniLM-L6": "MiniLM-L6",
        "BGE-small": "BGE-small",
        "EmbeddingGemma-300M": "EmbeddingGemma-300M",
        "Qwen3.5-0.8B": "Qwen3.5-0.8B",
        "Gemma-4-E2B": "Gemma-4-E2B",
        "Qwen3.5-2B": "Qwen3.5-2B",
        "GPT-5.6-luna (API)": "GPT-5.6-luna (API)",
    }
    checked = bad = 0
    for disp, key in display.items():
        line = next((x for x in blk.splitlines() if f"& {disp} " in x), None)
        a = rows.get(key)
        if line is None or a is None:
            logger.info(f"  {RED}MISMATCH{RESET} tab:timing {disp}: missing")
            bad += 1
            continue
        checked += 1
        # the docs/s column, allowing the thousands separator the table uses
        paper = float(
            line.split("&")[3]
            .replace("\\textbf{", "")
            .replace("}", "")
            .replace("{,}", "")
            .strip()
        )
        if abs(paper - a["docs_per_s"]) > 0.6:
            bad += 1
            logger.info(
                f"  {RED}MISMATCH{RESET} tab:timing {disp} docs/s: "
                f"paper={paper} artifact={a['docs_per_s']}"
            )
    return checked, bad


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
    for m in re.finditer(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", tex, re.S):
        if label in m.group(0):
            return m.group(0)
    return None


def scores(results: Path, task: str, metric: str | None) -> dict[str, float]:
    out: dict[str, float] = {}
    for f in glob.glob(str(results / f"*_{task}.json")):
        d = json.loads(Path(f).read_text())
        if "error" in d:
            continue
        if d.get("model_key") in RETIRED_MODELS:
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

    for label, (fname, tasks, metric, _stats) in SUMMARY.items():
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
            else:
                task = TASK_ROW_LABELS.get(label, {}).get(rowname)
                if task not in tasks:
                    continue
                task_scores = by_task[task]
                expected = [max(task_scores.values()), statistics.median(task_scores.values())]
                if len(vals) == 3 and "bm25" in task_scores:
                    expected.append(task_scores["bm25"])
                for stat, paper, art in zip(
                    _stats[: len(vals)], vals, expected, strict=True
                ):
                    checked += 1
                    if abs(round(art, 4) - paper) >= args.tol:
                        bad += 1
                        logger.info(
                            f"  {RED}MISMATCH{RESET} {label} {rowname} {stat}: "
                            f"paper={paper} artifact={round(art, 4)}"
                        )

    c2, b2 = check_artifact_tables(sections, args.tol)
    checked += c2
    bad += b2

    c3, b3 = check_decoder_table(sections, args.tol)
    checked += c3
    bad += b3

    c4, b4 = check_timing_table(sections)
    checked += c4
    bad += b4

    # Completeness. Everything above verifies that numbers PRESENT in a table
    # match their artifacts. It cannot see a model that was left out, and that
    # is exactly what happened: five models were added to the sweep and the
    # classification table kept showing the older 24, including after one of
    # the new models took first place.
    for label, (fname, cols, metric) in PER_MODEL.items():
        tex = (sections / fname).read_text()
        blk = table_block(tex, label)
        if blk is None:
            continue
        scored = set(scores(results, cols[0], metric))
        key2name = {k: n for n, k in name2key.items()}
        missing = []
        for key in sorted(scored):
            disp = key2name.get(key, key)
            alias = ALIAS.get(disp, disp)
            # a row is present if its display name, its alias, or a
            # distinctive prefix of either appears in the table body
            found = any(
                cand and cand in blk
                for cand in (disp, alias, disp.split()[0][:10], alias.split()[0][:10])
            )
            if not found:
                missing.append(key)
        checked += 1
        if missing:
            bad += 1
            logger.info(
                f"  {RED}INCOMPLETE{RESET} {label}: {len(missing)} model(s) have "
                f"results but no row: {', '.join(missing)}"
            )

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
