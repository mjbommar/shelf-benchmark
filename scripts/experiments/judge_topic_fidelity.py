"""Judge whether a document still covers its topics, without reading the words.

The embedding-cosine fidelity check in ``analyze_topic_leakage.py`` is
confounded by the very thing the experiment manipulates. Cosine between a
document and its topic *string* falls when the document stops containing that
string, whether or not the document is still about the subject. Suppressing
the word therefore looks like losing the topic even when coverage is intact.

This asks a model instead, one topic at a time, with the arm hidden. The
judge never sees which prompt produced the text, and is told explicitly that
the topic words may be absent.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shelf.llm.backends import GenerationParams, GenerationRequest  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

SYSTEM = (
    "You judge whether a document substantively covers a subject. The document "
    "may never use the subject's name -- that is expected and is NOT a reason "
    "to say no. Judge the substance: does the text develop this subject through "
    "its content, examples, or argument?\n\n"
    "Answer with exactly one word: YES or NO."
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="results/experiments/topic_ab_merged.jsonl")
    ap.add_argument("--out", default="results/experiments/topic_fidelity_judged.jsonl")
    ap.add_argument("--judge", default="anthropic:claude-sonnet-5")
    ap.add_argument("--max-per-arm", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from generate_documents import _make_backend

    with Path(args.input).open() as fh:
        rows = [json.loads(line) for line in fh]
    rows = [r for r in rows if (r.get("text") or "").strip()]

    # one judgement per (document, topic)
    items = []
    for i, r in enumerate(rows):
        for t in r["topics"]:
            items.append({"row": i, "arm": r["arm"], "model": r["model"], "topic": t})
    random.Random(args.seed).shuffle(items)

    per_arm: defaultdict[str, int] = defaultdict(int)
    keep = []
    for it in items:
        if per_arm[it["arm"]] < args.max_per_arm:
            per_arm[it["arm"]] += 1
            keep.append(it)
    logger.info(f"{len(keep)} judgements across {len(per_arm)} arms")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        with out.open() as fh:
            for line in fh:
                d = json.loads(line)
                done.add((d["row"], d["topic"]))

    provider, _, model = args.judge.partition(":")
    backend = _make_backend(provider=provider, model=model, service_tier=None)
    params = GenerationParams(temperature=0.0, top_p=1.0, max_output_tokens=2048)

    with out.open("a") as fh:
        for n, it in enumerate(keep):
            if (it["row"], it["topic"]) in done:
                continue
            text = rows[it["row"]]["text"][:6000]
            prompt = f"SUBJECT: {it['topic']}\n\nDOCUMENT:\n{text}"
            try:
                res = backend.generate(
                    GenerationRequest(prompt=prompt, system_prompt=SYSTEM), params
                )
                verdict = (res.text or "").strip().upper()
            except Exception as exc:
                logger.warning(f"  judge failed: {exc}")
                verdict = ""
            covered = verdict.startswith("YES")
            fh.write(
                json.dumps({**it, "verdict": verdict[:20], "covered": covered}) + "\n"
            )
            fh.flush()
            if (n + 1) % 40 == 0:
                logger.info(f"  {n + 1}/{len(keep)}")

    # ---- report -------------------------------------------------------
    with out.open() as fh:
        judged = [json.loads(line) for line in fh]
    judged = [j for j in judged if j["verdict"]]
    by_arm: defaultdict[str, list[bool]] = defaultdict(list)
    for j in judged:
        by_arm[j["arm"]].append(j["covered"])

    logger.info(f"\n{'arm':<15}{'judged':>8}{'topic covered':>16}")
    logger.info("-" * 39)
    for arm in sorted(by_arm):
        v = by_arm[arm]
        logger.info(f"{arm:<15}{len(v):>8}{statistics.mean(v) * 100:>15.1f}%")
    logger.info(
        "\nJudge was blind to the arm and told the topic words may be absent.\n"
        "This is the fidelity number to trust; embedding cosine is confounded."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
