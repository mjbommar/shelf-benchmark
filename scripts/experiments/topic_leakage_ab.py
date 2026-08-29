"""A/B/C test: can prompt changes cut verbatim topic echo without losing the topic?

The v0.4 prompt renders form and subject area as *semantic descriptions* but
passes topics through as exact strings (`generator.py`, "topics: {...}").
That asymmetry predicts what we measure: topics appear verbatim in 44.5% of
v0.4 documents against 1.5% for form.

Three arms, same specifications, so the comparison is paired:

  A  control     topics passed verbatim, as v0.4 does today
  B  described   topic name replaced by its sanitized description
  C  guided      topic name kept, plus an instruction not to use it as a label

**The trap this design exists to avoid.** The cheapest way to cut verbatim
rate is to write a document that is not about the topic. So every arm is
scored on two axes: how often the topic appears verbatim, and how close the
document still sits to its topic in embedding space. An arm that wins on
leakage while losing fidelity has not won.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shelf.llm.backends import GenerationParams, GenerationRequest  # noqa: E402
from shelf.sampler.dimensions import LCCClass, LCGFTTerm  # noqa: E402
from shelf.sampler.document import Document  # noqa: E402
from shelf.sampler.enriched import EnrichedDescriptions  # noqa: E402
from shelf.sampler.generator import (  # noqa: E402
    DocumentLength,
    PromptVariant,
    Register,
    build_generation_prompt,
    build_system_prompt,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ARMS = ("control", "guided", "guided_gloss")


#: Instruction shared by both treatment arms. It keeps the topic name in the
#: prompt -- removing it is not an option, see below -- and asks the model not
#: to echo it as a label.
_GUIDANCE = (
    "topic handling: develop these subjects through specific detail, examples, "
    "and argument. Do not use the topic words themselves as labels, headings, "
    "or announcements; a reader should infer the subject from the substance."
)


def _gloss(topic: str, enriched: EnrichedDescriptions) -> str | None:
    """A definitional gloss, but only where it can be trusted.

    Topic descriptions come from three sources. Only ``scope_note`` is quoted
    from LC and reliably about the sense we mean. The ``hierarchy`` source
    (1,546 of 1,983 topics) is templated from wherever the label sits in the
    LC tree and silently picks the wrong sense: "Information" resolves to
    "a topic within Criminal procedure", "Cloud computing" to "Electronic data
    processing--Distributed processing", "Security" to "Investments".

    Substituting those for the topic name would cut verbatim echo by writing
    documents about the wrong subject. So a gloss is only ever *added* beside
    the name, never swapped for it, and only from a scope note.
    """
    entry = enriched.topics.get(topic)
    if entry is None or entry.source != "scope_note":
        return None
    text = (entry.description or "").strip()
    return text if 20 <= len(text) <= 180 else None


def render_topics(arm: str, topics: list[str], enriched: EnrichedDescriptions) -> str:
    if arm == "control":
        return f"topics: {', '.join(topics)}"
    if arm == "guided":
        return f"topics: {', '.join(topics)}\n{_GUIDANCE}"
    if arm == "guided_gloss":
        parts = []
        for t in topics:
            g = _gloss(t, enriched)
            parts.append(f"{t} ({g})" if g else t)
        return f"topics: {'; '.join(parts)}\n{_GUIDANCE}"
    raise ValueError(arm)


def make_doc(spec: dict[str, Any]) -> Document:
    return Document(
        lcc=LCCClass(code=spec["lcc_code"], name=spec["lcc_name"]),
        lcgft=LCGFTTerm(category=spec["lcgft_category"], form=spec["lcgft_form"]),
        topics=spec["topics"],
        audience=spec.get("audience"),
        geographic=spec.get("geographic") or [],
    )


def build_prompt(arm: str, spec: dict[str, Any], enriched: EnrichedDescriptions) -> str:
    """Rebuild the v0.4 prompt with only the topics line swapped."""
    doc = make_doc(spec)
    base = build_generation_prompt(
        doc,
        length=DocumentLength.MEDIUM,
        register=Register.PROFESSIONAL,
        enriched=enriched,
    )
    lines = [ln for ln in base.split("\n") if not ln.startswith("topics:")]
    out: list[str] = []
    inserted = False
    for ln in lines:
        out.append(ln)
        if ln.startswith("subject area:") and not inserted:
            out.append(render_topics(arm, spec["topics"], enriched))
            inserted = True
    if not inserted:
        out.append(render_topics(arm, spec["topics"], enriched))
    return "\n".join(out)


def load_specs(n: int, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for f in sorted(glob.glob("data/artifacts/spec_blocks/*.jsonl")):
        with open(f) as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("topics") and r.get("lcc_code") and r.get("lcgft_form"):
                    rows.append(r)
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:n]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-specs", type=int, default=60)
    ap.add_argument(
        "--models",
        nargs="+",
        default=["anthropic:claude-haiku-4.5", "gemini:gemini-3.7-flash"],
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/experiments/topic_leakage_ab.jsonl")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    enriched = EnrichedDescriptions.load()
    specs = load_specs(args.n_specs, args.seed)
    total = len(specs) * len(ARMS) * len(args.models)
    logger.info(
        f"{len(specs)} specs x {len(ARMS)} arms x {len(args.models)} models = {total} docs"
    )

    if args.dry_run:
        s = specs[0]
        logger.info(f"\nspec topics: {s['topics']}")
        for arm in ARMS:
            logger.info(f"\n----- {arm} -----\n{build_prompt(arm, s, enriched)}")
        return 0

    from generate_documents import _make_backend as create_backend

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done: set[tuple[int, str, str]] = set()
    if out.exists():
        for line in out.open():
            r = json.loads(line)
            done.add((r["spec_idx"], r["arm"], r["model"]))
    if done:
        logger.info(f"resuming: {len(done)} already generated")

    params = GenerationParams(temperature=1.0, top_p=0.95, max_output_tokens=3072)

    backends = {}
    for model_spec in args.models:
        provider, _, model = model_spec.partition(":")
        backends[model] = create_backend(
            provider=provider, model=model, service_tier=None
        )

    with out.open("a") as fh:
        for spec_idx, spec in enumerate(specs):
            for model, backend in backends.items():
                for arm in ARMS:
                    if (spec_idx, arm, model) in done:
                        continue
                    try:
                        res = backend.generate(
                            GenerationRequest(
                                prompt=build_prompt(arm, spec, enriched),
                                system_prompt=build_system_prompt(
                                    make_doc(spec), variant=PromptVariant.DIRECT
                                ),
                            ),
                            params,
                        )
                        text = res.text or ""
                    except Exception as exc:
                        logger.warning(f"  {model} {arm} spec{spec_idx}: {exc}")
                        text = ""
                    fh.write(
                        json.dumps(
                            {
                                "spec_idx": spec_idx,
                                "arm": arm,
                                "model": model,
                                "topics": spec["topics"],
                                "lcc_code": spec["lcc_code"],
                                "lcc_name": spec["lcc_name"],
                                "lcgft_form": spec["lcgft_form"],
                                "text": text,
                            }
                        )
                        + "\n"
                    )
                    fh.flush()
            if (spec_idx + 1) % 10 == 0:
                logger.info(f"  {spec_idx + 1}/{len(specs)} specs")
    logger.info(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
