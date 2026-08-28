#!/usr/bin/env python
"""Rewrite LCC subclass captions into prose that does not hand over its own keywords.

The enriched LCC subclass descriptions are caption hierarchies -- comma-separated
term lists like *"geology, geographical divisions, dynamic and structural
geology, paleozoology, ..."*. Generators copy them: measured over a 240-document
pilot, **36.1% of each subclass's description vocabulary appeared verbatim in its
own documents**, and TF-IDF then recovered the subclass at 0.7585 macro-F1 (0.9286
within parent Q) on ~8 training examples per class. The tier exists to defeat
lexical saturation and was reproducing it at finer grain.

This rewrites each caption into one or two sentences describing what the field
studies -- its questions, materials and methods -- without enumerating the
taxonomy's own terms. The rewrite is then checked against the source caption and
retried once if it simply echoes it.

Nothing is overwritten: output goes to a separate file, and the original
enriched export stays the input.

Usage:
    uv run python scripts/rewrite_subclass_descriptions.py --dry-run
    uv run python scripts/rewrite_subclass_descriptions.py --limit 5
"""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rich.console import Console
from shelf.llm import (
    GenerationParams,
    GenerationRequest,
    OpenRouterBackend,
    ReasoningConfig,
)
from shelf.sampler.enriched import EnrichedDescriptions

console = Console()

SYSTEM_PROMPT = """You write one-sentence descriptions of academic fields for a
document-generation system.

You are given a field's name and a raw list of its subdivision headings. Write a
single sentence, 20-35 words, describing what someone working in this field
actually studies: the questions they ask, the materials they work with, the
methods they use.

Hard constraints:
- Do NOT reuse the nouns from the heading list. Describe the work, don't relabel it.
- Do NOT name the classification, the field's formal title, or any code.
- Write plain declarative prose. No lists, no semicolons, no "including".
- It must be specific enough that a reader could tell this field from a sibling
  field in the same discipline.

Respond with the sentence only."""


def significant_terms(text: str) -> set[str]:
    return set(re.findall(r"[a-z]{5,}", text.lower()))


def overlap(source: str, rewrite: str) -> float:
    src = significant_terms(source)
    if not src:
        return 0.0
    return len(src & significant_terms(rewrite)) / len(src)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="openai/gpt-5.6-luna")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/taxonomies/enriched/lcc_subclass_prose.json"),
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-overlap", type=float, default=0.20)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    enriched = EnrichedDescriptions.load()
    entries = sorted(enriched.lcc_subclasses.items())
    if args.limit:
        entries = entries[: args.limit]
    console.print(f"[bold]{len(entries)} subclass descriptions to rewrite[/bold]")

    if args.dry_run:
        for code, entry in entries[:5]:
            console.print(f"  {code}: {entry.description[:110]}")
        console.print("[yellow]DRY RUN - no API calls.[/yellow]")
        return 0

    backend = OpenRouterBackend(args.model, reasoning=ReasoningConfig.off())

    def rewrite(item):
        code, entry = item
        prompt = f"Field: {entry.label}\nSubdivision headings: {entry.description}"
        for attempt in range(2):
            try:
                result = backend.generate(
                    GenerationRequest(prompt=prompt, system_prompt=SYSTEM_PROMPT),
                    GenerationParams(
                        temperature=0.4 + 0.3 * attempt,
                        top_p=0.95,
                        max_output_tokens=200,
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                return code, None, f"{type(exc).__name__}: {exc}"[:120], 1.0
            text = " ".join(result.text.split()).strip().strip('"')
            ratio = overlap(entry.description, text)
            if ratio <= args.max_overlap:
                return code, text, None, ratio
        return code, text, "high_overlap", ratio

    with ThreadPoolExecutor(max_workers=10) as pool:
        results = list(pool.map(rewrite, entries))

    payload, failures, ratios = {}, [], []
    for code, text, err, ratio in results:
        if text:
            payload[code] = {
                "description": text,
                "source_overlap": round(ratio, 4),
                "flag": err,
            }
            ratios.append(ratio)
        if err:
            failures.append((code, err))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    console.print(f"  wrote {len(payload)} rewrites to {args.output}")
    if ratios:
        console.print(
            f"  mean source-vocabulary overlap: {sum(ratios) / len(ratios) * 100:.1f}%"
        )
    if failures:
        console.print(f"  [yellow]{len(failures)} flagged: {failures[:4]}[/yellow]")
    for code in list(payload)[:4]:
        console.print(f"    {code}: {payload[code]['description']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
