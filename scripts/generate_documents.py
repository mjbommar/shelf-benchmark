#!/usr/bin/env python
"""Generate benchmark documents with LC taxonomy labels.

Samples document specifications from LC taxonomies and generates text
using OpenAI gpt-5.1. Outputs artifacts to JSONL with full metadata.

Usage:
    python scripts/generate_documents.py --output data/benchmark.jsonl --limit 100
    python scripts/generate_documents.py --concurrency 20 --model gpt-5.1
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import openai
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from shelf.config import load_env
from shelf.sampler import (
    DocumentSampler,
    LengthSampler,
    RegisterSampler,
    SamplingParamsSampler,
    LENGTH_WORD_RANGES,
    REGISTER_DESCRIPTIONS,
    GENERATION_INSTRUCTIONS,
    build_generation_prompt,
)
from shelf.sampler.generator import _parse_generated_text
from shelf.sampler.artifacts import generate_artifact_id, get_git_version

console = Console()


async def generate_one(
    doc_id: str,
    sampler: DocumentSampler,
    length_sampler: LengthSampler,
    register_sampler: RegisterSampler,
    sampling_sampler: SamplingParamsSampler,
    client: openai.AsyncOpenAI,
    model: str,
    service_tier: str | None,
    semaphore: asyncio.Semaphore,
    git_info: dict[str, str | bool | None] | None = None,
) -> dict | None:
    """Generate a single document."""
    async with semaphore:
        try:
            # Sample document specification
            doc = sampler.sample(doc_id=doc_id)
            length = length_sampler.sample()
            register = register_sampler.sample()
            sampling = sampling_sampler.sample()

            # Build prompt
            input_text = build_generation_prompt(doc, length, register)

            # Calculate max_output_tokens based on target word range
            # Words are ~1.3 tokens on average, use 2x upper bound for safety
            word_range = LENGTH_WORD_RANGES.get(length)
            max_words = word_range[1] if word_range else 500
            max_output_tokens = max(256, min(16384, max_words * 2))

            # Generate
            request_kwargs = {
                "model": model,
                "instructions": GENERATION_INSTRUCTIONS,
                "input": input_text,
                "max_output_tokens": max_output_tokens,
                "temperature": sampling.temperature,
                "top_p": sampling.top_p,
            }
            if service_tier:
                request_kwargs["service_tier"] = service_tier

            response = await client.responses.create(**request_kwargs)
            raw_text = response.output_text or ""
            title, body = _parse_generated_text(raw_text)

            word_count = len(body.split())
            word_range = LENGTH_WORD_RANGES.get(length)

            result = {
                "id": doc_id,
                "title": title,
                "body": body,
                "word_count": word_count,
                # LC labels
                "lcc_code": doc.lcc.code,
                "lcc_name": doc.lcc.name,
                "lcc_uri": doc.lcc.uri,
                "lcgft_category": doc.lcgft.category,
                "lcgft_form": doc.lcgft.form,
                "topics": doc.topics,
                "audience": doc.audience,
                "geographic": doc.geographic,
                # Generation params
                "target_length": length.value,
                "target_word_range": list(word_range) if word_range else None,
                "register": register.value,
                "register_description": REGISTER_DESCRIPTIONS.get(register),
                "temperature": sampling.temperature,
                "top_p": sampling.top_p,
                # Metadata
                "model": model,
                "prompt": input_text,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }

            # Add git version info for reproducibility
            if git_info:
                result["git_commit"] = git_info.get("commit")
                result["git_dirty"] = git_info.get("dirty")
                result["git_branch"] = git_info.get("branch")
                result["code_version"] = git_info.get("version_string")

            return result
        except Exception as e:
            console.print(f"[red]✗[/red] {doc_id}: {e}")
            return None


async def generate_documents(
    count: int,
    output_path: Path,
    artifacts_dir: Path,
    model: str,
    service_tier: str | None,
    concurrency: int,
    seed: int | None,
    log_usage: bool,
) -> None:
    """Generate documents and write to JSONL + individual artifacts."""
    load_env()
    client = openai.AsyncOpenAI()
    semaphore = asyncio.Semaphore(concurrency)

    # Capture git info once for all artifacts in this run
    git_info = get_git_version()
    console.print(f"  Code version: {git_info.get('version_string', 'unknown')}")

    # Create artifacts directory
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Initialize samplers
    sampler = DocumentSampler(seed=seed)
    length_sampler = LengthSampler(seed=seed)
    register_sampler = RegisterSampler(seed=seed)
    sampling_sampler = SamplingParamsSampler(seed=seed)

    # Generate unique IDs for documents
    to_generate = [generate_artifact_id() for _ in range(count)]

    console.print(f"Generating {count:,} documents")

    total_in = 0
    total_out = 0
    processed = 0
    failed = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_file = open(output_path, "a")

    # Create tasks
    tasks = [
        asyncio.create_task(
            generate_one(
                doc_id=doc_id,
                sampler=sampler,
                length_sampler=length_sampler,
                register_sampler=register_sampler,
                sampling_sampler=sampling_sampler,
                client=client,
                model=model,
                service_tier=service_tier,
                semaphore=semaphore,
                git_info=git_info,
            )
        )
        for doc_id in to_generate
    ]

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )

    try:
        with progress:
            task_id = progress.add_task("Generating documents", total=len(tasks))

            for coro in asyncio.as_completed(tasks):
                result = await coro
                progress.advance(task_id)

                if result is None:
                    failed += 1
                    continue

                processed += 1
                total_in += result.get("input_tokens", 0)
                total_out += result.get("output_tokens", 0)

                if log_usage:
                    console.print(
                        f"[green]✓[/green] {result['id']} | {result['lcgft_form'][:20]:20} | "
                        f"{result['word_count']:4} words | "
                        f"t={result['temperature']:.2f} | "
                        f"in:{result['input_tokens']}, out:{result['output_tokens']}"
                    )

                # Remove token counts from output (internal tracking only)
                out_record = {
                    k: v
                    for k, v in result.items()
                    if k not in ("input_tokens", "output_tokens")
                }

                # Save individual artifact JSON
                artifact_path = artifacts_dir / f"{result['id']}.json"
                with open(artifact_path, "w") as af:
                    json.dump(out_record, af, indent=2)

                # Append to JSONL index
                output_file.write(json.dumps(out_record) + "\n")
                output_file.flush()
    finally:
        output_file.close()

    total_tokens = total_in + total_out
    console.print("\n[bold]Summary:[/bold]")
    console.print(f"  Processed: {processed:,} documents")
    console.print(f"  Failed: {failed:,}")
    console.print(
        f"  Tokens: {total_in:,} in + {total_out:,} out = {total_tokens:,} total"
    )
    # gpt-5.1 pricing: $0.625/1M input, $5.00/1M output
    est_cost = (total_in * 0.625 + total_out * 5.00) / 1_000_000
    console.print(f"  Est. cost: ${est_cost:.4f}")
    console.print(f"  JSONL index: {output_path}")
    console.print(f"  Artifacts: {artifacts_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Generate benchmark documents with LC taxonomy labels"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/benchmark.jsonl"),
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("data/artifacts"),
        help="Directory for individual artifact JSON files",
    )
    parser.add_argument(
        "--count", type=int, default=100, help="Number of documents to generate"
    )
    parser.add_argument(
        "--model", default="gpt-5.1", help="Model to use (default: gpt-5.1)"
    )
    parser.add_argument(
        "--service-tier", default="flex", help="Service tier (default: flex)"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=20,
        help="Max concurrent requests (default: 20)",
    )
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument(
        "--log-usage", action="store_true", help="Print per-document token usage"
    )

    args = parser.parse_args()

    console.print("[bold]SHELF Document Generator[/bold]")
    console.print(f"  Model: {args.model}")
    console.print(f"  Service tier: {args.service_tier}")
    console.print(f"  Concurrency: {args.concurrency}")
    console.print(f"  Target count: {args.count}")
    console.print(f"  Output: {args.output}")
    console.print(f"  Artifacts: {args.artifacts_dir}")
    if args.seed is not None:
        console.print(f"  Seed: {args.seed}")
    console.print()

    asyncio.run(
        generate_documents(
            count=args.count,
            output_path=args.output,
            artifacts_dir=args.artifacts_dir,
            model=args.model,
            service_tier=args.service_tier,
            concurrency=args.concurrency,
            seed=args.seed,
            log_usage=args.log_usage,
        )
    )


if __name__ == "__main__":
    main()
