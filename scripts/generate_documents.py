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
from shelf.llm import (
    AnthropicMessagesBackend,
    GenerationParams,
    GenerationRequest,
    GeminiBackend,
    LLMBackend,
    OpenAIResponsesBackend,
)
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

# =============================================================================
# Pricing table (USD per 1M tokens) - Updated December 2025
# Sources:
#   - OpenAI: https://openai.com/api/pricing/
#   - Anthropic: https://docs.anthropic.com/en/docs/about-claude/models
#   - Google: https://ai.google.dev/gemini-api/docs/pricing
# =============================================================================

MODEL_PRICING: dict[str, dict[str, tuple[float, float]]] = {
    # OpenAI models - flex tier pricing (input_per_1M, output_per_1M)
    "gpt-5.2": {"standard": (0.875, 7.00), "batch": (0.875, 7.00)},
    "gpt-5.1": {"standard": (0.625, 5.00), "batch": (0.625, 5.00)},
    "gpt-5": {"standard": (0.625, 5.00), "batch": (0.625, 5.00)},
    "gpt-5-mini": {"standard": (0.125, 1.00), "batch": (0.125, 1.00)},
    "gpt-5-nano": {"standard": (0.025, 0.20), "batch": (0.025, 0.20)},
    "o3": {"standard": (1.00, 4.00), "batch": (1.00, 4.00)},
    "o4-mini": {"standard": (0.55, 2.20), "batch": (0.55, 2.20)},
    # Anthropic models
    "claude-sonnet-4-5-20250929": {"standard": (3.00, 15.00), "batch": (1.50, 7.50)},
    "claude-opus-4-5-20250929": {"standard": (15.00, 75.00), "batch": (7.50, 37.50)},
    "claude-3-7-sonnet-20250219": {"standard": (3.00, 15.00), "batch": (1.50, 7.50)},
    "claude-3-5-sonnet-20241022": {"standard": (3.00, 15.00), "batch": (1.50, 7.50)},
    "claude-3-5-haiku-20241022": {"standard": (0.80, 4.00), "batch": (0.40, 2.00)},
    # Google Gemini models
    "gemini-3-pro": {"standard": (2.00, 12.00), "batch": (1.00, 6.00)},  # Nov 2025
    "gemini-2.5-pro": {"standard": (1.25, 10.00), "batch": (0.625, 5.00)},
    "gemini-2.5-flash": {"standard": (0.30, 2.50), "batch": (0.15, 1.25)},
    "gemini-2.0-flash": {"standard": (0.10, 0.40), "batch": (0.05, 0.20)},
    "gemini-2.5-flash-lite": {"standard": (0.10, 0.40), "batch": (0.05, 0.20)},
    "gemini-1.5-pro": {"standard": (1.25, 5.00), "batch": (0.625, 2.50)},
    "gemini-1.5-flash": {"standard": (0.075, 0.30), "batch": (0.0375, 0.15)},
}


def _estimate_cost(
    model: str, input_tokens: int, output_tokens: int, use_batch: bool = False
) -> tuple[float, str] | None:
    """Estimate cost for a model. Returns (cost, pricing_note) or None if unknown."""
    # Try exact match first
    pricing = MODEL_PRICING.get(model)

    # Try prefix match for versioned models
    if pricing is None:
        for model_prefix in MODEL_PRICING:
            if model.startswith(model_prefix.rsplit("-", 1)[0]):
                pricing = MODEL_PRICING[model_prefix]
                break

    if pricing is None:
        return None

    tier = "batch" if use_batch else "standard"
    input_rate, output_rate = pricing.get(tier, pricing["standard"])
    cost = (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000
    return cost, f"{tier} pricing"


def _make_backend(
    provider: str,
    model: str,
    service_tier: str | None,
    use_batch_api: bool = False,
    thinking_budget: int | None = None,
    token_multiplier: float = 1.0,
) -> LLMBackend:
    """Instantiate the requested backend."""
    provider = provider.lower()
    if provider == "openai":
        return OpenAIResponsesBackend(
            model=model,
            service_tier=service_tier,
        )
    if provider == "anthropic":
        return AnthropicMessagesBackend(
            model=model,
            use_batch_api=use_batch_api,
        )
    if provider == "gemini":
        return GeminiBackend(
            model=model,
            use_batch_api=use_batch_api,
            thinking_budget=thinking_budget,
            token_multiplier=token_multiplier,
        )
    raise ValueError(f"Unsupported provider: {provider}")


def _build_result_dict(
    doc_id: str,
    doc,
    title: str,
    body: str,
    length,
    register,
    sampling,
    backend: LLMBackend,
    gen_result,
    git_info: dict[str, str | bool | None] | None = None,
) -> dict:
    """Build result dictionary from generation outputs."""
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
        "model": backend.model,
        "input_tokens": gen_result.input_tokens if gen_result else None,
        "output_tokens": gen_result.output_tokens if gen_result else None,
    }

    # Add Gemini-specific settings if applicable
    if hasattr(backend, "_thinking_budget"):
        result["thinking_budget"] = backend._thinking_budget
    if hasattr(backend, "_token_multiplier"):
        result["token_multiplier"] = backend._token_multiplier

    if git_info:
        result["git_commit"] = git_info.get("commit")
        result["git_dirty"] = git_info.get("dirty")
        result["git_branch"] = git_info.get("branch")
        result["code_version"] = git_info.get("version_string")

    return result


def generate_batch_sync(
    doc_ids: list[str],
    sampler: DocumentSampler,
    length_sampler: LengthSampler,
    register_sampler: RegisterSampler,
    sampling_sampler: SamplingParamsSampler,
    backend: LLMBackend,
    git_info: dict[str, str | bool | None] | None = None,
) -> list[dict | None]:
    """Generate a batch of documents using backend's batch API."""
    # Sample all document specs
    docs = []
    lengths = []
    registers = []
    requests = []

    # Use single sampling params for the batch (required by batch API)
    sampling = sampling_sampler.sample()

    for doc_id in doc_ids:
        doc = sampler.sample(doc_id=doc_id)
        length = length_sampler.sample()
        register = register_sampler.sample()

        docs.append(doc)
        lengths.append(length)
        registers.append(register)

        input_text = build_generation_prompt(doc, length, register)

        requests.append(
            GenerationRequest(
                prompt=input_text,
                system_prompt=GENERATION_INSTRUCTIONS,
            )
        )

    # Call batch API
    params = GenerationParams(
        temperature=sampling.temperature,
        top_p=sampling.top_p,
        max_output_tokens=4096,  # Use conservative default for batch
    )

    try:
        gen_results = backend.generate_batch(requests, params)
    except Exception as e:
        console.print(f"[red]✗[/red] Batch failed: {e}")
        return [None] * len(doc_ids)

    # Build result dicts
    results = []
    for i, (doc_id, doc, length, register, gen_result) in enumerate(
        zip(doc_ids, docs, lengths, registers, gen_results)
    ):
        try:
            title, body = _parse_generated_text(gen_result.text)
            result = _build_result_dict(
                doc_id,
                doc,
                title,
                body,
                length,
                register,
                sampling,
                backend,
                gen_result,
                git_info,
            )
            results.append(result)
        except Exception as e:
            console.print(f"[red]✗[/red] {doc_id}: {e}")
            results.append(None)

    return results


async def generate_one(
    doc_id: str,
    sampler: DocumentSampler,
    length_sampler: LengthSampler,
    register_sampler: RegisterSampler,
    sampling_sampler: SamplingParamsSampler,
    backend: LLMBackend,
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

            # Generate via backend abstraction
            gen_result = await backend.generate_async(
                GenerationRequest(
                    prompt=input_text,
                    system_prompt=GENERATION_INSTRUCTIONS,
                ),
                GenerationParams(
                    temperature=sampling.temperature,
                    top_p=sampling.top_p,
                    max_output_tokens=max_output_tokens,
                ),
            )
            raw_text = gen_result.text
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
                "model": backend.model,
                "prompt": input_text,
                "input_tokens": gen_result.input_tokens,
                "output_tokens": gen_result.output_tokens,
            }

            # Add Gemini-specific settings if applicable
            if hasattr(backend, "_thinking_budget"):
                result["thinking_budget"] = backend._thinking_budget
            if hasattr(backend, "_token_multiplier"):
                result["token_multiplier"] = backend._token_multiplier

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
    backend: LLMBackend,
    provider: str,
    concurrency: int,
    seed: int | None,
    log_usage: bool,
    batch_size: int = 0,
) -> None:
    """Generate documents and write to JSONL + individual artifacts."""
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

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )

    # Helper to save a result
    def save_result(result: dict) -> None:
        nonlocal total_in, total_out, processed
        input_tokens = result.get("input_tokens") or 0
        output_tokens = result.get("output_tokens") or 0
        total_in += input_tokens
        total_out += output_tokens
        processed += 1

        if log_usage:
            console.print(
                f"[green]✓[/green] {result['id']} | {result['lcgft_form'][:20]:20} | "
                f"{result['word_count']:4} words | "
                f"t={result['temperature']:.2f} | "
                f"in:{result.get('input_tokens', '?')}, out:{result.get('output_tokens', '?')}"
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

    try:
        # Batch mode: use backend's batch API
        if batch_size > 0:
            with progress:
                task_id = progress.add_task("Generating documents", total=count)

                # Process in batches
                for i in range(0, len(to_generate), batch_size):
                    batch_ids = to_generate[i : i + batch_size]
                    results = generate_batch_sync(
                        batch_ids,
                        sampler,
                        length_sampler,
                        register_sampler,
                        sampling_sampler,
                        backend,
                        git_info,
                    )

                    for result in results:
                        if result is None:
                            failed += 1
                        else:
                            save_result(result)
                        progress.advance(task_id)

        # Concurrent async mode (default)
        else:
            semaphore = asyncio.Semaphore(concurrency)
            tasks = [
                asyncio.create_task(
                    generate_one(
                        doc_id=doc_id,
                        sampler=sampler,
                        length_sampler=length_sampler,
                        register_sampler=register_sampler,
                        sampling_sampler=sampling_sampler,
                        backend=backend,
                        semaphore=semaphore,
                        git_info=git_info,
                    )
                )
                for doc_id in to_generate
            ]

            with progress:
                task_id = progress.add_task("Generating documents", total=len(tasks))

                for coro in asyncio.as_completed(tasks):
                    result = await coro
                    progress.advance(task_id)

                    if result is None:
                        failed += 1
                    else:
                        save_result(result)
    finally:
        output_file.close()

    total_tokens = total_in + total_out
    console.print("\n[bold]Summary:[/bold]")
    console.print(f"  Processed: {processed:,} documents")
    console.print(f"  Failed: {failed:,}")
    console.print(
        f"  Tokens: {total_in:,} in + {total_out:,} out = {total_tokens:,} total"
    )

    # Estimate cost using pricing table
    cost_estimate = _estimate_cost(
        backend.model, total_in, total_out, use_batch=batch_size > 0
    )
    if cost_estimate:
        cost, pricing_note = cost_estimate
        console.print(f"  Est. cost ({pricing_note}): ${cost:.4f}")
    else:
        console.print(f"  Est. cost: (unknown model '{backend.model}')")

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
        "--provider",
        default=None,
        choices=["openai", "anthropic", "gemini"],
        help="LLM provider (default: inferred from model name)",
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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Batch size for batch API (0=disabled, use concurrent async). "
        "Anthropic/Gemini only. Note: Gemini batch has 24hr target turnaround.",
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=None,
        help="Gemini thinking budget (0=disable, >0=set budget, None=no config). "
        "Use 0 for flash models to save tokens. Pro models require thinking.",
    )
    parser.add_argument(
        "--token-multiplier",
        type=float,
        default=1.0,
        help="Multiply max_output_tokens (Gemini only). Use >1 to accommodate "
        "thinking overhead for pro models (e.g., 4.0 for 4x buffer).",
    )

    args = parser.parse_args()

    console.print("[bold]SHELF Document Generator[/bold]")
    console.print(f"  Model: {args.model}")
    console.print(f"  Service tier: {args.service_tier}")
    if args.batch_size > 0:
        console.print(f"  Batch size: {args.batch_size}")
    else:
        console.print(f"  Concurrency: {args.concurrency}")
    console.print(f"  Target count: {args.count}")
    console.print(f"  Output: {args.output}")
    console.print(f"  Artifacts: {args.artifacts_dir}")
    if args.seed is not None:
        console.print(f"  Seed: {args.seed}")
    console.print()

    # Load environment/API keys before creating backend
    load_env()

    # Infer provider if not supplied
    if args.provider:
        provider = args.provider
    elif args.model.startswith("claude"):
        provider = "anthropic"
    elif args.model.startswith("gemini"):
        provider = "gemini"
    else:
        provider = "openai"

    use_batch_api = args.batch_size > 0
    if use_batch_api and provider == "openai":
        console.print(
            "[yellow]Warning: OpenAI does not support batch API, using concurrent async[/yellow]"
        )
        use_batch_api = False

    backend = _make_backend(
        provider,
        args.model,
        args.service_tier,
        use_batch_api=use_batch_api,
        thinking_budget=args.thinking_budget,
        token_multiplier=args.token_multiplier,
    )
    console.print(f"  Provider: {provider}")
    if use_batch_api:
        console.print(f"  Mode: batch API (batch_size={args.batch_size})")
    else:
        console.print("  Mode: concurrent async")
    console.print()

    asyncio.run(
        generate_documents(
            count=args.count,
            output_path=args.output,
            artifacts_dir=args.artifacts_dir,
            backend=backend,
            provider=provider,
            concurrency=args.concurrency,
            seed=args.seed,
            log_usage=args.log_usage,
            batch_size=args.batch_size if use_batch_api else 0,
        )
    )


if __name__ == "__main__":
    main()
