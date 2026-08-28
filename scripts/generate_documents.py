#!/usr/bin/env python
"""Generate benchmark documents with LC taxonomy labels.

Phase 1 of the v0.4 data plan hands **the same spec block to every generator**,
so generator identity becomes the only varying factor. This script is the
money-spending path for that, and every design choice here is subordinate to one
rule: it must not be possible to overrun a budget.

Three mechanisms enforce that.

**Live pricing.** There is no hardcoded price table. Prices come from
``shelf.llm.pricing.PricingTable`` (OpenRouter catalogue, 24h disk cache), and
the snapshot's ``fetched_at`` is recorded in the run manifest. A model that
cannot be priced is refused rather than run blind -- pass
``--price-input-per-1m`` / ``--price-output-per-1m`` to state a price yourself.

**A hard cap held against reservations, not actuals.** Every request reserves
its *worst case* cost through ``BudgetGuard.reserve`` before the call goes out
and commits its *actual* cost afterwards. Twenty concurrent requests therefore
cannot each individually pass a check and then collectively blow the cap. When a
model hits its cap the run for that model stops cleanly; other models continue.

**A resumable, ledger-backed queue.** Spend and completed ``spec_id``s are read
back from ``data/artifacts/cost_ledger.jsonl`` on startup, so a crashed run
resumes without re-spending on work that already landed.

Usage:
    # Price a run without making a single API call (the safety gate).
    python scripts/generate_documents.py --dry-run --count 1500 \
        --model anthropic/claude-opus-5 openai/gpt-5.6-sol x-ai/grok-4.6

    # Draw three independent spec blocks and save them.
    python scripts/generate_documents.py --draw-spec-blocks 3 \
        --specs-per-block 500 --spec-block-dir data/spec_blocks --draw-only

    # Generate one block with one model, capped at $25.
    python scripts/generate_documents.py \
        --spec-block data/spec_blocks/block-00-seed42.jsonl \
        --model meta-llama/llama-4-maverick --provider openrouter \
        --budget-per-model 25

    # Legacy invocation still works (now with a budget guard in front of it).
    python scripts/generate_documents.py --count 100 --model gpt-5.1
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from shelf.llm import (
    REASONING_EFFORTS,
    AnthropicMessagesBackend,
    BudgetExceeded,
    BudgetGuard,
    CostLedger,
    GeminiBackend,
    GenerationParams,
    GenerationRequest,
    LLMBackend,
    OpenAIResponsesBackend,
    OpenRouterBackend,
    ProviderRouting,
    ReasoningConfig,
    XAIBackend,
    new_run_id,
)
from shelf.llm.ledger import DEFAULT_LEDGER_PATH, STATUS_OK
from shelf.llm.pricing import (
    DEFAULT_CACHE_PATH,
    VERIFIED_NATIVE_PRICES,
    ModelPrice,
    PriceOverride,
    PricingError,
    PricingTable,
    UnknownModelError,
)
from shelf.sampler import (
    LENGTH_WORD_RANGES,
    REGISTER_DESCRIPTIONS,
    DocumentSampler,
    LengthSampler,
    RegisterSampler,
    SamplingParamsSampler,
)
from shelf.sampler.artifacts import generate_artifact_id, get_git_version
from shelf.sampler.generator import (
    DEFAULT_PROMPT_VARIANT_WEIGHTS,
    PromptVariant,
    PromptVariantSampler,
    _parse_generated_text,
    build_generation_prompt,
    build_system_prompt,
)
from shelf.sampler.specs import (
    DocumentSpec,
    SpecBlock,
    draw_spec_blocks,
    load_spec_block,
    save_spec_block,
)

console = Console()

# =============================================================================
# Safety constants
# =============================================================================

#: Per-model cap applied when the operator does not pass ``--budget-per-model``.
#: The alternative -- refusing to start -- would break the documented legacy
#: invocation, and an unbounded run is not on the table. $5 covers the 100-doc
#: default on any roster model with room to spare while stopping a mistyped
#: ``--count 150000`` long before it costs real money. A Phase 1 run must state
#: its own budget explicitly.
DEFAULT_BUDGET_PER_MODEL_USD = 5.00

#: Conservative chars-per-token divisor for *bounding* prompt size. English
#: tokenizers rarely fall below ~3 chars/token; 2.5 leaves headroom so the
#: reservation is an upper bound rather than a guess.
WORST_CASE_CHARS_PER_TOKEN = 2.5

#: Realistic chars-per-token divisor, used only for the projected (expected)
#: cost shown by ``--dry-run``.
EXPECTED_CHARS_PER_TOKEN = 4.0

#: Fixed slack added to the worst-case input estimate (chat envelope, BOS/EOS,
#: provider-injected preamble).
WORST_CASE_INPUT_SLACK_TOKENS = 128

#: Tokens per word when projecting expected output length.
TOKENS_PER_WORD = 1.35

#: Tokens reserved for the ``Title: ...`` line.
TITLE_OVERHEAD_TOKENS = 40

#: Reasoning tokens bill as output and are the main overrun risk. Some providers
#: count them inside ``max_output_tokens`` and some do not, so when reasoning is
#: enabled the worst-case output bound is multiplied by this factor. With the
#: default ``--reasoning-effort none`` it never applies.
REASONING_WORST_CASE_MULTIPLIER = 2.0

#: Plan §5 assumption for a thinking model's output, used for the dry-run
#: projection only (never for the reservation).
EXPECTED_REASONING_TOKENS = 1500

#: ``--reasoning-effort default`` means "send no reasoning parameter at all".
REASONING_EFFORT_PROVIDER_DEFAULT = "default"

STATUS_GENERATED = "generated"
STATUS_FAILED = "failed"
STATUS_SKIPPED_BUDGET = "skipped_budget"


# =============================================================================
# Pricing
# =============================================================================


def load_pricing_table(
    *,
    cache_path: Path | None = DEFAULT_CACHE_PATH,
    refresh: bool = False,
    allow_network: bool = True,
    native_prices: bool = True,
    extra_overrides: dict[str, PriceOverride] | None = None,
) -> PricingTable:
    """Load the live OpenRouter catalogue, optionally overridden by native prices.

    Args:
        cache_path: Disk cache for the catalogue (24h TTL).
        refresh: Re-fetch even when the cache is fresh.
        allow_network: Set False to run strictly off the cache.
        native_prices: Apply :data:`VERIFIED_NATIVE_PRICES`. These are the
            provider's own published numbers, which are the *higher* of the two
            wherever OpenRouter disagrees -- the conservative choice for a cap.
        extra_overrides: Manual per-model prices, applied last.

    Returns:
        A :class:`PricingTable`.
    """
    overrides: dict[str, PriceOverride] = (
        dict(VERIFIED_NATIVE_PRICES) if native_prices else {}
    )
    if extra_overrides:
        overrides.update(extra_overrides)
    return PricingTable.load(
        cache_path=cache_path,
        force_refresh=refresh,
        allow_network=allow_network,
        overrides=overrides,
    )


def resolve_price(table: PricingTable, model: str, provider: str | None) -> ModelPrice:
    """Price a model or refuse to run.

    Raises:
        SystemExit: If the model cannot be priced. A budget cannot be enforced
            against an unknown price, so this is fatal by design.
    """
    try:
        return table.price_of(model, provider)
    except UnknownModelError as exc:
        raise SystemExit(
            f"Cannot price model {model!r} (provider={provider!r}): {exc}\n"
            "Refusing to spend against an unknown price. Either use an "
            "OpenRouter-style slug, or state the price explicitly with "
            "--price-input-per-1m / --price-output-per-1m."
        ) from exc


# =============================================================================
# Token / cost estimation
# =============================================================================


# Reasoning tokens are billed and budgeted as OUTPUT, so on a short target they
# can consume the entire cap and leave nothing for the document.
#
# Measured on 1,497 gemini-3.7-flash documents: the 218 that came back empty
# (14.6%) had reasoning median 244 tokens against an output median of 9, and
# were overwhelmingly micro/tiny/brief targets. The successful ones had
# reasoning 129 and output 448. A `micro` document asks for 10-25 words, so the
# v0.3.1 cap of max_words*2 = 50 tokens is consumed by thinking before a single
# word of content is written.
#
# Several providers refuse to disable reasoning at all (see the plan's §16.2),
# so the budget has to absorb it rather than assume it away.
# 1024 proved marginal: a `brief` document was observed spending 1,215 tokens on
# reasoning alone, leaving 65 of a 1,280 cap for the document itself.
_REASONING_HEADROOM_TOKENS = 3072


def max_output_tokens_for(length: Any, reasoning_on: bool = False) -> int:
    """Hard output cap for a document length bucket.

    With ``reasoning_on`` the cap gains headroom for thinking tokens; without
    it the value is unchanged from v0.3.1.
    """
    word_range = LENGTH_WORD_RANGES.get(length)
    max_words = word_range[1] if word_range else 500
    cap = max(256, min(16384, max_words * 2))
    if reasoning_on:
        cap = min(16384, cap + _REASONING_HEADROOM_TOKENS)
    return cap


@dataclass(frozen=True)
class TokenEstimate:
    """Expected and worst-case token counts for one request.

    ``expected_*`` drives the dry-run projection; ``worst_*`` drives the budget
    reservation. Only the worst case is load-bearing for the cap.
    """

    expected_input: int
    expected_output: int
    worst_input: int
    worst_output: int


def estimate_tokens(
    prompt: str,
    system_prompt: str,
    length: Any,
    *,
    reasoning_on: bool,
) -> TokenEstimate:
    """Bound and project the token usage of one generation request."""
    chars = len(prompt) + len(system_prompt)
    expected_input = max(1, int(chars / EXPECTED_CHARS_PER_TOKEN))
    worst_input = (
        int(math.ceil(chars / WORST_CASE_CHARS_PER_TOKEN))
        + WORST_CASE_INPUT_SLACK_TOKENS
    )

    word_range = LENGTH_WORD_RANGES.get(length) or (250, 500)
    midpoint = (word_range[0] + word_range[1]) / 2
    expected_output = int(midpoint * TOKENS_PER_WORD) + TITLE_OVERHEAD_TOKENS
    if reasoning_on:
        expected_output += EXPECTED_REASONING_TOKENS

    cap = max_output_tokens_for(length, reasoning_on)
    worst_output = int(cap * REASONING_WORST_CASE_MULTIPLIER) if reasoning_on else cap
    return TokenEstimate(
        expected_input=expected_input,
        expected_output=expected_output,
        worst_input=worst_input,
        worst_output=max(worst_output, expected_output),
    )


# =============================================================================
# Spec sourcing
# =============================================================================


def _stamp_prompt_variants(
    block: SpecBlock,
    *,
    fixed: PromptVariant | None,
    weights: dict[PromptVariant, float] | None,
    seed: int,
) -> SpecBlock:
    """Bake a prompt variant into every spec in a block.

    The variant is part of spec identity (it is hashed into ``spec_id``), which
    is exactly right for the paired design: every generator writing spec X uses
    the same system prompt, so generator stays the only varying factor.
    """
    if fixed is None and weights is None:
        return block
    sampler = PromptVariantSampler(weights, seed) if weights is not None else None
    specs = tuple(
        dataclasses.replace(
            spec, prompt_variant=(sampler.sample() if sampler else fixed)
        )
        for spec in block.specs
    )
    return SpecBlock(
        block_id=block.block_id,
        seed=block.seed,
        specs=specs,
        sampler_config=block.sampler_config,
    )


def draw_adhoc_block(
    count: int,
    *,
    seed: int | None,
    block_id: str,
    fixed_variant: PromptVariant | None,
    variant_weights: dict[PromptVariant, float] | None,
) -> SpecBlock:
    """Sample ``count`` fresh specs in-run (the legacy, non-block path).

    Kept so the documented ``--count 100 --model gpt-5.1`` invocation keeps
    working. The specs still get real ``spec_id``s and a synthetic ``block_id``,
    so provenance and resume behave identically to the block path.
    """
    sampler = DocumentSampler(seed=seed)
    lengths = LengthSampler(seed=seed)
    registers = RegisterSampler(seed=seed)
    variants = (
        PromptVariantSampler(variant_weights, seed)
        if variant_weights is not None
        else None
    )
    specs = tuple(
        DocumentSpec.from_document(
            sampler.sample(),
            target_length=lengths.sample(),
            register=registers.sample(),
            prompt_variant=(variants.sample() if variants else fixed_variant),
            block_id=block_id,
        )
        for _ in range(count)
    )
    return SpecBlock(
        block_id=block_id,
        seed=seed if seed is not None else -1,
        specs=specs,
        sampler_config={"source": "adhoc", "count": count},
    )


def resolve_spec_blocks(args: argparse.Namespace, run_id: str) -> list[SpecBlock]:
    """Build the spec blocks this run will generate from.

    Precedence: ``--spec-block`` files, then ``--draw-spec-blocks``, then an
    in-run ad-hoc draw of ``--count`` specs.
    """
    fixed_variant = PromptVariant(args.prompt_variant)
    weights = parse_variant_weights(args.prompt_variant_weights)

    if args.spec_block:
        blocks = [load_spec_block(path) for path in args.spec_block]
        if weights is not None:
            console.print(
                "[yellow]Note:[/yellow] --prompt-variant-weights is ignored for "
                "loaded spec blocks; the variant is part of spec identity and is "
                "assigned when the block is drawn."
            )
        return blocks

    if args.draw_spec_blocks:
        blocks = draw_spec_blocks(
            n_blocks=args.draw_spec_blocks,
            specs_per_block=args.specs_per_block,
            base_seed=args.seed if args.seed is not None else 42,
        )
        blocks = [
            _stamp_prompt_variants(
                block,
                fixed=fixed_variant,
                weights=weights,
                seed=block.seed,
            )
            for block in blocks
        ]
        if args.spec_block_dir:
            args.spec_block_dir.mkdir(parents=True, exist_ok=True)
            for block in blocks:
                path = save_spec_block(
                    block, args.spec_block_dir / f"{block.block_id}.jsonl"
                )
                console.print(f"  Saved spec block: {path} ({len(block)} specs)")
        return blocks

    return [
        draw_adhoc_block(
            args.count if args.count is not None else 100,
            seed=args.seed,
            block_id=f"adhoc-{run_id}",
            fixed_variant=fixed_variant,
            variant_weights=weights,
        )
    ]


def flatten_specs(blocks: list[SpecBlock], limit: int | None) -> list[DocumentSpec]:
    """Flatten blocks into a de-duplicated spec queue, optionally truncated."""
    seen: set[str] = set()
    specs: list[DocumentSpec] = []
    for block in blocks:
        for spec in block.specs:
            if spec.spec_id in seen:
                continue
            seen.add(spec.spec_id)
            specs.append(spec)
    if limit is not None and limit > 0:
        specs = specs[:limit]
    return specs


# =============================================================================
# CLI value parsing
# =============================================================================


def parse_variant_weights(raw: str | None) -> dict[PromptVariant, float] | None:
    """Parse ``"v0.4-direct=1,v0.4-archival=3"`` into normalized weights.

    ``"default"`` selects :data:`DEFAULT_PROMPT_VARIANT_WEIGHTS` (uniform over
    the four v0.4 variants).

    Raises:
        SystemExit: On an unknown variant name or a non-positive total.
    """
    if not raw:
        return None
    if raw.strip().lower() == "default":
        return dict(DEFAULT_PROMPT_VARIANT_WEIGHTS)

    weights: dict[PromptVariant, float] = {}
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        name, _, value = chunk.partition("=")
        try:
            variant = PromptVariant(name.strip())
        except ValueError as exc:
            valid = ", ".join(v.value for v in PromptVariant)
            raise SystemExit(
                f"Unknown prompt variant {name.strip()!r}. Valid: {valid}"
            ) from exc
        try:
            weights[variant] = float(value) if value else 1.0
        except ValueError as exc:
            raise SystemExit(f"Bad weight in {chunk!r}") from exc

    total = sum(weights.values())
    if total <= 0:
        raise SystemExit("--prompt-variant-weights must sum to a positive number")
    return {variant: weight / total for variant, weight in weights.items()}


def infer_provider(model: str) -> str:
    """Infer a provider from a model id.

    An OpenRouter-style ``vendor/model`` slug implies the gateway; otherwise the
    native provider is guessed from the family prefix.
    """
    if "/" in model:
        return "openrouter"
    lowered = model.lower()
    if lowered.startswith("claude"):
        return "anthropic"
    if lowered.startswith("gemini"):
        return "gemini"
    if lowered.startswith("grok"):
        return "xai"
    return "openai"


def build_reasoning_config(effort: str) -> ReasoningConfig | None:
    """Translate ``--reasoning-effort`` into a provider-neutral config.

    ``none`` (the default) is an explicit request to disable thinking, which is
    the cost-safe posture: reasoning tokens bill as output. ``default`` sends no
    reasoning parameter at all and leaves the provider's own default in place.
    """
    if effort == REASONING_EFFORT_PROVIDER_DEFAULT:
        return None
    if effort == "none":
        return ReasoningConfig.off()
    return ReasoningConfig(effort=effort, enabled=True, exclude=True)


# =============================================================================
# Backends
# =============================================================================


def _make_backend(
    provider: str,
    model: str,
    service_tier: str | None,
    use_batch_api: bool = False,
    thinking_budget: int | None = None,
    token_multiplier: float = 1.0,
    reasoning: ReasoningConfig | None = None,
    pin_provider: str | None = None,
) -> LLMBackend:
    """Instantiate the requested backend."""
    provider = provider.lower()
    if provider == "openai":
        return OpenAIResponsesBackend(
            model=model,
            service_tier=service_tier,
            reasoning=reasoning,
        )
    if provider == "anthropic":
        return AnthropicMessagesBackend(
            model=model,
            use_batch_api=use_batch_api,
            reasoning=reasoning,
        )
    if provider == "gemini":
        return GeminiBackend(
            model=model,
            use_batch_api=use_batch_api,
            thinking_budget=thinking_budget,
            token_multiplier=token_multiplier,
            reasoning=reasoning,
        )
    if provider == "openrouter":
        return OpenRouterBackend(
            model=model,
            reasoning=reasoning,
            routing=ProviderRouting.pin(pin_provider) if pin_provider else None,
            use_batch_api=use_batch_api,
        )
    if provider == "xai":
        return XAIBackend(model=model, reasoning=reasoning)
    raise ValueError(f"Unsupported provider: {provider}")


# =============================================================================
# Artifact construction
# =============================================================================


def build_artifact(
    doc_id: str,
    spec: DocumentSpec,
    variant: PromptVariant,
    title: str,
    body: str,
    sampling: Any,
    model: str,
    provider: str,
    prompt: str,
    gen_result: Any,
    cost_usd: float,
    run_id: str,
    git_info: dict[str, str | bool | None] | None = None,
) -> dict[str, Any]:
    """Build one artifact record, including full generation provenance."""
    doc = spec.to_document()
    word_range = LENGTH_WORD_RANGES.get(spec.target_length)
    reasoning_tokens = getattr(gen_result, "reasoning_tokens", None)

    record: dict[str, Any] = {
        "id": doc_id,
        "title": title,
        "body": body,
        "word_count": len(body.split()),
        # LC labels
        "lcc_code": doc.lcc.code,
        "lcc_name": doc.lcc.name,
        "lcc_uri": getattr(doc.lcc, "uri", None),
        "lcgft_category": doc.lcgft.category,
        "lcgft_form": doc.lcgft.form,
        "topics": list(spec.topics),
        "audience": spec.audience,
        "geographic": list(spec.geographic),
        # Generation params
        "target_length": spec.target_length.value,
        "target_word_range": list(word_range) if word_range else None,
        "register": spec.register.value,
        "register_description": REGISTER_DESCRIPTIONS.get(spec.register),
        "temperature": sampling.temperature,
        "top_p": sampling.top_p,
        # Provenance
        "model": model,
        "model_resolved": getattr(gen_result, "model_resolved", None),
        "provider": provider,
        "provider_served": getattr(gen_result, "provider_served", None),
        "prompt": prompt,
        "prompt_variant_id": variant.value,
        "spec_id": spec.spec_id,
        "block_id": spec.block_id,
        # Minimal-pair grouping (Phase 3). Without these the pair members
        # cannot be kept in the same split and the pair task leaks.
        "pair_id": spec.pair_id,
        "pair_role": spec.pair_role,
        "pair_axis": spec.pair_axis,
        "run_id": run_id,
        "input_tokens": getattr(gen_result, "input_tokens", None),
        "output_tokens": getattr(gen_result, "output_tokens", None),
        "reasoning_tokens": reasoning_tokens,
        "cost_usd": round(cost_usd, 8),
    }

    if git_info:
        record["git_commit"] = git_info.get("commit")
        record["git_dirty"] = git_info.get("dirty")
        record["git_branch"] = git_info.get("branch")
        record["code_version"] = git_info.get("version_string")

    return record


# =============================================================================
# Per-model run plan
# =============================================================================


@dataclass
class ModelPlan:
    """Everything decided about one model before any request goes out."""

    model: str
    provider: str
    price: ModelPrice
    specs: list[DocumentSpec]
    already_done: int
    cap: float | None
    committed: float
    expected_cost: float = 0.0
    worst_case_cost: float = 0.0

    @property
    def remaining(self) -> float | None:
        """Headroom under the cap after prior spend on this ledger."""
        if self.cap is None:
            return None
        return max(0.0, self.cap - self.committed)

    @property
    def within_budget(self) -> bool:
        """True when the projected cost fits under the cap."""
        remaining = self.remaining
        return remaining is None or self.expected_cost <= remaining

    def to_dict(self) -> dict[str, Any]:
        """Manifest form."""
        return {
            "model": self.model,
            "provider": self.provider,
            "price": self.price.to_dict(),
            "n_specs": len(self.specs),
            "already_completed": self.already_done,
            "cap": self.cap,
            "committed_before_run": round(self.committed, 6),
            "expected_cost": round(self.expected_cost, 6),
            "worst_case_cost": round(self.worst_case_cost, 6),
            "within_budget": self.within_budget,
        }


def build_model_plan(
    model: str,
    provider: str,
    price: ModelPrice,
    specs: list[DocumentSpec],
    *,
    guard: BudgetGuard,
    fixed_variant: PromptVariant,
    reasoning_on: bool,
    resume: bool,
) -> ModelPlan:
    """Cost the full queue for one model against live prices."""
    done = guard.ledger.completed_spec_ids(model=model) if resume else set()
    pending = [spec for spec in specs if spec.spec_id not in done]

    expected = 0.0
    worst = 0.0
    for spec in pending:
        variant = spec.prompt_variant or fixed_variant
        doc = spec.to_document()
        prompt = build_generation_prompt(doc, spec.target_length, spec.register)
        system_prompt = build_system_prompt(doc, variant)
        estimate = estimate_tokens(
            prompt, system_prompt, spec.target_length, reasoning_on=reasoning_on
        )
        expected += price.cost(estimate.expected_input, estimate.expected_output)
        worst += price.cost(estimate.worst_input, estimate.worst_output)

    return ModelPlan(
        model=model,
        provider=provider,
        price=price,
        specs=pending,
        already_done=len(specs) - len(pending),
        cap=guard.cap_for(model),
        committed=guard.committed(model),
        expected_cost=expected,
        worst_case_cost=worst,
    )


# =============================================================================
# Generation
# =============================================================================


@dataclass
class ModelOutcome:
    """What actually happened for one model."""

    model: str
    generated: int = 0
    failed: int = 0
    skipped: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0
    budget_aborted: bool = False
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Manifest form."""
        return {
            "model": self.model,
            "generated": self.generated,
            "failed": self.failed,
            "skipped": self.skipped,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "budget_aborted": self.budget_aborted,
            "errors": self.errors[:20],
        }


async def generate_one(
    spec: DocumentSpec,
    *,
    backend: LLMBackend,
    model: str,
    provider: str,
    price: ModelPrice,
    guard: BudgetGuard,
    sampling_sampler: SamplingParamsSampler,
    fixed_variant: PromptVariant,
    reasoning_on: bool,
    semaphore: asyncio.Semaphore,
    abort: asyncio.Event,
    run_id: str,
    block_id: str,
    git_info: dict[str, str | bool | None] | None = None,
) -> tuple[str, dict[str, Any] | None, str | None]:
    """Generate one document under a budget reservation.

    The reservation is taken *before* the request goes out and sized to the
    worst case, so concurrent in-flight requests are all charged against the cap
    simultaneously. The actual cost is committed when the response lands.

    Returns:
        ``(status, artifact, error)`` where status is one of
        ``generated`` / ``failed`` / ``skipped_budget``.
    """
    if abort.is_set():
        return STATUS_SKIPPED_BUDGET, None, None

    async with semaphore:
        if abort.is_set():
            return STATUS_SKIPPED_BUDGET, None, None

        variant = spec.prompt_variant or fixed_variant
        doc = spec.to_document()
        prompt = build_generation_prompt(doc, spec.target_length, spec.register)
        system_prompt = build_system_prompt(doc, variant)
        sampling = sampling_sampler.sample()
        estimate = estimate_tokens(
            prompt, system_prompt, spec.target_length, reasoning_on=reasoning_on
        )
        worst_case = price.cost(estimate.worst_input, estimate.worst_output)

        try:
            with guard.reserve(
                model,
                worst_case,
                spec_id=spec.spec_id,
                record_failure=True,
            ) as pending:
                gen_result = await backend.generate_async(
                    GenerationRequest(prompt=prompt, system_prompt=system_prompt),
                    GenerationParams(
                        temperature=sampling.temperature,
                        top_p=sampling.top_p,
                        max_output_tokens=max_output_tokens_for(
                            spec.target_length, reasoning_on
                        ),
                    ),
                )
                title, body = _parse_generated_text(gen_result.text)
                input_tokens = gen_result.input_tokens or 0
                output_tokens = gen_result.output_tokens or 0
                cost = price.cost(input_tokens, output_tokens)
                pending.commit(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    reasoning_tokens=gen_result.reasoning_tokens or 0,
                    cost_usd=cost,
                    provider_served=gen_result.provider_served,
                    status=STATUS_OK,
                    block_id=block_id,
                    prompt_variant_id=variant.value,
                    model_resolved=gen_result.model_resolved,
                )
        except BudgetExceeded as exc:
            abort.set()
            return STATUS_SKIPPED_BUDGET, None, str(exc)
        except Exception as exc:  # noqa: BLE001 - one bad doc must not kill a run
            return STATUS_FAILED, None, f"{spec.spec_id}: {exc}"

        artifact = build_artifact(
            generate_artifact_id(),
            spec,
            variant,
            title,
            body,
            sampling,
            model,
            provider,
            prompt,
            gen_result,
            cost,
            run_id,
            git_info,
        )
        return STATUS_GENERATED, artifact, None


def generate_batch_sync(
    specs: list[DocumentSpec],
    *,
    backend: LLMBackend,
    model: str,
    provider: str,
    price: ModelPrice,
    guard: BudgetGuard,
    sampling_sampler: SamplingParamsSampler,
    fixed_variant: PromptVariant,
    reasoning_on: bool,
    run_id: str,
    block_id: str,
    git_info: dict[str, str | bool | None] | None = None,
) -> tuple[list[dict[str, Any] | None], list[str]]:
    """Generate a batch through the backend's batch API, budget-checked.

    Batch submission is all-or-nothing, so the whole batch's worst case is
    checked against the cap before anything is submitted; per-spec costs are
    booked afterwards (``check=False``, because the money is already spent and
    losing the accounting is worse than overshooting the cap on paper).
    """
    sampling = sampling_sampler.sample()
    requests = []
    worst_total = 0.0
    for spec in specs:
        variant = spec.prompt_variant or fixed_variant
        doc = spec.to_document()
        prompt = build_generation_prompt(doc, spec.target_length, spec.register)
        system_prompt = build_system_prompt(doc, variant)
        estimate = estimate_tokens(
            prompt, system_prompt, spec.target_length, reasoning_on=reasoning_on
        )
        worst_total += price.cost(estimate.worst_input, estimate.worst_output)
        requests.append(GenerationRequest(prompt=prompt, system_prompt=system_prompt))

    guard.check(model, worst_total)

    params = GenerationParams(
        temperature=sampling.temperature,
        top_p=sampling.top_p,
        max_output_tokens=max(
            max_output_tokens_for(spec.target_length, reasoning_on) for spec in specs
        ),
    )
    gen_results = backend.generate_batch(requests, params)

    results: list[dict[str, Any] | None] = []
    errors: list[str] = []
    for spec, request, gen_result in zip(specs, requests, gen_results, strict=False):
        variant = spec.prompt_variant or fixed_variant
        input_tokens = gen_result.input_tokens or 0
        output_tokens = gen_result.output_tokens or 0
        cost = price.cost(input_tokens, output_tokens)
        try:
            title, body = _parse_generated_text(gen_result.text)
        except Exception as exc:  # noqa: BLE001
            guard.record(
                model,
                cost_usd=cost,
                check=False,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                spec_id=spec.spec_id,
                status="error",
                block_id=block_id,
            )
            results.append(None)
            errors.append(f"{spec.spec_id}: {exc}")
            continue

        guard.record(
            model,
            cost_usd=cost,
            check=False,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=gen_result.reasoning_tokens or 0,
            provider_served=gen_result.provider_served,
            spec_id=spec.spec_id,
            status=STATUS_OK,
            block_id=block_id,
            prompt_variant_id=variant.value,
        )
        results.append(
            build_artifact(
                generate_artifact_id(),
                spec,
                variant,
                title,
                body,
                sampling,
                model,
                provider,
                request.prompt,
                gen_result,
                cost,
                run_id,
                git_info,
            )
        )
    return results, errors


class ArtifactWriter:
    """Appends artifacts to a JSONL index and one JSON file each."""

    def __init__(self, output_path: Path, artifacts_dir: Path, log_usage: bool):
        self.output_path = output_path
        self.artifacts_dir = artifacts_dir
        self.log_usage = log_usage
        output_path.parent.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._handle = output_path.open("a", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        """Persist one artifact."""
        if self.log_usage:
            console.print(
                f"[green]OK[/green] {record['id']} | {record['lcgft_form'][:20]:20} | "
                f"{record['word_count']:4} words | ${record['cost_usd']:.5f} | "
                f"in:{record.get('input_tokens')} out:{record.get('output_tokens')}"
            )
        path = self.artifacts_dir / f"{record['id']}.json"
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        self._handle.write(json.dumps(record) + "\n")
        self._handle.flush()

    def close(self) -> None:
        """Close the JSONL index."""
        self._handle.close()


async def generate_for_model(
    plan: ModelPlan,
    *,
    args: argparse.Namespace,
    guard: BudgetGuard,
    writer: ArtifactWriter,
    run_id: str,
    git_info: dict[str, str | bool | None] | None,
    backend_factory: Any = _make_backend,
) -> ModelOutcome:
    """Run one model's whole queue, stopping cleanly if it hits its cap."""
    outcome = ModelOutcome(model=plan.model)
    if not plan.specs:
        return outcome

    reasoning = build_reasoning_config(args.reasoning_effort)
    reasoning_on = reasoning is not None and not reasoning.is_off
    fixed_variant = PromptVariant(args.prompt_variant)
    use_batch_api = args.batch_size > 0 and plan.provider != "openai"

    backend = backend_factory(
        plan.provider,
        plan.model,
        args.service_tier,
        use_batch_api=use_batch_api,
        thinking_budget=args.thinking_budget,
        token_multiplier=args.token_multiplier,
        reasoning=reasoning,
        pin_provider=args.pin_provider,
    )
    sampling_sampler = SamplingParamsSampler(seed=args.seed)
    block_id = plan.specs[0].block_id

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )

    def account(record: dict[str, Any]) -> None:
        outcome.generated += 1
        outcome.input_tokens += record.get("input_tokens") or 0
        outcome.output_tokens += record.get("output_tokens") or 0
        outcome.reasoning_tokens += record.get("reasoning_tokens") or 0
        outcome.cost_usd += record.get("cost_usd") or 0.0
        writer.write(record)

    if use_batch_api:
        with progress:
            task_id = progress.add_task(
                f"Generating [{plan.model}]", total=len(plan.specs)
            )
            for start in range(0, len(plan.specs), args.batch_size):
                chunk = plan.specs[start : start + args.batch_size]
                try:
                    records, errors = generate_batch_sync(
                        chunk,
                        backend=backend,
                        model=plan.model,
                        provider=plan.provider,
                        price=plan.price,
                        guard=guard,
                        sampling_sampler=sampling_sampler,
                        fixed_variant=fixed_variant,
                        reasoning_on=reasoning_on,
                        run_id=run_id,
                        block_id=block_id,
                        git_info=git_info,
                    )
                except BudgetExceeded as exc:
                    outcome.budget_aborted = True
                    outcome.skipped += len(plan.specs) - start
                    outcome.errors.append(str(exc))
                    console.print(f"[red]Budget cap reached:[/red] {exc}")
                    break
                except Exception as exc:  # noqa: BLE001
                    outcome.failed += len(chunk)
                    outcome.errors.append(f"batch failed: {exc}")
                    progress.advance(task_id, len(chunk))
                    continue
                outcome.errors.extend(errors)
                for record in records:
                    if record is None:
                        outcome.failed += 1
                    else:
                        account(record)
                    progress.advance(task_id)
        return outcome

    semaphore = asyncio.Semaphore(args.concurrency)
    abort = asyncio.Event()
    tasks = [
        asyncio.create_task(
            generate_one(
                spec,
                backend=backend,
                model=plan.model,
                provider=plan.provider,
                price=plan.price,
                guard=guard,
                sampling_sampler=sampling_sampler,
                fixed_variant=fixed_variant,
                reasoning_on=reasoning_on,
                semaphore=semaphore,
                abort=abort,
                run_id=run_id,
                block_id=spec.block_id or block_id,
                git_info=git_info,
            )
        )
        for spec in plan.specs
    ]

    with progress:
        task_id = progress.add_task(f"Generating [{plan.model}]", total=len(tasks))
        for coro in asyncio.as_completed(tasks):
            status, record, error = await coro
            progress.advance(task_id)
            if status == STATUS_GENERATED and record is not None:
                account(record)
            elif status == STATUS_SKIPPED_BUDGET:
                outcome.skipped += 1
                outcome.budget_aborted = True
                if error:
                    outcome.errors.append(error)
            else:
                outcome.failed += 1
                if error:
                    outcome.errors.append(error)

    if outcome.budget_aborted:
        console.print(
            f"[red]Budget cap reached for {plan.model}[/red] - "
            f"{outcome.generated:,} written, {outcome.skipped:,} skipped. "
            "Continuing with remaining models."
        )
    return outcome


# =============================================================================
# Reporting
# =============================================================================


def print_plan_table(plans: list[ModelPlan], budget_total: float | None) -> None:
    """Print the per-model cost projection."""
    table = Table(title="Projected cost", show_lines=False)
    table.add_column("Model")
    table.add_column("Provider")
    table.add_column("Docs", justify="right")
    table.add_column("Done", justify="right")
    table.add_column("$/1M in", justify="right")
    table.add_column("$/1M out", justify="right")
    table.add_column("Projected", justify="right")
    table.add_column("Worst case", justify="right")
    table.add_column("Cap", justify="right")
    table.add_column("Verdict")

    for plan in plans:
        remaining = plan.remaining
        if not plan.within_budget:
            verdict = "[red]OVER CAP[/red]"
        elif remaining is not None and plan.worst_case_cost > remaining:
            # The cap is enforced against worst-case reservations, so a run whose
            # ceiling exceeds the cap can still stop early even though its
            # projection fits. Say so before the money is committed.
            verdict = "[yellow]may abort early[/yellow]"
        else:
            verdict = "[green]within budget[/green]"

        table.add_row(
            plan.model,
            plan.provider,
            f"{len(plan.specs):,}",
            f"{plan.already_done:,}",
            f"{plan.price.input_per_1m:.3f}",
            f"{plan.price.output_per_1m:.3f}",
            f"${plan.expected_cost:.2f}",
            f"${plan.worst_case_cost:.2f}",
            "-" if plan.cap is None else f"${plan.cap:.2f}",
            verdict,
        )

    total_expected = sum(plan.expected_cost for plan in plans)
    total_worst = sum(plan.worst_case_cost for plan in plans)
    table.add_section()
    table.add_row(
        "[bold]TOTAL[/bold]",
        "",
        f"{sum(len(p.specs) for p in plans):,}",
        f"{sum(p.already_done for p in plans):,}",
        "",
        "",
        f"[bold]${total_expected:.2f}[/bold]",
        f"[bold]${total_worst:.2f}[/bold]",
        "-" if budget_total is None else f"${budget_total:.2f}",
        "",
    )
    console.print(table)


def write_manifest(
    path: Path,
    *,
    run_id: str,
    args: argparse.Namespace,
    pricing: PricingTable,
    blocks: list[SpecBlock],
    plans: list[ModelPlan],
    outcomes: list[ModelOutcome],
    guard: BudgetGuard | None,
    git_info: dict[str, str | bool | None] | None,
    dry_run: bool,
) -> Path:
    """Write the run manifest, including the pricing snapshot's ``fetched_at``."""
    manifest = {
        "run_id": run_id,
        "dry_run": dry_run,
        "git": git_info,
        "args": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in sorted(vars(args).items())
            if key != "func"
        },
        "pricing": pricing.manifest_entry([plan.model for plan in plans]),
        "pricing_fetched_at": pricing.fetched_at,
        "spec_blocks": [block.to_manifest() for block in blocks],
        "plans": [plan.to_dict() for plan in plans],
        "outcomes": [outcome.to_dict() for outcome in outcomes],
        "budget": (
            guard.report([plan.model for plan in plans]).to_dict() if guard else None
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return path


# =============================================================================
# Orchestration
# =============================================================================


async def run(
    args: argparse.Namespace,
    *,
    pricing: PricingTable,
    backend_factory: Any = _make_backend,
) -> int:
    """Plan, price, and (unless ``--dry-run``) generate. Returns an exit code."""
    run_id = new_run_id()
    git_info = get_git_version()
    console.print(f"  Run id: {run_id}")
    console.print(f"  Code version: {git_info.get('version_string', 'unknown')}")
    console.print(f"  Pricing fetched_at: {pricing.fetched_at}")

    blocks = resolve_spec_blocks(args, run_id)
    # In ad-hoc mode --count already bounded the draw; in block mode it is an
    # optional truncation, so a block of 500 is not silently cut to a default.
    from_blocks = bool(args.spec_block or args.draw_spec_blocks)
    specs = flatten_specs(blocks, args.count if from_blocks else None)
    console.print(
        f"  Spec blocks: {len(blocks)} "
        f"({', '.join(block.block_id for block in blocks)})"
    )
    console.print(f"  Specs queued: {len(specs):,}")

    if args.draw_only:
        console.print("[green]Spec blocks drawn; --draw-only, stopping here.[/green]")
        return 0

    ledger = CostLedger(args.ledger, run_id=run_id, fsync=not args.no_fsync)
    guard = BudgetGuard(
        ledger,
        per_model_cap=args.budget_per_model,
        global_cap=args.budget_total,
    )

    reasoning = build_reasoning_config(args.reasoning_effort)
    reasoning_on = reasoning is not None and not reasoning.is_off
    fixed_variant = PromptVariant(args.prompt_variant)

    plans: list[ModelPlan] = []
    for model in args.model:
        provider = args.provider or infer_provider(model)
        price = resolve_price(pricing, model, provider)
        plans.append(
            build_model_plan(
                model,
                provider,
                price,
                specs,
                guard=guard,
                fixed_variant=fixed_variant,
                reasoning_on=reasoning_on,
                resume=not args.no_resume,
            )
        )

    print_plan_table(plans, args.budget_total)

    if args.dry_run:
        over = [plan for plan in plans if not plan.within_budget]
        manifest_path = write_manifest(
            args.manifest or (args.artifacts_dir / "manifests" / f"{run_id}.json"),
            run_id=run_id,
            args=args,
            pricing=pricing,
            blocks=blocks,
            plans=plans,
            outcomes=[],
            guard=guard,
            git_info=git_info,
            dry_run=True,
        )
        ledger.close()
        console.print(f"  Manifest: {manifest_path}")
        console.print("[bold green]DRY RUN[/bold green] - no API calls were made.")
        if over:
            console.print(
                "[red]"
                + ", ".join(plan.model for plan in over)
                + " exceed their cap; raise --budget-per-model or lower --count."
                + "[/red]"
            )
            return 2
        return 0

    writer = ArtifactWriter(args.output, args.artifacts_dir, args.log_usage)
    outcomes: list[ModelOutcome] = []
    try:
        for plan in plans:
            if not plan.specs:
                console.print(
                    f"[yellow]Nothing to do for {plan.model}[/yellow] "
                    f"({plan.already_done:,} specs already completed)"
                )
                outcomes.append(ModelOutcome(model=plan.model))
                continue
            if guard.is_exhausted(plan.model):
                console.print(
                    f"[red]Skipping {plan.model}[/red]: budget already exhausted "
                    f"(${guard.committed(plan.model):.2f} of ${plan.cap})"
                )
                outcomes.append(
                    ModelOutcome(
                        model=plan.model,
                        skipped=len(plan.specs),
                        budget_aborted=True,
                    )
                )
                continue
            outcomes.append(
                await generate_for_model(
                    plan,
                    args=args,
                    guard=guard,
                    writer=writer,
                    run_id=run_id,
                    git_info=git_info,
                    backend_factory=backend_factory,
                )
            )
    finally:
        writer.close()
        manifest_path = write_manifest(
            args.manifest or (args.artifacts_dir / "manifests" / f"{run_id}.json"),
            run_id=run_id,
            args=args,
            pricing=pricing,
            blocks=blocks,
            plans=plans,
            outcomes=outcomes,
            guard=guard,
            git_info=git_info,
            dry_run=False,
        )
        ledger.close()

    # Failure reasons are only held in memory during the run, and the ledger's
    # error rows carry zero tokens and zero cost -- so a failed model is
    # invisible in a spend total and looks identical to a success in a row
    # count. Persist the reasons so a partial run can be diagnosed afterwards
    # instead of re-derived by reproducing the call. (A real run lost three
    # models entirely to one 400 and the reason was nowhere on disk.)
    error_log = Path(args.ledger).with_suffix(".errors.jsonl")
    total_failures = sum(o.failed for o in outcomes)
    if total_failures:
        error_log.parent.mkdir(parents=True, exist_ok=True)
        with error_log.open("a", encoding="utf-8") as handle:
            for outcome in outcomes:
                for reason in outcome.errors:
                    handle.write(
                        json.dumps(
                            {
                                "run_id": run_id,
                                "model": outcome.model,
                                "reason": str(reason)[:500],
                            }
                        )
                        + "\n"
                    )

    console.print("\n[bold]Summary:[/bold]")
    for outcome in outcomes:
        marker = ""
        if outcome.budget_aborted:
            marker = " [red](budget cap hit)[/red]"
        elif outcome.failed and outcome.generated == 0:
            marker = " [red](PRODUCED NOTHING)[/red]"
        elif outcome.failed:
            marker = f" [yellow]({outcome.failed:,} failed)[/yellow]"
        console.print(
            f"  {outcome.model}: {outcome.generated:,} generated, "
            f"{outcome.failed:,} failed, {outcome.skipped:,} skipped, "
            f"${outcome.cost_usd:.4f} spent" + marker
        )
    if total_failures:
        console.print(f"  [yellow]Failure reasons: {error_log}[/yellow]")
    console.print(f"  Total spend this run: ${sum(o.cost_usd for o in outcomes):.4f}")
    console.print(f"  Ledger: {args.ledger}")
    console.print(f"  Manifest: {manifest_path}")
    console.print(f"  JSONL index: {args.output}")
    console.print(f"  Artifacts: {args.artifacts_dir}/")

    return 3 if any(o.budget_aborted for o in outcomes) else 0


def load_api_keys() -> None:
    """Hydrate API keys from ``env.json``, best-effort.

    Imported lazily because ``shelf.config`` pulls the vendor SDKs in at module
    scope, which would make this script unimportable (and untestable) wherever
    the optional provider packages are not installed. A missing ``env.json`` or
    a missing SDK is not fatal: the keys may already be exported in the
    environment, and the backends raise their own clear error if they are not.
    """
    try:
        from shelf.config import load_env

        load_env()
    except (ImportError, FileNotFoundError) as exc:
        console.print(
            f"[yellow]  Not loading env.json ({exc}); using os.environ[/yellow]"
        )


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Generate benchmark documents with LC taxonomy labels"
    )

    # -- output ------------------------------------------------------------ #
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
        "--manifest",
        type=Path,
        default=None,
        help="Run manifest path (default: <artifacts-dir>/manifests/<run_id>.json)",
    )

    # -- what to generate --------------------------------------------------- #
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Documents to generate (default: 100 when sampling fresh specs). "
        "With --spec-block/--draw-spec-blocks this truncates the spec queue "
        "instead of drawing fresh specs; omit it to generate the whole block.",
    )
    parser.add_argument(
        "--model",
        nargs="+",
        default=["gpt-5.1"],
        help="One or more models. Each gets its own budget and is run in turn.",
    )
    parser.add_argument(
        "--provider",
        default=None,
        choices=["openai", "anthropic", "gemini", "openrouter", "xai"],
        help="LLM provider (default: inferred from model id)",
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
        "Anthropic/Gemini/OpenRouter only.",
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=None,
        help="Gemini thinking budget (0=disable, >0=set budget). Overrides "
        "--reasoning-effort for Gemini.",
    )
    parser.add_argument(
        "--token-multiplier",
        type=float,
        default=1.0,
        help="Multiply max_output_tokens (Gemini only)",
    )

    # -- budget ------------------------------------------------------------- #
    budget = parser.add_argument_group("budget")
    budget.add_argument(
        "--budget-per-model",
        type=float,
        default=DEFAULT_BUDGET_PER_MODEL_USD,
        help=f"Hard USD cap per model, enforced against the ledger's lifetime "
        f"spend for that model (default: {DEFAULT_BUDGET_PER_MODEL_USD:.2f}). "
        "A run is never unbounded.",
    )
    budget.add_argument(
        "--budget-total",
        type=float,
        default=None,
        help="Optional hard USD cap across every model in the ledger file",
    )
    budget.add_argument(
        "--ledger",
        type=Path,
        default=DEFAULT_LEDGER_PATH,
        help=f"Cost ledger JSONL (default: {DEFAULT_LEDGER_PATH})",
    )
    budget.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-generate spec ids already completed for a model (re-spends)",
    )
    budget.add_argument(
        "--no-fsync",
        action="store_true",
        help="Skip fsync on ledger writes (faster, less crash-durable)",
    )
    budget.add_argument(
        "--dry-run",
        action="store_true",
        help="Sample/load specs, price the run against live pricing, print the "
        "projection, and exit without making a single API call",
    )

    # -- pricing ------------------------------------------------------------ #
    pricing = parser.add_argument_group("pricing")
    pricing.add_argument(
        "--pricing-cache",
        type=Path,
        default=DEFAULT_CACHE_PATH,
        help=f"Pricing cache path (default: {DEFAULT_CACHE_PATH}, 24h TTL)",
    )
    pricing.add_argument(
        "--refresh-pricing",
        action="store_true",
        help="Re-fetch the OpenRouter catalogue even if the cache is fresh",
    )
    pricing.add_argument(
        "--offline-pricing",
        action="store_true",
        help="Never touch the network for pricing; use the cache only",
    )
    pricing.add_argument(
        "--no-native-prices",
        action="store_true",
        help="Price against OpenRouter's catalogue only, without the verified "
        "provider-page overrides (which are the higher, conservative figure "
        "wherever the two disagree)",
    )
    pricing.add_argument(
        "--price-input-per-1m",
        type=float,
        default=None,
        help="Manual input price for every --model (USD per 1M tokens). Use for "
        "models the catalogue does not list.",
    )
    pricing.add_argument(
        "--price-output-per-1m",
        type=float,
        default=None,
        help="Manual output price for every --model (USD per 1M tokens)",
    )

    # -- spec blocks -------------------------------------------------------- #
    specs = parser.add_argument_group("spec blocks")
    specs.add_argument(
        "--spec-block",
        type=Path,
        action="append",
        default=None,
        help="Generate from a saved spec block (repeatable)",
    )
    specs.add_argument(
        "--draw-spec-blocks",
        type=int,
        default=0,
        help="Draw N independent spec blocks (block i uses seed --seed + i)",
    )
    specs.add_argument(
        "--specs-per-block",
        type=int,
        default=500,
        help="Specs per drawn block (default: 500)",
    )
    specs.add_argument(
        "--spec-block-dir",
        type=Path,
        default=None,
        help="Directory to save drawn spec blocks into",
    )
    specs.add_argument(
        "--draw-only",
        action="store_true",
        help="Draw and save spec blocks, then exit without generating",
    )

    # -- prompts and reasoning ---------------------------------------------- #
    prompts = parser.add_argument_group("prompts and reasoning")
    prompts.add_argument(
        "--prompt-variant",
        default=PromptVariant.V0_3_1.value,
        choices=[variant.value for variant in PromptVariant],
        help=f"System prompt variant (default: {PromptVariant.V0_3_1.value}, the "
        "frozen v0.3.1 preset)",
    )
    prompts.add_argument(
        "--prompt-variant-weights",
        default=None,
        help='Sample variants instead, e.g. "v0.4-direct=1,v0.4-archival=3" or '
        '"default" for uniform over the four v0.4 variants. Applied when specs '
        "are drawn, so the variant becomes part of spec identity.",
    )
    prompts.add_argument(
        "--reasoning-effort",
        default="none",
        choices=[*REASONING_EFFORTS, REASONING_EFFORT_PROVIDER_DEFAULT],
        help="Reasoning effort (default: none). Reasoning tokens bill as output "
        "and are the main cost-overrun risk. 'default' sends no reasoning "
        "parameter at all.",
    )
    prompts.add_argument(
        "--pin-provider",
        default=None,
        help="Pin OpenRouter routing to one upstream provider slug "
        "(e.g. 'deepinfra') for reproducibility",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)

    if args.budget_per_model is not None and args.budget_per_model <= 0:
        raise SystemExit("--budget-per-model must be positive")
    if (args.price_input_per_1m is None) != (args.price_output_per_1m is None):
        raise SystemExit(
            "--price-input-per-1m and --price-output-per-1m must be given together"
        )

    console.print("[bold]SHELF Document Generator[/bold]")
    console.print(f"  Models: {', '.join(args.model)}")
    console.print(f"  Budget/model: ${args.budget_per_model:.2f}")
    if args.budget_total is not None:
        console.print(f"  Budget total: ${args.budget_total:.2f}")
    if args.budget_per_model == DEFAULT_BUDGET_PER_MODEL_USD:
        console.print(
            "[yellow]  Using the default per-model cap. Pass --budget-per-model "
            "for a real run.[/yellow]"
        )
    console.print(f"  Reasoning effort: {args.reasoning_effort}")
    console.print(f"  Prompt variant: {args.prompt_variant}")
    if args.batch_size > 0:
        console.print(f"  Batch size: {args.batch_size}")
    else:
        console.print(f"  Concurrency: {args.concurrency}")
    console.print(f"  Output: {args.output}")
    console.print(f"  Artifacts: {args.artifacts_dir}")
    if args.seed is not None:
        console.print(f"  Seed: {args.seed}")

    load_api_keys()

    extra_overrides = None
    if args.price_input_per_1m is not None:
        extra_overrides = {
            model: PriceOverride(
                args.price_input_per_1m,
                args.price_output_per_1m,
                note="manual --price-*-per-1m",
            )
            for model in args.model
        }

    try:
        pricing = load_pricing_table(
            cache_path=args.pricing_cache,
            refresh=args.refresh_pricing,
            allow_network=not args.offline_pricing,
            native_prices=not args.no_native_prices,
            extra_overrides=extra_overrides,
        )
    except PricingError as exc:
        raise SystemExit(f"Cannot load pricing: {exc}") from exc

    return asyncio.run(run(args, pricing=pricing))


if __name__ == "__main__":
    sys.exit(main())
