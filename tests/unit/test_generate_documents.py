"""Unit tests for scripts/generate_documents.py (the money-spending path).

The invariants under test are the ones that stop money being lost:

- ``--dry-run`` prices a run and makes **zero** calls -- neither to an LLM
  backend nor over the network.
- The per-model cap is never unbounded: omitting ``--budget-per-model`` falls
  back to a conservative default rather than to "no limit".
- A request reserves its *worst case* before the call and commits its *actual*
  cost afterwards, so the reservation is always an upper bound.
- A model that cannot be priced is refused rather than run blind.
- Every artifact carries the provenance Phase 1 depends on (``spec_id``,
  ``block_id``, ``prompt_variant_id``, ``provider_served``, ``model_resolved``,
  ``reasoning_tokens``).
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from shelf.llm import BudgetGuard, CostLedger, GenerationResult, ReasoningConfig
from shelf.llm.pricing import PricingTable
from shelf.sampler.generator import DocumentLength, PromptVariant, Register
from shelf.sampler.specs import DocumentSpec, SpecBlock, save_spec_block

# The generator is a script, not a package module, so load it by path.
_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "generate_documents.py"
_spec = importlib.util.spec_from_file_location("generate_documents", _SCRIPT)
assert _spec is not None and _spec.loader is not None
gd = importlib.util.module_from_spec(_spec)
sys.modules["generate_documents"] = gd
_spec.loader.exec_module(gd)


# =============================================================================
# Fixtures
# =============================================================================

#: A miniature OpenRouter catalogue. Prices are round numbers so expected costs
#: can be computed by hand in the assertions below.
CATALOGUE: list[dict[str, Any]] = [
    {
        "id": "test/cheap",
        "name": "Cheap Test Model",
        "context_length": 128_000,
        # $1 / 1M input, $10 / 1M output.
        "pricing": {"prompt": "0.000001", "completion": "0.00001"},
    },
    {
        "id": "test/pricey",
        "name": "Pricey Test Model",
        "context_length": 128_000,
        # $10 / 1M input, $100 / 1M output.
        "pricing": {"prompt": "0.00001", "completion": "0.0001"},
    },
]


@pytest.fixture
def pricing() -> PricingTable:
    """A pricing table built from a fixture payload; never touches the network."""
    return PricingTable.from_payload(CATALOGUE, fetched_at="2026-08-26T00:00:00+00:00")


def make_spec(index: int, *, block_id: str = "block-test") -> DocumentSpec:
    """Build a deterministic spec without going through the sampler."""
    return DocumentSpec(
        lcc_code="QA",
        lcc_name="Mathematics",
        lcgft_form="Textbooks",
        lcgft_category="Instructional and educational works",
        topics=(f"Topic {index}",),
        target_length=DocumentLength.BRIEF,
        register=Register.ACADEMIC,
        audience="students",
        geographic=(),
        block_id=block_id,
    )


def make_block(n: int, block_id: str = "block-test") -> SpecBlock:
    """A spec block of ``n`` distinct specs."""
    return SpecBlock(
        block_id=block_id,
        seed=1,
        specs=tuple(make_spec(i, block_id=block_id) for i in range(n)),
    )


class RecordingBackend:
    """A backend that never leaves the process and reports fixed usage."""

    provider = "test"

    def __init__(
        self,
        model: str = "test/cheap",
        *,
        input_tokens: int = 100,
        output_tokens: int = 200,
        reasoning_tokens: int = 0,
        provider_served: str = "DeepInfra",
        model_resolved: str = "test/cheap-0001",
        fail_every: int = 0,
    ) -> None:
        self.model = model
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.reasoning_tokens = reasoning_tokens
        self.provider_served = provider_served
        self.model_resolved = model_resolved
        self.fail_every = fail_every
        self.calls = 0
        self.max_in_flight = 0
        self._in_flight = 0

    async def generate_async(self, _request, _params) -> GenerationResult:
        self.calls += 1
        call = self.calls
        self._in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self._in_flight)
        try:
            # Yield control so concurrent callers really do overlap.
            await asyncio.sleep(0)
            if self.fail_every and call % self.fail_every == 0:
                raise RuntimeError("simulated provider error")
            return GenerationResult(
                text=f"Title: Doc {call}\n\nBody text for document {call}.",
                input_tokens=self.input_tokens,
                output_tokens=self.output_tokens,
                provider_served=self.provider_served,
                reasoning_tokens=self.reasoning_tokens,
                model_resolved=self.model_resolved,
            )
        finally:
            self._in_flight -= 1

    def generate(self, _request, _params) -> GenerationResult:
        raise AssertionError("sync generate should not be used by the async path")

    def generate_batch(self, _requests, _params) -> list[GenerationResult]:
        raise AssertionError("batch path not under test here")


class ExplodingBackend:
    """A backend whose construction is itself a test failure."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("a backend was constructed during a dry run")


# =============================================================================
# Token and cost estimation
# =============================================================================


class TestTokenEstimation:
    """The reservation must bound the request, not merely guess at it."""

    def test_worst_case_bounds_expected(self) -> None:
        estimate = gd.estimate_tokens(
            "prompt " * 100,
            "system " * 100,
            DocumentLength.MEDIUM,
            reasoning_on=False,
        )
        assert estimate.worst_input > estimate.expected_input
        assert estimate.worst_output >= estimate.expected_output

    def test_worst_output_matches_the_api_cap(self) -> None:
        # The reservation ceiling must equal the max_output_tokens actually sent,
        # or the cap could be overrun by a maximally verbose model.
        for length in DocumentLength:
            estimate = gd.estimate_tokens(
                "x" * 500, "y" * 500, length, reasoning_on=False
            )
            assert estimate.worst_output == gd.max_output_tokens_for(length)

    def test_reasoning_inflates_the_worst_case(self) -> None:
        off = gd.estimate_tokens(
            "x" * 500, "y" * 500, DocumentLength.SHORT, reasoning_on=False
        )
        on = gd.estimate_tokens(
            "x" * 500, "y" * 500, DocumentLength.SHORT, reasoning_on=True
        )
        assert on.worst_output > off.worst_output
        assert on.expected_output > off.expected_output

    def test_max_output_tokens_is_clamped(self) -> None:
        assert gd.max_output_tokens_for(DocumentLength.MICRO) == 256
        assert gd.max_output_tokens_for(DocumentLength.EXTENDED) == 8000


# =============================================================================
# Pricing
# =============================================================================


class TestPricing:
    """An unpriceable model is refused, not run blind."""

    def test_resolve_price_reads_the_catalogue(self, pricing: PricingTable) -> None:
        price = gd.resolve_price(pricing, "test/cheap", "openrouter")
        assert price.input_per_1m == pytest.approx(1.0)
        assert price.output_per_1m == pytest.approx(10.0)

    def test_unknown_model_is_fatal(self, pricing: PricingTable) -> None:
        with pytest.raises(SystemExit, match="Refusing to spend"):
            gd.resolve_price(pricing, "nobody/knows-me", "openrouter")

    def test_native_overrides_are_applied_by_default(self, monkeypatch) -> None:
        captured: dict[str, Any] = {}

        def fake_load(**kwargs: Any) -> str:
            captured.update(kwargs)
            return "table"

        monkeypatch.setattr(gd.PricingTable, "load", staticmethod(fake_load))
        gd.load_pricing_table()
        # The provider's own published price is the higher figure wherever it
        # disagrees with OpenRouter, which is the conservative choice for a cap.
        assert "openai/gpt-5.6-sol" in captured["overrides"]

        captured.clear()
        gd.load_pricing_table(native_prices=False)
        assert captured["overrides"] == {}


# =============================================================================
# CLI parsing
# =============================================================================


class TestCLI:
    """The CLI surface, including backward compatibility."""

    def test_legacy_invocation_still_parses(self) -> None:
        args = gd.build_parser().parse_args(["--count", "100", "--model", "gpt-5.1"])
        assert args.model == ["gpt-5.1"]
        assert args.count == 100
        assert args.concurrency == 20
        assert args.prompt_variant == PromptVariant.V0_3_1.value

    def test_budget_is_never_unbounded(self) -> None:
        args = gd.build_parser().parse_args([])
        assert args.budget_per_model == gd.DEFAULT_BUDGET_PER_MODEL_USD
        assert args.budget_per_model > 0

    def test_reasoning_defaults_to_off(self) -> None:
        args = gd.build_parser().parse_args([])
        assert args.reasoning_effort == "none"
        config = gd.build_reasoning_config(args.reasoning_effort)
        assert isinstance(config, ReasoningConfig)
        assert config.is_off

    def test_provider_default_sends_no_reasoning_parameter(self) -> None:
        assert gd.build_reasoning_config("default") is None

    def test_reasoning_effort_passthrough(self) -> None:
        config = gd.build_reasoning_config("low")
        assert config is not None and not config.is_off and config.effort == "low"

    def test_new_providers_are_accepted(self) -> None:
        for provider in ("openrouter", "xai"):
            args = gd.build_parser().parse_args(["--provider", provider])
            assert args.provider == provider

    def test_negative_budget_is_rejected(self) -> None:
        with pytest.raises(SystemExit):
            gd.main(["--budget-per-model", "-1"])

    def test_manual_prices_must_come_in_pairs(self) -> None:
        with pytest.raises(SystemExit, match="must be given together"):
            gd.main(["--price-input-per-1m", "1.0"])

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("gpt-5.1", "openai"),
            ("claude-opus-4-5-20250929", "anthropic"),
            ("gemini-3-pro", "gemini"),
            ("grok-4.6", "xai"),
            ("meta-llama/llama-4-maverick", "openrouter"),
        ],
    )
    def test_provider_inference(self, model: str, expected: str) -> None:
        assert gd.infer_provider(model) == expected


class TestPromptVariantWeights:
    """``--prompt-variant-weights`` parsing."""

    def test_none_means_none(self) -> None:
        assert gd.parse_variant_weights(None) is None
        assert gd.parse_variant_weights("") is None

    def test_weights_are_normalized(self) -> None:
        weights = gd.parse_variant_weights("v0.4-direct=1,v0.4-archival=3")
        assert weights is not None
        assert sum(weights.values()) == pytest.approx(1.0)
        assert weights[PromptVariant.ARCHIVAL] == pytest.approx(0.75)

    def test_default_keyword(self) -> None:
        weights = gd.parse_variant_weights("default")
        assert weights is not None
        assert set(weights) == {
            PromptVariant.DIRECT,
            PromptVariant.PRACTITIONER,
            PromptVariant.EDITORIAL,
            PromptVariant.ARCHIVAL,
        }

    def test_unknown_variant_is_fatal(self) -> None:
        with pytest.raises(SystemExit, match="Unknown prompt variant"):
            gd.parse_variant_weights("nope=1")


# =============================================================================
# Spec sourcing
# =============================================================================


class TestSpecSourcing:
    """Spec blocks are what make one spec runnable across many generators."""

    def test_load_from_spec_block_file(self, tmp_path: Path) -> None:
        path = save_spec_block(make_block(5), tmp_path / "block.jsonl")
        args = gd.build_parser().parse_args(["--spec-block", str(path)])
        blocks = gd.resolve_spec_blocks(args, "run-x")
        assert len(blocks) == 1
        assert [spec.spec_id for spec in blocks[0]] == list(make_block(5).spec_ids)

    def test_draw_and_save_blocks(self, tmp_path: Path) -> None:
        args = gd.build_parser().parse_args(
            [
                "--draw-spec-blocks",
                "2",
                "--specs-per-block",
                "4",
                "--spec-block-dir",
                str(tmp_path),
                "--seed",
                "7",
            ]
        )
        blocks = gd.resolve_spec_blocks(args, "run-x")
        assert len(blocks) == 2
        assert all(len(block) == 4 for block in blocks)
        saved = sorted(tmp_path.glob("*.jsonl"))
        assert len(saved) == 2
        # Round-trips, ids and all.
        assert gd.load_spec_block(saved[0]).spec_ids == blocks[0].spec_ids

    def test_blocks_drawn_with_different_seeds_differ(self) -> None:
        args = gd.build_parser().parse_args(
            ["--draw-spec-blocks", "2", "--specs-per-block", "8", "--seed", "11"]
        )
        blocks = gd.resolve_spec_blocks(args, "run-x")
        assert set(blocks[0].spec_ids) != set(blocks[1].spec_ids)

    def test_adhoc_specs_still_carry_provenance(self) -> None:
        args = gd.build_parser().parse_args(["--count", "3", "--seed", "5"])
        blocks = gd.resolve_spec_blocks(args, "run-abc")
        assert len(blocks) == 1
        assert blocks[0].block_id == "adhoc-run-abc"
        assert all(spec.spec_id and spec.block_id for spec in blocks[0])

    def test_prompt_variant_is_baked_into_spec_identity(self) -> None:
        plain = gd.build_parser().parse_args(
            ["--draw-spec-blocks", "1", "--specs-per-block", "4", "--seed", "3"]
        )
        stamped = gd.build_parser().parse_args(
            [
                "--draw-spec-blocks",
                "1",
                "--specs-per-block",
                "4",
                "--seed",
                "3",
                "--prompt-variant",
                "v0.4-direct",
            ]
        )
        plain_block = gd.resolve_spec_blocks(plain, "r")[0]
        stamped_block = gd.resolve_spec_blocks(stamped, "r")[0]
        assert all(
            spec.prompt_variant is PromptVariant.DIRECT for spec in stamped_block
        )
        # Same content, different prompt -> a genuinely different spec.
        assert set(plain_block.spec_ids).isdisjoint(stamped_block.spec_ids)

    def test_flatten_deduplicates_and_truncates(self) -> None:
        block = make_block(4)
        specs = gd.flatten_specs([block, block], limit=None)
        assert len(specs) == 4
        assert len(gd.flatten_specs([block], limit=2)) == 2


# =============================================================================
# Model planning
# =============================================================================


_OPEN_LEDGERS: list[CostLedger] = []


@pytest.fixture(autouse=True)
def _close_ledgers():
    """Close every ledger a test opened, so no file handle is left dangling."""
    yield
    while _OPEN_LEDGERS:
        _OPEN_LEDGERS.pop().close()


def build_guard(tmp_path: Path, cap: float | None = 10.0) -> BudgetGuard:
    """A guard over a throwaway ledger."""
    ledger = CostLedger(tmp_path / "ledger.jsonl", run_id="run-test", fsync=False)
    _OPEN_LEDGERS.append(ledger)
    return BudgetGuard(ledger, per_model_cap=cap)


class TestModelPlan:
    """Planning prices the queue and honours prior spend."""

    def test_plan_prices_the_queue(self, tmp_path: Path, pricing: PricingTable) -> None:
        guard = build_guard(tmp_path)
        plan = gd.build_model_plan(
            "test/cheap",
            "openrouter",
            pricing.price_of("test/cheap"),
            list(make_block(10)),
            guard=guard,
            fixed_variant=PromptVariant.V0_3_1,
            reasoning_on=False,
            resume=True,
        )
        assert len(plan.specs) == 10
        assert plan.expected_cost > 0
        assert plan.worst_case_cost > plan.expected_cost
        assert plan.cap == 10.0

    def test_plan_skips_completed_specs(
        self, tmp_path: Path, pricing: PricingTable
    ) -> None:
        guard = build_guard(tmp_path)
        specs = list(make_block(10))
        for spec in specs[:4]:
            guard.record(
                "test/cheap", cost_usd=0.001, spec_id=spec.spec_id, status="ok"
            )
        plan = gd.build_model_plan(
            "test/cheap",
            "openrouter",
            pricing.price_of("test/cheap"),
            specs,
            guard=guard,
            fixed_variant=PromptVariant.V0_3_1,
            reasoning_on=False,
            resume=True,
        )
        assert len(plan.specs) == 6
        assert plan.already_done == 4

    def test_no_resume_replans_everything(
        self, tmp_path: Path, pricing: PricingTable
    ) -> None:
        guard = build_guard(tmp_path)
        specs = list(make_block(10))
        guard.record("test/cheap", cost_usd=0.001, spec_id=specs[0].spec_id)
        plan = gd.build_model_plan(
            "test/cheap",
            "openrouter",
            pricing.price_of("test/cheap"),
            specs,
            guard=guard,
            fixed_variant=PromptVariant.V0_3_1,
            reasoning_on=False,
            resume=False,
        )
        assert len(plan.specs) == 10

    def test_within_budget_accounts_for_prior_spend(
        self, tmp_path: Path, pricing: PricingTable
    ) -> None:
        guard = build_guard(tmp_path, cap=1.0)
        guard.record("test/pricey", cost_usd=0.99, spec_id="prior")
        plan = gd.build_model_plan(
            "test/pricey",
            "openrouter",
            pricing.price_of("test/pricey"),
            list(make_block(50)),
            guard=guard,
            fixed_variant=PromptVariant.V0_3_1,
            reasoning_on=False,
            resume=True,
        )
        assert plan.remaining == pytest.approx(0.01)
        assert not plan.within_budget


# =============================================================================
# Generation
# =============================================================================


def run_model(
    plan: gd.ModelPlan,
    args,
    guard: BudgetGuard,
    tmp_path: Path,
    backend,
) -> tuple[gd.ModelOutcome, list[dict[str, Any]]]:
    """Drive ``generate_for_model`` with an in-process backend."""
    writer = gd.ArtifactWriter(
        tmp_path / "out.jsonl", tmp_path / "artifacts", log_usage=False
    )
    try:
        outcome = asyncio.run(
            gd.generate_for_model(
                plan,
                args=args,
                guard=guard,
                writer=writer,
                run_id="run-test",
                git_info={"commit": "abc123", "version_string": "abc123"},
                backend_factory=lambda *_a, **_k: backend,
            )
        )
    finally:
        writer.close()
    records = [
        json.loads(line)
        for line in (tmp_path / "out.jsonl").read_text().splitlines()
        if line.strip()
    ]
    return outcome, records


def make_plan(
    pricing: PricingTable, guard: BudgetGuard, n: int, model: str = "test/cheap"
) -> gd.ModelPlan:
    """A ready-to-run plan for ``n`` specs."""
    return gd.build_model_plan(
        model,
        "openrouter",
        pricing.price_of(model),
        list(make_block(n)),
        guard=guard,
        fixed_variant=PromptVariant.V0_3_1,
        reasoning_on=False,
        resume=True,
    )


class TestGeneration:
    """The generation loop and the provenance it records."""

    def test_happy_path_writes_artifacts(
        self, tmp_path: Path, pricing: PricingTable
    ) -> None:
        guard = build_guard(tmp_path)
        args = gd.build_parser().parse_args(["--concurrency", "4"])
        backend = RecordingBackend()
        outcome, records = run_model(
            make_plan(pricing, guard, 6), args, guard, tmp_path, backend
        )
        assert outcome.generated == 6
        assert outcome.failed == 0
        assert backend.calls == 6
        assert len(records) == 6
        assert len(list((tmp_path / "artifacts").glob("*.json"))) == 6

    def test_artifacts_carry_full_provenance(
        self, tmp_path: Path, pricing: PricingTable
    ) -> None:
        guard = build_guard(tmp_path)
        args = gd.build_parser().parse_args([])
        backend = RecordingBackend(reasoning_tokens=17)
        _, records = run_model(
            make_plan(pricing, guard, 3), args, guard, tmp_path, backend
        )
        for record in records:
            assert record["spec_id"]
            assert record["block_id"] == "block-test"
            assert record["prompt_variant_id"] == PromptVariant.V0_3_1.value
            assert record["provider_served"] == "DeepInfra"
            assert record["model_resolved"] == "test/cheap-0001"
            assert record["reasoning_tokens"] == 17
            assert record["model"] == "test/cheap"
            assert record["run_id"] == "run-test"
            assert record["cost_usd"] > 0
            # The v0.3.1 label fields are still present.
            assert record["lcc_code"] == "QA"
            assert record["register"] == Register.ACADEMIC.value

    def test_variant_recorded_from_the_spec(
        self, tmp_path: Path, pricing: PricingTable
    ) -> None:
        guard = build_guard(tmp_path)
        args = gd.build_parser().parse_args([])
        import dataclasses

        specs = [
            dataclasses.replace(spec, prompt_variant=PromptVariant.ARCHIVAL)
            for spec in make_block(2)
        ]
        plan = gd.build_model_plan(
            "test/cheap",
            "openrouter",
            pricing.price_of("test/cheap"),
            specs,
            guard=guard,
            fixed_variant=PromptVariant.V0_3_1,
            reasoning_on=False,
            resume=True,
        )
        _, records = run_model(plan, args, guard, tmp_path, RecordingBackend())
        assert {r["prompt_variant_id"] for r in records} == {
            PromptVariant.ARCHIVAL.value
        }

    def test_failures_do_not_kill_the_run(
        self, tmp_path: Path, pricing: PricingTable
    ) -> None:
        guard = build_guard(tmp_path)
        args = gd.build_parser().parse_args(["--concurrency", "1"])
        backend = RecordingBackend(fail_every=2)
        outcome, records = run_model(
            make_plan(pricing, guard, 6), args, guard, tmp_path, backend
        )
        assert outcome.failed == 3
        assert outcome.generated == 3
        assert len(records) == 3

    def test_failed_requests_are_not_charged(
        self, tmp_path: Path, pricing: PricingTable
    ) -> None:
        guard = build_guard(tmp_path)
        args = gd.build_parser().parse_args(["--concurrency", "1"])
        outcome, _ = run_model(
            make_plan(pricing, guard, 4),
            args,
            guard,
            tmp_path,
            RecordingBackend(fail_every=1),
        )
        assert outcome.generated == 0
        assert guard.committed("test/cheap") == pytest.approx(0.0)
        # But they are still visible in the ledger.
        errors = [r for r in guard.ledger.records() if r.status == "error"]
        assert len(errors) == 4

    def test_reservation_is_released_after_commit(
        self, tmp_path: Path, pricing: PricingTable
    ) -> None:
        guard = build_guard(tmp_path)
        args = gd.build_parser().parse_args(["--concurrency", "4"])
        run_model(
            make_plan(pricing, guard, 5), args, guard, tmp_path, RecordingBackend()
        )
        # No leaked reservations: spend equals committed.
        assert guard.spent("test/cheap") == pytest.approx(guard.committed("test/cheap"))

    def test_committed_cost_matches_actual_usage(
        self, tmp_path: Path, pricing: PricingTable
    ) -> None:
        guard = build_guard(tmp_path)
        args = gd.build_parser().parse_args([])
        backend = RecordingBackend(input_tokens=1000, output_tokens=2000)
        outcome, _ = run_model(
            make_plan(pricing, guard, 4), args, guard, tmp_path, backend
        )
        # 1000 in @ $1/1M + 2000 out @ $10/1M = $0.021 per document.
        assert outcome.cost_usd == pytest.approx(4 * 0.021)
        assert guard.committed("test/cheap") == pytest.approx(4 * 0.021)


# =============================================================================
# Dry run
# =============================================================================


class TestDryRun:
    """``--dry-run`` is the gate that gets used before real money moves."""

    def test_dry_run_makes_no_calls(
        self, tmp_path: Path, pricing: PricingTable, monkeypatch
    ) -> None:
        # Any attempt to reach the network is a test failure.
        import httpx

        def no_network(*_args: Any, **_kwargs: Any):
            raise AssertionError("dry run attempted a network request")

        monkeypatch.setattr(httpx.Client, "request", no_network)
        monkeypatch.setattr(httpx.Client, "send", no_network)
        monkeypatch.setattr(httpx.AsyncClient, "request", no_network)
        monkeypatch.setattr(httpx.AsyncClient, "send", no_network)

        args = gd.build_parser().parse_args(
            [
                "--dry-run",
                "--count",
                "25",
                "--model",
                "test/cheap",
                "test/pricey",
                "--budget-per-model",
                "50",
                "--ledger",
                str(tmp_path / "ledger.jsonl"),
                "--artifacts-dir",
                str(tmp_path / "artifacts"),
                "--output",
                str(tmp_path / "out.jsonl"),
            ]
        )
        code = asyncio.run(
            gd.run(args, pricing=pricing, backend_factory=ExplodingBackend)
        )
        assert code == 0
        # Nothing was generated and nothing was charged.
        assert not (tmp_path / "out.jsonl").exists()
        assert not list((tmp_path / "artifacts").glob("*.json"))
        with CostLedger(tmp_path / "ledger.jsonl") as ledger:
            assert ledger.total_spend() == 0.0

    def test_dry_run_writes_a_manifest_with_pricing_provenance(
        self, tmp_path: Path, pricing: PricingTable
    ) -> None:
        manifest_path = tmp_path / "manifest.json"
        args = gd.build_parser().parse_args(
            [
                "--dry-run",
                "--count",
                "5",
                "--model",
                "test/cheap",
                "--manifest",
                str(manifest_path),
                "--ledger",
                str(tmp_path / "ledger.jsonl"),
                "--artifacts-dir",
                str(tmp_path / "artifacts"),
            ]
        )
        asyncio.run(gd.run(args, pricing=pricing, backend_factory=ExplodingBackend))
        manifest = json.loads(manifest_path.read_text())
        assert manifest["dry_run"] is True
        assert manifest["pricing_fetched_at"] == "2026-08-26T00:00:00+00:00"
        assert manifest["pricing"]["fetched_at"] == "2026-08-26T00:00:00+00:00"
        assert manifest["plans"][0]["model"] == "test/cheap"
        assert manifest["plans"][0]["n_specs"] == 5
        assert manifest["spec_blocks"][0]["n_specs"] == 5

    def test_dry_run_signals_an_over_budget_plan(
        self, tmp_path: Path, pricing: PricingTable
    ) -> None:
        args = gd.build_parser().parse_args(
            [
                "--dry-run",
                "--count",
                "500",
                "--model",
                "test/pricey",
                "--budget-per-model",
                "0.01",
                "--ledger",
                str(tmp_path / "ledger.jsonl"),
                "--artifacts-dir",
                str(tmp_path / "artifacts"),
            ]
        )
        code = asyncio.run(
            gd.run(args, pricing=pricing, backend_factory=ExplodingBackend)
        )
        assert code == 2

    def test_draw_only_stops_before_the_ledger(
        self, tmp_path: Path, pricing: PricingTable
    ) -> None:
        args = gd.build_parser().parse_args(
            [
                "--draw-spec-blocks",
                "2",
                "--specs-per-block",
                "3",
                "--spec-block-dir",
                str(tmp_path / "blocks"),
                "--draw-only",
                "--seed",
                "9",
                "--ledger",
                str(tmp_path / "ledger.jsonl"),
            ]
        )
        code = asyncio.run(
            gd.run(args, pricing=pricing, backend_factory=ExplodingBackend)
        )
        assert code == 0
        assert len(list((tmp_path / "blocks").glob("*.jsonl"))) == 2
        assert not (tmp_path / "ledger.jsonl").exists()


# =============================================================================
# Backend construction
# =============================================================================


class TestBackendFactory:
    """``--provider openrouter`` / ``xai`` and the routing pin."""

    def test_openrouter_backend(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
        backend = gd._make_backend(
            "openrouter",
            "meta-llama/llama-4-maverick",
            None,
            reasoning=ReasoningConfig.off(),
            pin_provider="deepinfra",
        )
        assert backend.provider == "openrouter"
        assert backend.routing is not None
        assert backend.routing.to_dict() == {
            "order": ["deepinfra"],
            "allow_fallbacks": False,
        }

    def test_openrouter_reasoning_is_off_by_default_shape(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
        backend = gd._make_backend(
            "openrouter", "test/cheap", None, reasoning=ReasoningConfig.off()
        )
        payload: dict[str, Any] = {}
        backend._apply_reasoning(payload)
        assert payload["reasoning"] == {"enabled": False, "exclude": True}

    def test_xai_backend(self, monkeypatch) -> None:
        monkeypatch.setenv("XAI_API_KEY", "sk-test")
        backend = gd._make_backend(
            "xai", "grok-4.6", None, reasoning=ReasoningConfig.off()
        )
        assert backend.provider == "xai"
        payload: dict[str, Any] = {}
        backend._apply_reasoning(payload)
        assert payload["reasoning_effort"] == "none"

    def test_unknown_provider_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unsupported provider"):
            gd._make_backend("nope", "m", None)


# =============================================================================
# Batch path
# =============================================================================


class BatchBackend:
    """A batch backend that returns one result per request."""

    provider = "openrouter"

    def __init__(self, model: str = "test/cheap", *, output_tokens: int = 200):
        self.model = model
        self.output_tokens = output_tokens
        self.batches = 0
        self.requests_seen = 0

    def generate_batch(self, requests, _params) -> list[GenerationResult]:
        self.batches += 1
        self.requests_seen += len(requests)
        return [
            GenerationResult(
                text=f"Title: Batch {i}\n\nBatch body {i}.",
                input_tokens=100,
                output_tokens=self.output_tokens,
                provider_served="BatchProvider",
                reasoning_tokens=0,
                model_resolved=f"{self.model}-batch",
            )
            for i, _ in enumerate(requests)
        ]

    def generate(self, _request, _params) -> GenerationResult:
        raise AssertionError("sync path not used")

    async def generate_async(self, _request, _params) -> GenerationResult:
        raise AssertionError("async path not used in batch mode")


class TestBatchPath:
    """Batch submission is all-or-nothing, so it is checked before it is sent."""

    def test_batch_generates_and_books_every_spec(
        self, tmp_path: Path, pricing: PricingTable
    ) -> None:
        guard = build_guard(tmp_path, cap=10.0)
        args = gd.build_parser().parse_args(["--batch-size", "4"])
        backend = BatchBackend()
        plan = make_plan(pricing, guard, 8)
        outcome, records = run_model(plan, args, guard, tmp_path, backend)

        assert backend.batches == 2
        assert backend.requests_seen == 8
        assert outcome.generated == 8
        assert len(records) == 8
        # Every spec is individually recorded, which is what makes batch runs
        # resumable rather than all-or-nothing on restart.
        ok_ids = {r.spec_id for r in guard.ledger.records() if r.status == "ok"}
        assert ok_ids == {spec.spec_id for spec in plan.specs}
        assert {r["provider_served"] for r in records} == {"BatchProvider"}

    def test_batch_refuses_to_submit_over_the_cap(
        self, tmp_path: Path, pricing: PricingTable
    ) -> None:
        # A cap far below the batch's worst case: nothing may be submitted.
        guard = build_guard(tmp_path, cap=0.0001)
        args = gd.build_parser().parse_args(["--batch-size", "50"])
        backend = BatchBackend()
        outcome, records = run_model(
            make_plan(pricing, guard, 50), args, guard, tmp_path, backend
        )
        assert backend.batches == 0
        assert outcome.budget_aborted
        assert outcome.generated == 0
        assert outcome.skipped == 50
        assert records == []
        assert guard.committed("test/cheap") == pytest.approx(0.0)
