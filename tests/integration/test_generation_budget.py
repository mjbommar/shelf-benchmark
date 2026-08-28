"""End-to-end budget, resume, and dry-run safety for scripts/generate_documents.py.

These are the tests that stand between a typo and a five-figure API bill, so
they exercise the real generation loop (real ``asyncio``, real ``BudgetGuard``,
real ``CostLedger`` on disk) with only the HTTP call itself replaced.

Three properties are proven here:

1. **The cap holds under concurrency.** Twenty in-flight requests each reserve
   their worst case *before* the call goes out, so they cannot each pass an
   individual check against committed-only spend and then collectively overrun.
   The backend used here bills exactly its reservation, which is the adversarial
   case: if reservations were not held, the run would blow through the cap.
2. **Resume does not re-spend.** A second pass over the same ledger skips every
   ``spec_id`` already completed for that model, and the total on disk is the
   cost of the work done once.
3. **``--dry-run`` spends nothing.** No backend is constructed, no socket is
   opened, and no ledger file is even created.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from shelf.llm import BudgetGuard, CostLedger, GenerationResult
from shelf.llm.pricing import PricingTable
from shelf.sampler.generator import DocumentLength, PromptVariant, Register
from shelf.sampler.specs import DocumentSpec, SpecBlock, save_spec_block

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "generate_documents.py"
_spec = importlib.util.spec_from_file_location("generate_documents_it", _SCRIPT)
assert _spec is not None and _spec.loader is not None
gd = importlib.util.module_from_spec(_spec)
sys.modules["generate_documents_it"] = gd
_spec.loader.exec_module(gd)


# =============================================================================
# Fixtures
# =============================================================================

#: $1 / 1M input, $10 / 1M output -- round numbers so costs are hand-checkable.
CATALOGUE: list[dict[str, Any]] = [
    {
        "id": "test/model-a",
        "name": "Test Model A",
        "context_length": 128_000,
        "pricing": {"prompt": "0.000001", "completion": "0.00001"},
    },
    {
        "id": "test/model-b",
        "name": "Test Model B",
        "context_length": 128_000,
        "pricing": {"prompt": "0.000001", "completion": "0.00001"},
    },
]

#: ``BRIEF`` documents cap output at 256 tokens, so the worst-case reservation
#: for one request is dominated by 256 output tokens at $10/1M = $0.00256.
WORST_OUTPUT_TOKENS = 256


@pytest.fixture
def pricing() -> PricingTable:
    """Pricing built from a fixture payload; never touches the network."""
    return PricingTable.from_payload(CATALOGUE, fetched_at="2026-08-26T00:00:00+00:00")


def make_block(n: int, block_id: str = "block-it") -> SpecBlock:
    """A block of ``n`` distinct, deterministic specs."""
    return SpecBlock(
        block_id=block_id,
        seed=1,
        specs=tuple(
            DocumentSpec(
                lcc_code="QA",
                lcc_name="Mathematics",
                lcgft_form="Textbooks",
                lcgft_category="Instructional and educational works",
                topics=(f"Topic {i}",),
                target_length=DocumentLength.BRIEF,
                register=Register.ACADEMIC,
                block_id=block_id,
            )
            for i in range(n)
        ),
    )


class MaxCostBackend:
    """A backend that bills its full ``max_output_tokens`` on every call.

    This is the adversarial case for a budget guard: the actual cost equals the
    reserved worst case, so any headroom the guard hands out is money spent.
    """

    provider = "test"

    def __init__(
        self, model: str = "test/model-a", *, gate: asyncio.Event | None = None
    ):
        self.model = model
        self.calls = 0
        self.spec_prompts: list[str] = []
        self.max_in_flight = 0
        self._in_flight = 0
        self._gate = gate

    async def generate_async(self, request, params) -> GenerationResult:
        self.calls += 1
        call = self.calls
        self.spec_prompts.append(request.prompt)
        self._in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self._in_flight)
        try:
            if self._gate is not None:
                # Hold every request open until the test releases them, so the
                # guard really does face `concurrency` simultaneous reservations.
                await self._gate.wait()
            await asyncio.sleep(0)
            return GenerationResult(
                text=f"Title: Doc {call}\n\nBody for document {call}.",
                input_tokens=0,
                output_tokens=params.max_output_tokens,
                provider_served="TestProvider",
                reasoning_tokens=0,
                model_resolved=f"{self.model}-resolved",
            )
        finally:
            self._in_flight -= 1

    def generate(self, _request, _params) -> GenerationResult:
        raise AssertionError("sync path not used")

    def generate_batch(self, _requests, _params) -> list[GenerationResult]:
        raise AssertionError("batch path not used")


class ExplodingBackend:
    """Constructing this is itself a test failure."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("a backend was constructed during a dry run")


def run_plan(
    plan: gd.ModelPlan,
    args,
    guard: BudgetGuard,
    tmp_path: Path,
    backend,
    *,
    gate: asyncio.Event | None = None,
) -> gd.ModelOutcome:
    """Drive one model's queue through the real async loop."""
    writer = gd.ArtifactWriter(
        tmp_path / "out.jsonl", tmp_path / "artifacts", log_usage=False
    )

    async def drive() -> gd.ModelOutcome:
        if gate is not None:
            # Release the held requests once the loop has had a chance to fill.
            async def release() -> None:
                for _ in range(50):
                    await asyncio.sleep(0)
                gate.set()

            task = asyncio.create_task(release())
        outcome = await gd.generate_for_model(
            plan,
            args=args,
            guard=guard,
            writer=writer,
            run_id="run-it",
            git_info=None,
            backend_factory=lambda *_a, **_k: backend,
        )
        if gate is not None:
            await task
        return outcome

    try:
        return asyncio.run(drive())
    finally:
        writer.close()


def make_plan(
    pricing: PricingTable,
    guard: BudgetGuard,
    specs: list[DocumentSpec],
    model: str = "test/model-a",
) -> gd.ModelPlan:
    """Plan ``specs`` for ``model``, honouring what the ledger says is done."""
    return gd.build_model_plan(
        model,
        "openrouter",
        pricing.price_of(model),
        specs,
        guard=guard,
        fixed_variant=PromptVariant.V0_3_1,
        reasoning_on=False,
        resume=True,
    )


def worst_case_cost(pricing: PricingTable, spec: DocumentSpec) -> float:
    """The USD amount the guard will reserve for one request on ``spec``."""
    from shelf.sampler.generator import build_generation_prompt, build_system_prompt

    doc = spec.to_document()
    estimate = gd.estimate_tokens(
        build_generation_prompt(doc, spec.target_length, spec.register),
        build_system_prompt(doc, PromptVariant.V0_3_1),
        spec.target_length,
        reasoning_on=False,
    )
    return pricing.price_of("test/model-a").cost(
        estimate.worst_input, estimate.worst_output
    )


def open_guard(
    ledger_path: Path, cap: float | None, *, global_cap: float | None = None
) -> tuple[CostLedger, BudgetGuard]:
    """Open a ledger and a guard over it. The caller closes the ledger."""
    ledger = CostLedger(ledger_path, run_id="run-it", fsync=False)
    return ledger, BudgetGuard(ledger, per_model_cap=cap, global_cap=global_cap)


# =============================================================================
# 1. The cap holds under concurrency
# =============================================================================


class TestBudgetCapUnderConcurrency:
    """Twenty in-flight requests must not be able to overrun the cap."""

    def test_cap_is_not_overrun_by_twenty_concurrent_requests(
        self, tmp_path: Path, pricing: PricingTable
    ) -> None:
        # Room for roughly ten documents, against a queue of 200 and 20-way
        # concurrency. Every request in this test bills exactly what it
        # reserved, so if reservations were not held against the cap, all 20
        # in-flight requests would pass a committed-only check at $0 spent and
        # the run would overrun.
        spec = make_block(1).specs[0]
        worst_per_doc = worst_case_cost(pricing, spec)
        actual_per_doc = WORST_OUTPUT_TOKENS * 10 / 1_000_000
        cap = worst_per_doc * 10.5
        ledger, guard = open_guard(tmp_path / "ledger.jsonl", cap)
        args = gd.build_parser().parse_args(["--concurrency", "20"])
        gate = asyncio.Event()
        backend = MaxCostBackend(gate=gate)

        try:
            plan = make_plan(pricing, guard, list(make_block(200)))
            outcome = run_plan(plan, args, guard, tmp_path, backend, gate=gate)

            # The hard invariant: never more than the cap, ever.
            assert guard.committed("test/model-a") <= cap + 1e-9
            assert outcome.cost_usd <= cap + 1e-9
            # It stopped early rather than finishing the queue cheaply.
            assert outcome.budget_aborted
            assert outcome.generated < 200
            assert outcome.skipped > 0
            # Concurrency was genuinely in play, not serialized by accident.
            assert backend.max_in_flight > 1
            # The guard admitted fewer than the 20 the semaphore would allow...
            assert backend.calls < 20
            # ...and admitting all 20 would have overrun the cap, which is the
            # exact failure mode reservations exist to prevent.
            assert 20 * actual_per_doc > cap
        finally:
            ledger.close()

    def test_written_artifacts_match_what_was_charged(
        self, tmp_path: Path, pricing: PricingTable
    ) -> None:
        per_doc = WORST_OUTPUT_TOKENS * 10 / 1_000_000
        ledger, guard = open_guard(tmp_path / "ledger.jsonl", per_doc * 5.5)
        args = gd.build_parser().parse_args(["--concurrency", "20"])
        try:
            plan = make_plan(pricing, guard, list(make_block(100)))
            outcome = run_plan(plan, args, guard, tmp_path, MaxCostBackend())
            records = [
                json.loads(line)
                for line in (tmp_path / "out.jsonl").read_text().splitlines()
                if line.strip()
            ]
            assert len(records) == outcome.generated
            assert sum(r["cost_usd"] for r in records) == pytest.approx(
                outcome.cost_usd
            )
            # Every successful document is in the ledger exactly once.
            ok_ids = [r.spec_id for r in ledger.records() if r.status == "ok"]
            assert sorted(ok_ids) == sorted(r["spec_id"] for r in records)
        finally:
            ledger.close()

    def test_global_cap_also_holds(self, tmp_path: Path, pricing: PricingTable) -> None:
        per_doc = WORST_OUTPUT_TOKENS * 10 / 1_000_000
        global_cap = per_doc * 4.5
        ledger, guard = open_guard(
            tmp_path / "ledger.jsonl", cap=100.0, global_cap=global_cap
        )
        args = gd.build_parser().parse_args(["--concurrency", "20"])
        try:
            plan = make_plan(pricing, guard, list(make_block(100)))
            outcome = run_plan(plan, args, guard, tmp_path, MaxCostBackend())
            assert guard.total_spend() <= global_cap + 1e-9
            assert outcome.budget_aborted
        finally:
            ledger.close()

    def test_exhausted_model_does_not_stop_the_next_one(
        self, tmp_path: Path, pricing: PricingTable
    ) -> None:
        """A cap hit on one generator must not abort the whole Phase 1 run."""
        per_doc = WORST_OUTPUT_TOKENS * 10 / 1_000_000
        ledger_path = tmp_path / "ledger.jsonl"
        args = gd.build_parser().parse_args(
            [
                "--concurrency",
                "8",
                "--count",
                "40",
                "--model",
                "test/model-a",
                "test/model-b",
                "--budget-per-model",
                f"{per_doc * 3.5}",
                "--ledger",
                str(ledger_path),
                "--artifacts-dir",
                str(tmp_path / "artifacts"),
                "--output",
                str(tmp_path / "out.jsonl"),
                "--seed",
                "3",
                "--no-fsync",
            ]
        )
        backends = {
            "test/model-a": MaxCostBackend("test/model-a"),
            "test/model-b": MaxCostBackend("test/model-b"),
        }

        def factory(_provider, model, *_a, **_k):
            return backends[model]

        code = asyncio.run(gd.run(args, pricing=pricing, backend_factory=factory))
        # Exit code 3 signals "a model hit its cap", not a crash.
        assert code == 3
        # Both models were attempted despite the first one aborting.
        assert backends["test/model-a"].calls > 0
        assert backends["test/model-b"].calls > 0

        with CostLedger(ledger_path) as reader:
            spend = reader.spend_by_model()
        assert spend["test/model-a"] <= per_doc * 3.5 + 1e-9
        assert spend["test/model-b"] <= per_doc * 3.5 + 1e-9

    def test_a_prior_exhausted_ledger_blocks_the_run_entirely(
        self, tmp_path: Path, pricing: PricingTable
    ) -> None:
        ledger_path = tmp_path / "ledger.jsonl"
        with CostLedger(ledger_path, run_id="earlier", fsync=False) as earlier:
            earlier.record("test/model-a", cost_usd=1.0, spec_id="something-else")

        backend = MaxCostBackend()
        args = gd.build_parser().parse_args(
            [
                "--count",
                "10",
                "--model",
                "test/model-a",
                "--budget-per-model",
                "0.5",
                "--ledger",
                str(ledger_path),
                "--artifacts-dir",
                str(tmp_path / "artifacts"),
                "--output",
                str(tmp_path / "out.jsonl"),
                "--no-fsync",
            ]
        )
        code = asyncio.run(
            gd.run(args, pricing=pricing, backend_factory=lambda *_a, **_k: backend)
        )
        assert code == 3
        assert backend.calls == 0


# =============================================================================
# 2. Resume does not re-spend
# =============================================================================


class TestResumeDoesNotRespend:
    """A crashed run must resume, not restart."""

    def test_second_pass_skips_completed_specs(
        self, tmp_path: Path, pricing: PricingTable
    ) -> None:
        ledger_path = tmp_path / "ledger.jsonl"
        specs = list(make_block(10))
        args = gd.build_parser().parse_args(["--concurrency", "4"])

        # -- pass 1: the half that landed before the "crash" ---------------- #
        ledger, guard = open_guard(ledger_path, cap=10.0)
        first = MaxCostBackend()
        try:
            outcome1 = run_plan(
                make_plan(pricing, guard, specs[:5]), args, guard, tmp_path, first
            )
        finally:
            ledger.close()
        assert outcome1.generated == 5
        assert first.calls == 5
        done_after_first = {
            r.spec_id for r in CostLedger(ledger_path).records() if r.status == "ok"
        }
        assert done_after_first == {spec.spec_id for spec in specs[:5]}

        # -- pass 2: rerun the *whole* queue against the same ledger -------- #
        ledger2, guard2 = open_guard(ledger_path, cap=10.0)
        second = MaxCostBackend()
        try:
            plan2 = make_plan(pricing, guard2, specs)
            # Planning alone already knows five are done.
            assert plan2.already_done == 5
            assert len(plan2.specs) == 5
            outcome2 = run_plan(plan2, args, guard2, tmp_path, second)
        finally:
            ledger2.close()

        assert second.calls == 5
        assert outcome2.generated == 5

        with CostLedger(ledger_path) as reader:
            records = [r for r in reader.records() if r.status == "ok"]
        # Ten documents, ten records, no spec paid for twice.
        assert len(records) == 10
        assert len({r.spec_id for r in records}) == 10
        assert sum(r.cost_usd for r in records) == pytest.approx(
            outcome1.cost_usd + outcome2.cost_usd
        )

    def test_a_completed_queue_costs_nothing_to_rerun(
        self, tmp_path: Path, pricing: PricingTable
    ) -> None:
        ledger_path = tmp_path / "ledger.jsonl"
        specs = list(make_block(6))
        args = gd.build_parser().parse_args(["--concurrency", "3"])

        ledger, guard = open_guard(ledger_path, cap=10.0)
        try:
            run_plan(
                make_plan(pricing, guard, specs),
                args,
                guard,
                tmp_path,
                MaxCostBackend(),
            )
        finally:
            ledger.close()
        with CostLedger(ledger_path) as reader:
            spend_after_first = reader.total_spend()

        # Re-running the identical command is a no-op.
        backend = MaxCostBackend()
        args2 = gd.build_parser().parse_args(
            [
                "--model",
                "test/model-a",
                "--budget-per-model",
                "10",
                "--ledger",
                str(ledger_path),
                "--artifacts-dir",
                str(tmp_path / "artifacts"),
                "--output",
                str(tmp_path / "out.jsonl"),
                "--spec-block",
                str(save_spec_block(make_block(6), tmp_path / "b.jsonl")),
                "--no-fsync",
            ]
        )
        code = asyncio.run(
            gd.run(args2, pricing=pricing, backend_factory=lambda *_a, **_k: backend)
        )
        assert code == 0
        assert backend.calls == 0
        with CostLedger(ledger_path) as reader:
            assert reader.total_spend() == pytest.approx(spend_after_first)

    def test_resume_is_per_model(self, tmp_path: Path, pricing: PricingTable) -> None:
        """Model B must still generate specs model A already covered."""
        ledger_path = tmp_path / "ledger.jsonl"
        specs = list(make_block(4))
        args = gd.build_parser().parse_args(["--concurrency", "2"])

        ledger, guard = open_guard(ledger_path, cap=10.0)
        try:
            run_plan(
                make_plan(pricing, guard, specs, "test/model-a"),
                args,
                guard,
                tmp_path,
                MaxCostBackend("test/model-a"),
            )
        finally:
            ledger.close()

        ledger2, guard2 = open_guard(ledger_path, cap=10.0)
        try:
            plan_b = make_plan(pricing, guard2, specs, "test/model-b")
            assert plan_b.already_done == 0
            assert len(plan_b.specs) == 4
        finally:
            ledger2.close()

    def test_no_resume_re_spends(self, tmp_path: Path, pricing: PricingTable) -> None:
        """The opt-out exists, and it is the only way to pay twice."""
        ledger_path = tmp_path / "ledger.jsonl"
        specs = list(make_block(4))
        ledger, guard = open_guard(ledger_path, cap=10.0)
        args = gd.build_parser().parse_args(["--concurrency", "2"])
        try:
            run_plan(
                make_plan(pricing, guard, specs),
                args,
                guard,
                tmp_path,
                MaxCostBackend(),
            )
        finally:
            ledger.close()

        ledger2, guard2 = open_guard(ledger_path, cap=10.0)
        try:
            plan = gd.build_model_plan(
                "test/model-a",
                "openrouter",
                pricing.price_of("test/model-a"),
                specs,
                guard=guard2,
                fixed_variant=PromptVariant.V0_3_1,
                reasoning_on=False,
                resume=False,
            )
            assert len(plan.specs) == 4
        finally:
            ledger2.close()

    def test_failed_specs_are_retried_on_resume(
        self, tmp_path: Path, pricing: PricingTable
    ) -> None:
        """Only ``status="ok"`` counts as done, so a failure is retried."""
        ledger_path = tmp_path / "ledger.jsonl"
        specs = list(make_block(3))
        with CostLedger(ledger_path, run_id="earlier", fsync=False) as earlier:
            for spec in specs:
                earlier.record(
                    "test/model-a", cost_usd=0.0, spec_id=spec.spec_id, status="error"
                )

        ledger, guard = open_guard(ledger_path, cap=10.0)
        try:
            plan = make_plan(pricing, guard, specs)
            assert len(plan.specs) == 3
            assert plan.already_done == 0
        finally:
            ledger.close()


# =============================================================================
# 3. --dry-run spends nothing
# =============================================================================


class TestDryRunSpendsNothing:
    """The safety gate: price it, print it, touch nothing."""

    def test_dry_run_opens_no_socket_and_builds_no_backend(
        self, tmp_path: Path, pricing: PricingTable, monkeypatch
    ) -> None:
        import httpx

        def no_network(*_args: Any, **_kwargs: Any):
            raise AssertionError("dry run attempted a network request")

        for client in (httpx.Client, httpx.AsyncClient):
            monkeypatch.setattr(client, "request", no_network)
            monkeypatch.setattr(client, "send", no_network)

        block_path = save_spec_block(make_block(120), tmp_path / "block.jsonl")
        args = gd.build_parser().parse_args(
            [
                "--dry-run",
                "--spec-block",
                str(block_path),
                "--model",
                "test/model-a",
                "test/model-b",
                "--budget-per-model",
                "10",
                "--concurrency",
                "20",
                "--ledger",
                str(tmp_path / "ledger.jsonl"),
                "--artifacts-dir",
                str(tmp_path / "artifacts"),
                "--output",
                str(tmp_path / "out.jsonl"),
                "--manifest",
                str(tmp_path / "manifest.json"),
            ]
        )
        code = asyncio.run(
            gd.run(args, pricing=pricing, backend_factory=ExplodingBackend)
        )

        assert code == 0
        assert not (tmp_path / "out.jsonl").exists()
        assert not (tmp_path / "artifacts").exists() or not list(
            (tmp_path / "artifacts").glob("*.json")
        )
        # The ledger is never even created by a dry run.
        assert not (tmp_path / "ledger.jsonl").exists()

        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert manifest["dry_run"] is True
        assert [plan["n_specs"] for plan in manifest["plans"]] == [120, 120]
        assert all(plan["expected_cost"] > 0 for plan in manifest["plans"])
        assert manifest["pricing_fetched_at"] == "2026-08-26T00:00:00+00:00"

    def test_dry_run_projection_matches_the_priced_plan(
        self, tmp_path: Path, pricing: PricingTable
    ) -> None:
        """The projection is a real per-spec pricing pass, not a flat multiple."""
        block_path = save_spec_block(make_block(50), tmp_path / "block.jsonl")
        args = gd.build_parser().parse_args(
            [
                "--dry-run",
                "--spec-block",
                str(block_path),
                "--model",
                "test/model-a",
                "--budget-per-model",
                "10",
                "--ledger",
                str(tmp_path / "ledger.jsonl"),
                "--artifacts-dir",
                str(tmp_path / "artifacts"),
                "--manifest",
                str(tmp_path / "manifest.json"),
            ]
        )
        asyncio.run(gd.run(args, pricing=pricing, backend_factory=ExplodingBackend))
        plan = json.loads((tmp_path / "manifest.json").read_text())["plans"][0]

        # Worst case is 50 x 256 output tokens at $10/1M, plus input.
        floor = 50 * WORST_OUTPUT_TOKENS * 10 / 1_000_000
        assert plan["worst_case_cost"] > floor
        assert plan["expected_cost"] < plan["worst_case_cost"]

    def test_dry_run_accounts_for_work_already_done(
        self, tmp_path: Path, pricing: PricingTable
    ) -> None:
        ledger_path = tmp_path / "ledger.jsonl"
        specs = list(make_block(20))
        with CostLedger(ledger_path, run_id="earlier", fsync=False) as earlier:
            for spec in specs[:8]:
                earlier.record(
                    "test/model-a", cost_usd=0.001, spec_id=spec.spec_id, status="ok"
                )

        block_path = save_spec_block(make_block(20), tmp_path / "block.jsonl")
        args = gd.build_parser().parse_args(
            [
                "--dry-run",
                "--spec-block",
                str(block_path),
                "--model",
                "test/model-a",
                "--budget-per-model",
                "10",
                "--ledger",
                str(ledger_path),
                "--artifacts-dir",
                str(tmp_path / "artifacts"),
                "--manifest",
                str(tmp_path / "manifest.json"),
            ]
        )
        asyncio.run(gd.run(args, pricing=pricing, backend_factory=ExplodingBackend))
        plan = json.loads((tmp_path / "manifest.json").read_text())["plans"][0]
        assert plan["n_specs"] == 12
        assert plan["already_completed"] == 8
        assert plan["committed_before_run"] == pytest.approx(0.008)


# =============================================================================
# 4. Phase 1 shape: one spec block, many generators
# =============================================================================


class TestSpecBlockAcrossGenerators:
    """The same spec block handed to several models is what Phase 1 buys."""

    def test_same_specs_realized_by_two_models(
        self, tmp_path: Path, pricing: PricingTable
    ) -> None:
        block_path = save_spec_block(make_block(6), tmp_path / "block.jsonl")
        backends = {
            "test/model-a": MaxCostBackend("test/model-a"),
            "test/model-b": MaxCostBackend("test/model-b"),
        }
        args = gd.build_parser().parse_args(
            [
                "--spec-block",
                str(block_path),
                "--model",
                "test/model-a",
                "test/model-b",
                "--budget-per-model",
                "10",
                "--concurrency",
                "3",
                "--ledger",
                str(tmp_path / "ledger.jsonl"),
                "--artifacts-dir",
                str(tmp_path / "artifacts"),
                "--output",
                str(tmp_path / "out.jsonl"),
                "--no-fsync",
            ]
        )
        code = asyncio.run(
            gd.run(
                args,
                pricing=pricing,
                backend_factory=lambda _provider, model, *_a, **_k: backends[model],
            )
        )
        assert code == 0

        records = [
            json.loads(line)
            for line in (tmp_path / "out.jsonl").read_text().splitlines()
            if line.strip()
        ]
        assert len(records) == 12
        by_model: dict[str, set[str]] = {}
        for record in records:
            by_model.setdefault(record["model"], set()).add(record["spec_id"])
        # Both generators realized the identical six specs -- the paired design.
        assert by_model["test/model-a"] == by_model["test/model-b"]
        assert len(by_model["test/model-a"]) == 6
        assert {r["block_id"] for r in records} == {"block-it"}
        assert {r["provider_served"] for r in records} == {"TestProvider"}
