"""Unit tests for shelf.llm.ledger.

Covers the two properties the v0.4 budget depends on: a model cannot spend past
its cap, and a resumed run does not re-spend what a crashed run already spent.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
from shelf.llm.ledger import (
    STATUS_ERROR,
    STATUS_OK,
    BudgetError,
    BudgetExceeded,
    BudgetGuard,
    CostLedger,
    LedgerRecord,
    new_run_id,
    read_records,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def ledger_path(tmp_path: Path) -> Path:
    """Path to a ledger inside a nested (not yet created) directory."""
    return tmp_path / "artifacts" / "cost_ledger.jsonl"


@pytest.fixture
def ledger(ledger_path: Path) -> Iterator[CostLedger]:
    """A ledger with fsync disabled for test speed."""
    with CostLedger(ledger_path, run_id="run-a", fsync=False) as instance:
        yield instance


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #


class TestLedgerRecord:
    """Serialization round-trips."""

    def test_round_trip(self):
        record = LedgerRecord(
            timestamp="2026-08-26T12:00:00.000+00:00",
            run_id="run-a",
            model="anthropic/claude-opus-5:batch",
            input_tokens=600,
            output_tokens=2400,
            reasoning_tokens=1800,
            cost_usd=0.0315,
            provider_served="anthropic",
            spec_id="spec-0001",
        )
        restored = LedgerRecord.from_dict(json.loads(record.to_json()))
        assert restored == record

    def test_total_tokens(self):
        record = LedgerRecord("t", "r", "m", input_tokens=10, output_tokens=5)
        assert record.total_tokens == 15

    def test_from_dict_tolerates_missing_and_extra_keys(self):
        record = LedgerRecord.from_dict(
            {"model": "m", "cost_usd": 1.5, "unknown_field": "ignored"}
        )
        assert record.model == "m"
        assert record.cost_usd == 1.5
        assert record.input_tokens == 0
        assert record.status == STATUS_OK

    def test_new_run_id_is_unique(self):
        assert new_run_id() != new_run_id()


# --------------------------------------------------------------------------- #
# Ledger I/O
# --------------------------------------------------------------------------- #


class TestCostLedger:
    """Append-only JSONL behaviour."""

    def test_creates_parent_directories(self, ledger: CostLedger, ledger_path: Path):
        ledger.record("m", cost_usd=1.0)
        assert ledger_path.exists()

    def test_one_line_per_request(self, ledger: CostLedger, ledger_path: Path):
        for i in range(5):
            ledger.record("m", cost_usd=0.1, spec_id=f"s{i}")
        assert len(ledger_path.read_text().strip().splitlines()) == 5

    def test_append_does_not_truncate_existing_file(self, ledger_path: Path):
        with CostLedger(ledger_path, run_id="run-a", fsync=False) as first:
            first.record("m", cost_usd=1.0)
        with CostLedger(ledger_path, run_id="run-b", fsync=False) as second:
            second.record("m", cost_usd=2.0)
            assert second.total_spend() == pytest.approx(3.0)

    def test_extra_fields_are_preserved(self, ledger: CostLedger):
        ledger.record("m", cost_usd=0.1, spec_block_id="block-2")
        (record,) = ledger.records()
        assert record.extra == {"spec_block_id": "block-2"}

    def test_spend_by_model(self, ledger: CostLedger):
        ledger.record("a", cost_usd=1.0)
        ledger.record("a", cost_usd=2.0)
        ledger.record("b", cost_usd=4.0)
        assert ledger.spend_by_model() == {
            "a": pytest.approx(3.0),
            "b": pytest.approx(4.0),
        }

    def test_spend_by_run(self, ledger: CostLedger):
        ledger.record("a", cost_usd=1.0)
        ledger.record("a", cost_usd=2.0, run_id="run-b")
        assert ledger.spend_by_run() == {
            "run-a": pytest.approx(1.0),
            "run-b": pytest.approx(2.0),
        }

    def test_request_counts(self, ledger: CostLedger):
        ledger.record("a", cost_usd=1.0)
        ledger.record("a", cost_usd=1.0)
        assert ledger.request_counts() == {"a": 2}

    def test_completed_spec_ids_only_successful(self, ledger: CostLedger):
        ledger.record("a", cost_usd=1.0, spec_id="s1")
        ledger.record("a", cost_usd=0.0, spec_id="s2", status=STATUS_ERROR)
        assert ledger.completed_spec_ids() == {"s1"}

    def test_completed_spec_ids_filtered_by_model(self, ledger: CostLedger):
        ledger.record("a", cost_usd=1.0, spec_id="s1")
        ledger.record("b", cost_usd=1.0, spec_id="s2")
        assert ledger.completed_spec_ids("b") == {"s2"}

    def test_missing_file_reads_as_empty(self, tmp_path: Path):
        assert list(read_records(tmp_path / "nope.jsonl")) == []

    def test_torn_final_line_is_skipped(self, ledger_path: Path):
        """A crash mid-write leaves a partial line; the rest must stay readable."""
        ledger_path.parent.mkdir(parents=True)
        good = LedgerRecord("t", "r", "m", cost_usd=2.0).to_json()
        ledger_path.write_text(good + "\n" + good[: len(good) // 2])
        records = list(read_records(ledger_path))
        assert len(records) == 1
        assert records[0].cost_usd == pytest.approx(2.0)

    def test_blank_lines_skipped(self, ledger_path: Path):
        ledger_path.parent.mkdir(parents=True)
        good = LedgerRecord("t", "r", "m", cost_usd=1.0).to_json()
        ledger_path.write_text(f"\n{good}\n\n")
        assert len(list(read_records(ledger_path))) == 1

    def test_writes_after_close_are_rejected(self, ledger_path: Path):
        instance = CostLedger(ledger_path, fsync=False)
        instance.record("m", cost_usd=1.0)
        instance.close()
        with pytest.raises(ValueError):
            instance.record("m", cost_usd=1.0)

    def test_concurrent_appends_are_not_interleaved(self, ledger_path: Path):
        """20 threads (the generator's concurrency) must produce 20 valid lines."""
        with CostLedger(ledger_path, fsync=False) as instance:

            def worker(index: int) -> None:
                for _ in range(25):
                    instance.record("m", cost_usd=0.01, spec_id=f"s{index}")

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        lines = ledger_path.read_text().strip().splitlines()
        assert len(lines) == 500
        for line in lines:
            json.loads(line)  # every line is complete, valid JSON


# --------------------------------------------------------------------------- #
# Budget guard
# --------------------------------------------------------------------------- #


class TestBudgetGuardCaps:
    """The hard cap."""

    def test_cap_blocks_at_the_cap(self, ledger: CostLedger):
        """The headline requirement: spend reaching the cap blocks further requests."""
        guard = BudgetGuard(ledger, per_model_cap=1.0)
        for _ in range(10):
            with guard.reserve("m", estimated_cost=0.1) as request:
                request.commit(cost_usd=0.1)

        assert guard.spent("m") == pytest.approx(1.0)
        assert guard.is_exhausted("m") is True
        with (
            pytest.raises(BudgetExceeded) as excinfo,
            guard.reserve("m", estimated_cost=0.1),
        ):
            pytest.fail("reservation should not have been granted")
        assert excinfo.value.model == "m"
        assert excinfo.value.cap == pytest.approx(1.0)
        assert excinfo.value.scope == "model"

    def test_cap_blocks_before_overrun(self, ledger: CostLedger):
        guard = BudgetGuard(ledger, per_model_cap=1.0)
        with guard.reserve("m", estimated_cost=0.9) as request:
            request.commit(cost_usd=0.9)
        with pytest.raises(BudgetExceeded), guard.reserve("m", estimated_cost=0.2):
            pytest.fail("unreachable")
        # Nothing was written for the refused request.
        assert ledger.total_spend() == pytest.approx(0.9)

    def test_other_models_unaffected(self, ledger: CostLedger):
        guard = BudgetGuard(ledger, per_model_cap=1.0)
        with guard.reserve("a", estimated_cost=1.0) as request:
            request.commit(cost_usd=1.0)
        assert guard.is_exhausted("a") is True
        with guard.reserve("b", estimated_cost=0.5) as request:
            request.commit(cost_usd=0.5)
        assert guard.spent("b") == pytest.approx(0.5)

    def test_per_model_mapping_and_default(self, ledger: CostLedger):
        guard = BudgetGuard(ledger, per_model_cap={"expensive": 50.0}, default_cap=5.0)
        assert guard.cap_for("expensive") == pytest.approx(50.0)
        assert guard.cap_for("other") == pytest.approx(5.0)

    def test_uncapped_model_never_blocks(self, ledger: CostLedger):
        guard = BudgetGuard(ledger)
        with guard.reserve("m", estimated_cost=1000.0) as request:
            request.commit(cost_usd=1000.0)
        assert guard.remaining("m") == float("inf")
        assert guard.is_exhausted("m") is False

    def test_global_cap_enforced(self, ledger: CostLedger):
        guard = BudgetGuard(ledger, per_model_cap=100.0, global_cap=1.0)
        with guard.reserve("a", estimated_cost=0.6) as request:
            request.commit(cost_usd=0.6)
        with (
            pytest.raises(BudgetExceeded) as excinfo,
            guard.reserve("b", estimated_cost=0.6),
        ):
            pytest.fail("unreachable")
        assert excinfo.value.scope == "global"

    def test_set_cap_overrides(self, ledger: CostLedger):
        guard = BudgetGuard(ledger, per_model_cap=1.0)
        guard.set_cap("m", 10.0)
        with guard.reserve("m", estimated_cost=5.0) as request:
            request.commit(cost_usd=5.0)
        assert guard.remaining("m") == pytest.approx(5.0)

    def test_non_positive_cap_rejected(self, ledger: CostLedger):
        with pytest.raises(ValueError):
            BudgetGuard(ledger, per_model_cap=0.0)
        with pytest.raises(ValueError):
            BudgetGuard(ledger, global_cap=-1.0)

    def test_negative_estimate_rejected(self, ledger: CostLedger):
        guard = BudgetGuard(ledger, per_model_cap=1.0)
        with pytest.raises(ValueError):
            guard.check("m", -0.5)

    def test_check_does_not_raise_with_headroom(self, ledger: CostLedger):
        guard = BudgetGuard(ledger, per_model_cap=1.0)
        guard.check("m", 0.5)


class TestReservations:
    """Reserve / commit / release semantics."""

    def test_reservation_holds_budget_before_commit(self, ledger: CostLedger):
        guard = BudgetGuard(ledger, per_model_cap=1.0)
        with guard.reserve("m", estimated_cost=0.8) as request:
            # A second concurrent request cannot slip past the cap.
            with pytest.raises(BudgetExceeded), guard.reserve("m", estimated_cost=0.8):
                pytest.fail("unreachable")
            request.commit(cost_usd=0.1)
        # The unused reservation is released, leaving only the actual cost.
        assert guard.spent("m") == pytest.approx(0.1)

    def test_commit_writes_full_record(self, ledger: CostLedger):
        guard = BudgetGuard(ledger, per_model_cap=10.0)
        with guard.reserve("m", estimated_cost=0.5, spec_id="spec-7") as request:
            request.commit(
                input_tokens=600,
                output_tokens=2400,
                reasoning_tokens=1800,
                cost_usd=0.42,
                provider_served="anthropic",
                request_id="req-1",
                spec_block_id="block-2",
            )
        (record,) = ledger.records()
        assert record.spec_id == "spec-7"
        assert record.input_tokens == 600
        assert record.reasoning_tokens == 1800
        assert record.provider_served == "anthropic"
        assert record.request_id == "req-1"
        assert record.extra == {"spec_block_id": "block-2"}
        assert record.cost_usd == pytest.approx(0.42)

    def test_commit_defaults_to_the_estimate(self, ledger: CostLedger):
        guard = BudgetGuard(ledger, per_model_cap=10.0)
        with guard.reserve("m", estimated_cost=0.25) as request:
            request.commit()
        assert guard.committed("m") == pytest.approx(0.25)

    def test_exception_releases_reservation_and_charges_nothing(
        self, ledger: CostLedger
    ):
        guard = BudgetGuard(ledger, per_model_cap=1.0)
        with pytest.raises(RuntimeError), guard.reserve("m", estimated_cost=0.9):
            raise RuntimeError("api blew up")
        assert guard.spent("m") == 0.0
        assert ledger.records() == []

    def test_exception_can_record_a_failure_row(self, ledger: CostLedger):
        guard = BudgetGuard(ledger, per_model_cap=1.0)
        with (
            pytest.raises(RuntimeError),
            guard.reserve("m", 0.9, spec_id="s1", record_failure=True),
        ):
            raise RuntimeError("boom")
        (record,) = ledger.records()
        assert record.status == STATUS_ERROR
        assert record.cost_usd == 0.0
        assert guard.spent("m") == 0.0

    def test_double_commit_rejected(self, ledger: CostLedger):
        guard = BudgetGuard(ledger, per_model_cap=10.0)
        with guard.reserve("m", estimated_cost=0.1) as request:
            request.commit(cost_usd=0.1)
            with pytest.raises(BudgetError):
                request.commit(cost_usd=0.1)
        assert len(ledger.records()) == 1

    def test_settled_flag(self, ledger: CostLedger):
        guard = BudgetGuard(ledger, per_model_cap=10.0)
        with guard.reserve("m", estimated_cost=0.1) as request:
            assert request.settled is False
            request.commit(cost_usd=0.1)
            assert request.settled is True

    def test_direct_record_bypasses_reservation(self, ledger: CostLedger):
        guard = BudgetGuard(ledger, per_model_cap=10.0)
        guard.record("m", cost_usd=3.0, input_tokens=10, output_tokens=20)
        assert guard.committed("m") == pytest.approx(3.0)

    def test_direct_record_can_skip_the_check(self, ledger: CostLedger):
        """A batch job's final usage may legitimately overshoot; still book it."""
        guard = BudgetGuard(ledger, per_model_cap=1.0)
        guard.record("m", cost_usd=5.0, check=False)
        assert guard.committed("m") == pytest.approx(5.0)
        assert guard.is_exhausted("m") is True

    def test_direct_record_respects_the_check(self, ledger: CostLedger):
        guard = BudgetGuard(ledger, per_model_cap=1.0)
        with pytest.raises(BudgetExceeded):
            guard.record("m", cost_usd=5.0)


class TestResume:
    """Crash-and-restart must not double-spend."""

    def test_reload_recomputes_spend_from_disk(self, ledger_path: Path):
        with CostLedger(ledger_path, run_id="run-1", fsync=False) as first:
            guard = BudgetGuard(first, per_model_cap=1.0)
            for i in range(5):
                with guard.reserve("m", 0.1, spec_id=f"s{i}") as request:
                    request.commit(cost_usd=0.1)
            assert guard.spent("m") == pytest.approx(0.5)

        # "Crash", then restart with a brand-new run against the same file.
        with CostLedger(ledger_path, run_id="run-2", fsync=False) as second:
            resumed = BudgetGuard(second, per_model_cap=1.0)
            assert resumed.committed("m") == pytest.approx(0.5)
            assert resumed.remaining("m") == pytest.approx(0.5)

            # Only 5 more requests fit; the 6th is refused, not double-spent.
            for i in range(5, 10):
                with resumed.reserve("m", 0.1, spec_id=f"s{i}") as request:
                    request.commit(cost_usd=0.1)
            assert resumed.spent("m") == pytest.approx(1.0)
            with pytest.raises(BudgetExceeded), resumed.reserve("m", 0.1):
                pytest.fail("unreachable")

            # Total across both runs equals the cap exactly — nothing counted twice.
            assert second.total_spend() == pytest.approx(1.0)
            assert len(second.records()) == 10
            assert second.completed_spec_ids("m") == {f"s{i}" for i in range(10)}

    def test_resume_sees_records_from_a_foreign_run(self, ledger_path: Path):
        ledger_path.parent.mkdir(parents=True)
        ledger_path.write_text(
            LedgerRecord("t", "earlier-run", "m", cost_usd=40.0).to_json() + "\n"
        )
        with CostLedger(ledger_path, run_id="new", fsync=False) as instance:
            guard = BudgetGuard(instance, per_model_cap=50.0)
            assert guard.remaining("m") == pytest.approx(10.0)

    def test_explicit_reload_picks_up_external_writes(self, ledger_path: Path):
        ledger_path.parent.mkdir(parents=True)
        with CostLedger(ledger_path, run_id="a", fsync=False) as instance:
            guard = BudgetGuard(instance, per_model_cap=10.0)
            assert guard.committed("m") == 0.0
            with ledger_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    LedgerRecord("t", "other", "m", cost_usd=4.0).to_json() + "\n"
                )
            guard.reload()
            assert guard.committed("m") == pytest.approx(4.0)


class TestConcurrency:
    """The guard is shared across the generator's 20 concurrent workers."""

    def test_cap_holds_under_thread_contention(self, ledger: CostLedger):
        guard = BudgetGuard(ledger, per_model_cap=1.0)
        granted: list[int] = []
        refused: list[int] = []
        lock = threading.Lock()

        def worker() -> None:
            for _ in range(20):
                try:
                    with guard.reserve("m", estimated_cost=0.01) as request:
                        request.commit(cost_usd=0.01)
                    with lock:
                        granted.append(1)
                except BudgetExceeded:
                    with lock:
                        refused.append(1)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(granted) == 100  # $1.00 / $0.01
        assert len(refused) == 300
        assert guard.committed("m") == pytest.approx(1.0)
        assert ledger.total_spend() == pytest.approx(1.0)

    def test_reservations_are_released_exactly_once(self, ledger: CostLedger):
        guard = BudgetGuard(ledger, per_model_cap=100.0)

        def worker() -> None:
            for _ in range(50):
                with guard.reserve("m", estimated_cost=0.05) as request:
                    request.commit(cost_usd=0.01)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # All reservations settled: spent == committed, no leaked holds.
        assert guard.spent("m") == pytest.approx(guard.committed("m"))
        assert guard.committed("m") == pytest.approx(5.0)


class TestReporting:
    """Spend reporting."""

    def test_report_shape(self, ledger: CostLedger):
        guard = BudgetGuard(ledger, per_model_cap={"a": 2.0, "b": 4.0}, global_cap=10.0)
        with guard.reserve("a", 0.5) as request:
            request.commit(cost_usd=0.5)
        with guard.reserve("b", 1.0) as request:
            request.commit(cost_usd=1.0, run_id="run-a")

        report = guard.report()
        assert report.total_spend == pytest.approx(1.5)
        assert report.global_remaining == pytest.approx(8.5)
        assert report.by_model["a"].remaining == pytest.approx(1.5)
        assert report.by_model["b"].remaining == pytest.approx(3.0)
        assert report.by_model["a"].exhausted is False
        assert report.by_run["run-a"] == pytest.approx(1.5)
        assert report.request_counts == {"a": 1, "b": 1}

    def test_report_includes_configured_but_unused_models(self, ledger: CostLedger):
        guard = BudgetGuard(ledger, per_model_cap={"unused": 5.0})
        report = guard.report()
        assert report.by_model["unused"].spent == 0.0
        assert report.by_model["unused"].remaining == pytest.approx(5.0)

    def test_report_is_json_serializable(self, ledger: CostLedger):
        guard = BudgetGuard(ledger, per_model_cap=5.0, global_cap=10.0)
        with guard.reserve("a", 0.5) as request:
            request.commit(cost_usd=0.5)
        json.dumps(guard.report().to_dict())

    def test_uncapped_report_serializes_without_inf(self, ledger: CostLedger):
        guard = BudgetGuard(ledger)
        with guard.reserve("a", 0.5) as request:
            request.commit(cost_usd=0.5)
        payload = guard.report().to_dict()
        assert payload["global_remaining"] is None
        assert payload["by_model"]["a"]["remaining"] is None

    def test_budget_for_single_model(self, ledger: CostLedger):
        guard = BudgetGuard(ledger, per_model_cap=2.0)
        with guard.reserve("a", 0.5) as request:
            request.commit(cost_usd=0.5)
        budget = guard.budget_for("a")
        assert budget.committed == pytest.approx(0.5)
        assert budget.reserved == 0.0
        assert budget.remaining == pytest.approx(1.5)
