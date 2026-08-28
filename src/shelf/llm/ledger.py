"""Append-only cost ledger and hard budget guard for SHELF generation runs.

v0.4 spends real money across ~15 generator models under a hard $20–50 per-model
cap (``docs/data_plan_v0.4.md`` §4, §5, §16). Two properties are non-negotiable:

1. **A model cannot overrun its cap.** :class:`BudgetGuard` reserves the
   estimated cost of a request *before* the call is made and raises
   :class:`BudgetExceeded` once the cap is reached. The caller aborts that model
   and continues with the rest of the roster.
2. **A restart must not double-spend.** Every request appends one JSONL record
   to ``data/artifacts/cost_ledger.jsonl``. On construction the guard replays
   the ledger and recomputes spend-to-date, so a resumed run starts from the
   real number rather than zero.

Concurrency
-----------
The generator runs with ``--concurrency 20`` (asyncio tasks, with sync backends
dispatched to executor threads), so the ledger must be safe for both. A single
``threading.RLock`` guards the reserve/commit accounting *and* the file append;
no ``await`` ever happens while the lock is held, so it is safe to call from
coroutines. Records are written as one ``write()`` of a complete newline-
terminated line to a file opened in append mode, then flushed, so a crash can at
worst leave a torn final line — which :func:`read_records` skips.

Example
-------
    ledger = CostLedger("data/artifacts/cost_ledger.jsonl", run_id="phase1")
    guard = BudgetGuard(ledger, per_model_cap=50.0, global_cap=400.0)

    try:
        with guard.reserve(model, estimated_cost=0.02, spec_id=spec.id) as req:
            result = await backend.generate_async(...)
            req.commit(
                input_tokens=result.input_tokens or 0,
                output_tokens=result.output_tokens or 0,
                cost_usd=table.estimate_cost(model, ...),
                provider_served=getattr(result, "provider_served", None),
            )
    except BudgetExceeded:
        abort_model(model)
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_LEDGER_PATH = Path("data/artifacts/cost_ledger.jsonl")

#: Relative slack used when comparing accumulated float spend against a cap.
#: Summing thousands of small costs drifts by ~1e-16 per addition; without this
#: a run can be refused one request short of its cap (or, at the other end, a
#: model sitting exactly on its cap looks like it still has room).
_CAP_EPSILON = 1e-9

STATUS_OK = "ok"
STATUS_ERROR = "error"
STATUS_SKIPPED = "skipped"


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class BudgetError(RuntimeError):
    """Base class for budget failures."""


class BudgetExceeded(BudgetError):
    """A per-model or global spend cap would be exceeded by this request.

    Attributes:
        model: The model whose request was refused.
        spent: Committed + reserved spend for that model (USD).
        cap: The cap that was hit (USD).
        scope: ``"model"`` or ``"global"``.
    """

    def __init__(
        self, model: str, spent: float, cap: float, scope: str = "model"
    ) -> None:
        self.model = model
        self.spent = spent
        self.cap = cap
        self.scope = scope
        super().__init__(
            f"{scope} budget exceeded for {model!r}: "
            f"${spent:.4f} committed/reserved against a ${cap:.2f} cap"
        )


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def new_run_id(prefix: str = "run") -> str:
    """Generate a sortable, unique run id (``run-20260826T140501Z-3f9a1c``)."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{stamp}-{uuid.uuid4().hex[:6]}"


@dataclass(frozen=True, slots=True)
class LedgerRecord:
    """One LLM request's cost accounting."""

    timestamp: str
    run_id: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0
    provider_served: str | None = None
    spec_id: str | None = None
    request_id: str | None = None
    status: str = STATUS_OK
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        """Input + output tokens (reasoning is assumed folded into output)."""
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict form, suitable for JSON."""
        return asdict(self)

    def to_json(self) -> str:
        """Compact single-line JSON, no trailing newline."""
        return json.dumps(self.to_dict(), separators=(",", ":"), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LedgerRecord:
        """Rebuild a record from a parsed ledger line, tolerating extra keys."""
        extra = data.get("extra")
        return cls(
            timestamp=str(data.get("timestamp", "")),
            run_id=str(data.get("run_id", "")),
            model=str(data.get("model", "")),
            input_tokens=int(data.get("input_tokens") or 0),
            output_tokens=int(data.get("output_tokens") or 0),
            reasoning_tokens=int(data.get("reasoning_tokens") or 0),
            cost_usd=float(data.get("cost_usd") or 0.0),
            provider_served=data.get("provider_served"),
            spec_id=data.get("spec_id"),
            request_id=data.get("request_id"),
            status=str(data.get("status") or STATUS_OK),
            extra=dict(extra) if isinstance(extra, Mapping) else {},
        )


def read_records(path: Path | str) -> Iterator[LedgerRecord]:
    """Stream records from a ledger file.

    Malformed lines — including a torn final line left by a crash mid-write —
    are skipped rather than raising, so a partially written ledger stays usable.
    """
    ledger_path = Path(path)
    if not ledger_path.exists():
        return
    with ledger_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict) and data.get("model"):
                yield LedgerRecord.from_dict(data)


# --------------------------------------------------------------------------- #
# Ledger
# --------------------------------------------------------------------------- #


class CostLedger:
    """Append-only JSONL ledger, safe for threads and asyncio tasks.

    The file handle is opened lazily in append mode and kept open for the life
    of the ledger. Each :meth:`record` call writes exactly one complete line
    under a lock and flushes; ``fsync`` is optional (default on) for durability
    across a hard kill.
    """

    def __init__(
        self,
        path: Path | str = DEFAULT_LEDGER_PATH,
        *,
        run_id: str | None = None,
        fsync: bool = True,
    ) -> None:
        """Open (or prepare to open) a ledger.

        Args:
            path: JSONL file. Parent directories are created on first write.
            run_id: Identifier stamped on records written by this instance.
                Defaults to a fresh :func:`new_run_id`.
            fsync: Flush to disk after every record. Costs ~1ms per request and
                buys crash durability; disable only for tests or dry runs.
        """
        self.path = Path(path)
        self.run_id = run_id or new_run_id()
        self.fsync = fsync
        self._lock = threading.RLock()
        self._handle = None
        self._closed = False

    # -- lifecycle --------------------------------------------------------- #

    def _get_handle(self):
        if self._handle is None:
            if self._closed:
                raise ValueError("ledger is closed")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("a", encoding="utf-8")
        return self._handle

    def close(self) -> None:
        """Flush and close the underlying file handle."""
        with self._lock:
            if self._handle is not None:
                self._handle.flush()
                self._handle.close()
                self._handle = None
            self._closed = True

    def __enter__(self) -> CostLedger:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- writing ----------------------------------------------------------- #

    def append(self, record: LedgerRecord) -> LedgerRecord:
        """Append a pre-built record."""
        line = record.to_json()
        with self._lock:
            handle = self._get_handle()
            handle.write(line + "\n")
            handle.flush()
            if self.fsync:
                os.fsync(handle.fileno())
        return record

    def record(
        self,
        model: str,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        reasoning_tokens: int = 0,
        cost_usd: float = 0.0,
        provider_served: str | None = None,
        spec_id: str | None = None,
        request_id: str | None = None,
        status: str = STATUS_OK,
        run_id: str | None = None,
        timestamp: str | None = None,
        **extra: Any,
    ) -> LedgerRecord:
        """Build and append one request record.

        Args:
            model: Model id as requested (keep the ``:batch`` suffix if used).
            input_tokens: Billed prompt tokens.
            output_tokens: Billed completion tokens.
            reasoning_tokens: Thinking tokens as reported by the provider.
            cost_usd: Cost for this request, normally from
                ``shelf.llm.pricing.estimate_cost``.
            provider_served: The backend OpenRouter actually routed to.
            spec_id: Document spec this request belongs to (for resume).
            request_id: Provider-side request id, when available.
            status: ``"ok"``, ``"error"``, or ``"skipped"``.
            run_id: Overrides the ledger's run id for this record.
            timestamp: Overrides the generated UTC timestamp.
            **extra: Anything else worth keeping (e.g. ``spec_block_id``).

        Returns:
            The appended :class:`LedgerRecord`.
        """
        record = LedgerRecord(
            timestamp=timestamp or _utc_now_iso(),
            run_id=run_id or self.run_id,
            model=model,
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
            reasoning_tokens=int(reasoning_tokens),
            cost_usd=float(cost_usd),
            provider_served=provider_served,
            spec_id=spec_id,
            request_id=request_id,
            status=status,
            extra=dict(extra),
        )
        return self.append(record)

    # -- reading ----------------------------------------------------------- #

    def records(self) -> list[LedgerRecord]:
        """Read every record currently on disk."""
        with self._lock:
            if self._handle is not None:
                self._handle.flush()
        return list(read_records(self.path))

    def spend_by_model(self) -> dict[str, float]:
        """Total USD spend per model, across all runs in the file."""
        totals: dict[str, float] = {}
        for record in self.records():
            totals[record.model] = totals.get(record.model, 0.0) + record.cost_usd
        return totals

    def spend_by_run(self) -> dict[str, float]:
        """Total USD spend per run id."""
        totals: dict[str, float] = {}
        for record in self.records():
            totals[record.run_id] = totals.get(record.run_id, 0.0) + record.cost_usd
        return totals

    def total_spend(self) -> float:
        """Total USD spend in the file."""
        return sum(record.cost_usd for record in self.records())

    def request_counts(self) -> dict[str, int]:
        """Number of ledger records per model."""
        counts: dict[str, int] = {}
        for record in self.records():
            counts[record.model] = counts.get(record.model, 0) + 1
        return counts

    def completed_spec_ids(self, model: str | None = None) -> set[str]:
        """Spec ids already generated successfully (for a resumable queue)."""
        done: set[str] = set()
        for record in self.records():
            if record.status != STATUS_OK or not record.spec_id:
                continue
            if model is None or record.model == model:
                done.add(record.spec_id)
        return done


# --------------------------------------------------------------------------- #
# Budget guard
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ModelBudget:
    """Spend state for one model."""

    model: str
    cap: float | None
    committed: float
    reserved: float

    @property
    def spent(self) -> float:
        """Committed plus currently reserved spend."""
        return self.committed + self.reserved

    @property
    def remaining(self) -> float:
        """Headroom left under the cap (``inf`` when uncapped)."""
        if self.cap is None:
            return float("inf")
        return max(0.0, self.cap - self.spent)

    @property
    def exhausted(self) -> bool:
        """True when the cap has been reached."""
        return self.cap is not None and self.spent >= self.cap - _slack(self.cap)


@dataclass(frozen=True, slots=True)
class BudgetReport:
    """Snapshot of budget state, suitable for a manifest or a console table."""

    total_spend: float
    global_cap: float | None
    by_model: dict[str, ModelBudget]
    by_run: dict[str, float]
    request_counts: dict[str, int]

    @property
    def global_remaining(self) -> float:
        """Headroom left under the global cap (``inf`` when uncapped)."""
        if self.global_cap is None:
            return float("inf")
        return max(0.0, self.global_cap - self.total_spend)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable form."""
        return {
            "total_spend": round(self.total_spend, 6),
            "global_cap": self.global_cap,
            "global_remaining": (
                None if self.global_cap is None else round(self.global_remaining, 6)
            ),
            "by_model": {
                model: {
                    "cap": budget.cap,
                    "committed": round(budget.committed, 6),
                    "reserved": round(budget.reserved, 6),
                    "spent": round(budget.spent, 6),
                    "remaining": (
                        None if budget.cap is None else round(budget.remaining, 6)
                    ),
                    "exhausted": budget.exhausted,
                    "requests": self.request_counts.get(model, 0),
                }
                for model, budget in sorted(self.by_model.items())
            },
            "by_run": {
                run: round(cost, 6) for run, cost in sorted(self.by_run.items())
            },
        }


class PendingRequest:
    """Handle for an in-flight reservation. Obtained from :meth:`BudgetGuard.reserve`."""

    __slots__ = ("_guard", "_settled", "estimated_cost", "model", "spec_id", "record")

    def __init__(
        self,
        guard: BudgetGuard,
        model: str,
        estimated_cost: float,
        spec_id: str | None,
    ) -> None:
        self._guard = guard
        self.model = model
        self.estimated_cost = estimated_cost
        self.spec_id = spec_id
        self._settled = False
        self.record: LedgerRecord | None = None

    @property
    def settled(self) -> bool:
        """True once the reservation has been committed or released."""
        return self._settled

    def commit(
        self,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        reasoning_tokens: int = 0,
        cost_usd: float | None = None,
        provider_served: str | None = None,
        request_id: str | None = None,
        status: str = STATUS_OK,
        **extra: Any,
    ) -> LedgerRecord:
        """Write the actual cost to the ledger and release the reservation.

        Args:
            cost_usd: Actual cost. Falls back to the reserved estimate when the
                caller cannot compute one.
            (others): See :meth:`CostLedger.record`.
        """
        if self._settled:
            raise BudgetError(f"request for {self.model!r} already settled")
        actual = self.estimated_cost if cost_usd is None else float(cost_usd)
        self.record = self._guard._settle(
            self,
            actual_cost=actual,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            provider_served=provider_served,
            request_id=request_id,
            status=status,
            extra=extra,
        )
        self._settled = True
        return self.record

    def release(self, *, record_failure: bool = False, **extra: Any) -> None:
        """Drop the reservation without charging for it.

        Args:
            record_failure: Also write a zero-cost ``status="error"`` record, so
                failed attempts remain visible in the ledger.
            **extra: Extra fields for that record.
        """
        if self._settled:
            return
        self._guard._release(self, record_failure=record_failure, extra=extra)
        self._settled = True


class BudgetGuard:
    """Hard per-model (and optional global) spend caps over a :class:`CostLedger`.

    Spend-to-date is recomputed from the ledger file on construction, so a
    resumed run continues from the real number instead of re-spending. Both the
    accounting and the ledger append happen under one re-entrant lock, which
    makes the guard safe to share across the asyncio tasks and executor threads
    of the generation loop.
    """

    def __init__(
        self,
        ledger: CostLedger,
        *,
        per_model_cap: float | Mapping[str, float] | None = None,
        global_cap: float | None = None,
        default_cap: float | None = None,
    ) -> None:
        """Configure caps and replay the ledger.

        Args:
            ledger: Where records are written and spend-to-date is read from.
            per_model_cap: A single cap applied to every model, or a per-model
                mapping. ``None`` means uncapped unless ``default_cap`` is set.
            global_cap: Optional cap across all models combined.
            default_cap: Cap for models missing from a ``per_model_cap`` mapping.

        Raises:
            ValueError: If any cap is not positive.
        """
        self.ledger = ledger
        self.global_cap = _validate_cap("global_cap", global_cap)

        if isinstance(per_model_cap, Mapping):
            self._caps = {
                model: _require_cap(f"per_model_cap[{model!r}]", cap)
                for model, cap in per_model_cap.items()
            }
            self.default_cap = _validate_cap("default_cap", default_cap)
        else:
            self._caps = {}
            single = _validate_cap("per_model_cap", per_model_cap)
            self.default_cap = (
                _validate_cap("default_cap", default_cap) if single is None else single
            )

        self._lock = threading.RLock()
        self._committed: dict[str, float] = {}
        self._reserved: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        self.reload()

    # -- caps -------------------------------------------------------------- #

    def cap_for(self, model: str) -> float | None:
        """The cap applying to ``model`` (``None`` when uncapped)."""
        return self._caps.get(model, self.default_cap)

    def set_cap(self, model: str, cap: float | None) -> None:
        """Override the cap for one model."""
        with self._lock:
            if cap is None:
                self._caps.pop(model, None)
            else:
                self._caps[model] = _require_cap(f"cap for {model!r}", cap)

    # -- state ------------------------------------------------------------- #

    def reload(self) -> None:
        """Recompute committed spend from the ledger file (resume path)."""
        committed: dict[str, float] = {}
        counts: dict[str, int] = {}
        for record in self.ledger.records():
            committed[record.model] = committed.get(record.model, 0.0) + record.cost_usd
            counts[record.model] = counts.get(record.model, 0) + 1
        with self._lock:
            self._committed = committed
            self._counts = counts

    def spent(self, model: str) -> float:
        """Committed + reserved spend for a model."""
        with self._lock:
            return self._committed.get(model, 0.0) + self._reserved.get(model, 0.0)

    def committed(self, model: str) -> float:
        """Spend already written to the ledger for a model."""
        with self._lock:
            return self._committed.get(model, 0.0)

    def total_spend(self) -> float:
        """Committed + reserved spend across all models."""
        with self._lock:
            return sum(self._committed.values()) + sum(self._reserved.values())

    def remaining(self, model: str) -> float:
        """Headroom left for a model, bounded by the global cap too."""
        cap = self.cap_for(model)
        model_left = float("inf") if cap is None else max(0.0, cap - self.spent(model))
        if self.global_cap is None:
            return model_left
        return min(model_left, max(0.0, self.global_cap - self.total_spend()))

    def is_exhausted(self, model: str) -> bool:
        """True when the model (or the global budget) has hit its cap."""
        cap = self.cap_for(model)
        if cap is not None and self.spent(model) >= cap - _slack(cap):
            return True
        return (
            self.global_cap is not None
            and self.total_spend() >= self.global_cap - _slack(self.global_cap)
        )

    # -- enforcement ------------------------------------------------------- #

    def check(self, model: str, estimated_cost: float = 0.0) -> None:
        """Raise if this request cannot be afforded.

        A model is blocked once its committed + reserved spend has *reached* the
        cap, and also when adding ``estimated_cost`` would push it over.

        Raises:
            BudgetExceeded: With ``scope="model"`` or ``scope="global"``.
        """
        if estimated_cost < 0:
            raise ValueError("estimated_cost must be non-negative")
        with self._lock:
            self._check_locked(model, estimated_cost)

    def _check_locked(self, model: str, estimated_cost: float) -> None:
        cap = self.cap_for(model)
        if cap is not None:
            spent = self._committed.get(model, 0.0) + self._reserved.get(model, 0.0)
            slack = _slack(cap)
            if spent >= cap - slack or spent + estimated_cost > cap + slack:
                raise BudgetExceeded(model, spent, cap, scope="model")
        if self.global_cap is not None:
            total = sum(self._committed.values()) + sum(self._reserved.values())
            slack = _slack(self.global_cap)
            if (
                total >= self.global_cap - slack
                or total + estimated_cost > self.global_cap + slack
            ):
                raise BudgetExceeded(model, total, self.global_cap, scope="global")

    @contextmanager
    def reserve(
        self,
        model: str,
        estimated_cost: float = 0.0,
        *,
        spec_id: str | None = None,
        record_failure: bool = False,
    ) -> Iterator[PendingRequest]:
        """Reserve budget for one request, then settle it.

        The estimate is held against the cap for the duration of the call, so
        20 concurrent requests cannot each individually pass a check and then
        collectively blow the budget. Call :meth:`PendingRequest.commit` inside
        the block with the actual usage; leaving the block without committing
        (including via an exception) releases the reservation and charges $0.

        Args:
            model: Model id as requested.
            estimated_cost: Worst-case USD cost of the request.
            spec_id: Document spec id, recorded for resume.
            record_failure: Write a zero-cost ``status="error"`` record if the
                block raises.

        Yields:
            A :class:`PendingRequest`.

        Raises:
            BudgetExceeded: If the request cannot be afforded.
        """
        pending = self._reserve(model, estimated_cost, spec_id)
        try:
            yield pending
        except BaseException:
            pending.release(record_failure=record_failure)
            raise
        else:
            pending.release()

    def _reserve(
        self, model: str, estimated_cost: float, spec_id: str | None
    ) -> PendingRequest:
        if estimated_cost < 0:
            raise ValueError("estimated_cost must be non-negative")
        with self._lock:
            self._check_locked(model, estimated_cost)
            self._reserved[model] = self._reserved.get(model, 0.0) + estimated_cost
        return PendingRequest(self, model, estimated_cost, spec_id)

    def _settle(
        self,
        pending: PendingRequest,
        *,
        actual_cost: float,
        input_tokens: int,
        output_tokens: int,
        reasoning_tokens: int,
        provider_served: str | None,
        request_id: str | None,
        status: str,
        extra: Mapping[str, Any],
    ) -> LedgerRecord:
        with self._lock:
            self._drop_reservation(pending)
            record = self.ledger.record(
                pending.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                reasoning_tokens=reasoning_tokens,
                cost_usd=actual_cost,
                provider_served=provider_served,
                spec_id=pending.spec_id,
                request_id=request_id,
                status=status,
                **extra,
            )
            self._committed[pending.model] = (
                self._committed.get(pending.model, 0.0) + actual_cost
            )
            self._counts[pending.model] = self._counts.get(pending.model, 0) + 1
        return record

    def _release(
        self,
        pending: PendingRequest,
        *,
        record_failure: bool,
        extra: Mapping[str, Any],
    ) -> None:
        with self._lock:
            self._drop_reservation(pending)
            if record_failure:
                self.ledger.record(
                    pending.model,
                    cost_usd=0.0,
                    spec_id=pending.spec_id,
                    status=STATUS_ERROR,
                    **extra,
                )
                self._counts[pending.model] = self._counts.get(pending.model, 0) + 1

    def _drop_reservation(self, pending: PendingRequest) -> None:
        remaining = self._reserved.get(pending.model, 0.0) - pending.estimated_cost
        if remaining > 1e-12:
            self._reserved[pending.model] = remaining
        else:
            self._reserved.pop(pending.model, None)

    # -- direct recording -------------------------------------------------- #

    def record(
        self,
        model: str,
        *,
        cost_usd: float,
        check: bool = True,
        **kwargs: Any,
    ) -> LedgerRecord:
        """Record an already-incurred cost without a prior reservation.

        Use this for costs discovered after the fact (e.g. a batch job's final
        usage report). Set ``check=False`` to book a cost that legitimately
        overshoots the cap rather than losing the accounting.

        Raises:
            BudgetExceeded: When ``check`` is True and the cap is already hit.
        """
        with self._lock:
            if check:
                self._check_locked(model, cost_usd)
            record = self.ledger.record(model, cost_usd=cost_usd, **kwargs)
            self._committed[model] = self._committed.get(model, 0.0) + cost_usd
            self._counts[model] = self._counts.get(model, 0) + 1
        return record

    # -- reporting --------------------------------------------------------- #

    def budget_for(self, model: str) -> ModelBudget:
        """Spend state for a single model."""
        with self._lock:
            return ModelBudget(
                model=model,
                cap=self.cap_for(model),
                committed=self._committed.get(model, 0.0),
                reserved=self._reserved.get(model, 0.0),
            )

    def report(self, models: Iterable[str] | None = None) -> BudgetReport:
        """Build a :class:`BudgetReport` over known and configured models."""
        with self._lock:
            known = set(self._committed) | set(self._reserved) | set(self._caps)
            if models is not None:
                known |= set(models)
            by_model = {model: self.budget_for(model) for model in known}
            counts = dict(self._counts)
            total = sum(self._committed.values()) + sum(self._reserved.values())
        return BudgetReport(
            total_spend=total,
            global_cap=self.global_cap,
            by_model=by_model,
            by_run=self.ledger.spend_by_run(),
            request_counts=counts,
        )


def _slack(cap: float) -> float:
    """Float-comparison slack proportional to the cap."""
    return max(abs(cap), 1.0) * _CAP_EPSILON


def _validate_cap(name: str, cap: float | None) -> float | None:
    if cap is None:
        return None
    return _require_cap(name, cap)


def _require_cap(name: str, cap: Any) -> float:
    """Coerce a cap to a positive float or raise."""
    try:
        value = float(cap)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number, got {cap!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


__all__ = [
    "DEFAULT_LEDGER_PATH",
    "STATUS_ERROR",
    "STATUS_OK",
    "STATUS_SKIPPED",
    "BudgetError",
    "BudgetExceeded",
    "BudgetGuard",
    "BudgetReport",
    "CostLedger",
    "LedgerRecord",
    "ModelBudget",
    "PendingRequest",
    "new_run_id",
    "read_records",
]
