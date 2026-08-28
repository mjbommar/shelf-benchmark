"""Promotion checks (§12.1 of ``docs/data_plan_v0.4.md``).

A slice is not published until its manifest establishes every item in the
§12.1 checklist:

- exact rows generated, retained, and rejected per generator, with per-gate
  rejection counts
- unique/missing ``spec_id``s and realized coverage of every sampled
  dimension
- realized distributions by generator, difficulty, relation stratum, length
  bucket, and ``prompt_variant_id``
- split sizes plus every removed cross-split spec collision
- near-duplicate rate within and across generators
- SHA-256 hashes for every split

This module computes all of the above from plain document records plus the
QC gate results computed by :mod:`shelf.qc.gates` and :mod:`shelf.qc.dedup`.
It does not regenerate documents or call generator backends -- it is a pure
report over data that has already been produced and gated.

Several fields named in the data plan (``spec_id``, ``generator``,
``difficulty``, ``relation_stratum``, ``prompt_variant_id``) are v0.4
schema additions (§13) that do not exist in the published v0.3.1 corpus.
``run_promotion_checks`` treats every dimension field as optional: a field
absent from every record is reported as skipped in ``notes`` rather than
raising, so this module works unmodified once those columns land. Generator
identity falls back to ``model`` (the v0.3.1 field that plays that role)
when no explicit ``generator`` field is present.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from shelf.qc.gates import QCResult

__all__ = [
    "DEFAULT_DIMENSION_FIELDS",
    "GeneratorGateStats",
    "NearDuplicateReport",
    "PromotionReport",
    "SpecCoverageReport",
    "run_promotion_checks",
]

# Dimensions worth breaking realized distributions down by, per §12.1's list
# ("by generator, difficulty, relation stratum, length bucket, and
# prompt_variant_id") plus the label dimensions the sampler actually
# stratifies on today. Fields absent from the record schema are skipped
# (see module docstring); this list is deliberately a superset of both the
# v0.3.1 and the planned v0.4 schema.
DEFAULT_DIMENSION_FIELDS: tuple[str, ...] = (
    "lcc_code",
    "lcgft_category",
    "lcgft_form",
    "audience",
    "register",
    "target_length",
    "topics",
    "geographic",
    "difficulty",
    "relation_stratum",
    "prompt_variant_id",
    "slice",
)


@dataclass
class GeneratorGateStats:
    """Rows generated/retained/rejected for one generator, with per-gate
    rejection counts (§12.1's first bullet)."""

    generated: int = 0
    retained: int = 0
    rejected: int = 0
    rejected_by_gate: dict[str, int] = field(default_factory=dict)

    def _record_rejection(self, gate_name: str) -> None:
        self.rejected_by_gate[gate_name] = self.rejected_by_gate.get(gate_name, 0) + 1


@dataclass
class SpecCoverageReport:
    """Unique/missing ``spec_id`` counts and realized dimension coverage."""

    unique_spec_ids: int
    missing_spec_id_count: int
    dimension_coverage: dict[str, int]


@dataclass
class NearDuplicateReport:
    """Near-duplicate rate within and across generators (§12.1)."""

    total_documents: int
    duplicate_documents: int
    within_generator_pairs: int
    across_generator_pairs: int

    @property
    def duplicate_rate(self) -> float:
        return (
            self.duplicate_documents / self.total_documents
            if self.total_documents
            else 0.0
        )

    @property
    def total_duplicate_pairs(self) -> int:
        return self.within_generator_pairs + self.across_generator_pairs

    @property
    def within_generator_rate(self) -> float:
        """Fraction of near-duplicate *pairs* that share a generator.

        Near 1.0 is the expected/benign shape under the Phase 1 design:
        most near-duplicates should come from the same spec being written
        by generator X twice (a real defect), not from independent specs
        that happen to collide across generators.
        """
        total = self.total_duplicate_pairs
        return self.within_generator_pairs / total if total else 0.0


@dataclass
class PromotionReport:
    """The full §12.1 promotion manifest."""

    generator_stats: dict[str, GeneratorGateStats]
    dimension_distributions: dict[str, dict[str, dict[str, int]]]
    split_sizes: dict[str, int]
    split_hashes: dict[str, str]
    spec_coverage: SpecCoverageReport | None
    cross_split_spec_collisions: tuple[str, ...]
    near_duplicates: NearDuplicateReport
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable summary, e.g. for a manifest file."""
        return {
            "generator_stats": {
                name: {
                    "generated": stats.generated,
                    "retained": stats.retained,
                    "rejected": stats.rejected,
                    "rejected_by_gate": dict(stats.rejected_by_gate),
                }
                for name, stats in self.generator_stats.items()
            },
            "dimension_distributions": self.dimension_distributions,
            "split_sizes": self.split_sizes,
            "split_hashes": self.split_hashes,
            "spec_coverage": None
            if self.spec_coverage is None
            else {
                "unique_spec_ids": self.spec_coverage.unique_spec_ids,
                "missing_spec_id_count": self.spec_coverage.missing_spec_id_count,
                "dimension_coverage": self.spec_coverage.dimension_coverage,
            },
            "cross_split_spec_collisions": list(self.cross_split_spec_collisions),
            "near_duplicates": {
                "total_documents": self.near_duplicates.total_documents,
                "duplicate_documents": self.near_duplicates.duplicate_documents,
                "duplicate_rate": self.near_duplicates.duplicate_rate,
                "within_generator_pairs": self.near_duplicates.within_generator_pairs,
                "across_generator_pairs": self.near_duplicates.across_generator_pairs,
                "within_generator_rate": self.near_duplicates.within_generator_rate,
            },
            "notes": list(self.notes),
        }


def _resolve_generator(
    record: Mapping[str, Any],
    generator_field: str,
    fallback_fields: Sequence[str],
) -> str:
    value = record.get(generator_field)
    if value:
        return str(value)
    for fallback in fallback_fields:
        value = record.get(fallback)
        if value:
            return str(value)
    return "unknown"


def _hash_split(
    doc_ids: Sequence[str],
    by_id: Mapping[str, Mapping[str, Any]],
    text_field: str,
) -> str:
    """SHA-256 over a split's (id, text) pairs, order-independent.

    Sorting by id before hashing means the hash depends only on split
    membership and content, not on iteration/shard order -- so re-running
    the split machinery with the same documents reproduces the same hash.
    """
    hasher = hashlib.sha256()
    for doc_id in sorted(str(i) for i in doc_ids):
        record = by_id.get(doc_id, {})
        text = str(record.get(text_field, ""))
        hasher.update(doc_id.encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(text.encode("utf-8"))
        hasher.update(b"\x01")
    return hasher.hexdigest()


def run_promotion_checks(
    records: Sequence[Mapping[str, Any]],
    qc_results: Mapping[str, QCResult],
    *,
    splits: Mapping[str, Sequence[str]] | None = None,
    dedup_pairs: Sequence[tuple[str, str, float]] | None = None,
    id_field: str = "id",
    text_field: str = "text",
    generator_field: str = "generator",
    generator_fallback_fields: Sequence[str] = ("model",),
    spec_id_field: str = "spec_id",
    dimension_fields: Sequence[str] = DEFAULT_DIMENSION_FIELDS,
) -> PromotionReport:
    """Build the §12.1 promotion manifest for one slice.

    Args:
        records: Every generated candidate document (retained *and*
            rejected) as mappings with at least ``id_field``.
        qc_results: ``doc_id -> QCResult`` for every record in ``records``.
            A record with no entry here is treated as rejected (missing QC
            is itself a defect worth surfacing, not something to skip).
        splits: ``split_name -> doc_ids`` for the documents that made it
            into each published split (train/validation/test, or
            core/subclass/minimal_pair/holdout). Only ids present here
            contribute to split sizes, split hashes, dimension
            distributions, and cross-split collision detection. If omitted,
            those sections report zero/empty with a note.
        dedup_pairs: Near-duplicate pairs as ``(id_a, id_b, similarity)``,
            typically ``NearDuplicateIndex.find_all_duplicate_pairs()`` run
            over ``records``. If omitted, the near-duplicate report is all
            zeros with a note.
        id_field: Key for the document id in each record.
        text_field: Key for document text, used for split hashing.
        generator_field: Key naming the generator explicitly (the planned
            v0.4 ``generator_family``/``generator`` column).
        generator_fallback_fields: Tried in order if ``generator_field`` is
            absent or empty -- ``model`` covers the published v0.3.1 schema,
            where generator identity is only recorded as the exact model
            string.
        spec_id_field: Key for the shared-spec id (§13, Phase 1 of v0.4).
        dimension_fields: Which fields to break realized distributions down
            by. Defaults to :data:`DEFAULT_DIMENSION_FIELDS`.
    """
    notes: list[str] = []
    by_id: dict[str, Mapping[str, Any]] = {
        str(record[id_field]): record for record in records
    }

    # --- rows generated/retained/rejected per generator, with per-gate
    # rejection counts -----------------------------------------------------
    generator_stats: dict[str, GeneratorGateStats] = defaultdict(GeneratorGateStats)
    for record in records:
        doc_id = str(record[id_field])
        generator = _resolve_generator(
            record, generator_field, generator_fallback_fields
        )
        stats = generator_stats[generator]
        stats.generated += 1
        qc = qc_results.get(doc_id)
        if qc is None:
            stats.rejected += 1
            stats._record_rejection("missing_qc_result")
            continue
        if qc.passed:
            stats.retained += 1
        else:
            stats.rejected += 1
            for gate in qc.failed_gates():
                stats._record_rejection(gate.value)

    # --- spec_id coverage ---------------------------------------------------
    spec_ids_present = [record.get(spec_id_field) for record in records]
    if any(spec_ids_present):
        unique_spec_ids = len({sid for sid in spec_ids_present if sid})
        missing = sum(1 for sid in spec_ids_present if not sid)
        dimension_coverage = {
            dim: len({v for record in records for v in _as_iterable(record.get(dim))})
            for dim in dimension_fields
            if any(record.get(dim) not in (None, "", []) for record in records)
        }
        spec_coverage = SpecCoverageReport(
            unique_spec_ids=unique_spec_ids,
            missing_spec_id_count=missing,
            dimension_coverage=dimension_coverage,
        )
    else:
        spec_coverage = None
        notes.append(
            f"'{spec_id_field}' not present on any record; spec coverage skipped "
            "(this is a v0.4/Phase-1 schema addition, absent from v0.3.1)."
        )

    # --- split sizes, hashes, dimension distributions, cross-split collisions
    splits = splits or {}
    if not splits:
        notes.append("no 'splits' provided; split sizes/hashes/collisions skipped.")

    split_sizes = {name: len(ids) for name, ids in splits.items()}
    split_hashes = {
        name: _hash_split(ids, by_id, text_field) for name, ids in splits.items()
    }

    dimension_distributions: dict[str, dict[str, dict[str, int]]] = {}
    for dim in dimension_fields:
        per_split: dict[str, dict[str, int]] = {}
        any_present = False
        for split_name, ids in splits.items():
            counts: dict[str, int] = {}
            for doc_id in ids:
                record = by_id.get(str(doc_id))
                if record is None:
                    continue
                for value in _as_iterable(record.get(dim)):
                    counts[value] = counts.get(value, 0) + 1
                    any_present = True
            per_split[split_name] = counts
        if any_present:
            dimension_distributions[dim] = per_split
        elif splits:
            notes.append(f"dimension '{dim}' not present on any split record; skipped.")

    cross_split_spec_collisions: tuple[str, ...] = ()
    if spec_coverage is not None and splits:
        spec_to_splits: dict[str, set[str]] = defaultdict(set)
        for split_name, ids in splits.items():
            for doc_id in ids:
                record = by_id.get(str(doc_id))
                if record is None:
                    continue
                sid = record.get(spec_id_field)
                if sid:
                    spec_to_splits[sid].add(split_name)
        cross_split_spec_collisions = tuple(
            sorted(
                sid
                for sid, split_names in spec_to_splits.items()
                if len(split_names) > 1
            )
        )

    # --- near-duplicate rate within/across generators -----------------------
    if dedup_pairs is None:
        notes.append(
            "no 'dedup_pairs' provided; run NearDuplicateIndex over the corpus "
            "and pass find_all_duplicate_pairs() to populate near-duplicate stats."
        )
        near_duplicates = NearDuplicateReport(
            total_documents=len(records),
            duplicate_documents=0,
            within_generator_pairs=0,
            across_generator_pairs=0,
        )
    else:
        flagged_ids: set[str] = set()
        within = 0
        across = 0
        for id_a, id_b, _similarity in dedup_pairs:
            flagged_ids.add(id_a)
            flagged_ids.add(id_b)
            record_a = by_id.get(id_a)
            record_b = by_id.get(id_b)
            if record_a is None or record_b is None:
                continue
            gen_a = _resolve_generator(
                record_a, generator_field, generator_fallback_fields
            )
            gen_b = _resolve_generator(
                record_b, generator_field, generator_fallback_fields
            )
            if gen_a == gen_b:
                within += 1
            else:
                across += 1
        near_duplicates = NearDuplicateReport(
            total_documents=len(records),
            duplicate_documents=len(flagged_ids),
            within_generator_pairs=within,
            across_generator_pairs=across,
        )

    return PromotionReport(
        generator_stats=dict(generator_stats),
        dimension_distributions=dimension_distributions,
        split_sizes=split_sizes,
        split_hashes=split_hashes,
        spec_coverage=spec_coverage,
        cross_split_spec_collisions=cross_split_spec_collisions,
        near_duplicates=near_duplicates,
        notes=tuple(notes),
    )


def _as_iterable(value: Any) -> list[str]:
    """Normalize a record field (scalar, list, or missing) to string values."""
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value if v not in (None, "")]
    return [str(value)]
