"""QC gate suite (§12-§12.1 of ``docs/data_plan_v0.4.md``).

Every generated document passes through gates G1-G7 before it enters the
corpus; results are stored as columns (§13: ``qc_parse``, ``qc_language``,
``qc_length_delta``, ``qc_selflabel``, ``qc_topic_coverage``, ``qc_near_dup``,
``qc_refusal``) so pass rates per generator become a reportable result.

- :mod:`shelf.qc.gates` -- G1 (parse), G2 (language), G3 (length adherence),
  G4 (self-labeling, delegates to :mod:`shelf.sampler.leakage`), G5 (topic
  coverage), and G7 (refusal/boilerplate). See :func:`shelf.qc.gates.run_gates`.
- :mod:`shelf.qc.dedup` -- G6 (near-duplicate detection via MinHash + banded
  LSH, scaling to corpus size without pairwise comparison). See
  :class:`shelf.qc.dedup.NearDuplicateIndex`.
- :mod:`shelf.qc.promotion` -- §12.1 promotion checks: per-generator
  generated/retained/rejected counts, realized dimension distributions,
  split sizes and SHA-256 hashes, and near-duplicate rate within/across
  generators. See :func:`shelf.qc.promotion.run_promotion_checks`.
"""

from __future__ import annotations

from shelf.qc.dedup import (
    DuplicateMatch,
    MinHasher,
    NearDuplicateIndex,
    NearDuplicateStats,
    jaccard_estimate,
    scan_corpus,
)
from shelf.qc.gates import (
    Gate,
    GateResult,
    QCResult,
    check_language,
    check_length,
    check_parse,
    check_parse_fields,
    check_refusal,
    check_self_label,
    check_topic_coverage,
    run_gates,
)
from shelf.qc.promotion import (
    DEFAULT_DIMENSION_FIELDS,
    GeneratorGateStats,
    NearDuplicateReport,
    PromotionReport,
    SpecCoverageReport,
    run_promotion_checks,
)

__all__ = [
    "DEFAULT_DIMENSION_FIELDS",
    "DuplicateMatch",
    "Gate",
    "GateResult",
    "GeneratorGateStats",
    "MinHasher",
    "NearDuplicateIndex",
    "NearDuplicateReport",
    "NearDuplicateStats",
    "PromotionReport",
    "QCResult",
    "SpecCoverageReport",
    "check_language",
    "check_length",
    "check_parse",
    "check_parse_fields",
    "check_refusal",
    "check_self_label",
    "check_topic_coverage",
    "jaccard_estimate",
    "run_gates",
    "run_promotion_checks",
    "scan_corpus",
]
