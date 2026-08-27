"""Instruction-conditioned retrieval for SHELF.

Implements the task described in ``docs/data_plan_v0.4.md`` section 11.3.

SHELF's corpus is a near-factorial cross of independent facets: every LCC
subject class co-occurs with every LCGFT genre category, so the corpus contains
maps about philosophy and jokes about law. That independence is what makes a
request like *"find documents on the same subject but in a different genre"*
answerable here and ill-posed on a natural corpus, where genre and subject are
correlated and the "different genre" half of the request either has no
population or silently changes the subject.

The consequence for evaluation is the point of this module: **relevance is
defined by the instruction, not by a label field**. The same query document has
a different correct answer set under each instruction, and under two
complementary instructions those sets are disjoint by construction. A retriever
that scores the same regardless of the instruction is not following it, and that
now shows up as a number rather than as an intuition.

Each instruction is a pair of constraint groups over the facet axes:

* **anchor** -- facets the retrieved document must *share* with the query.
* **contrast** -- facets it must *differ* on.

Relevant = anchor satisfied AND no contrast violated.

Two diagnostics accompany the usual IR metrics, because NDCG alone cannot say
*why* a run failed:

* ``anchor_match@k`` -- how much of the top-k satisfies the anchor at all. Low
  values mean the model could not find the requested facet.
This metric is **V@k** in the published literature (Yin, Tang and Du, CoDeR,
arXiv:2606.13204, which also defines FVR, the first violating rank). The key
below keeps its original spelling so result files already on disk stay
readable; papers and documentation should say V@k.

* ``contrast_violation@k`` -- V@k: how much of the top-k violates the contrast, i.e.
  is exactly what the instruction asked to exclude. Reported with
  ``contrast_violation_lift@k``, the ratio to the rate a random ranking would
  produce. A lift near 1.0 means the model is indifferent to the contrast
  clause; a lift well above 1.0 means it is actively drawn to what the
  instruction forbids, which is the characteristic failure of a model that
  embeds documents by overall similarity and ignores the instruction.

Example:
    from shelf.evaluate.instructions import get_instruction, InstructionJudge

    spec = get_instruction("instruction_same_form_diff_subject")
    spec.render("A field diary from Seattle...")
    # 'Instruct: Given a document, retrieve documents in the same genre...'

    judge = InstructionJudge(spec, corpus_rows)
    judgments = judge.judge(query_rows, results, k_values=[10])
    judgments.relevance["q1"]                       # instruction-defined qrels
    judgments.metrics["contrast_violation_lift@10"]
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

__all__ = [
    "FacetConstraint",
    "InstructionJudge",
    "InstructionJudgments",
    "InstructionSpec",
    "INSTRUCTION_SPECS",
    "get_instruction",
    "is_instruction_task",
    "list_instruction_tasks",
]

FacetRelation = Literal["same", "different"]

#: Prompt shape used by the instruct-embedder families this task targets
#: (E5-instruct, Qwen3-Embedding, Instructor). The instruction is a query-side
#: prefix, so a plain document-similarity model reads it as a few extra words
#: and is unaffected -- which is the intended null.
DEFAULT_QUERY_TEMPLATE = "Instruct: {instruction}\nQuery: {text}"


def _normalize_scalar(value: Any) -> str | None:
    """Normalize a scalar facet value, treating blanks as missing.

    A missing facet can never satisfy a "same" constraint. Audience is present
    on only ~70% of the corpus, so silently matching two missing values would
    manufacture a large block of spurious relevance.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text.casefold()


def _normalize_multi(value: Any) -> tuple[str, ...]:
    """Normalize a list-valued facet (e.g. ``topics``) to a tuple of keys."""
    if value is None:
        return ()
    if isinstance(value, str):
        items: Iterable[Any] = [value]
    elif isinstance(value, np.ndarray):
        items = value.tolist()
    elif isinstance(value, Iterable):
        items = value
    else:
        items = [value]

    normalized = []
    for item in items:
        key = _normalize_scalar(item)
        if key is not None:
            normalized.append(key)
    return tuple(dict.fromkeys(normalized))


@dataclass(frozen=True)
class FacetConstraint:
    """A single requirement the instruction places on a retrieved document.

    Attributes:
        field: Dataset column holding the facet.
        relation: ``"same"`` to require a match, ``"different"`` to forbid one.
        multi_valued: Whether the column holds a list. For list facets "same"
            means *shares at least one value*, and "different" means *shares
            none* -- partial overlap counts as a match, so ``different`` is the
            strict complement of ``same`` on every axis.
    """

    field: str
    relation: FacetRelation
    multi_valued: bool = False

    def describe(self) -> str:
        """Human-readable rendering, e.g. ``same lcc_code``."""
        return f"{self.relation} {self.field}"


@dataclass(frozen=True)
class InstructionSpec:
    """An instruction plus the relevance definition it implies.

    Attributes:
        task_name: Registry task name this instruction backs.
        instruction: Natural-language instruction given to the model.
        anchor: Facets the retrieved document must share with the query.
        contrast: Facets it must differ on.
        description: Human-readable summary for reports.
        query_template: How the instruction is joined to the query text.
    """

    task_name: str
    instruction: str
    anchor: tuple[FacetConstraint, ...]
    contrast: tuple[FacetConstraint, ...]
    description: str
    query_template: str = DEFAULT_QUERY_TEMPLATE

    def __post_init__(self) -> None:
        if not self.anchor:
            raise ValueError(
                f"{self.task_name}: an instruction needs at least one anchor "
                "constraint, otherwise every corpus document is relevant"
            )
        for constraint in self.anchor:
            if constraint.relation != "same":
                raise ValueError(
                    f"{self.task_name}: anchor constraints must use "
                    f"relation='same', got {constraint.relation!r}"
                )
        for constraint in self.contrast:
            if constraint.relation != "different":
                raise ValueError(
                    f"{self.task_name}: contrast constraints must use "
                    f"relation='different', got {constraint.relation!r}"
                )

    @property
    def constraints(self) -> tuple[FacetConstraint, ...]:
        """All constraints, anchors first."""
        return self.anchor + self.contrast

    @property
    def required_fields(self) -> tuple[str, ...]:
        """Dataset columns this instruction reads."""
        return tuple(dict.fromkeys(c.field for c in self.constraints))

    def render(self, text: str) -> str:
        """Prefix a query document with the instruction."""
        return self.query_template.format(instruction=self.instruction, text=text)

    def supported_by(self, columns: Iterable[str]) -> bool:
        """Whether a dataset carrying ``columns`` can express this instruction."""
        available = set(columns)
        return all(field_name in available for field_name in self.required_fields)


# ---------------------------------------------------------------------------
# Instruction catalogue
# ---------------------------------------------------------------------------

_SPECS: tuple[InstructionSpec, ...] = (
    InstructionSpec(
        task_name="instruction_same_subject_diff_form",
        instruction=(
            "Given a document, retrieve documents about the same subject area "
            "but written in a different genre or form."
        ),
        anchor=(FacetConstraint("lcc_code", "same"),),
        contrast=(FacetConstraint("lcgft_form", "different"),),
        description=(
            "Same LCC class, different LCGFT form. The complement of "
            "instruction_same_form_diff_subject on the same queries. Its "
            "contrast clause is cheap to satisfy -- 132 of 133 forms differ -- "
            "so it doubles as the control that shows how much of a score comes "
            "from the anchor alone."
        ),
    ),
    InstructionSpec(
        task_name="instruction_same_form_diff_subject",
        instruction=(
            "Given a document, retrieve documents in the same genre or form "
            "but about a different subject area."
        ),
        anchor=(FacetConstraint("lcgft_form", "same"),),
        contrast=(FacetConstraint("lcc_code", "different"),),
        description=(
            "Same LCGFT form, different LCC class. The hard direction: the "
            "anchor is the axis embedders are worst at (form) and the contrast "
            "forbids the axis they default to (subject), so a similarity-only "
            "retriever is pulled straight into the excluded set."
        ),
    ),
    InstructionSpec(
        task_name="instruction_same_topic_diff_subject",
        instruction=(
            "Given a document, retrieve documents that discuss at least one of "
            "the same topics but are classified in a different subject area."
        ),
        anchor=(FacetConstraint("topics", "same", multi_valued=True),),
        contrast=(FacetConstraint("lcc_code", "different"),),
        description=(
            "Shares at least one LCSH topic, different LCC class. Topics appear "
            "verbatim in 87% of bodies, so this is the variant most exposed to a "
            "lexical shortcut on the anchor -- and the contrast is what a "
            "lexical model cannot express."
        ),
    ),
    InstructionSpec(
        task_name="instruction_same_audience_diff_register",
        instruction=(
            "Given a document, retrieve documents written for the same intended "
            "audience but in a different writing register."
        ),
        anchor=(FacetConstraint("audience", "same"),),
        contrast=(FacetConstraint("register", "different"),),
        description=(
            "Same audience, different register. Both axes are stylistic rather "
            "than topical, so subject similarity -- the signal that dominates "
            "every other SHELF retrieval task -- is uninformative here."
        ),
    ),
)

#: Instruction specifications keyed by registry task name.
INSTRUCTION_SPECS: dict[str, InstructionSpec] = {
    spec.task_name: spec for spec in _SPECS
}


def get_instruction(task_name: str) -> InstructionSpec | None:
    """Return the instruction backing ``task_name``, or None if it has none.

    Returning None rather than raising keeps the retrieval evaluator's single
    code path: an ordinary retrieval task simply has no instruction.
    """
    return INSTRUCTION_SPECS.get(task_name)


def is_instruction_task(task_name: str) -> bool:
    """Whether ``task_name`` is an instruction-conditioned retrieval task."""
    return task_name in INSTRUCTION_SPECS


def list_instruction_tasks() -> list[str]:
    """Sorted names of the instruction-conditioned retrieval tasks."""
    return sorted(INSTRUCTION_SPECS)


# ---------------------------------------------------------------------------
# Judging
# ---------------------------------------------------------------------------


@dataclass
class InstructionJudgments:
    """Instruction-defined qrels plus the constraint diagnostics.

    Attributes:
        relevance: query_id -> set of relevant corpus document IDs.
        metrics: Aggregate ``anchor_match@k`` / ``contrast_violation@k`` /
            ``contrast_violation_lift@k`` values.
        per_query: Per-query diagnostics, when requested.
        num_queries: Queries with a non-empty relevant set.
        num_empty: Queries whose instruction has no answer in this corpus.
    """

    relevance: dict[str, set[str]] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    per_query: dict[str, dict[str, float]] = field(default_factory=dict)
    num_queries: int = 0
    num_empty: int = 0


class _SingleValuedFacet:
    """Integer-coded corpus column supporting O(n) equality masks."""

    def __init__(self, values: Sequence[Any]):
        lookup: dict[str, int] = {}
        codes = np.full(len(values), -1, dtype=np.int32)
        for i, raw in enumerate(values):
            key = _normalize_scalar(raw)
            if key is None:
                continue
            code = lookup.get(key)
            if code is None:
                code = len(lookup)
                lookup[key] = code
            codes[i] = code
        self._lookup = lookup
        self._codes = codes

    def mask(self, value: Any) -> np.ndarray:
        """Boolean mask of corpus documents sharing ``value``."""
        key = _normalize_scalar(value)
        if key is None:
            return np.zeros(self._codes.shape[0], dtype=bool)
        code = self._lookup.get(key)
        if code is None:
            return np.zeros(self._codes.shape[0], dtype=bool)
        return self._codes == code


class _MultiValuedFacet:
    """Per-value membership masks for a list column such as ``topics``."""

    def __init__(self, values: Sequence[Any]):
        n = len(values)
        membership: dict[str, np.ndarray] = {}
        for i, raw in enumerate(values):
            for key in _normalize_multi(raw):
                column = membership.get(key)
                if column is None:
                    column = np.zeros(n, dtype=bool)
                    membership[key] = column
                column[i] = True
        self._membership = membership
        self._size = n

    def mask(self, value: Any) -> np.ndarray:
        """Boolean mask of corpus documents sharing at least one value."""
        result = np.zeros(self._size, dtype=bool)
        for key in _normalize_multi(value):
            column = self._membership.get(key)
            if column is not None:
                result |= column
        return result


class InstructionJudge:
    """Builds instruction-defined relevance over a fixed corpus.

    The corpus is indexed once and reused across every query, so judging
    8,500 queries against 34,000 documents is a handful of vectorized
    comparisons per query rather than a 289-million-pair Python loop.

    Example:
        judge = InstructionJudge(spec, corpus_rows)
        judgments = judge.judge(query_rows, results, k_values=[10])
    """

    def __init__(
        self,
        spec: InstructionSpec,
        corpus_rows: Sequence[Mapping[str, Any]],
        id_field: str = "id",
    ):
        """Index a corpus for one instruction.

        Args:
            spec: The instruction to judge against.
            corpus_rows: Corpus documents as mappings (e.g. polars ``iter_rows``).
            id_field: Column holding the document ID.

        Raises:
            ValueError: If the corpus lacks a column the instruction needs.
        """
        self.spec = spec
        self.id_field = id_field

        rows = list(corpus_rows)
        self._ids = np.array([str(row[id_field]) for row in rows], dtype=object)
        self._index = {doc_id: i for i, doc_id in enumerate(self._ids)}

        missing = [
            name
            for name in spec.required_fields
            if rows and name not in rows[0]  # noqa: SIM118 - Mapping, not dict
        ]
        if missing:
            raise ValueError(
                f"corpus is missing columns required by {spec.task_name}: {missing}"
            )

        self._facets: dict[str, _SingleValuedFacet | _MultiValuedFacet] = {}
        for constraint in spec.constraints:
            if constraint.field in self._facets:
                continue
            column = [row.get(constraint.field) for row in rows]
            self._facets[constraint.field] = (
                _MultiValuedFacet(column)
                if constraint.multi_valued
                else _SingleValuedFacet(column)
            )

    @property
    def corpus_size(self) -> int:
        """Number of indexed corpus documents."""
        return int(self._ids.shape[0])

    def anchor_mask(self, query_row: Mapping[str, Any]) -> np.ndarray:
        """Corpus documents sharing every anchor facet with the query."""
        mask = np.ones(self.corpus_size, dtype=bool)
        for constraint in self.spec.anchor:
            mask &= self._facets[constraint.field].mask(query_row.get(constraint.field))
        return mask

    def contrast_violation_mask(self, query_row: Mapping[str, Any]) -> np.ndarray:
        """Corpus documents the instruction asked to exclude.

        A document violates the contrast if it matches the query on *any*
        contrast facet, regardless of whether it satisfies the anchor. That is
        deliberate: the diagnostic question is "did the run return what the
        instruction forbade", and a same-subject document is forbidden by
        "different subject area" whether or not it also matches on form.
        """
        mask = np.zeros(self.corpus_size, dtype=bool)
        for constraint in self.spec.contrast:
            mask |= self._facets[constraint.field].mask(query_row.get(constraint.field))
        return mask

    def relevant_mask(self, query_row: Mapping[str, Any]) -> np.ndarray:
        """Corpus documents satisfying the instruction."""
        return self.anchor_mask(query_row) & ~self.contrast_violation_mask(query_row)

    def relevant_ids(self, query_row: Mapping[str, Any]) -> set[str]:
        """Instruction-defined relevant document IDs for one query."""
        return set(self._ids[self.relevant_mask(query_row)].tolist())

    def judge(
        self,
        query_rows: Sequence[Mapping[str, Any]],
        results: Mapping[str, Sequence[str]] | None = None,
        k_values: Sequence[int] | None = None,
        compute_per_query: bool = False,
    ) -> InstructionJudgments:
        """Judge every query, and score a run's rankings if one is supplied.

        Args:
            query_rows: Query documents as mappings.
            results: query_id -> ranked document IDs. Omit to build qrels only.
            k_values: Cutoffs for the constraint diagnostics.
            compute_per_query: Whether to keep per-query diagnostics.

        Returns:
            An :class:`InstructionJudgments` with qrels and diagnostics.
        """
        cutoffs = list(k_values) if k_values else [10]

        judgments = InstructionJudgments()
        anchor_hits: dict[int, list[float]] = {k: [] for k in cutoffs}
        violations: dict[int, list[float]] = {k: [] for k in cutoffs}
        violation_chance: list[float] = []

        for row in query_rows:
            query_id = str(row[self.id_field])
            anchor = self.anchor_mask(row)
            violated = self.contrast_violation_mask(row)
            relevant = anchor & ~violated

            relevant_ids = set(self._ids[relevant].tolist())
            judgments.relevance[query_id] = relevant_ids
            if relevant_ids:
                judgments.num_queries += 1
            else:
                # No document in this corpus satisfies the instruction for this
                # query. Counting it is how a mis-specified instruction becomes
                # visible instead of quietly shrinking the query set.
                judgments.num_empty += 1

            if results is None or not relevant_ids:
                continue

            ranked = results.get(query_id)
            if ranked is None:
                continue

            chance = float(violated.mean()) if self.corpus_size else 0.0
            violation_chance.append(chance)

            positions = [
                self._index[doc_id] for doc_id in ranked if doc_id in self._index
            ]
            query_metrics: dict[str, float] = {"contrast_violation_chance": chance}
            for k in cutoffs:
                top = positions[:k]
                if not top:
                    continue
                anchor_rate = float(anchor[top].mean())
                violation_rate = float(violated[top].mean())
                anchor_hits[k].append(anchor_rate)
                violations[k].append(violation_rate)
                query_metrics[f"anchor_match@{k}"] = anchor_rate
                query_metrics[f"contrast_violation@{k}"] = violation_rate

            if compute_per_query:
                judgments.per_query[query_id] = query_metrics

        mean_chance = float(np.mean(violation_chance)) if violation_chance else 0.0
        judgments.metrics["contrast_violation_chance"] = mean_chance
        for k in cutoffs:
            anchor_rate = float(np.mean(anchor_hits[k])) if anchor_hits[k] else 0.0
            violation_rate = float(np.mean(violations[k])) if violations[k] else 0.0
            judgments.metrics[f"anchor_match@{k}"] = anchor_rate
            judgments.metrics[f"contrast_violation@{k}"] = violation_rate
            # Lift against a random ranking. Without it a violation rate is
            # uninterpretable: on the v0.3.1 corpus a top-10 violation rate of
            # 0.05 is about chance for "different LCC class" (~0.048) but about
            # six times chance for "different LCGFT form" (~0.0086).
            judgments.metrics[f"contrast_violation_lift@{k}"] = (
                violation_rate / mean_chance if mean_chance > 0.0 else 0.0
            )

        return judgments
