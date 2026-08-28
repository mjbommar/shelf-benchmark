"""Unit tests for instruction-conditioned retrieval.

Covers the instruction catalogue, the constraint semantics that define
relevance, and the constraint diagnostics (anchor match / contrast violation)
that say *why* a run failed rather than only how much.
"""

from __future__ import annotations

from typing import Any

import pytest
from shelf.evaluate.instructions import (
    INSTRUCTION_SPECS,
    FacetConstraint,
    InstructionJudge,
    InstructionSpec,
    get_instruction,
    is_instruction_task,
    list_instruction_tasks,
)
from shelf.evaluate.registry import TASK_REGISTRY, get_task
from shelf.evaluate.tasks import TaskType

# ===========================================================================
# Fixtures
# ===========================================================================


def _doc(
    doc_id: str,
    lcc: str,
    form: str,
    category: str = "Discursive works",
    topics: tuple[str, ...] = (),
    audience: str = "General",
    register: str = "academic",
) -> dict[str, Any]:
    return {
        "id": doc_id,
        "lcc_code": lcc,
        "lcgft_form": form,
        "lcgft_category": category,
        "topics": list(topics),
        "audience": audience,
        "register": register,
    }


@pytest.fixture
def corpus() -> list[dict[str, Any]]:
    """Small factorial corpus: subject x form are independent."""
    return [
        _doc("d1", "Q", "Lectures", topics=("Physics",)),
        _doc("d2", "Q", "Jokes", topics=("Physics", "Humor")),
        _doc("d3", "K", "Lectures", topics=("Law",)),
        _doc("d4", "K", "Jokes", topics=("Humor",)),
        _doc("d5", "N", "Maps", topics=("Art",)),
    ]


@pytest.fixture
def query() -> dict[str, Any]:
    """A science lecture about physics."""
    return _doc("q1", "Q", "Lectures", topics=("Physics",))


@pytest.fixture
def same_subject_spec() -> InstructionSpec:
    return INSTRUCTION_SPECS["instruction_same_subject_diff_form"]


@pytest.fixture
def same_form_spec() -> InstructionSpec:
    return INSTRUCTION_SPECS["instruction_same_form_diff_subject"]


# ===========================================================================
# Catalogue
# ===========================================================================


class TestInstructionCatalogue:
    @pytest.mark.unit
    def test_every_spec_has_a_registered_task(self):
        """Every instruction must back a real retrieval task."""
        for name, spec in INSTRUCTION_SPECS.items():
            assert spec.task_name == name
            assert name in TASK_REGISTRY
            assert get_task(name).task_type == TaskType.RETRIEVAL

    @pytest.mark.unit
    def test_list_instruction_tasks_sorted(self):
        names = list_instruction_tasks()
        assert names == sorted(names)
        assert len(names) == len(INSTRUCTION_SPECS)

    @pytest.mark.unit
    def test_is_instruction_task_discriminates(self):
        assert is_instruction_task("instruction_same_form_diff_subject")
        assert not is_instruction_task("lcc_retrieval")

    @pytest.mark.unit
    def test_get_instruction_returns_none_for_plain_task(self):
        """A label-match task has no instruction, and that is not an error."""
        assert get_instruction("lcc_retrieval") is None

    @pytest.mark.unit
    def test_complementary_pair_exists(self):
        """The same/different pair is what makes the task non-trivial."""
        a = INSTRUCTION_SPECS["instruction_same_subject_diff_form"]
        b = INSTRUCTION_SPECS["instruction_same_form_diff_subject"]
        assert a.anchor[0].field == b.contrast[0].field
        assert a.contrast[0].field == b.anchor[0].field

    @pytest.mark.unit
    def test_render_includes_instruction_and_text(self, same_form_spec):
        rendered = same_form_spec.render("hello world")
        assert same_form_spec.instruction in rendered
        assert rendered.endswith("hello world")

    @pytest.mark.unit
    def test_required_fields(self, same_form_spec):
        assert set(same_form_spec.required_fields) == {"lcgft_form", "lcc_code"}

    @pytest.mark.unit
    def test_supported_by(self, same_form_spec):
        assert same_form_spec.supported_by({"lcgft_form", "lcc_code", "text"})
        assert not same_form_spec.supported_by({"lcc_code"})


class TestInstructionSpecValidation:
    @pytest.mark.unit
    def test_anchor_required(self):
        with pytest.raises(ValueError, match="at least one anchor"):
            InstructionSpec(
                task_name="bad",
                instruction="x",
                anchor=(),
                contrast=(FacetConstraint("lcc_code", "different"),),
                description="",
            )

    @pytest.mark.unit
    def test_anchor_must_be_same(self):
        with pytest.raises(ValueError, match="relation='same'"):
            InstructionSpec(
                task_name="bad",
                instruction="x",
                anchor=(FacetConstraint("lcc_code", "different"),),
                contrast=(),
                description="",
            )

    @pytest.mark.unit
    def test_contrast_must_be_different(self):
        with pytest.raises(ValueError, match="relation='different'"):
            InstructionSpec(
                task_name="bad",
                instruction="x",
                anchor=(FacetConstraint("lcc_code", "same"),),
                contrast=(FacetConstraint("lcgft_form", "same"),),
                description="",
            )

    @pytest.mark.unit
    def test_constraint_describe(self):
        assert FacetConstraint("lcc_code", "same").describe() == "same lcc_code"


# ===========================================================================
# Relevance semantics
# ===========================================================================


class TestInstructionRelevance:
    @pytest.mark.unit
    def test_same_subject_diff_form(self, same_subject_spec, corpus, query):
        judge = InstructionJudge(same_subject_spec, corpus)
        # Q docs other than the query's own form: d2 (Q/Jokes). d1 is Q/Lectures
        # and is excluded by the "different form" clause.
        assert judge.relevant_ids(query) == {"d2"}

    @pytest.mark.unit
    def test_same_form_diff_subject(self, same_form_spec, corpus, query):
        judge = InstructionJudge(same_form_spec, corpus)
        # Lectures outside class Q: d3 (K/Lectures).
        assert judge.relevant_ids(query) == {"d3"}

    @pytest.mark.unit
    def test_complementary_instructions_disagree(
        self, same_subject_spec, same_form_spec, corpus, query
    ):
        """The whole point: one query, two instructions, disjoint answers."""
        a = InstructionJudge(same_subject_spec, corpus).relevant_ids(query)
        b = InstructionJudge(same_form_spec, corpus).relevant_ids(query)
        assert a and b
        assert a.isdisjoint(b)

    @pytest.mark.unit
    def test_multi_valued_anchor_matches_on_overlap(self, corpus, query):
        spec = INSTRUCTION_SPECS["instruction_same_topic_diff_subject"]
        judge = InstructionJudge(spec, corpus)
        # Query topics = {Physics}. d2 shares Physics but is class Q (excluded).
        # No non-Q document shares Physics, so the answer set is empty.
        assert judge.relevant_ids(query) == set()

        humor_query = _doc("q2", "P", "Essays", topics=("Humor",))
        # d2 (Q) and d4 (K) both carry Humor and neither is class P.
        assert judge.relevant_ids(humor_query) == {"d2", "d4"}

    @pytest.mark.unit
    def test_missing_anchor_value_yields_no_relevance(self, corpus):
        """A blank facet cannot satisfy 'same' -- it is missing, not a value."""
        spec = INSTRUCTION_SPECS["instruction_same_audience_diff_register"]
        judge = InstructionJudge(spec, corpus)
        blank = _doc("q3", "Q", "Lectures", audience="   ")
        assert judge.relevant_ids(blank) == set()

    @pytest.mark.unit
    def test_unknown_anchor_value_yields_no_relevance(self, same_form_spec, corpus):
        unknown = _doc("q4", "Q", "Cookbooks")
        judge = InstructionJudge(same_form_spec, corpus)
        assert judge.relevant_ids(unknown) == set()

    @pytest.mark.unit
    def test_case_insensitive_matching(self, same_form_spec, corpus):
        judge = InstructionJudge(same_form_spec, corpus)
        upper = _doc("q5", "Q", "LECTURES")
        assert judge.relevant_ids(upper) == {"d3"}

    @pytest.mark.unit
    def test_missing_column_raises(self, same_form_spec):
        with pytest.raises(ValueError, match="missing columns"):
            InstructionJudge(same_form_spec, [{"id": "d1", "lcc_code": "Q"}])

    @pytest.mark.unit
    def test_corpus_size(self, same_form_spec, corpus):
        assert InstructionJudge(same_form_spec, corpus).corpus_size == 5


# ===========================================================================
# Constraint diagnostics
# ===========================================================================


class TestConstraintDiagnostics:
    @pytest.mark.unit
    def test_perfect_ranking(self, same_form_spec, corpus, query):
        judge = InstructionJudge(same_form_spec, corpus)
        judgments = judge.judge([query], {"q1": ["d3", "d5"]}, k_values=[2])

        assert judgments.num_queries == 1
        assert judgments.num_empty == 0
        # d3 is Lectures (anchor hit), d5 is Maps (anchor miss); neither is Q.
        assert judgments.metrics["anchor_match@2"] == pytest.approx(0.5)
        assert judgments.metrics["contrast_violation@2"] == pytest.approx(0.0)
        assert judgments.metrics["contrast_violation_lift@2"] == pytest.approx(0.0)

    @pytest.mark.unit
    def test_instruction_ignoring_ranking_shows_violation_lift(
        self, same_form_spec, corpus, query
    ):
        """A subject-similarity ranking returns exactly what was forbidden."""
        judge = InstructionJudge(same_form_spec, corpus)
        judgments = judge.judge([query], {"q1": ["d1", "d2"]}, k_values=[2])

        assert judgments.metrics["contrast_violation@2"] == pytest.approx(1.0)
        # 2 of 5 corpus docs are class Q, so chance is 0.4 and the lift is 2.5x.
        assert judgments.metrics["contrast_violation_chance"] == pytest.approx(0.4)
        assert judgments.metrics["contrast_violation_lift@2"] == pytest.approx(2.5)

    @pytest.mark.unit
    def test_violation_counts_anchor_misses_too(self, same_form_spec, corpus, query):
        """A forbidden document is forbidden whether or not it hits the anchor."""
        judge = InstructionJudge(same_form_spec, corpus)
        # d2 is class Q (violates) but form Jokes (misses the anchor).
        judgments = judge.judge([query], {"q1": ["d2"]}, k_values=[1])
        assert judgments.metrics["contrast_violation@1"] == pytest.approx(1.0)
        assert judgments.metrics["anchor_match@1"] == pytest.approx(0.0)

    @pytest.mark.unit
    def test_queries_without_answer_are_counted(self, same_form_spec, corpus):
        judge = InstructionJudge(same_form_spec, corpus)
        unanswerable = _doc("q9", "Q", "Cookbooks")
        judgments = judge.judge([unanswerable], {"q9": ["d1"]}, k_values=[1])
        assert judgments.num_queries == 0
        assert judgments.num_empty == 1
        assert judgments.relevance["q9"] == set()

    @pytest.mark.unit
    def test_per_query_breakdown(self, same_form_spec, corpus, query):
        judge = InstructionJudge(same_form_spec, corpus)
        judgments = judge.judge(
            [query], {"q1": ["d3"]}, k_values=[1], compute_per_query=True
        )
        assert "q1" in judgments.per_query
        assert judgments.per_query["q1"]["anchor_match@1"] == pytest.approx(1.0)

    @pytest.mark.unit
    def test_judge_without_results_builds_qrels_only(
        self, same_form_spec, corpus, query
    ):
        judge = InstructionJudge(same_form_spec, corpus)
        judgments = judge.judge([query])
        assert judgments.relevance["q1"] == {"d3"}
        assert judgments.metrics["contrast_violation@10"] == pytest.approx(0.0)
