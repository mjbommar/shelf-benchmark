"""Tests for relation and difficulty strata."""

from __future__ import annotations

import numpy as np
import pytest
from shelf.evaluate.strata import (
    DifficultyBand,
    DocumentFacets,
    FormRelation,
    RelationStratum,
    SubjectRelation,
    classify_relation,
    difficulty_from_margins,
    margin_from_probabilities,
    stratum_distribution,
)

CATEGORY_INSTRUCTIONAL = "Instructional and educational works"
CATEGORY_LITERATURE = "Literature"


def facets(
    lcc_code: str = "Q",
    form: str = "Lectures",
    category: str = CATEGORY_INSTRUCTIONAL,
    topics: tuple[str, ...] = ("Mathematics",),
    subclass: str | None = None,
) -> DocumentFacets:
    return DocumentFacets(
        lcc_code=lcc_code,
        lcgft_form=form,
        lcgft_category=category,
        topics=topics,
        lcc_subclass=subclass,
    )


class TestSubjectAxis:
    def test_same_subclass_detected(self):
        a = facets(subclass="QA")
        b = facets(subclass="QA", topics=("Physics",))
        assert classify_relation(a, b).subject_level is SubjectRelation.SAME_SUBCLASS

    def test_same_class_different_subclass(self):
        a = facets(subclass="QA")
        b = facets(subclass="QC")
        assert classify_relation(a, b).subject_level is SubjectRelation.SAME_CLASS

    def test_different_class(self):
        a = facets(lcc_code="Q", subclass="QA")
        b = facets(lcc_code="K", subclass="KF")
        assert classify_relation(a, b).subject_level is SubjectRelation.DIFFERENT

    def test_missing_subclass_degrades_to_class_level(self):
        """v0.3.1 has no subclass; the axis must not claim SAME_SUBCLASS."""
        a = facets(subclass=None)
        b = facets(subclass=None)
        rel = classify_relation(a, b)
        assert rel.subject_level is SubjectRelation.SAME_CLASS
        assert rel.stratum is not RelationStratum.S0_IDENTICAL_FACETS

    def test_one_sided_subclass_does_not_claim_same_subclass(self):
        a = facets(subclass="QA")
        b = facets(subclass=None)
        assert classify_relation(a, b).subject_level is SubjectRelation.SAME_CLASS


class TestFormAxis:
    def test_same_form(self):
        a, b = facets(), facets(topics=("Physics",))
        assert classify_relation(a, b).form_level is FormRelation.SAME_FORM

    def test_same_category_different_form(self):
        a = facets(form="Lectures")
        b = facets(form="Textbooks")
        assert classify_relation(a, b).form_level is FormRelation.SAME_CATEGORY

    def test_different_category(self):
        a = facets(form="Lectures", category=CATEGORY_INSTRUCTIONAL)
        b = facets(form="Poetry", category=CATEGORY_LITERATURE)
        assert classify_relation(a, b).form_level is FormRelation.DIFFERENT


class TestTopicOverlap:
    def test_jaccard_full_overlap(self):
        a = facets(topics=("Art", "Music"))
        b = facets(topics=("Music", "Art"))
        assert classify_relation(a, b).topic_jaccard == pytest.approx(1.0)

    def test_jaccard_partial_overlap(self):
        a = facets(topics=("Art", "Music"))
        b = facets(topics=("Music", "Ethics"))
        rel = classify_relation(a, b)
        assert rel.topic_jaccard == pytest.approx(1 / 3)
        assert rel.shared_topics == frozenset({"music"})

    def test_jaccard_no_overlap(self):
        a = facets(topics=("Art",))
        b = facets(topics=("Ethics",))
        rel = classify_relation(a, b)
        assert rel.topic_jaccard == 0.0
        assert not rel.shares_any_topic

    def test_empty_topics_do_not_divide_by_zero(self):
        a = facets(topics=())
        b = facets(topics=())
        assert classify_relation(a, b).topic_jaccard == 0.0

    def test_topic_matching_is_case_and_space_insensitive(self):
        a = facets(topics=("  Art ",))
        b = facets(topics=("art",))
        assert classify_relation(a, b).topic_jaccard == pytest.approx(1.0)


class TestStratumLadder:
    def test_s0_requires_all_three_facets_identical(self):
        a = facets(subclass="QA", form="Lectures", topics=("Art",))
        b = facets(subclass="QA", form="Lectures", topics=("Art",))
        assert classify_relation(a, b).stratum is RelationStratum.S0_IDENTICAL_FACETS

    def test_s1_same_subclass_differing_form(self):
        a = facets(subclass="QA", form="Lectures")
        b = facets(subclass="QA", form="Textbooks")
        assert classify_relation(a, b).stratum is RelationStratum.S1_SAME_SUBCLASS

    def test_s1_same_subclass_differing_topics(self):
        a = facets(subclass="QA", topics=("Art",))
        b = facets(subclass="QA", topics=("Ethics",))
        assert classify_relation(a, b).stratum is RelationStratum.S1_SAME_SUBCLASS

    def test_s2_same_class(self):
        a = facets(subclass="QA")
        b = facets(subclass="QC")
        assert classify_relation(a, b).stratum is RelationStratum.S2_SAME_CLASS

    def test_s3_same_form_only(self):
        a = facets(lcc_code="Q", subclass="QA", form="Lectures", topics=("Art",))
        b = facets(lcc_code="K", subclass="KF", form="Lectures", topics=("Ethics",))
        assert classify_relation(a, b).stratum is RelationStratum.S3_SAME_FORM_ONLY

    def test_s4_same_category_only(self):
        a = facets(lcc_code="Q", subclass="QA", form="Lectures", topics=("Art",))
        b = facets(lcc_code="K", subclass="KF", form="Textbooks", topics=("Ethics",))
        assert classify_relation(a, b).stratum is RelationStratum.S4_SAME_CATEGORY_ONLY

    def test_s5_same_topic_only(self):
        a = facets(
            lcc_code="Q",
            subclass="QA",
            form="Lectures",
            category=CATEGORY_INSTRUCTIONAL,
            topics=("Art",),
        )
        b = facets(
            lcc_code="K",
            subclass="KF",
            form="Poetry",
            category=CATEGORY_LITERATURE,
            topics=("Art",),
        )
        assert classify_relation(a, b).stratum is RelationStratum.S5_SAME_TOPIC_ONLY

    def test_s6_unrelated(self):
        a = facets(
            lcc_code="Q",
            subclass="QA",
            form="Lectures",
            category=CATEGORY_INSTRUCTIONAL,
            topics=("Art",),
        )
        b = facets(
            lcc_code="K",
            subclass="KF",
            form="Poetry",
            category=CATEGORY_LITERATURE,
            topics=("Ethics",),
        )
        assert classify_relation(a, b).stratum is RelationStratum.S6_UNRELATED

    def test_subject_dominates_form(self):
        """Same class beats same form when the two axes disagree."""
        a = facets(lcc_code="Q", subclass="QA", form="Lectures")
        b = facets(lcc_code="Q", subclass="QC", form="Poetry", category="Literature")
        assert classify_relation(a, b).stratum is RelationStratum.S2_SAME_CLASS

    def test_form_beats_category(self):
        """Same form is finer than same category and must rank closer."""
        a = facets(lcc_code="Q", subclass="QA", form="Lectures")
        b = facets(lcc_code="K", subclass="KF", form="Lectures")
        rel = classify_relation(a, b)
        assert rel.stratum is RelationStratum.S3_SAME_FORM_ONLY

    def test_relation_is_symmetric(self):
        a = facets(lcc_code="Q", subclass="QA", form="Lectures", topics=("Art",))
        b = facets(lcc_code="K", subclass="KF", form="Poetry", topics=("Art", "Ethics"))
        forward = classify_relation(a, b)
        backward = classify_relation(b, a)
        assert forward.stratum is backward.stratum
        assert forward.subject_level is backward.subject_level
        assert forward.form_level is backward.form_level
        assert forward.topic_jaccard == pytest.approx(backward.topic_jaccard)

    def test_ladder_is_totally_ordered_and_complete(self):
        """Every stratum value is reachable and the enum has no gaps."""
        assert len(RelationStratum) == 7
        names = [s.name for s in RelationStratum]
        assert names == sorted(names)


class TestStratumDistribution:
    def test_includes_empty_strata(self):
        a = facets(subclass="QA")
        rels = [classify_relation(a, a)]
        dist = stratum_distribution(rels)
        assert set(dist) == set(RelationStratum)
        assert dist[RelationStratum.S0_IDENTICAL_FACETS] == 1
        assert dist[RelationStratum.S6_UNRELATED] == 0

    def test_counts_sum_to_input_length(self):
        a = facets(lcc_code="Q", subclass="QA")
        b = facets(lcc_code="K", subclass="KF", form="Poetry", category="Literature")
        rels = [
            classify_relation(a, a),
            classify_relation(a, b),
            classify_relation(b, b),
        ]
        assert sum(stratum_distribution(rels).values()) == 3

    def test_empty_input(self):
        assert sum(stratum_distribution([]).values()) == 0


class TestMargin:
    def test_correct_prediction_gives_positive_margin(self):
        probs = np.array([[0.7, 0.2, 0.1]])
        assert margin_from_probabilities(probs, [0])[0] == pytest.approx(0.5)

    def test_incorrect_prediction_gives_negative_margin(self):
        probs = np.array([[0.7, 0.2, 0.1]])
        assert margin_from_probabilities(probs, [1])[0] == pytest.approx(-0.5)

    def test_tie_gives_zero_margin(self):
        probs = np.array([[0.5, 0.5]])
        assert margin_from_probabilities(probs, [0])[0] == pytest.approx(0.0)

    def test_single_class_uses_full_mass(self):
        probs = np.array([[1.0]])
        assert margin_from_probabilities(probs, [0])[0] == pytest.approx(1.0)

    def test_multiple_rows(self):
        probs = np.array([[0.9, 0.1], [0.1, 0.9], [0.5, 0.5]])
        margins = margin_from_probabilities(probs, [0, 0, 1])
        assert margins == pytest.approx([0.8, -0.8, 0.0])

    def test_rejects_1d_probabilities(self):
        with pytest.raises(ValueError, match="2-D"):
            margin_from_probabilities(np.array([0.5, 0.5]), [0])

    def test_rejects_length_mismatch(self):
        with pytest.raises(ValueError, match="entries"):
            margin_from_probabilities(np.array([[0.5, 0.5]]), [0, 1])

    def test_rejects_out_of_range_index(self):
        with pytest.raises(ValueError, match="out-of-range"):
            margin_from_probabilities(np.array([[0.5, 0.5]]), [5])


class TestDifficultyBands:
    def test_terciles_populate_every_band(self):
        margins = np.linspace(0.01, 1.0, 300)
        bands = difficulty_from_margins(margins)
        assert set(bands) == {
            DifficultyBand.HARD,
            DifficultyBand.MEDIUM,
            DifficultyBand.EASY,
        }

    def test_negative_margins_are_always_hard(self):
        """Baseline errors stay HARD even when they dominate the distribution."""
        margins = [-0.9, -0.8, -0.7, -0.6, 0.95, 0.99]
        bands = difficulty_from_margins(margins)
        assert bands[:4] == [DifficultyBand.HARD] * 4

    def test_ordering_is_monotone(self):
        margins = np.linspace(0.01, 1.0, 99)
        bands = difficulty_from_margins(margins)
        rank = {
            DifficultyBand.HARD: 0,
            DifficultyBand.MEDIUM: 1,
            DifficultyBand.EASY: 2,
        }
        ranks = [rank[b] for b in bands]
        assert ranks == sorted(ranks)

    def test_output_length_matches_input(self):
        assert len(difficulty_from_margins([0.1, 0.2, 0.3, 0.4])) == 4

    def test_empty_input(self):
        assert difficulty_from_margins([]) == []

    def test_all_identical_margins_do_not_crash(self):
        bands = difficulty_from_margins([0.5] * 10)
        assert len(bands) == 10

    def test_custom_quantiles(self):
        margins = np.linspace(0.01, 1.0, 100)
        bands = difficulty_from_margins(margins, hard_quantile=0.1, easy_quantile=0.9)
        assert bands.count(DifficultyBand.HARD) < bands.count(DifficultyBand.MEDIUM)

    def test_rejects_inverted_quantiles(self):
        with pytest.raises(ValueError, match="hard_quantile"):
            difficulty_from_margins([0.1, 0.2], hard_quantile=0.8, easy_quantile=0.2)

    def test_rejects_out_of_range_quantiles(self):
        with pytest.raises(ValueError, match="hard_quantile"):
            difficulty_from_margins([0.1], hard_quantile=-0.1, easy_quantile=0.5)
