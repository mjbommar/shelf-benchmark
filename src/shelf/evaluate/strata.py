"""Item strata for SHELF evaluation.

This module implements the two stratification axes required by the v0.4 data
plan (``docs/data_plan_v0.4.md`` sections 11.4 and 11.5):

**Relation strata** describe where a *pair* of documents sits on a bibliographic
relation ladder. SHELF's factorial design means "same label / different label"
throws away most of the available structure: two documents can share a subject
but not a form, share a form but not a subject, or share neither while still
sharing topics. Collapsing all of those into a single "negative" class makes a
pair task far easier than it should be and hides which relation a model actually
fails on.

**Difficulty strata** describe how hard a single item is *for a lexical
baseline*. SHELF's primary classification task is close to lexically saturated
-- a bag-of-words model recovers the LC class of a 22-word document 75% of the
time -- so an aggregate score can be dominated by items that require no document
understanding at all. Scoring every item by the margin of a lexical baseline
makes that shortcut measurable instead of merely suspected.

Both stratifications are deliberately *reproducible without human labelling*:
relation strata are derived from the taxonomy metadata already on every
document, and difficulty is derived from a fitted lexical baseline plus a fixed
seed.

Example:
    from shelf.evaluate.strata import (
        DocumentFacets,
        RelationStratum,
        classify_relation,
        difficulty_from_margins,
    )

    a = DocumentFacets(lcc_code="Q", lcc_subclass="QA", lcgft_form="Lectures",
                       lcgft_category="Instructional and educational works",
                       topics=("Mathematics", "Education"))
    b = DocumentFacets(lcc_code="Q", lcc_subclass="QC", lcgft_form="Lectures",
                       lcgft_category="Instructional and educational works",
                       topics=("Physics",))

    rel = classify_relation(a, b)
    rel.stratum          # RelationStratum.S2_SAME_CLASS
    rel.subject_level    # SubjectRelation.SAME_CLASS
    rel.topic_jaccard    # 0.0
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

__all__ = [
    "DifficultyBand",
    "DocumentFacets",
    "FormRelation",
    "PairRelation",
    "RelationStratum",
    "SubjectRelation",
    "classify_relation",
    "difficulty_from_margins",
    "margin_from_probabilities",
    "stratum_distribution",
]


# ---------------------------------------------------------------------------
# Relation strata
# ---------------------------------------------------------------------------


class SubjectRelation(str, Enum):
    """How two documents relate on the LCC subject axis."""

    SAME_SUBCLASS = "same_subclass"
    SAME_CLASS = "same_class"
    DIFFERENT = "different"


class FormRelation(str, Enum):
    """How two documents relate on the LCGFT genre/form axis."""

    SAME_FORM = "same_form"
    SAME_CATEGORY = "same_category"
    DIFFERENT = "different"


class RelationStratum(str, Enum):
    """Ordinal bibliographic relation ladder, closest first.

    The ladder is an ordered *summary* of the independent subject, form, and
    topic relations. It exists so results can be reported per stratum; the
    underlying :class:`PairRelation` retains the full structure and should be
    preferred when the axes need to be analysed separately.
    """

    S0_IDENTICAL_FACETS = "S0_identical_facets"
    S1_SAME_SUBCLASS = "S1_same_subclass"
    S2_SAME_CLASS = "S2_same_class"
    S3_SAME_FORM_ONLY = "S3_same_form_only"
    S4_SAME_CATEGORY_ONLY = "S4_same_category_only"
    S5_SAME_TOPIC_ONLY = "S5_same_topic_only"
    S6_UNRELATED = "S6_unrelated"


@dataclass(frozen=True)
class DocumentFacets:
    """The taxonomy facets of a single document.

    ``lcc_subclass`` is optional because the frozen v0.3.1 corpus carries only
    the 21 top-level LCC classes. When it is absent on either side, the subject
    axis degrades gracefully to class-level resolution and never reports
    :attr:`SubjectRelation.SAME_SUBCLASS`.
    """

    lcc_code: str
    lcgft_form: str
    lcgft_category: str
    topics: tuple[str, ...] = ()
    lcc_subclass: str | None = None

    def topic_set(self) -> frozenset[str]:
        """Return topics as a set, normalized for comparison."""
        return frozenset(t.strip().casefold() for t in self.topics if t and t.strip())


@dataclass(frozen=True)
class PairRelation:
    """The full facet relationship between two documents."""

    subject_level: SubjectRelation
    form_level: FormRelation
    topic_jaccard: float
    shared_topics: frozenset[str] = field(default_factory=frozenset)

    @property
    def shares_any_topic(self) -> bool:
        """Whether the two documents share at least one topic."""
        return len(self.shared_topics) > 0

    @property
    def stratum(self) -> RelationStratum:
        """Collapse the axes into the ordinal ladder.

        Precedence is deliberate and documented rather than incidental:

        1. Subject proximity dominates, because LCC subject is SHELF's primary
           label and the finest-grained axis available.
        2. Within "different subject", a shared *form* is treated as closer than
           a shared *category*, since LCGFT form (133-554 values) is strictly
           finer than category (14 values).
        3. A shared topic with no shared subject or form is the weakest
           non-empty relation.
        """
        if self.subject_level is SubjectRelation.SAME_SUBCLASS:
            if self.form_level is FormRelation.SAME_FORM and self.topic_jaccard == 1.0:
                return RelationStratum.S0_IDENTICAL_FACETS
            return RelationStratum.S1_SAME_SUBCLASS

        if self.subject_level is SubjectRelation.SAME_CLASS:
            return RelationStratum.S2_SAME_CLASS

        if self.form_level is FormRelation.SAME_FORM:
            return RelationStratum.S3_SAME_FORM_ONLY

        if self.form_level is FormRelation.SAME_CATEGORY:
            return RelationStratum.S4_SAME_CATEGORY_ONLY

        if self.shares_any_topic:
            return RelationStratum.S5_SAME_TOPIC_ONLY

        return RelationStratum.S6_UNRELATED


def _subject_relation(a: DocumentFacets, b: DocumentFacets) -> SubjectRelation:
    """Resolve the subject axis, degrading to class level without subclasses."""
    if a.lcc_subclass and b.lcc_subclass and a.lcc_subclass == b.lcc_subclass:
        return SubjectRelation.SAME_SUBCLASS
    if a.lcc_code and a.lcc_code == b.lcc_code:
        # Without subclass metadata a same-class pair is the finest resolution
        # available, so it is reported as SAME_CLASS rather than guessed upward.
        return SubjectRelation.SAME_CLASS
    return SubjectRelation.DIFFERENT


def _form_relation(a: DocumentFacets, b: DocumentFacets) -> FormRelation:
    """Resolve the genre/form axis."""
    if a.lcgft_form and a.lcgft_form == b.lcgft_form:
        return FormRelation.SAME_FORM
    if a.lcgft_category and a.lcgft_category == b.lcgft_category:
        return FormRelation.SAME_CATEGORY
    return FormRelation.DIFFERENT


def classify_relation(a: DocumentFacets, b: DocumentFacets) -> PairRelation:
    """Classify the bibliographic relation between two documents.

    Args:
        a: Facets of the first document.
        b: Facets of the second document.

    Returns:
        A :class:`PairRelation` carrying the independent subject, form, and
        topic relations plus the derived ordinal stratum.
    """
    topics_a = a.topic_set()
    topics_b = b.topic_set()
    shared = topics_a & topics_b
    union = topics_a | topics_b
    jaccard = len(shared) / len(union) if union else 0.0

    return PairRelation(
        subject_level=_subject_relation(a, b),
        form_level=_form_relation(a, b),
        topic_jaccard=jaccard,
        shared_topics=frozenset(shared),
    )


def stratum_distribution(
    relations: Sequence[PairRelation],
) -> dict[RelationStratum, int]:
    """Count relations by stratum, including strata with zero members.

    Zero-count strata are retained deliberately: a published item set that
    silently contains no S3 pairs is a sampling failure, and a distribution
    that omits the empty stratum hides it.
    """
    counts = dict.fromkeys(RelationStratum, 0)
    for relation in relations:
        counts[relation.stratum] += 1
    return counts


# ---------------------------------------------------------------------------
# Difficulty strata
# ---------------------------------------------------------------------------


class DifficultyBand(str, Enum):
    """Difficulty of an item relative to a lexical baseline."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


def margin_from_probabilities(
    probabilities: np.ndarray,
    true_indices: Sequence[int],
) -> np.ndarray:
    """Compute the decision margin of a probabilistic classifier per item.

    The margin is ``p(true) - max p(other)``. It is positive when the baseline
    is correct and negative when it is wrong, and its magnitude reflects
    confidence, so a single monotone quantity orders items from "trivially
    solved by vocabulary" to "the lexical baseline gets this wrong".

    Args:
        probabilities: Array of shape ``(n_items, n_classes)``.
        true_indices: Index of the true class for each item.

    Returns:
        Array of shape ``(n_items,)`` with the per-item margin.

    Raises:
        ValueError: If shapes are inconsistent or an index is out of range.
    """
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 2:
        raise ValueError(f"probabilities must be 2-D, got shape {probs.shape}")

    truth = np.asarray(true_indices, dtype=int)
    if truth.shape[0] != probs.shape[0]:
        raise ValueError(
            f"true_indices has {truth.shape[0]} entries but probabilities has "
            f"{probs.shape[0]} rows"
        )
    if truth.size and (truth.min() < 0 or truth.max() >= probs.shape[1]):
        raise ValueError("true_indices contains an out-of-range class index")

    rows = np.arange(probs.shape[0])
    true_prob = probs[rows, truth]

    masked = probs.copy()
    masked[rows, truth] = -np.inf
    best_other = masked.max(axis=1)
    # With a single class there is no competitor; the margin is the whole mass.
    best_other = np.where(np.isneginf(best_other), 0.0, best_other)

    return true_prob - best_other


def difficulty_from_margins(
    margins: Sequence[float] | np.ndarray,
    *,
    hard_quantile: float = 1 / 3,
    easy_quantile: float = 2 / 3,
) -> list[DifficultyBand]:
    """Assign difficulty bands by quantiles of a lexical-baseline margin.

    Items the lexical baseline resolves confidently are ``EASY``; items it gets
    wrong or resolves near its decision boundary are ``HARD``. Terciles are the
    default so every band is populated by construction.

    Any item with a negative margin -- one the baseline gets outright wrong --
    is always ``HARD`` regardless of quantile position. Without that floor, a
    corpus the baseline solves almost perfectly would relabel its own errors as
    "medium" purely because they are numerous enough to reach the middle
    tercile.

    Args:
        margins: Per-item margins, e.g. from :func:`margin_from_probabilities`.
        hard_quantile: Lower cut point in [0, 1].
        easy_quantile: Upper cut point in [0, 1].

    Returns:
        A difficulty band per item, in input order.

    Raises:
        ValueError: If the quantiles are not ``0 <= hard < easy <= 1``.
    """
    if not 0.0 <= hard_quantile < easy_quantile <= 1.0:
        raise ValueError(
            f"require 0 <= hard_quantile < easy_quantile <= 1, got "
            f"{hard_quantile} and {easy_quantile}"
        )

    values = np.asarray(margins, dtype=float)
    if values.size == 0:
        return []

    low = float(np.quantile(values, hard_quantile))
    high = float(np.quantile(values, easy_quantile))

    bands: list[DifficultyBand] = []
    for value in values:
        if value < 0.0 or value <= low:
            bands.append(DifficultyBand.HARD)
        elif value >= high:
            bands.append(DifficultyBand.EASY)
        else:
            bands.append(DifficultyBand.MEDIUM)
    return bands
