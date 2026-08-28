"""Stratum-balanced pair mining for SHELF.

The pair tasks published in v0.3.1 are built by :func:`shelf.hub.dataset.generate_pairs`,
which draws negatives with ``random.sample(all_labels, 2)``. Measured against the
real corpus, that produces:

- positives: **100%** ``S2_same_class``
- negatives: **81%** ``S6_unrelated``

In other words the task is "same subject vs. completely unrelated", which any
model can approach with coarse topical similarity. Genuinely hard negatives --
two documents that share a *form*, a *category*, or a *topic* while differing in
subject -- make up only ~13.7% of random draws, so they are effectively absent.

Those pairs already exist in the corpus. This module mines them directly with
per-stratum quotas, so a pair set can be balanced across the relation ladder
without generating a single new document.

Mining is index-driven rather than rejection-driven at the top level: drawing
``S3_same_form_only`` pairs by rejection sampling would burn ~113 random draws
per hit. Instead documents are inverted by facet and candidates are drawn from
the relevant index, then verified against the true stratum.

Example:
    from shelf.hub.hard_negatives import mine_stratified_pairs
    from shelf.evaluate.strata import RelationStratum

    pairs = mine_stratified_pairs(
        documents,
        label_field="lcc_code",
        quotas={
            RelationStratum.S2_SAME_CLASS: 1000,
            RelationStratum.S3_SAME_FORM_ONLY: 1000,
            RelationStratum.S4_SAME_CATEGORY_ONLY: 1000,
            RelationStratum.S5_SAME_TOPIC_ONLY: 1000,
        },
        seed=42,
    )
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from shelf.evaluate.strata import (
    DocumentFacets,
    RelationStratum,
    classify_relation,
)

__all__ = [
    "MiningReport",
    "balanced_quotas",
    "document_facets",
    "mine_stratified_pairs",
]

# Attempts allowed per requested pair before a stratum is declared exhausted.
# Bounded so an impossible quota degrades to a short-count with a warning rather
# than hanging: S0/S1 are unreachable without subclass metadata, and a caller
# asking for them must get an answer, not a spin.
_ATTEMPTS_PER_PAIR = 60


@dataclass
class MiningReport:
    """What mining actually produced, versus what was asked for.

    Short counts are a sampling failure that must be visible. A pair set that
    silently contains 40 S3 pairs instead of the 1,000 requested would be
    reported as "stratum-balanced" while being nothing of the kind.
    """

    requested: dict[RelationStratum, int]
    produced: dict[RelationStratum, int]
    attempts: dict[RelationStratum, int]

    @property
    def shortfalls(self) -> dict[RelationStratum, int]:
        """Strata that could not be filled, and by how many."""
        return {
            stratum: count - self.produced.get(stratum, 0)
            for stratum, count in self.requested.items()
            if self.produced.get(stratum, 0) < count
        }

    @property
    def is_complete(self) -> bool:
        """Whether every requested quota was met exactly."""
        return not self.shortfalls


def document_facets(doc: dict[str, Any]) -> DocumentFacets:
    """Build :class:`DocumentFacets` from a SHELF document dict."""
    topics = doc.get("topics") or []
    return DocumentFacets(
        lcc_code=str(doc.get("lcc_code") or ""),
        lcgft_form=str(doc.get("lcgft_form") or ""),
        lcgft_category=str(doc.get("lcgft_category") or ""),
        topics=tuple(str(t) for t in topics if t),
        lcc_subclass=(str(doc["lcc_subclass"]) if doc.get("lcc_subclass") else None),
    )


def _build_indexes(
    documents: list[dict[str, Any]],
) -> tuple[
    dict[str, list[int]],
    dict[str, list[int]],
    dict[str, list[int]],
    dict[str, list[int]],
]:
    """Invert documents by subject, form, category, and topic."""
    by_class: dict[str, list[int]] = defaultdict(list)
    by_form: dict[str, list[int]] = defaultdict(list)
    by_category: dict[str, list[int]] = defaultdict(list)
    by_topic: dict[str, list[int]] = defaultdict(list)

    for i, doc in enumerate(documents):
        facets = document_facets(doc)
        if facets.lcc_code:
            by_class[facets.lcc_code].append(i)
        if facets.lcgft_form:
            by_form[facets.lcgft_form].append(i)
        if facets.lcgft_category:
            by_category[facets.lcgft_category].append(i)
        for topic in facets.topic_set():
            by_topic[topic].append(i)

    return dict(by_class), dict(by_form), dict(by_category), dict(by_topic)


def _index_for_stratum(
    stratum: RelationStratum,
    by_class: dict[str, list[int]],
    by_form: dict[str, list[int]],
    by_category: dict[str, list[int]],
    by_topic: dict[str, list[int]],
) -> dict[str, list[int]] | None:
    """Choose the inverted index that makes a stratum cheap to hit.

    Returns ``None`` for strata with no useful index, which fall back to
    uniform sampling over the whole corpus (correct for ``S6_unrelated``,
    which is the common case in a random draw).
    """
    if stratum in (
        RelationStratum.S0_IDENTICAL_FACETS,
        RelationStratum.S1_SAME_SUBCLASS,
        RelationStratum.S2_SAME_CLASS,
    ):
        return by_class
    if stratum is RelationStratum.S3_SAME_FORM_ONLY:
        return by_form
    if stratum is RelationStratum.S4_SAME_CATEGORY_ONLY:
        return by_category
    if stratum is RelationStratum.S5_SAME_TOPIC_ONLY:
        return by_topic
    return None


def _pair_record(
    pair_id: str,
    doc_a: dict[str, Any],
    doc_b: dict[str, Any],
    label: int,
    label_field: str,
    stratum: RelationStratum,
) -> dict[str, Any]:
    """Build a pair record matching the schema of ``generate_pairs``.

    The extra ``relation_stratum`` field is additive: existing consumers that
    read only the v0.3.1 keys are unaffected.
    """
    return {
        "id": pair_id,
        "doc_a_id": doc_a["id"],
        "doc_a_title": doc_a.get("title", ""),
        "doc_a_body": doc_a.get("body", ""),
        "doc_b_id": doc_b["id"],
        "doc_b_title": doc_b.get("title", ""),
        "doc_b_body": doc_b.get("body", ""),
        "label": label,
        "label_field": label_field,
        "relation_stratum": stratum.value,
    }


def balanced_quotas(
    total: int,
    negative_strata: tuple[RelationStratum, ...] = (
        RelationStratum.S3_SAME_FORM_ONLY,
        RelationStratum.S4_SAME_CATEGORY_ONLY,
        RelationStratum.S5_SAME_TOPIC_ONLY,
        RelationStratum.S6_UNRELATED,
    ),
    positive_stratum: RelationStratum = RelationStratum.S2_SAME_CLASS,
) -> dict[RelationStratum, int]:
    """Build quotas that yield a 50/50 positive/negative pair set.

    Quotas control class balance implicitly, which is an easy trap: asking for
    800 pairs in each of one positive and four negative strata yields a 20/80
    split, not the balanced set most pair metrics assume. This helper sizes the
    positive stratum to match the negative strata combined.

    Args:
        total: Approximate total number of pairs wanted.
        negative_strata: Strata treated as negatives, split evenly.
        positive_stratum: Stratum treated as the positive class.

    Returns:
        Quota mapping summing to at most ``total``.

    Raises:
        ValueError: If ``total`` is negative or ``negative_strata`` is empty.
    """
    if total < 0:
        raise ValueError(f"total must be non-negative, got {total}")
    if not negative_strata:
        raise ValueError("negative_strata must not be empty")

    half = total // 2
    per_negative = half // len(negative_strata)
    quotas = dict.fromkeys(negative_strata, per_negative)
    quotas[positive_stratum] = per_negative * len(negative_strata)
    return quotas


def mine_stratified_pairs(
    documents: list[dict[str, Any]],
    label_field: str = "lcc_code",
    quotas: dict[RelationStratum, int] | None = None,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], MiningReport]:
    """Mine document pairs with per-stratum quotas.

    The binary ``label`` is derived from ``label_field`` exactly as in
    :func:`shelf.hub.dataset.generate_pairs`, so a stratum-balanced set is a
    drop-in replacement for a randomly-sampled one. What changes is *which*
    negatives appear: instead of 81% trivially-unrelated pairs, the caller
    controls how many share a form, a category, or a topic.

    Args:
        documents: SHELF document dicts. Each needs ``id``, ``lcc_code``,
            ``lcgft_form``, ``lcgft_category``, and ``topics``.
        label_field: Field whose equality defines a positive pair.
        quotas: Requested count per stratum. Defaults to an equal split across
            the four strata reachable without subclass metadata.
        seed: Random seed.

    Returns:
        ``(pairs, report)``. Pairs are shuffled. The report records requested
        versus produced counts so shortfalls are explicit rather than silent.

    Note:
        Quotas determine class balance implicitly. Equal quotas across one
        positive and four negative strata give a 20/80 split, not 50/50 -- use
        :func:`balanced_quotas` when a balanced set is wanted.

    Raises:
        ValueError: If ``documents`` has fewer than two entries or a quota is
            negative.
    """
    if len(documents) < 2:
        raise ValueError(
            f"need at least 2 documents to form pairs, got {len(documents)}"
        )

    if quotas is None:
        quotas = {
            RelationStratum.S2_SAME_CLASS: 500,
            RelationStratum.S3_SAME_FORM_ONLY: 500,
            RelationStratum.S4_SAME_CATEGORY_ONLY: 500,
            RelationStratum.S5_SAME_TOPIC_ONLY: 500,
        }
    for stratum, count in quotas.items():
        if count < 0:
            raise ValueError(f"quota for {stratum.value} is negative: {count}")

    rng = random.Random(seed)
    by_class, by_form, by_category, by_topic = _build_indexes(documents)
    facet_cache = [document_facets(doc) for doc in documents]

    produced: Counter[RelationStratum] = Counter()
    attempts: Counter[RelationStratum] = Counter()
    seen: set[tuple[int, int]] = set()
    pairs: list[dict[str, Any]] = []

    for stratum, wanted in quotas.items():
        if wanted == 0:
            continue
        index = _index_for_stratum(stratum, by_class, by_form, by_category, by_topic)
        buckets = (
            [key for key, members in index.items() if len(members) >= 2]
            if index
            else []
        )
        budget = wanted * _ATTEMPTS_PER_PAIR

        while produced[stratum] < wanted and attempts[stratum] < budget:
            attempts[stratum] += 1

            if index is not None:
                if not buckets:
                    break
                members = index[rng.choice(buckets)]
                i, j = rng.sample(members, 2)
            else:
                i, j = rng.sample(range(len(documents)), 2)

            key = (i, j) if i < j else (j, i)
            if key in seen:
                continue
            if classify_relation(facet_cache[i], facet_cache[j]).stratum is not stratum:
                continue

            seen.add(key)
            doc_a, doc_b = documents[i], documents[j]
            label = int(doc_a.get(label_field) == doc_b.get(label_field))
            pairs.append(
                _pair_record(
                    f"pair_{len(pairs):06d}", doc_a, doc_b, label, label_field, stratum
                )
            )
            produced[stratum] += 1

    rng.shuffle(pairs)
    # Re-key after shuffling so ids are stable and contiguous for the emitted set.
    for position, pair in enumerate(pairs):
        pair["id"] = f"pair_{position:06d}"

    report = MiningReport(
        requested=dict(quotas),
        produced=dict(produced),
        attempts=dict(attempts),
    )
    return pairs, report
