"""Tests for stratum-balanced pair mining."""

from __future__ import annotations

import pytest
from shelf.evaluate.strata import RelationStratum, classify_relation
from shelf.hub.hard_negatives import (
    balanced_quotas,
    document_facets,
    mine_stratified_pairs,
)

CAT_INSTRUCTIONAL = "Instructional and educational works"
CAT_LITERATURE = "Literature"
CAT_LAW = "Law materials"


def doc(
    doc_id: str,
    lcc_code: str,
    form: str,
    category: str,
    topics: tuple[str, ...] = ("Art",),
) -> dict:
    return {
        "id": doc_id,
        "title": f"title-{doc_id}",
        "body": f"body-{doc_id}",
        "lcc_code": lcc_code,
        "lcgft_form": form,
        "lcgft_category": category,
        "topics": list(topics),
    }


@pytest.fixture
def corpus() -> list[dict]:
    """A small corpus with every reachable stratum represented."""
    docs: list[dict] = []
    classes = ["Q", "K", "P", "T"]
    forms = [
        ("Lectures", CAT_INSTRUCTIONAL),
        ("Textbooks", CAT_INSTRUCTIONAL),
        ("Poetry", CAT_LITERATURE),
        ("Briefs", CAT_LAW),
    ]
    topics = [("Art",), ("Ethics",), ("Music",), ("Art", "Ethics")]
    n = 0
    for cls in classes:
        for form, cat in forms:
            for topic in topics:
                for _ in range(3):
                    docs.append(doc(f"d{n:04d}", cls, form, cat, topic))
                    n += 1
    return docs


class TestDocumentFacets:
    def test_extracts_all_fields(self):
        f = document_facets(doc("x", "Q", "Lectures", CAT_INSTRUCTIONAL, ("Art",)))
        assert f.lcc_code == "Q"
        assert f.lcgft_form == "Lectures"
        assert f.lcgft_category == CAT_INSTRUCTIONAL
        assert f.topics == ("Art",)
        assert f.lcc_subclass is None

    def test_handles_missing_and_null_fields(self):
        f = document_facets({"id": "x", "topics": None})
        assert f.lcc_code == ""
        assert f.topics == ()

    def test_reads_subclass_when_present(self):
        d = doc("x", "Q", "Lectures", CAT_INSTRUCTIONAL)
        d["lcc_subclass"] = "QA"
        assert document_facets(d).lcc_subclass == "QA"


class TestBalancedQuotas:
    def test_positives_equal_negatives(self):
        q = balanced_quotas(4000)
        pos = q[RelationStratum.S2_SAME_CLASS]
        neg = sum(v for k, v in q.items() if k is not RelationStratum.S2_SAME_CLASS)
        assert pos == neg == 2000

    def test_does_not_exceed_total(self):
        for total in (0, 1, 7, 100, 4001):
            assert sum(balanced_quotas(total).values()) <= total

    def test_rejects_negative_total(self):
        with pytest.raises(ValueError, match="non-negative"):
            balanced_quotas(-1)

    def test_rejects_empty_negative_strata(self):
        with pytest.raises(ValueError, match="must not be empty"):
            balanced_quotas(100, negative_strata=())


class TestMining:
    def test_every_emitted_pair_is_in_its_claimed_stratum(self, corpus):
        quotas = {
            RelationStratum.S2_SAME_CLASS: 20,
            RelationStratum.S3_SAME_FORM_ONLY: 20,
            RelationStratum.S4_SAME_CATEGORY_ONLY: 20,
            RelationStratum.S5_SAME_TOPIC_ONLY: 20,
        }
        pairs, _ = mine_stratified_pairs(corpus, "lcc_code", quotas, seed=42)
        by_id = {d["id"]: d for d in corpus}
        for pair in pairs:
            rel = classify_relation(
                document_facets(by_id[pair["doc_a_id"]]),
                document_facets(by_id[pair["doc_b_id"]]),
            )
            assert rel.stratum.value == pair["relation_stratum"]

    def test_quotas_are_met(self, corpus):
        quotas = {
            RelationStratum.S2_SAME_CLASS: 15,
            RelationStratum.S3_SAME_FORM_ONLY: 15,
        }
        _, report = mine_stratified_pairs(corpus, "lcc_code", quotas, seed=1)
        assert report.is_complete
        assert report.produced[RelationStratum.S2_SAME_CLASS] == 15
        assert report.produced[RelationStratum.S3_SAME_FORM_ONLY] == 15

    def test_unreachable_stratum_reports_shortfall_and_terminates(self, corpus):
        """S1 needs subclass metadata the corpus lacks; must not hang."""
        quotas = {RelationStratum.S1_SAME_SUBCLASS: 10}
        pairs, report = mine_stratified_pairs(corpus, "lcc_code", quotas, seed=1)
        assert pairs == []
        assert not report.is_complete
        assert report.shortfalls == {RelationStratum.S1_SAME_SUBCLASS: 10}

    def test_labels_derive_from_label_field(self, corpus):
        quotas = {
            RelationStratum.S2_SAME_CLASS: 10,
            RelationStratum.S3_SAME_FORM_ONLY: 10,
        }
        pairs, _ = mine_stratified_pairs(corpus, "lcc_code", quotas, seed=7)
        by_id = {d["id"]: d for d in corpus}
        for pair in pairs:
            same = (
                by_id[pair["doc_a_id"]]["lcc_code"]
                == by_id[pair["doc_b_id"]]["lcc_code"]
            )
            assert pair["label"] == int(same)

    def test_same_class_stratum_is_always_positive(self, corpus):
        quotas = {RelationStratum.S2_SAME_CLASS: 20}
        pairs, _ = mine_stratified_pairs(corpus, "lcc_code", quotas, seed=3)
        assert all(p["label"] == 1 for p in pairs)

    def test_cross_class_strata_are_always_negative(self, corpus):
        quotas = {
            RelationStratum.S3_SAME_FORM_ONLY: 20,
            RelationStratum.S4_SAME_CATEGORY_ONLY: 20,
            RelationStratum.S5_SAME_TOPIC_ONLY: 20,
        }
        pairs, _ = mine_stratified_pairs(corpus, "lcc_code", quotas, seed=3)
        assert all(p["label"] == 0 for p in pairs)

    def test_no_duplicate_document_pairs(self, corpus):
        quotas = {RelationStratum.S2_SAME_CLASS: 40}
        pairs, _ = mine_stratified_pairs(corpus, "lcc_code", quotas, seed=11)
        keys = {tuple(sorted((p["doc_a_id"], p["doc_b_id"]))) for p in pairs}
        assert len(keys) == len(pairs)

    def test_no_self_pairs(self, corpus):
        quotas = {RelationStratum.S2_SAME_CLASS: 40}
        pairs, _ = mine_stratified_pairs(corpus, "lcc_code", quotas, seed=11)
        assert all(p["doc_a_id"] != p["doc_b_id"] for p in pairs)

    def test_ids_are_unique_and_contiguous(self, corpus):
        quotas = {
            RelationStratum.S2_SAME_CLASS: 10,
            RelationStratum.S3_SAME_FORM_ONLY: 10,
        }
        pairs, _ = mine_stratified_pairs(corpus, "lcc_code", quotas, seed=5)
        ids = [p["id"] for p in pairs]
        assert len(set(ids)) == len(ids)
        assert ids == [f"pair_{i:06d}" for i in range(len(pairs))]

    def test_deterministic_under_seed(self, corpus):
        quotas = {RelationStratum.S2_SAME_CLASS: 20}
        a, _ = mine_stratified_pairs(corpus, "lcc_code", quotas, seed=99)
        b, _ = mine_stratified_pairs(corpus, "lcc_code", quotas, seed=99)
        assert [(p["doc_a_id"], p["doc_b_id"]) for p in a] == [
            (p["doc_a_id"], p["doc_b_id"]) for p in b
        ]

    def test_different_seeds_differ(self, corpus):
        quotas = {RelationStratum.S2_SAME_CLASS: 20}
        a, _ = mine_stratified_pairs(corpus, "lcc_code", quotas, seed=1)
        b, _ = mine_stratified_pairs(corpus, "lcc_code", quotas, seed=2)
        assert [p["doc_a_id"] for p in a] != [p["doc_a_id"] for p in b]

    def test_schema_matches_generate_pairs(self, corpus):
        quotas = {RelationStratum.S2_SAME_CLASS: 5}
        pairs, _ = mine_stratified_pairs(corpus, "lcc_code", quotas, seed=1)
        required = {
            "id",
            "doc_a_id",
            "doc_a_title",
            "doc_a_body",
            "doc_b_id",
            "doc_b_title",
            "doc_b_body",
            "label",
            "label_field",
        }
        assert required <= set(pairs[0])
        assert pairs[0]["relation_stratum"]

    def test_zero_quota_is_skipped(self, corpus):
        quotas = {
            RelationStratum.S2_SAME_CLASS: 0,
            RelationStratum.S3_SAME_FORM_ONLY: 5,
        }
        pairs, report = mine_stratified_pairs(corpus, "lcc_code", quotas, seed=1)
        assert len(pairs) == 5
        assert report.is_complete

    def test_default_quotas_used_when_none(self, corpus):
        pairs, report = mine_stratified_pairs(corpus, "lcc_code", None, seed=1)
        assert set(report.requested) == {
            RelationStratum.S2_SAME_CLASS,
            RelationStratum.S3_SAME_FORM_ONLY,
            RelationStratum.S4_SAME_CATEGORY_ONLY,
            RelationStratum.S5_SAME_TOPIC_ONLY,
        }

    def test_rejects_too_few_documents(self):
        with pytest.raises(ValueError, match="at least 2 documents"):
            mine_stratified_pairs([doc("a", "Q", "L", CAT_INSTRUCTIONAL)], "lcc_code")

    def test_rejects_negative_quota(self, corpus):
        with pytest.raises(ValueError, match="negative"):
            mine_stratified_pairs(
                corpus, "lcc_code", {RelationStratum.S2_SAME_CLASS: -1}
            )

    def test_alternate_label_field(self, corpus):
        quotas = {RelationStratum.S3_SAME_FORM_ONLY: 10}
        pairs, _ = mine_stratified_pairs(corpus, "lcgft_form", quotas, seed=4)
        # S3 means same form, so under lcgft_form every pair is positive.
        assert all(p["label"] == 1 for p in pairs)
        assert all(p["label_field"] == "lcgft_form" for p in pairs)
