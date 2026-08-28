"""Tests for G6 near-duplicate detection (shelf.qc.dedup)."""

from __future__ import annotations

import numpy as np
import pytest
from shelf.qc.dedup import (
    MinHasher,
    NearDuplicateIndex,
    jaccard_estimate,
    scan_corpus,
)
from shelf.qc.gates import Gate

BASE_TEXT = (
    "Solar panels convert sunlight directly into electricity through the "
    "photovoltaic effect, a process that has become increasingly efficient "
    "and affordable over the past two decades of manufacturing innovation, "
    "driven by improvements in silicon purity, cell architecture, and large "
    "scale production techniques that have steadily lowered the cost per "
    "watt for residential and utility scale installations around the world."
)

# A near-identical realization of the same "spec" -- the Phase 1 scenario
# where 15 generators write the same content brief and produce
# near-duplicates by construction. A single word changed in an otherwise
# long, shared passage: word-level 5-gram shingling means even one
# substitution touches several shingles, so the fixture uses a long shared
# span to keep the resulting similarity realistically high (~0.88) rather
# than picking a similarity value out of thin air.
NEAR_DUPLICATE_TEXT = BASE_TEXT.replace("increasingly", "remarkably")

# A second near-duplicate realization with a different single-word swap, so
# it is still close to both BASE_TEXT and NEAR_DUPLICATE_TEXT (a three-way
# cluster from one shared spec).
NEAR_DUPLICATE_TEXT_2 = BASE_TEXT.replace("efficient", "effective")

UNRELATED_TEXT = (
    "The migratory patterns of Arctic terns span both hemispheres each year, "
    "making them one of the longest traveling species on the planet by a "
    "wide margin, covering tens of thousands of kilometers annually as they "
    "move between polar breeding grounds and distant feeding areas in a "
    "remarkable annual cycle of endurance and navigation."
)


class TestMinHasher:
    def test_signature_shape(self):
        hasher = MinHasher(num_perm=64)
        sig = hasher.signature(BASE_TEXT)
        assert sig.shape == (64,)
        assert sig.dtype == np.int64

    def test_identical_text_has_identical_signature(self):
        hasher = MinHasher(num_perm=64)
        assert np.array_equal(hasher.signature(BASE_TEXT), hasher.signature(BASE_TEXT))

    def test_deterministic_across_instances_with_same_seed(self):
        a = MinHasher(num_perm=32, seed=7)
        b = MinHasher(num_perm=32, seed=7)
        assert np.array_equal(a.signature(BASE_TEXT), b.signature(BASE_TEXT))

    def test_different_seed_gives_different_hash_functions(self):
        a = MinHasher(num_perm=32, seed=7)
        b = MinHasher(num_perm=32, seed=99)
        assert not np.array_equal(a.signature(BASE_TEXT), b.signature(BASE_TEXT))

    def test_empty_text_signature_is_stable(self):
        hasher = MinHasher(num_perm=16)
        sig1 = hasher.signature("")
        sig2 = hasher.signature("")
        assert np.array_equal(sig1, sig2)

    def test_short_text_shorter_than_shingle_size_does_not_crash(self):
        hasher = MinHasher(num_perm=16, shingle_size=5)
        sig = hasher.signature("two words")
        assert sig.shape == (16,)


class TestJaccardEstimate:
    def test_identical_signatures_have_similarity_one(self):
        hasher = MinHasher(num_perm=64)
        sig = hasher.signature(BASE_TEXT)
        assert jaccard_estimate(sig, sig) == 1.0

    def test_near_duplicate_scores_high_similarity(self):
        hasher = MinHasher(num_perm=128)
        sim = jaccard_estimate(
            hasher.signature(BASE_TEXT), hasher.signature(NEAR_DUPLICATE_TEXT)
        )
        assert sim >= 0.8

    def test_unrelated_text_scores_low_similarity(self):
        hasher = MinHasher(num_perm=128)
        sim = jaccard_estimate(
            hasher.signature(BASE_TEXT), hasher.signature(UNRELATED_TEXT)
        )
        assert sim < 0.3

    def test_mismatched_shapes_return_zero(self):
        a = np.zeros(4, dtype=np.int64)
        b = np.zeros(8, dtype=np.int64)
        assert jaccard_estimate(a, b) == 0.0


class TestNearDuplicateIndex:
    def test_first_occurrence_passes(self):
        index = NearDuplicateIndex(threshold=0.9)
        result = index.add("doc-1", BASE_TEXT)
        assert result.gate is Gate.NEAR_DUPLICATE
        assert result.passed is True

    def test_near_duplicate_of_existing_doc_fails(self):
        index = NearDuplicateIndex(threshold=0.85, num_perm=128, bands=32)
        index.add("doc-1", BASE_TEXT)
        result = index.add("doc-2", NEAR_DUPLICATE_TEXT)
        assert result.passed is False
        assert result.detail["matched_id"] == "doc-1"
        assert result.detail["similarity"] >= 0.85

    def test_unrelated_document_passes(self):
        index = NearDuplicateIndex(threshold=0.9)
        index.add("doc-1", BASE_TEXT)
        result = index.add("doc-2", UNRELATED_TEXT)
        assert result.passed is True

    def test_rejects_num_perm_not_divisible_by_bands(self):
        with pytest.raises(ValueError):
            NearDuplicateIndex(num_perm=100, bands=32)

    def test_exact_duplicate_text_always_flagged(self):
        index = NearDuplicateIndex(threshold=0.9)
        index.add("doc-1", BASE_TEXT)
        result = index.add("doc-2", BASE_TEXT)
        assert result.passed is False
        assert result.value == 1.0

    def test_len_reflects_documents_added(self):
        index = NearDuplicateIndex()
        assert len(index) == 0
        index.add("doc-1", BASE_TEXT)
        index.add("doc-2", UNRELATED_TEXT)
        assert len(index) == 2

    def test_best_match_does_not_mutate_index(self):
        index = NearDuplicateIndex(threshold=0.9)
        index.add("doc-1", BASE_TEXT)
        match = index.best_match(NEAR_DUPLICATE_TEXT)
        assert match is not None
        assert match.matched_id == "doc-1"
        assert len(index) == 1  # query-only, not inserted

    def test_best_match_none_when_index_empty(self):
        index = NearDuplicateIndex()
        assert index.best_match(BASE_TEXT) is None

    def test_find_all_duplicate_pairs(self):
        index = NearDuplicateIndex(threshold=0.85, num_perm=128, bands=32)
        index.add("doc-1", BASE_TEXT)
        index.add("doc-2", NEAR_DUPLICATE_TEXT)
        index.add("doc-3", UNRELATED_TEXT)
        pairs = index.find_all_duplicate_pairs()
        assert len(pairs) == 1
        a, b, sim = pairs[0]
        assert {a, b} == {"doc-1", "doc-2"}
        assert sim >= 0.85

    def test_cluster_of_three_near_duplicates_all_pair_up(self):
        index = NearDuplicateIndex(threshold=0.7, num_perm=128, bands=32)
        index.add("doc-1", BASE_TEXT)
        index.add("doc-2", NEAR_DUPLICATE_TEXT)
        index.add("doc-3", NEAR_DUPLICATE_TEXT_2)
        pairs = index.find_all_duplicate_pairs()
        involved = {doc_id for pair in pairs for doc_id in pair[:2]}
        assert involved == {"doc-1", "doc-2", "doc-3"}


class TestScanCorpus:
    def test_scores_every_document_in_order(self):
        ids = ["a", "b", "c"]
        texts = [BASE_TEXT, NEAR_DUPLICATE_TEXT, UNRELATED_TEXT]
        index, results = scan_corpus(ids, texts, threshold=0.85)
        assert set(results) == set(ids)
        assert results["a"].passed is True
        assert results["b"].passed is False
        assert results["c"].passed is True
        assert len(index) == 3

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            scan_corpus(["a", "b"], ["only one text"])


class TestScalability:
    def test_handles_a_few_thousand_documents_quickly(self):
        """Not a strict perf assertion (machine-dependent) -- just a smoke
        test that the vectorized signature path handles a corpus-shaped
        batch without pathological blowup."""
        import time

        rng = np.random.default_rng(0)
        vocab = BASE_TEXT.split() + UNRELATED_TEXT.split()
        texts = [" ".join(rng.choice(vocab, size=60)) for _ in range(2000)]
        start = time.monotonic()
        index, results = scan_corpus(
            [f"doc-{i}" for i in range(len(texts))], texts, threshold=0.9
        )
        elapsed = time.monotonic() - start
        assert len(results) == 2000
        assert elapsed < 30  # generous ceiling; real run reports actual timing
