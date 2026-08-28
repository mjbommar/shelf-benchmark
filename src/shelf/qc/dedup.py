"""G6 -- near-duplicate detection at corpus scale (§12 of the data plan).

Phase 1 of the v0.4 data plan hands the *same spec* to every generator, so
15 generators write near-duplicates of each other by construction. G6 is the
gate that keeps that design honest: without it, a spec with a distinctive
phrase would enter the corpus 15 times over.

``datasketch`` is not a declared dependency, so this module implements
MinHash and banded LSH from scratch using stdlib ``hashlib`` for token
hashing and ``numpy`` (already a dependency) to vectorize the permutation
hashing that would otherwise be a Python-level loop over
``num_perm x num_shingles`` per document.

Approach:

1. Tokenize each document into lowercase word shingles (default: 5-grams).
2. Hash each shingle to a 31-bit integer with BLAKE2b (stable across runs,
   unlike Python's salted ``hash()``).
3. Compute a MinHash signature per document: for ``num_perm`` independent
   ``(a * x + b) mod p`` hash functions, take the minimum hashed shingle
   value. This is a standard unbiased estimator of Jaccard similarity
   between shingle sets.
4. Band the signature into ``bands`` groups of rows for LSH: documents that
   collide in *any* band are candidates. This turns all-pairs comparison
   (O(n^2)) into O(n) hashing plus small-bucket comparisons, which is what
   makes G6 tractable at corpus scale.
5. Candidates are verified against the exact signature-based similarity
   estimate before being reported as near-duplicates, so the banding
   parameters only affect recall/speed, never correctness of the final
   threshold check.

Example:
    from shelf.qc.dedup import NearDuplicateIndex

    index = NearDuplicateIndex(threshold=0.9)
    result = index.add("doc-1", "Solar panels convert sunlight into electricity...")
    result.passed  # True: first occurrence, nothing to collide with yet

    result2 = index.add("doc-2", "Solar panels convert sunlight into electricity...")
    result2.passed  # False: near-duplicate of doc-1
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from shelf.qc.gates import Gate, GateResult

__all__ = [
    "DuplicateMatch",
    "MinHasher",
    "NearDuplicateIndex",
    "NearDuplicateStats",
    "jaccard_estimate",
]

# A prime just under 2^31. Shingle hashes and the (a, b) coefficients are all
# kept below this bound so that `a * x` fits comfortably in an int64
# (< 2^62), letting the whole signature computation run as vectorized numpy
# arithmetic instead of a per-shingle Python loop.
_PRIME = (1 << 31) - 1

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def _hash_token(token: str) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % _PRIME


def _shingle_hashes(text: str, shingle_size: int) -> np.ndarray:
    tokens = _tokenize(text)
    if not tokens:
        return np.empty(0, dtype=np.int64)
    if len(tokens) < shingle_size:
        grams = [" ".join(tokens)]
    else:
        grams = [
            " ".join(tokens[i : i + shingle_size])
            for i in range(len(tokens) - shingle_size + 1)
        ]
    return np.array([_hash_token(g) for g in grams], dtype=np.int64)


class MinHasher:
    """Computes MinHash signatures for word-shingled text.

    Signatures are ``num_perm``-length int64 arrays; the fraction of
    matching positions between two signatures is an unbiased estimator of
    the Jaccard similarity of the underlying shingle sets.
    """

    def __init__(
        self,
        num_perm: int = 128,
        shingle_size: int = 5,
        seed: int = 42,
    ) -> None:
        self.num_perm = num_perm
        self.shingle_size = shingle_size
        rng = np.random.RandomState(seed)
        # a in [1, PRIME); b in [0, PRIME) -- standard universal hash family.
        self._a = rng.randint(1, _PRIME, size=num_perm, dtype=np.int64)
        self._b = rng.randint(0, _PRIME, size=num_perm, dtype=np.int64)

    def signature(self, text: str) -> np.ndarray:
        """Compute the MinHash signature for one document."""
        shingles = _shingle_hashes(text, self.shingle_size)
        if shingles.size == 0:
            # No content: every permutation "minimum" is the max possible
            # value, so an empty document only matches other empty ones.
            return np.full(self.num_perm, _PRIME - 1, dtype=np.int64)
        # (S, 1) x (1, P) -> (S, P), vectorized in numpy (C loop, not Python).
        hashed = (np.outer(shingles, self._a) + self._b) % _PRIME
        return hashed.min(axis=0)


def jaccard_estimate(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """Estimated Jaccard similarity between two MinHash signatures."""
    if sig_a.shape != sig_b.shape or sig_a.size == 0:
        return 0.0
    return float(np.count_nonzero(sig_a == sig_b)) / sig_a.size


@dataclass(frozen=True)
class DuplicateMatch:
    """One near-duplicate hit found while adding a document to the index."""

    matched_id: str
    similarity: float


@dataclass
class NearDuplicateStats:
    """Aggregate G6 statistics over a batch of documents."""

    total: int = 0
    duplicates: int = 0
    pairs: list[tuple[str, str, float]] = field(default_factory=list)

    @property
    def duplicate_rate(self) -> float:
        return self.duplicates / self.total if self.total else 0.0


class NearDuplicateIndex:
    """Incremental near-duplicate index (G6) using MinHash + banded LSH.

    Documents are added in order; each ``add`` call checks the new document
    against every document already in the index and returns a
    :class:`~shelf.qc.gates.GateResult` -- the first occurrence of a spec
    passes, later near-duplicate realizations fail. This mirrors how the
    gate must behave in the generation pipeline ("every generated document
    passes through these before entering the corpus") and also serves as
    the batch/corpus-validation entry point: call ``add`` for every document
    in a fixed order (e.g. by ``id``) to reproduce the full corpus's
    near-duplicate structure.

    LSH banding parameters (``bands`` groups of ``num_perm / bands`` rows
    each) only affect which candidate pairs get *considered*; every
    candidate is re-checked against the exact signature-based Jaccard
    estimate before being reported, so correctness does not depend on the
    banding choice, only recall/speed does. The defaults are recall-heavy
    (many small bands) since false-candidate verification is cheap and
    missed near-duplicates are the more expensive failure mode.
    """

    def __init__(
        self,
        threshold: float = 0.9,
        num_perm: int = 128,
        bands: int = 32,
        shingle_size: int = 5,
        seed: int = 42,
    ) -> None:
        if num_perm % bands != 0:
            raise ValueError("num_perm must be divisible by bands")
        self.threshold = threshold
        self.bands = bands
        self.rows = num_perm // bands
        self._hasher = MinHasher(
            num_perm=num_perm, shingle_size=shingle_size, seed=seed
        )
        self._signatures: dict[str, np.ndarray] = {}
        self._buckets: list[dict[bytes, list[str]]] = [
            defaultdict(list) for _ in range(bands)
        ]

    def _band_keys(self, signature: np.ndarray) -> list[bytes]:
        bands = signature.reshape(self.bands, self.rows)
        return [band.tobytes() for band in bands]

    def _candidates(self, band_keys: list[bytes]) -> set[str]:
        candidates: set[str] = set()
        for band_index, key in enumerate(band_keys):
            candidates.update(self._buckets[band_index].get(key, ()))
        return candidates

    def best_match(self, text: str) -> DuplicateMatch | None:
        """Return the closest existing match for ``text``, if any exists."""
        signature = self._hasher.signature(text)
        band_keys = self._band_keys(signature)
        candidates = self._candidates(band_keys)
        best: DuplicateMatch | None = None
        for candidate_id in candidates:
            sim = jaccard_estimate(signature, self._signatures[candidate_id])
            if best is None or sim > best.similarity:
                best = DuplicateMatch(candidate_id, sim)
        return best

    def add(self, doc_id: str, text: str) -> GateResult:
        """Score and insert one document, in corpus order.

        Returns a G6 :class:`~shelf.qc.gates.GateResult`: ``passed=False``
        with the matched id and similarity in ``detail`` if ``text`` is a
        near-duplicate (``similarity >= threshold``) of a document already
        in the index; the document is still added to the index either way
        so that clusters of near-duplicates are all detected against the
        first (retained) member.
        """
        signature = self._hasher.signature(text)
        band_keys = self._band_keys(signature)
        candidates = self._candidates(band_keys)

        best: DuplicateMatch | None = None
        for candidate_id in candidates:
            sim = jaccard_estimate(signature, self._signatures[candidate_id])
            if best is None or sim > best.similarity:
                best = DuplicateMatch(candidate_id, sim)

        self._signatures[doc_id] = signature
        for band_index, key in enumerate(band_keys):
            self._buckets[band_index][key].append(doc_id)

        is_duplicate = best is not None and best.similarity >= self.threshold
        detail: dict[str, Any] = {}
        if best is not None:
            detail = {"matched_id": best.matched_id, "similarity": best.similarity}
        return GateResult(
            Gate.NEAR_DUPLICATE,
            not is_duplicate,
            value=best.similarity if best is not None else 0.0,
            detail=detail,
        )

    def find_all_duplicate_pairs(self) -> list[tuple[str, str, float]]:
        """All near-duplicate pairs currently in the index (``sim >= threshold``).

        More expensive than :meth:`add`'s incremental checks (each bucket's
        members are compared pairwise), but useful for a post hoc audit of
        an already-built corpus, e.g. for
        :func:`shelf.qc.promotion.run_promotion_checks`.
        """
        seen_pairs: set[tuple[str, str]] = set()
        results: list[tuple[str, str, float]] = []
        for bucket in self._buckets:
            for members in bucket.values():
                if len(members) < 2:
                    continue
                for i in range(len(members)):
                    for j in range(i + 1, len(members)):
                        a, b = sorted((members[i], members[j]))
                        if (a, b) in seen_pairs:
                            continue
                        seen_pairs.add((a, b))
                        sim = jaccard_estimate(self._signatures[a], self._signatures[b])
                        if sim >= self.threshold:
                            results.append((a, b, sim))
        return results

    def __len__(self) -> int:
        return len(self._signatures)


def scan_corpus(
    ids: list[str],
    texts: list[str],
    *,
    threshold: float = 0.9,
    num_perm: int = 128,
    bands: int = 32,
    shingle_size: int = 5,
    seed: int = 42,
) -> tuple[NearDuplicateIndex, dict[str, GateResult]]:
    """Run G6 over a full corpus in order, returning per-document results.

    Convenience wrapper for batch/validation use (e.g. scoring the
    published corpus) so callers do not have to drive
    :class:`NearDuplicateIndex` by hand.

    Args:
        ids: Document identifiers, in the order they should be considered
            "generated" (first occurrence of a near-duplicate cluster wins).
        texts: Document bodies, same order/length as ``ids``.
        threshold, num_perm, bands, shingle_size, seed: Forwarded to
            :class:`NearDuplicateIndex`.

    Returns:
        The populated index (for further queries, e.g.
        :meth:`NearDuplicateIndex.find_all_duplicate_pairs`) and a mapping
        of ``doc_id -> GateResult``.
    """
    if len(ids) != len(texts):
        raise ValueError("ids and texts must be the same length")
    index = NearDuplicateIndex(
        threshold=threshold,
        num_perm=num_perm,
        bands=bands,
        shingle_size=shingle_size,
        seed=seed,
    )
    results = {
        doc_id: index.add(doc_id, text) for doc_id, text in zip(ids, texts, strict=True)
    }
    return index, results
