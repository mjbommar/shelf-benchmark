"""Tests for group-aware (spec-level) splitting.

Phase 1 hands the same spec to every generator, so a spec's realizations are
near-duplicates. Splitting them independently manufactures train/test leakage.
These tests pin the property that prevents it.
"""

from __future__ import annotations

import pytest
from shelf.hub.splitter import SplitConfig, StratifiedSplitter

GENERATORS = ("gpt", "claude", "gemini", "grok", "deepseek", "qwen", "glm", "llama")
CATEGORIES = ("Informational works", "Literature", "Law materials", "Music")
LCC_CODES = ("Q", "K", "P", "T", "H", "B")


def make_corpus(n_specs: int = 200, generators: tuple[str, ...] = GENERATORS):
    """Build a Phase-1-shaped corpus: every spec realized by every generator."""
    docs = []
    for i in range(n_specs):
        spec_id = f"spec{i:05d}"
        lcc = LCC_CODES[i % len(LCC_CODES)]
        cat = CATEGORIES[i % len(CATEGORIES)]
        block = f"block-{i % 3:02d}"
        for gen in generators:
            docs.append(
                {
                    "id": f"{spec_id}-{gen}",
                    "spec_id": spec_id,
                    "block_id": block,
                    "model": gen,
                    "lcc_code": lcc,
                    "lcgft_category": cat,
                }
            )
    return docs


def straddling_specs(result) -> set[str]:
    seen: dict[str, set[str]] = {}
    for name in ("train", "dev", "test"):
        for doc in result.get_split(name):
            seen.setdefault(doc["spec_id"], set()).add(name)
    return {k for k, v in seen.items() if len(v) > 1}


class TestLeakageIsRealWithoutGrouping:
    def test_document_level_split_leaks_specs(self):
        """The failure mode this feature exists to prevent."""
        docs = make_corpus()
        result = StratifiedSplitter(SplitConfig()).split(docs)
        assert len(straddling_specs(result)) > 100


class TestGroupedSplitting:
    def test_no_spec_straddles_splits(self):
        docs = make_corpus()
        result = StratifiedSplitter(SplitConfig(group_by="spec_id")).split(docs)
        assert straddling_specs(result) == set()

    def test_all_documents_are_retained(self):
        docs = make_corpus()
        result = StratifiedSplitter(SplitConfig(group_by="spec_id")).split(docs)
        assert len(result.train) + len(result.dev) + len(result.test) == len(docs)

    def test_group_members_stay_together(self):
        docs = make_corpus()
        result = StratifiedSplitter(SplitConfig(group_by="spec_id")).split(docs)
        for name in ("train", "dev", "test"):
            by_spec: dict[str, int] = {}
            for doc in result.get_split(name):
                by_spec[doc["spec_id"]] = by_spec.get(doc["spec_id"], 0) + 1
            assert all(count == len(GENERATORS) for count in by_spec.values())

    def test_ratios_are_respected_at_group_level(self):
        docs = make_corpus(n_specs=300)
        result = StratifiedSplitter(SplitConfig(group_by="spec_id")).split(docs)
        groups = result.statistics["grouping"]["groups_per_split"]
        assert groups["train"] == 180
        assert groups["dev"] == 60
        assert groups["test"] == 60

    def test_every_block_appears_in_every_split(self):
        """Split must not be confounded with which spec block a doc came from."""
        docs = make_corpus(n_specs=300)
        result = StratifiedSplitter(SplitConfig(group_by="spec_id")).split(docs)
        for name in ("train", "dev", "test"):
            blocks = {d["block_id"] for d in result.get_split(name)}
            assert blocks == {"block-00", "block-01", "block-02"}

    def test_deterministic_under_seed(self):
        docs = make_corpus()
        a = StratifiedSplitter(SplitConfig(group_by="spec_id")).split(docs)
        b = StratifiedSplitter(SplitConfig(group_by="spec_id")).split(docs)
        assert a.checksum == b.checksum

    def test_different_seed_changes_the_split(self):
        docs = make_corpus()
        a = StratifiedSplitter(SplitConfig(group_by="spec_id", random_seed=1)).split(
            docs
        )
        b = StratifiedSplitter(SplitConfig(group_by="spec_id", random_seed=2)).split(
            docs
        )
        assert a.checksum != b.checksum

    def test_statistics_record_the_grouping(self):
        docs = make_corpus()
        result = StratifiedSplitter(SplitConfig(group_by="spec_id")).split(docs)
        grouping = result.statistics["grouping"]
        assert grouping["group_by"] == "spec_id"
        assert grouping["n_groups"] == 200
        assert grouping["mean_docs_per_group"] == pytest.approx(len(GENERATORS))

    def test_config_is_carried_through(self):
        docs = make_corpus()
        result = StratifiedSplitter(SplitConfig(group_by="spec_id")).split(docs)
        assert result.config.group_by == "spec_id"

    def test_uneven_group_sizes_are_handled(self):
        """Not every generator succeeds on every spec."""
        docs = make_corpus()
        docs = [
            d for d in docs if not (d["spec_id"].endswith("7") and d["model"] == "glm")
        ]
        result = StratifiedSplitter(SplitConfig(group_by="spec_id")).split(docs)
        assert straddling_specs(result) == set()
        assert len(result.train) + len(result.dev) + len(result.test) == len(docs)


class TestValidation:
    def test_missing_group_field_is_rejected(self):
        docs = make_corpus()
        for doc in docs[:5]:
            del doc["spec_id"]
        with pytest.raises(ValueError, match="have no 'spec_id'"):
            StratifiedSplitter(SplitConfig(group_by="spec_id")).split(docs)

    def test_empty_group_value_is_rejected(self):
        docs = make_corpus()
        docs[0]["spec_id"] = ""
        with pytest.raises(ValueError, match="have no 'spec_id'"):
            StratifiedSplitter(SplitConfig(group_by="spec_id")).split(docs)

    def test_too_few_groups_gives_a_message_about_groups(self):
        """1,600 documents in 3 groups must not report a document shortage."""
        docs = make_corpus()
        with pytest.raises(ValueError, match="at least 100 distinct 'block_id' values"):
            StratifiedSplitter(SplitConfig(group_by="block_id")).split(docs)

    def test_grouping_by_an_arbitrary_field_works(self):
        docs = make_corpus(n_specs=400)
        for i, doc in enumerate(docs):
            doc["pair_id"] = f"pair{i // 4:05d}"
        result = StratifiedSplitter(SplitConfig(group_by="pair_id")).split(docs)
        seen: dict[str, set[str]] = {}
        for name in ("train", "dev", "test"):
            for doc in result.get_split(name):
                seen.setdefault(doc["pair_id"], set()).add(name)
        assert all(len(v) == 1 for v in seen.values())


class TestBackwardCompatibility:
    def test_default_config_has_no_grouping(self):
        assert SplitConfig().group_by is None

    def test_ungrouped_split_is_unchanged(self):
        docs = make_corpus(n_specs=100)
        a = StratifiedSplitter(SplitConfig()).split(docs)
        b = StratifiedSplitter(SplitConfig(group_by=None)).split(docs)
        assert a.checksum == b.checksum
