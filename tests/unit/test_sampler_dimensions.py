"""Unit tests for shelf.sampler.dimensions taxonomy pool expansion.

Tests cover:
- v0.3.1 preset pools are byte-for-byte reproductions of the frozen,
  hand-curated label sets (TOPICS_BY_DOMAIN / GEOGRAPHIC_AREAS / LCGFT_DATA)
  used to build the published v0.3.1 corpus, including RNG-sequence
  reproducibility for a fixed seed.
- Expanded (`pool_size=<int>`) pools reach the requested sizes, sourced
  from data/taxonomies/*.json via shelf.sampler.lc_data.
- Determinism: same seed -> identical sample sequence, for both preset and
  expanded pools.
- Uniform-vs-frequency weighting: uniform is the default and covers the
  full pool; frequency weighting concentrates draws on head terms.
- The top-level `LCCSampler` is unaffected by the optional `subclass` field
  added to `LCCClass` for the v0.4 difficulty tier (see test_lcc_subclass.py).
- Validation errors for out-of-range pool sizes.
"""

from __future__ import annotations

from collections import Counter

import pytest
from shelf.sampler.dimensions import (
    GEOGRAPHIC_AREAS,
    LCC_DATA,
    LCGFT_DATA,
    PRESET_V0_3_1,
    TOPICS_BY_DOMAIN,
    GeographicSampler,
    LCCSampler,
    LCGFTSampler,
    TopicSampler,
)
from shelf.sampler.lc_data import (
    FORM_POOL_MAX,
    GEOGRAPHIC_POOL_MAX,
    TOPIC_POOL_MAX,
    build_expanded_form_pool,
    load_form_pool,
    load_geographic_pool,
    load_topic_pool,
)

# =============================================================================
# v0.3.1 preset: exact reproducibility
# =============================================================================


class TestTopicSamplerV031Preset:
    """The default pool must remain exactly TOPICS_BY_DOMAIN."""

    def test_default_pool_size_is_v0_3_1(self):
        sampler = TopicSampler()
        assert sampler._pool_size == PRESET_V0_3_1

    def test_default_pool_matches_topics_by_domain(self):
        sampler = TopicSampler()
        expected = []
        for domain in TOPICS_BY_DOMAIN:
            expected.extend(TOPICS_BY_DOMAIN[domain])
        assert sampler.values() == expected

    def test_unique_topic_count_is_112(self):
        # 113 raw entries (TOPICS_BY_DOMAIN has one cross-domain duplicate:
        # "Finance" appears in both social_sciences and business), 112
        # distinct labels -- matches the measured-facts figure in
        # docs/data_plan_v0.4.md section 4.1.
        sampler = TopicSampler()
        assert len(sampler.values()) == 113
        assert len(set(sampler.values())) == 112

    def test_lcc_class_filtering_unchanged(self):
        sampler = TopicSampler(lcc_class="K")
        assert sampler.values() == TOPICS_BY_DOMAIN["law"]

    def test_domains_filtering_unchanged(self):
        sampler = TopicSampler(domains=["science", "medicine"])
        assert (
            sampler.values()
            == TOPICS_BY_DOMAIN["science"] + TOPICS_BY_DOMAIN["medicine"]
        )

    def test_reproduces_pinned_golden_sequence(self):
        """Pinned output for seed=42 against the pre-expansion implementation.

        If this ever changes, the v0.3.1 corpus is no longer reproducible.
        """
        sampler = TopicSampler(seed=42)
        assert sampler.sample_n(5) == [
            "Finance",
            "Finance",
            "Analysis",
            "International relations",
            "Geology",
        ]


class TestGeographicSamplerV031Preset:
    def test_default_pool_size_is_v0_3_1(self):
        sampler = GeographicSampler()
        assert sampler._pool_size == PRESET_V0_3_1

    def test_default_pool_matches_geographic_areas(self):
        sampler = GeographicSampler()
        expected = [area for areas in GEOGRAPHIC_AREAS.values() for area in areas]
        assert sampler.values() == expected
        assert len(sampler.values()) == 44

    def test_area_types_filtering_unchanged(self):
        sampler = GeographicSampler(area_types=["countries"])
        assert sampler.values() == GEOGRAPHIC_AREAS["countries"]

    def test_reproduces_pinned_golden_sequence(self):
        sampler = GeographicSampler(seed=42)
        assert sampler.sample_n(5) == [
            "United Kingdom",
            "California",
            None,
            "Berlin",
            "New York City",
        ]


class TestLCCSamplerV031Preset:
    """Adding an optional `subclass` to `LCCClass` must not disturb the frozen
    top-level draw: same pool, same RNG sequence, same rendering."""

    def test_default_pool_is_the_21_main_classes(self):
        sampler = LCCSampler()
        assert [c.code for c in sampler.values()] == list(LCC_DATA)

    def test_draws_carry_no_subclass(self):
        assert all(c.subclass is None for c in LCCSampler(seed=42).sample_n(20))

    def test_reproduces_pinned_golden_sequence(self):
        sampler = LCCSampler(seed=42)
        assert [str(c) for c in sampler.sample_n(5)] == [
            "Z: Bibliography, Library Science",
            "D: World History (except Americas)",
            "A: General Works",
            "J: Political Science",
            "H: Social Sciences",
        ]

    def test_pinned_uris(self):
        assert [c.uri for c in LCCSampler(seed=42).sample_n(3)] == [
            "http://id.loc.gov/authorities/classification/Z",
            "http://id.loc.gov/authorities/classification/D",
            "http://id.loc.gov/authorities/classification/A",
        ]


class TestLCGFTSamplerV031Preset:
    def test_default_pool_size_is_v0_3_1(self):
        sampler = LCGFTSampler()
        assert sampler._pool_size == PRESET_V0_3_1

    def test_default_pool_matches_lcgft_data(self):
        sampler = LCGFTSampler()
        assert sampler._data == LCGFT_DATA
        assert len(sampler.values()) == 14

    def test_unique_form_count_is_133(self):
        sampler = LCGFTSampler()
        all_forms = {form for forms in sampler._data.values() for form in forms}
        assert len(all_forms) == 133

    def test_categories_filtering_unchanged(self):
        sampler = LCGFTSampler(categories=["Literature", "Law materials"])
        assert set(sampler.values()) == {"Literature", "Law materials"}

    def test_reproduces_pinned_golden_sequence(self):
        sampler = LCGFTSampler(seed=42)
        assert [str(t) for t in sampler.sample_n(5)] == [
            "Ephemera > Calendars",
            "Informational works > Data sets",
            "Literature > Poetry",
            "Instructional and educational works > How-to guides",
            "Law materials > Contracts",
        ]


# =============================================================================
# Expanded pools: sizes and sourcing
# =============================================================================


class TestTopicSamplerExpandedPool:
    @pytest.mark.parametrize("size", [50, 500, 1000, 2000])
    def test_pool_reaches_requested_size(self, size):
        sampler = TopicSampler(seed=1, pool_size=size)
        assert len(sampler.values()) == size
        assert len(set(sampler.values())) == size  # all distinct

    def test_pool_sourced_from_lc_data(self):
        sampler = TopicSampler(seed=1, pool_size=500)
        entries = load_topic_pool(500)
        assert sampler.values() == [e.label for e in entries]

    def test_smaller_cut_is_prefix_of_larger_cut(self):
        small = TopicSampler(seed=1, pool_size=100).values()
        large = TopicSampler(seed=1, pool_size=1000).values()
        assert small == large[:100]

    def test_domains_ignored_for_expanded_pool(self):
        # domains/lcc_class only apply to the v0.3.1 preset; passing them
        # alongside a numeric pool_size must not raise, and must not
        # restrict the expanded pool.
        sampler = TopicSampler(domains=["law"], seed=1, pool_size=500)
        assert len(sampler.values()) == 500

    def test_rejects_pool_size_above_max(self):
        with pytest.raises(ValueError):
            TopicSampler(pool_size=TOPIC_POOL_MAX + 1)

    def test_rejects_pool_size_below_one(self):
        with pytest.raises(ValueError):
            TopicSampler(pool_size=0)


class TestGeographicSamplerExpandedPool:
    @pytest.mark.parametrize("size", [50, 200, 500])
    def test_pool_reaches_requested_size(self, size):
        sampler = GeographicSampler(seed=1, pool_size=size)
        assert len(sampler.values()) == size
        assert len(set(sampler.values())) == size

    def test_pool_sourced_from_lc_data(self):
        sampler = GeographicSampler(seed=1, pool_size=200)
        entries = load_geographic_pool(200)
        assert sampler.values() == [e.label for e in entries]

    def test_area_types_ignored_for_expanded_pool(self):
        sampler = GeographicSampler(area_types=["countries"], seed=1, pool_size=200)
        assert len(sampler.values()) == 200

    def test_rejects_pool_size_above_max(self):
        with pytest.raises(ValueError):
            GeographicSampler(pool_size=GEOGRAPHIC_POOL_MAX + 1)

    def test_rejects_pool_size_below_one(self):
        with pytest.raises(ValueError):
            GeographicSampler(pool_size=0)


class TestLCGFTSamplerExpandedPool:
    @pytest.mark.parametrize("size", [133, 300, 400, 554])
    def test_pool_reaches_requested_size(self, size):
        sampler = LCGFTSampler(seed=1, pool_size=size)
        all_forms = {form for forms in sampler._data.values() for form in forms}
        assert len(all_forms) == size

    def test_curated_forms_are_a_subset_of_expanded_pool(self):
        sampler = LCGFTSampler(seed=1, pool_size=300)
        expanded_forms = {form for forms in sampler._data.values() for form in forms}
        curated_forms = {form for forms in LCGFT_DATA.values() for form in forms}
        assert curated_forms <= expanded_forms

    def test_curated_categories_keep_their_forms(self):
        # Categories from the curated pool must keep at least their
        # original forms after expansion (extension, not replacement).
        expanded = build_expanded_form_pool(300)
        for category, forms in LCGFT_DATA.items():
            assert set(forms) <= set(expanded[category])

    def test_rejects_pool_size_below_curated_count(self):
        with pytest.raises(ValueError):
            LCGFTSampler(pool_size=132)

    def test_rejects_pool_size_above_max(self):
        with pytest.raises(ValueError):
            LCGFTSampler(pool_size=FORM_POOL_MAX + 1)

    def test_categories_filter_applies_to_expanded_pool(self):
        sampler = LCGFTSampler(
            categories=["Literature", "Law materials"], seed=1, pool_size=300
        )
        assert set(sampler.values()) <= {"Literature", "Law materials"}


# =============================================================================
# Determinism
# =============================================================================


class TestDeterminism:
    def test_topic_sampler_same_seed_same_sequence(self):
        a = TopicSampler(seed=123, pool_size=500).sample_n(50)
        b = TopicSampler(seed=123, pool_size=500).sample_n(50)
        assert a == b

    def test_topic_sampler_v031_same_seed_same_sequence(self):
        a = TopicSampler(seed=123).sample_n(50)
        b = TopicSampler(seed=123).sample_n(50)
        assert a == b

    def test_geographic_sampler_same_seed_same_sequence(self):
        a = GeographicSampler(seed=123, pool_size=200).sample_n(50)
        b = GeographicSampler(seed=123, pool_size=200).sample_n(50)
        assert a == b

    def test_lcgft_sampler_same_seed_same_sequence(self):
        a = [str(t) for t in LCGFTSampler(seed=123, pool_size=300).sample_n(50)]
        b = [str(t) for t in LCGFTSampler(seed=123, pool_size=300).sample_n(50)]
        assert a == b

    def test_different_seeds_diverge(self):
        a = TopicSampler(seed=1, pool_size=500).sample_n(50)
        b = TopicSampler(seed=2, pool_size=500).sample_n(50)
        assert a != b


# =============================================================================
# Weighting: uniform (default) vs frequency
# =============================================================================


class TestWeighting:
    def test_uniform_is_default(self):
        assert TopicSampler(pool_size=500)._weighting == "uniform"
        assert GeographicSampler(pool_size=200)._weighting == "uniform"
        assert LCGFTSampler(pool_size=300)._weighting == "uniform"

    def test_uniform_sampling_covers_full_pool(self):
        # Over many draws, uniform sampling should touch (nearly) every
        # label in the pool -- this is the whole point of pool expansion.
        sampler = GeographicSampler(
            seed=1, pool_size=500, weighting="uniform", include_none=False
        )
        draws = sampler.sample_n(20_000)
        assert len(set(draws)) == 500

    def test_frequency_weighting_concentrates_on_head_term(self):
        # "United States" carries ~45% of total MARC frequency mass in the
        # top-500 geographic pool; frequency weighting must reflect that
        # skew, while uniform must not.
        weighted = GeographicSampler(
            seed=1, pool_size=500, weighting="frequency", include_none=False
        )
        uniform = GeographicSampler(
            seed=1, pool_size=500, weighting="uniform", include_none=False
        )
        n = 20_000
        weighted_counts = Counter(weighted.sample_n(n))
        uniform_counts = Counter(uniform.sample_n(n))

        weighted_share = weighted_counts["United States"] / n
        uniform_share = uniform_counts["United States"] / n

        assert weighted_share > 0.35  # measured mass share is ~0.451
        assert uniform_share < 0.01  # ~1/500 = 0.002 expected

    def test_lcgft_frequency_weighting_biases_within_category(self):
        weighted = LCGFTSampler(
            seed=1,
            pool_size=554,
            weighting="frequency",
            categories=["Cartographic materials"],
        )
        uniform = LCGFTSampler(
            seed=1,
            pool_size=554,
            weighting="uniform",
            categories=["Cartographic materials"],
        )
        n = 5_000
        weighted_forms = [str(t.form) for t in weighted.sample_n(n)]
        uniform_forms = [str(t.form) for t in uniform.sample_n(n)]

        # "Maps" is the single highest-frequency LCGFT term overall (70,971)
        # and belongs to Cartographic materials, so it should dominate under
        # frequency weighting far more than under uniform sampling.
        weighted_maps_share = weighted_forms.count("Maps") / n
        uniform_maps_share = uniform_forms.count("Maps") / n
        assert weighted_maps_share > uniform_maps_share


# =============================================================================
# lc_data pool loaders
# =============================================================================


class TestLoadPools:
    def test_load_topic_pool_size_and_order(self):
        entries = load_topic_pool(10)
        assert len(entries) == 10
        assert [e.rank for e in entries] == list(range(1, 11))
        # descending frequency (rank-ordered)
        freqs = [e.frequency for e in entries]
        assert freqs == sorted(freqs, reverse=True)

    def test_load_geographic_pool_top_entry_is_united_states(self):
        entries = load_geographic_pool(1)
        assert entries[0].label == "United States"

    def test_load_form_pool_top_entry_is_maps(self):
        entries = load_form_pool(1)
        assert entries[0].label == "Maps"

    def test_pool_description_fields_are_unpopulated(self):
        # Documented coverage gap: none of the extracted taxonomy files
        # carry populated description/uri/alt_labels/broader/narrower --
        # only id/label/frequency/rank are real. If this ever changes
        # (e.g. a richer extraction pass), this test should start failing
        # and the sampler docstrings/description fields should be revisited.
        entries = load_topic_pool(500)
        assert sum(1 for e in entries if e.description) == 0
        assert sum(1 for e in entries if e.uri) == 0
        assert sum(1 for e in entries if e.alt_labels) == 0

    @pytest.mark.parametrize(
        "loader,pool_max",
        [
            (load_topic_pool, TOPIC_POOL_MAX),
            (load_geographic_pool, GEOGRAPHIC_POOL_MAX),
            (load_form_pool, FORM_POOL_MAX),
        ],
    )
    def test_loaders_reject_out_of_range_sizes(self, loader, pool_max):
        with pytest.raises(ValueError):
            loader(0)
        with pytest.raises(ValueError):
            loader(pool_max + 1)

    def test_build_expanded_form_pool_category_coverage_is_documented(self):
        # At pool_size=300, ~19% of the 167 forms added beyond the 133
        # curated ones get a real category via lcgft_hierarchy.json; the
        # rest fall back to "Uncategorized". This test pins the measured
        # ratio range rather than an exact count so it isn't overly brittle
        # to taxonomy file edits, while still catching a regression to 0%
        # or 100% coverage.
        expanded = build_expanded_form_pool(300)
        curated_forms = {form for forms in LCGFT_DATA.values() for form in forms}
        added_forms = {
            form for forms in expanded.values() for form in forms
        } - curated_forms
        assert len(added_forms) == 300 - len(curated_forms)
        uncategorized = set(expanded.get("Uncategorized", []))
        assert uncategorized <= added_forms
        matched_fraction = 1 - (len(uncategorized) / len(added_forms))
        assert 0.10 < matched_fraction < 0.35
