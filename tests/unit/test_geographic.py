"""Unit tests for the geographic region mapping and multi-tag label policy.

Covers the v0.4 additions to ``shelf.taxonomies.geographic``:
- The three-way :class:`GeographicLabelPolicy` (``FIRST``, ``UNAMBIGUOUS_ONLY``,
  ``ALL_REGIONS``) and its entry point, :func:`get_region_with_policy`.
- :func:`get_regions_from_list` / :func:`is_unambiguous_region`.
- :func:`analyze_geographic_ambiguity`, which makes the multi-tag defect
  documented in ``docs/data_plan_v0.4.md`` section 11.7 measurable.
- That the pre-existing default behavior (``FIRST``) is unchanged, so
  ``get_region_from_list``, ``filter_documents_for_clustering``, and
  ``add_geographic_region_field`` all remain backward compatible.
"""

from __future__ import annotations

import pytest
from shelf.taxonomies.geographic import (
    GeographicAmbiguityReport,
    GeographicLabelPolicy,
    add_geographic_region_field,
    analyze_geographic_ambiguity,
    filter_documents_for_clustering,
    get_region_from_list,
    get_region_with_policy,
    get_regions_from_list,
    is_unambiguous_region,
)

# ===========================================================================
# get_regions_from_list / is_unambiguous_region
# ===========================================================================


@pytest.mark.unit
class TestGetRegionsFromList:
    def test_single_recognized_tag(self):
        assert get_regions_from_list(["Tokyo"]) == frozenset({"East Asia"})

    def test_multiple_tags_same_region(self):
        assert get_regions_from_list(["Paris", "London"]) == frozenset({"Europe"})

    def test_multiple_tags_different_regions(self):
        assert get_regions_from_list(["Paris", "Brazil"]) == frozenset(
            {"Europe", "South America"}
        )

    def test_unrecognized_tag_ignored(self):
        assert get_regions_from_list(["Wakanda"]) == frozenset()

    def test_empty_list(self):
        assert get_regions_from_list([]) == frozenset()

    def test_mixed_recognized_and_unrecognized(self):
        assert get_regions_from_list(["Wakanda", "Tokyo"]) == frozenset({"East Asia"})


@pytest.mark.unit
class TestIsUnambiguousRegion:
    def test_single_tag_is_unambiguous(self):
        assert is_unambiguous_region(["Tokyo"]) is True

    def test_same_region_tags_are_unambiguous(self):
        assert is_unambiguous_region(["Paris", "London"]) is True

    def test_different_region_tags_are_ambiguous(self):
        assert is_unambiguous_region(["Paris", "Brazil"]) is False

    def test_no_tags_is_unambiguous(self):
        assert is_unambiguous_region([]) is True

    def test_no_recognized_tags_is_unambiguous(self):
        assert is_unambiguous_region(["Wakanda", "Narnia"]) is True


# ===========================================================================
# get_region_with_policy
# ===========================================================================


@pytest.mark.unit
class TestGetRegionWithPolicy:
    def test_first_policy_matches_get_region_from_list(self):
        """The default policy must reproduce the historical function exactly."""
        cases = [
            ["Paris", "Brazil"],
            ["Tokyo", "Beijing"],
            [],
            ["Unrecognized"],
            ["Paris", "London"],
        ]
        for locations in cases:
            assert get_region_with_policy(
                locations, GeographicLabelPolicy.FIRST
            ) == get_region_from_list(locations)

    def test_default_policy_is_first(self):
        """Calling without an explicit policy must not change behavior."""
        assert get_region_with_policy(["Paris", "Brazil"]) == get_region_from_list(
            ["Paris", "Brazil"]
        )

    def test_unambiguous_only_returns_region_for_agreeing_tags(self):
        assert (
            get_region_with_policy(
                ["Paris", "London"], GeographicLabelPolicy.UNAMBIGUOUS_ONLY
            )
            == "Europe"
        )

    def test_unambiguous_only_returns_none_for_ambiguous_tags(self):
        assert (
            get_region_with_policy(
                ["Paris", "Brazil"], GeographicLabelPolicy.UNAMBIGUOUS_ONLY
            )
            is None
        )

    def test_unambiguous_only_returns_none_for_no_recognized_tags(self):
        assert (
            get_region_with_policy([], GeographicLabelPolicy.UNAMBIGUOUS_ONLY) is None
        )

    def test_all_regions_returns_full_set(self):
        result = get_region_with_policy(
            ["Paris", "Brazil"], GeographicLabelPolicy.ALL_REGIONS
        )
        assert result == frozenset({"Europe", "South America"})

    def test_all_regions_returns_none_when_empty(self):
        assert get_region_with_policy([], GeographicLabelPolicy.ALL_REGIONS) is None

    def test_all_regions_single_tag(self):
        result = get_region_with_policy(["Tokyo"], GeographicLabelPolicy.ALL_REGIONS)
        assert result == frozenset({"East Asia"})

    def test_unknown_policy_raises(self):
        with pytest.raises(ValueError, match="Unknown GeographicLabelPolicy"):
            get_region_with_policy(["Tokyo"], "not_a_policy")  # type: ignore[arg-type]


# ===========================================================================
# analyze_geographic_ambiguity / GeographicAmbiguityReport
# ===========================================================================


@pytest.mark.unit
class TestAnalyzeGeographicAmbiguity:
    @pytest.fixture
    def sample_docs(self) -> list[dict]:
        return [
            {"geographic": ["Paris", "Brazil"]},  # ambiguous: Europe + South America
            {"geographic": ["Beijing", "Florida"]},  # ambiguous: East Asia + N. America
            {"geographic": ["Paris", "London"]},  # multi-tag but same region (Europe)
            {"geographic": ["Tokyo"]},  # single tag, unambiguous
            {"geographic": []},  # no geo tag at all
            {"geographic": ["Unrecognized Place"]},  # tag present, unrecognized
        ]

    def test_counts(self, sample_docs):
        report = analyze_geographic_ambiguity(sample_docs)
        assert isinstance(report, GeographicAmbiguityReport)
        assert report.total_documents == 6
        # with_geo: Paris/Brazil, Beijing/Florida, Paris/London, Tokyo -> 4
        assert report.documents_with_geo == 4
        # multi-tag (raw tag count >= 2): 3 docs
        assert report.documents_with_multiple_tags == 3
        # ambiguous (regions differ): Paris/Brazil, Beijing/Florida -> 2
        assert report.documents_ambiguous == 2

    def test_region_pair_counts(self, sample_docs):
        report = analyze_geographic_ambiguity(sample_docs)
        assert report.region_pair_counts[("Europe", "South America")] == 1
        assert report.region_pair_counts[("East Asia", "North America")] == 1
        assert len(report.region_pair_counts) == 2

    def test_example_ambiguous_tags_capped(self, sample_docs):
        report = analyze_geographic_ambiguity(sample_docs, max_examples=1)
        assert len(report.example_ambiguous_tags) == 1

    def test_fractions(self, sample_docs):
        report = analyze_geographic_ambiguity(sample_docs)
        assert report.multi_tag_fraction == pytest.approx(3 / 6)
        assert report.ambiguous_fraction_of_multi_tag == pytest.approx(2 / 3)
        assert report.ambiguous_fraction_of_geo_labelled == pytest.approx(2 / 4)

    def test_fractions_with_empty_corpus_do_not_divide_by_zero(self):
        report = analyze_geographic_ambiguity([])
        assert report.total_documents == 0
        assert report.multi_tag_fraction == 0.0
        assert report.ambiguous_fraction_of_multi_tag == 0.0
        assert report.ambiguous_fraction_of_geo_labelled == 0.0

    def test_summary_is_a_nonempty_string(self, sample_docs):
        report = analyze_geographic_ambiguity(sample_docs)
        summary = report.summary()
        assert isinstance(summary, str)
        assert "2/3" in summary  # documents_ambiguous / documents_with_multiple_tags

    def test_no_ambiguity_case(self):
        docs = [{"geographic": ["Paris", "London"]}, {"geographic": ["Tokyo"]}]
        report = analyze_geographic_ambiguity(docs)
        assert report.documents_ambiguous == 0
        assert report.region_pair_counts == {}

    def test_missing_geographic_key_treated_as_empty(self):
        report = analyze_geographic_ambiguity([{"id": "1"}])
        assert report.total_documents == 1
        assert report.documents_with_geo == 0


# ===========================================================================
# Backward compatibility: default behavior must not change
# ===========================================================================


@pytest.mark.unit
class TestBackwardCompatibility:
    def test_get_region_from_list_unchanged(self):
        assert get_region_from_list(["Paris", "Brazil"]) == "Europe"
        assert get_region_from_list(["Beijing", "Florida"]) == "East Asia"
        assert get_region_from_list([]) is None

    def test_filter_documents_for_clustering_default_policy_unchanged(self):
        docs = [
            {"id": "1", "geographic": ["Tokyo"]},
            {"id": "2", "geographic": []},
            {"id": "3", "geographic": ["Unknown"]},
            {"id": "4", "geographic": ["Paris", "London"]},
        ]
        filtered = filter_documents_for_clustering(docs)
        assert [d["id"] for d in filtered] == ["1", "4"]

    def test_filter_documents_for_clustering_unambiguous_only_drops_ambiguous(self):
        docs = [
            {"id": "1", "geographic": ["Tokyo"]},
            {"id": "2", "geographic": ["Paris", "Brazil"]},  # ambiguous
            {"id": "3", "geographic": ["Paris", "London"]},  # unambiguous (both Europe)
        ]
        filtered = filter_documents_for_clustering(
            docs, policy=GeographicLabelPolicy.UNAMBIGUOUS_ONLY
        )
        assert [d["id"] for d in filtered] == ["1", "3"]

    def test_add_geographic_region_field_default_policy_unchanged(self):
        docs = [
            {"id": "1", "geographic": ["Tokyo", "Beijing"]},
            {"id": "2", "geographic": []},
        ]
        updated = add_geographic_region_field(docs)
        assert updated[0]["geographic_region"] == "East Asia"
        assert updated[1]["geographic_region"] is None

    def test_add_geographic_region_field_unambiguous_only(self):
        docs = [
            {"id": "1", "geographic": ["Paris", "Brazil"]},  # ambiguous -> None
            {"id": "2", "geographic": ["Paris", "London"]},  # unambiguous -> Europe
        ]
        updated = add_geographic_region_field(
            docs, policy=GeographicLabelPolicy.UNAMBIGUOUS_ONLY
        )
        assert updated[0]["geographic_region"] is None
        assert updated[1]["geographic_region"] == "Europe"

    def test_add_geographic_region_field_all_regions(self):
        docs = [{"id": "1", "geographic": ["Paris", "Brazil"]}]
        updated = add_geographic_region_field(
            docs, policy=GeographicLabelPolicy.ALL_REGIONS
        )
        assert updated[0]["geographic_region"] == frozenset({"Europe", "South America"})
