"""Unit tests for the Phase 6 natural-data transfer slice.

Two things are being protected here.

**The contamination boundary.** The slice's whole value is that it is the
*contaminated-natural* condition set against SHELF's *clean-synthetic* one. If a
record could claim it was not contaminated, or if synthetic and natural records
could be pooled into one metric, the comparison would silently stop meaning
anything. Both are tested as hard failures, not warnings.

**The label fidelity.** Labels come from Gutenberg's cataloguers, so the parser
has to read the RDF exactly, the boilerplate stripper must not leave Project
Gutenberg's licence text inside a passage, and the chunker must not hand back
front matter as if it were the classified work.

Every network path is mocked. No test in this file touches gutenberg.org.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import urllib.error
from pathlib import Path
from typing import Any

import pytest
from shelf.hub.transfer import (
    CONTAMINATION_NOTICE,
    SKEW_NOTICE,
    ContaminationStatus,
    NaturalSource,
    SourceType,
    SourceTypePoolingError,
    TransferValidationError,
    assert_single_source_type,
    build_report,
    compare_to_synthetic,
    describe_slice,
    format_distribution_report,
    load_transfer_slice,
    split_by_source_type,
    total_variation_distance,
    validate_record,
    validate_records,
)

# The builder is a script, not a package module, so load it by path (same
# pattern as tests/unit/test_generate_documents.py).
_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "build_transfer_slice.py"
_spec = importlib.util.spec_from_file_location("build_transfer_slice", _SCRIPT)
assert _spec is not None and _spec.loader is not None
bts = importlib.util.module_from_spec(_spec)
sys.modules["build_transfer_slice"] = bts
_spec.loader.exec_module(bts)


# =============================================================================
# Fixtures
# =============================================================================

#: A faithful miniature of a real ``pg<id>.rdf``: same namespaces, same
#: ``dcam:memberOf`` keying of LCC vs LCSH, same nested plain-text format block.
RDF_TEMPLATE = """<?xml version="1.0" encoding="utf-8"?>
<rdf:RDF xml:base="http://www.gutenberg.org/"
  xmlns:pgterms="http://www.gutenberg.org/2009/pgterms/"
  xmlns:dcterms="http://purl.org/dc/terms/"
  xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
  xmlns:dcam="http://purl.org/dc/dcam/"
>
  <pgterms:ebook rdf:about="ebooks/{gid}">
    <dcterms:title>{title}</dcterms:title>
    <dcterms:issued rdf:datatype="http://www.w3.org/2001/XMLSchema#date">1998-06-01</dcterms:issued>
    <dcterms:rights>{rights}</dcterms:rights>
    <pgterms:downloads rdf:datatype="http://www.w3.org/2001/XMLSchema#integer">4242</pgterms:downloads>
    <dcterms:creator>
      <pgterms:agent rdf:about="2009/agents/68">
        <pgterms:name>{author}</pgterms:name>
        <pgterms:birthdate rdf:datatype="http://www.w3.org/2001/XMLSchema#integer">1775</pgterms:birthdate>
        <pgterms:deathdate rdf:datatype="http://www.w3.org/2001/XMLSchema#integer">1817</pgterms:deathdate>
      </pgterms:agent>
    </dcterms:creator>
    <dcterms:language>
      <rdf:Description rdf:nodeID="Nlang">
        <rdf:value rdf:datatype="http://purl.org/dc/terms/RFC4646">{language}</rdf:value>
      </rdf:Description>
    </dcterms:language>
    {subjects}
    <dcterms:type>
      <rdf:Description rdf:nodeID="Ntype">
        <dcam:memberOf rdf:resource="http://purl.org/dc/terms/DCMIType"/>
        <rdf:value>{dcmi_type}</rdf:value>
      </rdf:Description>
    </dcterms:type>
    <pgterms:bookshelf>
      <rdf:Description rdf:nodeID="Nshelf">
        <dcam:memberOf rdf:resource="2009/pgterms/Bookshelf"/>
        <rdf:value>Category: Novels</rdf:value>
      </rdf:Description>
    </pgterms:bookshelf>
    <dcterms:hasFormat>
      <pgterms:file rdf:about="https://www.gutenberg.org/ebooks/{gid}.txt.utf-8">
        <dcterms:extent rdf:datatype="http://www.w3.org/2001/XMLSchema#integer">{extent}</dcterms:extent>
        <dcterms:format>
          <rdf:Description rdf:nodeID="Nimt">
            <dcam:memberOf rdf:resource="http://purl.org/dc/terms/IMT"/>
            <rdf:value rdf:datatype="http://purl.org/dc/terms/IMT">text/plain; charset=utf-8</rdf:value>
          </rdf:Description>
        </dcterms:format>
      </pgterms:file>
    </dcterms:hasFormat>
    <dcterms:hasFormat>
      <pgterms:file rdf:about="https://www.gutenberg.org/ebooks/{gid}.epub.images">
        <dcterms:extent rdf:datatype="http://www.w3.org/2001/XMLSchema#integer">9999999</dcterms:extent>
        <dcterms:format>
          <rdf:Description rdf:nodeID="Nimt2">
            <dcam:memberOf rdf:resource="http://purl.org/dc/terms/IMT"/>
            <rdf:value rdf:datatype="http://purl.org/dc/terms/IMT">application/epub+zip</rdf:value>
          </rdf:Description>
        </dcterms:format>
      </pgterms:file>
    </dcterms:hasFormat>
  </pgterms:ebook>
</rdf:RDF>
"""

_SUBJECT_BLOCK = """    <dcterms:subject>
      <rdf:Description rdf:nodeID="N{i}">
        <dcam:memberOf rdf:resource="http://purl.org/dc/terms/{scheme}"/>
        <rdf:value>{value}</rdf:value>
      </rdf:Description>
    </dcterms:subject>"""


def make_rdf(
    gid: int = 1342,
    *,
    title: str = "Pride and Prejudice",
    author: str = "Austen, Jane",
    language: str = "en",
    lcc: tuple[str, ...] = ("PR",),
    lcsh: tuple[str, ...] = ("England -- Fiction", "Love stories"),
    dcmi_type: str = "Text",
    rights: str = "Public domain in the USA.",
    extent: int = 772386,
) -> bytes:
    subjects = "\n".join(
        [
            _SUBJECT_BLOCK.format(i=i, scheme="LCC", value=value)
            for i, value in enumerate(lcc)
        ]
        + [
            _SUBJECT_BLOCK.format(i=100 + i, scheme="LCSH", value=value)
            for i, value in enumerate(lcsh)
        ]
    )
    return RDF_TEMPLATE.format(
        gid=gid,
        title=title,
        author=author,
        language=language,
        subjects=subjects,
        dcmi_type=dcmi_type,
        rights=rights,
        extent=extent,
    ).encode("utf-8")


def make_gutenberg_text(paragraphs: int = 60, words: int = 60) -> str:
    """A plausible PG plain-text file: licence header, credits, body, footer."""
    body = "\n\n".join(
        f"Paragraph {i} " + " ".join(["word"] * words) for i in range(paragraphs)
    )
    return (
        "The Project Gutenberg eBook of Something\n\n"
        "This ebook is for the use of anyone anywhere in the United States...\n\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK SOMETHING ***\n\n"
        "Produced by A Volunteer and the Online Distributed Proofreading Team.\n\n"
        f"{body}\n\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK SOMETHING ***\n\n"
        "Updated editions will replace the previous one—the old editions will be renamed.\n"
        "START: FULL LICENSE\n"
    )


def make_record(
    rec_id: str = "pg-1342-0002",
    *,
    lcc: str = "P",
    source_type: str = SourceType.NATURAL.value,
    status: str = ContaminationStatus.KNOWN_CONTAMINATED.value,
    text: str = "some natural prose about a country estate and five daughters",
    **overrides: Any,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "id": rec_id,
        "text": text,
        "body": text,
        "title": "Pride and Prejudice",
        "word_count": len(text.split()),
        "lcc_code": lcc,
        "lcc_name": "Language and Literature",
        "lcgft_category": "Literature",
        "lcgft_form": "Fiction",
        "topics": ["England"],
        "language": "en",
        "source_type": source_type,
        "source": NaturalSource.PROJECT_GUTENBERG.value,
        "contamination_status": status,
        "provenance": {
            "gutenberg_id": 1342,
            "lcc_letter": lcc,
            "lcc_subclasses": ["PR"],
            "chunk_index": 2,
            "chunk_count": 40,
            "text_url": "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
            "author_death_year": 1817,
            "gutenberg_issued": "1998-06-01",
        },
    }
    record.update(overrides)
    return record


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any real HTTP attempt an immediate, loud failure."""

    def explode(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("network access attempted in a unit test")

    monkeypatch.setattr(bts.urllib.request, "urlopen", explode)


# =============================================================================
# RDF parsing
# =============================================================================


class TestParseEbookRdf:
    def test_extracts_lcc_lcsh_title_author_language(self) -> None:
        entry = bts.parse_ebook_rdf(make_rdf())
        assert entry is not None
        assert entry.gutenberg_id == 1342
        assert entry.title == "Pride and Prejudice"
        assert entry.authors == ["Austen, Jane"]
        assert entry.author_birth_year == 1775
        assert entry.author_death_year == 1817
        assert entry.languages == ["en"]
        assert entry.lcc_subclasses == ["PR"]
        assert entry.lcsh_headings == ["England -- Fiction", "Love stories"]
        assert entry.dcmi_types == ["Text"]
        assert entry.rights == "Public domain in the USA."
        assert entry.downloads == 4242

    def test_main_class_is_the_first_letter_of_the_subclass(self) -> None:
        entry = bts.parse_ebook_rdf(make_rdf(lcc=("PR",)))
        assert entry is not None
        assert entry.lcc_letters == ["P"]

    def test_multiple_subclasses_may_span_main_classes(self) -> None:
        entry = bts.parse_ebook_rdf(make_rdf(lcc=("PR", "DA")))
        assert entry is not None
        assert entry.lcc_letters == ["D", "P"]

    def test_only_plain_text_format_extent_is_recorded(self) -> None:
        # The epub in the fixture is 9,999,999 bytes; it must not be picked up.
        entry = bts.parse_ebook_rdf(make_rdf(extent=772386))
        assert entry is not None
        assert entry.text_bytes == 772386

    def test_work_without_lcc_yields_no_letters(self) -> None:
        entry = bts.parse_ebook_rdf(make_rdf(lcc=()))
        assert entry is not None
        assert entry.lcc_subclasses == []
        assert entry.lcc_letters == []

    def test_non_alphabetic_lcc_value_is_ignored_as_a_main_class(self) -> None:
        entry = bts.parse_ebook_rdf(make_rdf(lcc=("2",)))
        assert entry is not None
        assert entry.lcc_letters == []

    def test_malformed_xml_returns_none(self) -> None:
        assert bts.parse_ebook_rdf(b"<rdf:RDF><not closed") is None

    def test_non_ebook_document_returns_none(self) -> None:
        assert (
            bts.parse_ebook_rdf(b'<?xml version="1.0"?><rdf:RDF xmlns:rdf="x"/>')
            is None
        )


# =============================================================================
# LCSH-derived labels
# =============================================================================


class TestDerivedLabels:
    def test_fiction_subdivision_maps_to_literature(self) -> None:
        assert bts.derive_lcgft(["England -- Fiction"]) == ("Fiction", "Literature")

    def test_more_specific_genre_wins_over_bare_fiction(self) -> None:
        form, category = bts.derive_lcgft(["Science fiction", "Adventure -- Fiction"])
        assert (form, category) == ("Science fiction", "Literature")

    def test_unmatched_headings_yield_no_form_rather_than_a_guess(self) -> None:
        assert bts.derive_lcgft(["Bees", "Apiculture"]) == ("", "")

    @pytest.mark.parametrize(
        ("heading", "expected"),
        [
            ("Sermons, English", "Sermons"),
            ("Cookery, American", "Cookbooks"),
            ("Shakespeare, William -- Criticism and interpretation", "Criticism"),
        ],
    )
    def test_representative_markers(self, heading: str, expected: str) -> None:
        assert bts.derive_lcgft([heading])[0] == expected

    def test_derived_form_is_a_real_shelf_label(self) -> None:
        from shelf.sampler.dimensions import LCGFT_DATA

        for _marker, form, category in bts.LCSH_FORM_MARKERS:
            assert category in LCGFT_DATA, category
            assert form in LCGFT_DATA[category], (category, form)

    def test_juvenile_headings_derive_audience(self) -> None:
        assert bts.derive_audience(["Adventure -- Juvenile fiction"]) == "Children"
        assert bts.derive_audience(["Bees"]) == ""

    def test_topics_are_lcsh_main_headings_deduplicated_in_order(self) -> None:
        topics = bts.lcsh_main_headings(
            ["England -- Fiction", "England -- History", "Love stories"]
        )
        assert topics == ["England", "Love stories"]


# =============================================================================
# Text cleaning and chunking
# =============================================================================


class TestBoilerplate:
    def test_licence_header_and_footer_are_removed(self) -> None:
        cleaned = bts.strip_gutenberg_boilerplate(
            make_gutenberg_text(paragraphs=5, words=10)
        )
        assert "PROJECT GUTENBERG EBOOK" not in cleaned
        assert "FULL LICENSE" not in cleaned
        assert "Paragraph 0" in cleaned

    def test_transcription_credits_are_removed(self) -> None:
        cleaned = bts.strip_gutenberg_boilerplate(
            make_gutenberg_text(paragraphs=5, words=10)
        )
        assert "Produced by" not in cleaned

    def test_legacy_footer_without_star_fence_is_handled(self) -> None:
        raw = (
            "*** START OF THE PROJECT GUTENBERG EBOOK X ***\n\n"
            "Real body text here.\n\n"
            "End of the Project Gutenberg EBook of X, by Y\n\n"
            "Licence blather.\n"
        )
        cleaned = bts.strip_gutenberg_boilerplate(raw)
        assert "Real body text here." in cleaned
        assert "Licence blather" not in cleaned

    def test_text_without_markers_survives_unchanged(self) -> None:
        assert (
            bts.strip_gutenberg_boilerplate("Just a paragraph.") == "Just a paragraph."
        )

    def test_hard_wrapped_lines_rejoin_into_one_paragraph(self) -> None:
        paragraphs = bts.split_paragraphs("one two\nthree four\n\nnext para")
        assert paragraphs == ["one two three four", "next para"]


class TestTitleLeakage:
    def test_title_and_byline_paragraphs_are_dropped(self) -> None:
        paragraphs = [
            "Pride and Prejudice",
            "By Jane Austen",
            "It is a truth universally acknowledged.",
        ]
        kept = bts.drop_title_lines(paragraphs, "Pride and Prejudice", ["Austen, Jane"])
        assert kept == ["It is a truth universally acknowledged."]

    def test_body_mentioning_the_title_is_kept(self) -> None:
        paragraphs = ["Pride and Prejudice is the name of the estate she inherited."]
        assert bts.drop_title_lines(paragraphs, "Pride and Prejudice", []) == paragraphs


class TestChunking:
    def test_chunks_land_near_the_target_length(self) -> None:
        config = bts.ChunkConfig(target_words=100, min_words=40, max_words=300)
        paragraphs = [" ".join(["w"] * 30) for _ in range(20)]
        chunks = bts.chunk_paragraphs(paragraphs, config)
        assert chunks
        for chunk in chunks:
            assert 40 <= len(chunk.split()) <= 300

    def test_short_trailing_remainder_is_discarded(self) -> None:
        config = bts.ChunkConfig(target_words=100, min_words=90, max_words=300)
        chunks = bts.chunk_paragraphs([" ".join(["w"] * 10)], config)
        assert chunks == []

    def test_max_words_is_a_hard_ceiling_on_a_passage(self) -> None:
        config = bts.ChunkConfig(target_words=100, min_words=20, max_words=150)
        paragraphs = [" ".join(["w"] * n) for n in (90, 90, 30, 140, 20)]
        chunks = bts.chunk_paragraphs(paragraphs, config)
        assert chunks
        assert all(len(c.split()) <= 150 for c in chunks)

    def test_oversized_paragraph_is_split_not_dropped(self) -> None:
        config = bts.ChunkConfig(target_words=100, min_words=50, max_words=200)
        chunks = bts.chunk_paragraphs([" ".join(["w"] * 1000)], config)
        assert len(chunks) >= 9
        assert all(len(c.split()) <= 200 for c in chunks)

    def test_selection_skips_head_and_tail_chunks(self) -> None:
        config = bts.ChunkConfig(
            skip_head_chunks=1, skip_tail_chunks=1, max_per_work=10
        )
        chunks = [f"c{i}" for i in range(6)]
        picked = bts.select_chunks(chunks, config)
        indices = [i for i, _ in picked]
        assert 0 not in indices
        assert 5 not in indices

    def test_selection_spreads_across_the_work(self) -> None:
        config = bts.ChunkConfig(skip_head_chunks=1, skip_tail_chunks=1, max_per_work=3)
        chunks = [f"c{i}" for i in range(100)]
        indices = [i for i, _ in bts.select_chunks(chunks, config)]
        assert len(indices) == 3
        # Spread, not the first three consecutive chunks.
        assert max(indices) - min(indices) > 50

    def test_short_work_still_yields_a_chunk(self) -> None:
        config = bts.ChunkConfig(skip_head_chunks=1, skip_tail_chunks=1, max_per_work=3)
        assert bts.select_chunks(["only"], config) == [(0, "only")]

    def test_no_chunks_means_no_selection(self) -> None:
        assert bts.select_chunks([], bts.ChunkConfig()) == []


# =============================================================================
# Selection / stratification
# =============================================================================


def _entry(gid: int, letter: str, **kwargs: Any) -> Any:
    defaults: dict[str, Any] = {
        "title": f"Work {gid}",
        "authors": [f"Author {gid}"],
        "languages": ["en"],
        "lcc_subclasses": [f"{letter}R"],
        "lcsh_headings": ["Something"],
        "dcmi_types": ["Text"],
        "rights": "Public domain in the USA.",
        "text_bytes": 500_000,
    }
    defaults.update(kwargs)
    return bts.CatalogEntry(gutenberg_id=gid, **defaults)


class TestEligibility:
    @pytest.mark.parametrize(
        ("kwargs", "reason"),
        [
            ({"dcmi_types": ["Sound"]}, "not_text"),
            ({"title": ""}, "no_title"),
            ({"lcc_subclasses": []}, "no_lcc"),
            ({"lcc_subclasses": ["99"]}, "lcc_not_main_class"),
            ({"lcc_subclasses": ["PR", "DA"]}, "multi_lcc_letter"),
            ({"languages": ["fr"]}, "language"),
            (
                {"rights": "Copyrighted. Read the copyright notice."},
                "not_public_domain",
            ),
            ({"text_bytes": 100}, "too_short"),
            ({"text_bytes": 99_000_000}, "too_long"),
        ],
    )
    def test_rejection_reasons(self, kwargs: dict[str, Any], reason: str) -> None:
        config = bts.SelectionConfig()
        assert bts.eligible(_entry(1, "P", **kwargs), config) == reason

    def test_eligible_entry_passes(self) -> None:
        assert bts.eligible(_entry(1, "P"), bts.SelectionConfig()) is None

    def test_multi_class_works_can_be_admitted_explicitly(self) -> None:
        config = bts.SelectionConfig(require_single_lcc_letter=False)
        assert bts.eligible(_entry(1, "P", lcc_subclasses=["PR", "DA"]), config) is None


class TestSelectWorks:
    def test_per_class_quota_is_identical_and_shortfall_is_not_redistributed(
        self,
    ) -> None:
        # 200 P works, 2 Q works, nothing else: a naive "fill to target" would
        # hand the slack back to P, which is exactly what must not happen.
        entries = [_entry(i, "P") for i in range(200)] + [
            _entry(1000 + i, "Q") for i in range(2)
        ]
        chunk_config = bts.ChunkConfig(max_per_work=3)
        config = bts.SelectionConfig(target_passages=63, max_works_per_author=99)

        selection = bts.select_works(entries, config, chunk_config)

        assert selection.per_class_work_quota == 1
        assert selection.selected_per_class["P"] == 1
        assert selection.selected_per_class["Q"] == 1
        assert selection.selected_per_class["B"] == 0
        assert len(selection.works) == 2

    def test_works_per_author_is_capped(self) -> None:
        entries = [_entry(i, "P", authors=["One Author"]) for i in range(50)]
        config = bts.SelectionConfig(target_passages=21 * 30, max_works_per_author=2)
        selection = bts.select_works(entries, config, bts.ChunkConfig(max_per_work=1))
        assert selection.selected_per_class["P"] == 2

    def test_duplicate_title_author_pairs_are_dropped(self) -> None:
        entries = [
            _entry(1, "P", title="Ivanhoe", authors=["Scott, Walter"]),
            _entry(2, "P", title="ivanhoe.", authors=["Scott, Walter"]),
        ]
        selection = bts.select_works(entries, bts.SelectionConfig(), bts.ChunkConfig())
        assert selection.rejections.get("duplicate_work") == 1

    def test_selection_is_deterministic_under_a_seed(self) -> None:
        entries = [_entry(i, "P") for i in range(100)]
        config = bts.SelectionConfig(target_passages=21 * 9, seed=7)
        first = bts.select_works(entries, config, bts.ChunkConfig(max_per_work=3))
        second = bts.select_works(entries, config, bts.ChunkConfig(max_per_work=3))
        assert [w["gutenberg_id"] for w in first.works] == [
            w["gutenberg_id"] for w in second.works
        ]

    def test_rejection_reasons_are_counted_for_the_manifest(self) -> None:
        entries = [
            _entry(1, "P", languages=["fr"]),
            _entry(2, "P", dcmi_types=["Sound"]),
        ]
        selection = bts.select_works(entries, bts.SelectionConfig(), bts.ChunkConfig())
        assert selection.rejections["language"] == 1
        assert selection.rejections["not_text"] == 1


# =============================================================================
# Record construction
# =============================================================================


class TestBuildRecords:
    def _work(self, gid: int = 1342, letter: str = "P") -> dict[str, Any]:
        return _entry(
            gid,
            letter,
            title="Pride and Prejudice",
            authors=["Austen, Jane"],
            lcsh_headings=["England -- Fiction"],
            author_birth_year=1775,
            author_death_year=1817,
            issued="1998-06-01",
        ).to_dict()

    def test_records_carry_the_full_natural_schema(self, tmp_path: Path) -> None:
        texts = tmp_path / "texts"
        texts.mkdir()
        (texts / "1342.txt").write_text(make_gutenberg_text(), encoding="utf-8")

        records, problems = build = bts.build_records(
            [self._work()],
            texts,
            bts.ChunkConfig(target_words=200, min_words=50, max_per_work=3),
            retrieved_at="2026-08-26T00:00:00+00:00",
        )
        assert build
        assert not problems
        assert records

        record = records[0]
        assert record["id"].startswith("pg-1342-")
        assert record["lcc_code"] == "P"
        assert record["lcc_name"] == "Language and Literature"
        assert record["lcgft_form"] == "Fiction"
        assert record["topics"] == ["England"]
        assert record["source_type"] == SourceType.NATURAL.value
        assert record["source"] == NaturalSource.PROJECT_GUTENBERG.value
        assert (
            record["contamination_status"]
            == ContaminationStatus.KNOWN_CONTAMINATED.value
        )
        assert record["provenance"]["gutenberg_id"] == 1342
        assert record["provenance"]["lcc_subclasses"] == ["PR"]
        assert "chunk_index" in record["provenance"]
        assert record["provenance"]["text_url"].endswith("/cache/epub/1342/pg1342.txt")

    def test_label_space_is_declared_so_topics_are_not_confused_with_shelfs(
        self, tmp_path: Path
    ) -> None:
        texts = tmp_path / "texts"
        texts.mkdir()
        (texts / "1342.txt").write_text(make_gutenberg_text(), encoding="utf-8")
        records, _ = bts.build_records(
            [self._work()],
            texts,
            bts.ChunkConfig(),
            retrieved_at="2026-08-26T00:00:00+00:00",
        )
        assert records[0]["label_space"]["topics"] == "lcsh_main_headings"

    def test_no_licence_text_survives_into_a_passage(self, tmp_path: Path) -> None:
        texts = tmp_path / "texts"
        texts.mkdir()
        (texts / "1342.txt").write_text(make_gutenberg_text(), encoding="utf-8")
        records, _ = bts.build_records(
            [self._work()],
            texts,
            bts.ChunkConfig(),
            retrieved_at="2026-08-26T00:00:00+00:00",
        )
        for record in records:
            assert "PROJECT GUTENBERG" not in record["text"].upper()

    def test_missing_text_is_reported_not_fatal(self, tmp_path: Path) -> None:
        records, problems = bts.build_records(
            [self._work()], tmp_path, bts.ChunkConfig(), retrieved_at="x"
        )
        assert records == []
        assert problems["text_missing"] == 1

    def test_unchunkable_text_is_reported(self, tmp_path: Path) -> None:
        texts = tmp_path / "texts"
        texts.mkdir()
        (texts / "1342.txt").write_text("tiny", encoding="utf-8")
        records, problems = bts.build_records(
            [self._work()], texts, bts.ChunkConfig(), retrieved_at="x"
        )
        assert records == []
        assert problems["no_usable_chunks"] == 1


# =============================================================================
# Polite fetching (all network mocked)
# =============================================================================


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self, _size: int | None = None) -> bytes:
        payload, self._payload = self._payload, b""
        return payload

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *args: Any) -> bool:
        return False


class TestPoliteFetcher:
    def test_rate_limit_is_enforced_between_requests(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        slept: list[float] = []
        monkeypatch.setattr(bts.time, "sleep", slept.append)
        monkeypatch.setattr(
            bts.urllib.request, "urlopen", lambda *_a, **_k: _FakeResponse(b"payload")
        )

        fetcher = bts.PoliteFetcher(min_interval=5.0, respect_robots=False)
        fetcher.get("https://www.gutenberg.org/cache/epub/1/pg1.txt")
        fetcher.get("https://www.gutenberg.org/cache/epub/2/pg2.txt")

        assert slept, "second request must wait for the rate-limit interval"
        assert max(slept) <= 5.0

    def test_user_agent_identifies_the_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[Any] = []

        def fake_urlopen(request: Any, **_kwargs: Any) -> _FakeResponse:
            seen.append(request)
            return _FakeResponse(b"body")

        monkeypatch.setattr(bts.time, "sleep", lambda _s: None)
        monkeypatch.setattr(bts.urllib.request, "urlopen", fake_urlopen)

        bts.PoliteFetcher(min_interval=0.0, respect_robots=False).get(
            "https://example.org/x"
        )
        assert "SHELF" in seen[0].get_header("User-agent")

    def test_robots_disallow_refuses_the_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(bts.time, "sleep", lambda _s: None)
        monkeypatch.setattr(
            bts.urllib.request,
            "urlopen",
            lambda *_a, **_k: _FakeResponse(
                b"User-agent: *\nDisallow: /ebooks/search\n"
            ),
        )
        fetcher = bts.PoliteFetcher(min_interval=0.0)
        assert fetcher.allowed("https://www.gutenberg.org/cache/epub/1/pg1.txt")
        with pytest.raises(PermissionError):
            fetcher.get("https://www.gutenberg.org/ebooks/search?q=x")

    def test_transient_failures_are_retried_then_surfaced(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        attempts: list[int] = []

        def flaky(*_args: Any, **_kwargs: Any) -> Any:
            attempts.append(1)
            raise urllib.error.URLError("boom")

        monkeypatch.setattr(bts.time, "sleep", lambda _s: None)
        monkeypatch.setattr(bts.urllib.request, "urlopen", flaky)

        fetcher = bts.PoliteFetcher(
            min_interval=0.0, max_retries=3, respect_robots=False
        )
        with pytest.raises(urllib.error.URLError):
            fetcher.get("https://example.org/x")
        assert len(attempts) == 3

    @pytest.mark.usefixtures("no_network")
    def test_download_is_skipped_when_already_cached(self, tmp_path: Path) -> None:
        destination = tmp_path / "rdf-files.tar.bz2"
        destination.write_bytes(b"already here")
        fetcher = bts.PoliteFetcher(min_interval=0.0, respect_robots=False)
        assert fetcher.download("https://example.org/x", destination) == destination
        assert destination.read_bytes() == b"already here"


# =============================================================================
# CLI behaviour
# =============================================================================


class TestCli:
    @pytest.mark.usefixtures("no_network")
    def test_dry_run_touches_no_network_and_writes_nothing(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cache = tmp_path / "cache"
        out = tmp_path / "out"
        code = bts.main(
            ["--dry-run", "--cache-dir", str(cache), "--output-dir", str(out)],
        )
        assert code == 0
        assert not cache.exists()
        assert not out.exists()
        assert "[dry-run]" in capsys.readouterr().out

    @pytest.mark.usefixtures("no_network")
    def test_contamination_notice_is_printed_before_any_work(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        bts.main(
            [
                "--dry-run",
                "--cache-dir",
                str(tmp_path / "c"),
                "--output-dir",
                str(tmp_path / "o"),
            ]
        )
        out = capsys.readouterr().out
        assert "NOT contamination-free" in out
        assert "SKEW:" in out

    @pytest.mark.usefixtures("no_network")
    def test_missing_prerequisite_stage_fails_cleanly(self, tmp_path: Path) -> None:
        code = bts.main(
            [
                "--stage",
                "select",
                "--cache-dir",
                str(tmp_path / "c"),
                "--output-dir",
                str(tmp_path / "o"),
            ]
        )
        assert code == 1

    @pytest.mark.usefixtures("no_network")
    def test_limit_caps_the_number_of_selected_works(self, tmp_path: Path) -> None:
        cache = tmp_path / "cache"
        cache.mkdir(parents=True)
        catalog = cache / "catalog.jsonl"
        with catalog.open("w", encoding="utf-8") as handle:
            for i in range(60):
                handle.write(json.dumps(_entry(i, "P").to_dict()) + "\n")

        out = tmp_path / "out"
        code = bts.main(
            [
                "--stage",
                "select",
                "--cache-dir",
                str(cache),
                "--output-dir",
                str(out),
                "--limit",
                "3",
                "--max-works-per-author",
                "99",
            ]
        )
        assert code == 0
        selection = json.loads((out / "selection.json").read_text())
        assert len(selection["works"]) == 3

    @pytest.mark.usefixtures("no_network")
    def test_end_to_end_build_from_cached_inputs(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cache = tmp_path / "cache"
        texts = cache / "texts"
        texts.mkdir(parents=True)
        out = tmp_path / "out"
        out.mkdir()

        works = []
        for i, letter in enumerate(["P", "Q", "B"]):
            gid = 100 + i
            works.append(
                _entry(
                    gid,
                    letter,
                    title=f"Work {gid}",
                    lcsh_headings=["Adventure -- Fiction"],
                    author_death_year=1900 + i,
                    issued="2001-01-01",
                ).to_dict()
            )
            (texts / f"{gid}.txt").write_text(make_gutenberg_text(), encoding="utf-8")

        (out / "selection.json").write_text(
            json.dumps({"works": works, "config": {}, "rejections": {}}),
            encoding="utf-8",
        )

        code = bts.main(
            ["--stage", "build", "--cache-dir", str(cache), "--output-dir", str(out)]
        )
        assert code == 0

        records = [
            json.loads(line)
            for line in (out / "records.jsonl").read_text().splitlines()
        ]
        assert records
        assert {r["lcc_code"] for r in records} == {"P", "Q", "B"}

        manifest = json.loads((out / "manifest.json").read_text())
        assert (
            manifest["contamination_status"]
            == ContaminationStatus.KNOWN_CONTAMINATED.value
        )
        assert "NOT contamination-free" in manifest["contamination_notice"]
        assert manifest["records_sha256"]

        report = (out / "distribution_report.txt").read_text()
        assert "REALIZED DISTRIBUTION" in report
        capsys.readouterr()


# =============================================================================
# shelf.hub.transfer: validation
# =============================================================================


class TestValidation:
    def test_a_well_formed_record_validates(self) -> None:
        assert validate_record(make_record()) == []

    def test_missing_required_field_is_an_error(self) -> None:
        record = make_record()
        del record["lcc_code"]
        errors = validate_record(record)
        assert any("lcc_code" in e for e in errors)

    def test_lcc_code_must_be_one_of_the_21_main_classes(self) -> None:
        errors = validate_record(make_record(lcc="PR"))
        assert any("21 LCC main classes" in e for e in errors)

    def test_a_synthetic_record_is_rejected_from_a_transfer_slice(self) -> None:
        errors = validate_record(make_record(source_type=SourceType.SYNTHETIC.value))
        assert any("source_type" in e for e in errors)

    def test_a_natural_record_may_not_claim_it_is_uncontaminated(self) -> None:
        errors = validate_record(
            make_record(status=ContaminationStatus.NOT_KNOWN_CONTAMINATED.value)
        )
        assert errors
        assert any("not_known_contaminated" in e for e in errors)

    def test_contamination_status_must_match_the_source(self) -> None:
        errors = validate_record(make_record(status=ContaminationStatus.UNKNOWN.value))
        assert any("property of the source" in e for e in errors)

    def test_gutenberg_records_need_their_provenance(self) -> None:
        record = make_record()
        del record["provenance"]["gutenberg_id"]
        assert any("gutenberg_id" in e for e in validate_record(record))

    def test_duplicate_ids_are_an_error(self) -> None:
        report = validate_records([make_record(), make_record()])
        assert not report.ok
        assert any("duplicate id" in e for e in report.errors)

    def test_duplicate_text_is_a_warning_not_an_error(self) -> None:
        report = validate_records([make_record("a"), make_record("b")])
        assert report.ok
        assert any("duplicates" in w for w in report.warnings)

    def test_missing_form_labels_are_surfaced_as_a_warning(self) -> None:
        report = validate_records([make_record("a", lcgft_form="")])
        assert report.ok
        assert any("lcgft_form" in w for w in report.warnings)

    def test_summary_states_pass_or_fail(self) -> None:
        assert validate_records([make_record()]).summary().startswith("PASS")
        assert validate_records([make_record(lcc="ZZ")]).summary().startswith("FAIL")


class TestLoading:
    def test_round_trip_through_jsonl(self, tmp_path: Path) -> None:
        path = tmp_path / "records.jsonl"
        path.write_text(
            "\n".join(json.dumps(make_record(f"pg-1-{i:04d}")) for i in range(3))
            + "\n",
            encoding="utf-8",
        )
        records = load_transfer_slice(path)
        assert len(records) == 3

    def test_strict_mode_raises_on_an_invalid_slice(self, tmp_path: Path) -> None:
        path = tmp_path / "records.jsonl"
        path.write_text(json.dumps(make_record(lcc="ZZ")) + "\n", encoding="utf-8")
        with pytest.raises(TransferValidationError):
            load_transfer_slice(path)
        assert load_transfer_slice(path, strict=False)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_transfer_slice(tmp_path / "nope.jsonl")


# =============================================================================
# shelf.hub.transfer: the no-pooling rule
# =============================================================================


class TestSourceTypeSeparation:
    def test_pooling_synthetic_and_natural_raises(self) -> None:
        records = [
            make_record("a"),
            make_record("b", source_type=SourceType.SYNTHETIC.value),
        ]
        with pytest.raises(SourceTypePoolingError):
            assert_single_source_type(records)

    def test_a_homogeneous_slice_returns_its_source_type(self) -> None:
        assert assert_single_source_type([make_record()]) == SourceType.NATURAL.value

    def test_split_gives_one_bucket_per_condition(self) -> None:
        buckets = split_by_source_type(
            [make_record("a"), make_record("b", source_type=SourceType.SYNTHETIC.value)]
        )
        assert set(buckets) == {SourceType.NATURAL.value, SourceType.SYNTHETIC.value}
        assert len(buckets[SourceType.NATURAL.value]) == 1

    def test_records_without_a_source_type_default_to_synthetic(self) -> None:
        record = make_record()
        del record["source_type"]
        assert assert_single_source_type([record]) == SourceType.SYNTHETIC.value


# =============================================================================
# shelf.hub.transfer: distribution reporting
# =============================================================================


class TestDistributionReporting:
    def _slice(self) -> list[dict[str, Any]]:
        records = []
        for i in range(10):
            records.append(make_record(f"pg-1-{i:04d}", lcc="P"))
        for i in range(2):
            records.append(make_record(f"pg-2-{i:04d}", lcc="Q"))
        return records

    def test_profile_counts_passages_and_works(self) -> None:
        profile = describe_slice(self._slice())
        assert profile.n_records == 12
        # All fixture records share gutenberg_id 1342, so this is one work.
        assert profile.n_works == 1
        assert profile.lcc_counts["P"] == 10
        assert profile.lcc_counts["Q"] == 2

    def test_empty_slice_profiles_cleanly(self) -> None:
        assert describe_slice([]).n_records == 0

    def test_length_and_language_are_measured(self) -> None:
        profile = describe_slice(self._slice())
        assert profile.length.n == 12
        assert profile.language_counts["en"] == 12

    def test_author_date_range_is_reported(self) -> None:
        profile = describe_slice(self._slice())
        assert profile.author_year_range == (1817, 1817)
        assert profile.author_century_counts == {"1801-1900": 12}

    def test_author_centuries_sort_chronologically_not_lexically(self) -> None:
        records = [
            make_record("a", provenance={"gutenberg_id": 1, "author_death_year": 280}),
            make_record("b", provenance={"gutenberg_id": 2, "author_death_year": 1850}),
        ]
        assert list(describe_slice(records).author_century_counts) == [
            "201-300",
            "1801-1900",
        ]

    def test_entropy_is_low_for_a_skewed_slice(self) -> None:
        profile = describe_slice(self._slice())
        assert 0.0 < profile.lcc_normalized_entropy < 0.3

    def test_comparison_names_the_missing_classes(self) -> None:
        comparison = compare_to_synthetic(describe_slice(self._slice()))
        assert "B" in comparison.missing_classes
        assert comparison.natural_entropy < comparison.synthetic_entropy
        assert comparison.total_variation > 0.5

    def test_over_and_under_representation_are_signed_correctly(self) -> None:
        comparison = compare_to_synthetic(describe_slice(self._slice()))
        assert comparison.over_represented[0][0] == "P"
        assert comparison.over_represented[0][1] > 0
        assert all(delta < 0 for _c, delta in comparison.under_represented)

    def test_total_variation_of_identical_distributions_is_zero(self) -> None:
        counts = {"P": 3, "Q": 7}
        assert total_variation_distance(counts, counts) == pytest.approx(0.0)

    def test_report_leads_with_the_caveats(self) -> None:
        _profile, _comparison, report = build_report(self._slice())
        assert CONTAMINATION_NOTICE in report
        assert SKEW_NOTICE in report
        # The caveats precede the numbers, not the other way round.
        assert report.index(CONTAMINATION_NOTICE) < report.index("LCC MAIN CLASS")

    def test_report_shows_every_class_including_the_empty_ones(self) -> None:
        _profile, _comparison, report = build_report(self._slice())
        for letter in "ABCDEFGHJKLMNPQRSTUVZ":
            assert f"\n  {letter} " in report

    def test_pool_coverage_section_exposes_near_exhausted_classes(self) -> None:
        profile = describe_slice(self._slice())
        report = format_distribution_report(
            profile, pool_counts={"P": 31_524, "Q": 103, "B": 0}
        )
        assert "SOURCE POOL COVERAGE" in report
        assert "31,524" in report
        assert "sampling fraction spans" in report

    def test_pool_coverage_section_is_omitted_when_unknown(self) -> None:
        report = format_distribution_report(describe_slice(self._slice()))
        assert "SOURCE POOL COVERAGE" not in report

    def test_lcgft_coverage_percentage_is_stated(self) -> None:
        _profile, _comparison, report = build_report(self._slice())
        assert "% of passages" in report

    def test_profile_serializes_with_its_notices_attached(self) -> None:
        payload = describe_slice(self._slice()).to_dict()
        assert payload["contamination_notice"] == CONTAMINATION_NOTICE
        assert payload["skew_notice"] == SKEW_NOTICE
        assert payload["lcc_counts"]["P"] == 10
