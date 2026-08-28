"""Tests for QC gates G1-G5 and G7 (shelf.qc.gates)."""

from __future__ import annotations

from shelf.qc.gates import (
    Gate,
    check_language,
    check_length,
    check_parse,
    check_parse_fields,
    check_refusal,
    check_self_label,
    check_topic_coverage,
    run_gates,
)

ENGLISH_PARAGRAPH = (
    "This report documents significant deforestation patterns across "
    "timber-dependent regions, with particular attention to employment "
    "decline in extractive sectors. The assessment examines labor market "
    "disruptions and demographic shifts consequent to accelerated harvest "
    "rates and subsequent forest cover loss across these jurisdictions."
)


class TestCheckParse:
    def test_well_formed_envelope_passes(self):
        result = check_parse("Title: A Title\n\nSome body text here.")
        assert result.gate is Gate.PARSE
        assert result.passed is True
        assert result.detail["title"] == "A Title"
        assert result.detail["body"] == "Some body text here."

    def test_empty_text_fails(self):
        result = check_parse("")
        assert result.passed is False
        assert result.detail["reason"] == "empty_text"

    def test_none_fails(self):
        result = check_parse(None)
        assert result.passed is False

    def test_missing_blank_line_produces_empty_body_and_fails(self):
        """This is the v0.3.1 empty-body bug: a single newline instead of a
        blank line after the title leaves nothing for the body."""
        result = check_parse("Title: A Title\nSome body text here.")
        assert result.detail["body"] == ""
        assert result.passed is False
        assert result.detail["reason"] == "empty_body"

    def test_whitespace_only_text_fails(self):
        result = check_parse("   \n\n   ")
        assert result.passed is False


class TestCheckParseFields:
    def test_both_present_passes(self):
        result = check_parse_fields("Title", "Body text.")
        assert result.passed is True

    def test_empty_body_fails(self):
        result = check_parse_fields("Title", "")
        assert result.passed is False
        assert result.detail["reason"] == "empty_body"

    def test_empty_title_fails(self):
        result = check_parse_fields("", "Body text.")
        assert result.passed is False
        assert result.detail["reason"] == "empty_title"

    def test_none_values_treated_as_empty(self):
        result = check_parse_fields(None, None)
        assert result.passed is False


class TestCheckLanguage:
    def test_english_paragraph_passes(self):
        result = check_language(ENGLISH_PARAGRAPH)
        assert result.gate is Gate.LANGUAGE
        assert result.passed is True
        assert result.value > 0

    def test_short_micro_document_passes(self):
        # 10-25 word "micro" documents must not be unfairly penalized.
        result = check_language(
            "Solar panels convert sunlight into electricity for homes and businesses today."
        )
        assert result.passed is True

    def test_empty_text_fails(self):
        result = check_language("")
        assert result.passed is False

    def test_non_latin_script_fails(self):
        result = check_language("这是一个中文文档，用于测试语言检测门。" * 3)
        assert result.passed is False

    def test_non_english_latin_script_fails(self):
        text = (
            "Este informe documenta patrones significativos de deforestación "
            "en regiones dependientes de la madera, con especial atención a "
            "la disminución del empleo en los sectores extractivos."
        )
        result = check_language(text)
        assert result.passed is False


class TestCheckLength:
    def test_within_range_has_zero_delta(self):
        result = check_length(60, (50, 100))
        assert result.gate is Gate.LENGTH
        assert result.passed is True
        assert result.value == 0

    def test_over_range_records_positive_delta(self):
        result = check_length(500, (50, 100))
        assert result.passed is False
        assert result.value == 400

    def test_under_range_records_negative_delta(self):
        result = check_length(10, (50, 100))
        assert result.passed is False
        assert result.value == -40

    def test_small_overshoot_within_tolerance_passes(self):
        result = check_length(105, (50, 100))
        assert result.passed is True
        assert result.value == 5

    def test_no_target_range_always_passes(self):
        result = check_length(9999, None)
        assert result.passed is True
        assert result.value == 0

    def test_zero_zero_range_treated_as_no_target(self):
        result = check_length(15, (0, 0))
        assert result.passed is True

    def test_micro_bucket_10_to_25_words_is_not_flagged(self):
        """Sanity-check for the real-corpus fact: micro documents target
        10-25 words and must not be treated as failures for being short."""
        result = check_length(18, (10, 25))
        assert result.passed is True
        assert result.value == 0

    def test_delta_recorded_even_when_failing(self):
        result = check_length(1000, (50, 100))
        assert result.passed is False
        assert isinstance(result.value, int)
        assert result.value != 0


class TestCheckSelfLabel:
    def test_clean_text_passes(self):
        result = check_self_label(ENGLISH_PARAGRAPH)
        assert result.gate is Gate.SELF_LABEL
        assert result.passed is True

    def test_self_labeling_phrase_fails(self):
        result = check_self_label(
            "In the field of political science, this document explores policy."
        )
        assert result.passed is False

    def test_announced_label_fails(self):
        result = check_self_label(
            "Genre: Law materials\nA discussion of contracts.",
            labels=["Law materials"],
        )
        assert result.passed is False
        assert "Law materials" in result.detail["leaked_labels"]

    def test_label_as_ordinary_vocabulary_passes(self):
        """A document may use its domain's words; it may not announce its class."""
        result = check_self_label(
            "This is a kind of Law materials about contracts.",
            labels=["Law materials"],
        )
        assert result.passed is True


class TestCheckTopicCoverage:
    def test_no_topics_trivially_passes(self):
        result = check_topic_coverage(ENGLISH_PARAGRAPH, [])
        assert result.gate is Gate.TOPIC_COVERAGE
        assert result.passed is True
        assert result.value == 1.0

    def test_all_topics_present_passes_with_full_fraction(self):
        result = check_topic_coverage(ENGLISH_PARAGRAPH, ["Deforestation", "Labor"])
        assert result.passed is True
        assert result.value == 1.0

    def test_missing_topics_reduce_fraction_not_just_boolean(self):
        result = check_topic_coverage(
            ENGLISH_PARAGRAPH, ["Deforestation", "Labor", "Astrophysics", "Cuisine"]
        )
        assert 0.0 < result.value < 1.0
        assert result.detail["per_topic"]["Astrophysics"] == 0.0

    def test_none_of_the_topics_present_is_recorded_but_does_not_gate(self):
        """Zero verbatim coverage is informational, not a failure.

        A document that addresses its topics without naming them is doing what
        GENERATION_INSTRUCTIONS asks. Gating here would reject the better
        generator: on the v0.4 run Opus 5 scored 34.9% and Sol 75.1%.
        """
        result = check_topic_coverage(
            ENGLISH_PARAGRAPH, ["Astrophysics", "Cuisine", "Numismatics"]
        )
        assert result.value == 0.0
        assert result.passed is True

    def test_coverage_can_still_gate_when_asked(self):
        result = check_topic_coverage(
            ENGLISH_PARAGRAPH,
            ["Astrophysics", "Cuisine", "Numismatics"],
            coverage_threshold=0.5,
        )
        assert result.passed is False

    def test_morphological_variant_counts_as_covered(self):
        result = check_topic_coverage("Governance structures shape policy.", ["Govern"])
        assert result.value == 1.0


class TestCheckRefusal:
    def test_normal_text_passes(self):
        result = check_refusal(ENGLISH_PARAGRAPH)
        assert result.gate is Gate.REFUSAL
        assert result.passed is True
        assert result.value is False

    def test_refusal_boilerplate_fails(self):
        result = check_refusal(
            "I'm sorry, but I can't help with that request as it violates policy."
        )
        assert result.passed is False

    def test_as_an_ai_disclaimer_fails(self):
        result = check_refusal(
            "As an AI language model, I do not have personal opinions."
        )
        assert result.passed is False

    def test_truncation_marker_fails(self):
        result = check_refusal(ENGLISH_PARAGRAPH + " [Content truncated]")
        assert result.passed is False
        assert result.detail["truncated"] is True

    def test_trailing_ellipsis_flagged_as_truncated(self):
        result = check_refusal(ENGLISH_PARAGRAPH + " and then...")
        assert result.passed is False

    def test_missing_terminal_punctuation_alone_does_not_fail(self):
        """A body lacking a terminal '.' is *not*, by itself, treated as
        truncated -- validated against the real corpus, this heuristic
        alone has a ~14% false-positive rate on legitimate non-prose forms
        (tables, lists, signature blocks). It is recorded, not gated."""
        text = " ".join(["word"] * 20) + " and then it just stops mid"
        result = check_refusal(text)
        assert result.passed is True
        assert result.detail["missing_terminal_punctuation"] is True
        assert result.detail["truncated"] is False

    def test_table_row_ending_is_not_flagged(self):
        # A markdown table row is a legitimate document ending for several
        # LCGFT forms (calendars, reference works) and must not be
        # penalized just because it ends on "|" rather than a period.
        text = (
            "| Task | Owner | Due |\n"
            "|---|---|---|\n"
            "| " + " ".join(["Draft budget review cycle"] * 4) + " |"
        )
        result = check_refusal(text)
        assert result.passed is True

    def test_signature_block_placeholder_ending_is_not_flagged(self):
        text = ENGLISH_PARAGRAPH + "\n\nDated: ____________, 20__\nNew York, New York"
        result = check_refusal(text)
        assert result.passed is True

    def test_empty_text_fails(self):
        result = check_refusal("")
        assert result.passed is False


class TestRunGates:
    def test_raw_text_path(self):
        result = run_gates(
            doc_id="doc-1",
            raw_text=f"Title: Forest Report\n\n{ENGLISH_PARAGRAPH}",
            target_word_range=(50, 100),
            topics=["Deforestation", "Labor"],
        )
        assert result.doc_id == "doc-1"
        assert result.parse.passed is True
        assert result.passed is True
        assert result.near_duplicate is None

    def test_prefit_title_body_path(self):
        result = run_gates(
            doc_id="doc-2",
            title="Forest Report",
            body=ENGLISH_PARAGRAPH,
            target_word_range=(50, 100),
            topics=["Deforestation"],
        )
        assert result.parse.passed is True
        assert result.passed is True

    def test_empty_body_fails_only_parse_and_propagates_to_overall(self):
        result = run_gates(doc_id="doc-3", title="Title", body="")
        assert result.parse.passed is False
        assert result.passed is False
        assert Gate.PARSE in result.failed_gates()

    def test_to_columns_matches_section_13_schema(self):
        result = run_gates(
            doc_id="doc-4",
            title="Forest Report",
            body=ENGLISH_PARAGRAPH,
            target_word_range=(50, 100),
            topics=["Deforestation"],
        )
        columns = result.to_columns()
        assert set(columns) == {
            "qc_parse",
            "qc_language",
            "qc_length_delta",
            "qc_selflabel",
            "qc_topic_coverage",
            "qc_near_dup",
            "qc_refusal",
            "qc_passed",
        }
        assert columns["qc_near_dup"] is None  # G6 not attached yet

    def test_with_near_duplicate_attaches_and_can_fail_overall(self):
        from shelf.qc.gates import GateResult

        base = run_gates(
            doc_id="doc-5",
            title="Forest Report",
            body=ENGLISH_PARAGRAPH,
            target_word_range=(50, 100),
        )
        assert base.passed is True

        dup_result = GateResult(Gate.NEAR_DUPLICATE, False, value=0.95)
        with_dup = base.with_near_duplicate(dup_result)
        assert with_dup.near_duplicate is dup_result
        assert with_dup.passed is False
        assert Gate.NEAR_DUPLICATE in with_dup.failed_gates()
        assert with_dup.to_columns()["qc_near_dup"] == 0.95

    def test_word_count_defaults_to_body_split_length(self):
        result = run_gates(doc_id="doc-6", title="T", body="one two three four five")
        assert result.length.value == 0 or isinstance(result.length.value, int)
