"""Tests for §12.1 promotion checks (shelf.qc.promotion)."""

from __future__ import annotations

from shelf.qc.gates import Gate, GateResult, run_gates
from shelf.qc.promotion import run_promotion_checks

GOOD_BODY = (
    "This report documents significant deforestation patterns across "
    "timber-dependent regions, with particular attention to employment "
    "decline in extractive sectors and demographic shifts across the region."
)


def _make_record(doc_id, generator, lcc_code="G", topics=None, spec_id=None):
    return {
        "id": doc_id,
        "text": GOOD_BODY,
        "title": "Regional Report",
        "generator": generator,
        "lcc_code": lcc_code,
        "topics": topics or ["Deforestation"],
        "spec_id": spec_id,
    }


class TestGeneratorStats:
    def test_all_pass_counts_as_retained(self):
        records = [_make_record("1", "gpt-5.1"), _make_record("2", "gpt-5.1")]
        qc = {
            r["id"]: run_gates(
                doc_id=r["id"], title=r["title"], body=r["text"], topics=r["topics"]
            )
            for r in records
        }
        report = run_promotion_checks(records, qc)
        stats = report.generator_stats["gpt-5.1"]
        assert stats.generated == 2
        assert stats.retained == 2
        assert stats.rejected == 0

    def test_failed_gate_counted_as_rejection_with_reason(self):
        records = [_make_record("1", "gpt-5.1")]
        # Force an empty body -> fails G1.
        qc = {"1": run_gates(doc_id="1", title="Title", body="")}
        report = run_promotion_checks(records, qc)
        stats = report.generator_stats["gpt-5.1"]
        assert stats.generated == 1
        assert stats.retained == 0
        assert stats.rejected == 1
        assert stats.rejected_by_gate[Gate.PARSE.value] == 1

    def test_missing_qc_result_counts_as_rejection(self):
        records = [_make_record("1", "gpt-5.1")]
        report = run_promotion_checks(records, {})
        stats = report.generator_stats["gpt-5.1"]
        assert stats.rejected == 1
        assert stats.rejected_by_gate["missing_qc_result"] == 1

    def test_generator_field_falls_back_to_model(self):
        record = {"id": "1", "text": GOOD_BODY, "model": "claude-haiku-4-5"}
        qc = {"1": run_gates(doc_id="1", title="T", body=GOOD_BODY)}
        report = run_promotion_checks([record], qc)
        assert "claude-haiku-4-5" in report.generator_stats

    def test_unknown_generator_when_no_field_present(self):
        record = {"id": "1", "text": GOOD_BODY}
        qc = {"1": run_gates(doc_id="1", title="T", body=GOOD_BODY)}
        report = run_promotion_checks([record], qc)
        assert "unknown" in report.generator_stats

    def test_multiple_generators_tracked_independently(self):
        records = [
            _make_record("1", "gpt-5.1"),
            _make_record("2", "claude-opus-4.5"),
        ]
        qc = {
            r["id"]: run_gates(
                doc_id=r["id"], title=r["title"], body=r["text"], topics=r["topics"]
            )
            for r in records
        }
        report = run_promotion_checks(records, qc)
        assert set(report.generator_stats) == {"gpt-5.1", "claude-opus-4.5"}


class TestSpecCoverage:
    def test_spec_ids_present_computes_coverage(self):
        records = [
            _make_record("1", "gpt-5.1", spec_id="spec-a"),
            _make_record("2", "gpt-5.1", spec_id="spec-b"),
            _make_record("3", "gpt-5.1", spec_id=None),
        ]
        qc = {
            r["id"]: run_gates(
                doc_id=r["id"], title=r["title"], body=r["text"], topics=r["topics"]
            )
            for r in records
        }
        report = run_promotion_checks(records, qc)
        assert report.spec_coverage is not None
        assert report.spec_coverage.unique_spec_ids == 2
        assert report.spec_coverage.missing_spec_id_count == 1

    def test_no_spec_ids_reports_none_with_note(self):
        records = [_make_record("1", "gpt-5.1")]
        qc = {
            "1": run_gates(
                doc_id="1", title="T", body=GOOD_BODY, topics=["Deforestation"]
            )
        }
        report = run_promotion_checks(records, qc)
        assert report.spec_coverage is None
        assert any("spec_id" in note for note in report.notes)


class TestSplitsAndHashing:
    def test_split_sizes_reported(self):
        records = [_make_record(str(i), "gpt-5.1") for i in range(5)]
        qc = {
            r["id"]: run_gates(
                doc_id=r["id"], title=r["title"], body=r["text"], topics=r["topics"]
            )
            for r in records
        }
        splits = {"train": ["0", "1", "2"], "test": ["3", "4"]}
        report = run_promotion_checks(records, qc, splits=splits)
        assert report.split_sizes == {"train": 3, "test": 2}

    def test_split_hash_is_stable_sha256_hex(self):
        records = [_make_record(str(i), "gpt-5.1") for i in range(3)]
        qc = {
            r["id"]: run_gates(
                doc_id=r["id"], title=r["title"], body=r["text"], topics=r["topics"]
            )
            for r in records
        }
        splits = {"train": ["0", "1", "2"]}
        report_a = run_promotion_checks(records, qc, splits=splits)
        report_b = run_promotion_checks(records, qc, splits=splits)
        assert report_a.split_hashes["train"] == report_b.split_hashes["train"]
        assert len(report_a.split_hashes["train"]) == 64  # hex sha256

    def test_split_hash_order_independent(self):
        records = [_make_record(str(i), "gpt-5.1") for i in range(3)]
        qc = {
            r["id"]: run_gates(
                doc_id=r["id"], title=r["title"], body=r["text"], topics=r["topics"]
            )
            for r in records
        }
        forward = run_promotion_checks(records, qc, splits={"train": ["0", "1", "2"]})
        shuffled = run_promotion_checks(records, qc, splits={"train": ["2", "0", "1"]})
        assert forward.split_hashes["train"] == shuffled.split_hashes["train"]

    def test_different_content_gives_different_hash(self):
        records_a = [_make_record("0", "gpt-5.1")]
        records_b = [{**_make_record("0", "gpt-5.1"), "text": "totally different text"}]
        qc_a = {"0": run_gates(doc_id="0", title="T", body=records_a[0]["text"])}
        qc_b = {"0": run_gates(doc_id="0", title="T", body=records_b[0]["text"])}
        report_a = run_promotion_checks(records_a, qc_a, splits={"train": ["0"]})
        report_b = run_promotion_checks(records_b, qc_b, splits={"train": ["0"]})
        assert report_a.split_hashes["train"] != report_b.split_hashes["train"]

    def test_no_splits_notes_skip(self):
        records = [_make_record("1", "gpt-5.1")]
        qc = {
            "1": run_gates(
                doc_id="1", title="T", body=GOOD_BODY, topics=["Deforestation"]
            )
        }
        report = run_promotion_checks(records, qc)
        assert report.split_sizes == {}
        assert any("splits" in note for note in report.notes)


class TestDimensionDistributions:
    def test_realized_counts_by_lcc_code(self):
        records = [
            _make_record("0", "gpt-5.1", lcc_code="G"),
            _make_record("1", "gpt-5.1", lcc_code="G"),
            _make_record("2", "gpt-5.1", lcc_code="K"),
        ]
        qc = {
            r["id"]: run_gates(
                doc_id=r["id"], title=r["title"], body=r["text"], topics=r["topics"]
            )
            for r in records
        }
        splits = {"train": ["0", "1", "2"]}
        report = run_promotion_checks(records, qc, splits=splits)
        assert report.dimension_distributions["lcc_code"]["train"] == {"G": 2, "K": 1}

    def test_multivalued_field_counts_each_value(self):
        records = [_make_record("0", "gpt-5.1", topics=["Labor", "Population"])]
        qc = {
            "0": run_gates(
                doc_id="0", title="T", body=GOOD_BODY, topics=["Labor", "Population"]
            )
        }
        splits = {"train": ["0"]}
        report = run_promotion_checks(records, qc, splits=splits)
        assert report.dimension_distributions["topics"]["train"] == {
            "Labor": 1,
            "Population": 1,
        }

    def test_absent_dimension_noted_and_skipped(self):
        records = [_make_record("0", "gpt-5.1")]
        qc = {
            "0": run_gates(
                doc_id="0", title="T", body=GOOD_BODY, topics=["Deforestation"]
            )
        }
        report = run_promotion_checks(
            records, qc, splits={"train": ["0"]}, dimension_fields=("difficulty",)
        )
        assert "difficulty" not in report.dimension_distributions
        assert any("difficulty" in note for note in report.notes)


class TestCrossSplitCollisions:
    def test_spec_id_in_two_splits_is_a_collision(self):
        records = [
            _make_record("0", "gpt-5.1", spec_id="spec-a"),
            _make_record("1", "gpt-5.1", spec_id="spec-a"),
        ]
        qc = {
            r["id"]: run_gates(
                doc_id=r["id"], title=r["title"], body=r["text"], topics=r["topics"]
            )
            for r in records
        }
        splits = {"train": ["0"], "test": ["1"]}
        report = run_promotion_checks(records, qc, splits=splits)
        assert report.cross_split_spec_collisions == ("spec-a",)

    def test_spec_id_confined_to_one_split_is_not_a_collision(self):
        records = [
            _make_record("0", "gpt-5.1", spec_id="spec-a"),
            _make_record("1", "gpt-5.1", spec_id="spec-b"),
        ]
        qc = {
            r["id"]: run_gates(
                doc_id=r["id"], title=r["title"], body=r["text"], topics=r["topics"]
            )
            for r in records
        }
        splits = {"train": ["0"], "test": ["1"]}
        report = run_promotion_checks(records, qc, splits=splits)
        assert report.cross_split_spec_collisions == ()


class TestNearDuplicateReport:
    def test_no_dedup_pairs_gives_zeroed_report_with_note(self):
        records = [_make_record("0", "gpt-5.1")]
        qc = {
            "0": run_gates(
                doc_id="0", title="T", body=GOOD_BODY, topics=["Deforestation"]
            )
        }
        report = run_promotion_checks(records, qc)
        assert report.near_duplicates.duplicate_documents == 0
        assert any("dedup_pairs" in note for note in report.notes)

    def test_within_generator_pair_counted(self):
        records = [
            _make_record("0", "gpt-5.1"),
            _make_record("1", "gpt-5.1"),
        ]
        qc = {
            r["id"]: run_gates(
                doc_id=r["id"], title=r["title"], body=r["text"], topics=r["topics"]
            )
            for r in records
        }
        report = run_promotion_checks(records, qc, dedup_pairs=[("0", "1", 0.95)])
        assert report.near_duplicates.within_generator_pairs == 1
        assert report.near_duplicates.across_generator_pairs == 0
        assert report.near_duplicates.duplicate_documents == 2
        assert report.near_duplicates.within_generator_rate == 1.0

    def test_across_generator_pair_counted(self):
        records = [
            _make_record("0", "gpt-5.1"),
            _make_record("1", "claude-opus-4.5"),
        ]
        qc = {
            r["id"]: run_gates(
                doc_id=r["id"], title=r["title"], body=r["text"], topics=r["topics"]
            )
            for r in records
        }
        report = run_promotion_checks(records, qc, dedup_pairs=[("0", "1", 0.95)])
        assert report.near_duplicates.across_generator_pairs == 1
        assert report.near_duplicates.within_generator_pairs == 0
        assert report.near_duplicates.within_generator_rate == 0.0

    def test_duplicate_rate_property(self):
        records = [_make_record(str(i), "gpt-5.1") for i in range(4)]
        qc = {
            r["id"]: run_gates(
                doc_id=r["id"], title=r["title"], body=r["text"], topics=r["topics"]
            )
            for r in records
        }
        report = run_promotion_checks(records, qc, dedup_pairs=[("0", "1", 0.95)])
        assert report.near_duplicates.duplicate_rate == 0.5  # 2 of 4 flagged


class TestToDict:
    def test_to_dict_is_json_serializable(self):
        import json

        records = [_make_record("0", "gpt-5.1", spec_id="spec-a")]
        qc = {
            "0": run_gates(
                doc_id="0", title="T", body=GOOD_BODY, topics=["Deforestation"]
            )
        }
        report = run_promotion_checks(
            records,
            qc,
            splits={"train": ["0"]},
            dedup_pairs=[],
        )
        payload = json.dumps(report.to_dict())
        assert "generator_stats" in payload
        assert "gpt-5.1" in payload


class TestQCResultWithNearDuplicateFeedsGeneratorStats:
    def test_near_duplicate_failure_counts_as_rejection(self):
        base = run_gates(
            doc_id="1", title="T", body=GOOD_BODY, topics=["Deforestation"]
        )
        dup = base.with_near_duplicate(
            GateResult(Gate.NEAR_DUPLICATE, False, value=0.95)
        )
        records = [_make_record("1", "gpt-5.1")]
        report = run_promotion_checks(records, {"1": dup})
        stats = report.generator_stats["gpt-5.1"]
        assert stats.rejected == 1
        assert stats.rejected_by_gate[Gate.NEAR_DUPLICATE.value] == 1
