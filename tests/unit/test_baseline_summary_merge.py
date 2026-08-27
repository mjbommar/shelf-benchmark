"""Regression tests for scripts/baselines/run_all.py summary aggregation.

``summary.json`` describes the whole output directory, not the invocation
that happened to write it last. Before ``harvest_existing_results`` existed,
evaluating one model rewrote the summary with that single model and silently
discarded every other result. That is how a 22-model table was reduced to one
fine-tune entry, and how a published headline came to cite a run that no
longer existed on disk.

The invariants under test:

- A partial run recovers the results it did not produce.
- Results from the current run win over stale copies on disk.
- Unreadable files are skipped with a warning rather than killing the run.
- Only files matching a configured model and task are picked up.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

# run_all is a script, not a package module, so load it by path.
_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "baselines" / "run_all.py"
_spec = importlib.util.spec_from_file_location("baselines_run_all", _SCRIPT)
assert _spec is not None and _spec.loader is not None
run_all = importlib.util.module_from_spec(_spec)
sys.modules["baselines_run_all"] = run_all
_spec.loader.exec_module(run_all)


MODELS = {"tfidf": {"type": "tfidf"}, "bert": {"type": "sentence_transformer"}}
TASKS = {
    "classification": ["lcc_classification"],
    "retrieval": ["lcc_retrieval"],
}


def _write(out: Path, key: str, score: float) -> None:
    (out / f"{key}.json").write_text(json.dumps({"task": key, "primary_score": score}))


def test_partial_run_recovers_results_it_did_not_produce(tmp_path: Path) -> None:
    """The bug: a one-model run must not erase the other models."""
    for key in ("tfidf_lcc_classification", "tfidf_lcc_retrieval"):
        _write(tmp_path, key, 0.5)

    # This run only produced one cell, for a different model.
    produced = {"bert_lcc_classification": {"primary_score": 0.9}}

    merged, recovered = run_all.harvest_existing_results(
        tmp_path, MODELS, TASKS, produced
    )

    assert recovered == 2
    assert set(merged) == {
        "bert_lcc_classification",
        "tfidf_lcc_classification",
        "tfidf_lcc_retrieval",
    }
    # The freshly produced result is untouched.
    assert merged["bert_lcc_classification"]["primary_score"] == 0.9


def test_current_run_wins_over_stale_disk_copy(tmp_path: Path) -> None:
    """Re-running a cell replaces it; the old file does not shadow the new."""
    _write(tmp_path, "tfidf_lcc_classification", 0.11)
    produced = {"tfidf_lcc_classification": {"primary_score": 0.99}}

    merged, recovered = run_all.harvest_existing_results(
        tmp_path, MODELS, TASKS, produced
    )

    assert recovered == 0
    assert merged["tfidf_lcc_classification"]["primary_score"] == 0.99


def test_unreadable_file_is_skipped_not_fatal(tmp_path: Path) -> None:
    """A corrupt result must not take the whole aggregation down with it."""
    _write(tmp_path, "tfidf_lcc_classification", 0.5)
    (tmp_path / "tfidf_lcc_retrieval.json").write_text("{ truncated")

    merged, recovered = run_all.harvest_existing_results(tmp_path, MODELS, TASKS, {})

    assert recovered == 1
    assert "tfidf_lcc_classification" in merged
    assert "tfidf_lcc_retrieval" not in merged


def test_unrelated_files_are_ignored(tmp_path: Path) -> None:
    """Only configured model/task pairs are harvested."""
    _write(tmp_path, "tfidf_lcc_classification", 0.5)
    _write(tmp_path, "summary", 0.0)
    _write(tmp_path, "manifest", 0.0)
    _write(tmp_path, "unknown_model_lcc_classification", 0.7)

    merged, recovered = run_all.harvest_existing_results(tmp_path, MODELS, TASKS, {})

    assert recovered == 1
    assert set(merged) == {"tfidf_lcc_classification"}


def test_empty_directory_is_harmless(tmp_path: Path) -> None:
    merged, recovered = run_all.harvest_existing_results(tmp_path, MODELS, TASKS, {})
    assert recovered == 0
    assert merged == {}


@pytest.mark.parametrize("n_models", [1, 5, 25])
def test_summary_covers_directory_regardless_of_run_size(
    tmp_path: Path, n_models: int
) -> None:
    """Whatever subset runs, the merged view is the directory."""
    models = {f"m{i}": {"type": "tfidf"} for i in range(25)}
    for i in range(25):
        _write(tmp_path, f"m{i}_lcc_classification", 0.5)

    produced = {
        f"m{i}_lcc_classification": {"primary_score": 0.9} for i in range(n_models)
    }
    merged, _ = run_all.harvest_existing_results(tmp_path, models, TASKS, produced)

    assert len(merged) == 25
