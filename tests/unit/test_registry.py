"""Unit tests for shelf.evaluate.registry module.

Tests cover:
- Getting tasks by name with get_task()
- Listing all available tasks with list_tasks()
- Filtering tasks by type with list_tasks(task_type=...)
- Specialized list functions (list_retrieval_tasks, etc.)
- Error handling for unknown tasks
- Task specification validation
- Registry constants (LCC_CODES, LCGFT_CATEGORIES, etc.)
- TASK_REGISTRY structure and completeness
"""

from __future__ import annotations

from pathlib import Path

import pytest
from shelf.evaluate.registry import (
    GEOGRAPHIC_REGIONS,
    LCC_CODES,
    LCGFT_CATEGORIES,
    LCGFT_FORMS,
    REGISTERS,
    TASK_REGISTRY,
    TOPICS,
    get_task,
    list_classification_tasks,
    list_clustering_tasks,
    list_multilabel_tasks,
    list_pair_tasks,
    list_retrieval_tasks,
    list_tasks,
)
from shelf.evaluate.tasks import TaskSpec, TaskType


class TestRegistryConstants:
    """Tests for registry constants (LCC_CODES, LCGFT_CATEGORIES, etc.)."""

    @pytest.mark.unit
    def test_lcc_codes_count(self):
        """Test LCC_CODES has exactly 21 classes."""
        assert len(LCC_CODES) == 21

    @pytest.mark.unit
    def test_lcc_codes_values(self):
        """Test LCC_CODES contains expected values."""
        assert "A" in LCC_CODES
        assert "B" in LCC_CODES
        assert "Z" in LCC_CODES
        # Note: I and O are not used in LCC
        assert "I" not in LCC_CODES
        assert "O" not in LCC_CODES

    @pytest.mark.unit
    def test_lcc_codes_are_uppercase(self):
        """Test all LCC codes are uppercase single letters."""
        for code in LCC_CODES:
            assert isinstance(code, str)
            assert len(code) == 1
            assert code.isupper()

    @pytest.mark.unit
    def test_lcc_codes_unique(self):
        """Test LCC_CODES has no duplicates."""
        assert len(LCC_CODES) == len(set(LCC_CODES))

    @pytest.mark.unit
    def test_lcgft_categories_count(self):
        """Test LCGFT_CATEGORIES has exactly 14 categories."""
        assert len(LCGFT_CATEGORIES) == 14

    @pytest.mark.unit
    def test_lcgft_categories_values(self):
        """Test LCGFT_CATEGORIES contains expected values."""
        assert "Literature" in LCGFT_CATEGORIES
        assert "Music" in LCGFT_CATEGORIES
        assert "Law materials" in LCGFT_CATEGORIES
        assert "Visual works" in LCGFT_CATEGORIES

    @pytest.mark.unit
    def test_lcgft_categories_unique(self):
        """Test LCGFT_CATEGORIES has no duplicates."""
        assert len(LCGFT_CATEGORIES) == len(set(LCGFT_CATEGORIES))

    @pytest.mark.unit
    def test_registers_count(self):
        """Test REGISTERS has exactly 8 registers."""
        assert len(REGISTERS) == 8

    @pytest.mark.unit
    def test_registers_values(self):
        """Test REGISTERS contains expected values."""
        assert "academic" in REGISTERS
        assert "casual" in REGISTERS
        assert "formal" in REGISTERS
        assert "technical" in REGISTERS

    @pytest.mark.unit
    def test_registers_are_lowercase(self):
        """Test all register names are lowercase."""
        for register in REGISTERS:
            assert isinstance(register, str)
            assert register.islower()

    @pytest.mark.unit
    def test_registers_unique(self):
        """Test REGISTERS has no duplicates."""
        assert len(REGISTERS) == len(set(REGISTERS))

    @pytest.mark.unit
    def test_geographic_regions_count(self):
        """Test GEOGRAPHIC_REGIONS has exactly 8 regions."""
        assert len(GEOGRAPHIC_REGIONS) == 8

    @pytest.mark.unit
    def test_geographic_regions_values(self):
        """Test GEOGRAPHIC_REGIONS contains expected values."""
        assert "North America" in GEOGRAPHIC_REGIONS
        assert "Europe" in GEOGRAPHIC_REGIONS
        assert "East Asia" in GEOGRAPHIC_REGIONS
        assert "Sub-Saharan Africa" in GEOGRAPHIC_REGIONS

    @pytest.mark.unit
    def test_geographic_regions_unique(self):
        """Test GEOGRAPHIC_REGIONS has no duplicates."""
        assert len(GEOGRAPHIC_REGIONS) == len(set(GEOGRAPHIC_REGIONS))


class TestTaskRegistry:
    """Tests for TASK_REGISTRY structure and contents."""

    @pytest.mark.unit
    def test_registry_is_dict(self):
        """Test TASK_REGISTRY is a dictionary."""
        assert isinstance(TASK_REGISTRY, dict)

    @pytest.mark.unit
    def test_registry_not_empty(self):
        """Test TASK_REGISTRY is not empty."""
        assert len(TASK_REGISTRY) > 0

    @pytest.mark.unit
    def test_all_registry_values_are_taskspecs(self):
        """Test all registry values are TaskSpec instances."""
        for task_spec in TASK_REGISTRY.values():
            assert isinstance(task_spec, TaskSpec)

    @pytest.mark.unit
    def test_all_registry_keys_match_task_names(self):
        """Test registry keys match their TaskSpec names."""
        for key, spec in TASK_REGISTRY.items():
            assert key == spec.name

    @pytest.mark.unit
    def test_registry_has_retrieval_tasks(self):
        """Test registry contains retrieval tasks."""
        retrieval_tasks = [
            spec
            for spec in TASK_REGISTRY.values()
            if spec.task_type == TaskType.RETRIEVAL
        ]
        assert len(retrieval_tasks) > 0

    @pytest.mark.unit
    def test_registry_has_classification_tasks(self):
        """Test registry contains classification tasks."""
        classification_tasks = [
            spec
            for spec in TASK_REGISTRY.values()
            if spec.task_type == TaskType.CLASSIFICATION
        ]
        assert len(classification_tasks) > 0

    @pytest.mark.unit
    def test_registry_has_clustering_tasks(self):
        """Test registry contains clustering tasks."""
        clustering_tasks = [
            spec
            for spec in TASK_REGISTRY.values()
            if spec.task_type == TaskType.CLUSTERING
        ]
        assert len(clustering_tasks) > 0

    @pytest.mark.unit
    def test_registry_has_pair_tasks(self):
        """Test registry contains pair classification tasks."""
        pair_tasks = [
            spec
            for spec in TASK_REGISTRY.values()
            if spec.task_type == TaskType.PAIR_CLASSIFICATION
        ]
        assert len(pair_tasks) > 0

    @pytest.mark.unit
    def test_specific_tasks_exist(self):
        """Test specific expected tasks exist in registry."""
        expected_tasks = [
            "lcc_classification",
            "lcc_retrieval",
            "lcc_clustering",
            "same_lcc_pairs",
            "form_retrieval",
            "lcgft_category_classification",
        ]
        for task_name in expected_tasks:
            assert task_name in TASK_REGISTRY

    @pytest.mark.unit
    def test_all_tasks_have_valid_metrics(self):
        """Test all tasks have non-empty primary and secondary metrics."""
        for spec in TASK_REGISTRY.values():
            assert isinstance(spec.primary_metric, str)
            assert len(spec.primary_metric) > 0
            assert isinstance(spec.secondary_metrics, tuple)

    @pytest.mark.unit
    def test_all_tasks_have_dataset_info(self):
        """Test all tasks have dataset configuration."""
        for spec in TASK_REGISTRY.values():
            assert isinstance(spec.dataset_name, str)
            assert len(spec.dataset_name) > 0
            assert isinstance(spec.default_split, str)
            assert len(spec.default_split) > 0


class TestGetTask:
    """Tests for get_task() function."""

    @pytest.mark.unit
    def test_get_task_returns_taskspec(self):
        """Test get_task returns a TaskSpec instance."""
        task = get_task("lcc_classification")
        assert isinstance(task, TaskSpec)

    @pytest.mark.unit
    def test_get_task_by_name(self):
        """Test get_task retrieves correct task by name."""
        task = get_task("lcc_classification")
        assert task.name == "lcc_classification"
        assert task.task_type == TaskType.CLASSIFICATION

    @pytest.mark.unit
    def test_get_task_lcc_retrieval(self):
        """Test getting lcc_retrieval task."""
        task = get_task("lcc_retrieval")
        assert task.name == "lcc_retrieval"
        assert task.task_type == TaskType.RETRIEVAL
        assert task.label_field == "lcc_code"
        assert task.label_space == LCC_CODES
        assert task.primary_metric == "ndcg@10"

    @pytest.mark.unit
    def test_get_task_lcc_classification(self):
        """Test getting lcc_classification task."""
        task = get_task("lcc_classification")
        assert task.name == "lcc_classification"
        assert task.task_type == TaskType.CLASSIFICATION
        assert task.label_field == "lcc_code"
        assert task.label_space == LCC_CODES
        assert task.primary_metric == "macro_f1"

    @pytest.mark.unit
    def test_get_task_form_retrieval(self):
        """Test getting form_retrieval task."""
        task = get_task("form_retrieval")
        assert task.name == "form_retrieval"
        assert task.task_type == TaskType.RETRIEVAL
        assert task.label_field == "lcgft_form"
        assert task.label_space is None  # Open vocabulary
        assert task.primary_metric == "ndcg@10"

    @pytest.mark.unit
    def test_get_task_lcc_clustering(self):
        """Test getting lcc_clustering task."""
        task = get_task("lcc_clustering")
        assert task.name == "lcc_clustering"
        assert task.task_type == TaskType.CLUSTERING
        assert task.label_field == "lcc_code"
        assert task.label_space == LCC_CODES
        assert task.primary_metric == "v_measure"

    @pytest.mark.unit
    def test_get_task_same_lcc_pairs(self):
        """Test getting same_lcc_pairs task."""
        task = get_task("same_lcc_pairs")
        assert task.name == "same_lcc_pairs"
        assert task.task_type == TaskType.PAIR_CLASSIFICATION
        assert task.label_field == "label"
        assert task.label_space == ("0", "1")
        assert task.primary_metric == "auc_roc"
        assert task.dataset_config == "same_lcc_pairs"

    @pytest.mark.unit
    def test_get_task_register_classification(self):
        """Test getting register_classification task."""
        task = get_task("register_classification")
        assert task.name == "register_classification"
        assert task.task_type == TaskType.CLASSIFICATION
        assert task.label_field == "register"
        assert task.label_space == REGISTERS
        assert task.primary_metric == "macro_f1"

    @pytest.mark.unit
    def test_get_task_geographic_clustering(self):
        """Test getting geographic_clustering task."""
        task = get_task("geographic_clustering")
        assert task.name == "geographic_clustering"
        assert task.task_type == TaskType.CLUSTERING
        assert task.label_field == "geographic_region"
        assert task.label_space == GEOGRAPHIC_REGIONS
        assert task.primary_metric == "v_measure"

    @pytest.mark.unit
    def test_get_task_unknown_raises_error(self):
        """Test get_task raises ValueError for unknown task."""
        with pytest.raises(ValueError, match="Unknown task"):
            get_task("nonexistent_task")

    @pytest.mark.unit
    def test_get_task_error_includes_available_tasks(self):
        """Test error message includes list of available tasks."""
        with pytest.raises(ValueError, match="Available tasks:"):
            get_task("invalid_task")

    @pytest.mark.unit
    def test_get_task_error_message_format(self):
        """Test error message format for unknown task."""
        with pytest.raises(ValueError) as exc_info:
            get_task("bad_task")
        error_msg = str(exc_info.value)
        assert "bad_task" in error_msg
        assert "Available tasks:" in error_msg

    @pytest.mark.unit
    def test_get_task_empty_string(self):
        """Test get_task with empty string raises ValueError."""
        with pytest.raises(ValueError):
            get_task("")

    @pytest.mark.unit
    def test_get_task_case_sensitive(self):
        """Test get_task is case-sensitive."""
        with pytest.raises(ValueError):
            get_task("LCC_CLASSIFICATION")  # Wrong case

    @pytest.mark.unit
    def test_get_task_returns_same_instance(self):
        """Test get_task returns the same TaskSpec instance."""
        task1 = get_task("lcc_classification")
        task2 = get_task("lcc_classification")
        # TaskSpec is frozen (immutable), so instances should be equal
        assert task1 == task2
        assert task1 is task2  # Should be same object from registry


class TestListTasks:
    """Tests for list_tasks() function."""

    @pytest.mark.unit
    def test_list_tasks_returns_list(self):
        """Test list_tasks returns a list."""
        tasks = list_tasks()
        assert isinstance(tasks, list)

    @pytest.mark.unit
    def test_list_tasks_not_empty(self):
        """Test list_tasks returns non-empty list."""
        tasks = list_tasks()
        assert len(tasks) > 0

    @pytest.mark.unit
    def test_list_tasks_all_strings(self):
        """Test list_tasks returns list of strings."""
        tasks = list_tasks()
        assert all(isinstance(task, str) for task in tasks)

    @pytest.mark.unit
    def test_list_tasks_sorted(self):
        """Test list_tasks returns sorted list."""
        tasks = list_tasks()
        assert tasks == sorted(tasks)

    @pytest.mark.unit
    def test_list_tasks_no_duplicates(self):
        """Test list_tasks has no duplicates."""
        tasks = list_tasks()
        assert len(tasks) == len(set(tasks))

    @pytest.mark.unit
    def test_list_tasks_matches_registry(self):
        """Test list_tasks returns all registry keys."""
        tasks = list_tasks()
        registry_keys = sorted(TASK_REGISTRY.keys())
        assert tasks == registry_keys

    @pytest.mark.unit
    def test_list_tasks_contains_known_tasks(self):
        """Test list_tasks contains known task names."""
        tasks = list_tasks()
        assert "lcc_classification" in tasks
        assert "lcc_retrieval" in tasks
        assert "lcc_clustering" in tasks
        assert "same_lcc_pairs" in tasks


class TestListTasksByType:
    """Tests for list_tasks(task_type=...) filtering."""

    @pytest.mark.unit
    def test_list_tasks_filter_retrieval(self):
        """Test list_tasks filters by RETRIEVAL type."""
        tasks = list_tasks(TaskType.RETRIEVAL)
        assert isinstance(tasks, list)
        assert len(tasks) > 0
        # Verify all are retrieval tasks
        for task_name in tasks:
            task_spec = get_task(task_name)
            assert task_spec.task_type == TaskType.RETRIEVAL

    @pytest.mark.unit
    def test_list_tasks_filter_classification(self):
        """Test list_tasks filters by CLASSIFICATION type."""
        tasks = list_tasks(TaskType.CLASSIFICATION)
        assert len(tasks) > 0
        for task_name in tasks:
            task_spec = get_task(task_name)
            assert task_spec.task_type == TaskType.CLASSIFICATION

    @pytest.mark.unit
    def test_list_tasks_filter_clustering(self):
        """Test list_tasks filters by CLUSTERING type."""
        tasks = list_tasks(TaskType.CLUSTERING)
        assert len(tasks) > 0
        for task_name in tasks:
            task_spec = get_task(task_name)
            assert task_spec.task_type == TaskType.CLUSTERING

    @pytest.mark.unit
    def test_list_tasks_filter_pair_classification(self):
        """Test list_tasks filters by PAIR_CLASSIFICATION type."""
        tasks = list_tasks(TaskType.PAIR_CLASSIFICATION)
        assert len(tasks) > 0
        for task_name in tasks:
            task_spec = get_task(task_name)
            assert task_spec.task_type == TaskType.PAIR_CLASSIFICATION

    @pytest.mark.unit
    def test_list_tasks_filter_sorted(self):
        """Test filtered list_tasks returns sorted results."""
        tasks = list_tasks(TaskType.CLASSIFICATION)
        assert tasks == sorted(tasks)

    @pytest.mark.unit
    def test_list_tasks_filter_no_duplicates(self):
        """Test filtered list_tasks has no duplicates."""
        tasks = list_tasks(TaskType.RETRIEVAL)
        assert len(tasks) == len(set(tasks))

    @pytest.mark.unit
    def test_list_tasks_filter_none_returns_all(self):
        """Test list_tasks(None) returns all tasks."""
        all_tasks = list_tasks()
        none_tasks = list_tasks(None)
        assert all_tasks == none_tasks

    @pytest.mark.unit
    def test_list_tasks_all_types_cover_registry(self):
        """Test filtering by all types covers all registry tasks."""
        all_tasks = set(list_tasks())
        retrieval = set(list_tasks(TaskType.RETRIEVAL))
        classification = set(list_tasks(TaskType.CLASSIFICATION))
        clustering = set(list_tasks(TaskType.CLUSTERING))
        pair = set(list_tasks(TaskType.PAIR_CLASSIFICATION))
        multilabel = set(list_tasks(TaskType.MULTILABEL))

        combined = retrieval | classification | clustering | pair | multilabel
        assert combined == all_tasks


class TestSpecializedListFunctions:
    """Tests for specialized task listing functions."""

    @pytest.mark.unit
    def test_list_retrieval_tasks_returns_list(self):
        """Test list_retrieval_tasks returns a list."""
        tasks = list_retrieval_tasks()
        assert isinstance(tasks, list)

    @pytest.mark.unit
    def test_list_retrieval_tasks_matches_filter(self):
        """Test list_retrieval_tasks matches list_tasks(RETRIEVAL)."""
        tasks1 = list_retrieval_tasks()
        tasks2 = list_tasks(TaskType.RETRIEVAL)
        assert tasks1 == tasks2

    @pytest.mark.unit
    def test_list_retrieval_tasks_contains_expected(self):
        """Test list_retrieval_tasks contains expected tasks."""
        tasks = list_retrieval_tasks()
        assert "lcc_retrieval" in tasks
        assert "form_retrieval" in tasks
        assert "category_retrieval" in tasks

    @pytest.mark.unit
    def test_list_classification_tasks_returns_list(self):
        """Test list_classification_tasks returns a list."""
        tasks = list_classification_tasks()
        assert isinstance(tasks, list)

    @pytest.mark.unit
    def test_list_classification_tasks_matches_filter(self):
        """Test list_classification_tasks matches list_tasks(CLASSIFICATION)."""
        tasks1 = list_classification_tasks()
        tasks2 = list_tasks(TaskType.CLASSIFICATION)
        assert tasks1 == tasks2

    @pytest.mark.unit
    def test_list_classification_tasks_contains_expected(self):
        """Test list_classification_tasks contains expected tasks."""
        tasks = list_classification_tasks()
        assert "lcc_classification" in tasks
        assert "lcgft_category_classification" in tasks
        assert "register_classification" in tasks

    @pytest.mark.unit
    def test_list_clustering_tasks_returns_list(self):
        """Test list_clustering_tasks returns a list."""
        tasks = list_clustering_tasks()
        assert isinstance(tasks, list)

    @pytest.mark.unit
    def test_list_clustering_tasks_matches_filter(self):
        """Test list_clustering_tasks matches list_tasks(CLUSTERING)."""
        tasks1 = list_clustering_tasks()
        tasks2 = list_tasks(TaskType.CLUSTERING)
        assert tasks1 == tasks2

    @pytest.mark.unit
    def test_list_clustering_tasks_contains_expected(self):
        """Test list_clustering_tasks contains expected tasks."""
        tasks = list_clustering_tasks()
        assert "lcc_clustering" in tasks
        assert "lcgft_clustering" in tasks
        assert "register_clustering" in tasks
        assert "geographic_clustering" in tasks

    @pytest.mark.unit
    def test_list_pair_tasks_returns_list(self):
        """Test list_pair_tasks returns a list."""
        tasks = list_pair_tasks()
        assert isinstance(tasks, list)

    @pytest.mark.unit
    def test_list_pair_tasks_matches_filter(self):
        """Test list_pair_tasks matches list_tasks(PAIR_CLASSIFICATION)."""
        tasks1 = list_pair_tasks()
        tasks2 = list_tasks(TaskType.PAIR_CLASSIFICATION)
        assert tasks1 == tasks2

    @pytest.mark.unit
    def test_list_pair_tasks_contains_expected(self):
        """Test list_pair_tasks contains expected tasks."""
        tasks = list_pair_tasks()
        assert "same_lcc_pairs" in tasks
        assert "same_form_pairs" in tasks
        assert "same_register_pairs" in tasks
        assert "same_audience_pairs" in tasks
        assert "same_topic_pairs" in tasks
        assert "topic_overlap_pairs" in tasks

    @pytest.mark.unit
    def test_all_specialized_functions_are_disjoint(self):
        """Test specialized list functions return disjoint sets (except multilabel)."""
        retrieval = set(list_retrieval_tasks())
        classification = set(list_classification_tasks())
        clustering = set(list_clustering_tasks())
        pair = set(list_pair_tasks())

        # These sets should be disjoint
        assert retrieval.isdisjoint(classification)
        assert retrieval.isdisjoint(clustering)
        assert retrieval.isdisjoint(pair)
        assert classification.isdisjoint(clustering)
        assert classification.isdisjoint(pair)
        assert clustering.isdisjoint(pair)


class TestTaskSpecificationValidation:
    """Tests for validating task specifications in registry."""

    @pytest.mark.unit
    def test_all_retrieval_tasks_have_correct_metrics(self):
        """Test all retrieval tasks have appropriate metrics."""
        for task_name in list_retrieval_tasks():
            task = get_task(task_name)
            # Retrieval tasks should have ndcg or mrr as primary metric
            assert task.primary_metric in ("ndcg@10", "mrr", "map@10")
            # Should have retrieval-specific secondary metrics
            assert isinstance(task.secondary_metrics, tuple)

    @pytest.mark.unit
    def test_all_classification_tasks_have_correct_metrics(self):
        """Test all classification tasks have appropriate metrics."""
        for task_name in list_classification_tasks():
            task = get_task(task_name)
            # Classification tasks should have F1 or accuracy as primary
            assert task.primary_metric in ("macro_f1", "accuracy", "f1", "auc_roc")
            # Should have classification-specific secondary metrics
            assert isinstance(task.secondary_metrics, tuple)

    @pytest.mark.unit
    def test_all_clustering_tasks_have_correct_metrics(self):
        """Test all clustering tasks have appropriate metrics."""
        for task_name in list_clustering_tasks():
            task = get_task(task_name)
            # Clustering tasks should have v_measure, nmi, or ari as primary.
            # Subject-conditional tasks additionally distinguish macro from
            # pooled, because they are different quantities: macro averages
            # over classes and flatters small ones, pooled does not. They use
            # ARI rather than V-measure because V-measure is not
            # chance-corrected and inflates under conditioning (a shuffled-label
            # control scores V-measure 0.0869 macro but ARI -0.0015).
            assert task.primary_metric in (
                "v_measure",
                "nmi",
                "ari",
                "ari_pooled",
                "ari_macro",
                "v_measure_pooled",
                "v_measure_macro",
            )
            assert isinstance(task.secondary_metrics, tuple)

    @pytest.mark.unit
    def test_all_pair_tasks_have_binary_labels(self):
        """Test all pair classification tasks have binary label space."""
        for task_name in list_pair_tasks():
            task = get_task(task_name)
            # Most pair tasks should be binary (except topic_overlap_pairs)
            if task_name == "topic_overlap_pairs":
                assert task.label_space == ("0", "1", "2", "3")  # 4-class
            else:
                assert task.label_space == ("0", "1")  # Binary

    @pytest.mark.unit
    def test_all_tasks_have_text_field(self):
        """Test all tasks specify a text field."""
        for task_name in list_tasks():
            task = get_task(task_name)
            assert isinstance(task.text_field, str)
            assert len(task.text_field) > 0
            assert task.text_field == "text"  # Should all use "text"

    @pytest.mark.unit
    def test_all_tasks_have_id_field(self):
        """Test all tasks specify an ID field."""
        for task_name in list_tasks():
            task = get_task(task_name)
            assert isinstance(task.id_field, str)
            assert len(task.id_field) > 0

    @pytest.mark.unit
    def test_all_tasks_have_label_field(self):
        """Test all tasks specify a label field."""
        for task_name in list_tasks():
            task = get_task(task_name)
            assert isinstance(task.label_field, str)
            assert len(task.label_field) > 0

    @pytest.mark.unit
    def test_all_tasks_have_description(self):
        """Test all tasks have non-empty descriptions."""
        for task_name in list_tasks():
            task = get_task(task_name)
            assert isinstance(task.description, str)
            assert len(task.description) > 10  # Should be meaningful

    @pytest.mark.unit
    def test_all_tasks_use_shelf_dataset(self):
        """Test all tasks use the SHELF dataset."""
        for task_name in list_tasks():
            task = get_task(task_name)
            assert task.dataset_name == "mjbommar/SHELF"

    @pytest.mark.unit
    def test_all_tasks_have_default_split(self):
        """Test all tasks specify a default split."""
        for task_name in list_tasks():
            task = get_task(task_name)
            assert task.default_split in ("train", "validation", "test")

    @pytest.mark.unit
    def test_pair_tasks_have_correct_config(self):
        """Test pair tasks have appropriate dataset configs."""
        pair_tasks = list_pair_tasks()
        for task_name in pair_tasks:
            task = get_task(task_name)
            # Pair tasks should have matching config names
            if task_name.startswith("same_"):
                assert task.dataset_config == task_name
            elif task_name == "topic_overlap_pairs":
                assert task.dataset_config == "topic_overlap_pairs"

    @pytest.mark.unit
    def test_lcc_tasks_use_lcc_codes(self):
        """Test LCC-related tasks use LCC_CODES label space."""
        lcc_tasks = ["lcc_classification", "lcc_retrieval", "lcc_clustering"]
        for task_name in lcc_tasks:
            task = get_task(task_name)
            assert task.label_space == LCC_CODES

    @pytest.mark.unit
    def test_lcgft_category_tasks_use_lcgft_categories(self):
        """Test LCGFT category tasks use LCGFT_CATEGORIES label space."""
        category_tasks = [
            "lcgft_category_classification",
            "category_retrieval",
            "lcgft_clustering",
        ]
        for task_name in category_tasks:
            task = get_task(task_name)
            assert task.label_space == LCGFT_CATEGORIES

    @pytest.mark.unit
    def test_register_tasks_use_registers(self):
        """Test register tasks use REGISTERS label space."""
        register_tasks = ["register_classification", "register_clustering"]
        for task_name in register_tasks:
            task = get_task(task_name)
            assert task.label_space == REGISTERS

    @pytest.mark.unit
    def test_geographic_clustering_uses_regions(self):
        """Test geographic_clustering uses GEOGRAPHIC_REGIONS label space."""
        task = get_task("geographic_clustering")
        assert task.label_space == GEOGRAPHIC_REGIONS

    @pytest.mark.unit
    def test_form_retrieval_has_open_vocabulary(self):
        """Test form_retrieval has None label_space (open vocabulary)."""
        task = get_task("form_retrieval")
        assert task.label_space is None
        assert task.num_classes is None


class TestTaskSpecImmutability:
    """Tests that TaskSpecs from registry are immutable."""

    @pytest.mark.unit
    def test_taskspec_is_frozen(self):
        """Test TaskSpec from registry is frozen (immutable)."""
        task = get_task("lcc_classification")
        with pytest.raises(AttributeError):
            task.name = "modified_name"

    @pytest.mark.unit
    def test_cannot_modify_task_type(self):
        """Test cannot modify task_type after creation."""
        task = get_task("lcc_retrieval")
        with pytest.raises(AttributeError):
            task.task_type = TaskType.CLASSIFICATION

    @pytest.mark.unit
    def test_cannot_modify_label_space(self):
        """Test cannot modify label_space after creation."""
        task = get_task("lcc_classification")
        with pytest.raises(AttributeError):
            task.label_space = ("X", "Y", "Z")


class TestFormClassificationTask:
    """Tests for the 133-way form_classification task."""

    @pytest.mark.unit
    def test_lcgft_forms_count(self):
        """Test LCGFT_FORMS has exactly 133 genre/form terms."""
        assert len(LCGFT_FORMS) == 133

    @pytest.mark.unit
    def test_lcgft_forms_unique_and_sorted(self):
        """Test LCGFT_FORMS has no duplicates and is sorted."""
        assert len(set(LCGFT_FORMS)) == len(LCGFT_FORMS)
        assert list(LCGFT_FORMS) == sorted(LCGFT_FORMS)

    @pytest.mark.unit
    def test_lcgft_forms_sample_values(self):
        """Test LCGFT_FORMS contains expected terms."""
        for form in ("Lectures", "Maps", "Prayers", "Jokes", "Biographies"):
            assert form in LCGFT_FORMS

    @pytest.mark.unit
    def test_task_registered(self):
        """Test form_classification is in the registry."""
        assert "form_classification" in TASK_REGISTRY
        task = get_task("form_classification")
        assert isinstance(task, TaskSpec)
        assert task.name == "form_classification"

    @pytest.mark.unit
    def test_task_type_is_classification(self):
        """Test form_classification is a single-label classification task."""
        task = get_task("form_classification")
        assert task.task_type == TaskType.CLASSIFICATION
        assert "form_classification" in list_classification_tasks()

    @pytest.mark.unit
    def test_task_fields(self):
        """Test form_classification reads lcgft_form from the default config."""
        task = get_task("form_classification")
        assert task.text_field == "text"
        assert task.label_field == "lcgft_form"
        assert task.id_field == "id"
        assert task.dataset_name == "mjbommar/SHELF"
        assert task.dataset_config == "default"
        assert task.default_split == "test"

    @pytest.mark.unit
    def test_task_label_space_is_explicit(self):
        """Test form_classification declares the full 133-term label space.

        An explicit space keeps the macro-F1 denominator and the confusion
        matrix ordering stable even on a filtered subset. This is the one
        difference from form_retrieval, which uses an open vocabulary.
        """
        task = get_task("form_classification")
        assert task.label_space == LCGFT_FORMS
        assert task.num_classes == 133
        assert get_task("form_retrieval").label_space is None

    @pytest.mark.unit
    def test_task_metrics(self):
        """Test form_classification reports the full classification metric set."""
        task = get_task("form_classification")
        assert task.primary_metric == "macro_f1"
        assert set(task.secondary_metrics) == {
            "micro_f1",
            "accuracy",
            "weighted_f1",
        }


class TestTopicClassificationTask:
    """Tests for the multi-label topic_classification task."""

    @pytest.mark.unit
    def test_topics_count(self):
        """Test TOPICS has exactly 112 topical terms."""
        assert len(TOPICS) == 112

    @pytest.mark.unit
    def test_topics_unique_and_sorted(self):
        """Test TOPICS has no duplicates and is sorted."""
        assert len(set(TOPICS)) == len(TOPICS)
        assert list(TOPICS) == sorted(TOPICS)

    @pytest.mark.unit
    def test_topics_sample_values(self):
        """Test TOPICS contains expected terms."""
        for topic in ("Art", "Ethics", "Democracy", "Globalization"):
            assert topic in TOPICS

    @pytest.mark.unit
    def test_task_registered(self):
        """Test topic_classification is in the registry."""
        assert "topic_classification" in TASK_REGISTRY
        task = get_task("topic_classification")
        assert isinstance(task, TaskSpec)
        assert task.name == "topic_classification"

    @pytest.mark.unit
    def test_task_type_is_multilabel(self):
        """Test topic_classification is SHELF's first multi-label task."""
        task = get_task("topic_classification")
        assert task.task_type == TaskType.MULTILABEL
        assert list_multilabel_tasks() == ["topic_classification"]

    @pytest.mark.unit
    def test_multilabel_not_listed_as_classification(self):
        """Test the multi-label task is not mixed into classification tasks."""
        assert "topic_classification" not in list_classification_tasks()
        assert "topic_classification" not in list_clustering_tasks()
        assert "topic_classification" not in list_pair_tasks()
        assert "topic_classification" not in list_retrieval_tasks()

    @pytest.mark.unit
    def test_task_fields(self):
        """Test topic_classification reads the list-valued topics column."""
        task = get_task("topic_classification")
        assert task.text_field == "text"
        assert task.label_field == "topics"
        assert task.id_field == "id"
        assert task.dataset_name == "mjbommar/SHELF"
        assert task.dataset_config == "default"
        assert task.default_split == "test"

    @pytest.mark.unit
    def test_task_label_space(self):
        """Test topic_classification declares the 112-term label space."""
        task = get_task("topic_classification")
        assert task.label_space == TOPICS
        assert task.num_classes == 112

    @pytest.mark.unit
    def test_task_metrics(self):
        """Test topic_classification reports the full multi-label metric set."""
        task = get_task("topic_classification")
        assert task.primary_metric == "macro_f1"
        # All four averagings plus set-level and ranking metrics, so no
        # single number can be cherry-picked.
        for metric in (
            "micro_f1",
            "samples_f1",
            "weighted_f1",
            "subset_accuracy",
            "hamming_loss",
            "lrap",
            "map_micro",
        ):
            assert metric in task.secondary_metrics


class TestNewTasksInBaselineConfig:
    """Tests that the new tasks are wired into the baseline runner config."""

    @pytest.mark.unit
    def test_tasks_present_in_config(self):
        """Test both new tasks appear under the config's tasks section."""
        import yaml

        config_path = (
            Path(__file__).resolve().parents[2]
            / "scripts"
            / "baselines"
            / "config.yaml"
        )
        config = yaml.safe_load(config_path.read_text())
        tasks = config["tasks"]

        assert "form_classification" in tasks["classification"]
        assert "topic_classification" in tasks["multilabel"]

    @pytest.mark.unit
    def test_all_configured_tasks_exist_in_registry(self):
        """Test every task named in the config resolves to a TaskSpec."""
        import yaml

        config_path = (
            Path(__file__).resolve().parents[2]
            / "scripts"
            / "baselines"
            / "config.yaml"
        )
        config = yaml.safe_load(config_path.read_text())

        for task_type, task_names in config["tasks"].items():
            for task_name in task_names:
                task = get_task(task_name)
                assert task.task_type.value == task_type
