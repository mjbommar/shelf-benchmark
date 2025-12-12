"""Unit tests for shelf.evaluate.tasks module.

Tests cover:
- TaskType enum values and behavior
- TaskSpec creation with all fields
- TaskSpec immutability (frozen=True)
- num_classes property (with and without label_space)
- __str__ and __repr__ methods
- TaskSpec equality
- TaskSpec with None label_space (open vocabulary)
- Edge cases and validation
"""

from __future__ import annotations

import pytest

from shelf.evaluate.tasks import TaskSpec, TaskType


class TestTaskType:
    """Tests for TaskType enum."""

    @pytest.mark.unit
    def test_all_task_types_exist(self):
        """Test all expected task types are defined."""
        assert hasattr(TaskType, "CLASSIFICATION")
        assert hasattr(TaskType, "MULTILABEL")
        assert hasattr(TaskType, "RETRIEVAL")
        assert hasattr(TaskType, "CLUSTERING")
        assert hasattr(TaskType, "PAIR_CLASSIFICATION")

    @pytest.mark.unit
    def test_task_type_values(self):
        """Test task type enum values are correct."""
        assert TaskType.CLASSIFICATION.value == "classification"
        assert TaskType.MULTILABEL.value == "multilabel"
        assert TaskType.RETRIEVAL.value == "retrieval"
        assert TaskType.CLUSTERING.value == "clustering"
        assert TaskType.PAIR_CLASSIFICATION.value == "pair_classification"

    @pytest.mark.unit
    def test_task_type_equality(self):
        """Test task type enum equality."""
        assert TaskType.CLASSIFICATION == TaskType.CLASSIFICATION
        assert TaskType.CLASSIFICATION != TaskType.RETRIEVAL

    @pytest.mark.unit
    def test_task_type_from_value(self):
        """Test creating TaskType from string value."""
        assert TaskType("classification") == TaskType.CLASSIFICATION
        assert TaskType("retrieval") == TaskType.RETRIEVAL

    @pytest.mark.unit
    def test_task_type_invalid_value(self):
        """Test invalid task type value raises error."""
        with pytest.raises(ValueError):
            TaskType("invalid_task_type")

    @pytest.mark.unit
    def test_task_type_is_hashable(self):
        """Test TaskType can be used as dict key."""
        task_map = {
            TaskType.CLASSIFICATION: "clf",
            TaskType.RETRIEVAL: "ret",
        }
        assert task_map[TaskType.CLASSIFICATION] == "clf"
        assert task_map[TaskType.RETRIEVAL] == "ret"


class TestTaskSpecCreation:
    """Tests for TaskSpec creation and initialization."""

    @pytest.mark.unit
    def test_create_with_all_fields(self, sample_task_spec):
        """Test creating TaskSpec with all required fields."""
        spec = TaskSpec(
            name=sample_task_spec["name"],
            task_type=TaskType.CLASSIFICATION,
            description=sample_task_spec["description"],
            text_field=sample_task_spec["text_field"],
            label_field=sample_task_spec["label_field"],
            id_field=sample_task_spec["id_field"],
            label_space=tuple(sample_task_spec["label_space"]),
            primary_metric="macro_f1",
            secondary_metrics=tuple(sample_task_spec["metrics"][1:]),
            dataset_name="mjbommar/SHELF",
            dataset_config="default",
            default_split="test",
        )

        assert spec.name == "test_classification"
        assert spec.task_type == TaskType.CLASSIFICATION
        assert spec.description == "Test classification task"
        assert spec.text_field == "body"
        assert spec.label_field == "lcc"
        assert spec.id_field == "id"
        assert spec.label_space == ("A", "B", "C", "D")
        assert spec.primary_metric == "macro_f1"
        assert spec.secondary_metrics == ("micro_f1", "accuracy")
        assert spec.dataset_name == "mjbommar/SHELF"
        assert spec.dataset_config == "default"
        assert spec.default_split == "test"

    @pytest.mark.unit
    def test_create_with_none_label_space(self):
        """Test creating TaskSpec with open vocabulary (None label_space)."""
        spec = TaskSpec(
            name="open_vocab_task",
            task_type=TaskType.MULTILABEL,
            description="Task with open vocabulary",
            text_field="text",
            label_field="topics",
            id_field="id",
            label_space=None,  # Open vocabulary
            primary_metric="macro_f1",
            secondary_metrics=("micro_f1",),
            dataset_name="test_dataset",
            dataset_config=None,
            default_split="test",
        )

        assert spec.label_space is None
        assert spec.num_classes is None

    @pytest.mark.unit
    def test_create_with_none_dataset_config(self):
        """Test creating TaskSpec with None dataset_config."""
        spec = TaskSpec(
            name="simple_task",
            task_type=TaskType.CLASSIFICATION,
            description="Simple task",
            text_field="text",
            label_field="label",
            id_field="id",
            label_space=("A", "B"),
            primary_metric="accuracy",
            secondary_metrics=(),
            dataset_name="test_dataset",
            dataset_config=None,  # No config needed
            default_split="test",
        )

        assert spec.dataset_config is None

    @pytest.mark.unit
    def test_create_with_empty_secondary_metrics(self):
        """Test creating TaskSpec with no secondary metrics."""
        spec = TaskSpec(
            name="minimal_metrics",
            task_type=TaskType.CLASSIFICATION,
            description="Task with only primary metric",
            text_field="text",
            label_field="label",
            id_field="id",
            label_space=("A", "B"),
            primary_metric="accuracy",
            secondary_metrics=(),  # Empty tuple
            dataset_name="test_dataset",
            dataset_config="default",
            default_split="test",
        )

        assert spec.secondary_metrics == ()
        assert len(spec.secondary_metrics) == 0

    @pytest.mark.unit
    def test_create_retrieval_task(self):
        """Test creating retrieval task spec."""
        spec = TaskSpec(
            name="lcc_retrieval",
            task_type=TaskType.RETRIEVAL,
            description="Retrieve documents by LCC class",
            text_field="body",
            label_field="lcc",
            id_field="id",
            label_space=tuple("ABCDEFGHJKLMNPQRSTUVZ"),
            primary_metric="ndcg@10",
            secondary_metrics=("map@10", "recall@10"),
            dataset_name="mjbommar/SHELF",
            dataset_config="default",
            default_split="test",
        )

        assert spec.task_type == TaskType.RETRIEVAL
        assert spec.primary_metric == "ndcg@10"
        assert "map@10" in spec.secondary_metrics

    @pytest.mark.unit
    def test_create_pair_classification_task(self):
        """Test creating pair classification task spec."""
        spec = TaskSpec(
            name="same_lcc_pairs",
            task_type=TaskType.PAIR_CLASSIFICATION,
            description="Classify document pairs by same LCC",
            text_field="text_pair",
            label_field="same_lcc",
            id_field="pair_id",
            label_space=("0", "1"),  # Binary classification
            primary_metric="accuracy",
            secondary_metrics=("f1", "precision", "recall"),
            dataset_name="mjbommar/SHELF",
            dataset_config="same_lcc_pairs",
            default_split="test",
        )

        assert spec.task_type == TaskType.PAIR_CLASSIFICATION
        assert spec.dataset_config == "same_lcc_pairs"
        assert spec.num_classes == 2


class TestTaskSpecImmutability:
    """Tests for TaskSpec immutability (frozen=True)."""

    @pytest.mark.unit
    def test_cannot_modify_name(self):
        """Test that name field cannot be modified."""
        spec = TaskSpec(
            name="test",
            task_type=TaskType.CLASSIFICATION,
            description="Test",
            text_field="text",
            label_field="label",
            id_field="id",
            label_space=("A", "B"),
            primary_metric="accuracy",
            secondary_metrics=(),
            dataset_name="test",
            dataset_config=None,
            default_split="test",
        )

        with pytest.raises(AttributeError):
            spec.name = "modified"

    @pytest.mark.unit
    def test_cannot_modify_task_type(self):
        """Test that task_type field cannot be modified."""
        spec = TaskSpec(
            name="test",
            task_type=TaskType.CLASSIFICATION,
            description="Test",
            text_field="text",
            label_field="label",
            id_field="id",
            label_space=("A", "B"),
            primary_metric="accuracy",
            secondary_metrics=(),
            dataset_name="test",
            dataset_config=None,
            default_split="test",
        )

        with pytest.raises(AttributeError):
            spec.task_type = TaskType.RETRIEVAL

    @pytest.mark.unit
    def test_cannot_modify_label_space(self):
        """Test that label_space field cannot be modified."""
        spec = TaskSpec(
            name="test",
            task_type=TaskType.CLASSIFICATION,
            description="Test",
            text_field="text",
            label_field="label",
            id_field="id",
            label_space=("A", "B"),
            primary_metric="accuracy",
            secondary_metrics=(),
            dataset_name="test",
            dataset_config=None,
            default_split="test",
        )

        with pytest.raises(AttributeError):
            spec.label_space = ("C", "D")

    @pytest.mark.unit
    def test_cannot_modify_metrics(self):
        """Test that metric fields cannot be modified."""
        spec = TaskSpec(
            name="test",
            task_type=TaskType.CLASSIFICATION,
            description="Test",
            text_field="text",
            label_field="label",
            id_field="id",
            label_space=("A", "B"),
            primary_metric="accuracy",
            secondary_metrics=("f1",),
            dataset_name="test",
            dataset_config=None,
            default_split="test",
        )

        with pytest.raises(AttributeError):
            spec.primary_metric = "f1"

        with pytest.raises(AttributeError):
            spec.secondary_metrics = ("precision",)

    @pytest.mark.unit
    def test_cannot_add_new_attributes(self):
        """Test that new attributes cannot be added."""
        spec = TaskSpec(
            name="test",
            task_type=TaskType.CLASSIFICATION,
            description="Test",
            text_field="text",
            label_field="label",
            id_field="id",
            label_space=("A", "B"),
            primary_metric="accuracy",
            secondary_metrics=(),
            dataset_name="test",
            dataset_config=None,
            default_split="test",
        )

        with pytest.raises(AttributeError):
            spec.new_field = "value"


class TestTaskSpecNumClasses:
    """Tests for num_classes property."""

    @pytest.mark.unit
    def test_num_classes_with_label_space(self):
        """Test num_classes returns correct count when label_space is defined."""
        spec = TaskSpec(
            name="test",
            task_type=TaskType.CLASSIFICATION,
            description="Test",
            text_field="text",
            label_field="label",
            id_field="id",
            label_space=("A", "B", "C", "D"),
            primary_metric="accuracy",
            secondary_metrics=(),
            dataset_name="test",
            dataset_config=None,
            default_split="test",
        )

        assert spec.num_classes == 4

    @pytest.mark.unit
    def test_num_classes_with_lcc_labels(self):
        """Test num_classes with full LCC label space (21 classes)."""
        lcc_labels = tuple("ABCDEFGHJKLMNPQRSTUVZ")
        spec = TaskSpec(
            name="lcc_classification",
            task_type=TaskType.CLASSIFICATION,
            description="LCC classification",
            text_field="body",
            label_field="lcc",
            id_field="id",
            label_space=lcc_labels,
            primary_metric="macro_f1",
            secondary_metrics=("micro_f1",),
            dataset_name="mjbommar/SHELF",
            dataset_config="default",
            default_split="test",
        )

        assert spec.num_classes == 21

    @pytest.mark.unit
    def test_num_classes_binary(self):
        """Test num_classes for binary classification."""
        spec = TaskSpec(
            name="binary_task",
            task_type=TaskType.PAIR_CLASSIFICATION,
            description="Binary task",
            text_field="text",
            label_field="label",
            id_field="id",
            label_space=("0", "1"),
            primary_metric="accuracy",
            secondary_metrics=(),
            dataset_name="test",
            dataset_config=None,
            default_split="test",
        )

        assert spec.num_classes == 2

    @pytest.mark.unit
    def test_num_classes_with_none_label_space(self):
        """Test num_classes returns None when label_space is None."""
        spec = TaskSpec(
            name="open_vocab",
            task_type=TaskType.MULTILABEL,
            description="Open vocabulary task",
            text_field="text",
            label_field="topics",
            id_field="id",
            label_space=None,  # Open vocabulary
            primary_metric="macro_f1",
            secondary_metrics=(),
            dataset_name="test",
            dataset_config=None,
            default_split="test",
        )

        assert spec.num_classes is None

    @pytest.mark.unit
    def test_num_classes_with_empty_label_space(self):
        """Test num_classes with empty label space returns None (falsy empty tuple)."""
        spec = TaskSpec(
            name="empty_labels",
            task_type=TaskType.CLASSIFICATION,
            description="Task with empty label space",
            text_field="text",
            label_field="label",
            id_field="id",
            label_space=(),  # Empty tuple is falsy, so num_classes returns None
            primary_metric="accuracy",
            secondary_metrics=(),
            dataset_name="test",
            dataset_config=None,
            default_split="test",
        )

        # Empty tuple is falsy in Python, so num_classes returns None
        assert spec.num_classes is None

    @pytest.mark.unit
    def test_num_classes_single_class(self):
        """Test num_classes with single class (edge case)."""
        spec = TaskSpec(
            name="single_class",
            task_type=TaskType.CLASSIFICATION,
            description="Single class task",
            text_field="text",
            label_field="label",
            id_field="id",
            label_space=("A",),
            primary_metric="accuracy",
            secondary_metrics=(),
            dataset_name="test",
            dataset_config=None,
            default_split="test",
        )

        assert spec.num_classes == 1


class TestTaskSpecStringRepresentation:
    """Tests for __str__ and __repr__ methods."""

    @pytest.mark.unit
    def test_str_format(self):
        """Test __str__ output format."""
        spec = TaskSpec(
            name="lcc_classification",
            task_type=TaskType.CLASSIFICATION,
            description="LCC classification task",
            text_field="body",
            label_field="lcc",
            id_field="id",
            label_space=("A", "B", "C"),
            primary_metric="macro_f1",
            secondary_metrics=("micro_f1",),
            dataset_name="mjbommar/SHELF",
            dataset_config="default",
            default_split="test",
        )

        s = str(spec)
        assert "TaskSpec(" in s
        assert "lcc_classification" in s
        assert "type=classification" in s
        assert "metric=macro_f1" in s

    @pytest.mark.unit
    def test_str_contains_key_info(self):
        """Test __str__ contains essential information."""
        spec = TaskSpec(
            name="test_task",
            task_type=TaskType.RETRIEVAL,
            description="Test",
            text_field="text",
            label_field="label",
            id_field="id",
            label_space=None,
            primary_metric="ndcg@10",
            secondary_metrics=(),
            dataset_name="test",
            dataset_config=None,
            default_split="test",
        )

        s = str(spec)
        assert "test_task" in s
        assert "retrieval" in s
        assert "ndcg@10" in s

    @pytest.mark.unit
    def test_repr_format(self):
        """Test __repr__ output format."""
        spec = TaskSpec(
            name="lcc_classification",
            task_type=TaskType.CLASSIFICATION,
            description="LCC classification task",
            text_field="body",
            label_field="lcc",
            id_field="id",
            label_space=("A", "B", "C"),
            primary_metric="macro_f1",
            secondary_metrics=("micro_f1",),
            dataset_name="mjbommar/SHELF",
            dataset_config="default",
            default_split="test",
        )

        r = repr(spec)
        assert "TaskSpec(" in r
        assert "name='lcc_classification'" in r
        assert "task_type=TaskType.CLASSIFICATION" in r
        assert "label_field='lcc'" in r
        assert "primary_metric='macro_f1'" in r

    @pytest.mark.unit
    def test_repr_contains_essential_fields(self):
        """Test __repr__ contains essential fields for debugging."""
        spec = TaskSpec(
            name="test",
            task_type=TaskType.CLUSTERING,
            description="Test",
            text_field="embeddings",
            label_field="cluster",
            id_field="doc_id",
            label_space=None,
            primary_metric="ari",
            secondary_metrics=("nmi", "v_measure"),
            dataset_name="test",
            dataset_config=None,
            default_split="test",
        )

        r = repr(spec)
        assert "name='test'" in r
        assert "TaskType.CLUSTERING" in r
        assert "label_field='cluster'" in r
        assert "primary_metric='ari'" in r

    @pytest.mark.unit
    def test_str_and_repr_different(self):
        """Test that __str__ and __repr__ produce different outputs."""
        spec = TaskSpec(
            name="test",
            task_type=TaskType.CLASSIFICATION,
            description="Test",
            text_field="text",
            label_field="label",
            id_field="id",
            label_space=("A", "B"),
            primary_metric="accuracy",
            secondary_metrics=(),
            dataset_name="test",
            dataset_config=None,
            default_split="test",
        )

        assert str(spec) != repr(spec)
        # __str__ is more concise
        assert len(str(spec)) < len(repr(spec))


class TestTaskSpecEquality:
    """Tests for TaskSpec equality comparison."""

    @pytest.mark.unit
    def test_equality_same_specs(self):
        """Test two identical TaskSpecs are equal."""
        spec1 = TaskSpec(
            name="test",
            task_type=TaskType.CLASSIFICATION,
            description="Test task",
            text_field="text",
            label_field="label",
            id_field="id",
            label_space=("A", "B"),
            primary_metric="accuracy",
            secondary_metrics=("f1",),
            dataset_name="test",
            dataset_config="default",
            default_split="test",
        )

        spec2 = TaskSpec(
            name="test",
            task_type=TaskType.CLASSIFICATION,
            description="Test task",
            text_field="text",
            label_field="label",
            id_field="id",
            label_space=("A", "B"),
            primary_metric="accuracy",
            secondary_metrics=("f1",),
            dataset_name="test",
            dataset_config="default",
            default_split="test",
        )

        assert spec1 == spec2

    @pytest.mark.unit
    def test_inequality_different_names(self):
        """Test TaskSpecs with different names are not equal."""
        spec1 = TaskSpec(
            name="test1",
            task_type=TaskType.CLASSIFICATION,
            description="Test",
            text_field="text",
            label_field="label",
            id_field="id",
            label_space=("A", "B"),
            primary_metric="accuracy",
            secondary_metrics=(),
            dataset_name="test",
            dataset_config=None,
            default_split="test",
        )

        spec2 = TaskSpec(
            name="test2",
            task_type=TaskType.CLASSIFICATION,
            description="Test",
            text_field="text",
            label_field="label",
            id_field="id",
            label_space=("A", "B"),
            primary_metric="accuracy",
            secondary_metrics=(),
            dataset_name="test",
            dataset_config=None,
            default_split="test",
        )

        assert spec1 != spec2

    @pytest.mark.unit
    def test_inequality_different_task_types(self):
        """Test TaskSpecs with different task types are not equal."""
        spec1 = TaskSpec(
            name="test",
            task_type=TaskType.CLASSIFICATION,
            description="Test",
            text_field="text",
            label_field="label",
            id_field="id",
            label_space=("A", "B"),
            primary_metric="accuracy",
            secondary_metrics=(),
            dataset_name="test",
            dataset_config=None,
            default_split="test",
        )

        spec2 = TaskSpec(
            name="test",
            task_type=TaskType.RETRIEVAL,
            description="Test",
            text_field="text",
            label_field="label",
            id_field="id",
            label_space=("A", "B"),
            primary_metric="accuracy",
            secondary_metrics=(),
            dataset_name="test",
            dataset_config=None,
            default_split="test",
        )

        assert spec1 != spec2

    @pytest.mark.unit
    def test_inequality_different_label_space(self):
        """Test TaskSpecs with different label spaces are not equal."""
        spec1 = TaskSpec(
            name="test",
            task_type=TaskType.CLASSIFICATION,
            description="Test",
            text_field="text",
            label_field="label",
            id_field="id",
            label_space=("A", "B"),
            primary_metric="accuracy",
            secondary_metrics=(),
            dataset_name="test",
            dataset_config=None,
            default_split="test",
        )

        spec2 = TaskSpec(
            name="test",
            task_type=TaskType.CLASSIFICATION,
            description="Test",
            text_field="text",
            label_field="label",
            id_field="id",
            label_space=("C", "D"),
            primary_metric="accuracy",
            secondary_metrics=(),
            dataset_name="test",
            dataset_config=None,
            default_split="test",
        )

        assert spec1 != spec2

    @pytest.mark.unit
    def test_inequality_different_metrics(self):
        """Test TaskSpecs with different metrics are not equal."""
        spec1 = TaskSpec(
            name="test",
            task_type=TaskType.CLASSIFICATION,
            description="Test",
            text_field="text",
            label_field="label",
            id_field="id",
            label_space=("A", "B"),
            primary_metric="accuracy",
            secondary_metrics=("f1",),
            dataset_name="test",
            dataset_config=None,
            default_split="test",
        )

        spec2 = TaskSpec(
            name="test",
            task_type=TaskType.CLASSIFICATION,
            description="Test",
            text_field="text",
            label_field="label",
            id_field="id",
            label_space=("A", "B"),
            primary_metric="accuracy",
            secondary_metrics=("precision",),
            dataset_name="test",
            dataset_config=None,
            default_split="test",
        )

        assert spec1 != spec2

    @pytest.mark.unit
    def test_hashable_as_dict_key(self):
        """Test TaskSpec can be used as dictionary key."""
        spec1 = TaskSpec(
            name="test1",
            task_type=TaskType.CLASSIFICATION,
            description="Test",
            text_field="text",
            label_field="label",
            id_field="id",
            label_space=("A", "B"),
            primary_metric="accuracy",
            secondary_metrics=(),
            dataset_name="test",
            dataset_config=None,
            default_split="test",
        )

        spec2 = TaskSpec(
            name="test2",
            task_type=TaskType.RETRIEVAL,
            description="Test",
            text_field="text",
            label_field="label",
            id_field="id",
            label_space=None,
            primary_metric="ndcg@10",
            secondary_metrics=(),
            dataset_name="test",
            dataset_config=None,
            default_split="test",
        )

        task_map = {
            spec1: "classification_task",
            spec2: "retrieval_task",
        }

        assert task_map[spec1] == "classification_task"
        assert task_map[spec2] == "retrieval_task"


class TestTaskSpecEdgeCases:
    """Edge case tests for TaskSpec."""

    @pytest.mark.unit
    def test_label_space_is_tuple_not_list(self):
        """Test that label_space must be tuple (immutable)."""
        # This should work (tuple)
        spec = TaskSpec(
            name="test",
            task_type=TaskType.CLASSIFICATION,
            description="Test",
            text_field="text",
            label_field="label",
            id_field="id",
            label_space=("A", "B", "C"),
            primary_metric="accuracy",
            secondary_metrics=(),
            dataset_name="test",
            dataset_config=None,
            default_split="test",
        )

        assert isinstance(spec.label_space, tuple)

    @pytest.mark.unit
    def test_secondary_metrics_is_tuple(self):
        """Test that secondary_metrics is stored as tuple."""
        spec = TaskSpec(
            name="test",
            task_type=TaskType.CLASSIFICATION,
            description="Test",
            text_field="text",
            label_field="label",
            id_field="id",
            label_space=("A", "B"),
            primary_metric="accuracy",
            secondary_metrics=("f1", "precision", "recall"),
            dataset_name="test",
            dataset_config=None,
            default_split="test",
        )

        assert isinstance(spec.secondary_metrics, tuple)
        assert len(spec.secondary_metrics) == 3

    @pytest.mark.unit
    def test_large_label_space(self):
        """Test TaskSpec with large label space."""
        # Create 133 labels (like LCGFT forms)
        large_label_space = tuple(f"form_{i:03d}" for i in range(133))

        spec = TaskSpec(
            name="lcgft_form",
            task_type=TaskType.CLASSIFICATION,
            description="LCGFT form classification",
            text_field="body",
            label_field="form",
            id_field="id",
            label_space=large_label_space,
            primary_metric="macro_f1",
            secondary_metrics=("micro_f1", "weighted_f1"),
            dataset_name="mjbommar/SHELF",
            dataset_config="default",
            default_split="test",
        )

        assert spec.num_classes == 133
        assert len(spec.label_space) == 133

    @pytest.mark.unit
    def test_special_characters_in_name(self):
        """Test TaskSpec with special characters in name."""
        spec = TaskSpec(
            name="lcc_classification_v2.1",
            task_type=TaskType.CLASSIFICATION,
            description="LCC classification (version 2.1)",
            text_field="body",
            label_field="lcc",
            id_field="id",
            label_space=("A", "B"),
            primary_metric="macro_f1",
            secondary_metrics=(),
            dataset_name="mjbommar/SHELF",
            dataset_config=None,
            default_split="test",
        )

        assert spec.name == "lcc_classification_v2.1"

    @pytest.mark.unit
    def test_metric_names_with_at_symbol(self):
        """Test TaskSpec with @ in metric names (e.g., ndcg@10)."""
        spec = TaskSpec(
            name="retrieval",
            task_type=TaskType.RETRIEVAL,
            description="Retrieval task",
            text_field="text",
            label_field="label",
            id_field="id",
            label_space=None,
            primary_metric="ndcg@10",
            secondary_metrics=("map@10", "recall@100", "mrr@10"),
            dataset_name="test",
            dataset_config=None,
            default_split="test",
        )

        assert spec.primary_metric == "ndcg@10"
        assert "map@10" in spec.secondary_metrics
        assert "recall@100" in spec.secondary_metrics

    @pytest.mark.unit
    def test_different_splits(self):
        """Test TaskSpec with different default splits."""
        train_spec = TaskSpec(
            name="test",
            task_type=TaskType.CLASSIFICATION,
            description="Test",
            text_field="text",
            label_field="label",
            id_field="id",
            label_space=("A", "B"),
            primary_metric="accuracy",
            secondary_metrics=(),
            dataset_name="test",
            dataset_config=None,
            default_split="train",
        )

        val_spec = TaskSpec(
            name="test",
            task_type=TaskType.CLASSIFICATION,
            description="Test",
            text_field="text",
            label_field="label",
            id_field="id",
            label_space=("A", "B"),
            primary_metric="accuracy",
            secondary_metrics=(),
            dataset_name="test",
            dataset_config=None,
            default_split="validation",
        )

        test_spec = TaskSpec(
            name="test",
            task_type=TaskType.CLASSIFICATION,
            description="Test",
            text_field="text",
            label_field="label",
            id_field="id",
            label_space=("A", "B"),
            primary_metric="accuracy",
            secondary_metrics=(),
            dataset_name="test",
            dataset_config=None,
            default_split="test",
        )

        assert train_spec.default_split == "train"
        assert val_spec.default_split == "validation"
        assert test_spec.default_split == "test"
        # Different splits mean specs are not equal
        assert train_spec != val_spec != test_spec
