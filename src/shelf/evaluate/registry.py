"""Task registry for SHELF evaluation.

This module defines all available evaluation tasks with their
specifications.
"""

from __future__ import annotations

from shelf.evaluate.tasks import TaskSpec, TaskType

# Library of Congress Classification codes (21 classes)
LCC_CODES = (
    "A",  # General Works
    "B",  # Philosophy, Psychology, Religion
    "C",  # Auxiliary Sciences of History
    "D",  # World History
    "E",  # History of the Americas (General)
    "F",  # History of the Americas (Local)
    "G",  # Geography, Anthropology, Recreation
    "H",  # Social Sciences
    "J",  # Political Science
    "K",  # Law
    "L",  # Education
    "M",  # Music
    "N",  # Fine Arts
    "P",  # Language and Literature
    "Q",  # Science
    "R",  # Medicine
    "S",  # Agriculture
    "T",  # Technology
    "U",  # Military Science
    "V",  # Naval Science
    "Z",  # Bibliography, Library Science
)

# LCGFT Categories (14 categories)
LCGFT_CATEGORIES = (
    "Cartographic materials",
    "Commemorative works",
    "Creative nonfiction",
    "Discursive works",
    "Ephemera",
    "Informational works",
    "Instructional and educational works",
    "Law materials",
    "Literature",
    "Music",
    "Recreational works",
    "Religious materials",
    "Sound recordings",
    "Visual works",
)

# Register of writing styles
REGISTERS = (
    "academic",
    "casual",
    "conversational",
    "creative",
    "formal",
    "journalistic",
    "professional",
    "technical",
)

# Geographic regions (8 clusters for geographic clustering)
GEOGRAPHIC_REGIONS = (
    "North America",
    "South America",
    "Europe",
    "East Asia",
    "South/Southeast Asia",
    "Middle East & North Africa",
    "Sub-Saharan Africa",
    "Central America & Caribbean",
)


# ============================================================================
# TASK REGISTRY
# ============================================================================

TASK_REGISTRY: dict[str, TaskSpec] = {
    # -------------------------------------------------------------------------
    # RETRIEVAL TASKS (highest priority for RAG)
    # -------------------------------------------------------------------------
    "lcc_retrieval": TaskSpec(
        name="lcc_retrieval",
        task_type=TaskType.RETRIEVAL,
        description="Retrieve documents with the same Library of Congress Classification (subject area)",
        text_field="text",
        label_field="lcc_code",
        id_field="id",
        label_space=LCC_CODES,
        primary_metric="ndcg@10",
        secondary_metrics=("mrr", "recall@10", "recall@100", "map@10"),
        dataset_name="mjbommar/SHELF",
        dataset_config="default",
        default_split="test",
    ),
    "form_retrieval": TaskSpec(
        name="form_retrieval",
        task_type=TaskType.RETRIEVAL,
        description="Retrieve documents with the same LCGFT genre/form",
        text_field="text",
        label_field="lcgft_form",
        id_field="id",
        label_space=None,  # 133 forms, open vocabulary
        primary_metric="ndcg@10",
        secondary_metrics=("mrr", "recall@10", "recall@100", "map@10"),
        dataset_name="mjbommar/SHELF",
        dataset_config="default",
        default_split="test",
    ),
    "category_retrieval": TaskSpec(
        name="category_retrieval",
        task_type=TaskType.RETRIEVAL,
        description="Retrieve documents with the same LCGFT category",
        text_field="text",
        label_field="lcgft_category",
        id_field="id",
        label_space=LCGFT_CATEGORIES,
        primary_metric="ndcg@10",
        secondary_metrics=("mrr", "recall@10", "recall@100", "map@10"),
        dataset_name="mjbommar/SHELF",
        dataset_config="default",
        default_split="test",
    ),
    # -------------------------------------------------------------------------
    # CLASSIFICATION TASKS
    # -------------------------------------------------------------------------
    "lcc_classification": TaskSpec(
        name="lcc_classification",
        task_type=TaskType.CLASSIFICATION,
        description="Classify documents into 21 Library of Congress subject classes",
        text_field="text",
        label_field="lcc_code",
        id_field="id",
        label_space=LCC_CODES,
        primary_metric="macro_f1",
        secondary_metrics=("micro_f1", "accuracy", "weighted_f1"),
        dataset_name="mjbommar/SHELF",
        dataset_config="default",
        default_split="test",
    ),
    "lcgft_category_classification": TaskSpec(
        name="lcgft_category_classification",
        task_type=TaskType.CLASSIFICATION,
        description="Classify documents into 14 LCGFT genre categories",
        text_field="text",
        label_field="lcgft_category",
        id_field="id",
        label_space=LCGFT_CATEGORIES,
        primary_metric="macro_f1",
        secondary_metrics=("micro_f1", "accuracy", "weighted_f1"),
        dataset_name="mjbommar/SHELF",
        dataset_config="default",
        default_split="test",
    ),
    "register_classification": TaskSpec(
        name="register_classification",
        task_type=TaskType.CLASSIFICATION,
        description="Classify documents by writing register/style",
        text_field="text",
        label_field="register",
        id_field="id",
        label_space=REGISTERS,
        primary_metric="macro_f1",
        secondary_metrics=("micro_f1", "accuracy", "weighted_f1"),
        dataset_name="mjbommar/SHELF",
        dataset_config="default",
        default_split="test",
    ),
    # -------------------------------------------------------------------------
    # CLUSTERING TASKS
    # -------------------------------------------------------------------------
    "lcc_clustering": TaskSpec(
        name="lcc_clustering",
        task_type=TaskType.CLUSTERING,
        description="Cluster documents into 21 subject groups",
        text_field="text",
        label_field="lcc_code",
        id_field="id",
        label_space=LCC_CODES,
        primary_metric="v_measure",
        secondary_metrics=("nmi", "ari"),
        dataset_name="mjbommar/SHELF",
        dataset_config="default",
        default_split="test",
    ),
    "lcgft_clustering": TaskSpec(
        name="lcgft_clustering",
        task_type=TaskType.CLUSTERING,
        description="Cluster documents into 14 genre categories",
        text_field="text",
        label_field="lcgft_category",
        id_field="id",
        label_space=LCGFT_CATEGORIES,
        primary_metric="v_measure",
        secondary_metrics=("nmi", "ari"),
        dataset_name="mjbommar/SHELF",
        dataset_config="default",
        default_split="test",
    ),
    "register_clustering": TaskSpec(
        name="register_clustering",
        task_type=TaskType.CLUSTERING,
        description="Cluster documents into 8 writing register/style groups",
        text_field="text",
        label_field="register",
        id_field="id",
        label_space=REGISTERS,
        primary_metric="v_measure",
        secondary_metrics=("nmi", "ari"),
        dataset_name="mjbommar/SHELF",
        dataset_config="default",
        default_split="test",
    ),
    "geographic_clustering": TaskSpec(
        name="geographic_clustering",
        task_type=TaskType.CLUSTERING,
        description="Cluster documents into 8 geographic regions based on content",
        text_field="text",
        label_field="geographic_region",  # Requires preprocessing with geographic.py
        id_field="id",
        label_space=GEOGRAPHIC_REGIONS,
        primary_metric="v_measure",
        secondary_metrics=("nmi", "ari"),
        dataset_name="mjbommar/SHELF",
        dataset_config="default",
        default_split="test",
    ),
    # -------------------------------------------------------------------------
    # CLUSTERING TASKS - HDBSCAN DISCOVERY (no fixed k)
    # -------------------------------------------------------------------------
    "lcc_clustering_hdbscan": TaskSpec(
        name="lcc_clustering_hdbscan",
        task_type=TaskType.CLUSTERING,
        description="Discover document clusters using HDBSCAN (no fixed k)",
        text_field="text",
        label_field="lcc_code",
        id_field="id",
        label_space=LCC_CODES,  # Used for evaluation, not clustering
        primary_metric="v_measure",
        secondary_metrics=("nmi", "ari", "noise_ratio", "cluster_k_error"),
        dataset_name="mjbommar/SHELF",
        dataset_config="default",
        default_split="test",
    ),
    "lcgft_clustering_hdbscan": TaskSpec(
        name="lcgft_clustering_hdbscan",
        task_type=TaskType.CLUSTERING,
        description="Discover genre clusters using HDBSCAN (no fixed k)",
        text_field="text",
        label_field="lcgft_category",
        id_field="id",
        label_space=LCGFT_CATEGORIES,
        primary_metric="v_measure",
        secondary_metrics=("nmi", "ari", "noise_ratio", "cluster_k_error"),
        dataset_name="mjbommar/SHELF",
        dataset_config="default",
        default_split="test",
    ),
    "register_clustering_hdbscan": TaskSpec(
        name="register_clustering_hdbscan",
        task_type=TaskType.CLUSTERING,
        description="Discover register clusters using HDBSCAN (no fixed k)",
        text_field="text",
        label_field="register",
        id_field="id",
        label_space=REGISTERS,
        primary_metric="v_measure",
        secondary_metrics=("nmi", "ari", "noise_ratio", "cluster_k_error"),
        dataset_name="mjbommar/SHELF",
        dataset_config="default",
        default_split="test",
    ),
    "geographic_clustering_hdbscan": TaskSpec(
        name="geographic_clustering_hdbscan",
        task_type=TaskType.CLUSTERING,
        description="Discover geographic clusters using HDBSCAN (no fixed k)",
        text_field="text",
        label_field="geographic_region",
        id_field="id",
        label_space=GEOGRAPHIC_REGIONS,
        primary_metric="v_measure",
        secondary_metrics=("nmi", "ari", "noise_ratio", "cluster_k_error"),
        dataset_name="mjbommar/SHELF",
        dataset_config="default",
        default_split="test",
    ),
    # -------------------------------------------------------------------------
    # CLUSTERING TASKS - AGGLOMERATIVE WITH COSINE DISTANCE
    # -------------------------------------------------------------------------
    "lcc_clustering_agglomerative": TaskSpec(
        name="lcc_clustering_agglomerative",
        task_type=TaskType.CLUSTERING,
        description="Cluster documents using agglomerative clustering with cosine distance",
        text_field="text",
        label_field="lcc_code",
        id_field="id",
        label_space=LCC_CODES,
        primary_metric="v_measure",
        secondary_metrics=("nmi", "ari"),
        dataset_name="mjbommar/SHELF",
        dataset_config="default",
        default_split="test",
    ),
    "lcgft_clustering_agglomerative": TaskSpec(
        name="lcgft_clustering_agglomerative",
        task_type=TaskType.CLUSTERING,
        description="Cluster genres using agglomerative clustering with cosine distance",
        text_field="text",
        label_field="lcgft_category",
        id_field="id",
        label_space=LCGFT_CATEGORIES,
        primary_metric="v_measure",
        secondary_metrics=("nmi", "ari"),
        dataset_name="mjbommar/SHELF",
        dataset_config="default",
        default_split="test",
    ),
    "register_clustering_agglomerative": TaskSpec(
        name="register_clustering_agglomerative",
        task_type=TaskType.CLUSTERING,
        description="Cluster registers using agglomerative clustering with cosine distance",
        text_field="text",
        label_field="register",
        id_field="id",
        label_space=REGISTERS,
        primary_metric="v_measure",
        secondary_metrics=("nmi", "ari"),
        dataset_name="mjbommar/SHELF",
        dataset_config="default",
        default_split="test",
    ),
    "geographic_clustering_agglomerative": TaskSpec(
        name="geographic_clustering_agglomerative",
        task_type=TaskType.CLUSTERING,
        description="Cluster geography using agglomerative clustering with cosine distance",
        text_field="text",
        label_field="geographic_region",
        id_field="id",
        label_space=GEOGRAPHIC_REGIONS,
        primary_metric="v_measure",
        secondary_metrics=("nmi", "ari"),
        dataset_name="mjbommar/SHELF",
        dataset_config="default",
        default_split="test",
    ),
    # -------------------------------------------------------------------------
    # PAIR CLASSIFICATION TASKS
    # -------------------------------------------------------------------------
    "same_lcc_pairs": TaskSpec(
        name="same_lcc_pairs",
        task_type=TaskType.PAIR_CLASSIFICATION,
        description="Predict whether two documents share the same LCC class",
        text_field="text",  # Will need special handling for pairs
        label_field="label",
        id_field="pair_id",
        label_space=("0", "1"),
        primary_metric="f1",
        secondary_metrics=("accuracy", "mcc"),
        dataset_name="mjbommar/SHELF",
        dataset_config="same_lcc_pairs",
        default_split="test",
    ),
    "same_form_pairs": TaskSpec(
        name="same_form_pairs",
        task_type=TaskType.PAIR_CLASSIFICATION,
        description="Predict whether two documents share the same LCGFT form",
        text_field="text",  # Will need special handling for pairs
        label_field="label",
        id_field="pair_id",
        label_space=("0", "1"),
        primary_metric="f1",
        secondary_metrics=("accuracy", "mcc"),
        dataset_name="mjbommar/SHELF",
        dataset_config="same_form_pairs",
        default_split="test",
    ),
    "same_register_pairs": TaskSpec(
        name="same_register_pairs",
        task_type=TaskType.PAIR_CLASSIFICATION,
        description="Predict whether two documents share the same writing register/style",
        text_field="text",  # Will need special handling for pairs
        label_field="label",
        id_field="pair_id",
        label_space=("0", "1"),
        primary_metric="f1",
        secondary_metrics=("accuracy", "mcc"),
        dataset_name="mjbommar/SHELF",
        dataset_config="same_register_pairs",
        default_split="test",
    ),
    "same_audience_pairs": TaskSpec(
        name="same_audience_pairs",
        task_type=TaskType.PAIR_CLASSIFICATION,
        description="Predict whether two documents share the same target audience",
        text_field="text",  # Will need special handling for pairs
        label_field="label",
        id_field="pair_id",
        label_space=("0", "1"),
        primary_metric="f1",
        secondary_metrics=("accuracy", "mcc"),
        dataset_name="mjbommar/SHELF",
        dataset_config="same_audience_pairs",
        default_split="test",
    ),
    "same_topic_pairs": TaskSpec(
        name="same_topic_pairs",
        task_type=TaskType.PAIR_CLASSIFICATION,
        description="Predict whether two documents share ANY topic (binary multi-label overlap)",
        text_field="text",  # Will need special handling for pairs
        label_field="label",
        id_field="pair_id",
        label_space=("0", "1"),
        primary_metric="f1",
        secondary_metrics=("accuracy", "mcc"),
        dataset_name="mjbommar/SHELF",
        dataset_config="same_topic_pairs",
        default_split="test",
    ),
    "topic_overlap_pairs": TaskSpec(
        name="topic_overlap_pairs",
        task_type=TaskType.PAIR_CLASSIFICATION,
        # Note: Original labels are 0-3+ overlap levels, but evaluated as binary
        # (0 = no overlap, 1+ = any overlap) since pair evaluators use similarity thresholds
        description="Predict whether two documents share any topics (binarized from 4-class overlap)",
        text_field="text",  # Will need special handling for pairs
        label_field="label",
        id_field="pair_id",
        label_space=("0", "1"),  # Evaluated as binary (any overlap vs none)
        primary_metric="f1",
        secondary_metrics=("accuracy", "auc_roc"),
        dataset_name="mjbommar/SHELF",
        dataset_config="topic_overlap_pairs",
        default_split="test",
    ),
}


def get_task(name: str) -> TaskSpec:
    """Get task specification by name.

    Args:
        name: Task name (e.g., "lcc_retrieval")

    Returns:
        TaskSpec for the requested task

    Raises:
        ValueError: If task name is not found
    """
    if name not in TASK_REGISTRY:
        available = sorted(TASK_REGISTRY.keys())
        raise ValueError(f"Unknown task: {name!r}. Available tasks: {available}")
    return TASK_REGISTRY[name]


def list_tasks(task_type: TaskType | None = None) -> list[str]:
    """List available task names.

    Args:
        task_type: Optional filter by task type

    Returns:
        List of task names
    """
    if task_type is None:
        return sorted(TASK_REGISTRY.keys())

    return sorted(
        name for name, spec in TASK_REGISTRY.items() if spec.task_type == task_type
    )


def list_retrieval_tasks() -> list[str]:
    """List all retrieval task names."""
    return list_tasks(TaskType.RETRIEVAL)


def list_classification_tasks() -> list[str]:
    """List all classification task names."""
    return list_tasks(TaskType.CLASSIFICATION)


def list_clustering_tasks() -> list[str]:
    """List all clustering task names."""
    return list_tasks(TaskType.CLUSTERING)


def list_pair_tasks() -> list[str]:
    """List all pair classification task names."""
    return list_tasks(TaskType.PAIR_CLASSIFICATION)


def list_clustering_kmeans_tasks() -> list[str]:
    """List k-means clustering task names (default algorithm)."""
    return [
        name
        for name in list_tasks(TaskType.CLUSTERING)
        if "_hdbscan" not in name and "_agglomerative" not in name
    ]


def list_clustering_hdbscan_tasks() -> list[str]:
    """List HDBSCAN clustering task names."""
    return [name for name in list_tasks(TaskType.CLUSTERING) if "_hdbscan" in name]


def list_clustering_agglomerative_tasks() -> list[str]:
    """List agglomerative clustering task names."""
    return [
        name for name in list_tasks(TaskType.CLUSTERING) if "_agglomerative" in name
    ]
