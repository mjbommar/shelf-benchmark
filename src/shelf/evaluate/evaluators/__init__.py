"""Task evaluators for SHELF benchmark."""

from shelf.evaluate.evaluators.base import TaskEvaluator
from shelf.evaluate.evaluators.classification import ClassificationEvaluator
from shelf.evaluate.evaluators.clustering import ClusteringEvaluator
from shelf.evaluate.evaluators.clustering_agglomerative import (
    AgglomerativeClusteringEvaluator,
)
from shelf.evaluate.evaluators.clustering_conditional import (
    SubjectConditionalClusteringEvaluator,
)
from shelf.evaluate.evaluators.clustering_hdbscan import HDBSCANClusteringEvaluator
from shelf.evaluate.evaluators.multilabel import (
    MultiLabelClassificationEvaluator,
    validate_multilabel_predictions,
)
from shelf.evaluate.evaluators.pair import PairClassificationEvaluator
from shelf.evaluate.evaluators.retrieval import (
    GRADED_SCHEMES,
    GradedJudge,
    GradedScheme,
    RetrievalEvaluator,
)

__all__ = [
    "TaskEvaluator",
    "ClassificationEvaluator",
    "ClusteringEvaluator",
    "AgglomerativeClusteringEvaluator",
    "SubjectConditionalClusteringEvaluator",
    "HDBSCANClusteringEvaluator",
    "MultiLabelClassificationEvaluator",
    "validate_multilabel_predictions",
    "PairClassificationEvaluator",
    "RetrievalEvaluator",
    "GradedJudge",
    "GradedScheme",
    "GRADED_SCHEMES",
]
