# ClassificationEvaluator Unit Tests

## Overview

Comprehensive unit tests for `/home/mjbommar/src/shelf-benchmark/src/shelf/evaluate/evaluators/classification.py`.

Test file: `/home/mjbommar/src/shelf-benchmark/tests/unit/test_evaluators_classification.py`

## Test Coverage

### 1. Initialization Tests (`TestClassificationEvaluatorInit`)

- **test_init_basic**: Basic initialization with task spec
- **test_init_with_random_seed**: Custom random seed
- **test_init_stores_task_spec**: Task spec is properly stored

### 2. Perfect Predictions Tests (`TestClassificationEvaluatorPerfect`)

- **test_evaluate_perfect_predictions**: 100% accuracy scenario
- **test_perfect_predictions_no_misclassified**: No misclassified IDs
- **test_perfect_predictions_confusion_matrix**: Diagonal confusion matrix

### 3. Partial Predictions Tests (`TestClassificationEvaluatorPartial`)

- **test_evaluate_partial_predictions**: 50% accuracy scenario
- **test_partial_predictions_misclassified_ids**: Correct misclassified ID tracking
- **test_partial_predictions_per_class_metrics**: Per-class breakdown (precision, recall, F1, support)
- **test_partial_predictions_confusion_matrix**: Confusion matrix validation

### 4. Edge Cases Tests (`TestClassificationEvaluatorEdgeCases`)

- **test_evaluate_with_confidence_scores**: Optional confidence scores
- **test_evaluate_missing_prediction_raises_validation_error**: Missing predictions error
- **test_evaluate_duplicate_id_raises_validation_error**: Duplicate ID error
- **test_evaluate_unknown_id_raises_validation_error**: Unknown document ID error
- **test_evaluate_invalid_label_raises_validation_error**: Invalid label error
- **test_evaluate_single_class**: Single class evaluation
- **test_evaluate_all_wrong**: All predictions wrong (0% accuracy)
- **test_evaluate_misclassified_ids_limited_to_100**: Misclassified IDs capped at 100

### 5. Provenance and Context Tests (`TestClassificationEvaluatorProvenance`)

- **test_evaluate_includes_context**: EvaluationContext tracking
- **test_evaluate_includes_provenance**: DataProvenance tracking

### 6. Stratified Metrics Tests (`TestClassificationEvaluatorStratified`)

- **test_evaluate_without_stratification**: No stratification by default

### 7. Bootstrap CI Tests (`TestClassificationEvaluatorBootstrap`)

- **test_evaluate_compute_ci_false**: CIs not computed when compute_ci=False
- **test_evaluate_compute_ci_true_not_implemented**: compute_ci=True accepted (impl TBD)

### 8. Classifier Evaluation Tests (`TestClassificationEvaluatorClassifier`)

- **test_evaluate_classifier_perfect**: Mock perfect classifier
- **test_evaluate_classifier_partial**: Mock partial classifier
- **test_evaluate_classifier_wrong_prediction_count_raises**: Prediction count mismatch error
- **test_evaluate_classifier_uses_batch_size**: Batch size parameter passing

### 9. Embedder Evaluation Tests (`TestClassificationEvaluatorEmbedder`)

- **test_evaluate_embedder_with_classifier**: Mock embedder with LogisticRegression
- **test_evaluate_embedder_wrong_train_embedding_count_raises**: Train embedding count mismatch
- **test_evaluate_embedder_wrong_test_embedding_count_raises**: Test embedding count mismatch

### 10. Integration Tests (`TestClassificationEvaluatorIntegration`)

- **test_end_to_end_perfect**: Complete workflow with perfect predictions
- **test_end_to_end_partial**: Complete workflow with partial predictions
- **test_result_serialization**: JSON serialization/deserialization
- **test_result_to_dict**: Dictionary conversion
- **test_result_summary**: Human-readable summary generation

### 11. Label Space Tests (`TestClassificationEvaluatorLabelSpace`)

- **test_evaluate_with_explicit_label_space**: Explicit label space usage
- **test_evaluate_with_no_label_space**: Label inference from data

## Test Fixtures

### Ground Truth DataFrames

- **classification_task_spec**: Sample TaskSpec for LCC classification
- **ground_truth_df**: 6-document ground truth with LCC labels
- **ground_truth_df_larger**: 20-document ground truth for stratification tests

### Prediction Fixtures

- **perfect_predictions**: 100% accuracy predictions
- **partial_predictions**: 50% accuracy predictions
- **predictions_with_confidence**: Predictions with confidence scores

## Running Tests

```bash
# Run all classification evaluator tests
python -m pytest tests/unit/test_evaluators_classification.py -v

# Run specific test class
python -m pytest tests/unit/test_evaluators_classification.py::TestClassificationEvaluatorInit -v

# Run specific test
python -m pytest tests/unit/test_evaluators_classification.py::TestClassificationEvaluatorInit::test_init_basic -v

# Run with coverage
python -m pytest tests/unit/test_evaluators_classification.py --cov=shelf.evaluate.evaluators.classification

# Quick runner script
bash run_classification_tests.sh
```

## Test Markers

All tests use:
- `@pytest.mark.unit`: Unit test marker
- `@pytest.mark.evaluator`: Evaluator-specific marker

## Key Test Scenarios

### Happy Path
1. Perfect predictions (100% accuracy)
2. Partial predictions (50% accuracy)
3. Predictions with confidence scores

### Error Handling
1. Missing predictions → ValidationError
2. Duplicate IDs → ValidationError
3. Unknown document IDs → ValidationError
4. Invalid labels (not in label space) → ValidationError
5. Wrong prediction count from classifier → ValueError
6. Wrong embedding count from embedder → ValueError

### Metrics Validation
- Accuracy in [0, 1]
- F1 scores (macro, micro, weighted) in [0, 1]
- Per-class metrics: precision, recall, F1, support
- Confusion matrix: correct shape and values
- Misclassified IDs tracked correctly (limited to 100)

### Provenance Tracking
- EvaluationContext includes: random_seed, dataset_checksum, versions
- DataProvenance includes: commits, models, distributions

### Result Serialization
- to_dict() conversion
- to_json() / from_json() round-trip
- summary() human-readable output

## Test Statistics

- **Total test methods**: 41
- **Test classes**: 11
- **Fixtures**: 7
- **Code coverage target**: >95% for classification.py

## Dependencies

- pytest
- polars
- unittest.mock (MagicMock, patch)
- shelf.evaluate.evaluators.classification
- shelf.evaluate.results
- shelf.evaluate.schemas
- shelf.evaluate.tasks

## Notes

1. **Mocking**: Uses unittest.mock for classifier and embedder protocols
2. **Polars DataFrames**: All ground truth uses Polars (not Pandas)
3. **ValidationError**: Pydantic-based validation in schemas.py
4. **Bootstrap CIs**: compute_ci=True parameter exists but implementation pending
5. **Stratified metrics**: Feature defined but not yet implemented in base class

## Future Enhancements

1. Add stratified metrics tests when feature is implemented
2. Add bootstrap CI tests when implementation is complete
3. Add tests for evaluate_from_file() method
4. Add performance/benchmark tests for large datasets
