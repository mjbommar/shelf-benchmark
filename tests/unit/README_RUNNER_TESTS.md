# Runner Module Unit Tests

## File: test_runner.py

Comprehensive unit tests for `src/shelf/evaluate/runner.py`.

## What is Tested

The runner module provides the main evaluation API:
- `evaluate()` - Main entry point for evaluating tasks
- `evaluate_all()` - Batch evaluation across multiple tasks
- `_create_evaluator()` - Factory for creating appropriate evaluators

## Test Structure

```
test_runner.py
├── Mock Objects
│   ├── MockEmbedder (for embedding-based evaluation)
│   ├── MockClassifier (for classification models)
│   └── MockRetriever (for retrieval models)
│
├── TestCreateEvaluator (6 tests)
│   └── Tests evaluator factory for all task types
│
├── TestEvaluateWithPredictions (4 tests)
│   └── Tests evaluate() with predictions (files and lists)
│
├── TestEvaluateWithModels (10 tests)
│   └── Tests evaluate() with model objects
│
├── TestEvaluateErrorHandling (2 tests)
│   └── Tests error conditions and edge cases
│
├── TestEvaluateOutputSaving (2 tests)
│   └── Tests result file saving
│
├── TestEvaluateAll (11 tests)
│   └── Tests batch evaluation functionality
│
└── TestPredictionFileChecksum (2 tests)
    └── Tests provenance tracking for predictions
```

## Running Tests

```bash
# All runner tests
uv run pytest tests/unit/test_runner.py -v

# Specific test class
uv run pytest tests/unit/test_runner.py::TestEvaluateAll -v

# Single test
uv run pytest tests/unit/test_runner.py::TestEvaluateAll::test_evaluate_all_default_retrieval_tasks -v

# Without coverage (faster)
uv run pytest tests/unit/test_runner.py --no-cov -v
```

## Key Testing Patterns

### 1. Mocking Evaluators
All evaluator classes are mocked to avoid loading datasets:

```python
@patch("shelf.evaluate.runner.ClassificationEvaluator")
def test_example(mock_evaluator_cls):
    mock_evaluator = MagicMock()
    mock_result = EvaluationResult(...)
    mock_evaluator.evaluate.return_value = mock_result
    mock_evaluator_cls.return_value = mock_evaluator
```

### 2. Testing Parameter Propagation
Verify parameters are passed through correctly:

```python
result = evaluate("lcc_retrieval", model=embedder, batch_size=64)

# Verify batch_size was passed
call_kwargs = mock_evaluator.evaluate_embedder.call_args[1]
assert call_kwargs["batch_size"] == 64
```

### 3. Testing Different Model Types
The runner detects model type by checking for specific methods:
- Embedder: Has `embed()` method
- Classifier: Has `predict()` method
- Retriever: Has `retrieve()` method

### 4. Testing File I/O
Uses temporary files for testing prediction file loading:

```python
predictions = [{"id": "doc_001", "prediction": "A"}]
pred_file = create_temp_predictions_file(predictions)
result = evaluate("lcc_classification", predictions=pred_file)
```

## Coverage

**37 tests** covering:
- ✅ All task types (retrieval, classification, clustering, pair)
- ✅ All input modes (file, list, model)
- ✅ All model types (embedder, classifier, retriever)
- ✅ Parameter passing and validation
- ✅ Error handling
- ✅ Output file saving
- ✅ Batch evaluation
- ✅ Provenance tracking

## Related Files

- Source: `src/shelf/evaluate/runner.py`
- Evaluators: `src/shelf/evaluate/evaluators/`
- Tasks: `src/shelf/evaluate/tasks.py`
- Registry: `src/shelf/evaluate/registry.py`
- Results: `src/shelf/evaluate/results.py`

## Notes

- All tests use `@pytest.mark.unit` marker
- Tests avoid network calls and dataset loading
- Mock objects simulate model interfaces without dependencies
- Follows same patterns as other unit tests in the suite
