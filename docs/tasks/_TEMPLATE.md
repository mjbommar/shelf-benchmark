# Task: {TASK_NAME}

> **Task Type**: {classification | retrieval | clustering | pair_classification}
> **Difficulty**: {easy | medium | hard}
> **Primary Metric**: {metric_name}

## Overview

{1-2 paragraph description of what this task evaluates and why it matters for NLP/ML systems. What capability does it test?}

## Task Definition

### Input
{Describe the input format - what does a model receive?}

### Output
{Describe the expected output format - what should a model produce?}

### Formal Definition
{Mathematical or formal specification if applicable}

## Dataset

### Source
{Where does the data come from? How was it generated?}

### Statistics
| Split | Documents | Label Distribution |
|-------|-----------|-------------------|
| Train | N | {summary} |
| Dev   | N | {summary} |
| Test  | N | {summary} |

### Label Space
{List all possible labels/classes with descriptions}

### Data Format
```json
{
  "id": "...",
  "text": "...",
  "label": "..."
}
```

## Evaluation

### Primary Metric
{Name}: {Definition and why this metric is appropriate}

### Secondary Metrics
{Additional metrics reported}

### Evaluation Protocol
{How evaluation is performed - any special considerations}

## Baselines

| Model | {Primary Metric} | {Secondary Metric} | Notes |
|-------|------------------|-------------------|-------|
| Random | X.X | X.X | Lower bound |
| TF-IDF + LR | X.X | X.X | Simple baseline |
| BERT | X.X | X.X | Neural baseline |
| Human | X.X | X.X | Upper bound estimate |

## Related Work

### Similar Tasks in Other Benchmarks
{How does this compare to tasks in GLUE, SuperGLUE, MTEB, etc.?}

### Relevant Literature
{Academic papers on similar classification/retrieval problems}

## Implementation Notes

### Running Evaluation
```bash
shelf evaluate --task {task_name} --model {model_path}
```

### Submission Format
{Format for leaderboard submissions}

## References

1. {Citation with URL}
2. {Citation with URL}

---
*Last updated: {DATE}*
