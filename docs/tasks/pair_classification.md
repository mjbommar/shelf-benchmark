# Task: Pair Classification

> **Task Type**: pair_classification
> **Difficulty**: medium
> **Primary Metric**: F1 Score

## Overview

The Pair Classification tasks evaluate a model's ability to determine semantic relationships between pairs of documents based on their Library of Congress metadata. These tasks test whether models can understand document similarity at different levels of granularity: subject matter (LCC classes) and document form/genre (LCGFT forms).

Unlike single-document classification, pair classification requires models to process two documents jointly and reason about their relationship. This mirrors real-world applications such as duplicate detection, document clustering preprocessing, and recommendation systems. The tasks are inspired by GLUE benchmark pair classification tasks like MRPC (Microsoft Research Paraphrase Corpus) and QQP (Quora Question Pairs), but adapted for the bibliographic domain with Library of Congress taxonomy.

## Task Definition

SHELF includes two binary pair classification tasks:

1. **Same-LCC Prediction**: Predict whether two documents share the same Library of Congress Classification (LCC) code
2. **Same-Form Prediction**: Predict whether two documents share the same Library of Congress Genre/Form Term (LCGFT)

### Input

Each instance consists of two documents, represented by their textual content:
- **Document A**: A title and body text from the SHELF corpus
- **Document B**: A second title and body text from the SHELF corpus

Models receive the concatenation or paired encoding of these documents, depending on the model architecture.

### Output

A binary prediction:
- **1 (Positive)**: The documents share the same label (LCC code or LCGFT form)
- **0 (Negative)**: The documents have different labels

### Formal Definition

For Same-LCC Prediction:
```
f: (doc_a, doc_b) -> {0, 1}
where f(doc_a, doc_b) = 1 iff lcc_code(doc_a) = lcc_code(doc_b)
```

For Same-Form Prediction:
```
f: (doc_a, doc_b) -> {0, 1}
where f(doc_a, doc_b) = 1 iff lcgft_form(doc_a) = lcgft_form(doc_b)
```

## Dataset Construction

### Pair Generation Strategy

Pairs are constructed using a balanced sampling approach:

**Positive Pairs (label = 1)**:
- Sample two distinct documents with the same target label
- For Same-LCC: both documents have identical `lcc_code` values
- For Same-Form: both documents have identical `lcgft_form` values
- Documents are sampled to represent the distribution of labels in the corpus

**Negative Pairs (label = 0)**:
- Sample two documents with different target labels
- Random sampling ensures negative pairs span diverse label combinations
- Avoid trivially easy pairs (e.g., very different document lengths or topics)

### Balancing Strategy

To prevent models from exploiting class imbalance:
- Target distribution: 50% positive pairs, 50% negative pairs
- Similar to GLUE tasks, though MRPC and QQP have natural imbalances (37% and 37% positive, respectively)
- Stratified splits ensure train/dev/test maintain similar balance

### Statistics

| Split | Pairs | Positive % | Negative % |
|-------|-------|-----------|-----------|
| Train | TBD | ~50% | ~50% |
| Dev   | TBD | ~50% | ~50% |
| Test  | TBD | ~50% | ~50% |

### Label Space

**Binary Classification**:
- `0`: Different labels (negative pair)
- `1`: Same label (positive pair)

### Data Format

```json
{
  "id": "pair_12345",
  "doc_a": {
    "id": "20251211_030155_57dfc238",
    "title": "Quarterly Agribusiness Risk Memo on Emerging Externalities",
    "body": "In light of climate change, migratory wildlife now attends...",
    "lcc_code": "S",
    "lcgft_form": "Satire"
  },
  "doc_b": {
    "id": "20251211_030155_f3221f5a",
    "title": "Canticle over the Cuyahoga",
    "body": "In an Akron church basement, a gospel choir braids...",
    "lcc_code": "P",
    "lcgft_form": "Theological works"
  },
  "label_lcc": 0,
  "label_form": 0
}
```

## Input Encoding Strategies

Different model architectures require different input formats:

### Concatenation (for BERT-style models)

```
[CLS] doc_a_title [SEP] doc_a_body [SEP] doc_b_title [SEP] doc_b_body [SEP]
```

Using BERT's segment embeddings to distinguish documents:
- Segment A: doc_a (title + body)
- Segment B: doc_b (title + body)

### Separate Encoding (for bi-encoder models)

Encode each document independently, then combine representations:
```
repr_a = encode(doc_a)
repr_b = encode(doc_b)
combined = [repr_a; repr_b; |repr_a - repr_b|; repr_a * repr_b]
```

### Symmetric Encoding

Since pair order shouldn't matter (similarity is commutative):
```
f(doc_a, doc_b) = f(doc_b, doc_a)
```

Models should be trained with data augmentation that swaps pair order.

## Evaluation

### Primary Metric

**F1 Score**: The harmonic mean of precision and recall, computed for the positive class (same label).

```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

Where:
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)

F1 is the primary metric because it balances precision and recall, which is especially important when positive and negative pairs may have subtle differences. This follows the GLUE benchmark convention for MRPC and QQP tasks.

### Secondary Metrics

**Accuracy**: Fraction of correctly classified pairs
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Average Metric**: Following GLUE, we report the unweighted average of F1 and Accuracy
```
Average = (F1 + Accuracy) / 2
```

This average is used for leaderboard ranking to provide a balanced view of model performance.

**Matthews Correlation Coefficient (MCC)**: A more robust metric for binary classification
```
MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
```

MCC is reported as a diagnostic metric to detect overoptimistic results on imbalanced data.

### Evaluation Protocol

1. Generate predictions for all pairs in the test set
2. Compute TP, TN, FP, FN from predictions vs. ground truth
3. Calculate F1, Accuracy, and MCC
4. Report Average of F1 and Accuracy as primary score
5. Statistical significance testing using bootstrap resampling (1000 iterations)

### Why F1 and Accuracy?

The combination of F1 and Accuracy provides complementary information:
- **Accuracy** shows overall correctness but can be misleading on imbalanced data
- **F1** focuses on positive class performance, balancing precision (are positive predictions correct?) and recall (are positive instances found?)
- **Average** prevents models from optimizing only one metric at the expense of the other

This approach mirrors GLUE's MRPC and QQP tasks, which also report both metrics due to their class imbalances.

## Baselines

| Model | Same-LCC F1/Acc | Same-Form F1/Acc | Notes |
|-------|-----------------|------------------|-------|
| Random | 50.0 / 50.0 | 50.0 / 50.0 | Lower bound (random guessing) |
| Majority Class | 0.0 / 50.0 | 0.0 / 50.0 | Always predicts negative |
| TF-IDF + Cosine | TBD | TBD | Similarity threshold baseline |
| TF-IDF + LR | TBD | TBD | Logistic regression on concatenated TF-IDF vectors |
| SBERT Cosine | TBD | TBD | Sentence-BERT embeddings with cosine similarity |
| BERT Cross-Encoder | TBD | TBD | Fine-tuned BERT with [CLS] token classification |
| Human Agreement | N/A | N/A | Inter-annotator agreement (ground truth from LOC) |

## Relationship to GLUE-Style Tasks

### Comparison to MRPC and QQP

The SHELF pair classification tasks are structurally similar to GLUE's paraphrase and semantic equivalence tasks:

| Aspect | MRPC | QQP | SHELF Same-LCC/Form |
|--------|------|-----|------------------------|
| Domain | News paraphrases | Community Q&A | Library catalog documents |
| Label | Binary (paraphrase) | Binary (duplicate) | Binary (same label) |
| Avg Length | ~20 words/sentence | ~10 words/question | ~50-200 words/document |
| Positive % | 37% | 37% | ~50% |
| Similarity Type | Semantic equivalence | Question equivalence | Subject/form equivalence |
| Metrics | F1, Accuracy | F1, Accuracy | F1, Accuracy |

### Key Differences

**1. Similarity Definition**:
- MRPC/QQP: Direct semantic equivalence (same meaning)
- SHELF: Metadata-based equivalence (same subject/form, but different content)

**2. Document Length**:
- MRPC/QQP: Short texts (10-20 words)
- SHELF: Longer documents (25-500+ words)

**3. Ground Truth**:
- MRPC/QQP: Human-annotated similarity judgments
- SHELF: Library of Congress professional cataloging

**4. Task Granularity**:
- MRPC/QQP: Single semantic equivalence task
- SHELF: Multiple levels (subject classification, genre/form)

### What This Tests: Document Similarity Understanding

The pair classification tasks evaluate several model capabilities:

**1. Semantic Abstraction**: Can the model recognize that two documents about different topics (e.g., "climate policy in California" vs. "sustainable farming in Iowa") belong to the same high-level category (Agriculture)?

**2. Genre Recognition**: Can the model distinguish document forms (e.g., satire vs. theological work) beyond content similarity?

**3. Cross-Document Reasoning**: Unlike single-document classification, models must jointly process and compare two documents, requiring:
   - Attention mechanisms that span both documents
   - Representation learning that captures comparative features
   - Ability to focus on relevant signals (subject matter vs. style)

**4. Long-Context Understanding**: With documents averaging 50-200 words, models must maintain coherent representations over longer spans than typical GLUE tasks.

**5. Robustness to Surface Variation**: Documents with the same LCC code may have very different vocabulary, topics, and styles. Models must learn abstract categorical boundaries rather than lexical overlap.

### Relationship to NLI Tasks

The pair tasks differ from Natural Language Inference (SNLI, MultiNLI):

| Aspect | NLI (SNLI/MultiNLI) | SHELF Pair Classification |
|--------|---------------------|------------------------------|
| Labels | 3-way (entailment, neutral, contradiction) | Binary (same/different) |
| Relationship | Logical inference | Categorical equivalence |
| Asymmetric | Yes (premise → hypothesis) | No (symmetric) |
| Task | Does B follow from A? | Are A and B in the same category? |

NLI requires directional reasoning (premise entails hypothesis), while pair classification requires symmetric similarity assessment.

## Related Work

### Similar Tasks in Other Benchmarks

**GLUE (Wang et al., 2019)**:
- MRPC: Microsoft Research Paraphrase Corpus
- QQP: Quora Question Pairs
- STS-B: Semantic Textual Similarity (regression, not classification)

**SuperGLUE (Wang et al., 2019)**:
- Does not include pair classification tasks (focuses on harder reasoning)

**MTEB (Muennighoff et al., 2023)**:
- Includes pair classification in the "PairClassification" task category
- Tasks: SprintDuplicateQuestions, TwitterSemEval2015, TwitterURLCorpus

**SemEval STS Tasks (2012-2017)**:
- Semantic Textual Similarity with continuous scores [0-5]
- SHELF uses binary labels instead of continuous similarity

### Relevant Literature

1. **Paraphrase and Semantic Similarity**:
   - Dolan & Brockett (2005): "Automatically Constructing a Corpus of Sentential Paraphrases"
   - Agirre et al. (2012): "SemEval-2012 Task 6: A Pilot on Semantic Textual Similarity"

2. **Question Similarity**:
   - Lei et al. (2016): "Semi-supervised Question Retrieval with Gated Convolutions"
   - Wang et al. (2017): "Bilateral Multi-Perspective Matching for Natural Language Sentences"

3. **Document Classification and Similarity**:
   - Reimers & Gurevych (2019): "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"
   - Devlin et al. (2019): "BERT: Pre-training of Deep Bidirectional Transformers"

4. **Library Science and Classification**:
   - Mai (2011): "The Quality and Qualities of Information Retrieval Thesauri"
   - Tennis (2002): "Subject Ontogeny: Subject Access through Time and the Dimensionality of Classification"

## Implementation Notes

### Running Evaluation

```bash
# Evaluate Same-LCC Prediction
shelf evaluate --task same_lcc --model <model_path>

# Evaluate Same-Form Prediction
shelf evaluate --task same_form --model <model_path>

# Evaluate both tasks
shelf evaluate --task pair_classification --model <model_path>
```

### Submission Format

Predictions should be submitted as a JSON file:

```json
{
  "task": "same_lcc",  // or "same_form"
  "predictions": [
    {
      "id": "pair_12345",
      "prediction": 1,  // 0 or 1
      "confidence": 0.87  // optional
    },
    ...
  ]
}
```

### Training Tips

1. **Data Augmentation**: Swap document order to ensure symmetric learning
2. **Hard Negative Mining**: Include pairs from closely related categories (e.g., adjacent LCC classes)
3. **Curriculum Learning**: Start with easy pairs (very different categories) and progress to hard pairs
4. **Cross-Encoder vs. Bi-Encoder**:
   - Cross-encoders (BERT with concatenation) achieve higher accuracy but slower inference
   - Bi-encoders (SBERT-style) enable efficient retrieval but may have lower accuracy
5. **Regularization**: Use dropout and early stopping to prevent overfitting on pair patterns

## References

1. [GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding](https://openreview.net/pdf?id=rJ4km2R5t7) (Wang et al., 2019)
2. [GLUE Benchmark Official Website](https://gluebenchmark.com/)
3. [The Stanford Natural Language Inference (SNLI) Corpus](https://nlp.stanford.edu/projects/snli/)
4. [Natural Language Inference - Sentence Transformers](https://sbert.net/examples/training/nli/README.html)
5. [Evaluation Metrics for Binary Classification](https://neptune.ai/blog/evaluation-metrics-binary-classification)
6. [The Advantages of the Matthews Correlation Coefficient (MCC) over F1 Score](https://link.springer.com/article/10.1186/s12864-019-6413-7)
7. [Semantic Textual Similarity Methods, Tools, and Applications: A Survey](https://www.scielo.org.mx/scielo.php?script=sci_arttext&pid=S1405-55462016000400647)
8. [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084) (Reimers & Gurevych, 2019)
9. [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) (Devlin et al., 2019)
10. [Library of Congress Classification](https://www.loc.gov/catdir/cpso/lcc.html)
11. [Library of Congress Genre/Form Terms](https://www.loc.gov/aba/publications/FreeLCGFT/freelcgft.html)

---
*Last updated: 2025-12-10*
