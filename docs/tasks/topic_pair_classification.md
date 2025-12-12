# Task: Topic Pair Classification

> **Task Type**: pair_classification
> **Difficulty**: medium-hard
> **Primary Metric**: F1 Score (Binary), Macro-F1 (Graded)

## Overview

The Topic Pair Classification tasks evaluate a model's ability to determine topical relationships between pairs of documents based on their Library of Congress Subject Headings (LCSH). Unlike the Same-LCC and Same-Form tasks which test categorical similarity at a single granularity level, topic pair classification tests **multi-label semantic overlap** — whether documents share subject matter despite potentially belonging to different LCC classes or document forms.

These tasks mirror real-world applications such as:
- **Topical clustering**: Grouping documents by shared themes across disciplinary boundaries
- **Related article recommendation**: Finding documents with overlapping subjects
- **Cross-domain research discovery**: Identifying connections between fields
- **Document deduplication**: Detecting content similarity beyond surface features

The key challenge is that SHELF documents have **1-4 topics per document** from a vocabulary of **112 unique topics**, creating a multi-label overlap problem. A document about "Ethics" and "Philosophy" might share one topic with a document about "Philosophy" and "Religion", requiring models to understand partial semantic overlap rather than strict categorical equivalence.

## Task Definition

SHELF includes two topic overlap pair classification tasks:

1. **Same-Topic Prediction (Binary)**: Predict whether two documents share ANY topic (binary classification)
2. **Topic Overlap Count (Graded)**: Predict HOW MANY topics two documents share (4-class classification: 0, 1, 2, 3+)

### Input

Each instance consists of two documents, represented by their textual content:
- **Document A**: A title and body text from the SHELF corpus
- **Document B**: A second title and body text from the SHELF corpus

Models receive the concatenation or paired encoding of these documents, depending on the model architecture.

### Output

**Binary Task (Same-Topic)**:
- **1 (Positive)**: The documents share at least one topic
- **0 (Negative)**: The documents have no topics in common

**Graded Task (Topic Overlap)**:
- **0**: No shared topics
- **1**: Exactly 1 shared topic
- **2**: Exactly 2 shared topics
- **3**: 3 or more shared topics (capped at 3+ due to rarity)

### Formal Definition

For Same-Topic Prediction:
```
f: (doc_a, doc_b) -> {0, 1}
where f(doc_a, doc_b) = 1 iff |topics(doc_a) ∩ topics(doc_b)| > 0
```

For Topic Overlap Count:
```
f: (doc_a, doc_b) -> {0, 1, 2, 3}
where f(doc_a, doc_b) = min(|topics(doc_a) ∩ topics(doc_b)|, 3)
```

## Dataset Construction

### Multi-Label Topic Structure

SHELF documents are annotated with 1-4 Library of Congress Subject Headings from a vocabulary of 112 unique topics. Examples include:
- Philosophy, Culture, Ethics, Religion
- Science, Technology, Environment, Health
- Law, Politics, Government, Democracy
- Art, Music, Literature, Film
- Business, Economics, Trade, Finance

Topic co-occurrence is **designed to be independent** of LCC class and document form, meaning:
- A Map (visual work) and a Lecture (instructional work) can both be about "Climate change"
- A document in class Q (Science) and class K (Law) can both cover "Ethics"
- Documents with different forms but shared topics test semantic understanding beyond genre

### Pair Generation Strategy

Pairs are constructed using a balanced sampling approach with controlled overlap distributions:

**Binary Task Distribution**:
- **50% negative pairs** (label=0): No shared topics
- **50% positive pairs** (label=1): At least one shared topic

**Graded Task Distribution**:
- **40% zero overlap** (label=0): No shared topics
- **30% single overlap** (label=1): Exactly 1 shared topic
- **20% double overlap** (label=2): Exactly 2 shared topics
- **10% triple+ overlap** (label=3): 3 or more shared topics

**Sampling Algorithm**:

For **zero overlap** pairs:
1. Randomly sample two documents with topics
2. Accept if they have no topics in common
3. Repeat until target count reached

For **positive overlap** pairs:
1. Sample document A with topics
2. For exact overlap counts (graded mode): Search for document B that shares exactly N topics
3. For any overlap (binary mode): Sample from documents sharing at least one topic with A
4. Build inverted index (topic → documents) for efficient candidate retrieval

**Balancing Challenges**:
- Finding exact overlap counts (especially 2 or 3+) is harder than any/none
- Algorithm uses rejection sampling with up to 3× attempts per target
- Topic distribution affects pair availability (common topics have more candidates)

### Statistics

| Split | Pairs | Label Distribution (Binary) |
|-------|-------|----------------------------|
| Train | 20,000 | 50% pos / 50% neg |
| Validation | 4,000 | 50% pos / 50% neg |
| Test | 4,000 | 50% pos / 50% neg |

| Split | Pairs | Label Distribution (Graded) |
|-------|-------|----------------------------|
| Train | 20,000 | 40% L0 / 30% L1 / 20% L2 / 10% L3+ |
| Validation | 4,000 | 40% L0 / 30% L1 / 20% L2 / 10% L3+ |
| Test | 4,000 | 40% L0 / 30% L1 / 20% L2 / 10% L3+ |

### Label Space

**Binary Classification**:
- `0`: No shared topics (negative pair)
- `1`: At least one shared topic (positive pair)

**Graded Classification**:
- `0`: No shared topics
- `1`: Exactly 1 shared topic
- `2`: Exactly 2 shared topics
- `3`: 3 or more shared topics

### Data Format

```json
{
  "id": "pair_012345",
  "doc_a_id": "20251211_030155_57dfc238",
  "doc_a_title": "Ethics in Modern Technology Development",
  "doc_a_body": "As artificial intelligence systems become more prevalent...",
  "doc_b_id": "20251211_030155_a8f92b4c",
  "doc_b_title": "Philosophical Approaches to Machine Learning",
  "doc_b_body": "The intersection of philosophy and computer science...",
  "label": 2,
  "overlap_count": 2,
  "shared_topics": ["Ethics", "Philosophy"]
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
```python
repr_a = encode(doc_a)
repr_b = encode(doc_b)
# Combine with concatenation, difference, and element-wise product
combined = [repr_a; repr_b; |repr_a - repr_b|; repr_a * repr_b]
```

### Symmetric Encoding

Since topic overlap is commutative:
```
f(doc_a, doc_b) = f(doc_b, doc_a)
```

Models should be trained with data augmentation that swaps pair order to enforce symmetry.

## Evaluation

### Primary Metric

**Binary Task: F1 Score**
- The harmonic mean of precision and recall for the positive class (shared topics)
- Balances the model's ability to identify true overlaps (recall) with accuracy (precision)

**Graded Task: Macro-F1 Score**
- Unweighted average of F1 scores across all 4 classes
- Ensures the model performs well on all overlap counts, not just common ones
- Prevents models from ignoring rare classes (2+ overlap)

```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
Macro-F1 = (F1_0 + F1_1 + F1_2 + F1_3) / 4
```

### Secondary Metrics

**Accuracy**: Fraction of correctly classified pairs
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Per-Class F1** (Graded task only): F1 for each overlap count separately
- Shows model performance on specific overlap levels
- Diagnostic for identifying which overlap counts are harder

**Confusion Matrix**: Full confusion matrix for error analysis
- Reveals whether model confuses adjacent classes (e.g., 1 vs 2 overlap)
- Helps identify systematic biases

**Matthews Correlation Coefficient (MCC)** (Binary task only):
```
MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
```
- More robust than accuracy for binary classification
- Ranges from -1 (total disagreement) to +1 (perfect prediction)

### Evaluation Protocol

1. Generate predictions for all pairs in the test set
2. Compute primary and secondary metrics
3. Report per-class breakdown (graded task)
4. Statistical significance testing using bootstrap resampling (1000 iterations)
5. Error analysis: Sample misclassified pairs for qualitative review

### Why These Metrics?

**Binary Task**: F1 balances precision and recall, crucial when false positives (claiming overlap when none exists) and false negatives (missing actual overlap) have different costs. Following GLUE MRPC/QQP conventions.

**Graded Task**: Macro-F1 prevents models from achieving high accuracy by only predicting common classes (0 or 1 overlap). The model must understand all levels of overlap to score well.

## Baselines

| Model | Binary F1/Acc | Graded Macro-F1/Acc | Notes |
|-------|---------------|---------------------|-------|
| Random | 50.0 / 50.0 | 25.0 / 40.0 | Lower bound (random guessing) |
| Majority Class | 0.0 / 50.0 | 13.3 / 40.0 | Always predicts most common class |
| TF-IDF + Cosine | TBD | TBD | Threshold on cosine similarity |
| TF-IDF + LR | TBD | TBD | Logistic regression on concatenated vectors |
| Topic Jaccard | TBD | TBD | Oracle using true topics (upper bound) |
| SBERT Cosine | TBD | TBD | Sentence-BERT embeddings |
| BERT Cross-Encoder | TBD | TBD | Fine-tuned BERT classification |

**Note**: The "Topic Jaccard" baseline represents an oracle that can perfectly identify topics. It provides an upper bound since it has access to ground truth topic labels, showing the maximum achievable performance if the model could perfectly extract topics from text.

## Relationship to Other Pair Classification Tasks

### Comparison to GLUE-Style Tasks

| Aspect | MRPC/QQP | SHELF Same-LCC/Form | SHELF Topic Overlap |
|--------|----------|---------------------|---------------------|
| Domain | News/Q&A | Bibliographic | Bibliographic |
| Label | Binary (paraphrase) | Binary (same category) | Binary or 4-class (overlap) |
| Similarity Type | Semantic equivalence | Categorical equivalence | Multi-label overlap |
| Avg Length | 10-20 words | 50-200 words | 50-200 words |
| Positive % | 37% | ~50% | 50% (binary) / varied (graded) |
| Ground Truth | Human annotation | LOC cataloging | LOC subject headings |

### Key Differences from Same-LCC/Form Tasks

**1. Multi-Label vs Single-Label**:
- Same-LCC/Form: Each document has exactly one LCC code and one form → binary match
- Topic Overlap: Each document has 1-4 topics → partial overlap possible

**2. Overlap Semantics**:
- Same-LCC/Form: All-or-nothing (same category or different)
- Topic Overlap: Gradual (0, 1, 2, 3+ shared topics)

**3. Cross-Domain Possibility**:
- Same-LCC/Form: Documents with same label are usually thematically similar
- Topic Overlap: Documents from different LCC classes/forms can share topics
  - Example: A Law document (class K) and a Science document (class Q) can both discuss "Ethics"

**4. Task Difficulty**:
- Same-LCC/Form: Medium (21 and 133 classes respectively)
- Topic Overlap Binary: Medium-Hard (must identify semantic overlap across 112 topics)
- Topic Overlap Graded: Hard (must count exact overlap, not just detect it)

## What This Tests: Multi-Label Semantic Understanding

The topic pair classification tasks evaluate several advanced model capabilities:

**1. Multi-Label Reasoning**:
- Can the model understand that documents can belong to multiple topics simultaneously?
- Does it recognize partial overlap (sharing some but not all topics)?

**2. Fine-Grained Semantic Similarity**:
- Beyond detecting "similar" vs "different", can it quantify similarity (1, 2, or 3+ shared topics)?
- Can it distinguish between documents sharing "Philosophy" vs sharing both "Philosophy" and "Ethics"?

**3. Cross-Domain Topic Detection**:
- Can the model identify shared topics even when documents differ in:
  - LCC class (subject discipline)
  - Document form (lecture vs map vs satire)
  - Writing register (academic vs casual)
  - Document length (50 vs 500 words)

**4. Topic Independence from Genre**:
- SHELF's topic distribution is designed to be independent of genre/form
- Models must learn topic semantics, not genre-topic correlations
- Example: "Ethics" appears in Legal briefs, Philosophical essays, and Scientific reports

**5. Abstraction and Generalization**:
- Topics are high-level Library of Congress Subject Headings
- Models must abstract from specific terms in text to broad topical categories
- Example: Document discussing "machine learning fairness" → topic "Ethics"

## Related Work

### Similar Tasks in Other Benchmarks

**GLUE (Wang et al., 2019)**:
- MRPC: Binary paraphrase detection
- QQP: Binary duplicate question detection
- STS-B: Semantic Textual Similarity (regression, 0-5 scale)

**SuperGLUE (Wang et al., 2019)**:
- Does not include pair classification (focuses on reasoning)

**MTEB (Muennighoff et al., 2023)**:
- PairClassification tasks: SprintDuplicateQuestions, TwitterSemEval2015
- Mostly binary classification, not graded

**SemEval STS Tasks (2012-2017)**:
- Continuous similarity scores [0-5]
- SHELF uses discrete bins (0, 1, 2, 3+) for interpretability

### Unique Contributions of SHELF Topic Tasks

**1. Multi-Label Pair Classification**:
- Most pair tasks (MRPC, QQP) are binary "same/different"
- SHELF adds graded overlap counting for multi-label scenarios
- Novel task design inspired by set intersection problems

**2. Bibliographic Domain**:
- Library of Congress subject headings provide expert-curated topics
- More structured than open-domain similarity (STS)
- More diverse than domain-specific tasks (e.g., medical entity linking)

**3. Independence Design**:
- Topic distribution is explicitly independent of LCC/form
- Prevents models from exploiting genre-topic correlations
- Tests pure semantic understanding

### Relevant Literature

1. **Multi-Label Text Classification**:
   - Tsoumakas & Katakis (2007): "Multi-Label Classification: An Overview"
   - Nam et al. (2014): "Large-scale Multi-label Text Classification"

2. **Semantic Similarity**:
   - Reimers & Gurevych (2019): "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"
   - Cer et al. (2017): "SemEval-2017 Task 1: Semantic Textual Similarity"

3. **Set Overlap and Jaccard Similarity**:
   - Jaccard (1912): "The Distribution of the Flora in the Alpine Zone"
   - Broder (1997): "On the Resemblance and Containment of Documents"

4. **Library Science**:
   - Mai (2011): "The Quality and Qualities of Information Retrieval Thesauri"
   - Library of Congress Subject Headings (LCSH): https://www.loc.gov/aba/cataloging/subject/

## Implementation Notes

### Running Evaluation

```bash
# Evaluate Same-Topic Prediction (Binary)
shelf evaluate --task same_topic --model <model_path>

# Evaluate Topic Overlap Count (Graded)
shelf evaluate --task topic_overlap --model <model_path>

# Evaluate both tasks
shelf evaluate --task topic_pair_classification --model <model_path>
```

### Submission Format

Predictions should be submitted as a JSONL file with one prediction per line:

```json
{
  "task": "same_topic",  // or "topic_overlap"
  "predictions": [
    {
      "id": "pair_012345",
      "prediction": 1,  // Binary: 0 or 1; Graded: 0, 1, 2, or 3
      "confidence": 0.87,  // optional probability
      "predicted_topics_a": ["Ethics", "Philosophy"],  // optional
      "predicted_topics_b": ["Philosophy", "Science"]   // optional
    }
  ]
}
```

### Loading from HuggingFace

```python
from datasets import load_dataset

# Load binary topic pairs
binary_pairs = load_dataset("mjbommar/SHELF", name="same_topic_pairs")

# Load graded topic overlap pairs
graded_pairs = load_dataset("mjbommar/SHELF", name="topic_overlap_pairs")

# Inspect data
print(binary_pairs["train"][0])
# {'id': 'pair_000000', 'doc_a_id': '...', 'doc_a_title': '...',
#  'doc_a_body': '...', 'doc_b_id': '...', 'doc_b_title': '...',
#  'doc_b_body': '...', 'label': 1, 'overlap_count': 2,
#  'shared_topics': ['Ethics', 'Philosophy']}
```

### Training Tips

**1. Data Augmentation**:
- Swap document order to ensure symmetric learning: f(A,B) = f(B,A)
- Randomly shuffle topics within documents during training (augmentation)

**2. Hard Negative Mining**:
- Include pairs from documents with similar but non-overlapping topics
- Example: ("Philosophy", "Ethics") vs ("Science", "Technology")

**3. Curriculum Learning**:
- Start with easy pairs (0 vs 3+ overlap)
- Progress to hard pairs (1 vs 2 overlap)
- Helps model learn decision boundaries

**4. Multi-Task Learning**:
- Jointly train on binary and graded tasks
- Share encoder, separate classification heads
- Graded task provides richer supervision signal

**5. Topic-Aware Pretraining**:
- Pretrain on topic classification (single documents)
- Fine-tune on pair classification
- Helps model learn topic representations

**6. Handling Class Imbalance (Graded)**:
- Use class weights inversely proportional to frequency
- Weight 0: 0.25, 1: 0.33, 2: 0.50, 3+: 1.0
- Prevents model from ignoring rare overlap counts

**7. Architecture Choices**:
- **Cross-encoders** (BERT with [CLS] token): Higher accuracy, slower
- **Bi-encoders** (SBERT-style): Faster, can cache embeddings, slightly lower accuracy
- **Late interaction** (ColBERT-style): Balance of speed and accuracy

## Considerations and Limitations

### Synthetic Data Implications

- Topics assigned during generation, may not perfectly match text content
- Real-world topic assignment involves human cataloger judgment
- Some topic combinations may be rare or artificial

### Topic Granularity

- 112 topics is coarser than full LCSH (thousands of headings)
- Selected topics cover broad categories, not fine-grained subjects
- Trade-off between coverage and tractability

### Overlap Distribution

- Natural topic co-occurrence is non-uniform
- Some topics co-occur more frequently (e.g., "Law" + "Politics")
- Controlled sampling ensures balanced training, but test data may have biases

### Evaluation Challenges

- Graded task is harder due to class granularity
- Models may struggle to distinguish 1 vs 2 overlap
- Adjacent class confusion (predicting 1 when true label is 2) should be penalized less than distant class confusion (predicting 0 when true label is 3)

## References

1. [GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding](https://openreview.net/pdf?id=rJ4km2R5t7) (Wang et al., 2019)
2. [MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316) (Muennighoff et al., 2023)
3. [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084) (Reimers & Gurevych, 2019)
4. [SemEval-2017 Task 1: Semantic Textual Similarity](https://arxiv.org/abs/1708.00055) (Cer et al., 2017)
5. [Multi-Label Classification: An Overview](https://doi.org/10.1007/s10618-007-0064-6) (Tsoumakas & Katakis, 2007)
6. [Large-scale Multi-label Text Classification — Revisiting Neural Networks](https://arxiv.org/abs/1312.5419) (Nam et al., 2014)
7. [On the Resemblance and Containment of Documents](https://ieeexplore.ieee.org/document/666900) (Broder, 1997)
8. [Library of Congress Subject Headings](https://www.loc.gov/aba/cataloging/subject/)
9. [Library of Congress Classification](https://www.loc.gov/catdir/cpso/lcc.html)
10. [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) (Devlin et al., 2019)

---
*Last updated: 2025-12-12*
