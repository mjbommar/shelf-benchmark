# Task: Audience Pair Classification

> **Task Type**: pair_classification
> **Difficulty**: medium
> **Primary Metric**: F1 Score

## Overview

The Audience Pair Classification task evaluates a model's ability to determine whether two documents share the same target audience. Unlike single-document audience classification, this task requires models to process two documents jointly and reason about their similarity in terms of reader-level characteristics: vocabulary complexity, assumed prior knowledge, explanatory depth, and mode of address.

This task tests whether models can recognize that two documents with different content may be calibrated for the same demographic group, requiring abstraction beyond surface-level lexical similarity to underlying sociolinguistic features. The task is inspired by GLUE benchmark pair classification tasks like MRPC and QQP, but adapted for the bibliographic domain with Library of Congress Demographic Group Terms (LCDGT).

## Task Definition

**Same-Audience Prediction**: Predict whether two documents share the same target audience demographic.

### Input

Each instance consists of two documents, represented by their textual content:
- **Document A**: A title and body text from the SHELF corpus
- **Document B**: A second title and body text from the SHELF corpus

Models receive the concatenation or paired encoding of these documents, depending on the model architecture.

### Output

A binary prediction:
- **1 (Positive)**: The documents share the same audience (including both being null/general audience)
- **0 (Negative)**: The documents have different audiences

### Formal Definition

```
f: (doc_a, doc_b) -> {0, 1}
where f(doc_a, doc_b) = 1 iff audience(doc_a) = audience(doc_b)
```

**Null Handling**: Two documents with `null` audience are considered to have the same audience (both are general-purpose documents with no specific demographic target).

## Dataset Construction

### Pair Generation Strategy

Pairs are constructed using a balanced sampling approach with special handling for null audiences:

**Positive Pairs (label = 1)**:
- Sample two distinct documents with the same target audience
- Includes null-null pairs (both documents have no specific audience)
- Documents are sampled to represent the distribution of audiences in the corpus

**Negative Pairs (label = 0)**:
- Sample two documents with different target audiences
- Can include null vs. non-null pairs (general vs. specific audience)
- Random sampling ensures negative pairs span diverse audience combinations

### Null Audience Handling

A key design decision: **null is treated as a valid audience class**.

**Rationale**:
- Approximately 30% of SHELF documents have null audience (general-purpose content)
- Two documents with null audience share a meaningful property: both are written for general readership
- Excluding null-null pairs would bias the task toward specialized audiences
- Real-world applications must handle both targeted and general-purpose content

**Implications**:
- Null-null pairs are **positive** examples (label = 1)
- Null vs. non-null pairs are **negative** examples (label = 0)
- Models must recognize general-purpose writing style as distinct from audience-targeted writing

### Balancing Strategy

To prevent models from exploiting class imbalance:
- Target distribution: 50% positive pairs, 50% negative pairs
- Similar to GLUE tasks (MRPC and QQP have 37% positive)
- Stratified splits ensure train/dev/test maintain similar balance

### Statistics

| Split | Pairs | Positive % | Negative % |
|-------|-------|-----------|-----------|
| Train | 20,000 | ~50% | ~50% |
| Validation | 4,000 | ~50% | ~50% |
| Test | 4,000 | ~50% | ~50% |

### Label Space

**Binary Classification**:
- `0`: Different audiences (negative pair)
- `1`: Same audience (positive pair)

**Audience Types** (25 total, including null):

**Age Groups**:
- Children, Adolescents, Young adults, Adults, Older adults

**Educational Levels**:
- Students, Graduate students, Researchers, Scholars

**Professional Roles**:
- Professionals, Practitioners, Specialists, Experts
- Scientists, Engineers, Physicians, Lawyers, Educators
- Business professionals, Policy makers

**General Audiences**:
- General public, Beginners, Non-specialists, Lay readers

**Null Class**:
- `null` (no specific target audience, ~30% of corpus)

### Data Format

```json
{
  "id": "pair_000123",
  "doc_a_id": "20251211_030155_abcd1234",
  "doc_a_title": "Introduction to Constitutional Law",
  "doc_a_body": "This primer introduces the fundamental principles...",
  "doc_b_id": "20251211_030155_efgh5678",
  "doc_b_title": "The Physics of Everyday Objects",
  "doc_b_body": "Have you ever wondered why objects fall...",
  "label": 1,
  "label_field": "audience"
}
```

In this example, both documents might be written for "General public" or "Lay readers", resulting in `label: 1` despite different subjects.

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

**F1 Score**: The harmonic mean of precision and recall, computed for the positive class (same audience).

```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

Where:
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)

F1 is the primary metric because it balances precision and recall, which is especially important when audience signals may be subtle. This follows the GLUE benchmark convention for MRPC and QQP tasks.

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

## What This Tests: Audience Understanding

The audience pair classification task evaluates several model capabilities:

### 1. Sociolinguistic Abstraction

Can the model recognize that two documents about completely different topics (e.g., "Constitutional Law for Beginners" vs. "Basic Chemistry Explained") belong to the same audience category (e.g., "Beginners" or "Lay readers")?

This requires detecting:
- Vocabulary complexity level (everyday words vs. technical jargon)
- Explanatory depth (concepts explained from first principles vs. assumed known)
- Mode of address (direct "you" vs. impersonal third person)
- Sentence complexity (simple vs. complex syntactic structures)

### 2. Null Audience Recognition

Can the model distinguish documents written for general readership (null) from those targeting specific demographics?

General-purpose documents typically:
- Use moderate vocabulary (neither simplified nor specialized)
- Assume educated adult readership
- Avoid both oversimplification and excessive technicality
- Use professional/formal register

### 3. Cross-Document Reasoning

Unlike single-document classification, models must:
- Process two documents jointly
- Compare readability and complexity levels
- Focus on sociolinguistic features rather than content similarity
- Ignore topic differences while detecting audience similarity

### 4. Fine-Grained Audience Distinctions

The 24 non-null audience types include overlapping categories that require nuanced understanding:
- **Students** vs. **Graduate students** vs. **Researchers**
- **Professionals** vs. **Specialists** vs. **Experts**
- **General public** vs. **Non-specialists** vs. **Lay readers**

Models must detect subtle differences in:
- Assumed educational background
- Prior domain knowledge
- Familiarity with specialized terminology
- Expected reading proficiency

### 5. Robustness to Topic Variation

Documents with the same audience may discuss:
- Completely different subjects (law, science, arts)
- Different LCC classes (all 21 classes appear with all audiences)
- Different forms (textbooks, lectures, blogs, etc.)

Models must learn audience features that generalize across content domains.

## Baselines

| Model | F1 | Accuracy | Average | Notes |
|-------|-----|----------|---------|-------|
| Random | 50.0 | 50.0 | 50.0 | Lower bound (random guessing) |
| Majority Class | 0.0 | 50.0 | 25.0 | Always predicts negative |
| TF-IDF + Cosine | TBD | TBD | TBD | Similarity threshold baseline |
| TF-IDF + LR | TBD | TBD | TBD | Logistic regression on concatenated TF-IDF vectors |
| Readability Features + LR | TBD | TBD | TBD | Flesch-Kincaid, SMOG, vocabulary complexity |
| SBERT Cosine | TBD | TBD | TBD | Sentence-BERT embeddings with cosine similarity |
| BERT Cross-Encoder | TBD | TBD | TBD | Fine-tuned BERT with [CLS] token classification |
| Human Agreement | N/A | N/A | N/A | Inter-annotator agreement (ground truth from generation) |

**Expected Performance**:
- Audience pair classification is likely **harder** than LCC or form pair classification
- Audience signals are more subtle than subject matter or genre
- Null handling adds complexity (must recognize "general-purpose" as a category)
- Fine-grained audience distinctions (Students vs. Graduate students) may cause confusion

## Relationship to GLUE-Style Tasks

### Comparison to MRPC and QQP

| Aspect | MRPC | QQP | SHELF Audience Pairs |
|--------|------|-----|----------------------|
| Domain | News paraphrases | Community Q&A | Library catalog documents |
| Label | Binary (paraphrase) | Binary (duplicate) | Binary (same audience) |
| Avg Length | ~20 words/sentence | ~10 words/question | ~50-200 words/document |
| Positive % | 37% | 37% | ~50% |
| Similarity Type | Semantic equivalence | Question equivalence | Audience equivalence |
| Metrics | F1, Accuracy | F1, Accuracy | F1, Accuracy |
| Null Handling | N/A | N/A | Null as valid class |

### Key Differences

**1. Similarity Definition**:
- MRPC/QQP: Direct semantic equivalence (same meaning)
- Audience Pairs: Sociolinguistic equivalence (same reader level, different content)

**2. Signal Sparsity**:
- MRPC/QQP: Rich lexical overlap signals
- Audience Pairs: Subtle readability and formality markers

**3. Null Values**:
- MRPC/QQP: No null handling required
- Audience Pairs: 30% of documents have null audience (general-purpose)

**4. Document Length**:
- MRPC/QQP: Short texts (10-20 words)
- Audience Pairs: Longer documents (25-500+ words)

**5. Cross-Domain Generalization**:
- MRPC/QQP: Same domain for both texts
- Audience Pairs: Documents can be from completely different domains (law, science, arts)

## Relationship to Readability Assessment

Audience pair classification is closely related to readability assessment but differs in key ways:

| Aspect | Readability Assessment | Audience Pair Classification |
|--------|------------------------|------------------------------|
| Task | Predict reading grade level | Predict if two documents share audience |
| Output | Continuous score or grade level | Binary same/different |
| Focus | Absolute complexity | Relative similarity |
| Features | Sentence length, syllable count | Comparative readability patterns |
| Datasets | OneStopEnglish, NewsELA | SHELF (synthetic, audience-labeled) |

**Why Pair Classification?**
- Tests comparative judgment (are these similar?) rather than absolute scaling
- More robust to calibration differences across models
- Mirrors real-world tasks (clustering, deduplication)
- Reduces annotation burden (relative judgments easier than absolute scores)

## Related Work

### Similar Tasks in Other Benchmarks

**GLUE (Wang et al., 2019)**:
- MRPC: Microsoft Research Paraphrase Corpus
- QQP: Quora Question Pairs
- STS-B: Semantic Textual Similarity (regression, not classification)

**MTEB (Muennighoff et al., 2023)**:
- Includes pair classification in the "PairClassification" task category
- Tasks: SprintDuplicateQuestions, TwitterSemEval2015, TwitterURLCorpus
- No readability or audience-focused tasks

**Readability Benchmarks**:
- OneStopEnglish: Articles at elementary, intermediate, advanced levels
- NewsELA: News articles rewritten for different grade levels
- CommonLit Readability Prize: Predict reading ease scores

**Formality Detection**:
- GYAFC: Grammarly's Yahoo Answers Formality Corpus (binary formal/informal)
- X-FORMAL: Multilingual formality dataset
- See audience_register_classification.md for full references

### Relevant Literature

**Paraphrase and Semantic Similarity**:
1. Dolan & Brockett (2005): "Automatically Constructing a Corpus of Sentential Paraphrases"
2. Agirre et al. (2012): "SemEval-2012 Task 6: A Pilot on Semantic Textual Similarity"

**Readability and Audience Detection**:
1. [Supervised and Unsupervised Neural Approaches to Text Readability](https://direct.mit.edu/coli/article/47/1/141/97334) (Computational Linguistics, MIT Press)
2. [Exploring the Effectiveness of Shallow and L2 Learner-Suitable Textual Features for Readability](https://www.mdpi.com/2076-3417/14/17/7997) (MDPI 2024)
3. [Readability prediction: How many features are necessary?](https://projecteuclid.org/journals/annals-of-applied-statistics/volume-18/issue-2/Readability-prediction-How-many-features-are-necessary/10.1214/23-AOAS1820.short) (Annals of Applied Statistics 2024)

**Library Science**:
1. [Library of Congress Demographic Group Terms Manual](https://loc.gov/aba/publications/FreeLCDGT/Introduction-to-LCDGT.pdf) (LC, February 2024)
2. [What are the Library of Congress Demographic Group Terms?](https://acrl.ala.org/anss/index.php/publications/cataloging-qa/what-are-the-library-of-congress-demographic-group-terms-and-how-are-they-used/)

## Implementation Notes

### Running Evaluation

```bash
# Evaluate Audience Pair Classification
shelf evaluate --task same_audience --model <model_path>

# Compare with other pair tasks
shelf evaluate --task pair_classification --model <model_path>
```

### Submission Format

Predictions should be submitted as a JSON file:

```json
{
  "task": "same_audience",
  "predictions": [
    {
      "id": "pair_000123",
      "prediction": 1,  // 0 or 1
      "confidence": 0.73  // optional
    },
    ...
  ]
}
```

### Training Tips

**1. Handle Null Audiences**:
- Treat null as a valid class (don't skip null-null pairs)
- Ensure model learns to recognize general-purpose writing

**2. Data Augmentation**:
- Swap document order to ensure symmetric learning
- Both (doc_a, doc_b) and (doc_b, doc_a) should yield same prediction

**3. Feature Engineering** (for non-neural baselines):
- Compute readability scores (Flesch-Kincaid, SMOG, ARI)
- Measure vocabulary complexity (word frequency, syllable counts)
- Analyze POS distributions (pronouns, passives, hedging)
- Compare sentence length and syntactic complexity

**4. Hard Negative Mining**:
- Include pairs from closely related audiences (e.g., "Students" vs. "Graduate students")
- Sample null vs. non-null pairs to force distinction

**5. Cross-Encoder vs. Bi-Encoder**:
- Cross-encoders (BERT with concatenation) achieve higher accuracy but slower inference
- Bi-encoders (SBERT-style) enable efficient retrieval but may have lower accuracy
- Audience detection may favor cross-encoders due to subtle comparative signals

**6. Multi-Task Learning**:
- Joint training with single-document audience classification
- Shared representations between audience and register tasks
- Auxiliary tasks: readability prediction, formality detection

### Expected Challenges

**1. Overlapping Audiences**:
- "Researchers" vs. "Scholars" vs. "Graduate students"
- "Professionals" vs. "Specialists" vs. "Experts"
- "General public" vs. "Non-specialists" vs. "Lay readers"

These fine-grained distinctions may cause systematic confusion.

**2. Null Handling**:
- Models must learn that null is a valid category
- General-purpose writing has distinct features (moderate complexity)
- Null-null pairs should be positive, null-X pairs negative

**3. Topic Independence**:
- Same audience can appear with any topic or subject
- Model must ignore content similarity and focus on readability
- Cross-domain generalization is critical

**4. Length Variation**:
- Documents range from 10 to 4000+ words
- Readability features may vary with length
- Short documents provide fewer signals

**5. Subtle Signals**:
- Audience features are more subtle than topic or genre
- Requires sensitivity to vocabulary, syntax, and discourse patterns
- May require more training data than LCC or form pair tasks

## References

1. [GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding](https://openreview.net/pdf?id=rJ4km2R5t7) (Wang et al., 2019)
2. [GLUE Benchmark Official Website](https://gluebenchmark.com/)
3. [Supervised and Unsupervised Neural Approaches to Text Readability](https://direct.mit.edu/coli/article/47/1/141/97334) (Computational Linguistics, MIT Press)
4. [Exploring the Effectiveness of Shallow and L2 Learner-Suitable Textual Features for Readability](https://www.mdpi.com/2076-3417/14/17/7997) (MDPI 2024)
5. [Library of Congress Demographic Group Terms Manual](https://loc.gov/aba/publications/FreeLCDGT/Introduction-to-LCDGT.pdf) (LC, February 2024)
6. [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084) (Reimers & Gurevych, 2019)
7. [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) (Devlin et al., 2019)
8. [The Advantages of the Matthews Correlation Coefficient (MCC) over F1 Score](https://link.springer.com/article/10.1186/s12864-019-6413-7)
9. [MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316) (Muennighoff et al., 2023)
10. [Library of Congress Demographic Group Categories](https://www.loc.gov/standards/valuelist/lcdgt.html)

---
*Last updated: 2025-12-12*
