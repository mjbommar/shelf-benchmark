# Task: Register Pair Classification

> **Task Type**: pair_classification
> **Difficulty**: hard
> **Primary Metric**: F1 Score

## Overview

The Register Pair Classification task evaluates a model's ability to determine whether two documents share the same writing style or register. Unlike Same-LCC and Same-Form pair classification tasks that focus on subject matter and genre, this task tests the model's sensitivity to **stylistic features** such as formality level, vocabulary choice, sentence structure, and tone.

This task is particularly challenging because:
1. **Style is orthogonal to content**: Documents about the same topic can be written in vastly different styles
2. **Subtle linguistic differences**: Distinguishing "professional" from "formal" or "conversational" from "casual" requires fine-grained understanding
3. **Multi-dimensional features**: Register encompasses vocabulary, syntax, discourse markers, hedging, and more

The task mirrors real-world applications such as authorship attribution, style transfer evaluation, genre detection, and content moderation systems that need to identify tone violations.

## Task Definition

### Input

Each instance consists of two documents, represented by their textual content:
- **Document A**: A title and body text from the SHELF corpus
- **Document B**: A second title and body text from the SHELF corpus

Models receive the concatenation or paired encoding of these documents, depending on the model architecture.

### Output

A binary prediction:
- **1 (Positive)**: The documents share the same writing register
- **0 (Negative)**: The documents have different writing registers

### Formal Definition

```
f: (doc_a, doc_b) -> {0, 1}
where f(doc_a, doc_b) = 1 iff register(doc_a) = register(doc_b)
```

## Dataset Construction

### Register Categories

SHELF includes 8 distinct writing registers:

| Register | Description | Example Markers |
|----------|-------------|-----------------|
| **casual** | Informal, conversational | Contractions, colloquialisms, first-person |
| **conversational** | Friendly but clear | Personal pronouns, active voice, accessible vocabulary |
| **professional** | Standard business tone | Clear structure, neutral vocabulary, directive language |
| **formal** | Official, ceremonial | Passive voice, nominalizations, complex sentences |
| **academic** | Scholarly, precise | Citations, hedging ("may", "suggests"), technical terms |
| **technical** | Specialized, expert-level | Jargon, precise terminology, domain expertise assumed |
| **journalistic** | News style, factual | Inverted pyramid, attribution, third-person |
| **creative** | Literary, expressive | Metaphor, imagery, varied syntax, artistic language |

### Pair Generation Strategy

Pairs are constructed using balanced sampling:

**Positive Pairs (label = 1)**:
- Sample two distinct documents with the same register
- For example: two "academic" documents, or two "casual" documents
- Documents are sampled to represent the distribution of registers in the corpus

**Negative Pairs (label = 0)**:
- Sample two documents with different registers
- Random sampling ensures negative pairs span diverse register combinations
- Examples: academic + casual, formal + creative, professional + journalistic

### Balancing Strategy

To prevent models from exploiting class imbalance:
- Target distribution: 50% positive pairs, 50% negative pairs
- Stratified splits ensure train/dev/test maintain similar balance
- Random seed (42) ensures reproducibility

### Statistics

| Split | Pairs | Positive % | Negative % |
|-------|-------|-----------|-----------|
| Train | 20,000 | ~50% | ~50% |
| Validation | 4,000 | ~50% | ~50% |
| Test | 4,000 | ~50% | ~50% |

### Label Space

**Binary Classification**:
- `0`: Different registers (negative pair)
- `1`: Same register (positive pair)

### Data Format

```json
{
  "id": "pair_12345",
  "doc_a_id": "20251211_030155_57dfc238",
  "doc_a_title": "Quarterly Risk Assessment in Agricultural Markets",
  "doc_a_body": "This report analyzes emerging externalities in the agribusiness sector...",
  "doc_b_id": "20251211_030155_f3221f5a",
  "doc_b_title": "Strategic Framework for Supply Chain Optimization",
  "doc_b_body": "The following document outlines key performance indicators for operational efficiency...",
  "label": 1,  // Both are "professional" register
  "label_field": "register"
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

**F1 Score**: The harmonic mean of precision and recall, computed for the positive class (same register).

```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

Where:
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)

F1 is the primary metric because it balances precision and recall, which is especially important when positive and negative pairs may have subtle stylistic differences.

### Secondary Metrics

**Accuracy**: Fraction of correctly classified pairs
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Average Metric**: Following GLUE, we report the unweighted average of F1 and Accuracy
```
Average = (F1 + Accuracy) / 2
```

**Matthews Correlation Coefficient (MCC)**: A more robust metric for binary classification
```
MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
```

MCC is reported as a diagnostic metric to detect overoptimistic results.

### Evaluation Protocol

1. Generate predictions for all pairs in the test set
2. Compute TP, TN, FP, FN from predictions vs. ground truth
3. Calculate F1, Accuracy, and MCC
4. Report Average of F1 and Accuracy as primary score
5. Statistical significance testing using bootstrap resampling (1000 iterations)

### Per-Register Analysis

In addition to overall metrics, we recommend reporting:
- **Confusion matrix** by register pair type
- **Per-register precision and recall** to identify which registers are easiest/hardest to distinguish
- **Hard negative analysis**: Which register pairs are most confusable?

Example hard pairs:
- Casual vs. Conversational (both informal)
- Professional vs. Formal (both structured)
- Academic vs. Technical (both specialized)

## Why This Task is Harder Than LCC/Form Pair Classification

### 1. Style is Content-Independent

**LCC and Form pairs** have strong lexical signals:
- Documents about Law (K) use terms like "statute", "plaintiff", "jurisdiction"
- Maps have geographic vocabulary and spatial descriptions
- Subject matter provides clear topical clusters

**Register pairs** cut across all content:
- An academic paper about Law and an academic paper about Science share stylistic features (citations, hedging, formal vocabulary) despite different topics
- A casual blog post about Technology and a casual diary entry about History both use contractions, colloquialisms, and first-person voice

### 2. Subtle Linguistic Distinctions

The 8 registers exist on a continuum:

```
Casual <-> Conversational <-> Professional <-> Formal
                    |
              Creative/Journalistic
                    |
              Academic/Technical
```

Adjacent registers share many features:
- **Casual vs. Conversational**: Both use contractions, but conversational is more structured
- **Professional vs. Formal**: Both avoid slang, but formal uses more passive voice and nominalizations
- **Academic vs. Technical**: Both use specialized vocabulary, but academic includes more hedging and citations

Models must learn these fine-grained distinctions rather than coarse topic boundaries.

### 3. Multi-Dimensional Stylistic Features

Register encompasses multiple linguistic dimensions:

| Dimension | Casual | Formal | Academic | Technical |
|-----------|--------|--------|----------|-----------|
| **Vocabulary** | Common words | Latinate words | Domain terms + hedges | Jargon + precision |
| **Syntax** | Simple sentences | Complex sentences | Subordinate clauses | Definitions + specifications |
| **Voice** | First-person | Passive | Third-person | Impersonal |
| **Contractions** | Common | Rare | Never | Never |
| **Hedging** | Rare | Some | Frequent | Rare |
| **Citations** | Never | Rare | Always | Sometimes |

Models must integrate evidence across all dimensions to make correct predictions.

### 4. Requires Deeper Language Understanding

While LCC and Form can be predicted from:
- **Bag-of-words features** (topic keywords, genre markers)
- **Document structure** (e.g., legal documents have sections, maps have coordinates)

Register requires:
- **Syntactic parsing** (sentence complexity, voice, clause structure)
- **Pragmatic understanding** (formality, intended audience, social context)
- **Discourse analysis** (cohesion markers, text organization)
- **Lexical sophistication** (word frequency, abstractness, technicality)

## Baselines

| Model | F1 / Accuracy | MCC | Notes |
|-------|---------------|-----|-------|
| Random | 50.0 / 50.0 | 0.00 | Lower bound (random guessing) |
| Majority Class | 0.0 / 50.0 | 0.00 | Always predicts negative |
| TF-IDF + Cosine | TBD | TBD | Similarity threshold baseline |
| TF-IDF + LR | TBD | TBD | Logistic regression on concatenated TF-IDF vectors |
| SBERT Cosine | TBD | TBD | Sentence-BERT embeddings with cosine similarity |
| BERT Cross-Encoder | TBD | TBD | Fine-tuned BERT with [CLS] token classification |
| Stylometric Features + RF | TBD | TBD | Hand-crafted features (sentence length, vocabulary diversity, etc.) |
| Human Agreement | N/A | N/A | Inter-annotator agreement (ground truth from generation prompts) |

### Expected Performance

Based on stylistic analysis literature and preliminary experiments:
- **TF-IDF baselines**: 55-65% accuracy (some lexical patterns but weak)
- **BERT fine-tuned**: 70-85% accuracy (good at learning register patterns)
- **Specialized style models**: 80-90% accuracy (e.g., models pretrained on authorship tasks)
- **Human performance**: 90-95% (some register boundaries are genuinely ambiguous)

## Relationship to Other NLP Tasks

### Comparison to Authorship Attribution

| Aspect | Authorship Attribution | Register Pair Classification |
|--------|----------------------|------------------------------|
| **Signal** | Individual author style | Sociolinguistic register |
| **Stability** | Stable across contexts | Varies by context/audience |
| **Features** | Idiosyncratic patterns | Conventional style patterns |
| **Training data** | Multiple texts by same author | Documents labeled by register |

Register is more **general** than authorship: many authors can write in the same register, and one author can use multiple registers.

### Comparison to Genre Classification

| Aspect | Genre Classification | Register Classification |
|--------|---------------------|-------------------------|
| **Focus** | Document type/purpose | Writing style/tone |
| **Example** | "News article" vs. "Blog post" | "Journalistic" vs. "Casual" |
| **Overlap** | High correlation | Orthogonal |

In SHELF:
- **Genre** (LCGFT form): "Map", "Satire", "Legal brief"
- **Register**: "Technical", "Creative", "Formal"
- A **Map** can be written in **technical** register (engineering diagram) or **casual** register (hand-drawn tourist map)

### Comparison to Formality Detection

Register classification is a **generalization** of formality detection:
- Formality detection: Binary or 3-way classification (informal/neutral/formal)
- Register classification: 8-way classification with multiple dimensions beyond formality

## Related Work

### Stylometry and Register Studies

1. **Biber (1988)**: "Variation Across Speech and Writing" - foundational work on register variation
2. **Heylighen & Dewaele (1999)**: "Formality of Language: definition, measurement and behavioral determinants"
3. **Pavlick & Tetreault (2016)**: "An Empirical Analysis of Formality in Online Communication"

### Computational Stylistics

1. **Burrows (2002)**: "Delta: a Measure of Stylistic Difference and a Guide to Likely Authorship"
2. **Stamatatos (2009)**: "A Survey of Modern Authorship Attribution Methods"
3. **Argamon et al. (2009)**: "Automatically Profiling the Author of an Anonymous Text"

### Register and Formality Detection

1. **Lahiri (2015)**: "Lexical Formality in Context: Joint Modeling of Formality and Contextual Factors"
2. **Rao & Tetreault (2018)**: "Dear Sir or Madam, May I Introduce the GYAFC Dataset: Corpus, Benchmarks and Metrics for Formality Style Transfer"
3. **Brooke et al. (2010)**: "A Multi-Dimensional Bayesian Approach to Lexical Style"

### Neural Approaches to Style

1. **Ficler & Goldberg (2017)**: "Controlling Linguistic Style Aspects in Neural Language Generation"
2. **Shen et al. (2017)**: "Style Transfer from Non-Parallel Text by Cross-Alignment"
3. **Tikhonov & Yamshchikov (2018)**: "What is Wrong with Style Transfer for Texts?"

## Implementation Notes

### Running Evaluation

```bash
# Evaluate Same-Register Prediction
shelf evaluate --task same_register --model <model_path>

# Evaluate all pair classification tasks
shelf evaluate --task pair_classification --model <model_path>
```

### Submission Format

Predictions should be submitted as a JSON file:

```json
{
  "task": "same_register",
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

1. **Data Augmentation**:
   - Swap document order to ensure symmetric learning
   - Mix register pairs at training time to expose the model to all combinations

2. **Hard Negative Mining**:
   - Include pairs from similar registers (e.g., casual + conversational, professional + formal)
   - These hard negatives force the model to learn fine-grained distinctions

3. **Feature Engineering (for classical ML)**:
   - Sentence length distribution
   - Type-token ratio (vocabulary diversity)
   - POS tag frequencies
   - Passive voice ratio
   - Average word frequency/concreteness
   - Punctuation patterns

4. **Cross-Encoder vs. Bi-Encoder**:
   - **Cross-encoders** (BERT with concatenation) achieve higher accuracy but slower inference
   - **Bi-encoders** (SBERT-style) enable efficient retrieval but may have lower accuracy
   - For register classification, cross-encoders are recommended (style requires joint reasoning)

5. **Multi-Task Learning**:
   - Train jointly with single-document register classification
   - Auxiliary task: predict which register is used by each document independently
   - This helps the model learn robust register representations

6. **Regularization**:
   - Use dropout to prevent overfitting on superficial patterns
   - Early stopping based on validation F1
   - Consider adversarial training to remove topic/content leakage

### Error Analysis

When evaluating, consider:

1. **Confusion Matrix by Register Pair**:
   - Which register pairs are most confusable?
   - Are errors symmetric (A→B = B→A)?

2. **Content Leakage**:
   - Does the model rely on topic keywords instead of style?
   - Test: Evaluate on same-topic document pairs

3. **Length Bias**:
   - Do longer documents perform better?
   - Are certain registers associated with specific lengths?

4. **Qualitative Analysis**:
   - Sample misclassified pairs
   - Identify linguistic features the model missed
   - Check for annotation errors in ground truth

## References

1. [GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding](https://openreview.net/pdf?id=rJ4km2R5t7) (Wang et al., 2019)
2. [Biber's Framework of Register Variation](https://books.google.com/books?id=UJ8gAwAAQBAJ) (Biber, 1988)
3. [Formality of Language: Definition and Measurement](https://www.tandfonline.com/doi/abs/10.1080/23273798.2019.1585866)
4. [An Empirical Analysis of Formality in Online Communication](https://aclanthology.org/Q16-1005/) (Pavlick & Tetreault, 2016)
5. [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084) (Reimers & Gurevych, 2019)
6. [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) (Devlin et al., 2019)
7. [Authorship Attribution](https://www.annualreviews.org/doi/abs/10.1146/annurev-linguistics-011516-034229) (Stamatatos, 2009)
8. [Controlling Linguistic Style in Neural Language Generation](https://arxiv.org/abs/1707.02633) (Ficler & Goldberg, 2017)
9. [GYAFC Dataset for Formality Style Transfer](https://arxiv.org/abs/1803.06535) (Rao & Tetreault, 2018)
10. [Library of Congress Classification](https://www.loc.gov/catdir/cpso/lcc.html)

---
*Last updated: 2025-12-12*
