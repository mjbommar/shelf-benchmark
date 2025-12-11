# Task: Audience and Register Classification

> **Task Type**: classification (multi-class)
> **Difficulty**: medium
> **Primary Metric**: macro-F1

## Overview

This document describes two related but distinct classification tasks that evaluate a model's ability to detect linguistic variation along two key dimensions: intended audience (who the text is written for) and linguistic register (the formality/style of writing). Together, these tasks test whether models can recover sociolinguistic metadata from generated text, assessing their sensitivity to vocabulary complexity, formality markers, and discourse-level stylistic choices.

**Audience Classification** asks models to predict the target demographic group for which a document was written, using 24 classes derived from Library of Congress Demographic Group Terms (LCDGT). This tests readability detection and audience-appropriate language use.

**Register Classification** asks models to identify the writing register/tone used in a document across 8 classes (casual, conversational, professional, formal, academic, technical, journalistic, creative). This tests formality detection and style identification.

Both tasks are particularly interesting because they evaluate whether linguistic features intentionally encoded during text generation remain detectable and classifiable, serving as a test of both generation fidelity and classification robustness.

## Task Definitions

### 1. Audience Classification

#### Input
A document consisting of a title and body text, ranging from 10 to 4000 words.

#### Output
One of 24 audience categories, or null if the document has no specific target audience.

#### Label Space
The 24 audience classes are organized into conceptual groups:

**Age Groups:**
- Children
- Adolescents
- Young adults
- Adults
- Older adults

**Educational Levels:**
- Students
- Graduate students
- Researchers
- Scholars

**Professional Roles:**
- Professionals
- Practitioners
- Specialists
- Experts
- Scientists
- Engineers
- Physicians
- Lawyers
- Educators
- Business professionals
- Policy makers

**General Audiences:**
- General public
- Beginners
- Non-specialists
- Lay readers

**Null Class:**
- null (no specific target audience)

#### Formal Definition
Given a document d = (title, body), classify into audience class a ∈ A where A is the set of 24 audience labels plus null. The classification should identify which demographic group the text's vocabulary, explanatory depth, assumed prior knowledge, and mode of address are calibrated for.

### 2. Register Classification

#### Input
A document consisting of a title and body text, ranging from 10 to 4000 words.

#### Output
One of 8 writing register categories.

#### Label Space
The 8 register classes represent distinct linguistic styles:

1. **casual** - Informal, conversational, like a blog post or social media
2. **conversational** - Friendly and approachable, like talking to a colleague
3. **professional** - Clear and professional, standard business tone
4. **formal** - Formal and official, appropriate for legal or governmental contexts
5. **academic** - Scholarly and precise, with citations and hedged claims
6. **technical** - Technical and specialized, assuming domain expertise
7. **journalistic** - Clear and factual, inverted pyramid news style
8. **creative** - Expressive and literary, with vivid language and style

#### Formal Definition
Given a document d = (title, body), classify into register class r ∈ R where R = {casual, conversational, professional, formal, academic, technical, journalistic, creative}. The classification should identify the dominant linguistic register based on formality markers, vocabulary choices, syntactic complexity, and discourse structure.

## Dataset

### Source
All documents in SHELF are synthetically generated using OpenAI's GPT-5.1 model via the Responses API. During generation, each document was assigned:
- **Audience**: Sampled from the 24 LCDGT-style demographic categories (with 30% probability of null)
- **Register**: Sampled from 8 register classes with weighted distribution (professional: 25%, conversational: 15%, academic: 15%, formal: 15%, casual: 10%, technical: 10%, journalistic: 5%, creative: 5%)

The generation prompt explicitly instructed the model to calibrate vocabulary complexity, assumed prior knowledge, explanatory depth, and mode of address to match the target audience, and to write in the specified register/tone. These tasks evaluate whether those stylistic features remain detectable in the final text.

### Statistics
Exact statistics depend on dataset split configuration. Typical distribution:

| Split | Documents | Audience Distribution | Register Distribution |
|-------|-----------|----------------------|----------------------|
| Train | ~7,000 | Balanced across 24+null classes | Weighted: professional 25%, academic/formal/conversational 15% each |
| Dev   | ~1,500 | Proportional to training | Proportional to training |
| Test  | ~1,500 | Proportional to training | Proportional to training |

### Data Format
```json
{
  "id": "20251211_030155_0d645df6",
  "title": "What Los Angeles Can Learn from Asia's Smoggy Skies...",
  "text": "Los Angeles likes to think of itself as a reformed smog capital...",
  "word_count": 369,
  "labels": {
    "audience": "Lay readers",
    "register": "conversational",
    "lcc_code": "G",
    "lcgft_form": "Editorials",
    ...
  }
}
```

For Audience Classification, the target label is `labels.audience` (string or null).
For Register Classification, the target label is `labels.register` (string).

## Linguistic Features Tested

Both tasks evaluate models' sensitivity to sociolinguistic variation across multiple levels of linguistic structure:

### Lexical Features
- **Vocabulary complexity**: Technical jargon vs. everyday language
- **Word choice sophistication**: Latinate vs. Germanic roots, polysyllabic vs. simple words
- **Domain-specific terminology**: Specialized vocabulary indicating expert vs. lay audience
- **Lexical density**: Ratio of content words to function words
- **Type-token ratio**: Vocabulary richness and repetition

### Syntactic Features
- **Sentence length and complexity**: Simple vs. complex sentences
- **Passive voice frequency**: More common in formal and academic registers
- **Nominalization**: Converting verbs to nouns (formal marker)
- **Subordinate clause density**: Embedding and syntactic depth
- **Hedging constructions**: "may", "could", "appears to" (academic register)

### Formality Markers
Research shows that nouns, adjectives, articles, and prepositions are more frequent in formal styles, while pronouns, adverbs, verbs, and interjections are more frequent in informal styles. Models must learn these distributional patterns to succeed.

**Informal markers:**
- First/second person pronouns (I, you, we)
- Contractions (don't, can't, it's)
- Colloquialisms and slang
- Exclamations and interjections
- Sentence fragments
- Rhetorical questions

**Formal markers:**
- Third person and passive constructions
- Expanded forms (do not, cannot, it is)
- Latinate vocabulary
- Abstract nouns
- Complex prepositions (in accordance with, with respect to)
- Impersonal constructions

### Discourse Features
- **Explanatory depth**: How much background information is provided
- **Mode of address**: Direct address (you) vs. impersonal constructions
- **Hedging and certainty**: Academic hedging vs. assertive claims
- **Structure and organization**: Inverted pyramid (journalistic) vs. narrative arc (creative)
- **Citation and attribution**: Explicit in academic, implicit in other registers

### Pragmatic Features
- **Assumed prior knowledge**: Technical terms defined or assumed known
- **Reader engagement strategies**: Questions, imperatives, inclusive pronouns
- **Tone and stance**: Objective, persuasive, entertaining, instructional
- **Cultural references**: Age-appropriate or domain-specific allusions

## Evaluation

### Primary Metric
**Macro-F1**: Unweighted average of F1 scores across all classes. This metric is appropriate because:
1. Class distributions are imbalanced (especially for audience, with null class)
2. All classes are equally important for evaluation
3. It balances precision and recall across all categories
4. It penalizes models that ignore minority classes

### Secondary Metrics
- **Micro-F1**: Overall accuracy weighted by class frequency
- **Weighted-F1**: F1 averaged by class support
- **Per-class F1**: Detailed breakdown showing which audiences/registers are hardest to detect
- **Confusion matrices**: Reveal systematic confusions (e.g., "professional" vs. "formal")
- **Accuracy**: Simple correct/total ratio

### Evaluation Protocol
Standard multi-class classification evaluation:
1. Models predict a single class label for each document
2. Predictions are compared against gold labels
3. Metrics computed using scikit-learn classification_report
4. For audience classification, null predictions are treated as a valid class

**Special considerations:**
- Documents with null audience should not be ignored but classified as "null"
- Register classification has no null class (every document has a register)
- Both tasks use the same document pool but different target labels

## Baselines

Expected baseline performance (to be empirically validated):

| Model | Audience Macro-F1 | Register Macro-F1 | Notes |
|-------|------------------|-------------------|-------|
| Random | ~0.04 | ~0.125 | Random guessing among 25/8 classes |
| Majority class | ~0.02 | ~0.03 | Always predicting most common class |
| TF-IDF + Logistic Regression | 0.45-0.55 | 0.60-0.70 | Lexical features baseline |
| BERT-base | 0.65-0.75 | 0.75-0.85 | Contextual embeddings |
| RoBERTa-large | 0.70-0.80 | 0.80-0.90 | State-of-art encoder |
| GPT-4 (zero-shot) | 0.60-0.70 | 0.70-0.80 | Generative model without fine-tuning |
| Human (expert annotator) | 0.85-0.95 | 0.90-0.95 | Upper bound estimate |

Note: Audience classification is expected to be harder than register classification because:
1. More fine-grained distinctions (24 vs. 8 classes)
2. Audience signals may be more subtle than register markers
3. Some audiences overlap conceptually (e.g., "Researchers" vs. "Scholars")

## Related Work

### Similar Tasks in Other Benchmarks

**Readability Assessment:**
- Traditional readability formulas (Flesch-Kincaid, Dale-Chall) predict reading grade level
- OneStopEnglish corpus: Articles at elementary, intermediate, advanced levels
- NewsELA corpus: News articles rewritten for different grade levels
- CommonLit Readability Prize (Kaggle 2021): Predict reading ease

**Formality Detection:**
- GYAFC (Grammarly's Yahoo Answers Formality Corpus): Formal/informal binary classification
- X-FORMAL: Multilingual formality dataset
- FAME-MT: 11.2M translations with formality labels across 15 European languages
- Pavlick & Tetreault (2016): Formality scoring for online communication

**Register/Genre Classification:**
- Brown Corpus: 15 genre categories (news, fiction, academic, etc.)
- BNC (British National Corpus): Register and genre tags
- CORE (Common Online Register of English): Web-based register classification

**Style Transfer:**
- Formality transfer (informal→formal, formal→informal)
- Simplification (complex→simple for different reading levels)

### Relevant Literature

**Readability and Audience Detection:**
- [Supervised and Unsupervised Neural Approaches to Text Readability](https://direct.mit.edu/coli/article/47/1/141/97334/Supervised-and-Unsupervised-Neural-Approaches-to) (Computational Linguistics, MIT Press)
- [Exploring the Effectiveness of Shallow and L2 Learner-Suitable Textual Features](https://www.mdpi.com/2076-3417/14/17/7997) (MDPI 2024) - Found Random Forest achieved 74% accuracy on sentence-level simple/complex classification
- [Readability prediction: How many features are necessary?](https://projecteuclid.org/journals/annals-of-applied-statistics/volume-18/issue-2/Readability-prediction-How-many-features-are-necessary/10.1214/23-AOAS1820.short) (Annals of Applied Statistics 2024) - Analyzes up to 200 features for readability
- WeeklyReader Corpus: SVM classifier for age-based audience classification

**Formality and Register Detection:**
- [Automatic classification of documents by formality](https://www.researchgate.net/publication/224178419_Automatic_classification_of_documents_by_formality) (Sheikha & Inkpen)
- [Detecting Text Formality: A Study of Text Classification Approaches](https://aclanthology.org/2023.ranlp-1.31.pdf) (RANLP 2023) - Systematic comparison of statistical, neural, and Transformer models; Char BiLSTM outperformed Transformers for monolingual formality
- [An Empirical Analysis of Formality in Online Communication](https://cs.brown.edu/people/epavlick/papers/formality.pdf) (Pavlick & Tetreault)
- [FAME-MT Dataset: Formality Awareness Made Easy for Machine Translation](https://arxiv.org/html/2405.11942v1) (ArXiv 2024)
- [Do LLMs write like humans? Variation in grammatical and rhetorical styles](https://www.pnas.org/doi/10.1073/pnas.2422455122) (PNAS) - Shows instruction-tuned models have distinct noun-heavy style

**Stylometry and Writing Style:**
- [Detection of changes in literary writing style using N-grams](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0267590) (PLOS ONE)
- [Stylistic text classification using functional lexical features](https://www.researchgate.net/publication/220435559_Stylistic_text_classification_using_functional_lexical_features)

**LC Controlled Vocabularies:**
- [Library of Congress Demographic Group Terms Manual](https://loc.gov/aba/publications/FreeLCDGT/Introduction-to-LCDGT.pdf) (LC, February 2024)
- [What are the Library of Congress Demographic Group Terms?](https://acrl.ala.org/anss/index.php/publications/cataloging-qa/what-are-the-library-of-congress-demographic-group-terms-and-how-are-they-used/)
- [Library of Congress Demographic Group Categories](https://www.loc.gov/standards/valuelist/lcdgt.html)

**Key Finding from Literature:**
- **F-score formality measure**: Nouns, adjectives, articles, prepositions → formal; Pronouns, adverbs, verbs, interjections → informal
- **Char BiLSTM vs. Transformers**: Character-level BiLSTMs sometimes outperform Transformers for formality detection
- **Vocabulary choice**: "Perhaps the biggest style marker" (Sheikha & Inkpen 2010)
- **Neural approaches**: Transfer well across languages for readability assessment
- **POS distribution**: Most reliable signal for register classification

## Implementation Notes

### Running Evaluation

Audience Classification:
```bash
shelf evaluate --task audience_classification --model {model_path} --split test
```

Register Classification:
```bash
shelf evaluate --task register_classification --model {model_path} --split test
```

Both tasks:
```bash
shelf evaluate --task audience_classification,register_classification --model {model_path}
```

### Submission Format

Predictions should be submitted as JSONL (one prediction per line):

**Audience Classification:**
```json
{"id": "20251211_030155_0d645df6", "prediction": "Lay readers"}
{"id": "20251211_030155_06dea11a", "prediction": null}
```

**Register Classification:**
```json
{"id": "20251211_030155_0d645df6", "prediction": "conversational"}
{"id": "20251211_030155_06dea11a", "prediction": "formal"}
```

Predictions must:
- Include all test set document IDs
- Use exact label strings from the label space (case-sensitive)
- Use `null` (not "null" string) for no-audience predictions in audience task
- Contain no extra fields (id and prediction only)

### Model Development Tips

**Feature Engineering:**
- Extract POS tag distributions (formality markers)
- Compute readability scores (Flesch-Kincaid, SMOG, etc.)
- Measure vocabulary sophistication (word frequency, syllable count)
- Analyze syntactic complexity (parse tree depth, clause counts)
- Count personal pronouns, passive voice, hedging expressions

**Neural Approaches:**
- Fine-tune BERT/RoBERTa on document-level classification
- Use hierarchical models for long documents (sentence → document)
- Try character-level models for formality detection
- Experiment with multi-task learning (joint audience + register prediction)

**Prompt Engineering (for LLMs):**
- Provide clear definitions and examples of each class
- Use chain-of-thought: ask model to identify key linguistic features first
- Few-shot prompting with diverse examples from each class
- For audience: prompt to identify vocabulary level, assumed knowledge, explanations
- For register: prompt to identify formality markers, sentence structure, tone

**Error Analysis:**
- Check confusion matrices for systematic errors
- Identify challenging audience pairs (e.g., Researchers vs. Scholars)
- Analyze register confusions (casual vs. conversational, formal vs. academic)
- Examine documents with null audience - what makes them general?
- Review short vs. long documents - does length affect performance?

## References

1. [Supervised and Unsupervised Neural Approaches to Text Readability](https://direct.mit.edu/coli/article/47/1/141/97334/Supervised-and-Unsupervised-Neural-Approaches-to), Computational Linguistics, MIT Press
2. [Exploring the Effectiveness of Shallow and L2 Learner-Suitable Textual Features for Readability](https://www.mdpi.com/2076-3417/14/17/7997), MDPI Applied Sciences, 2024
3. [Readability prediction: How many features are necessary?](https://projecteuclid.org/journals/annals-of-applied-statistics/volume-18/issue-2/Readability-prediction-How-many-features-are-necessary/10.1214/23-AOAS1820.short), Annals of Applied Statistics, 2024
4. [Automatic classification of documents by formality](https://www.researchgate.net/publication/224178419_Automatic_classification_of_documents_by_formality), Sheikha & Inkpen
5. [Detecting Text Formality: A Study of Text Classification Approaches](https://aclanthology.org/2023.ranlp-1.31.pdf), RANLP 2023
6. [An Empirical Analysis of Formality in Online Communication](https://cs.brown.edu/people/epavlick/papers/formality.pdf), Pavlick & Tetreault
7. [FAME-MT Dataset: Formality Awareness Made Easy](https://arxiv.org/html/2405.11942v1), ArXiv 2024
8. [Do LLMs write like humans? Variation in grammatical and rhetorical styles](https://www.pnas.org/doi/10.1073/pnas.2422455122), PNAS
9. [Detection of changes in literary writing style using N-grams](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0267590), PLOS ONE
10. [Library of Congress Demographic Group Terms Manual](https://loc.gov/aba/publications/FreeLCDGT/Introduction-to-LCDGT.pdf), Library of Congress, February 2024
11. [What are the Library of Congress Demographic Group Terms?](https://acrl.ala.org/anss/index.php/publications/cataloging-qa/what-are-the-library-of-congress-demographic-group-terms-and-how-are-they-used/), ACRL/ANSS
12. [Library of Congress Demographic Group Categories](https://www.loc.gov/standards/valuelist/lcdgt.html), LC Network Development and MARC Standards

---
*Last updated: 2025-12-10*
