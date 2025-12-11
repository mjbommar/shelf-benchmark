# Task: LCGFT Genre/Form Classification

> **Task Type**: classification
> **Difficulty**: hard
> **Primary Metric**: Macro-F1

## Overview

This task evaluates a model's ability to classify documents according to the Library of Congress Genre/Form Terms (LCGFT), a controlled vocabulary that describes what a document *is* rather than what it is *about*. LCGFT classification is fundamental to library cataloging, information retrieval, and document organization systems.

The task presents two complementary classification challenges: (1) fine-grained **form classification** across 133 distinct genre/form terms (e.g., "Treaties", "Sermons", "Podcasts", "Maps"), and (2) coarser **category classification** across 14 high-level categories (e.g., "Literature", "Law materials", "Religious materials"). The hierarchical relationship between forms and categories provides an opportunity to evaluate both fine-grained discrimination and structural understanding of genre taxonomies.

## Task Definition

### Input
A document consisting of:
- **Title**: A short descriptive title (typically 5-15 words)
- **Body**: The main document text (ranging from 25 to 500+ words)

Documents span diverse genres including legal materials, creative works, informational content, educational materials, religious texts, and ephemera.

### Output

#### Variant 1: Form Classification
A single label from 133 possible LCGFT form terms.

#### Variant 2: Category Classification
A single label from 14 possible LCGFT categories.

#### Variant 3: Joint Hierarchical Classification
Both a form label and its corresponding category, where the model must predict the hierarchical relationship correctly.

### Formal Definition

**Form Classification:**
Given document *d* = (title, body), predict form label *f* ∈ F where |F| = 133.

**Category Classification:**
Given document *d* = (title, body), predict category label *c* ∈ C where |C| = 14.

**Hierarchical Classification:**
Given document *d*, predict (*c*, *f*) where *f* ∈ forms(*c*), and forms(*c*) denotes the set of forms belonging to category *c*.

## Dataset

### Source
Documents are synthetically generated using GPT-based language models, controlled by metadata templates that specify:
- LCC classification (21 main subject classes)
- LCGFT category and form
- Topic keywords (1-5 domain-specific subjects)
- Target length (tiny: 25-50, brief: 50-100, short: 100-200, medium: 200-300, long: 300-500 words)
- Register/tone (academic, professional, creative, technical)
- Optional audience and geographic focus

This generation approach ensures balanced coverage across all 133 forms and 14 categories while maintaining realistic document characteristics and genre-appropriate language.

### Statistics
| Split | Documents | Forms | Categories | Avg Length |
|-------|-----------|-------|------------|------------|
| Train | TBD | 133 | 14 | ~150 words |
| Dev   | TBD | 133 | 14 | ~150 words |
| Test  | TBD | 133 | 14 | ~150 words |

### Label Space

#### Categories (14 classes)
1. **Informational works**: Reference materials, reports, data (22 forms)
2. **Law materials**: Legal documents and resources (10 forms)
3. **Instructional and educational works**: Teaching and learning materials (13 forms)
4. **Literature**: Creative fiction and drama (14 forms)
5. **Discursive works**: Commentary, analysis, debate (11 forms)
6. **Creative nonfiction**: Memoirs, essays, journalism (9 forms)
7. **Sound recordings**: Audio content (8 forms)
8. **Visual works**: Images, films, video (10 forms)
9. **Music**: Musical compositions and performances (9 forms)
10. **Religious materials**: Sacred and theological works (7 forms)
11. **Ephemera**: Temporary or promotional materials (9 forms)
12. **Commemorative works**: Memorial and celebratory works (7 forms)
13. **Cartographic materials**: Maps and geographic representations (8 forms)
14. **Recreational works**: Games, puzzles, humor (6 forms)

#### Forms (133 unique classes)
The complete list includes diverse forms such as:
- **Legal**: Treaties, Constitutions, Court decisions, Legal briefs, Statutes
- **Literary**: Novels, Poetry, Drama, Short stories, Satire, Science fiction
- **Informational**: Academic theses, Technical reports, Statistics, Databases
- **Educational**: Textbooks, Tutorials, Handbooks, Study guides
- **Media**: Podcasts, Motion pictures, Photographs, Music recordings
- **Religious**: Sermons, Prayers, Sacred works, Liturgical texts
- **Ephemeral**: Broadsides, Posters, Pamphlets, Menus

*Note*: Some forms appear in multiple categories (e.g., "Lectures" in Instructional works, Discursive works, and Sound recordings), creating subtle ambiguities that require contextual understanding.

### Data Format
```json
{
  "id": "20251211_030155_57dfc238",
  "title": "Quarterly Agribusiness Risk Memo on Emerging Externalities",
  "body": "In light of climate change, migratory wildlife now attends board meetings...",
  "word_count": 37,
  "lcc_code": "S",
  "lcc_name": "Agriculture",
  "lcgft_category": "Literature",
  "lcgft_form": "Satire",
  "topics": ["Climate change", "Wildlife", "Neuroscience", "Ocean conservation"],
  "audience": null,
  "geographic": [],
  "target_length": "tiny",
  "register": "professional"
}
```

## Evaluation

### Primary Metric: Macro-F1

**Definition**: The unweighted mean of per-class F1 scores across all forms (or categories). For each class *i*, compute F1*ᵢ* = 2·(precision*ᵢ*·recall*ᵢ*)/(precision*ᵢ*+recall*ᵢ*), then average: Macro-F1 = (1/|C|)·Σ F1*ᵢ*.

**Rationale**: With 133 fine-grained classes, the dataset exhibits significant class diversity and potential imbalance. Macro-F1 treats all forms equally regardless of frequency, ensuring that model performance on rare forms (e.g., "Festschriften", "Globes") is weighted equally with common forms (e.g., "Essays", "Reports"). This is critical for library cataloging applications where accurate classification of rare genres is as important as frequent ones.

Unlike micro-F1 or accuracy, macro-F1 prevents dominant classes from masking poor performance on minority classes. In imbalanced many-class problems, macro-F1 highlights weaknesses across the label space and better reflects the model's ability to discriminate fine-grained distinctions.

### Secondary Metrics

1. **Weighted-F1**: Per-class F1 scores weighted by support (number of instances per class). Provides a balanced view between micro and macro averaging, accounting for actual class distribution in the test set.

2. **Accuracy**: Overall proportion of correct predictions. Less informative for imbalanced distributions but useful as a simple baseline reference.

3. **Per-Category Accuracy**: For hierarchical evaluation, measures whether predictions fall within the correct category even if the specific form is wrong. Assesses whether errors are "close" in the taxonomy.

4. **Hierarchical Precision/Recall**: For joint predictions, measures consistency between predicted forms and categories. A prediction is partially correct if the category matches but the form is wrong.

5. **Confusion Matrix Analysis**: Per-class precision and recall, identifying which forms are systematically confused (e.g., "Novels" vs. "Short stories", "Maps" vs. "Atlases").

### Evaluation Protocol

**Form Classification (Primary)**:
- Models predict one of 133 form labels
- Compute macro-F1, weighted-F1, and accuracy
- Report per-class F1 for all forms with support > 10 in test set
- Generate confusion matrices for frequently-confused form pairs

**Category Classification (Secondary)**:
- Models predict one of 14 category labels
- Compute macro-F1 and accuracy
- Serves as an easier baseline to contextualize form classification difficulty

**Hierarchical Consistency (Optional)**:
- Models predict both category and form
- Evaluate: (a) category accuracy, (b) form accuracy, (c) joint accuracy
- Measure whether form predictions are consistent with category predictions
- Compare hierarchical models vs. independent classifiers

**Robustness Evaluation**:
- Stratify performance by document length (tiny/brief/short/medium/long)
- Analyze performance by register (academic/professional/creative/technical)
- Evaluate cross-subject generalization (train on subset of LCC classes, test on others)

## Baselines

| Model | Form Macro-F1 | Category Macro-F1 | Notes |
|-------|---------------|-------------------|-------|
| Random | 0.015 | 0.071 | Uniform random selection |
| Majority Class | 0.001 | 0.014 | Always predict most frequent class |
| TF-IDF + Logistic Regression | TBD | TBD | Bag-of-words baseline |
| FastText | TBD | TBD | Shallow neural baseline |
| BERT-base | TBD | TBD | Pretrained transformer |
| RoBERTa-large | TBD | TBD | Larger pretrained model |
| Hierarchical BERT | TBD | TBD | Joint category+form prediction |
| GPT-4 (zero-shot) | TBD | TBD | With label descriptions |
| Human Expert | TBD | TBD | Professional cataloger |

*Note*: Random baseline for 133 classes: 1/133 ≈ 0.0075 accuracy. Expected macro-F1 ≈ 0.015 due to random precision/recall.

## Related Work

### Similar Tasks in Other Benchmarks

**GLUE/SuperGLUE**: These benchmarks focus on sentence-level understanding tasks with typically 2-5 classes. LCGFT form classification is significantly more challenging with 133 fine-grained classes requiring document-level understanding.

**AG News, DBpedia, Yahoo Answers**: Topic classification benchmarks with 4-14 classes. AG News has 4 news categories, DBpedia has 14 ontology classes, Yahoo Answers has 10 topic categories. LCGFT's 133 forms represents an order of magnitude more classes and focuses on *genre* (document type) rather than *topic* (subject matter).

**20 Newsgroups, Reuters-21578**: Classic benchmarks with 20 and 118 classes respectively. Reuters-21578 is most comparable in scale to LCGFT form classification. However, Reuters focuses on news topic classification while LCGFT classifies genre/form across diverse document types.

**MTEB (Massive Text Embedding Benchmark)**: Includes classification tasks but primarily focuses on embedding quality across retrieval, clustering, and pair classification. LCGFT provides a complementary genre classification challenge.

**PaperNet**: Academic paper classification with 20 fine-grained CS classes. Similar difficulty level but domain-specific. PaperNet reported <80% accuracy, demonstrating challenges of fine-grained classification.

### Hierarchical and Multi-Class Classification

**Hierarchical Text Classification**: LCGFT's category→form structure resembles hierarchical classification tasks in patent classification (IPC codes), product categorization (Amazon products), and academic taxonomies (arXiv categories). The key distinction is enforcing hierarchical consistency: predicted forms must belong to predicted categories.

**CLEF-IP, BioASQ**: Domain-specific hierarchical classification tasks in patents and biomedical literature. These demonstrate the value of exploiting hierarchical structure for improved fine-grained classification.

### Relevant Literature

**Genre Classification**: Research on book genre classification (Novels, Drama, Poetry) using stylistic features shows that genre is often more difficult than topic classification due to subtle linguistic patterns rather than keyword-based differences. LCGFT extends this to a broader range of functional genres (Treaties, Maps, Sermons).

**Library Science Applications**: Automated cataloging using LCSH (Library of Congress Subject Headings) and Dewey Decimal classification. LCGFT complements subject classification by categorizing document *form* rather than subject *topic*, requiring different discriminative features.

**Fine-Grained Classification**: Recent work on fine-grained emotion classification (GoEmotions with 27 emotions), intent classification (100+ intents), and entity typing (hundreds of types) demonstrates challenges similar to LCGFT: class imbalance, subtle distinctions between classes, and the need for contextual understanding beyond surface keywords.

**Document-Level Understanding**: Genre classification requires document-level coherence and structure understanding (e.g., distinguishing "Academic theses" from "Essays" based on structure and purpose) rather than sentence-level semantics, aligning with recent trends toward longer-context models.

## Implementation Notes

### Running Evaluation
```bash
# Form classification (133 classes)
shelf evaluate --task lcgft_form --model path/to/model

# Category classification (14 classes)
shelf evaluate --task lcgft_category --model path/to/model

# Hierarchical joint evaluation
shelf evaluate --task lcgft_hierarchical --model path/to/model

# With detailed per-class metrics
shelf evaluate --task lcgft_form --model path/to/model --detailed
```

### Submission Format
```json
{
  "task": "lcgft_form",
  "predictions": [
    {
      "id": "20251211_030155_57dfc238",
      "predicted_form": "Satire",
      "confidence": 0.92
    }
  ]
}
```

For hierarchical evaluation:
```json
{
  "task": "lcgft_hierarchical",
  "predictions": [
    {
      "id": "20251211_030155_57dfc238",
      "predicted_category": "Literature",
      "predicted_form": "Satire",
      "confidence": 0.92
    }
  ]
}
```

### Implementation Recommendations

1. **Handle Label Ambiguity**: Nine forms appear in multiple categories. Models should use contextual cues to resolve category membership.

2. **Exploit Hierarchy**: Consider two-stage models (predict category → predict form within category) or joint models with hierarchical loss functions.

3. **Address Imbalance**: With 133 classes and synthetic data, implement class balancing through weighted loss, oversampling rare forms, or focal loss.

4. **Feature Engineering**: Genre/form classification benefits from structural features (document length, section markers, citation patterns) beyond content semantics.

5. **Prompt Engineering for LLMs**: Zero-shot and few-shot evaluation with large language models should include:
   - Clear definitions of LCGFT (form vs. topic distinction)
   - Category and form label lists with definitions
   - Example documents for each form

## Challenges and Open Questions

### Multi-Label Extension
While this task is defined as single-label classification, real library documents often receive multiple LCGFT terms. Future work could extend to multi-label classification where documents may have 2-5 forms (e.g., a document that is both a "Conference paper" and a "Technical report").

### Cross-Domain Generalization
How well do models trained on documents from certain LCC classes (e.g., Science, Law) generalize to documents from other classes (e.g., Literature, Music)? Evaluating cross-domain transfer reveals whether models learn genre-specific linguistic patterns vs. subject-specific keywords.

### Form vs. Topic Disambiguation
A key challenge is distinguishing form from topic. For example, a document *about* sermons (topic) should not be classified as "Sermons" (form) unless it *is* a sermon. Models must learn functional and structural properties rather than just topical keywords.

### Hierarchical Consistency
Should models be penalized for predicting the correct form but wrong category (if a form appears in multiple categories)? How to balance form-level accuracy vs. hierarchical consistency?

### Rare Form Performance
With 133 forms, many will have limited training examples. Can models achieve reasonable performance on rare forms like "Festschriften" or "Nautical charts" through transfer learning from similar forms or category-level knowledge?

## References

1. [Library of Congress Genre/Form Terms (LCGFT)](https://www.loc.gov/aba/publications/FreeLCGFT/freelcgft.html) - Official LC documentation
2. [Genre/Form Headings at the Library of Congress](https://www.loc.gov/catdir/cpso/genreformgeneral.html) - Background and guidelines
3. [Micro, Macro & Weighted Averages of F1 Score](https://towardsdatascience.com/micro-macro-weighted-averages-of-f1-score-clearly-explained-b603420b292f/) - Metric explanation
4. [Understanding Macro F1 Score in Multi-Class Classification](https://medium.com/@sushma.mullamuri420/understanding-the-macro-f1-score-in-multi-class-classification-21ca00c200da) - Evaluation metrics
5. [PaperNet: Fine-Grained Paper Classification](https://www.mdpi.com/2076-3417/12/9/4554) - Comparable fine-grained classification task
6. [Deep Learning Based Text Classification: A Comprehensive Review](https://arxiv.org/pdf/2004.03705) - Survey of classification methods
7. [Genre Identification and the Compositional Effect of Genre in Literature](https://aclweb.org/anthology/C18-1167) - Academic work on genre classification
8. [A Thorough Benchmark of Automatic Text Classification](https://arxiv.org/html/2504.01930v1) - 2025 benchmark study with LLMs

---
*Last updated: December 10, 2025*
