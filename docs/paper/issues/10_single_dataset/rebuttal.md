# Peer Review Response: Single Dataset Limitation

## Reviewer Concern

> "Results on a single benchmark may not generalize to other tasks or domains."

---

## Response Summary

We appreciate the reviewer's attention to generalization, which is a critical concern for benchmark papers. However, we respectfully argue that **characterizing SHELF as a "single dataset" misrepresents its structure and scope**.

SHELF is more accurately described as a **benchmark suite with massive internal diversity** achieved through systematic cross-product sampling across independent taxonomic dimensions. By this measure, SHELF contains 21-294 effective task variants (depending on granularity), making it comparable in scope to established multi-dataset benchmarks like GLUE (9 tasks) and approaching MTEB (58 datasets).

**Key evidence**:
1. **558,600 unique document configurations** (21 subjects × 133 forms × 8 registers × 25 audiences)
2. **18 distinct evaluation tasks** across classification, retrieval, clustering, and pair classification
3. **Independent taxonomic dimensions** that create genuine cross-product diversity (every subject appears with every form, unlike real corpora with strong correlations)
4. **Domain-complete coverage** via Library of Congress Classification, which spans all human knowledge, not a narrow vertical

We have prepared detailed analyses in supplementary materials and propose specific revisions to clarify SHELF's scope and positioning.

---

## Detailed Response

### 1. SHELF Is Not a "Single Dataset" in the Traditional Sense

**Traditional single dataset**:
- Narrow task (e.g., sentiment analysis on movie reviews)
- Limited label diversity (2-5 classes)
- Single domain (e.g., e-commerce, news)
- Homogeneous document types

**SHELF**:
- 18 evaluation tasks across 4 task types
- 21-133 classes per classification task
- 21 subject domains spanning all knowledge (Philosophy, Law, Science, Medicine, Technology, Fine Arts, etc.)
- 133 heterogeneous document forms (Maps, Lectures, Prayers, Jokes, Legal briefs, Satellite imagery, etc.)

**Quantitative comparison**:

| Benchmark | Datasets | Tasks | Subject Domains | Document Forms |
|-----------|----------|-------|-----------------|----------------|
| GLUE | 9 | 9 | General NLU | Varies |
| SuperGLUE | 10 | 10 | General NLU | Varies |
| MTEB | 58 | 58+ | Implicit | Dataset-dependent |
| **SHELF** | **1 (7 configs)** | **18** | **21 explicit** | **133 explicit** |

SHELF is task-comparable to GLUE/SuperGLUE (18 vs. 9-10 tasks) while offering deeper classification granularity (133 vs. typical 5-20 classes in genre datasets).

### 2. Cross-Product Diversity Creates Effective Task Multiplicity

SHELF's taxonomic dimensions are **statistically independent**, as evidenced by:
- Near-uniform LCC distribution (4.6-4.9% per class)
- Complete cross-product coverage (every subject appears with every genre category)
- Co-occurrence matrices showing independence

**What this means**:
Each (subject, form, register, audience) combination tests different capabilities. Consider:

- **Maps about Philosophy** (rare in real corpora)
- **Jokes about Law** (uncommon in real datasets)
- **Prayers about Technology** (virtually absent in natural data)
- **Satellite imagery in Academic register for Specialists** (novel combination)

Real-world corpora exhibit strong correlations:
- Medical texts → Academic register → Research papers → Scholarly audience
- Legal texts → Formal register → Legal briefs → Professional audience

**SHELF intentionally breaks these correlations** to create more comprehensive coverage than aggregating real-world datasets would provide.

**Result**: 558,600 core configurations, each testing a distinct combination of capabilities.

### 3. Domain-Complete vs. Domain-Specific

Unlike domain-specific benchmarks (FinMTEB for finance, ChemTEB for chemistry), SHELF uses **Library of Congress Classification**, the most comprehensive bibliographic taxonomy ever developed, designed to organize **all human knowledge**.

**SHELF's 21 LCC classes**:
- A: General Works
- B: Philosophy, Psychology, Religion
- C: Auxiliary Sciences of History
- D: World History
- E-F: History of Americas
- G: Geography, Anthropology, Recreation
- H: Social Sciences
- J: Political Science
- K: Law
- L: Education
- M: Music
- N: Fine Arts
- P: Language and Literature
- Q: Science
- R: Medicine
- S: Agriculture
- T: Technology
- U: Military Science
- V: Naval Science
- Z: Bibliography, Library Science

**Coverage**: Not a narrow vertical, but the full horizontal breadth of human intellectual output.

**Positioning**: SHELF is "domain-complete" rather than "domain-specific." Strong SHELF performance indicates genuine document understanding across comprehensive taxonomies, not just narrow task optimization.

### 4. Comparison to MTEB

MTEB (Massive Text Embedding Benchmark) is widely accepted as a comprehensive evaluation framework, yet it uses a fundamentally similar approach to SHELF—aggregating diverse tasks to test broad capabilities.

**Two roads to diversity**:

| Approach | MTEB | SHELF |
|----------|------|-------|
| **Strategy** | Dataset aggregation | Taxonomic cross-products |
| **Diversity source** | 58 different datasets | 558,600 configurations |
| **Coverage** | Broad task types (8) | Deep classification (133 forms) |
| **Data** | Real-world | Synthetic |
| **Advantage** | Established baselines | Controlled methodology, low contamination |

**Key insight**: MTEB achieves diversity through **aggregation** (combining many datasets). SHELF achieves diversity through **cross-products** (combining independent dimensions). Both create comprehensive evaluation; neither approach is superior.

**Complementarity**:
- MTEB: "Can embeddings generalize across diverse real-world tasks?" (breadth)
- SHELF: "Can embeddings understand comprehensive document taxonomies?" (depth)

Both are needed for complete model evaluation.

### 5. What SHELF Measures That Other Benchmarks Don't

**Unique contributions**:

1. **Comprehensive genre/form classification**
   - Most genre datasets: 5-10 classes (news, reviews, fiction)
   - SHELF: 133 forms across 14 categories
   - Tests ability to distinguish Maps from Lectures, Prayers from Jokes, Satellite imagery from Floor plans

2. **Subject-form interaction**
   - Real datasets have correlated dimensions (medical papers, legal briefs)
   - SHELF tests uncorrelated combinations (Maps about Philosophy, Jokes about Law)
   - Requires genuine understanding, not pattern matching on correlated features

3. **Authoritative taxonomic grounding**
   - Most benchmarks: Ad hoc labels or narrow domains
   - SHELF: Library of Congress taxonomies (100+ years of expert curation)
   - Comprehensive, stable, professionally maintained

4. **Multi-faceted simultaneous classification**
   - Most benchmarks: Single-label classification
   - SHELF: Simultaneous classification on 6 independent dimensions
   - Tests compositional understanding

### 6. Generalization Evidence and Future Work

**Within-benchmark generalization**:
Models trained on SHELF must already generalize to unseen:
- Subject × Form combinations (train on Science+Maps, test on Law+Maps)
- Subject × Register combinations (train on Medicine+Academic, test on Medicine+Casual)
- Multi-dimensional interactions

This is **not mere memorization**—it requires genuine understanding of independent taxonomic dimensions.

**Cross-benchmark generalization** (future work):
We acknowledge the need for validation beyond SHELF and propose:

1. **Correlation study**: Analyze correlation between SHELF scores and performance on real bibliographic data (MARC records, library catalogs)

2. **Transfer learning**: Fine-tune on SHELF, evaluate on real library classification tasks

3. **Multi-language SHELF**: Generate same taxonomic structure in Spanish, French, Chinese to test cross-lingual consistency

4. **SHELF-v2**: Use different generation methodology to validate robustness to synthetic data approach

5. **Domain-specific extensions**: SHELF-Legal, SHELF-Science to test whether general SHELF performance predicts domain-specific performance

6. **Longitudinal study**: Track model improvements on SHELF vs. MTEB over time

### 7. Acknowledged Limitations

We acknowledge legitimate limitations:

1. **Single generation methodology**: All documents generated via similar prompting approach (mitigated by using 9 different models: GPT-5.1/5.2, Gemini 2.5/3, Claude Haiku/Sonnet/Opus 4.5)

2. **Synthetic data differences**: Generated text may differ from real documents in subtle ways (future work: validation study with human experts)

3. **Bibliographic focus**: SHELF tests document classification, not general NLU (this is intentional scope, not a limitation)

4. **English-only**: Current version is monolingual (multilingual extensions planned)

**These are scope decisions, not fundamental flaws.** Every benchmark has a focus. MTEB focuses on embedding quality across tasks; SHELF focuses on document understanding across taxonomies.

---

## Proposed Revisions

To address the reviewer's concern, we propose the following revisions to the paper:

### 1. Clarify Positioning (Introduction)

**Add**:
> "While SHELF consists of a single unified dataset, it is more accurately characterized as a **benchmark suite with internal diversity**. Through systematic cross-product sampling across independent taxonomic dimensions (21 subjects × 133 forms × 8 registers × 25 audiences), SHELF generates 558,600 unique document configurations, creating 18 distinct evaluation tasks comparable in scope to established multi-task benchmarks like GLUE (9 tasks)."

### 2. Add Diversity Analysis Section

**New section**: "Internal Diversity and Effective Task Multiplicity"

Include:
- Quantification of cross-product diversity (558,600 configurations)
- Analysis of dimension independence (co-occurrence matrices)
- Comparison to multi-dataset benchmarks (GLUE, MTEB)
- Effective benchmark count (21-294 depending on granularity)

### 3. Expand Related Work

**Add subsection**: "Approaches to Benchmark Diversity"

Compare:
- Dataset aggregation (MTEB, GLUE)
- Internal diversity (SHELF)
- Domain-specific (FinMTEB, ChemTEB)

Position SHELF as complementary to MTEB, not competing.

### 4. Add Generalization Discussion

**New subsection**: "Generalization and Validation"

Include:
- Within-benchmark generalization (unseen combinations)
- Expected correlation with real bibliographic tasks
- Limitations of synthetic data
- Future work: cross-benchmark validation

### 5. Update Abstract

**Add phrase**:
> "...creating a benchmark suite with 18 evaluation tasks and 558,600 unique document configurations across independent taxonomic dimensions."

---

## Conclusion

The "single dataset limitation" concern is valid for narrow, homogeneous benchmarks. **It does not apply to SHELF's design.**

SHELF achieves diversity through **controlled cross-product sampling** rather than **dataset aggregation**—a different approach to the same goal. With 21 subject domains, 133 document forms, 18 evaluation tasks, and 558,600 unique configurations, SHELF offers comparable scope to multi-dataset benchmarks while providing advantages of consistent methodology, low contamination risk, and explicit taxonomic grounding.

We have provided detailed analyses demonstrating SHELF's internal diversity and propose specific revisions to clarify this in the paper. We believe these changes adequately address the generalization concern while maintaining our contribution's integrity.

**SHELF is not a "single dataset" in the traditional sense. It is a benchmark suite with unified methodology and massive internal diversity.**

---

## Supplementary Materials

We have prepared the following supplementary materials to support this response:

1. **`diversity_analysis.py`**: Quantitative analysis script computing:
   - Cross-product diversity (558,600 configurations)
   - Task variety (18 tasks across 4 types)
   - Effective benchmark count (21-294)
   - Comparison to MTEB

2. **`analysis.md`**: Comprehensive generalization analysis covering:
   - Research background on benchmark generalization
   - SHELF's internal diversity quantification
   - Comparison to GLUE, SuperGLUE, MTEB
   - Future work to strengthen generalization claims

3. **`comparison_to_mteb.md`**: Detailed MTEB comparison explaining:
   - Structural differences (aggregation vs. cross-products)
   - Complementary purposes (breadth vs. depth)
   - What each benchmark measures
   - Why both are needed

4. **`diversity_analysis_output.json`**: Machine-readable analysis results

These materials are available in `/docs/paper/issues/10_single_dataset/` for reviewer reference.

---

**Response version**: 1.0
**Date**: 2025-12-14
**Authors**: SHELF team
