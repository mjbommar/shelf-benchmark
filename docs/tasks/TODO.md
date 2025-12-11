# SHELF Development Roadmap

> **Status**: Active Development
> **Created**: 2025-12-11
> **Last Updated**: 2025-12-11
> **Version**: 0.2.0 (Data Complete)

## Executive Summary

SHELF is a Library of Congress taxonomy benchmark for evaluating NLP models on bibliographic classification, retrieval, and clustering tasks. This document synthesizes research findings from comprehensive analysis of task documentation, data artifacts, evaluation metrics, baseline models, and benchmark design patterns (GLUE, MTEB, HELM, BigBench).

**Current State**:
- ✅ 10,000 synthetic documents generated via GPT-5.1
- ✅ Quality filtering applied (non-English, empty docs removed)
- ✅ Distribution report generated (`docs/CORPUS_DISTRIBUTION_REPORT.md`)
- 8 task definitions documented (4 classification, 3 retrieval, 2 pair, 2 clustering)
- Core generation infrastructure functional
- No evaluation harness, baselines, or leaderboard yet

**Target State**:
- 10,000+ documents with quality filtering
- Complete evaluation infrastructure
- Published baselines with reproducible results
- Public leaderboard and submission system
- Community contribution framework

---

## Phase 0: Current Foundation (Complete)

### What Exists

| Component | Status | Location |
|-----------|--------|----------|
| Document sampler framework | ✅ Complete | `src/shelf/sampler/` |
| Fluent API for metadata | ✅ Complete | `sampler/document.py` |
| GPT-5.1 generation pipeline | ✅ Complete | `sampler/generator.py` |
| Artifact storage | ✅ Complete | `data/artifacts/` |
| Distribution analysis script | ✅ Complete | `scripts/analyze_distribution.py` |
| Task documentation (8 tasks) | ✅ Complete | `docs/tasks/*.md` |

### Quality Assessment (from Data Analysis)

**Corpus Statistics** (see `docs/CORPUS_DISTRIBUTION_REPORT.md`):
- 10,000 documents, ~7.38 million words total
- 21 LCC codes (4.4-5.2% each, well-balanced)
- 14 LCGFT categories (6.6-7.5% each, well-balanced)
- 133 LCGFT forms (long-tail distribution)
- 25 audience types (30% None, remainder balanced)
- 8 registers, 8 target lengths, 44 geographic locations, 112 topics

**Quality Filtering Applied**:
- ✅ Removed 20 empty-body documents (regenerated)
- ✅ Removed 286 non-English documents (German, Italian, Spanish, Korean, Latin, French)
- ✅ Final corpus: 10,000 high-quality English documents

**Remaining Known Issues**:
- 5-8% register-content mismatches (detectable but not filtered)
- Single-model generation bias (GPT-5.1 only)
- Fabricated citations not marked as synthetic

---

## Phase 1: Documentation & Guides

> **Objective**: Create comprehensive documentation that enables reproducibility and community adoption.
> **Dependencies**: None (can start immediately)
> **Estimated Effort**: 1-2 weeks

### Task 1.1: Evaluation Metrics Guide
**Deliverable**: `docs/tasks/EVALUATION_METRICS_GUIDE.md`

Create comprehensive guide covering:
- [ ] Classification metrics (Macro-F1, Micro-F1, Weighted-F1, per-class analysis)
- [ ] Multi-label metrics (Hamming Loss, Subset Accuracy, label-wise F1)
- [ ] Retrieval metrics (NDCG@k, MRR, Recall@k, MAP@k)
- [ ] Clustering metrics (V-measure, NMI, ARI, internal metrics)
- [ ] Pair classification metrics (F1, MCC, balanced accuracy)
- [ ] Statistical significance testing (bootstrap CI, McNemar's, Mann-Whitney U)
- [ ] Code snippets for all metrics (scikit-learn implementations)
- [ ] When to use which metric decision tree
- [ ] Cross-task metric comparison guidance

**Acceptance Criteria**:
- All SHELF metrics documented with formulas
- Working code examples for each metric
- Statistical testing workflow documented

### Task 1.2: Baseline Models Guide
**Deliverable**: `docs/tasks/BASELINE_MODELS_GUIDE.md`

Create practical baseline implementation guide:
- [ ] Model taxonomy (classical ML, embeddings, fine-tuned transformers, LLMs)
- [ ] Expected performance ranges per task
- [ ] Cost/performance tradeoff analysis
- [ ] Implementation code for each baseline category:
  - TF-IDF + Logistic Regression
  - Sentence-BERT (zero-shot)
  - BERT/RoBERTa fine-tuning
  - LLM few-shot (GPT-4o, Claude)
  - Ensemble/hybrid approaches
- [ ] Hyperparameter recommendations
- [ ] Training time and resource estimates
- [ ] Model selection decision framework

**Acceptance Criteria**:
- 5+ baseline implementations with code
- Performance estimates for all 8 tasks
- Clear model selection guidance

### Task 1.3: Benchmark Design Document
**Deliverable**: `docs/BENCHMARK_DESIGN.md`

Document benchmark architecture decisions:
- [ ] Comparison to GLUE/MTEB/HELM/BigBench
- [ ] Task organization hierarchy
- [ ] Aggregate scoring methodology (SHELF Score formula)
- [ ] Metric normalization approach
- [ ] Leaderboard tier structure (Official, Community, Research)
- [ ] Submission format specification (JSON schema)
- [ ] Versioning scheme (semantic versioning)
- [ ] Contamination prevention strategies
- [ ] Reproducibility requirements
- [ ] Community contribution guidelines

**Acceptance Criteria**:
- Clear rationale for all design decisions
- Submission format validated with JSON schema
- Version compatibility guidelines documented

### Task 1.4: Data Quality Analysis Report ✅ COMPLETE
**Deliverable**: `docs/CORPUS_DISTRIBUTION_REPORT.md`
**Status**: ✅ Complete (2025-12-11)

Document data generation quality:
- [x] Sampling methodology and statistics
- [x] Quality metrics by dimension (LCC, LCGFT, audience, register, length)
- [x] Distribution coverage analysis
- [x] Identified issues and mitigation strategies
- [x] Recommendations for train/dev/test splits

**Results**:
- Comprehensive 12-section report generated
- All dimensions analyzed with percentages
- Quality filtering documented
- Split recommendations provided

### Task 1.5: Task Enhancement Observations
**Deliverable**: "Observations and Enhancement Ideas" sections in existing task docs

Add research findings to each task file:
- [ ] `lcc_classification.md` - hierarchical loss, boundary cases, title-body fusion
- [ ] `lcgft_classification.md` - rare-form handling, hierarchy exploitation, form similarity
- [ ] `topic_classification.md` - label dependency modeling, threshold optimization
- [ ] `audience_register_classification.md` - class imbalance handling, linguistic features
- [ ] `retrieval.md` - hard negatives, query variants, cross-lingual extension
- [ ] `clustering.md` - hierarchical evaluation, internal metrics, S2S/P2P variants
- [ ] `pair_classification.md` - hard negative mining, similarity scoring, partial match

**Acceptance Criteria**:
- Each task doc has enhancement section
- Specific, actionable recommendations

---

## Phase 2: Data Generation & Quality

> **Objective**: Scale to 10,000+ high-quality documents with proper train/dev/test splits.
> **Dependencies**: Phase 1 (quality criteria defined)
> **Estimated Effort**: 2-3 weeks

### Task 2.1: Quality Filtering Pipeline
**Deliverable**: `src/shelf/quality/` module
**Status**: 🟡 Partial (ad-hoc filtering done, module not yet created)

Implement automated quality checks:
- [x] Empty body detection and removal
- [x] Non-English content detection (multi-language pattern matching)
- [x] Removal manifest for traceability (`data/removed_non_english_manifest.json`)
- [ ] Length validation (within 5% of target range)
- [ ] Register consistency classifier (detect mismatches)
- [ ] Coherence scoring (perplexity-based or model-based)
- [ ] Metadata-content alignment verification
- [ ] Duplicate/near-duplicate detection
- [ ] Quality score aggregation per document
- [ ] Filtering thresholds and rejection logging

**Acceptance Criteria**:
- Quality module with clear API
- >95% of passing documents are high-quality (human validation on sample)
- Rejection reasons logged for analysis

### Task 2.2: Cross-Model Generation
**Deliverable**: Diverse document corpus

Reduce single-model bias:
- [ ] Implement Claude generation backend
- [ ] Implement Gemini generation backend (optional)
- [ ] Generate 20-30% of corpus with non-GPT models
- [ ] Compare quality distributions across models
- [ ] Document model-specific patterns

**Acceptance Criteria**:
- At least 2 generation models used
- Model source tracked in metadata
- No significant quality degradation

### Task 2.3: Scale to 10,000 Documents ✅ COMPLETE
**Deliverable**: Full corpus in `data/`
**Status**: ✅ Complete (2025-12-11)

Generate production dataset:
- [x] Target distribution: balanced across all dimensions
- [x] 10,000 documents minimum
- [x] Apply quality filtering (306 rejected, regenerated)
- [x] Create distribution report (`docs/CORPUS_DISTRIBUTION_REPORT.md`)

**Results**:
- 10,000 documents in `data/artifacts/`
- All 21 LCC codes represented (4.4-5.2% each)
- All 133 LCGFT forms represented
- ~7.38 million words total
- ~80 MB storage

### Task 2.4: Stratified Train/Dev/Test Splits
**Deliverable**: `data/splits/` with train.jsonl, dev.jsonl, test.jsonl

Create evaluation splits:
- [ ] Stratified sampling preserving label distributions
- [ ] Split ratios: 60% train, 20% dev, 20% test
- [ ] Ensure all labels appear in all splits (minimum 3 per class per split)
- [ ] No document overlap between splits
- [ ] Generate split statistics report
- [ ] Create checksums for reproducibility

**Acceptance Criteria**:
- Balanced splits with documented statistics
- All labels represented in each split
- Checksums published

### Task 2.5: Test Set Embargo & Security
**Deliverable**: Secure test set distribution

Prevent contamination:
- [ ] Encrypt test set for embargoed distribution
- [ ] Create test set submission API (predictions only)
- [ ] Implement n-gram overlap detection
- [ ] Document contamination detection protocol
- [ ] Plan periodic test set refresh schedule

**Acceptance Criteria**:
- Test set not publicly accessible in raw form
- Contamination detection pipeline functional

---

## Phase 3: Evaluation Infrastructure

> **Objective**: Build complete evaluation harness with reproducible baselines.
> **Dependencies**: Phase 2 (data splits available)
> **Estimated Effort**: 3-4 weeks

### Task 3.1: Core Evaluation Harness
**Deliverable**: `src/shelf/evaluate/` module

Implement evaluation framework:
- [ ] Task-agnostic evaluation runner
- [ ] Metric computation for all task types
- [ ] Per-class breakdown generation
- [ ] Confusion matrix generation
- [ ] Statistical significance testing utilities
- [ ] Results serialization (JSON format)
- [ ] CLI interface: `shelf evaluate --task <task> --predictions <file>`

**Acceptance Criteria**:
- All 8 tasks evaluable via CLI
- Reproducible results (fixed seeds)
- Clear error messages for malformed inputs

### Task 3.2: Classification Evaluation
**Deliverable**: Classification task evaluators

Implement for each classification task:
- [ ] LCC Classification (21 classes, Macro-F1)
- [ ] LCGFT Form Classification (133 classes, Macro-F1)
- [ ] Topic Classification (113 labels, Micro-F1, multi-label)
- [ ] Audience Classification (25 classes, Macro-F1)
- [ ] Register Classification (8 classes, Macro-F1)

Per task:
- [ ] Primary metric computation
- [ ] Secondary metrics (Weighted-F1, Accuracy)
- [ ] Per-class precision/recall/F1
- [ ] Confusion matrix for classes with errors
- [ ] Error analysis output

**Acceptance Criteria**:
- Verified against manual calculation on sample
- Handles edge cases (zero-support classes)

### Task 3.3: Retrieval Evaluation
**Deliverable**: Retrieval task evaluators

Implement for each retrieval task:
- [ ] LCC Retrieval (subject similarity)
- [ ] Form Retrieval (genre similarity)
- [ ] Topic Retrieval (topic relevance)

Per task:
- [ ] Query generation from test documents
- [ ] Relevance judgment computation
- [ ] NDCG@k (k=1,5,10,50,100)
- [ ] MRR computation
- [ ] Recall@k computation
- [ ] MAP@k computation
- [ ] Per-query breakdown

**Acceptance Criteria**:
- Matches MTEB/BEIR evaluation methodology
- Handles ties correctly

### Task 3.4: Clustering Evaluation
**Deliverable**: Clustering task evaluators

Implement for each clustering task:
- [ ] LCC Clustering (21 clusters)
- [ ] LCGFT Category Clustering (14 clusters)

Per task:
- [ ] k-means clustering with configurable k
- [ ] V-measure computation
- [ ] NMI computation
- [ ] ARI computation
- [ ] Per-cluster homogeneity/completeness
- [ ] Internal metrics (Silhouette, optional)

**Acceptance Criteria**:
- Matches MTEB clustering evaluation
- Reproducible with fixed seeds

### Task 3.5: Pair Classification Evaluation
**Deliverable**: Pair classification task evaluators

Implement for each pair task:
- [ ] Same-LCC Prediction
- [ ] Same-Form Prediction

Per task:
- [ ] Pair generation (balanced positive/negative)
- [ ] F1 score (positive class)
- [ ] Accuracy
- [ ] MCC (Matthews Correlation Coefficient)
- [ ] (F1 + Accuracy) / 2 average (GLUE style)

**Acceptance Criteria**:
- Balanced pair generation
- Follows GLUE pair classification protocol

### Task 3.6: Baseline Implementations
**Deliverable**: `src/shelf/baselines/` module

Implement reference baselines:
- [ ] TF-IDF + Logistic Regression (all classification tasks)
- [ ] Sentence-BERT zero-shot (all tasks)
- [ ] BM25 (retrieval tasks)
- [ ] BERT-base fine-tuning (classification tasks)
- [ ] Cross-encoder (pair classification)

Per baseline:
- [ ] Training script (if applicable)
- [ ] Inference script
- [ ] Hyperparameter configuration
- [ ] Published results on dev set

**Acceptance Criteria**:
- All baselines reproducible from scratch
- Results within expected ranges (per Baseline Models Guide)
- Training time and resources documented

### Task 3.7: Aggregate Scoring
**Deliverable**: SHELF Score computation

Implement aggregate scoring:
- [ ] Metric normalization to [0, 1]
- [ ] Task family aggregation (Classification, Retrieval, Pair, Clustering)
- [ ] Weighted SHELF Score computation
- [ ] Configurable weights
- [ ] Score breakdown visualization

**Acceptance Criteria**:
- Clear, documented formula
- Reproducible aggregate scores

---

## Phase 4: Task Enhancements

> **Objective**: Implement task variants and improvements identified in research.
> **Dependencies**: Phase 3 (base evaluation working)
> **Estimated Effort**: 4-6 weeks (can be parallelized)

### Task 4.1: Hard Negative Mining
**Deliverable**: Enhanced negative sampling for retrieval and pair tasks

Implement difficulty-stratified negatives:
- [ ] Adjacent LCC class negative mining
- [ ] Similar LCGFT form negative mining
- [ ] Content-based hard negatives (embedding similarity)
- [ ] Difficulty tier labeling (Easy/Medium/Hard)
- [ ] Curriculum learning support

**Acceptance Criteria**:
- Hard negatives demonstrably harder (baseline accuracy lower)
- Difficulty distribution documented

### Task 4.2: Hierarchical Classification Enhancement
**Deliverable**: Hierarchical evaluation for LCGFT

Implement hierarchy-aware evaluation:
- [ ] Category → Form hierarchy encoding
- [ ] Hierarchical consistency metric
- [ ] Two-stage prediction support (predict category, then form)
- [ ] Partial credit for correct category, wrong form

**Acceptance Criteria**:
- Hierarchical metrics implemented
- Comparison with flat classification

### Task 4.3: Few-Shot Task Variants
**Deliverable**: Few-shot versions of classification tasks

Create few-shot evaluation:
- [ ] 5-shot variant for each classification task
- [ ] 10-shot variant
- [ ] 50-shot variant
- [ ] Prompt templates for LLM evaluation
- [ ] SetFit-style evaluation support

**Acceptance Criteria**:
- Few-shot baselines published
- Clear evaluation protocol

### Task 4.4: Cross-Lingual Retrieval (SHELF-XLR)
**Deliverable**: Multilingual retrieval task

Extend to cross-lingual:
- [ ] Translate 20% of documents to Spanish, Portuguese, Chinese
- [ ] Cross-lingual query-document pairs
- [ ] Language-specific evaluation breakdown
- [ ] Multilingual embedding baseline (BGE-m3, mE5)

**Acceptance Criteria**:
- At least 2 non-English languages
- Baseline results for multilingual models

### Task 4.5: Adversarial/Robustness Variants
**Deliverable**: SHELF-Hard adversarial subset

Create challenging evaluation:
- [ ] Confusable LCC class pairs only
- [ ] Rare LCGFT forms only
- [ ] Length-extreme documents (very short, very long)
- [ ] Register-mismatched evaluation
- [ ] Robustness metrics (performance drop vs. standard)

**Acceptance Criteria**:
- Adversarial subset demonstrably harder
- Robustness analysis for baselines

### Task 4.6: Multi-Hop Retrieval
**Deliverable**: SHELF-MultiHop reasoning task

Create compound retrieval:
- [ ] Multi-constraint queries (LCC AND Topic AND Geographic)
- [ ] Boolean retrieval evaluation
- [ ] Per-constraint recall analysis
- [ ] Compound query baselines

**Acceptance Criteria**:
- 100+ multi-hop queries
- Baseline results published

---

## Phase 5: Community & Launch

> **Objective**: Public release with leaderboard, website, and community infrastructure.
> **Dependencies**: Phase 3 (evaluation harness), Phase 4 (optional enhancements)
> **Estimated Effort**: 4-6 weeks

### Task 5.1: Leaderboard Backend
**Deliverable**: Submission and ranking system

Implement leaderboard infrastructure:
- [ ] Submission API (receive predictions JSON)
- [ ] Automatic evaluation pipeline
- [ ] Results database (PostgreSQL)
- [ ] Ranking computation
- [ ] Historical tracking
- [ ] API endpoints for leaderboard data

**Acceptance Criteria**:
- End-to-end submission → evaluation → ranking
- <5 minute turnaround for evaluation

### Task 5.2: Leaderboard Frontend
**Deliverable**: Public leaderboard website

Build interactive leaderboard:
- [ ] Sortable/filterable leaderboard table
- [ ] Per-task breakdown views
- [ ] Model comparison tool (side-by-side)
- [ ] Visualizations (radar charts, bar charts)
- [ ] Results export (CSV, JSON)
- [ ] Responsive design (mobile-friendly)

**Acceptance Criteria**:
- Intuitive, fast UI
- All metrics visible
- Shareable permalinks

### Task 5.3: Documentation Website
**Deliverable**: docs.shelf.org

Create documentation portal:
- [ ] Getting started guide
- [ ] Task specifications
- [ ] Evaluation protocol
- [ ] Submission guide
- [ ] API reference
- [ ] FAQ
- [ ] Search functionality

**Acceptance Criteria**:
- All documentation accessible
- Clear navigation
- Code examples runnable

### Task 5.4: Submission Format Validation
**Deliverable**: Submission validation tools

Implement submission checking:
- [ ] JSON schema validation
- [ ] Required fields verification
- [ ] Metric range validation
- [ ] Duplicate detection
- [ ] Clear error messages
- [ ] Pre-submission validation CLI tool

**Acceptance Criteria**:
- All malformed submissions rejected with clear errors
- Validation available locally before submission

### Task 5.5: PyPI Package
**Deliverable**: `pip install shelf`

Package for distribution:
- [ ] Package structure (src layout)
- [ ] Dependencies specification
- [ ] CLI entry points
- [ ] Data download utilities
- [ ] Version management
- [ ] PyPI publication
- [ ] Conda-forge (optional)

**Acceptance Criteria**:
- `pip install shelf` works
- `shelf evaluate` CLI functional
- Data downloadable via package

### Task 5.6: Public Baselines Publication
**Deliverable**: Published baseline results

Release official baselines:
- [ ] Run all baselines on test set
- [ ] Publish results with confidence intervals
- [ ] Release model weights (where applicable)
- [ ] Document reproduction steps
- [ ] Create baseline leaderboard entries

**Acceptance Criteria**:
- All baselines reproducible by external users
- Results match documented ranges

### Task 5.7: Launch Announcement
**Deliverable**: Public release

Coordinate launch:
- [ ] ArXiv paper (benchmark description)
- [ ] Blog post announcement
- [ ] Twitter/social media
- [ ] HuggingFace dataset publication
- [ ] Papers With Code integration
- [ ] Community outreach (NLP conferences, forums)

**Acceptance Criteria**:
- Benchmark discoverable via standard channels
- Clear call for submissions

### Task 5.8: Community Contribution Framework
**Deliverable**: Contribution guidelines and infrastructure

Enable community participation:
- [ ] GitHub issue templates (bug, feature, new task)
- [ ] Pull request template
- [ ] Contribution guidelines (CONTRIBUTING.md)
- [ ] Code review process
- [ ] Contributor recognition system
- [ ] Discussion forum (GitHub Discussions or Discord)

**Acceptance Criteria**:
- Clear path for external contributions
- First external contribution merged

---

## Phase 6: Maintenance & Evolution

> **Objective**: Long-term sustainability and benchmark evolution.
> **Dependencies**: Phase 5 (launched)
> **Ongoing**

### Task 6.1: Contamination Monitoring
**Deliverable**: Ongoing contamination detection

Implement monitoring:
- [ ] Periodic n-gram overlap checks against new models
- [ ] Suspicious result flagging
- [ ] Contamination audit reports
- [ ] Test set refresh schedule (quarterly)

### Task 6.2: Version Updates
**Deliverable**: SHELF 1.1, 1.2, 2.0

Plan evolution:
- [ ] v1.1 (3 months): Data refresh, new baselines, bug fixes
- [ ] v1.2 (6 months): New task variants (few-shot, multilingual)
- [ ] v2.0 (12 months): Major update (new tasks, expanded corpus)

### Task 6.3: Community Engagement
**Deliverable**: Active community

Maintain engagement:
- [ ] Regular leaderboard updates
- [ ] Highlight interesting submissions
- [ ] Workshops at conferences
- [ ] Collaboration with library science community
- [ ] Annual benchmark report

---

## Priority Matrix

### P0: Critical Path (Blocks Everything)
1. ~~Task 2.3: Scale to 10,000 Documents~~ ✅ COMPLETE
2. **Task 2.4: Stratified Train/Dev/Test Splits** ← NEXT
3. Task 3.1: Core Evaluation Harness
4. Task 3.6: Baseline Implementations

### P1: High Priority (Required for Launch)
1. Task 1.1-1.4: Documentation & Guides
2. Task 2.1: Quality Filtering Pipeline
3. Task 3.2-3.5: Task-Specific Evaluators
4. Task 5.1-5.2: Leaderboard
5. Task 5.5: PyPI Package

### P2: Important (Enhances Value)
1. Task 1.5: Task Enhancement Observations
2. Task 2.2: Cross-Model Generation
3. Task 3.7: Aggregate Scoring
4. Task 4.1: Hard Negative Mining
5. Task 4.3: Few-Shot Variants

### P3: Nice to Have (Future Work)
1. Task 4.2: Hierarchical Classification
2. Task 4.4: Cross-Lingual Retrieval
3. Task 4.5: Adversarial Variants
4. Task 4.6: Multi-Hop Retrieval

---

## Success Metrics

### Launch Criteria (MVP)
- [ ] 10,000+ documents with train/dev/test splits
- [ ] All 8 tasks evaluable via CLI
- [ ] 3+ baselines published with reproducible results
- [ ] Public leaderboard accepting submissions
- [ ] Documentation complete for all tasks
- [ ] PyPI package installable

### 6-Month Goals
- [ ] 50+ unique model submissions
- [ ] 10+ institutions participating
- [ ] 5+ papers citing SHELF
- [ ] 1,000+ GitHub stars
- [ ] 3+ community-contributed task variants
- [ ] Zero contamination incidents

### 12-Month Goals
- [ ] 200+ leaderboard entries
- [ ] SHELF v2.0 released
- [ ] Workshop at major NLP conference
- [ ] Integration with MTEB/HELM ecosystems
- [ ] Multilingual expansion (5+ languages)
- [ ] Library science community adoption

---

## Resource Estimates

### Compute Requirements
| Task | GPU Hours | Cost Estimate |
|------|-----------|---------------|
| Document Generation (10K) | 20-40 | $50-100 |
| Quality Filtering | 5-10 | $10-25 |
| Baseline Training (all) | 50-100 | $100-250 |
| Evaluation Runs | 10-20 | $25-50 |
| **Total** | **85-170** | **$185-425** |

### API Costs (Generation)
| Model | Documents | Estimated Cost |
|-------|-----------|----------------|
| GPT-5.1 | 7,000 | $35-70 |
| Claude | 2,000 | $10-20 |
| Gemini | 1,000 | $5-10 |
| **Total** | **10,000** | **$50-100** |

### Human Effort
| Phase | Person-Weeks | Skills Needed |
|-------|--------------|---------------|
| Phase 1 (Docs) | 2 | Technical writing |
| Phase 2 (Data) | 3 | ML engineering |
| Phase 3 (Eval) | 4 | ML engineering |
| Phase 4 (Tasks) | 4-6 | ML research |
| Phase 5 (Launch) | 4-6 | Full-stack, DevOps |
| **Total** | **17-21** | Mixed |

---

## Risks and Mitigations

### Risk 1: Data Quality Issues at Scale
**Impact**: High
**Likelihood**: Medium
**Mitigation**:
- Implement quality filtering before scaling
- Human validation on 5% sample
- Iterative generation with feedback

### Risk 2: Contamination in Foundation Models
**Impact**: High
**Likelihood**: Medium
**Mitigation**:
- Test set encryption and embargo
- N-gram detection pipeline
- Periodic test set refresh
- Clear contamination policy

### Risk 3: Low Community Adoption
**Impact**: Medium
**Likelihood**: Medium
**Mitigation**:
- Strong baseline suite
- Easy-to-use package
- Integration with existing ecosystems (MTEB)
- Academic paper for visibility

### Risk 4: Benchmark Saturation
**Impact**: Medium
**Likelihood**: Low (long-term)
**Mitigation**:
- Plan for v2.0 with harder tasks
- Adversarial variants ready
- Community task contributions

### Risk 5: Maintenance Burden
**Impact**: Medium
**Likelihood**: Medium
**Mitigation**:
- Automated evaluation pipeline
- Clear contribution guidelines
- Community maintainers

---

## Appendix A: Research Findings Summary

### Classification Task Insights
- LCGFT 133-class task is "extreme fine-grained" (comparable to CUB-200, CIFAR-100)
- Audience task has 30% null class imbalance requiring focal loss or stratification
- Hierarchical loss can improve LCGFT by 3-5%
- Register detection benefits from character-level features

### Retrieval Task Insights
- SHELF fills unique niche: library science taxonomies not in BEIR/MTEB
- Hard negative mining from adjacent LCC classes improves NDCG by 3-5%
- Cross-lingual variant would differentiate from existing benchmarks
- Multi-hop reasoning variant tests compound query capabilities

### Clustering Task Insights
- V-measure appropriate but NOT adjusted for chance (use with ARI)
- 21 clusters can inflate NMI; need careful interpretation
- S2S/P2P variants (title-only vs. full doc) would strengthen evaluation
- Hierarchical clustering evaluation aligns with actual LCC structure

### Pair Classification Insights
- Fundamentally different from STS/NLI (categorical equivalence vs. semantic similarity)
- Hard negatives from adjacent categories most valuable
- Continuous similarity scoring would add value beyond binary
- Symmetric evaluation important (order shouldn't matter)

### Data Quality Insights
- 92-98% coherence across sampled documents
- 5-8% register mismatches detectable with automated classifier
- Academic/professional registers highest quality
- Creative/casual registers more variable
- Cross-model generation reduces single-model bias

### Baseline Model Insights
- Expected ranges: TF-IDF 45-65%, BERT 70-85%, RoBERTa-large 80-90%
- GPT-4o/Claude few-shot: 75-85% with good prompting
- NV-Embed leads MTEB (69.32), BGE-multilingual strong alternative
- SetFit excellent for few-shot scenarios

---

## Appendix B: Task Specification Quick Reference

| Task | Classes | Primary Metric | Expected Baseline |
|------|---------|----------------|-------------------|
| LCC Classification | 21 | Macro-F1 | 0.70-0.85 |
| LCGFT Form | 133 | Macro-F1 | 0.55-0.75 |
| Topic (multi-label) | 113 | Micro-F1 | 0.70-0.85 |
| Audience | 25 | Macro-F1 | 0.60-0.80 |
| Register | 8 | Macro-F1 | 0.70-0.85 |
| LCC Retrieval | - | NDCG@10 | 0.60-0.75 |
| Form Retrieval | - | NDCG@10 | 0.55-0.70 |
| Topic Retrieval | - | NDCG@10 | 0.60-0.75 |
| Same-LCC Pair | 2 | F1 | 0.75-0.88 |
| Same-Form Pair | 2 | F1 | 0.70-0.85 |
| LCC Clustering | 21 | V-measure | 0.55-0.75 |
| LCGFT Clustering | 14 | V-measure | 0.60-0.80 |

---

## Appendix C: SHELF Score Formula

```
SHELF Score = 0.40 × Classification + 0.30 × Retrieval +
                 0.15 × PairClassification + 0.15 × Clustering

Where:
  Classification = mean(LCC_F1, LCGFT_F1, Topic_F1, Audience_F1, Register_F1)
  Retrieval = mean(LCC_NDCG, Form_NDCG, Topic_NDCG)
  PairClassification = mean(SameLCC_F1, SameForm_F1)
  Clustering = mean(LCC_Vmeasure, LCGFT_Vmeasure)
```

---

*Document maintained by SHELF development team.*
