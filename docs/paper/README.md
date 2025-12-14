# SHELF Paper Drafting

This folder contains materials for the SHELF benchmark paper submission.

## Target Venues

| Venue | Deadline | Track | Notes |
|-------|----------|-------|-------|
| NeurIPS 2025 | May 2025 | Datasets & Benchmarks | Strong fit for methodology |
| ICLR 2026 | Sep 2025 | Main | Surprising findings angle |
| ACL 2025 | Feb 2025 | Main | NLP benchmark contribution |
| EMNLP 2025 | Jun 2025 | Main | Document understanding |
| COLM 2025 | TBD | Main | LLM evaluation focus |

## Document Structure

```
docs/paper/
├── README.md                 # This file
├── abstract.md               # Title options and abstract
├── key_findings.md           # Main empirical findings
├── statistical_findings.md   # Statistical analysis summary (NEW)
├── contributions.md          # Paper contributions list
├── reviewer_concerns.md      # Anticipated reviewer concerns
├── related_work_notes.md     # Notes on related benchmarks
├── issues/                   # Detailed reviewer concern rebuttals
│   ├── 01_tfidf_leakage/     # TF-IDF information leakage analysis
│   ├── 02_synthetic_quality/ # Synthetic data quality validation
│   ├── 03_contamination/     # Data contamination concerns
│   ├── 04_lc_bias/           # Library of Congress bias analysis
│   ├── 05_statistical_significance/  # Statistical significance (CRITICAL)
│   ├── 06_task_independence/ # Task correlation analysis
│   ├── 07_baseline_fairness/ # Baseline hyperparameter fairness
│   ├── 08_document_length/   # Document length effects
│   ├── 09_reproducibility/   # Reproducibility checklist
│   └── 10_single_dataset/    # Single dataset limitation
└── figures/                  # Figure concepts and drafts
    └── README.md
```

## Current Status

- [x] Initial abstract drafted
- [x] Key findings documented
- [x] Contributions identified
- [x] Statistical analysis complete
- [x] Reviewer concerns addressed (10 issues)
- [ ] Related work survey
- [ ] Introduction draft
- [ ] Methods section
- [ ] Experiments section
- [ ] Figure designs
- [ ] Camera-ready preparation

## Quick Stats (v0.3.0)

- **Documents**: 42,616
- **Generation Models**: 9 (GPT-5.1/5.2, Gemini 2.5 Flash/Pro, Gemini 3 Pro, Claude 4.5 Haiku/Sonnet/Opus)
- **Evaluation Models**: 24 (21 dense + 3 sparse)
- **Task Types**: 4 (Classification, Retrieval, Clustering, Pair Classification)
- **Total Tasks**: 16
- **Total Evaluations**: 358

## Key Result (Updated with Statistical Analysis)

**Raw scores**: TF-IDF+SVD (0.679) achieves highest SHELF score vs BGE-large (0.513)

**Statistical reality**: After multiple comparison correction, **no model differences are statistically significant** (Friedman p = 0.958). All 24 models form a single equivalence group.

**Task independence**: Mean task correlation r = 0.088 (< 0.3 threshold), confirming tasks measure independent capabilities.

**Implication**: SHELF is best positioned as a **model characterization tool** revealing task-specific capability profiles, not as a ranking system.

## How to Run Analysis

```bash
# Full statistical analysis
shelf eval analyze --verbose

# Export to JSON for paper figures
shelf eval analyze --verbose --export-json analysis.json
```
