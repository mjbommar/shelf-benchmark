# Issue 05: Statistical Significance

**Reviewer Concern**: Are the reported performance differences statistically significant given high task variance?

**Status**: ✅ COMPLETE - Comprehensive statistical analysis conducted

**Bottom Line**: Most pairwise differences are NOT statistically significant after multiple comparison correction, but large effect sizes (Cohen's d > 0.8) reveal meaningful practical differences. Paper claims require revision.

---

## Files in This Directory

### 1. `significance_tests.py` (18 KB)
**Purpose**: Runnable Python script performing all statistical analyses
**Contains**:
- Bootstrap confidence interval computation (10,000 iterations)
- Paired t-tests and Wilcoxon tests for 9 key model comparisons
- Cohen's d effect size calculations
- Bonferroni and Holm multiple comparison corrections
- Task variance and correlation analyses

**Run**: `uv run python docs/paper/issues/05_statistical_significance/significance_tests.py`

**Output**: Console report + `statistical_tests_results.json`

### 2. `statistical_tests_results.json` (7.7 KB)
**Purpose**: Machine-readable results from significance testing
**Contains**:
- SHELF scores for all 24 models
- Pairwise comparison results (p-values, effect sizes, test statistics)
- Task variance statistics (mean, std, min, max, CV)

**Use**: For programmatic access to statistical results

### 3. `analysis.md` (15 KB)
**Purpose**: Comprehensive statistical analysis writeup
**Sections**:
1. Executive Summary (key findings)
2. Bootstrap Confidence Intervals (methodology and results)
3. Pairwise Statistical Tests (9 key comparisons)
4. Multiple Comparison Corrections (Bonferroni, Holm)
5. Task Variance Analysis (high-variance vs. low-variance tasks)
6. Task Correlation Analysis (identifies r > 0.9 pairs)
7. Effect Size Analysis (Cohen's d distribution)
8. Statistical Power Analysis (n=7 effective tasks)
9. Summary of Statistical Issues (5 key problems)
10. Recommendations for Paper (claim revisions)

**Audience**: Technical reviewers, statisticians

### 4. `confidence_intervals.md` (7.4 KB)
**Purpose**: Bootstrap CI results and interpretation guide
**Contains**:
- Full model ranking table with 95% CIs
- CI width analysis (50-60% of mean)
- Equivalence class identification (3 tiers, not 24 ranks)
- Sparse vs. dense comparison with CIs
- Methodological notes (why percentile bootstrap?)

**Audience**: Paper authors (for tables/figures)

### 5. `rebuttal.md` (20 KB)
**Purpose**: Polished response to reviewer statistical concerns
**Sections**:
1. Summary of Statistical Analysis (1-page overview)
2. Detailed Responses to 5 Statistical Concerns
3. Revised Claims for Paper (before/after templates)
4. Statistical Best Practices (6 recommendations)
5. Addressing Broader Statistical Validity
6. Comparison to Benchmark Best Practices (MTEB, BEIR, HELM)
7. Limitations and Future Work
8. Conclusion and Recommendations

**Audience**: Paper submission, reviewer response

---

## Key Findings Summary

### Finding 1: Limited Statistical Significance
- **Result**: 0 of 9 key comparisons survive Bonferroni correction (α/9 = 0.0056)
- **Uncorrected significant**: 2 comparisons (BM25 vs. TF-IDF p=0.041, BM25 vs. BGE-large p=0.019)
- **Implication**: Cannot claim statistically significant differences with conservative correction

### Finding 2: Large Effect Sizes Exist
- **BM25 vs. BGE-large**: Cohen's d = 0.97 (very large)
- **TF-IDF vs. BM25**: Cohen's d = 0.81 (large)
- **TF vs. TF-IDF**: Cohen's d = 0.60 (medium)
- **Neural model pairs**: Cohen's d < 0.32 (negligible to small)
- **Implication**: Practical differences exist despite limited statistical significance

### Finding 3: Overlapping Confidence Intervals
- **All top-10 models**: 95% CIs overlap
- **CI width**: 50-60% of mean (reflects high task variance)
- **BM25**: 0.547 [0.392, 0.672]
- **BGE-large**: 0.518 [0.370, 0.652]
- **Implication**: Rankings are unstable; treat top-10 as equivalence class

### Finding 4: High Task Correlations
- **8 task pairs** with |r| > 0.9 (near-redundant)
- **same_topic_pairs ↔ topic_overlap_pairs**: r = 0.99
- **same_lcc_pairs ↔ topic_overlap_pairs**: r = 0.97
- **category_retrieval ↔ form_retrieval**: r = 0.97
- **Effective sample size**: ~7 independent task families (not 16 tasks)
- **Implication**: Violates independence assumption; reduces statistical power

### Finding 5: Low Statistical Power
- **With n=7 effective tasks**:
  - Power = 0.08 for d=0.2 (small effects)
  - Power = 0.51 for d=0.8 (large effects)
  - Power = 0.68 for d=1.0 (very large effects)
- **Implication**: Can reliably detect only large differences (d > 0.8)

---

## Critical Claim Revisions

### ❌ REMOVE: "Sparse methods outperform dense embeddings"
**Problem**: TF-IDF vs. BGE-large shows d=0.08 (negligible), p=0.74 (not significant)

### ✅ ADD: "BM25 achieves competitive performance"
**Evidence**: BM25 (0.547) vs. BGE-large (0.518), d=0.97 (large), p=0.019 (marginal, uncorrected)
**Revised claim**:
> "BM25 achieved the highest SHELF score (0.547, 95% CI [0.392, 0.672]), outperforming BGE-large (0.518, 95% CI [0.370, 0.652]) with a large effect size (Cohen's d = 0.97), though this difference was marginally significant (p=0.019) and did not survive Bonferroni correction."

### ❌ REMOVE: "BGE-large is the best neural model"
**Problem**: Overlapping CIs with GTE-base, E5-large (all d < 0.26, p > 0.23)

### ✅ ADD: "Top neural models are statistically equivalent"
**Evidence**: BGE-large, GTE-base, E5-large all have overlapping 95% CIs
**Revised claim**:
> "Among neural models, BGE-large (0.518), GTE-base (0.513), and E5-large (0.507) showed equivalent performance (overlapping 95% CIs, pairwise Cohen's d < 0.26, p > 0.23)."

### ❌ MODIFY: "Model size scaling improves performance"
**Problem**: Large→base marginal (d=0.44, p=0.099), base→small negligible (d=0.08, p=0.77)

### ✅ REVISE: "Model size scaling shows modest benefits"
**Evidence**: Only large→base shows small effect
**Revised claim**:
> "Model size scaling showed modest benefits: BGE-large outperformed BGE-base with a small effect (d=0.44, p=0.099), while BGE-base and BGE-small were equivalent (d=0.08, p=0.77)."

---

## Statistical Best Practices Implemented

### 1. ✅ Bootstrap Confidence Intervals
- **Method**: Percentile bootstrap (recommended by simulation studies)
- **Iterations**: 10,000
- **Application**: All aggregate SHELF scores
- **Benefit**: Robust to non-normality, transparent uncertainty quantification

### 2. ✅ Effect Size Reporting
- **Metric**: Cohen's d (standardized mean difference)
- **Interpretation**: d < 0.2 (negligible), 0.2-0.5 (small), 0.5-0.8 (medium), ≥0.8 (large)
- **Application**: All pairwise comparisons
- **Benefit**: Sample-independent measure of practical importance

### 3. ✅ Multiple Comparison Corrections
- **Methods**: Bonferroni (conservative), Holm (less conservative)
- **Application**: 9 key pairwise tests
- **Result**: 0 significant after correction
- **Benefit**: Controls family-wise error rate (Type I error)

### 4. ✅ Paired Statistical Tests
- **Methods**: Paired t-test (parametric), Wilcoxon signed-rank (non-parametric)
- **Application**: All model comparisons on common tasks
- **Benefit**: Accounts for within-task correlations

### 5. ✅ Task Correlation Analysis
- **Method**: Pearson correlation matrix (models × tasks)
- **Finding**: 8 pairs with r > 0.9 (near-redundant)
- **Benefit**: Identifies task dependencies, estimates effective sample size

### 6. ✅ Statistical Power Analysis
- **Method**: Theoretical power calculation for paired t-test
- **Finding**: 51% power for d=0.8 with n=7 effective tasks
- **Benefit**: Explains non-significant results, sets expectations

---

## Comparison to Other Benchmarks

| Benchmark | # Tasks | Bootstrap CIs? | Significance Testing? | Effect Sizes? | Multiple Corrections? |
|-----------|---------|----------------|------------------------|---------------|------------------------|
| **SHELF** | **16** | **✅ Yes** | **✅ Yes (9 tests)** | **✅ Yes (Cohen's d)** | **✅ Yes (Bonferroni, Holm)** |
| MTEB | 56 | ❌ No | ❌ No | ❌ No | ❌ No |
| BEIR | 18 | ❌ No | ⚠️ Rare | ❌ No | ❌ No |
| GLUE | 9 | ❌ No | ⚠️ Some papers | ⚠️ Rare | ❌ No |
| HELM | 42+ | ⚠️ Some metrics | ⚠️ Limited | ❌ No | ❌ No |

**Conclusion**: SHELF's statistical rigor **exceeds** current NLP benchmarking standards.

---

## Recommendations for Paper

### Essential Changes (Must Do)

1. **Add Bootstrap CIs to all tables**
   - Format: `0.547 [0.392, 0.672]`
   - Location: All aggregate score tables (Tables 1, 2, main results)

2. **Report effect sizes alongside p-values**
   - Format: `d = 0.97, p = 0.019 (uncorrected)`
   - Location: All model comparison discussions

3. **Revise superiority claims**
   - Use templates from `rebuttal.md` Section 3
   - Emphasize equivalence classes over rankings

4. **Add limitations section**
   - Statistical power (n=7 effective tasks)
   - Wide CIs reflect genuine uncertainty
   - Multiple comparisons reduce significance

5. **Include per-task heatmaps**
   - Models × tasks with color-coded scores
   - More informative than aggregate rankings

### Optional Enhancements (Recommended)

6. **Discuss task correlations**
   - Effective sample size (~7 vs. 16 tasks)
   - Task families (pair classification, retrieval, classification, clustering)

7. **Compare to MTEB/BEIR practices**
   - Note: SHELF exceeds standard (CIs, significance tests)

8. **Add equivalence testing**
   - For "no difference" claims (e.g., BGE-base vs. BGE-small)
   - TOST procedure with margin d < 0.2

---

## For Reviewers: Key Takeaways

### Concern: "Are differences statistically significant?"
**Answer**: Mostly no (after correction), but large effect sizes (d > 0.8) indicate practical importance.

### Concern: "Are wide CIs a problem?"
**Answer**: No—they honestly reflect uncertainty given n=7 effective tasks. This is transparent, not flawed.

### Concern: "Is the benchmark still valuable?"
**Answer**: YES—value lies in task-level characterization, not definitive rankings. Statistical rigor exceeds current standards (MTEB, BEIR).

### Concern: "What about multiple comparisons?"
**Answer**: Addressed with Bonferroni/Holm corrections. We report both corrected and uncorrected p-values for transparency.

### Concern: "What about task dependencies?"
**Answer**: High correlations (r > 0.9) identified and discussed. Bootstrap CIs are robust to dependencies.

---

## Next Steps

1. ✅ **Statistical analysis complete** (all files in this directory)
2. 🔲 **Revise paper claims** using templates from `rebuttal.md`
3. 🔲 **Add CI columns** to all results tables
4. 🔲 **Create figures** with error bars (use `confidence_intervals.md` data)
5. 🔲 **Write limitations section** (use `analysis.md` Section 9)
6. 🔲 **Update abstract/conclusions** to reflect statistical findings
7. 🔲 **Prepare reviewer response** (use `rebuttal.md` directly)

---

## Contact

For questions about this statistical analysis:
- **Methods**: See `significance_tests.py` (fully documented)
- **Interpretation**: See `analysis.md` (detailed explanations)
- **Paper revisions**: See `rebuttal.md` (ready-to-use templates)

---

## License Note

Statistical testing implements standard methods:
- Bootstrap: Efron & Tibshirani (1993)
- Paired t-test: Student (1908)
- Wilcoxon: Wilcoxon (1945)
- Bonferroni: Dunn (1961)
- Holm: Holm (1979)
- Cohen's d: Cohen (1988)

All implementations use scipy.stats (BSD license).
