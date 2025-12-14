# Quick Reference: TF-IDF Train/Test Leakage Investigation

**TL;DR**: ✅ **NO LEAKAGE FOUND** - All implementations maintain proper train/test separation.

## 30-Second Summary

| Task           | Leakage? | Protocol      | Details |
|----------------|----------|---------------|---------|
| Classification | ✅ No    | Inductive     | Vocab + SVD from train only |
| Retrieval      | ✅ No    | Corpus-fitted | Vocab + SVD from corpus only |
| Clustering     | ⚠️ N/A   | Transductive  | Expected and standard |

## 2-Minute Summary

**What was the concern?**
Reviewer worried that TF-IDF vocabulary and SVD components might be fitted on test data, causing train/test leakage.

**What did we find?**
- **Classification**: ✅ Proper separation. Vocabulary and SVD fitted on train, transformed on test.
- **Retrieval**: ✅ Proper separation. Vocabulary and SVD fitted on corpus, transformed on queries.
- **Clustering**: ⚠️ Transductive by design (standard practice, not leakage).

**How did we verify?**
1. Code analysis of adapters and evaluators
2. Runnable verification script (6 tests, all pass)
3. Comparison with best practices from IR/NLP literature

**What's the verdict?**
No train/test leakage. Implementation is correct and defensible.

## File Guide

**Start here**:
- `SUMMARY.md` - Full investigation summary
- `README.md` - Navigation guide

**For reviewers**:
- `rebuttal.md` - Polished response to peer review concerns

**For technical details**:
- `analysis.md` - Deep dive with code traces
- `data_flow.md` - Visual diagrams of data flow
- `verification.py` - Runnable tests

## Run Verification

```bash
# From repo root
uv run python docs/paper/issues/01_tfidf_leakage/verification.py
```

Expected: All 6 tests pass in ~5 seconds.

## Key Code Locations

**TF-IDF Adapter** (`src/shelf/evaluate/adapters/tfidf.py`):
- Line 406: `encode()` method - fit-or-transform logic
- Lines 233-262: `fit()` method
- Lines 307-384: `fit_transform()` method

**Classification** (`src/shelf/evaluate/evaluators/classification.py`):
- Line 309: Train encoding (fits)
- Line 317: Test encoding (transforms)

**Retrieval** (`src/shelf/evaluate/evaluators/retrieval.py`):
- Line 211: Corpus encoding (fits)
- Line 219: Query encoding (transforms)

**Clustering** (`src/shelf/evaluate/evaluators/clustering.py`):
- Line 246: Test encoding (fits - transductive)

## One-Liner Responses

**"Does classification have train/test leakage?"**
No. Vocabulary and SVD fitted on train only, test uses transform.

**"Does retrieval have leakage?"**
No. Corpus-fitted, query-transformed. Standard IR protocol.

**"Why does clustering fit on test?"**
Transductive by design. Standard practice. No label leakage.

**"Can we trust the results?"**
Yes. Implementation follows best practices for each task type.

## For Paper Revision

**Minimal addition** (1 sentence):
> "TF-IDF models use standard fit-transform protocols: fitted on training/corpus data, transformed on test/query data (classification and retrieval), or fitted tranductively on the evaluation split (clustering, following MTEB protocols)."

**Where to add**:
- Baselines section (methodology)
- After describing TF-IDF/TF+SVD baselines

**Why to add**:
- Preempts reviewer concerns
- Clarifies transductive clustering
- Shows awareness of best practices

## Bottom Line

✅ Implementation is **correct**
✅ Results are **valid**
✅ Concern is **unfounded**
✅ Evidence is **comprehensive**

**Confidence level**: Very high. Multiple verification methods confirm proper separation.
