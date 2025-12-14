# TF-IDF Train/Test Leakage Investigation

**Issue**: Reviewer concern about potential train/test leakage in TF-IDF/TF+SVD evaluation
**Status**: ✅ **RESOLVED** - No leakage found
**Date**: 2025-12-14

## Quick Summary

After comprehensive investigation, we confirm that SHELF's TF-IDF and TF+SVD implementations maintain **proper train/test separation** for all task types. The concern is **unfounded**.

## Files in this Directory

1. **`analysis.md`** - Detailed technical analysis
   - Code architecture review
   - Data flow traces for each task type
   - Comparison with best practices
   - References to code locations

2. **`verification.py`** - Runnable verification script
   - Demonstrates sklearn TF-IDF behavior
   - Tests SHELF implementation for each task type
   - Shows proper separation for classification and retrieval
   - Explains transductive nature of clustering

3. **`rebuttal.md`** - Polished response for reviewers
   - Summary of findings
   - Evidence of proper separation
   - Addresses potential confusions
   - References to best practices

## How to Run Verification

```bash
# From the repo root
uv run python docs/paper/issues/01_tfidf_leakage/verification.py
```

Expected output: All tests pass, demonstrating:
- ✅ Classification: Proper train/test separation (inductive)
- ✅ Retrieval: Proper corpus/query separation (corpus-fitted)
- ✅ Clustering: Transductive (expected and acceptable)

## Key Findings

### Classification (Inductive)
- Vocabulary fitted on **train only**
- IDF weights from **train only**
- SVD components from **train embeddings only**
- Test data transformed using train-fitted models
- **Verdict**: ✅ No leakage

### Retrieval (Corpus-Fitted)
- Vocabulary fitted on **corpus (train+validation)**
- IDF weights from **corpus only**
- SVD components from **corpus embeddings only**
- Queries transformed using corpus-fitted models
- **Verdict**: ✅ No leakage (standard IR protocol)

### Clustering (Transductive)
- Vocabulary and SVD fitted on **test split**
- This is **transductive learning**
- Standard for clustering tasks
- Ground truth labels not used during embedding/clustering
- **Verdict**: ⚠️ Transductive (expected and acceptable)

## Code References

**Adapters**:
- `/home/mjbommar/src/shelf-benchmark/src/shelf/evaluate/adapters/tfidf.py`
- `/home/mjbommar/src/shelf-benchmark/src/shelf/evaluate/adapters/tf.py`

**Evaluators**:
- `/home/mjbommar/src/shelf-benchmark/src/shelf/evaluate/evaluators/classification.py` (lines 254-396)
- `/home/mjbommar/src/shelf-benchmark/src/shelf/evaluate/evaluators/retrieval.py` (lines 145-260)
- `/home/mjbommar/src/shelf-benchmark/src/shelf/evaluate/evaluators/clustering.py` (lines 191-285)

## Next Steps

If needed for paper revision:
1. Add clarification about fit-transform pattern in methodology section
2. Explicitly note transductive nature of clustering evaluation
3. Add references to standard IR and clustering protocols
4. Include verification script as supplementary material

## References

- [RapidMiner: Apply IDF of training set in test](https://community.rapidminer.com/discussion/16987/solved-apply-idf-of-training-set-in-test)
- [Using LSA to Score Short Answer Responses](https://pmc.ncbi.nlm.nih.gov/articles/PMC7047257/)
- [Stanford NLP: LSI](https://nlp.stanford.edu/IR-book/pdf/18lsi.pdf)
- [TF-IDF with Scikit-Learn](https://www.pythontutorials.net/blog/tfidf-sklearn/)
