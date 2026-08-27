# Reducing verbatim topic echo

Measured 2026-08-27. 40 specifications x 3 arms, generated with
`gemini-3.7-flash`; a second model (`claude-haiku-4-5-20251001`) was added
after a model-id error and is not in the judged sample below.

## The mechanism

`generator.py` renders form and subject area as *semantic descriptions* but
passes topics through verbatim:

```python
f"topics: {', '.join(doc.topics)}"
```

That asymmetry predicts what the corpus shows: topics appear verbatim in
44.5% of v0.4 documents (length-controlled) against 1.5% for form.

## Arms

| arm | topics line |
|---|---|
| `control` | names only, as v0.4 ships today |
| `guided` | names plus an instruction not to use them as labels or headings |
| `guided_gloss` | as `guided`, plus a scope-note gloss beside the name |

**Substituting descriptions for names was tried and rejected before spending
anything.** Only 382 of 1,983 topic descriptions come from LC scope notes;
the other 1,546 are templated from tree position and pick the wrong sense --
"Information" resolves to *a topic within Criminal procedure*, "Cloud
computing" to *Electronic data processing--Distributed processing*,
"Security" to *Investments*. Substituting those would have cut verbatim
echo by generating documents about the wrong subject.

## Results

| arm | verbatim echo | topic covered (blind judge) |
|---|---|---|
| `control` | 82.2% | 92.2% |
| `guided` | **4.4%** | 85.4% |
| `guided_gloss` | 6.6% | 85.7% |

Paired bootstrap against control:

| arm | d verbatim | 95% CI | d covered | 95% CI |
|---|---|---|---|---|
| `guided` | **-77.9pp** | [-86.9, -67.8] | -7.0pp | [-15.8, **+2.1**] |
| `guided_gloss` | -75.6pp | [-84.7, -66.3] | -6.6pp | [-15.4, **+2.3**] |

## The measurement that nearly went wrong

Fidelity was first scored as cosine between a document and its topic string,
embedded with a sentence encoder. By that metric both arms lost fidelity
significantly, and the experiment read as a failure.

That metric is confounded by the manipulation. Cosine to a topic *string*
falls when the document stops containing that string, whether or not the
document still covers the subject -- which is exactly what the treatment was
designed to cause. It cannot separate "suppressed the word" from "lost the
subject."

Replacing it with a blind judge, shown one topic at a time, never told which
arm produced the text, and told explicitly that the topic words may be absent
and that their absence is not grounds to answer no, reversed the reading.

## What to conclude

**Adopt `guided`.** An 18-fold reduction in verbatim echo for a coverage
change that is not statistically distinguishable from zero.

**Do not describe fidelity as preserved.** The point estimate is -7pp with an
upper bound of +2.1 at n = 90 per arm. That is underpowered to resolve a
7-point effect, not evidence of no effect. Before regenerating a corpus, run
a larger confirmation -- 300 or more judgements per arm -- so a real 5-to-7
point coverage cost would be detected if it exists.

**Drop the gloss.** `guided_gloss` matches `guided` on both axes, so the
scope-note lookup adds cost and complexity for nothing.

**Single generator.** These numbers are one model. The second model's
documents are generated but not yet judged, and the effect could be
model-dependent; instructions asking a model not to use particular words have
a mixed record across families.
