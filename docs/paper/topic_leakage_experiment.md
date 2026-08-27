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

Verbatim echo, by generator:

| model | control | `guided` | `guided_gloss` |
|---|---|---|---|
| gemini-3.7-flash | 82.2% | **4.4%** | 6.6% |
| claude-haiku-4-5 | 61.5% | 23.1% | 26.4% |

**The effect is strongly model-dependent.** Gemini's echo falls 95%,
Haiku's 62%, from different baselines. An instruction not to use particular
words does not land the same way across model families.

Topic coverage, blind judge, with a bootstrap interval against control:

| model | arm | n | covered | change | 95% CI |
|---|---|---|---|---|---|
| gemini | control | 90 | 92.2% | — | — |
| gemini | `guided` | 89 | 85.4% | -7.0pp | [-15.8, +2.1] |
| gemini | `guided_gloss` | 91 | 85.7% | -6.6pp | [-15.4, +2.3] |
| haiku | control | 67 | 82.1% | — | — |
| haiku | `guided` | 75 | 61.3% | **-20.7pp** | [-34.7, **-6.5**] |
| haiku | `guided_gloss` | 77 | 72.7% | -9.6pp | [-23.3, +4.2] |
| **pooled** | `guided` | 164 | 74.4% | **-13.5pp** | [-22.2, **-4.9**] |
| **pooled** | `guided_gloss` | 168 | 79.8% | **-8.1pp** | [-16.0, **-0.1**] |

Haiku judging was at 214 of 272 when this was written; the numbers may move
slightly.

## Correction

**An earlier version of this file recommended `guided` and advised dropping
the gloss. That was wrong, and it was wrong because it rested on one
model.** On gemini alone the coverage change was not distinguishable from
zero, and `guided_gloss` looked like a pointless complication. Adding a
second generator reversed both readings: `guided` costs 20.7 points of topic
coverage on Haiku, and pooled across models both arms now show a
*significant* coverage loss.

## What to conclude

**Prefer `guided_gloss` over `guided`.** It keeps most of the leakage
reduction at roughly half the coverage cost (-8.1pp against -13.5pp pooled;
-9.6pp against -20.7pp on Haiku). The mechanism is plausible: telling a model
not to name a subject, without telling it what the subject is, invites drift.
A scope-note gloss gives it something concrete to write about instead. This
is the opposite of the earlier recommendation.

**Neither arm is free.** Both cost real topic coverage once more than one
generator is measured. A full regeneration must weigh an 18-fold leakage
reduction against roughly 8 points of coverage, and that is a judgement call
rather than a measurement.

**Report the leakage reduction per generator, not pooled.** A 95% reduction
on one family and 62% on another averages into a number describing neither.

## Method note that generalises

Fidelity was first scored as cosine between a document and its topic string.
By that metric both arms failed. The metric is confounded by the
manipulation: cosine to a *string* falls when the document stops containing
that string, whether or not the subject survives. Replacing it with a blind
judge -- one topic at a time, arm hidden, told the topic words may be absent
and that their absence is not grounds to answer no -- changed the reading.

The general lesson is that an automatic metric sharing vocabulary with the
treatment cannot referee that treatment.
