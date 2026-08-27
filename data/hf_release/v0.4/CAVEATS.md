## Known limitations in the v0.4 slices

These are measured, not suspected, and are published alongside the data rather
than left for a reader to discover.

**No subclass tier.** An LCC subclass tier was planned and is not shipped: the
specification blocks assigned 80 subclasses but generation used the parent-class
description, so those documents carry 16 parent classes and no subclass label.
They ship as `v0_4_supplement`.

**Empty-document rate.** `v0_4_supplement` and `v0_4_minimal_pairs` were generated
before a fix for a reasoning-budget defect landed: on short length targets,
reasoning tokens consumed the entire output cap, yielding a title truncated
mid-word and no body. 13-15% of raw generations were affected. Those documents
are removed by the QC gates, so the published slices are clean, but they are
~14% smaller than their nominal spec count.

**Mild generator confound on register and length.** In `v0_4_core`, generator is
independent of the labels that matter -- LCC class (Cramer's V 0.018) and LCGFT
category (0.028) -- but is measurably correlated with `register` (0.062),
`target_length` (0.066) and `prompt_variant_id` (0.085). The cause is
non-uniform QC removal: the empty-body defect hit short documents hardest, and
two generators failed entirely for part of the run. Effect sizes are small but
the p-values are unambiguous. Any analysis conditioned on register or length
carries this confound.

**Do not pool `transfer_gutenberg` with synthetic slices.** It is in the
pretraining data of essentially every model that would be evaluated on it. SHELF
is the clean-synthetic condition; Gutenberg is the contaminated-natural one; the
gap between them is the measurement. Pooling them measures neither.

**Prompt variants differ from v0.3.1.** v0.4 documents were generated with four
new system-prompt variants and form-conditional output formatting; `default` was
generated with a single prompt. `prompt_variant_id` records which. A controlled
A/B measured spurious markdown on non-markdown forms dropping from 26.7% to
~1.3% (Fisher exact p < 0.00001).
