# Taxonomy enrichment coverage

Generated: 2026-08-27T01:58:12.773473+00:00
Script version: 1.0.0

Source strategy: bulk SKOS/RDF dumps for LCSH and LCGFT (one request each, streamed); the id.loc.gov classification API for LCC (no bulk download exists). The LC Name Authority File bulk dump (1.69 GB) is not used -- NAF place records carry no definitions.

| Taxonomy | Labels | Matched | Any description | Verbatim LC scope note | No description |
|---|---:|---:|---:|---:|---:|
| lcsh_topical | 2000 | 1966 | 1932 | 369 | 68 |
| lcsh_geo | 500 | 497 | 296 | 45 | 204 |
| lcgft | 554 | 474 | 451 | 162 | 103 |
| curated_forms | 133 | 123 | 123 | 53 | 10 |
| curated_topics | 112 | 103 | 94 | 32 | 18 |
| curated_geographic | 44 | 35 | 8 | 3 | 36 |
| lcc_subclass | 100 | 96 | 96 | 18 | 4 |

## Fallback chain

1. `scope_note` -- LC's own prose, boilerplate openings and
   cross-reference tails stripped. Verbatim.
2. `hierarchy` (LCSH/LCGFT) / `caption_hierarchy` (LCC) -- a
   deterministic template over LC broader/narrower labels or over
   the classification caption plus its widest subdivisions.
3. `variant_labels` -- LC variant labels, excluding those that are
   only re-orderings of the label itself.
4. `none` -- no LC record, or a record with no usable content. The
   consumer must keep its existing fallback for these.

## Per-taxonomy detail

### lcsh_topical

- description sources: `{'hierarchy': 1515, 'none': 68, 'scope_note': 369, 'variant_labels': 48}`
- match kinds: `{'lcsh:alt_label': 11, 'lcsh:pref_label': 1954, 'lcsh:pref_label_variant': 1, 'none': 34}`
- descriptions containing their own label (self-labeling risk): 63
- unmatched examples: Diplomatic relations, Computerized simulation, Design analysis, Wind tunnel tests, Flow distribution, Data acquisition, Government Operations and Politics, Applications programs (computers), Structural analysis, Flight tests

### lcsh_geo

- description sources: `{'hierarchy': 245, 'none': 204, 'scope_note': 45, 'variant_labels': 6}`
- match kinds: `{'lcsh:alt_label': 4, 'lcsh:alt_label_variant': 2, 'lcsh:pref_label': 292, 'lcsh:pref_label_variant': 1, 'naf:label_service': 198, 'none': 3}`
- descriptions containing their own label (self-labeling risk): 7
- unmatched examples: United States, West, West United States, Washington (D.C.) Metropolitan Area

### lcgft

- description sources: `{'hierarchy': 272, 'none': 103, 'scope_note': 162, 'variant_labels': 17}`
- match kinds: `{'lcgft:alt_label': 19, 'lcgft:alt_label_variant': 5, 'lcgft:pref_label': 296, 'lcgft:pref_label_variant': 28, 'lcsh:alt_label': 10, 'lcsh:alt_label_variant': 8, 'lcsh:pref_label': 101, 'lcsh:pref_label_variant': 7, 'none': 80}`
- descriptions containing their own label (self-labeling risk): 7
- unmatched examples: Online resources, Juvenile works, Statistiques, Handbook, Audiences législatives, Documents législatifs, Census, 1990, United States, Atlas, Guideline

### curated_forms

- description sources: `{'hierarchy': 69, 'none': 10, 'scope_note': 53, 'variant_labels': 1}`
- match kinds: `{'lcgft:alt_label': 7, 'lcgft:pref_label': 103, 'lcgft:pref_label_variant': 1, 'lcsh:alt_label': 3, 'lcsh:pref_label': 9, 'none': 10}`
- descriptions containing their own label (self-labeling risk): 4
- unmatched examples: Anniversary publications, Flyers, Music recordings, News articles, Opinion pieces, Profiles, Satellite imagery, Theological works, Tributes, Tutorials

### curated_topics

- description sources: `{'hierarchy': 55, 'none': 18, 'scope_note': 32, 'variant_labels': 7}`
- match kinds: `{'lcsh:alt_label': 11, 'lcsh:pref_label': 89, 'lcsh:pref_label_variant': 3, 'none': 9}`
- descriptions containing their own label (self-labeling risk): 10
- unmatched examples: Carbon emissions, Data science, Innovation, Knowledge, Networks, Ocean conservation, Operations, Renewable energy, Supply chain

### curated_geographic

- description sources: `{'hierarchy': 2, 'none': 36, 'scope_note': 3, 'variant_labels': 3}`
- match kinds: `{'lcsh:pref_label': 8, 'lcsh:pref_label_variant': 1, 'naf:label_service': 26, 'none': 9}`
- descriptions containing their own label (self-labeling risk): 0
- unmatched examples: Beijing, Berlin, Chicago, Los Angeles, Mumbai, New York, New York City, São Paulo, Tokyo

### lcc_subclass

- description sources: `{'caption_hierarchy': 78, 'none': 4, 'scope_note': 18}`
- match kinds: `{'classification_api': 96, 'none': 4}`
- descriptions containing their own label (self-labeling risk): 0
- unmatched examples: JX, IN, PAR, NOT

