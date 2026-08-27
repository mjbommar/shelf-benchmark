## The `all` config

A single pooled corpus of every synthetic SHELF document: the v0.3.1
`default` corpus plus every v0.4 slice.

| | documents |
|---|---|
| `default` (v0.3.1) | 42,532 |
| `v0_4_core` | 18,345 |
| `v0_4_supplement` | 1,043 |
| `v0_4_minimal_pairs` | 687 |
| `v0_4_holdout` | 292 |
| **`all`** | **62,899** |

Splits: train 37,795 / validation 12,600 / test 12,504. Each
document keeps the split it was assigned in its source config.

**This config is not generator balanced, and that is the trade.** Pooling
returns the largest generator to 47.7% of the corpus, against
9.2% in `v0_4_core`. Use `all` when sample count matters more than
balance, and `v0_4_core` when it does not. Reporting a generator-sensitive
result on `all` without saying so would be misleading.

Every row carries `source_config` and `source_version`, so any component
slice can be recovered exactly:

```python
from datasets import load_dataset
ds = load_dataset("mjbommar/SHELF", "all")
core = ds["train"].filter(lambda r: r["source_config"] == "v0_4_core")
```

Schema is the union of both generations (44 columns), so no column is
dropped; columns absent from a source are null. `text` is always
populated. Provider routing prefixes are normalised, so one model is one
id. Titles carrying a leading markdown heading or `Title:` label were
cleaned (169 rows). Deduplicated on normalised body text: zero duplicates
were found across the two corpora, as expected from disjoint spec blocks.

**The Gutenberg transfer control is deliberately excluded.** It is natural
text used to measure whether SHELF scores transfer, and pooling it into
the corpus would destroy that measurement. It remains a separate config.
