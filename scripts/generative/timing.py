"""Compare inference rate across the three families on matched work.

The stored encoder results carry a FLOPs estimate but `throughput_bytes_sec`
is null, so nothing in the paper measures what any arm costs to run.

Fairness is the whole difficulty here, and three things are controlled:

* **Same documents.** One sample, one order, every arm.
* **Same token budget.** A decoder at 2048 tokens does four times the work of
  one at 512, so every arm is timed at 512 tokens of its own tokenizer. Each
  decoder is also timed at the budget its frozen configuration actually uses,
  because that is the rate its reported score was produced at.
* **Same hardware**, one card, with a warm-up pass excluded so model load and
  CUDA context creation are not charged to the first arm measured.

What is timed is inference: an already-fitted model turning documents into
predictions. Fitting the vectoriser and the logistic head is one-off and is
reported separately rather than amortised into the rate.

One asymmetry is not removed and must be read with the numbers. Encoders are
batched, which is how they are deployed and how the paper's scores were
produced. The decoder harness scores one document at a time, because each
needs its own prompt and its own final-position logits. A batched decoder
implementation would narrow the gap; these rates are for the harness as run,
not a claim about the best achievable decoder throughput.

Usage:
    uv run python scripts/generative/timing.py --n 1000
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

MATCHED_BUDGET = 512

ENCODERS = [
    ("MiniLM-L6", "sentence-transformers/all-MiniLM-L6-v2", 22_700_000),
    ("BGE-small", "BAAI/bge-small-en-v1.5", 33_400_000),
    ("EmbeddingGemma-300M", "google/embeddinggemma-300m", 307_600_000),
]
DECODERS = [
    ("Qwen3.5-0.8B", "Qwen/Qwen3.5-0.8B", "letter_forced", 2048, 870_000_000),
    ("Qwen3.5-2B", "Qwen/Qwen3.5-2B", "cataloguer", 0, 2_270_000_000),
    ("Gemma-4-E2B", "google/gemma-4-E2B-it", "cataloguer", 512, 5_120_000_000),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--train", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--output", default="results/generative/timing.json")
    args = ap.parse_args()

    import polars as pl
    import torch
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    te = pl.read_parquet("data/hf_dataset/all/test.parquet").head(args.n)
    tr = pl.read_parquet("data/hf_dataset/all/train.parquet").head(args.train)
    Xte = te["text"].to_list()
    Xtr, ytr = tr["text"].to_list(), tr["lcc_code"].to_list()
    rows: list[dict] = []
    logger.info(
        f"  {len(Xte)} documents, budget {MATCHED_BUDGET} tokens where applicable\n"
    )
    logger.info(f"  {'arm':<22}{'params':>9}{'docs/s':>10}{'ms/doc':>9}  note")
    logger.info("  " + "-" * 62)

    def record(family, model, params, n, secs, note, **extra):
        rows.append(
            {
                "family": family,
                "model": model,
                "params": params,
                "n": n,
                "inference_s": round(secs, 3),
                "docs_per_s": round(n / secs, 2),
                "ms_per_doc": round(1000 * secs / n, 2),
                "note": note,
                **extra,
            }
        )
        p = f"{params / 1e6:.0f}M" if params < 1e9 else f"{params / 1e9:.2f}B"
        logger.info(
            f"  {model:<22}{p:>9}{n / secs:>10.1f}{1000 * secs / n:>9.2f}  {note}"
        )

    # ---- lexical -----------------------------------------------------
    vec = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2))
    t0 = time.perf_counter()
    clf = LogisticRegression(max_iter=1000).fit(vec.fit_transform(Xtr), ytr)
    fit_s = time.perf_counter() - t0
    vec.transform(Xte[:50])  # warm-up
    t0 = time.perf_counter()
    clf.predict(vec.transform(Xte))
    record(
        "lexical",
        "TF-IDF + logistic",
        0,
        len(Xte),
        time.perf_counter() - t0,
        "cpu, whole document",
        one_off_fit_s=round(fit_s, 1),
    )

    # ---- encoders ----------------------------------------------------
    from sentence_transformers import SentenceTransformer

    for name, path, params in ENCODERS:
        try:
            m = SentenceTransformer(path, device="cuda")
            m.max_seq_length = min(m.max_seq_length, MATCHED_BUDGET)
            head = LogisticRegression(max_iter=1000).fit(
                m.encode(Xtr[:2000], batch_size=args.batch, show_progress_bar=False),
                ytr[:2000],
            )
            m.encode(Xte[:64], batch_size=args.batch, show_progress_bar=False)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            head.predict(m.encode(Xte, batch_size=args.batch, show_progress_bar=False))
            torch.cuda.synchronize()
            record(
                "encoder+probe",
                name,
                params,
                len(Xte),
                time.perf_counter() - t0,
                f"cuda, batch {args.batch}",
                token_budget=MATCHED_BUDGET,
            )
            del m
            torch.cuda.empty_cache()
        except Exception as exc:
            logger.warning(f"  {name}: {type(exc).__name__}: {str(exc)[:80]}")

    # ---- decoders ----------------------------------------------------
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from classify import LCC, render, truncate_tokens
    from transformers import AutoModelForCausalLM, AutoTokenizer

    codes = list(LCC)
    for name, path, prompt, frozen_budget, params in DECODERS:
        try:
            tok = AutoTokenizer.from_pretrained(path)
            mdl = AutoModelForCausalLM.from_pretrained(
                path, dtype=torch.bfloat16, device_map="cuda:0"
            ).eval()
            variants = [
                [
                    t[0]
                    for t in (
                        tok(f, add_special_tokens=False).input_ids for f in (c, f" {c}")
                    )
                    if len(t) == 1
                ]
                for c in codes
            ]

            def run(docs, budget, tok=tok, mdl=mdl, prompt=prompt, variants=variants):
                for d in docs:
                    s = render(prompt, truncate_tokens(tok, d, budget))
                    s = tok.apply_chat_template(
                        [{"role": "user", "content": s}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    ids = tok(s, return_tensors="pt").input_ids.to(mdl.device)
                    with torch.no_grad():
                        lg = mdl(ids, logits_to_keep=1).logits[0, -1].float()
                    lp = torch.log_softmax(lg, dim=-1)
                    torch.stack(
                        [torch.logsumexp(lp[v], dim=0) for v in variants]
                    ).argmax()

            for budget, tag in ((MATCHED_BUDGET, "matched"), (frozen_budget, "frozen")):
                if tag == "frozen" and budget == MATCHED_BUDGET:
                    continue
                run(Xte[:16], budget)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                run(Xte, budget)
                torch.cuda.synchronize()
                label = name if tag == "matched" else f"{name} (frozen)"
                record(
                    "decoder",
                    label,
                    params,
                    len(Xte),
                    time.perf_counter() - t0,
                    f"cuda, one at a time, {budget or 'full'} tok",
                    token_budget=budget,
                    config=tag,
                )
            del mdl
            torch.cuda.empty_cache()
        except Exception as exc:
            logger.warning(f"  {name}: {type(exc).__name__}: {str(exc)[:80]}")

    out = {
        "n_documents": len(Xte),
        "matched_token_budget": MATCHED_BUDGET,
        "encoder_batch_size": args.batch,
        "device": "single RTX 4070 Ti SUPER",
        "timed": "inference only; fitting the head is one-off and reported separately",
        "caveat": (
            "Encoders are batched, as deployed and as the paper's scores were "
            "produced. The decoder harness scores one document at a time "
            "because each needs its own prompt and final-position logits. A "
            "batched decoder would be faster than these rates."
        ),
        "rows": rows,
    }
    q = Path(args.output)
    q.parent.mkdir(parents=True, exist_ok=True)
    q.write_text(json.dumps(out, indent=2))
    logger.info(f"\n  wrote {q}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
