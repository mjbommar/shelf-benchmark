"""Classify SHELF documents with a small decoder, by scoring labels not parsing text.

The paper reports macro-F1 for encoders. To compare, a decoder must produce
exactly one label from a closed set. Free generation does not: it emits
"Class H", "H.", "Social Sciences", or a refusal, and every parser for that
adds error belonging to the parser rather than the model. So each candidate
label is scored and the highest-likelihood one wins. The prediction is always
in the label set and a weak model is measured as weak, not as malformed.

Two corrections, both found by a degenerate first run that used one class of
21 and scored 0.0048:

* Instruct checkpoints are scored through their chat template. Scoring a raw
  completion asks the model a question in a format it was not tuned for.
* Label priors are calibrated. Uncalibrated, this model ranks ``A`` first for
  every document by roughly 3.5 nats, which is its prior rather than a
  decision. Calibration scores the labels against a content-free prompt and
  subtracts, following Zhao et al. 2021. Raw and calibrated are both kept:
  the size of the correction is itself a finding.

Every LCC class is one token, so all 21 are read from a single forward pass.

Usage:
    uv run python scripts/generative/classify.py --model Qwen/Qwen3.5-0.8B \\
        --prompt letter_only --chat --calibrate --n 300
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

LCC = {
    "A": "General Works",
    "B": "Philosophy, Psychology, Religion",
    "C": "Auxiliary Sciences of History",
    "D": "World History (except Americas)",
    "E": "History of the Americas (general, US)",
    "F": "History of the Americas (local)",
    "G": "Geography, Anthropology, Recreation",
    "H": "Social Sciences",
    "J": "Political Science",
    "K": "Law",
    "L": "Education",
    "M": "Music",
    "N": "Fine Arts",
    "P": "Language and Literature",
    "Q": "Science",
    "R": "Medicine",
    "S": "Agriculture",
    "T": "Technology",
    "U": "Military Science",
    "V": "Naval Science",
    "Z": "Bibliography, Library Science",
}
MENU = "\n".join(f"{c} = {n}" for c, n in LCC.items())

# Content-free stand-in for the document, for the calibration pass.
NULL_DOC = "N/A"


def truncate_tokens(tok, text: str, budget: int) -> str:
    """Cut to a token budget, not a word count.

    Words are the wrong unit: 1.47 tokens per word here, and the length
    distribution is heavy-tailed (median 456 tokens, mean 947, max 6341), so a
    fixed word count truncates documents unevenly. A budget of 0 means the
    whole document, which is what a decoder with a long context can actually
    read, and 512 or 2048 match the caps the encoders in the paper run under.
    """
    if budget <= 0:
        return text
    ids = tok.encode(text, add_special_tokens=False)
    return text if len(ids) <= budget else tok.decode(ids[:budget])


def render(style: str, text: str) -> str:
    if style == "letter_only":
        return (
            f"Document:\n{text}\n\nLibrary of Congress classes:\n{MENU}\n\n"
            "The single best class letter is:"
        )
    if style == "no_menu":
        return (
            f"Document:\n{text}\n\nThe single best Library of Congress class letter is:"
        )
    if style == "cataloguer":
        return (
            "You are a Library of Congress cataloguer assigning a subject class.\n\n"
            f"Document:\n{text}\n\nClasses:\n{MENU}\n\nAssigned class letter:"
        )
    if style == "terse":
        return f"{MENU}\n\nText:\n{text}\n\nClass:"
    if style == "letter_forced":
        # The model's preferred continuations are "Based", "The", "To": it
        # wants to write a sentence. Demanding a bare letter moves mass onto
        # the label set instead of measuring format compliance.
        return (
            f"Document:\n{text}\n\nLibrary of Congress classes:\n{MENU}\n\n"
            "Reply with exactly one capital letter from the list above and "
            "nothing else.\nAnswer:"
        )
    raise SystemExit(f"unknown prompt style: {style}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="letter_only")
    ap.add_argument("--words", type=int, default=250)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chat", action="store_true", help="use the chat template")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    import polars as pl
    import torch
    from sklearn.metrics import f1_score
    from transformers import AutoModelForCausalLM, AutoTokenizer

    df = pl.read_parquet(f"data/hf_dataset/all/{args.split}.parquet")
    df = df.sample(n=min(args.n, len(df)), seed=args.seed)
    docs, gold = df["text"].to_list(), df["lcc_code"].to_list()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0"
    ).eval()

    codes = list(LCC)
    # Single-token ids for each class letter, with the leading space the
    # model would actually emit after a colon.
    tid = []
    for c in codes:
        ids = tok(f" {c}", add_special_tokens=False).input_ids
        if len(ids) != 1:
            raise SystemExit(f"class {c} is not one token: {ids}")
        tid.append(ids[0])

    def wrap(prompt: str) -> str:
        if not args.chat:
            return prompt
        return tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def label_logprobs(text: str) -> torch.Tensor:
        s = wrap(render(args.prompt, text))
        ids = tok(s, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            # Only the final position is ever read. Without logits_to_keep the
            # model materialises logits for every position: a 6341-token
            # document against Gemma's 262k vocabulary is 3.3 GB in bf16, and
            # .float() doubles it. That is what OOM'd a 16 GB card.
            lg = model(ids, logits_to_keep=1).logits[0, -1].float()
        return torch.log_softmax(lg, dim=-1)[tid]

    bias = label_logprobs(NULL_DOC) if args.calibrate else None

    raw_pred, cal_pred = [], []
    t0 = time.time()
    for i, doc in enumerate(docs):
        lp = label_logprobs(" ".join(doc.split()[: args.words]))
        raw_pred.append(codes[int(lp.argmax())])
        cal_pred.append(
            codes[int((lp - bias).argmax())] if bias is not None else raw_pred[-1]
        )
        if (i + 1) % 100 == 0:
            logger.info(
                f"    {i + 1}/{len(docs)}  {(time.time() - t0) / (i + 1):.3f}s/doc"
            )

    def score(p: list[str]) -> dict:
        return {
            "macro_f1": f1_score(gold, p, average="macro", zero_division=0.0),
            "accuracy": sum(a == b for a, b in zip(gold, p, strict=True)) / len(gold),
            "classes_used": len(set(p)),
        }

    out = {
        "model": args.model,
        "prompt_style": args.prompt,
        "chat_template": args.chat,
        "calibrated": args.calibrate,
        "words": args.words,
        "n": len(docs),
        "split": args.split,
        "seed": args.seed,
        "raw": score(raw_pred),
        "calibrated_scores": score(cal_pred) if bias is not None else None,
        "seconds": round(time.time() - t0, 1),
    }
    r = out["raw"]
    c = out["calibrated_scores"]
    logger.info(
        f"  {args.model.split('/')[-1]:<18} {args.prompt:<12} "
        f"chat={int(args.chat)} w={args.words:<4} "
        f"raw F1={r['macro_f1']:.4f} ({r['classes_used']}/21)"
        + (f"  cal F1={c['macro_f1']:.4f} ({c['classes_used']}/21)" if c else "")
    )
    if args.output:
        p = Path(args.output)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
