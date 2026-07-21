"""
Run this FIRST, before writing/training anything else.

Checks whether the two chosen pretrained models actually have "genuinely
different" tokenizers. If there's heavy overlap (e.g. two models from the
same lab sharing a tokenizer -- this happens more often than you'd expect,
e.g. Qwen2.5 vs Qwen2.5-Coder), the whole "disjoint vocabulary" framing of
this project doesn't hold for that pair, and you should pick a different one.

Usage:
    python check_tokenizer_overlap.py                 # real pair (config.Config.default())
    python check_tokenizer_overlap.py --debug          # tiny stand-in pair
    python check_tokenizer_overlap.py --a MODEL_A --b MODEL_B   # arbitrary pair
"""
import argparse
from transformers import AutoTokenizer

from config import Config


SAMPLE_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr",
    "This function returns the sum of two integers.",
    "import numpy as np\nx = np.array([1, 2, 3])",
    "Please explain how binary search works in O(log n) time.",
]


def overlap_report(tok_a, tok_b, name_a, name_b):
    vocab_a = set(tok_a.get_vocab().keys())
    vocab_b = set(tok_b.get_vocab().keys())
    shared = vocab_a & vocab_b
    smaller = min(len(vocab_a), len(vocab_b))
    overlap_frac = len(shared) / smaller

    print(f"\n=== Vocab-level overlap: {name_a} vs {name_b} ===")
    print(f"{name_a} vocab size: {len(vocab_a)}")
    print(f"{name_b} vocab size: {len(vocab_b)}")
    print(f"Shared surface-form tokens: {len(shared)} ({100 * overlap_frac:.1f}% of smaller vocab)")

    # NOTE on calibration: surface-string overlap is naturally HIGH for any
    # two general-purpose BPE tokenizers trained on English-heavy corpora --
    # common words, whitespace, and punctuation converge across almost any
    # tokenizer trained on internet-scale text, regardless of lab or domain
    # mix. A high percentage here does NOT by itself mean the tokenizers are
    # secretly the same file. What actually matters for this project:
    #   1. Vocab SIZES should differ meaningfully (rules out literal reuse)
    #   2. Even shared surface strings get different integer IDs in each
    #      tokenizer, feeding two INDEPENDENTLY TRAINED embedding matrices --
    #      so there's no shared embedding table to exploit either way.
    # The one pattern that IS a red flag: near-identical vocab size (e.g.
    # both exactly 49152) combined with high overlap -- that suggests literal
    # shared lineage, not independent convergence.
    size_ratio = smaller / max(len(vocab_a), len(vocab_b))
    if size_ratio > 0.95 and overlap_frac > 0.30:
        print("!! WARNING: near-identical vocab sizes AND high overlap -- "
              "these tokenizers likely share literal lineage (e.g. same "
              "training data/config, possibly the same file). Pick a "
              "different pair.")
    elif overlap_frac > 0.60:
        print("NOTE: high surface overlap, but vocab sizes differ meaningfully "
              "-- likely just the normal convergence of two independently "
              "trained general-purpose BPE tokenizers on English-heavy data, "
              "not shared lineage. Check the sample tokenizations below: if "
              "segmentation differs on code-heavy text even when overlap is "
              "high, the tokenizers are functioning independently.")
    else:
        print("OK: overlap and vocab sizes both look consistent with "
              "independently-trained tokenizers.")

    print(f"\n=== Sample tokenization comparison ===")
    for text in SAMPLE_TEXTS:
        ids_a = tok_a.encode(text)
        ids_b = tok_b.encode(text)
        toks_a = tok_a.convert_ids_to_tokens(ids_a)
        toks_b = tok_b.convert_ids_to_tokens(ids_b)
        print(f"\nText: {text!r}")
        print(f"  {name_a:10s} ({len(ids_a):2d} tokens): {toks_a}")
        print(f"  {name_b:10s} ({len(ids_b):2d} tokens): {toks_b}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true", help="use tiny debug model pair")
    ap.add_argument("--a", type=str, default=None, help="override model A hf id")
    ap.add_argument("--b", type=str, default=None, help="override model B hf id")
    args = ap.parse_args()

    cfg = Config.debug() if args.debug else Config.default()

    id_a = args.a or cfg.pair.english.hf_model_id
    id_b = args.b or cfg.pair.python.hf_model_id
    name_a = cfg.pair.english.name if not args.a else "A"
    name_b = cfg.pair.python.name if not args.b else "B"

    print(f"Loading tokenizers:\n  {name_a}: {id_a}\n  {name_b}: {id_b}")
    tok_a = AutoTokenizer.from_pretrained(id_a)
    tok_b = AutoTokenizer.from_pretrained(id_b)

    overlap_report(tok_a, tok_b, name_a, name_b)


if __name__ == "__main__":
    main()
