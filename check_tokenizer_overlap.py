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

    print(f"\n=== Vocab-level overlap: {name_a} vs {name_b} ===")
    print(f"{name_a} vocab size: {len(vocab_a)}")
    print(f"{name_b} vocab size: {len(vocab_b)}")
    print(f"Shared surface-form tokens: {len(shared)} "
          f"({100 * len(shared) / min(len(vocab_a), len(vocab_b)):.1f}% of smaller vocab)")
    if len(shared) / min(len(vocab_a), len(vocab_b)) > 0.3:
        print("!! WARNING: >30% overlap -- these tokenizers may be too similar "
              "for the 'genuinely different vocabulary' premise of this project.")
    else:
        print("OK: overlap looks low enough to treat these as genuinely different vocabularies.")

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
