"""
Checks how many of the (docstring, code) pairs actually used for training
are stub-like (pass / ... / NotImplementedError / docstring-only bodies
with no real logic), and prints examples. This is specifically checking a
hypothesis: that the model's regression toward emitting stub bodies at
step ~335 (falling val loss but WORSE generation quality) is being
reinforced by stub functions being a non-trivial fraction of the real
CodeSearchNet training data, not just a decoding fluke.

Usage:
    python check_stub_functions.py
    python check_stub_functions.py --show 10   # print more examples
"""
import argparse
import ast
import re

from data import load_pairs
from config import DataConfig


STUB_BODY_PATTERNS = [
    re.compile(r"^\s*pass\s*$", re.MULTILINE),
    re.compile(r"^\s*\.\.\.\s*$", re.MULTILINE),
    re.compile(r"raise\s+NotImplementedError"),
]


def strip_def_and_docstring(code: str):
    """Best-effort: parse the function, return the body source with the
    docstring (if any) removed, so a docstring-only function isn't
    miscounted as having 'real' logic just because it has a long comment."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code  # fall back to raw text if it doesn't parse standalone
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(
                    getattr(body[0], "value", None), (ast.Constant, ast.Str)):
                body = body[1:]  # drop docstring node
            if not body:
                return ""  # docstring (or nothing) was the ENTIRE body
            # crude: return source lines covered by remaining body nodes
            return "\n".join(
                ast.get_source_segment(code, n) or "" for n in body
            )
    return code


def is_stub(code: str) -> bool:
    remainder = strip_def_and_docstring(code)
    if remainder.strip() == "":
        return True
    for pat in STUB_BODY_PATTERNS:
        if pat.search(remainder):
            return True
    # very short remaining body (e.g. just "return None") is also
    # stub-suspicious, though this threshold is a heuristic, not a proof
    non_empty_lines = [l for l in remainder.splitlines() if l.strip()]
    if len(non_empty_lines) <= 1 and len(remainder.strip()) < 20:
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-path", default=DataConfig.processed_path)
    ap.add_argument("--show", type=int, default=5)
    args = ap.parse_args()

    pairs = load_pairs(args.processed_path)
    print(f"Loaded {len(pairs)} pairs from {args.processed_path}")

    stubs = [p for p in pairs if is_stub(p["code"])]
    frac = len(stubs) / len(pairs) if pairs else 0.0
    print(f"\nStub-like functions: {len(stubs)} / {len(pairs)} ({100*frac:.1f}%)")

    if frac > 0.05:
        print("!! This is a meaningful fraction of the training data. "
              "Worth considering filtering these out of DataConfig / "
              "data.py's build_pairs() min_function_lines / a body-check, "
              "and regenerating handoff_pairs.jsonl, before further "
              "training -- the model may be partially learning that "
              "'pass' or a short stub is an acceptable completion for a "
              "meaningful fraction of docstrings, which would explain "
              "generation regressing toward stubs even as val loss (which "
              "doesn't distinguish stub-correctness from real-logic-"
              "correctness) keeps improving.")
    else:
        print("Stub fraction is low -- probably not the primary explanation "
              "for the step-335 regression. Worth looking elsewhere (e.g. "
              "overfitting to a different data pattern, or just sample "
              "variance -- check generations across saved checkpoints with "
              "compare_checkpoints.py before concluding anything).")

    print(f"\n=== {min(args.show, len(stubs))} example stub pairs ===")
    for p in stubs[:args.show]:
        print(f"\n--- {p['id']} ---")
        print(f"docstring: {p['docstring'][:150]!r}")
        print(f"code:\n{p['code']}")


if __name__ == "__main__":
    main()
