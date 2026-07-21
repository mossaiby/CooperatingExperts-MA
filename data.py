"""
Loads CodeSearchNet Python, cleans/dedups it, and writes handoff pairs:
each pair is (docstring, function_body) used directly as Phase-2
(English-prefix -> Python-continuation) and, reversed, (Python-prefix ->
English-continuation) training data.

No LLM generation step needed -- this is what lets you skip the local-Gemma
bottleneck entirely for Phase 2.
"""
import json
import os
import random
import re
from dataclasses import asdict

from datasets import load_dataset

from config import DataConfig


def _clean_docstring(doc: str) -> str:
    doc = doc.strip()
    doc = re.sub(r"\s+", " ", doc)
    return doc


def _line_count(code: str) -> int:
    return len([l for l in code.splitlines() if l.strip()])


def build_pairs(cfg: DataConfig):
    """
    Loads a CodeSearchNet Python mirror and extracts (docstring, code) pairs.

    NOTE: the original `code_search_net` dataset repo on the HF hub ships as
    a legacy loading SCRIPT, and `datasets>=4.0` dropped script-based
    dataset support entirely (not just deprecated -- it's a hard error now,
    `HfUriError`/`trust_remote_code` no longer works around it). We use a
    pre-converted, script-free mirror instead. Field names vary slightly
    across mirrors, so this pulls from several plausible column names
    rather than hardcoding one.
    """
    print(f"Loading CodeSearchNet Python mirror ({cfg.hf_dataset_id}) ...")
    ds = load_dataset(cfg.hf_dataset_id, split=cfg.hf_dataset_split,
                       cache_dir=cfg.cache_dir)

    code_field_candidates = ["code", "func_code_string", "whole_func_string", "content"]
    doc_field_candidates = ["docstring", "func_documentation_string", "summary", "description"]

    cols = set(ds.column_names)
    code_field = next((f for f in code_field_candidates if f in cols), None)
    doc_field = next((f for f in doc_field_candidates if f in cols), None)
    if code_field is None or doc_field is None:
        raise ValueError(
            f"Couldn't find recognizable code/docstring columns in {cfg.hf_dataset_id}. "
            f"Available columns: {sorted(cols)}. Update code_field_candidates/"
            f"doc_field_candidates in data.py to match this mirror's schema."
        )
    print(f"Using columns: code={code_field!r}, docstring={doc_field!r}")

    seen_funcs = set()
    seen_docs = set()
    pairs = []
    rng = random.Random(cfg.seed)

    indices = list(range(len(ds)))
    rng.shuffle(indices)

    for idx in indices:
        if len(pairs) >= cfg.n_pairs:
            break
        row = ds[idx]
        code = row.get(code_field) or ""
        doc = _clean_docstring(row.get(doc_field) or "")

        if len(doc.split()) < cfg.min_docstring_words:
            continue
        n_lines = _line_count(code)
        if n_lines < cfg.min_function_lines or n_lines > cfg.max_function_lines:
            continue

        # cheap dedup: exact-match on normalized code / docstring
        code_key = re.sub(r"\s+", "", code)
        if code_key in seen_funcs or doc in seen_docs:
            continue
        seen_funcs.add(code_key)
        seen_docs.add(doc)

        pairs.append({
            "id": f"csn_{idx:07d}",
            "docstring": doc,
            "code": code.strip(),
        })

    print(f"Collected {len(pairs)} clean, deduped pairs "
          f"(requested {cfg.n_pairs}).")
    return pairs


def write_pairs(pairs, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"Wrote {len(pairs)} pairs to {path}")


def load_pairs(path: str):
    pairs = []
    with open(path) as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


def train_val_split(pairs, val_fraction: float, seed: int):
    rng = random.Random(seed)
    shuffled = pairs[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_fraction))
    return shuffled[n_val:], shuffled[:n_val]


if __name__ == "__main__":
    cfg = DataConfig()
    pairs = build_pairs(cfg)
    write_pairs(pairs, cfg.processed_path)
