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
    print(f"Loading CodeSearchNet ({cfg.codesearchnet_config}) ...")
    ds = load_dataset(
        "code_search_net", cfg.codesearchnet_config,
        cache_dir=cfg.cache_dir, trust_remote_code=True,
    )["train"]

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
        code = row.get("func_code_string") or row.get("whole_func_string") or ""
        doc = row.get("func_documentation_string") or ""
        doc = _clean_docstring(doc)

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
