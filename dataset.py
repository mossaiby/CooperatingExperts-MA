"""
HandoffDataset for Phase 2 (stitching).

Each raw pair is (docstring, code). We build BOTH directions:
  english -> python : prefix = docstring (english tokenizer),
                       continuation = code (python tokenizer)
  python -> english : prefix = code (python tokenizer),
                       continuation = docstring (english tokenizer)

This mirrors v1's HandoffDataset but tokenizes each side with a different
HF tokenizer instead of a shared from-scratch one.
"""
import torch
from torch.utils.data import Dataset


class HandoffDataset(Dataset):
    def __init__(self, pairs, english_tokenizer, python_tokenizer, max_seq_len=256):
        self.pairs = pairs
        self.tok_en = english_tokenizer
        self.tok_py = python_tokenizer
        self.max_seq_len = max_seq_len

        # each raw pair yields two directed examples
        self.examples = []
        for p in pairs:
            self.examples.append({"direction": "en2py", "prefix": p["docstring"], "cont": p["code"]})
            self.examples.append({"direction": "py2en", "prefix": p["code"], "cont": p["docstring"]})

    def __len__(self):
        return len(self.examples)

    def _encode(self, tokenizer, text):
        ids = tokenizer(
            text, truncation=True, max_length=self.max_seq_len,
            return_tensors=None,
        )["input_ids"]
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        if ex["direction"] == "en2py":
            prefix_ids = self._encode(self.tok_en, ex["prefix"])
            cont_ids = self._encode(self.tok_py, ex["cont"])
            src, dst = "english", "python"
        else:
            prefix_ids = self._encode(self.tok_py, ex["prefix"])
            cont_ids = self._encode(self.tok_en, ex["cont"])
            src, dst = "python", "english"
        return {
            "src": src, "dst": dst,
            "prefix_ids": prefix_ids, "cont_ids": cont_ids,
        }


def collate_handoff(batch, pad_id_by_expert):
    """
    Batches a list of same-direction examples. Caller is expected to group
    by direction (see train_stitch.py's per-direction dataloaders) since
    prefix/cont come from different tokenizers per direction and padding
    ids differ per expert.
    """
    src = batch[0]["src"]
    dst = batch[0]["dst"]
    assert all(b["src"] == src and b["dst"] == dst for b in batch), \
        "collate_handoff expects a single-direction batch"

    prefix_pad = pad_id_by_expert[src]
    cont_pad = pad_id_by_expert[dst]

    max_prefix = max(b["prefix_ids"].shape[0] for b in batch)
    max_cont = max(b["cont_ids"].shape[0] for b in batch)

    prefix_ids = torch.full((len(batch), max_prefix), prefix_pad, dtype=torch.long)
    prefix_mask = torch.zeros((len(batch), max_prefix), dtype=torch.long)
    cont_ids = torch.full((len(batch), max_cont), cont_pad, dtype=torch.long)
    cont_mask = torch.zeros((len(batch), max_cont), dtype=torch.long)

    for i, b in enumerate(batch):
        L = b["prefix_ids"].shape[0]
        prefix_ids[i, :L] = b["prefix_ids"]
        prefix_mask[i, :L] = 1
        Lc = b["cont_ids"].shape[0]
        cont_ids[i, :Lc] = b["cont_ids"]
        cont_mask[i, :Lc] = 1

    return {
        "src": src, "dst": dst,
        "prefix_ids": prefix_ids, "prefix_mask": prefix_mask,
        "cont_ids": cont_ids, "cont_mask": cont_mask,
    }


class DirectionalBatcher:
    """
    Simple helper: splits a HandoffDataset's examples into two lists (one per
    direction) and yields batches by cycling both, so a single training step
    can always get one en2py batch and one py2en batch (matches v1's
    "both directions per step" behaviour).
    """
    def __init__(self, dataset: HandoffDataset, batch_size: int, pad_id_by_expert,
                 shuffle=True, seed=0):
        import random
        self.pad_id_by_expert = pad_id_by_expert
        self.batch_size = batch_size
        self.rng = random.Random(seed)

        self.dataset = dataset
        idx_en2py, idx_py2en = [], []
        for i, ex in enumerate(dataset.examples):
            (idx_en2py if ex["direction"] == "en2py" else idx_py2en).append(i)
        self.idx_by_dir = {"en2py": idx_en2py, "py2en": idx_py2en}
        self.shuffle = shuffle

    def _cycle(self, direction):
        idxs = self.idx_by_dir[direction][:]
        while True:
            if self.shuffle:
                self.rng.shuffle(idxs)
            for i in range(0, len(idxs), self.batch_size):
                batch_idx = idxs[i:i + self.batch_size]
                if len(batch_idx) < 1:
                    continue
                items = [self.dataset[j] for j in batch_idx]
                yield collate_handoff(items, self.pad_id_by_expert)

    def infinite_pairs(self):
        """Yields (en2py_batch, py2en_batch) forever."""
        gen_a = self._cycle("en2py")
        gen_b = self._cycle("py2en")
        while True:
            yield next(gen_a), next(gen_b)
