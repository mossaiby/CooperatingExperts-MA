"""
Sanity check: does the injected handoff vector actually matter?

Compares dst's next-token loss on val continuations under three conditions:
  1. REAL handoff vector (normal Phase-2 forward)
  2. ZEROED handoff vector (same code path, vector replaced with zeros)
  3. NO injection at all (dst just sees cont_ids cold, no virtual position 0)

If (1) isn't meaningfully better than (2)/(3), the bridge isn't carrying
useful signal yet, regardless of how good the training/val loss number
looks -- the model may just be leaning on its own strong language-modeling
prior for the continuation almost independent of the prefix.

Usage:
    python check_handoff_signal.py --bridge-ckpt checkpoints/model_stitched_best.pt
    python check_handoff_signal.py --bridge-ckpt checkpoints/model_stitched_best.pt --debug
"""
import argparse

import torch
import torch.nn.functional as F

from config import Config
from models import FrozenExpert
from data import load_pairs, train_val_split
from dataset import HandoffDataset, DirectionalBatcher
from train_stitch import load_bridge_checkpoint


@torch.no_grad()
def lm_loss_given_injection(dst: FrozenExpert, cont_ids, cont_mask, injected_vec):
    if injected_vec is not None:
        logits = dst.forward_with_injected_prefix(injected_vec, cont_ids, cont_mask)
    else:
        out = dst.backbone(input_ids=cont_ids, attention_mask=cont_mask)
        logits = out.logits
    targets = cont_ids.clone()
    targets[cont_mask == 0] = -100
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        targets.reshape(-1),
        ignore_index=-100,
    ).item()


@torch.no_grad()
def run_check(cfg: Config, bridge_ckpt: str, device="cuda", n_batches=30):
    experts = {
        "english": FrozenExpert(cfg.pair.english, cfg.shared, cfg.switch, cfg.handoff_layer, device),
        "python": FrozenExpert(cfg.pair.python, cfg.shared, cfg.switch, cfg.handoff_layer, device),
    }
    load_bridge_checkpoint(experts, bridge_ckpt)
    for e in experts.values():
        e.backbone.eval()

    pad_id_by_expert = {name: e.tokenizer.pad_token_id for name, e in experts.items()}
    pairs = load_pairs(cfg.data.processed_path)
    _, val_pairs = train_val_split(pairs, cfg.data.val_fraction, cfg.data.seed)
    val_ds = HandoffDataset(val_pairs, experts["english"].tokenizer,
                             experts["python"].tokenizer, cfg.stitch.max_seq_len)
    batcher = DirectionalBatcher(val_ds, cfg.stitch.batch_size, pad_id_by_expert, shuffle=False)
    gen = batcher.infinite_pairs()

    totals = {"real": 0.0, "zeroed": 0.0, "none": 0.0}
    for _ in range(n_batches):
        en2py_batch, py2en_batch = next(gen)
        for batch in (en2py_batch, py2en_batch):
            src = experts[batch["src"]]
            dst = experts[batch["dst"]]
            prefix_ids = batch["prefix_ids"].to(device)
            prefix_mask = batch["prefix_mask"].to(device)
            cont_ids = batch["cont_ids"].to(device)
            cont_mask = batch["cont_mask"].to(device)

            h_src = src.encode_handoff_vector(prefix_ids, prefix_mask)
            z = src.to_shared_space(h_src)
            h0_real = dst.from_shared_space(z)
            h0_zero = torch.zeros_like(h0_real)

            totals["real"] += lm_loss_given_injection(dst, cont_ids, cont_mask, h0_real)
            totals["zeroed"] += lm_loss_given_injection(dst, cont_ids, cont_mask, h0_zero)
            totals["none"] += lm_loss_given_injection(dst, cont_ids, cont_mask, None)

    n = 2 * n_batches
    for k in totals:
        totals[k] /= n

    print("\n=== Handoff signal check (avg cross-entropy over val continuations) ===")
    print(f"  REAL handoff vector : {totals['real']:.4f}")
    print(f"  ZEROED handoff vector: {totals['zeroed']:.4f}")
    print(f"  NO injection at all  : {totals['none']:.4f}")
    gap_zero = totals['zeroed'] - totals['real']
    gap_none = totals['none'] - totals['real']
    print(f"\n  real vs zeroed gap: {gap_zero:+.4f}  (higher = bridge matters more)")
    print(f"  real vs none gap:   {gap_none:+.4f}")
    if gap_zero < 0.05 and gap_none < 0.05:
        print("\n  !! WARNING: real handoff barely beats zeroed/no injection. "
              "The bridge may not be carrying meaningful signal yet -- the "
              "target expert is likely leaning mostly on its own language "
              "prior rather than the injected vector.")
    else:
        print("\n  OK: real handoff meaningfully beats the ablations -- the "
              "bridge appears to be carrying real signal.")
    return totals


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bridge-ckpt", required=True)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-batches", type=int, default=30)
    args = ap.parse_args()
    cfg = Config.debug() if args.debug else Config.default()
    run_check(cfg, args.bridge_ckpt, device=args.device, n_batches=args.n_batches)
