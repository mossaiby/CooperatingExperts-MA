"""
Phase 2: joint stitching.

Backbones are frozen (see models.FrozenExpert). Only the bridge parameters
train: to_shared / from_shared for both experts, plus the handful of new
<switch:*> embedding rows.

Loss per direction (src -> dst), mirrors v1's joint_loss:
  1. Encode src's prefix, take handoff hidden vector h_src.
  2. z = src.to_shared_space(h_src)
  3. h0 = dst.from_shared_space(z)
  4. Run dst on cont_ids with h0 injected as virtual position 0.
  5. LM loss: dst's next-token loss on cont_ids.
  6. Alignment regularizer: ||from_shared(to_shared(h)) - h||^2 for both
     src and dst's own last hidden state (keeps projections ~invertible).
Total = lm_loss + align_weight * (align_src + align_dst), summed over both
directions per step (en2py and py2en), matching v1.
"""
import os
import time

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from config import Config
from models import FrozenExpert
from data import load_pairs, train_val_split
from dataset import HandoffDataset, DirectionalBatcher


def _cosine_warmup_lr(step, warmup_steps, total_steps, base_lr):
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    import math
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * base_lr * (1 + math.cos(math.pi * min(progress, 1.0)))


def directed_loss(experts, batch, align_weight, device):
    src = experts[batch["src"]]
    dst = experts[batch["dst"]]

    prefix_ids = batch["prefix_ids"].to(device)
    prefix_mask = batch["prefix_mask"].to(device)
    cont_ids = batch["cont_ids"].to(device)
    cont_mask = batch["cont_mask"].to(device)

    h_src = src.encode_handoff_vector(prefix_ids, prefix_mask)  # [B, H_src], float32
    z = src.to_shared_space(h_src)                              # [B, shared_dim]
    h0 = dst.from_shared_space(z)                               # [B, H_dst]

    logits = dst.forward_with_injected_prefix(h0, cont_ids, cont_mask)  # [B, T, V]

    # mask out padded continuation positions from the loss
    targets = cont_ids.clone()
    targets[cont_mask == 0] = -100
    lm_loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        targets.reshape(-1),
        ignore_index=-100,
    )

    # alignment regularizer on both experts' own round-trip
    h_dst = dst.encode_handoff_vector(cont_ids, cont_mask)
    align_src = F.mse_loss(src.from_shared_space(src.to_shared_space(h_src)).float(), h_src)
    align_dst = F.mse_loss(dst.from_shared_space(dst.to_shared_space(h_dst)).float(), h_dst)

    total = lm_loss + align_weight * (align_src + align_dst)
    return total, lm_loss.item(), (align_src.item() + align_dst.item())


def train_stitch(cfg: Config, device="cuda"):
    os.makedirs(cfg.stitch.ckpt_dir, exist_ok=True)

    print("Loading frozen experts ...")
    experts = {
        "english": FrozenExpert(cfg.pair.english, cfg.shared, cfg.switch, cfg.handoff_layer, device),
        "python": FrozenExpert(cfg.pair.python, cfg.shared, cfg.switch, cfg.handoff_layer, device),
    }
    if cfg.stitch.grad_checkpointing:
        for e in experts.values():
            e.gradient_checkpointing_enable()

    pad_id_by_expert = {name: e.tokenizer.pad_token_id for name, e in experts.items()}

    print("Loading data ...")
    pairs = load_pairs(cfg.data.processed_path)
    train_pairs, val_pairs = train_val_split(pairs, cfg.data.val_fraction, cfg.data.seed)
    print(f"train pairs: {len(train_pairs)}, val pairs: {len(val_pairs)}")

    train_ds = HandoffDataset(train_pairs, experts["english"].tokenizer,
                               experts["python"].tokenizer, cfg.stitch.max_seq_len)
    val_ds = HandoffDataset(val_pairs, experts["english"].tokenizer,
                             experts["python"].tokenizer, cfg.stitch.max_seq_len)

    train_batcher = DirectionalBatcher(train_ds, cfg.stitch.batch_size, pad_id_by_expert, shuffle=True)
    val_batcher = DirectionalBatcher(val_ds, cfg.stitch.batch_size, pad_id_by_expert, shuffle=False)

    trainable_params = []
    for e in experts.values():
        trainable_params.extend(e.trainable_parameters())
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"Trainable bridge params: {n_trainable:,}")

    optimizer = AdamW(trainable_params, lr=cfg.stitch.lr, weight_decay=cfg.stitch.weight_decay)

    best_val = float("inf")
    patience_left = cfg.stitch.early_stop_patience
    train_gen = train_batcher.infinite_pairs()

    t0 = time.time()
    for step in range(cfg.stitch.steps_max):
        lr = _cosine_warmup_lr(step, cfg.stitch.warmup_steps, cfg.stitch.steps_max, cfg.stitch.lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad()
        step_lm_loss, step_align_loss = 0.0, 0.0
        for _ in range(cfg.stitch.grad_accum):
            en2py_batch, py2en_batch = next(train_gen)
            for batch in (en2py_batch, py2en_batch):
                loss, lm_l, align_l = directed_loss(experts, batch, cfg.stitch.align_weight, device)
                (loss / (2 * cfg.stitch.grad_accum)).backward()
                step_lm_loss += lm_l / (2 * cfg.stitch.grad_accum)
                step_align_loss += align_l / (2 * cfg.stitch.grad_accum)

        torch.nn.utils.clip_grad_norm_(trainable_params, cfg.stitch.grad_clip)
        optimizer.step()

        if step % cfg.stitch.log_every == 0:
            elapsed = time.time() - t0
            print(f"[stitch] step {step:5d}/{cfg.stitch.steps_max} "
                  f"lr={lr:.2e} lm_loss={step_lm_loss:.4f} align_loss={step_align_loss:.4f} "
                  f"({elapsed:.0f}s elapsed)")

        if step > 0 and step % cfg.stitch.val_every == 0:
            val_loss = evaluate(experts, val_batcher, cfg.stitch.align_weight, device, n_batches=20)
            print(f"[stitch] step {step:5d} VAL lm_loss={val_loss:.4f}")
            if val_loss < best_val - cfg.stitch.early_stop_min_delta:
                best_val = val_loss
                patience_left = cfg.stitch.early_stop_patience
                save_bridge_checkpoint(experts, cfg.stitch.ckpt_dir, "model_stitched_best.pt")
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print(f"[stitch] early stopping at step {step} (best val_lm_loss={best_val:.4f})")
                    break

        if step > 0 and step % cfg.stitch.ckpt_every == 0:
            save_bridge_checkpoint(experts, cfg.stitch.ckpt_dir, f"model_stitched_step{step}.pt")

    save_bridge_checkpoint(experts, cfg.stitch.ckpt_dir, "model_stitched_final.pt")
    return experts


@torch.no_grad()
def evaluate(experts, batcher, align_weight, device, n_batches=20):
    total_lm = 0.0
    gen = batcher.infinite_pairs()
    for _ in range(n_batches):
        en2py_batch, py2en_batch = next(gen)
        for batch in (en2py_batch, py2en_batch):
            _, lm_l, _ = directed_loss(experts, batch, align_weight, device)
            total_lm += lm_l
    return total_lm / (2 * n_batches)


def save_bridge_checkpoint(experts, ckpt_dir, filename):
    """
    Only saves the trainable bridge params (to_shared/from_shared + new
    embedding rows), NOT the frozen backbone weights -- those are re-loaded
    from the HF hub id at inference time, so checkpoints stay tiny (MBs, not
    GBs).
    """
    state = {}
    for name, e in experts.items():
        state[name] = {
            "to_shared": e.to_shared.state_dict(),
            "from_shared": e.from_shared.state_dict(),
            "new_token_embeddings": {
                str(i): e.backbone.get_input_embeddings().weight[i].detach().cpu()
                for i in e.new_token_ids
            },
            "hf_model_id": e.spec.hf_model_id,
            "handoff_layer_index": e.handoff_layer_index,
        }
    path = os.path.join(ckpt_dir, filename)
    torch.save(state, path)
    print(f"Saved bridge checkpoint -> {path}")


def load_bridge_checkpoint(experts, path):
    state = torch.load(path, map_location="cpu")
    for name, e in experts.items():
        s = state[name]
        e.to_shared.load_state_dict(s["to_shared"])
        e.from_shared.load_state_dict(s["from_shared"])
        emb_weight = e.backbone.get_input_embeddings().weight
        with torch.no_grad():
            for i_str, row in s["new_token_embeddings"].items():
                emb_weight[int(i_str)] = row.to(emb_weight.device, emb_weight.dtype)
    print(f"Loaded bridge checkpoint from {path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cfg = Config.debug() if args.debug else Config.default()
    train_stitch(cfg, device=args.device)
