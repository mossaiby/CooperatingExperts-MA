"""
Phase 3: mixed end-to-end training.

Unlike v1 (which fully unfroze its 44.6M-param from-scratch experts), we
can't afford to fully unfreeze 1-3B pretrained backbones -- both compute
cost and catastrophic-forgetting risk are too high. Instead we attach LoRA
adapters (via peft) to each frozen backbone, and continue training the
bridge (to_shared/from_shared + new-token rows) together with the LoRA
params on whole interleaved sessions with <switch:NAME> tokens.

Sessions here are built cheaply from the same CodeSearchNet pairs: each
"session" is docstring -> <switch:python> -> code -> <switch:english> -> (a
short closing sentence). This is a simple 2-switch session, not the richer
multi-switch-per-reply structure v1's synthetic generator produced -- see
README for the note on optionally bringing an LLM-generation step back for
richer multi-switch sessions once this baseline works.
"""
import os
import time

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model, PeftModel

from config import Config
from models import FrozenExpert
from data import load_pairs, train_val_split
from train_stitch import load_bridge_checkpoint, save_bridge_checkpoint, _cosine_warmup_lr


CLOSERS = [
    "That should do it.",
    "Let me know if you'd like a different approach.",
    "This should handle the typical cases.",
    "Happy to adjust this further if needed.",
]


def attach_lora(expert: FrozenExpert, lora_cfg):
    lora_config = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.alpha,
        lora_dropout=lora_cfg.dropout,
        target_modules=list(lora_cfg.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    expert.backbone = get_peft_model(expert.backbone, lora_config)
    expert.backbone.print_trainable_parameters()
    return expert


def attach_lora_from_checkpoint(expert: FrozenExpert, adapter_dir: str):
    """Loads a previously-saved LoRA adapter (from save_lora_checkpoint) in
    trainable mode, for resuming a Phase-3 run after a disconnect."""
    expert.backbone = PeftModel.from_pretrained(expert.backbone, adapter_dir, is_trainable=True)
    expert.backbone.print_trainable_parameters()
    return expert


@torch.no_grad()
def evaluate_mixed(val_sessions, experts, device, switch_loss_weight, n_sessions=20, num_vectors=1):
    import random
    rng = random.Random(0)
    total = 0.0
    sample = rng.sample(val_sessions, min(n_sessions, len(val_sessions)))
    for session in sample:
        chunks = encode_session(session, experts, max_seq_len=256)
        loss = mixed_loss_for_session(chunks, experts, device, switch_loss_weight, num_vectors=num_vectors)
        total += loss.item()
    return total / len(sample)


def build_sessions(pairs, seed=0):
    import random
    rng = random.Random(seed)
    sessions = []
    for p in pairs:
        closer = rng.choice(CLOSERS)
        sessions.append({
            "segments": [
                {"expert": "english", "text": p["docstring"]},
                {"expert": "python", "text": p["code"]},
                {"expert": "english", "text": closer},
            ]
        })
    return sessions


def encode_session(session, experts, max_seq_len):
    """
    Tokenizes each segment with its own expert's tokenizer, inserts the
    switch token BEFORE each segment (using the tokenizer of the segment
    that is about to start, matching v1's boundary-marker convention), and
    returns a list of (expert_name, token_ids) chunks plus the switch ids
    used between them. Actual cross-tokenizer concatenation happens at the
    embedding level in train_step below (can't concat ids from different
    vocabs into one tensor).
    """
    chunks = []
    for i, seg in enumerate(session["segments"]):
        e = experts[seg["expert"]]
        ids = e.tokenizer(seg["text"], truncation=True, max_length=max_seq_len,
                           return_tensors=None)["input_ids"]
        switch_id = e.switch_id(seg["expert"])
        chunks.append({"expert": seg["expert"], "ids": ids, "switch_id": switch_id})
    return chunks


def mixed_loss_for_session(chunks, experts, device, switch_loss_weight, num_vectors=1):
    """
    Walks the session segment by segment. For segment i>0, the carried
    hidden state (or, if num_vectors>1, short sequence of hidden states) is
    produced by encoding segment i-1 with ITS expert and projecting through
    the shared space into segment i's expert -- exactly the Phase-2 handoff
    mechanism, now with LoRA-adapted (not fully frozen) backbones. Loss is
    accumulated per segment and averaged.
    """
    total_loss = 0.0
    n = 0
    carried_vec = None  # [1, H_dst] (num_vectors=1) or [1, K, H_dst] once we have one

    for i, chunk in enumerate(chunks):
        e = experts[chunk["expert"]]
        ids = torch.tensor([chunk["ids"]], dtype=torch.long, device=device)
        mask = torch.ones_like(ids)

        if carried_vec is None:
            # first segment: normal LM loss, no injected prefix
            out = e.backbone(input_ids=ids, attention_mask=mask, labels=ids)
            loss = out.loss
        else:
            logits = e.forward_with_injected_prefix(carried_vec, ids, mask)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                ids.reshape(-1),
            )
        total_loss += loss
        n += 1

        # prepare handoff vector for the NEXT segment (if any)
        if i < len(chunks) - 1:
            if num_vectors > 1:
                h = e.encode_handoff_vectors(ids, mask, num_vectors)  # [1, K, H_src]
            else:
                h = e.encode_handoff_vector(ids, mask)         # [1, H_src]
            z = e.to_shared_space(h)                        # [1, (K,) shared_dim]
            next_expert = experts[chunks[i + 1]["expert"]]
            carried_vec = next_expert.from_shared_space(z)   # [1, (K,) H_dst]

    return total_loss / max(1, n)


def train_lora(cfg: Config, bridge_ckpt_path: str, device="cuda", resume_from: str = None):
    os.makedirs(cfg.lora.ckpt_dir, exist_ok=True)

    print("Loading frozen experts + Phase-2 bridge checkpoint ...")
    experts = {
        "english": FrozenExpert(cfg.pair.english, cfg.shared, cfg.switch, cfg.handoff_layer, device),
        "python": FrozenExpert(cfg.pair.python, cfg.shared, cfg.switch, cfg.handoff_layer, device),
    }

    start_step = 0
    if resume_from is not None:
        print(f"Resuming Phase 3 from {resume_from} ...")
        for name, e in experts.items():
            attach_lora_from_checkpoint(e, os.path.join(resume_from, f"{name}_lora"))
            e.gradient_checkpointing_enable()
        early_stop_state = load_bridge_checkpoint(experts, os.path.join(resume_from, "bridge.pt"))
        import re as _re
        m = _re.search(r"step(\d+)", os.path.basename(resume_from.rstrip("/")))
        start_step = int(m.group(1)) if m else 0
        print(f"Resumed at inferred start_step={start_step}. Note optimizer "
              f"(AdamW) state is NOT restored, only LoRA + bridge weights.")
    else:
        print("Loading Phase-2 bridge checkpoint (fresh LoRA attach) ...")
        load_bridge_checkpoint(experts, bridge_ckpt_path)
        print("Attaching LoRA adapters ...")
        for e in experts.values():
            attach_lora(e, cfg.lora)
            e.gradient_checkpointing_enable()

    pairs = load_pairs(cfg.data.processed_path)
    train_pairs, val_pairs = train_val_split(pairs, cfg.data.val_fraction, cfg.data.seed)
    train_sessions = build_sessions(train_pairs, cfg.data.seed)
    val_sessions = build_sessions(val_pairs, cfg.data.seed + 1)

    trainable_params = []
    for e in experts.values():
        trainable_params.extend(e.trainable_parameters())          # bridge params
        trainable_params.extend([p for p in e.backbone.parameters() if p.requires_grad])  # LoRA
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"Trainable params (bridge + LoRA): {n_trainable:,}")

    optimizer = AdamW(trainable_params, lr=cfg.lora.lr)

    import random
    rng = random.Random(cfg.data.seed)
    t0 = time.time()
    best_val = float("inf")

    for step in range(start_step, cfg.lora.steps_max):
        lr = _cosine_warmup_lr(step, cfg.lora.warmup_steps, cfg.lora.steps_max, cfg.lora.lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad()
        step_loss = 0.0
        for _ in range(cfg.lora.grad_accum):
            session = rng.choice(train_sessions)
            chunks = encode_session(session, experts, max_seq_len=256)
            loss = mixed_loss_for_session(chunks, experts, device, cfg.lora.switch_loss_weight,
                                           num_vectors=cfg.shared.num_vectors)
            (loss / cfg.lora.grad_accum).backward()
            step_loss += loss.item() / cfg.lora.grad_accum

        torch.nn.utils.clip_grad_norm_(trainable_params, cfg.lora.grad_clip)
        optimizer.step()

        if step % cfg.lora.log_every == 0:
            elapsed = time.time() - t0
            print(f"[lora] step {step:5d}/{cfg.lora.steps_max} lr={lr:.2e} "
                  f"loss={step_loss:.4f} ({elapsed:.0f}s elapsed)")

        if step > 0 and step % cfg.lora.ckpt_every == 0:
            val_loss = evaluate_mixed(val_sessions, experts, device, cfg.lora.switch_loss_weight,
                                       num_vectors=cfg.shared.num_vectors)
            print(f"[lora] step {step:5d} VAL loss={val_loss:.4f}")
            save_lora_checkpoint(experts, cfg.lora.ckpt_dir, f"lora_step{step}")
            if val_loss < best_val:
                best_val = val_loss
                save_lora_checkpoint(experts, cfg.lora.ckpt_dir, "lora_best")

    save_lora_checkpoint(experts, cfg.lora.ckpt_dir, "lora_final")
    return experts


def save_lora_checkpoint(experts, ckpt_dir, tag):
    out_dir = os.path.join(ckpt_dir, tag)
    os.makedirs(out_dir, exist_ok=True)
    for name, e in experts.items():
        e.backbone.save_pretrained(os.path.join(out_dir, f"{name}_lora"))
    save_bridge_checkpoint(experts, out_dir, "bridge.pt")
    print(f"Saved LoRA + bridge checkpoint -> {out_dir}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--bridge-ckpt", default=None,
                     help="path to a checkpoint saved by train_stitch.py, e.g. "
                          "checkpoints/model_stitched_best.pt (required unless "
                          "--resume-from is given)")
    ap.add_argument("--resume-from", default=None,
                     help="path to a directory saved by this script's "
                          "save_lora_checkpoint, e.g. checkpoints/lora_step200, "
                          "to continue a Phase-3 run after a disconnect")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.bridge_ckpt is None and args.resume_from is None:
        ap.error("must pass either --bridge-ckpt (fresh Phase 3 start) or "
                  "--resume-from (continue a Phase 3 run)")
    cfg = Config.debug() if args.debug else Config.default()
    train_lora(cfg, args.bridge_ckpt, device=args.device, resume_from=args.resume_from)
