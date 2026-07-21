"""
Smoke test: runs the whole pipeline on tiny debug models (distilgpt2 +
codegen-350M-mono) with a handful of fake pairs, entirely on CPU or a small
GPU. Meant to catch shape/logic bugs BEFORE spending Colab session time on
the real 1-3B pair.

This does NOT use CodeSearchNet (avoids a slow download for a smoke test) --
it uses a few hand-written fake pairs instead.

Usage:
    python smoke_test.py
"""
import os
import torch

from config import Config
from models import FrozenExpert
from dataset import HandoffDataset, DirectionalBatcher
from train_stitch import directed_loss, save_bridge_checkpoint, load_bridge_checkpoint
from train_lora import build_sessions, encode_session, mixed_loss_for_session, attach_lora
from generate import load_experts_for_generation, generate


FAKE_PAIRS = [
    {"id": "fake_0", "docstring": "Return the sum of two numbers.",
     "code": "def add(a, b):\n    return a + b"},
    {"id": "fake_1", "docstring": "Check if a number is even.",
     "code": "def is_even(n):\n    return n % 2 == 0"},
    {"id": "fake_2", "docstring": "Reverse a string.",
     "code": "def reverse(s):\n    return s[::-1]"},
    {"id": "fake_3", "docstring": "Find the maximum value in a list.",
     "code": "def find_max(xs):\n    return max(xs)"},
]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running smoke test on device={device}")

    cfg = Config.debug()
    os.makedirs("smoke_ckpt", exist_ok=True)
    cfg.stitch.ckpt_dir = "smoke_ckpt"
    cfg.lora.ckpt_dir = "smoke_ckpt"

    print("\n--- Loading frozen tiny experts ---")
    experts = {
        "english": FrozenExpert(cfg.pair.english, cfg.shared, cfg.switch, cfg.handoff_layer, device),
        "python": FrozenExpert(cfg.pair.python, cfg.shared, cfg.switch, cfg.handoff_layer, device),
    }

    print("\n--- Building a tiny HandoffDataset + one directed loss step ---")
    pad_id_by_expert = {name: e.tokenizer.pad_token_id for name, e in experts.items()}
    ds = HandoffDataset(FAKE_PAIRS, experts["english"].tokenizer, experts["python"].tokenizer,
                         max_seq_len=cfg.stitch.max_seq_len)
    batcher = DirectionalBatcher(ds, batch_size=2, pad_id_by_expert=pad_id_by_expert)
    en2py_batch, py2en_batch = next(batcher.infinite_pairs())
    loss, lm_l, align_l = directed_loss(experts, en2py_batch, cfg.stitch.align_weight, device)
    loss.backward()
    print(f"en2py directed_loss OK: total={loss.item():.4f} lm={lm_l:.4f} align={align_l:.4f}")
    for e in experts.values():
        e.to_shared.zero_grad()
        e.from_shared.zero_grad()
        e.backbone.get_input_embeddings().weight.grad = None

    print("\n--- Checkpoint save/load round trip ---")
    save_bridge_checkpoint(experts, cfg.stitch.ckpt_dir, "smoke_bridge.pt")
    load_bridge_checkpoint(experts, os.path.join(cfg.stitch.ckpt_dir, "smoke_bridge.pt"))
    print("checkpoint round trip OK")

    print("\n--- Mixed session loss (phase 3 shape check, no LoRA attached) ---")
    sessions = build_sessions(FAKE_PAIRS, seed=0)
    chunks = encode_session(sessions[0], experts, max_seq_len=64)
    mloss = mixed_loss_for_session(chunks, experts, device, switch_loss_weight=0.1)
    mloss.backward()
    print(f"mixed_loss_for_session OK: loss={mloss.item():.4f}")

    print("\n--- Generation smoke test ---")
    for e in experts.values():
        e.backbone.eval()
    pieces = generate(experts, "def add(a, b):", "python", cfg, device=device)
    print("Generation pieces:")
    for name, text in pieces:
        print(f"  [{name}] {text[:80]!r}")

    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
