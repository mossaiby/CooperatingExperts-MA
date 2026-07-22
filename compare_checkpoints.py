"""
Runs the same fixed prompts against multiple saved checkpoints (Phase 2
bridge-only, and/or Phase 3 lora_step* dirs) and prints them grouped by
checkpoint, so you can compare quality across training progress instead of
trusting single noisy samples at any one point.

Usage:
    python compare_checkpoints.py \
        --lora-ckpts checkpoints/lora_step100 checkpoints/lora_step160 checkpoints/lora_step320 \
        --temperature 0.8 --seed 0

    # include a Phase-2-only bridge checkpoint in the comparison too:
    python compare_checkpoints.py \
        --bridge-ckpts checkpoints/model_stitched_best.pt \
        --lora-ckpts checkpoints/lora_step160 checkpoints/lora_step320
"""
import argparse

import torch

from config import Config
from generate import load_experts_for_generation, generate


DEFAULT_PROMPTS = [
    ("python", "def factorial(n):"),
    ("english", "This function calculates the factorial of a number recursively."),
    ("python", "def is_palindrome(s):"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bridge-ckpts", nargs="*", default=[],
                     help="Phase-2-only checkpoint .pt files to include")
    ap.add_argument("--lora-ckpts", nargs="*", default=[],
                     help="Phase-3 checkpoint directories to include")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--max-new-tokens", nargs="*", type=int, default=[150],
                     help="one or more generation lengths to test per "
                          "checkpoint -- pass several (e.g. 20 50 100 150) "
                          "to check whether quality degrades with length "
                          "(exposure bias / compounding errors) vs. is "
                          "broken even at short lengths (more structural)")
    ap.add_argument("--seed", type=int, default=0,
                     help="fixed seed so the SAME sampling noise is used "
                          "across checkpoints -- otherwise differences "
                          "could just be random draw, not real quality "
                          "differences between checkpoints")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if not args.bridge_ckpts and not args.lora_ckpts:
        ap.error("pass at least one of --bridge-ckpts / --lora-ckpts")

    cfg = Config.debug() if args.debug else Config.default()
    cfg.gen.temperature = args.temperature
    cfg.gen.top_k = args.top_k

    checkpoints = [("bridge", p) for p in args.bridge_ckpts] + \
                  [("lora", p) for p in args.lora_ckpts]

    for kind, ckpt_path in checkpoints:
        print(f"\n{'='*70}\nCHECKPOINT: {ckpt_path} ({kind})\n{'='*70}")
        if kind == "bridge":
            experts = load_experts_for_generation(cfg, bridge_ckpt_path=ckpt_path, device=args.device)
        else:
            experts = load_experts_for_generation(cfg, lora_ckpt_dir=ckpt_path, device=args.device)

        for start_expert, prompt in DEFAULT_PROMPTS:
            for max_new in args.max_new_tokens:
                cfg.gen.max_new_tokens = max_new
                torch.manual_seed(args.seed)  # same noise draw across lengths/checkpoints
                pieces = generate(experts, prompt, start_expert, cfg, device=args.device)
                print(f"\n--- prompt ({start_expert}, max_new_tokens={max_new}): {prompt!r} ---")
                for name, text in pieces:
                    print(f"  [{name}] {text}")

        del experts
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
