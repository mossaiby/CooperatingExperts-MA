"""
One-stop launcher for Phase 3 (LoRA) training, platform-agnostic (works on
Colab, Lightning.ai, or anywhere else with a persistent filesystem at
--ckpt-dir). Auto-detects the most recent lora_step<N> checkpoint directory
in --ckpt-dir and resumes from it if present; otherwise starts fresh from
--bridge-ckpt.

Usage:
    python run_lora.py --bridge-ckpt checkpoints/model_stitched_best.pt
    python run_lora.py --bridge-ckpt checkpoints/model_stitched_best.pt --ckpt-dir /my/persistent/dir
    python run_lora.py --bridge-ckpt ... --no-resume   # ignore existing checkpoints, start fresh

If you're on Colab and want checkpoints on Drive, mount it first (see
mount_drive.py) and pass that path as --ckpt-dir.
"""
import argparse
import glob
import os
import re

from config import Config
from train_lora import train_lora


def find_latest_lora_checkpoint(ckpt_dir: str):
    candidates = [
        d for d in glob.glob(os.path.join(ckpt_dir, "lora_step*"))
        if os.path.isdir(d)
    ]
    if not candidates:
        return None
    def step_of(path):
        m = re.search(r"step(\d+)", os.path.basename(path.rstrip("/")))
        return int(m.group(1)) if m else -1
    return max(candidates, key=step_of)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bridge-ckpt", required=True,
                     help="Phase-2 checkpoint, used only if no existing "
                          "Phase-3 checkpoint is found to resume from")
    ap.add_argument("--ckpt-dir", default="checkpoints",
                     help="where Phase-3 checkpoints live / will be written")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps-max", type=int, default=None)
    ap.add_argument("--switch-loss-weight", type=float, default=None,
                     help="override cfg.lora.switch_loss_weight")
    ap.add_argument("--no-resume", action="store_true")
    args, _ = ap.parse_known_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)

    cfg = Config.debug() if args.debug else Config.default()
    cfg.lora.ckpt_dir = args.ckpt_dir
    if args.steps_max is not None:
        cfg.lora.steps_max = args.steps_max
    if args.switch_loss_weight is not None:
        cfg.lora.switch_loss_weight = args.switch_loss_weight

    resume_from = None
    if not args.no_resume:
        resume_from = find_latest_lora_checkpoint(args.ckpt_dir)
        if resume_from:
            print(f"Found existing Phase-3 checkpoint, will resume: {resume_from}")
        else:
            print("No existing Phase-3 checkpoint found -- starting fresh from "
                  f"{args.bridge_ckpt}")
    else:
        print("--no-resume passed, starting fresh even if checkpoints exist.")

    train_lora(cfg, args.bridge_ckpt, device=args.device, resume_from=resume_from)


if __name__ == "__main__":
    main()
