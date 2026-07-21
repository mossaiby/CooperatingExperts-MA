"""
One-stop Colab launcher for Phase 2 (stitching) training.

Run this at the start of every fresh Colab session -- it:
  1. Mounts Google Drive (needed every session; Colab wipes /content on
     each new runtime, but Drive itself persists).
  2. Points checkpointing at a Drive folder so nothing is lost on disconnect.
  3. Auto-detects the most recent checkpoint in that folder and resumes
     from it, if one exists -- no need to hunt for the exact filename.
  4. Starts train_stitch.py's training loop.

Usage (in a Colab cell):
    %run run_colab_stitch.py

Or with overrides:
    %run run_colab_stitch.py --ckpt-subdir my_run --steps-max 5000

If you truly want to start over and ignore any existing checkpoint in that
Drive folder, pass --no-resume.
"""
import argparse
import glob
import os
import re

from config import Config
from mount_drive import mount_and_get_ckpt_dir
from train_stitch import train_stitch


def find_latest_checkpoint(ckpt_dir: str):
    """
    Looks for model_stitched_step<N>.pt files and returns the one with the
    highest N, or None if the folder is empty / has no step checkpoints.
    model_stitched_best.pt and model_stitched_final.pt are intentionally
    NOT auto-resumed from here, since "best" isn't necessarily "most
    recent" and "final" implies a prior run already completed -- resuming
    training should default to the latest STEP checkpoint, matching what
    train_stitch.py's own step-count inference (via the step<N> filename
    pattern) expects.
    """
    candidates = glob.glob(os.path.join(ckpt_dir, "model_stitched_step*.pt"))
    if not candidates:
        return None
    def step_of(path):
        m = re.search(r"step(\d+)", os.path.basename(path))
        return int(m.group(1)) if m else -1
    return max(candidates, key=step_of)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-subdir", default="cooperating_experts_ckpts",
                     help="folder name under My Drive/ for checkpoints")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps-max", type=int, default=None,
                     help="override cfg.stitch.steps_max")
    ap.add_argument("--no-resume", action="store_true",
                     help="ignore any existing checkpoint and start fresh")
    args, _ = ap.parse_known_args()  # known_args tolerates Colab's own extra argv

    ckpt_dir = mount_and_get_ckpt_dir(args.ckpt_subdir)

    cfg = Config.debug() if args.debug else Config.default()
    cfg.stitch.ckpt_dir = ckpt_dir
    if args.steps_max is not None:
        cfg.stitch.steps_max = args.steps_max

    resume_from = None
    if not args.no_resume:
        resume_from = find_latest_checkpoint(ckpt_dir)
        if resume_from:
            print(f"Found existing checkpoint, will resume: {resume_from}")
        else:
            print("No existing checkpoint found in this Drive folder -- starting fresh.")
    else:
        print("--no-resume passed, starting fresh even if checkpoints exist.")

    train_stitch(cfg, device=args.device, resume_from=resume_from)


if __name__ == "__main__":
    main()
