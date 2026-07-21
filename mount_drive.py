"""
Mounts Google Drive (Colab only) and creates/prints the checkpoint
directory, so train_stitch.py / train_lora.py can point at persistent
storage instead of Colab's ephemeral local disk.

IMPORTANT: this only works inside a Colab runtime, and it works most
reliably run as a notebook cell via:

    %run mount_drive.py

rather than `!python mount_drive.py` -- the `!` form runs in a subprocess,
and Drive's auth flow (which needs to show you a link/prompt the first
time) doesn't always surface cleanly through a subprocess. `%run` executes
in-process in the same kernel as the rest of your notebook, same as a
normal cell, so the auth prompt behaves normally.

Usage (in a Colab cell):
    %run mount_drive.py --ckpt-subdir cooperating_experts_ckpts

Then in your training command:
    python train_stitch.py   # after editing cfg.stitch.ckpt_dir, see below

or, simplest, just import the path this script prints and set it directly:

    from mount_drive import mount_and_get_ckpt_dir
    ckpt_dir = mount_and_get_ckpt_dir("cooperating_experts_ckpts")
    cfg.stitch.ckpt_dir = ckpt_dir
    cfg.lora.ckpt_dir = ckpt_dir
"""
import argparse
import os


def mount_and_get_ckpt_dir(ckpt_subdir: str = "cooperating_experts_ckpts",
                            drive_mount_point: str = "/content/drive") -> str:
    try:
        from google.colab import drive
    except ImportError:
        raise RuntimeError(
            "google.colab not available -- this script only works inside "
            "a Colab runtime. If you're running locally, just set "
            "cfg.stitch.ckpt_dir / cfg.lora.ckpt_dir to a normal local or "
            "synced-folder path instead."
        )

    if not os.path.isdir(drive_mount_point):
        print(f"Mounting Google Drive at {drive_mount_point} ...")
        drive.mount(drive_mount_point)
    else:
        print(f"Google Drive already mounted at {drive_mount_point}")

    ckpt_dir = os.path.join(drive_mount_point, "MyDrive", ckpt_subdir)
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Checkpoint directory ready: {ckpt_dir}")
    return ckpt_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-subdir", default="cooperating_experts_ckpts",
                     help="folder name to create under My Drive/")
    ap.add_argument("--drive-mount-point", default="/content/drive")
    args = ap.parse_args()

    path = mount_and_get_ckpt_dir(args.ckpt_subdir, args.drive_mount_point)
    print(f"\nSet this in your training config:\n"
          f"  cfg.stitch.ckpt_dir = {path!r}\n"
          f"  cfg.lora.ckpt_dir   = {path!r}")
