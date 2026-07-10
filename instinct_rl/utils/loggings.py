"""Utilities for managing training log directories."""

import glob
import os
import re
import shutil


def pack_checkpoint_folder(
    log_dir: str,
    resume_path: str,
    num_iter: int,
    additional_file_regex: list[str] | None = None,
) -> str:
    """Copy essential training artifacts into a standalone checkpoint folder.

    Creates a sibling folder next to ``log_dir`` named
    ``{YYYYMMDD}_{HHMMSS}_{num_iter}_ckpt`` and copies the instinct_rl-owned
    artifacts (git diffs, tensorboard events, loaded model file). Additional
    files or directories can be matched via ``additional_file_regex``.

    Args:
        log_dir: Path to the original training run directory.
        resume_path: Absolute path to the model ``.pt`` file that was loaded.
        num_iter: The iteration number of the loaded checkpoint.
        additional_file_regex: Optional list of regex patterns to match extra
            files or directories in ``log_dir`` to copy.

    Returns:
        The absolute path to the newly created checkpoint folder.
    """
    basename = os.path.basename(os.path.normpath(log_dir))
    parts = basename.split("_")
    prefix = f"{parts[0]}_{parts[1]}"
    iter_str = f"{num_iter // 1000}k" if num_iter >= 1000 else str(num_iter)
    new_name = f"{prefix}_{iter_str}_ckpt"
    new_dir = os.path.join(os.path.dirname(log_dir), new_name)
    if os.path.exists(new_dir):
        print(f"\033[93m[WARN] Checkpoint folder already exists, overwriting: {new_dir}\033[0m")
    os.makedirs(new_dir, exist_ok=True)

    # instinct_rl-owned artifacts
    for subdir in ("git", "params", "exported"):
        src = os.path.join(log_dir, subdir)
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(new_dir, subdir), dirs_exist_ok=True)

    for event_file in glob.glob(os.path.join(log_dir, "events.*")):
        shutil.copy2(event_file, new_dir)

    if os.path.isfile(resume_path):
        shutil.copy2(resume_path, new_dir)

    if additional_file_regex:
        for entry in os.listdir(log_dir):
            if any(re.search(pattern, entry) for pattern in additional_file_regex):
                src = os.path.join(log_dir, entry)
                dst = os.path.join(new_dir, entry)
                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                elif os.path.isfile(src):
                    shutil.copy2(src, new_dir)

    print(f"[INFO] Packed checkpoint folder: {new_dir}")
    return new_dir
