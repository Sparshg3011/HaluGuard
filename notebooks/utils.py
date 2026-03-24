"""
utils.py — Shared utilities for HaluGuard Colab notebooks.

Import this at the top of each notebook:
    import sys; sys.path.insert(0, "..")
    from notebooks.utils import check_gpu, mount_drive, save_checkpoint, load_checkpoint
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# GPU / environment checks
# ---------------------------------------------------------------------------

def check_gpu() -> str:
    """Print and return a summary of the current GPU environment.

    Checks for CUDA availability via PyTorch and calls ``nvidia-smi`` for
    hardware details.  Prints a warning if no GPU is found (T4 expected on
    Colab Pro).

    Returns:
        Device string: ``"cuda"`` if a GPU is available, else ``"cpu"``.
    """
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        if has_cuda:
            device_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU found: {device_name} ({vram_gb:.1f} GB VRAM)")
        else:
            print("WARNING: No GPU found. Running on CPU — embedding will be slow.")
        return "cuda" if has_cuda else "cpu"
    except ImportError:
        print("PyTorch not installed. Run: pip install torch")
        return "cpu"


def print_env_summary() -> None:
    """Print Python version, PyTorch version, and transformers version."""
    import sys
    print(f"Python:       {sys.version.split()[0]}")
    try:
        import torch
        print(f"PyTorch:      {torch.__version__}")
    except ImportError:
        print("PyTorch:      NOT INSTALLED")
    try:
        import transformers
        print(f"Transformers: {transformers.__version__}")
    except ImportError:
        print("Transformers: NOT INSTALLED")


# ---------------------------------------------------------------------------
# Google Drive helpers (Colab-specific)
# ---------------------------------------------------------------------------

def mount_drive(mount_point: str = "/content/drive") -> Path:
    """Mount Google Drive in Colab and return the MyDrive path.

    No-ops gracefully when not running in Colab (e.g. local Jupyter).

    Args:
        mount_point: Where to mount the drive.  Default ``/content/drive``.

    Returns:
        ``Path`` to ``MyDrive`` directory.
    """
    try:
        from google.colab import drive  # type: ignore[import]
        drive.mount(mount_point)
        my_drive = Path(mount_point) / "MyDrive"
        print(f"Drive mounted at {my_drive}")
        return my_drive
    except ImportError:
        print("Not running in Colab — Drive mount skipped.")
        return Path.home() / "HaluGuard_drive_sim"


def get_drive_path(subdir: str = "HaluGuard") -> Path:
    """Return (and create) the project directory inside MyDrive.

    Args:
        subdir: Subfolder name inside MyDrive.  Default ``"HaluGuard"``.

    Returns:
        ``Path`` to the project folder on Drive.
    """
    drive_root = Path("/content/drive/MyDrive")
    if not drive_root.exists():
        # Fallback for local development
        drive_root = Path.home() / "HaluGuard_drive_sim"
    project_dir = drive_root / subdir
    project_dir.mkdir(parents=True, exist_ok=True)
    return project_dir


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    obj: Any,
    name: str,
    drive_subdir: str = "HaluGuard/checkpoints",
) -> Path:
    """Save a PyTorch state dict or arbitrary object to Google Drive.

    Args:
        obj:          Object to save.  If it has a ``state_dict()`` method
                      (i.e. it's an ``nn.Module``), the state dict is saved.
                      Otherwise the object itself is saved via ``torch.save``.
        name:         Filename (e.g. ``"hccs_epoch_5.pt"``).
        drive_subdir: Subfolder inside MyDrive.

    Returns:
        Path where the checkpoint was saved.
    """
    import torch

    drive_root = Path("/content/drive/MyDrive")
    if not drive_root.exists():
        drive_root = Path.home() / "HaluGuard_drive_sim"

    out_dir = drive_root / drive_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / name

    payload = obj.state_dict() if hasattr(obj, "state_dict") else obj
    torch.save(payload, out_path)
    print(f"Saved checkpoint: {out_path}")
    return out_path


def load_checkpoint(
    name: str,
    drive_subdir: str = "HaluGuard/checkpoints",
) -> Any:
    """Load a checkpoint saved by ``save_checkpoint``.

    Args:
        name:         Filename, e.g. ``"hccs_best.pt"``.
        drive_subdir: Subfolder inside MyDrive.

    Returns:
        The loaded object (state dict or arbitrary).

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
    """
    import torch

    drive_root = Path("/content/drive/MyDrive")
    if not drive_root.exists():
        drive_root = Path.home() / "HaluGuard_drive_sim"

    path = drive_root / drive_subdir / name
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    obj = torch.load(path, map_location="cpu")
    print(f"Loaded checkpoint: {path}")
    return obj


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

def append_jsonl(record: dict, path: Path) -> None:
    """Append a single JSON record to a JSONL file (create if needed).

    Useful for incremental checkpoint-style saving during long data pipelines.

    Args:
        record: Dict to serialise.
        path:   Destination ``.jsonl`` file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def count_jsonl(path: Path) -> int:
    """Count the number of records in a JSONL file.

    Args:
        path: Path to the ``.jsonl`` file.

    Returns:
        Number of non-empty lines.  0 if the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())
