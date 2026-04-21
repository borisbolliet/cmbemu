"""Hugging Face Hub download + local cache for the competition datasets.

First call to ``load_train`` / ``load_test`` pulls the .npz files from the
HF dataset repo and caches them under ``~/.cache/cmbemu/<version>/``. Override
the repo by setting the ``CMBEMU_HF_REPO`` environment variable.
"""
from __future__ import annotations

import os
from pathlib import Path

from .io import load_dataset

# Update this to the real HF dataset repo once it is created.
DEFAULT_HF_REPO = "borisbolliet/cmbemu-competition-v1"

# Filenames in the HF repo
FNAME_TRAIN = "train.npz"
FNAME_TEST = "test.npz"
FNAME_CHI2_TRUE = "chi2_true.npy"  # optional precomputed NxN on test

# Map the "size" arg of the loaders to the actual filenames
_SIZE_MAP = {
    "full":  {"train": "train.npz",       "test": "test.npz"},
    "small": {"train": "train_small.npz", "test": "test_small.npz"},
}


def _repo_id() -> str:
    return os.environ.get("CMBEMU_HF_REPO", DEFAULT_HF_REPO)


def _cache_dir() -> Path:
    root = Path(os.environ.get("CMBEMU_CACHE_DIR",
                               Path.home() / ".cache" / "cmbemu"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def _fetch(filename: str, local_first: Path | None = None) -> Path:
    """Resolve a dataset file, preferring a local path if provided."""
    if local_first is not None and local_first.exists():
        return local_first
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(
        repo_id=_repo_id(),
        filename=filename,
        repo_type="dataset",
        cache_dir=str(_cache_dir()),
    ))


def download_data(size: str = "full") -> dict[str, Path]:
    """Trigger the HF download for both train and test up front, return paths."""
    files = _SIZE_MAP[size]
    out = {}
    for tag, fname in files.items():
        out[tag] = _fetch(fname)
    return out


def load_train(size: str = "full", local_dir: str | Path | None = None) -> dict:
    """Return the training dataset as a dict of arrays (see io.load_dataset).

    Downloads + caches on first call; subsequent calls read from cache.

    Parameters
    ----------
    size : "full" (50k) or "small" (5k). "small" is for quick dev loops.
    local_dir : if given, looks for the file here before falling back to HF.
        Useful during development before the HF repo exists.
    """
    fname = _SIZE_MAP[size]["train"]
    local = Path(local_dir) / fname if local_dir else None
    return load_dataset(_fetch(fname, local))


def load_test(size: str = "full", local_dir: str | Path | None = None) -> dict:
    """Return the test dataset as a dict of arrays."""
    fname = _SIZE_MAP[size]["test"]
    local = Path(local_dir) / fname if local_dir else None
    return load_dataset(_fetch(fname, local))
