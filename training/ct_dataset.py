"""
PyTorch Dataset for the limited-angle CT training pairs.
========================================================

WHAT THIS FILE DOES
-------------------
It is the bridge between the .npy files that generate_dataset.py wrote to disk
and the tensors that the training loop consumes. PyTorch's DataLoader knows how
to batch, shuffle and parallel-load *if* you give it a Dataset object that can
(a) say how many samples exist and (b) return sample number i. This file
provides exactly that, plus the train/val split logic.

Each sample is a pair of single-channel 256x256 float32 images:
    input  = data/<set>/inputs/xxxxx.npy   (limited-angle FBP reconstruction)
    target = data/<set>/targets/xxxxx.npy  (ground-truth phantom, in [0, 1])

DESIGN DECISIONS (worth knowing for the thesis write-up)
--------------------------------------------------------
* No per-image normalisation. The targets are already in [0, 1] and the FBP
  inputs sit in roughly the same scale (with the over/undershoot that
  limited-angle FBP produces). Keeping the native scale means PSNR/SSIM are
  computed against a data_range of 1.0 -- directly comparable to the FBP and
  OS-SART baselines reported in limited_angle_demo.py.
* The train/val split is DETERMINISTIC: we seed a shuffle of the sorted file
  list, so every run sees the identical split. Without this, learning curves
  from two runs would not be comparable.
* Shepp-Logan is intentionally NOT in this random-ellipse pilot; it is held out
  as the standard, unseen test phantom (used later in evaluate_unet.py).
"""

import os
import glob

import numpy as np
import torch
from torch.utils.data import Dataset


def list_pairs(data_dir):
    """Find and pair up every input with its matching target.

    Returns a list of (input_path, target_path) tuples. Both lists are sorted,
    so the i-th input lines up with the i-th target. The guards below catch the
    silent, dangerous failure modes -- a missing file or a mismatch would train
    the network on wrong (input, target) pairs and quietly ruin everything.
    """
    in_dir = os.path.join(data_dir, "inputs")
    tg_dir = os.path.join(data_dir, "targets")
    inputs = sorted(glob.glob(os.path.join(in_dir, "*.npy")))
    targets = sorted(glob.glob(os.path.join(tg_dir, "*.npy")))
    if not inputs:
        raise FileNotFoundError(f"No .npy inputs found in {in_dir}")
    if len(inputs) != len(targets):
        raise ValueError(
            f"input/target count mismatch: {len(inputs)} inputs vs "
            f"{len(targets)} targets in {data_dir}")
    # Confirm names align index-for-index (00007.npy <-> 00007.npy).
    for ip, tp in zip(inputs, targets):
        if os.path.basename(ip) != os.path.basename(tp):
            raise ValueError(
                f"input/target filename mismatch: {ip} vs {tp}")
    return list(zip(inputs, targets))


def split_pairs(pairs, val_frac=0.1, seed=42):
    """Deterministically shuffle, then carve off a validation fraction.

    Using a seeded numpy Generator means the same `seed` always produces the
    same train/val membership -- reproducibility you can rely on across runs.
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(len(pairs))
    rng.shuffle(idx)                       # shuffle the *indices*, not the files
    n_val = max(1, int(round(len(pairs) * val_frac)))
    val_idx = set(idx[:n_val].tolist())    # first n_val shuffled indices = val
    train = [pairs[i] for i in range(len(pairs)) if i not in val_idx]
    val = [pairs[i] for i in range(len(pairs)) if i in val_idx]
    return train, val


class LimitedAngleCTDataset(Dataset):
    """Wraps a list of (input, target) paths as a PyTorch Dataset.

    PyTorch needs two methods: __len__ (how many samples) and __getitem__ (give
    me sample i). The DataLoader calls these to assemble batches. We load lazily
    -- one pair off disk per call -- which keeps memory low and scales to the
    larger production dataset for free.
    """

    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        in_path, tg_path = self.pairs[i]
        x = np.load(in_path).astype(np.float32)   # (H, W) numpy array
        y = np.load(tg_path).astype(np.float32)   # (H, W) numpy array
        # PyTorch convs expect (Channels, H, W); these are grayscale so we add a
        # single channel axis with unsqueeze(0) -> (1, H, W). The DataLoader then
        # stacks a batch of these into (B, 1, H, W).
        x = torch.from_numpy(x).unsqueeze(0)
        y = torch.from_numpy(y).unsqueeze(0)
        return x, y


def build_datasets(data_dir, val_frac=0.1, seed=42):
    """Convenience wrapper: list -> split -> wrap.
    Returns (train_ds, val_ds, train_pairs, val_pairs)."""
    pairs = list_pairs(data_dir)
    train_pairs, val_pairs = split_pairs(pairs, val_frac=val_frac, seed=seed)
    return (LimitedAngleCTDataset(train_pairs),
            LimitedAngleCTDataset(val_pairs),
            train_pairs, val_pairs)


if __name__ == "__main__":
    # `python training/ct_dataset.py` self-tests against the real pilot data and
    # prints the split sizes and one sample's shape/range. A quick way to confirm
    # the data is wired up correctly before launching a training run.
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(here, "..", "data", "pilot_v1")
    pairs = list_pairs(data_dir)
    tr, va = split_pairs(pairs)
    print(f"total pairs: {len(pairs)} | train: {len(tr)} | val: {len(va)}")
    ds = LimitedAngleCTDataset(tr)
    x, y = ds[0]
    print("input tensor ", tuple(x.shape), x.dtype,
          f"range [{x.min():.3f}, {x.max():.3f}]")
    print("target tensor", tuple(y.shape), y.dtype,
          f"range [{y.min():.3f}, {y.max():.3f}]")
