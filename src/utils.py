"""Shared helpers: seeding, class weights, datasets, and plotting."""
from __future__ import annotations

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BeatDataset(Dataset):
    """Wraps (beats, rr, labels) arrays as tensors."""

    def __init__(self, beats, rr, labels):
        self.beats = torch.as_tensor(beats, dtype=torch.float32)
        self.rr = torch.as_tensor(rr, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.beats[i], self.rr[i], self.labels[i]


def make_loader(beats, rr, labels, batch_size=128, shuffle=False):
    return DataLoader(BeatDataset(beats, rr, labels), batch_size=batch_size,
                      shuffle=shuffle, drop_last=False)


def balanced_class_weights(labels: np.ndarray, n_classes: int = 4) -> torch.Tensor:
    """Balanced weights computed from the TRAINING split only (no leakage)."""
    classes = np.arange(n_classes)
    w = compute_class_weight("balanced", classes=classes, y=labels)
    return torch.as_tensor(w, dtype=torch.float32)


def plot_learning_curves(history: dict, out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(history["train_loss"], label="train")
    ax[0].plot(history["val_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(history["train_f1"], label="train")
    ax[1].plot(history["val_f1"], label="val")
    ax[1].set_title("Macro-F1"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
