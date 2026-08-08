"""Shared helpers: seeding, class weights, RR standardization, datasets, plotting."""
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
    """Yields (beat[1,360], rr[3], label) so all three models share one loader."""

    def __init__(self, beats, rr, labels):
        # channel-first (N, 1, 360) to match the notebook's convention
        self.X = torch.as_tensor(beats, dtype=torch.float32).unsqueeze(1)
        self.rr = torch.as_tensor(rr, dtype=torch.float32)
        self.y = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.rr[i], self.y[i]


def make_loader(beats, rr, labels, batch_size=128, shuffle=False):
    return DataLoader(BeatDataset(beats, rr, labels), batch_size=batch_size,
                      shuffle=shuffle, drop_last=False)


def balanced_class_weights(labels: np.ndarray, n_classes: int = 4) -> torch.Tensor:
    """Balanced weights from the TRAINING split only; robust to absent classes."""
    present = np.unique(labels)
    w = compute_class_weight("balanced", classes=present, y=labels)
    weights = np.ones(n_classes, dtype=np.float32)
    weights[present] = w
    return torch.as_tensor(weights, dtype=torch.float32)


def rr_standardizer(rr_train: np.ndarray):
    """Return (mu, sd) from the training RR features to standardize all splits."""
    mu = rr_train.mean(0)
    sd = rr_train.std(0) + 1e-8
    return mu.astype(np.float32), sd.astype(np.float32)


def apply_rr_standardization(rr, mu, sd):
    return ((rr - mu) / sd).astype(np.float32)


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
