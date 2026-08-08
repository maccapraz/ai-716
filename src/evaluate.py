"""Evaluate trained models on the held-out DS2 inter-patient test set.

Reports macro-F1, per-class precision/recall/F1, and confusion matrices, and
writes results/results_summary.csv plus confusion-matrix figures.

Run directly (after training):
    python -m src.evaluate
"""
from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                             confusion_matrix)

from .data import load_split
from .models import build_model
from .preprocessing import CLASSES
from .utils import get_device, make_loader

CKPT_DIR = "results/checkpoints"
RESULTS_DIR = "results"
FIG_DIR = "results/figures"
MODELS = ["cnn1d", "lstm", "cnn_lstm"]


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, gts = [], []
    for beats, rr, y in loader:
        logits = model(beats.to(device), rr.to(device))
        preds.append(logits.argmax(1).cpu().numpy()); gts.append(y.numpy())
    return np.concatenate(gts), np.concatenate(preds)


def plot_confusion(cm, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG_DIR, exist_ok=True)
    cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(4), CLASSES); ax.set_yticks(range(4), CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"{name} (recall)")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, fraction=0.046); fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"cm_{name}.png"), dpi=150); plt.close(fig)


def evaluate_all():
    device = get_device()
    test_loader = make_loader(*load_split("test"), batch_size=256, shuffle=False)
    rows, full = [], {}
    for name in MODELS:
        model = build_model(name).to(device)
        model.load_state_dict(torch.load(os.path.join(CKPT_DIR, f"{name}.pt"),
                                         map_location=device))
        y, p = predict(model, test_loader, device)
        acc = accuracy_score(y, p)
        macro = f1_score(y, p, average="macro", zero_division=0)
        pr, rc, f1, _ = precision_recall_fscore_support(
            y, p, labels=range(4), zero_division=0)
        cm = confusion_matrix(y, p, labels=range(4))
        plot_confusion(cm, name)
        rows.append({"model": name, "accuracy": round(acc, 3),
                     "macro_f1": round(macro, 3),
                     **{f"f1_{c}": round(f1[i], 3) for i, c in enumerate(CLASSES)}})
        full[name] = {"accuracy": acc, "macro_f1": macro,
                      "precision": pr.tolist(), "recall": rc.tolist(),
                      "f1": f1.tolist(), "confusion_matrix": cm.tolist()}
        print(f"{name}: acc={acc:.3f} macro-F1={macro:.3f} "
              f"per-class F1={[round(x, 3) for x in f1]}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "results_summary.csv"), index=False)
    with open(os.path.join(RESULTS_DIR, "results_full.json"), "w") as f:
        json.dump(full, f, indent=2)
    best = max(rows, key=lambda r: r["macro_f1"])
    print(f"\nBest model by macro-F1: {best['model']} ({best['macro_f1']})")


if __name__ == "__main__":
    evaluate_all()
