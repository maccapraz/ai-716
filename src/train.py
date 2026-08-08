"""Shared training loop for all three architectures.

Config (matches the report):
    AdamW (lr 1e-3, weight decay 1e-4), batch 128, weighted cross-entropy,
    ReduceLROnPlateau (x0.5, patience 3), gradient clipping (5.0),
    early stopping (patience 5) on validation macro-F1, up to 20 epochs.

Run directly to train all three models and cache their weights:
    python -m src.train
"""
from __future__ import annotations

import os
import copy
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from .data import load_split
from .models import build_model, count_params
from .utils import (set_seed, get_device, make_loader, balanced_class_weights,
                    rr_standardizer, apply_rr_standardization)

MODELS = ["cnn1d", "lstm", "cnn_lstm"]
MAX_EPOCHS = 20
PATIENCE = 5
CKPT_DIR = "results/checkpoints"
RR_STATS = "results/rr_stats.npz"


def _epoch(model, loader, device, criterion, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total_loss, preds, gts = 0.0, [], []
    for beats, rr, y in loader:
        beats, rr, y = beats.to(device), rr.to(device), y.to(device)
        with torch.set_grad_enabled(train):
            logits = model(beats, rr)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        total_loss += loss.item() * len(y)
        preds.append(logits.argmax(1).cpu().numpy()); gts.append(y.cpu().numpy())
    preds, gts = np.concatenate(preds), np.concatenate(gts)
    f1 = f1_score(gts, preds, average="macro", zero_division=0)
    return total_loss / len(loader.dataset), f1


def train_model(name: str, device=None, seed: int = 42):
    set_seed(seed)
    device = device or get_device()

    (Xtr, rr_tr, ytr) = load_split("train")
    (Xva, rr_va, yva) = load_split("val")

    # Standardize RR features with TRAINING statistics only (no leakage); cache them.
    mu, sd = rr_standardizer(rr_tr)
    os.makedirs(os.path.dirname(RR_STATS), exist_ok=True)
    np.savez(RR_STATS, mu=mu, sd=sd)
    rr_tr = apply_rr_standardization(rr_tr, mu, sd)
    rr_va = apply_rr_standardization(rr_va, mu, sd)

    train_loader = make_loader(Xtr, rr_tr, ytr, batch_size=128, shuffle=True)
    val_loader = make_loader(Xva, rr_va, yva, batch_size=256, shuffle=False)

    class_w = balanced_class_weights(ytr).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_w)

    model = build_model(name).to(device)
    print(f"{name}: {count_params(model):,} trainable params")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3)

    history = {k: [] for k in ("train_loss", "val_loss", "train_f1", "val_f1")}
    best_f1, best_state, since_improved = -1.0, None, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        tl, tf = _epoch(model, train_loader, device, criterion, optimizer)
        vl, vf = _epoch(model, val_loader, device, criterion)
        scheduler.step(vf)
        for k, v in zip(history, (tl, vl, tf, vf)):
            history[k].append(v)
        print(f"  epoch {epoch:2d}  train_f1={tf:.3f}  val_f1={vf:.3f}  val_loss={vl:.3f}")
        if vf > best_f1:
            best_f1, best_state, since_improved = vf, copy.deepcopy(model.state_dict()), 0
        else:
            since_improved += 1
            if since_improved >= PATIENCE:
                print(f"  early stopping at epoch {epoch} (best val macro-F1={best_f1:.3f})")
                break

    model.load_state_dict(best_state)          # restore best weights
    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(CKPT_DIR, f"{name}.pt"))
    return model, history, best_f1


if __name__ == "__main__":
    for m in MODELS:
        print(f"\n=== Training {m} ===")
        train_model(m)
