"""The three architectures, all behind a shared (beat, rr) -> logits interface.

    CNN1D     ~36k params  -- beat morphology only
    LSTMNet   ~55k params  -- beat (subsampled to 180) + RR features
    CNN-LSTM  ~86k params  -- conv front-end + bidirectional LSTM + RR features
"""
from __future__ import annotations

import torch
import torch.nn as nn

N_CLASSES = 4
RR_DIM = 3


class CNN1D(nn.Module):
    """Three-block 1-D CNN over beat morphology (RR features ignored)."""

    def __init__(self, n_classes: int = N_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.5), nn.Linear(128, n_classes))

    def forward(self, x, rr=None):            # rr ignored by the CNN baseline
        # x: (B, 360) -> (B, 1, 360)
        return self.head(self.features(x.unsqueeze(1)))


class LSTMNet(nn.Module):
    """Two-layer LSTM over the beat (subsampled 2x) fused with RR features."""

    def __init__(self, n_classes: int = N_CLASSES, hidden: int = 128, subsample: int = 2):
        super().__init__()
        self.subsample = subsample
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, num_layers=2,
                            batch_first=True, dropout=0.3)
        self.head = nn.Sequential(
            nn.Linear(hidden + RR_DIM, 64), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, n_classes),
        )

    def forward(self, x, rr):
        seq = x[:, ::self.subsample].unsqueeze(-1)   # (B, 180, 1)
        _, (h, _) = self.lstm(seq)
        feat = torch.cat([h[-1], rr], dim=1)
        return self.head(feat)


class CNNLSTM(nn.Module):
    """Convolutional front-end feeding a bidirectional LSTM, fused with RR features."""

    def __init__(self, n_classes: int = N_CLASSES, hidden: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden, num_layers=1,
                            batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden + RR_DIM, 64), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, n_classes),
        )

    def forward(self, x, rr):
        c = self.conv(x.unsqueeze(1))          # (B, 64, T)
        seq = c.permute(0, 2, 1)               # (B, T, 64)
        _, (h, _) = self.lstm(seq)
        feat = torch.cat([h[-2], h[-1], rr], dim=1)   # both directions + RR
        return self.head(feat)


def build_model(name: str) -> nn.Module:
    return {"cnn1d": CNN1D, "lstm": LSTMNet, "cnn_lstm": CNNLSTM}[name]()


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
