"""The three architectures, all behind a shared (beat, rr) -> logits interface.

Inputs: beat x of shape (B, 1, 360) and RR features rr of shape (B, 3).
Matches the Milestone C notebook (CNN1D ignores rr; the recurrent models fuse it).
"""
from __future__ import annotations

import torch
import torch.nn as nn

N_CLASSES = 4
RR_DIM = 3


class CNN1D(nn.Module):
    """1-D CNN baseline over beat morphology. Ignores rr for a uniform signature."""

    def __init__(self, n_classes: int = N_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),                       # global average pooling
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.5), nn.Linear(128, n_classes))

    def forward(self, x, rr=None):                         # x: (B, 1, 360)
        return self.head(self.features(x))


class LSTMNet(nn.Module):
    """Two-layer LSTM over the subsampled beat, fused with the 3 RR features."""

    def __init__(self, n_classes=N_CLASSES, hidden=64, layers=2, rr_dim=RR_DIM, subsample=2):
        super().__init__()
        self.subsample = subsample                          # 360 -> 180 timesteps
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, num_layers=layers,
                            batch_first=True, dropout=0.3)
        self.head = nn.Sequential(
            nn.Linear(hidden + rr_dim, 64), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, n_classes),
        )

    def forward(self, x, rr):                               # x: (B,1,360) rr: (B,3)
        seq = x[:, :, ::self.subsample].transpose(1, 2)     # -> (B, 180, 1)
        _, (h_n, _) = self.lstm(seq)
        return self.head(torch.cat([h_n[-1], rr], dim=1))   # last layer's hidden state


class CNNLSTM(nn.Module):
    """Conv front-end compresses the beat; a bidirectional LSTM orders the features."""

    def __init__(self, n_classes=N_CLASSES, hidden=64, rr_dim=RR_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden, num_layers=1,
                            batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden + rr_dim, 64), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, n_classes),
        )

    def forward(self, x, rr):                               # x: (B,1,360)
        f = self.conv(x)                                    # -> (B, 64, 90)
        f = f.transpose(1, 2)                               # -> (B, 90, 64)
        _, (h_n, _) = self.lstm(f)                          # h_n: (2, B, hidden)
        h = torch.cat([h_n[0], h_n[1]], dim=1)              # forward + backward
        return self.head(torch.cat([h, rr], dim=1))


def build_model(name: str) -> nn.Module:
    return {"cnn1d": CNN1D, "lstm": LSTMNet, "cnn_lstm": CNNLSTM}[name]()


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
