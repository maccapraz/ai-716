"""Signal preprocessing and beat segmentation for MIT-BIH.

Pipeline (matches the report's Methodology):
    1. Third-order zero-phase Butterworth band-pass filter (0.5-40 Hz).
    2. Per-record z-score normalization.
    3. 360-sample beat windows (+/-0.5 s) centered on each annotated R-peak.
    4. Three RR-interval features per beat (previous, current, 10-beat local average).
    5. AAMI EC57 class mapping to N, S, V, F (paced/unknown Q dropped).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt

FS = 360                      # MIT-BIH sampling rate (Hz)
WINDOW = 360                  # samples per beat (~+/-0.5 s)
HALF = WINDOW // 2

# AAMI EC57 mapping: raw annotation symbol -> superclass.
# The Q class (/, f, Q) is intentionally excluded from modeling.
AAMI = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",   # Normal
    "A": "S", "a": "S", "J": "S", "S": "S",             # Supraventricular ectopic
    "V": "V", "E": "V",                                 # Ventricular ectopic
    "F": "F",                                           # Fusion
}
CLASSES = ["N", "S", "V", "F"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def bandpass_filter(signal: np.ndarray, fs: int = FS,
                    low: float = 0.5, high: float = 40.0, order: int = 3) -> np.ndarray:
    """Zero-phase Butterworth band-pass to remove baseline wander and HF noise."""
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal)


def zscore(signal: np.ndarray) -> np.ndarray:
    """Per-record z-score so amplitude differences do not dominate learning."""
    mu, sd = signal.mean(), signal.std()
    return (signal - mu) / (sd + 1e-8)


def rr_features(samples: np.ndarray, idx: int) -> np.ndarray:
    """Previous RR, current RR, and 10-beat local-average RR (in seconds)."""
    prev_rr = (samples[idx] - samples[idx - 1]) / FS if idx > 0 else 0.0
    next_rr = (samples[idx + 1] - samples[idx]) / FS if idx < len(samples) - 1 else 0.0
    lo = max(0, idx - 10)
    local = np.diff(samples[lo:idx + 1]) / FS
    local_avg = float(local.mean()) if local.size else 0.0
    return np.array([prev_rr, next_rr, local_avg], dtype=np.float32)


def segment_record(signal: np.ndarray, ann_samples: np.ndarray, ann_symbols: list[str]):
    """Filter, normalize, then extract labeled beats and RR features from one record.

    Returns (beats [N,360], rr [N,3], labels [N]) keeping only AAMI N/S/V/F beats
    that have a full window.
    """
    sig = zscore(bandpass_filter(signal))
    beats, rr, labels = [], [], []
    for i, (s, sym) in enumerate(zip(ann_samples, ann_symbols)):
        cls = AAMI.get(sym)
        if cls is None:                       # non-beat annotation or Q class
            continue
        start, end = s - HALF, s + HALF
        if start < 0 or end > len(sig):       # incomplete window at recording edge
            continue
        beats.append(sig[start:end].astype(np.float32))
        rr.append(rr_features(ann_samples, i))
        labels.append(CLASS_TO_IDX[cls])
    if not beats:
        return (np.empty((0, WINDOW), np.float32),
                np.empty((0, 3), np.float32),
                np.empty((0,), np.int64))
    return np.stack(beats), np.stack(rr), np.array(labels, dtype=np.int64)
