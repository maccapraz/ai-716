"""Signal preprocessing and beat segmentation for MIT-BIH.

Pipeline (matches the Milestone C notebook):
    1. Third-order zero-phase Butterworth band-pass filter (0.5-40 Hz).
    2. Per-record z-score normalization.
    3. 360-sample beat windows (+/-180 samples ~ +/-0.5 s) centered on each R-peak.
    4. Three RR-interval features per beat: previous RR, current RR, and a
       10-beat centered local-average RR (seconds).
    5. AAMI EC57 class mapping to N, S, V, F (paced/unknown Q dropped).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

FS = 360                      # MIT-BIH sampling rate (Hz)
BEAT_WIN = 180                # samples each side of the R-peak -> 360-sample beats
LOWCUT = 0.5
HIGHCUT = 40.0

# AAMI EC57 mapping: superclass -> raw annotation symbols.
# The Q class (/, f, Q) is intentionally excluded from modeling.
AAMI = {
    "N": list("NLRej"),   # Normal + bundle-branch + atrial/nodal escape
    "S": list("AaJS"),    # Supraventricular ectopic
    "V": list("VE"),      # Ventricular ectopic
    "F": list("F"),       # Fusion
}
SYM2AAMI = {s: cls for cls, syms in AAMI.items() for s in syms}
CLASSES = ["N", "S", "V", "F"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def bandpass(signal, fs=FS, low=LOWCUT, high=HIGHCUT, order=3):
    """Zero-phase Butterworth band-pass: removes baseline wander + HF noise."""
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal)


def zscore(signal):
    """Per-record standardization so amplitude scale is comparable across patients."""
    return (signal - signal.mean()) / (signal.std() + 1e-8)


def rr_features(beat_peaks, fs=FS):
    """Previous RR, current RR, and 10-beat centered local-average RR (seconds)."""
    beat_peaks = np.asarray(beat_peaks)
    if len(beat_peaks) < 2:
        return np.zeros((len(beat_peaks), 3), dtype=float)
    rr = np.diff(beat_peaks) / fs                     # seconds between R-peaks
    rr = np.concatenate([[rr[0]], rr])                # pad the first beat
    prev_rr = np.concatenate([[rr[0]], rr[:-1]])
    local_avg = pd.Series(rr).rolling(10, min_periods=1, center=True).mean().values
    return np.column_stack([prev_rr, rr, local_avg])  # (n_beats, 3)


def segment_record(signal, ann_samples, ann_symbols):
    """Filter, normalize, then extract labeled beats and aligned RR features.

    Returns (beats [N,360], rr [N,3], labels [N]) keeping only AAMI N/S/V/F beats
    that have a full window.
    """
    sig = zscore(bandpass(signal))
    peaks, labels = [], []
    for peak, sym in zip(ann_samples, ann_symbols):
        if sym not in SYM2AAMI:                       # non-beat / Q annotation
            continue
        if peak - BEAT_WIN < 0 or peak + BEAT_WIN >= len(sig):
            continue                                  # incomplete edge beat
        peaks.append(peak)
        labels.append(CLASS_TO_IDX[SYM2AAMI[sym]])
    peaks = np.array(peaks)
    if len(peaks) == 0:
        return (np.empty((0, 2 * BEAT_WIN), np.float32),
                np.empty((0, 3), np.float32),
                np.empty((0,), np.int64))
    beats = np.stack([sig[p - BEAT_WIN:p + BEAT_WIN] for p in peaks]).astype(np.float32)
    rr = rr_features(peaks).astype(np.float32)        # aligned 1:1 with beats
    return beats, rr, np.array(labels, dtype=np.int64)
