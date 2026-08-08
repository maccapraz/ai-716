"""Download MIT-BIH, apply the de Chazal DS1/DS2 inter-patient split, and cache beats.

Run directly to build all cached splits:
    python -m src.data
"""
from __future__ import annotations

import os
import numpy as np

from .preprocessing import segment_record, CLASSES

# de Chazal et al. (2004) inter-patient partition.
# The four paced records (102, 104, 107, 217) are excluded (AAMI Q class).
DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122,
       124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210,
       212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]
PACED = [102, 104, 107, 217]

# Patient-disjoint validation carved out of DS1 (18 train / 4 validation records).
# These four records are chosen because between them they carry a usable number
# of the rare S and F beats, so the macro-F1 early-stopping signal is meaningful.
VAL_RECORDS = [114, 201, 209, 223]
TRAIN_RECORDS = [r for r in DS1 if r not in VAL_RECORDS]

DATA_DIR = "data/mitdb"
CACHE_DIR = "cache"
MLII_PREFERRED = ("MLII", "II")


def download_mitdb(data_dir: str = DATA_DIR) -> None:
    """Download the MIT-BIH Arrhythmia Database from PhysioNet via wfdb."""
    import wfdb
    os.makedirs(data_dir, exist_ok=True)
    if not os.listdir(data_dir):
        print("Downloading MIT-BIH Arrhythmia Database...")
        wfdb.dl_database("mitdb", data_dir)
    else:
        print(f"MIT-BIH already present in {data_dir}")


def _mlii_channel(sig_names: list[str]) -> int:
    for name in MLII_PREFERRED:
        if name in sig_names:
            return sig_names.index(name)
    return 0  # fall back to the first channel


def build_split(records: list[int], data_dir: str = DATA_DIR):
    """Segment a list of records into (beats, rr, labels) arrays."""
    import wfdb
    beats, rr, labels = [], [], []
    for rec in records:
        path = os.path.join(data_dir, str(rec))
        record = wfdb.rdrecord(path)
        ann = wfdb.rdann(path, "atr")
        ch = _mlii_channel(list(record.sig_name))
        b, r, y = segment_record(record.p_signal[:, ch],
                                 np.asarray(ann.sample), list(ann.symbol))
        beats.append(b); rr.append(r); labels.append(y)
    return (np.concatenate(beats), np.concatenate(rr), np.concatenate(labels))


def build_and_cache(data_dir: str = DATA_DIR, cache_dir: str = CACHE_DIR) -> None:
    """Build train/val/test splits and cache them to compressed .npz files."""
    download_mitdb(data_dir)
    os.makedirs(cache_dir, exist_ok=True)
    for name, recs in [("train", TRAIN_RECORDS), ("val", VAL_RECORDS), ("test", DS2)]:
        out = os.path.join(cache_dir, f"{name}.npz")
        if os.path.exists(out):
            print(f"[cache] {name} already built -> {out}")
            continue
        b, r, y = build_split(recs, data_dir)
        np.savez_compressed(out, beats=b, rr=r, labels=y)
        dist = {CLASSES[i]: int((y == i).sum()) for i in range(len(CLASSES))}
        print(f"[cache] {name}: {len(y)} beats {dist} -> {out}")


def load_split(name: str, cache_dir: str = CACHE_DIR):
    """Load a cached split as (beats, rr, labels)."""
    d = np.load(os.path.join(cache_dir, f"{name}.npz"))
    return d["beats"], d["rr"], d["labels"]


if __name__ == "__main__":
    build_and_cache()
