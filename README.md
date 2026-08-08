# ECG Arrhythmia Classification — Comparing CNN and Recurrent Architectures

Inter-patient heartbeat classification on the **MIT-BIH Arrhythmia Database** under the
ANSI/AAMI EC57 four-class scheme (Normal, Supraventricular ectopic, Ventricular ectopic,
Fusion). This repository accompanies the AI-716 Final Project Report and contains the
complete, reproducible pipeline for the CNN1D, LSTM, and CNN-LSTM models.

> **Course:** AI-716 Advanced Artificial Intelligence · Capitol Technology University
> **Author:** Carlos Prazeres

---

## Key result (DS2 inter-patient test)

| Model | Accuracy | Macro-F1 | F1 N | F1 S | F1 V | F1 F |
|-----------|:--------:|:--------:|:----:|:----:|:----:|:----:|
| **CNN1D** | 0.770 | **0.395** | 0.872 | 0.156 | 0.552 | 0.001 |
| LSTM | 0.710 | 0.345 | 0.841 | 0.106 | 0.353 | 0.079 |
| CNN-LSTM | 0.716 | 0.335 | 0.835 | 0.066 | 0.415 | 0.023 |

The simplest model (CNN1D) generalized best. Macro-F1 is the headline metric because
~89% of beats are Normal; a majority-class baseline scores only ~0.24 macro-F1.

---

## Repository structure

```
.
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── notebooks/
│   └── ecg_arrhythmia.ipynb      # End-to-end Colab notebook (main entry point)
├── src/
│   ├── data.py                   # Download, filtering, segmentation, DS1/DS2 split
│   ├── preprocessing.py          # Butterworth band-pass, z-score, RR-interval features
│   ├── models.py                 # CNN1D, LSTMNet, CNN-LSTM definitions
│   ├── train.py                  # Shared training loop (AdamW, weighted CE, early stop)
│   ├── evaluate.py               # Macro-F1, per-class metrics, confusion matrices
│   └── utils.py                  # Caching, seeding, plotting helpers
├── cache/                        # Cached .npz beat tensors (git-ignored)
├── results/
│   ├── results_summary.csv       # Per-model metrics
│   ├── milestone_c_results.json  # Full run snapshot
│   └── figures/                  # Learning curves, confusion matrices
└── report/
    └── Milestone_E_Final_Report.docx
```

> If your work currently lives in a single Colab notebook, you can start by committing it
> under `notebooks/` and splitting out `src/` modules over time. The structure above is the
> target; a single well-commented notebook plus this README already satisfies the deliverable.

---

## File purposes

- **`src/data.py`** — Downloads MIT-BIH via `wfdb`, applies the de Chazal DS1/DS2 inter-patient
  partition, excludes the four paced records (102, 104, 107, 217), and segments 360-sample
  beat windows centered on annotated R-peaks.
- **`src/preprocessing.py`** — Third-order zero-phase Butterworth band-pass (0.5–40 Hz),
  per-record z-score normalization, and the three RR-interval features (previous, current,
  10-beat local average).
- **`src/models.py`** — The three architectures behind a shared `(beat, rr) -> logits`
  interface: CNN1D (36,228 params), LSTMNet (55,044), CNN-LSTM (86,020).
- **`src/train.py`** — One training loop for all models: AdamW (lr 1e-3, wd 1e-4), weighted
  cross-entropy with training-split class weights, ReduceLROnPlateau, gradient clipping (5.0),
  and early stopping on validation macro-F1 with best-weight restoration.
- **`src/evaluate.py`** — Computes macro-F1, per-class precision/recall/F1, and confusion
  matrices on the held-out DS2 test set.

---

## Environment setup

```bash
# Python 3.10+ recommended
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**`requirements.txt`**
```
torch>=2.1
wfdb>=4.3
numpy
scipy
scikit-learn
matplotlib
pandas
```

---

## Run instructions

**Option A — Colab (as originally executed):**
Open `notebooks/ecg_arrhythmia.ipynb`, set runtime to GPU (Tesla T4), and run all cells
top to bottom. The first run downloads and caches beat tensors to `cache/`; later runs skip
re-segmentation.

**Option B — Local scripts (if modularized):**
```bash
python -m src.data          # download + preprocess + cache
python -m src.train         # train CNN1D, LSTM, CNN-LSTM
python -m src.evaluate      # metrics + confusion matrices -> results/
```

The full pipeline runs in well under a minute per model once the cache is built.

---

## Notes on reproducibility

- Random seeds are fixed for NumPy and PyTorch.
- Class weights and normalization statistics are computed from the **training split only**
  to prevent leakage.
- Evaluation uses the **patient-disjoint** DS1/DS2 protocol; no subject appears in both
  training and test.

## Data

MIT-BIH Arrhythmia Database is publicly available from PhysioNet
(https://physionet.org/content/mitdb/). It is downloaded automatically by `wfdb`; raw
recordings are **not** committed to this repository.
