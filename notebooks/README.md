# Notebooks

Place your main Colab notebook here, e.g. `ecg_arrhythmia.ipynb`.

The notebook is the original end-to-end entry point used to produce the reported
results (Milestone C). The `src/` modules mirror its stages so the pipeline can also
be run as importable, testable Python:

- dataset loading & preprocessing  → `src/data.py`, `src/preprocessing.py`
- model construction               → `src/models.py`
- training loop                    → `src/train.py`
- evaluation & figures             → `src/evaluate.py`

To export your Colab notebook: **File → Download → Download .ipynb**, then drop it here.
