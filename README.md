# VSFC Low-resource Data Augmentation

Experimental pipeline for label-preserving data augmentation on Vietnamese
student feedback sentiment classification.

## Scope

This repository is organized around the execution plan in
`../codex_execution_plan.md`. The initial bootstrap includes reproducible
configuration, utility helpers, and the project layout needed for later data,
augmentation, training, and analysis phases.

## Quick Checks

```bash
pip install -r requirements.txt
python -c "from src.utils.seed import set_seed; set_seed(42)"
python -c "import yaml; print(yaml.safe_load(open('configs/base.yaml')))"
```

## Phase 3 GPU Run

Local CPU can verify code, but PhoBERT acceptance should run on a CUDA GPU
such as Kaggle/Colab T4.

```bash
python scripts/check_gpu.py
python scripts/setup_vncorenlp.py
python scripts/download_data.py
python -m src.data.subsample --train-csv data/raw/train.csv --seed 42
python -m src.experiments.run_phobert --ratio 1.00 --seed 42 --augmentation none --decision-rule tune_logit_bias --logging-steps 25
```

The notebook `notebooks/phase3_gpu_run.ipynb` contains the same flow for
Kaggle/Colab. It first runs a tiny GPU smoke test, then runs the full Phase 3
gate. On Kaggle, keep Internet on and use a GPU accelerator; the notebook
defaults to a single visible T4 GPU to avoid multi-GPU `DataParallel` stalls.
The Phase 3 gate is `test.macro_f1 >= 0.85` for `phobert_none_1.00_42`.

## Directory Overview

- `configs/`: shared experiment and model settings.
- `src/`: reusable Python package code.
- `scripts/`: command-line scripts for setup and orchestration.
- `data/`: raw, split, and augmented data. This is gitignored.
- `results/`: logs, predictions, tables, and figures.
- `notebooks/`: exploratory analysis and reporting notebooks.
