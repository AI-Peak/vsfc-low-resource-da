# Execution Log

Generated at: 2026-06-27 13:00:02 +07:00

## Cost Log

- Paid API calls: none.
- Gemini API usage: none.
- Cloud/GPU usage: none.
- Package/data/model downloads: free public downloads only.
- Estimated direct money spent by Codex work so far: 0 VND.

Downloaded/cached resources:

- Python packages from package registry into `.venv/`.
- UIT-VSFC dataset from HuggingFace.
- PhoBERT tokenizer files from HuggingFace for notebook/tokenizer sanity check.
- VnCoreNLP model files from GitHub into `vncorenlp/`.

## Phase 0 - Bootstrap

Status: completed and verified.

Actions:

- Read `codex_execution_plan.md`.
- Created project directory `vsfc-low-resource-da/`.
- Created repository structure:
  - `configs/`
  - `src/`
  - `src/data/`
  - `src/augmentation/`
  - `src/models/`
  - `src/evaluation/`
  - `src/experiments/`
  - `src/utils/`
  - `data/raw/`
  - `data/splits/`
  - `data/augmented/`
  - `results/logs/`
  - `results/predictions/`
  - `results/tables/`
  - `notebooks/`
  - `scripts/`
- Added `README.md`.
- Added `requirements.txt`.
- Added `.gitignore`, including the planned ignored paths plus `.venv/`.
- Added config files:
  - `configs/base.yaml`
  - `configs/phobert.yaml`
  - `configs/augmentation.yaml`
- Added package/module stubs for later phases.
- Implemented:
  - `src/utils/seed.py`
  - `src/utils/io.py`
  - `src/utils/logging.py`
- Created local virtual environment `.venv/`.
- Installed all dependencies from `requirements.txt`.

Verification:

- `from src.utils.seed import set_seed; set_seed(42)` passed.
- `yaml.safe_load(open('configs/base.yaml'))` passed.
- `pip check` passed with `No broken requirements found`.

Notes:

- `python` was not available globally in the shell, so all checks use `.\.venv\Scripts\python.exe`.
- Notebooks and non-Phase-0 modules were intentionally created as placeholders at this stage.

## Phase 1 - Data Pipeline

Status: completed and verified.

Actions:

- Reviewed Phase 0 feedback supplied by the user.
- Implemented `src/data/load.py`:
  - Loads `uitnlp/vietnamese_students_feedback` from HuggingFace.
  - Keeps only `sentence` and `sentiment`.
  - Resolves train/dev/test split names.
  - Prints split size and label distribution.
  - Saves raw split CSVs when requested.
- Implemented `scripts/download_data.py`:
  - Downloads UIT-VSFC.
  - Saves `data/raw/train.csv`, `data/raw/dev.csv`, `data/raw/test.csv`.
  - Adds project root to `sys.path` for direct script execution.
- Implemented `src/data/preprocess.py`:
  - Unicode NFC normalization.
  - Whitespace cleanup.
  - Optional lowercasing, default off.
  - VnCoreNLP word segmentation.
  - Reuses a single segmenter instance.
  - Auto-detects `JAVA_HOME`.
  - Uses a temp runtime copy of VnCoreNLP on Windows when the project path contains spaces.
- Implemented `scripts/setup_vncorenlp.py`:
  - Downloads VnCoreNLP model files.
  - Uses `py_vncorenlp.download_model` first.
  - Adds a Python `urlretrieve` fallback because the upstream package calls `wget`, which is fragile on Windows.
  - Smoke-tests segmentation on sample Vietnamese sentences.
- Implemented `src/data/subsample.py`:
  - Stratified sampling by `sentiment`.
  - Generates 5%, 10%, 20%, and 100% subsets.
  - Saves stable filenames like `train_0.05_42.csv`.
- Implemented `src/data/dataset.py`:
  - PyTorch `Dataset`.
  - Wraps HuggingFace tokenizer.
  - Uses truncation and padding to `max_length=128`.
- Replaced empty `notebooks/01_data_exploration.ipynb` with real exploration cells:
  - Split sizes.
  - Sentence length distribution.
  - Mean, median, and 95th percentile length stats.
  - Label distribution per split.
  - Five sample sentences per class.
  - PhoBERT tokenizer vocabulary/OOV sanity stats.

Dependency change:

- Changed `requirements.txt` from `datasets>=2.14.0` to `datasets>=2.14.0,<3.0.0`.
- Reason: `datasets 5.0.0` no longer supports dataset scripts, and UIT-VSFC currently uses a dataset script.
- Updated loader to call `load_dataset(..., trust_remote_code=True)`.

Downloaded data:

- `data/raw/train.csv`: 11,426 rows.
- `data/raw/dev.csv`: 1,583 rows.
- `data/raw/test.csv`: 3,166 rows.

Label distribution observed:

- Train:
  - negative: 5,325 rows, 46.60%.
  - neutral: 458 rows, 4.01%.
  - positive: 5,643 rows, 49.39%.
- Dev:
  - negative: 705 rows, 44.54%.
  - neutral: 73 rows, 4.61%.
  - positive: 805 rows, 50.85%.
- Test:
  - negative: 1,409 rows, 44.50%.
  - neutral: 167 rows, 5.27%.
  - positive: 1,590 rows, 50.22%.

Generated split files:

- `data/splits/train_0.05_42.csv`: 571 rows.
- `data/splits/train_0.10_42.csv`: 1,142 rows.
- `data/splits/train_0.20_42.csv`: 2,286 rows.
- `data/splits/train_1.00_42.csv`: 11,426 rows.

Subset distribution check:

- `train_0.05_42.csv`: max percentage-point difference from full train = 0.02.
- `train_0.10_42.csv`: max percentage-point difference from full train = 0.02.
- `train_0.20_42.csv`: max percentage-point difference from full train = 0.01.
- `train_1.00_42.csv`: max percentage-point difference from full train = 0.00.

VnCoreNLP verification:

- Java exists on the machine.
- `JAVA_HOME` was missing, so the code now auto-detects `C:/Program Files/Java/jdk-22`.
- VnCoreNLP model files were downloaded into `vncorenlp/`.
- Direct segmentation smoke test passed:
  - Input: `Giảng viên dạy rất dễ hiểu.`
  - Output: `Giảng_viên dạy rất dễ hiểu .`

Notebook verification:

- `.venv` does not include `jupyter-nbconvert`, because Jupyter is not in the planned requirements.
- Verified `01_data_exploration.ipynb` by executing its code cells through a small JSON code-cell runner.
- Notebook logic ran successfully.
- PhoBERT tokenizer sanity stats:
  - Tokenizer vocab size: 64,000.
  - Observed token types in train: 2,578.
  - Total tokens: 166,830.
  - Unknown tokens: 4.
  - Unknown-token rate: 0.0024%.

Verification commands/results:

- `python -m compileall src scripts`: passed.
- `python -m src.data.load --save-raw`: passed.
- `python -m src.data.subsample --train-csv data/raw/train.csv --seed 42 --force`: passed.
- `pip check`: passed.
- Imports for `load_uit_vsfc`, `create_subsamples`, and `VSFCDataset`: passed.

## Phase 2 - Classical Baseline

Updated at: 2026-06-27 13:08:18 +07:00

Status: completed and verified.

Actions:

- Implemented `src/models/classical.py`.
  - Added TF-IDF + LogisticRegression pipeline.
  - TF-IDF settings: 1-2 grams, `max_features=20000`, `min_df=2`.
  - LogisticRegression settings: `class_weight="balanced"`, `max_iter=1000`, seeded `random_state`.
  - Added prediction output dataclasses for labels and probabilities.
- Implemented `src/evaluation/metrics.py`.
  - Added `compute_metrics`.
  - Returns `macro_f1`, `weighted_f1`, `accuracy`, and `per_class_f1`.
  - Added `confusion_matrix_plot` using seaborn heatmap.
- Implemented `src/experiments/run_baseline.py`.
  - CLI args: `--ratio`, `--seed`, `--data-dir`, `--results-dir`, `--force-subsample`.
  - Loads `data/splits/train_{ratio}_{seed}.csv`.
  - Loads dev/test from `data/raw/`.
  - Creates missing train subset if needed.
  - Saves predictions to `results/predictions/baseline_{ratio}_{seed}.csv`.
  - Saves metrics to `results/logs/baseline_{ratio}_{seed}.json`.

Generated artifacts:

- `results/predictions/baseline_1.00_42.csv`.
- `results/logs/baseline_1.00_42.json`.

Acceptance run:

- Command: `python -m src.experiments.run_baseline --ratio 1.00 --seed 42`.
- Train size: 11,426.
- Dev size: 1,583.
- Test size: 3,166.
- Dev macro-F1: 0.7500.
- Test macro-F1: 0.7147.
- Acceptance threshold: test macro-F1 >= 0.70.
- Result: passed.

Prediction CSV schema:

- `sentence`
- `true_label`
- `predicted_label`
- `prob_0`
- `prob_1`
- `prob_2`

Verification commands/results:

- `python -m compileall src scripts`: passed.
- Small in-memory baseline/schema smoke test: passed.
- `pip check`: passed.
- Prediction CSV shape: 3,166 rows x 6 columns.

Phase 2 cleanup update:

- Updated at: 2026-06-27 13:17:20 +07:00.
- Added confusion matrix artifact generation to `src/experiments/run_baseline.py`.
- Figure path: `results/tables/figures/baseline_1.00_42_confusion_matrix.png`.
- Added default skip-if-exists behavior for Phase 7 orchestration.
- Added `--overwrite` to force reruns when needed.
- If predictions and metrics exist but the figure is missing, the runner now creates the figure from existing predictions without retraining.
- Verified rerun behavior:
  - First rerun created the missing confusion matrix from `baseline_1.00_42.csv`.
  - Second rerun skipped because predictions, metrics, and figure all existed.

## Current Stop Point

Per the execution plan, work stopped after Phase 2 to report status before Phase 3.

Phase 3 reminder:

- Wire `src.data.preprocess.preprocess_text` into the PhoBERT training pipeline.
- PhoBERT should receive VnCoreNLP word-segmented text before tokenization.
- Practical options:
  - preprocess train/dev/test inside `run_phobert.py` before creating `VSFCDataset`; or
  - cache preprocessed CSVs and load them idempotently.
- Be careful not to leave PhoBERT consuming raw unsegmented sentences.

## GPU Environment Check Before Phase 3

Checked at: 2026-06-27 13:24:51 +07:00.

Local machine status:

- `nvidia-smi`: not found.
- `nvcc`: not found.
- Windows video controller: AMD Radeon(TM) Graphics.
- PyTorch version in `.venv`: `2.12.1+cpu`.
- `torch.version.cuda`: `None`.
- `torch.cuda.is_available()`: `False`.
- `torch.cuda.device_count()`: `0`.
- cuDNN available: `False`.
- `transformers` installed: yes.
- `accelerate` installed: no.
- `Trainer` import: passed.

Conclusion:

- The current local environment is not GPU-ready for PhoBERT fine-tuning.
- It can be used for Phase 3 code implementation and small CPU smoke tests.
- The Phase 3 100% gating run should be done on Kaggle/Colab T4 or another NVIDIA CUDA environment.
- Before a real GPU run, verify:
  - `nvidia-smi`
  - `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"`
  - install `accelerate` if HuggingFace `Trainer` requires it in that environment.

Next phase:

- Phase 3 - PhoBERT Fine-tuning.
- Implement `PhoBERTTrainer`.
- Implement `src/experiments/run_phobert.py`.
- Run 100% no-augmentation gating experiment.

## Phase 3 - PhoBERT Fine-tuning

Updated at: 2026-06-27 13:52:00 +07:00.

Status: implementation completed; GPU gating run not executed locally because this
machine has no CUDA GPU.

Actions:

- Added `accelerate>=0.25.0` to `requirements.txt`.
- Installed `accelerate` into local `.venv`.
- Added `.gitignore` entries:
  - `data/preprocessed/`
  - `results/models/`
- Implemented `src/models/phobert_trainer.py`.
  - Loads `vinai/phobert-base` with `AutoTokenizer` and
    `AutoModelForSequenceClassification`.
  - Wraps HuggingFace `Trainer`.
  - Uses dev macro-F1 as `metric_for_best_model`.
  - Adds early stopping with patience from `configs/phobert.yaml`.
  - Supports current/new `transformers` `processing_class` and older
    `tokenizer` Trainer APIs.
  - Saves lightweight artifacts under `results/models/{run_name}/`:
    classifier head weights, tokenizer files, model config, and metadata.
  - Removes full Trainer checkpoints after training unless
    `--keep-checkpoints` is passed.
- Implemented `src/experiments/run_phobert.py`.
  - CLI args: `--ratio`, `--seed`, `--augmentation`, `--overwrite`,
    `--force-preprocess`, `--cpu`, `--keep-checkpoints`, sample-limit smoke
    args, and config overrides.
  - Loads low-resource train split and raw dev/test CSVs.
  - Supports future augmentation inputs:
    - `data/augmented/eda_{ratio}_{seed}.csv`
    - `data/augmented/llm_raw_{ratio}_{seed}.csv`
    - `data/augmented/llm_filtered_{ratio}_{seed}.csv`
  - Combines original training data with augmented rows when augmentation is
    selected.
  - Wires `src.data.preprocess.preprocess_frame` into the training path before
    tokenization.
  - Caches segmented CSVs under `data/preprocessed/`.
  - Saves predictions to
    `results/predictions/phobert_{augmentation}_{ratio}_{seed}.csv`.
  - Saves metrics to
    `results/logs/phobert_{augmentation}_{ratio}_{seed}.json`.
  - Saves confusion matrix figures to `results/tables/figures/`.
  - Has skip-if-exists behavior and `--overwrite`, matching the baseline
    runner for Phase 7 orchestration.
  - Enforces the Phase 3 gate for `augmentation=none`, `ratio=1.00`,
    `seed=42`: test macro-F1 must be at least `0.85`.
- Added `scripts/check_gpu.py`.
  - Prints Python, torch, transformers, accelerate, `nvidia-smi`, CUDA, and
    cuDNN status.
- Added `notebooks/phase3_gpu_run.ipynb`.
  - Kaggle/Colab notebook for installing requirements, checking GPU, setting up
    VnCoreNLP, downloading UIT-VSFC, creating splits, and running the Phase 3
    gate.
- Updated `README.md` with Phase 3 GPU run commands.

Verification:

- `python -m compileall src scripts`: passed.
- `python -m src.experiments.run_phobert --help`: passed.
- `notebooks/phase3_gpu_run.ipynb` JSON parse: passed.
- `pip check`: passed with `No broken requirements found`.
- `scripts/check_gpu.py`: passed and confirms local CPU-only environment.
- VnCoreNLP preprocessing smoke test:
  - Input: `Giảng viên dạy rất dễ hiểu.`
  - Output: `Giảng_viên dạy rất dễ hiểu .`

Local GPU status after dependency update:

- `nvidia-smi`: not found.
- PyTorch: `2.12.1+cpu`.
- `torch.version.cuda`: `None`.
- `torch.cuda.is_available()`: `False`.
- `torch.cuda.device_count()`: `0`.
- `accelerate`: installed.

Phase 3 acceptance status:

- Code and GPU run notebook are ready.
- Full gating run is still pending on Kaggle/Colab T4 or another CUDA GPU:

```powershell
python -m src.experiments.run_phobert --ratio 1.00 --seed 42 --augmentation none
```

- Required result: `test.macro_f1 >= 0.85`.

## Phase 3 Kaggle Timeout Triage

Observation:

- Kaggle Version 2 was canceled after `43200.4s` with `timeout exceeded`
  (`Exit code: 137`).
- GPU was configured correctly in the run:
  - `torch.cuda.is_available(): True`
  - `torch.cuda.device_count(): 2`
  - Accelerator: `GPU T4 x2`
- Setup, VnCoreNLP, data download, split creation, and preprocessing completed.
- The log reached PhoBERT training startup (`0/3580`) and then produced no
  useful progress before Kaggle's 12-hour limit.

Conclusion:

- This is a Kaggle runtime timeout/stall, not a Phase 3 metric failure.
- Phase 3 acceptance is still pending because no final
  `results/logs/phobert_none_1.00_42.json` metric was produced.

Remediation implemented:

- Pinned stable HuggingFace training dependencies in `requirements.txt`:
  - `transformers==4.37.2`
  - `accelerate==0.25.0`
  - `peft==0.7.1`
- Updated `VSFCDataset` to tokenize each split once during dataset creation
  instead of tokenizing inside every training `__getitem__`.
- Changed Trainer logging from epoch-only to step logging with
  `logging_first_step=True` and configurable `--logging-steps`.
- Disabled `full_determinism` by default for PhoBERT Trainer; deterministic
  seeds are still set, but CUDA deterministic algorithm enforcement is avoided.
- Updated `notebooks/phase3_gpu_run.ipynb`:
  - forces `CUDA_VISIBLE_DEVICES=0` so Kaggle uses one T4 instead of
    multi-GPU `DataParallel`;
  - pulls the latest GitHub code when an existing Kaggle clone is present;
  - checks that installed `transformers` is 4.x after `pip install`;
  - runs a tiny GPU smoke test before the full Phase 3 gate;
  - runs the full gate with `--num-epochs 5 --logging-steps 25`.
- Added a Kaggle-side safeguard in `PhoBERTTrainer` so even an older imported
  notebook defaults to one visible GPU unless `VSFC_USE_ALL_GPUS=1` is set.
- Pinned `peft==0.7.1` after Kaggle smoke test exposed an import mismatch:
  Kaggle's preinstalled newer `peft` expected
  `accelerate.utils.memory.clear_device_cache`, which is not available in
  `accelerate==0.25.0`.

Next action:

- Rerun Kaggle from the updated GitHub notebook/repo.
- If the smoke test does not show step logs within a few minutes, stop the run
  and inspect the new log immediately instead of waiting for the 12-hour limit.
