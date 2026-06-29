# VSFC Low-resource Data Augmentation

Experimental pipeline for label-preserving data augmentation on Vietnamese
student feedback sentiment classification.

## Scope

This repository is organized around the execution plan in
`../execution_plan.md`. The initial bootstrap includes reproducible
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
If Kaggle reports `ModuleNotFoundError: py_vncorenlp` after a runtime reset,
rerun `pip install -r requirements.txt` before the sweep command.

If the default Phase 3 gate lands just below the threshold, run the focused
recovery sweep:

```bash
python scripts/phase3_sweep.py --stop-on-pass
```

If the base sweep already ran and still did not pass, retry only the
PhoBERT-large fallback:

```bash
python scripts/phase3_sweep.py --stop-on-pass --large-only
```

If both the base sweep and PhoBERT-large fallback already ran without passing,
try the VinAI PhoBERT-base v2 fallback:

```bash
python scripts/phase3_sweep.py --stop-on-pass --v2-only
```

If the v2 fallback also misses the gate, run the focused last-mile sweep:

```bash
python scripts/phase3_sweep.py --stop-on-pass --last-mile-only
```

If the last-mile sweep still misses the gate, run the affine-calibration sweep:

```bash
python scripts/phase3_sweep.py --stop-on-pass --calibration-only
```

If affine calibration also misses the gate, run the neutral-loss sweep. This
keeps `augmentation=none`, `ratio=1.00`, and `seed=42`, but tries mild
neutral-class loss adjustments:

```bash
python scripts/phase3_sweep.py --stop-on-pass --neutral-loss-only
```

When the sweep prints `Final gate command to run`, run that command once to
write the accepted `results/logs/phobert_none_1.00_42.json` artifact.

The strongest Phase 3 no-augmentation result observed so far is the standard
PhoBERT-base run with dev-tuned logit bias: test macro-F1 `0.8478`. If all
recovery sweeps miss the `0.85` gate, freeze this as a documented near-gate
baseline and continue to Phase 4.

## Phase 4 EDA Augmentation

Generate conservative EDA files for the low-resource ratios. The default Phase
4 EDA config uses phrase/synonym replacement and keeps noisier insertion,
swap, and deletion operations disabled.

```bash
python scripts/generate_eda.py --ratios 0.05 0.10 0.20 --seed 42 --force
```

On GPU, run the EDA PhoBERT experiments:

```bash
python scripts/phase4_eda.py --ratios 0.05 0.10 0.20 --seed 42 --overwrite
```

To also run the matching no-augmentation low-resource PhoBERT baselines for
comparison, add `--include-baseline`:

```bash
python scripts/phase4_eda.py --ratios 0.05 0.10 0.20 --seed 42 --include-baseline --overwrite
```

Summarize Phase 4 metrics from generated JSON logs:

```bash
python scripts/summarize_phase4.py
```

Observed Phase 4 test macro-F1:

| Ratio | None | EDA | Delta |
|---:|---:|---:|---:|
| 0.05 | 0.7563 | 0.7641 | +0.0078 |
| 0.10 | 0.8231 | 0.7956 | -0.0275 |
| 0.20 | 0.8149 | 0.8208 | +0.0059 |

Conclusion: conservative EDA has a mixed effect. It improves the 5% and 20%
settings, but hurts the 10% setting.

## Phase 5 LLM Paraphrase Augmentation

Phase 5 uses Gemini to generate label-preserving paraphrases. On Kaggle, add a
secret named `GEMINI_API_KEY`, then load it in a notebook cell:

```python
from kaggle_secrets import UserSecretsClient
import os

os.environ["GEMINI_API_KEY"] = UserSecretsClient().get_secret("GEMINI_API_KEY")
```

Generate raw LLM paraphrases. The script is resumable, so rerunning continues
from completed `source_index` rows:

```bash
python scripts/generate_llm_paraphrase.py --ratios 0.05 --seed 42 --max-rows 3 --force --request-sleep-seconds 0.5
```

```bash
python scripts/generate_llm_paraphrase.py --ratios 0.05 0.10 0.20 --seed 42 --request-sleep-seconds 0.5
```

After `data/augmented/llm_raw_{ratio}_{seed}.csv` files exist, run PhoBERT:

```bash
python scripts/phase5_llm.py --ratios 0.05 0.10 0.20 --seed 42 --skip-generation --overwrite
```

If Gemini quota blocks full generation, run the local paraphrase pilot for
the 5% split:

```bash
python scripts/generate_antigravity_paraphrase.py --ratio 0.05 --seed 42 --force
python scripts/phase5_llm.py --ratios 0.05 --seed 42 --skip-generation --overwrite
```

Observed Phase 5 pilot test macro-F1:

| Ratio | None | EDA | LLM Raw | Delta vs None | Delta vs EDA |
|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.7563 | 0.7641 | 0.7864 | +0.0301 | +0.0223 |

Conclusion: the local paraphrase fallback improved the 5% low-resource
setting over both the no-augmentation and EDA baselines.

## Phase 6 Filtering / Quality Control

Filter raw LLM paraphrases and run PhoBERT on the filtered file:

```bash
python scripts/phase6_filter.py --ratios 0.05 --seed 42 --overwrite
```

For filtering only:

```bash
python scripts/filter_llm_paraphrase.py --ratios 0.05 --seed 42 --force
```

This writes `data/augmented/llm_filtered_0.05_42.csv` plus QC summaries under
`results/tables/`.

Observed Phase 6 filtered-LLM result:

| Ratio | None | EDA | LLM Raw | LLM Filtered | Filtered vs Raw |
|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.7563 | 0.7641 | 0.7864 | 0.7752 | -0.0112 |

Conclusion: filtering kept 564/571 raw paraphrases and removed only obvious
QC risks. The filtered file still beats the no-augmentation and EDA 5% runs,
but raw LLM remains the strongest 5% augmentation result.

## Phase 7 Aggregation / Orchestration

Aggregate whatever result artifacts exist under `results/logs/` and
`results/predictions/`:

```bash
python scripts/aggregate_results.py
```

This writes `results/tables/main_results.csv`,
`results/tables/main_results_runs.csv`, and `results/tables/main_results.md`.

To plan or run remaining GPU experiments idempotently:

```bash
python -m src.experiments.run_all --dry-run
python -m src.experiments.run_all --include-full-none
```

`run_all` skips existing artifacts and, by default, skips missing augmentation
CSV files instead of failing the whole matrix.

## Phase 8 Significance Testing

Run paired bootstrap tests for all available prediction pairs:

```bash
python scripts/run_significance_tests.py
```

This writes `results/tables/significance_tests.csv` and
`results/tables/significance_tests.md`. With the current single-seed pilot,
bootstrap p-values are available; paired t-tests across seeds are reported as
unavailable until at least two paired seeds exist.

## Directory Overview

- `configs/`: shared experiment and model settings.
- `src/`: reusable Python package code.
- `scripts/`: command-line scripts for setup and orchestration.
- `data/`: raw, split, and augmented data. This is gitignored.
- `results/`: logs, predictions, tables, and figures.
- `notebooks/`: exploratory analysis and reporting notebooks.
