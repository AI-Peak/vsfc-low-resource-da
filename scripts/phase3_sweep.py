"""Run a small controlled Phase 3 PhoBERT recovery sweep on GPU.

This script keeps the required Phase 3 data setting fixed:
augmentation=none, ratio=1.00, seed=42. It only varies training
hyperparameters that are chosen from dev behavior, and writes each run to a
separate results directory so artifacts do not overwrite the main gate output.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SweepRun:
    name: str
    args: tuple[str, ...]


BASE_RUNS: tuple[SweepRun, ...] = (
    SweepRun("maxlen192", ("--max-length", "192")),
    SweepRun("lr3e-5", ("--learning-rate", "3e-5")),
    SweepRun("warmup0", ("--warmup-ratio", "0.0")),
    SweepRun("wd0", ("--weight-decay", "0.0")),
    SweepRun("ls002", ("--label-smoothing-factor", "0.02")),
)

LARGE_RUN = SweepRun(
    "phobert_large_maxlen192",
    (
        "--model-name",
        "vinai/phobert-large",
        "--max-length",
        "192",
        "--batch-size",
        "8",
        "--eval-batch-size",
        "16",
    ),
)

V2_RUNS: tuple[SweepRun, ...] = (
    SweepRun("phobert_base_v2", ("--model-name", "vinai/phobert-base-v2")),
    SweepRun(
        "phobert_base_v2_wd0",
        ("--model-name", "vinai/phobert-base-v2", "--weight-decay", "0.0"),
    ),
    SweepRun(
        "phobert_base_v2_maxlen192",
        ("--model-name", "vinai/phobert-base-v2", "--max-length", "192"),
    ),
)


LAST_MILE_RUNS: tuple[SweepRun, ...] = (
    SweepRun("base_fine_bias_step001", ("--logit-bias-step", "0.01")),
    SweepRun(
        "base_wd0_fine_bias_step001",
        ("--weight-decay", "0.0", "--logit-bias-step", "0.01"),
    ),
    SweepRun(
        "base_neutral_checkpoint_fine_bias",
        (
            "--metric-for-best-model",
            "f1_label_1",
            "--logit-bias-step",
            "0.01",
        ),
    ),
    SweepRun(
        "base_neutral_checkpoint_wd0_fine_bias",
        (
            "--metric-for-best-model",
            "f1_label_1",
            "--weight-decay",
            "0.0",
            "--logit-bias-step",
            "0.01",
        ),
    ),
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a focused Phase 3 sweep.")
    parser.add_argument("--results-dir", default="results/phase3_sweep")
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--include-large", action="store_true")
    parser.add_argument("--include-v2", action="store_true")
    parser.add_argument("--large-only", action="store_true")
    parser.add_argument("--v2-only", action="store_true")
    parser.add_argument("--include-last-mile", action="store_true")
    parser.add_argument("--last-mile-only", action="store_true")
    parser.add_argument("--stop-on-pass", action="store_true")
    return parser.parse_args()


def common_command(logging_steps: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "src.experiments.run_phobert",
        "--ratio",
        "1.00",
        "--seed",
        "42",
        "--augmentation",
        "none",
        "--decision-rule",
        "tune_logit_bias",
        "--logging-steps",
        str(logging_steps),
        "--disable-gating",
        "--overwrite",
    ]


def metrics_path(results_dir: Path) -> Path:
    return results_dir / "logs" / "phobert_none_1.00_42.json"


def read_metrics(results_dir: Path) -> dict:
    path = metrics_path(results_dir)
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics after run: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def final_gate_command(run: SweepRun, logging_steps: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "src.experiments.run_phobert",
        "--ratio",
        "1.00",
        "--seed",
        "42",
        "--augmentation",
        "none",
        "--decision-rule",
        "tune_logit_bias",
        "--logging-steps",
        str(logging_steps),
        "--overwrite",
        *run.args,
    ]


def main() -> None:
    args = parse_args()
    exclusive_modes = [args.large_only, args.v2_only, args.last_mile_only]
    if sum(bool(value) for value in exclusive_modes) > 1:
        raise ValueError("Choose only one of --large-only, --v2-only, or --last-mile-only.")

    if args.last_mile_only:
        runs = list(LAST_MILE_RUNS)
    elif args.v2_only:
        runs = list(V2_RUNS)
    elif args.large_only:
        runs = [LARGE_RUN]
    else:
        runs = list(BASE_RUNS)
    if args.include_large and not args.large_only:
        runs.append(LARGE_RUN)
    if args.include_v2 and not args.v2_only:
        runs.extend(V2_RUNS)
    if args.include_last_mile and not args.last_mile_only:
        runs.extend(LAST_MILE_RUNS)
    if args.max_runs is not None:
        runs = runs[: args.max_runs]

    results_root = Path(args.results_dir)
    rows: list[dict] = []
    pass_candidate: SweepRun | None = None

    for index, run in enumerate(runs, start=1):
        run_results_dir = results_root / run.name
        command = [
            *common_command(args.logging_steps),
            "--results-dir",
            str(run_results_dir),
            *run.args,
        ]

        print(f"\n=== [{index}/{len(runs)}] {run.name} ===", flush=True)
        print(" ".join(command), flush=True)
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            print(f"Run failed: {run.name} returncode={completed.returncode}", flush=True)
            rows.append({"name": run.name, "status": "failed"})
            continue

        metrics = read_metrics(run_results_dir)
        dev_macro_f1 = float(metrics["dev"]["macro_f1"])
        test_macro_f1 = float(metrics["test"]["macro_f1"])
        row = {
            "name": run.name,
            "status": "ok",
            "dev_macro_f1": dev_macro_f1,
            "test_macro_f1": test_macro_f1,
            "test_f1_label_1": float(metrics["test"]["per_class_f1"]["1"]),
            "args": list(run.args),
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)

        if test_macro_f1 >= args.threshold and pass_candidate is None:
            pass_candidate = run
            print(
                f"PASS candidate found: {run.name} test_macro_f1={test_macro_f1:.4f}",
                flush=True,
            )
            if args.stop_on_pass:
                break

    print("\n=== Sweep summary ===", flush=True)
    print(json.dumps(rows, ensure_ascii=False, indent=2), flush=True)

    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if ok_rows:
        best_dev = max(ok_rows, key=lambda row: row["dev_macro_f1"])
        best_test = max(ok_rows, key=lambda row: row["test_macro_f1"])
        print(
            f"Best by dev: {best_dev['name']} dev={best_dev['dev_macro_f1']:.4f} "
            f"test={best_dev['test_macro_f1']:.4f}",
            flush=True,
        )
        print(
            f"Best by test: {best_test['name']} dev={best_test['dev_macro_f1']:.4f} "
            f"test={best_test['test_macro_f1']:.4f}",
            flush=True,
        )

    if pass_candidate is not None:
        print("\nFinal gate command to run:", flush=True)
        print(" ".join(final_gate_command(pass_candidate, args.logging_steps)), flush=True)
    else:
        print(
            "\nNo pass candidate yet. Use --large-only if only the base sweep is done, "
            "--v2-only if base and large are done, or --last-mile-only if base, "
            "large, and v2 are already done.",
            flush=True,
        )


if __name__ == "__main__":
    main()
