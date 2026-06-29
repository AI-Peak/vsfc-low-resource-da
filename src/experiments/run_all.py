"""Run the PhoBERT experiment matrix idempotently."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.subsample import format_ratio
from src.experiments.run_phobert import VALID_AUGMENTATIONS
from src.experiments.run_phobert import augmented_path
from src.experiments.run_phobert import output_paths

DEFAULT_LOW_RESOURCE_RATIOS = (0.05, 0.10, 0.20)
DEFAULT_SEEDS = (42,)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a PhoBERT experiment matrix.")
    parser.add_argument(
        "--ratios",
        nargs="+",
        type=float,
        default=list(DEFAULT_LOW_RESOURCE_RATIOS),
        help="Low-resource ratios to run.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=VALID_AUGMENTATIONS,
        default=list(VALID_AUGMENTATIONS),
        help="Augmentation methods to run.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-fail", action="store_true")
    parser.add_argument(
        "--include-full-none",
        action="store_true",
        help="Also include the 100 percent no-augmentation ceiling for each seed.",
    )
    parser.add_argument(
        "--no-skip-missing-augmentation",
        action="store_true",
        help="Let run_phobert fail when an augmentation CSV is missing.",
    )
    return parser.parse_args()

def experiment_exists(
    augmentation: str,
    ratio: float,
    seed: int,
    results_dir: str | Path,
) -> tuple[bool, Path, Path]:
    prediction_path, metrics_path, _ = output_paths(
        augmentation=augmentation,
        ratio=ratio,
        seed=seed,
        results_dir=results_dir,
    )
    return prediction_path.exists() and metrics_path.exists(), prediction_path, metrics_path

def has_required_augmentation(
    augmentation: str,
    ratio: float,
    seed: int,
    data_dir: str | Path,
) -> tuple[bool, str]:
    if augmentation == "none":
        return True, ""

    path = augmented_path(
        augmentation=augmentation,
        ratio=ratio,
        seed=seed,
        data_dir=data_dir,
    )
    if path.exists():
        return True, ""
    return False, f"missing augmentation file: {path}"

def build_command(
    augmentation: str,
    ratio: float,
    seed: int,
    data_dir: str | Path,
    results_dir: str | Path,
    logging_steps: int,
    overwrite: bool,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "src.experiments.run_phobert",
        "--ratio",
        format_ratio(ratio),
        "--seed",
        str(seed),
        "--augmentation",
        augmentation,
        "--data-dir",
        str(data_dir),
        "--results-dir",
        str(results_dir),
        "--decision-rule",
        "tune_logit_bias",
        "--logging-steps",
        str(logging_steps),
        "--disable-gating",
    ]
    if overwrite:
        command.append("--overwrite")
    return command

def write_run_summary(rows: list[dict[str, object]], results_dir: str | Path) -> Path:
    output_path = Path(results_dir) / "tables" / "run_all_plan.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "ratio",
        "seed",
        "augmentation",
        "status",
        "returncode",
        "prediction_path",
        "metrics_path",
        "elapsed_seconds",
        "reason",
        "command",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path

def main() -> None:
    args = parse_args()
    skip_missing_augmentation = not args.no_skip_missing_augmentation

    jobs: list[tuple[float, str, int]] = [
        (ratio, method, seed)
        for seed in args.seeds
        for ratio in args.ratios
        for method in args.methods
    ]
    if args.include_full_none:
        jobs.extend((1.00, "none", seed) for seed in args.seeds)

    rows: list[dict[str, object]] = []
    for index, (ratio, augmentation, seed) in enumerate(jobs, start=1):
        ratio_name = format_ratio(ratio)
        exists, prediction_path, metrics_path = experiment_exists(
            augmentation=augmentation,
            ratio=ratio,
            seed=seed,
            results_dir=args.results_dir,
        )
        command = build_command(
            augmentation=augmentation,
            ratio=ratio,
            seed=seed,
            data_dir=args.data_dir,
            results_dir=args.results_dir,
            logging_steps=args.logging_steps,
            overwrite=args.overwrite,
        )

        base_row = {
            "ratio": ratio_name,
            "seed": seed,
            "augmentation": augmentation,
            "returncode": "",
            "prediction_path": str(prediction_path),
            "metrics_path": str(metrics_path),
            "elapsed_seconds": "",
            "reason": "",
            "command": " ".join(command),
        }

        if exists and not args.overwrite:
            print(f"[{index}/{len(jobs)}] skip existing {augmentation} ratio={ratio_name} seed={seed}")
            rows.append({**base_row, "status": "skipped_existing"})
            continue

        has_aug, reason = has_required_augmentation(
            augmentation=augmentation,
            ratio=ratio,
            seed=seed,
            data_dir=args.data_dir,
        )
        if not has_aug and skip_missing_augmentation:
            print(f"[{index}/{len(jobs)}] skip {augmentation} ratio={ratio_name} seed={seed}: {reason}")
            rows.append({**base_row, "status": "skipped_missing_augmentation", "reason": reason})
            continue

        if args.dry_run:
            print(f"[{index}/{len(jobs)}] dry-run {augmentation} ratio={ratio_name} seed={seed}")
            rows.append({**base_row, "status": "dry_run"})
            continue

        print(f"\n[{index}/{len(jobs)}] run {augmentation} ratio={ratio_name} seed={seed}", flush=True)
        print(" ".join(command), flush=True)
        start = time.perf_counter()
        completed = subprocess.run(command, check=False)
        elapsed = time.perf_counter() - start
        status = "ok" if completed.returncode == 0 else "failed"
        rows.append(
            {
                **base_row,
                "status": status,
                "returncode": completed.returncode,
                "elapsed_seconds": f"{elapsed:.1f}",
            }
        )
        if completed.returncode != 0 and args.stop_on_fail:
            break

    summary_path = write_run_summary(rows, args.results_dir)
    ok_count = sum(1 for row in rows if row["status"] in {"ok", "skipped_existing"})
    print(f"\nSaved run summary: {summary_path}", flush=True)
    print(f"Usable/completed jobs: {ok_count}/{len(rows)}", flush=True)

if __name__ == "__main__":
    main()
