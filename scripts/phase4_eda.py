"""Run Phase 4 EDA augmentation experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_eda import DEFAULT_PHASE4_RATIOS
from scripts.generate_eda import generate_eda_for_ratio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate EDA data and run PhoBERT.")
    parser.add_argument("--ratios", nargs="+", type=float, default=list(DEFAULT_PHASE4_RATIOS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--num-aug", type=int, default=None)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--include-baseline", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--stop-on-fail", action="store_true")
    return parser.parse_args()


def phobert_command(
    ratio: float,
    seed: int,
    augmentation: str,
    data_dir: str,
    results_dir: str,
    logging_steps: int,
    overwrite: bool,
) -> list[str]:
    """Build a PhoBERT command for one Phase 4 run."""
    command = [
        sys.executable,
        "-m",
        "src.experiments.run_phobert",
        "--ratio",
        f"{ratio:.2f}",
        "--seed",
        str(seed),
        "--augmentation",
        augmentation,
        "--data-dir",
        data_dir,
        "--results-dir",
        results_dir,
        "--decision-rule",
        "tune_logit_bias",
        "--logging-steps",
        str(logging_steps),
    ]
    if overwrite:
        command.append("--overwrite")
    return command


def run_command(command: list[str], stop_on_fail: bool) -> int:
    """Run one subprocess command with visible logging."""
    print("\n" + " ".join(command), flush=True)
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0 and stop_on_fail:
        raise SystemExit(completed.returncode)
    return int(completed.returncode)


def main() -> None:
    args = parse_args()

    if not args.skip_generation:
        for ratio in args.ratios:
            generate_eda_for_ratio(
                ratio=ratio,
                seed=args.seed,
                data_dir=args.data_dir,
                num_aug=args.num_aug,
                force=args.overwrite,
            )

    if args.generate_only:
        return

    rows: list[dict[str, object]] = []
    for ratio in args.ratios:
        augmentations = ["eda"]
        if args.include_baseline:
            augmentations.insert(0, "none")

        for augmentation in augmentations:
            command = phobert_command(
                ratio=ratio,
                seed=args.seed,
                augmentation=augmentation,
                data_dir=args.data_dir,
                results_dir=args.results_dir,
                logging_steps=args.logging_steps,
                overwrite=args.overwrite,
            )
            returncode = run_command(command, stop_on_fail=args.stop_on_fail)
            rows.append(
                {
                    "ratio": f"{ratio:.2f}",
                    "augmentation": augmentation,
                    "returncode": returncode,
                }
            )

    print("\n=== Phase 4 run summary ===", flush=True)
    for row in rows:
        print(row, flush=True)


if __name__ == "__main__":
    main()
