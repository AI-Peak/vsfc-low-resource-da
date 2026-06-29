"""Run Phase 6 LLM paraphrase filtering experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.filter_llm_paraphrase import build_filter_config
from scripts.filter_llm_paraphrase import filter_for_ratio
from scripts.filter_llm_paraphrase import filtered_output_path

DEFAULT_PHASE6_RATIOS = (0.05,)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter LLM data and run PhoBERT.")
    parser.add_argument("--ratios", nargs="+", type=float, default=list(DEFAULT_PHASE6_RATIOS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--config", default="configs/augmentation.yaml")
    parser.add_argument("--min-length-ratio", type=float, default=None)
    parser.add_argument("--max-length-ratio", type=float, default=None)
    parser.add_argument("--min-token-overlap", type=float, default=None)
    parser.add_argument("--min-augmented-tokens", type=int, default=None)
    parser.add_argument("--skip-filtering", action="store_true")
    parser.add_argument("--filter-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--stop-on-fail", action="store_true")
    return parser.parse_args()

def phobert_command(
    ratio: float,
    seed: int,
    data_dir: str,
    results_dir: str,
    logging_steps: int,
    overwrite: bool,
) -> list[str]:
    """Build a PhoBERT command for one filtered LLM run."""
    command = [
        sys.executable,
        "-m",
        "src.experiments.run_phobert",
        "--ratio",
        f"{ratio:.2f}",
        "--seed",
        str(seed),
        "--augmentation",
        "llm_paraphrase_filtered",
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
    """Run one visible subprocess command."""
    print("\n" + " ".join(command), flush=True)
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0 and stop_on_fail:
        raise SystemExit(completed.returncode)
    return int(completed.returncode)

def ensure_filtered_files_exist(ratios: list[float], seed: int, data_dir: str) -> None:
    """Fail early if filtered LLM files are missing before training."""
    missing = []
    for ratio in ratios:
        path = filtered_output_path(data_dir=data_dir, output_dir=None, ratio=ratio, seed=seed)
        if not path.exists():
            missing.append(path)
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "Missing filtered LLM augmentation file(s). Run filtering first:\n"
            f"{formatted}"
        )

def main() -> None:
    args = parse_args()

    if not args.skip_filtering:
        filter_config = build_filter_config(args.config, args)
        for ratio in args.ratios:
            filter_for_ratio(
                ratio=ratio,
                seed=args.seed,
                data_dir=args.data_dir,
                output_dir=None,
                results_dir=args.results_dir,
                config=filter_config,
                force=args.overwrite,
            )

    if args.filter_only:
        return

    ensure_filtered_files_exist(ratios=args.ratios, seed=args.seed, data_dir=args.data_dir)

    rows: list[dict[str, object]] = []
    for ratio in args.ratios:
        command = phobert_command(
            ratio=ratio,
            seed=args.seed,
            data_dir=args.data_dir,
            results_dir=args.results_dir,
            logging_steps=args.logging_steps,
            overwrite=args.overwrite,
        )
        returncode = run_command(command, stop_on_fail=args.stop_on_fail)
        rows.append(
            {
                "ratio": f"{ratio:.2f}",
                "augmentation": "llm_paraphrase_filtered",
                "returncode": returncode,
            }
        )

    print("\n=== Phase 6 run summary ===", flush=True)
    for row in rows:
        print(row, flush=True)

if __name__ == "__main__":
    main()
