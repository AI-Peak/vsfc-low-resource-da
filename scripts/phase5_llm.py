"""Run Phase 5 LLM paraphrase augmentation experiments."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_eda import DEFAULT_PHASE4_RATIOS
from scripts.generate_llm_paraphrase import build_paraphraser
from scripts.generate_llm_paraphrase import generate_for_ratio
from scripts.generate_llm_paraphrase import output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate LLM data and run PhoBERT.")
    parser.add_argument("--ratios", nargs="+", type=float, default=list(DEFAULT_PHASE4_RATIOS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--config", default="configs/augmentation.yaml")
    parser.add_argument("--api-key-env", default="GEMINI_API_KEY")
    parser.add_argument("--model", default=None)
    parser.add_argument("--num-aug", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--request-sleep-seconds", type=float, default=0.0)
    parser.add_argument("--retry-sleep-seconds", type=float, default=3.0)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--force-generation", action="store_true")
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--overwrite", action="store_true")
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
    """Build a PhoBERT command for one LLM-raw run."""
    command = [
        sys.executable,
        "-m",
        "src.experiments.run_phobert",
        "--ratio",
        f"{ratio:.2f}",
        "--seed",
        str(seed),
        "--augmentation",
        "llm_paraphrase_raw",
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


def ensure_llm_files_exist(ratios: list[float], seed: int, data_dir: str) -> None:
    """Fail early if raw LLM files are missing before training."""
    missing = []
    for ratio in ratios:
        path = output_path(data_dir=data_dir, output_dir=None, ratio=ratio, seed=seed)
        if not path.exists():
            missing.append(path)

    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "Missing raw LLM augmentation file(s). Run generation first:\n"
            f"{formatted}"
        )

def main() -> None:
    args = parse_args()

    if not args.skip_generation:
        paraphraser = build_paraphraser(args.config, args)
        num_aug = int(args.num_aug or paraphraser.config.num_aug)
        for ratio in args.ratios:
            generate_for_ratio(
                ratio=ratio,
                seed=args.seed,
                data_dir=args.data_dir,
                output_dir=None,
                paraphraser=paraphraser,
                num_aug=num_aug,
                force=args.force_generation,
                max_rows=args.max_rows,
            )

    if args.generate_only:
        return

    ensure_llm_files_exist(ratios=args.ratios, seed=args.seed, data_dir=args.data_dir)

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
                "augmentation": "llm_paraphrase_raw",
                "returncode": returncode,
            }
        )

    print("\n=== Phase 5 run summary ===", flush=True)
    for row in rows:
        print(row, flush=True)


if __name__ == "__main__":
    main()
