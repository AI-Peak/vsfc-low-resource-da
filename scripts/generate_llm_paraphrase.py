"""Generate raw LLM paraphrase augmentation files for Phase 5."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_eda import DEFAULT_PHASE4_RATIOS
from scripts.generate_eda import load_train_split
from src.augmentation.base import TextAugmenter
from src.augmentation.base import deduplicate_augmented_rows
from src.augmentation.base import label_counts
from src.augmentation.llm_paraphrase import GeminiParaphraser
from src.data.subsample import format_ratio
from src.utils.io import read_yaml

FIELDNAMES = [
    "source_index",
    "original_sentence",
    "augmented_sentence",
    "label",
    "method",
    "ratio",
    "seed",
    "model",
]


def output_path(data_dir: str | Path, output_dir: str | Path | None, ratio: float, seed: int) -> Path:
    """Return the output CSV path consumed by run_phobert."""
    base_dir = Path(output_dir) if output_dir else Path(data_dir) / "augmented"
    return base_dir / f"llm_raw_{format_ratio(ratio)}_{seed}.csv"


def read_existing_counts(path: Path) -> dict[int, int]:
    """Read completed source-index counts from an existing output CSV."""
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    if frame.empty or "source_index" not in frame.columns:
        return {}
    counts = frame["source_index"].astype(int).value_counts()
    return {int(index): int(count) for index, count in counts.items()}


def append_rows(path: Path, rows: Iterable[dict[str, object]]) -> None:
    """Append generated rows to the output CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def finalize_file(path: Path) -> None:
    """Deduplicate and normalize a generated LLM file."""
    if not path.exists():
        return
    frame = pd.read_csv(path)
    if frame.empty:
        return
    frame = deduplicate_augmented_rows(frame)
    frame.to_csv(path, index=False)
    print(f"Finalized {path}: rows={len(frame)} labels={label_counts(frame)}", flush=True)


def build_paraphraser(config_path: str | Path, args: argparse.Namespace) -> GeminiParaphraser:
    """Build a Gemini paraphraser from config plus CLI overrides."""
    config = read_yaml(config_path)
    llm_config = config.get("llm", {})
    return GeminiParaphraser(
        model=args.model or llm_config.get("model", "gemini-2.5-flash-lite"),
        num_aug=args.num_aug or int(llm_config.get("num_aug", 1)),
        temperature=(
            args.temperature
            if args.temperature is not None
            else float(llm_config.get("temperature", 0.7))
        ),
        max_retries=int(llm_config.get("max_retries", 3)),
        retry_sleep_seconds=args.retry_sleep_seconds,
        request_sleep_seconds=args.request_sleep_seconds,
        api_key_env=args.api_key_env,
        seed=args.seed,
    )


def generate_for_ratio(
    ratio: float,
    seed: int,
    data_dir: str | Path,
    output_dir: str | Path | None,
    paraphraser: GeminiParaphraser,
    num_aug: int,
    force: bool,
    max_rows: int | None = None,
) -> Path:
    """Generate one LLM raw augmentation file."""
    path = output_path(data_dir=data_dir, output_dir=output_dir, ratio=ratio, seed=seed)
    if force and path.exists():
        path.unlink()

    train_df = load_train_split(data_dir=data_dir, ratio=ratio, seed=seed).reset_index(drop=True)
    if max_rows is not None:
        train_df = train_df.head(max_rows).reset_index(drop=True)

    existing_counts = read_existing_counts(path)
    print(
        f"Generating {path}: source_rows={len(train_df)} existing_sources={len(existing_counts)}",
        flush=True,
    )

    for source_index, row in train_df.iterrows():
        completed = existing_counts.get(int(source_index), 0)
        if completed >= num_aug:
            continue

        sentence = TextAugmenter.normalize_text(row["sentence"])
        label = int(row["sentiment"])
        needed = num_aug - completed
        try:
            paraphrases = paraphraser.paraphrase(sentence, label=label, num_aug=needed)
        except Exception as exc:
            print(f"Failed source_index={source_index} label={label}: {exc}", flush=True)
            continue

        rows = [
            {
                "source_index": int(source_index),
                "original_sentence": sentence,
                "augmented_sentence": paraphrase,
                "label": label,
                "method": paraphraser.method_name,
                "ratio": format_ratio(ratio),
                "seed": int(seed),
                "model": paraphraser.config.model,
            }
            for paraphrase in paraphrases
        ]
        append_rows(path, rows)
        existing_counts[int(source_index)] = completed + len(rows)
        print(
            f"{format_ratio(ratio)} source_index={source_index + 1}/{len(train_df)} "
            f"generated={len(rows)}",
            flush=True,
        )

    finalize_file(path)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Phase 5 LLM paraphrases.")
    parser.add_argument("--ratios", nargs="+", type=float, default=list(DEFAULT_PHASE4_RATIOS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--config", default="configs/augmentation.yaml")
    parser.add_argument("--api-key-env", default="GEMINI_API_KEY")
    parser.add_argument("--model", default=None)
    parser.add_argument("--num-aug", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--request-sleep-seconds", type=float, default=0.0)
    parser.add_argument("--retry-sleep-seconds", type=float, default=3.0)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paraphraser = build_paraphraser(args.config, args)
    num_aug = int(args.num_aug or paraphraser.config.num_aug)
    for ratio in args.ratios:
        generate_for_ratio(
            ratio=ratio,
            seed=args.seed,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            paraphraser=paraphraser,
            num_aug=num_aug,
            force=args.force,
            max_rows=args.max_rows,
        )


if __name__ == "__main__":
    main()
