"""Dataset loading utilities for UIT-VSFC."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd
from datasets import DatasetDict, load_dataset


DEFAULT_DATASET_NAME = "uitnlp/vietnamese_students_feedback"
REQUIRED_COLUMNS = ("sentence", "sentiment")
SPLIT_ALIASES = {
    "train": ("train",),
    "dev": ("validation", "dev", "valid"),
    "test": ("test",),
}


def _resolve_split(dataset: DatasetDict, canonical_name: str) -> str:
    for split_name in SPLIT_ALIASES[canonical_name]:
        if split_name in dataset:
            return split_name
    available = ", ".join(dataset.keys())
    expected = ", ".join(SPLIT_ALIASES[canonical_name])
    raise KeyError(
        f"Could not find a {canonical_name!r} split. "
        f"Expected one of: {expected}. Available splits: {available}"
    )


def _split_to_frame(dataset: DatasetDict, split_name: str) -> pd.DataFrame:
    frame = pd.DataFrame(dataset[split_name])
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Split {split_name!r} is missing columns: {missing}")

    frame = frame.loc[:, list(REQUIRED_COLUMNS)].copy()
    frame["sentence"] = frame["sentence"].astype(str)
    frame["sentiment"] = frame["sentiment"].astype(int)
    return frame


def label_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    """Return label counts and percentages sorted by sentiment id."""
    counts = frame["sentiment"].value_counts().sort_index()
    distribution = counts.rename("count").to_frame()
    distribution["pct"] = (distribution["count"] / len(frame) * 100).round(2)
    return distribution


def print_split_summary(name: str, frame: pd.DataFrame) -> None:
    """Print size and label distribution for one split."""
    print(f"\n{name}: {len(frame):,} rows")
    print(label_distribution(frame).to_string())


def load_uit_vsfc(
    dataset_name: str = DEFAULT_DATASET_NAME,
    cache_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load UIT-VSFC from HuggingFace and return train/dev/test DataFrames."""
    dataset = load_dataset(
        dataset_name,
        cache_dir=str(cache_dir) if cache_dir else None,
        trust_remote_code=True,
    )

    train_split = _resolve_split(dataset, "train")
    dev_split = _resolve_split(dataset, "dev")
    test_split = _resolve_split(dataset, "test")

    train_df = _split_to_frame(dataset, train_split)
    dev_df = _split_to_frame(dataset, dev_split)
    test_df = _split_to_frame(dataset, test_split)

    for name, frame in (
        ("train", train_df),
        ("dev", dev_df),
        ("test", test_df),
    ):
        print_split_summary(name, frame)

    return train_df, dev_df, test_df


def save_raw_splits(
    splits: Iterable[tuple[str, pd.DataFrame]],
    output_dir: str | Path = "data/raw",
    force: bool = False,
) -> None:
    """Save split DataFrames as CSV files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for name, frame in splits:
        csv_path = output_path / f"{name}.csv"
        if csv_path.exists() and not force:
            print(f"Skipping existing file: {csv_path}")
            continue
        frame.to_csv(csv_path, index=False)
        print(f"Saved {name}: {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load UIT-VSFC and print summaries.")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--save-raw", action="store_true")
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_df, dev_df, test_df = load_uit_vsfc(
        dataset_name=args.dataset_name,
        cache_dir=args.cache_dir,
    )

    if args.save_raw:
        save_raw_splits(
            (("train", train_df), ("dev", dev_df), ("test", test_df)),
            output_dir=args.output_dir,
            force=args.force,
        )


if __name__ == "__main__":
    main()
