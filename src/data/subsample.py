"""Create stratified low-resource training subsets."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

from src.data.load import label_distribution, load_uit_vsfc


DEFAULT_RATIOS = (0.05, 0.10, 0.20, 1.00)


def format_ratio(ratio: float) -> str:
    """Format ratios for stable filenames."""
    return f"{ratio:.2f}"


def stratified_sample(
    train_df: pd.DataFrame,
    ratio: float,
    seed: int = 42,
    label_col: str = "sentiment",
) -> pd.DataFrame:
    """Sample each label group independently to preserve label distribution."""
    if ratio <= 0 or ratio > 1:
        raise ValueError(f"ratio must be in (0, 1], got {ratio}")

    if ratio == 1:
        return train_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    sampled_parts: list[pd.DataFrame] = []
    for label, group in train_df.groupby(label_col, sort=True):
        n_rows = max(1, int(round(len(group) * ratio)))
        sampled = group.sample(n=n_rows, random_state=seed + int(label))
        sampled_parts.append(sampled)

    subset = pd.concat(sampled_parts, axis=0)
    subset = subset.sample(frac=1, random_state=seed).reset_index(drop=True)
    return subset


def create_subsamples(
    train_df: pd.DataFrame,
    ratios: Sequence[float] = DEFAULT_RATIOS,
    seed: int = 42,
    output_dir: str | Path = "data/splits",
    label_col: str = "sentiment",
    force: bool = False,
) -> dict[float, pd.DataFrame]:
    """Create and save stratified subsets for all requested ratios."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    subsets: dict[float, pd.DataFrame] = {}
    original_dist = label_distribution(train_df)
    print("\nOriginal train distribution")
    print(original_dist.to_string())

    for ratio in ratios:
        subset = stratified_sample(
            train_df,
            ratio=ratio,
            seed=seed,
            label_col=label_col,
        )
        ratio_name = format_ratio(ratio)
        csv_path = output_path / f"train_{ratio_name}_{seed}.csv"

        if csv_path.exists() and not force:
            print(f"\nSkipping existing subset: {csv_path}")
            subset = pd.read_csv(csv_path)
        else:
            subset.to_csv(csv_path, index=False)
            print(f"\nSaved subset: {csv_path}")

        print(f"ratio={ratio_name}, rows={len(subset):,}")
        print(label_distribution(subset).to_string())
        subsets[ratio] = subset

    return subsets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create stratified UIT-VSFC subsets.")
    parser.add_argument("--train-csv", default=None)
    parser.add_argument("--ratios", nargs="+", type=float, default=list(DEFAULT_RATIOS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="data/splits")
    parser.add_argument("--dataset-name", default="uitnlp/vietnamese_students_feedback")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.train_csv:
        train_df = pd.read_csv(args.train_csv)
    else:
        train_df, _, _ = load_uit_vsfc(
            dataset_name=args.dataset_name,
            cache_dir=args.cache_dir,
        )

    create_subsamples(
        train_df,
        ratios=args.ratios,
        seed=args.seed,
        output_dir=args.output_dir,
        force=args.force,
    )


if __name__ == "__main__":
    main()
