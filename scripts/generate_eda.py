"""Generate EDA augmented CSV files for Phase 4."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.augmentation.base import deduplicate_augmented_rows
from src.augmentation.base import label_counts
from src.augmentation.base import validate_augmented_schema
from src.augmentation.eda import EDAAugmenter
from src.data.subsample import create_subsamples
from src.data.subsample import format_ratio
from src.utils.io import read_yaml

DEFAULT_PHASE4_RATIOS = (0.05, 0.10, 0.20)


def load_train_split(data_dir: str | Path, ratio: float, seed: int) -> pd.DataFrame:
    """Load or create the stratified train split for one ratio."""
    data_path = Path(data_dir)
    ratio_name = format_ratio(ratio)
    split_path = data_path / "splits" / f"train_{ratio_name}_{seed}.csv"
    if split_path.exists():
        return pd.read_csv(split_path)

    raw_train_path = data_path / "raw" / "train.csv"
    if not raw_train_path.exists():
        raise FileNotFoundError(
            f"Missing {raw_train_path}. Run scripts/download_data.py before EDA generation."
        )

    print(f"Creating missing train split: {split_path}", flush=True)
    raw_train = pd.read_csv(raw_train_path)
    create_subsamples(
        raw_train,
        ratios=[ratio],
        seed=seed,
        output_dir=data_path / "splits",
        force=True,
    )
    return pd.read_csv(split_path)


def build_augmenter(config_path: str | Path, seed: int) -> tuple[EDAAugmenter, int]:
    """Build an EDA augmenter from configs/augmentation.yaml."""
    config = read_yaml(config_path)
    eda_config = config.get("eda", {})
    augmenter = EDAAugmenter(
        seed=seed,
        alpha_sr=float(eda_config.get("alpha_sr", 0.1)),
        alpha_ri=float(eda_config.get("alpha_ri", 0.1)),
        alpha_rs=float(eda_config.get("alpha_rs", 0.1)),
        p_rd=float(eda_config.get("p_rd", 0.1)),
    )
    return augmenter, int(eda_config.get("num_aug", 1))


def generate_eda_for_ratio(
    ratio: float,
    seed: int = 42,
    data_dir: str | Path = "data",
    output_dir: str | Path | None = None,
    config_path: str | Path = "configs/augmentation.yaml",
    num_aug: int | None = None,
    force: bool = False,
) -> Path:
    """Generate and save one EDA file."""
    data_path = Path(data_dir)
    output_path = Path(output_dir) if output_dir else data_path / "augmented"
    output_path.mkdir(parents=True, exist_ok=True)

    ratio_name = format_ratio(ratio)
    csv_path = output_path / f"eda_{ratio_name}_{seed}.csv"
    if csv_path.exists() and not force:
        print(f"Skipping existing EDA file: {csv_path}", flush=True)
        return csv_path

    train_df = load_train_split(data_dir=data_path, ratio=ratio, seed=seed)
    augmenter, default_num_aug = build_augmenter(config_path=config_path, seed=seed)
    resolved_num_aug = int(num_aug if num_aug is not None else default_num_aug)
    augmented = augmenter.augment_frame(train_df, num_aug=resolved_num_aug)
    augmented = deduplicate_augmented_rows(augmented)
    validate_augmented_schema(augmented)
    augmented["ratio"] = ratio_name
    augmented["seed"] = int(seed)
    augmented.to_csv(csv_path, index=False)

    print(
        f"Saved {csv_path}: source_rows={len(train_df)} augmented_rows={len(augmented)} "
        f"labels={label_counts(augmented)}",
        flush=True,
    )
    return csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Phase 4 EDA data.")
    parser.add_argument("--ratios", nargs="+", type=float, default=list(DEFAULT_PHASE4_RATIOS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--config", default="configs/augmentation.yaml")
    parser.add_argument("--num-aug", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for ratio in args.ratios:
        generate_eda_for_ratio(
            ratio=ratio,
            seed=args.seed,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            config_path=args.config,
            num_aug=args.num_aug,
            force=args.force,
        )


if __name__ == "__main__":
    main()
