"""Download UIT-VSFC and cache raw CSV splits."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load import load_uit_vsfc, save_raw_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download UIT-VSFC raw splits.")
    parser.add_argument("--dataset-name", default="uitnlp/vietnamese_students_feedback")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_df, dev_df, test_df = load_uit_vsfc(
        dataset_name=args.dataset_name,
        cache_dir=args.cache_dir,
    )
    save_raw_splits(
        (("train", train_df), ("dev", dev_df), ("test", test_df)),
        output_dir=args.output_dir,
        force=args.force,
    )


if __name__ == "__main__":
    main()
