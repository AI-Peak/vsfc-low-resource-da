"""Filter raw LLM paraphrase augmentation files for Phase 6."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_eda import DEFAULT_PHASE4_RATIOS
from scripts.generate_llm_paraphrase import output_path
from src.augmentation.base import label_counts
from src.augmentation.filter import AugmentationFilterConfig
from src.augmentation.filter import diagnostics_summary
from src.augmentation.filter import filter_augmented_frame
from src.data.subsample import format_ratio
from src.utils.io import read_yaml

def filtered_output_path(
    data_dir: str | Path,
    output_dir: str | Path | None,
    ratio: float,
    seed: int,
) -> Path:
    """Return the filtered CSV path consumed by run_phobert."""
    base_dir = Path(output_dir) if output_dir else Path(data_dir) / "augmented"
    return base_dir / f"llm_filtered_{format_ratio(ratio)}_{seed}.csv"

def build_filter_config(config_path: str | Path, args: argparse.Namespace) -> AugmentationFilterConfig:
    """Build filter thresholds from YAML plus CLI overrides."""
    config = read_yaml(config_path)
    filter_config = config.get("filter", {})
    return AugmentationFilterConfig(
        min_length_ratio=(
            args.min_length_ratio
            if args.min_length_ratio is not None
            else float(filter_config.get("min_length_ratio", 0.45))
        ),
        max_length_ratio=(
            args.max_length_ratio
            if args.max_length_ratio is not None
            else float(filter_config.get("max_length_ratio", 2.20))
        ),
        min_token_overlap=(
            args.min_token_overlap
            if args.min_token_overlap is not None
            else float(filter_config.get("min_token_overlap", 0.20))
        ),
        min_augmented_tokens=(
            args.min_augmented_tokens
            if args.min_augmented_tokens is not None
            else int(filter_config.get("min_augmented_tokens", 2))
        ),
    )

def write_summary_markdown(rows: list[dict[str, object]], path: str | Path) -> None:
    """Write Phase 6 filtering summary as Markdown."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 6 LLM Filtering Summary",
        "",
        "| Ratio | Raw Rows | Kept Rows | Dropped Rows | Keep Rate | Label Counts |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {ratio} | {raw_rows} | {kept_rows} | {dropped_rows} | {keep_rate:.4f} | {label_counts} |".format(
                ratio=row["ratio"],
                raw_rows=int(row["raw_rows"]),
                kept_rows=int(row["kept_rows"]),
                dropped_rows=int(row["dropped_rows"]),
                keep_rate=float(row["keep_rate"]),
                label_counts=row["label_counts"],
            )
        )
    lines.extend(
        [
            "",
            "Conclusion: heuristic filtering removes only obvious low-quality or "
            "label-risk paraphrases before the filtered PhoBERT run.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")

def filter_for_ratio(
    ratio: float,
    seed: int,
    data_dir: str | Path,
    output_dir: str | Path | None,
    results_dir: str | Path,
    config: AugmentationFilterConfig,
    force: bool,
) -> dict[str, object]:
    """Filter one raw LLM augmentation file and return a summary row."""
    raw_path = output_path(data_dir=data_dir, output_dir=None, ratio=ratio, seed=seed)
    filtered_path = filtered_output_path(
        data_dir=data_dir,
        output_dir=output_dir,
        ratio=ratio,
        seed=seed,
    )
    if filtered_path.exists() and not force:
        print(f"Skipping existing filtered file: {filtered_path}", flush=True)
        raw = pd.read_csv(raw_path)
        filtered = pd.read_csv(filtered_path)
        dropped_rows = max(len(raw) - len(filtered), 0)
        return {
            "ratio": format_ratio(ratio),
            "raw_rows": len(raw),
            "kept_rows": len(filtered),
            "dropped_rows": dropped_rows,
            "keep_rate": len(filtered) / len(raw) if len(raw) else 0.0,
            "label_counts": label_counts(filtered),
            "filtered_path": str(filtered_path),
        }

    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw LLM file: {raw_path}")

    raw = pd.read_csv(raw_path)
    filtered, diagnostics = filter_augmented_frame(raw, config=config)
    filtered_path.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(filtered_path, index=False)

    tables_dir = Path(results_dir) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = tables_dir / f"phase6_filter_qc_{format_ratio(ratio)}_{seed}.csv"
    diagnostics.to_csv(diagnostics_path, index=False)

    summary = diagnostics_summary(diagnostics)
    keep_rate = len(filtered) / len(raw) if len(raw) else 0.0
    print(
        f"Saved {filtered_path}: raw_rows={len(raw)} kept_rows={len(filtered)} "
        f"dropped_rows={len(raw) - len(filtered)} keep_rate={keep_rate:.4f} "
        f"labels={label_counts(filtered)} summary={summary}",
        flush=True,
    )
    print(f"Saved diagnostics: {diagnostics_path}", flush=True)

    return {
        "ratio": format_ratio(ratio),
        "raw_rows": len(raw),
        "kept_rows": len(filtered),
        "dropped_rows": len(raw) - len(filtered),
        "keep_rate": keep_rate,
        "label_counts": label_counts(filtered),
        "filtered_path": str(filtered_path),
    }

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter Phase 6 LLM paraphrases.")
    parser.add_argument("--ratios", nargs="+", type=float, default=list(DEFAULT_PHASE4_RATIOS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--config", default="configs/augmentation.yaml")
    parser.add_argument("--min-length-ratio", type=float, default=None)
    parser.add_argument("--max-length-ratio", type=float, default=None)
    parser.add_argument("--min-token-overlap", type=float, default=None)
    parser.add_argument("--min-augmented-tokens", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    filter_config = build_filter_config(args.config, args)
    rows = [
        filter_for_ratio(
            ratio=ratio,
            seed=args.seed,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            results_dir=args.results_dir,
            config=filter_config,
            force=args.force,
        )
        for ratio in args.ratios
    ]
    tables_dir = Path(args.results_dir) / "tables"
    summary_csv = tables_dir / "phase6_filter_summary.csv"
    summary_md = tables_dir / "phase6_filter_summary.md"
    pd.DataFrame(rows).to_csv(summary_csv, index=False)
    write_summary_markdown(rows, summary_md)
    print(f"Saved {summary_csv}", flush=True)
    print(f"Saved {summary_md}", flush=True)

if __name__ == "__main__":
    main()
