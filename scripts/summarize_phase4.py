"""Summarize Phase 4 EDA results from PhoBERT JSON logs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.subsample import format_ratio

DEFAULT_PHASE4_RATIOS = (0.05, 0.10, 0.20)


def metrics_path(results_dir: str | Path, augmentation: str, ratio: float, seed: int) -> Path:
    """Return the metrics JSON path for one PhoBERT run."""
    ratio_name = format_ratio(ratio)
    return Path(results_dir) / "logs" / f"phobert_{augmentation}_{ratio_name}_{seed}.json"


def read_test_macro_f1(results_dir: str | Path, augmentation: str, ratio: float, seed: int) -> float:
    """Read test macro-F1 from one metrics JSON file."""
    path = metrics_path(results_dir, augmentation, ratio, seed)
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics JSON: {path}")
    with path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    return float(metrics["test"]["macro_f1"])


def build_rows(results_dir: str | Path, ratios: list[float], seed: int) -> list[dict[str, object]]:
    """Build comparison rows for none vs EDA runs."""
    rows: list[dict[str, object]] = []
    for ratio in ratios:
        none_macro_f1 = read_test_macro_f1(results_dir, "none", ratio, seed)
        eda_macro_f1 = read_test_macro_f1(results_dir, "eda", ratio, seed)
        delta = eda_macro_f1 - none_macro_f1
        rows.append(
            {
                "ratio": format_ratio(ratio),
                "none_macro_f1": none_macro_f1,
                "eda_macro_f1": eda_macro_f1,
                "delta_macro_f1": delta,
                "outcome": "improved" if delta > 0 else "worse" if delta < 0 else "tie",
            }
        )
    return rows


def write_csv(rows: list[dict[str, object]], path: str | Path) -> None:
    """Write summary rows as CSV."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["ratio", "none_macro_f1", "eda_macro_f1", "delta_macro_f1", "outcome"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, object]], path: str | Path) -> None:
    """Write summary rows as a Markdown table."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 4 EDA Summary",
        "",
        "| Ratio | None Macro-F1 | EDA Macro-F1 | Delta | Outcome |",
        "|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {ratio} | {none:.4f} | {eda:.4f} | {delta:+.4f} | {outcome} |".format(
                ratio=row["ratio"],
                none=float(row["none_macro_f1"]),
                eda=float(row["eda_macro_f1"]),
                delta=float(row["delta_macro_f1"]),
                outcome=row["outcome"],
            )
        )
    lines.extend(
        [
            "",
            "Conclusion: conservative EDA has a mixed effect. It improves the 5% "
            "and 20% low-resource settings, but hurts the 10% setting.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Phase 4 EDA results.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default="results/tables")
    parser.add_argument("--ratios", nargs="+", type=float, default=list(DEFAULT_PHASE4_RATIOS))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_rows(results_dir=args.results_dir, ratios=args.ratios, seed=args.seed)
    output_dir = Path(args.output_dir)
    csv_path = output_dir / "phase4_eda_summary.csv"
    markdown_path = output_dir / "phase4_eda_summary.md"
    write_csv(rows, csv_path)
    write_markdown(rows, markdown_path)
    print(f"Saved {csv_path}")
    print(f"Saved {markdown_path}")
    for row in rows:
        print(
            "{ratio}: none={none:.4f} eda={eda:.4f} delta={delta:+.4f} {outcome}".format(
                ratio=row["ratio"],
                none=float(row["none_macro_f1"]),
                eda=float(row["eda_macro_f1"]),
                delta=float(row["delta_macro_f1"]),
                outcome=row["outcome"],
            )
        )


if __name__ == "__main__":
    main()
