"""Aggregate experiment metrics into paper-ready result tables."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.subsample import format_ratio
from src.evaluation.metrics import compute_metrics
from src.utils.io import read_json

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate PhoBERT result artifacts.")
    parser.add_argument("--results-dir", default="results")
    return parser.parse_args()

def identity_key(model: str, augmentation: str, ratio: str, seed: int) -> tuple[str, str, str, int]:
    return model, augmentation, ratio, int(seed)

def row_from_metrics(path: Path) -> dict[str, object]:
    metrics = read_json(path)
    model = str(metrics.get("model", "phobert"))
    augmentation = str(metrics.get("augmentation", model))
    ratio = format_ratio(float(metrics["ratio"]))
    seed = int(metrics["seed"])
    test = metrics["test"]
    per_class = test.get("per_class_f1", {})
    return {
        "model": model,
        "augmentation": augmentation,
        "ratio": ratio,
        "seed": seed,
        "macro_f1": float(test["macro_f1"]),
        "weighted_f1": float(test["weighted_f1"]),
        "accuracy": float(test["accuracy"]),
        "f1_label_0": float(per_class.get("0", 0.0)),
        "f1_label_1": float(per_class.get("1", 0.0)),
        "f1_label_2": float(per_class.get("2", 0.0)),
        "source": str(path),
    }

def parse_prediction_identity(path: Path) -> tuple[str, str, str, int] | None:
    parts = path.stem.split("_")
    if path.stem.startswith("baseline_") and len(parts) == 3:
        _, ratio, seed = parts
        return "tfidf_logreg", "tfidf_logreg", ratio, int(seed)
    if path.stem.startswith("phobert_") and len(parts) >= 4:
        seed = int(parts[-1])
        ratio = parts[-2]
        augmentation = "_".join(parts[1:-2])
        return "phobert", augmentation, ratio, seed
    return None

def row_from_predictions(path: Path) -> dict[str, object] | None:
    identity = parse_prediction_identity(path)
    if identity is None:
        return None
    model, augmentation, ratio, seed = identity
    predictions = pd.read_csv(path)
    required = {"true_label", "predicted_label"}
    if not required.issubset(predictions.columns):
        return None
    metrics = compute_metrics(
        predictions["true_label"].astype(int),
        predictions["predicted_label"].astype(int),
    )
    per_class = metrics["per_class_f1"]
    return {
        "model": model,
        "augmentation": augmentation,
        "ratio": ratio,
        "seed": seed,
        "macro_f1": float(metrics["macro_f1"]),
        "weighted_f1": float(metrics["weighted_f1"]),
        "accuracy": float(metrics["accuracy"]),
        "f1_label_0": float(per_class.get("0", 0.0)),
        "f1_label_1": float(per_class.get("1", 0.0)),
        "f1_label_2": float(per_class.get("2", 0.0)),
        "source": str(path),
    }

def collect_result_rows(results_dir: str | Path) -> list[dict[str, object]]:
    results_path = Path(results_dir)
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str, str, int]] = set()

    for path in sorted((results_path / "logs").glob("*.json")):
        row = row_from_metrics(path)
        key = identity_key(
            str(row["model"]),
            str(row["augmentation"]),
            str(row["ratio"]),
            int(row["seed"]),
        )
        seen.add(key)
        rows.append(row)

    for path in sorted((results_path / "predictions").glob("*.csv")):
        row = row_from_predictions(path)
        if row is None:
            continue
        key = identity_key(
            str(row["model"]),
            str(row["augmentation"]),
            str(row["ratio"]),
            int(row["seed"]),
        )
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)

    return rows

def aggregate(rows: list[dict[str, object]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = pd.DataFrame(rows)
    if runs.empty:
        return runs, runs

    runs["ratio_float"] = runs["ratio"].astype(float)
    runs = runs.sort_values(["model", "augmentation", "ratio_float", "seed"]).drop(
        columns=["ratio_float"]
    )
    grouped = runs.groupby(["model", "augmentation", "ratio"], as_index=False)
    summary = grouped.agg(
        seed_count=("seed", "nunique"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        weighted_f1_mean=("weighted_f1", "mean"),
        weighted_f1_std=("weighted_f1", "std"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        f1_label_0_mean=("f1_label_0", "mean"),
        f1_label_1_mean=("f1_label_1", "mean"),
        f1_label_2_mean=("f1_label_2", "mean"),
    )
    std_columns = [column for column in summary.columns if column.endswith("_std")]
    summary[std_columns] = summary[std_columns].fillna(0.0)
    summary["ratio_float"] = summary["ratio"].astype(float)
    summary = summary.sort_values(["model", "augmentation", "ratio_float"]).drop(
        columns=["ratio_float"]
    )
    return runs, summary

def write_markdown(summary: pd.DataFrame, path: Path) -> None:
    lines = ["# Main Results", ""]
    if summary.empty:
        lines.append("No result artifacts found.")
    else:
        display = summary.copy()
        for column in display.columns:
            if column.endswith("_mean") or column.endswith("_std"):
                display[column] = display[column].map(lambda value: f"{value:.4f}")
        columns = list(display.columns)
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("|" + "|".join("---" for _ in columns) + "|")
        for _, row in display.iterrows():
            values = [str(row[column]) for column in columns]
            lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")

def main() -> None:
    args = parse_args()
    results_path = Path(args.results_dir)
    tables_dir = results_path / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_result_rows(results_path)
    runs, summary = aggregate(rows)

    runs_path = tables_dir / "main_results_runs.csv"
    summary_path = tables_dir / "main_results.csv"
    markdown_path = tables_dir / "main_results.md"
    runs.to_csv(runs_path, index=False)
    summary.to_csv(summary_path, index=False)
    write_markdown(summary, markdown_path)

    print(f"Saved {runs_path}")
    print(f"Saved {summary_path}")
    print(f"Saved {markdown_path}")
    print(f"Aggregated result groups: {len(summary)}")

if __name__ == "__main__":
    main()
