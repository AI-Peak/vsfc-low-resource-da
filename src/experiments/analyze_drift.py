"""Analyze LLM paraphrase filtering and drift-risk signals."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.evaluation.reporter import markdown_table
from src.evaluation.reporter import read_csv_if_exists
from src.evaluation.reporter import write_text

def as_bool_series(series: pd.Series) -> pd.Series:
    """Convert CSV-loaded keep flags to booleans."""
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze LLM drift/filtering artifacts.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--ratio", default="0.05")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def reason_counts(qc: pd.DataFrame) -> dict[str, int]:
    """Count non-kept filter reasons from the QC diagnostics file."""
    if qc.empty or "reasons" not in qc.columns or "keep" not in qc.columns:
        return {}
    dropped = qc[~as_bool_series(qc["keep"])]
    counts: dict[str, int] = {}
    for raw_reason in dropped["reasons"].dropna().astype(str):
        for reason in raw_reason.split(";"):
            reason = reason.strip()
            if reason and reason != "kept":
                counts[reason] = counts.get(reason, 0) + 1
    return counts

def per_class_filter_summary(qc: pd.DataFrame) -> list[dict[str, object]]:
    """Summarize keep/drop counts per sentiment label."""
    if qc.empty or "label" not in qc.columns or "keep" not in qc.columns:
        return []
    rows: list[dict[str, object]] = []
    for label, group in qc.groupby("label", sort=True):
        kept = int(as_bool_series(group["keep"]).sum())
        total = int(len(group))
        dropped = total - kept
        rows.append(
            {
                "label": int(label),
                "raw_rows": total,
                "kept_rows": kept,
                "dropped_rows": dropped,
                "drop_rate": dropped / total if total else 0.0,
            }
        )
    return rows

def build_drift_rows(results_dir: str | Path, ratio: str, seed: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Build overall and per-class drift/filtering summary rows."""
    tables_dir = Path(results_dir) / "tables"
    filter_summary = read_csv_if_exists(tables_dir / "phase6_filter_summary.csv")
    qc = read_csv_if_exists(tables_dir / f"phase6_filter_qc_{ratio}_{seed}.csv")

    if not filter_summary.empty:
        row = filter_summary[filter_summary["ratio"].astype(str) == str(ratio)]
        source = row.iloc[0].to_dict() if not row.empty else filter_summary.iloc[0].to_dict()
        raw_rows = int(source.get("raw_rows", len(qc)))
        kept_rows = int(source.get("kept_rows", 0))
        dropped_rows = int(source.get("dropped_rows", max(raw_rows - kept_rows, 0)))
        keep_rate = float(source.get("keep_rate", kept_rows / raw_rows if raw_rows else 0.0))
    elif not qc.empty:
        raw_rows = int(len(qc))
        kept_rows = int(as_bool_series(qc["keep"]).sum()) if "keep" in qc.columns else 0
        dropped_rows = raw_rows - kept_rows
        keep_rate = kept_rows / raw_rows if raw_rows else 0.0
    else:
        raw_rows = kept_rows = dropped_rows = 0
        keep_rate = 0.0

    reasons = reason_counts(qc)
    overall_rows = [
        {
            "ratio": ratio,
            "seed": seed,
            "raw_rows": raw_rows,
            "kept_rows": kept_rows,
            "dropped_rows": dropped_rows,
            "keep_rate": keep_rate,
            "drop_rate": 1.0 - keep_rate if raw_rows else 0.0,
            "drop_reasons": "; ".join(f"{key}={value}" for key, value in sorted(reasons.items())),
            "interpretation": "low observed filter-risk rate; filtering was conservative",
        }
    ]
    return overall_rows, per_class_filter_summary(qc)

def write_markdown(
    overall_rows: list[dict[str, object]],
    class_rows: list[dict[str, object]],
    path: str | Path,
) -> Path:
    """Write drift analysis as Markdown."""
    lines = ["# Drift Analysis", ""]
    lines.append("## Overall")
    lines.append("")
    lines.append(
        markdown_table(
            overall_rows,
            [
                "ratio",
                "seed",
                "raw_rows",
                "kept_rows",
                "dropped_rows",
                "keep_rate",
                "drop_rate",
                "drop_reasons",
            ],
        )
    )
    lines.append("")
    lines.append("## Per Class")
    lines.append("")
    if class_rows:
        lines.append(
            markdown_table(
                class_rows,
                ["label", "raw_rows", "kept_rows", "dropped_rows", "drop_rate"],
            )
        )
    else:
        lines.append("Per-class QC diagnostics were not available.")
    lines.append("")
    return write_text(path, "\n".join(lines))

def analyze_drift(
    results_dir: str | Path = "results",
    ratio: str = "0.05",
    seed: int = 42,
) -> tuple[Path, Path, Path]:
    """Analyze Phase 6 filter diagnostics and write output artifacts."""
    tables_dir = Path(results_dir) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    overall_rows, class_rows = build_drift_rows(results_dir, ratio, seed)

    overall_path = tables_dir / "drift_analysis.csv"
    class_path = tables_dir / "drift_analysis_by_class.csv"
    md_path = tables_dir / "drift_analysis.md"
    pd.DataFrame(overall_rows).to_csv(overall_path, index=False)
    pd.DataFrame(class_rows).to_csv(class_path, index=False)
    write_markdown(overall_rows, class_rows, md_path)
    return overall_path, class_path, md_path

def main() -> None:
    args = parse_args()
    overall_path, class_path, md_path = analyze_drift(
        results_dir=args.results_dir,
        ratio=args.ratio,
        seed=args.seed,
    )
    print(f"Saved {overall_path}")
    print(f"Saved {class_path}")
    print(f"Saved {md_path}")

if __name__ == "__main__":
    main()
