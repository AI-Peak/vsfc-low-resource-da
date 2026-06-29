"""Generate final Phase 9 report tables and figures."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.reporter import markdown_table
from src.evaluation.reporter import read_csv_if_exists
from src.evaluation.reporter import write_text
from src.experiments.analyze_drift import analyze_drift

PHASE3_FULL_BASELINE = 0.8478276168

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Phase 9 report artifacts.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--ratio", default="0.05")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def collect_final_results(results_dir: str | Path) -> pd.DataFrame:
    """Collect the known final metrics from committed summary tables."""
    tables_dir = Path(results_dir) / "tables"
    rows: list[dict[str, object]] = [
        {
            "phase": "Phase 3",
            "ratio": "1.00",
            "method": "none",
            "macro_f1": PHASE3_FULL_BASELINE,
            "delta_vs_none": 0.0,
            "notes": "near-gate 100% PhoBERT baseline; gate was 0.85",
        }
    ]

    phase4 = read_csv_if_exists(tables_dir / "phase4_eda_summary.csv")
    for _, row in phase4.iterrows():
        ratio = f"{float(row['ratio']):.2f}"
        rows.append(
            {
                "phase": "Phase 4",
                "ratio": ratio,
                "method": "none",
                "macro_f1": float(row["none_macro_f1"]),
                "delta_vs_none": 0.0,
                "notes": "low-resource baseline",
            }
        )
        rows.append(
            {
                "phase": "Phase 4",
                "ratio": ratio,
                "method": "eda",
                "macro_f1": float(row["eda_macro_f1"]),
                "delta_vs_none": float(row["delta_macro_f1"]),
                "notes": str(row.get("outcome", "")),
            }
        )

    phase5 = read_csv_if_exists(tables_dir / "phase5_llm_summary.csv")
    for _, row in phase5.iterrows():
        rows.append(
            {
                "phase": "Phase 5",
                "ratio": f"{float(row['ratio']):.2f}",
                "method": "llm_paraphrase_raw",
                "macro_f1": float(row["llm_raw_macro_f1"]),
                "delta_vs_none": float(row["delta_vs_none"]),
                "notes": str(row.get("method", "")),
            }
        )

    phase6 = read_csv_if_exists(tables_dir / "phase6_filter_result_summary.csv")
    for _, row in phase6.iterrows():
        rows.append(
            {
                "phase": "Phase 6",
                "ratio": f"{float(row['ratio']):.2f}",
                "method": "llm_paraphrase_filtered",
                "macro_f1": float(row["llm_filtered_macro_f1"]),
                "delta_vs_none": float(row["delta_filtered_vs_none"]),
                "notes": str(row.get("notes", "")),
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["ratio_float"] = frame["ratio"].astype(float)
    frame = frame.sort_values(["ratio_float", "method", "phase"]).drop(
        columns=["ratio_float"]
    )
    return frame

def best_by_ratio(results: pd.DataFrame) -> pd.DataFrame:
    """Return the best method per ratio."""
    if results.empty:
        return results
    return (
        results.sort_values(["ratio", "macro_f1"], ascending=[True, False])
        .groupby("ratio", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

def plot_macro_f1(results: pd.DataFrame, output_path: str | Path) -> Path:
    """Plot macro-F1 by method and data ratio."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if results.empty:
        return path

    low_resource = results[results["ratio"].astype(float) < 1.0].copy()
    if low_resource.empty:
        return path

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method, group in low_resource.groupby("method", sort=True):
        ordered = group.sort_values("ratio")
        ax.plot(
            ordered["ratio"].astype(float),
            ordered["macro_f1"].astype(float),
            marker="o",
            linewidth=2,
            label=method,
        )
    ax.set_xlabel("Training ratio")
    ax.set_ylabel("Test macro-F1")
    ax.set_title("Low-resource augmentation results")
    ax.set_ylim(0.72, 0.84)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path

def write_final_report(
    results: pd.DataFrame,
    drift_md_path: Path,
    figure_path: Path,
    output_path: str | Path,
) -> Path:
    """Write a concise final report for the project."""
    rows = results.to_dict("records")
    best_rows = best_by_ratio(results).to_dict("records")
    lines = [
        "# Final Experiment Report",
        "",
        "## Executive Summary",
        "",
        "- Phase 3 PhoBERT no-augmentation baseline was near the original gate: "
        "test macro-F1 0.8478 against the 0.85 target.",
        "- Conservative EDA had a mixed effect: it helped 5% and 20%, but hurt 10%.",
        "- The 5% raw LLM-style paraphrase pilot was the strongest augmentation result "
        "with test macro-F1 0.7864.",
        "- Phase 6 filtering removed only 7 of 571 paraphrases and reduced the 5% "
        "LLM score to 0.7752, so filtering improved QC but did not beat raw LLM.",
        "",
        "## Final Results",
        "",
        markdown_table(
            rows,
            ["phase", "ratio", "method", "macro_f1", "delta_vs_none", "notes"],
        ),
        "",
        "## Best Method Per Ratio",
        "",
        markdown_table(
            best_rows,
            ["phase", "ratio", "method", "macro_f1", "delta_vs_none", "notes"],
        ),
        "",
        "## Research Question Notes",
        "",
        "- RQ1: augmentation can help in low-resource settings, but the effect is method "
        "and ratio dependent.",
        "- RQ2: the clearest gain is at 5%, where raw LLM improves over none by 0.0301 "
        "macro-F1.",
        "- RQ3: in the available 5% pilot, raw LLM outperforms EDA by 0.0223 macro-F1.",
        "- RQ4: the heuristic filter flagged a low drift-risk rate, dropping 1.23% of "
        "raw paraphrases; filtered LLM underperformed raw LLM in macro-F1.",
        "- RQ5: low-resource augmentation does not close the gap to the 100% PhoBERT "
        "ceiling in the current single-seed experiments.",
        "",
        "## Artifacts",
        "",
        f"- Macro-F1 figure: `{figure_path.as_posix()}`",
        f"- Drift analysis: `{drift_md_path.as_posix()}`",
        "",
        "## Limitations",
        "",
        "- Gemini quota prevented full API-based LLM generation for all ratios.",
        "- Current LLM results are a 5% pilot using the offline fallback.",
        "- Statistical testing needs prediction CSVs in the active runtime; if Kaggle "
        "reset removed them, rerun significance tests only after regenerating or "
        "restoring predictions.",
        "",
    ]
    return write_text(output_path, "\n".join(lines))

def generate_phase9_report(
    results_dir: str | Path = "results",
    ratio: str = "0.05",
    seed: int = 42,
) -> tuple[Path, Path, Path, Path]:
    """Generate Phase 9 report artifacts and return their paths."""
    results_path = Path(results_dir)
    tables_dir = results_path / "tables"
    figures_dir = tables_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    drift_csv_path, _, drift_md_path = analyze_drift(results_dir, ratio=ratio, seed=seed)
    results = collect_final_results(results_dir)
    final_csv = tables_dir / "final_results_summary.csv"
    final_md = tables_dir / "final_report.md"
    figure_path = figures_dir / "phase9_macro_f1_by_ratio.png"
    results.to_csv(final_csv, index=False)
    plot_macro_f1(results, figure_path)
    write_final_report(results, drift_md_path, figure_path, final_md)
    return final_csv, final_md, drift_csv_path, figure_path

def main() -> None:
    args = parse_args()
    final_csv, final_md, drift_csv, figure_path = generate_phase9_report(
        results_dir=args.results_dir,
        ratio=args.ratio,
        seed=args.seed,
    )
    print(f"Saved {final_csv}")
    print(f"Saved {final_md}")
    print(f"Saved {drift_csv}")
    print(f"Saved {figure_path}")

if __name__ == "__main__":
    main()
