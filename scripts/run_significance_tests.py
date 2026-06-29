"""Run Phase 8 significance tests for available prediction artifacts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.aggregate_results import parse_prediction_identity
from src.evaluation.stats_test import macro_f1
from src.evaluation.stats_test import paired_bootstrap_test
from src.evaluation.stats_test import paired_t_test_across_seeds

DEFAULT_COMPARISONS = (
    ("none", "eda"),
    ("none", "llm_paraphrase_raw"),
    ("none", "llm_paraphrase_filtered"),
    ("eda", "llm_paraphrase_raw"),
    ("llm_paraphrase_raw", "llm_paraphrase_filtered"),
)

@dataclass(frozen=True)
class PredictionArtifact:
    """Loaded prediction file plus experiment identity."""

    model: str
    augmentation: str
    ratio: str
    seed: int
    path: Path
    frame: pd.DataFrame

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paired significance tests.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--ratios", nargs="+", default=None)
    parser.add_argument("--n-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args()

def load_prediction_artifacts(results_dir: str | Path) -> dict[tuple[str, str, int], PredictionArtifact]:
    """Load PhoBERT prediction artifacts keyed by augmentation, ratio, and seed."""
    predictions_dir = Path(results_dir) / "predictions"
    artifacts: dict[tuple[str, str, int], PredictionArtifact] = {}
    for path in sorted(predictions_dir.glob("phobert_*.csv")):
        identity = parse_prediction_identity(path)
        if identity is None:
            continue
        model, augmentation, ratio, seed = identity
        frame = pd.read_csv(path)
        required = {"true_label", "predicted_label"}
        if not required.issubset(frame.columns):
            continue
        artifacts[(augmentation, ratio, seed)] = PredictionArtifact(
            model=model,
            augmentation=augmentation,
            ratio=ratio,
            seed=seed,
            path=path,
            frame=frame,
        )
    return artifacts

def aligned_frames(
    artifact_a: PredictionArtifact,
    artifact_b: PredictionArtifact,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return aligned y_true/pred_a/pred_b arrays for one paired seed."""
    frame_a = artifact_a.frame.reset_index(drop=True)
    frame_b = artifact_b.frame.reset_index(drop=True)
    if len(frame_a) != len(frame_b):
        raise ValueError(
            f"Prediction length mismatch: {artifact_a.path} vs {artifact_b.path}"
        )

    y_a = frame_a["true_label"].astype(int).to_numpy()
    y_b = frame_b["true_label"].astype(int).to_numpy()
    if not np.array_equal(y_a, y_b):
        raise ValueError(
            f"true_label mismatch between paired files: {artifact_a.path} vs {artifact_b.path}"
        )

    return (
        y_a,
        frame_a["predicted_label"].astype(int).to_numpy(),
        frame_b["predicted_label"].astype(int).to_numpy(),
    )

def run_one_comparison(
    artifacts: dict[tuple[str, str, int], PredictionArtifact],
    ratio: str,
    method_a: str,
    method_b: str,
    n_resamples: int,
    seed: int,
    alpha: float,
    comparison_count: int,
) -> dict[str, object] | None:
    seeds_a = {key[2] for key in artifacts if key[0] == method_a and key[1] == ratio}
    seeds_b = {key[2] for key in artifacts if key[0] == method_b and key[1] == ratio}
    paired_seeds = sorted(seeds_a.intersection(seeds_b))
    if not paired_seeds:
        return None

    y_parts: list[np.ndarray] = []
    pred_a_parts: list[np.ndarray] = []
    pred_b_parts: list[np.ndarray] = []
    metric_a: list[float] = []
    metric_b: list[float] = []
    for paired_seed in paired_seeds:
        artifact_a = artifacts[(method_a, ratio, paired_seed)]
        artifact_b = artifacts[(method_b, ratio, paired_seed)]
        y_true, preds_a, preds_b = aligned_frames(artifact_a, artifact_b)
        y_parts.append(y_true)
        pred_a_parts.append(preds_a)
        pred_b_parts.append(preds_b)
        metric_a.append(macro_f1(y_true, preds_a))
        metric_b.append(macro_f1(y_true, preds_b))

    bootstrap = paired_bootstrap_test(
        preds_a=np.concatenate(pred_a_parts),
        preds_b=np.concatenate(pred_b_parts),
        y_true=np.concatenate(y_parts),
        n_resamples=n_resamples,
        seed=seed,
    )
    t_statistic, t_p_value = paired_t_test_across_seeds(metric_a, metric_b)
    corrected_alpha = alpha / comparison_count
    return {
        "ratio": ratio,
        "comparison": f"{method_a}_vs_{method_b}",
        "method_a": method_a,
        "method_b": method_b,
        "seed_count": len(paired_seeds),
        "seeds": ",".join(str(value) for value in paired_seeds),
        "macro_f1_a_mean": float(np.mean(metric_a)),
        "macro_f1_b_mean": float(np.mean(metric_b)),
        "mean_diff_b_minus_a": float(np.mean(np.asarray(metric_b) - np.asarray(metric_a))),
        "bootstrap_diff_b_minus_a": bootstrap.observed_diff,
        "bootstrap_ci_low": bootstrap.ci_low,
        "bootstrap_ci_high": bootstrap.ci_high,
        "p_value_bootstrap": bootstrap.p_value,
        "t_statistic": t_statistic,
        "p_value_ttest": t_p_value,
        "alpha": alpha,
        "bonferroni_alpha": corrected_alpha,
        "significant_at_05": bool(bootstrap.p_value < corrected_alpha),
        "n_resamples": bootstrap.n_resamples,
        "note": "" if len(paired_seeds) >= 2 else "ttest_unavailable_single_seed",
    }

def write_markdown(rows: list[dict[str, object]], path: Path) -> None:
    lines = ["# Significance Tests", ""]
    if not rows:
        lines.append("No paired comparisons were available.")
    else:
        frame = pd.DataFrame(rows)
        display_columns = [
            "ratio",
            "comparison",
            "seed_count",
            "macro_f1_a_mean",
            "macro_f1_b_mean",
            "mean_diff_b_minus_a",
            "bootstrap_ci_low",
            "bootstrap_ci_high",
            "p_value_bootstrap",
            "bonferroni_alpha",
            "significant_at_05",
            "note",
        ]
        display = frame[display_columns].copy()
        numeric_columns = [
            "macro_f1_a_mean",
            "macro_f1_b_mean",
            "mean_diff_b_minus_a",
            "bootstrap_ci_low",
            "bootstrap_ci_high",
            "p_value_bootstrap",
            "bonferroni_alpha",
        ]
        for column in numeric_columns:
            display[column] = display[column].map(lambda value: f"{float(value):.4f}")
        columns = list(display.columns)
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("|" + "|".join("---" for _ in columns) + "|")
        for _, row in display.iterrows():
            lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")

def main() -> None:
    args = parse_args()
    artifacts = load_prediction_artifacts(args.results_dir)
    available_ratios = sorted({key[1] for key in artifacts}, key=float)
    ratios = args.ratios or available_ratios

    available_comparisons = [
        (ratio, method_a, method_b)
        for ratio in ratios
        for method_a, method_b in DEFAULT_COMPARISONS
        if any(key[0] == method_a and key[1] == ratio for key in artifacts)
        and any(key[0] == method_b and key[1] == ratio for key in artifacts)
    ]
    comparison_count = max(len(available_comparisons), 1)
    rows = [
        row
        for ratio, method_a, method_b in available_comparisons
        for row in [
            run_one_comparison(
                artifacts=artifacts,
                ratio=ratio,
                method_a=method_a,
                method_b=method_b,
                n_resamples=args.n_resamples,
                seed=args.seed,
                alpha=args.alpha,
                comparison_count=comparison_count,
            )
        ]
        if row is not None
    ]

    tables_dir = Path(args.results_dir) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tables_dir / "significance_tests.csv"
    md_path = tables_dir / "significance_tests.md"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    write_markdown(rows, md_path)

    print(f"Saved {csv_path}")
    print(f"Saved {md_path}")
    print(f"Completed paired comparisons: {len(rows)}")

if __name__ == "__main__":
    main()
