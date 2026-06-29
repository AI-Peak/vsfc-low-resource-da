"""Statistical tests for paired model comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats
from sklearn.metrics import f1_score

DEFAULT_LABELS = (0, 1, 2)

@dataclass(frozen=True)
class BootstrapResult:
    """Result from a paired bootstrap test."""

    observed_diff: float
    p_value: float
    ci_low: float
    ci_high: float
    n_resamples: int

def macro_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Compute macro-F1 with the project label set."""
    return float(
        f1_score(
            y_true,
            y_pred,
            labels=list(DEFAULT_LABELS),
            average="macro",
            zero_division=0,
        )
    )

def paired_bootstrap_test(
    preds_a: Sequence[int],
    preds_b: Sequence[int],
    y_true: Sequence[int],
    n_resamples: int = 10000,
    seed: int = 42,
    confidence: float = 0.95,
) -> BootstrapResult:
    """Compare two paired prediction vectors with bootstrap macro-F1.

    The reported difference is ``macro_f1(b) - macro_f1(a)``. A positive value
    means the second method in the comparison is better.
    """
    if n_resamples <= 0:
        raise ValueError(f"n_resamples must be positive, got {n_resamples}")

    y_array = np.asarray(y_true, dtype=int)
    a_array = np.asarray(preds_a, dtype=int)
    b_array = np.asarray(preds_b, dtype=int)
    if not (len(y_array) == len(a_array) == len(b_array)):
        raise ValueError(
            "paired_bootstrap_test requires equal-length y_true, preds_a, and preds_b"
        )
    if len(y_array) == 0:
        raise ValueError("paired_bootstrap_test requires at least one example")

    observed = macro_f1(y_array, b_array) - macro_f1(y_array, a_array)
    rng = np.random.default_rng(seed)
    n_examples = len(y_array)
    differences = np.empty(n_resamples, dtype=float)
    for index in range(n_resamples):
        sample_indices = rng.integers(0, n_examples, size=n_examples)
        sample_y = y_array[sample_indices]
        differences[index] = macro_f1(sample_y, b_array[sample_indices]) - macro_f1(
            sample_y,
            a_array[sample_indices],
        )

    alpha = 1.0 - confidence
    ci_low, ci_high = np.quantile(differences, [alpha / 2.0, 1.0 - (alpha / 2.0)])
    centered = differences - observed
    p_value = float(np.mean(np.abs(centered) >= abs(observed)))
    return BootstrapResult(
        observed_diff=float(observed),
        p_value=p_value,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        n_resamples=int(n_resamples),
    )

def paired_t_test_across_seeds(
    metrics_a: Sequence[float],
    metrics_b: Sequence[float],
) -> tuple[float, float]:
    """Run a paired t-test over per-seed metrics.

    Returns ``(nan, nan)`` when fewer than two paired seeds are available.
    """
    a_array = np.asarray(metrics_a, dtype=float)
    b_array = np.asarray(metrics_b, dtype=float)
    if len(a_array) != len(b_array):
        raise ValueError("paired t-test requires equal numbers of seed metrics")
    if len(a_array) < 2:
        return float("nan"), float("nan")
    result = stats.ttest_rel(b_array, a_array)
    return float(result.statistic), float(result.pvalue)
