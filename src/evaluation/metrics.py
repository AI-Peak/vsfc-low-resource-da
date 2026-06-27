"""Evaluation metrics and plots."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score


DEFAULT_LABELS = (0, 1, 2)
DEFAULT_LABEL_NAMES = ("negative", "neutral", "positive")


def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Sequence[int] = DEFAULT_LABELS,
) -> dict[str, object]:
    """Compute accuracy, macro/weighted F1, and per-class F1."""
    per_class = f1_score(
        y_true,
        y_pred,
        labels=list(labels),
        average=None,
        zero_division=0,
    )
    return {
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=list(labels),
                average="weighted",
                zero_division=0,
            )
        ),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class_f1": {
            str(label): float(score) for label, score in zip(labels, per_class)
        },
    }


def confusion_matrix_plot(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    save_path: str | Path,
    labels: Sequence[int] = DEFAULT_LABELS,
    label_names: Sequence[str] = DEFAULT_LABEL_NAMES,
) -> None:
    """Save a confusion matrix heatmap."""
    matrix = confusion_matrix(y_true, y_pred, labels=list(labels))
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=list(label_names),
        yticklabels=list(label_names),
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
