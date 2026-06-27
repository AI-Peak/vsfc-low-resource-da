"""Classical baseline experiment runner."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.subsample import create_subsamples
from src.data.subsample import format_ratio
from src.evaluation.metrics import compute_metrics
from src.evaluation.metrics import confusion_matrix_plot
from src.models.classical import train_tfidf_logreg
from src.utils.io import ensure_parent_dir
from src.utils.io import read_json
from src.utils.io import save_json
from src.utils.logging import get_logger
from src.utils.seed import set_seed


LOGGER = get_logger(__name__)
VALID_RATIOS = (0.05, 0.10, 0.20, 1.00)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TF-IDF + LogisticRegression baseline.")
    parser.add_argument("--ratio", type=float, required=True, choices=VALID_RATIOS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--force-subsample", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_split(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV file: {path}")
    frame = pd.read_csv(path)
    frame["sentence"] = frame["sentence"].astype(str)
    frame["sentiment"] = frame["sentiment"].astype(int)
    return frame


def load_train_subset(
    ratio: float,
    seed: int,
    data_dir: str | Path,
    force_subsample: bool = False,
) -> pd.DataFrame:
    data_path = Path(data_dir)
    ratio_name = format_ratio(ratio)
    train_path = data_path / "splits" / f"train_{ratio_name}_{seed}.csv"

    if force_subsample or not train_path.exists():
        LOGGER.info("Creating missing subset: %s", train_path)
        full_train = load_split(data_path / "raw" / "train.csv")
        create_subsamples(
            full_train,
            ratios=[ratio],
            seed=seed,
            output_dir=data_path / "splits",
            force=True,
        )

    return load_split(train_path)


def build_prediction_frame(
    frame: pd.DataFrame,
    predictions,
    probabilities,
) -> pd.DataFrame:
    output = pd.DataFrame(
        {
            "sentence": frame["sentence"].astype(str),
            "true_label": frame["sentiment"].astype(int),
            "predicted_label": predictions.astype(int),
        }
    )
    for class_id in range(probabilities.shape[1]):
        output[f"prob_{class_id}"] = probabilities[:, class_id]
    return output


def output_paths(
    ratio: float,
    seed: int,
    results_dir: str | Path = "results",
) -> tuple[Path, Path, Path]:
    """Return prediction, metric, and figure output paths."""
    results_path = Path(results_dir)
    ratio_name = format_ratio(ratio)
    return (
        results_path / "predictions" / f"baseline_{ratio_name}_{seed}.csv",
        results_path / "logs" / f"baseline_{ratio_name}_{seed}.json",
        results_path
        / "tables"
        / "figures"
        / f"baseline_{ratio_name}_{seed}_confusion_matrix.png",
    )


def create_confusion_matrix_from_predictions(
    prediction_path: str | Path,
    figure_path: str | Path,
) -> None:
    """Create a confusion matrix plot from an existing prediction CSV."""
    predictions = pd.read_csv(prediction_path)
    confusion_matrix_plot(
        predictions["true_label"].astype(int),
        predictions["predicted_label"].astype(int),
        save_path=figure_path,
    )


def run_baseline(
    ratio: float,
    seed: int,
    data_dir: str | Path = "data",
    results_dir: str | Path = "results",
    force_subsample: bool = False,
    overwrite: bool = False,
) -> dict[str, object]:
    set_seed(seed)
    data_path = Path(data_dir)
    ratio_name = format_ratio(ratio)
    prediction_path, metrics_path, figure_path = output_paths(
        ratio=ratio,
        seed=seed,
        results_dir=results_dir,
    )

    if not overwrite and prediction_path.exists() and metrics_path.exists():
        if not figure_path.exists():
            LOGGER.info("Creating missing confusion matrix from existing predictions.")
            create_confusion_matrix_from_predictions(prediction_path, figure_path)
            LOGGER.info("Saved confusion matrix: %s", figure_path)
        else:
            LOGGER.info(
                "Skipping existing baseline artifacts for ratio=%s seed=%s",
                ratio_name,
                seed,
            )
        return read_json(metrics_path)

    train_df = load_train_subset(
        ratio=ratio,
        seed=seed,
        data_dir=data_path,
        force_subsample=force_subsample,
    )
    dev_df = load_split(data_path / "raw" / "dev.csv")
    test_df = load_split(data_path / "raw" / "test.csv")

    LOGGER.info(
        "Training baseline ratio=%s seed=%s train=%s dev=%s test=%s",
        ratio_name,
        seed,
        len(train_df),
        len(dev_df),
        len(test_df),
    )
    output = train_tfidf_logreg(
        train_df=train_df,
        dev_df=dev_df,
        test_df=test_df,
        config={"random_state": seed},
    )

    dev_metrics = compute_metrics(dev_df["sentiment"], output.dev.predictions)
    test_metrics = compute_metrics(test_df["sentiment"], output.test.predictions)
    metrics = {
        "model": "tfidf_logreg",
        "ratio": ratio,
        "seed": seed,
        "train_size": int(len(train_df)),
        "dev_size": int(len(dev_df)),
        "test_size": int(len(test_df)),
        "dev": dev_metrics,
        "test": test_metrics,
        "config": {
            "tfidf": {
                "ngram_range": [1, 2],
                "max_features": 20000,
                "min_df": 2,
            },
            "logistic_regression": {
                "class_weight": "balanced",
                "max_iter": 1000,
                "random_state": seed,
            },
        },
    }

    prediction_frame = build_prediction_frame(
        test_df,
        output.test.predictions,
        output.test.probabilities,
    )
    ensure_parent_dir(prediction_path)
    prediction_frame.to_csv(prediction_path, index=False)
    save_json(metrics, metrics_path)
    confusion_matrix_plot(
        test_df["sentiment"],
        output.test.predictions,
        save_path=figure_path,
    )

    LOGGER.info("Saved predictions: %s", prediction_path)
    LOGGER.info("Saved metrics: %s", metrics_path)
    LOGGER.info("Saved confusion matrix: %s", figure_path)
    LOGGER.info("Test macro_f1=%.4f", test_metrics["macro_f1"])
    return metrics


def main() -> None:
    args = parse_args()
    run_baseline(
        ratio=args.ratio,
        seed=args.seed,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        force_subsample=args.force_subsample,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
