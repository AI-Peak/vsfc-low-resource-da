"""PhoBERT experiment runner."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.preprocess import preprocess_frame
from src.data.subsample import create_subsamples
from src.data.subsample import format_ratio
from src.evaluation.metrics import compute_metrics
from src.evaluation.metrics import confusion_matrix_plot
from src.models.phobert_trainer import PhoBERTTrainer
from src.utils.io import ensure_parent_dir
from src.utils.io import read_json
from src.utils.io import read_yaml
from src.utils.io import save_json
from src.utils.logging import get_logger
from src.utils.seed import set_seed

LOGGER = get_logger(__name__)
VALID_RATIOS = (0.05, 0.10, 0.20, 1.00)
VALID_AUGMENTATIONS = (
    "none",
    "eda",
    "llm_paraphrase_raw",
    "llm_paraphrase_filtered",
)
VALID_DECISION_RULES = ("argmax", "tune_logit_bias", "tune_logit_affine")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PhoBERT fine-tuning.")
    parser.add_argument("--ratio", type=float, required=True, choices=VALID_RATIOS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augmentation", choices=VALID_AUGMENTATIONS, default="none")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--phobert-config", default="configs/phobert.yaml")
    parser.add_argument("--force-subsample", action="store_true")
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-segment", action="store_true")
    parser.add_argument("--lowercase", action="store_true")
    parser.add_argument("--vncorenlp-dir", default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--keep-checkpoints", action="store_true")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--num-epochs", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=None)
    parser.add_argument("--label-smoothing-factor", type=float, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--metric-for-best-model", default=None)
    parser.add_argument(
        "--class-weighting",
        choices=("none", "balanced", "sqrt_balanced"),
        default=None,
    )
    parser.add_argument("--logging-steps", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-dev-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--decision-rule", choices=VALID_DECISION_RULES, default="argmax")
    parser.add_argument("--logit-bias-min", type=float, default=-2.0)
    parser.add_argument("--logit-bias-max", type=float, default=2.0)
    parser.add_argument("--logit-bias-step", type=float, default=0.05)
    parser.add_argument("--logit-scale-values", default="0.8,0.9,1.0,1.1,1.2")
    parser.add_argument("--disable-gating", action="store_true")
    parser.add_argument("--gating-threshold", type=float, default=0.85)
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


def augmented_path(
    augmentation: str,
    ratio: float,
    seed: int,
    data_dir: str | Path,
) -> Path:
    ratio_name = format_ratio(ratio)
    name_map = {
        "eda": f"eda_{ratio_name}_{seed}.csv",
        "llm_paraphrase_raw": f"llm_raw_{ratio_name}_{seed}.csv",
        "llm_paraphrase_filtered": f"llm_filtered_{ratio_name}_{seed}.csv",
    }
    return Path(data_dir) / "augmented" / name_map[augmentation]


def load_augmented_training_data(
    base_train_df: pd.DataFrame,
    augmentation: str,
    ratio: float,
    seed: int,
    data_dir: str | Path,
) -> pd.DataFrame:
    if augmentation == "none":
        return base_train_df

    path = augmented_path(augmentation, ratio, seed, data_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing augmentation file: {path}. "
            f"Generate {augmentation} data before running this experiment."
        )

    augmented = pd.read_csv(path)
    required = {"augmented_sentence", "label"}
    missing = required.difference(augmented.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")

    augmented_rows = pd.DataFrame(
        {
            "sentence": augmented["augmented_sentence"].astype(str),
            "sentiment": augmented["label"].astype(int),
        }
    )
    combined = pd.concat([base_train_df, augmented_rows], axis=0, ignore_index=True)
    combined = combined.dropna(subset=["sentence", "sentiment"])
    combined = combined.drop_duplicates(subset=["sentence", "sentiment"])
    return combined.sample(frac=1, random_state=seed).reset_index(drop=True)


def limit_frame(
    frame: pd.DataFrame,
    max_samples: int | None,
    seed: int,
) -> pd.DataFrame:
    if max_samples is None or max_samples >= len(frame):
        return frame
    return frame.sample(n=max_samples, random_state=seed).reset_index(drop=True)


def preprocess_cache_path(
    split_name: str,
    data_dir: str | Path,
    lowercase: bool,
    segment: bool,
) -> Path:
    suffix = "seg" if segment else "norm"
    case = "lower" if lowercase else "case"
    return Path(data_dir) / "preprocessed" / f"{split_name}_{suffix}_{case}.csv"


def prepare_preprocessed_frame(
    frame: pd.DataFrame,
    split_name: str,
    data_dir: str | Path,
    lowercase: bool = False,
    segment: bool = True,
    vncorenlp_dir: str | Path | None = None,
    force: bool = False,
) -> pd.DataFrame:
    cache_path = preprocess_cache_path(
        split_name=split_name,
        data_dir=data_dir,
        lowercase=lowercase,
        segment=segment,
    )
    if cache_path.exists() and not force:
        cached = load_split(cache_path)
        if len(cached) == len(frame):
            LOGGER.info("Loading preprocessed cache: %s", cache_path)
            return cached
        LOGGER.info(
            "Ignoring preprocessed cache with row mismatch: %s has %s rows, expected %s",
            cache_path,
            len(cached),
            len(frame),
        )

    LOGGER.info(
        "Preprocessing %s rows for %s (segment=%s lowercase=%s)",
        len(frame),
        split_name,
        segment,
        lowercase,
    )
    processed = preprocess_frame(
        frame,
        text_col="sentence",
        output_col="sentence",
        lowercase=lowercase,
        segment=segment,
        save_dir=vncorenlp_dir,
    )
    ensure_parent_dir(cache_path)
    processed.to_csv(cache_path, index=False)
    LOGGER.info("Saved preprocessed cache: %s", cache_path)
    return processed


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

def predict_with_logit_bias(logits, bias: list[float] | np.ndarray) -> np.ndarray:
    """Predict labels after adding a class-specific logit bias."""
    return np.argmax(np.asarray(logits) + np.asarray(bias), axis=-1)

def predict_with_logit_affine(
    logits,
    scale: list[float] | np.ndarray,
    bias: list[float] | np.ndarray,
) -> np.ndarray:
    """Predict labels after class-specific logit scaling and bias."""
    return np.argmax((np.asarray(logits) * np.asarray(scale)) + np.asarray(bias), axis=-1)

def macro_f1_from_predictions(
    labels: np.ndarray,
    predictions: np.ndarray,
    num_labels: int,
) -> float:
    """Compute macro-F1 without sklearn overhead inside tight tuning loops."""
    scores = []
    for label in range(num_labels):
        true_positive = np.sum((labels == label) & (predictions == label))
        false_positive = np.sum((labels != label) & (predictions == label))
        false_negative = np.sum((labels == label) & (predictions != label))
        denominator = (2 * true_positive) + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else (2 * true_positive) / denominator)
    return float(np.mean(scores))

def tune_logit_bias(
    logits,
    labels,
    min_bias: float = -2.0,
    max_bias: float = 2.0,
    step: float = 0.05,
) -> tuple[list[float], dict[str, Any]]:
    """Tune a 3-class logit bias on dev macro-F1.

    Biases are relative, so class 0 is fixed at 0 and classes 1/2 are swept.
    """
    if step <= 0:
        raise ValueError(f"logit-bias-step must be positive, got {step}")

    logits_array = np.asarray(logits)
    labels_array = np.asarray(labels, dtype=int)
    values = np.arange(min_bias, max_bias + (step / 2), step)
    best_score = -1.0
    best_bias = np.zeros(logits_array.shape[1], dtype=float)
    best_predictions = np.argmax(logits_array, axis=-1)

    for label_1_bias in values:
        for label_2_bias in values:
            bias = np.array([0.0, label_1_bias, label_2_bias], dtype=float)
            predictions = predict_with_logit_bias(logits_array, bias)
            score = macro_f1_from_predictions(
                labels_array,
                predictions,
                num_labels=logits_array.shape[1],
            )
            best_norm = float(np.linalg.norm(best_bias))
            current_norm = float(np.linalg.norm(bias))
            if score > best_score or (
                np.isclose(score, best_score) and current_norm < best_norm
            ):
                best_score = score
                best_bias = bias
                best_predictions = predictions

    return (
        [float(value) for value in best_bias],
        {
            "best_dev_macro_f1": float(best_score),
            "grid": {
                "min": float(min_bias),
                "max": float(max_bias),
                "step": float(step),
            },
            "predictions": best_predictions,
        },
    )

def parse_float_values(raw_values: str) -> list[float]:
    values = [float(value.strip()) for value in raw_values.split(",") if value.strip()]
    if not values:
        raise ValueError("logit-scale-values must include at least one float")
    if any(value <= 0 for value in values):
        raise ValueError(f"logit-scale-values must be positive, got {values}")
    return values

def tune_logit_affine(
    logits,
    labels,
    min_bias: float = -2.0,
    max_bias: float = 2.0,
    step: float = 0.05,
    scale_values: list[float] | None = None,
) -> tuple[list[float], list[float], dict[str, Any]]:
    """Tune class-specific logit scale and bias on dev macro-F1.

    Class 0 is fixed at scale=1 and bias=0 so the search stays relative.
    """
    if step <= 0:
        raise ValueError(f"logit-bias-step must be positive, got {step}")

    scale_candidates = np.asarray(scale_values or [0.8, 0.9, 1.0, 1.1, 1.2], dtype=float)
    if np.any(scale_candidates <= 0):
        raise ValueError(f"logit scale values must be positive: {scale_candidates.tolist()}")

    logits_array = np.asarray(logits)
    labels_array = np.asarray(labels, dtype=int)
    bias_values = np.arange(min_bias, max_bias + (step / 2), step)
    best_score = -1.0
    best_scale = np.ones(logits_array.shape[1], dtype=float)
    best_bias = np.zeros(logits_array.shape[1], dtype=float)
    best_predictions = np.argmax(logits_array, axis=-1)

    for label_1_scale in scale_candidates:
        for label_2_scale in scale_candidates:
            scale = np.array([1.0, label_1_scale, label_2_scale], dtype=float)
            scaled_logits = logits_array * scale
            for label_1_bias in bias_values:
                for label_2_bias in bias_values:
                    bias = np.array([0.0, label_1_bias, label_2_bias], dtype=float)
                    predictions = predict_with_logit_bias(scaled_logits, bias)
                    score = macro_f1_from_predictions(
                        labels_array,
                        predictions,
                        num_labels=logits_array.shape[1],
                    )
                    best_complexity = float(
                        np.linalg.norm(best_bias) + np.linalg.norm(best_scale - 1.0)
                    )
                    current_complexity = float(
                        np.linalg.norm(bias) + np.linalg.norm(scale - 1.0)
                    )
                    if score > best_score or (
                        np.isclose(score, best_score)
                        and current_complexity < best_complexity
                    ):
                        best_score = score
                        best_scale = scale
                        best_bias = bias
                        best_predictions = predictions

    return (
        [float(value) for value in best_scale],
        [float(value) for value in best_bias],
        {
            "best_dev_macro_f1": float(best_score),
            "grid": {
                "bias_min": float(min_bias),
                "bias_max": float(max_bias),
                "bias_step": float(step),
                "scale_values": [float(value) for value in scale_candidates],
            },
            "predictions": best_predictions,
        },
    )


def output_paths(
    augmentation: str,
    ratio: float,
    seed: int,
    results_dir: str | Path = "results",
) -> tuple[Path, Path, Path]:
    results_path = Path(results_dir)
    ratio_name = format_ratio(ratio)
    stem = f"phobert_{augmentation}_{ratio_name}_{seed}"
    return (
        results_path / "predictions" / f"{stem}.csv",
        results_path / "logs" / f"{stem}.json",
        results_path / "tables" / "figures" / f"{stem}_confusion_matrix.png",
    )


def create_confusion_matrix_from_predictions(
    prediction_path: str | Path,
    figure_path: str | Path,
) -> None:
    predictions = pd.read_csv(prediction_path)
    confusion_matrix_plot(
        predictions["true_label"].astype(int),
        predictions["predicted_label"].astype(int),
        save_path=figure_path,
    )


def read_configs(
    base_config_path: str | Path,
    phobert_config_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    base_config = read_yaml(base_config_path)
    phobert_config = read_yaml(phobert_config_path)
    return base_config, phobert_config


def run_phobert(
    ratio: float,
    seed: int,
    augmentation: str = "none",
    data_dir: str | Path = "data",
    results_dir: str | Path = "results",
    base_config_path: str | Path = "configs/base.yaml",
    phobert_config_path: str | Path = "configs/phobert.yaml",
    force_subsample: bool = False,
    force_preprocess: bool = False,
    overwrite: bool = False,
    segment: bool = True,
    lowercase: bool = False,
    vncorenlp_dir: str | Path | None = None,
    use_cpu: bool | None = None,
    keep_checkpoints: bool = False,
    model_name: str | None = None,
    max_length: int | None = None,
    batch_size: int | None = None,
    eval_batch_size: int | None = None,
    num_epochs: float | None = None,
    learning_rate: float | None = None,
    weight_decay: float | None = None,
    warmup_ratio: float | None = None,
    label_smoothing_factor: float | None = None,
    early_stopping_patience: int | None = None,
    metric_for_best_model: str | None = None,
    class_weighting: str | None = None,
    logging_steps: int | None = None,
    max_train_samples: int | None = None,
    max_dev_samples: int | None = None,
    max_test_samples: int | None = None,
    decision_rule: str = "argmax",
    logit_bias_min: float = -2.0,
    logit_bias_max: float = 2.0,
    logit_bias_step: float = 0.05,
    logit_scale_values: str = "0.8,0.9,1.0,1.1,1.2",
    disable_gating: bool = False,
    gating_threshold: float = 0.85,
) -> dict[str, Any]:
    set_seed(seed)
    data_path = Path(data_dir)
    ratio_name = format_ratio(ratio)
    prediction_path, metrics_path, figure_path = output_paths(
        augmentation=augmentation,
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
                "Skipping existing PhoBERT artifacts for augmentation=%s ratio=%s seed=%s",
                augmentation,
                ratio_name,
                seed,
            )
        return read_json(metrics_path)

    base_config, phobert_config = read_configs(base_config_path, phobert_config_path)
    if model_name:
        phobert_config["model_name"] = model_name
    if max_length is not None:
        phobert_config["max_length"] = max_length
    if batch_size is not None:
        phobert_config["batch_size"] = batch_size
    if eval_batch_size is not None:
        phobert_config["eval_batch_size"] = eval_batch_size
    if num_epochs is not None:
        phobert_config["num_epochs"] = num_epochs
    if learning_rate is not None:
        phobert_config["learning_rate"] = learning_rate
    if weight_decay is not None:
        phobert_config["weight_decay"] = weight_decay
    if warmup_ratio is not None:
        phobert_config["warmup_ratio"] = warmup_ratio
    if label_smoothing_factor is not None:
        phobert_config["label_smoothing_factor"] = label_smoothing_factor
    if early_stopping_patience is not None:
        phobert_config["early_stopping_patience"] = early_stopping_patience
    if metric_for_best_model is not None:
        phobert_config["metric_for_best_model"] = metric_for_best_model
    if class_weighting is not None:
        phobert_config["class_weighting"] = class_weighting
    if logging_steps is not None:
        phobert_config["logging_steps"] = logging_steps

    base_train_df = load_train_subset(
        ratio=ratio,
        seed=seed,
        data_dir=data_path,
        force_subsample=force_subsample,
    )
    train_df = load_augmented_training_data(
        base_train_df=base_train_df,
        augmentation=augmentation,
        ratio=ratio,
        seed=seed,
        data_dir=data_path,
    )
    dev_df = load_split(data_path / "raw" / "dev.csv")
    test_df = load_split(data_path / "raw" / "test.csv")

    train_df = limit_frame(train_df, max_train_samples, seed)
    dev_df = limit_frame(dev_df, max_dev_samples, seed)
    test_df = limit_frame(test_df, max_test_samples, seed)

    sample_limited = any(
        value is not None for value in (max_train_samples, max_dev_samples, max_test_samples)
    )
    train_split_name = f"train_{augmentation}_{ratio_name}_{seed}"
    dev_split_name = "dev"
    test_split_name = "test"
    if sample_limited:
        train_split_name += "_sampled"
        dev_split_name += "_sampled"
        test_split_name += "_sampled"
        force_preprocess = True

    train_df = prepare_preprocessed_frame(
        train_df,
        split_name=train_split_name,
        data_dir=data_path,
        lowercase=lowercase,
        segment=segment,
        vncorenlp_dir=vncorenlp_dir,
        force=force_preprocess,
    )
    dev_df = prepare_preprocessed_frame(
        dev_df,
        split_name=dev_split_name,
        data_dir=data_path,
        lowercase=lowercase,
        segment=segment,
        vncorenlp_dir=vncorenlp_dir,
        force=force_preprocess,
    )
    test_df = prepare_preprocessed_frame(
        test_df,
        split_name=test_split_name,
        data_dir=data_path,
        lowercase=lowercase,
        segment=segment,
        vncorenlp_dir=vncorenlp_dir,
        force=force_preprocess,
    )

    LOGGER.info(
        "Training PhoBERT augmentation=%s ratio=%s seed=%s train=%s dev=%s test=%s",
        augmentation,
        ratio_name,
        seed,
        len(train_df),
        len(dev_df),
        len(test_df),
    )
    run_name = f"phobert_{augmentation}_{ratio_name}_{seed}"
    trainer = PhoBERTTrainer(
        config=phobert_config,
        seed=seed,
        num_labels=int(base_config.get("data", {}).get("num_labels", 3)),
        run_name=run_name,
        results_dir=results_dir,
        use_cpu=use_cpu,
        keep_checkpoints=keep_checkpoints,
    )
    trainer.train(train_df=train_df, dev_df=dev_df)
    prediction_output = trainer.predict(test_df)

    dev_output = trainer.predict(dev_df)
    raw_dev_metrics = compute_metrics(dev_df["sentiment"], dev_output.predictions)
    raw_test_metrics = compute_metrics(test_df["sentiment"], prediction_output.predictions)
    dev_predictions = dev_output.predictions
    test_predictions = prediction_output.predictions
    decision: dict[str, Any] = {"rule": decision_rule}

    if decision_rule == "tune_logit_bias":
        bias, tuning_info = tune_logit_bias(
            dev_output.logits,
            dev_df["sentiment"],
            min_bias=logit_bias_min,
            max_bias=logit_bias_max,
            step=logit_bias_step,
        )
        dev_predictions = tuning_info.pop("predictions")
        test_predictions = predict_with_logit_bias(prediction_output.logits, bias)
        decision.update(
            {
                "logit_bias": bias,
                "tuning": tuning_info,
            }
        )
        LOGGER.info(
            "Tuned logit bias on dev: bias=%s dev_macro_f1=%.4f",
            [round(value, 4) for value in bias],
            tuning_info["best_dev_macro_f1"],
        )
    elif decision_rule == "tune_logit_affine":
        scale, bias, tuning_info = tune_logit_affine(
            dev_output.logits,
            dev_df["sentiment"],
            min_bias=logit_bias_min,
            max_bias=logit_bias_max,
            step=logit_bias_step,
            scale_values=parse_float_values(logit_scale_values),
        )
        dev_predictions = tuning_info.pop("predictions")
        test_predictions = predict_with_logit_affine(
            prediction_output.logits,
            scale,
            bias,
        )
        decision.update(
            {
                "logit_scale": scale,
                "logit_bias": bias,
                "tuning": tuning_info,
            }
        )
        LOGGER.info(
            "Tuned logit affine rule on dev: scale=%s bias=%s dev_macro_f1=%.4f",
            [round(value, 4) for value in scale],
            [round(value, 4) for value in bias],
            tuning_info["best_dev_macro_f1"],
        )
    elif decision_rule != "argmax":
        raise ValueError(f"Unsupported decision_rule: {decision_rule}")

    dev_metrics = compute_metrics(dev_df["sentiment"], dev_predictions)
    test_metrics = compute_metrics(test_df["sentiment"], test_predictions)
    metrics: dict[str, Any] = {
        "model": "phobert",
        "augmentation": augmentation,
        "ratio": ratio,
        "seed": seed,
        "base_train_size": int(len(base_train_df)),
        "train_size": int(len(train_df)),
        "dev_size": int(len(dev_df)),
        "test_size": int(len(test_df)),
        "preprocessing": {
            "segment": segment,
            "lowercase": lowercase,
            "vncorenlp_dir": str(vncorenlp_dir) if vncorenlp_dir else None,
        },
        "dev": dev_metrics,
        "test": test_metrics,
        "raw_dev": raw_dev_metrics,
        "raw_test": raw_test_metrics,
        "config": phobert_config,
        "class_weights": trainer.resolved_class_weights,
        "decision": decision,
    }

    prediction_frame = build_prediction_frame(
        test_df,
        test_predictions,
        prediction_output.probabilities,
    )
    ensure_parent_dir(prediction_path)
    prediction_frame.to_csv(prediction_path, index=False)
    save_json(metrics, metrics_path)
    confusion_matrix_plot(
        test_df["sentiment"],
        test_predictions,
        save_path=figure_path,
    )

    LOGGER.info("Saved predictions: %s", prediction_path)
    LOGGER.info("Saved metrics: %s", metrics_path)
    LOGGER.info("Saved confusion matrix: %s", figure_path)
    LOGGER.info("Test macro_f1=%.4f", test_metrics["macro_f1"])

    is_phase3_gate = (
        augmentation == "none"
        and ratio == 1.00
        and seed == 42
        and not sample_limited
        and not disable_gating
    )
    if is_phase3_gate and float(test_metrics["macro_f1"]) < gating_threshold:
        raise RuntimeError(
            "Phase 3 gating failed: "
            f"test macro_f1={test_metrics['macro_f1']:.4f} < {gating_threshold:.2f}. "
            "Halt before augmentation and inspect preprocessing/tokenization."
        )

    return metrics


def main() -> None:
    args = parse_args()
    run_phobert(
        ratio=args.ratio,
        seed=args.seed,
        augmentation=args.augmentation,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        base_config_path=args.base_config,
        phobert_config_path=args.phobert_config,
        force_subsample=args.force_subsample,
        force_preprocess=args.force_preprocess,
        overwrite=args.overwrite,
        segment=not args.no_segment,
        lowercase=args.lowercase,
        vncorenlp_dir=args.vncorenlp_dir,
        use_cpu=True if args.cpu else None,
        keep_checkpoints=args.keep_checkpoints,
        model_name=args.model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        label_smoothing_factor=args.label_smoothing_factor,
        early_stopping_patience=args.early_stopping_patience,
        metric_for_best_model=args.metric_for_best_model,
        class_weighting=args.class_weighting,
        logging_steps=args.logging_steps,
        max_train_samples=args.max_train_samples,
        max_dev_samples=args.max_dev_samples,
        max_test_samples=args.max_test_samples,
        decision_rule=args.decision_rule,
        logit_bias_min=args.logit_bias_min,
        logit_bias_max=args.logit_bias_max,
        logit_bias_step=args.logit_bias_step,
        logit_scale_values=args.logit_scale_values,
        disable_gating=args.disable_gating,
        gating_threshold=args.gating_threshold,
    )


if __name__ == "__main__":
    main()
