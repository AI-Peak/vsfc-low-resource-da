"""PhoBERT experiment runner."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

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
    parser.add_argument("--num-epochs", type=float, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-dev-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
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
        LOGGER.info("Loading preprocessed cache: %s", cache_path)
        return load_split(cache_path)

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
    num_epochs: float | None = None,
    max_train_samples: int | None = None,
    max_dev_samples: int | None = None,
    max_test_samples: int | None = None,
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
    if num_epochs is not None:
        phobert_config["num_epochs"] = num_epochs

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

    train_split_name = f"train_{augmentation}_{ratio_name}_{seed}"
    if any(value is not None for value in (max_train_samples, max_dev_samples, max_test_samples)):
        train_split_name += "_sampled"
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
        split_name="dev",
        data_dir=data_path,
        lowercase=lowercase,
        segment=segment,
        vncorenlp_dir=vncorenlp_dir,
        force=force_preprocess,
    )
    test_df = prepare_preprocessed_frame(
        test_df,
        split_name="test",
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
    dev_metrics = compute_metrics(dev_df["sentiment"], dev_output.predictions)
    test_metrics = compute_metrics(test_df["sentiment"], prediction_output.predictions)
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
        "config": phobert_config,
    }

    prediction_frame = build_prediction_frame(
        test_df,
        prediction_output.predictions,
        prediction_output.probabilities,
    )
    ensure_parent_dir(prediction_path)
    prediction_frame.to_csv(prediction_path, index=False)
    save_json(metrics, metrics_path)
    confusion_matrix_plot(
        test_df["sentiment"],
        prediction_output.predictions,
        save_path=figure_path,
    )

    LOGGER.info("Saved predictions: %s", prediction_path)
    LOGGER.info("Saved metrics: %s", metrics_path)
    LOGGER.info("Saved confusion matrix: %s", figure_path)
    LOGGER.info("Test macro_f1=%.4f", test_metrics["macro_f1"])

    sample_limited = any(
        value is not None for value in (max_train_samples, max_dev_samples, max_test_samples)
    )
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
        num_epochs=args.num_epochs,
        max_train_samples=args.max_train_samples,
        max_dev_samples=args.max_dev_samples,
        max_test_samples=args.max_test_samples,
        disable_gating=args.disable_gating,
        gating_threshold=args.gating_threshold,
    )


if __name__ == "__main__":
    main()
