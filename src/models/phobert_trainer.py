"""PhoBERT trainer wrapper."""

from __future__ import annotations

import inspect
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import EarlyStoppingCallback
from transformers import Trainer
from transformers import TrainingArguments

from src.data.dataset import VSFCDataset
from src.evaluation.metrics import compute_metrics as compute_classification_metrics
from src.utils.io import save_json

DEFAULT_PHOBERT_CONFIG: dict[str, Any] = {
    "model_name": "vinai/phobert-base",
    "max_length": 128,
    "batch_size": 16,
    "eval_batch_size": 32,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "num_epochs": 10,
    "warmup_ratio": 0.1,
    "early_stopping_patience": 3,
    "metric_for_best_model": "macro_f1",
}


@dataclass
class PhoBERTPredictionOutput:
    """Predicted labels and class probabilities."""

    predictions: np.ndarray
    probabilities: np.ndarray
    logits: np.ndarray


class PhoBERTTrainer:
    """Thin wrapper around HuggingFace Trainer for UIT-VSFC PhoBERT runs."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        seed: int = 42,
        num_labels: int = 3,
        run_name: str = "phobert",
        results_dir: str | Path = "results",
        use_cpu: bool | None = None,
        keep_checkpoints: bool = False,
    ) -> None:
        self.config = DEFAULT_PHOBERT_CONFIG.copy()
        if config:
            self.config.update(config)
        self.seed = int(seed)
        self.num_labels = int(num_labels)
        self.run_name = run_name
        self.results_dir = Path(results_dir)
        self.use_cpu = use_cpu
        self.keep_checkpoints = keep_checkpoints
        self.checkpoint_dir = self.results_dir / "models" / "checkpoints" / run_name
        self.artifact_dir = self.results_dir / "models" / run_name

        model_name = str(self.config["model_name"])
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=self.num_labels,
        )
        self.trainer: Trainer | None = None

    def _training_arguments(self) -> TrainingArguments:
        params = inspect.signature(TrainingArguments.__init__).parameters
        kwargs: dict[str, Any] = {
            "output_dir": str(self.checkpoint_dir),
            "per_device_train_batch_size": int(self.config["batch_size"]),
            "per_device_eval_batch_size": int(self.config["eval_batch_size"]),
            "learning_rate": float(self.config["learning_rate"]),
            "weight_decay": float(self.config["weight_decay"]),
            "num_train_epochs": float(self.config["num_epochs"]),
            "warmup_ratio": float(self.config["warmup_ratio"]),
            "logging_strategy": "epoch",
            "save_strategy": "epoch",
            "save_total_limit": 1,
            "load_best_model_at_end": True,
            "metric_for_best_model": str(self.config["metric_for_best_model"]),
            "greater_is_better": True,
            "seed": self.seed,
            "data_seed": self.seed,
            "report_to": [],
            "run_name": self.run_name,
            "disable_tqdm": False,
            "remove_unused_columns": False,
        }

        if "eval_strategy" in params:
            kwargs["eval_strategy"] = "epoch"
        else:
            kwargs["evaluation_strategy"] = "epoch"

        if "full_determinism" in params:
            kwargs["full_determinism"] = True

        if "save_only_model" in params:
            kwargs["save_only_model"] = True

        if torch.cuda.is_available() and not self.use_cpu:
            if "fp16" in params:
                kwargs["fp16"] = True
        elif self.use_cpu is not False:
            if "use_cpu" in params:
                kwargs["use_cpu"] = True
            elif "no_cuda" in params:
                kwargs["no_cuda"] = True

        return TrainingArguments(**kwargs)

    def _make_dataset(self, frame: pd.DataFrame) -> VSFCDataset:
        return VSFCDataset(
            frame=frame,
            tokenizer=self.tokenizer,
            max_length=int(self.config["max_length"]),
        )

    @staticmethod
    def _compute_trainer_metrics(eval_pred: Any) -> dict[str, float]:
        if hasattr(eval_pred, "predictions"):
            logits = eval_pred.predictions
            labels = eval_pred.label_ids
        else:
            logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        metrics = compute_classification_metrics(labels, predictions)
        flat_metrics = {
            "macro_f1": float(metrics["macro_f1"]),
            "weighted_f1": float(metrics["weighted_f1"]),
            "accuracy": float(metrics["accuracy"]),
        }
        for label, score in metrics["per_class_f1"].items():
            flat_metrics[f"f1_label_{label}"] = float(score)
        return flat_metrics

    def _trainer_kwargs(self) -> dict[str, Any]:
        """Return Trainer kwargs compatible with installed transformers version."""
        params = inspect.signature(Trainer.__init__).parameters
        if "processing_class" in params:
            return {"processing_class": self.tokenizer}
        if "tokenizer" in params:
            return {"tokenizer": self.tokenizer}
        return {}

    def train(self, train_df: pd.DataFrame, dev_df: pd.DataFrame) -> AutoModelForSequenceClassification:
        """Fine-tune PhoBERT and keep the best checkpoint loaded in memory."""
        train_dataset = self._make_dataset(train_df)
        dev_dataset = self._make_dataset(dev_df)
        args = self._training_arguments()
        callbacks = [
            EarlyStoppingCallback(
                early_stopping_patience=int(self.config["early_stopping_patience"])
            )
        ]

        self.trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            compute_metrics=self._compute_trainer_metrics,
            callbacks=callbacks,
            **self._trainer_kwargs(),
        )
        self.trainer.train()
        self.save_lightweight_artifacts()
        if not self.keep_checkpoints:
            self.cleanup_checkpoints()
        return self.model

    def predict(self, test_df: pd.DataFrame) -> PhoBERTPredictionOutput:
        """Predict labels and probabilities for a test DataFrame."""
        if self.trainer is None:
            self.trainer = Trainer(
                model=self.model,
                args=self._training_arguments(),
                compute_metrics=self._compute_trainer_metrics,
                **self._trainer_kwargs(),
            )

        output = self.trainer.predict(self._make_dataset(test_df))
        logits = output.predictions
        probabilities = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        predictions = probabilities.argmax(axis=1)
        return PhoBERTPredictionOutput(
            predictions=predictions,
            probabilities=probabilities,
            logits=logits,
        )

    def save_lightweight_artifacts(self) -> None:
        """Save classifier head weights, tokenizer files, and metadata only."""
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        classifier = getattr(self.model, "classifier", None)
        if classifier is not None:
            torch.save(classifier.state_dict(), self.artifact_dir / "classifier_head.pt")
        self.tokenizer.save_pretrained(self.artifact_dir / "tokenizer")
        self.model.config.save_pretrained(self.artifact_dir / "config")
        save_json(
            {
                "model_name": self.config["model_name"],
                "num_labels": self.num_labels,
                "seed": self.seed,
                "run_name": self.run_name,
                "saved_artifacts": [
                    "classifier_head.pt",
                    "tokenizer/",
                    "config/",
                ],
                "full_checkpoint_saved": self.keep_checkpoints,
            },
            self.artifact_dir / "metadata.json",
        )

    def cleanup_checkpoints(self) -> None:
        """Remove full Trainer checkpoints after the best model is loaded."""
        if self.checkpoint_dir.exists():
            shutil.rmtree(self.checkpoint_dir)
