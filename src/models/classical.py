"""Classical baseline models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


DEFAULT_CLASSICAL_CONFIG = {
    "ngram_range": (1, 2),
    "max_features": 20000,
    "min_df": 2,
    "class_weight": "balanced",
    "max_iter": 1000,
    "random_state": 42,
}


@dataclass
class PredictionOutput:
    """Predicted labels and class probabilities."""

    predictions: np.ndarray
    probabilities: np.ndarray


@dataclass
class ClassicalBaselineOutput:
    """Output bundle from the classical baseline training function."""

    model: Pipeline
    dev: PredictionOutput
    test: PredictionOutput


def _merged_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    merged = DEFAULT_CLASSICAL_CONFIG.copy()
    if config:
        merged.update(config)
    return merged


def build_tfidf_logreg(config: dict[str, Any] | None = None) -> Pipeline:
    """Build the TF-IDF + LogisticRegression pipeline."""
    cfg = _merged_config(config)
    vectorizer = TfidfVectorizer(
        ngram_range=tuple(cfg["ngram_range"]),
        max_features=int(cfg["max_features"]),
        min_df=int(cfg["min_df"]),
    )
    classifier = LogisticRegression(
        class_weight=cfg["class_weight"],
        max_iter=int(cfg["max_iter"]),
        random_state=int(cfg["random_state"]),
    )
    return Pipeline(
        steps=[
            ("tfidf", vectorizer),
            ("classifier", classifier),
        ]
    )


def predict_with_probabilities(model: Pipeline, frame: pd.DataFrame) -> PredictionOutput:
    """Predict labels and class probabilities for a DataFrame."""
    texts = frame["sentence"].astype(str)
    predictions = model.predict(texts)
    probabilities = model.predict_proba(texts)
    return PredictionOutput(predictions=predictions, probabilities=probabilities)


def train_tfidf_logreg(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> ClassicalBaselineOutput:
    """Train TF-IDF + LogisticRegression and predict on dev/test."""
    model = build_tfidf_logreg(config=config)
    model.fit(
        train_df["sentence"].astype(str),
        train_df["sentiment"].astype(int),
    )
    return ClassicalBaselineOutput(
        model=model,
        dev=predict_with_probabilities(model, dev_df),
        test=predict_with_probabilities(model, test_df),
    )
