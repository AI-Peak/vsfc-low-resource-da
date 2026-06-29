"""Base utilities for label-preserving text augmentation."""

from __future__ import annotations

from dataclasses import dataclass
import random
import re
from typing import Iterable

import pandas as pd

WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class AugmentedExample:
    """One augmented sentence with provenance."""

    source_index: int
    original_sentence: str
    augmented_sentence: str
    label: int
    method: str


class TextAugmenter:
    """Base class for deterministic DataFrame augmentation."""

    method_name = "base"

    def __init__(self, seed: int = 42) -> None:
        self.seed = int(seed)
        self.rng = random.Random(self.seed)

    @staticmethod
    def normalize_text(text: str) -> str:
        """Collapse whitespace and strip the augmented sentence."""
        return WHITESPACE_RE.sub(" ", str(text)).strip()

    def augment_sentence(self, sentence: str) -> str:
        """Return one augmented variant of a sentence."""
        raise NotImplementedError

    def augment_frame(
        self,
        frame: pd.DataFrame,
        num_aug: int = 1,
        text_col: str = "sentence",
        label_col: str = "sentiment",
        max_attempts_per_aug: int = 8,
    ) -> pd.DataFrame:
        """Augment every row in a DataFrame and return a traceable DataFrame."""
        if num_aug < 1:
            raise ValueError(f"num_aug must be >= 1, got {num_aug}")

        rows: list[AugmentedExample] = []
        for source_index, row in frame.reset_index(drop=True).iterrows():
            sentence = self.normalize_text(row[text_col])
            label = int(row[label_col])
            seen = {sentence}

            for _ in range(num_aug):
                augmented = sentence
                for _attempt in range(max_attempts_per_aug):
                    candidate = self.normalize_text(self.augment_sentence(sentence))
                    if candidate and candidate not in seen:
                        augmented = candidate
                        break
                seen.add(augmented)
                rows.append(
                    AugmentedExample(
                        source_index=int(source_index),
                        original_sentence=sentence,
                        augmented_sentence=augmented,
                        label=label,
                        method=self.method_name,
                    )
                )

        return pd.DataFrame([row.__dict__ for row in rows])


def deduplicate_augmented_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Drop empty and duplicate augmented examples while preserving order."""
    if rows.empty:
        return rows
    required = {"augmented_sentence", "label"}
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"Augmented rows missing columns: {sorted(missing)}")

    output = rows.copy()
    output["augmented_sentence"] = output["augmented_sentence"].map(TextAugmenter.normalize_text)
    output = output[output["augmented_sentence"].astype(bool)]
    output = output.drop_duplicates(subset=["augmented_sentence", "label"], keep="first")
    return output.reset_index(drop=True)


def label_counts(rows: pd.DataFrame, label_col: str = "label") -> dict[int, int]:
    """Return sorted label counts for concise CLI logging."""
    if rows.empty:
        return {}
    counts = rows[label_col].astype(int).value_counts().sort_index()
    return {int(label): int(count) for label, count in counts.items()}


def validate_augmented_schema(rows: pd.DataFrame, required: Iterable[str] | None = None) -> None:
    """Validate the schema consumed by the PhoBERT runner."""
    expected = set(required or ("augmented_sentence", "label"))
    missing = expected.difference(rows.columns)
    if missing:
        raise ValueError(f"Augmented DataFrame missing columns: {sorted(missing)}")
