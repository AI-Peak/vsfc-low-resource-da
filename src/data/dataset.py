"""PyTorch dataset wrappers for PhoBERT fine-tuning."""

from __future__ import annotations

from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset


class VSFCDataset(Dataset):
    """Wrap a VSFC DataFrame for HuggingFace/PyTorch training."""

    def __init__(
        self,
        frame: pd.DataFrame,
        tokenizer: Any,
        max_length: int = 128,
        text_col: str = "sentence",
        label_col: str = "sentiment",
    ) -> None:
        self.texts = frame[text_col].astype(str).tolist()
        self.encodings = tokenizer(
            self.texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(frame[label_col].astype(int).tolist(), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {key: value[index] for key, value in self.encodings.items()}
        item["labels"] = self.labels[index]
        return item
