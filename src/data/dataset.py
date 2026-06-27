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
        self.labels = frame[label_col].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[index],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item
