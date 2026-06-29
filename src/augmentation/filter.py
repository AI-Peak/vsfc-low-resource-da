"""Quality-control filters for generated augmentation rows."""

from __future__ import annotations

from dataclasses import dataclass
import re

import pandas as pd

from src.augmentation.base import TextAugmenter
from src.augmentation.base import deduplicate_augmented_rows
from src.augmentation.base import validate_augmented_schema

TOKEN_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)

@dataclass(frozen=True)
class AugmentationFilterConfig:
    """Conservative heuristic thresholds for label-preserving paraphrases."""

    min_length_ratio: float = 0.45
    max_length_ratio: float = 2.20
    min_token_overlap: float = 0.20
    min_augmented_tokens: int = 2

def tokenize(text: str) -> list[str]:
    """Tokenize text for lightweight overlap checks."""
    normalized = TextAugmenter.normalize_text(text).lower()
    return TOKEN_RE.findall(normalized)

def token_overlap(original: str, augmented: str) -> float:
    """Return Jaccard token overlap between original and augmented text."""
    original_tokens = set(tokenize(original))
    augmented_tokens = set(tokenize(augmented))
    if not original_tokens or not augmented_tokens:
        return 0.0
    return len(original_tokens & augmented_tokens) / len(original_tokens | augmented_tokens)

def length_ratio(original: str, augmented: str) -> float:
    """Return augmented/original token length ratio."""
    original_len = max(len(tokenize(original)), 1)
    return len(tokenize(augmented)) / original_len

def rejection_reasons(
    original: str,
    augmented: str,
    label: int,
    config: AugmentationFilterConfig,
) -> list[str]:
    """List quality-control rejection reasons for one augmented row."""
    original_text = TextAugmenter.normalize_text(original)
    augmented_text = TextAugmenter.normalize_text(augmented)
    reasons: list[str] = []

    if not original_text or not augmented_text:
        reasons.append("empty_text")
    if original_text.lower() == augmented_text.lower():
        reasons.append("unchanged")
    if int(label) not in {0, 1, 2}:
        reasons.append("invalid_label")

    augmented_tokens = tokenize(augmented_text)
    ratio = length_ratio(original_text, augmented_text)
    overlap = token_overlap(original_text, augmented_text)

    if len(augmented_tokens) < config.min_augmented_tokens:
        reasons.append("too_few_tokens")
    if ratio < config.min_length_ratio:
        reasons.append("too_short_relative")
    if ratio > config.max_length_ratio:
        reasons.append("too_long_relative")
    if overlap < config.min_token_overlap:
        reasons.append("low_token_overlap")

    return reasons

def filter_augmented_frame(
    rows: pd.DataFrame,
    config: AugmentationFilterConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter augmented rows and return filtered rows plus diagnostics."""
    validate_augmented_schema(rows, required=("original_sentence", "augmented_sentence", "label"))
    resolved_config = config or AugmentationFilterConfig()

    diagnostics: list[dict[str, object]] = []
    keep_indices: list[int] = []
    for index, row in rows.reset_index(drop=True).iterrows():
        original = TextAugmenter.normalize_text(row["original_sentence"])
        augmented = TextAugmenter.normalize_text(row["augmented_sentence"])
        label = int(row["label"])
        reasons = rejection_reasons(original, augmented, label, resolved_config)
        keep = not reasons
        if keep:
            keep_indices.append(index)

        diagnostics.append(
            {
                "source_index": int(row.get("source_index", index)),
                "label": label,
                "keep": bool(keep),
                "reasons": ";".join(reasons) if reasons else "kept",
                "length_ratio": length_ratio(original, augmented),
                "token_overlap": token_overlap(original, augmented),
            }
        )

    filtered = rows.reset_index(drop=True).iloc[keep_indices].copy()
    filtered = deduplicate_augmented_rows(filtered)
    return filtered.reset_index(drop=True), pd.DataFrame(diagnostics)

def diagnostics_summary(diagnostics: pd.DataFrame) -> dict[str, int]:
    """Return compact keep/drop counts by reason."""
    if diagnostics.empty:
        return {"kept": 0, "dropped": 0}

    kept = int(diagnostics["keep"].sum())
    output: dict[str, int] = {
        "kept": kept,
        "dropped": int(len(diagnostics) - kept),
    }
    dropped = diagnostics[~diagnostics["keep"]]
    for reasons in dropped["reasons"].astype(str):
        for reason in reasons.split(";"):
            output[reason] = output.get(reason, 0) + 1
    return output
