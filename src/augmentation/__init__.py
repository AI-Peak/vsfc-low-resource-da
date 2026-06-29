"""Data augmentation methods."""

from src.augmentation.base import TextAugmenter
from src.augmentation.eda import EDAAugmenter
from src.augmentation.llm_paraphrase import GeminiParaphraser

__all__ = ["EDAAugmenter", "GeminiParaphraser", "TextAugmenter"]
