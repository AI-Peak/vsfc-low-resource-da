"""Prompt templates for LLM-based augmentation."""

from __future__ import annotations

LABEL_NAMES = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


def label_name(label: int) -> str:
    """Return a readable label name for prompts."""
    return LABEL_NAMES.get(int(label), str(label))


def build_paraphrase_prompt(sentence: str, label: int, num_aug: int = 1) -> str:
    """Build a strict label-preserving Vietnamese paraphrase prompt."""
    return f"""You are creating data augmentation for Vietnamese student feedback sentiment classification.

The sentiment label must stay unchanged: {label_name(label)}.

Original Vietnamese sentence:
\"\"\"{sentence}\"\"\"

Requirements:
- Write {num_aug} natural Vietnamese paraphrase(s).
- Preserve the main meaning and the sentiment label.
- Do not add new information.
- Do not make the sentiment noticeably stronger or weaker.
- Do not switch to another sentiment.
- Do not explain.
- Return only a JSON array of strings, for example: ["mot cau paraphrase"].
"""
