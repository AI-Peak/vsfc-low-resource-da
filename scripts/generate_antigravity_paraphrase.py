"""Generate a local paraphrase file for Phase 5 pilot runs."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_eda import load_train_split
from scripts.generate_llm_paraphrase import FIELDNAMES
from scripts.generate_llm_paraphrase import finalize_file
from scripts.generate_llm_paraphrase import output_path
from src.augmentation.base import TextAugmenter
from src.data.subsample import format_ratio

REPLACEMENTS = (
    ("giảng dạy", "truyền đạt"),
    ("dễ hiểu", "rõ ràng"),
    ("khó hiểu", "chưa rõ"),
    ("tận tình", "nhiệt tình"),
    ("nhiệt tình", "tận tâm"),
    ("vui tính", "thân thiện"),
    ("quan tâm", "chú ý hỗ trợ"),
    ("bài tập", "bài luyện tập"),
    ("kiến thức", "nội dung kiến thức"),
    ("môn học", "học phần"),
    ("sinh viên", "người học"),
    ("học sinh", "sinh viên"),
    ("phòng thực hành", "phòng lab"),
    ("máy tính", "máy"),
    ("wifi", "mạng wifi"),
    ("nên", "cần"),
    ("cần", "nên"),
    ("chưa", "vẫn chưa"),
    ("rất", "khá"),
)

PREFIXES = (
    "theo em , ",
    "nhìn chung , ",
    "về cơ bản , ",
    "có thể nói là , ",
)

TRAILING_PUNCT_RE = re.compile(r"\s+([,.!?;:])")

def normalize_feedback(text: str) -> str:
    """Normalize feedback text while preserving the repository's token spacing."""
    text = TextAugmenter.normalize_text(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def replace_first_phrase(text: str, source_index: int) -> str:
    """Apply one conservative phrase replacement when available."""
    start = source_index % len(REPLACEMENTS)
    ordered = REPLACEMENTS[start:] + REPLACEMENTS[:start]
    for old, new in ordered:
        if old in text:
            return text.replace(old, new, 1)
    return text

def add_discourse_frame(text: str, source_index: int) -> str:
    """Add a light discourse frame that keeps the sentiment unchanged."""
    prefix = PREFIXES[source_index % len(PREFIXES)]
    if text.startswith(prefix):
        return text
    return prefix + text

def clean_output(text: str) -> str:
    """Clean spacing and ensure the paraphrase ends with punctuation."""
    text = TextAugmenter.normalize_text(text)
    text = TRAILING_PUNCT_RE.sub(r" \1", text)
    if not text.endswith((".", "!", "?")):
        text = f"{text} ."
    return text

def paraphrase(sentence: str, source_index: int) -> str:
    """Create one label-preserving offline paraphrase."""
    original = normalize_feedback(sentence)
    candidate = replace_first_phrase(original, source_index)
    if candidate == original:
        candidate = add_discourse_frame(original, source_index)
    elif source_index % 3 == 0:
        candidate = add_discourse_frame(candidate, source_index)

    candidate = clean_output(candidate)
    if candidate == clean_output(original):
        candidate = clean_output(add_discourse_frame(original, source_index + 1))
    return candidate

def generate_file(
    ratio: float,
    seed: int,
    data_dir: str | Path,
    output_dir: str | Path | None,
    force: bool,
) -> Path:
    """Generate the raw LLM-compatible CSV file without external API calls."""
    path = output_path(data_dir=data_dir, output_dir=output_dir, ratio=ratio, seed=seed)
    if path.exists() and not force:
        print(f"Skipping existing file: {path}", flush=True)
        return path

    train_df = load_train_split(data_dir=data_dir, ratio=ratio, seed=seed).reset_index(drop=True)
    rows = []
    for source_index, row in train_df.iterrows():
        sentence = TextAugmenter.normalize_text(row["sentence"])
        rows.append(
            {
                "source_index": int(source_index),
                "original_sentence": sentence,
                "augmented_sentence": paraphrase(sentence, int(source_index)),
                "label": int(row["sentiment"]),
                "method": "paraphrase_local",
                "ratio": format_ratio(ratio),
                "seed": int(seed),
                "model": "local_backup",
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=FIELDNAMES).to_csv(path, index=False)
    finalize_file(path)
    return path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate local paraphrases.")
    parser.add_argument("--ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    path = generate_file(
        ratio=args.ratio,
        seed=args.seed,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        force=args.force,
    )
    print(f"Saved local paraphrase file: {path}", flush=True)

if __name__ == "__main__":
    main()
