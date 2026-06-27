"""Text preprocessing utilities for PhoBERT.

PhoBERT is normally fed VnCoreNLP word-segmented text. This module keeps
lowercasing disabled by default because casing can carry lexical information,
and the original UIT-VSFC text is already student-feedback style Vietnamese.
Experiments can opt into lowercasing with ``lowercase=True``.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import tempfile
import unicodedata
from pathlib import Path
from typing import Iterable

import pandas as pd
import py_vncorenlp


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VNCORENLP_DIR = Path(os.environ.get("VNCORENLP_DIR", REPO_ROOT / "vncorenlp"))
_SEGMENTER: py_vncorenlp.VnCoreNLP | None = None
_SEGMENTER_DIR: Path | None = None


def ensure_java_home() -> None:
    """Set JAVA_HOME from common local JDK locations when it is missing."""
    if os.environ.get("JAVA_HOME"):
        return

    candidates: list[Path] = []

    java_executable = shutil.which("java")
    if java_executable:
        java_path = Path(java_executable).resolve()
        if java_path.parent.name.lower() == "bin":
            candidates.append(java_path.parent.parent)

    if os.name == "nt":
        java_root = Path("C:/Program Files/Java")
        if java_root.exists():
            candidates.extend(sorted(java_root.glob("jdk*"), reverse=True))

    for candidate in candidates:
        java_binary = candidate / "bin" / ("java.exe" if os.name == "nt" else "java")
        if java_binary.exists():
            os.environ["JAVA_HOME"] = str(candidate)
            return


def runtime_model_dir(model_dir: Path) -> Path:
    """Return a VnCoreNLP path safe for Java on Windows.

    The upstream Java code can misread paths with spaces as URL-encoded paths.
    When the project lives under such a path, mirror the model directory to a
    no-space temp path and load VnCoreNLP from there.
    """
    if os.name != "nt" or " " not in str(model_dir):
        return model_dir

    runtime_dir = Path(tempfile.gettempdir()) / "vsfc_vncorenlp"
    shutil.copytree(model_dir, runtime_dir, dirs_exist_ok=True)
    return runtime_dir


def normalize_text(text: str) -> str:
    """Normalize Unicode to NFC and collapse surrounding/internal whitespace."""
    normalized = unicodedata.normalize("NFC", str(text))
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def get_segmenter(save_dir: str | Path | None = None) -> py_vncorenlp.VnCoreNLP:
    """Load the VnCoreNLP word segmenter once and reuse it."""
    global _SEGMENTER, _SEGMENTER_DIR

    model_dir = Path(save_dir) if save_dir else DEFAULT_VNCORENLP_DIR
    model_dir = model_dir.resolve()

    if _SEGMENTER is not None and _SEGMENTER_DIR == model_dir:
        return _SEGMENTER

    if not model_dir.exists():
        raise FileNotFoundError(
            f"VnCoreNLP model directory not found: {model_dir}. "
            "Run `python scripts/setup_vncorenlp.py` first."
        )

    ensure_java_home()
    load_dir = runtime_model_dir(model_dir)
    original_cwd = Path.cwd()

    try:
        _SEGMENTER = py_vncorenlp.VnCoreNLP(
            annotators=["wseg"],
            save_dir=str(load_dir),
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not load VnCoreNLP from {model_dir}. "
            "Make sure Java is installed and run `python scripts/setup_vncorenlp.py`."
        ) from exc
    finally:
        os.chdir(original_cwd)

    _SEGMENTER_DIR = model_dir
    return _SEGMENTER


def word_segment(text: str, save_dir: str | Path | None = None) -> str:
    """Word-segment text with VnCoreNLP and return a space-separated string."""
    segmenter = get_segmenter(save_dir=save_dir)
    segmented = segmenter.word_segment(text)

    if isinstance(segmented, list):
        return " ".join(sentence.strip() for sentence in segmented if sentence.strip())
    return str(segmented).strip()


def preprocess_text(
    text: str,
    lowercase: bool = False,
    segment: bool = True,
    save_dir: str | Path | None = None,
) -> str:
    """Normalize and optionally word-segment one sentence."""
    processed = normalize_text(text)
    if lowercase:
        processed = processed.lower()
    if segment and processed:
        processed = word_segment(processed, save_dir=save_dir)
    return processed


def preprocess_frame(
    frame: pd.DataFrame,
    text_col: str = "sentence",
    output_col: str = "sentence",
    lowercase: bool = False,
    segment: bool = True,
    save_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Preprocess a DataFrame text column."""
    output = frame.copy()
    output[output_col] = [
        preprocess_text(
            text,
            lowercase=lowercase,
            segment=segment,
            save_dir=save_dir,
        )
        for text in output[text_col].astype(str)
    ]
    return output


def preprocess_texts(
    texts: Iterable[str],
    lowercase: bool = False,
    segment: bool = True,
    save_dir: str | Path | None = None,
) -> list[str]:
    """Preprocess an iterable of texts."""
    return [
        preprocess_text(
            text,
            lowercase=lowercase,
            segment=segment,
            save_dir=save_dir,
        )
        for text in texts
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess Vietnamese text.")
    parser.add_argument("text", nargs="*", help="Text to preprocess.")
    parser.add_argument("--lowercase", action="store_true")
    parser.add_argument("--no-segment", action="store_true")
    parser.add_argument("--vncorenlp-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = args.text or [
        "Giảng viên dạy rất dễ hiểu.",
        "Cơ sở vật chất chưa tốt lắm.",
    ]
    for text in examples:
        print(
            preprocess_text(
                text,
                lowercase=args.lowercase,
                segment=not args.no_segment,
                save_dir=args.vncorenlp_dir,
            )
        )


if __name__ == "__main__":
    main()
