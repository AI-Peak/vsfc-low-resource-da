"""Download and smoke-test the VnCoreNLP word segmenter."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys
import tempfile
from urllib.request import urlretrieve

import py_vncorenlp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


EXAMPLES = [
    "Giảng viên dạy rất dễ hiểu.",
    "Môn học này quá khó và bài tập hơi nhiều.",
    "Cơ sở vật chất chưa tốt lắm.",
    "Thầy cô hỗ trợ sinh viên nhiệt tình.",
    "Em không thích cách chấm điểm của môn này.",
]
VNCORENLP_BASE_URL = "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master"
REQUIRED_FILES = {
    "VnCoreNLP-1.2.jar": f"{VNCORENLP_BASE_URL}/VnCoreNLP-1.2.jar",
    "models/wordsegmenter/vi-vocab": f"{VNCORENLP_BASE_URL}/models/wordsegmenter/vi-vocab",
    "models/wordsegmenter/wordsegmenter.rdr": (
        f"{VNCORENLP_BASE_URL}/models/wordsegmenter/wordsegmenter.rdr"
    ),
    "models/postagger/vi-tagger": f"{VNCORENLP_BASE_URL}/models/postagger/vi-tagger",
    "models/ner/vi-500brownclusters.xz": (
        f"{VNCORENLP_BASE_URL}/models/ner/vi-500brownclusters.xz"
    ),
    "models/ner/vi-ner.xz": f"{VNCORENLP_BASE_URL}/models/ner/vi-ner.xz",
    "models/ner/vi-pretrainedembeddings.xz": (
        f"{VNCORENLP_BASE_URL}/models/ner/vi-pretrainedembeddings.xz"
    ),
    "models/dep/vi-dep.xz": f"{VNCORENLP_BASE_URL}/models/dep/vi-dep.xz",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download VnCoreNLP model files.")
    parser.add_argument("--save-dir", default="./vncorenlp")
    return parser.parse_args()


def _has_required_files(save_dir: Path) -> bool:
    return all((save_dir / relative_path).is_file() for relative_path in REQUIRED_FILES)


def _download_missing_files(save_dir: Path) -> None:
    for relative_path, url in REQUIRED_FILES.items():
        output_path = save_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists() and output_path.stat().st_size > 0:
            continue

        print(f"Downloading {relative_path}")
        urlretrieve(url, output_path)


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
            print(f"Using JAVA_HOME={candidate}")
            return


def runtime_model_dir(save_dir: Path) -> Path:
    """Return a VnCoreNLP path safe for Java on Windows."""
    if os.name != "nt" or " " not in str(save_dir):
        return save_dir

    runtime_dir = Path(tempfile.gettempdir()) / "vsfc_vncorenlp"
    shutil.copytree(save_dir, runtime_dir, dirs_exist_ok=True)
    return runtime_dir


def ensure_vncorenlp_files(save_dir: Path) -> None:
    """Download files, using py_vncorenlp first and a Windows-safe fallback."""
    if _has_required_files(save_dir):
        print(f"VnCoreNLP model folder already exists: {save_dir}")
        return

    try:
        py_vncorenlp.download_model(save_dir=str(save_dir))
    except Exception as exc:
        print(f"py_vncorenlp.download_model failed, using fallback: {exc}")

    if not _has_required_files(save_dir):
        _download_missing_files(save_dir)

    if not _has_required_files(save_dir):
        missing = [
            relative_path
            for relative_path in REQUIRED_FILES
            if not (save_dir / relative_path).is_file()
        ]
        raise RuntimeError(f"Missing VnCoreNLP files after download: {missing}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading VnCoreNLP model to: {save_dir}")
    ensure_vncorenlp_files(save_dir)

    print("\nSmoke-testing word segmentation")
    ensure_java_home()
    load_dir = runtime_model_dir(save_dir)
    original_cwd = Path.cwd()
    try:
        segmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=str(load_dir))
    finally:
        os.chdir(original_cwd)
    for sentence in EXAMPLES:
        segmented = segmenter.word_segment(sentence)
        if isinstance(segmented, list):
            segmented_text = " ".join(segmented)
        else:
            segmented_text = str(segmented)
        print(f"- {sentence}")
        print(f"  {segmented_text}")


if __name__ == "__main__":
    main()
