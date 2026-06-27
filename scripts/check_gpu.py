"""Print GPU and training-environment diagnostics."""

from __future__ import annotations

import importlib.util
import subprocess
import sys


def module_version(name: str) -> str:
    try:
        module = __import__(name)
    except Exception as exc:
        return f"not importable ({exc.__class__.__name__}: {exc})"
    return str(getattr(module, "__version__", "unknown"))


def command_output(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except FileNotFoundError:
        return "not found"
    except Exception as exc:
        return f"failed ({exc.__class__.__name__}: {exc})"

    output = (completed.stdout or completed.stderr).strip()
    if not output:
        output = f"exit_code={completed.returncode}"
    return output


def main() -> None:
    print(f"python: {sys.version.split()[0]}")
    print(f"torch: {module_version('torch')}")
    print(f"transformers: {module_version('transformers')}")
    print(f"accelerate installed: {importlib.util.find_spec('accelerate') is not None}")
    print()
    print("nvidia-smi:")
    print(command_output(["nvidia-smi"]))
    print()

    try:
        import torch
    except Exception as exc:
        print(f"torch import failed: {exc}")
        return

    print(f"torch.version.cuda: {torch.version.cuda}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            print(f"cuda:{index}: {torch.cuda.get_device_name(index)}")
        print(f"cudnn available: {torch.backends.cudnn.is_available()}")


if __name__ == "__main__":
    main()
