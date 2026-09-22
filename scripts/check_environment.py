#!/usr/bin/env python
"""Report the current execution environment (Python, torch, CUDA, GPU).

Never fabricates GPU information - absent components are reported as absent.

Usage:
    python scripts/check_environment.py
"""

import argparse
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the DocuTune environment")
    parser.add_argument("--json", action="store_true", help="Machine-readable output")
    args = parser.parse_args()

    info: dict = {
        "os": platform.platform(),
        "python_version": platform.python_version(),
    }
    for pkg in ("torch", "transformers", "peft", "bitsandbytes", "accelerate", "fastapi"):
        try:
            info[pkg] = version(pkg)
        except PackageNotFoundError:
            info[pkg] = None

    cuda = {"available": False, "version": None, "gpu_name": None, "gpu_memory_gb": None}
    try:
        import torch

        cuda["available"] = bool(torch.cuda.is_available())
        if cuda["available"]:
            cuda["version"] = torch.version.cuda
            cuda["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            cuda["gpu_memory_gb"] = round(props.total_memory / 1024**3, 1)
    except ImportError:
        pass
    info["cuda"] = cuda

    if args.json:
        import json

        print(json.dumps(info, indent=2))
    else:
        print("DocuTune environment check")
        print("--------------------------")
        print(f"OS               : {info['os']}")
        print(f"Python           : {info['python_version']}")
        for pkg in ("torch", "transformers", "peft", "bitsandbytes", "accelerate", "fastapi"):
            print(f"{pkg:<16}: {info[pkg] or 'NOT INSTALLED'}")
        print(f"CUDA available   : {cuda['available']}")
        if cuda["available"]:
            print(f"CUDA version     : {cuda['version']}")
            print(f"GPU              : {cuda['gpu_name']} ({cuda['gpu_memory_gb']} GB)")
        print()
        if cuda["available"] and cuda["gpu_memory_gb"] and cuda["gpu_memory_gb"] >= 14:
            print("Verdict: suitable for QLoRA fine-tuning of the configured 3-4B model.")
        elif cuda["available"]:
            print("Verdict: GPU present but may be tight for QLoRA of a 3-4B model; "
                  "reduce max_length or batch size, or use Google Colab (T4/L4).")
        else:
            print("Verdict: NO GPU detected. Fine-tuning must run on Google Colab or another "
                  "CUDA machine. CPU inference of the 3-4B model works but is slow.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
