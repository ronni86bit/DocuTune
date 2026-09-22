#!/usr/bin/env python
"""Optionally publish the trained LoRA adapter to the Hugging Face Hub.

Explicit and manual by design: requires --repo-id AND HF_TOKEN in the
environment. Nothing is uploaded automatically, and no secrets or private
data are included (the adapter folder contains only adapter weights/config).

Usage:
    export HF_TOKEN=hf_...
    python scripts/push_adapter.py --repo-id <user>/docutune-adapter
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docutune.config import EvalConfig, project_path  # noqa: E402
from docutune.utils.logging import get_logger  # noqa: E402

logger = get_logger("push_adapter")


def main() -> int:
    parser = argparse.ArgumentParser(description="Push the LoRA adapter to the HF Hub")
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--repo-id", required=True,
                        help="Target HF repo, e.g. yourname/docutune-phi3-adapter")
    parser.add_argument("--private", action="store_true", default=True)
    parser.add_argument("--public", dest="private", action="store_false")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        logger.error("HF_TOKEN is not set. Export your Hugging Face token first.")
        return 1

    cfg = EvalConfig.from_yaml(args.config)
    adapter_dir = project_path(args.adapter or cfg.adapter_path)
    if not Path(adapter_dir).is_dir():
        logger.error("Adapter not found at %s", adapter_dir)
        return 1

    try:
        from huggingface_hub import HfApi
    except ImportError:
        logger.error("huggingface_hub is not installed: pip install huggingface_hub")
        return 1

    api = HfApi(token=token)
    logger.info("Creating repo %s (private=%s)", args.repo_id, args.private)
    api.create_repo(repo_id=args.repo_id, private=args.private, exist_ok=True)
    logger.info("Uploading adapter from %s ...", adapter_dir)
    api.upload_folder(folder_path=str(adapter_dir), repo_id=args.repo_id,
                      repo_type="model")
    logger.info("Adapter published to https://huggingface.co/%s", args.repo_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
