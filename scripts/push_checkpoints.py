#!/usr/bin/env python
"""Manually mirror local Trainer checkpoints to the private HF Hub repo.

Recovery utility for the case where an in-run checkpoint upload failed
(transient Hub/network error) and the checkpoint never made it to the
repository. Uses exactly the same validation + atomic-commit upload as the
trainer (`docutune/training/remote_persistence.py`); already-persisted steps
are skipped. Never deletes or modifies the local checkpoints.

Usage:
    export DOCUTUNE_HF_TOKEN=hf_...        # write-enabled token
    python scripts/push_checkpoints.py --repo-id <user>/docutune-phi3-adapter
    python scripts/push_checkpoints.py --repo-id <user>/docutune-phi3-adapter \
        --checkpoints-dir artifacts/training --steps 25,50
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docutune.config import TrainConfig, project_path  # noqa: E402
from docutune.training.checkpoints import CHECKPOINT_DIR_RE, validate_checkpoint_dir  # noqa: E402
from docutune.training.remote_persistence import (  # noqa: E402
    RemoteCheckpointStore,
    checkpoint_fingerprint,
)
from docutune.utils.logging import get_logger  # noqa: E402

logger = get_logger("push_checkpoints")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Upload local Trainer checkpoints to the HF Hub (manual recovery)")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--checkpoints-dir", default=None,
                        help="Directory containing checkpoint-<step>/ dirs "
                             "(default: the config's training output_dir)")
    parser.add_argument("--repo-id", required=True,
                        help="Target private HF repo, e.g. <user>/docutune-phi3-adapter")
    parser.add_argument("--steps", default=None,
                        help="Comma-separated checkpoint steps to push (default: all valid)")
    args = parser.parse_args()

    token = os.environ.get("DOCUTUNE_HF_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        logger.error("No token found. Export DOCUTUNE_HF_TOKEN (or HF_TOKEN) first.")
        return 1

    config = TrainConfig.from_yaml(args.config)
    checkpoints_dir = project_path(args.checkpoints_dir or config.training.output_dir)
    if not checkpoints_dir.is_dir():
        logger.error("Checkpoints directory not found: %s", checkpoints_dir)
        return 1

    steps: list[int] = []
    if args.steps:
        steps = [int(s) for s in args.steps.split(",")]
    else:
        for entry in sorted(checkpoints_dir.iterdir()):
            match = CHECKPOINT_DIR_RE.match(entry.name)
            if entry.is_dir() and match:
                steps.append(int(match.group(1)))
    if not steps:
        logger.error("No checkpoint-* directories found in %s", checkpoints_dir)
        return 1

    store = RemoteCheckpointStore(args.repo_id, token,
                                  fingerprint=checkpoint_fingerprint(config))
    failed = 0
    for step in sorted(steps):
        ckpt = checkpoints_dir / f"checkpoint-{step}"
        valid, problems = validate_checkpoint_dir(ckpt)
        if not valid:
            logger.error("checkpoint-%d is invalid (%s) - skipping", step, "; ".join(problems))
            failed += 1
            continue
        record = store.upload_checkpoint(ckpt, step)
        if record["upload_status"] == "succeeded" and record["verified"]:
            logger.info("checkpoint-%d: OK", step)
        elif record["upload_status"] == "skipped":
            logger.info("checkpoint-%d: %s", step, record["reason"])
        else:
            logger.error("checkpoint-%d FAILED: %s", step, record["reason"])
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
