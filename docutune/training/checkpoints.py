"""Checkpoint discovery and validation for interrupted training runs.

torch-free so it is unit-testable anywhere.

Why this exists: on ephemeral remote runtimes (Kaggle/Colab) a training
process can die mid-checkpoint-write, leaving a partial ``checkpoint-*``
directory. Hugging Face Trainer's built-in "resume from last checkpoint"
picks the highest-numbered ``checkpoint-*`` directory without checking
completeness, which can crash the resumed run. The trainer entry point
therefore uses :func:`find_latest_valid_checkpoint`, which selects the
highest-numbered checkpoint that actually contains everything needed to
resume optimizer/scheduler/model/Trainer state.

Checkpoint layout (produced by transformers Trainer with save_strategy=steps
and a PEFT model): checkpoint-<step>/{trainer_state.json, optimizer.pt,
scheduler.pt, adapter_model.safetensors, tokenizer files, rng_state.pth}.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

CHECKPOINT_DIR_RE = re.compile(r"^checkpoint-(\d+)$")

# Files required to resume full Trainer state (model/adapter, optimizer,
# scheduler, trainer state incl. step counter and log history).
REQUIRED_CHECKPOINT_FILES = (
    "trainer_state.json",
    "optimizer.pt",
    "scheduler.pt",
)
# Adapter weights: PEFT checkpoints store adapter_model.safetensors
# (safetensors is the only format in transformers 5.x); the model.* variants
# are accepted for full-model checkpoints.
CHECKPOINT_WEIGHT_FILES = (
    "adapter_model.safetensors",
    "adapter_model.bin",
    "model.safetensors",
    "pytorch_model.bin",
)


def validate_checkpoint_dir(path: str | Path) -> tuple[bool, list[str]]:
    """Check that a checkpoint directory is complete enough to resume from.

    Returns (is_valid, problems). A checkpoint is valid when it contains all
    required state files, at least one set of model/adapter weights, and a
    parseable trainer_state.json.
    """
    path = Path(path)
    problems: list[str] = []
    if not path.is_dir():
        return False, [f"not a directory: {path}"]
    for name in REQUIRED_CHECKPOINT_FILES:
        if not (path / name).is_file():
            problems.append(f"missing {name}")
    if not any((path / name).is_file() for name in CHECKPOINT_WEIGHT_FILES):
        problems.append(f"missing model/adapter weights (one of {CHECKPOINT_WEIGHT_FILES})")
    state_file = path / "trainer_state.json"
    if state_file.is_file():
        try:
            state = json.loads(state_file.read_text(encoding="utf-8"))
            if not isinstance(state, dict) or "global_step" not in state:
                problems.append("trainer_state.json missing global_step")
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
            problems.append(f"trainer_state.json unparseable ({exc})")
    return not problems, problems


def find_latest_valid_checkpoint(output_dir: str | Path) -> Path | None:
    """Return the highest-step VALID checkpoint under output_dir, else None.

    Invalid/incomplete checkpoint directories are skipped (not selected), so
    a crash mid-write can never poison a resume.
    """
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        return None
    candidates: list[tuple[int, Path]] = []
    for entry in output_dir.iterdir():
        match = CHECKPOINT_DIR_RE.match(entry.name)
        if entry.is_dir() and match:
            candidates.append((int(match.group(1)), entry))
    candidates.sort(reverse=True)
    for _step, path in candidates:
        valid, _problems = validate_checkpoint_dir(path)
        if valid:
            return path
    return None
