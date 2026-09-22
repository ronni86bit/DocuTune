"""Training schedule math: optimizer-step totals and warmup conversion.

Pure Python (no torch/transformers) so it is unit-testable anywhere.

Why this module exists (BUG 1 compatibility): Transformers 5.x removed
``TrainingArguments(warmup_ratio=...)``; the supported argument is
``warmup_steps``. The YAML deliberately keeps expressing warmup as a ratio
(``warmup_ratio: 0.05``) - the ratio is converted into an exact warmup step
count here, based on the actual number of optimizer/update steps, so the
intended schedule is preserved on any transformers version and any dataset
size / batch size / gradient-accumulation / epoch / max_steps combination.
"""

from __future__ import annotations

import math


def compute_total_update_steps(
    num_train_examples: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    num_epochs: float,
    override_max_steps: int = 0,
) -> int:
    """Total optimizer (update) steps the Trainer will run.

    Mirrors the Trainer/accelerate computation for the default
    drop_last=False dataloader:
        loader_steps_per_epoch   = ceil(N / batch_size)
        update_steps_per_epoch   = ceil(loader_steps / grad_accum)
        total                    = ceil(update_steps_per_epoch * epochs)

    ``override_max_steps > 0`` (e.g. the smoke test) wins over the epoch
    computation, exactly like ``TrainingArguments(max_steps=...)``.
    """
    if num_train_examples <= 0:
        raise ValueError(f"num_train_examples must be positive, got {num_train_examples}")
    if per_device_train_batch_size <= 0:
        raise ValueError("per_device_train_batch_size must be positive")
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")

    if override_max_steps > 0:
        return int(override_max_steps)
    if num_epochs <= 0:
        raise ValueError(f"num_epochs must be positive, got {num_epochs}")

    loader_steps_per_epoch = math.ceil(num_train_examples / per_device_train_batch_size)
    update_steps_per_epoch = math.ceil(loader_steps_per_epoch / gradient_accumulation_steps)
    return max(1, math.ceil(update_steps_per_epoch * num_epochs))


def compute_warmup_steps(warmup_ratio: float, total_steps: int) -> int:
    """Convert a warmup *ratio* into an absolute warmup step count.

    ``ceil(ratio * total_steps)`` guarantees at least the requested fraction
    of the run warms up, is clamped to ``[0, total_steps]``, and returns 0
    for non-positive ratios (warmup disabled).
    """
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    if warmup_ratio <= 0:
        return 0
    return min(total_steps, max(1, math.ceil(warmup_ratio * total_steps)))
