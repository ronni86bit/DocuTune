"""Tokenization + label masking for instruction fine-tuning.

build_features is torch-free (plain lists) so it can be unit-tested without
installing torch. Completion-only loss: prompt tokens are masked to -100;
only the target JSON (+ EOS) contributes to the loss.

The TEST SPLIT IS NEVER REFERENCED ANYWHERE IN TRAINING. Only train.jsonl and
validation.jsonl are accepted here.
"""

from __future__ import annotations

from typing import Any

from docutune.inference.prompt import build_prompt_text, serialize_target

ALLOWED_TRAINING_FILES = ("train", "validation")


def build_features(tokenizer: Any, text: str, target: dict[str, Any],
                   max_length: int = 2048) -> dict[str, list[int]]:
    """Build input_ids / attention_mask / labels with prompt masking.

    Prompt and target are tokenized separately and concatenated, so the
    prompt/target boundary is exact regardless of tokenizer merging behavior.
    """
    prompt_text = build_prompt_text(tokenizer, text)
    eos = tokenizer.eos_token or ""
    target_text = serialize_target(target) + eos

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]

    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
        truncated = True
    else:
        truncated = False

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "truncated": truncated,
        "prompt_tokens": len(prompt_ids),
        "target_tokens": len(target_ids),
    }


class SFTDataset:
    """Protocol-based map-style dataset over a JSONL split.

    Implements __len__/__getitem__, which is the entire protocol
    torch DataLoader and Hugging Face Trainer require - there is no need to
    subclass torch.utils.data.Dataset (doing so would also drag a torch
    import into this module's class definition). torch is imported lazily in
    the constructor purely to validate availability early with a clear
    error. as_torch_dataset() returns self: it previously tried to
    INSTANTIATE torch.utils.data.Dataset(self.features), which fails with
    "TypeError: Dataset() takes no arguments" (regression-tested).
    """

    def __init__(self, jsonl_path: str, tokenizer: Any, max_length: int = 2048,
                 max_samples: int | None = None):
        import torch  # noqa: F401 - validates torch availability at construction

        from docutune.utils.io import read_jsonl
        from docutune.utils.logging import get_logger

        logger = get_logger(__name__)
        self.rows = read_jsonl(jsonl_path)
        if max_samples is not None:
            self.rows = self.rows[:max_samples]
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.features: list[dict[str, list[int]]] = []
        self.n_truncated = 0
        for row in self.rows:
            feats = build_features(tokenizer, row["text"], row["target"], max_length)
            self.n_truncated += int(feats.pop("truncated"))
            feats.pop("prompt_tokens")
            feats.pop("target_tokens")
            self.features.append(feats)
        if self.n_truncated:
            logger.warning("%d/%d examples exceeded max_length=%d and were truncated",
                           self.n_truncated, len(self.features), max_length)
        logger.info("Tokenized %d examples from %s", len(self.features), jsonl_path)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        return self.features[idx]

    def as_torch_dataset(self) -> SFTDataset:
        """Return self: SFTDataset already satisfies the Trainer protocol."""
        return self
