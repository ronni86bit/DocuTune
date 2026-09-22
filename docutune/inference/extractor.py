"""Resume extraction via a loaded model bundle.

Deterministic decoding (greedy), identical prompt for base and fine-tuned
models. Latency covers tokenize + generate + detokenize; model loading and
output parsing are excluded and warm-up calls are handled by callers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from docutune.evaluation.parser import parse_model_output
from docutune.inference.loader import ModelBundle
from docutune.inference.prompt import build_prompt_text
from docutune.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ExtractionOutcome:
    raw_output: str
    parsed_output: Any
    json_valid: bool
    schema_valid: bool
    schema_error: str | None = None
    latency_ms: float = 0.0
    model_name: str = ""
    adapter_path: str | None = None
    prompt_version: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_output": self.raw_output,
            "parsed_output": self.parsed_output,
            "json_valid": self.json_valid,
            "schema_valid": self.schema_valid,
            "schema_error": self.schema_error,
            "latency_ms": self.latency_ms,
            "model_name": self.model_name,
            "adapter_path": self.adapter_path,
        }


class ResumeExtractor:
    """Deterministic text -> JSON extractor backed by a ModelBundle."""

    def __init__(self, bundle: ModelBundle, max_new_tokens: int = 700):
        self.bundle = bundle
        self.max_new_tokens = max_new_tokens

    def _generate(self, text: str) -> tuple[str, float]:
        import torch

        tokenizer = self.bundle.tokenizer
        prompt_text = build_prompt_text(tokenizer, text)
        started = time.perf_counter()
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True,
                           max_length=4096).to(self.bundle.model.device)
        with torch.no_grad():
            output_ids = self.bundle.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
        latency_ms = (time.perf_counter() - started) * 1000
        return raw, latency_ms

    def extract(self, text: str) -> ExtractionOutcome:
        from docutune.config import PROMPT_VERSION
        from docutune.schema.resume import ResumeExtraction, schema_error_message

        raw, latency_ms = self._generate(text)
        parsed = parse_model_output(raw)
        schema_valid = False
        schema_error = None
        if parsed.is_object:
            try:
                parsed.parsed = ResumeExtraction.model_validate(parsed.parsed).model_dump()
                schema_valid = True
            except Exception as exc:
                schema_error = schema_error_message(exc)
        return ExtractionOutcome(
            raw_output=raw,
            parsed_output=parsed.parsed if parsed.is_object else None,
            json_valid=parsed.json_valid,
            schema_valid=schema_valid,
            schema_error=schema_error,
            latency_ms=latency_ms,
            model_name=self.bundle.model_name,
            adapter_path=self.bundle.adapter_path,
            prompt_version=PROMPT_VERSION,
            error=parsed.error,
        )
