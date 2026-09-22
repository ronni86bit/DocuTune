"""Robust model-output parsing.

Accepts plain JSON, surrounding whitespace and accidental Markdown code
fences. No LLM-based repair, no semantic fixing - a malformed output stays
malformed (json_valid=False) and feeds the JSON-validity metric as a failure.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*\n?(.*?)```", re.DOTALL)


@dataclass
class ParsedOutput:
    raw_output: str
    parsed: Any = None
    json_valid: bool = False
    error: str | None = None
    matched_via: str | None = field(default=None)  # diagnostic: how JSON was located

    @property
    def is_object(self) -> bool:
        return isinstance(self.parsed, dict)


def _try_loads(text: str) -> tuple[Any | None, str | None]:
    try:
        return json.loads(text), None
    except (json.JSONDecodeError, ValueError) as exc:
        return None, str(exc)


def parse_model_output(raw_output: str) -> ParsedOutput:
    """Parse raw model text into a ParsedOutput.

    Strategy (strict, in order):
      1. direct json.loads of the stripped text
      2. each fenced code block (```json ... ```)
      3. substring from the first '{' through the last '}'
    Only syntactic recovery - never semantic repair.
    """
    raw = raw_output or ""
    stripped = raw.strip()

    if not stripped:
        return ParsedOutput(raw_output=raw, error="empty output")

    parsed, error = _try_loads(stripped)
    if parsed is not None:
        return ParsedOutput(raw_output=raw, parsed=parsed, json_valid=True, matched_via="direct")

    for block in _FENCE_RE.findall(raw):
        parsed, block_error = _try_loads(block.strip())
        if parsed is not None:
            return ParsedOutput(raw_output=raw, parsed=parsed, json_valid=True,
                                matched_via="code_fence")
        error = block_error

    start, end = stripped.find("{"), stripped.rfind("}")
    if start != -1 and end > start:
        parsed, brace_error = _try_loads(stripped[start : end + 1])
        if parsed is not None:
            return ParsedOutput(raw_output=raw, parsed=parsed, json_valid=True,
                                matched_via="brace_slice")
        error = brace_error

    return ParsedOutput(raw_output=raw, parsed=None, json_valid=False,
                        error=error or "unparseable output")
