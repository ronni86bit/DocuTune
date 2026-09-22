"""The ONE canonical prompt, shared by training, baseline inference,
fine-tuned inference, benchmark and the API.

Do not duplicate or re-define prompt strings anywhere else. Import from here.
"""

from __future__ import annotations

import json
from typing import Any

from docutune.config import PROMPT_VERSION  # re-exported single source of truth

__all__ = ["PROMPT_VERSION", "SYSTEM_PROMPT", "build_user_message", "build_messages",
           "build_prompt_text", "serialize_target", "canonicalize_target",
           "TARGET_JSON_INSTRUCTION"]

PROMPT_VERSION = PROMPT_VERSION

SYSTEM_PROMPT = (
    "You are a resume information extraction model.\n"
    "Extract only information explicitly supported by the input resume.\n"
    "Return exactly one JSON object conforming to the required schema.\n"
    "Do not provide explanations.\n"
    "Do not invent missing information.\n"
    "Use null for missing scalar values.\n"
    "Use [] for missing list fields."
)

USER_INSTRUCTION = "Extract the structured information from the following resume:"

# The exact schema field contract, appended to the user message so both the
# base and the fine-tuned model see identical, explicit instructions.
TARGET_JSON_INSTRUCTION = (
    "The JSON object must have exactly these keys:\n"
    '{"name": str|null, "email": str|null, "phone": str|null, "location": str|null, '
    '"summary": str|null, "skills": [str],\n'
    ' "education": [{"degree": str|null, "institution": str|null, "field": str|null, '
    '"start_year": int|null, "end_year": int|null, "grade": str|null}],\n'
    ' "experience": [{"title": str|null, "company": str|null, "location": str|null, '
    '"start_date": "YYYY-MM"|"YYYY"|null, "end_date": "YYYY-MM"|"YYYY"|null, '
    '"current": bool, "responsibilities": [str]}],\n'
    ' "projects": [{"name": str|null, "technologies": [str], "description": str|null}],\n'
    ' "certifications": [{"name": str|null, "issuer": str|null, "year": int|null}]}\n'
    "Normalize experience dates to YYYY-MM (or YYYY when only the year is given). "
    "Use null for end_date when the role is current."
)


def build_user_message(resume_text: str) -> str:
    """The user turn: instruction + resume wrapped in RESUME tags."""
    return (
        f"{USER_INSTRUCTION}\n\n<RESUME>\n{resume_text.strip()}\n</RESUME>\n\n"
        f"{TARGET_JSON_INSTRUCTION}"
    )


def build_messages(resume_text: str) -> list[dict[str, str]]:
    """Chat messages consumed by tokenizer.apply_chat_template."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(resume_text)},
    ]


def build_prompt_text(tokenizer: Any, resume_text: str) -> str:
    """Render the full prompt through the model's chat template.

    Used identically at training time (prompt side) and inference time.
    """
    return tokenizer.apply_chat_template(
        build_messages(resume_text),
        tokenize=False,
        add_generation_prompt=True,
    )


def canonicalize_target(target: dict[str, Any]) -> dict[str, Any]:
    """Validate + canonicalize a target record through the Pydantic schema.

    Guarantees key order, types and defaults are identical everywhere the
    target JSON is serialized (training labels, gold comparison).
    """
    from docutune.schema.resume import ResumeExtraction

    return ResumeExtraction.model_validate(target).model_dump()


def serialize_target(target: dict[str, Any]) -> str:
    """Deterministic JSON string used as the training target."""
    return json.dumps(canonicalize_target(target), ensure_ascii=False, indent=2)
