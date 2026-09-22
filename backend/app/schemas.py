"""FastAPI request/response schemas (clean OpenAPI documentation)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

MAX_INPUT_CHARS_DEFAULT = 20000


class ExtractRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_INPUT_CHARS_DEFAULT,
                      description="Unstructured resume text")
    model: Literal["base", "finetuned"] | None = Field(
        default=None, description="Override the server's default model mode")

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be empty or whitespace only")
        return value


class ExtractionResult(BaseModel):
    model: str
    raw_output: str = ""
    parsed_output: dict[str, Any] | None = None
    json_valid: bool = False
    schema_valid: bool = False
    latency_ms: float = 0.0
    error: str | None = None
    model_name: str | None = None
    adapter_path: str | None = None


class CompareResult(BaseModel):
    input: str
    base: ExtractionResult
    finetuned: ExtractionResult | None = None
    finetuned_error: str | None = None


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "docutune-backend"
    device: str = "unknown"
    adapter_available: bool = False
    model_mode: str


class MetadataResponse(BaseModel):
    base_model: str
    model_revision: str | None = None
    default_model_mode: str
    adapter_path: str
    adapter_available: bool
    schema_version: str
    prompt_version: str
    evaluator_version: str
    dataset_version: str
    device: str
    quantized: bool
    max_input_chars: int


class BenchmarkUnavailable(BaseModel):
    available: bool = False
    message: str
