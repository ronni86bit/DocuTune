"""Extraction routes: /api/extract and /api/compare."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from backend.app.dependencies import get_inference_service
from backend.app.schemas import CompareResult, ExtractionResult, ExtractRequest
from backend.app.services.inference import InferenceService, ModelUnavailableError

router = APIRouter(tags=["extraction"])


def _to_result(outcome, mode: str) -> ExtractionResult:
    return ExtractionResult(
        model=mode,
        raw_output=outcome.raw_output,
        parsed_output=outcome.parsed_output,
        json_valid=outcome.json_valid,
        schema_valid=outcome.schema_valid,
        latency_ms=round(outcome.latency_ms, 1),
        error=outcome.error,
        model_name=outcome.model_name,
        adapter_path=outcome.adapter_path,
    )


@router.post("/api/extract", response_model=ExtractionResult)
def extract(request: ExtractRequest,
            service: InferenceService = Depends(get_inference_service)) -> ExtractionResult:
    """Extract structured JSON from resume text with the selected model."""
    try:
        outcome = service.extract(request.text, request.model)
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=exc.user_message) from exc
    return _to_result(outcome, request.model or service.manager.settings.model_mode)


@router.post("/api/compare", response_model=CompareResult)
def compare(request: ExtractRequest,
            service: InferenceService = Depends(get_inference_service)) -> CompareResult:
    """Run BOTH models on the same input with the same settings."""
    try:
        result = service.compare(request.text)
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=exc.user_message) from exc
    return CompareResult(
        input=request.text,
        base=_to_result(result["base"], "base"),
        finetuned=_to_result(result["finetuned"], "finetuned") if result["finetuned"] else None,
        finetuned_error=result["finetuned_error"],
    )
