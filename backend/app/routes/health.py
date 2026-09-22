"""Health + metadata routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.app.config import Settings
from backend.app.dependencies import get_manager, get_settings
from backend.app.schemas import HealthResponse, MetadataResponse
from backend.app.services.model_manager import ModelManager
from docutune.config import (
    DATASET_VERSION,
    EVALUATOR_VERSION,
    PROMPT_VERSION,
    SCHEMA_VERSION,
)

router = APIRouter(tags=["health"])


@router.get("/api/health", response_model=HealthResponse)
def health(settings: Settings = Depends(get_settings),
           manager: ModelManager = Depends(get_manager)) -> HealthResponse:
    """Liveness probe. Deliberately never loads models (cheap for orchestration)."""
    return HealthResponse(
        status="ok",
        device=manager.describe()["device"],
        adapter_available=manager.adapter_available(),
        model_mode=settings.model_mode,
    )


@router.get("/api/metadata", response_model=MetadataResponse)
def metadata(manager: ModelManager = Depends(get_manager)) -> MetadataResponse:
    """Active model + version metadata. The active model is never hidden."""
    info = manager.describe()
    return MetadataResponse(
        base_model=info["base_model"],
        model_revision=info["model_revision"],
        default_model_mode=info["default_model_mode"],
        adapter_path=info["adapter_path"],
        adapter_available=info["adapter_available"],
        schema_version=SCHEMA_VERSION,
        prompt_version=PROMPT_VERSION,
        evaluator_version=EVALUATOR_VERSION,
        dataset_version=DATASET_VERSION,
        device=info["device"],
        quantized=info["quantized"],
        max_input_chars=manager.settings.max_input_chars,
    )
