"""FastAPI dependency wiring (overridable in tests).

Note: services must receive their collaborators through Depends(...) so that
app.dependency_overrides works - calling get_manager() directly inside a
dependency body would bypass the override mechanism.
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import Depends

from backend.app.config import Settings, load_settings
from backend.app.services.inference import InferenceService
from backend.app.services.model_manager import ModelManager


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()


@lru_cache(maxsize=1)
def get_manager() -> ModelManager:
    return ModelManager(get_settings())


def get_inference_service(manager: ModelManager = Depends(get_manager)) -> InferenceService:
    return InferenceService(manager)
