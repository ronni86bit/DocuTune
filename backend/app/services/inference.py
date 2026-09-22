"""Inference service bridging routes and the model manager."""

from __future__ import annotations

from typing import Any

from backend.app.services.model_manager import ModelManager, ModelUnavailableError
from docutune.inference.extractor import ExtractionOutcome

__all__ = ["InferenceService", "ModelUnavailableError"]


class InferenceService:
    """Thin service used by routes; injectable/fake-able in tests."""

    def __init__(self, manager: ModelManager):
        self.manager = manager

    def extract(self, text: str, mode: str | None = None) -> ExtractionOutcome:
        resolved = mode or self.manager.settings.model_mode
        return self.manager.extract(text, resolved)

    def compare(self, text: str) -> dict[str, Any]:
        base = self.manager.extract(text, "base")
        try:
            finetuned: ExtractionOutcome | None = self.manager.extract(text, "finetuned")
            ft_error = None
        except ModelUnavailableError as exc:
            finetuned = None
            ft_error = exc.user_message
        return {"base": base, "finetuned": finetuned, "finetuned_error": ft_error}
