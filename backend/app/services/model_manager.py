"""Application-level model manager.

Loads each model at most once (never per request), guarded by a lock, and
translates loader errors into clear, user-facing API errors:
- missing adapter  -> "Fine-tuned adapter not found. Run the Colab training
   pipeline and configure ADAPTER_PATH."
- base load failure -> clear message, HTTP 503 at the route layer.
"""

from __future__ import annotations

import threading

from backend.app.config import Settings
from docutune.inference.extractor import ExtractionOutcome, ResumeExtractor
from docutune.inference.loader import (
    AdapterNotFoundError,
    ModelLoadError,
    load_base_model,
    load_finetuned_model,
    resolve_device,
)
from docutune.utils.logging import get_logger

logger = get_logger("docutune.backend")

ADAPTER_MESSAGE = (
    "Fine-tuned adapter not found. Run the Colab training pipeline and configure ADAPTER_PATH."
)


class ModelUnavailableError(RuntimeError):
    """Raised when a requested model cannot be served (mapped to HTTP 503)."""

    def __init__(self, message: str):
        super().__init__(message)
        self.user_message = message


class ModelManager:
    """Caches base and fine-tuned extractors for the whole app lifetime."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = threading.Lock()
        self._base: ResumeExtractor | None = None
        self._finetuned: ResumeExtractor | None = None

    # -- availability ---------------------------------------------------
    def adapter_available(self) -> bool:
        """A directory without adapter_config.json is not a trained adapter."""
        return (self.settings.adapter_abs_path / "adapter_config.json").is_file()

    # -- loading --------------------------------------------------------
    def extractor_for(self, mode: str) -> ResumeExtractor:
        if mode not in ("base", "finetuned"):
            raise ValueError(f"unknown model mode: {mode}")
        cached = self._base if mode == "base" else self._finetuned
        if cached is not None:
            return cached
        with self._lock:
            cached = self._base if mode == "base" else self._finetuned
            if cached is not None:
                return cached
            try:
                if mode == "base":
                    logger.info("Loading base model %s", self.settings.base_model)
                    bundle = load_base_model(
                        model_name=self.settings.base_model,
                        revision=self.settings.model_revision,
                        device=self.settings.device,
                        quantized=self.settings.quantized,
                    )
                    extractor = ResumeExtractor(bundle, self.settings.max_new_tokens)
                    self._base = extractor
                else:
                    if not self.adapter_available():
                        raise ModelUnavailableError(ADAPTER_MESSAGE)
                    logger.info("Loading fine-tuned model (base + adapter %s)",
                                self.settings.adapter_abs_path)
                    bundle = load_finetuned_model(
                        adapter_path=str(self.settings.adapter_abs_path),
                        model_name=self.settings.base_model,
                        revision=self.settings.model_revision,
                        device=self.settings.device,
                        quantized=self.settings.quantized,
                    )
                    extractor = ResumeExtractor(bundle, self.settings.max_new_tokens)
                    self._finetuned = extractor
                return extractor
            except AdapterNotFoundError as exc:
                logger.error("Adapter unavailable: %s", exc)
                raise ModelUnavailableError(ADAPTER_MESSAGE) from exc
            except ModelLoadError as exc:
                logger.error("Model load failure: %s", exc)
                raise ModelUnavailableError(
                    f"Model could not be loaded: {exc}"
                ) from exc

    # -- inference ------------------------------------------------------
    def extract(self, text: str, mode: str) -> ExtractionOutcome:
        extractor = self.extractor_for(mode)
        logger.info("Extraction request received (mode=%s, chars=%d)", mode, len(text))
        outcome = extractor.extract(text)
        logger.info("Extraction completed (mode=%s, latency_ms=%.0f, json_valid=%s)",
                    mode, outcome.latency_ms, outcome.json_valid)
        return outcome

    def describe(self) -> dict:
        device = "unknown"
        try:
            device = resolve_device(self.settings.device)
        except Exception:  # noqa: BLE001 - describe() must never raise
            pass
        return {
            "base_model": self.settings.base_model,
            "model_revision": self.settings.model_revision,
            "default_model_mode": self.settings.model_mode,
            "adapter_path": str(self.settings.adapter_abs_path),
            "adapter_available": self.adapter_available(),
            "device": device,
            "quantized": self.settings.quantized,
        }
