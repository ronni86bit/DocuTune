"""Model loading for inference (base model and base + LoRA adapter).

torch/transformers/peft are imported lazily inside functions so that GPU-free
environments (tests, CPU-only API smoke checks) can import this module safely.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from docutune.config import DEFAULT_ADAPTER_PATH, DEFAULT_BASE_MODEL
from docutune.utils.logging import get_logger

logger = get_logger(__name__)

# Windows without Developer Mode: HF cache symlinks degrade gracefully; the
# warning is noise in every log line, so silence it once.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


class ModelLoadError(RuntimeError):
    """Raised when the base model cannot be loaded (clear user-facing cause)."""


class AdapterNotFoundError(FileNotFoundError):
    """Raised when the fine-tuned LoRA adapter is missing."""


@dataclass
class ModelBundle:
    model: Any
    tokenizer: Any
    device: str
    model_name: str
    model_revision: str | None = None
    adapter_path: str | None = None
    quantized: bool = False
    extras: dict[str, Any] = field(default_factory=dict)


def resolve_device(device: str = "auto") -> str:
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ModelLoadError("DEVICE=cuda requested but no CUDA device is available.")
    return device


def _resolve_compute_dtype(device: str):
    import torch

    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device == "cuda":
        return torch.float16
    return torch.float32


def _dtype_kwarg(torch_dtype) -> dict[str, Any]:
    """transformers v5 renamed torch_dtype -> dtype; support both lines."""
    import transformers

    major = int(transformers.__version__.split(".")[0])
    key = "dtype" if major >= 5 else "torch_dtype"
    return {key: torch_dtype}


def _load_tokenizer(model_name: str, revision: str | None):
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision or None)
    except Exception as exc:
        raise ModelLoadError(
            f"Failed to load tokenizer for '{model_name}' ({exc}). "
            "Check the model name and your network connection or HF_TOKEN."
        ) from exc
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _quantization_config_or_none(quantized: bool, device: str):
    """4-bit NF4 config; None when quantization is not possible (with warning)."""
    if not quantized:
        return None
    if device != "cuda":
        logger.warning(
            "Quantization requested but device is %s; loading in full precision instead.", device
        )
        return None
    try:
        from transformers import BitsAndBytesConfig

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=_resolve_compute_dtype(device),
        )
    except ImportError:
        logger.warning("bitsandbytes not installed; loading in full precision instead.")
        return None


def load_base_model(
    model_name: str = DEFAULT_BASE_MODEL,
    revision: str | None = None,
    device: str = "auto",
    quantized: bool = False,
) -> ModelBundle:
    """Load the untouched base instruction model."""
    from transformers import AutoModelForCausalLM

    resolved = resolve_device(device)
    logger.info(
        "Loading base model %s (revision=%s) on %s",
        model_name, revision or "default", resolved,
    )
    tokenizer = _load_tokenizer(model_name, revision)
    quant_config = _quantization_config_or_none(quantized, resolved)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision or None,
            quantization_config=quant_config,
            **(_dtype_kwarg(_resolve_compute_dtype(resolved)) if quant_config is None else {}),
            device_map="auto" if resolved == "cuda" else None,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        )
    except Exception as exc:
        raise ModelLoadError(
            f"Failed to load base model '{model_name}' ({exc}). "
            "Verify the model id, network access and available memory."
        ) from exc
    if resolved == "cpu":
        model.to("cpu")
    model.eval()
    logger.info("Base model loaded")
    return ModelBundle(
        model=model,
        tokenizer=tokenizer,
        device=resolved,
        model_name=model_name,
        model_revision=revision,
        quantized=quant_config is not None,
    )


def load_finetuned_model(
    adapter_path: str = DEFAULT_ADAPTER_PATH,
    model_name: str = DEFAULT_BASE_MODEL,
    revision: str | None = None,
    device: str = "auto",
    quantized: bool = False,
) -> ModelBundle:
    """Load base model + LoRA adapter (merged weights are NOT required)."""
    import os

    from peft import PeftModel

    from docutune.config import project_path

    resolved_adapter = project_path(adapter_path)
    if not os.path.isdir(resolved_adapter):
        raise AdapterNotFoundError(
            f"Fine-tuned adapter not found at '{resolved_adapter}'. "
            "Run the Colab training pipeline and configure ADAPTER_PATH."
        )
    bundle = load_base_model(model_name=model_name, revision=revision,
                             device=device, quantized=quantized)
    logger.info("Loading LoRA adapter from %s", resolved_adapter)
    try:
        bundle.model = PeftModel.from_pretrained(bundle.model, str(resolved_adapter))
    except Exception as exc:
        raise ModelLoadError(
            f"Failed to load LoRA adapter from '{resolved_adapter}' ({exc}). "
            "The adapter must match the configured base model."
        ) from exc
    bundle.model.eval()
    bundle.adapter_path = str(resolved_adapter)
    logger.info("Fine-tuned model ready (base + adapter)")
    return bundle
