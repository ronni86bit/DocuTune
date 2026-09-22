"""Model construction for QLoRA/LoRA training.

All heavy imports are function-local. Target modules are verified against the
actual architecture: training fails with an informative listing of available
Linear modules if a requested module does not exist.
"""

from __future__ import annotations

from typing import Any

from docutune.config import TrainConfig
from docutune.utils.logging import get_logger

logger = get_logger(__name__)


def list_linear_module_names(model: Any) -> list[str]:
    """Unique suffix names of all Linear submodules in the model."""
    import torch.nn as nn

    names = set()
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear,)):
            names.add(name)
    # Also collect the last path component (e.g. "q_proj") for suffix matching.
    for name in list(names):
        names.add(name.split(".")[-1])
    return sorted(names)


def verify_target_modules(model: Any, requested: list[str]) -> list[str]:
    """Return requested modules that exist in the model; raise otherwise.

    Matches either full submodule paths or final path components, and only
    counts modules that are Linear layers (valid LoRA targets).
    """
    available = set(list_linear_module_names(model))
    verified: list[str] = []
    missing: list[str] = []
    for target in requested:
        if target in available:
            verified.append(target)
        else:
            missing.append(target)
    if not verified:
        raise ValueError(
            "None of the requested LoRA target modules exist in this model. "
            f"Requested: {requested}. "
            f"Available Linear module names include: {sorted(available)[:40]}. "
            "Update lora.target_modules in configs/train.yaml for the selected model."
        )
    if missing:
        logger.warning("Requested LoRA target modules not found (skipped): %s", missing)
    return verified


def resolve_compute_dtype(device: str):
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


def build_quantization_config(cfg: TrainConfig, device: str):
    """BitsAndBytesConfig for QLoRA, or None when quantization cannot be used."""
    if not cfg.quantization.enabled or device != "cuda":
        if cfg.quantization.enabled:
            logger.warning(
                "Quantization is enabled in the config but no CUDA device is available; "
                "continuing WITHOUT quantization (fine on CPU/GPU-full-precision, but not QLoRA)."
            )
        return None
    import torch
    from transformers import BitsAndBytesConfig

    compute_dtype = resolve_compute_dtype(device)
    if cfg.quantization.compute_dtype not in (None, "auto"):
        compute_dtype = getattr(torch, cfg.quantization.compute_dtype)
    logger.info("Quantization: %s-bit %s, double_quant=%s, compute_dtype=%s",
                cfg.quantization.bits, cfg.quantization.quant_type,
                cfg.quantization.use_double_quant, compute_dtype)
    return BitsAndBytesConfig(
        load_in_4bit=cfg.quantization.bits == 4,
        bnb_4bit_quant_type=cfg.quantization.quant_type,
        bnb_4bit_use_double_quant=cfg.quantization.use_double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def load_tokenizer(model_name: str, revision: str | None = None):
    from transformers import AutoTokenizer

    logger.info("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision or None)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_training_model(cfg: TrainConfig, device: str) -> tuple[Any, Any]:
    """Load the (possibly quantized) causal LM for LoRA training."""
    from transformers import AutoModelForCausalLM

    quant_config = build_quantization_config(cfg, device)
    logger.info("Loading model %s (revision=%s)", cfg.model.name, cfg.model.revision or "default")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        revision=cfg.model.revision or None,
        quantization_config=quant_config,
        **({} if quant_config else _dtype_kwarg(resolve_compute_dtype(device))),
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=False,
    )
    if device == "cpu":
        model.to("cpu")
    if quant_config is not None:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    return model


def attach_lora(model: Any, cfg: TrainConfig) -> Any:
    """Verify target modules, attach LoRA, and report trainable parameters."""
    from peft import LoraConfig, get_peft_model

    verified = verify_target_modules(model, cfg.lora.target_modules)
    logger.info("LoRA target modules (verified): %s", verified)
    lora_config = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=verified,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model
