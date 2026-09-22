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


def resolve_lora_target_modules(model: Any, requested: list[str]) -> list[str]:
    """Resolve requested LoRA target modules against the ACTUAL architecture.

    Resolution rules (in order):
      1. Exact match against the model's Linear submodules (matched by name or
         by final path component, e.g. "o_proj" ~ "...self_attn.o_proj").
      2. Fused-QKV aliasing - architectures fuse the three attention input
         projections into one Linear, others keep them separate:
           - request {q_proj, k_proj, v_proj} + model has ``qkv_proj``
             (e.g. Phi-3) -> adapt ``qkv_proj``
           - request ``qkv_proj`` + model has separate q/k/v -> adapt all three
         Only a complete trio is aliased, so a partially-specified request
         still fails loudly instead of guessing.

    Fails LOUDLY (ValueError) when:
      - any requested module resolves neither exactly nor via alias
        (no silent skipping - a skipped attention projection silently
        shrinks the adaptation and invalidates the experiment);
      - the resolved set is empty.

    Returns the sorted list of module names that will actually be adapted;
    this is what gets logged and recorded in the training manifest.
    """
    available = set(list_linear_module_names(model))
    requested = list(dict.fromkeys(requested))
    resolved = {name for name in requested if name in available}
    unmatched = [name for name in requested if name not in resolved]

    qkv_trio = ("q_proj", "k_proj", "v_proj")
    # Phi-3 style: q/k/v requested, fused qkv_proj present.
    trio_requested = set(qkv_trio).issubset(requested)
    if set(unmatched) & set(qkv_trio) and "qkv_proj" in available and trio_requested:
        resolved.add("qkv_proj")
        unmatched = [name for name in unmatched if name not in qkv_trio]
        logger.info("Fused attention detected: q/k/v requests mapped to the fused 'qkv_proj' layer")
    # Llama-style: fused qkv_proj requested, separate projections present.
    if "qkv_proj" in unmatched:
        separate = [name for name in qkv_trio if name in available]
        if separate:
            resolved.update(separate)
            unmatched.remove("qkv_proj")
            logger.info(
                "Separate attention projections detected: fused request mapped to %s", separate
            )

    if unmatched:
        raise ValueError(
            f"Requested LoRA target modules do not exist in this model and no "
            f"architecture alias applies. Requested: {requested}. Unresolved: {unmatched}. "
            f"Available Linear module names include: {sorted(available)[:40]}. "
            f"Update lora.target_modules in configs/train.yaml for the selected model "
            f"(never leave attention projections silently unadapted)."
        )
    if not resolved:
        raise ValueError(
            "LoRA target-module resolution produced an EMPTY set - refusing to "
            "train an adapter that adapts nothing. Requested: "
            f"{requested}. Available Linear module names include: {sorted(available)[:40]}."
        )
    ordered = sorted(resolved)
    logger.info("LoRA target modules (actually adapted): %s", ordered)
    return ordered


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


def attach_lora(model: Any, cfg: TrainConfig) -> tuple[Any, list[str]]:
    """Resolve target modules against the real architecture, attach LoRA.

    Returns (peft_model, resolved_modules) - the resolved list is what the
    caller must record in the training manifest (exactly which projections
    are adapted, e.g. qkv_proj + o_proj on Phi-3).
    """
    from peft import LoraConfig, get_peft_model

    resolved = resolve_lora_target_modules(model, cfg.lora.target_modules)
    lora_config = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=resolved,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, resolved
