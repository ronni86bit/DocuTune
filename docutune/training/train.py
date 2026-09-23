"""LoRA / QLoRA fine-tuning entry point.

Usage:
    python -m docutune.training.train --config configs/train.yaml
    python -m docutune.training.train --config configs/train.yaml --resume
    python -m docutune.training.train --smoke-test

Strict separation of concerns:
- loads ONLY train.jsonl / validation.jsonl (the test split is never touched)
- saves adapter + tokenizer + resolved config + manifest + loss history
- supports checkpoint resume (--resume, --checkpoint)
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path
from typing import Any

from docutune.config import (
    DATASET_VERSION,
    PROMPT_VERSION,
    SCHEMA_VERSION,
    TrainConfig,
    project_path,
)
from docutune.seed import set_seeds
from docutune.training.checkpoints import (
    find_latest_valid_checkpoint,
    validate_checkpoint_dir,
)
from docutune.training.persistence import persist_adapter, resolve_upload_settings
from docutune.training.schedule import compute_total_update_steps, compute_warmup_steps
from docutune.utils.io import read_jsonl, write_json
from docutune.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DocuTune LoRA/QLoRA training")
    parser.add_argument("--config", default="configs/train.yaml", help="Training YAML config")
    parser.add_argument("--output-dir", default=None, help="Override training output dir")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the latest checkpoint in the output dir")
    parser.add_argument("--checkpoint", default=None,
                        help="Resume from a specific checkpoint path")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap training samples (debugging)")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Validate the full training setup on a tiny subset")
    return parser.parse_args(argv)


def software_versions() -> dict[str, str | None]:
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str | None] = {"python_version": platform.python_version()}
    for pkg, key in [
        ("torch", "torch_version"),
        ("transformers", "transformers_version"),
        ("peft", "peft_version"),
        ("bitsandbytes", "bitsandbytes_version"),
        ("accelerate", "accelerate_version"),
        ("datasets", "datasets_version"),
    ]:
        try:
            out[key] = version(pkg)
        except PackageNotFoundError:
            out[key] = None
    return out


def gpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {"cuda_version": None, "gpu_name": None, "gpu_memory_total_gb": None}
    try:
        import torch

        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["gpu_memory_total_gb"] = round(props.total_memory / 1024**3, 1)
    except ImportError:
        pass
    return info


def validate_environment(config: TrainConfig) -> dict[str, Any]:
    """Step 1: imports, CUDA and quantization feasibility."""
    try:
        import peft  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            f"Training requires torch, transformers and peft ({exc}). "
            "Install with: pip install -e \".[training]\""
        ) from exc
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if config.quantization.enabled and device != "cpu":
        try:
            import bitsandbytes  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Quantization (QLoRA) requires bitsandbytes. Install it or set "
                "quantization.enabled: false in the training config."
            ) from exc
    env = {"device": device, **gpu_info(), **software_versions()}
    logger.info("CUDA available: %s", torch.cuda.is_available())
    if env["gpu_name"]:
        logger.info("GPU detected: %s (%.1f GB)", env["gpu_name"], env["gpu_memory_total_gb"])
    return env


def build_training_arguments(
    config: TrainConfig,
    output_dir: Path,
    num_train_examples: int,
    bf16: bool = False,
    smoke_test: bool = False,
):
    """Construct TrainingArguments against the INSTALLED transformers version.

    Compatibility notes (transformers 5.x):
    - ``warmup_ratio`` was REMOVED. The YAML keeps ``warmup_ratio: 0.05``;
      it is converted here into an exact ``warmup_steps`` value based on the
      actual number of optimizer steps (docutune/training/schedule.py).
    - ``save_safetensors`` was REMOVED (safetensors is the only format now).
    """
    import inspect

    from transformers import TrainingArguments

    hp = config.training
    override_max_steps = 2 if smoke_test else 0
    total_steps = compute_total_update_steps(
        num_train_examples=num_train_examples,
        per_device_train_batch_size=hp.per_device_train_batch_size,
        gradient_accumulation_steps=hp.gradient_accumulation_steps,
        num_epochs=hp.epochs,
        override_max_steps=override_max_steps,
    )
    warmup_steps = compute_warmup_steps(hp.warmup_ratio, total_steps)
    logger.info(
        "Schedule: %d optimizer steps total, %d warmup steps (from warmup_ratio=%s)",
        total_steps, warmup_steps, hp.warmup_ratio,
    )

    params = inspect.signature(TrainingArguments.__init__).parameters
    if "warmup_steps" not in params:
        raise RuntimeError(
            "Installed transformers has no TrainingArguments(warmup_steps=...). "
            "Adapt docutune/training/schedule.py + build_training_arguments for "
            f"this version (transformers {software_versions().get('transformers_version')})."
        )

    kwargs: dict[str, Any] = dict(
        output_dir=str(output_dir),
        learning_rate=hp.learning_rate,
        weight_decay=hp.weight_decay,
        warmup_steps=warmup_steps,
        per_device_train_batch_size=hp.per_device_train_batch_size,
        gradient_accumulation_steps=hp.gradient_accumulation_steps,
        gradient_checkpointing=hp.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=hp.logging_steps,
        lr_scheduler_type="linear",
        bf16=bf16,
        seed=hp.seed,
        data_seed=hp.seed,
        report_to=[],
        load_best_model_at_end=False,
        remove_unused_columns=False,
    )
    if smoke_test:
        # Tiny run: fixed 2 steps, no periodic eval/checkpointing.
        kwargs["max_steps"] = total_steps
        kwargs["eval_strategy"] = "no"
        kwargs["save_strategy"] = "no"
    else:
        kwargs["num_train_epochs"] = hp.epochs
        kwargs["eval_strategy"] = hp.eval_strategy
        kwargs["eval_steps"] = hp.eval_steps
        kwargs["save_strategy"] = hp.save_strategy
        kwargs["save_steps"] = hp.save_steps
        kwargs["save_total_limit"] = hp.save_total_limit
    return TrainingArguments(**kwargs)


def run_training(config: TrainConfig, args: argparse.Namespace) -> Path:
    """The full training pipeline (steps 1-23 of the spec)."""
    from docutune.training.dataset import SFTDataset
    from docutune.training.model import attach_lora, load_tokenizer, load_training_model

    started = time.time()
    env = validate_environment(config)  # 1 validate environment
    device = env["device"]
    set_seeds(config.training.seed)  # 2 seeds

    output_dir = project_path(args.output_dir or config.training.output_dir)
    final_adapter_dir = project_path(config.training.final_adapter_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_adapter_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(config.model.name, config.model.revision)  # 3 tokenizer
    max_length = 512 if args.smoke_test else config.training.max_length

    if args.smoke_test:
        logger.info("SMOKE TEST: tiny subset, 2 steps only")

    model = load_training_model(config, device)  # 4/5/6/7 model + cuda + quantization
    if config.training.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    model, resolved_modules = attach_lora(model, config)  # 8-10 LoRA + target-module resolution

    train_file = project_path(config.data.train_file)
    val_file = project_path(config.data.validation_file)
    for path in (train_file, val_file):
        if not Path(path).is_file():
            raise FileNotFoundError(
                f"{path} not found. Generate the dataset first: python scripts/generate_data.py"
            )
    # 11-15 tokenization + labels + prompt masking (train/validation ONLY)
    train_dataset = SFTDataset(str(train_file), tokenizer, max_length,
                               max_samples=args.max_samples)
    val_dataset = SFTDataset(str(val_file), tokenizer, max_length,
                             max_samples=16 if args.smoke_test else None)
    logger.info("Train=%d Validation=%d (test split is NEVER loaded here)",
                len(train_dataset), len(val_dataset))

    from transformers import DataCollatorForSeq2Seq, Trainer

    collator = DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100)
    training_args = build_training_arguments(
        config,
        output_dir,
        num_train_examples=len(train_dataset),
        bf16=(device == "cuda" and env.get("torch_version", "").startswith("2")),
        smoke_test=args.smoke_test,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset.as_torch_dataset(),  # 13 tokenized
        eval_dataset=val_dataset.as_torch_dataset(),
        data_collator=collator,
        processing_class=tokenizer,
    )

    # 16 train (with checkpoint resume support)
    resume_from = None
    if args.checkpoint:
        explicit = Path(args.checkpoint)
        valid, problems = validate_checkpoint_dir(explicit)
        if not valid:
            raise RuntimeError(
                f"Explicit checkpoint {explicit} is not a valid resume point: "
                f"{problems}. Use --resume to auto-discover the latest valid "
                "checkpoint instead."
            )
        resume_from = str(explicit)
        logger.info("Resuming from explicit checkpoint: %s", resume_from)
    elif args.resume:
        latest = find_latest_valid_checkpoint(output_dir)
        if latest is not None:
            resume_from = str(latest)
            logger.info("Resuming from the latest valid checkpoint: %s", resume_from)
        else:
            resume_from = None
            logger.warning(
                "No valid checkpoint found in %s - training from scratch "
                "(incomplete checkpoints are ignored)", output_dir,
            )
    trainer.train(resume_from_checkpoint=resume_from)

    # 17 validation evaluation
    eval_metrics: dict[str, Any] = {}
    try:
        eval_metrics = trainer.evaluate()
        logger.info("Validation metrics: %s", eval_metrics)
    except Exception as exc:  # noqa: BLE001 - evaluation failure must not lose the adapter
        logger.warning("Final validation evaluation failed: %s", exc)

    # 18-20 save final adapter + tokenizer
    logger.info("Saving final adapter to %s", final_adapter_dir)
    trainer.save_model(str(final_adapter_dir))
    tokenizer.save_pretrained(str(final_adapter_dir))

    # 21-23 configuration, metrics, manifest
    config_copy = dict(
        model={"name": config.model.name, "revision": config.model.revision},
        quantization=vars(config.quantization),
        lora={
            **vars(config.lora),
            "target_modules_requested": config.lora.target_modules,
            "target_modules_resolved": resolved_modules,
        },
        training={**vars(config.training)},
        data=vars(config.data),
    )
    write_json(final_adapter_dir / "resolved_training_config.json", config_copy)

    loss_history = [
        {k: v for k, v in entry.items() if isinstance(v, (int, float, str))}
        for entry in trainer.state.log_history
    ]
    write_json(output_dir / "training_log.json", loss_history)

    manifest = {
        "model_name": config.model.name,
        "model_revision": config.model.revision,
        "schema_version": SCHEMA_VERSION,
        "dataset_version": DATASET_VERSION,
        "prompt_version": PROMPT_VERSION,
        "train_size": len(train_dataset),
        "validation_size": len(val_dataset),
        "test_size": "held out - not used by training",
        "seed": config.training.seed,
        "epochs": config.training.epochs,
        "learning_rate": config.training.learning_rate,
        "batch_size": config.training.per_device_train_batch_size,
        "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "max_length": max_length,
        "lora_r": config.lora.r,
        "lora_alpha": config.lora.alpha,
        "lora_dropout": config.lora.dropout,
        "target_modules_requested": config.lora.target_modules,
        "target_modules": resolved_modules,
        "quantization": {
            "enabled": config.quantization.enabled and device == "cuda",
            "bits": config.quantization.bits,
            "quant_type": config.quantization.quant_type,
            "use_double_quant": config.quantization.use_double_quant,
        },
        "smoke_test": bool(args.smoke_test),
        "resumed_from_checkpoint": resume_from,
        "device": device,
        **{k: v for k, v in env.items() if k != "device"},
        "eval_metrics": eval_metrics,
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started)),
        "end_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "duration_seconds": round(time.time() - started, 1),
    }

    # Write the manifest into the adapter directory BEFORE persistence so the
    # uploaded artifact always contains the reproduction metadata (the final
    # local copies below additionally record the upload status).
    write_json(final_adapter_dir / "training_manifest.json", manifest)

    # Optional persistence: push the verified adapter (+ manifest + resolved
    # config, which live in the same directory) to a private HF repo. Local
    # files are never modified by this step; failures are reported honestly.
    if args.smoke_test:
        upload_result = {"status": "skipped (smoke test)", "repo_id": None,
                         "url": None, "reason": None}
        logger.info("Smoke test - adapter upload skipped")
    else:
        settings = resolve_upload_settings()
        if not settings.enabled:
            logger.info(
                "Adapter upload DISABLED (%s). The adapter is saved locally at %s "
                "- copy it off the runtime manually, or set DOCUTUNE_HF_REPO_ID "
                "(+ DOCUTUNE_HF_TOKEN / HF_TOKEN) to enable automatic upload.",
                settings.reason, final_adapter_dir,
            )
        else:
            logger.info("Adapter upload ENABLED -> private repo %s", settings.repo_id)
        upload_result = persist_adapter(final_adapter_dir, settings)
        if upload_result["status"] == "succeeded":
            logger.info("Adapter upload SUCCEEDED: %s", upload_result["url"])
        elif upload_result["status"] == "failed":
            logger.error("Adapter upload FAILED: %s", upload_result["reason"])
    manifest["artifact_upload"] = upload_result

    write_json(project_path("artifacts/training/training_manifest.json"), manifest)
    write_json(final_adapter_dir / "training_manifest.json", manifest)
    logger.info("Training manifest written; total duration %.1fs", manifest["duration_seconds"])
    return final_adapter_dir


def _resolve_against_real_phi3(config: TrainConfig) -> list[str]:
    """Resolve configured target modules against a real Phi3Attention module.

    Builds the actual `Phi3Attention` layer from the installed transformers
    (random-initialized, no weights download) so the resolution is verified
    against the true architecture naming (fused qkv_proj + o_proj), not a
    stub. Falls back to a hand-built fused-attention stub only if the local
    transformers does not ship the phi3 module.
    """
    import torch.nn as nn

    from docutune.training.model import resolve_lora_target_modules

    try:
        from transformers.models.phi3 import modeling_phi3

        attention = modeling_phi3.Phi3Attention(
            modeling_phi3.Phi3Config(hidden_size=32, num_attention_heads=4,
                                     num_key_value_heads=2),
            layer_idx=0,
        )
        source = f"real Phi3Attention from transformers {modeling_phi3.__name__}"
    except Exception as exc:  # noqa: BLE001 - fall back to an equivalent stub
        logger.warning("Could not build Phi3Attention (%s); using an equivalent stub", exc)

        class _Phi3LikeAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.qkv_proj = nn.Linear(32, 96)
                self.o_proj = nn.Linear(32, 32)

        attention = _Phi3LikeAttention()
        source = "equivalent fused-attention stub"

    resolved = resolve_lora_target_modules(attention, config.lora.target_modules)
    logger.info("Resolved against %s -> %s", source, resolved)
    if "qkv_proj" not in resolved or "o_proj" not in resolved:
        raise RuntimeError(
            f"Phi-3 target resolution unexpected: {resolved} "
            "(expected the fused qkv_proj and o_proj to be adapted)"
        )
    return resolved


def run_smoke_test(config: TrainConfig, args: argparse.Namespace) -> int:
    """GPU-free-friendly validation of the whole training stack."""
    logger.info("=== DocuTune training smoke test ===")
    env = validate_environment(config)
    logger.info("Imports OK (torch/transformers/peft). Device: %s", env["device"])
    set_seeds(config.training.seed)

    from docutune.training.model import (
        load_tokenizer,
    )

    tokenizer = load_tokenizer(config.model.name, config.model.revision)
    logger.info("Tokenizer OK: vocab=%s, eos=%r", tokenizer.vocab_size, tokenizer.eos_token)

    from docutune.training.dataset import build_features

    train_rows = read_jsonl(project_path(config.data.train_file))[:4]
    if not train_rows:
        raise RuntimeError("No training rows found - run scripts/generate_data.py first")
    feats = [build_features(tokenizer, r["text"], r["target"], max_length=1024) for r in train_rows]
    n_masked = sum(1 for f in feats if all(v == -100 for v in f["labels"]))
    if n_masked:
        raise RuntimeError("Label masking is broken: an example has no supervised tokens")
    logger.info("Tokenization + prompt masking OK (example lengths: %s)",
                [len(f["input_ids"]) for f in feats])

    if env["device"] == "cuda":
        import copy

        logger.info("CUDA available - constructing the model and running 2 real steps")
        smoke_cfg = copy.deepcopy(config)
        smoke_cfg.training.output_dir = "artifacts/training/smoke"
        args.max_samples = 4
        args.output_dir = "artifacts/training/smoke"
        final_dir = run_training(smoke_cfg, args)
        logger.info("Smoke training completed; adapter at %s", final_dir)
    else:
        logger.info(
            "No CUDA device - skipping model construction and training steps. "
            "Everything else (imports, config, tokenizer, tokenization, masking, "
            "target-module resolution) validated."
        )
        # Resolve the configured target modules against the REAL Phi-3
        # attention module from the installed transformers (tiny random
        # weights, no download - the class constructor builds its Linears).

        resolved = _resolve_against_real_phi3(config)
        logger.info("Phi-3 target-module resolution OK (resolved=%s)", resolved)
        logger.info("SMOKE TEST PASSED (CPU mode; real training steps require a GPU)")

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "ok").write_text("smoke", encoding="utf-8")
    logger.info("Temporary output handling OK")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = TrainConfig.from_yaml(args.config)
    if args.output_dir:
        config.training.output_dir = args.output_dir
    logger.info(
        "DocuTune training | model=%s | lora r=%d alpha=%d | epochs=%s | smoke_test=%s",
        config.model.name, config.lora.r, config.lora.alpha, config.training.epochs,
        args.smoke_test,
    )
    if args.smoke_test:
        try:
            return run_smoke_test(config, args)
        except Exception as exc:  # noqa: BLE001 - report cleanly, exit non-zero
            logger.error("SMOKE TEST FAILED: %s: %s", type(exc).__name__, exc)
            return 1
    final_dir = run_training(config, args)
    logger.info("Done. Final adapter: %s", final_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
