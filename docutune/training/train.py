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


def _training_arguments(config: TrainConfig, output_dir: Path, bf16: bool):
    from transformers import TrainingArguments

    hp = config.training
    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=hp.epochs,
        learning_rate=hp.learning_rate,
        weight_decay=hp.weight_decay,
        warmup_ratio=hp.warmup_ratio,
        per_device_train_batch_size=hp.per_device_train_batch_size,
        gradient_accumulation_steps=hp.gradient_accumulation_steps,
        gradient_checkpointing=hp.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy=hp.eval_strategy,
        eval_steps=hp.eval_steps,
        save_strategy=hp.save_strategy,
        save_steps=hp.save_steps,
        save_total_limit=hp.save_total_limit,
        logging_steps=hp.logging_steps,
        lr_scheduler_type="linear",
        bf16=bf16,
        seed=hp.seed,
        data_seed=hp.seed,
        report_to=[],
        save_safetensors=True,
        load_best_model_at_end=False,
        remove_unused_columns=False,
    )


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
    model = attach_lora(model, config)  # 8-10 LoRA + target module verification

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
    training_args = _training_arguments(
        config, output_dir, bf16=(device == "cuda" and env.get("torch_version", "").startswith("2"))
    )
    if args.smoke_test:
        training_args.max_steps = 2
        training_args.eval_strategy = "no"
        training_args.save_strategy = "no"
        training_args.eval_steps = 500

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
        resume_from = args.checkpoint
        logger.info("Resuming from explicit checkpoint: %s", resume_from)
    elif args.resume:
        resume_from = True  # Trainer finds the latest checkpoint in output_dir
        logger.info("Resuming from the latest checkpoint in %s", output_dir)
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
        lora={**vars(config.lora), "target_modules": config.lora.target_modules},
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
        "target_modules": config.lora.target_modules,
        "quantization": {
            "enabled": config.quantization.enabled and device == "cuda",
            "bits": config.quantization.bits,
            "quant_type": config.quantization.quant_type,
            "use_double_quant": config.quantization.use_double_quant,
        },
        "smoke_test": bool(args.smoke_test),
        "device": device,
        **{k: v for k, v in env.items() if k != "device"},
        "eval_metrics": eval_metrics,
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started)),
        "end_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "duration_seconds": round(time.time() - started, 1),
    }
    write_json(project_path("artifacts/training/training_manifest.json"), manifest)
    write_json(final_adapter_dir / "training_manifest.json", manifest)
    logger.info("Training manifest written; total duration %.1fs", manifest["duration_seconds"])
    return final_adapter_dir


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
            "Everything else (imports, config, tokenizer, tokenization, masking) validated."
        )
        # Validate the LoRA target-module verification logic structurally,
        # using tiny real nn.Linear layers (no model weights involved).
        import torch.nn as nn

        from docutune.training.model import verify_target_modules

        class _StubModel:
            def named_modules(self):
                return iter([
                    ("model.layers.0.self_attn.q_proj", nn.Linear(8, 8)),
                    ("model.layers.0.self_attn.k_proj", nn.Linear(8, 8)),
                    ("model.layers.0.mlp.up_proj", nn.Linear(8, 8)),
                ])

        verified = verify_target_modules(_StubModel(), ["q_proj", "k_proj"])
        logger.info("Target-module verification logic OK (verified=%s on stub)", verified)
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
