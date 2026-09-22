"""Central configuration for DocuTune.

Versioning constants, the active base model, and portable path resolution
live here. This module must stay importable without torch/transformers so
that GPU-free tests and the API can import it freely.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Versions (single source of truth)
# ---------------------------------------------------------------------------
SCHEMA_VERSION = "1.0"
DATASET_VERSION = "1.0"
PROMPT_VERSION = "1.0"
EVALUATOR_VERSION = "1.0"
# Bumped only when the adapter format/semantics change.
ADAPTER_VERSION = "1.0"

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
DEFAULT_BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
# Documented fallback configurations. The active model is NEVER switched
# automatically; these exist only as documented alternatives (see README).
FALLBACK_MODELS = [
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
]

DEFAULT_ADAPTER_PATH = "artifacts/adapters/final"
DEFAULT_TRAINING_OUTPUT_DIR = "artifacts/training"
DEFAULT_RESULTS_DIR = "results"

RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Portable paths (Windows-safe; no hard-coded user directories)
# ---------------------------------------------------------------------------
def find_project_root() -> Path:
    """Locate the repository root.

    Order: DOCUTUNE_ROOT env var, then walk up from the current working
    directory looking for the pyproject.toml marker, then fall back to the
    current working directory.
    """
    env_root = os.environ.get("DOCUTUNE_ROOT")
    if env_root:
        return Path(env_root).resolve()
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path.cwd().resolve()


def project_path(relative: str | os.PathLike[str]) -> Path:
    """Resolve a repo-relative path into an absolute path."""
    path = Path(relative)
    if path.is_absolute():
        return path
    return find_project_root() / path


# ---------------------------------------------------------------------------
# YAML config loading
# ---------------------------------------------------------------------------
def load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    resolved = project_path(path) if not Path(path).is_absolute() else Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Configuration file not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Configuration root must be a mapping: {resolved}")
    return data


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    name: str = DEFAULT_BASE_MODEL
    revision: str | None = None


@dataclass
class QuantizationConfig:
    enabled: bool = True
    bits: int = 4
    quant_type: str = "nf4"
    use_double_quant: bool = True
    compute_dtype: str = "auto"  # auto -> bfloat16 if supported else float16


@dataclass
class LoRAConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )


@dataclass
class TrainingHyperparams:
    output_dir: str = DEFAULT_TRAINING_OUTPUT_DIR
    final_adapter_dir: str = DEFAULT_ADAPTER_PATH
    epochs: float = 3
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_length: int = 2048
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    gradient_checkpointing: bool = True
    eval_strategy: str = "steps"
    eval_steps: int = 25
    save_strategy: str = "steps"
    save_steps: int = 25
    logging_steps: int = 10
    save_total_limit: int = 2
    seed: int = RANDOM_SEED


@dataclass
class DataPaths:
    train_file: str = "data/splits/train.jsonl"
    validation_file: str = "data/splits/validation.jsonl"


@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingHyperparams = field(default_factory=TrainingHyperparams)
    data: DataPaths = field(default_factory=DataPaths)
    source_path: str | None = None

    @classmethod
    def from_yaml(cls, path: str | os.PathLike[str]) -> TrainConfig:
        raw = load_yaml(path)
        model = ModelConfig(**raw.get("model", {}))
        quantization = QuantizationConfig(**raw.get("quantization", {}))
        lora = LoRAConfig(**raw.get("lora", {}))
        training = TrainingHyperparams(**raw.get("training", {}))
        data = DataPaths(**raw.get("data", {}))
        return cls(
            model=model,
            quantization=quantization,
            lora=lora,
            training=training,
            data=data,
            source_path=str(path),
        )


# ---------------------------------------------------------------------------
# Evaluation configuration
# ---------------------------------------------------------------------------
@dataclass
class GenerationConfig:
    max_new_tokens: int = 700
    do_sample: bool = False
    num_beams: int = 1


@dataclass
class EvalConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    adapter_path: str = DEFAULT_ADAPTER_PATH
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    test_file: str = "data/splits/test.jsonl"
    baseline_predictions: str = "results/baseline_predictions.jsonl"
    finetuned_predictions: str = "results/finetuned_predictions.jsonl"
    bootstrap_samples: int = 1000
    bootstrap_seed: int = RANDOM_SEED
    warmup_examples: int = 3
    source_path: str | None = None

    @classmethod
    def from_yaml(cls, path: str | os.PathLike[str]) -> EvalConfig:
        raw = load_yaml(path)
        model = ModelConfig(**raw.get("model", {}))
        adapter = raw.get("adapter", {})
        generation = GenerationConfig(**raw.get("generation", {}))
        evaluation = raw.get("evaluation", {})
        return cls(
            model=model,
            adapter_path=adapter.get("path", cls.adapter_path),
            generation=generation,
            test_file=evaluation.get("test_file", cls.test_file),
            baseline_predictions=evaluation.get(
                "baseline_predictions", cls.baseline_predictions
            ),
            finetuned_predictions=evaluation.get(
                "finetuned_predictions", cls.finetuned_predictions
            ),
            bootstrap_samples=evaluation.get("bootstrap_samples", cls.bootstrap_samples),
            bootstrap_seed=evaluation.get("bootstrap_seed", cls.bootstrap_seed),
            warmup_examples=evaluation.get("warmup_examples", cls.warmup_examples),
            source_path=str(path),
        )


def generation_fingerprint(gen: GenerationConfig) -> dict[str, Any]:
    """Stable fingerprint of decoding settings, used in cache metadata."""
    return {
        "max_new_tokens": gen.max_new_tokens,
        "do_sample": gen.do_sample,
        "num_beams": gen.num_beams,
    }
