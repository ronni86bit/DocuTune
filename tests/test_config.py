"""Configuration loading: YAML defaults, dataclasses, env overrides."""

from docutune.config import (
    DEFAULT_BASE_MODEL,
    FALLBACK_MODELS,
    EvalConfig,
    TrainConfig,
    load_yaml,
)


def test_train_config_defaults():
    cfg = TrainConfig.from_yaml("configs/train.yaml")
    assert cfg.model.name == DEFAULT_BASE_MODEL
    assert cfg.lora.r == 16
    assert cfg.lora.alpha == 32
    assert cfg.lora.dropout == 0.05
    assert cfg.lora.target_modules == ["q_proj", "k_proj", "v_proj", "o_proj"]
    assert cfg.quantization.bits == 4
    assert cfg.quantization.quant_type == "nf4"
    assert cfg.quantization.use_double_quant is True
    assert cfg.training.epochs == 3
    assert abs(cfg.training.learning_rate - 2e-4) < 1e-9
    assert cfg.training.max_length == 2048
    assert cfg.training.per_device_train_batch_size == 2
    assert cfg.training.gradient_accumulation_steps == 8
    assert cfg.data.train_file == "data/splits/train.jsonl"
    # The test split must not be referenced anywhere in the training config.
    import json

    assert "test" not in json.dumps(vars(cfg.data))


def test_eval_config_defaults():
    cfg = EvalConfig.from_yaml("configs/eval.yaml")
    assert cfg.adapter_path == "artifacts/adapters/final"
    assert cfg.generation.max_new_tokens == 700
    assert cfg.generation.do_sample is False
    assert cfg.generation.num_beams == 1
    assert cfg.test_file == "data/splits/test.jsonl"
    assert cfg.bootstrap_seed == 42


def test_fallback_models_documented():
    assert "meta-llama/Llama-3.2-3B-Instruct" in FALLBACK_MODELS
    assert "Qwen/Qwen2.5-3B-Instruct" in FALLBACK_MODELS


def test_load_yaml_missing_raises():
    import pytest

    with pytest.raises(FileNotFoundError):
        load_yaml("configs/does_not_exist.yaml")


def test_backend_settings_env_overrides(monkeypatch, tmp_path):
    from backend.app.config import load_settings

    monkeypatch.setenv("BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
    monkeypatch.setenv("MODEL_MODE", "base")
    monkeypatch.setenv("ADAPTER_PATH", str(tmp_path))
    monkeypatch.setenv("MAX_NEW_TOKENS", "256")
    monkeypatch.setenv("QUANTIZED", "true")
    settings = load_settings()
    assert settings.base_model == "Qwen/Qwen2.5-3B-Instruct"
    assert settings.model_mode == "base"
    assert settings.max_new_tokens == 256
    assert settings.quantized is True
    assert settings.adapter_abs_path == tmp_path


def test_backend_settings_rejects_bad_mode(monkeypatch):
    from backend.app.config import load_settings

    monkeypatch.setenv("MODEL_MODE", "banana")
    import pytest

    with pytest.raises(ValueError):
        load_settings()
