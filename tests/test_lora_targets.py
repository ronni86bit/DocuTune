"""LoRA target-module resolution (BUG 2: Phi-3 fuses q/k/v into qkv_proj)
and TrainingArguments construction against the INSTALLED transformers.

Torch/transformers-dependent tests skip gracefully when the training
extras are not installed (e.g. in CI, which never downloads models).
"""

import pytest

from docutune.training.model import resolve_lora_target_modules
from docutune.training.schedule import compute_warmup_steps

# Every test in this module needs torch: the stub models build real
# nn.Linear layers and the integration tests build real Phi-3 modules.
# CI installs no training extras, so skip the whole module there.
pytest.importorskip("torch", reason="stub/real models require torch")

DEFAULT_REQUEST = ["q_proj", "k_proj", "v_proj", "o_proj"]


class _StubModel:
    """Minimal named_modules stand-in over real nn.Linear layers."""

    def __init__(self, linear_names):
        import torch.nn as nn

        self._modules_by_name = {
            name: nn.Linear(8, 8) for name in linear_names
        }

    def named_modules(self):
        return iter(self._modules_by_name.items())


def _separate_attention_model():
    # Llama-style: separate q/k/v projections plus o_proj and MLP.
    return _StubModel([
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.v_proj",
        "model.layers.0.self_attn.o_proj",
        "model.layers.0.mlp.gate_proj",
        "model.layers.0.mlp.up_proj",
        "model.layers.0.mlp.down_proj",
    ])


def _phi3_style_model():
    # Phi-3-style: FUSED qkv_proj plus o_proj and MLP.
    return _StubModel([
        "model.layers.0.self_attn.qkv_proj",
        "model.layers.0.self_attn.o_proj",
        "model.layers.0.mlp.gate_up_proj",
        "model.layers.0.mlp.down_proj",
    ])


class TestResolutionOnStubs:
    def test_separate_qkv_resolves_exactly(self):
        resolved = resolve_lora_target_modules(_separate_attention_model(), DEFAULT_REQUEST)
        assert set(resolved) == {"q_proj", "k_proj", "v_proj", "o_proj"}

    def test_phi3_fused_qkv_mapped_from_trio(self):
        resolved = resolve_lora_target_modules(_phi3_style_model(), DEFAULT_REQUEST)
        # q/k/v requests collapse into the single fused qkv_proj adapter
        assert set(resolved) == {"qkv_proj", "o_proj"}

    def test_phi3_fused_request_expands_to_separate(self):
        resolved = resolve_lora_target_modules(
            _separate_attention_model(), ["qkv_proj", "o_proj"])
        assert set(resolved) == {"q_proj", "k_proj", "v_proj", "o_proj"}

    def test_unknown_module_fails_loudly(self):
        with pytest.raises(ValueError, match="Unresolved"):
            resolve_lora_target_modules(_phi3_style_model(), ["q_proj", "w_proj", "o_proj"])

    def test_partial_trio_does_not_silently_alias(self):
        # only q_proj requested of the trio: ambiguous -> must fail loudly
        with pytest.raises(ValueError, match="Unresolved"):
            resolve_lora_target_modules(_phi3_style_model(), ["q_proj"])

    def test_empty_resolution_impossible(self):
        with pytest.raises(ValueError):
            resolve_lora_target_modules(_phi3_style_model(), ["nonexistent"])

    def test_duplicates_deduplicated(self):
        resolved = resolve_lora_target_modules(
            _phi3_style_model(), ["o_proj", "o_proj", "q_proj", "k_proj", "v_proj"])
        assert resolved.count("o_proj") == 1


class TestOnRealPhi3:
    """Integration-style: resolve against the REAL Phi3Attention class from
    the installed transformers (constructor builds its Linears in-process,
    no model download)."""

    def test_phi3_attention_targets(self):
        pytest.importorskip("torch")
        pytest.importorskip("transformers")
        from transformers.models.phi3 import modeling_phi3

        attention = modeling_phi3.Phi3Attention(
            modeling_phi3.Phi3Config(hidden_size=32, num_attention_heads=4,
                                     num_key_value_heads=2),
            layer_idx=0,
        )
        resolved = resolve_lora_target_modules(attention, DEFAULT_REQUEST)
        assert set(resolved) == {"qkv_proj", "o_proj"}, (
            f"Phi-3 resolution wrong: {resolved} - the fused qkv_proj and "
            "o_proj must both be adapted"
        )
        # ... and the model source really uses a fused qkv projection
        import inspect

        src = inspect.getsource(modeling_phi3)
        assert "self.qkv_proj" in src

    def test_full_phi3_model_targets(self):
        pytest.importorskip("torch")
        pytest.importorskip("transformers")
        from transformers.models.phi3 import modeling_phi3

        tiny = modeling_phi3.Phi3Config(
            hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, vocab_size=1000,
            pad_token_id=0, max_position_embeddings=512,
        )
        model = modeling_phi3.Phi3Model(tiny)
        resolved = resolve_lora_target_modules(model, DEFAULT_REQUEST)
        assert set(resolved) == {"qkv_proj", "o_proj"}
        # every resolved name matches at least one actual Linear submodule
        available = set()
        import torch.nn as nn

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                available.add(name)
                available.add(name.split(".")[-1])
        assert set(resolved) <= available


class TestTrainingArgumentsConstruction:
    """BUG 1 regression guard: TrainingArguments must construct on the
    installed transformers with the exact kwargs train.py uses."""

    def test_construction_with_computed_warmup(self, tmp_path):
        pytest.importorskip("torch")
        pytest.importorskip("transformers")

        from docutune.config import TrainConfig
        from docutune.training.train import build_training_arguments

        cfg = TrainConfig.from_yaml("configs/train.yaml")
        args = build_training_arguments(
            cfg, tmp_path, num_train_examples=450, bf16=False, smoke_test=False,
        )
        # warmup converted from the 0.05 ratio over 87 total steps
        assert args.warmup_steps == compute_warmup_steps(0.05, 87) == 5
        assert args.per_device_train_batch_size == 2
        assert args.gradient_accumulation_steps == 8

    def test_construction_smoke_mode(self, tmp_path):
        pytest.importorskip("torch")
        pytest.importorskip("transformers")
        from docutune.config import TrainConfig
        from docutune.training.train import build_training_arguments

        cfg = TrainConfig.from_yaml("configs/train.yaml")
        args = build_training_arguments(
            cfg, tmp_path, num_train_examples=16, bf16=False, smoke_test=True,
        )
        assert args.max_steps == 2
        assert args.warmup_steps == 1
        assert args.eval_strategy == "no"
        assert args.save_strategy == "no"
