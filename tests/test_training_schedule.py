"""Warmup/schedule math (BUG 1 compatibility: transformers 5.x removed
TrainingArguments(warmup_ratio=...), so the ratio is converted to steps)."""

import pytest

from docutune.training.schedule import compute_total_update_steps, compute_warmup_steps


class TestTotalUpdateSteps:
    def test_default_dataset_shape(self):
        # 450 examples, batch 2, grad-accum 8, 3 epochs:
        # ceil(450/2)=225 -> ceil(225/8)=29 -> 29*3 = 87
        assert compute_total_update_steps(450, 2, 8, 3) == 87

    def test_single_step_run(self):
        # 16 examples, batch 2, accum 8, 1 epoch: ceil(16/2)=8 -> ceil(8/8)=1
        assert compute_total_update_steps(16, 2, 8, 1) == 1

    def test_scales_with_batch_size(self):
        base = compute_total_update_steps(450, 2, 8, 3)
        larger_batch = compute_total_update_steps(450, 4, 8, 3)
        assert larger_batch < base

    def test_scales_with_gradient_accumulation(self):
        base = compute_total_update_steps(450, 2, 8, 3)
        more_accum = compute_total_update_steps(450, 2, 16, 3)
        assert more_accum < base

    def test_scales_with_dataset_size(self):
        assert compute_total_update_steps(100, 2, 8, 3) < compute_total_update_steps(900, 2, 8, 3)

    def test_fractional_epochs(self):
        # 29 update steps/epoch * 0.5 -> ceil(14.5) = 15
        assert compute_total_update_steps(450, 2, 8, 0.5) == 15

    def test_override_max_steps_wins(self):
        assert compute_total_update_steps(450, 2, 8, 3, override_max_steps=2) == 2
        assert compute_total_update_steps(16, 2, 8, 1, override_max_steps=10) == 10

    def test_minimum_one_step(self):
        assert compute_total_update_steps(1, 8, 64, 1) == 1

    def test_rejects_invalid_inputs(self):
        with pytest.raises(ValueError):
            compute_total_update_steps(0, 2, 8, 3)
        with pytest.raises(ValueError):
            compute_total_update_steps(10, 0, 8, 3)
        with pytest.raises(ValueError):
            compute_total_update_steps(10, 2, 0, 3)
        with pytest.raises(ValueError):
            compute_total_update_steps(10, 2, 8, 0)


class TestWarmupSteps:
    def test_configured_ratio_on_default_dataset(self):
        # total 87 steps * 0.05 = 4.35 -> ceil = 5 warmup steps
        assert compute_warmup_steps(0.05, 87) == 5

    def test_ceil_guarantees_requested_fraction(self):
        assert compute_warmup_steps(0.05, 10) == 1  # ceil(0.5)
        assert compute_warmup_steps(0.1, 25) == 3  # ceil(2.5)

    def test_disabled_for_zero_or_negative_ratio(self):
        assert compute_warmup_steps(0.0, 100) == 0
        assert compute_warmup_steps(-0.05, 100) == 0

    def test_clamped_to_total_steps(self):
        assert compute_warmup_steps(1.0, 40) == 40
        assert compute_warmup_steps(0.9, 10) == 9

    def test_smoke_run_total_is_two(self):
        assert compute_warmup_steps(0.05, 2) == 1

    def test_rejects_non_positive_total(self):
        with pytest.raises(ValueError):
            compute_warmup_steps(0.05, 0)
