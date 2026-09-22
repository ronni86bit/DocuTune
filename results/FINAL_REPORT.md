# DocuTune Before/After Evaluation

> **STATUS: NOT YET EXECUTED.**
> This file is a template. Running `python scripts/benchmark.py` (or
> `python scripts/run_experiment.py`) after a real training run overwrites it with the
> actual measured report. Until then every metric is TBD — no numbers are fabricated.

## Experiment

Untouched base model vs. LoRA/QLoRA fine-tuned model on the exact same held-out
synthetic test set, with the exact same prompt, decoding configuration, parser,
normalization and evaluator. Only the model state differs.

## Dataset

- 600 synthetic examples (seed 42): 450 train / 75 validation / 75 test
- Test set uses held-out rendering templates 11–12
- Status: **TBD** (run the benchmark to fill)

## Model

- Base model: `microsoft/Phi-3-mini-4k-instruct`
- Fine-tuned: base + LoRA adapter (`artifacts/adapters/final`)
- Status: **TBD**

## Fine-Tuning Configuration

See `artifacts/training/training_manifest.json` (written by a real training run).

## Baseline Results

TBD

## Fine-Tuned Results

TBD

## Metric Deltas

TBD

## Per-Field Metrics

TBD

## Error Analysis

TBD

## Latency

TBD

## Confidence Intervals

TBD — 95% bootstrap intervals on this held-out benchmark only.

## Representative Examples

TBD — deterministic selection including failures (random seeded sample, biggest
improvement, regression, and both base/fine-tuned failure-swap directions).

## Limitations

- Synthetic dataset and synthetic language bias
- Small held-out test set (75 examples by default)
- `unsupported_value_rate` is defined against the synthetic gold record only
- Claims are scoped to this benchmark configuration

## Reproducibility

```bash
python scripts/generate_data.py --seed 42
python scripts/validate_data.py
python scripts/run_baseline.py
python -m docutune.training.train --config configs/train.yaml   # GPU (Colab)
python scripts/run_finetuned.py
python scripts/benchmark.py                                      # writes this file
```
