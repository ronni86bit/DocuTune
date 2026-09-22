"""Shared model-backed inference runner used by the benchmark scripts.

Loads the requested model ONCE per pass, performs warm-up generations that
are excluded from latency statistics, and exposes an extract() function
compatible with docutune.evaluation.benchmark.run_inference_pass.
"""

from __future__ import annotations

from typing import Any

from docutune.config import EvalConfig, generation_fingerprint, project_path
from docutune.utils.logging import get_logger

logger = get_logger(__name__)

WARMUP_TEXT = (
    "Priya Sharma\nBengaluru, Karnataka\n\n"
    "Data Analyst with 2 years of experience.\n\n"
    "Skills:\nPython, SQL, Tableau\n\n"
    "Experience:\nData Analyst at Skyline Analytics, January 2023 - Present.\n"
    "Built dashboards for sales metrics."
)


class ModelRunner:
    """Thin wrapper bundling model loading + deterministic extraction."""

    def __init__(self, kind: str, eval_cfg: EvalConfig, device: str = "auto",
                 quantized: bool = False):
        if kind not in ("base", "finetuned"):
            raise ValueError(f"unknown model kind: {kind}")
        self.kind = kind
        self.eval_cfg = eval_cfg
        if kind == "base":
            from docutune.inference.loader import load_base_model

            self.bundle = load_base_model(
                model_name=eval_cfg.model.name,
                revision=eval_cfg.model.revision,
                device=device,
                quantized=quantized,
            )
        else:
            from docutune.inference.loader import load_finetuned_model

            self.bundle = load_finetuned_model(
                adapter_path=eval_cfg.adapter_path,
                model_name=eval_cfg.model.name,
                revision=eval_cfg.model.revision,
                device=device,
                quantized=quantized,
            )
        from docutune.inference.extractor import ResumeExtractor

        self.extractor = ResumeExtractor(self.bundle,
                                         max_new_tokens=eval_cfg.generation.max_new_tokens)

    def warmup(self, repeats: int = 2) -> None:
        """Warm-up generations; results and latencies are discarded."""
        for _ in range(max(0, repeats)):
            self.extractor.extract(WARMUP_TEXT)
        logger.info("[%s] warm-up done (not included in latency statistics)", self.kind)

    def extract(self, text: str) -> dict[str, Any]:
        return self.extractor.extract(text).to_dict()


def ensure_predictions(
    kind: str,
    eval_cfg: EvalConfig,
    examples: list[dict[str, Any]],
    output_path: str,
    resume: bool = True,
    force: bool = False,
    max_examples: int | None = None,
    device: str = "auto",
    quantized: bool = False,
) -> list[dict[str, Any]]:
    """Run or resume one model's inference pass over the test set."""
    from docutune.evaluation.benchmark import (
        cache_fingerprint,
        predictions_have_all,
        run_inference_pass,
    )

    adapter_for_cache = project_path(eval_cfg.adapter_path) if kind == "finetuned" else None
    fingerprint = cache_fingerprint(
        kind=kind,
        model_name=eval_cfg.model.name,
        model_revision=eval_cfg.model.revision,
        adapter_path=str(adapter_for_cache) if adapter_for_cache else None,
        generation_config=generation_fingerprint(eval_cfg.generation),
        test_file=str(project_path(eval_cfg.test_file)),
    )

    if (
        not force
        and resume
        and predictions_have_all(output_path, examples[:max_examples] if max_examples else examples,
                                 fingerprint)
    ):
        from docutune.utils.io import read_jsonl
        from docutune.utils.logging import get_logger

        get_logger(__name__).info("[%s] cached predictions are current - skipping inference", kind)
        return read_jsonl(output_path)

    runner = ModelRunner(kind, eval_cfg, device=device, quantized=quantized)
    runner.warmup(repeats=eval_cfg.warmup_examples)
    return run_inference_pass(
        kind=kind,
        examples=examples,
        predictions_path=output_path,
        fingerprint=fingerprint,
        extract_fn=runner.extract,
        resume=resume,
        force=force,
        max_examples=max_examples,
    )
