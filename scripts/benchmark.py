#!/usr/bin/env python
"""Compute the full before/after benchmark and write every results artifact.

Runs inference when prediction files are missing (or with --force), then
produces: metrics.json, metrics.csv, per_field_metrics.csv, comparison.jsonl,
error analysis, bootstrap CIs, charts, benchmark_report.md/.html,
benchmark_manifest.json, FINAL_REPORT.md and the README metric update.

Usage:
    python scripts/benchmark.py [--resume] [--force] [--max-examples N]
                                [--skip-baseline] [--skip-finetuned]
"""

import argparse
import csv
import platform
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docutune.config import (  # noqa: E402
    EvalConfig,
    generation_fingerprint,
    project_path,
)
from docutune.data.validator import validate_dataset  # noqa: E402
from docutune.evaluation.benchmark import (  # noqa: E402
    build_comparison_rows,  # noqa: E402
    evaluate_model_predictions,
    per_example_evaluations,
)
from docutune.evaluation.bootstrap import summarize_bootstrap  # noqa: E402
from docutune.evaluation.error_analysis import (  # noqa: E402
    build_error_records,
    select_representative_examples,
    summarize_errors,
    write_error_summary_csv,
)
from docutune.evaluation.metrics import delta_metrics, per_field_table  # noqa: E402
from docutune.evaluation.reporting import (  # noqa: E402
    generate_charts,
    markdown_to_html,
    render_benchmark_report,
    update_readme_metrics,
    write_headline_metrics_csv,
    write_per_field_csv,
)
from docutune.evaluation.runner import ensure_predictions  # noqa: E402
from docutune.utils.io import (  # noqa: E402
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
    write_text,
)
from docutune.utils.logging import get_logger  # noqa: E402

logger = get_logger("benchmark")


def _software_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {"python": platform.python_version()}
    for pkg in ("torch", "transformers", "peft", "bitsandbytes", "accelerate"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    return out


def _device_and_gpu() -> tuple[str, str]:
    device, gpu = "cpu", "none"
    try:
        import torch

        if torch.cuda.is_available():
            device = "cuda"
            gpu = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return device, gpu


def run_full_benchmark(
    cfg: EvalConfig,
    max_examples: int | None,
    resume: bool,
    force: bool,
    skip_baseline: bool,
    skip_finetuned: bool,
    device: str = "auto",
    update_readme: bool = True,
) -> dict:
    """Ensure predictions, then compute and write every benchmark artifact."""
    results_dir = project_path("results")
    test_file = project_path(cfg.test_file)
    examples = read_jsonl(test_file)
    if max_examples:
        examples = examples[:max_examples]
    logger.info("Benchmark over %d held-out test examples", len(examples))

    # ------------------------------------------------------------------
    # 1-2. Inference passes (identical settings for both models)
    # ------------------------------------------------------------------
    base_rows: list = []
    ft_rows: list = []
    if not skip_baseline:
        logger.info("=== Baseline (base model) pass ===")
        base_rows = ensure_predictions(
            "base", cfg, examples, str(project_path(cfg.baseline_predictions)),
            resume=resume, force=force, max_examples=max_examples, device=device,
        )
    elif Path(project_path(cfg.baseline_predictions)).is_file():
        base_rows = read_jsonl(project_path(cfg.baseline_predictions))

    if not skip_finetuned:
        adapter_dir = project_path(cfg.adapter_path)
        if not (adapter_dir / "adapter_config.json").is_file():
            logger.warning(
                "Fine-tuned adapter not found at %s - skipping fine-tuned pass. "
                "Run the Colab training pipeline and configure ADAPTER_PATH.", adapter_dir,
            )
        else:
            logger.info("=== Fine-tuned (base + adapter) pass ===")
            ft_rows = ensure_predictions(
                "finetuned", cfg, examples, str(project_path(cfg.finetuned_predictions)),
                resume=resume, force=force, max_examples=max_examples, device=device,
            )
    elif Path(project_path(cfg.finetuned_predictions)).is_file():
        ft_rows = read_jsonl(project_path(cfg.finetuned_predictions))

    if not base_rows and not ft_rows:
        raise RuntimeError("No predictions available; nothing to benchmark.")

    # ------------------------------------------------------------------
    # 3-7. Metrics via the ONE shared evaluator
    # ------------------------------------------------------------------
    logger.info("Computing metrics with the shared evaluator")
    base_metrics = evaluate_model_predictions(base_rows, examples)
    ft_metrics = evaluate_model_predictions(ft_rows, examples) if ft_rows else {}

    train_manifest_path = project_path("data/dataset_manifest.json")
    train_size = 450
    if Path(train_manifest_path).is_file():
        train_size = read_json(train_manifest_path).get("counts", {}).get("train", 450)

    device_label, gpu_label = _device_and_gpu()
    metadata = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "base_model": cfg.model.name,
        "base_model_revision": cfg.model.revision,
        "adapter": cfg.adapter_path if ft_rows else None,
        "dataset_version": "1.0",
        "schema_version": "1.0",
        "prompt_version": "1.0",
        "evaluator_version": "1.0",
        "test_size": len(examples),
        "train_size": train_size,
        "seed": cfg.bootstrap_seed,
        "generation_config": generation_fingerprint(cfg.generation),
        "device": device_label,
        "gpu": gpu_label,
        "software_versions": _software_versions(),
    }
    metrics = {
        "metadata": metadata,
        "base": {k: v for k, v in base_metrics.items() if not k.startswith("_")},
        "finetuned": (
            {k: v for k, v in ft_metrics.items() if not k.startswith("_")}
            if ft_metrics else None
        ),
        "delta": delta_metrics(base_metrics, ft_metrics) if ft_metrics else {},
    }

    # Per-example evaluations (shared by bootstrap CIs and example selection)
    base_evals_map = per_example_evaluations(base_rows, examples) if base_rows else {}
    ft_evals_map = per_example_evaluations(ft_rows, examples) if ft_rows else {}

    # Bootstrap CIs (per-example, deterministic seed)
    if base_evals_map:
        metrics["bootstrap"] = {"base": summarize_bootstrap(
            list(base_evals_map.values()), n_boot=cfg.bootstrap_samples, seed=cfg.bootstrap_seed)}
        if ft_evals_map:
            metrics["bootstrap"]["finetuned"] = summarize_bootstrap(
                list(ft_evals_map.values()), n_boot=cfg.bootstrap_samples, seed=cfg.bootstrap_seed)

    # Per-field table
    if ft_metrics:
        metrics["per_field"] = per_field_table(base_metrics, ft_metrics)
    write_json(results_dir / "metrics.json", metrics)
    write_headline_metrics_csv(str(results_dir / "metrics.csv"), metrics)
    if metrics.get("per_field"):
        write_per_field_csv(str(results_dir / "per_field_metrics.csv"), metrics["per_field"])

    # comparison.jsonl (per-example, both models)
    if base_rows and ft_rows:
        comparison = build_comparison_rows(base_rows, ft_rows, examples)
        write_jsonl(results_dir / "comparison.jsonl", comparison)

    # ------------------------------------------------------------------
    # 8. Error analysis + representative examples (includes failures)
    # ------------------------------------------------------------------
    if base_rows and ft_rows:
        logger.info("Running error analysis")
        base_by_id = {r["id"]: r for r in base_rows}
        ft_by_id = {r["id"]: r for r in ft_rows}
        records = build_error_records(examples, base_by_id, ft_by_id)
        write_jsonl(results_dir / "error_analysis.jsonl", records)
        summary = summarize_errors(records)
        write_error_summary_csv(str(results_dir / "error_analysis_summary.csv"), summary)

        f1_by_id = {
            ex["id"]: {
                "base": base_evals_map.get(ex["id"], {}).get("field_f1"),
                "finetuned": ft_evals_map.get(ex["id"], {}).get("field_f1"),
            }
            for ex in examples
        }
        representative = select_representative_examples(records, f1_by_id,
                                                        seed=cfg.bootstrap_seed)
        write_json(results_dir / "reports" / "representative_examples.json", representative)
    elif base_rows:
        base_by_id = {r["id"]: r for r in base_rows}
        records = build_error_records(examples, base_by_id, {})
        write_jsonl(results_dir / "error_analysis.jsonl", records)
        write_error_summary_csv(str(results_dir / "error_analysis_summary.csv"),
                                summarize_errors(records))

    # ------------------------------------------------------------------
    # 9-10. Charts + reports + manifest + README
    # ------------------------------------------------------------------
    training_log = project_path("artifacts/training/training_log.json")
    charts = generate_charts(metrics, str(results_dir / "charts"),
                             training_log_path=str(training_log))
    logger.info("Charts written: %d", len(charts))

    error_summary_rows = []
    summary_path = results_dir / "error_analysis_summary.csv"
    if summary_path.is_file():
        with summary_path.open("r", encoding="utf-8", newline="") as fh:
            error_summary_rows = [
                {**row,
                 "base_count": int(row["base_count"]),
                 "finetuned_count": int(row["finetuned_count"]),
                 "base_pct": float(row["base_pct"]),
                 "finetuned_pct": float(row["finetuned_pct"])}
                for row in csv.DictReader(fh)
            ]

    benchmark_manifest = {
        "timestamp": metadata["timestamp"],
        "base_model": metadata["base_model"],
        "base_model_revision": metadata["base_model_revision"],
        "adapter": metadata["adapter"],
        "dataset_version": metadata["dataset_version"],
        "schema_version": metadata["schema_version"],
        "prompt_version": metadata["prompt_version"],
        "evaluator_version": metadata["evaluator_version"],
        "test_size": metadata["test_size"],
        "generation_config": metadata["generation_config"],
        "device": metadata["device"],
        "gpu": metadata["gpu"],
        "software_versions": metadata["software_versions"],
        "decoding": "greedy (do_sample=false, num_beams=1); identical for both models",
        "latency_note": "warm model; warm-up generations excluded from statistics",
        "scope_note": "results apply to this held-out synthetic benchmark only",
    }
    write_json(results_dir / "benchmark_manifest.json", benchmark_manifest)

    report_md = render_benchmark_report(metrics, error_summary_rows, benchmark_manifest)
    write_text(results_dir / "benchmark_report.md", report_md)
    write_text(results_dir / "benchmark_report.html", markdown_to_html(report_md))
    write_text(results_dir / "FINAL_REPORT.md", render_benchmark_report(
        metrics, error_summary_rows, benchmark_manifest, final=True))

    if update_readme:
        try:
            update_readme_metrics(str(results_dir / "metrics.json"),
                                  str(project_path("README.md")))
        except FileNotFoundError as exc:
            logger.warning("%s", exc)

    logger.info("Benchmark complete: results/metrics.json + reports written")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="DocuTune before/after benchmark")
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-finetuned", action="store_true")
    parser.add_argument("--skip-inference", action="store_true",
                        help="Only compute artifacts from existing prediction files")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--no-readme", action="store_true", help="Do not touch README.md")
    args = parser.parse_args()

    report = validate_dataset()
    if not report.ok:
        logger.error("Dataset validation failed - fix data before benchmarking.")
        return 1

    cfg = EvalConfig.from_yaml(args.config)
    if args.skip_inference:
        args.skip_baseline = True
        args.skip_finetuned = True
    run_full_benchmark(
        cfg,
        max_examples=args.max_examples,
        resume=args.resume,
        force=args.force,
        skip_baseline=args.skip_baseline,
        skip_finetuned=args.skip_finetuned,
        device=args.device,
        update_readme=not args.no_readme,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
