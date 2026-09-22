#!/usr/bin/env python
"""Run the complete experiment end to end.

Pipeline: environment validation -> dataset validation -> baseline inference
-> adapter availability check -> fine-tuned inference -> metric calculation
-> bootstrap analysis -> error analysis -> charts -> reports -> README update.

Usage:
    python scripts/run_experiment.py [--resume] [--force] [--max-examples N]
                                     [--skip-baseline] [--skip-finetuned]
"""

import argparse
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docutune.config import EvalConfig  # noqa: E402
from docutune.data.validator import format_report, validate_dataset  # noqa: E402
from docutune.utils.logging import get_logger  # noqa: E402

logger = get_logger("run_experiment")

# Sibling script reuse: scripts/benchmark.py holds the artifact pipeline.
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _log_environment() -> None:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "none"
    versions = {}
    for pkg in ("torch", "transformers", "peft", "bitsandbytes", "accelerate"):
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = None
    logger.info("Environment: device=%s gpu=%s torch=%s transformers=%s peft=%s",
                device, gpu, versions.get("torch"), versions.get("transformers"),
                versions.get("peft"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full DocuTune experiment")
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-finetuned", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--no-readme", action="store_true")
    args = parser.parse_args()

    # 1. Environment validation
    logger.info("Step 1/11: environment validation")
    try:
        _log_environment()
    except ImportError as exc:
        logger.error("Missing required training/serving libraries: %s", exc)
        return 1

    # 2. Dataset validation (hard gate)
    logger.info("Step 2/11: dataset validation")
    report = validate_dataset()
    print(format_report(report))
    if not report.ok:
        logger.error("Dataset invalid - aborting. Run scripts/generate_data.py first.")
        return 1

    cfg = EvalConfig.from_yaml(args.config)

    # 3-11. Inference, metrics, bootstrap, error analysis, charts, reports, README
    from benchmark import run_full_benchmark  # noqa: E402 - sibling script

    logger.info("Steps 3-11: benchmark pipeline (baseline, fine-tuned, metrics, "
                "bootstrap, error analysis, charts, reports, README)")
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
    logger.info("Full experiment finished. See results/metrics.json and results/FINAL_REPORT.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
