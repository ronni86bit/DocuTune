#!/usr/bin/env python
"""Run baseline inference: the UNTOUCHED base model over the held-out test set.

Uses the canonical prompt, deterministic decoding, the shared parser and the
shared evaluator - identical to the fine-tuned pass in every respect except
the model state.

Usage:
    python scripts/run_baseline.py [--resume] [--force] [--max-examples 75]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docutune.config import EvalConfig, project_path  # noqa: E402
from docutune.evaluation.runner import ensure_predictions  # noqa: E402
from docutune.utils.io import read_jsonl  # noqa: E402
from docutune.utils.logging import get_logger  # noqa: E402

logger = get_logger("run_baseline")


def main() -> int:
    parser = argparse.ArgumentParser(description="Baseline (base model) inference pass")
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--output", default=None, help="Override predictions output path")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--force", action="store_true", help="Recompute even if cache matches")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    cfg = EvalConfig.from_yaml(args.config)
    output = args.output or cfg.baseline_predictions
    test_file = project_path(cfg.test_file)
    if not Path(test_file).is_file():
        logger.error("Test set not found at %s - run scripts/generate_data.py first", test_file)
        return 1
    examples = read_jsonl(test_file)
    logger.info("Baseline inference over %d test examples (model=%s)",
                min(len(examples), args.max_examples or len(examples)), cfg.model.name)

    rows = ensure_predictions(
        kind="base",
        eval_cfg=cfg,
        examples=examples,
        output_path=str(project_path(output)),
        resume=args.resume,
        force=args.force,
        max_examples=args.max_examples,
        device=args.device,
    )
    logger.info("Wrote %d baseline predictions to %s", len(rows), output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
