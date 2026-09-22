#!/usr/bin/env python
"""Run fine-tuned inference: base model + LoRA adapter over the same test set.

Usage:
    python scripts/run_finetuned.py [--resume] [--force] [--max-examples 75]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docutune.config import EvalConfig, project_path  # noqa: E402
from docutune.evaluation.runner import ensure_predictions  # noqa: E402
from docutune.inference.loader import AdapterNotFoundError  # noqa: E402
from docutune.utils.io import read_jsonl  # noqa: E402
from docutune.utils.logging import get_logger  # noqa: E402

logger = get_logger("run_finetuned")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tuned (base + adapter) inference pass")
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--output", default=None, help="Override predictions output path")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--adapter", default=None, help="Override adapter path")
    args = parser.parse_args()

    cfg = EvalConfig.from_yaml(args.config)
    if args.adapter:
        cfg.adapter_path = args.adapter
    output = args.output or cfg.finetuned_predictions
    test_file = project_path(cfg.test_file)
    if not Path(test_file).is_file():
        logger.error("Test set not found at %s - run scripts/generate_data.py first", test_file)
        return 1

    adapter_path = project_path(cfg.adapter_path)
    if not (adapter_path / "adapter_config.json").is_file():
        logger.error(
            "Fine-tuned adapter not found. Run the Colab training pipeline and "
            "configure ADAPTER_PATH (expected: %s)", adapter_path,
        )
        return 2

    examples = read_jsonl(test_file)
    logger.info("Fine-tuned inference over %d test examples (adapter=%s)",
                min(len(examples), args.max_examples or len(examples)), adapter_path)
    try:
        rows = ensure_predictions(
            kind="finetuned",
            eval_cfg=cfg,
            examples=examples,
            output_path=str(project_path(output)),
            resume=args.resume,
            force=args.force,
            max_examples=args.max_examples,
            device=args.device,
        )
    except AdapterNotFoundError as exc:
        logger.error("%s", exc)
        return 2
    logger.info("Wrote %d fine-tuned predictions to %s", len(rows), output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
