#!/usr/bin/env python
"""Optionally merge the LoRA adapter into the base model weights.

Normal serving uses base + adapter and does NOT need this. Merged weights are
only useful for runtimes without PEFT support.

Usage:
    python scripts/merge_adapter.py [--adapter artifacts/adapters/final]
                                    [--output artifacts/adapters/merged]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docutune.config import EvalConfig, project_path  # noqa: E402
from docutune.utils.logging import get_logger  # noqa: E402

logger = get_logger("merge_adapter")


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge the LoRA adapter into base weights")
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--output", default="artifacts/adapters/merged")
    args = parser.parse_args()

    try:
        import torch  # noqa: F401
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        logger.error("Merging requires torch, transformers and peft (%s)", exc)
        return 1

    cfg = EvalConfig.from_yaml(args.config)
    adapter_dir = project_path(args.adapter or cfg.adapter_path)
    if not Path(adapter_dir).is_dir():
        logger.error("Adapter not found at %s", adapter_dir)
        return 1
    out_dir = project_path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    import transformers

    dtype_key = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    logger.info("Loading base model %s", cfg.model.name)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name, revision=cfg.model.revision or None,
        **{dtype_key: "auto"}, low_cpu_mem_usage=True,
    )
    logger.info("Attaching and merging adapter from %s", adapter_dir)
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model = model.merge_and_unload()
    logger.info("Saving merged model to %s (this is several GB)", out_dir)
    model.save_pretrained(str(out_dir))
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, revision=cfg.model.revision or None)
    tokenizer.save_pretrained(str(out_dir))
    logger.info("Merged model written. Serving does not require this artifact.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
