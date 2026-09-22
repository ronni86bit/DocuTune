#!/usr/bin/env python
"""Generate the deterministic synthetic resume dataset.

Usage:
    python scripts/generate_data.py --seed 42
    python scripts/generate_data.py --train 20 --validation 6 --test 6   # dev subset
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docutune.config import project_path  # noqa: E402
from docutune.data.generator import DEFAULT_COUNTS, dataset_manifest, generate_dataset  # noqa: E402
from docutune.data.splitter import split_examples  # noqa: E402
from docutune.utils.io import sha256_file, write_json, write_jsonl, write_text  # noqa: E402
from docutune.utils.logging import get_logger  # noqa: E402

logger = get_logger("generate_data")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the DocuTune synthetic dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", type=int, default=DEFAULT_COUNTS["train"])
    parser.add_argument("--validation", type=int, default=DEFAULT_COUNTS["validation"])
    parser.add_argument("--test", type=int, default=DEFAULT_COUNTS["test"])
    parser.add_argument("--output-dir", default="data", help="Dataset root (contains splits/)")
    args = parser.parse_args()

    counts = {"train": args.train, "validation": args.validation, "test": args.test}
    out_dir = project_path(args.output_dir)
    splits_dir = out_dir / "splits"
    processed_dir = out_dir / "processed"
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)

    logger.info("Generating dataset: seed=%d counts=%s", args.seed, counts)
    examples = generate_dataset(seed=args.seed, counts=counts)
    by_split = split_examples(examples)

    for split, split_examples_list in by_split.items():
        path = splits_dir / f"{split}.jsonl"
        write_jsonl(path, split_examples_list)
        logger.info("wrote %s (%d examples)", path, len(split_examples_list))
    write_jsonl(processed_dir / "all_examples.jsonl", examples)

    hashes = {
        f"{split}.jsonl": sha256_file(splits_dir / f"{split}.jsonl")
        for split in by_split
    }
    manifest = dataset_manifest(examples, seed=args.seed, file_hashes=hashes)
    manifest["counts"] = counts
    write_json(out_dir / "dataset_manifest.json", manifest)
    write_text(
        out_dir / "DATASET_CARD.md",
        "# DocuTune Dataset\n\n"
        "Fully synthetic, deterministic, generated locally - no real personal resumes, "
        "no external LLM APIs. See docs/DATASET.md for the full methodology and "
        "data/dataset_manifest.json for the exact generation fingerprint.\n",
    )
    logger.info("Dataset manifest written (total=%d). Next: python scripts/validate_data.py",
                len(examples))
    return 0


if __name__ == "__main__":
    sys.exit(main())
