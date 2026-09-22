#!/usr/bin/env python
"""Validate the generated dataset (leakage, schema, counts, distributions).

Usage:
    python scripts/validate_data.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docutune.data.validator import format_report, validate_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the DocuTune dataset")
    parser.add_argument("--splits-dir", default=None,
                        help="Optional custom directory containing train/validation/test.jsonl")
    args = parser.parse_args()

    report = validate_dataset(splits_dir=args.splits_dir)
    print(format_report(report))
    if not report.ok:
        print("\nValidation FAILED - fix the errors above before training.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
