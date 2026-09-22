#!/usr/bin/env python
"""Update the README benchmark table + resume bullet from results/metrics.json.

Never invents numbers: refuses to run when metrics.json does not exist.

Usage:
    python scripts/update_readme_metrics.py            # update README.md
    python scripts/update_readme_metrics.py --check    # exit 1 if out of sync
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docutune.config import project_path  # noqa: E402
from docutune.evaluation.reporting import (  # noqa: E402
    README_TABLE_END,
    README_TABLE_START,
    render_readme_table,
    update_readme_metrics,
)
from docutune.utils.io import read_json, read_text  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync README metrics with measured results")
    parser.add_argument("--metrics", default="results/metrics.json")
    parser.add_argument("--readme", default="README.md")
    parser.add_argument("--check", action="store_true",
                        help="Only verify that README matches metrics.json")
    args = parser.parse_args()

    metrics_path = project_path(args.metrics)
    readme_path = project_path(args.readme)

    if args.check:
        if not metrics_path.is_file():
            print("metrics.json not found; README must still contain TBD values")
            return 0
        metrics = read_json(metrics_path)
        readme = read_text(readme_path)
        expected_table = render_readme_table(metrics)
        start = readme.find(README_TABLE_START)
        end = readme.find(README_TABLE_END)
        ok = start != -1 and end != -1 and expected_table in readme[start:end]
        if not ok:
            print("README benchmark table is out of sync with results/metrics.json")
            return 1
        print("README metrics are in sync")
        return 0

    try:
        changed = update_readme_metrics(str(metrics_path), str(readme_path))
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1
    print("README updated." if changed else "README already up to date.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
