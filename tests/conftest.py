"""Shared test fixtures. All tests are GPU-free: no model downloads, no
Hugging Face authentication, no real inference."""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from docutune.data.generator import generate_dataset  # noqa: E402
from docutune.data.splitter import split_examples  # noqa: E402
from docutune.utils.io import write_jsonl  # noqa: E402


@pytest.fixture(scope="session")
def small_dataset(tmp_path_factory):
    """A tiny deterministic dataset written to a temp directory."""
    examples = generate_dataset(seed=42, counts={"train": 16, "validation": 4, "test": 4})
    out = tmp_path_factory.mktemp("splits")
    for split, split_rows in split_examples(examples).items():
        write_jsonl(out / f"{split}.jsonl", split_rows)
    return {"dir": out, "examples": examples}


class FakeExtractor:
    """Deterministic fake 'model' for benchmark/resume tests: echoes the
    gold target for even indices, emits garbage for odd ones."""

    def __init__(self, examples):
        self.gold_by_id = {ex["id"]: ex["target"] for ex in examples}
        self.calls = 0

    def __call__(self, text: str):
        import json
        import time

        self.calls += 1
        ex_id = f"resume_{self.calls:06d}"
        gold = self.gold_by_id.get(ex_id)
        time.sleep(0.001)
        if gold is None or _hash(text) % 2 == 1:
            return {
                "raw_output": "sorry, I cannot produce JSON",
                "parsed_output": None,
                "json_valid": False,
                "schema_valid": False,
                "latency_ms": 1.0,
            }
        raw = json.dumps(gold)
        if _hash(text) % 4 == 2:
            raw = f"```json\n{raw}\n```"
        return {
            "raw_output": raw,
            "parsed_output": gold,
            "json_valid": True,
            "schema_valid": True,
            "latency_ms": 2.5,
        }


def _hash(text: str) -> int:
    value = 0
    for ch in text:
        value = (value * 31 + ord(ch)) % 1000
    return value


def _offset(_text: str) -> int:
    return 0


@pytest.fixture
def fake_extractor(small_dataset):
    return FakeExtractor(small_dataset["examples"])
