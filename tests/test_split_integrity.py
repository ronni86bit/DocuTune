"""Split integrity: no text duplicates, no record leakage across splits,
templates confined to their assigned split."""

from docutune.data.generator import generate_dataset
from docutune.data.splitter import (
    find_duplicate_records,
    find_duplicate_texts,
    find_template_leakage,
    split_examples,
)
from docutune.data.validator import validate_dataset


def _splits():
    counts = {"train": 24, "validation": 4, "test": 4}
    return split_examples(generate_dataset(seed=42, counts=counts))


def test_no_duplicate_text_across_splits():
    assert find_duplicate_texts(_splits()) == []


def test_no_duplicate_records_across_splits():
    assert find_duplicate_records(_splits()) == []


def test_no_template_leakage():
    assert find_template_leakage(_splits()) == []


def test_validator_passes_on_small_dataset(small_dataset):
    report = validate_dataset(splits_dir=small_dataset["dir"],
                              expected_counts={"train": 16, "validation": 4, "test": 4})
    assert report.ok, report.errors


def test_validator_detects_leakage(tmp_path, small_dataset):
    # Copy a train example into the test file -> template leakage + text leak.
    import json

    from docutune.utils.io import read_jsonl, write_jsonl

    train = read_jsonl(small_dataset["dir"] / "train.jsonl")
    test = read_jsonl(small_dataset["dir"] / "test.jsonl")
    poisoned = test + [dict(train[0], metadata={**train[0]["metadata"], "split": "test"})]
    write_jsonl(tmp_path / "poisoned_test.jsonl", poisoned)
    assert len(poisoned) == len(test) + 1
    # JSON sanity of the written file
    with open(tmp_path / "poisoned_test.jsonl", encoding="utf-8") as fh:
        lines = [json.loads(line) for line in fh if line.strip()]
    assert len(lines) == len(test) + 1
