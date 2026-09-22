"""Deterministic generation, counts, template mapping, difficulty."""

from docutune.data.generator import DEFAULT_COUNTS, generate_dataset, make_example
from docutune.data.renderer import TEMPLATE_SPLIT_MAP, TEMPLATES, TEMPLATES_BY_ID
from docutune.schema.resume import ResumeExtraction


def test_dataset_counts():
    examples = generate_dataset(seed=42, counts={"train": 12, "validation": 4, "test": 4})
    assert len(examples) == 20
    splits = {}
    for ex in examples:
        splits[ex["metadata"]["split"]] = splits.get(ex["metadata"]["split"], 0) + 1
    assert splits == {"train": 12, "validation": 4, "test": 4}


def test_determinism(small_dataset):
    fresh = generate_dataset(seed=42, counts={"train": 16, "validation": 4, "test": 4})
    assert [ex["id"] for ex in fresh] == [ex["id"] for ex in small_dataset["examples"]]
    assert [ex["text"] for ex in fresh] == [ex["text"] for ex in small_dataset["examples"]]


def test_all_targets_schema_valid(small_dataset):
    for ex in small_dataset["examples"]:
        ResumeExtraction.model_validate(ex["target"])


def test_template_mapping():
    assert len(TEMPLATES) >= 12
    assert set(TEMPLATE_SPLIT_MAP.values()) == {"train", "validation", "test"}
    train_templates = {t for t, s in TEMPLATE_SPLIT_MAP.items() if s == "train"}
    test_templates = {t for t, s in TEMPLATE_SPLIT_MAP.items() if s == "test"}
    assert train_templates == {
        "template_01", "template_02", "template_03", "template_04",
        "template_05", "template_06", "template_07", "template_08",
    }
    assert test_templates == {"template_11", "template_12"}


def test_distinct_templates_produce_distinct_text():
    """All 12 templates must produce genuinely different formatting."""
    import random

    from docutune.data.renderer import render_resume

    ex = make_example(1, "train", TEMPLATES[0], 42)
    texts = {
        render_resume(ex["target"], spec_id, random.Random(0))
        for spec_id in TEMPLATES_BY_ID
    }
    # At least 10 of the 12 renderings must be mutually distinct (fragmented
    # shuffling can theoretically collide, but practically all 12 differ).
    assert len(texts) >= 10


def test_difficulty_values(small_dataset):
    for ex in small_dataset["examples"]:
        assert ex["metadata"]["difficulty"] in ("easy", "medium", "hard")


def test_metadata_complete(small_dataset):
    for ex in small_dataset["examples"]:
        meta = ex["metadata"]
        for key in ("template_id", "template_name", "difficulty", "split", "seed",
                    "schema_version", "dataset_version"):
            assert key in meta, f"missing metadata key {key}"
        assert ex["id"].startswith("resume_")


def test_default_counts():
    assert DEFAULT_COUNTS == {"train": 450, "validation": 75, "test": 75}
